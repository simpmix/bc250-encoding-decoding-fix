/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * va_decode_hevc.c - the VA-API decode entry point, VAEntrypointVLD for
 * H.265.
 *
 * The same shape as the H.264 one: the application has already parsed the
 * parameter sets and hands over VAPictureParameterBufferHEVC and one
 * VASliceParameterBufferHEVC per slice, so this file is a translation
 * layer that turns VA's structures into the decoder's own.
 *
 * ⚠️ With one difference. The slice header is parsed here, out of the
 * bytes the application handed over, rather than rebuilt from
 * VASliceParameterBufferHEVC. It has to be: wavefront parallelism needs
 * the entry point offsets, VA says how many of them there are and never
 * says what they are, and working out where a substream ends without them
 * is wrong by a byte or two often enough to matter. Everything else in
 * the slice parameters is cross-checked against what the header says and
 * the header wins.
 */
#include "va_backend.h"
#include "decoder_h265.h"
#include "bitreader.h"
#include "hevc_dec_tables.h"   /* the diagonal scans, for the matrices */

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

/* One picture, everything vaEndPicture had, handed to the decode thread:
 * the parameter buffers copied, the slices and their data moved out of the
 * context rather than copied - the context starts the next picture with
 * buffers of its own. */
/* The picture's references, found in the decoder's buffer once, before
 * the picture is begun. ⚠️ Not looked up per slice: with two pictures in
 * flight the next one's reference set may already have let go of a
 * picture this one still reads - the lookup would miss it, where the
 * pointer taken beforehand is held and stays good. */
struct resolved_refs {
    const void *img[15];            /* by index into ReferenceFrames */
    const void *closest;            /* the stand-in when a list is empty */
};

struct bc250_hevc_job {
    VAPictureParameterBufferHEVC pic;
    VAIQMatrixBufferHEVC iq;
    int has_iq;
    struct bc250_hevc_dec_slice *slices;
    int n_slices;
    uint8_t *data;
    int width, height;
    VASurfaceID target;
    gpu_image_t img;
    gpu_memory_t mem;
    struct bc250_hevc_job *next;
    /* Filled by begin_job(). */
    int frame;
    bool begun;
    hevc_sps_t sps;
    hevc_pps_t pps;
    struct resolved_refs rr;
};

/* The context's decode thread and its queue.
 *
 * ⚠️ One thread per context, and pictures decoded in the order they were
 * ended: the decoder keeps its own reference pictures and every picture
 * predicts from the ones before it. The point is not decoding pictures at
 * the same time but letting the application prepare the next one - parse
 * it, fill its buffers - while this one decodes. */
/* Queued or running: one per decode thread and two waiting, so that a
 * thread finishing a picture always finds the next one ready. */
#define ASYNC_DEPTH (HEVC_DECODER_FRAMES + 2)
struct bc250_hevc_async;
struct decode_worker {
    struct bc250_hevc_async *a;
    int frame;                      /* the decoder frame this thread owns */
    pthread_t thread;
    uint8_t *rbsp;
    size_t cap_rbsp;
};
struct bc250_hevc_async {
    struct decode_worker w[HEVC_DECODER_FRAMES];
    int n_workers;
    pthread_mutex_t m;
    pthread_cond_t cv;
    struct bc250_hevc_job *head, *tail;
    int queued;                     /* queued or running */
    bool quit;
    bc250_driver_data *data;
    hevc_decoder_t *dec;
};

void bc250_hevc_dec_reset(bc250_context *c)
{
    c->hevc_dec_state.has_pic = 0;
    c->hevc_dec_state.has_iq = 0;
    c->hevc_dec_state.n_slices = 0;
    c->hevc_dec_state.n_data = 0;
}

void bc250_hevc_dec_free(bc250_context *c)
{
    free(c->hevc_dec_state.slices);
    free(c->hevc_dec_state.data);
    free(c->hevc_dec_state.rbsp);
    c->hevc_dec_state.slices = NULL;
    c->hevc_dec_state.data = NULL;
    c->hevc_dec_state.rbsp = NULL;
    c->hevc_dec_state.cap_slices = 0;
    c->hevc_dec_state.cap_data = 0;
    c->hevc_dec_state.cap_rbsp = 0;
    bc250_hevc_dec_reset(c);
}

static bool grow_slices(bc250_context *c)
{
    if (c->hevc_dec_state.n_slices < c->hevc_dec_state.cap_slices) return true;
    const int new_one = c->hevc_dec_state.cap_slices
        ? c->hevc_dec_state.cap_slices * 2 : 16;
    void *p = realloc(c->hevc_dec_state.slices,
                      (size_t)new_one * sizeof(*c->hevc_dec_state.slices));
    if (!p) return false;
    c->hevc_dec_state.slices = p;
    c->hevc_dec_state.cap_slices = new_one;
    return true;
}

static bool grow_data(bc250_context *c, size_t count)
{
    const size_t serve = c->hevc_dec_state.n_data + count;
    if (serve <= c->hevc_dec_state.cap_data) return true;
    size_t new_one = c->hevc_dec_state.cap_data ? c->hevc_dec_state.cap_data : 65536;
    while (new_one < serve) new_one *= 2;
    uint8_t *p = realloc(c->hevc_dec_state.data, new_one);
    if (!p) return false;
    c->hevc_dec_state.data = p;
    c->hevc_dec_state.cap_data = new_one;
    return true;
}

VAStatus bc250_hevc_dec_render(bc250_context *c, bc250_buffer *b)
{
    if (!b->data) return VA_STATUS_SUCCESS;

    switch (b->type) {
    case VAPictureParameterBufferType:
        if (b->size < sizeof(VAPictureParameterBufferHEVC))
            return VA_STATUS_ERROR_INVALID_PARAMETER;
        memcpy(&c->hevc_dec_state.pic, b->data,
               sizeof(VAPictureParameterBufferHEVC));
        c->hevc_dec_state.has_pic = 1;
        return VA_STATUS_SUCCESS;

    case VAIQMatrixBufferType:
        /* The quantisation matrices, already resolved: defaults, copies
         * and the PPS overriding the SPS are the application's business,
         * and what arrives is the list each block uses. See
         * scaling_from_va() for the order they arrive in. */
        if (b->size < sizeof(VAIQMatrixBufferHEVC))
            return VA_STATUS_ERROR_INVALID_PARAMETER;
        memcpy(&c->hevc_dec_state.iq, b->data, sizeof(VAIQMatrixBufferHEVC));
        c->hevc_dec_state.has_iq = 1;
        return VA_STATUS_SUCCESS;

    case VASliceParameterBufferType: {
        const unsigned n = b->num_elements ? b->num_elements : 1;
        if (b->size < sizeof(VASliceParameterBufferHEVC))
            return VA_STATUS_ERROR_INVALID_PARAMETER;
        for (unsigned k = 0; k < n; k++) {
            if (!grow_slices(c)) return VA_STATUS_ERROR_ALLOCATION_FAILED;
            const VASliceParameterBufferHEVC *src =
                (const VASliceParameterBufferHEVC *)
                ((const uint8_t *)b->data + (size_t)k * b->size);
            const int i = c->hevc_dec_state.n_slices;
            c->hevc_dec_state.slices[i].p = *src;
            c->hevc_dec_state.slices[i].off = (size_t)-1;
            c->hevc_dec_state.slices[i].len = 0;
            c->hevc_dec_state.n_slices++;
        }
        return VA_STATUS_SUCCESS;
    }

    case VASliceDataBufferType: {
        const size_t total = (size_t)b->size
            * (b->num_elements ? b->num_elements : 1);
        for (int i = 0; i < c->hevc_dec_state.n_slices; i++) {
            if (c->hevc_dec_state.slices[i].off != (size_t)-1) continue;
            const VASliceParameterBufferHEVC *p = &c->hevc_dec_state.slices[i].p;
            if (p->slice_data_flag != VA_SLICE_DATA_FLAG_ALL)
                return VA_STATUS_ERROR_UNIMPLEMENTED;
            if ((size_t)p->slice_data_offset + p->slice_data_size > total)
                return VA_STATUS_ERROR_INVALID_PARAMETER;
            if (!grow_data(c, p->slice_data_size))
                return VA_STATUS_ERROR_ALLOCATION_FAILED;
            memcpy(c->hevc_dec_state.data + c->hevc_dec_state.n_data,
                   (const uint8_t *)b->data + p->slice_data_offset,
                   p->slice_data_size);
            c->hevc_dec_state.slices[i].off = c->hevc_dec_state.n_data;
            c->hevc_dec_state.slices[i].len = p->slice_data_size;
            c->hevc_dec_state.n_data += p->slice_data_size;
        }
        return VA_STATUS_SUCCESS;
    }

    default:
        return VA_STATUS_SUCCESS;
    }
}

/* --------------------------------------------------------- translation */

static void fill_sps(const VAPictureParameterBufferHEVC *p, hevc_sps_t *s)
{
    memset(s, 0, sizeof *s);
    s->valid = true;
    s->chroma_format_idc = p->pic_fields.bits.chroma_format_idc;
    s->width = p->pic_width_in_luma_samples;
    s->height = p->pic_height_in_luma_samples;
    s->bit_depth_luma = 8 + p->bit_depth_luma_minus8;
    s->bit_depth_chroma = 8 + p->bit_depth_chroma_minus8;
    s->log2_max_poc_lsb = 4 + p->log2_max_pic_order_cnt_lsb_minus4;
    s->max_dec_pic_buffering = p->sps_max_dec_pic_buffering_minus1 + 1;

    s->log2_min_cb = 3 + p->log2_min_luma_coding_block_size_minus3;
    s->log2_ctb = s->log2_min_cb + p->log2_diff_max_min_luma_coding_block_size;
    s->log2_min_tb = 2 + p->log2_min_transform_block_size_minus2;
    s->log2_max_tb = s->log2_min_tb + p->log2_diff_max_min_transform_block_size;
    s->max_transform_hierarchy_depth_intra = p->max_transform_hierarchy_depth_intra;
    s->max_transform_hierarchy_depth_inter = p->max_transform_hierarchy_depth_inter;

    s->scaling_list_enabled = p->pic_fields.bits.scaling_list_enabled_flag;
    s->amp_enabled = p->pic_fields.bits.amp_enabled_flag;
    s->sao_enabled = p->slice_parsing_fields.bits.sample_adaptive_offset_enabled_flag;
    s->pcm_enabled = p->pic_fields.bits.pcm_enabled_flag;
    s->pcm_bit_depth_luma = p->pcm_sample_bit_depth_luma_minus1 + 1;
    s->pcm_bit_depth_chroma = p->pcm_sample_bit_depth_chroma_minus1 + 1;
    s->log2_min_pcm_cb = 3 + p->log2_min_pcm_luma_coding_block_size_minus3;
    s->log2_max_pcm_cb = s->log2_min_pcm_cb
        + p->log2_diff_max_min_pcm_luma_coding_block_size;
    s->pcm_loop_filter_disabled = p->pic_fields.bits.pcm_loop_filter_disabled_flag;
    s->long_term_ref_pics_present = p->slice_parsing_fields.bits.long_term_ref_pics_present_flag;
    s->num_long_term_sps = p->num_long_term_ref_pic_sps;
    s->temporal_mvp_enabled = p->slice_parsing_fields.bits.sps_temporal_mvp_enabled_flag;
    s->strong_intra_smoothing = p->pic_fields.bits.strong_intra_smoothing_enabled_flag;

    /* ⚠️ The application has already applied the conformance window: the
     * size it asks for is the size it wants back. Cropping again here
     * would take it off twice. */
    s->crop_left = s->crop_right = s->crop_top = s->crop_bottom = 0;

    s->ctb_size = 1 << s->log2_ctb;
    s->ctb_width = (s->width + s->ctb_size - 1) >> s->log2_ctb;
    s->ctb_height = (s->height + s->ctb_size - 1) >> s->log2_ctb;
    s->ctb_count = s->ctb_width * s->ctb_height;
    s->min_cb_width = s->width >> s->log2_min_cb;
    s->min_cb_height = s->height >> s->log2_min_cb;

    /* ⚠️ The short-term sets are NOT always inline in the slice header.
     * HM - and most MP4 and MKV files - keep them in the SPS and send only
     * an index. The header is still re-read here for its entry points and
     * its exact data offset, and with zero sets declared that read failed
     * on every such slice. VA does not pass the sets' contents, but it
     * passes how many there are, which is what the index needs, and
     * st_rps_bits, which is how long an inline set is, so the parser can
     * step over it. Neither is needed for anything else: the reference
     * lists arrive finished. */
    s->num_st_rps = p->num_short_term_ref_pic_sets;
    s->st_rps_bits = (int)p->st_rps_bits;
}

static void fill_pps(const VAPictureParameterBufferHEVC *p, hevc_pps_t *q)
{
    memset(q, 0, sizeof *q);
    q->valid = true;
    q->pps_id = 0;
    q->sps_id = 0;
    q->dependent_slice_segments_enabled =
        p->slice_parsing_fields.bits.dependent_slice_segments_enabled_flag;
    q->output_flag_present = p->slice_parsing_fields.bits.output_flag_present_flag;
    q->num_extra_slice_header_bits = p->num_extra_slice_header_bits;
    q->sign_data_hiding = p->pic_fields.bits.sign_data_hiding_enabled_flag;
    q->cabac_init_present = p->slice_parsing_fields.bits.cabac_init_present_flag;
    q->num_ref_idx_default[0] = p->num_ref_idx_l0_default_active_minus1 + 1;
    q->num_ref_idx_default[1] = p->num_ref_idx_l1_default_active_minus1 + 1;
    q->init_qp = 26 + p->init_qp_minus26;
    q->constrained_intra_pred = p->pic_fields.bits.constrained_intra_pred_flag;
    q->transform_skip_enabled = p->pic_fields.bits.transform_skip_enabled_flag;
    q->cu_qp_delta_enabled = p->pic_fields.bits.cu_qp_delta_enabled_flag;
    q->diff_cu_qp_delta_depth = p->diff_cu_qp_delta_depth;
    q->cb_qp_offset = p->pps_cb_qp_offset;
    q->cr_qp_offset = p->pps_cr_qp_offset;
    q->slice_chroma_qp_offsets_present =
        p->slice_parsing_fields.bits.pps_slice_chroma_qp_offsets_present_flag;
    q->weighted_pred = p->pic_fields.bits.weighted_pred_flag;
    q->weighted_bipred = p->pic_fields.bits.weighted_bipred_flag;
    q->transquant_bypass_enabled = p->pic_fields.bits.transquant_bypass_enabled_flag;
    q->tiles_enabled = p->pic_fields.bits.tiles_enabled_flag;
    q->entropy_coding_sync_enabled = p->pic_fields.bits.entropy_coding_sync_enabled_flag;
    q->num_tile_columns = p->num_tile_columns_minus1 + 1;
    q->num_tile_rows = p->num_tile_rows_minus1 + 1;
    q->loop_filter_across_tiles = p->pic_fields.bits.loop_filter_across_tiles_enabled_flag;

    /* ⚠️ VA carries no uniform_spacing flag. Its own header says the
     * application fills the width arrays whatever the stream said, so
     * the layout that arrives here is always the explicit one - which is
     * why this sets uniform_spacing false rather than passing something
     * through. hevcd_prepare_tiles() then derives the last column and
     * the last row, as it does for an explicit layout out of a
     * bitstream, and the totals have to come out right or it refuses.
     *
     * The bounds are VA's own array sizes, checked before the copy in
     * bc250_hevc_dec_decode(). */
    q->uniform_spacing = false;
    for (int i = 0; i < q->num_tile_columns - 1; i++)
        q->column_width[i] = p->column_width_minus1[i] + 1;
    for (int i = 0; i < q->num_tile_rows - 1; i++)
        q->row_height[i] = p->row_height_minus1[i] + 1;
    q->loop_filter_across_slices =
        p->pic_fields.bits.pps_loop_filter_across_slices_enabled_flag;
    q->deblocking_filter_override_enabled =
        p->slice_parsing_fields.bits.deblocking_filter_override_enabled_flag;
    q->deblocking_filter_disabled =
        p->slice_parsing_fields.bits.pps_disable_deblocking_filter_flag;
    q->deblocking_filter_control_present = 1;
    q->beta_offset = 2 * p->pps_beta_offset_div2;
    q->tc_offset = 2 * p->pps_tc_offset_div2;
    q->pps_scaling_list_present = 0;
    q->lists_modification_present =
        p->slice_parsing_fields.bits.lists_modification_present_flag;
    q->log2_parallel_merge_level = 2 + p->log2_parallel_merge_level_minus2;
    q->slice_segment_header_extension_present =
        p->slice_parsing_fields.bits.slice_segment_header_extension_present_flag;
}

static bool is_valid(const VAPictureHEVC *p)
{
    return !(p->flags & VA_PICTURE_HEVC_INVALID)
        && p->picture_id != VA_INVALID_SURFACE;
}

/* VA's matrices into the parser's layout: the 32x32 ones moved to
 * matrixId 0 and 3, and every list put back into the up-right diagonal
 * order the bitstream sends.
 *
 * ⚠️ VA holds them in RASTER order - row after row of the 4x4 or 8x8 base
 * matrix, as the comment above the structure in va_dec_hevc.h says, and as
 * a dump of what ffmpeg sends for SLIST_A shows (6 9 13 18 25 35 36 37,
 * then 9 10 15 21 ...). Taken as if they were already diagonal, every
 * factor but the first lands on the wrong frequency. */
static void scaling_from_va(const VAIQMatrixBufferHEVC *q, hevc_scaling_t *s)
{
    hevc_scaling_defaults(s);
    for (int m = 0; m < 6; m++) {
        for (int i = 0; i < 16; i++)
            s->list[0][m][i] =
                q->ScalingList4x4[m][hevcd_diag4_y[i] * 4 + hevcd_diag4_x[i]];
        for (int i = 0; i < 64; i++) {
            const int r = hevcd_diag8_y[i] * 8 + hevcd_diag8_x[i];
            s->list[1][m][i] = q->ScalingList8x8[m][r];
            s->list[2][m][i] = q->ScalingList16x16[m][r];
            if (m < 2) s->list[3][3 * m][i] = q->ScalingList32x32[m][r];
        }
        s->dc[2][m] = q->ScalingListDC16x16[m];
    }
    for (int k = 0; k < 2; k++)
        s->dc[3][3 * k] = q->ScalingListDC32x32[k];
}

static void fill_slice(const VAPictureParameterBufferHEVC *pp,
                       const VASliceParameterBufferHEVC *sp,
                       const hevc_pps_t *pps,
                       int nal_type,
                       const struct resolved_refs *rr,
                       hevc_slice_t *sl)
{
    memset(sl, 0, sizeof(*sl));
    sl->nal_type = nal_type;
    sl->first_slice_in_pic = (sp->slice_segment_address == 0);
    sl->dependent_slice_segment = sp->LongSliceFlags.fields.dependent_slice_segment_flag != 0;
    sl->segment_address = sp->slice_segment_address;
    sl->type = sp->LongSliceFlags.fields.slice_type; /* 0: B, 1: P, 2: I */
    sl->pic_output_flag = true;
    sl->poc = pp->CurrPic.pic_order_cnt;
    sl->poc_lsb = pp->CurrPic.pic_order_cnt & ((1 << (4 + pp->log2_max_pic_order_cnt_lsb_minus4)) - 1);

    sl->temporal_mvp_enabled = sp->LongSliceFlags.fields.slice_temporal_mvp_enabled_flag != 0;
    sl->sao_luma = sp->LongSliceFlags.fields.slice_sao_luma_flag != 0;
    sl->sao_chroma = sp->LongSliceFlags.fields.slice_sao_chroma_flag != 0;

    sl->num_ref_idx[0] = (sl->type == 2) ? 0 : (sp->num_ref_idx_l0_active_minus1 + 1);
    sl->num_ref_idx[1] = (sl->type == 0) ? (sp->num_ref_idx_l1_active_minus1 + 1) : 0;
    if (sl->num_ref_idx[0] > 15) sl->num_ref_idx[0] = 15;
    if (sl->num_ref_idx[1] > 15) sl->num_ref_idx[1] = 15;

    sl->mvd_l1_zero = sp->LongSliceFlags.fields.mvd_l1_zero_flag != 0;
    sl->cabac_init_flag = sp->LongSliceFlags.fields.cabac_init_flag != 0;
    sl->collocated_from_l0 = sp->LongSliceFlags.fields.collocated_from_l0_flag != 0;
    sl->collocated_ref_idx = sp->collocated_ref_idx;
    sl->five_minus_max_num_merge_cand = sp->five_minus_max_num_merge_cand;

    sl->qp = pps->init_qp + sp->slice_qp_delta;
    /* ⚠️ The slice's own offset only. block_qp() adds the PPS's to it, as
     * 8.6.1 does; adding it here as well counted it twice, which is
     * invisible whenever the PPS offset is zero - x265's default - and a
     * wrong chroma quantiser whenever it is not. */
    sl->cb_qp_offset = sp->slice_cb_qp_offset;
    sl->cr_qp_offset = sp->slice_cr_qp_offset;
    sl->deblocking_filter_disabled = sp->LongSliceFlags.fields.slice_deblocking_filter_disabled_flag != 0;
    sl->beta_offset = 2 * sp->slice_beta_offset_div2;
    sl->tc_offset = 2 * sp->slice_tc_offset_div2;
    sl->loop_filter_across_slices = sp->LongSliceFlags.fields.slice_loop_filter_across_slices_enabled_flag != 0;

    /* Weighted prediction */
    sl->luma_log2_weight_denom = sp->luma_log2_weight_denom;
    sl->chroma_log2_weight_denom = sp->luma_log2_weight_denom + sp->delta_chroma_log2_weight_denom;

    const int16_t one_l = (int16_t)(1 << sp->luma_log2_weight_denom);
    const int16_t one_c = (int16_t)(1 << sl->chroma_log2_weight_denom);

    for (int r = 0; r < 16; r++) {
        if (r < 15) {
            sl->luma_weight[0][r] = one_l + sp->delta_luma_weight_l0[r];
            sl->luma_offset[0][r] = sp->luma_offset_l0[r];
            sl->luma_weight[1][r] = one_l + sp->delta_luma_weight_l1[r];
            sl->luma_offset[1][r] = sp->luma_offset_l1[r];
            for (int j = 0; j < 2; j++) {
                sl->chroma_weight[0][r][j] = one_c + sp->delta_chroma_weight_l0[r][j];
                sl->chroma_offset[0][r][j] = sp->ChromaOffsetL0[r][j];
                sl->chroma_weight[1][r][j] = one_c + sp->delta_chroma_weight_l1[r][j];
                sl->chroma_offset[1][r][j] = sp->ChromaOffsetL1[r][j];
            }
        } else {
            sl->luma_weight[0][r] = one_l;
            sl->luma_weight[1][r] = one_l;
            for (int j = 0; j < 2; j++) {
                sl->chroma_weight[0][r][j] = one_c;
                sl->chroma_weight[1][r][j] = one_c;
            }
        }
    }

    sl->data_bit_offset = (size_t)sp->slice_data_byte_offset << 3;
    sl->num_entry_point_offsets = sp->num_entry_point_offsets;

    /* Build explicit reference picture lists from RefPicList[2][15] */
    sl->has_explicit_rpl = true;
    for (int l = 0; l < 2; l++) {
        sl->explicit_n_refs[l] = 0;
        const int want = sl->num_ref_idx[l];
        for (int r = 0; r < want && r < 15; r++) {
            uint8_t ref_idx = sp->RefPicList[l][r];
            if (ref_idx < 15 && is_valid(&pp->ReferenceFrames[ref_idx])) {
                uintptr_t ref_surf = (uintptr_t)pp->ReferenceFrames[ref_idx].picture_id;
                int ref_poc = pp->ReferenceFrames[ref_idx].pic_order_cnt;
                (void)ref_surf; (void)ref_poc;
                const void *img = rr->img[ref_idx];
                if (img) {
                    /* Long-term or not decides whether a motion vector
                     * pointing at it may be scaled, 8.5.3.2.7. */
                    sl->explicit_lt[l][sl->explicit_n_refs[l]] =
                        (pp->ReferenceFrames[ref_idx].flags
                         & VA_PICTURE_HEVC_LONG_TERM_REFERENCE) != 0;
                    sl->explicit_ref_pic[l][sl->explicit_n_refs[l]++] = img;
                }
            }
        }
        /* Concealment fallback if reference frame was dropped or lost */
        if (sl->explicit_n_refs[l] == 0 && want > 0) {
            const void *any = rr->closest;
            if (any) {
                while (sl->explicit_n_refs[l] < want)
                    sl->explicit_ref_pic[l][sl->explicit_n_refs[l]++] = any;
            }
        } else {
            while (sl->explicit_n_refs[l] < want && sl->explicit_n_refs[l] > 0) {
                sl->explicit_ref_pic[l][sl->explicit_n_refs[l]] =
                    sl->explicit_ref_pic[l][sl->explicit_n_refs[l] - 1];
                sl->explicit_lt[l][sl->explicit_n_refs[l]] =
                    sl->explicit_lt[l][sl->explicit_n_refs[l] - 1];
                sl->explicit_n_refs[l]++;
            }
        }
        sl->num_ref_idx[l] = sl->explicit_n_refs[l];
    }

    /* Collocated picture */
    sl->explicit_col = NULL;
    if (sl->temporal_mvp_enabled && sp->collocated_ref_idx < 15) {
        const int l = sl->collocated_from_l0 ? 0 : 1;
        if (sp->collocated_ref_idx < sl->explicit_n_refs[l])
            sl->explicit_col = sl->explicit_ref_pic[l][sp->collocated_ref_idx];
    }
}

/* The checks vaEndPicture can still answer with an error. Everything here
 * reads only the picture parameters, so it is cheap, and it is what used to
 * be refused at the top of the decode. */
VAStatus bc250_hevc_dec_check(bc250_context *c)
{
    if (!c->h265_dec) return VA_STATUS_ERROR_INVALID_CONTEXT;
    if (!c->hevc_dec_state.has_pic || c->hevc_dec_state.n_slices == 0)
        return VA_STATUS_ERROR_INVALID_PARAMETER;

    const VAPictureParameterBufferHEVC *pp = &c->hevc_dec_state.pic;

    /* 4:2:0 at eight or ten bits, and the two depths equal. Everything
     * below eight bits and everything above ten is a profile this
     * decoder does not claim; luma and chroma differing is one the
     * templates cannot serve, because every line that picks between the
     * two builds reads the luma depth. */
    if (pp->pic_fields.bits.chroma_format_idc != 1
        || (pp->bit_depth_luma_minus8 != 0 && pp->bit_depth_luma_minus8 != 2)
        || pp->bit_depth_chroma_minus8 != pp->bit_depth_luma_minus8)
        return VA_STATUS_ERROR_UNSUPPORTED_PROFILE;
    /* ⚠️ VA's own arrays hold nineteen column widths and twenty-one row
     * heights. A layout with more than that has nowhere to have come
     * from, and copying it would read past the end of the structure the
     * application handed us. */
    if (pp->num_tile_columns_minus1 > 19 || pp->num_tile_rows_minus1 > 21)
        return VA_STATUS_ERROR_UNSUPPORTED_PROFILE;
    /* ⚠️ Tiles and wavefront together are refused in the parameter set
     * parse for the same reason they are refused there: the substreams
     * would run per tile per row and nothing here walks that. Checked
     * again on this side because the parameter set never gets parsed on
     * the VA path - it arrives already taken apart. */
    if (pp->pic_fields.bits.tiles_enabled_flag
        && pp->pic_fields.bits.entropy_coding_sync_enabled_flag)
        return VA_STATUS_ERROR_UNSUPPORTED_PROFILE;
    hevc_sps_t sps;
    fill_sps(pp, &sps);
    if (sps.width != c->width || sps.height != c->height)
        return VA_STATUS_ERROR_RESOLUTION_NOT_SUPPORTED;
    return VA_STATUS_SUCCESS;
}

struct bc250_hevc_job *bc250_hevc_dec_take(bc250_context *c, VASurfaceID target,
                                           gpu_image_t img, gpu_memory_t mem)
{
    struct bc250_hevc_job *job = calloc(1, sizeof *job);
    if (!job) return NULL;
    job->pic = c->hevc_dec_state.pic;
    job->iq = c->hevc_dec_state.iq;
    job->has_iq = c->hevc_dec_state.has_iq;
    job->slices = c->hevc_dec_state.slices;
    job->n_slices = c->hevc_dec_state.n_slices;
    job->data = c->hevc_dec_state.data;
    job->width = c->width;
    job->height = c->height;
    job->target = target;
    job->img = img;
    job->mem = mem;
    /* Moved, not copied: the next picture grows buffers of its own. */
    c->hevc_dec_state.slices = NULL;
    c->hevc_dec_state.cap_slices = 0;
    c->hevc_dec_state.data = NULL;
    c->hevc_dec_state.cap_data = 0;
    bc250_hevc_dec_reset(c);
    return job;
}

static void free_job(struct bc250_hevc_job *job)
{
    free(job->slices);
    free(job->data);
    free(job);
}

/* The first half of a picture, which has to happen in decoding order: the
 * parameter sets, which pictures are still references, pointers to them,
 * and the picture opened on frame `frame`. Cheap; the decode thread does it
 * with the queue lock held, which is what keeps the order. */
static VAStatus begin_job(hevc_decoder_t *dec, int frame,
                          struct bc250_hevc_job *job)
{
    const VAPictureParameterBufferHEVC *pp = &job->pic;
    hevc_sps_t *sps = &job->sps;
    hevc_pps_t *pps = &job->pps;
    fill_sps(pp, sps);
    fill_pps(pp, pps);
    /* Enabled with no matrix buffer means the defaults, as it does in a
     * bitstream. */
    if (sps->scaling_list_enabled) {
        if (job->has_iq)
            scaling_from_va(&job->iq, &sps->scaling);
        else
            hevc_scaling_defaults(&sps->scaling);
    }
    /* ⚠️ NumPicTotalCurr sets the width of every list_entry in the slice
     * header the fallback parse below reads, and it counts long-term
     * pictures the SPS marks as used - flags VA does not pass on. VA does
     * say which reference is in which current subset, and those three
     * counted together ARE the number, as va.h itself puts it. */
    for (int i = 0; i < 15; i++)
        if (is_valid(&pp->ReferenceFrames[i])
            && (pp->ReferenceFrames[i].flags
                & (VA_PICTURE_HEVC_RPS_ST_CURR_BEFORE
                   | VA_PICTURE_HEVC_RPS_ST_CURR_AFTER
                   | VA_PICTURE_HEVC_RPS_LT_CURR)))
            pps->num_pic_total_curr++;

    /* Which surfaces are still references. This runs before the picture is
     * opened, so that the one about to be decoded cannot be handed a slot
     * it is about to read from. */
    uintptr_t ref_pic[16];
    int poc[16];
    int n_refs = 0;
    for (int i = 0; i < 15 && n_refs < 16; i++) {
        if (!is_valid(&pp->ReferenceFrames[i])) continue;
        ref_pic[n_refs] = (uintptr_t)pp->ReferenceFrames[i].picture_id;
        poc[n_refs] = pp->ReferenceFrames[i].pic_order_cnt;
        n_refs++;
    }
    hevc_decoder_set_references(dec, ref_pic, poc, n_refs);

    /* Every reference found now, and held from begin_frame on. */
    const void *held[16];
    int n_held = 0;
    memset(&job->rr, 0, sizeof job->rr);
    for (int i = 0; i < 15; i++) {
        if (!is_valid(&pp->ReferenceFrames[i])) continue;
        job->rr.img[i] = hevc_decoder_find_ref(
            dec, (uintptr_t)pp->ReferenceFrames[i].picture_id,
            pp->ReferenceFrames[i].pic_order_cnt);
        if (job->rr.img[i] && n_held < 16) held[n_held++] = job->rr.img[i];
    }
    job->rr.closest = hevc_decoder_find_closest(dec, pp->CurrPic.pic_order_cnt);
    if (job->rr.closest && n_held < 16) held[n_held++] = job->rr.closest;

    job->frame = frame;
    if (hevc_decoder_begin_frame(dec, frame, sps, pps, (uintptr_t)job->target,
                                 pp->CurrPic.pic_order_cnt, held, n_held) != 0)
        return VA_STATUS_ERROR_OPERATION_FAILED;
    job->begun = true;
    return VA_STATUS_SUCCESS;
}

/* The rest, which may run beside the next picture's: every slice, the loop
 * filters, and the copy into the surface. */
static VAStatus finish_job(hevc_decoder_t *dec, struct bc250_hevc_job *job,
                           uint8_t **rbsp_p, size_t *cap_rbsp)
{
    const VAPictureParameterBufferHEVC *pp = &job->pic;
    const int frame = job->frame;
    const hevc_pps_t *pps = &job->pps;

    /* One set of parameter sets for optional bitstream entry-point fallback */
    hevc_sps_t sps_store[16];
    hevc_pps_t pps_store[64];
    for (int i = 0; i < 16; i++) sps_store[i] = job->sps;
    for (int i = 0; i < 64; i++) { pps_store[i] = *pps; pps_store[i].pps_id = i; }

    for (int i = 0; i < job->n_slices; i++) {
        if (job->slices[i].off == (size_t)-1) continue;
        const uint8_t *raw = job->data + job->slices[i].off;
        const size_t raw_len = job->slices[i].len;
        if (raw_len < 3) continue;

        /* Skip start code prefix if present in buffer */
        size_t nal_off = 0;
        if (raw[0] == 0 && raw[1] == 0) {
            if (raw[2] == 1) nal_off = 3;
            else if (raw_len >= 4 && raw[2] == 0 && raw[3] == 1) nal_off = 4;
        }
        const uint8_t *nal = raw + nal_off;
        const size_t len = raw_len - nal_off;
        if (len < 3) continue;

        if (*cap_rbsp < len) {
            uint8_t *p = realloc(*rbsp_p, len);
            if (!p) {
                hevc_decoder_end_frame(dec, frame);
                return VA_STATUS_ERROR_ALLOCATION_FAILED;
            }
            *rbsp_p = p;
            *cap_rbsp = len;
        }
        uint8_t *rbsp = *rbsp_p;
        const size_t n = br_extract_rbsp(rbsp, len, nal, len);

        const int kind = (nal[0] >> 1) & 0x3f;
        const VASliceParameterBufferHEVC *sp = &job->slices[i].p;

        hevc_slice_t sl;
        fill_slice(pp, sp, pps, kind, &job->rr, &sl);

        /* The header re-read, for what VA does not carry in the form the
         * decoder wants. ⚠️ Always tried, and when it reads cleanly it
         * decides two things:
         *   - the entry points, which VA counts but does not list;
         *   - where the slice data begins. The decoder reads the payload
         *     with its emulation prevention bytes removed, and an offset
         *     measured on the NAL unit as sent is past the right place by
         *     however many of them the header held. The parser's offset is
         *     on the same payload the decoder reads.
         * When it does not read, VA's figures stand. */
        {
            hevc_slice_t parsed;
            if (hevc_ps_read_slice(&parsed, rbsp, n, kind,
                                   sps_store, pps_store) == 0) {
                sl.data_bit_offset = parsed.data_bit_offset;
                sl.num_entry_point_offsets = parsed.num_entry_point_offsets;
                memcpy(sl.entry_point, parsed.entry_point, sizeof(sl.entry_point));
                if (sl.num_entry_point_offsets > 0)
                    hevc_decoder_shift_entry_points(&sl, nal, len,
                                                    sl.data_bit_offset >> 3);
            }
        }

        hevc_decoder_frame_slice(dec, frame, &sl, rbsp, n);
    }

    /* ⚠️ Always finished, even when every slice was refused: a picture
     * that is never ended leaves the buffer holding a slot no later one
     * can reuse, the application still gets its surface back, and a
     * picture decoding beside this one may be waiting for its rows. */
    hevc_decoder_end_frame(dec, frame);
    if (hevc_decoder_load_frame(dec, frame, job->img, job->mem) != 0)
        return VA_STATUS_ERROR_OPERATION_FAILED;
    return VA_STATUS_SUCCESS;
}

/* Both halves in a row, on frame 0: the synchronous path. */
static VAStatus run_job(hevc_decoder_t *dec, struct bc250_hevc_job *job,
                        uint8_t **rbsp_p, size_t *cap_rbsp)
{
    const VAStatus st = begin_job(dec, 0, job);
    if (st != VA_STATUS_SUCCESS) return st;
    return finish_job(dec, job, rbsp_p, cap_rbsp);
}

/* One of the context's decode threads. It owns one decoder frame. It takes
 * the next picture and begins it with the queue lock still held - so that
 * pictures are begun strictly in the order they were ended, whichever
 * thread takes them - then decodes it with the lock dropped, beside
 * whatever the other thread is decoding. A picture that reads one still
 * being decoded waits inside the decoder for the rows it needs. */
static void *decode_thread(void *arg)
{
    struct decode_worker *w = arg;
    struct bc250_hevc_async *a = w->a;
    for (;;) {
        pthread_mutex_lock(&a->m);
        while (!a->head && !a->quit) pthread_cond_wait(&a->cv, &a->m);
        struct bc250_hevc_job *job = a->head;
        if (!job) {                         /* quit, and nothing left */
            pthread_mutex_unlock(&a->m);
            break;
        }
        a->head = job->next;
        if (!a->head) a->tail = NULL;
        VAStatus st = begin_job(a->dec, w->frame, job);
        pthread_mutex_unlock(&a->m);

        if (st == VA_STATUS_SUCCESS)
            st = finish_job(a->dec, job, &w->rbsp, &w->cap_rbsp);
        bc250_decode_finished(a->data, job->target, st);
        free_job(job);

        pthread_mutex_lock(&a->m);
        a->queued--;
        pthread_cond_broadcast(&a->cv);     /* room in the queue */
        pthread_mutex_unlock(&a->m);
    }
    return NULL;
}

static struct bc250_hevc_async *async_start(bc250_driver_data *data,
                                            hevc_decoder_t *dec)
{
    struct bc250_hevc_async *a = calloc(1, sizeof *a);
    if (!a) return NULL;
    a->data = data;
    a->dec = dec;
    pthread_mutex_init(&a->m, NULL);
    pthread_cond_init(&a->cv, NULL);

    /* BC250_HEVC_FRAMES=1 decodes one picture at a time, in order - still
     * after vaEndPicture has returned. */
    int want = HEVC_DECODER_FRAMES;
    const char *s = getenv("BC250_HEVC_FRAMES");
    if (s && atoi(s) >= 1 && atoi(s) < want) want = atoi(s);
    for (int k = 0; k < want; k++) {
        struct decode_worker *w = &a->w[a->n_workers];
        w->a = a;
        w->frame = k;
        if (pthread_create(&w->thread, NULL, decode_thread, w) == 0)
            a->n_workers++;
    }
    if (!a->n_workers) {
        pthread_mutex_destroy(&a->m);
        pthread_cond_destroy(&a->cv);
        free(a);
        return NULL;
    }
    return a;
}

void bc250_hevc_async_stop(struct bc250_hevc_async *a)
{
    if (!a) return;
    pthread_mutex_lock(&a->m);
    a->quit = true;
    pthread_cond_broadcast(&a->cv);
    pthread_mutex_unlock(&a->m);
    for (int k = 0; k < a->n_workers; k++) {
        pthread_join(a->w[k].thread, NULL);
        free(a->w[k].rbsp);
    }
    pthread_mutex_destroy(&a->m);
    pthread_cond_destroy(&a->cv);
    free(a);
}

VAStatus bc250_hevc_dec_submit(bc250_driver_data *data, bc250_context *c,
                               struct bc250_hevc_job *job)
{
    /* Asked every picture rather than remembered: two contexts can end
     * pictures at the same moment, and a remembered answer written by
     * both is a data race, harmless or not. getenv is cheap next to a
     * picture. */
    const bool sync_mode = getenv("BC250_HEVC_SYNC") != NULL;

    if (!c->hevc_async && !sync_mode)
        c->hevc_async = async_start(data, c->h265_dec);
    struct bc250_hevc_async *a = c->hevc_async;
    if (!a) {
        /* Synchronous, as before: asked for, or no thread to be had. */
        const VAStatus st = run_job(c->h265_dec, job,
                                    &c->hevc_dec_state.rbsp,
                                    &c->hevc_dec_state.cap_rbsp);
        bc250_decode_finished(data, job->target, st);
        free_job(job);
        return st;
    }

    pthread_mutex_lock(&a->m);
    /* A few pictures ahead at most: past that the application is only
     * filling memory, and a player's latency grows with every one. */
    while (a->queued >= ASYNC_DEPTH) pthread_cond_wait(&a->cv, &a->m);
    job->next = NULL;
    if (a->tail) a->tail->next = job;
    else a->head = job;
    a->tail = job;
    a->queued++;
    pthread_cond_broadcast(&a->cv);
    pthread_mutex_unlock(&a->m);
    return VA_STATUS_SUCCESS;
}
