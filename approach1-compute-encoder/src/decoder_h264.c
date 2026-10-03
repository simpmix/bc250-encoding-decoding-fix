/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * decoder_h264.c - the picture level: the frame store, the slice loop and
 * the hand-over of the finished picture.
 */
#include "h264_dec_internal.h"
#include "h264_simd.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------ lifetime */

bool h264_decoder_supports(int profile_idc, int chroma_format_idc,
                           int bit_depth_luma, int bit_depth_chroma,
                           bool frame_mbs_only, bool mb_adaptive)
{
    (void)profile_idc;          /* High, Main and Baseline are all fine */
    if (chroma_format_idc != 1) return false;      /* 4:2:0 only */
    if (bit_depth_luma != 8 || bit_depth_chroma != 8) return false;
    if (!frame_mbs_only || mb_adaptive) return false;   /* progressive only */
    return true;
}

static void free_frame(h264d_frame_t *f)
{
    free(f->y);
    free(f->col_ref);
    free(f->col_mv);
    free(f->col_poc);
    f->y = f->cb = f->cr = NULL;
    f->col_ref = NULL;
    f->col_mv = NULL;
    f->col_poc = NULL;
    f->used = false;
    f->surface = ~0u;
}

static int alloc_frame(h264d_frame_t *f, int w, int h)
{
    const int sy = (w + 31) & ~31;
    const int sc = (w / 2 + 31) & ~31;
    const size_t n = (size_t)sy * h + 2 * (size_t)sc * (h / 2);
    f->y = calloc(1, n);
    if (!f->y) return -1;
    f->cb = f->y + (size_t)sy * h;
    f->cr = f->cb + (size_t)sc * (h / 2);
    f->stride_y = sy;
    f->stride_c = sc;
    f->surface = ~0u;
    f->used = false;
    return 0;
}

h264_decoder_t *h264_decoder_create(bc250_gpu_context_t *gpu_ctx,
                                    uint32_t width, uint32_t height)
{
    h264_decoder_t *d = calloc(1, sizeof(*d));
    if (!d) return NULL;

    d->gpu = gpu_ctx;
    d->width = (int)width;
    d->height = (int)height;
    d->mb_w = ((int)width + 15) / 16;
    d->mb_h = ((int)height + 15) / 16;
    d->mb_count = d->mb_w * d->mb_h;

    d->mbs = calloc((size_t)d->mb_count, sizeof(h264d_mb_t));
    d->slice_of_mb = calloc((size_t)d->mb_count, 1);
    /* Default to full picture height (d->mb_h) for zero-stall wavefront
     * reconstruction across all worker threads. On the BC-250's 16 GB unified
     * GDDR6 memory (448 GB/s bandwidth), allocating full-frame residuals (~43 MB at 4K,
     * ~11 MB at 1080p) eliminates 34+ blocking thread synchronization barriers per frame
     * and avoids worker thread starvation across 8 CPU cores.
     * Retain BC250_H264_BAND override for manual cache-tuning. */
    {
        int b = d->mb_h;
        const char *e = getenv("BC250_H264_BAND");
        if (e) b = atoi(e);
        if (b < 4) b = 4;
        if (b > d->mb_h) b = d->mb_h;
        d->band_rows = b;
    }
    d->slices = calloc(H264D_MAX_SLICES, sizeof(h264d_slice_t));
    d->dequant = calloc(1, sizeof(h264d_dequant_set_t));
    d->n_residuals = (d->band_rows + 1) * d->mb_w;
    if (d->n_residuals > d->mb_count) d->n_residuals = d->mb_count;
    d->residuals = calloc((size_t)d->n_residuals, sizeof(h264d_residual_t));
    d->res = d->residuals;
    d->uv_buf = NULL;
    d->uv_cap = 0;
    d->rbsp_cap = (size_t)d->mb_count * 512 + 65536;
    d->rbsp = malloc(d->rbsp_cap);
    if (!d->mbs || !d->slice_of_mb || !d->residuals || !d->rbsp
        || !d->slices || !d->dequant) {
        h264_decoder_destroy(d);
        return NULL;
    }

    for (int i = 0; i < H264D_DPB_SIZE; i++) {
        if (alloc_frame(&d->dpb[i], d->mb_w * 16, d->mb_h * 16) != 0) {
            h264_decoder_destroy(d);
            return NULL;
        }
        d->dpb[i].col_ref = calloc((size_t)d->mb_count * 2 * 4, sizeof(int8_t));
        d->dpb[i].col_mv = calloc((size_t)d->mb_count * 2 * 16 * 2, sizeof(int16_t));
        d->dpb[i].col_poc = calloc((size_t)d->mb_count * 2 * 4, sizeof(int32_t));
        if (!d->dpb[i].col_ref || !d->dpb[i].col_mv || !d->dpb[i].col_poc) {
            h264_decoder_destroy(d);
            return NULL;
        }
    }
    d->cur = -1;
    return d;
}

void h264_decoder_destroy(h264_decoder_t *d)
{
    if (!d) return;
    for (int i = 0; i < H264D_DPB_SIZE; i++)
        free_frame(&d->dpb[i]);
    h264d_pool_stop(d);
    free(d->mbs);
    free(d->slice_of_mb);
    free(d->residuals);
    free(d->slices);
    free(d->dequant);
    free(d->rbsp);
    free(d->uv_buf);
    free(d);
}

h264d_frame_t *h264_decoder_frame_for(h264_decoder_t *d, uint32_t surface)
{
    for (int i = 0; i < H264D_DPB_SIZE; i++)
        if (d->dpb[i].used && d->dpb[i].surface == surface)
            return &d->dpb[i];
    return NULL;
}

int h264_decoder_slot_of(const h264_decoder_t *d, const h264d_frame_t *f)
{
    const int i = (int)(f - d->dpb);
    return (i >= 0 && i < H264D_DPB_SIZE) ? i : 0;
}

/* The slot a surface already occupies, or a free one. */
static int slot_per(h264_decoder_t *d, uint32_t surface)
{
    for (int i = 0; i < H264D_DPB_SIZE; i++)
        if (d->dpb[i].used && d->dpb[i].surface == surface)
            return i;
    for (int i = 0; i < H264D_DPB_SIZE; i++)
        if (!d->dpb[i].used)
            return i;
    /* Everything is claimed: take the one the application has not listed as
     * a reference for the longest, which is the smallest POC. */
    int worst_one = 0;
    for (int i = 1; i < H264D_DPB_SIZE; i++)
        if (d->dpb[i].poc < d->dpb[worst_one].poc)
            worst_one = i;
    return worst_one;
}

void h264_decoder_set_references(h264_decoder_t *d, const uint32_t *refs,
                                 const int *pocs, const bool *long_term, int n)
{
    bool keep[H264D_DPB_SIZE] = { false };
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < H264D_DPB_SIZE; i++) {
            if (d->dpb[i].used && d->dpb[i].surface == refs[k]) {
                keep[i] = true;
                d->dpb[i].poc = pocs[k];
                d->dpb[i].is_long_term = long_term[k];
            }
        }
    }
    if (d->cur >= 0) keep[d->cur] = true;
    for (int i = 0; i < H264D_DPB_SIZE; i++)
        if (d->dpb[i].used && !keep[i])
            d->dpb[i].used = false;
}

/* ------------------------------------------------------------- picture */

int h264_decoder_begin_picture(h264_decoder_t *d, const h264d_pic_t *pic,
                               uint32_t surface, int poc, int frame_num)
{
    if (!d->pool) h264d_pool_start(d);

    d->pic = *pic;
    d->cur = slot_per(d, surface);
    h264d_frame_t *f = &d->dpb[d->cur];
    f->surface = surface;
    f->poc = poc;
    f->frame_num = frame_num;
    f->used = true;

    memset(d->mbs, 0, (size_t)d->mb_count * sizeof(h264d_mb_t));
    for (int i = 0; i < d->mb_count; i++) {
        memset(d->mbs[i].ref, -1, sizeof(d->mbs[i].ref));
        memset(d->mbs[i].ref_idx, -1, sizeof(d->mbs[i].ref_idx));
    }
    memset(d->slice_of_mb, 0xff, (size_t)d->mb_count);
    d->n_slices = 0;
    /* Rebuilt only when the scaling lists could have changed, which is
     * here: a new picture may carry a new picture parameter set. */
    h264d_dequant_build_all(d->dequant, d->pic.scaling4, d->pic.scaling8);
    return 0;
}

/* One line per macroblock when BC250_H264_TRACE is set. The single most
 * useful debugging tool this decoder has: a diff against the same line out
 * of a reference decoder says which syntax element went wrong, and usually
 * which macroblock it went wrong at. */
static void trace_mb(const h264_decoder_t *d)
{
    const h264d_mb_t *m = &d->mbs[d->mb_idx];
    fprintf(stderr, "mb %4d (%2d,%2d) type %d intra %d t8 %d cbp %02x qp %2d",
            d->mb_idx, d->mb_x, d->mb_y, m->type, m->intra,
            m->transform8x8, m->cbp, m->qpy);
    if (m->intra) {
        fprintf(stderr, " chroma %d modes", m->chroma_pred_mode);
        for (int k = 0; k < 16; k++) fprintf(stderr, " %d", m->ipred[k]);
    } else {
        fprintf(stderr, " sub %d %d %d %d", m->sub_type[0],
                m->sub_type[1], m->sub_type[2], m->sub_type[3]);
        fprintf(stderr, " r1");
        for (int p8 = 0; p8 < 4; p8++)
            fprintf(stderr, " %d", m->ref_idx[1][p8]);
        fprintf(stderr, " ref");
        for (int p8 = 0; p8 < 4; p8++)
            fprintf(stderr, " %d/%d", m->ref_idx[0][p8], m->ref[0][p8]);
        fprintf(stderr, " mv");
        for (int bb = 0; bb < 16; bb++)
            fprintf(stderr, " %d,%d", m->mv[0][bb][0], m->mv[0][bb][1]);
    }
    fprintf(stderr, "\n");
}

/* Reconstruct the rows the entropy decoder has left behind, once a band of
 * them has piled up. Called at the start of every macroblock row.
 *
 * ⚠️ This is what keeps the residual ring small enough to stay in cache,
 * and it is also what bounds how far apart the wavefront's threads can
 * get. The two pull in opposite directions. */
static inline void flush(h264_decoder_t *d, int number)
{
    const int band = d->band_rows * d->mb_w;
    if (d->mb_idx - d->to_reconstruct < band) return;
    h264d_reconstruct_range(d, d->to_reconstruct,
                            d->mb_idx - d->to_reconstruct, number);
    d->to_reconstruct = d->mb_idx;
}

/* One slice, on whatever cursor it is handed. `number` is the slice's
 * number within the picture, assigned by the caller before any of them
 * start so that the per-slice tables can be filled in serially. */
int h264d_decode_slice(h264_decoder_t *d, const h264d_slice_input_t *in,
                           int number)
{
    const h264d_slice_t *slice = &in->slice;
    const uint8_t *data = in->data;
    const size_t size = in->size;
    const int bit_offset = in->bit_offset;

    d->slice = *slice;
    if (getenv("BC250_H264_TRACE"))
        fprintf(stderr, "slice %d: deblk idc %d alpha %d beta %d, qp %d, "
                        "cabac_idc %d, type %d\n", number,
                slice->disable_deblocking_filter_idc, slice->alpha_c0_offset,
                slice->beta_offset, slice->qpy, slice->cabac_init_idc,
                slice->type);

    /* The slice data starts at a byte boundary once the CABAC alignment bits
     * are skipped, and the emulation prevention bytes have to come out
     * before the engine ever sees them. */
    d->cabac_mode = d->pic.entropy_coding_mode;

    /* ⚠️ The whole NAL is un-escaped, from its first byte, and the offset
     * travels with it. Un-escaping only the tail would miss a sequence that
     * straddles the boundary, and leaving the offset alone would put the
     * first macroblock in the wrong place as soon as a slice header
     * contains an escaped byte. */
    if (bit_offset < 0 || (size_t)bit_offset >= size * 8) return -1;
    size_t first_bit = (size_t)bit_offset;
    const size_t n = br_extract_rbsp_map(d->rbsp, d->rbsp_cap, data, size,
                                         &first_bit);
    if (first_bit >= n * 8) return -1;

    d->qpy = slice->qpy;
    d->last_qp_delta_nonzero = 0;
    d->to_reconstruct = slice->first_mb;
    d->mb_idx = slice->first_mb;
    if (d->mb_idx >= d->mb_count) return -1;
    d->mb_x = d->mb_idx % d->mb_w;
    d->mb_y = d->mb_idx / d->mb_w;

    if (d->cabac_mode) {
        /* cabac_alignment_one_bit: the arithmetic decoder loads bytes, so
         * it starts on one. */
        const size_t first_byte = (first_bit + 7) / 8;
        if (first_byte >= n) return -1;
        h264d_cabac_init(&d->cabac, d->rbsp + first_byte, n - first_byte,
                         slice->type == 2, slice->cabac_init_idc, slice->qpy);
    } else {
        /* CAVLC has no alignment element: slice_data() begins at the very
         * next bit. */
        br_init(&d->br, d->rbsp, n);
        br_skip(&d->br, (int)first_bit);
    }

    const char *trace = getenv("BC250_H264_TRACE");
    int count = 0;

#define ADVANCE() do {                              \
        d->mb_idx++;                                   \
        d->mb_x = d->mb_idx % d->mb_w;                 \
        d->mb_y = d->mb_idx / d->mb_w;                 \
        if (d->mb_x == 0) flush(d, number);          \
    } while (0)

    if (d->cabac_mode) {
        for (;;) {
            d->slice_of_mb[d->mb_idx] = (uint8_t)number;
            const int r = h264d_decode_mb_cabac(d);
            if (r) return r;
            count++;
            if (trace) trace_mb(d);

            if (h264d_cabac_terminate(&d->cabac))
                break;
            if (h264d_cabac_overrun(&d->cabac))
                return -4;

            ADVANCE();
            if (d->mb_idx >= d->mb_count)
                break;
        }
    } else {
        /* Clause 7.3.4. A run of skipped macroblocks is counted, not
         * flagged, and more_rbsp_data() ends the slice.
         *
         * ⚠️ The run is read before every coded macroblock of a P or B
         * slice, including when it is zero, and the standard only consults
         * more_rbsp_data() after a run that was not zero. Consulting it
         * unconditionally would be right in practice and wrong in
         * principle, so it is not done. */
        for (;;) {
            if (slice->type != 2) {
                int skips = (int)br_read_ue(&d->br);
                const int n_skips = skips;
                while (skips-- > 0) {
                    if (d->mb_idx >= d->mb_count) return -1;
                    d->slice_of_mb[d->mb_idx] = (uint8_t)number;
                    if (h264d_cavlc_skip(d)) return -5;
                    count++;
                    if (trace) trace_mb(d);
                    ADVANCE();
                }
                if (n_skips > 0 && !br_more_rbsp_data(&d->br))
                    break;
            }
            if (d->mb_idx >= d->mb_count) break;
            if (br_overrun(&d->br)) return -4;

            d->slice_of_mb[d->mb_idx] = (uint8_t)number;
            const int r = h264d_decode_mb_cavlc(d);
            if (r) return r;
            count++;
            if (trace) trace_mb(d);

            ADVANCE();
            if (d->mb_idx >= d->mb_count) break;
            if (!br_more_rbsp_data(&d->br)) break;
        }
        if (br_overrun(&d->br)) return -4;
    }
#undef ADVANCE

    if (trace)
        fprintf(stderr, "slice done: %d macroblocks, overrun %d\n",
                count, d->cabac_mode
                        ? (int)h264d_cabac_overrun(&d->cabac)
                        : (int)br_overrun(&d->br));

    /* Whatever is left of the last band. */
    {
        const int fine = slice->first_mb + count;
        if (fine > d->to_reconstruct)
            h264d_reconstruct_range(d, d->to_reconstruct,
                                    fine - d->to_reconstruct, number);
    }
    return 0;
}


int h264_decoder_slices(h264_decoder_t *d, const h264d_slice_input_t *in, int n)
{
    if (d->cur < 0) return -1;
    if (n <= 0) return 0;
    if (d->n_slices + n > H264D_MAX_SLICES) return -1;

    /* The numbers and the per-slice tables are filled in here, before
     * anything starts: a worker writes only its own entries, and nothing
     * has to be locked. */
    const int first = d->n_slices;
    for (int i = 0; i < n; i++) {
        const h264d_slice_t *s = &in[i].slice;
        d->slices[first + i] = *s;
        d->deblock[first + i].disable_idc =
            (int8_t)s->disable_deblocking_filter_idc;
        d->deblock[first + i].alpha_offset = (int8_t)s->alpha_c0_offset;
        d->deblock[first + i].beta_offset = (int8_t)s->beta_offset;
    }
    d->n_slices += n;

    if (n >= 2 && d->pool)
        return h264d_slices_pool(d, in, n, first);

    int first_error = 0;
    for (int i = 0; i < n; i++) {
        const int r = h264d_decode_slice(d, &in[i], first + i);
        if (r && !first_error) first_error = r;
    }
    return first_error;
}

int h264_decoder_slice(h264_decoder_t *d, const h264d_slice_t *slice,
                       const uint8_t *data, size_t size, int bit_offset)
{
    const h264d_slice_input_t one_pred = { *slice, data, size, bit_offset, 0 };
    return h264_decoder_slices(d, &one_pred, 1);
}

int h264_decoder_end_picture(h264_decoder_t *d, gpu_image_t out,
                             gpu_memory_t out_memory)
{
    if (d->cur < 0) return -1;
    h264d_frame_t *f = &d->dpb[d->cur];

    /* Hand the motion field over to the picture before the macroblock store
     * is reused by the next one. A direct macroblock of a later B picture
     * reads it, and by then this picture is only a reference. */
    bool solo_intra = true;
    for (int i = 0; i < d->mb_count; i++) {
        const h264d_mb_t *m = &d->mbs[i];
        int8_t *cr8 = f->col_ref + (size_t)i * 8;
        int16_t *cmv = f->col_mv + (size_t)i * 64;
        int32_t *cpc = f->col_poc + (size_t)i * 8;
        for (int l = 0; l < 2; l++) {
            /* ⚠️ The index, not the slot. colZeroFlag asks whether the
             * co-located block used index zero of ITS OWN picture's list,
             * which is a question about the index. */
            for (int p = 0; p < 4; p++) {
                cr8[l * 4 + p] = m->intra ? -1 : m->ref_idx[l][p];
                /* The slot is still valid here: this runs the moment the
                 * picture is decoded, before any of its references can be
                 * pushed out of the store. */
                const int slot = m->intra ? -1 : m->ref[l][p];
                cpc[l * 4 + p] = (slot >= 0 && slot < H264D_DPB_SIZE)
                               ? d->dpb[slot].poc : 0;
            }
            for (int b = 0; b < 16; b++) {
                cmv[(l * 16 + b) * 2 + 0] = m->intra ? 0 : m->mv[l][b][0];
                cmv[(l * 16 + b) * 2 + 1] = m->intra ? 0 : m->mv[l][b][1];
            }
        }
        if (!m->intra) solo_intra = false;
    }
    f->col_intra_only = solo_intra;

    /* Any macroblock no slice covered is left as it was allocated. Marking
     * them as belonging to no slice keeps the deblocking filter from
     * treating them as part of their neighbour's. */
    if (getenv("BC250_H264_TRACE"))
        fprintf(stderr, "picture done, slot %d poc %d: row 0 before "
                        "%d %d %d %d\n", d->cur, f->poc,
                f->y[0], f->y[1], f->y[2], f->y[3]);

    {
        const h264d_deblock_pic_t dp = {
            f->y, f->cb, f->cr, f->stride_y, f->stride_c,
            d->mb_w, d->mb_h, d->mbs, d->slice_of_mb, d->deblock,
            d->pic.chroma_qp_index_offset,
            d->pic.second_chroma_qp_index_offset
        };
        h264d_deblock_wavefront(d, &dp);
    }

    if (getenv("BC250_H264_TRACE"))
        fprintf(stderr, "                          row 0 after  "
                        "%d %d %d %d\n", f->y[0], f->y[1], f->y[2], f->y[3]);

    if (!d->gpu)
        return 0;            /* the standalone harness keeps the planes */

    /* The surface wants NV12, so the two chroma planes are interleaved on
     * the way out. Written once, at the end: surface memory is
     * write-combining, which is fast to write and very slow to read. */
    const int cw = d->width / 2, ch = d->height / 2;
    const size_t uv_needed = (size_t)cw * 2 * ch;
    if (d->uv_cap < uv_needed) {
        uint8_t *new_uv = realloc(d->uv_buf, uv_needed);
        if (!new_uv) return -1;
        d->uv_buf = new_uv;
        d->uv_cap = uv_needed;
    }
    uint8_t *uv = d->uv_buf;

    for (int y = 0; y < ch; y++) {
        const uint8_t *a = f->cb + (size_t)y * f->stride_c;
        const uint8_t *b = f->cr + (size_t)y * f->stride_c;
        uint8_t *o = uv + (size_t)y * cw * 2;
        int x = 0;
#if BC250_H264_SSE2
        for (; x + 15 < cw; x += 16) {
            __m128i cb_vec = _mm_loadu_si128((const __m128i *)(a + x));
            __m128i cr_vec = _mm_loadu_si128((const __m128i *)(b + x));
            __m128i uv_lo = _mm_unpacklo_epi8(cb_vec, cr_vec);
            __m128i uv_hi = _mm_unpackhi_epi8(cb_vec, cr_vec);
            _mm_storeu_si128((__m128i *)(o + 2 * x), uv_lo);
            _mm_storeu_si128((__m128i *)(o + 2 * x + 16), uv_hi);
        }
#endif
        for (; x < cw; x++) {
            o[2 * x] = a[x];
            o[2 * x + 1] = b[x];
        }
    }
    return gpu_compute_upload_nv12(d->gpu, &out, out_memory,
                                  f->y, f->stride_y,
                                  uv, cw * 2,
                                  d->width, d->height);
}
