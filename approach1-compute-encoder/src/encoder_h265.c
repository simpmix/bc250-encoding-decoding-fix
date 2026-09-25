/* bc250-encoding-decoding-fix v0.4.3 - https://github.com/simpmix/bc250-encoding-decoding-fix */
/*
 * Copyright (c) 2026 BC-250 Project Contributors
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * encoder_h265.c - H.265/HEVC encoder supporting IDR (I-slices) and
 *                  inter-frame prediction (P-slices with zero-motion CU skip).
 *
 * ============================================================================
 * DESIGN, in one place (see docs/hevc_scope_note.md and DEVLOG.md Sec. 6/8/27
 * for the history of why this replaced a non-functional stub)
 * ============================================================================
 *
 * This encoder supports GOP structures with periodic/forced IDR frames and
 * inter-predicted P-frames. For P-frames:
 *   - Reference picture set (RPS) is configured with DeltaPOC = -1 pointing
 *     to the previous reconstructed frame in the DPB.
 *   - NAL unit type is NAL_UNIT_CODED_SLICE_TRAIL_R (1) with 4-bit POC LSB.
 *   - Each 8x8 CU evaluates temporal difference against the reference picture.
 *     Static / low-motion blocks are coded as SKIP CUs (cu_skip_flag = 1,
 *     merge_idx = 0) with zero residual and zero motion vector, yielding
 *     immense bitrate reduction on typical desktop / streaming video.
 *   - Dynamic blocks are coded with cu_skip_flag = 0 and pred_mode_flag = 1
 *     (MODE_INTRA) falling back to full intra prediction and transform coding.
 *
 * Picture structure, chosen to keep every stage genuinely simple AND
 * genuinely spec-correct at the same time (see hevc_intra.h's top comment
 * for why this is NOT a case of reusing the existing GPU DCT/quantize
 * shaders - HEVC's mandatory 4x4-luma-intra DST-VII transform and its own
 * QP-to-quantizer-step mapping make that numerically wrong, not just a
 * block-size mismatch):
 *
 *   - CTU size = 16x16 (the minimum ITU-T H.265 allows - CtbLog2SizeY must
 *     be 4..6). One split_cu_flag=1 per CTU (always forced - condL/condA
 *     context still real, computed from real neighbor availability), giving
 *     exactly four 8x8 CUs per CTU, in z-order (TL,TR,BL,BR).
 *   - Every CU is intra, PartMode=PART_NxN (legal only at minimum CU size,
 *     which 8x8 always is here) - four independent 4x4 luma PUs per CU, each
 *     with its own real intra_luma_pred_mode. PartMode=NxN makes
 *     IntraSplitFlag=1, which per 7.4.9.8 FORCES (infers, no bit spent) the
 *     transform tree to split once at trafoDepth==0, landing exactly on the
 *     four 4x4 luma PUs as their own leaf TUs - no separate transform-size
 *     decision needed anywhere in this encoder.
 *   - Chroma (4:2:0) is one 4x4 Cb + one 4x4 Cr block per CU (8x8 luma / 1
 *     chroma shift = 4x4 chroma, coded once at the CU's own transform-tree
 *     root per the spec's "chroma stops splitting at the 4x4 floor" rule -
 *     see encode_cu()'s comment).
 *   - Every 4x4 TU (luma AND chroma) is therefore always exactly one
 *     coefficient group - no sig_coeff_group_flag/coded_sub_block_flag
 *     complexity anywhere (see hevc_cabac.h's scope note).
 *
 * Real per-4x4-block intra prediction (Planar/DC/Horizontal/Vertical, the
 * same four candidates the GPU's own I16x16 SAD decision already knows how
 * to choose between, conceptually) with proper z-scan reconstruction
 * chaining, real DST-VII (luma) / DCT-II (chroma) transform + real HEVC
 * quantization, and real CABAC entropy coding are implemented in
 * hevc_intra.c and hevc_cabac.c respectively - see those files.
 *
 * The one piece of the existing GPU/Vulkan infrastructure this file DOES
 * reuse unmodified is gpu_compute_download_nv12() - the real, already-
 * uploaded picture is read back from the GPU surface into host memory once
 * per frame, exactly the way va_backend.c's own CPU-side surface access
 * (bc250_MapBuffer et al) already does, and the existing
 * gpu_compute_begin_picture/dispatch_encode/end_picture/sync() sequence is
 * still called first (with its result discarded) purely to preserve the
 * exact same Vulkan image layout transitions and fence/staging-buffer
 * bookkeeping the rest of this driver (va_backend.c's EndPicture) already
 * depends on - see that call site's comment below.
 */

#include "encoder_h265.h"
#include "bitstream.h"
#include "hevc_cabac.h"
#include "hevc_intra.h"
#include "hevc_inter.h"
#include "dynamic_governor.h"
#include "cpu_simd_me.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sched.h>
#include <stdatomic.h>
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__SSE2__) || defined(__x86_64__) || defined(_M_X64)
#include <emmintrin.h>
#endif

#define NAL_UNIT_CODED_SLICE_TRAIL_R     1
#define NAL_UNIT_CODED_SLICE_IDR_W_RADL 19
#define NAL_UNIT_VPS               32
#define NAL_UNIT_SPS               33
#define NAL_UNIT_PPS               34
#define NAL_UNIT_AUD               35

#define HEVC_MAX_SLICES 16
#define HEVC_CTU_SIZE 16
#define HEVC_CU_SIZE   8
#define HEVC_PU_SIZE   4
/* Samples of repeated edge around the search planes - see build_hpel(). */
#define HPEL_MARGIN   16
/* Rows a slice's entry point list is sized for: 8192 samples of CTU rows. */
#define HEVC_WPP_MAX_ROWS 512

/* ============================================================================
 * Level selection (Annex A.3 MaxLumaPs table, picture-size-only heuristic -
 * a real encoder would also check bitrate/CPB constraints; this project's
 * one-QP-for-the-whole-stream design has no rate-control loop to check
 * against, so this picks the smallest level whose MaxLumaPs covers the
 * picture, which is what every simple/embedded HEVC encoder does in
 * practice for a "just make it playable" level tag).
 * ==========================================================================*/
static int hevc_pick_level_idc(uint32_t width, uint32_t height) {
    static const struct { uint64_t max_luma_ps; int level_idc; } table[] = {
        { 36864UL,        30 },
        { 122880UL,       60 },
        { 245760UL,       63 },
        { 552960UL,       90 },
        { 983040UL,       93 },
        { 2228224UL,     120 },
        { 2228224UL,     123 },
        { 8912896UL,     150 },
        { 8912896UL,     153 },
        { 8912896UL,     156 },
        { 35651584UL,    180 },
        { 35651584UL,    183 },
        { 35651584UL,    186 },
    };
    uint64_t pic_size = (uint64_t)width * (uint64_t)height;
    for (size_t i = 0; i < sizeof(table) / sizeof(table[0]); i++)
        if (pic_size <= table[i].max_luma_ps) return table[i].level_idc;
    return 186;
}

/* ============================================================================
 * VPS / SPS / PPS
 * ==========================================================================*/

/* Main (1) at eight bits, Main 10 (2) at ten - Annex A.3.2/A.3.3. A Main 10
 * stream claims only Main 10: its samples do not fit a Main decoder. */
static void write_profile_tier_level(bitstream_t *bs, int level_idc, int bit_depth) {
    const int profile_idc = bit_depth > 8 ? 2 : 1;
    bs_write_u(bs, 2, 0);   /* general_profile_space */
    bs_write1(bs, 0);       /* general_tier_flag (Main tier) */
    bs_write_u(bs, 5, profile_idc); /* general_profile_idc */
    for (int i = 0; i < 32; i++)
        bs_write1(bs, i == profile_idc ? 1 : 0); /* general_profile_compatibility_flag[] */
    bs_write1(bs, 1); /* general_progressive_source_flag */
    bs_write1(bs, 0); /* general_interlaced_source_flag */
    bs_write1(bs, 0); /* general_non_packed_constraint_flag */
    bs_write1(bs, 1); /* general_frame_only_constraint_flag */
    bs_write_u(bs, 16, 0);
    bs_write_u(bs, 16, 0);
    bs_write_u(bs, 12, 0); /* reserved_zero_44bits */
    bs_write_u(bs, 8, level_idc);
}

static size_t write_aud_hevc(uint8_t *buf, size_t buf_size, bool is_idr) {
    uint8_t rbsp[8];
    bitstream_t bs;
    bs_init(&bs, rbsp, sizeof(rbsp));
    /* pic_type: 0 for I-slices only, 1 for P and I slices */
    bs_write_u(&bs, 3, is_idr ? 0 : 1);
    bs_rbsp_trailing_bits(&bs);

    bitstream_t out_bs;
    bs_init(&out_bs, buf, buf_size);
    bs_write_nal_header_hevc(&out_bs, NAL_UNIT_AUD);
    size_t off = bs_bytes_written(&out_bs);
    if (off >= buf_size) return 0;
    return off + bs_rbsp_to_ebsp(buf + off, buf_size - off, rbsp, bs_bytes_written(&bs));
}

static size_t write_vps(uint8_t *buf, size_t buf_size, int bit_depth) {
    uint8_t rbsp[128];
    bitstream_t bs;
    bs_init(&bs, rbsp, sizeof(rbsp));

    bs_write_u(&bs, 4, 0);   /* vps_video_parameter_set_id */
    bs_write_u(&bs, 2, 3);   /* vps_base_layer_internal/available_flag */
    bs_write_u(&bs, 6, 0);   /* vps_max_layers_minus1 */
    bs_write_u(&bs, 3, 0);   /* vps_max_sub_layers_minus1 */
    bs_write1(&bs, 1);       /* vps_temporal_id_nesting_flag */
    bs_write_u(&bs, 16, 0xffff);

    write_profile_tier_level(&bs, 30, bit_depth); /* level is irrelevant here; SPS carries the real one */

    bs_write1(&bs, 1); /* vps_sub_layer_ordering_info_present_flag */
    bs_write_ue(&bs, 1); /* vps_max_dec_pic_buffering_minus1 = 1 (1 ref + 1 current pic) */
    bs_write_ue(&bs, 0); /* vps_num_reorder_pics */
    bs_write_ue(&bs, 0); /* vps_max_latency_increase_plus1 */

    bs_write_u(&bs, 6, 0); /* vps_max_nuh_reserved_zero_layer_id */
    bs_write_ue(&bs, 0);   /* vps_max_op_sets_minus1 */
    bs_write1(&bs, 0);     /* vps_timing_info_present_flag */
    bs_write1(&bs, 0);     /* vps_extension_flag */

    bs_rbsp_trailing_bits(&bs);

    bitstream_t out_bs;
    bs_init(&out_bs, buf, buf_size);
    bs_write_nal_header_hevc(&out_bs, NAL_UNIT_VPS);
    size_t off = bs_bytes_written(&out_bs);
    if (off >= buf_size) return 0;
    return off + bs_rbsp_to_ebsp(buf + off, buf_size - off, rbsp, bs_bytes_written(&bs));
}

static size_t write_sps(uint8_t *buf, size_t buf_size, uint32_t coded_w, uint32_t coded_h,
                         uint32_t real_w, uint32_t real_h, int level_idc, int bit_depth, int tu8) {
    uint8_t rbsp[256];
    bitstream_t bs;
    bs_init(&bs, rbsp, sizeof(rbsp));

    bs_write_u(&bs, 4, 0);  /* sps_video_parameter_set_id */
    bs_write_u(&bs, 3, 0);  /* sps_max_sub_layers_minus1 */
    bs_write1(&bs, 1);      /* sps_temporal_id_nesting_flag */

    write_profile_tier_level(&bs, level_idc, bit_depth);

    bs_write_ue(&bs, 0); /* sps_seq_parameter_set_id */
    bs_write_ue(&bs, 1); /* chroma_format_idc = 1 (4:2:0) */

    bs_write_ue(&bs, coded_w);
    bs_write_ue(&bs, coded_h);

    int need_crop = (coded_w != real_w) || (coded_h != real_h);
    bs_write1(&bs, need_crop ? 1 : 0);
    if (need_crop) {
        bs_write_ue(&bs, 0);
        bs_write_ue(&bs, (coded_w - real_w) / 2);
        bs_write_ue(&bs, 0);
        bs_write_ue(&bs, (coded_h - real_h) / 2);
    }

    bs_write_ue(&bs, bit_depth - 8); /* bit_depth_luma_minus8 */
    bs_write_ue(&bs, bit_depth - 8); /* bit_depth_chroma_minus8 */
    bs_write_ue(&bs, 4); /* log2_max_pic_order_cnt_lsb_minus4 = 4 -> log2=8 (0..255) */

    bs_write1(&bs, 1); /* sps_sub_layer_ordering_info_present_flag */
    bs_write_ue(&bs, 1); /* sps_max_dec_pic_buffering_minus1 = 1 (1 ref + 1 current pic) */
    bs_write_ue(&bs, 0); /* sps_num_reorder_pics */
    bs_write_ue(&bs, 0); /* sps_max_latency_increase_plus1 */

    bs_write_ue(&bs, 0); /* log2_min_luma_coding_block_size_minus3 -> MinCb = 8 */
    bs_write_ue(&bs, 1); /* log2_diff_max_min_coding_block_size -> Ctb = 16 */
    bs_write_ue(&bs, 0); /* log2_min_luma_transform_block_size_minus2 -> MinTb = 4 */
    /* With 8x8 transforms, MaxTb is 8 and an inter CU's transform tree may
     * split once: split_transform_flag is coded and chooses. Intra NxN
     * splits to 4x4 regardless (IntraSplitFlag). */
    bs_write_ue(&bs, tu8 ? 1 : 0); /* log2_diff_max_min_transform_block_size */
    bs_write_ue(&bs, tu8 ? 1 : 0); /* max_transform_hierarchy_depth_inter */
    bs_write_ue(&bs, 0); /* max_transform_hierarchy_depth_intra (IntraSplitFlag adds +1 -> MaxTrafoDepth=1) */

    bs_write1(&bs, 0); /* scaling_list_enabled_flag */
    bs_write1(&bs, 0); /* amp_enabled_flag */
    bs_write1(&bs, 0); /* sample_adaptive_offset_enabled_flag */
    bs_write1(&bs, 0); /* pcm_enabled_flag */

    bs_write_ue(&bs, 1); /* num_short_term_ref_pic_sets = 1 */
    /* short_term_ref_pic_set(0) per Rec. ITU-T H.265 7.3.7 */
    bs_write_ue(&bs, 1); /* num_negative_pics = 1 */
    bs_write_ue(&bs, 0); /* num_positive_pics = 0 */
    bs_write_ue(&bs, 0); /* delta_poc_s0_minus1[0] = 0 -> DeltaPoc = -(0+1) = -1 */
    bs_write1(&bs, 1);   /* used_by_curr_pic_s0_flag[0] = 1 */

    bs_write1(&bs, 0);   /* long_term_ref_pics_present_flag */
    bs_write1(&bs, 0);   /* sps_temporal_mvp_enable_flag */
    bs_write1(&bs, 0);   /* sps_strong_intra_smoothing_enable_flag */
    bs_write1(&bs, 0);   /* vui_parameters_present_flag */
    bs_write1(&bs, 0);   /* sps_extension_present_flag */

    bs_rbsp_trailing_bits(&bs);

    bitstream_t out_bs;
    bs_init(&out_bs, buf, buf_size);
    bs_write_nal_header_hevc(&out_bs, NAL_UNIT_SPS);
    size_t off = bs_bytes_written(&out_bs);
    if (off >= buf_size) return 0;
    return off + bs_rbsp_to_ebsp(buf + off, buf_size - off, rbsp, bs_bytes_written(&bs));
}

static size_t write_pps(uint8_t *buf, size_t buf_size, int init_qp, int wpp) {
    uint8_t rbsp[64];
    bitstream_t bs;
    bs_init(&bs, rbsp, sizeof(rbsp));

    bs_write_ue(&bs, 0); /* pps_pic_parameter_set_id */
    bs_write_ue(&bs, 0); /* pps_seq_parameter_set_id */
    bs_write1(&bs, 0);   /* dependent_slice_segments_enabled_flag */
    bs_write1(&bs, 0);   /* output_flag_present_flag */
    bs_write_u(&bs, 3, 0); /* num_extra_slice_header_bits */
    bs_write1(&bs, 0);   /* sign_data_hiding_flag */
    bs_write1(&bs, 0);   /* cabac_init_present_flag */
    bs_write_ue(&bs, 0); /* num_ref_idx_l0_default_active_minus1 */
    bs_write_ue(&bs, 0); /* num_ref_idx_l1_default_active_minus1 */
    bs_write_se(&bs, init_qp - 26); /* init_qp_minus26 */
    bs_write1(&bs, 0);   /* constrained_intra_pred_flag */
    bs_write1(&bs, 0);   /* transform_skip_enabled_flag */
    bs_write1(&bs, 0);   /* cu_qp_delta_enabled_flag */
    bs_write_se(&bs, 0); /* pps_cb_qp_offset */
    bs_write_se(&bs, 0); /* pps_cr_qp_offset */
    bs_write1(&bs, 0);   /* pps_slice_chroma_qp_offsets_present_flag */
    bs_write1(&bs, 0);   /* weighted_pred_flag */
    bs_write1(&bs, 0);   /* weighted_bipred_flag */
    bs_write1(&bs, 0);   /* transquant_bypass_enable_flag */
    bs_write1(&bs, 0);   /* tiles_enabled_flag */
    bs_write1(&bs, wpp ? 1 : 0); /* entropy_coding_sync_enabled_flag */
    bs_write1(&bs, 0);   /* pps_loop_filter_across_slices_enabled_flag = 0 */
    bs_write1(&bs, 1);   /* deblocking_filter_control_present_flag = 1 (we need to disable deblock) */
    bs_write1(&bs, 0);   /* deblocking_filter_override_enabled_flag = 0 */
    bs_write1(&bs, 1);   /* pps_deblocking_filter_disabled_flag = 1 (encoder has no deblock filter;
                           * leaving this enabled causes reference-frame mismatch drift on P-frames
                           * because the decoder deblocks its reference but our encoder doesn't) */
    bs_write1(&bs, 0);   /* pps_scaling_list_data_present_flag */
    bs_write1(&bs, 0);   /* lists_modification_present_flag */
    bs_write_ue(&bs, 0); /* log2_parallel_merge_level_minus2 */
    bs_write1(&bs, 0);   /* slice_segment_header_extension_present_flag */
    bs_write1(&bs, 0);   /* pps_extension_present_flag */

    bs_rbsp_trailing_bits(&bs);

    bitstream_t out_bs;
    bs_init(&out_bs, buf, buf_size);
    bs_write_nal_header_hevc(&out_bs, NAL_UNIT_PPS);
    size_t off = bs_bytes_written(&out_bs);
    if (off >= buf_size) return 0;
    return off + bs_rbsp_to_ebsp(buf + off, buf_size - off, rbsp, bs_bytes_written(&bs));
}

/* ============================================================================
 * Encoder state
 * ==========================================================================*/

struct hevc_encoder {
    bc250_gpu_context_t *gpu;
    uint32_t width, height;               /* real: the picture being coded */
    uint32_t ctx_width, ctx_height;       /* what the context was opened for */
    uint32_t coded_width, coded_height;   /* rounded up to a 16px CTU multiple */
    uint32_t width_ctu, height_ctu;
    uint32_t fps, bitrate;
    uint32_t frame_count;
    uint32_t gop_size;
    uint32_t poc;
    bool     force_idr;
    bool     has_ref;
    int qp;
    int pps_init_qp;
    int qp_hint_applied;         /* Last QP explicitly handed to hevc_encoder_set_qp(), or -1 if never called yet */
    rate_control_t rc;
    bool cbr_intent;
    uint32_t quality_level;      /* 1..7 (1 = Quality, 4 = Balanced, 7 = Speed) */
    uint32_t max_frame_bits;     /* Maximum frame size in bits (0 = unlimited) */

    /* 8 or 10. It decides what a sample is in the planes below, the
     * profile the stream declares, and which copy of hevc_enc_template.c
     * encodes it. */
    int bit_depth;

    /* Source (post-download, padded/replicated to coded dimensions) and
     * reconstructed planes. Luma at coded_w x coded_h; chroma at
     * coded_w/2 x coded_h/2 (4:2:0). A sample is a uint8_t at eight bits
     * and a uint16_t at ten, which is why these are void: only the
     * template, which knows which, reads them. */
    void *src_y, *src_cb, *src_cr;
    void *recon_y, *recon_cb, *recon_cr;
    void *prev_recon_y, *prev_recon_cb, *prev_recon_cr;

    /* Per-CU skip tracking for current frame (for condL/condA context derivation).
     * Size: (width_ctu * 2) * (height_ctu * 2). */
    uint8_t *cu_skip_map;

    /* Inter prediction & motion vector maps (for spatial merge candidate derivation).
     * Size: (width_ctu * 2) * (height_ctu * 2). MVs in 1/4-pel units. */
    uint8_t *cu_is_inter;
    /* Quadtree depth of the CU covering each 8x8 cell: 0 for a whole-CTU
     * CU, 1 for an 8x8 one. split_cu_flag's context reads it. */
    uint8_t *cu_depth;
    /* Whole-CTU skips: on unless BC250_HEVC_CU16=0. */
    bool cu16;
    /* 8x8 transforms for inter CUs: on unless BC250_HEVC_TU8=0. */
    int tu8;
    /* The quantizer's dead zone: on unless BC250_HEVC_DEADZONE=0, which
     * rounds every level to nearest. The offsets, in twelfths of a step
     * (hevc_intra.h), are set per picture by encode_core(). */
    bool deadzone;
    int quant_round_inter, quant_round_intra;
    /* A CU whose merge residual quantizes to nothing is a skip, decided
     * there: on unless BC250_HEVC_EARLY_SKIP=0. */
    bool early_skip;
    int16_t *mv_x_map;
    int16_t *mv_y_map;
    uint32_t last_frame_sad;

    /* GPU compute motion vector readback for acceleration */
    gpu_mv_t *gpu_mvs;
    uint32_t num_gpu_mvs;

    /* Real per-4x4-luma-PU intra mode, for MPM derivation - one entry per
     * 4x4 position, persistent scratch (positional availability checks
     * gate every read, so stale cross-frame content is never read - see
     * hevc_derive_mpm() call sites below). */
    int8_t *luma_mode_map;
    uint32_t mode_map_stride;

    /* The previous picture's luma at whole and half samples, for the
     * motion search only - see build_hpel() in hevc_enc_template.c. */
    void *hpel[4];
    int32_t *hpel_tmp;
    int hpel_stride;

    /* Wavefront parallel processing: one substream per CTU row, see
     * encode_wpp_row(). On unless BC250_HEVC_WPP=0. */
    bool wpp;
    uint8_t **row_buf;
    size_t row_buf_cap;
    size_t *row_len;
    uint32_t *row_sad;
    uint8_t (*row_ctx)[HEVC_NUM_CTX];
    atomic_int *row_progress;

    /* This frame's Lagrange multipliers - see lambda_sse_q8(). */
    int64_t lambda_sse_q8;
    int lambda_sad_q8;


    /* Raw NV12 download scratch, real width x height - P010 at ten bits,
     * so two bytes a sample. */
    uint8_t *dl_y;
    uint8_t *dl_uv;

    uint8_t *slice_rbsp;
    size_t   slice_rbsp_cap;

    /* One slice per thread's worth of state. A slice is a whole number of CTU
     * rows, which is what makes a slice boundary also a CTU-row boundary - the
     * MPM derivation already refuses to look above a CTU row, so that part was
     * slice-safe before this existed. */
    int      num_slices;
    uint8_t *slice_buf[HEVC_MAX_SLICES];
    size_t   slice_len[HEVC_MAX_SLICES];
    uint32_t slice_sad[HEVC_MAX_SLICES];

    size_t   slice_buf_cap;
    uint8_t *scratch_out;
    size_t   scratch_out_cap;

    /* Dynamic asymmetric CPU/GPU load balancing governor & SIMD ME config */
    dynamic_governor_t governor;
    /* Frames the governor has told us to keep off the GPU, counted so one
     * in every step_down_hysteresis can go anyway and bring back a
     * measurement. See hevc_encoder_encode_frame(). */
    uint32_t governor_skips;
    cpu_simd_me_config_t me_cfg;
};

static uint32_t round_up16(uint32_t v) { return (v + 15u) & ~15u; }

hevc_encoder_t *hevc_encoder_create(bc250_gpu_context_t *gpu_ctx,
                                    uint32_t width, uint32_t height,
                                    uint32_t fps, uint32_t bitrate)
{
    return hevc_encoder_create_depth(gpu_ctx, width, height, fps, bitrate, 8);
}

hevc_encoder_t *hevc_encoder_create_depth(bc250_gpu_context_t *gpu_ctx,
                                          uint32_t width, uint32_t height,
                                          uint32_t fps, uint32_t bitrate,
                                          int bit_depth)
{
    if (width == 0 || height == 0) return NULL;
    if (bit_depth != 8 && bit_depth != 10) return NULL;
    hevc_encoder_t *enc = calloc(1, sizeof(hevc_encoder_t));
    if (!enc) return NULL;

    enc->gpu = gpu_ctx;
    enc->ctx_width = width;
    enc->ctx_height = height;
    enc->bit_depth = bit_depth;
    enc->width = width;
    enc->height = height;
    enc->fps = fps ? fps : 30;
    enc->bitrate = bitrate;
    enc->qp = 27;
    bool qp_pinned = false;
    {
        const char *qp_env = getenv("BC250_HEVC_QP");
        if (qp_env) {
            int q = atoi(qp_env);
            if (q >= 1 && q <= 51) { enc->qp = q; qp_pinned = true; }
        }
    }
    enc->pps_init_qp = enc->qp;
    /* The rate control was built and then switched off.
     *
     * encode_core() already asks rc_get_frame_qp() for this frame's QP, the
     * tail already calls rc_update_stats() with the bits produced, and the
     * slice header already writes slice_qp_delta so a per-frame QP reaches
     * the decoder. All three are guarded by `rc.mode != RC_CQP`, and this
     * call passed RC_CQP - so none of them ever ran: every frame went out at
     * QP 27 no matter what bitrate the caller asked for, and asking for more
     * bitrate changed nothing.
     *
     * Measured on a BC-250 at 1920x1080, testsrc, 8 Mbit/s requested: the
     * stream came out at 1.5 Mbit/s and 25.3 dB PSNR, where h264_vaapi on the
     * same clip and the same request gives 51.4 dB. The H.264 path defaults
     * to RC_LOW_LATENCY (see its rc_init() call), so HEVC now does the same.
     *
     * BC250_HEVC_QP still pins the QP: somebody who names a QP is asking for
     * constant QP, and that is what RC_CQP is for.
     */
    rc_init(&enc->rc, qp_pinned ? RC_CQP : RC_LOW_LATENCY, bitrate,
            (double)enc->fps, width, height);
    /* QP from a model of what pictures cost (rate_control.c), unless
     * BC250_HEVC_RC_MODEL=0 asks for the old buffer-feedback loop. */
    {
        const char *m = getenv("BC250_HEVC_RC_MODEL");
        enc->rc.model = !(m && strcmp(m, "0") == 0);
    }
    if (qp_pinned) {
        enc->rc.current_qp = enc->qp;
        enc->rc.base_qp = enc->qp;
    } else {
        /* HEVC achieves target bitrates at roughly ~3 to 5 QP higher than H.264
         * for the same content complexity. Offset initial base_qp so the first
         * GOP doesn't start severely over-quantized at QP 12. */
        if (enc->rc.base_qp + 4 <= enc->rc.qp_max) {
            enc->rc.base_qp += 4;
            enc->rc.current_qp = enc->rc.base_qp;
        }
    }
    enc->pps_init_qp = enc->qp;
    enc->cbr_intent = false;
    enc->qp_hint_applied = -1; /* no explicit QP hint applied yet - see hevc_encoder_set_qp() */
    enc->quality_level = 4;
    enc->max_frame_bits = 0;

    enc->gop_size = enc->fps;
    {
        const char *gop_env = getenv("BC250_HEVC_GOP");
        if (gop_env) {
            int g = atoi(gop_env);
            if (g >= 1) enc->gop_size = (uint32_t)g;
        }
    }
    enc->poc = 0;
    enc->force_idr = false;
    enc->has_ref = false;

    enc->coded_width = round_up16(width);
    enc->coded_height = round_up16(height);
    enc->width_ctu = enc->coded_width / HEVC_CTU_SIZE;
    enc->height_ctu = enc->coded_height / HEVC_CTU_SIZE;

    /* Bytes, not samples: every buffer below that holds samples is sized
     * with this. */
    const size_t bps = bit_depth > 8 ? 2 : 1;
    size_t luma_size = (size_t)enc->coded_width * enc->coded_height;
    size_t chroma_size = (size_t)(enc->coded_width / 2) * (enc->coded_height / 2);
    luma_size *= bps;
    chroma_size *= bps;

    enc->src_y = malloc(luma_size);
    enc->src_cb = malloc(chroma_size);
    enc->src_cr = malloc(chroma_size);
    enc->recon_y = malloc(luma_size);
    enc->recon_cb = malloc(chroma_size);
    enc->recon_cr = malloc(chroma_size);
    enc->prev_recon_y = malloc(luma_size);
    enc->prev_recon_cb = malloc(chroma_size);
    enc->prev_recon_cr = malloc(chroma_size);

    size_t num_cus = (size_t)(enc->width_ctu * 2) * (enc->height_ctu * 2);
    enc->cu_skip_map = calloc(num_cus, 1);
    enc->cu_is_inter = calloc(num_cus, 1);
    enc->cu_depth = calloc(num_cus, 1);
    {
        const char *e = getenv("BC250_HEVC_CU16");
        enc->cu16 = !(e && strcmp(e, "0") == 0);
        e = getenv("BC250_HEVC_TU8");
        enc->tu8 = !(e && strcmp(e, "0") == 0);
        e = getenv("BC250_HEVC_DEADZONE");
        enc->deadzone = !(e && strcmp(e, "0") == 0);
        e = getenv("BC250_HEVC_EARLY_SKIP");
        enc->early_skip = !(e && strcmp(e, "0") == 0);
    }
    enc->mv_x_map = calloc(num_cus, sizeof(int16_t));
    enc->mv_y_map = calloc(num_cus, sizeof(int16_t));

    enc->mode_map_stride = enc->coded_width / HEVC_PU_SIZE;
    enc->luma_mode_map = malloc((size_t)enc->mode_map_stride * (enc->coded_height / HEVC_PU_SIZE));

    enc->row_buf_cap = (size_t)enc->coded_width * HEVC_CTU_SIZE * bps * 3 + 65536;
    enc->row_buf = calloc(enc->height_ctu, sizeof(*enc->row_buf));
    enc->row_len = calloc(enc->height_ctu, sizeof(*enc->row_len));
    enc->row_sad = calloc(enc->height_ctu, sizeof(*enc->row_sad));
    enc->row_ctx = calloc(enc->height_ctu, sizeof(*enc->row_ctx));
    enc->row_progress = calloc(enc->height_ctu, sizeof(*enc->row_progress));
    if (enc->row_buf)
        for (uint32_t r = 0; r < enc->height_ctu; r++) enc->row_buf[r] = malloc(enc->row_buf_cap);
    enc->hpel_stride = (int)enc->coded_width + 2 * HPEL_MARGIN;
    {
        const size_t rows = enc->coded_height + 2 * HPEL_MARGIN;
        for (int i = 0; i < 4; i++) enc->hpel[i] = malloc((size_t)enc->hpel_stride * rows * bps);
        enc->hpel_tmp = malloc((size_t)enc->hpel_stride * (rows + 7) * sizeof(int32_t));
    }
    enc->dl_y = malloc((size_t)width * height * bps);
    enc->dl_uv = malloc((size_t)(width / 2) * (height / 2) * 2 * bps);

    enc->slice_rbsp_cap = luma_size + 65536;
    enc->slice_rbsp = malloc(enc->slice_rbsp_cap);

    /* Four by default, measured rather than guessed.
     *
     * On a BC-250 at 1920x1080 at 60 fps, testsrc, 8 Mbit/s requested:
     *    1 slice    78.2 fps   46.41 dB   4.09 Mbit/s
     *    4 slices  111.2 fps   45.83 dB   3.85 Mbit/s
     *    8 slices  118.9 fps   45.63 dB   3.78 Mbit/s
     *   16 slices  138.7 fps   44.74 dB   3.31 Mbit/s
     * Prediction restarts at every slice boundary, so more slices cost
     * quality. Four buys 42% more speed for half a dB, which on a box whose
     * job is streaming a game while the game is running is a trade worth
     * making; sixteen gives up too much for the rest. BC250_HEVC_SLICES
     * overrides it, and 1 restores exactly the old single-slice bitstream. */
    {
        const char *e = getenv("BC250_HEVC_WPP");
        enc->wpp = !(e && strcmp(e, "0") == 0);
    }
    /* With the wavefront the rows run in parallel without cutting the
     * picture into slices, and every slice boundary is prediction lost. */
    enc->num_slices = enc->wpp ? 1 : 4;
    if (enc->num_slices > (int)enc->height_ctu) enc->num_slices = (int)enc->height_ctu;
    {
        const char *s = getenv("BC250_HEVC_SLICES");
        if (s) {
            int n = atoi(s);
            if (n >= 1 && n <= HEVC_MAX_SLICES) enc->num_slices = n;
            if (enc->num_slices > (int)enc->height_ctu) enc->num_slices = (int)enc->height_ctu;
        }
    }
    {
        size_t per = luma_size / (size_t)enc->num_slices + 262144;
        for (int i = 0; i < enc->num_slices; i++) enc->slice_buf[i] = malloc(per);
        enc->slice_buf_cap = per;
    }

    enc->scratch_out_cap = luma_size + 131072;
    enc->scratch_out = malloc(enc->scratch_out_cap);

    size_t num_mbs = (size_t)enc->width_ctu * enc->height_ctu;
    enc->gpu_mvs = calloc(num_mbs, sizeof(gpu_mv_t));

    if (!enc->src_y || !enc->src_cb || !enc->src_cr ||
        !enc->recon_y || !enc->recon_cb || !enc->recon_cr ||
        !enc->prev_recon_y || !enc->prev_recon_cb || !enc->prev_recon_cr ||
        !enc->cu_skip_map || !enc->cu_is_inter || !enc->cu_depth || !enc->mv_x_map || !enc->mv_y_map ||
        !enc->luma_mode_map || !enc->dl_y || !enc->dl_uv ||
        !enc->slice_rbsp || !enc->scratch_out || !enc->gpu_mvs ||
        !enc->hpel[0] || !enc->hpel[1] || !enc->hpel[2] || !enc->hpel[3] || !enc->hpel_tmp ||
        !enc->row_buf || !enc->row_len || !enc->row_sad || !enc->row_ctx || !enc->row_progress ||
        !enc->row_buf[enc->height_ctu - 1]) {
        hevc_encoder_destroy(enc);
        return NULL;
    }

    dynamic_governor_init(&enc->governor);
    cpu_simd_me_config_init(&enc->me_cfg, width, height);

    return enc;
}

void hevc_encoder_set_force_idr(hevc_encoder_t *encoder)
{
    if (encoder) encoder->force_idr = true;
}

void hevc_encoder_set_gop_size(hevc_encoder_t *encoder, uint32_t gop_size)
{
    if (encoder && gop_size >= 1) encoder->gop_size = gop_size;
}

uint32_t hevc_encoder_get_gop_size(const hevc_encoder_t *encoder)
{
    return encoder ? encoder->gop_size : 30;
}

void hevc_encoder_set_qp(hevc_encoder_t *encoder, int qp)
{
    if (encoder) {
        if (qp < 0) qp = 0;
        if (qp > 51) qp = 51;
        if (qp != encoder->qp_hint_applied) {
            encoder->rc.base_qp = qp;
            encoder->rc.current_qp = qp;
            encoder->qp_hint_applied = qp;
        }
        encoder->qp = qp;
    }
}

void hevc_encoder_set_cbr_intent(hevc_encoder_t *encoder, bool cbr_intent)
{
    if (encoder) {
        encoder->cbr_intent = cbr_intent;
    }
}

int hevc_encoder_get_qp(const hevc_encoder_t *encoder)
{
    return encoder ? encoder->qp : 27;
}

void hevc_encoder_set_bitrate(hevc_encoder_t *encoder, uint32_t bitrate)
{
    if (encoder && bitrate > 0 && bitrate != encoder->rc.target_bitrate) {
        encoder->bitrate = bitrate;
        rc_init(&encoder->rc, encoder->rc.mode, bitrate, (double)encoder->fps,
                encoder->width, encoder->height);
        if (encoder->rc.mode != RC_CQP && encoder->rc.base_qp + 4 <= encoder->rc.qp_max) {
            encoder->rc.base_qp += 4;
            encoder->rc.current_qp = encoder->rc.base_qp;
        }
    }
}

uint32_t hevc_encoder_get_bitrate(const hevc_encoder_t *encoder)
{
    return encoder ? encoder->bitrate : 0;
}

void hevc_encoder_set_fps(hevc_encoder_t *encoder, uint32_t fps)
{
    if (encoder && fps > 0 && fps != encoder->fps) {
        encoder->fps = fps;
        rc_init(&encoder->rc, encoder->rc.mode, encoder->rc.target_bitrate,
                (double)fps, encoder->width, encoder->height);
        if (encoder->rc.mode != RC_CQP && encoder->rc.base_qp + 4 <= encoder->rc.qp_max) {
            encoder->rc.base_qp += 4;
            encoder->rc.current_qp = encoder->rc.base_qp;
        }
    }
}

uint32_t hevc_encoder_get_fps(const hevc_encoder_t *encoder)
{
    return encoder ? encoder->fps : 30;
}

void hevc_encoder_set_rc_mode(hevc_encoder_t *encoder, rc_mode_t mode)
{
    if (encoder) {
        encoder->rc.mode = mode;
        if (mode == RC_LOW_LATENCY) {
            encoder->rc.buffer_size = encoder->rc.target_bits_per_frame * 2;
        } else if (mode == RC_CBR || mode == RC_VBR) {
            encoder->rc.buffer_size = encoder->rc.target_bitrate;
        }
        if (encoder->rc.buffer_size < 1000) encoder->rc.buffer_size = 1000;
        encoder->rc.buffer_fullness = encoder->rc.buffer_size / 2;
        encoder->rc.error_integral = 0;
    }
}

rc_mode_t hevc_encoder_get_rc_mode(const hevc_encoder_t *encoder)
{
    return encoder ? encoder->rc.mode : RC_CQP;
}

uint32_t hevc_encoder_get_last_frame_sad(const hevc_encoder_t *encoder)
{
    return encoder ? encoder->last_frame_sad : 0;
}

void hevc_encoder_set_quality_level(hevc_encoder_t *encoder, uint32_t quality_level)
{
    if (encoder) {
        if (quality_level < 1) quality_level = 1;
        if (quality_level > 7) quality_level = 7;
        encoder->quality_level = quality_level;
        rc_set_quality_level(&encoder->rc, quality_level);
    }
}

uint32_t hevc_encoder_get_quality_level(const hevc_encoder_t *encoder)
{
    return encoder ? encoder->quality_level : 4;
}

void hevc_encoder_set_max_frame_size(hevc_encoder_t *encoder, uint32_t max_frame_bits)
{
    if (encoder) {
        encoder->max_frame_bits = max_frame_bits;
        rc_set_max_frame_size(&encoder->rc, max_frame_bits);
    }
}

uint32_t hevc_encoder_get_max_frame_size(const hevc_encoder_t *encoder)
{
    return encoder ? encoder->max_frame_bits : 0;
}

int hevc_encoder_get_bit_depth(const hevc_encoder_t *encoder)
{
    return encoder ? encoder->bit_depth : 8;
}

int hevc_encoder_get_governor_tier(const hevc_encoder_t *encoder)
{
    return encoder ? (int)dynamic_governor_get_tier(&encoder->governor) : 0;
}

void hevc_encoder_destroy(hevc_encoder_t *encoder)
{
    if (!encoder) return;
    free(encoder->src_y); free(encoder->src_cb); free(encoder->src_cr);
    free(encoder->recon_y); free(encoder->recon_cb); free(encoder->recon_cr);
    free(encoder->prev_recon_y); free(encoder->prev_recon_cb); free(encoder->prev_recon_cr);
    free(encoder->cu_skip_map);
    free(encoder->cu_is_inter);
    free(encoder->cu_depth);
    free(encoder->mv_x_map);
    free(encoder->mv_y_map);
    free(encoder->luma_mode_map);
    free(encoder->dl_y); free(encoder->dl_uv);
    for (int i = 0; i < encoder->num_slices; i++) free(encoder->slice_buf[i]);
    free(encoder->slice_rbsp);
    free(encoder->scratch_out);
    free(encoder->gpu_mvs);
    for (int i = 0; i < 4; i++) free(encoder->hpel[i]);
    if (encoder->row_buf)
        for (uint32_t r = 0; r < encoder->height_ctu; r++) free(encoder->row_buf[r]);
    free(encoder->row_buf); free(encoder->row_len); free(encoder->row_sad);
    free(encoder->row_ctx); free((void *)encoder->row_progress);
    free(encoder->hpel_tmp);
    free(encoder);
}

/* ============================================================================
 * Per-CU encoding
 * ==========================================================================*/

static const int pu_off_x[4] = { 0, 4, 0, 4 };
static const int pu_off_y[4] = { 0, 0, 4, 4 };

static int any_nonzero16(const int16_t *c) {
    for (int i = 0; i < 16; i++) if (c[i]) return 1;
    return 0;
}

typedef struct {
    int16_t x;
    int16_t y;
} hevc_mv_t;

static inline uint32_t hevc_cu_rank(uint32_t width_ctu, int cux, int cuy) {
    uint32_t ctu_col = (uint32_t)cux / 2;
    uint32_t ctu_row = (uint32_t)cuy / 2;
    uint32_t cu_sub = ((uint32_t)cuy & 1) * 2 + ((uint32_t)cux & 1);
    return (ctu_row * width_ctu + ctu_col) * 4 + cu_sub;
}

static inline bool hevc_cu_is_available(uint32_t width_ctu, uint32_t height_ctu,
                                        int cur_cux, int cur_cuy,
                                        int nb_cux, int nb_cuy, int cuy_min)
{
    if (nb_cux < 0 || nb_cuy < cuy_min) return false;
    if (nb_cux >= (int)(width_ctu * 2) || nb_cuy >= (int)(height_ctu * 2)) return false;
    uint32_t cur_rank = hevc_cu_rank(width_ctu, cur_cux, cur_cuy);
    uint32_t nb_rank = hevc_cu_rank(width_ctu, nb_cux, nb_cuy);
    return nb_rank < cur_rank;
}

/* Derives spatial merge candidates matching ITU-T H.265 Section 8.5.3.2.2.
 * Output cand_mvs has exactly 5 candidates (padded with (0,0)), in 1/4-pel units.
 * All candidates are strictly derived from spatial neighbors or zero-vectors,
 * ensuring 100% bit-exact candidate derivation matching hardware decoders. */
static int derive_merge_candidates_n(int cuy_min, const hevc_encoder_t *enc,
                                     int cux, int cuy, int n,
                                     hevc_mv_t cand_mvs[5]){
    uint32_t w_cu = enc->width_ctu * 2;
    uint32_t h_cu = enc->height_ctu * 2;

    hevc_mv_t spatial_cand[5];
    int num_spatial = 0;

    /* 1. Candidate A1 (Left): (cux - 1, cuy) */
    bool a1_has_inter = false;
    hevc_mv_t mv_a1 = {0, 0};
    if (hevc_cu_is_available(enc->width_ctu, enc->height_ctu, cux, cuy, cux - 1, cuy + n - 1, cuy_min)) {
        uint32_t a1_idx = (uint32_t)(cuy + n - 1) * w_cu + (uint32_t)(cux - 1);
        if (enc->cu_is_inter[a1_idx]) {
            a1_has_inter = true;
            mv_a1.x = enc->mv_x_map[a1_idx];
            mv_a1.y = enc->mv_y_map[a1_idx];
            spatial_cand[num_spatial++] = mv_a1;
        }
    }

    /* 2. Candidate B1 (Above): (cux, cuy - 1) */
    bool b1_has_inter = false;
    hevc_mv_t mv_b1 = {0, 0};
    if (hevc_cu_is_available(enc->width_ctu, enc->height_ctu, cux, cuy, cux + n - 1, cuy - 1, cuy_min)) {
        uint32_t b1_idx = (uint32_t)(cuy - 1) * w_cu + (uint32_t)(cux + n - 1);
        if (enc->cu_is_inter[b1_idx]) {
            b1_has_inter = true;
            mv_b1.x = enc->mv_x_map[b1_idx];
            mv_b1.y = enc->mv_y_map[b1_idx];
            /* Pruning: B1 against A1 */
            if (!a1_has_inter || mv_b1.x != mv_a1.x || mv_b1.y != mv_a1.y) {
                spatial_cand[num_spatial++] = mv_b1;
            }
        }
    }

    /* 3. Candidate B0 (Above-Right): (cux + 1, cuy - 1) */
    bool b0_has_inter = false;
    hevc_mv_t mv_b0 = {0, 0};
    if (hevc_cu_is_available(enc->width_ctu, enc->height_ctu, cux, cuy, cux + n, cuy - 1, cuy_min)) {
        uint32_t b0_idx = (uint32_t)(cuy - 1) * w_cu + (uint32_t)(cux + n);
        if (enc->cu_is_inter[b0_idx]) {
            b0_has_inter = true;
            mv_b0.x = enc->mv_x_map[b0_idx];
            mv_b0.y = enc->mv_y_map[b0_idx];
            /* Pruning: B0 against B1 */
            if (!b1_has_inter || mv_b0.x != mv_b1.x || mv_b0.y != mv_b1.y) {
                spatial_cand[num_spatial++] = mv_b0;
            }
        }
    }

    /* 4. Candidate A0 (Below-Left): (cux - 1, cuy + 1) */
    bool a0_has_inter = false;
    hevc_mv_t mv_a0 = {0, 0};
    if (hevc_cu_is_available(enc->width_ctu, enc->height_ctu, cux, cuy, cux - 1, cuy + n, cuy_min)) {
        uint32_t a0_idx = (uint32_t)(cuy + n) * w_cu + (uint32_t)(cux - 1);
        if (enc->cu_is_inter[a0_idx]) {
            a0_has_inter = true;
            mv_a0.x = enc->mv_x_map[a0_idx];
            mv_a0.y = enc->mv_y_map[a0_idx];
            /* Pruning: A0 against A1 */
            if (!a1_has_inter || mv_a0.x != mv_a1.x || mv_a0.y != mv_a1.y) {
                spatial_cand[num_spatial++] = mv_a0;
            }
        }
    }

    /* 5. Candidate B2 (Above-Left): (cux - 1, cuy - 1) */
    if (num_spatial < 4) {
        if (hevc_cu_is_available(enc->width_ctu, enc->height_ctu, cux, cuy, cux - 1, cuy - 1, cuy_min)) {
            uint32_t b2_idx = (uint32_t)(cuy - 1) * w_cu + (uint32_t)(cux - 1);
            if (enc->cu_is_inter[b2_idx]) {
                hevc_mv_t mv_b2;
                mv_b2.x = enc->mv_x_map[b2_idx];
                mv_b2.y = enc->mv_y_map[b2_idx];
                /* Pruning: B2 against A1 and B1 */
                if ((!a1_has_inter || mv_b2.x != mv_a1.x || mv_b2.y != mv_a1.y) &&
                    (!b1_has_inter || mv_b2.x != mv_b1.x || mv_b2.y != mv_b1.y)) {
                    spatial_cand[num_spatial++] = mv_b2;
                }
            }
        }
    }

    int num_cand = 0;
    for (int i = 0; i < num_spatial && num_cand < 5; i++) {
        cand_mvs[num_cand++] = spatial_cand[i];
    }

    while (num_cand < 5) {
        cand_mvs[num_cand].x = 0;
        cand_mvs[num_cand].y = 0;
        num_cand++;
    }

    return num_cand;
}

/* The 8x8 CU case, which is every CU but a whole-CTU skip. */
static inline int derive_merge_candidates(int cuy_min, const hevc_encoder_t *enc,
                                          int cux, int cuy, hevc_mv_t cand_mvs[5])
{
    return derive_merge_candidates_n(cuy_min, enc, cux, cuy, 1, cand_mvs);
}

/* ============================================================================
 * Frame entry point
 * ==========================================================================*/

/* Core encode: assumes encoder->dl_y / encoder->dl_uv (real width x height,
 * NV12: dl_y row-pitch == width, dl_uv row-pitch == width with Cb/Cr
 * interleaved) are already populated. Both hevc_encoder_encode_frame()
 * (GPU-surface readback) and hevc_encoder_encode_raw() (direct host
 * pointers, no GPU involved - see encoder_h265.h) fill those in their own
 * way and then share everything from here on. */
/* Split NV12's interleaved chroma into separate Cb and Cr planes.
 *
 * This ran one byte at a time: 518400 iterations per 1080p frame, inside
 * encode_core(), which the profiler put at the top of the HEVC encode once
 * the surface readback had been dealt with. PSHUFB gathers the even bytes
 * into one half of a register and the odd bytes into the other, so sixteen
 * samples of each plane come out per pair of loads.
 *
 * Runtime-dispatched like cpu_simd_me.c, and the scalar tail keeps widths
 * that are not a multiple of sixteen working.
 */
#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("ssse3")))
static void deinterleava_uv_ssse3(uint8_t *cb, uint8_t *cr,
                                  const uint8_t *uv, size_t n) {
    const __m128i sh = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14,
                                     1, 3, 5, 7, 9, 11, 13, 15);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i a = _mm_loadu_si128((const __m128i *)(uv + 2 * i));
        __m128i b = _mm_loadu_si128((const __m128i *)(uv + 2 * i + 16));
        a = _mm_shuffle_epi8(a, sh);
        b = _mm_shuffle_epi8(b, sh);
        _mm_storeu_si128((__m128i *)(cb + i), _mm_unpacklo_epi64(a, b));
        _mm_storeu_si128((__m128i *)(cr + i), _mm_unpackhi_epi64(a, b));
    }
    for (; i < n; i++) { cb[i] = uv[2 * i]; cr[i] = uv[2 * i + 1]; }
}
#endif

static void deinterleava_uv(uint8_t *cb, uint8_t *cr, const uint8_t *uv, size_t n) {
#if defined(__x86_64__) || defined(_M_X64)
    static int ha_ssse3 = -1;
    if (ha_ssse3 < 0) ha_ssse3 = __builtin_cpu_supports("ssse3") ? 1 : 0;
    if (ha_ssse3) { deinterleava_uv_ssse3(cb, cr, uv, n); return; }
#endif
    for (size_t i = 0; i < n; i++) { cb[i] = uv[2 * i]; cr[i] = uv[2 * i + 1]; }
}

/* ============================================================================
 * Inter prediction: what does not depend on the sample depth
 * ==========================================================================*/

/* Whether the CU at (nb_cux, nb_cuy) is there to lend its motion: coded
 * already, in this slice, and inter. 6.4.2 counts an intra neighbour as
 * unavailable for motion. */
static inline bool motion_available(const hevc_encoder_t *enc, int cuy_min,
                                    int cux, int cuy, int nb_cux, int nb_cuy)
{
    if (!hevc_cu_is_available(enc->width_ctu, enc->height_ctu, cux, cuy,
                              nb_cux, nb_cuy, cuy_min))
        return false;
    return enc->cu_is_inter[(uint32_t)nb_cuy * enc->width_ctu * 2 + (uint32_t)nb_cux] != 0;
}

static inline hevc_mv_t motion_at(const hevc_encoder_t *enc, int cux, int cuy)
{
    const uint32_t i = (uint32_t)cuy * enc->width_ctu * 2 + (uint32_t)cux;
    hevc_mv_t m = { enc->mv_x_map[i], enc->mv_y_map[i] };
    return m;
}

/* 8.5.3.2.6 and 8.5.3.2.7 for this encoder's only prediction unit shape, an
 * 8x8 2Nx2N CU, with one reference picture and no temporal candidate.
 *
 * With every neighbour pointing at the same picture, the spec's second,
 * scaled pass over each group finds exactly what the first one found, so
 * it reduces to this: A is the first of A0, A1 with motion; B the first of
 * B0, B1, B2. When neither A0 nor A1 is even there (isScaledFlag 0), A
 * takes B's vector and B is looked for again - the same one - so the list
 * collapses to B alone. A duplicate is dropped, and zeros fill the list to
 * two. */
static void derive_amvp_candidates(int cuy_min, const hevc_encoder_t *enc,
                                   int cux, int cuy, hevc_mv_t out[2])
{
    const bool a0 = motion_available(enc, cuy_min, cux, cuy, cux - 1, cuy + 1);
    const bool a1 = motion_available(enc, cuy_min, cux, cuy, cux - 1, cuy);
    bool has_a = false, has_b = false;
    hevc_mv_t mv_a = { 0, 0 }, mv_b = { 0, 0 };
    if (a0)      { has_a = true; mv_a = motion_at(enc, cux - 1, cuy + 1); }
    else if (a1) { has_a = true; mv_a = motion_at(enc, cux - 1, cuy); }

    static const int bx[3] = { 1, 0, -1 };
    for (int k = 0; k < 3 && !has_b; k++) {
        if (motion_available(enc, cuy_min, cux, cuy, cux + bx[k], cuy - 1)) {
            has_b = true;
            mv_b = motion_at(enc, cux + bx[k], cuy - 1);
        }
    }
    if (!a0 && !a1) {
        /* isScaledFlag 0: A takes B, and B found again is B. */
        has_a = has_b;
        mv_a = mv_b;
    }

    int n = 0;
    if (has_a) out[n++] = mv_a;
    if (has_b && !(has_a && mv_a.x == mv_b.x && mv_a.y == mv_b.y)) out[n++] = mv_b;
    while (n < 2) { out[n].x = 0; out[n].y = 0; n++; }
}

/* Rough bit counts, for choosing between candidates, not for the rate
 * control. One motion vector component in quarter samples, as mvd_coding()
 * spends it: greater0, greater1, an order-1 Exp-Golomb remainder, a sign. */
static inline int mvd_bits(int d)
{
    const unsigned a = (unsigned)(d < 0 ? -d : d);
    if (a == 0) return 1;
    if (a == 1) return 3;
    return 3 + 2 * (31 - __builtin_clz((a - 2) / 2 + 1)) + 2;
}

/* Bits a quantized 4x4 block costs, roughly: where the last coefficient
 * is, a significance flag up to it, and per coefficient a sign and its
 * size. */
static int coeff_bits(const int16_t c[16])
{
    int last = -1, bits = 0;
    for (int i = 15; i >= 0; i--) if (c[i]) { last = i; break; }
    if (last < 0) return 0;
    bits = 4 + last;
    for (int i = 0; i <= last; i++) {
        const int a = c[i] < 0 ? -c[i] : c[i];
        if (a) bits += 2 + (a > 1 ? 1 + 2 * (31 - __builtin_clz((unsigned)a)) : 0);
    }
    return bits;
}

/* The Lagrange multipliers for this frame's QP: for sums of squared errors
 * (0.57 * 2^((QP-12)/3), HM's value for P pictures) and for sums of absolute
 * differences, its square root. Eight-bit units: the ten-bit path scales
 * its distortions down before comparing. Fixed point, eight fraction bits. */
static inline int64_t lambda_sse_q8(int qp)
{
    return (int64_t)(0.57 * pow(2.0, (qp - 12) / 3.0) * 256.0 + 0.5);
}

static inline int lambda_sad_q8(int qp)
{
    return (int)(sqrt(0.57 * pow(2.0, (qp - 12) / 3.0)) * 256.0 + 0.5);
}

/* What an 8x8 CU of a P picture was decided to be. */
enum { CU_SKIP, CU_MERGE, CU_AMVP, CU_INTRA };

/* Everything that touches samples, once per bit depth. */
#define BIT_DEPTH 8
#include "hevc_pixel.h"
#include "hevc_enc_template.c"
#undef BIT_DEPTH

#define BIT_DEPTH 10
#include "hevc_pixel.h"
#include "hevc_enc_template.c"
#undef BIT_DEPTH

/* Waiting for the row above: spin briefly, then give the core away - on a
 * machine also running the game being streamed, a spinning thread is a
 * stolen one. */
static inline void wpp_pause(unsigned *spins)
{
    if (++*spins < 64) _mm_pause();
    else sched_yield();
}
#define WPP_PAUSE() wpp_pause(&spins_)

/* ============================================================================
 * Wavefront parallel processing
 * ==========================================================================*/

/* Emulation prevention over a run of RBSP bytes, continuing the zero count
 * across calls, so that a slice's data can be escaped row by row and each
 * row's escaped size known - the entry points count escaped bytes (7.4.7.1).
 * `out` may be NULL to count only. Returns the bytes written. */
static size_t ebsp_escape(const uint8_t *in, size_t n, uint8_t *out, int *zeros)
{
    size_t o = 0;
    for (size_t i = 0; i < n; i++) {
        const uint8_t b = in[i];
        if (*zeros >= 2 && b <= 3) {
            if (out) out[o] = 0x03;
            o++;
            *zeros = 0;
        }
        if (out) out[o] = b;
        o++;
        *zeros = b == 0 ? *zeros + 1 : 0;
    }
    return o;
}

/* One CTU row of a WPP picture, into its own substream.
 *
 * The row waits for the one above to be two CTUs ahead: a CTU reads the
 * CTU above and to the right of it - intra reference samples, the B0 merge
 * and AMVP candidates. Its CABAC contexts start from those the row above had
 * after its second CTU (9.3.1: the synchronization), or fresh on the first
 * row of a slice, where the CTU above-right is in another slice. */
static void encode_wpp_row(hevc_encoder_t *e, int row, int r0, int r1, bool is_idr, bool ten_bit)
{
    const int w = (int)e->width_ctu;
    const int y_min = r0 * HEVC_CTU_SIZE;
    bitstream_t bs;
    bs_init(&bs, e->row_buf[row], e->row_buf_cap);
    hevc_cabac_t cab;
    hevc_cabac_init(&cab, &bs);
    uint32_t sad = 0;
    unsigned spins_ = 0;

    if (row == r0 || w < 2) {
        hevc_cabac_reset_contexts(&cab, e->qp, is_idr ? 2 : 1);
    } else {
        while (atomic_load_explicit(&e->row_progress[row - 1], memory_order_acquire) < 2)
            WPP_PAUSE();
        memcpy(cab.ctx, e->row_ctx[row - 1], sizeof(cab.ctx));
    }
    hevc_cabac_start(&cab);

    for (int col = 0; col < w; col++) {
        if (row > r0) {
            const int need = col + 2 < w ? col + 2 : w;
            while (atomic_load_explicit(&e->row_progress[row - 1], memory_order_acquire) < need)
                WPP_PAUSE();
        }
        if (ten_bit) encode_ctu_10(e, &cab, col, row, is_idr, y_min, &sad);
        else         encode_ctu_8(e, &cab, col, row, is_idr, y_min, &sad);
        if (col == 1) memcpy(e->row_ctx[row], cab.ctx, sizeof(cab.ctx));

        const bool last_of_slice = row == r1 - 1 && col == w - 1;
        hevc_cabac_encode_terminate(&cab, last_of_slice ? 1 : 0);   /* end_of_slice_segment_flag */
        if (!last_of_slice && col == w - 1) {
            hevc_cabac_encode_terminate(&cab, 1);                   /* end_of_subset_one_bit */
        }
        atomic_store_explicit(&e->row_progress[row], col + 1, memory_order_release);
    }
    hevc_cabac_finish(&cab);
    bs_rbsp_trailing_bits(&bs);   /* byte_alignment(), or the slice's trailing bits */
    e->row_len[row] = bs_bytes_written(&bs);
    e->row_sad[row] = sad;
}

static size_t maybe_append_filler_hevc(hevc_encoder_t *encoder, size_t total_written)
{
    if (!encoder->cbr_intent ||
        (encoder->rc.mode != RC_CBR && encoder->rc.mode != RC_LOW_LATENCY)) {
        return total_written;
    }

    uint32_t target_bytes = (encoder->rc.target_bits_per_frame + 7) / 8;
    /* With the rate model the stream's account decides, the way a real
     * buffer would: fill only what brings the debt back to zero. Filling
     * every small picture up to one picture's budget, while the large ones
     * stay large, put CBR 4-9% over its bitrate (and 34-57% with the old
     * loop). */
    if (encoder->rc.model) {
        const double want = (double)encoder->rc.target_bits_per_frame - encoder->rc.debt;
        target_bytes = want > 0.0 ? (uint32_t)(want / 8.0) : 0;
    }
    if (target_bytes <= total_written) {
        return total_written;
    }

    size_t shortfall = (size_t)target_bytes - total_written;
    if (shortfall < BS_HEVC_FILLER_MIN_NAL_SIZE) {
        return total_written;
    }

    size_t ff_count = shortfall - BS_HEVC_FILLER_MIN_NAL_SIZE;
    if (total_written + shortfall > encoder->scratch_out_cap) {
        size_t new_cap = total_written + shortfall + 131072;
        uint8_t *new_buf = realloc(encoder->scratch_out, new_cap);
        if (new_buf) {
            encoder->scratch_out = new_buf;
            encoder->scratch_out_cap = new_cap;
        } else {
            return total_written;
        }
    }

    size_t written = bs_write_filler_hevc(encoder->scratch_out + total_written,
                                          encoder->scratch_out_cap - total_written,
                                          ff_count);
    return total_written + written;
}

static int encode_core(hevc_encoder_t *encoder, uint8_t *output_buf, size_t output_size)
{
    bool is_idr = (encoder->frame_count % encoder->gop_size == 0) || encoder->force_idr || !encoder->has_ref;
    encoder->force_idr = false;
    if (is_idr) {
        encoder->poc = 0;
    }

    /* In VBR/CBR/LOW_LATENCY mode, update QP via rate control model */
    if (encoder->rc.mode != RC_CQP) {
        int target_qp = encoder->rc.model ? rc_model_frame_qp(&encoder->rc, is_idr)
                                          : rc_get_frame_qp(&encoder->rc, is_idr ? 0 : encoder->last_frame_sad);
        if (target_qp >= 1 && target_qp <= 51) {
            encoder->qp = target_qp;
        }
    }
    encoder->last_frame_sad = 0;
    encoder->lambda_sse_q8 = lambda_sse_q8(encoder->qp);
    encoder->lambda_sad_q8 = lambda_sad_q8(encoder->qp);
    /* Rounding to nearest cost a fifth more bits for the same picture than
     * the dead zone does. */
    encoder->quant_round_inter = encoder->deadzone ? HEVC_QUANT_ROUND_INTER : HEVC_QUANT_ROUND_NEAREST;
    encoder->quant_round_intra = !encoder->deadzone ? HEVC_QUANT_ROUND_NEAREST
                               : is_idr ? HEVC_QUANT_ROUND_INTRA_I : HEVC_QUANT_ROUND_INTRA_P;

    const bool ten_bit = encoder->bit_depth > 8;
    if (ten_bit) load_source_10(encoder);
    else         load_source_8(encoder);

    size_t num_cus = (size_t)(encoder->width_ctu * 2) * (encoder->height_ctu * 2);
    memset(encoder->luma_mode_map, 0, (size_t)encoder->mode_map_stride * (encoder->coded_height / HEVC_PU_SIZE));
    memset(encoder->cu_skip_map, 0, num_cus);
    memset(encoder->cu_is_inter, 0, num_cus);
    memset(encoder->cu_depth, 0, num_cus);
    memset(encoder->mv_x_map, 0, num_cus * sizeof(int16_t));
    memset(encoder->mv_y_map, 0, num_cus * sizeof(int16_t));

    bool write_param_sets = is_idr;
    if (getenv("BC250_HEVC_REPEAT_HEADERS")) {
        write_param_sets = true;
    }
    if (write_param_sets) {
        encoder->pps_init_qp = encoder->qp;
    }
    int slice_qp_delta = encoder->qp - encoder->pps_init_qp;

    /* One slice per region of CTU rows.
     *
     * A slice is a whole number of CTU rows, so a slice boundary is also a CTU
     * row boundary - which is why the MPM derivation, which already refuses to
     * look above a CTU row, needed nothing. Everything else that reads a
     * neighbour takes y_min and stops there: the intra reference samples, the
     * merge candidates and the two CABAC contexts that ask whether the block
     * above was skipped.
     *
     * Each slice gets its own CABAC engine, its own buffer and its own SAD
     * accumulator, so nothing is shared while a slice is being encoded. That
     * is what lets the loop be handed to OpenMP.
     */
    if (!is_idr && encoder->has_ref) {
        if (ten_bit) build_hpel_10(encoder);
        else         build_hpel_8(encoder);
    }

    const int ns = encoder->num_slices;
    const uint32_t ctu_rows = encoder->height_ctu;
    uint32_t bit_address = 0;
    {
        uint32_t n = encoder->width_ctu * encoder->height_ctu;
        while ((1u << bit_address) < n) bit_address++;
    }

    /* Nothing is shared while a slice is being encoded.
     *
     * Each slice has its own CABAC engine, its own output buffer, its own SAD
     * accumulator and its own region of the reconstruction and of the mode and
     * skip maps. Every neighbour lookup stops at y_min, so no slice ever reads
     * a pixel or a mode another slice is writing. The shared reads - the source
     * planes and the previous reconstruction - are read-only for the whole
     * frame.
     */
    if (encoder->wpp) {
        for (uint32_t r = 0; r < ctu_rows; r++)
            atomic_store_explicit(&encoder->row_progress[r], 0, memory_order_relaxed);
#ifdef _OPENMP
        int max_t = omp_get_max_threads();
        int th = (int)ctu_rows < max_t ? (int)ctu_rows : max_t;
        /* schedule(static, 1): thread t takes rows t, t + T, ... in order,
         * so a row only ever waits for one that is already running or
         * done. */
#pragma omp parallel for schedule(static, 1) num_threads(th)
#endif
        for (int r = 0; r < (int)ctu_rows; r++) {
            int s = 0;
            while (s + 1 < ns && (uint32_t)r >= (uint32_t)(((uint64_t)ctu_rows * (uint32_t)(s + 1)) / (uint32_t)ns)) s++;
            const int r0 = (int)(((uint64_t)ctu_rows * (uint32_t)s) / (uint32_t)ns);
            const int r1 = (int)(((uint64_t)ctu_rows * (uint32_t)(s + 1)) / (uint32_t)ns);
            encode_wpp_row(encoder, r, r0, r1, is_idr, ten_bit);
        }
        for (uint32_t r = 0; r < ctu_rows; r++) encoder->last_frame_sad += encoder->row_sad[r];
    } else {
#ifdef _OPENMP
    int max_t = omp_get_max_threads();
    int slice_threads = (ns < max_t) ? ns : max_t;
    if (slice_threads < 1) slice_threads = 1;
#pragma omp parallel for schedule(static) num_threads(slice_threads) if (ns > 1 && slice_threads > 1)
#endif
    for (int s = 0; s < ns; s++) {
        uint32_t r0 = (uint32_t)(((uint64_t)ctu_rows * (uint32_t)s) / (uint32_t)ns);
        uint32_t r1 = (uint32_t)(((uint64_t)ctu_rows * (uint32_t)(s + 1)) / (uint32_t)ns);
        int y_min = (int)(r0 * HEVC_CTU_SIZE);
        uint32_t sad = 0;

        bitstream_t slice_bs;
        bs_init(&slice_bs, encoder->slice_buf[s], encoder->slice_buf_cap);

        bs_write1(&slice_bs, s == 0 ? 1 : 0);   /* first_slice_segment_in_pic_flag */
        if (is_idr) {
            bs_write1(&slice_bs, 1);            /* no_output_of_prior_pics_flag */
        }
        bs_write_ue(&slice_bs, 0);              /* slice_pic_parameter_set_id */
        /* dependent_slice_segments_enabled_flag is 0 in the PPS, so no
         * dependent_slice_segment_flag here - just the address, in
         * Ceil(Log2(PicSizeInCtbsY)) bits, per Rec. ITU-T H.265 7.3.6.1. */
        if (s != 0) {
            bs_write_u(&slice_bs, (int)bit_address, r0 * encoder->width_ctu);
        }
        bs_write_ue(&slice_bs, is_idr ? 2 : 1); /* slice_type: 2 = I, 1 = P */

        if (!is_idr) {
            bs_write_u(&slice_bs, 8, encoder->poc & 0xFF);
            bs_write1(&slice_bs, 1);
            bs_write1(&slice_bs, 0);
            bs_write_ue(&slice_bs, 0);
        }

        bs_write_se(&slice_bs, slice_qp_delta);
        bs_rbsp_trailing_bits(&slice_bs);

        hevc_cabac_t cab;
        hevc_cabac_init(&cab, &slice_bs);
        hevc_cabac_reset_contexts(&cab, encoder->qp, is_idr ? 2 : 1);
        hevc_cabac_start(&cab);

        uint32_t ctus_slice = (r1 - r0) * encoder->width_ctu;
        uint32_t k = 0;
        for (uint32_t row = r0; row < r1; row++) {
            for (uint32_t col = 0; col < encoder->width_ctu; col++) {
                if (ten_bit) encode_ctu_10(encoder, &cab, (int)col, (int)row, is_idr, y_min, &sad);
                else         encode_ctu_8(encoder, &cab, (int)col, (int)row, is_idr, y_min, &sad);
                k++;
                /* end_of_slice_segment_flag: the last CTU of THIS slice */
                hevc_cabac_encode_terminate(&cab, k == ctus_slice ? 1 : 0);
            }
        }

        hevc_cabac_finish(&cab);
        bs_rbsp_trailing_bits(&slice_bs);
        encoder->slice_len[s] = bs_bytes_written(&slice_bs);
        encoder->slice_sad[s] = sad;
    }

    for (int s = 0; s < ns; s++) encoder->last_frame_sad += encoder->slice_sad[s];
    }

    size_t total = 0;
    total += write_aud_hevc(encoder->scratch_out + total, encoder->scratch_out_cap - total, is_idr);
    if (write_param_sets) {
        total += write_vps(encoder->scratch_out + total, encoder->scratch_out_cap - total,
                           encoder->bit_depth);
        total += write_sps(encoder->scratch_out + total, encoder->scratch_out_cap - total,
                            encoder->coded_width, encoder->coded_height,
                            encoder->width, encoder->height,
                            hevc_pick_level_idc(encoder->coded_width, encoder->coded_height),
                            encoder->bit_depth, encoder->tu8);
        total += write_pps(encoder->scratch_out + total, encoder->scratch_out_cap - total, encoder->qp,
                           encoder->wpp);
    }

    for (int s = 0; encoder->wpp && s < ns; s++) {
        const uint32_t r0 = (uint32_t)(((uint64_t)ctu_rows * (uint32_t)s) / (uint32_t)ns);
        const uint32_t r1 = (uint32_t)(((uint64_t)ctu_rows * (uint32_t)(s + 1)) / (uint32_t)ns);

        /* The rows' sizes once escaped: the entry points count escaped
         * bytes. The header ends in a non-zero byte - the alignment bit -
         * so the data's zero count starts from nothing. */
        size_t sizes[HEVC_WPP_MAX_ROWS], data = 0, max_size = 0;
        int zeros = 0;
        for (uint32_t r = r0; r < r1; r++) {
            sizes[r - r0] = ebsp_escape(encoder->row_buf[r], encoder->row_len[r], NULL, &zeros);
            data += sizes[r - r0];
            if (r + 1 < r1 && sizes[r - r0] > max_size) max_size = sizes[r - r0];
        }

        uint8_t hdr[HEVC_WPP_MAX_ROWS * 5 + 64];
        bitstream_t hb;
        bs_init(&hb, hdr, sizeof(hdr));
        bs_write1(&hb, s == 0 ? 1 : 0);          /* first_slice_segment_in_pic_flag */
        if (is_idr) bs_write1(&hb, 1);           /* no_output_of_prior_pics_flag */
        bs_write_ue(&hb, 0);                     /* slice_pic_parameter_set_id */
        if (s != 0) bs_write_u(&hb, (int)bit_address, r0 * encoder->width_ctu);
        bs_write_ue(&hb, is_idr ? 2 : 1);        /* slice_type */
        if (!is_idr) {
            bs_write_u(&hb, 8, encoder->poc & 0xFF);
            bs_write1(&hb, 1);
            bs_write1(&hb, 0);
            bs_write_ue(&hb, 0);
        }
        bs_write_se(&hb, slice_qp_delta);
        /* 7.3.6.1: the entry points, one per row after the first. */
        const uint32_t entries = r1 - r0 - 1;
        bs_write_ue(&hb, entries);               /* num_entry_point_offsets */
        if (entries > 0) {
            int len = 1;
            while (len < 32 && (max_size - 1) >> len) len++;
            bs_write_ue(&hb, (uint32_t)(len - 1)); /* offset_len_minus1 */
            for (uint32_t i = 0; i < entries; i++)
                bs_write_u(&hb, len, (uint32_t)(sizes[i] - 1));
        }
        bs_rbsp_trailing_bits(&hb);              /* byte_alignment() */

        size_t needed = total + 64 + sizeof(hdr) * 2 + data;
        if (needed > encoder->scratch_out_cap) {
            uint8_t *nb = realloc(encoder->scratch_out, needed + 131072);
            if (!nb) return -1;
            encoder->scratch_out = nb;
            encoder->scratch_out_cap = needed + 131072;
        }
        bitstream_t out_bs;
        bs_init(&out_bs, encoder->scratch_out + total, encoder->scratch_out_cap - total);
        bs_write_nal_header_hevc(&out_bs, is_idr ? NAL_UNIT_CODED_SLICE_IDR_W_RADL : NAL_UNIT_CODED_SLICE_TRAIL_R);
        total += bs_bytes_written(&out_bs);
        total += bs_rbsp_to_ebsp(encoder->scratch_out + total, encoder->scratch_out_cap - total,
                                 hdr, bs_bytes_written(&hb));
        zeros = 0;
        for (uint32_t r = r0; r < r1; r++)
            total += ebsp_escape(encoder->row_buf[r], encoder->row_len[r],
                                 encoder->scratch_out + total, &zeros);
    }

    for (int s = 0; !encoder->wpp && s < ns; s++) {
        size_t needed = total + 32 + encoder->slice_len[s] * 2;
        if (needed > encoder->scratch_out_cap) {
            size_t new_cap = encoder->scratch_out_cap * 2;
            if (new_cap < needed + 131072) new_cap = needed + 131072;
            uint8_t *new_buf = realloc(encoder->scratch_out, new_cap);
            if (new_buf) {
                encoder->scratch_out = new_buf;
                encoder->scratch_out_cap = new_cap;
            }
        }
        bitstream_t out_bs;
        bs_init(&out_bs, encoder->scratch_out + total, encoder->scratch_out_cap - total);
        bs_write_nal_header_hevc(&out_bs, is_idr ? NAL_UNIT_CODED_SLICE_IDR_W_RADL : NAL_UNIT_CODED_SLICE_TRAIL_R);
        size_t off = bs_bytes_written(&out_bs);
        size_t ebsp = bs_rbsp_to_ebsp(encoder->scratch_out + total + off,
                                      encoder->scratch_out_cap - total - off,
                                      encoder->slice_buf[s], encoder->slice_len[s]);
        total += off + ebsp;
    }

    size_t real_coded = total;
    total = maybe_append_filler_hevc(encoder, total);

    if (total > output_size) return -1;
    memcpy(output_buf, encoder->scratch_out, total);

    if (real_coded > 0 && encoder->rc.mode != RC_CQP) {
        rc_update_stats(&encoder->rc, (int)(real_coded * 8));
        if (encoder->rc.model) rc_model_frame_coded(&encoder->rc, is_idr, encoder->qp, (int)(real_coded * 8));
    }

    /* Update reference buffers for subsequent P-frames */
    size_t luma_size = (size_t)encoder->coded_width * encoder->coded_height;
    size_t chroma_size = (size_t)(encoder->coded_width / 2) * (encoder->coded_height / 2);
    /* The reference picture changes hands, it does not get copied.
     *
     * This was three megabytes of memcpy per 1080p frame. prev_recon_* is only
     * ever READ (motion search and the skip path's copy) and recon_* is only
     * ever WRITTEN - every pixel of the coded area, by one CU or another - so
     * swapping the pointers leaves both sides holding exactly what the copy
     * used to give them. Both buffers stay allocated and are freed together,
     * so nothing changes for hevc_encoder_destroy().
     */
    {
        void *t;
        t = encoder->prev_recon_y;  encoder->prev_recon_y  = encoder->recon_y;  encoder->recon_y  = t;
        t = encoder->prev_recon_cb; encoder->prev_recon_cb = encoder->recon_cb; encoder->recon_cb = t;
        t = encoder->prev_recon_cr; encoder->prev_recon_cr = encoder->recon_cr; encoder->recon_cr = t;
    }
    (void)luma_size; (void)chroma_size;
    encoder->has_ref = true;
    encoder->poc++;

    /* Debug-only: dump this encoder's own idea of the reconstructed picture
     * (i.e. what a bug-free decoder given this exact bitstream SHOULD
     * reproduce) - lets a diff against a real decoder's actual output
     * localize whether a mismatch is in the prediction/transform/quant
     * math (this dump would ALSO look wrong) or in CABAC/bitstream framing
     * (this dump looks right, but a real decoder's output doesn't). */
    if (getenv("BC250_HEVC_DEBUG_RECON")) {
        /* After the swap it is prev_recon_y that holds this frame. */
        /* ⚠️ Not the working directory: that is whatever process
         * loaded the driver happened to be started in. */
        FILE *fy = bc250_debug_dump_open("bc250_hevc_debug_recon_y.raw",
                                         "BC250_HEVC_DEBUG_RECON");
        if (fy) {
            fwrite(encoder->prev_recon_y, ten_bit ? 2 : 1,
                   (size_t)encoder->coded_width * encoder->coded_height, fy);
            fclose(fy);
        }
    }

    encoder->frame_count++;
    return (int)total;
}

/* The picture is as big as the surface it arrives in, not as big as the
 * context was opened for.
 *
 * ⚠️ They differ, and not by accident: ffmpeg opens a Main 10 context at
 * 1920x1088, the height rounded up to a whole CTB, and then hands it
 * 1920x1080 surfaces. Taking the context's word for it, this encoder read
 * 1088 rows out of a 1080-row surface - past the end of its memory at two
 * bytes a sample, which crashed. At one byte a sample the same mistake
 * reads whatever follows the plane instead, and codes it, with no cropping
 * window to hide it. So each frame is the smaller of the two in each
 * direction.
 *
 * The buffers and the maps are sized for the context's CTB grid, so a size
 * that needs another grid is refused. One that fits it is taken on, with an
 * IDR, whose SPS carries the new conformance window. */
static int fit_to_surface(hevc_encoder_t *encoder, uint32_t surface_w, uint32_t surface_h)
{
    if (!surface_w || !surface_h) return 0;
    uint32_t w = surface_w < encoder->ctx_width ? surface_w : encoder->ctx_width;
    uint32_t h = surface_h < encoder->ctx_height ? surface_h : encoder->ctx_height;
    if (w == encoder->width && h == encoder->height) return 0;
    if (round_up16(w) != encoder->coded_width || round_up16(h) != encoder->coded_height)
        return -1;
    encoder->width = w;
    encoder->height = h;
    encoder->force_idr = true;
    return 0;
}

int hevc_encoder_encode_frame(hevc_encoder_t *encoder,
                              bc250_gpu_context_t *gpu_ctx,
                              gpu_image_t input_surface,
                              gpu_memory_t input_memory,
                              uint8_t *output_buf, size_t output_size)
{
    if (!encoder || !output_buf) return -1;

    encoder->num_gpu_mvs = 0;
    bool is_idr = (encoder->frame_count % encoder->gop_size == 0) || encoder->force_idr || !encoder->has_ref;
    const bool ten_bit = encoder->bit_depth > 8;

    if (gpu_ctx && input_surface.y_plane != VK_NULL_HANDLE) {
        /* An eight-bit encoder reading a P010 surface, or the reverse,
         * would encode half a picture of garbage without a complaint. */
        if ((input_surface.format == GPU_IMAGE_P010) != ten_bit) return -1;
        if (fit_to_surface(encoder, input_surface.width, input_surface.height) != 0) return -1;

        /* ⚠️ The governor's tiers mean something narrower here than in the
         * H.264 encoder. The GPU's only job in this one is the motion
         * search, so there is no reduced-search dispatch to fall back on:
         * tiers 0 and 1 both run it, and tiers 2 and 3 do not run it at
         * all. Not running it is already a complete fallback - the CPU
         * search is what happens when num_gpu_mvs stays at zero, which is
         * exactly the state an I frame is in. */
        const governor_tier_t tier = dynamic_governor_get_tier(&encoder->governor);
        bool use_gpu_me = (tier < GOV_TIER_2_CPU_OFFLOAD);

        /* ⚠️ Not at ten bits. The motion search shader reads eight-bit
         * luma, and all this encoder takes from it is the hint that a block
         * has not moved; the CPU makes that decision on its own without it,
         * which is what every I frame does anyway. */
        if (ten_bit) use_gpu_me = false;

        /* ⚠️ And that search is also the only thing that measures the GPU.
         * Skipping it leaves the governor with no new latency, so the
         * moving average never decays and the encoder would stay on the
         * CPU for the rest of the stream. One frame in every
         * step_down_hysteresis goes to the GPU anyway, purely to bring
         * back a reading. */
        if (!use_gpu_me && !ten_bit) {
            const uint32_t every = encoder->governor.step_down_hysteresis;
            encoder->governor_skips++;
            use_gpu_me = every && (encoder->governor_skips % every == 0);
        }

        /* Run lightweight subgroup-accelerated GPU motion estimation on P-frames (~0.4ms) */
        if (!is_idr && encoder->has_ref && use_gpu_me) {
            gpu_compute_begin_picture(gpu_ctx, input_surface);
            gpu_compute_dispatch_me_only(gpu_ctx, input_surface, (int)encoder->width, (int)encoder->height);
            gpu_compute_end_picture(gpu_ctx);
            gpu_compute_sync(gpu_ctx);
            dynamic_governor_update(&encoder->governor,
                                    gpu_compute_get_last_latency_ms(gpu_ctx));

            void *mv_data = NULL;
            size_t mv_size = 0;
            if (gpu_compute_get_mv_staging_data(gpu_ctx, &mv_data, &mv_size) == 0 && mv_data) {
                size_t max_bytes = (size_t)encoder->width_ctu * encoder->height_ctu * sizeof(gpu_mv_t);
                size_t copy_bytes = (mv_size < max_bytes) ? mv_size : max_bytes;
                memcpy(encoder->gpu_mvs, mv_data, copy_bytes);
                encoder->num_gpu_mvs = (uint32_t)(copy_bytes / sizeof(gpu_mv_t));
            }
        } else if (tier == GOV_TIER_3_FAILOVER && !is_idr && encoder->has_ref) {
            /* One skipped frame is the whole emergency. Step back down so
             * the next frame tries the GPU again instead of waiting for a
             * measurement that can only come from trying. */
            dynamic_governor_notify_failover_handled(&encoder->governor);
        }

        /* The pitches are in bytes, and a P010 row is two bytes a sample. */
        const int pitch = (int)encoder->width * (ten_bit ? 2 : 1);
        gpu_compute_download_nv12(gpu_ctx, &input_surface, input_memory,
                                   encoder->dl_y, pitch,
                                   encoder->dl_uv, pitch,
                                   (int)encoder->width, (int)encoder->height);
    } else if (ten_bit) {
        /* Mid-grey, as P010 carries it: 512 in the top ten bits. */
        const size_t n = (size_t)encoder->width * encoder->height;
        uint16_t *y = (uint16_t *)encoder->dl_y, *uv = (uint16_t *)encoder->dl_uv;
        for (size_t i = 0; i < n; i++) y[i] = 512 << 6;
        for (size_t i = 0; i < n / 2; i++) uv[i] = 512 << 6;
    } else {
        memset(encoder->dl_y, 128, (size_t)encoder->width * encoder->height);
        memset(encoder->dl_uv, 128, (size_t)(encoder->width / 2) * (encoder->height / 2) * 2);
    }

    return encode_core(encoder, output_buf, output_size);
}

int hevc_encoder_encode_raw(hevc_encoder_t *encoder,
                            const uint8_t *y_plane, int y_pitch,
                            const uint8_t *uv_plane, int uv_pitch,
                            uint8_t *output_buf, size_t output_size)
{
    if (!encoder || !output_buf || !y_plane || !uv_plane) return -1;

    encoder->num_gpu_mvs = 0;

    /* A row is `width` samples; at ten bits a sample is two bytes. */
    const size_t row = (size_t)encoder->width * (encoder->bit_depth > 8 ? 2 : 1);
    for (uint32_t y = 0; y < encoder->height; y++)
        memcpy(encoder->dl_y + (size_t)y * row, y_plane + (size_t)y * y_pitch, row);
    for (uint32_t y = 0; y < encoder->height / 2; y++)
        memcpy(encoder->dl_uv + (size_t)y * row, uv_plane + (size_t)y * uv_pitch, row);

    return encode_core(encoder, output_buf, output_size);
}
