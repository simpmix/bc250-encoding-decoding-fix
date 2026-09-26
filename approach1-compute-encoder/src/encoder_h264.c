/* bc250-encoding-decoding-fix v0.4.3 - https://github.com/simpmix/bc250-encoding-decoding-fix */
/*
 * Copyright (c) 2026 BC-250 Project Contributors
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * encoder_h264.c - H.264/AVC Compute Shader Encoder Implementation
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "bitstream.h"
#include "cavlc.h"
#include "cabac.h"
#include "rate_control.h"
#include "gpu_compute.h"
#include "cpu_simd_me.h"
#include "dynamic_governor.h"
#include "encoder_h264.h"
#ifdef BC250_HAVE_X264
#include "encoder_x264.h"
#endif

/* Decoded Picture Buffer entry */
typedef struct dpb_entry {
    gpu_image_t image;          /* Reconstructed frame on GPU */
    gpu_memory_t memory;
    int frame_num;              /* H.264 frame_num */
    int poc;                    /* Picture order count */
    bool is_reference;          /* Used as reference? */
    bool is_long_term;          /* Long-term reference */
} dpb_entry_t;

/*
 * ============================================================================
 * Spec (clause 6.4.3) luma4x4BlkIdx <-> GPU raster block-index remapping
 * ============================================================================
 *
 * The GPU buffers (quant_levels_buffer/coeff_buffer) lay out the 16 luma 4x4
 * blocks of a macroblock in plain raster order: GPU block index = row*4+col,
 * where (row,col) are in 4x4-pixel-block units (block 0 = MB pixel offset
 * (0,0), block 1 = (4,0), block 4 = (0,4), etc - confirmed by re-reading
 * dct_transform.comp/quantize.comp, whose global_block_idx = mb_idx*24+block_idx
 * assumes exactly this).
 *
 * H.264's bitstream, however, must present luma4x4BlkIdx in *spec* order
 * (0..15), which maps to pixel offsets via clause 6.4.3's InverseRasterScan
 * quadrant math: quadrant q = blkIdx/4 gives a (2*(q%2), 2*(q/2)) 4x4-block
 * offset, and sub = blkIdx%4 gives a (sub%2, sub/2) offset within that
 * quadrant. blk_x4[]/blk_y4[] below were hand-derived from that formula and
 * verified against every entry (see task notes / commit message).
 */
static const int blk_x4[16] = {0,1,0,1, 2,3,2,3, 0,1,0,1, 2,3,2,3};
static const int blk_y4[16] = {0,0,1,1, 0,0,1,1, 2,2,3,3, 2,2,3,3};

/* Inverse of (blk_x4,blk_y4): spec_blk_idx_from_xy[y4][x4] -> spec blkIdx.
 * Built by hand-inverting the table above (blkIdx -> (blk_x4[blkIdx], blk_y4[blkIdx])). */
static const int spec_blk_idx_from_xy[4][4] = {
    {  0,  1,  4,  5 },
    {  2,  3,  6,  7 },
    {  8,  9, 12, 13 },
    { 10, 11, 14, 15 }
};

static inline int gpu_raster_block_idx(int blk_idx) {
    return blk_y4[blk_idx] * 4 + blk_x4[blk_idx];
}

/* Chroma 4x4 blocks are a plain 2x2 grid with no further sub-quadrant
 * structure (unlike luma's 16-block, 4-quadrant layout), so clause 6.4.3's
 * chroma equivalent (8.5.11.2-ish 2x2 InverseRasterScan) reduces to the same
 * raster order the GPU buffers already use (block (row,col) = row*2+col,
 * TL=0,TR=1,BL=2,BR=3). No remap table is needed for chroma - checked by
 * working the quadrant formula for a 2x2 (not 4x4) grid by hand: with only
 * one level of blocks there is no quadrant/sub split to reorder. */

/*
 * ============================================================================
 * H.264 encoder state
 * ============================================================================
 */

/* Per-slice/per-frame CAVLC neighbor (nC) context, indexed by absolute
 * macroblock address. Persists across frames (see h264_encoder_create) since
 * every entry is unconditionally overwritten before any later macroblock in
 * the same frame could read it as a neighbor (neighbors always have a
 * strictly smaller mb address, and mbs are processed in increasing order),
 * so stale data from a previous frame is never observed. */
typedef struct {
    uint8_t (*nz_luma)[16];  /* [total_mbs][16], indexed by SPEC blkIdx */
    uint8_t (*nz_cb)[4];     /* [total_mbs][4] */
    uint8_t (*nz_cr)[4];     /* [total_mbs][4] */
    uint32_t width_in_mbs;
    uint32_t start_mb;       /* current slice's first_mb_in_slice */
    /* PERF: quantize.comp/intra_wavefront.comp's per-4x4-block nonzero
     * bitmask, [total_mbs*24] uint32, bit p set iff that block's raster
     * position p quantized nonzero (see gpu_compute.h's nz_staging_buffers).
     * block_any_nonzero() answers from this instead of reading the block, so
     * the ~90-96% of blocks that are entirely zero are never touched in the
     * 22MB (at 1440p) quant_levels buffer at all. NULL is legal and means
     * "fall back to scanning" - the mask is purely an accelerator, never a
     * source of truth the encoder cannot do without. */
    const uint32_t *nz_mask;
} nc_ctx_t;

/* H.264 encoder state */
struct h264_encoder {
    /* Dimensions */
    uint32_t width, height;
    uint32_t width_in_mbs, height_in_mbs;
    uint32_t total_mbs;

    /* GOP & Stream structure */
    uint32_t fps;
    uint32_t gop_size;          /* IDR interval */
    uint32_t frame_count;       /* Total frames encoded */
    uint32_t frame_num;         /* H.264 frame_num (resets at IDR) */
    uint32_t idr_pic_id;        /* Increments at each IDR */
    int poc;                    /* Picture order count */
    bool force_idr;             /* Dynamic keyframe request flag */

    /* Parameters */
    h264_sps_t sps;
    h264_pps_t pps;
    rate_control_t rc;
    int qp_hint_applied;         /* Last QP explicitly handed to h264_encoder_set_qp(), or -1 if never called yet */
    bool cbr_intent;             /* see h264_encoder_set_cbr_intent's doc comment */
    uint64_t last_frame_sad;     /* Sum of macroblock motion SAD from previous frame */
    int num_slices;              /* Configured slices per frame (1..16) */
    uint32_t quality_level;      /* 1..7 (1 = Quality, 4 = Balanced, 7 = Speed) */
    uint32_t max_frame_bits;     /* Maximum frame size in bits (0 = unlimited) */

    /* DPB */
    dpb_entry_t dpb[16];
    int dpb_count;
    int dpb_max;

    /* GPU context reference */
    bc250_gpu_context_t *gpu;

    /* Output buffer */
    uint8_t *output_buf;
    size_t output_buf_size;

    /* Software reference frame for host/CPU encoding and test paths */
    uint8_t *prev_y_frame;
    bool has_prev_frame;

    /* CAVLC neighbor (nC) context, persistent across frames (see nc_ctx_t doc).
     * When use_cabac is set, these SAME arrays are reused to hold binary
     * coded_block_flag values (0/1) instead of CAVLC's TotalCoeff counts -
     * see cabac_neighbor.c-equivalent helpers below (luma_cbf_neighbors()/
     * chroma_cbf_neighbors()). The two modes are mutually exclusive per
     * encoder instance (use_cabac is fixed at h264_encoder_create() time),
     * so there is no risk of one mode misreading the other's convention. */
    uint8_t (*nz_luma)[16];
    uint8_t (*nz_cb)[4];
    uint8_t (*nz_cr)[4];

    /* CABAC-only persistent per-frame neighbor state (unused/unallocated
     * when use_cabac is false) - see encode_mb_i16x16_cabac()/
     * encode_mb_p16x16_cabac() and the luma_cbf_neighbors()/
     * chroma_cbf_neighbors()/cbp_nb helpers below for how each is read. */
    bool use_cabac;
    uint8_t *dc_cbf_luma;       /* [total_mbs]: I16x16 luma DC coded_block_flag */
    uint8_t (*dc_cbf_chroma)[2]; /* [total_mbs][2]: {Cb, Cr} DC coded_block_flag */
    int16_t *cbp_nb;            /* [total_mbs]: combined (cbp_chroma<<4)|cbp_luma
                                  * of the last MB encoded at this address this
                                  * slice/frame, or -1 if not yet written this
                                  * slice (used as the CABAC "neighbor
                                  * unavailable" sentinel, matching x264's -1
                                  * convention exactly - see cabac_write_cbp_luma/
                                  * _chroma's doc comments in cabac.h). */
    uint8_t *mvd_x_abs;         /* [total_mbs]: abs(mvd_l0 x), capped at 66 */
    uint8_t *mvd_y_abs;         /* [total_mbs]: abs(mvd_l0 y), capped at 66 */
    uint8_t *skip_flag;         /* [total_mbs]: P-slice mb_skip_flag of the
                                  * last MB encoded at this address this
                                  * slice/frame (for mb_skip_flag's own
                                  * ctxIdxInc neighbor lookup, ITU-T
                                  * 9.3.3.1.1.1) */

    /* PERF: host-side cacheable shadow copies of the GPU staging-buffer
     * readback data (quant_levels/coeff/pred_modes/mvs) - see
     * h264_encoder_encode_frame's "shadow_copy" comment for why these exist.
     * Persistent/growable across frames (realloc'd only when a frame needs a
     * bigger buffer than before, e.g. the very first frame at a given
     * resolution) so a steady-state encode does zero allocation per frame. */
    int16_t *quant_levels_shadow; size_t quant_levels_shadow_cap;
    int *dc_coeff_shadow;     size_t dc_coeff_shadow_cap;
    uint32_t *pred_modes_shadow; size_t pred_modes_shadow_cap;
    void *mvs_shadow;         size_t mvs_shadow_cap; /* actually gpu_mv_t*, typedef'd later in this file */
    /* The nonzero-mask readback needs a shadow for the same reason the others
     * do, and more so: block_any_nonzero() hits it once per block per query,
     * which is precisely the scattered small-read pattern the GPU staging
     * memory is worst at. It is 1/16th the size of quant_levels_shadow. */
    uint32_t *nz_masks_shadow; size_t nz_masks_shadow_cap;

    /* Dynamic CPU/GPU Load Governor & CPU SIMD Motion Estimation */
    dynamic_governor_t governor;
    cpu_simd_me_config_t me_cfg;
    gpu_mv_t *cpu_mvs;
    size_t cpu_mvs_cap;

#ifdef BC250_HAVE_X264
    /* Non-NULL when H.264 goes through libx264 - see encoder_x264.h. The
     * compute pipeline above stays allocated either way: encode_raw() and
     * BC250_H264_BACKEND=compute still use it. */
    h264_x264_t *x264;
    uint8_t *x264_y, *x264_uv;      /* the surface, read back as NV12 */
    size_t x264_cap;
#endif
    int icq_quality;               /* > 0: VA ICQ at this quality factor */
};

static void manage_dpb(h264_encoder_t *encoder, int new_frame_num, int new_poc)
{
    if (encoder->dpb_count >= encoder->dpb_max) {
        memmove(&encoder->dpb[0], &encoder->dpb[1],
                sizeof(dpb_entry_t) * (encoder->dpb_max - 1));
        encoder->dpb_count--;
    }

    dpb_entry_t *entry = &encoder->dpb[encoder->dpb_count++];
    memset(entry, 0, sizeof(*entry));
    entry->frame_num = new_frame_num;
    entry->poc = new_poc;
    entry->is_reference = true;
    entry->is_long_term = false;
}

/*
 * apply_qp_override - test-only hook (BC250_FORCE_QP), same convention as
 * this file's other BC250_* debug env vars (BC250_FAST_MODE,
 * BC250_SLICES_PER_FRAME, BC250_DUMP_INPUT_FRAMES): lets a surgical
 * round-trip test drive the encoder at an exact, deterministic QP instead of
 * whatever the CBR/VBR rate_control.c feedback loop would pick (rate_control
 * clamps QP drift to +-2/frame around base_qp=26 and is bitrate-driven, so
 * hitting a specific QP like 12 or 51 through it deterministically on frame
 * 0 isn't otherwise possible). No effect unless BC250_FORCE_QP is set; does
 * not change any CAVLC/MVD/skip-decision logic, only which qp value those
 * paths are handed.
 */
static int apply_qp_override(int qp) {
    const char *force_qp_env = getenv("BC250_FORCE_QP");
    if (force_qp_env) {
        int forced = atoi(force_qp_env);
        if (forced >= 0 && forced <= 51) return forced;
    }
    return qp;
}

/*
 * write_aud - Writes Access Unit Delimiter (NAL type 9)
 * Essential for Sunshine / Moonlight / WebRTC to identify frame boundaries.
 */
static size_t write_aud(uint8_t *buf, size_t buf_size, bool is_idr) {
    if (buf_size < 6) return 0;
    buf[0] = 0x00;
    buf[1] = 0x00;
    buf[2] = 0x00;
    buf[3] = 0x01;
    buf[4] = 0x09; /* NAL header: forbidden=0, ref_idc=0, type=9 (AUD) */
    buf[5] = is_idr ? 0x10 : 0x30; /* primary_pic_type: 0 for I, 1 for P (shifted) + stop bit */
    return 6;
}

/*
 * maybe_append_filler - close the gap between what real coded slice data
 * (plus AUD/SPS/PPS) actually used and rate_control.c's per-frame target,
 * by appending a spec-defined filler_data_rbsp() NAL (bs_write_filler(),
 * NAL unit type 12) - but ONLY when the caller has signaled genuine CBR
 * intent (h264_encoder_set_cbr_intent()).
 *
 * This is the fix for the gap docs/rate_control_audit.md and the
 * fix(rate_control) commit before this one both documented and explicitly
 * left open: once rate_control.c's feedback loop drives QP down to its
 * floor (qp_min=12) and the content still doesn't need as many bits as a
 * high requested bitrate calls for, there was previously nothing to make
 * up the difference - the encoder just produced whatever bits the content
 * actually cost and stopped, so e.g. an 8 Mbps and a 20 Mbps request for
 * the same content converged to the exact same real output size. Filler
 * NALs are the standard way real encoders (x264 included - see
 * bs_write_filler()'s doc comment) manufacture the remaining bytes to
 * actually reach a constant-bitrate target.
 *
 * Deliberately NOT applied when cbr_intent is false (VBR, or a caller that
 * never set it): the same audit correctly identified that VBR content
 * legitimately using fewer bits than a loose ceiling is correct behavior,
 * not a bug, and padding it would manufacture bits nobody asked for.
 *
 * RC_LOW_LATENCY is treated the same as RC_CBR here (not excluded the way
 * true VBR is) - it's a tighter-buffer variant of hitting a bitrate target,
 * not a different rate-control philosophy, so a caller that asked for CBR
 * intent under low-latency mode should still get real bitrate-target
 * padding. See rc_init()'s call site for why this driver now defaults to
 * RC_LOW_LATENCY instead of RC_CBR.
 *
 * `total_written` on entry is the frame's real byte count so far (AUD +
 * SPS/PPS on IDR + all coded slice NAL(s)), already sitting in
 * encoder->output_buf. Returns the (possibly unchanged) new total_written;
 * never writes past encoder->output_buf_size.
 */
static size_t maybe_append_filler(h264_encoder_t *encoder, size_t total_written) {
    if (!encoder->cbr_intent ||
        (encoder->rc.mode != RC_CBR && encoder->rc.mode != RC_LOW_LATENCY)) {
        return total_written;
    }

    /* rate_control.c's target_bits_per_frame already accounts for
     * target_bitrate/framerate (and va_backend.c's target_percentage
     * scaling for whatever buffer last called h264_encoder_set_bitrate) -
     * reuse it directly rather than recomputing anything here. Round up:
     * an encoder that pads should err toward meeting the target, not
     * quietly falling half a byte short of it every frame. */
    uint32_t target_bytes = (encoder->rc.target_bits_per_frame + 7) / 8;
    if (target_bytes <= total_written) {
        return total_written; /* content already met or exceeded the target */
    }

    size_t shortfall = (size_t)target_bytes - total_written;

    /* A filler NAL's minimum possible size (zero 0xFF payload bytes) is
     * BS_FILLER_MIN_NAL_SIZE - see that macro's doc comment. A shortfall
     * smaller than that can't be closed without overshooting the target,
     * so it's left alone (this is a real, but small and expected, residual
     * - not the multi-Mbps gap this change targets). */
    if (shortfall < BS_FILLER_MIN_NAL_SIZE) {
        return total_written;
    }

    /* By construction, a filler NAL of exactly `shortfall` total bytes
     * needs (shortfall - BS_FILLER_MIN_NAL_SIZE) 0xFF payload bytes. */
    size_t filler_ff_count = shortfall - BS_FILLER_MIN_NAL_SIZE;

    if (total_written + shortfall > encoder->output_buf_size) {
        /* Not enough room in the frame's own scratch buffer - extremely
         * unlikely given output_buf_size's width*height*2+65536 sizing,
         * but fail safe (skip padding) rather than ever writing OOB or
         * emitting a truncated, non-byte-aligned filler NAL. */
        return total_written;
    }

    size_t written = bs_write_filler(encoder->output_buf + total_written,
                                     encoder->output_buf_size - total_written,
                                     filler_ff_count);
    return total_written + written;
}

/*
 * ============================================================================
 * Residual coefficient helpers (GPU staging buffer access, quantization,
 * Hadamard transforms, neighbor-context derivation)
 * ============================================================================
 */

/*
 * shadow_copy - bulk-copy one GPU-readback staging buffer into a persistent,
 * growable, normal malloc'd (cacheable) host buffer, returning a pointer to
 * the copy (or, on an allocation failure, to the original source - never
 * fails the encode over a perf-only optimization).
 *
 * PERF (found via on-target `BC250_PERF_STATS=1` timing added directly
 * inside h264_encoder_encode_frame's per-MB loop - real board measurement,
 * not a synthetic offline harness): the ~44-63us/macroblock CPU-side CAVLC
 * cost this project has been chasing is NOT dominated by entropy-coding bit
 * writes at all (batching cavlc.c's unary zero-bit writes into single
 * bs_write_u() calls - this branch's other commit - moved the board-measured
 * fps by under 0.6%). Per-MB timing brackets isolated the real cost: the
 * unconditional-per-MB skip decision (mb_has_any_luma_nonzero() +
 * mv_predictor(), called for EVERY macroblock before any entropy coding
 * happens at all) alone averaged ~36-38us/MB at every tested resolution
 * (640x480/1280x720/1920x1080) - a cost independent of resolution, which
 * rules out a working-set/cache-capacity explanation (a bigger frame's
 * larger buffer would show a worse, not identical, per-MB cost if this were
 * an ordinary cache-capacity effect) and points instead at a fixed
 * per-access latency.
 *
 * That fixed latency traces to gpu_compute.c's create_buffer_with_memory()
 * calls for quant_staging_buffers/coeff_staging_buffers/pred_mode_staging_
 * buffers/mv_staging_buffers: VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
 * VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, deliberately WITHOUT
 * VK_MEMORY_PROPERTY_HOST_CACHED_BIT (reasonable for the driver's own
 * one-shot vkCmdCopyBuffer GPU->host writes into them, which don't care
 * about CPU cacheability). On this hardware/driver that memory type is
 * uncached/write-combined from the CPU's read side - fine for a single
 * linear write, but every one of this file's per-MB, per-block, per-
 * neighbor reads (mb_has_any_luma_nonzero() rescanning up to 16 blocks,
 * cavlc_write_4x4_block()/_ac_block() reading the same coefficients again
 * a few lines later, mv_predictor()/neighbor_mv() re-touching adjacent
 * macroblocks' motion vectors) was a fresh scattered read of that same
 * uncached memory, over and over, for the whole frame.
 *
 * DO NOT DELETE THIS AS REDUNDANT. gpu_compute.c later started requesting
 * HOST_CACHED for those same staging buffers, which looks like it addresses
 * the identical root cause and makes this memcpy pure overhead. It does not -
 * the two fixes are complementary, measured as a 2x2 on real hardware at
 * 1440p (BC250_STAGING_CACHED x BC250_SHADOW_COPY, mean P-frame ms):
 *
 *            cached        uncached
 *   shadow    13.76         329.92     <- memcpy itself: 3.0ms vs 319.3ms
 *   no shadow 14.57         862.26     <- CAVLC: 5.6ms vs 857.1ms
 *
 * HOST_CACHED is what makes the bulk read affordable (319ms -> 3.0ms); this
 * function is what keeps the per-MB scattered reads off that memory at all
 * (a HOST_CACHED mapping still costs CAVLC 5.6 -> 13.1ms when read directly,
 * so the flag alone does not make the staging buffer behave like ordinary
 * cacheable RAM). Removing either one regresses; removing both is the
 * original ~860ms/frame pathology.
 *
 * This function turns that into ONE per-frame sequential streaming read
 * (the access pattern uncached/write-combined memory penalizes least) of
 * each GPU staging buffer into ordinary, cacheable malloc'd memory, done
 * once immediately after gpu_compute_get_*_staging_data() hands back the
 * raw mapped pointer and before ANY of this file's repeated-access CPU code
 * (the debug dumps, the skip decision, the CAVLC loop) touches it. Every
 * value is copied verbatim (memcpy, no transformation) - this changes WHERE
 * the CPU reads a byte from, never WHAT byte it reads, so it cannot change
 * the encoded bitstream in any way (see this change's commit message for
 * the byte-exact before/after board verification, the same methodology
 * used for the CAVLC bit-batching fix above).
 */
static const void *shadow_copy(void **shadow_ptr, size_t *cap, const void *src, size_t size) {
    if (!src || size == 0) return src;
    if (*cap < size) {
        void *newbuf = realloc(*shadow_ptr, size);
        if (!newbuf) return src; /* OOM: fall back to the raw (slower but correct) mapped pointer */
        *shadow_ptr = newbuf;
        *cap = size;
    }
    /* The source is mapped write-combining staging memory: an ordinary load
     * fetches part of a line at a time. Measured on a BC-250, this memcpy was
     * 14% of the H.264 encoder thread. MOVNTDQA reads a whole line into a fill
     * buffer, and degrades to an ordinary load if the memory turns out cached. */
    gpu_compute_copy_from_wc(*shadow_ptr, src, size);
    return *shadow_ptr;
}

/* quant_levels/coeff buffers are laid out as num_mbs*24*16 ints; block index
 * (0-23) is the GPU RASTER convention (see file-top comment), position (0-15)
 * is raster within the 4x4 block (index = row*4+col, NOT zigzag). */
static inline const int16_t *quant_block_ptr(const int16_t *quant_levels, uint32_t mb_idx, int raster_block) {
    return quant_levels + ((size_t)mb_idx * 24 + raster_block) * 16;
}
/* One 4x4 block's PRE-quantization DC term (what used to be
 * dc_coeff_at(dc_coeff, mb, blk)).
 *
 * PERF: every CPU read of the pre-quant coefficient buffer was position 0 of
 * some block - 24 of the 384 ints per macroblock, for the I16x16 luma DC and
 * the two chroma DC Hadamards - so the GPU now writes those values to a
 * compact num_mbs*24 buffer and only that crosses to the host. The full
 * coefficient buffer stays device-local (quantize.comp reads it as input,
 * reconstruct.comp reads it for its own DC path). At 1440p this drops 44.2 MB
 * of host-visible staging per encoder context, 22.1 MB of per-frame
 * GPU->host copy, and 22.1 MB of per-frame shadow_copy(). See
 * gpu_compute.h's dc_coeff_buffer. */
static inline int dc_coeff_at(const int *dc_coeff, uint32_t mb_idx, int raster_block) {
    return dc_coeff[(size_t)mb_idx * 24 + (uint32_t)raster_block];
}

/*
 * "Does this 4x4 block have a nonzero level anywhere in [start_pos, end_pos)?"
 *
 * PERF: `nz` is quantize.comp/intra_wavefront.comp's per-block nonzero bitmask
 * (see nc_ctx_t::nz_mask). When present this answers from one 4-byte mask read
 * instead of walking up to 16 ints out of the quant_levels readback, which at
 * 1440p is a 22MB buffer that a BC250_NZ_AUDIT run measured to be 89.8-95.8%
 * entirely-zero blocks - i.e. the overwhelmingly common answer to this
 * question was being paid for at full memory cost. The bit test is EQUIVALENT
 * to the scan, not an approximation: the shaders set bit p from the same
 * `level != 0` test on the same value this loop would read, and that
 * equivalence is checked block-for-block against a CPU recompute under
 * BC250_NZ_AUDIT=1.
 *
 * `nz == NULL` falls back to the scan, so nothing here depends on the mask
 * being available.
 */
static inline int block_any_nonzero(const int16_t *quant_levels, const uint32_t *nz,
                                     uint32_t mb_idx, int raster_block, int start_pos, int end_pos) {
    if (nz) {
        uint32_t mask = nz[(size_t)mb_idx * 24 + (uint32_t)raster_block] & 0xFFFFu;
        /* bits [start_pos, end_pos) */
        uint32_t range = ((end_pos >= 16) ? 0xFFFFu : ((1u << end_pos) - 1u)) & ~((1u << start_pos) - 1u);
        return (mask & range) != 0;
    }
    const int16_t *blk = quant_block_ptr(quant_levels, mb_idx, raster_block);
    for (int p = start_pos; p < end_pos; p++) if (blk[p] != 0) return 1;
    return 0;
}

/*
 * DC_LEVEL_SCALE0 - LevelScale4x4(qP%6, 0, 0) under the default (flat) scaling
 * list, i.e. 16 * V(qP%6, pos_type=0), where V is ITU-T H.264 Table 8-15's
 * "norm_adjust" matrix, position-type-0 (both-indices-even) column:
 * V(0..5,0) = {10,11,13,14,16,18} (cross-checked against x264's
 * common/tables.c dequant4_mf[][0] initializer and ffmpeg's
 * libavcodec/h264_ps.c / h264idct_template.c dequant tables - both reduce to
 * this same 16*V constant for the DC position). This is the SAME table
 * already used by this codebase's AC dequant at position 0
 * (quantize.comp's/reconstruct.comp's V[][0]) - luma DC (8.5.10) and chroma
 * DC (8.5.11.2) both scale by this exact value, they just apply a different
 * qP-dependent shift/rounding around it (see quantize_dc_luma/_chroma below).
 */
static const int DC_LEVEL_SCALE0[6] = {160, 176, 208, 224, 256, 288};

/*
 * quantize_dc_luma / quantize_dc_chroma - forward-quantize one
 * Hadamard-transformed DC coefficient.
 *
 * REPLACES the former "DELIBERATE SIMPLIFICATION" quantize_dc(), which
 * reused the AC quantizer's MF0/shift-by-(15+qp/6) formula for DC too. That
 * was confirmed wrong: real decoders (and ITU-T H.264 8.5.10/8.5.11.2)
 * dequantize DC coefficients with a DC-specific piecewise formula built
 * around DC_LEVEL_SCALE0 above, not the AC MF/V multiplier pairing at all.
 * Verified against x264's actual common/quant.c:
 *
 *   dequant_4x4_dc() (luma I16x16 DC, 8.5.10):
 *     qbits = qp/6 - 6
 *     qbits >= 0 (qp >= 36): recovered = c * (LS << qbits)
 *     qbits <  0 (qp <  36): recovered = (c * LS + (1 << (-qbits-1))) >> -qbits
 *   dequant_2x2_dc() (chroma DC, 8.5.11.2 - a DIFFERENT function, not a
 *   reuse of the luma one with a different rounding offset):
 *     qbits = qp/6 - 5
 *     qbits >= 0 (qp >= 30): recovered = c * (LS << qbits)
 *     qbits <  0 (qp <  30): recovered = (c * LS) >> -qbits   (NO rounding
 *                                          term added in this branch - the
 *                                          spec text and x264 both omit it
 *                                          here, unlike the luma DC low-QP
 *                                          branch above)
 *   where LS = DC_LEVEL_SCALE0[qp%6] in both cases.
 *
 * So chroma DC differs from luma DC in TWO ways, not one: a different
 * qP>=30 (vs qP>=36) branch threshold, and no rounding addend on the low-QP
 * side - not just "the same table with a different f". Both branches above
 * are the same underlying relationship recovered ~= c * LS * 2^(qp/6 - K)
 * (K=6 luma, K=5 chroma), just written as a left-shift-after-multiply
 * (qp>=6K... i.e. high QP) or a multiply-then-round-right-shift (low QP) for
 * integer-friendliness - not two structurally different formulas.
 *
 * These functions are near-inverses of that real dequant relationship
 * (choose the integer level c minimizing error against
 * recovered = c * LS * 2^(qp/6-K), i.e. c = round(v * 2^(K-qp/6) / LS)),
 * with two encoder-side (non-normative) design choices layered on top:
 *
 * 1. Rounding: plain round-to-nearest (bias = denom/2) rather than this
 *    file's AC quantizer's asymmetric intra=1/3 / inter=1/6 "dead-zone"
 *    bias. That asymmetric bias is a rate-distortion tuning choice (bias
 *    towards transmitting zero) which trades away worst-case reconstruction
 *    accuracy - fine for AC energy, but directly counterproductive for DC,
 *    where the goal (per the surgical round-trip validation this fix is
 *    judged on) is minimizing |recovered - original|, and round-to-nearest
 *    roughly halves the worst-case error the dead-zone bias would otherwise
 *    leave on the table. Neither choice affects CAVLC bitstream validity,
 *    only decoded numeric accuracy. No separate intra/inter parameter is
 *    needed here as a result (unlike the AC path).
 *
 * 2. quantize_dc_luma's K is 5, NOT 6, even though dequant_dc_luma's real,
 *    decoder-matching K is 6 - see quantize_dc_luma's own comment for why
 *    this asymmetry is required (it compensates for this codebase's
 *    specific luma_dc_hadamard() forward-transform gain, found empirically
 *    via a board round-trip test after the K=6/K=6 symmetric version
 *    clipped every reconstructed pixel to white regardless of QP).
 *    quantize_dc_chroma has no such adjustment (K=5 both directions,
 *    symmetric) - chroma_dc_hadamard's forward gain already matches what's
 *    needed, verified the same way.
 *
 * Validated empirically via a surgical encode/ffmpeg-decode round-trip
 * across qp in {12,24,26,30,36,40,51} (spanning both branches of both
 * dequant functions) - see commit message.
 */
/*
 * quantize_dc_luma's forward threshold is 5, NOT 6, even though
 * dequant_dc_luma (below) correctly uses 6 to match real decoders. This is
 * NOT a typo/copy-paste of quantize_dc_chroma - it is required, and was
 * found empirically (see commit message for the full derivation and the
 * board round-trip numbers that exposed it): this codebase's OWN
 * luma_dc_hadamard()/luma_dc_hadamard_inv() pair (untouched, out of scope
 * for this fix) is not gain-neutral the way a textbook normalized Hadamard
 * would be. luma_dc_hadamard's forward pass includes exactly ONE
 * normalizing ">>1" (only in its second/final stage), so a constant 4x4
 * input of per-block-DC value X concentrates to a single Hadamard-domain
 * value of 8*X (verified by hand-tracing the butterfly for constant input,
 * and confirmed by the board test below) - i.e. HALF the 16*X gain a fully
 * unnormalized 4x4 Hadamard would give, since real reference Hadamard
 * transforms (e.g. x264's dct4x4dc()) apply NO normalization anywhere in
 * the forward pass. luma_dc_hadamard_inv (decode-side, correctly
 * unnormalized to match real decoders) does not undo any of that gain - it
 * just replicates a DC-only level to all 16 positions unchanged. Meanwhile
 * the value dequant_dc_luma() must hand to inverse_4x4() needs to be on the
 * SAME scale a plain dequant_ac() coefficient would be for the identical
 * raw value (since inverse_4x4/dequant_ac is the shared, already-correct
 * AC path used verbatim by P16x16 luma, which has no DC/AC split at all) -
 * and MF[qp_rem][0]*LevelScale4x4(qp_rem,0,0) is, by a well-known H.264
 * design property, approximately 2^21 for every qp_rem (13107*160=2097120
 * vs 2^21=2097152, a <0.002% design-rounding gap - confirmed by direct
 * computation), making that AC round-trip's coefficient gain a QP-
 * independent constant of exactly 4x the raw value. So the DC path needs
 * dc_val = 4 * (raw per-block DC), but the composed
 * quantize(spec-K=6)+dequant(spec-K=6) pair here would instead deliver
 * 8 * (raw per-block DC) - a 2x-too-large value that clips reconstruction
 * to white regardless of QP (exactly what an unfixed board round-trip
 * test showed: every QP from 12 to 51 decoded pure white/255 for a flat
 * non-128 macroblock). A forward-quantizer near-inverse threshold of 5
 * (dividing by exactly double what threshold-6 would) exactly cancels that
 * extra 2x, independent of qp_per, while dequant_dc_luma stays at the
 * spec-correct threshold 6 that a real decoder actually applies - this
 * asymmetry is deliberate, not an oversight. (chroma_dc_hadamard's forward
 * gain for constant input is only 4x - already equal to the AC-path
 * target - so quantize_dc_chroma below needs no such adjustment and stays
 * symmetric with dequant_dc_chroma at threshold 5.) Confirmed empirically:
 * after this fix, the same surgical round-trip decodes within a few
 * pixels of the source value at low/mid QP - see commit message.
 */
static int quantize_dc_luma(int v, int qp) {
    int qp_per = qp / 6;
    int qp_rem = qp % 6;
    int64_t LS = DC_LEVEL_SCALE0[qp_rem];
    int sign = (v < 0) ? -1 : 1;
    int64_t av = (v < 0) ? -(int64_t)v : (int64_t)v;

    int64_t numer_av, denom;
    if (qp_per >= 5) {
        numer_av = av;
        denom = LS << (qp_per - 5);
    } else {
        numer_av = av << (5 - qp_per);
        denom = LS;
    }
    int64_t level = (numer_av + denom / 2) / denom;
    return (int)(sign * level);
}

/* ITU-T H.264 Table 8-15: QPc = f(qPI), qPI = Clip3(-QpBdOffsetC, 51, QPy +
 * chroma_qp_index_offset). This project always signals
 * chroma_qp_index_offset = 0 (bitstream.c) and is 8-bit-only
 * (QpBdOffsetC = 0), so qPI reduces to QPy clamped to [0,51] (already true
 * by construction). Kept in sync with the identical copies in
 * quantize.comp/reconstruct.comp/intra_wavefront.comp - see those for the
 * full derivation and the real-hardware measurement (~13dB of chroma PSNR
 * against a real decode at QP=42, luma unaffected) that this fixes. */
static int chroma_qp(int qpy) {
    if (qpy < 30) return qpy;
    static const int table[22] = {29,30,31,32,32,33,34,34,35,35,36,36,37,37,37,38,38,38,39,39,39,39};
    int idx = qpy - 30;
    return table[idx < 22 ? idx : 21];
}

static int quantize_dc_chroma(int v, int qp) {
    int qp_per = qp / 6;
    int qp_rem = qp % 6;
    int64_t LS = DC_LEVEL_SCALE0[qp_rem];
    int sign = (v < 0) ? -1 : 1;
    int64_t av = (v < 0) ? -(int64_t)v : (int64_t)v;

    int64_t numer_av, denom;
    if (qp_per >= 5) {
        numer_av = av;
        denom = LS << (qp_per - 5);
    } else {
        numer_av = av << (5 - qp_per);
        denom = LS;
    }
    int64_t level = (numer_av + denom / 2) / denom;
    return (int)(sign * level);
}

/* Forward Hadamard transform of the 16 luma DC coefficients (one MB's worth),
 * standard x264-style 2-pass butterfly construction - implemented exactly as
 * specified (the write pattern into tmp[] transposes implicitly, so this is
 * the usual "rows, transpose, columns" 2D Hadamard without an explicit
 * transpose step). dc_in/dc_out are in raster (row*4+col) order. */
static void luma_dc_hadamard(const int dc_in[4][4], int dc_out[16]) {
    int tmp[16];
    for (int i = 0; i < 4; i++) {
        int s01 = dc_in[i][0] + dc_in[i][1], d01 = dc_in[i][0] - dc_in[i][1];
        int s23 = dc_in[i][2] + dc_in[i][3], d23 = dc_in[i][2] - dc_in[i][3];
        tmp[0*4+i] = s01+s23; tmp[1*4+i] = s01-s23; tmp[2*4+i] = d01-d23; tmp[3*4+i] = d01+d23;
    }
    for (int i = 0; i < 4; i++) {
        int s01 = tmp[i*4+0]+tmp[i*4+1], d01 = tmp[i*4+0]-tmp[i*4+1];
        int s23 = tmp[i*4+2]+tmp[i*4+3], d23 = tmp[i*4+2]-tmp[i*4+3];
        dc_out[i*4+0] = (s01+s23+1)>>1; dc_out[i*4+1] = (s01-s23+1)>>1;
        dc_out[i*4+2] = (d01-d23+1)>>1; dc_out[i*4+3] = (d01+d23+1)>>1;
    }
}

/*
 * h264_intra16_luma_dc_transform - forward Hadamard, quantize, and transpose
 * the 16 luma DC coefficients of one Intra16x16 macroblock into the array
 * layout cavlc_write_4x4_block() (and a real decoder's inverse-scan) expect.
 *
 * BUG FIX (commit d95b840, "transpose Intra16x16 luma DC array before CAVLC
 * - closes remaining luma corruption"): luma_dc_hadamard()'s two-pass
 * butterfly structure (see its own doc comment: the write pattern into
 * tmp[] transposes implicitly, so this is the usual "rows, transpose,
 * columns" 2D Hadamard without an explicit transpose step) means
 * dc_out_raw[i*4+j] is NOT simply "the Hadamard-domain value at row i,
 * column j" the way a plain 2D transform's output would be. That is
 * harmless for the GPU reconstruction path (reconstruct.comp /
 * intra_wavefront.comp independently re-derive the same forward Hadamard
 * from the same coeff buffer and apply their own inverse using the
 * identical raster block index both times, so their round trip is
 * self-consistent by construction), but it is NOT harmless here: this
 * array is serialized into the actual bitstream (zigzag-scanned via
 * cavlc_write_4x4_block), and a real, independent decoder inverse-scans and
 * inverse-Hadamards it assuming the natural (non-transposed) row/column
 * labeling - which the raw Hadamard output does not have. A byte-level "DC
 * shuffle" test (one macroblock, 16 sub-blocks each a distinct known flat
 * shade) showed every one of the 12 off-diagonal sub-blocks decoding, via
 * ffmpeg on real hardware, to EXACTLY the value belonging at its
 * (col,row)-transposed position before this fix - invisible on flat
 * content (a transpose of a constant is itself), but the dominant cause of
 * quality_test.sh's luma-specific corruption on real, spatially-varying
 * content. The transpose below is the fix. Chroma needs no equivalent:
 * chroma_dc_hadamard() is a single symmetric closed-form 2x2 transform with
 * no two-pass butterfly (hence no implicit transpose), and
 * cavlc_write_chroma_dc_block() scans its 4 values directly in row-major
 * order with no zigzag step.
 *
 * Exposed (non-static) so unit tests can exercise this pure-math path in
 * isolation without a GPU context - see tests/test_encode.c's
 * test_intra16_dc_transpose() regression test. encode_mb_i16x16() below is
 * the only production caller.
 *
 * @param dc_in                16 pre-quant luma DC values (one per luma 4x4
 *                             sub-block), raster (row*4+col) order.
 * @param qp                   Quantization parameter for this macroblock.
 * @param dc_out               Output: quantized DC array in the natural
 *                             row/column order CAVLC/a real decoder expect.
 * @param dc_out_pretranspose  Optional (may be NULL): filled with the
 *                             quantized array BEFORE the transpose fix is
 *                             applied (i.e. raw luma_dc_hadamard() output
 *                             order) - for regression testing only; no
 *                             production caller needs this.
 */
void h264_intra16_luma_dc_transform(const int dc_in[4][4], int qp,
                                     int dc_out[16], int dc_out_pretranspose[16]) {
    int dc_out_raw[16];
    luma_dc_hadamard(dc_in, dc_out_raw);
    int dc_out_natural[16];
    for (int i = 0; i < 16; i++) dc_out_natural[i] = quantize_dc_luma(dc_out_raw[i], qp);

    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            dc_out[r * 4 + c] = dc_out_natural[c * 4 + r];

    if (dc_out_pretranspose) memcpy(dc_out_pretranspose, dc_out_natural, sizeof(dc_out_natural));
}

/* Forward Hadamard transform of a 2x2 chroma DC block. c[]/out[] are raster
 * (TL,TR,BL,BR) order, matching the GPU Cb/Cr DC block layout. */
static void chroma_dc_hadamard(const int c[4], int out[4]) {
    int c00 = c[0], c01 = c[1], c10 = c[2], c11 = c[3];
    out[0] = c00 + c01 + c10 + c11;
    out[1] = c00 - c01 + c10 - c11;
    out[2] = c00 + c01 - c10 - c11;
    out[3] = c00 - c01 - c10 + c11;
}

/* nC derivation for a luma 4x4 block at spec blkIdx, per ITU-T 9.2.1. */
static int luma_nc(const nc_ctx_t *nc, uint32_t mb, uint32_t mbx, uint32_t mby, int blk_idx) {
    int x4 = blk_x4[blk_idx], y4 = blk_y4[blk_idx];
    int nA = -1, nB = -1;

    if (x4 > 0) {
        nA = nc->nz_luma[mb][spec_blk_idx_from_xy[y4][x4 - 1]];
    } else if (mbx > 0 && (mb - 1) >= nc->start_mb) {
        nA = nc->nz_luma[mb - 1][spec_blk_idx_from_xy[y4][3]];
    }

    if (y4 > 0) {
        nB = nc->nz_luma[mb][spec_blk_idx_from_xy[y4 - 1][x4]];
    } else if (mby > 0 && (mb - nc->width_in_mbs) >= nc->start_mb) {
        nB = nc->nz_luma[mb - nc->width_in_mbs][spec_blk_idx_from_xy[3][x4]];
    }

    if (nA >= 0 && nB >= 0) return (nA + nB + 1) >> 1;
    if (nA >= 0) return nA;
    if (nB >= 0) return nB;
    return 0;
}

/* nC derivation for a chroma 4x4 block (2x2 grid, blk_idx 0..3, raster==spec order). */
static int chroma_nc(uint8_t (*nz_c)[4], uint32_t mb, uint32_t mbx, uint32_t mby,
                      uint32_t width_in_mbs, uint32_t start_mb, int blk_idx) {
    int x2 = blk_idx % 2, y2 = blk_idx / 2;
    int nA = -1, nB = -1;

    if (x2 > 0) {
        nA = nz_c[mb][y2 * 2 + (x2 - 1)];
    } else if (mbx > 0 && (mb - 1) >= start_mb) {
        nA = nz_c[mb - 1][y2 * 2 + 1];
    }

    if (y2 > 0) {
        nB = nz_c[mb][(y2 - 1) * 2 + x2];
    } else if (mby > 0 && (mb - width_in_mbs) >= start_mb) {
        nB = nz_c[mb - width_in_mbs][1 * 2 + x2];
    }

    if (nA >= 0 && nB >= 0) return (nA + nB + 1) >> 1;
    if (nA >= 0) return nA;
    if (nB >= 0) return nB;
    return 0;
}

/*
 * nC derivation for the I16x16 luma DC block.
 *
 * CORRECTED (found via ffmpeg round-trip validation - this directly
 * contradicts what an earlier version of this file, and its originating
 * task brief, assumed): ITU-T H.264 9.2.1 does NOT give the luma DC block
 * its own independent whole-MB neighbor chain. Per spec (and confirmed by
 * ffmpeg's libavcodec/h264_cavlc.c decode_residual(), which for the DC
 * block calls `pred_non_zero_count(h, sl, (n-LUMA_DC_BLOCK_INDEX)*16)` -
 * i.e. index 0 - while STORING the DC block's own decoded TotalCoeff at a
 * completely different cache slot reserved for LUMA_DC_BLOCK_INDEX), the DC
 * block's nC is derived EXACTLY as if it were luma4x4BlkIdx 0 (i.e. the
 * same left/top neighbor derivation as the first luma AC block), and the
 * DC block's own TotalCoeff is never itself used as anyone's neighbor
 * value - it is immediately superseded once AC block 0 of the same MB is
 * decoded (which writes ITS total_coeff to the real block-0 slot). Since
 * this encoder always writes the DC block before any AC block of the same
 * MB, nz_luma[mb][0] at DC-encode time still holds whatever the LEFT/TOP
 * NEIGHBOR MB left there, exactly matching this rule - so DC's nC is simply
 * `luma_nc(nc, mb, mbx, mby, blk_idx=0)`, and no separate dc_nz array is
 * needed at all (a prior version of this file added one; it produced a
 * bitstream ffmpeg rejected with "negative number of zero coeffs" on any
 * macroblock with a nonzero-DC-total_coeff neighbor, traced and confirmed
 * via an independent reference CAVLC decoder cross-checked against this
 * exact ffmpeg source function).
 */

/* Real per-MB motion vector, as written back by mv_staging (see
 * gpu_compute_get_mv_staging_data()'s doc comment) - mirrors the GPU's
 * std430 MotionVector struct {ivec2 mv; uint sad;} byte-for-byte (16 bytes:
 * two int32 + one uint32 + 4 bytes of std430 struct-alignment padding).
 *
 * UNITS: mvx/mvy are in QUARTER-LUMA-SAMPLE units (motion_estimation.comp's
 * integer-pel diamond search result, refined to half-pel then quarter-pel
 * via the real H.264 8.4.2.2.1 interpolation filter - see that shader's
 * top-of-file comment). This is the ONE consistent unit used everywhere
 * this struct's values flow: mv_predictor()/neighbor_mv()'s median
 * predictor, encode_mb_p16x16()'s MVD, the P_Skip "matches predictor" check
 * in h264_encoder_encode_frame(), and residual_predict.comp's P-slice
 * prediction (which reads the same mv_buffer values directly, no CPU-side
 * rescaling in between). Defined as gpu_mv_t in gpu_compute.h. */

/*
 * gpu_pred_mode_i16 - I16x16 prediction mode, as chosen by
 * residual_predict.comp's real SAD-based mode decision (DC/Vertical/
 * Horizontal/Plane against actual neighbor pixels).
 *
 * REPLACES a former CPU-side heuristic that inferred mode from post-quant
 * AC coefficient activity - that approach became structurally impossible
 * once residual generation itself needs to know the prediction mode BEFORE
 * DCT/quantize even run (the residual IS source-minus-prediction). Mode
 * decision now happens on the GPU, before DCT, directly from pixel-domain
 * SAD; this just reads back what it decided. See residual_predict.comp's
 * top-of-file comment for the full design, including the neighbor-pixel-
 * source and slice-boundary simplifications it documents (this readback
 * inherits both: it does not re-derive or gate on start_mb the way the old
 * heuristic did, because the GPU's mode decision already didn't either).
 */
static int gpu_pred_mode_i16(const uint32_t *pred_modes, uint32_t mb_idx) {
    return pred_modes ? (int)(pred_modes[mb_idx] & 0x3u) : H264_I16x16_DC;
}

/* Real per-MB chroma intra prediction mode (ITU-T 8.3.4/Table 8-3), packed by
 * residual_predict.comp into bits 2-3 of the same pred_modes[] word the luma
 * I16x16 mode (bits 0-1) already uses - see that shader's chroma mode-
 * decision comment for why chroma needs its own real SAD-based mode decision
 * (DC-only unconditionally produced badly wrong chroma for any macroblock
 * whose top spatial neighbor is very different content from its own - the
 * root cause of the gradient-boundary row artifact this fixes). */
static int gpu_chroma_pred_mode(const uint32_t *pred_modes, uint32_t mb_idx) {
    return pred_modes ? (int)((pred_modes[mb_idx] >> 2) & 0x3u) : H264_CHROMA_DC;
}

int h264_sanitize_i16_mode(int mode, bool top_avail, bool left_avail) {
    if (!top_avail && !left_avail) {
        return H264_I16x16_DC;
    }
    if (!top_avail && (mode == H264_I16x16_VERT || mode == H264_I16x16_PLANE)) {
        return left_avail ? H264_I16x16_HORIZ : H264_I16x16_DC;
    }
    if (!left_avail && (mode == H264_I16x16_HORIZ || mode == H264_I16x16_PLANE)) {
        return top_avail ? H264_I16x16_VERT : H264_I16x16_DC;
    }
    return mode;
}

int h264_sanitize_chroma_mode(int mode, bool top_avail, bool left_avail) {
    if (!top_avail && !left_avail) {
        return H264_CHROMA_DC;
    }
    if (!top_avail && (mode == H264_CHROMA_VERT || mode == H264_CHROMA_PLANE)) {
        return left_avail ? H264_CHROMA_HORIZ : H264_CHROMA_DC;
    }
    if (!left_avail && (mode == H264_CHROMA_HORIZ || mode == H264_CHROMA_PLANE)) {
        return top_avail ? H264_CHROMA_VERT : H264_CHROMA_DC;
    }
    return mode;
}

/* Neighbor MV lookup for the P16x16 MVD predictor below: (dx,dy) is a
 * neighbor offset in MB units (e.g. left=(-1,0), top=(0,-1)). Unavailable
 * (off-picture, or belongs to an earlier slice) is reported via *avail. */
static void neighbor_mv(const gpu_mv_t *mvs, uint32_t mbx, uint32_t mby,
                         uint32_t width_in_mbs, uint32_t start_mb, int dx, int dy,
                         int *mvx, int *mvy, bool *avail) {
    int nbx = (int)mbx + dx, nby = (int)mby + dy;
    if (nbx < 0 || nby < 0 || (uint32_t)nbx >= width_in_mbs) { *avail = false; *mvx = 0; *mvy = 0; return; }
    uint32_t nb = (uint32_t)nby * width_in_mbs + (uint32_t)nbx;
    if (nb < start_mb) { *avail = false; *mvx = 0; *mvy = 0; return; }
    *avail = true;
    *mvx = mvs[nb].mvx;
    *mvy = mvs[nb].mvy;
}

static inline int median3(int a, int b, int c) {
    return a + b + c - (a < b ? (a < c ? a : c) : (b < c ? b : c)) - (a > b ? (a > c ? a : c) : (b > c ? b : c));
}

/*
 * mv_predictor - ITU-T H.264 8.4.1.3 median motion vector predictor (A=left,
 * B=top, C=top-right, substituting D=top-left when C is unavailable). This
 * MUST match a real decoder's predictor exactly (ffmpeg included) - the
 * encoder transmits mv-minus-predictor (MVD) and the decoder reconstructs
 * mv=predictor+MVD using its OWN predictor computed the same way, so any
 * mismatch here corrupts every subsequent motion vector, not just this one.
 * All MBs in a P slice are P16x16 in this encoder (no intra-in-P mixing), so
 * the spec's ref-idx-equality special cases never apply here.
 */
static void mv_predictor(const gpu_mv_t *mvs, uint32_t mbx, uint32_t mby,
                          uint32_t width_in_mbs, uint32_t start_mb, int *px, int *py) {
    int ax, ay, bx, by, cx, cy;
    bool a_ok, b_ok, c_ok;
    neighbor_mv(mvs, mbx, mby, width_in_mbs, start_mb, -1, 0, &ax, &ay, &a_ok);   /* A: left */
    neighbor_mv(mvs, mbx, mby, width_in_mbs, start_mb, 0, -1, &bx, &by, &b_ok);   /* B: top */
    neighbor_mv(mvs, mbx, mby, width_in_mbs, start_mb, 1, -1, &cx, &cy, &c_ok);   /* C: top-right */
    if (!c_ok) {
        neighbor_mv(mvs, mbx, mby, width_in_mbs, start_mb, -1, -1, &cx, &cy, &c_ok); /* D substitutes C */
    }

    if (!b_ok && !c_ok && a_ok) {
        *px = ax; *py = ay;
        return;
    }
    /* An unavailable neighbor contributes (0,0) to the median (per spec) once
     * the single-predictor special case above doesn't apply. */
    if (!a_ok) { ax = 0; ay = 0; }
    if (!b_ok) { bx = 0; by = 0; }
    if (!c_ok) { cx = 0; cy = 0; }
    *px = median3(ax, bx, cx);
    *py = median3(ay, by, cy);
}

/*
 * skip_mv_predictor - ITU-T H.264 8.4.1.1 "Derivation process for luma
 * motion vector prediction for skipped macroblocks in P slices". This is a
 * DIFFERENT rule from mv_predictor()'s plain 8.4.1.3 median, and a real
 * decoder uses THIS rule - not 8.4.1.3 - to reconstruct the motion vector of
 * any macroblock whose mb_skip_flag/P_Skip run says "skipped".
 *
 * Per spec: both components of mvL0 are forced to (0,0) if EITHER of the
 * left (A) or top (B) neighbors is unavailable, OR if an available A or B
 * has refIdxL0==0 and mvL0==(0,0). This project always uses a single
 * reference frame and never mixes Intra into P slices (see mv_predictor()'s
 * doc comment), so every available P-slice neighbor always has refIdxL0==0
 * - the spec's "refIdxL0==0 && mv==(0,0)" condition therefore reduces to
 * simply "mv==(0,0)" here. Only when NONE of the zero-forcing conditions
 * hold does 8.4.1.1 fall back to the plain 8.4.1.3 median (mv_predictor()).
 *
 * THE BUG THIS FIXES: prior to this function existing, the P_Skip
 * legality check in h264_encoder_encode_frame() (both the CAVLC
 * mb_skip_run path and the CABAC mb_skip_flag path) compared the real
 * searched motion vector against mv_predictor()'s plain median - not this
 * zero-forcing rule - to decide whether a macroblock could legally be
 * skipped. Whenever the zero-forcing condition actually applied (e.g. this
 * MB's left or top neighbor itself has zero motion, or is off-picture) but
 * the plain median happened to be nonzero AND equal to this MB's own real
 * searched motion, the old check wrongly certified the MB as skip-legal.
 * The bitstream then encoded it as skip, but a real decoder - applying THIS
 * zero-forcing rule - reconstructs mvL0=(0,0), not the plain median the
 * encoder matched against. That silently diverges the decoder's motion (and,
 * because mv_predictor() unconditionally treats every neighbor's raw
 * committed mv[] as truth without re-deriving whatever a decoder would
 * actually have reconstructed for a skipped neighbor, this wrong value then
 * poisons the median-of-neighbors predictor of every later macroblock that
 * looks at this one as A, B, C, or D) - even though the transmitted MVD of
 * every OTHER, explicitly-coded macroblock was always individually correct
 * in isolation. Confirmed via BC250_DEBUG_MV_ROW cross-referenced against
 * ffmpeg's own decoded per-MB motion vectors (-debug mv / codecview=mv=pf):
 * see docs/DEVLOG.md for the concrete before/after macroblock trace.
 */
static void skip_mv_predictor_ex(const gpu_mv_t *mvs, uint32_t mbx, uint32_t mby,
                                  uint32_t width_in_mbs, uint32_t start_mb, int *px, int *py,
                                  bool *out_zero_force) {
    int ax, ay, bx, by;
    bool a_ok, b_ok;
    neighbor_mv(mvs, mbx, mby, width_in_mbs, start_mb, -1, 0, &ax, &ay, &a_ok);  /* A: left */
    neighbor_mv(mvs, mbx, mby, width_in_mbs, start_mb, 0, -1, &bx, &by, &b_ok);  /* B: top */

    bool zero_force = !a_ok || !b_ok || (a_ok && ax == 0 && ay == 0) || (b_ok && bx == 0 && by == 0);
    if (out_zero_force) *out_zero_force = zero_force;
    if (zero_force) {
        *px = 0; *py = 0;
        return;
    }
    mv_predictor(mvs, mbx, mby, width_in_mbs, start_mb, px, py);
}

static void skip_mv_predictor(const gpu_mv_t *mvs, uint32_t mbx, uint32_t mby,
                               uint32_t width_in_mbs, uint32_t start_mb, int *px, int *py) {
    skip_mv_predictor_ex(mvs, mbx, mby, width_in_mbs, start_mb, px, py, NULL);
}

/* Whole-MB "does this P16x16 MB have any nonzero luma coefficient" skip
 * decision. Previously this read the lossy packed entropy summary
 * (mb_blocks[b]&0xFF); it now reads the real quant_levels data directly -
 * strictly more accurate, same semantic heuristic. */
static int mb_has_any_luma_nonzero(const int16_t *quant_levels, const uint32_t *nz, uint32_t mb_idx) {
    for (int b = 0; b < 16; b++) {
        if (block_any_nonzero(quant_levels, nz, mb_idx, b, 0, 16)) return 1;
    }
    return 0;
}

/* Whole-MB "does this P16x16 MB have any nonzero CHROMA coefficient" -
 * the skip decision's missing other half (see mb_has_any_luma_nonzero()
 * above and its two call sites' doc comments). A P_Skip macroblock per
 * ITU-T 8.4.1.1 carries ZERO residual for the ENTIRE macroblock, chroma
 * included - not just luma. Before this function existed, the skip
 * decision at both call sites checked luma alone, so any MB with a
 * nonzero chroma residual but a zero luma residual (and an MV matching
 * the predictor) was still wrongly certified skip-legal, silently
 * dropping its real chroma correction from the transmitted bitstream.
 *
 * That mismatch is invisible to this encoder itself: gpu_compute.c's
 * reconstruct.comp shader (which builds the GPU-side reference image
 * used for every LATER frame's motion search and skip decisions) applies
 * the full chroma residual unconditionally, independent of what the CPU
 * later decides to transmit. So the encoder's own future-frame reference
 * silently keeps the "corrected" chroma that was never actually sent to
 * the real decoder - meaning the real client's chroma is now wrong, but
 * this encoder's own quant_levels for that position keep computing a
 * near-zero residual on subsequent frames too (relative to its own,
 * already-"corrected" internal reference), so the missing correction is
 * never retransmitted. This is a one-way, compounding, chroma-only drift
 * that only a future IDR (full intra, no skip) can reset - matching the
 * real-client symptom of a live stream's color slowly collapsing to
 * grayscale/incorrect color over a GOP, then resetting at the next IDR.
 *
 * Mirrors the exact same chroma-DC-Hadamard-then-nonzero-check /
 * chroma-AC-nonzero-check used everywhere else in this file to derive
 * cbp_chroma (see encode_mb_p16x16/encode_mb_i16x16 and their CABAC
 * counterparts) - this is not a new heuristic, just applying the same
 * existing test to the skip decision. */
static int mb_has_any_chroma_nonzero(const int16_t *quant_levels, const int *dc_coeff,
                                      const uint32_t *nz, uint32_t mb_idx) {
    int cb_dc_raw[4], cr_dc_raw[4];
    for (int i = 0; i < 4; i++) cb_dc_raw[i] = dc_coeff_at(dc_coeff, mb_idx, 16 + i);
    for (int i = 0; i < 4; i++) cr_dc_raw[i] = dc_coeff_at(dc_coeff, mb_idx, 20 + i);
    int cb_dc[4], cr_dc[4];
    chroma_dc_hadamard(cb_dc_raw, cb_dc);
    chroma_dc_hadamard(cr_dc_raw, cr_dc);
    for (int i = 0; i < 4; i++) if (cb_dc[i] != 0 || cr_dc[i] != 0) return 1;

    for (int b = 16; b < 24; b++) {
        if (block_any_nonzero(quant_levels, nz, mb_idx, b, 1, 16)) return 1;
    }
    return 0;
}

/*
 * encode_mb_i16x16 - Encode one Intra 16x16 macroblock: header, luma DC
 * (Hadamard), luma AC (16 blocks, spec order), chroma DC (Hadamard x2) and
 * chroma AC (8 blocks), updating the nC neighbor-context arrays as it goes.
 */
static void encode_mb_i16x16(bitstream_t *bs, const int16_t *quant_levels, const int *dc_coeff,
                              const uint32_t *pred_modes,
                              uint32_t mb, uint32_t mbx, uint32_t mby, nc_ctx_t *nc, int qp) {
    bool left_avail = (mbx > 0 && (mb - 1) >= nc->start_mb);
    bool top_avail  = (mby > 0 && (mb - nc->width_in_mbs) >= nc->start_mb);
    int raw_luma_mode = gpu_pred_mode_i16(pred_modes, mb);
    int raw_chroma_mode = gpu_chroma_pred_mode(pred_modes, mb);
    int pred_mode = h264_sanitize_i16_mode(raw_luma_mode, top_avail, left_avail);
    int chroma_pred_mode = h264_sanitize_chroma_mode(raw_chroma_mode, top_avail, left_avail);
    {
        const char *dbg = getenv("BC250_DEBUG_I16_MB");
        if (dbg && (uint32_t)atoi(dbg) == mb) {
            fprintf(stderr, "[BC250_DEBUG_I16_MB] mb=%u mbx=%u mby=%u raw_pred_modes=%u luma_mode=%d chroma_mode=%d\n",
                    mb, mbx, mby, pred_modes ? pred_modes[mb] : 0xFFFFFFFFu, pred_mode, chroma_pred_mode);
        }
    }

    /* Luma DC: gather PRE-quant DC (coeff buffer, position 0) of the 16
     * raster blocks into the natural 4x4 grid, then forward-Hadamard,
     * quantize, and transpose into the row/column order CAVLC (and a real
     * decoder) expect - see h264_intra16_luma_dc_transform()'s doc comment
     * for why the transpose is required (commit d95b840 fix). */
    int dc_in[4][4];
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            dc_in[r][c] = dc_coeff_at(dc_coeff, mb, r * 4 + c);
    int dc_out[16];
    h264_intra16_luma_dc_transform(dc_in, qp, dc_out, NULL);

    /* cbp_luma: any nonzero AC (raster positions 1..15) across all 16 luma blocks. */
    int cbp_luma_flag = 0;
    for (int blk = 0; blk < 16 && !cbp_luma_flag; blk++) {
        if (block_any_nonzero(quant_levels, nc->nz_mask, mb, blk, 1, 16)) cbp_luma_flag = 1;
    }

    /* Chroma DC (Hadamard) + cbp_chroma. */
    int cb_dc_raw[4], cr_dc_raw[4];
    for (int i = 0; i < 4; i++) cb_dc_raw[i] = dc_coeff_at(dc_coeff, mb, 16 + i);
    for (int i = 0; i < 4; i++) cr_dc_raw[i] = dc_coeff_at(dc_coeff, mb, 20 + i);
    int cb_dc[4], cr_dc[4];
    chroma_dc_hadamard(cb_dc_raw, cb_dc);
    chroma_dc_hadamard(cr_dc_raw, cr_dc);
    for (int i = 0; i < 4; i++) { cb_dc[i] = quantize_dc_chroma(cb_dc[i], chroma_qp(qp)); cr_dc[i] = quantize_dc_chroma(cr_dc[i], chroma_qp(qp)); }

    {
        const char *dbg = getenv("BC250_DEBUG_I16_MB");
        if (dbg && (uint32_t)atoi(dbg) == mb) {
            fprintf(stderr, "[BC250_DEBUG_I16_MB] qp=%d cb_dc_raw=(%d,%d,%d,%d) cr_dc_raw=(%d,%d,%d,%d) cb_dc_tx=(%d,%d,%d,%d) cr_dc_tx=(%d,%d,%d,%d)\n",
                    qp, cb_dc_raw[0],cb_dc_raw[1],cb_dc_raw[2],cb_dc_raw[3],
                    cr_dc_raw[0],cr_dc_raw[1],cr_dc_raw[2],cr_dc_raw[3],
                    cb_dc[0],cb_dc[1],cb_dc[2],cb_dc[3], cr_dc[0],cr_dc[1],cr_dc[2],cr_dc[3]);
            for (int blk = 16; blk < 24; blk++) {
                const int16_t *b = quant_block_ptr(quant_levels, mb, blk);
                fprintf(stderr, "[BC250_DEBUG_I16_MB]   quant_block=%d dc=%d vals=[%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d]\n",
                        blk, b[0], b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7],b[8],b[9],b[10],b[11],b[12],b[13],b[14],b[15]);
            }
        }
    }

    int chroma_dc_nonzero = 0;
    for (int i = 0; i < 4; i++) if (cb_dc[i] != 0 || cr_dc[i] != 0) { chroma_dc_nonzero = 1; break; }
    int chroma_ac_nonzero = 0;
    for (int blk = 16; blk < 24 && !chroma_ac_nonzero; blk++) {
        if (block_any_nonzero(quant_levels, nc->nz_mask, mb, blk, 1, 16)) chroma_ac_nonzero = 1;
    }
    int cbp_chroma = chroma_ac_nonzero ? 2 : (chroma_dc_nonzero ? 1 : 0);

    /* Header first (mb_type encodes pred_mode/cbp_chroma/cbp_luma_flag,
     * followed by intra_chroma_pred_mode and mb_qp_delta), THEN residual. */
    cavlc_write_mb_i16x16_header(bs, pred_mode, chroma_pred_mode, cbp_chroma, cbp_luma_flag ? 15 : 0, 0);

    /* Luma DC block (always present for I16x16, first in residual order).
     * nC uses luma4x4BlkIdx=0's neighbor chain (see dc_nc's replacement
     * comment above) - this MUST run before the AC loop below writes
     * nz_luma[mb][0], since the DC block's own nC needs to see the LEFT/TOP
     * NEIGHBOR MB's block-0-adjacent values, not this MB's own (not-yet-
     * decoded) block 0. */
    int nC_dc = luma_nc(nc, mb, mbx, mby, 0);
    /* dc_out is the Hadamard-transformed luma DC (plain int[16], computed
     * above by h264_intra16_luma_dc_transform) - narrow to int16_t only for
     * this call, rather than changing that function's output type, for the
     * same reason cabac_write_quant_block() exists: keep this one CAVLC
     * call site's needs from rippling into a shared computation. */
    int16_t dc_out16[16];
    for (int i = 0; i < 16; i++) dc_out16[i] = (int16_t)dc_out[i];
    cavlc_write_4x4_block(bs, dc_out16, nC_dc);

    /* Luma AC blocks, spec blkIdx order (0..15). */
    if (cbp_luma_flag) {
        for (int blk_idx = 0; blk_idx < 16; blk_idx++) {
            int raster = gpu_raster_block_idx(blk_idx);
            int nC = luma_nc(nc, mb, mbx, mby, blk_idx);
            int tc = cavlc_write_4x4_ac_block(bs, quant_block_ptr(quant_levels, mb, raster), nC);
            nc->nz_luma[mb][blk_idx] = (uint8_t)tc;
        }
    } else {
        memset(nc->nz_luma[mb], 0, sizeof(nc->nz_luma[mb]));
    }

    /* Chroma DC (Cb then Cr) if any chroma residual at all. */
    if (cbp_chroma >= 1) {
        cavlc_write_chroma_dc_block(bs, cb_dc);
        cavlc_write_chroma_dc_block(bs, cr_dc);
    }

    /* Chroma AC (4 Cb blocks then 4 Cr blocks) only if cbp_chroma==2. */
    if (cbp_chroma == 2) {
        for (int blk_idx = 0; blk_idx < 4; blk_idx++) {
            int nC = chroma_nc(nc->nz_cb, mb, mbx, mby, nc->width_in_mbs, nc->start_mb, blk_idx);
            int tc = cavlc_write_4x4_ac_block(bs, quant_block_ptr(quant_levels, mb, 16 + blk_idx), nC);
            nc->nz_cb[mb][blk_idx] = (uint8_t)tc;
        }
        for (int blk_idx = 0; blk_idx < 4; blk_idx++) {
            int nC = chroma_nc(nc->nz_cr, mb, mbx, mby, nc->width_in_mbs, nc->start_mb, blk_idx);
            int tc = cavlc_write_4x4_ac_block(bs, quant_block_ptr(quant_levels, mb, 20 + blk_idx), nC);
            nc->nz_cr[mb][blk_idx] = (uint8_t)tc;
        }
    } else {
        memset(nc->nz_cb[mb], 0, sizeof(nc->nz_cb[mb]));
        memset(nc->nz_cr[mb], 0, sizeof(nc->nz_cr[mb]));
    }
}

/*
 * encode_mb_p16x16 - Encode one Inter P_L0_16x16 macroblock: header (MVD +
 * real 6-bit CBP), luma (16 FULL blocks, DC included, spec order), chroma DC
 * (Hadamard, same as I16x16) and chroma AC (8 blocks).
 */
static void encode_mb_p16x16(bitstream_t *bs, const int16_t *quant_levels, const int *dc_coeff,
                              const gpu_mv_t *mvs,
                              uint32_t mb, uint32_t mbx, uint32_t mby, nc_ctx_t *nc, int qp) {
    int pred_x, pred_y;
    mv_predictor(mvs, mbx, mby, nc->width_in_mbs, nc->start_mb, &pred_x, &pred_y);
    /* UNITS (see motion_estimation.comp's sub-pel refinement comment and
     * gpu_mv_t's doc comment above): mvs[].mvx/mvy are now real QUARTER-
     * LUMA-SAMPLE-precision motion vectors, computed by an integer-pel
     * diamond search followed by a half-pel-then-quarter-pel refinement
     * pass, and stored in quarter-pel units directly - the SAME units ITU-T
     * H.264 7.4.5.3 requires mvd_l0[][][compIdx] to be coded in. mv_predictor()'s
     * median-of-neighbors predictor operates on those same quarter-pel
     * gpu_mv_t values, so pred_x/pred_y are quarter-pel too - the difference
     * below needs NO further scaling (a prior version of this fix scaled by
     * 4 here to correct for the search being integer-pel-only at the time;
     * that scaling is now WRONG and would quadruple every real motion
     * vector's magnitude, since mvs[] itself already carries the fractional
     * precision - see git history for that superseded fix's rationale). The
     * legacy CPU-heuristic path lower in this file (h264_encoder_encode_raw)
     * is a separate, self-contained integer-pel-only path that does its own
     * `best_dx * 4` scaling and does NOT read mvs[] - it is unaffected by
     * this change. */
    int mvd_x = mvs[mb].mvx - pred_x;
    int mvd_y = mvs[mb].mvy - pred_y;

    /* Luma CBP: one bit per 8x8 quadrant (spec blkIdx/4), based on full
     * 16-coefficient (DC+AC) nonzero-anywhere check since P16x16 luma has
     * no separate DC/AC split. */
    int luma_cbp = 0;
    for (int q = 0; q < 4; q++) {
        int any = 0;
        for (int sub = 0; sub < 4 && !any; sub++) {
            int blk_idx = q * 4 + sub;
            int raster = gpu_raster_block_idx(blk_idx);
            if (block_any_nonzero(quant_levels, nc->nz_mask, mb, raster, 0, 16)) any = 1;
        }
        if (any) luma_cbp |= (1 << q);
    }

    /* Chroma DC (Hadamard, inter (/6) rounding) + cbp_chroma - identical
     * structure to encode_mb_i16x16's chroma handling. */
    int cb_dc_raw[4], cr_dc_raw[4];
    for (int i = 0; i < 4; i++) cb_dc_raw[i] = dc_coeff_at(dc_coeff, mb, 16 + i);
    for (int i = 0; i < 4; i++) cr_dc_raw[i] = dc_coeff_at(dc_coeff, mb, 20 + i);
    int cb_dc[4], cr_dc[4];
    chroma_dc_hadamard(cb_dc_raw, cb_dc);
    chroma_dc_hadamard(cr_dc_raw, cr_dc);
    for (int i = 0; i < 4; i++) { cb_dc[i] = quantize_dc_chroma(cb_dc[i], chroma_qp(qp)); cr_dc[i] = quantize_dc_chroma(cr_dc[i], chroma_qp(qp)); }

    int chroma_dc_nonzero = 0;
    for (int i = 0; i < 4; i++) if (cb_dc[i] != 0 || cr_dc[i] != 0) { chroma_dc_nonzero = 1; break; }
    int chroma_ac_nonzero = 0;
    for (int blk = 16; blk < 24 && !chroma_ac_nonzero; blk++) {
        if (block_any_nonzero(quant_levels, nc->nz_mask, mb, blk, 1, 16)) chroma_ac_nonzero = 1;
    }
    int cbp_chroma = chroma_ac_nonzero ? 2 : (chroma_dc_nonzero ? 1 : 0);

    /* Standard H.264 CodedBlockPattern = CBPChroma*16 + CBPLuma; cavlc_write_mb_p16x16_header
     * maps this through map_inter_cbp()/Table 9-4 internally. */
    int cbp = (luma_cbp & 0xF) | (cbp_chroma << 4);
    cavlc_write_mb_p16x16_header(bs, mvd_x, mvd_y, cbp, 0);

    for (int blk_idx = 0; blk_idx < 16; blk_idx++) {
        int q = blk_idx / 4;
        if (luma_cbp & (1 << q)) {
            int raster = gpu_raster_block_idx(blk_idx);
            int nC = luma_nc(nc, mb, mbx, mby, blk_idx);
            int tc = cavlc_write_4x4_block(bs, quant_block_ptr(quant_levels, mb, raster), nC);
            nc->nz_luma[mb][blk_idx] = (uint8_t)tc;
        } else {
            nc->nz_luma[mb][blk_idx] = 0;
        }
    }

    if (cbp_chroma >= 1) {
        cavlc_write_chroma_dc_block(bs, cb_dc);
        cavlc_write_chroma_dc_block(bs, cr_dc);
    }
    if (cbp_chroma == 2) {
        for (int blk_idx = 0; blk_idx < 4; blk_idx++) {
            int nC = chroma_nc(nc->nz_cb, mb, mbx, mby, nc->width_in_mbs, nc->start_mb, blk_idx);
            int tc = cavlc_write_4x4_ac_block(bs, quant_block_ptr(quant_levels, mb, 16 + blk_idx), nC);
            nc->nz_cb[mb][blk_idx] = (uint8_t)tc;
        }
        for (int blk_idx = 0; blk_idx < 4; blk_idx++) {
            int nC = chroma_nc(nc->nz_cr, mb, mbx, mby, nc->width_in_mbs, nc->start_mb, blk_idx);
            int tc = cavlc_write_4x4_ac_block(bs, quant_block_ptr(quant_levels, mb, 20 + blk_idx), nC);
            nc->nz_cr[mb][blk_idx] = (uint8_t)tc;
        }
    } else {
        memset(nc->nz_cb[mb], 0, sizeof(nc->nz_cb[mb]));
        memset(nc->nz_cr[mb], 0, sizeof(nc->nz_cr[mb]));
    }
}

/*
 * ============================================================================
 * CABAC macroblock encoding
 * ============================================================================
 * Control flow independently written (see cabac.h's top-of-file provenance
 * comment) to this project's own call shape, mirroring encode_mb_i16x16()/
 * encode_mb_p16x16() above so the two entropy coders plug into the same GPU
 * readback data the same way. The actual context-index/ctxIdxInc formulas
 * these call into (cabac_write_*() in cabac.c) are adapted from x264.
 *
 * Neighbor-context helpers below parallel luma_nc()/chroma_nc() above but
 * derive a binary coded_block_flag (ITU-T 9.3.3.1.1.9) instead of CAVLC's
 * averaged nC, with the "neighbor unavailable" default depending on whether
 * the CURRENT macroblock is intra (b_intra) - unlike nC's default of 0.
 */
static void luma_cbf_neighbors(const nc_ctx_t *nc, uint32_t mb, uint32_t mbx, uint32_t mby,
                                int blk_idx, bool b_intra, int *nA, int *nB) {
    int x4 = blk_x4[blk_idx], y4 = blk_y4[blk_idx];
    int def = b_intra ? 1 : 0;

    if (x4 > 0) {
        *nA = nc->nz_luma[mb][spec_blk_idx_from_xy[y4][x4 - 1]] ? 1 : 0;
    } else if (mbx > 0 && (mb - 1) >= nc->start_mb) {
        *nA = nc->nz_luma[mb - 1][spec_blk_idx_from_xy[y4][3]] ? 1 : 0;
    } else {
        *nA = def;
    }

    if (y4 > 0) {
        *nB = nc->nz_luma[mb][spec_blk_idx_from_xy[y4 - 1][x4]] ? 1 : 0;
    } else if (mby > 0 && (mb - nc->width_in_mbs) >= nc->start_mb) {
        *nB = nc->nz_luma[mb - nc->width_in_mbs][spec_blk_idx_from_xy[3][x4]] ? 1 : 0;
    } else {
        *nB = def;
    }
}

static void chroma_cbf_neighbors(uint8_t (*nz_c)[4], uint32_t mb, uint32_t mbx, uint32_t mby,
                                  uint32_t width_in_mbs, uint32_t start_mb, int blk_idx, bool b_intra,
                                  int *nA, int *nB) {
    int x2 = blk_idx % 2, y2 = blk_idx / 2;
    int def = b_intra ? 1 : 0;

    if (x2 > 0) {
        *nA = nz_c[mb][y2 * 2 + (x2 - 1)] ? 1 : 0;
    } else if (mbx > 0 && (mb - 1) >= start_mb) {
        *nA = nz_c[mb - 1][y2 * 2 + 1] ? 1 : 0;
    } else {
        *nA = def;
    }

    if (y2 > 0) {
        *nB = nz_c[mb][(y2 - 1) * 2 + x2] ? 1 : 0;
    } else if (mby > 0 && (mb - width_in_mbs) >= start_mb) {
        *nB = nz_c[mb - width_in_mbs][1 * 2 + x2] ? 1 : 0;
    } else {
        *nB = def;
    }
}

/* DC-block coded_block_flag neighbor derivation (ITU-T 9.3.3.1.1.9,
 * ctxBlockCat 0/3): unlike the AC/4x4 sub-block case above, this is a
 * simple whole-macroblock lookup - the spec's transBlockA/B for a DC block
 * is just "the DC block of the immediate left/top neighbor MB", with no
 * luma4x4BlkIdx-style sub-position math (DC is a single per-MB/per-component
 * entity, not a 4x4 grid). */
static void dc_cbf_neighbors(const uint8_t *dc_store, uint32_t mb, uint32_t mbx, uint32_t mby,
                              uint32_t width_in_mbs, uint32_t start_mb, bool b_intra, int *nA, int *nB) {
    int def = b_intra ? 1 : 0;
    *nA = (mbx > 0 && (mb - 1) >= start_mb) ? (dc_store[mb - 1] ? 1 : 0) : def;
    *nB = (mby > 0 && (mb - width_in_mbs) >= start_mb) ? (dc_store[mb - width_in_mbs] ? 1 : 0) : def;
}

static void chroma_dc_cbf_neighbors(uint8_t (*store)[2], int comp, uint32_t mb, uint32_t mbx, uint32_t mby,
                                     uint32_t width_in_mbs, uint32_t start_mb, bool b_intra, int *nA, int *nB) {
    int def = b_intra ? 1 : 0;
    *nA = (mbx > 0 && (mb - 1) >= start_mb) ? (store[mb - 1][comp] ? 1 : 0) : def;
    *nB = (mby > 0 && (mb - width_in_mbs) >= start_mb) ? (store[mb - width_in_mbs][comp] ? 1 : 0) : def;
}

/* H.264 zigzag scan order for a 4x4 block (ITU-T Figure 8-8) - identical
 * table to cavlc.c's own (private, static) copy; duplicated here rather than
 * exposed via a header since it's a small spec-mandated constant, not
 * project-specific logic. */
static const int cabac_zigzag_4x4[16] = {
     0,  1,  4,  8,
     5,  2,  3,  6,
     9, 12, 13, 10,
     7, 11, 14, 15
};

/* Encode one residual block: coded_block_flag, then (if set) the residual
 * itself. `raw` is the FULL 16-value raster block (or, for chroma DC, the
 * 4-value raw block) in the same convention this file's other helpers use;
 * `skip_dc` requests the AC-only 15-coefficient scan (zigzag_4x4[1..15]),
 * matching cavlc_write_4x4_ac_block()'s convention. Returns the cbf value
 * (for the caller's own nz_luma/nz_cb/nz_cr/dc_cbf_* bookkeeping). */
static bool cabac_write_block(cabac_engine_t *cb, cabac_ctx_block_cat_t cat,
                               int nA, int nB, const int *raw, bool skip_dc) {
    int scanned[16];
    int n = cabac_count_coeffs(cat);
    if (cat == CABAC_CAT_CHROMA_DC) {
        for (int i = 0; i < n; i++) scanned[i] = raw[i]; /* no zigzag for 2x2 chroma DC */
    } else if (skip_dc) {
        for (int i = 0; i < n; i++) scanned[i] = raw[cabac_zigzag_4x4[i + 1]];
    } else {
        for (int i = 0; i < n; i++) scanned[i] = raw[cabac_zigzag_4x4[i]];
    }
    bool nz = false;
    for (int i = 0; i < n; i++) if (scanned[i] != 0) { nz = true; break; }
    cabac_write_coded_block_flag(cb, cat, nA, nB, nz);
    if (nz) cabac_write_residual_block(cb, cat, scanned);
    return nz;
}

/* cabac_write_block() takes `const int *raw` because its callers are a
 * genuine mix: DC-Hadamard results (dc_out/cb_dc/cr_dc, plain local `int`
 * arrays) alongside quant_block_ptr()'s int16_t-backed pointers into the
 * quant_levels readback. Rather than change cabac_write_block()'s shared
 * signature - which would ripple into the Hadamard-result call sites too,
 * for no benefit there - this is a thin widen-and-forward wrapper used only
 * at the quant_block_ptr() call sites. The 16-int scratch copy is noise
 * next to everything else in this pipeline; the point is keeping the
 * shared function's contract simple and not touching the DC-Hadamard
 * variables' types at all. */
static bool cabac_write_quant_block(cabac_engine_t *cb, cabac_ctx_block_cat_t cat,
                                     int nA, int nB, const int16_t *raw16, bool skip_dc) {
    int raw[16];
    for (int i = 0; i < 16; i++) raw[i] = raw16[i];
    return cabac_write_block(cb, cat, nA, nB, raw, skip_dc);
}

/*
 * encode_mb_i16x16_cabac - CABAC equivalent of encode_mb_i16x16(): mb_type
 * (with embedded cbp/pred-mode), intra_chroma_pred_mode, mb_qp_delta,
 * luma DC/AC and chroma DC/AC residual, updating this encoder's CABAC
 * neighbor-context arrays (dc_cbf_luma/dc_cbf_chroma/nz_luma/nz_cb/nz_cr)
 * as it goes - mirrors encode_mb_i16x16()'s structure and comments exactly,
 * see that function for the GPU-buffer-layout rationale shared by both.
 */
static void encode_mb_i16x16_cabac(cabac_engine_t *cb, h264_encoder_t *encoder,
                                    const int16_t *quant_levels, const int *dc_coeff,
                                    const uint32_t *pred_modes,
                                    uint32_t mb, uint32_t mbx, uint32_t mby, nc_ctx_t *nc, int qp,
                                    bool *last_dqp_nonzero) {
    bool left_avail = (mbx > 0 && (mb - 1) >= nc->start_mb);
    bool top_avail  = (mby > 0 && (mb - nc->width_in_mbs) >= nc->start_mb);
    int raw_luma_mode = gpu_pred_mode_i16(pred_modes, mb);
    int pred_mode = h264_sanitize_i16_mode(raw_luma_mode, top_avail, left_avail);

    int dc_in[4][4];
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            dc_in[r][c] = dc_coeff_at(dc_coeff, mb, r * 4 + c);
    int dc_out[16];
    h264_intra16_luma_dc_transform(dc_in, qp, dc_out, NULL);

    int cbp_luma_flag = 0;
    for (int blk = 0; blk < 16 && !cbp_luma_flag; blk++) {
        if (block_any_nonzero(quant_levels, nc->nz_mask, mb, blk, 1, 16)) cbp_luma_flag = 1;
    }

    int cb_dc_raw[4], cr_dc_raw[4];
    for (int i = 0; i < 4; i++) cb_dc_raw[i] = dc_coeff_at(dc_coeff, mb, 16 + i);
    for (int i = 0; i < 4; i++) cr_dc_raw[i] = dc_coeff_at(dc_coeff, mb, 20 + i);
    int cb_dc[4], cr_dc[4];
    chroma_dc_hadamard(cb_dc_raw, cb_dc);
    chroma_dc_hadamard(cr_dc_raw, cr_dc);
    for (int i = 0; i < 4; i++) { cb_dc[i] = quantize_dc_chroma(cb_dc[i], chroma_qp(qp)); cr_dc[i] = quantize_dc_chroma(cr_dc[i], chroma_qp(qp)); }

    int chroma_dc_nonzero = 0;
    for (int i = 0; i < 4; i++) if (cb_dc[i] != 0 || cr_dc[i] != 0) { chroma_dc_nonzero = 1; break; }
    int chroma_ac_nonzero = 0;
    for (int blk = 16; blk < 24 && !chroma_ac_nonzero; blk++) {
        if (block_any_nonzero(quant_levels, nc->nz_mask, mb, blk, 1, 16)) chroma_ac_nonzero = 1;
    }
    int cbp_chroma = chroma_ac_nonzero ? 2 : (chroma_dc_nonzero ? 1 : 0);

    /* mb_type (ITU-T 9.3.3.1.1.3): ctxIdxInc counts available {left,top}
     * neighbors that are NOT I_NxN. This project's I-slices are
     * homogeneously I_16x16 (never I_NxN/I_PCM), so this reduces to a plain
     * available-neighbor count. */
    int ctx_intra = (mbx > 0 && (mb - 1) >= nc->start_mb ? 1 : 0) +
                    (mby > 0 && (mb - nc->width_in_mbs) >= nc->start_mb ? 1 : 0);
    cabac_write_mb_type_i16x16(cb, ctx_intra, cbp_luma_flag != 0, cbp_chroma, pred_mode);

    /* intra_chroma_pred_mode: this project always transmits mode 0 (DC) -
     * see cavlc_write_mb_i16x16_header()'s hardcoded 0 - so every possible
     * neighbor also always has mode 0, making the ctxIdxInc formula's
     * "neighbor's mode != 0" test always false regardless of availability;
     * ctx is therefore always 0 without needing a dedicated neighbor-mode
     * tracking array. */
    cabac_write_intra_chroma_pred_mode(cb, 0, 0);

    *last_dqp_nonzero = cabac_write_qp_delta(cb, 0, *last_dqp_nonzero);

    /* Luma DC (always present for I16x16, always intra-defaulted). */
    int nA, nB;
    dc_cbf_neighbors(encoder->dc_cbf_luma, mb, mbx, mby, nc->width_in_mbs, nc->start_mb, true, &nA, &nB);
    bool dc_nz = cabac_write_block(cb, CABAC_CAT_LUMA_DC, nA, nB, dc_out, false);
    encoder->dc_cbf_luma[mb] = dc_nz ? 1 : 0;

    /* Luma AC blocks, spec blkIdx order (0..15). */
    if (cbp_luma_flag) {
        for (int blk_idx = 0; blk_idx < 16; blk_idx++) {
            int raster = gpu_raster_block_idx(blk_idx);
            luma_cbf_neighbors(nc, mb, mbx, mby, blk_idx, true, &nA, &nB);
            bool nz = cabac_write_quant_block(cb, CABAC_CAT_LUMA_AC, nA, nB,
                                         quant_block_ptr(quant_levels, mb, raster), true);
            nc->nz_luma[mb][blk_idx] = nz ? 1 : 0;
        }
    } else {
        memset(nc->nz_luma[mb], 0, sizeof(nc->nz_luma[mb]));
    }

    /* Chroma DC (Cb then Cr) if any chroma residual at all. */
    if (cbp_chroma >= 1) {
        chroma_dc_cbf_neighbors(encoder->dc_cbf_chroma, 0, mb, mbx, mby, nc->width_in_mbs, nc->start_mb, true, &nA, &nB);
        bool cb_nz = cabac_write_block(cb, CABAC_CAT_CHROMA_DC, nA, nB, cb_dc, false);
        chroma_dc_cbf_neighbors(encoder->dc_cbf_chroma, 1, mb, mbx, mby, nc->width_in_mbs, nc->start_mb, true, &nA, &nB);
        bool cr_nz = cabac_write_block(cb, CABAC_CAT_CHROMA_DC, nA, nB, cr_dc, false);
        encoder->dc_cbf_chroma[mb][0] = cb_nz ? 1 : 0;
        encoder->dc_cbf_chroma[mb][1] = cr_nz ? 1 : 0;
    } else {
        encoder->dc_cbf_chroma[mb][0] = 0;
        encoder->dc_cbf_chroma[mb][1] = 0;
    }

    /* Chroma AC (4 Cb blocks then 4 Cr blocks) only if cbp_chroma==2. */
    if (cbp_chroma == 2) {
        for (int blk_idx = 0; blk_idx < 4; blk_idx++) {
            chroma_cbf_neighbors(nc->nz_cb, mb, mbx, mby, nc->width_in_mbs, nc->start_mb, blk_idx, true, &nA, &nB);
            bool nz = cabac_write_quant_block(cb, CABAC_CAT_CHROMA_AC, nA, nB,
                                         quant_block_ptr(quant_levels, mb, 16 + blk_idx), true);
            nc->nz_cb[mb][blk_idx] = nz ? 1 : 0;
        }
        for (int blk_idx = 0; blk_idx < 4; blk_idx++) {
            chroma_cbf_neighbors(nc->nz_cr, mb, mbx, mby, nc->width_in_mbs, nc->start_mb, blk_idx, true, &nA, &nB);
            bool nz = cabac_write_quant_block(cb, CABAC_CAT_CHROMA_AC, nA, nB,
                                         quant_block_ptr(quant_levels, mb, 20 + blk_idx), true);
            nc->nz_cr[mb][blk_idx] = nz ? 1 : 0;
        }
    } else {
        memset(nc->nz_cb[mb], 0, sizeof(nc->nz_cb[mb]));
        memset(nc->nz_cr[mb], 0, sizeof(nc->nz_cr[mb]));
    }
}

/*
 * encode_mb_p16x16_cabac - CABAC equivalent of encode_mb_p16x16(), called
 * only for a macroblock the caller has already decided is NOT skipped (see
 * h264_encoder_encode_frame()'s CABAC P-slice loop for the skip decision and
 * the mb_skip_flag / mb_type prefix bins, which are coded by the caller
 * before this function runs, matching cavlc_write_p_skip_run()'s equivalent
 * split in the CAVLC path).
 */
static void encode_mb_p16x16_cabac(cabac_engine_t *cb, h264_encoder_t *encoder,
                                    const int16_t *quant_levels, const int *dc_coeff,
                                    const gpu_mv_t *mvs,
                                    uint32_t mb, uint32_t mbx, uint32_t mby, nc_ctx_t *nc, int qp,
                                    bool *last_dqp_nonzero) {
    int pred_x, pred_y;
    mv_predictor(mvs, mbx, mby, nc->width_in_mbs, nc->start_mb, &pred_x, &pred_y);
    int mvd_x = mvs[mb].mvx - pred_x;
    int mvd_y = mvs[mb].mvy - pred_y;

    int luma_cbp = 0;
    for (int q = 0; q < 4; q++) {
        int any = 0;
        for (int sub = 0; sub < 4 && !any; sub++) {
            int blk_idx = q * 4 + sub;
            int raster = gpu_raster_block_idx(blk_idx);
            if (block_any_nonzero(quant_levels, nc->nz_mask, mb, raster, 0, 16)) any = 1;
        }
        if (any) luma_cbp |= (1 << q);
    }

    int cb_dc_raw[4], cr_dc_raw[4];
    for (int i = 0; i < 4; i++) cb_dc_raw[i] = dc_coeff_at(dc_coeff, mb, 16 + i);
    for (int i = 0; i < 4; i++) cr_dc_raw[i] = dc_coeff_at(dc_coeff, mb, 20 + i);
    int cb_dc[4], cr_dc[4];
    chroma_dc_hadamard(cb_dc_raw, cb_dc);
    chroma_dc_hadamard(cr_dc_raw, cr_dc);
    for (int i = 0; i < 4; i++) { cb_dc[i] = quantize_dc_chroma(cb_dc[i], chroma_qp(qp)); cr_dc[i] = quantize_dc_chroma(cr_dc[i], chroma_qp(qp)); }

    int chroma_dc_nonzero = 0;
    for (int i = 0; i < 4; i++) if (cb_dc[i] != 0 || cr_dc[i] != 0) { chroma_dc_nonzero = 1; break; }
    int chroma_ac_nonzero = 0;
    for (int blk = 16; blk < 24 && !chroma_ac_nonzero; blk++) {
        if (block_any_nonzero(quant_levels, nc->nz_mask, mb, blk, 1, 16)) chroma_ac_nonzero = 1;
    }
    int cbp_chroma = chroma_ac_nonzero ? 2 : (chroma_dc_nonzero ? 1 : 0);

    cabac_write_mb_type_p_l0_16x16(cb);

    /* mvd_l0 (ITU-T 9.3.3.1.1.7): ctxIdxInc from the clip-summed abs(mvd) of
     * the left/top neighbor for each component - unavailable or skipped
     * neighbors contribute 0 (their stored mvd_x_abs/mvd_y_abs is 0, since
     * this array is either never-written-this-slice (calloc'd 0 at create,
     * and skip MBs explicitly zero it below) or a real coded value). */
    int left_x = (mbx > 0 && (mb - 1) >= nc->start_mb) ? encoder->mvd_x_abs[mb - 1] : 0;
    int top_x  = (mby > 0 && (mb - nc->width_in_mbs) >= nc->start_mb) ? encoder->mvd_x_abs[mb - nc->width_in_mbs] : 0;
    int left_y = (mbx > 0 && (mb - 1) >= nc->start_mb) ? encoder->mvd_y_abs[mb - 1] : 0;
    int top_y  = (mby > 0 && (mb - nc->width_in_mbs) >= nc->start_mb) ? encoder->mvd_y_abs[mb - nc->width_in_mbs] : 0;
    int ctx_x = cabac_mvd_ctx_from_neighbors(left_x, top_x);
    int ctx_y = cabac_mvd_ctx_from_neighbors(left_y, top_y);
    int abs_x = cabac_write_mvd_component(cb, 0, ctx_x, mvd_x);
    int abs_y = cabac_write_mvd_component(cb, 1, ctx_y, mvd_y);
    encoder->mvd_x_abs[mb] = (uint8_t)abs_x;
    encoder->mvd_y_abs[mb] = (uint8_t)abs_y;

    int cbp_l = (mbx > 0 && (mb - 1) >= nc->start_mb) ? encoder->cbp_nb[mb - 1] : -1;
    int cbp_t = (mby > 0 && (mb - nc->width_in_mbs) >= nc->start_mb) ? encoder->cbp_nb[mb - nc->width_in_mbs] : -1;
    cabac_write_cbp_luma(cb, luma_cbp, cbp_l, cbp_t);
    cabac_write_cbp_chroma(cb, cbp_chroma, cbp_l, cbp_t);
    encoder->cbp_nb[mb] = (int16_t)((cbp_chroma << 4) | (luma_cbp & 0xF));

    if (luma_cbp != 0 || cbp_chroma != 0) {
        *last_dqp_nonzero = cabac_write_qp_delta(cb, 0, *last_dqp_nonzero);
    }

    int nA, nB;
    for (int blk_idx = 0; blk_idx < 16; blk_idx++) {
        int q = blk_idx / 4;
        if (luma_cbp & (1 << q)) {
            int raster = gpu_raster_block_idx(blk_idx);
            luma_cbf_neighbors(nc, mb, mbx, mby, blk_idx, false, &nA, &nB);
            bool nz = cabac_write_quant_block(cb, CABAC_CAT_LUMA_4x4, nA, nB,
                                         quant_block_ptr(quant_levels, mb, raster), false);
            nc->nz_luma[mb][blk_idx] = nz ? 1 : 0;
        } else {
            nc->nz_luma[mb][blk_idx] = 0;
        }
    }

    if (cbp_chroma >= 1) {
        chroma_dc_cbf_neighbors(encoder->dc_cbf_chroma, 0, mb, mbx, mby, nc->width_in_mbs, nc->start_mb, false, &nA, &nB);
        bool cb_nz = cabac_write_block(cb, CABAC_CAT_CHROMA_DC, nA, nB, cb_dc, false);
        chroma_dc_cbf_neighbors(encoder->dc_cbf_chroma, 1, mb, mbx, mby, nc->width_in_mbs, nc->start_mb, false, &nA, &nB);
        bool cr_nz = cabac_write_block(cb, CABAC_CAT_CHROMA_DC, nA, nB, cr_dc, false);
        encoder->dc_cbf_chroma[mb][0] = cb_nz ? 1 : 0;
        encoder->dc_cbf_chroma[mb][1] = cr_nz ? 1 : 0;
    } else {
        encoder->dc_cbf_chroma[mb][0] = 0;
        encoder->dc_cbf_chroma[mb][1] = 0;
    }
    if (cbp_chroma == 2) {
        for (int blk_idx = 0; blk_idx < 4; blk_idx++) {
            chroma_cbf_neighbors(nc->nz_cb, mb, mbx, mby, nc->width_in_mbs, nc->start_mb, blk_idx, false, &nA, &nB);
            bool nz = cabac_write_quant_block(cb, CABAC_CAT_CHROMA_AC, nA, nB,
                                         quant_block_ptr(quant_levels, mb, 16 + blk_idx), true);
            nc->nz_cb[mb][blk_idx] = nz ? 1 : 0;
        }
        for (int blk_idx = 0; blk_idx < 4; blk_idx++) {
            chroma_cbf_neighbors(nc->nz_cr, mb, mbx, mby, nc->width_in_mbs, nc->start_mb, blk_idx, false, &nA, &nB);
            bool nz = cabac_write_quant_block(cb, CABAC_CAT_CHROMA_AC, nA, nB,
                                         quant_block_ptr(quant_levels, mb, 20 + blk_idx), true);
            nc->nz_cr[mb][blk_idx] = nz ? 1 : 0;
        }
    } else {
        memset(nc->nz_cb[mb], 0, sizeof(nc->nz_cb[mb]));
        memset(nc->nz_cr[mb], 0, sizeof(nc->nz_cr[mb]));
    }
}

h264_encoder_t *h264_encoder_create(bc250_gpu_context_t *gpu_ctx,
                                    uint32_t width, uint32_t height,
                                    uint32_t fps, uint32_t bitrate,
                                    int profile)
{
    if (width == 0 || height == 0) return NULL;
    h264_encoder_t *encoder = calloc(1, sizeof(h264_encoder_t));
    if (!encoder) return NULL;

    encoder->gpu = gpu_ctx;
    encoder->width = width;
    encoder->height = height;
    encoder->width_in_mbs = (width + 15) / 16;
    encoder->height_in_mbs = (height + 15) / 16;
    encoder->total_mbs = encoder->width_in_mbs * encoder->height_in_mbs;

    encoder->fps = fps > 0 ? fps : 60;
    encoder->gop_size = encoder->fps; /* 1-second default keyframe interval */
    encoder->dpb_max = 16;
    encoder->force_idr = false;
    encoder->num_slices = 1;
    const char *slice_env = getenv("BC250_SLICES_PER_FRAME");
    if (slice_env) {
        int s = atoi(slice_env);
        if (s >= 1 && s <= 16) encoder->num_slices = s;
    }
#if defined(__linux__)
    else if (program_invocation_short_name &&
             (strcmp(program_invocation_short_name, "wivrn-server") == 0 ||
              strcmp(program_invocation_short_name, "wivrn") == 0)) {
        /* WiVRn encodes high-res VR streams (typically >= 1800x1800 per eye).
         * Multi-slice H.264 enables parallel entropy coding and drastically reduces VR latency. */
        encoder->num_slices = 4;
    }
#endif
    encoder->quality_level = 4;
    encoder->max_frame_bits = 0;

    /*
     * BUG FIX (found while wiring up CABAC's Main/High-profile auto-select -
     * see this task's commit message): the two constants below used to be
     * 0x64 (100) and 0x4D (77), commented "VAProfileH264High"/
     * "VAProfileH264Main" - but those are H.264 profile_idc VALUES (100=High,
     * 77=Main, see bitstream.h's PROFILE_HIGH/PROFILE_MAIN), not real libva
     * VAProfile ENUM ordinals. Checked against the actual system
     * /usr/include/va/va.h: VAProfileH264Main=6, VAProfileH264High=7. Since
     * va_backend.c's bc250_CreateContext() passes the real VAProfile enum
     * value straight through as this function's `profile` parameter, the
     * old comparison (profile==100 / profile==77) could never be true for
     * any real VA-API caller (ffmpeg's h264_vaapi included) requesting Main
     * or High profile via vaCreateConfig() - prof_idc silently stayed
     * PROFILE_BASELINE regardless of what was actually requested, which
     * would have made CABAC's profile-based auto-selection below a no-op in
     * the one path (real VA-API use) that matters. tests/test_encode.c's own
     * direct call (`h264_encoder_create(..., PROFILE_BASELINE)`, i.e. 66)
     * happened to still "work" only because Baseline's value (66) doesn't
     * collide with either the old or new constants either way. Now checks
     * BOTH the real VAProfile enum values (6/7) AND the raw profile_idc
     * values (77/100, PROFILE_MAIN/PROFILE_HIGH) so a caller using either
     * convention gets the right result.
     */
    uint8_t prof_idc = PROFILE_BASELINE;
    if (profile == 7 /* VAProfileH264High */ || profile == PROFILE_HIGH) prof_idc = PROFILE_HIGH;
    else if (profile == 6 /* VAProfileH264Main */ || profile == PROFILE_MAIN) prof_idc = PROFILE_MAIN;

    /*
     * CABAC selection (new capability - see cabac.h/cabac.c). H.264 requires
     * entropy_coding_mode_flag=1 (CABAC) to only be signaled for Main/High
     * profile (Annex A) - Baseline decoders are not required to support it.
     * Default: auto-select CABAC whenever the negotiated profile is Main or
     * High (i.e. whenever a real VA-API caller actually asked for it via
     * vaCreateConfig), CAVLC for Baseline - matching real-world encoder
     * behavior and requiring no caller changes. BC250_USE_CABAC=0/1
     * overrides this either direction for testing (including forcing CABAC
     * on with a Baseline profile_idc, which is spec-nonconformant but useful
     * for isolating CABAC correctness from the profile-negotiation path;
     * BC250_USE_CABAC=1 does NOT itself change prof_idc/SPS profile_idc -
     * see this task's final report for why that combination is for testing
     * only, not a supported deployment configuration).
     */
    bool use_cabac = (prof_idc != PROFILE_BASELINE);
    const char *cabac_env = getenv("BC250_USE_CABAC");
    if (cabac_env) {
        if (strcmp(cabac_env, "1") == 0 || strcmp(cabac_env, "true") == 0) use_cabac = true;
        else if (strcmp(cabac_env, "0") == 0 || strcmp(cabac_env, "false") == 0) use_cabac = false;
    }
    encoder->use_cabac = use_cabac;

    h264_sps_default(&encoder->sps, width, height, encoder->fps, prof_idc);
    h264_pps_default(&encoder->pps, encoder->sps.sps_id, use_cabac, 26);

    /* RC_LOW_LATENCY, not RC_CBR: this driver's only real consumer is
     * real-time game streaming (Sunshine/Moonlight), which is exactly what
     * RC_LOW_LATENCY's 2-frame buffer (vs. RC_CBR's 1-second buffer) is
     * documented in rate_control.h to be for - it was implemented but never
     * actually selected here. Real-world effect measured on-hardware
     * (docs/DEVLOG.md Â§10.8): a single legitimate bitrate spike (e.g. a
     * large real screen change) saturates a 1-second buffer, and
     * rc_get_frame_qp()'s deliberately-clamped max QP step then takes many
     * frames - up to a full GOP, since nothing else resets it sooner - to
     * walk QP back down as that buffer slowly drains. A 2-frame buffer
     * reaches the same proportional error immediately but drains back to
     * its 50% target within a couple of normal frames, so the same clamped
     * per-frame QP step recovers in a couple of frames instead of several
     * seconds. See maybe_append_filler()'s mode check just above this
     * function for the one other place RC_CBR was special-cased - updated
     * to treat RC_LOW_LATENCY as CBR-intent-compatible too, since it is
     * still fundamentally a bitrate-target mode, just tuned for faster
     * reaction, not an opt-out of hitting the target the way real VBR is. */
    rc_init(&encoder->rc, RC_LOW_LATENCY, bitrate, (double)encoder->fps, width, height);
    encoder->last_frame_sad = 0;
    encoder->qp_hint_applied = -1; /* no explicit QP hint applied yet - see h264_encoder_set_qp() */

    /* 4 bytes/pixel + slack. Was 2 bytes/pixel, which is ~50x more than
     * real 1440p desktop content needs at QP 12 (measured max 148,542
     * bytes) but NOT enough for worst-case incompressible content: 1440p
     * random noise overflowed the old 7.4 MB at both QP 12 and QP 8 (the
     * latter wanting ~13.5 MB/frame), and an overflow means the frame is
     * refused outright rather than silently truncated (see slice_overflow
     * below). At 1440p this is 14.8 MB vs 7.4 MB - an irrelevant amount of
     * host memory for one encoder instance - and it took the pathological
     * case from 12-of-12 frames refused to 0. */
    encoder->output_buf_size = (size_t)width * height * 4 + 65536;
    encoder->output_buf = malloc(encoder->output_buf_size);
    if (!encoder->output_buf) {
        free(encoder);
        return NULL;
    }

    encoder->prev_y_frame = malloc((size_t)width * height);
    encoder->has_prev_frame = false;

    encoder->nz_luma = calloc(encoder->total_mbs, sizeof(*encoder->nz_luma));
    encoder->nz_cb = calloc(encoder->total_mbs, sizeof(*encoder->nz_cb));
    encoder->nz_cr = calloc(encoder->total_mbs, sizeof(*encoder->nz_cr));
    if (!encoder->nz_luma || !encoder->nz_cb || !encoder->nz_cr) {
        free(encoder->nz_luma); free(encoder->nz_cb); free(encoder->nz_cr);
        free(encoder->output_buf);
        free(encoder->prev_y_frame);
        free(encoder);
        return NULL;
    }

    if (use_cabac) {
        encoder->dc_cbf_luma = calloc(encoder->total_mbs, sizeof(*encoder->dc_cbf_luma));
        encoder->dc_cbf_chroma = calloc(encoder->total_mbs, sizeof(*encoder->dc_cbf_chroma));
        encoder->cbp_nb = malloc(encoder->total_mbs * sizeof(*encoder->cbp_nb));
        encoder->mvd_x_abs = calloc(encoder->total_mbs, sizeof(*encoder->mvd_x_abs));
        encoder->mvd_y_abs = calloc(encoder->total_mbs, sizeof(*encoder->mvd_y_abs));
        encoder->skip_flag = calloc(encoder->total_mbs, sizeof(*encoder->skip_flag));
        if (!encoder->dc_cbf_luma || !encoder->dc_cbf_chroma || !encoder->cbp_nb ||
            !encoder->mvd_x_abs || !encoder->mvd_y_abs || !encoder->skip_flag) {
            free(encoder->dc_cbf_luma); free(encoder->dc_cbf_chroma); free(encoder->cbp_nb);
            free(encoder->mvd_x_abs); free(encoder->mvd_y_abs); free(encoder->skip_flag);
            free(encoder->nz_luma); free(encoder->nz_cb); free(encoder->nz_cr);
            free(encoder->output_buf);
            free(encoder->prev_y_frame);
            free(encoder);
            return NULL;
        }
        /* -1 sentinel ("no neighbor written yet this slice") in every slot -
         * memset 0xFF gives int16_t -1 for every element (two's complement),
         * matching cbp_nb's documented convention. */
        memset(encoder->cbp_nb, 0xFF, encoder->total_mbs * sizeof(*encoder->cbp_nb));
    }

    dynamic_governor_init(&encoder->governor);
    cpu_simd_me_config_init(&encoder->me_cfg, width, height);
    encoder->cpu_mvs = calloc(encoder->total_mbs, sizeof(gpu_mv_t));
    encoder->cpu_mvs_cap = encoder->total_mbs;

#ifdef BC250_HAVE_X264
    {
        const char *be = getenv("BC250_H264_BACKEND");
        if (!be || (strcmp(be, "compute") != 0 && strcmp(be, "hybrid") != 0 && strcmp(be, "gpu") != 0))
            encoder->x264 = h264_x264_create();
    }
    if (encoder->x264) {
        fprintf(stderr, "[bc250-h264] Encoder initialized: %ux%u, profile %d, backend=x264 (CPU libx264; set BC250_H264_BACKEND=compute for GPU/hybrid)\n",
                width, height, prof_idc);
        return encoder;
    }
    /* Compute/Hybrid GPU+CPU backend */
    const char *be = getenv("BC250_H264_BACKEND");
    bool gpu_only = be && strcmp(be, "gpu") == 0;
    if (gpu_only) {
        encoder->governor.enabled = false;
        encoder->governor.cpu_offload_enabled = false;
    } else {
        encoder->governor.enabled = true;
        encoder->governor.cpu_offload_enabled = true;
    }
    fprintf(stderr, "[bc250-h264] Encoder initialized: %ux%u @ %u fps, %u bps, profile %d, backend=%s, entropy=%s, hybrid_governor=%s\n",
            width, height, encoder->fps, bitrate, prof_idc,
            gpu_only ? "gpu" : (be && strcmp(be, "hybrid") == 0 ? "hybrid" : "compute"),
            use_cabac ? "CABAC" : "CAVLC",
            encoder->governor.enabled ? "enabled" : "disabled");

    return encoder;
}

void h264_encoder_force_idr(h264_encoder_t *encoder) {
    if (encoder) {
        encoder->force_idr = true;
    }
}

void h264_encoder_set_bitrate(h264_encoder_t *encoder, uint32_t bitrate_bps) {
    /* Guard against reinitializing when the requested bitrate hasn't
     * actually changed. rc_init() is a full reset of the feedback loop's
     * accumulated state (buffer_fullness back to 50%, current_qp back to
     * base_qp, error_integral back to 0) - appropriate when the bitrate
     * genuinely changes, but this is called from bc250_RenderPicture()
     * (va_backend.c) for both VAEncSequenceParameterBufferType and
     * VAEncMiscParameterTypeRateControl, and docs/rate_control_audit.md
     * section 4 point 6 flags that it was never confirmed whether a real
     * VA-API caller resends one of those buffers with an unchanged value
     * every frame. If it does, resetting on every call would silently
     * throw away the integral term's whole reason for existing (letting
     * *sustained* error accumulate across frames) every single frame.
     * Preserving the already-selected mode (rather than hardcoding RC_CBR
     * again here) is likewise just "don't reset state that didn't need to
     * change." */
    if (encoder && bitrate_bps > 0 && bitrate_bps != encoder->rc.target_bitrate) {
        rc_init(&encoder->rc, encoder->rc.mode, bitrate_bps, (double)encoder->fps,
                encoder->width, encoder->height);
    }
}

void h264_encoder_set_cbr_intent(h264_encoder_t *encoder, bool cbr_intent) {
    /* Deliberately NOT routed through rc_init() and not touched by
     * h264_encoder_set_bitrate(): this is VA-API session intent (does the
     * caller want a real constant-bitrate contract, independent of the
     * bitrate number itself changing), not rate-controller numeric state.
     * Sticky across calls for the same reason set_bitrate's own comment
     * gives for preserving rc state across resends - see this function's
     * header-comment doc for how va_backend.c derives the value it passes
     * here and how often it's expected to be called. */
    if (encoder) {
        encoder->cbr_intent = cbr_intent;
    }
}

void h264_encoder_set_gop_size(h264_encoder_t *encoder, uint32_t gop_size) {
    if (encoder && gop_size > 0) {
        encoder->gop_size = gop_size;
    }
}

void h264_encoder_set_cropping(h264_encoder_t *encoder, int enable,
                               uint32_t left, uint32_t right,
                               uint32_t top, uint32_t bottom) {
    if (!encoder) return;
    encoder->sps.frame_cropping = enable ? true : false;
    encoder->sps.crop_left   = enable ? left   : 0;
    encoder->sps.crop_right  = enable ? right  : 0;
    encoder->sps.crop_top    = enable ? top    : 0;
    encoder->sps.crop_bottom = enable ? bottom : 0;
}

void h264_encoder_set_num_slices(h264_encoder_t *encoder, int num_slices) {
    if (encoder && num_slices >= 1 && num_slices <= 16) {
        encoder->num_slices = num_slices;
    }
}

int h264_encoder_get_num_slices(const h264_encoder_t *encoder) {
    return encoder ? encoder->num_slices : 1;
}

void h264_encoder_set_fps(h264_encoder_t *encoder, uint32_t fps) {
    /* h264_encoder_create() is always called with a hardcoded fps=30
     * (va_backend.c's bc250_CreateContext) regardless of what the real
     * VA-API caller actually negotiates - the real value, when a caller
     * sends VAEncMiscParameterTypeFrameRate, only arrives here, later, via
     * bc250_RenderPicture(). Previously this only updated encoder->fps and
     * rc.framerate (a display-only field never read by rc_get_frame_qp()/
     * rc_update_stats()) - rc.target_bits_per_frame and rc.buffer_size,
     * the fields that actually gate every QP/buffer decision, are computed
     * once by rc_init() from framerate and were never recomputed here. For
     * a real 60fps session (encoder created assuming 30fps, real frames
     * arriving twice as fast), that leaves target_bits_per_frame at double
     * its correct value: at 60 real fps against a 30fps-sized per-frame
     * budget, the encoder emits roughly 2x the intended real bitrate
     * before rate control's own feedback (itself measuring error against
     * that same wrong per-frame target) has any correct target to
     * converge toward. This is a real, independent contributor to the
     * "same issue, no change" real-client report after the RC_LOW_LATENCY
     * fix above - that fix corrects recovery *speed* under a correctly-
     * calibrated budget, not a budget calibrated for the wrong framerate
     * in the first place.
     *
     * Fixed the same way h264_encoder_set_bitrate() already handles a
     * genuine bitrate change just below: a full rc_init() (which also
     * updates rc.framerate) whenever fps actually changes, guarded so a
     * caller resending the same value every frame doesn't repeatedly reset
     * the feedback loop's accumulated state. */
    if (encoder && fps > 0 && fps != encoder->fps) {
        /* Permanent, low-risk diagnostic: whether a real VA-API caller ever
         * actually sends this at all, and what real value, was previously
         * unconfirmed (the encoder's own "Encoder initialized" log only
         * ever shows the hardcoded creation-time 30). Real fps mismatches
         * of this kind are silent otherwise. */
        fprintf(stderr, "[bc250-h264] fps updated: %u -> %u (rate control re-initialized)\n",
                encoder->fps, fps);
        encoder->fps = fps;
        rc_init(&encoder->rc, encoder->rc.mode, encoder->rc.target_bitrate,
                (double)fps, encoder->width, encoder->height);
    }
}

uint32_t h264_encoder_get_fps(const h264_encoder_t *encoder) {
    return encoder ? encoder->fps : 0;
}

uint32_t h264_encoder_get_bitrate(const h264_encoder_t *encoder) {
    return encoder ? encoder->rc.target_bitrate : 0;
}

void h264_encoder_set_rc_mode(h264_encoder_t *encoder, rc_mode_t mode) {
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

rc_mode_t h264_encoder_get_rc_mode(const h264_encoder_t *encoder) {
    return encoder ? encoder->rc.mode : RC_LOW_LATENCY;
}

void h264_encoder_set_qp(h264_encoder_t *encoder, int qp) {
    if (encoder) {
        if (qp < 0) qp = 0;
        if (qp > 51) qp = 51;
        if (qp != encoder->qp_hint_applied) {
            encoder->rc.base_qp = qp;
            encoder->rc.current_qp = qp;
            encoder->qp_hint_applied = qp;
        }
        encoder->pps.pic_init_qp = qp - 26;
    }
}

int h264_encoder_get_qp(const h264_encoder_t *encoder) {
    return encoder ? encoder->rc.current_qp : 0;
}

void h264_encoder_set_quality_level(h264_encoder_t *encoder, uint32_t quality_level) {
    if (encoder) {
        if (quality_level < 1) quality_level = 1;
        if (quality_level > 7) quality_level = 7;
        encoder->quality_level = quality_level;
        rc_set_quality_level(&encoder->rc, quality_level);
    }
}

uint32_t h264_encoder_get_quality_level(const h264_encoder_t *encoder) {
    return encoder ? encoder->quality_level : 4;
}

void h264_encoder_set_max_frame_size(h264_encoder_t *encoder, uint32_t max_frame_bits) {
    if (encoder) {
        encoder->max_frame_bits = max_frame_bits;
        rc_set_max_frame_size(&encoder->rc, max_frame_bits);
    }
}

uint32_t h264_encoder_get_max_frame_size(const h264_encoder_t *encoder) {
    return encoder ? encoder->max_frame_bits : 0;
}

void h264_encoder_set_icq_quality(h264_encoder_t *encoder, int quality) {
    if (encoder) encoder->icq_quality = quality > 0 && quality <= 51 ? quality : 0;
}

bool h264_encoder_uses_x264(const h264_encoder_t *encoder) {
#ifdef BC250_HAVE_X264
    return encoder && encoder->x264;
#else
    (void)encoder;
    return false;
#endif
}

/* Choose frame type and QP and put this frame's GPU work in flight without
 * waiting for it. See h264_encoder_submit_frame()'s header doc for why the
 * split exists and what it costs (rate control runs one frame ahead). */
int h264_encoder_get_governor_tier(const h264_encoder_t *encoder) {
    return encoder ? (int)dynamic_governor_get_tier(&encoder->governor) : 0;
}

/* Choose frame type and QP and put this frame's GPU work in flight without
 * waiting for it. When GPU contention is high, dynamically offload Motion
 * Estimation to the CPU Zen 2 SIMD engine. */
int h264_encoder_submit_frame_ext(h264_encoder_t *encoder,
                                  bc250_gpu_context_t *gpu_ctx,
                                  gpu_image_t input_surface,
                                  gpu_memory_t input_memory,
                                  h264_pending_frame_t *pending)
{
    if (!encoder || !pending) return -1;

    memset(pending, 0, sizeof(*pending));

    bool is_idr = (encoder->frame_count % encoder->gop_size == 0) || encoder->force_idr;
    encoder->force_idr = false;

    if (is_idr) {
        encoder->frame_num = 0;
        encoder->idr_pic_id++;
        encoder->poc = 0;
        encoder->dpb_count = 0;
        encoder->last_frame_sad = 0;
    }

    int qp = rc_get_frame_qp(&encoder->rc, encoder->last_frame_sad);
    /* Must track rate_control.c's rc->qp_min - see the comment there for
     * why 12 is deliberate and what lowering it measured. */
    if (qp < encoder->rc.qp_min) qp = encoder->rc.qp_min;
    if (qp > 51) qp = 51;
    qp = apply_qp_override(qp);

    /* Kept identical to the synchronous path's own slice-count logic so that
     * submit+finish back to back is byte-for-byte the old behaviour. */
    int num_slices = (encoder->num_slices >= 1 && encoder->num_slices <= 16) ? encoder->num_slices : 1;

    pending->valid      = true;
    pending->is_idr     = is_idr;
    pending->qp         = qp;
    pending->num_slices = num_slices;

    governor_tier_t tier = dynamic_governor_get_tier(&encoder->governor);

    /* Dynamic Governor Tier 3: Emergency Failover.
     * When GPU is in severe lockup/contention (>15.5ms), skip submitting new GPU work
     * this frame. pending->gpu_submitted remains false, causing finish_frame()
     * to safely emit an immediate P_Skip frame in 0.1ms, maintaining stream deadline. */
    if (!is_idr && tier == GOV_TIER_3_FAILOVER) {
        dynamic_governor_notify_failover_handled(&encoder->governor);
        return 0;
    }

    /* The dispatch/submit half of what used to be one synchronous block. */
    if (gpu_ctx && input_surface.y_plane != VK_NULL_HANDLE
        && gpu_compute_begin_picture(gpu_ctx, input_surface) == 0) {

        gpu_mv_t *cpu_mvs = NULL;
        int me_mode = (tier >= GOV_TIER_1_GPU_FAST) ? 1 : 0;

        /* Dynamic Governor Tier 2: CPU SIMD Motion Estimation Offload.
         * Only run if explicitly opted in via BC250_ENABLE_CPU_ME=1 / cpu_offload_enabled. */
        if (!is_idr && tier == GOV_TIER_2_CPU_OFFLOAD && encoder->governor.cpu_offload_enabled &&
            input_memory.memory != VK_NULL_HANDLE &&
            gpu_ctx->has_recon_frame && gpu_ctx->recon_image.y_plane != VK_NULL_HANDLE &&
            gpu_ctx->recon_memory.memory != VK_NULL_HANDLE) {

            gpu_nv12_layout_t in_layout, ref_layout;
            bool unmap_in = false, unmap_ref = false;
            uint8_t *in_base = gpu_compute_map_surface(gpu_ctx, &input_surface, input_memory, &in_layout, &unmap_in);
            uint8_t *ref_base = gpu_compute_map_surface(gpu_ctx, &gpu_ctx->recon_image, gpu_ctx->recon_memory, &ref_layout, &unmap_ref);

            if (in_base && ref_base) {
                const uint8_t *src_y = in_base + in_layout.y_offset;
                const uint8_t *ref_y = ref_base + ref_layout.y_offset;

                if (cpu_simd_me_search_frame(src_y, (int)in_layout.y_pitch,
                                             ref_y, (int)ref_layout.y_pitch,
                                             encoder->width, encoder->height,
                                             encoder->cpu_mvs,
                                             &encoder->me_cfg) == 0) {
                    cpu_mvs = encoder->cpu_mvs;
                    me_mode = 2;
                }
            }

            gpu_compute_unmap_surface(gpu_ctx, gpu_ctx->recon_memory, unmap_ref);
            gpu_compute_unmap_surface(gpu_ctx, input_memory, unmap_in);
        }

        gpu_compute_dispatch_encode_ext(gpu_ctx, input_surface, encoder->width, encoder->height,
                                        qp, is_idr ? 1 : 0, num_slices, me_mode, cpu_mvs);
        if (gpu_compute_end_picture(gpu_ctx) == 0) {
            pending->gpu_submitted = true;
            /* Captured HERE, not read back at finish time: this is the frame
             * whose data the matching finish must wait on and read. */
            pending->gpu_slot = gpu_compute_submitted_slot(gpu_ctx);
        }
    }

    return 0;
}

int h264_encoder_submit_frame(h264_encoder_t *encoder,
                              bc250_gpu_context_t *gpu_ctx,
                              gpu_image_t input_surface,
                              h264_pending_frame_t *pending)
{
    gpu_memory_t dummy_mem = {0};
    return h264_encoder_submit_frame_ext(encoder, gpu_ctx, input_surface, dummy_mem, pending);
}

#ifdef BC250_HAVE_X264
static bool is_live_caller(void)
{
#if defined(__linux__)
    return program_invocation_short_name &&
           (strcmp(program_invocation_short_name, "sunshine") == 0 ||
            strcmp(program_invocation_short_name, "wivrn-server") == 0 ||
            strcmp(program_invocation_short_name, "wivrn") == 0 ||
            strcmp(program_invocation_short_name, "steam") == 0 ||
            strcmp(program_invocation_short_name, "streaming_client") == 0);
#else
    return false;
#endif
}

/* The picture through x264: read the surface back, hand it over with the
 * settings as they stand now. */
static int encode_frame_x264(h264_encoder_t *encoder, bc250_gpu_context_t *gpu_ctx,
                             gpu_image_t input_surface, gpu_memory_t input_memory,
                             uint8_t *output_buf, size_t output_size)
{
    if (!gpu_ctx || input_surface.y_plane == VK_NULL_HANDLE) return -1;
    /* An NV12 encoder has nothing to say about a P010 surface. */
    if (input_surface.format == GPU_IMAGE_P010) return -1;

    /* The picture is the cropped one the caller described, and never more
     * than the surface holds: ffmpeg opens H.264 contexts at 1920x1088 and
     * hands them 1920x1080 surfaces. */
    uint32_t w = encoder->width_in_mbs * 16, h = encoder->height_in_mbs * 16;
    if (encoder->sps.frame_cropping) {
        w -= 2 * (encoder->sps.crop_left + encoder->sps.crop_right);
        h -= 2 * (encoder->sps.crop_top + encoder->sps.crop_bottom);
    } else {
        w = encoder->width;
        h = encoder->height;
    }
    if (input_surface.width && input_surface.width < w) w = input_surface.width;
    if (input_surface.height && input_surface.height < h) h = input_surface.height;
    w &= ~1u;
    h &= ~1u;
    if (w == 0 || h == 0) return -1;

    const size_t need = (size_t)w * h;
    if (need > encoder->x264_cap) {
        uint8_t *ny = realloc(encoder->x264_y, need);
        if (ny) encoder->x264_y = ny;
        uint8_t *nuv = realloc(encoder->x264_uv, need / 2);
        if (nuv) encoder->x264_uv = nuv;
        if (!ny || !nuv) return -1;
        encoder->x264_cap = need;
    }
    if (gpu_compute_download_nv12(gpu_ctx, &input_surface, input_memory,
                                  encoder->x264_y, (int)w, encoder->x264_uv, (int)w,
                                  (int)w, (int)h) != 0)
        return -1;

    h264_x264_config_t cfg = {
        .width = w, .height = h,
        .fps = encoder->fps,
        .gop = encoder->gop_size,
        .rc_mode = encoder->rc.mode,
        .bitrate = encoder->rc.target_bitrate,
        .qp = encoder->rc.current_qp,
        .crf = encoder->icq_quality,
        .quality_level = encoder->quality_level,
        .profile_idc = encoder->sps.profile_idc,
        .cbr_intent = encoder->cbr_intent,
        .live = is_live_caller(),
    };
    const bool idr = encoder->force_idr;
    encoder->force_idr = false;
    int n = h264_x264_encode(encoder->x264, &cfg,
                             encoder->x264_y, (int)w, encoder->x264_uv, (int)w,
                             idr, cfg.rc_mode == RC_CQP ? encoder->rc.current_qp : 0,
                             output_buf, output_size);
    if (n > 0) encoder->frame_count++;
    return n;
}
#endif

int h264_encoder_encode_frame_ext(h264_encoder_t *encoder,
                                  bc250_gpu_context_t *gpu_ctx,
                                  gpu_image_t input_surface,
                                  gpu_memory_t input_memory,
                                  uint8_t *output_buf, size_t output_size)
{
    if (!encoder || !output_buf) return -1;
#ifdef BC250_HAVE_X264
    if (encoder->x264)
        return encode_frame_x264(encoder, gpu_ctx, input_surface, input_memory,
                                 output_buf, output_size);
#endif

    h264_pending_frame_t pending;
    if (h264_encoder_submit_frame_ext(encoder, gpu_ctx, input_surface, input_memory, &pending) != 0)
        return -1;
    return h264_encoder_finish_frame(encoder, gpu_ctx,
                                     output_buf, output_size, &pending);
}

int h264_encoder_encode_frame(h264_encoder_t *encoder,
                              bc250_gpu_context_t *gpu_ctx,
                              gpu_image_t input_surface,
                              uint8_t *output_buf, size_t output_size)
{
    gpu_memory_t dummy_mem = {0};
    return h264_encoder_encode_frame_ext(encoder, gpu_ctx, input_surface, dummy_mem,
                                         output_buf, output_size);
}

static int get_default_slice_threads(int num_slices) {
    int threads = 1;
    const char *env_threads = getenv("BC250_MAX_CPU_THREADS");
    if (!env_threads) env_threads = getenv("BC250_THREADS");
    if (!env_threads) env_threads = getenv("BC250_CPU_THREADS");
    if (env_threads) {
        int t = atoi(env_threads);
        if (t >= 1 && t <= 16) return (num_slices < t) ? num_slices : t;
    }
    const char *env_no_omp = getenv("BC250_DISABLE_OPENMP");
    if (env_no_omp && (strcmp(env_no_omp, "0") != 0 && strcmp(env_no_omp, "false") != 0)) {
        return 1;
    }
#if defined(__linux__)
    if (program_invocation_short_name) {
        if (strcmp(program_invocation_short_name, "sunshine") == 0 ||
            strcmp(program_invocation_short_name, "steam") == 0 ||
            strcmp(program_invocation_short_name, "streaming_client") == 0) {
            return (num_slices < 2) ? 1 : 2;
        } else if (strcmp(program_invocation_short_name, "wivrn-server") == 0 ||
                   strcmp(program_invocation_short_name, "wivrn") == 0) {
            return (num_slices < 4) ? num_slices : 4;
        } else if (strcmp(program_invocation_short_name, "ffmpeg") == 0) {
            return (num_slices < 4) ? num_slices : 4;
        }
    }
#endif
    return (num_slices < 4) ? num_slices : 4;
}

int h264_encoder_finish_frame(h264_encoder_t *encoder,
                              bc250_gpu_context_t *gpu_ctx,
                              uint8_t *output_buf, size_t output_size,
                              const h264_pending_frame_t *pending)
{
    if (!encoder || !output_buf || !pending || !pending->valid) return -1;

    /* Opt-in CPU-side timing (BC250_PERF_STATS=1), added for real-time
     * throughput diagnosis. Two brackets: `frame_t0` covers this entire
     * function call (wall clock, as seen by the caller - includes the
     * synchronous GPU dispatch+sync below, all CPU CAVLC/bitstream work,
     * and rate-control/DPB bookkeeping), while `cavlc_t0` (set later, right
     * before the per-slice macroblock loop) isolates just the CPU-side
     * CAVLC entropy coding + bitstream writing, matching gpu_compute.c's
     * "[BC250_PERF_GPU] ..." lines so the two can be correlated per frame
     * (both are printed with the same `type=I|P` tag; join on stream order,
     * since the GPU line is emitted a few lines above this function's own
     * synchronous gpu_compute_sync() call within the same call). */
    const char *perf_env = getenv("BC250_PERF_STATS");
    bool perf_stats = perf_env && (strcmp(perf_env, "1") == 0 || strcmp(perf_env, "true") == 0);
    struct timespec frame_t0;
    if (perf_stats) clock_gettime(CLOCK_MONOTONIC, &frame_t0);

    /* PERF toggle (BC250_SHADOW_COPY=0 disables), read once per process.
     * See the shadow_copy() call site below and shadow_copy()'s doc comment. */
    static int shadow_copy_enabled = -1;
    if (shadow_copy_enabled < 0) {
        const char *sc_env = getenv("BC250_SHADOW_COPY");
        shadow_copy_enabled = (sc_env && strcmp(sc_env, "0") == 0) ? 0 : 1;
    }
    /* PERF/verification toggle (BC250_NZ_MASK=0 disables) - see the
     * gpu_compute_get_nz_staging_data() call below. */
    static int nz_mask_disabled = -1;
    if (nz_mask_disabled < 0) {
        const char *nz_env = getenv("BC250_NZ_MASK");
        nz_mask_disabled = (nz_env && strcmp(nz_env, "0") == 0) ? 1 : 0;
    }

    /* Frame type and QP were decided, and the IDR state reset applied, by the
     * matching h264_encoder_submit_frame() - they had to be, because the
     * quantize dispatch already consumed them. Recomputing them here would
     * re-read a frame_count that has not advanced yet and double-apply the
     * IDR side effects. */
    const bool is_idr = pending->is_idr;
    const int qp = pending->qp;

    size_t total_written = 0;

    /* 1. Write AUD (Access Unit Delimiter) NALU */
    total_written += write_aud(encoder->output_buf + total_written,
                               encoder->output_buf_size - total_written,
                               is_idr);

    /* 2. Write SPS and PPS NALUs on IDR frames */
    if (is_idr) {
        size_t sps_size = bs_write_sps(
            encoder->output_buf + total_written,
            encoder->output_buf_size - total_written,
            &encoder->sps);
        total_written += sps_size;

        size_t pps_size = bs_write_pps(
            encoder->output_buf + total_written,
            encoder->output_buf_size - total_written,
            &encoder->pps);
        total_written += pps_size;
    }

    /* 3a. Multi-slice partitioning (Sunshine/Moonlight network resilience) -
     * computed BEFORE the GPU dispatch below, since residual_predict.comp
     * needs num_slices too (see gpu_compute_dispatch_encode's doc comment
     * and that shader's SLICE BOUNDARIES note) to correctly treat a
     * different-slice neighbor MB as unavailable for intra prediction. */
    const int num_slices = pending->num_slices;

    /* 3b. Dispatch GPU compute encoding pipeline if available, and fetch the
     * REAL per-coefficient residual data (post-quant levels + pre-quant
     * transform coefficients) - not just the lossy packed entropy summary
     * the old heuristic-only path used. */
    const int16_t *quant_levels = NULL;
    const int *dc_coeff = NULL;
    const uint32_t *pred_modes = NULL;
    const gpu_mv_t *mvs = NULL;
    /* Per-4x4-block nonzero bitmask (see nc_ctx_t::nz_mask). Left NULL on
     * every path that produces no new GPU output this frame, exactly like the
     * four above; block_any_nonzero() then falls back to scanning. */
    const uint32_t *nz_masks = NULL;
    /* Opt-in phase brackets (BC250_PERF_STATS=1). Added because the existing
     * per-stage brackets accounted for only ~52% of frame time once the GPU
     * was contended: wait(sync) + cavlc + shadow came to 348 ms of a 675 ms
     * frame, with begin_picture and submit both ~0. Rather than guess where
     * the other 327 ms went, bracket the phases. dispatch is the interesting
     * one - it records the command buffer, updates descriptors and does
     * per-frame image-layout transitions, none of which was ever timed. */
    struct timespec ph_a, ph_b, ph_c, ph_d;
    bool ph = perf_stats;
    double ph_begin_ms = 0.0, ph_dispatch_ms = 0.0, ph_end_ms = 0.0;

    if (ph) clock_gettime(CLOCK_MONOTONIC, &ph_a);
    /* begin_picture/dispatch/submit all happened in h264_encoder_submit_frame().
     * gpu_submitted false means it refused to touch the command buffer or every
     * submit retry failed, so there is nothing to wait on and nothing fresh in
     * the staging slot - fall through with quant_levels/etc. left NULL, same as
     * every other "no new GPU work this frame" path below.
     *
     * begin_ms/dispatch_ms are now both zero by construction; the phases they
     * measured moved into the submit half, which in pipelined use is deliberately
     * no longer inside this frame's critical path. end_sync_ms remains the
     * interesting one - in pipelined use it should collapse toward zero, because
     * the GPU ran this frame while the CPU was entropy-coding the previous one. */
    if (pending->gpu_submitted) {
        if (ph) clock_gettime(CLOCK_MONOTONIC, &ph_b);
        if (ph) clock_gettime(CLOCK_MONOTONIC, &ph_c);
        /* gpu_compute_end_picture() now retries vkQueueSubmit internally
         * (real GPU contention can transiently fail a submit the same way
         * it can transiently fail an allocation - see that function's doc
         * comment) and only returns nonzero if every retry failed. In that
         * case the GPU never actually did new work this frame: calling
         * gpu_compute_sync() would wait on a fence that was reset but will
         * now never be signaled (a permanent hang), and every staging
         * buffer below would still hold whatever the last *successful*
         * dispatch left in it - stale data that must not be read back and
         * treated as this frame's real residual/MV/mode data. Skipping
         * straight past this block leaves quant_levels/coeff/pred_modes/mvs
         * at their NULL default, which the skip-decision logic below
         * already handles correctly and safely: a NULL quant_levels
         * certifies the whole frame P_Skip, matching what a real decoder
         * does when it receives no new information - the frame repeats the
         * last reference picture, rather than fabricated stale content
         * being sent as if it were genuinely this frame's.
         *
         * gpu_compute_sync()'s own vkWaitForFences is now checked too (see
         * its doc comment) - it's the exact call this comment already
         * identified as the vector for a stale read, just also guarding the
         * case where the wait itself fails even though the submit
         * succeeded. Folded into the same condition so either failure takes
         * the same safe fallback. */
        if (gpu_compute_sync_slot(gpu_ctx, pending->gpu_slot) == 0) {
            double last_gpu_lat = gpu_compute_get_last_latency_ms(gpu_ctx);
            if (!is_idr) {
                dynamic_governor_update(&encoder->governor, last_gpu_lat);
            }
        if (ph) {
            clock_gettime(CLOCK_MONOTONIC, &ph_d);
            ph_begin_ms    = (double)(ph_b.tv_sec - ph_a.tv_sec) * 1000.0 +
                             (double)(ph_b.tv_nsec - ph_a.tv_nsec) / 1e6;
            ph_dispatch_ms = (double)(ph_c.tv_sec - ph_b.tv_sec) * 1000.0 +
                             (double)(ph_c.tv_nsec - ph_b.tv_nsec) / 1e6;
            ph_end_ms      = (double)(ph_d.tv_sec - ph_c.tv_sec) * 1000.0 +
                             (double)(ph_d.tv_nsec - ph_c.tv_nsec) / 1e6;
            fprintf(stderr, "[BC250_PERF_PHASE] frame=%u type=%s begin_ms=%.3f "
                            "dispatch_ms=%.3f end_sync_ms=%.3f\n",
                    encoder->frame_count, is_idr ? "I" : "P",
                    ph_begin_ms, ph_dispatch_ms, ph_end_ms);
        }
        gpu_compute_debug_dump_recon(gpu_ctx, (int)encoder->width, (int)encoder->height);

        void *quant_data = NULL, *dc_data = NULL, *pred_mode_data = NULL, *mv_data = NULL;
        size_t quant_size = 0, dc_size = 0, pred_mode_size = 0, mv_size = 0;
        /* All five readbacks MUST name this frame's own slot. Using the
         * implicit "most recently submitted" variants here is correct only in
         * the synchronous path and silently reads the next frame's data in the
         * pipelined one. */
        const int slot = pending->gpu_slot;
        if (gpu_compute_get_quant_staging_data_slot(gpu_ctx, slot, &quant_data, &quant_size) == 0) {
            quant_levels = (const int16_t *)quant_data;
        }
        if (gpu_compute_get_dc_staging_data_slot(gpu_ctx, slot, &dc_data, &dc_size) == 0) {
            dc_coeff = (const int *)dc_data;
        }
        if (gpu_compute_get_pred_mode_staging_data_slot(gpu_ctx, slot, &pred_mode_data, &pred_mode_size) == 0) {
            pred_modes = (const uint32_t *)pred_mode_data;
        }
        if (gpu_compute_get_mv_staging_data_slot(gpu_ctx, slot, &mv_data, &mv_size) == 0) {
            mvs = (const gpu_mv_t *)mv_data;
        }
        void *nz_data = NULL;
        size_t nz_size = 0;
        /* BC250_NZ_MASK=0 forces the pre-mask behaviour (block_any_nonzero()
         * scans quant_levels). The mask is an accelerator that must be exactly
         * equivalent to the scan, so this toggle is the A/B that proves it:
         * one binary, same content, output must be byte-identical. */
        if (!nz_mask_disabled &&
            gpu_compute_get_nz_staging_data_slot(gpu_ctx, slot, &nz_data, &nz_size) == 0) {
            nz_masks = (const uint32_t *)nz_data;
        }

        /* PERF: copy each GPU staging buffer once into cacheable host memory
         * before any of this function's repeated per-MB/per-block reads
         * touch it - see shadow_copy()'s doc comment for the full board-
         * measured rationale (this is the dominant real CPU-side cost, not
         * CAVLC bit-writing). Every subsequent use of quant_levels/coeff/
         * pred_modes/mvs in this function reads the copy, not the raw
         * uncached mapped buffer. */
        /* DIAGNOSTIC ONLY (BC250_PERF_STATS=1): isolate the four shadow_copy()
         * bulk-memcpy calls as their own wall-clock bracket, to test whether
         * reading FROM the uncached/write-combined GPU staging memory (even
         * as one linear sequential pass) is itself the dominant per-frame
         * cost, separate from the per-MB CAVLC access pattern shadow_copy was
         * originally written to fix. See shadow_copy()'s doc comment above. */
        struct timespec shadow_t0, shadow_t1;
        if (perf_stats) clock_gettime(CLOCK_MONOTONIC, &shadow_t0);

        /* BC250_SHADOW_COPY=0 skips the four bulk memcpys and reads the mapped
         * staging buffers directly. This is a measurement toggle, not a
         * correctness one: shadow_copy() only changes WHERE a byte is read
         * from, so output is byte-identical either way (verified). See
         * shadow_copy()'s doc comment for why it may now be redundant. */
        if (shadow_copy_enabled) {
        quant_levels = (const int16_t *)shadow_copy((void **)&encoder->quant_levels_shadow,
                                                 &encoder->quant_levels_shadow_cap,
                                                 quant_levels, quant_size);
        dc_coeff = (const int *)shadow_copy((void **)&encoder->dc_coeff_shadow,
                                             &encoder->dc_coeff_shadow_cap,
                                             dc_coeff, dc_size);
        pred_modes = (const uint32_t *)shadow_copy((void **)&encoder->pred_modes_shadow,
                                                     &encoder->pred_modes_shadow_cap,
                                                     pred_modes, pred_mode_size);
        mvs = (const gpu_mv_t *)shadow_copy((void **)&encoder->mvs_shadow,
                                             &encoder->mvs_shadow_cap,
                                             mvs, mv_size);
        nz_masks = (const uint32_t *)shadow_copy((void **)&encoder->nz_masks_shadow,
                                                  &encoder->nz_masks_shadow_cap,
                                                  nz_masks, nz_size);
        }

        if (mvs) {
            uint64_t total_sad = 0;
            for (uint32_t mb = 0; mb < encoder->total_mbs; mb++) {
                total_sad += mvs[mb].sad;
            }
            encoder->last_frame_sad = total_sad;
        }

        if (perf_stats) {
            clock_gettime(CLOCK_MONOTONIC, &shadow_t1);
            double shadow_ms = (double)(shadow_t1.tv_sec - shadow_t0.tv_sec) * 1000.0 +
                                (double)(shadow_t1.tv_nsec - shadow_t0.tv_nsec) / 1e6;
            fprintf(stderr, "[BC250_PERF_SHADOW] frame=%u type=%s shadow_copy_ms=%.3f "
                            "quant_bytes=%zu dc_bytes=%zu pred_mode_bytes=%zu mv_bytes=%zu "
                            "nz_bytes=%zu total_bytes=%zu\n",
                    encoder->frame_count, is_idr ? "I" : "P", shadow_ms,
                    quant_size, dc_size, pred_mode_size, mv_size, nz_size,
                    quant_size + dc_size + pred_mode_size + mv_size + nz_size);
        }

        /* DIAGNOSTIC ONLY (BC250_NZ_AUDIT=1): before anything is allowed to
         * DEPEND on quantize.comp's per-block nonzero bitmask, prove two
         * things about it on real board data:
         *
         *   1. CORRECTNESS - recompute the mask on the CPU straight from
         *      quant_levels and require an exact match on every block. The
         *      mask is meant to be an exact restatement of `level != 0`, so
         *      any mismatch at all means the GPU buffer, the std430 layout
         *      assumption, or the double-buffer/fence contract is wrong, and
         *      a bit test against it would silently corrupt cbp / the P_Skip
         *      decision rather than just running slow.
         *   2. HEADROOM - report how many blocks are entirely zero, and how
         *      many have zero AC (positions 1..15). Those are exactly the
         *      blocks a mask-guided CAVLC would never have to read, so this
         *      is the measured ceiling on the optimization BEFORE writing it.
         *      If the density says there is little to skip, the refactor is
         *      not worth doing.
         *
         * Deliberately runs OUTSIDE the perf_stats timing brackets and is off
         * by default: it reads all of quant_levels a second time. */
        if (nz_masks && quant_levels && getenv("BC250_NZ_AUDIT") &&
            nz_size >= (size_t)encoder->total_mbs * 24 * sizeof(uint32_t) &&
            quant_size >= (size_t)encoder->total_mbs * 24 * 16 * sizeof(int16_t)) {
            uint32_t total_blocks = encoder->total_mbs * 24;
            uint32_t mismatches = 0, all_zero = 0, ac_zero = 0;
            uint32_t first_bad_block = 0;
            for (uint32_t b = 0; b < total_blocks; b++) {
                const int16_t *blk = quant_levels + (size_t)b * 16;
                uint32_t cpu_mask = 0;
                for (int p = 0; p < 16; p++) if (blk[p] != 0) cpu_mask |= (1u << p);
                uint32_t gpu_mask = nz_masks[b] & 0xFFFFu;
                if (cpu_mask != gpu_mask) {
                    if (mismatches == 0) first_bad_block = b;
                    mismatches++;
                }
                if (cpu_mask == 0) all_zero++;
                if ((cpu_mask & 0xFFFEu) == 0) ac_zero++;
            }
            fprintf(stderr, "[BC250_NZ_AUDIT] frame=%u type=%s blocks=%u mismatches=%u "
                            "first_bad=%u all_zero=%u (%.1f%%) ac_zero=%u (%.1f%%)\n",
                    encoder->frame_count, is_idr ? "I" : "P", total_blocks, mismatches,
                    mismatches ? first_bad_block : 0,
                    all_zero, total_blocks ? 100.0 * all_zero / total_blocks : 0.0,
                    ac_zero, total_blocks ? 100.0 * ac_zero / total_blocks : 0.0);
        }

        /* Opt-in debug instrumentation (BC250_DUMP_QUANT_LEVELS=1), kept as
         * a permanent low-risk diagnostic: dumps the exact post-quant
         * coefficient buffer this frame is about to hand to CAVLC, so an
         * external tool can independently decode the produced bitstream and
         * diff against known-good ground truth without guessing at
         * intermediate values. Dumps every frame while set; caller should
         * limit clip length. */
        if (quant_levels && getenv("BC250_DUMP_QUANT_LEVELS")) {
            char name[64];
            snprintf(name, sizeof(name), "quant_levels_%05u.bin", encoder->frame_count);
            FILE *qf = bc250_debug_dump_open(name, "BC250_DUMP_QUANT_LEVELS");
            if (qf) {
                fwrite(quant_levels, sizeof(int), (size_t)encoder->total_mbs * 24 * 16, qf);
                fclose(qf);
            }
            snprintf(name, sizeof(name), "quant_levels_%05u.meta", encoder->frame_count);
            FILE *mf = bc250_debug_dump_open(name, "BC250_DUMP_QUANT_LEVELS");
            if (mf) {
                fprintf(mf, "width_in_mbs=%u\nheight_in_mbs=%u\ntotal_mbs=%u\nqp=%d\nis_idr=%d\n",
                        encoder->width_in_mbs, encoder->height_in_mbs, encoder->total_mbs, qp, is_idr ? 1 : 0);
                fclose(mf);
            }
        }

        /* Opt-in diagnostic (BC250_DEBUG_MV_STATS=1): confirms the sub-pel
         * refinement in motion_estimation.comp is actually landing on
         * fractional (non-multiple-of-4) quarter-pel positions, not just
         * silently degenerating back to integer-only motion - added while
         * validating the sub-pel motion compensation change, kept as a
         * permanent low-risk diagnostic since "is the search really finding
         * sub-pel matches on this content" is a real, recurring question
         * that's otherwise invisible from the outside (unlike PSNR, which
         * conflates this with every other stage, including this test
         * content's own DCT-ringing on hard edges - see the sub-pel
         * validation notes in this change's commit history). */
        if (mvs && !is_idr && getenv("BC250_DEBUG_MV_STATS")) {
            uint32_t total = encoder->total_mbs;
            uint32_t subpel_x = 0, subpel_y = 0, subpel_any = 0;
            int32_t min_x = INT32_MAX, max_x = INT32_MIN, min_y = INT32_MAX, max_y = INT32_MIN;
            for (uint32_t m = 0; m < total; m++) {
                int32_t vx = mvs[m].mvx, vy = mvs[m].mvy;
                if (vx & 3) subpel_x++;
                if (vy & 3) subpel_y++;
                if ((vx & 3) || (vy & 3)) subpel_any++;
                if (vx < min_x) min_x = vx;
                if (vx > max_x) max_x = vx;
                if (vy < min_y) min_y = vy;
                if (vy > max_y) max_y = vy;
            }
            fprintf(stderr, "[BC250_DEBUG_MV_STATS] frame=%u total_mbs=%u sub_pel_x=%u sub_pel_y=%u sub_pel_any=%u "
                            "mvx_range=[%d,%d] mvy_range=[%d,%d] (units: quarter-luma-pel)\n",
                    encoder->frame_count, total, subpel_x, subpel_y, subpel_any, min_x, max_x, min_y, max_y);
        }

        /* TEMPORARY investigation diagnostic (BC250_DEBUG_MV_ROW=<mby>) for
         * the gradient-boundary block-displacement artifact task: dumps the
         * per-MB raw searched MV (as read back from mv_staging - i.e.
         * exactly what mv_predictor()/encode_mb_p16x16() will see) AND the
         * median predictor + resulting MVD for every macroblock in ONE
         * requested row, on every P frame. Lets a specific boundary row be
         * cross-referenced frame-by-frame against BC250_DUMP_RECON_FRAMES
         * output without eyeballing full aggregate stats. Not wired to any
         * permanent feature flag - safe to leave (no-op unless the env var
         * is set), but intended to be removed or promoted after the
         * investigation concludes. */
        if (mvs && !is_idr && getenv("BC250_DEBUG_MV_ROW")) {
            int dbg_row = atoi(getenv("BC250_DEBUG_MV_ROW"));
            if (dbg_row >= 0 && (uint32_t)dbg_row < encoder->height_in_mbs) {
                for (uint32_t mbx = 0; mbx < encoder->width_in_mbs; mbx++) {
                    uint32_t mb = (uint32_t)dbg_row * encoder->width_in_mbs + mbx;
                    int pred_x, pred_y;
                    mv_predictor(mvs, mbx, (uint32_t)dbg_row, encoder->width_in_mbs, 0, &pred_x, &pred_y);
                    int mvd_x = mvs[mb].mvx - pred_x;
                    int mvd_y = mvs[mb].mvy - pred_y;
                    /* Cross-check against the REAL P_Skip derivation
                     * (ITU-T 8.4.1.1, skip_mv_predictor() above) - see that
                     * function's doc comment for why this can legitimately
                     * differ from the plain median (pred_x/pred_y) used for
                     * MVD, and how that divergence corrupts downstream
                     * macroblocks whenever the OLD (pre-fix) skip-legality
                     * check certified this MB as skip using the wrong rule. */
                    int skip_pred_x, skip_pred_y;
                    bool zero_force = false;
                    skip_mv_predictor_ex(mvs, mbx, (uint32_t)dbg_row, encoder->width_in_mbs, 0,
                                          &skip_pred_x, &skip_pred_y, &zero_force);
                    bool old_skip_legal = (mvs[mb].mvx == pred_x && mvs[mb].mvy == pred_y);
                    bool new_skip_legal = (mvs[mb].mvx == skip_pred_x && mvs[mb].mvy == skip_pred_y);
                    fprintf(stderr, "[BC250_DEBUG_MV_ROW] frame=%u row=%d mbx=%u mb=%u "
                                    "mv=(%d,%d) sad=%u pred=(%d,%d) mvd=(%d,%d) "
                                    "zero_force=%d skip_pred=(%d,%d) old_skip_legal=%d new_skip_legal=%d%s\n",
                            encoder->frame_count, dbg_row, mbx, mb,
                            mvs[mb].mvx, mvs[mb].mvy, mvs[mb].sad, pred_x, pred_y, mvd_x, mvd_y,
                            zero_force, skip_pred_x, skip_pred_y, old_skip_legal, new_skip_legal,
                            (old_skip_legal != new_skip_legal) ? "  <== SKIP-LEGALITY DIVERGES" : "");
                }
            }
        }

        /* TEMPORARY investigation diagnostic (BC250_DEBUG_MB=<mb_index>,
         * BC250_DEBUG_FRAME=<frame_count>) for the gradient-boundary
         * artifact task: dumps the exact chroma DC/AC data one MB will
         * transmit (pre-Hadamard raw values, post-quant transmitted levels,
         * and the raw quant_levels AC content for its 8 chroma blocks), to
         * inspect the actual residual data at a specific macroblock/frame.
         *
         * CAUTION - a per-plane PSNR comparison that motivated adding this
         * (GPU recon_image, via BC250_DUMP_RECON_FRAMES, vs. real decoded
         * output) initially looked chroma-specific (Y~40dB vs U~20dB/
         * V~18.5dB) and pointed straight at this code path. That comparison
         * turned out to be comparing against the WRONG reference: repeating
         * it against the true source frames (BC250_DUMP_INPUT_FRAMES, the
         * same ground truth tools/quality_test.sh uses) instead of the GPU
         * recon dump showed decoded chroma is actually fine (U/V in the high
         * 30s dB, matching quality_test.sh's own healthy board-validated
         * numbers) - so this data dump did not end up implicating the
         * chroma DC/quant path itself. Kept as a safe, no-op-unless-set
         * diagnostic in case a future investigation wants it, but do not
         * treat its original chroma hypothesis as confirmed - see this
         * branch's final report for the corrected methodology and numbers.
         *
         * WIDENED (fix/chroma-boundary-residual): dropped the `!is_idr`
         * guard this originally shipped with. The gradient-boundary chroma
         * defect being investigated here is present at full severity on the
         * very first IDR frame already (see f7c478c's commit message,
         * confirmed via BC250_DUMP_RECON_FRAMES) - excluding IDR frames made
         * this diagnostic unusable for exactly the frame that matters. The
         * mvs printout below is separately still guarded on `mvs` being
         * non-NULL, which is naturally false on an IDR frame (no motion
         * vectors), so this is a strict widening, not a behavior change for
         * P-frame callers. */
        if (quant_levels && dc_coeff &&
            getenv("BC250_DEBUG_MB") && getenv("BC250_DEBUG_FRAME")) {
            uint32_t dbg_mb = (uint32_t)atoi(getenv("BC250_DEBUG_MB"));
            uint32_t dbg_frame = (uint32_t)atoi(getenv("BC250_DEBUG_FRAME"));
            if (encoder->frame_count == dbg_frame && dbg_mb < encoder->total_mbs) {
                uint32_t mb = dbg_mb;
                int cb_dc_raw[4], cr_dc_raw[4];
                for (int i = 0; i < 4; i++) cb_dc_raw[i] = dc_coeff_at(dc_coeff, mb, 16 + i);
                for (int i = 0; i < 4; i++) cr_dc_raw[i] = dc_coeff_at(dc_coeff, mb, 20 + i);
                int cb_dc[4], cr_dc[4];
                chroma_dc_hadamard(cb_dc_raw, cb_dc);
                chroma_dc_hadamard(cr_dc_raw, cr_dc);
                for (int i = 0; i < 4; i++) { cb_dc[i] = quantize_dc_chroma(cb_dc[i], chroma_qp(qp)); cr_dc[i] = quantize_dc_chroma(cr_dc[i], chroma_qp(qp)); }
                fprintf(stderr, "[BC250_DEBUG_MB] frame=%u mb=%u qp=%d cb_dc_raw=(%d,%d,%d,%d) cr_dc_raw=(%d,%d,%d,%d) "
                                "cb_dc_tx=(%d,%d,%d,%d) cr_dc_tx=(%d,%d,%d,%d)\n",
                        dbg_frame, mb, qp,
                        cb_dc_raw[0], cb_dc_raw[1], cb_dc_raw[2], cb_dc_raw[3],
                        cr_dc_raw[0], cr_dc_raw[1], cr_dc_raw[2], cr_dc_raw[3],
                        cb_dc[0], cb_dc[1], cb_dc[2], cb_dc[3],
                        cr_dc[0], cr_dc[1], cr_dc[2], cr_dc[3]);
                for (int blk = 16; blk < 24; blk++) {
                    const int16_t *b = quant_block_ptr(quant_levels, mb, blk);
                    int any_ac = 0;
                    for (int p = 1; p < 16; p++) if (b[p] != 0) any_ac = 1;
                    fprintf(stderr, "[BC250_DEBUG_MB]   quant_block=%d dc=%d any_ac=%d vals=[%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d]\n",
                            blk, b[0], any_ac,
                            b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7],
                            b[8],b[9],b[10],b[11],b[12],b[13],b[14],b[15]);
                }
                /* TEMPORARY: also dump the 16 LUMA blocks' real transmitted
                 * quant_levels + this MB's committed motion vector, to check
                 * real coefficient magnitude/count against CAVLC's escape-code
                 * thresholds for the gradient-row investigation. */
                if (mvs) {
                    fprintf(stderr, "[BC250_DEBUG_MB] mv=(%d,%d) sad=%u\n",
                            mvs[mb].mvx, mvs[mb].mvy, mvs[mb].sad);
                }
                for (int blk = 0; blk < 16; blk++) {
                    const int16_t *b = quant_block_ptr(quant_levels, mb, blk);
                    int nz = 0, maxabs = 0;
                    for (int p = 0; p < 16; p++) { if (b[p] != 0) nz++; if (abs(b[p]) > maxabs) maxabs = abs(b[p]); }
                    fprintf(stderr, "[BC250_DEBUG_MB]   luma_block=%d nz=%d maxabs=%d vals=[%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d]\n",
                            blk, nz, maxabs,
                            b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7],
                            b[8],b[9],b[10],b[11],b[12],b[13],b[14],b[15]);
                }
            }
        }
        } /* gpu_compute_sync() == 0 */
    } else {
        /* Emergency Failover or submit failure: no GPU work was submitted for this frame.
         * Notify the governor so any Tier 3 Failover steps down to Tier 2 CPU offload. */
        dynamic_governor_notify_failover_handled(&encoder->governor);
    } /* pending->gpu_submitted */

    /* 4. Encode Slices */

    const char *fm = getenv("BC250_FAST_MODE");
    int deblock_idc = (fm && (strcmp(fm, "1") == 0 || strcmp(fm, "true") == 0)) ? 1 : 0;
    int slice_type = is_idr ? SLICE_TYPE_I : SLICE_TYPE_P;
    int poc_bits = encoder->sps.log2_max_poc_lsb + 4;
    int slice_qp_delta = qp - 26 - encoder->pps.pic_init_qp;

    /* Opt-in CPU-side CAVLC timing (BC250_PERF_STATS=1) - see this
     * function's top-of-body comment. Brackets the whole per-slice loop
     * below: slice header bits, the CAVLC macroblock layer itself
     * (encode_mb_i16x16()/encode_mb_p16x16(), the actual entropy coding),
     * and RBSP->EBSP/NAL assembly - i.e. everything this driver does on the
     * CPU per frame besides GPU dispatch/readback and rate-control/DPB
     * bookkeeping (which stay outside this bracket). */
    struct timespec cavlc_t0;
    if (perf_stats) clock_gettime(CLOCK_MONOTONIC, &cavlc_t0);

    /* Per-slice output tracking for OpenMP parallel entropy coding */
    typedef struct {
        uint8_t *slice_rbsp;
        size_t   rbsp_len;
        bool     overflow;
        bool     alloc_failed;
    } slice_output_t;

    slice_output_t slices[16];
    memset(slices, 0, sizeof(slices));

    int threads = get_default_slice_threads(num_slices);
    int num_threads_used = 1;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads) if(threads > 1 && num_slices > 1)
#endif
    for (int s = 0; s < num_slices; s++) {
#ifdef _OPENMP
        if (s == 0) num_threads_used = omp_get_num_threads();
#endif
        uint32_t start_mb = (uint32_t)(s * encoder->total_mbs / num_slices);
        uint32_t end_mb = (uint32_t)((s + 1) * encoder->total_mbs / num_slices);

        /* 64 bytes/MB was sized against this project's only tested content
         * (synthetic ffmpeg testsrc: flat gradients/simple motion, which
         * never has much real per-block AC energy) and was never enough for
         * real, busy content at a real client's negotiated QP - confirmed
         * on-hardware via a real live Sunshine/Moonlight session: a 2560x1440
         * real desktop capture at QP=12 overflowed this buffer partway
         * through a P-frame, and both bs_write_*() (bitstream.c's own
         * bs->overflow check) and cabac_encode_flush() (cabac.c's
         * cb->overflow check, see its "CABAC slice buffer overflow" log)
         * correctly detected it and stopped writing rather than corrupting
         * memory - but the resulting bitstream was silently truncated
         * exactly at that point, desyncing any real decoder from there
         * onward (reproduced offline: "error while decoding MB 72 73,
         * bytestream -59" then error-concealment bleeding stale reference
         * content into every subsequent P-frame, matching the real-client
         * corruption reports this was root-caused from). 768 bytes/MB gives
         * real headroom above a real near-lossless macroblock's worst case
         * (24 4x4 blocks/MB, each block's CABAC bypass-coded coefficients
         * bounded well under 32 bytes even at high magnitude) while still
         * being a trivial, transient per-slice allocation (~11MB for a full
         * 2560x1440 frame in one slice, freed immediately after this loop
         * body). See docs/DEVLOG.md for the full investigation.
         *
         * That 768 figure is CABAC-only reasoning and does not hold for
         * CAVLC (docs/DEVLOG.md Â§26.1.1, "ruled out as the SIGSEGV cause but
         * worth fixing on its own merits" - undersizing truncates the
         * bitstream rather than corrupting the heap, since bs_write_u() above
         * bounds-checks byte_offset before every store, but truncation still
         * desyncs the decoder exactly like the CABAC case above). CAVLC's
         * `level_prefix >= 15` escape path, combined with this driver's
         * int16_t-clamped `quant_levels`, lets a single coefficient level
         * reach `level_code = 65534`, `total = level_code + 4066`,
         * `suffix_size = 16` (since 2^16 <= total < 2^17), `prefix =
         * suffix_size + 3 = 19` - about 36 bits (~4.5 bytes) for that one
         * level, and a 4x4 block can carry up to 16 of them plus
         * coeff_token/total_zeros/run_before overhead, putting a single
         * worst-case CAVLC 4x4 block near 90-100 bytes rather than CABAC's
         * <32. Scaled to 24 blocks/MB that is ~2.0-2.5 KB/MB; 2560 bytes/MB
         * is used below to round up generously (correctness matters far more
         * than a few transient KB of malloc). This buffer is allocated fresh
         * per slice per frame, so keying it off encoder->use_cabac keeps the
         * common CABAC path at its original, already-validated size instead
         * of paying the CAVLC allowance unconditionally. */
        /* Give CABAC generous 1536 bytes/MB headroom (preventing overflows on dense
         * high-entropy 31M content) while preserving CAVLC's 2560 bytes/MB allowance. */
        size_t bytes_per_mb = encoder->use_cabac ? 1536 : 2560;
        size_t rbsp_buf_size = (end_mb - start_mb) * bytes_per_mb + 4096;
        uint8_t *slice_rbsp = malloc(rbsp_buf_size);
        if (!slice_rbsp) {
            slices[s].alloc_failed = true;
            continue;
        }
        slices[s].slice_rbsp = slice_rbsp;

        bitstream_t bs;
        bs_init(&bs, slice_rbsp, rbsp_buf_size);

        /* 4a. Slice Header per H.264 Section 7.3.3 */
        bs_write_ue(&bs, start_mb); /* first_mb_in_slice */
        bs_write_ue(&bs, (uint32_t)slice_type);
        bs_write_ue(&bs, (uint32_t)encoder->pps.pps_id);
        bs_write_u(&bs, encoder->sps.log2_max_frame_num + 4, encoder->frame_num);

        if (is_idr) {
            bs_write_ue(&bs, encoder->idr_pic_id);
        }

        bs_write_u(&bs, poc_bits, encoder->poc & ((1 << poc_bits) - 1));

        if (!is_idr) {
            bs_write1(&bs, 0); /* num_ref_idx_active_override_flag = 0 */
            bs_write1(&bs, 0); /* ref_pic_list_modification_flag_l0 = 0 */
            bs_write1(&bs, 0); /* adaptive_ref_pic_marking_mode_flag = 0 */
        } else {
            bs_write1(&bs, 0); /* no_output_of_prior_pics_flag = 0 */
            bs_write1(&bs, 0); /* long_term_reference_flag = 0 */
        }

        /* Per ITU-T H.264 7.3.3: cabac_init_idc is present iff
         * entropy_coding_mode_flag==1 AND slice_type is P/SP/B (never I/SI) -
         * and MUST come before slice_qp_delta, not after (easy to get
         * backwards - cabac_init_idc precedes slice_qp_delta in decoding
         * order per the spec's slice_header() syntax table). This project
         * always signals cabac_init_idc=0 - see cabac.c's top-of-file
         * comment for why only the idc==0 context-init table is needed. */
        if (encoder->use_cabac && !is_idr) {
            bs_write_ue(&bs, 0); /* cabac_init_idc = 0 */
        }

        bs_write_se(&bs, slice_qp_delta);
        bs_write_ue(&bs, (uint32_t)deblock_idc);
        /* Per ITU-T H.264 7.3.3: slice_alpha_c0_offset_div2/slice_beta_offset_div2
         * are only present when disable_deblocking_filter_idc != 1. Writing them
         * unconditionally (as this used to) inserts two spurious se(v) values into
         * the slice header whenever BC250_FAST_MODE=1 sets deblock_idc=1, silently
         * desyncing every bit of macroblock data that follows - confirmed via
         * real decode: FAST_MODE=1 produced a cascade of varied CAVLC/mb_type/qp
         * errors from MB 0 onward, while deblock_idc=0 (the default, and the only
         * value exercised by this session's earlier testing) was always clean. */
        if (deblock_idc != 1) {
            bs_write_se(&bs, 0);
            bs_write_se(&bs, 0);
        }

        /* 4b. Slice Data (Macroblock Layer) using CAVLC per Section 7.3.4 */
        nc_ctx_t nc = {
            .nz_luma = encoder->nz_luma,
            .nz_cb = encoder->nz_cb,
            .nz_cr = encoder->nz_cr,
            .width_in_mbs = encoder->width_in_mbs,
            .start_mb = start_mb,
            .nz_mask = nz_masks,
        };

        size_t rbsp_len;

        if (!encoder->use_cabac) {
        if (is_idr) {
            for (uint32_t mb = start_mb; mb < end_mb; mb++) {
                uint32_t mbx = mb % encoder->width_in_mbs;
                uint32_t mby = mb / encoder->width_in_mbs;
                if (quant_levels && dc_coeff) {
                    encode_mb_i16x16(&bs, quant_levels, dc_coeff, pred_modes, mb, mbx, mby, &nc, qp);
                } else {
                    /* No GPU residual data available (e.g. gpu_ctx==NULL) -
                     * fall back to an all-zero-residual I16x16 MB so the
                     * bitstream stays structurally valid. */
                    cavlc_write_mb_i16x16_header(&bs, H264_I16x16_DC, H264_CHROMA_DC, 0, 0, 0);
                    int16_t zero16[16] = {0};
                    cavlc_write_4x4_block(&bs, zero16, luma_nc(&nc, mb, mbx, mby, 0));
                    memset(nc.nz_luma[mb], 0, sizeof(nc.nz_luma[mb]));
                    memset(nc.nz_cb[mb], 0, sizeof(nc.nz_cb[mb]));
                    memset(nc.nz_cr[mb], 0, sizeof(nc.nz_cr[mb]));
                }
            }
        } else {
            uint32_t current_skip_run = 0;
            for (uint32_t mb = start_mb; mb < end_mb; mb++) {
                uint32_t mbx = mb % encoder->width_in_mbs;
                uint32_t mby = mb / encoder->width_in_mbs;
                /* BUG FIX (second, independent root cause of the ~13 dB
                 * quality_test.sh FAIL - confirmed via a minimal repro whose
                 * ffmpeg decode trace showed EVERY single P-frame macroblock
                 * decoding as skip ('S'), even under a rigid 16px/frame
                 * moving test box motion_estimation.comp tracked perfectly -
                 * see commit message): this used to treat "zero luma
                 * residual" as sufficient, alone, to emit a P_Skip macroblock.
                 * But per ITU-T H.264 8.4.1.1 / 7.4.5, a decoder reconstructs
                 * a P_Skip macroblock using mvL0 = the SAME median-of-
                 * neighbors predictor mv_predictor() computes below - NOT
                 * the encoder's actual searched motion vector - plus a
                 * zero residual. Skip is only a legal encoding when the real
                 * searched MV *equals* that predictor; otherwise the encoder
                 * MUST code the MB (as P_L0_16x16, mvd != 0) even though its
                 * residual is zero, purely to transmit the real motion. The
                 * old check ignored this entirely, so any MB whose optimal
                 * motion happened to produce a perfect (zero-residual) match
                 * - common when the search was integer-pel-only, since a
                 * plain per-pixel copy from the previous frame often matched
                 * exactly - got silently skipped regardless of how large its
                 * real motion was, forcing the decoder to reuse
                 * (0,0)-chained neighbor predictors and freeze that content
                 * at its previous position. This is exactly the "frames
                 * barely change despite real motion" symptom, independent
                 * of the mvd quarter-pel scaling bug fixed in
                 * encode_mb_p16x16 above (that bug corrupted MBs that DO get
                 * coded; this one wrongly avoided coding MBs that should
                 * be). UNCHANGED by the later sub-pel motion search update
                 * (see gpu_mv_t's UNITS note): mvx/mvy and pred_x/pred_y are
                 * compared in the same quarter-pel units on both sides here,
                 * so equality still means exactly what it needs to.
                 *
                 * SECOND, INDEPENDENT BUG FIX (P-slice MV predictor/decoder
                 * mismatch investigation): "that predictor" a decoder
                 * actually uses for a SKIPPED macroblock is NOT
                 * mv_predictor()'s plain 8.4.1.3 median - it is the P_Skip-
                 * specific 8.4.1.1 derivation (skip_mv_predictor(), see its
                 * doc comment), which zero-forces mvL0 whenever the left or
                 * top neighbor is unavailable or has mv==(0,0). Comparing
                 * against the plain median here (as this comment previously
                 * described and the code previously did) let a macroblock
                 * whose real motion matched the *plain* median, but which
                 * ALSO met the 8.4.1.1 zero-forcing condition, get certified
                 * skip-legal even though a real decoder reconstructs
                 * mvL0=(0,0) for it, not the real motion. That silently
                 * diverges the decoder's reference frame from the encoder's
                 * own at exactly that macroblock, which then poisons every
                 * later macroblock (in this frame and, via the P-chain,
                 * every subsequent frame) whose own median predictor reads
                 * this position as a neighbor - confirmed via
                 * BC250_DEBUG_MV_ROW showing real, on-the-wire wrong-skip
                 * events (e.g. frame=139 mbx=22 mby=58: real motion (35,0)
                 * wrongly certified skip-legal because a neighbor's mv was
                 * (0,0)) a few dozen frames upstream of, and one macroblock
                 * away from, the originally-reported gradient-boundary pixel
                 * defect at mbx=21/mby=59. See docs/DEVLOG.md. */
                bool zero_luma_residual = quant_levels ? (mb_has_any_luma_nonzero(quant_levels, nz_masks, mb) == 0) : true;
                /* See mb_has_any_chroma_nonzero()'s doc comment: a P_Skip MB
                 * must have zero residual for chroma too, not just luma. */
                bool zero_chroma_residual = (quant_levels && dc_coeff) ? (mb_has_any_chroma_nonzero(quant_levels, dc_coeff, nz_masks, mb) == 0) : true;
                if (encoder->quality_level >= 5 && zero_luma_residual && !zero_chroma_residual && dc_coeff) {
                    if (abs(dc_coeff[mb * 2 + 0]) <= 1 && abs(dc_coeff[mb * 2 + 1]) <= 1) {
                        zero_chroma_residual = true;
                    }
                }
                bool mv_matches_predictor = true;
                if (mvs) {
                    int pred_x, pred_y;
                    skip_mv_predictor(mvs, mbx, mby, nc.width_in_mbs, nc.start_mb, &pred_x, &pred_y);
                    mv_matches_predictor = (mvs[mb].mvx == pred_x && mvs[mb].mvy == pred_y);
                }
                bool mb_changed = quant_levels ? !(zero_luma_residual && zero_chroma_residual && mv_matches_predictor) : false;

                if (!mb_changed) {
                    current_skip_run++;
                    memset(nc.nz_luma[mb], 0, sizeof(nc.nz_luma[mb]));
                    memset(nc.nz_cb[mb], 0, sizeof(nc.nz_cb[mb]));
                    memset(nc.nz_cr[mb], 0, sizeof(nc.nz_cr[mb]));
                } else {
                    /* mb_skip_run MUST be written exactly once, unconditionally
                     * (even when 0), before every coded macroblock - see
                     * cavlc_write_p_skip_run's doc comment. */
                    cavlc_write_p_skip_run(&bs, current_skip_run);
                    current_skip_run = 0;
                    if (quant_levels && dc_coeff) {
                        encode_mb_p16x16(&bs, quant_levels, dc_coeff, mvs, mb, mbx, mby, &nc, qp);
                    } else {
                        cavlc_write_mb_p16x16_header(&bs, 0, 0, 0, 0);
                        memset(nc.nz_luma[mb], 0, sizeof(nc.nz_luma[mb]));
                        memset(nc.nz_cb[mb], 0, sizeof(nc.nz_cb[mb]));
                        memset(nc.nz_cr[mb], 0, sizeof(nc.nz_cr[mb]));
                    }
                }
            }
            if (current_skip_run > 0) {
                cavlc_write_p_skip_run(&bs, current_skip_run);
            }
        }

        /* 4c. RBSP Trailing bits (1 followed by zero bits to byte boundary) */
        cavlc_write_slice_trailing_bits(&bs);
        bs_flush(&bs);

        rbsp_len = bs_bytes_written(&bs);

        } else {
            /* ================================================================
             * CABAC slice data (ITU-T 7.3.4 with entropy_coding_mode_flag==1)
             * ================================================================
             * cabac_alignment_one_bit: byte-align the slice header (already
             * written bit-for-bit identically to the CAVLC path above, up to
             * and including deblocking params) with 1-bits, per spec -
             * NOT bs_flush()'s zero-fill, which is only correct for CAVLC's
             * own rbsp_trailing_bits(). If the header already landed on a
             * byte boundary this loop does nothing (bit_offset==0). */
            while (bs.bit_offset != 0) bs_write1(&bs, 1);

            uint8_t *cabac_start = slice_rbsp + bs.byte_offset;
            uint8_t *cabac_end = slice_rbsp + rbsp_buf_size;

            cabac_engine_t cabac;
            cabac_engine_init(&cabac, cabac_start, cabac_end);
            cabac_context_init(&cabac, is_idr /* is_intra_slice */, 0 /* cabac_init_idc */, qp);

            bool last_dqp_nonzero = false;

            if (is_idr) {
                for (uint32_t mb = start_mb; mb < end_mb; mb++) {
                    uint32_t mbx = mb % encoder->width_in_mbs;
                    uint32_t mby = mb / encoder->width_in_mbs;
                    if (quant_levels && dc_coeff) {
                        encode_mb_i16x16_cabac(&cabac, encoder, quant_levels, dc_coeff, pred_modes,
                                               mb, mbx, mby, &nc, qp, &last_dqp_nonzero);
                    } else {
                        /* No GPU residual data available - same all-zero
                         * fallback as the CAVLC path above, CABAC-coded. */
                        int ctx_intra = (mbx > 0 && (mb - 1) >= nc.start_mb ? 1 : 0) +
                                        (mby > 0 && (mb - nc.width_in_mbs) >= nc.start_mb ? 1 : 0);
                        cabac_write_mb_type_i16x16(&cabac, ctx_intra, false, 0, H264_I16x16_DC);
                        cabac_write_intra_chroma_pred_mode(&cabac, 0, 0);
                        last_dqp_nonzero = cabac_write_qp_delta(&cabac, 0, last_dqp_nonzero);
                        int nA, nB;
                        dc_cbf_neighbors(encoder->dc_cbf_luma, mb, mbx, mby, nc.width_in_mbs, nc.start_mb, true, &nA, &nB);
                        cabac_write_coded_block_flag(&cabac, CABAC_CAT_LUMA_DC, nA, nB, false);
                        encoder->dc_cbf_luma[mb] = 0;
                        encoder->dc_cbf_chroma[mb][0] = 0;
                        encoder->dc_cbf_chroma[mb][1] = 0;
                        memset(nc.nz_luma[mb], 0, sizeof(nc.nz_luma[mb]));
                        memset(nc.nz_cb[mb], 0, sizeof(nc.nz_cb[mb]));
                        memset(nc.nz_cr[mb], 0, sizeof(nc.nz_cr[mb]));
                    }
                    bool is_last_mb = (mb == end_mb - 1);
                    if (cabac_write_end_of_slice_flag(&cabac, is_last_mb)) break;
                }
            } else {
                for (uint32_t mb = start_mb; mb < end_mb; mb++) {
                    uint32_t mbx = mb % encoder->width_in_mbs;
                    uint32_t mby = mb / encoder->width_in_mbs;

                    /* Same skip decision as the CAVLC path above (see that
                     * branch's extensive doc comment on why zero-residual
                     * alone is not sufficient, AND on the second,
                     * independent skip_mv_predictor()/8.4.1.1 fix below) -
                     * CABAC just codes the decision differently (an explicit
                     * mb_skip_flag per MB, ITU-T 9.3.3.1.1.1, instead of an
                     * accumulated mb_skip_run). */
                    bool zero_luma_residual = quant_levels ? (mb_has_any_luma_nonzero(quant_levels, nz_masks, mb) == 0) : true;
                    /* See mb_has_any_chroma_nonzero()'s doc comment: a P_Skip
                     * MB must have zero residual for chroma too, not just
                     * luma. */
                    bool zero_chroma_residual = (quant_levels && dc_coeff) ? (mb_has_any_chroma_nonzero(quant_levels, dc_coeff, nz_masks, mb) == 0) : true;
                    if (encoder->quality_level >= 5 && zero_luma_residual && !zero_chroma_residual && dc_coeff) {
                        if (abs(dc_coeff[mb * 2 + 0]) <= 1 && abs(dc_coeff[mb * 2 + 1]) <= 1) {
                            zero_chroma_residual = true;
                        }
                    }
                    bool mv_matches_predictor = true;
                    if (mvs) {
                        int pred_x, pred_y;
                        skip_mv_predictor(mvs, mbx, mby, nc.width_in_mbs, nc.start_mb, &pred_x, &pred_y);
                        mv_matches_predictor = (mvs[mb].mvx == pred_x && mvs[mb].mvy == pred_y);
                    }
                    bool mb_changed = quant_levels ? !(zero_luma_residual && zero_chroma_residual && mv_matches_predictor) : false;
                    bool skip = !mb_changed;

                    int ctx_skip = ((mbx > 0 && (mb - 1) >= nc.start_mb && !encoder->skip_flag[mb - 1]) ? 1 : 0) +
                                   ((mby > 0 && (mb - nc.width_in_mbs) >= nc.start_mb && !encoder->skip_flag[mb - nc.width_in_mbs]) ? 1 : 0);
                    cabac_write_mb_skip_p(&cabac, ctx_skip, skip);

                    if (skip) {
                        encoder->skip_flag[mb] = 1;
                        encoder->mvd_x_abs[mb] = 0;
                        encoder->mvd_y_abs[mb] = 0;
                        encoder->cbp_nb[mb] = 0;
                        encoder->dc_cbf_chroma[mb][0] = 0;
                        encoder->dc_cbf_chroma[mb][1] = 0;
                        memset(nc.nz_luma[mb], 0, sizeof(nc.nz_luma[mb]));
                        memset(nc.nz_cb[mb], 0, sizeof(nc.nz_cb[mb]));
                        memset(nc.nz_cr[mb], 0, sizeof(nc.nz_cr[mb]));
                    } else {
                        encoder->skip_flag[mb] = 0;
                        if (quant_levels && dc_coeff) {
                            encode_mb_p16x16_cabac(&cabac, encoder, quant_levels, dc_coeff, mvs,
                                                   mb, mbx, mby, &nc, qp, &last_dqp_nonzero);
                        } else {
                            cabac_write_mb_type_p_l0_16x16(&cabac);
                            int left_x = (mbx > 0 && (mb - 1) >= nc.start_mb) ? encoder->mvd_x_abs[mb - 1] : 0;
                            int top_x  = (mby > 0 && (mb - nc.width_in_mbs) >= nc.start_mb) ? encoder->mvd_x_abs[mb - nc.width_in_mbs] : 0;
                            int left_y = (mbx > 0 && (mb - 1) >= nc.start_mb) ? encoder->mvd_y_abs[mb - 1] : 0;
                            int top_y  = (mby > 0 && (mb - nc.width_in_mbs) >= nc.start_mb) ? encoder->mvd_y_abs[mb - nc.width_in_mbs] : 0;
                            cabac_write_mvd_component(&cabac, 0, cabac_mvd_ctx_from_neighbors(left_x, top_x), 0);
                            cabac_write_mvd_component(&cabac, 1, cabac_mvd_ctx_from_neighbors(left_y, top_y), 0);
                            int cbp_l = (mbx > 0 && (mb - 1) >= nc.start_mb) ? encoder->cbp_nb[mb - 1] : -1;
                            int cbp_t = (mby > 0 && (mb - nc.width_in_mbs) >= nc.start_mb) ? encoder->cbp_nb[mb - nc.width_in_mbs] : -1;
                            cabac_write_cbp_luma(&cabac, 0, cbp_l, cbp_t);
                            cabac_write_cbp_chroma(&cabac, 0, cbp_l, cbp_t);
                            encoder->cbp_nb[mb] = 0;
                            encoder->mvd_x_abs[mb] = 0;
                            encoder->mvd_y_abs[mb] = 0;
                            encoder->dc_cbf_chroma[mb][0] = 0;
                            encoder->dc_cbf_chroma[mb][1] = 0;
                            memset(nc.nz_luma[mb], 0, sizeof(nc.nz_luma[mb]));
                            memset(nc.nz_cb[mb], 0, sizeof(nc.nz_cb[mb]));
                            memset(nc.nz_cr[mb], 0, sizeof(nc.nz_cr[mb]));
                        }
                    }

                    bool is_last_mb = (mb == end_mb - 1);
                    if (cabac_write_end_of_slice_flag(&cabac, is_last_mb)) break;
                }
            }

            /* cabac_encode_flush() returns the byte count relative to
             * cabac_start (== cabac.p_start), NOT from the beginning of
             * slice_rbsp - the full RBSP length step 5 below needs also
             * includes the slice-header + alignment bytes already committed
             * to slice_rbsp[0 .. bs.byte_offset) before cabac_start. */
            rbsp_len = (size_t)bs.byte_offset + cabac_encode_flush(&cabac, encoder->frame_count);
            if (cabac.overflow) {
                fprintf(stderr, "[bc250-h264] CABAC slice buffer overflow (frame=%u slice=%d)\n",
                        encoder->frame_count, s);
                slices[s].overflow = true;
            }
        }

        slices[s].rbsp_len = rbsp_len;
    }

    /* 5. Assemble Slice NAL units sequentially: 4-byte start code + NAL header + EBSP */
    bool slice_overflow = false;
    for (int s = 0; s < num_slices; s++) {
        if (slices[s].alloc_failed || slices[s].overflow) {
            slice_overflow = true;
        }

        if (!slice_overflow) {
            size_t rbsp_len = slices[s].rbsp_len;
            size_t needed = total_written + 5 + rbsp_len * 2;
            if (needed > encoder->output_buf_size) {
                size_t new_cap = encoder->output_buf_size * 2;
                if (new_cap < needed + 65536) new_cap = needed + 65536;
                uint8_t *new_buf = realloc(encoder->output_buf, new_cap);
                if (new_buf) {
                    encoder->output_buf = new_buf;
                    encoder->output_buf_size = new_cap;
                }
            }

            if (total_written + 5 + rbsp_len * 2 <= encoder->output_buf_size) {
                uint8_t *nal_dst = encoder->output_buf + total_written;
                nal_dst[0] = 0x00;
                nal_dst[1] = 0x00;
                nal_dst[2] = 0x00;
                nal_dst[3] = 0x01;
                nal_dst[4] = is_idr ? ((NAL_REF_IDC_HIGH << 5) | NAL_TYPE_IDR_SLICE)
                                    : ((NAL_REF_IDC_MEDIUM << 5) | NAL_TYPE_SLICE);

                size_t ebsp_len = bs_rbsp_to_ebsp(nal_dst + 5,
                                                  encoder->output_buf_size - total_written - 5,
                                                  slices[s].slice_rbsp,
                                                  rbsp_len);
                total_written += 5 + ebsp_len;
            } else {
                fprintf(stderr, "[bc250-h264] slice %d/%d does not fit output_buf "
                                "(have %zu, used %zu, need %zu) - abandoning frame %u at qp=%d\n",
                        s, num_slices, encoder->output_buf_size, total_written,
                        (size_t)(5 + rbsp_len * 2), encoder->frame_count, qp);
                slice_overflow = true;
            }
        }

        if (slices[s].slice_rbsp) {
            free(slices[s].slice_rbsp);
            slices[s].slice_rbsp = NULL;
        }
    }

    if (slice_overflow) {
        return -1;
    }

    /* CBR filler padding - see maybe_append_filler()'s doc comment. Applied
     * after all real slice data is written and before the perf/size/copy
     * bookkeeping below, so BC250_PERF_STATS' bytes=%zu, the output_size
     * guard and the memcpy all see the real final (possibly padded) frame
     * size - exactly what a downstream consumer would actually receive.
     *
     * rc_update_stats() is the ONE consumer that must NOT see the padded
     * size: filler is manufactured to make bits_used equal the per-frame
     * target, so feeding the padded total back into the buffer model makes
     * bits_used cancel the drain exactly and buffer_fullness can never
     * fall. Measured consequence on real hardware (docs/DEVLOG.md Â§16): a
     * large opening IDR pins buffer_fullness at its clamp, error stays
     * positive forever, the integral term winds to full range, and QP
     * sticks at qp_max=51 for the entire session while every frame is
     * padded back up to the target - i.e. the picture is coded at the
     * worst possible quality and the bitrate is spent on padding, which no
     * amount of extra requested bitrate can improve. Real coded bits are
     * what the feedback loop has to see; with them, QP 51 produces a tiny
     * frame, the buffer drains, and QP recovers the way a rate controller
     * is supposed to. */
    size_t real_coded_bytes = total_written;
    total_written = maybe_append_filler(encoder, total_written);

    if (perf_stats) {
        struct timespec cavlc_t1;
        clock_gettime(CLOCK_MONOTONIC, &cavlc_t1);
        double cavlc_ms = (double)(cavlc_t1.tv_sec - cavlc_t0.tv_sec) * 1000.0 +
                          (double)(cavlc_t1.tv_nsec - cavlc_t0.tv_nsec) / 1e6;
        fprintf(stderr, "[BC250_PERF_CPU] frame=%u type=%s cavlc_ms=%.3f slices=%d threads=%d coder=%s\n",
                encoder->frame_count, is_idr ? "I" : "P", cavlc_ms,
                num_slices, num_threads_used, encoder->use_cabac ? "CABAC" : "CAVLC");
    }

    if (output_size < total_written) {
        fprintf(stderr, "[bc250-h264] Output buffer too small: need %zu, have %zu\n",
                total_written, output_size);
        return -1;
    }
    memcpy(output_buf, encoder->output_buf, total_written);

    /* Real coded bits only - see the maybe_append_filler() comment above. */
    rc_update_stats(&encoder->rc, (int)(real_coded_bytes * 8));
    manage_dpb(encoder, encoder->frame_num, encoder->poc);

    encoder->frame_num++;
    encoder->poc += 2;
    encoder->frame_count++;

    if (perf_stats) {
        struct timespec frame_t1;
        clock_gettime(CLOCK_MONOTONIC, &frame_t1);
        double wall_ms = (double)(frame_t1.tv_sec - frame_t0.tv_sec) * 1000.0 +
                         (double)(frame_t1.tv_nsec - frame_t0.tv_nsec) / 1e6;
        fprintf(stderr, "[BC250_PERF_FRAME] frame=%u type=%s wall_ms=%.3f bytes=%zu qp=%d "
                        "meas_fps=%.1f target_bpf=%u\n",
                encoder->frame_count - 1, is_idr ? "I" : "P", wall_ms, total_written,
                qp, encoder->rc.measured_fps, encoder->rc.target_bits_per_frame);
    }

    return (int)total_written;
}

int h264_encoder_encode_raw(h264_encoder_t *encoder,
                            const uint8_t *y_plane, int y_pitch,
                            const uint8_t *uv_plane, int uv_pitch,
                            uint8_t *output_buf, size_t output_size)
{
    (void)uv_plane; (void)uv_pitch;
    if (!encoder || !output_buf || !y_plane) return -1;

    bool is_idr = (encoder->frame_count % encoder->gop_size == 0) || encoder->force_idr;
    encoder->force_idr = false;

    if (is_idr) {
        encoder->frame_num = 0;
        encoder->idr_pic_id++;
        encoder->poc = 0;
        encoder->dpb_count = 0;
        encoder->last_frame_sad = 0;
    }

    int qp = rc_get_frame_qp(&encoder->rc, encoder->last_frame_sad);
    /* Must track rate_control.c's rc->qp_min - see the comment there for
     * why 12 is deliberate and what lowering it measured. */
    if (qp < encoder->rc.qp_min) qp = encoder->rc.qp_min;
    if (qp > 51) qp = 51;
    qp = apply_qp_override(qp);

    size_t total_written = 0;

    /* 1. Write AUD */
    total_written += write_aud(encoder->output_buf + total_written,
                               encoder->output_buf_size - total_written,
                               is_idr);

    /* 2. Write SPS / PPS on IDR */
    if (is_idr) {
        total_written += bs_write_sps(encoder->output_buf + total_written,
                                      encoder->output_buf_size - total_written,
                                      &encoder->sps);
        total_written += bs_write_pps(encoder->output_buf + total_written,
                                      encoder->output_buf_size - total_written,
                                      &encoder->pps);
    }

    /* 3. Encode Slices (supporting multi-slice partitioning for network resilience) */
    int num_slices = (encoder->num_slices >= 1 && encoder->num_slices <= 16) ? encoder->num_slices : 1;

    const char *fm = getenv("BC250_FAST_MODE");
    int deblock_idc = (fm && (strcmp(fm, "1") == 0 || strcmp(fm, "true") == 0)) ? 1 : 0;
    int slice_type = is_idr ? SLICE_TYPE_I : SLICE_TYPE_P;
    int poc_bits = encoder->sps.log2_max_poc_lsb + 4;
    int slice_qp_delta = qp - 26 - encoder->pps.pic_init_qp;

    typedef struct {
        uint8_t *slice_rbsp;
        size_t   rbsp_len;
        bool     overflow;
        bool     alloc_failed;
    } raw_slice_output_t;

    raw_slice_output_t slices[16];
    memset(slices, 0, sizeof(slices));

    int threads = get_default_slice_threads(num_slices);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads) if(threads > 1 && num_slices > 1)
#endif
    for (int s = 0; s < num_slices; s++) {
        uint32_t start_mb = (uint32_t)(s * encoder->total_mbs / num_slices);
        uint32_t end_mb = (uint32_t)((s + 1) * encoder->total_mbs / num_slices);

        size_t rbsp_buf_size = (end_mb - start_mb) * 64 + 4096;
        uint8_t *slice_rbsp = malloc(rbsp_buf_size);
        if (!slice_rbsp) {
            slices[s].alloc_failed = true;
            continue;
        }
        slices[s].slice_rbsp = slice_rbsp;

        bitstream_t bs;
        bs_init(&bs, slice_rbsp, rbsp_buf_size);

        bs_write_ue(&bs, start_mb); /* first_mb_in_slice */
        bs_write_ue(&bs, (uint32_t)slice_type);
        bs_write_ue(&bs, (uint32_t)encoder->pps.pps_id);
        bs_write_u(&bs, encoder->sps.log2_max_frame_num + 4, encoder->frame_num);

        if (is_idr) {
            bs_write_ue(&bs, encoder->idr_pic_id);
        }

        bs_write_u(&bs, poc_bits, encoder->poc & ((1 << poc_bits) - 1));

        if (!is_idr) {
            bs_write1(&bs, 0);
            bs_write1(&bs, 0);
            bs_write1(&bs, 0);
        } else {
            bs_write1(&bs, 0);
            bs_write1(&bs, 0);
        }

        bs_write_se(&bs, slice_qp_delta);
        bs_write_ue(&bs, (uint32_t)deblock_idc);
        /* Per ITU-T H.264 7.3.3: slice_alpha_c0_offset_div2/slice_beta_offset_div2
         * are only present when disable_deblocking_filter_idc != 1. Writing them
         * unconditionally (as this used to) inserts two spurious se(v) values into
         * the slice header whenever BC250_FAST_MODE=1 sets deblock_idc=1, silently
         * desyncing every bit of macroblock data that follows - confirmed via
         * real decode: FAST_MODE=1 produced a cascade of varied CAVLC/mb_type/qp
         * errors from MB 0 onward, while deblock_idc=0 (the default, and the only
         * value exercised by this session's earlier testing) was always clean. */
        if (deblock_idc != 1) {
            bs_write_se(&bs, 0);
            bs_write_se(&bs, 0);
        }

        /* Macroblock layer */
        if (is_idr) {
            for (uint32_t mb = start_mb; mb < end_mb; mb++) {
                uint32_t mby = mb / encoder->width_in_mbs;
                uint32_t mbx = mb % encoder->width_in_mbs;
                uint32_t v_diff = 0, h_diff = 0;
                for (int r = 0; r < 15; r++) {
                    uint32_t py = mby * 16 + r;
                    if (py >= encoder->height - 1) break;
                    for (int c = 0; c < 15; c++) {
                        uint32_t px = mbx * 16 + c;
                        if (px >= encoder->width - 1) break;
                        const uint8_t *p = y_plane + py * y_pitch + px;
                        v_diff += abs((int)p[0] - (int)p[y_pitch]);
                        h_diff += abs((int)p[0] - (int)p[1]);
                    }
                }
                int mode = H264_I16x16_DC;
                if (v_diff * 3 < h_diff * 2) mode = H264_I16x16_VERT;
                else if (h_diff * 3 < v_diff * 2) mode = H264_I16x16_HORIZ;
                bool left_avail = (mbx > 0 && (mb - 1) >= start_mb);
                bool top_avail  = (mby > 0 && (mb - encoder->width_in_mbs) >= start_mb);
                mode = h264_sanitize_i16_mode(mode, top_avail, left_avail);
                cavlc_write_mb_i16x16_header(&bs, mode, H264_CHROMA_DC, 0, 0, 0);
            }
        } else {
            uint32_t current_skip_run = 0;
            for (uint32_t mb = start_mb; mb < end_mb; mb++) {
                uint32_t mby = mb / encoder->width_in_mbs;
                uint32_t mbx = mb % encoder->width_in_mbs;
                uint32_t sad = 0;
                if (encoder->has_prev_frame && encoder->prev_y_frame) {
                    for (int r = 0; r < 16; r++) {
                        uint32_t py = mby * 16 + r;
                        if (py >= encoder->height) break;
                        for (int c = 0; c < 16; c++) {
                            uint32_t px = mbx * 16 + c;
                            if (px >= encoder->width) break;
                            int curr = y_plane[py * y_pitch + px];
                            int prev = encoder->prev_y_frame[py * encoder->width + px];
                            sad += abs(curr - prev);
                        }
                    }
                }
                if (sad < 512) {
                    current_skip_run++;
                } else {
                    /* mb_skip_run MUST be written exactly once, unconditionally
                     * (even when 0), before every coded macroblock - see
                     * cavlc_write_p_skip_run's doc comment. */
                    cavlc_write_p_skip_run(&bs, current_skip_run);
                    current_skip_run = 0;
                    int best_dx = 0, best_dy = 0;
                    uint32_t best_sad = sad;
                    if (encoder->has_prev_frame && encoder->prev_y_frame) {
                        static const int cand_mvs[8][2] = {
                            {-1, 0}, {1, 0}, {0, -1}, {0, 1},
                            {-2, 0}, {2, 0}, {0, -2}, {0, 2}
                        };
                        for (int d = 0; d < 8; d++) {
                            int dx = cand_mvs[d][0];
                            int dy = cand_mvs[d][1];
                            uint32_t cand_sad = 0;
                            for (int r = 0; r < 16; r += 2) {
                                int py = (int)(mby * 16 + r);
                                int ref_py = py + dy;
                                if (py >= (int)encoder->height || ref_py < 0 || ref_py >= (int)encoder->height) {
                                    cand_sad += 255 * 8;
                                    continue;
                                }
                                for (int c = 0; c < 16; c += 2) {
                                    int px = (int)(mbx * 16 + c);
                                    int ref_px = px + dx;
                                    if (px >= (int)encoder->width || ref_px < 0 || ref_px >= (int)encoder->width) {
                                        cand_sad += 255;
                                        continue;
                                    }
                                    cand_sad += abs((int)y_plane[py * y_pitch + px] -
                                                    (int)encoder->prev_y_frame[ref_py * encoder->width + ref_px]) * 4;
                                }
                            }
                            if (cand_sad < best_sad) {
                                best_sad = cand_sad;
                                best_dx = dx;
                                best_dy = dy;
                            }
                        }
                    }
                    cavlc_write_mb_p16x16_header(&bs, best_dx * 4, best_dy * 4, 0, 0);
                }
            }
            if (current_skip_run > 0) {
                cavlc_write_p_skip_run(&bs, current_skip_run);
            }
        }

        cavlc_write_slice_trailing_bits(&bs);
        bs_flush(&bs);

        slices[s].rbsp_len = bs_bytes_written(&bs);
    }

    bool raw_slice_overflow = false;
    for (int s = 0; s < num_slices; s++) {
        if (slices[s].alloc_failed || slices[s].overflow) {
            raw_slice_overflow = true;
        }

        if (!raw_slice_overflow) {
            size_t rbsp_len = slices[s].rbsp_len;
            size_t needed = total_written + 5 + rbsp_len * 2;
            if (needed > encoder->output_buf_size) {
                size_t new_cap = encoder->output_buf_size * 2;
                if (new_cap < needed + 65536) new_cap = needed + 65536;
                uint8_t *new_buf = realloc(encoder->output_buf, new_cap);
                if (new_buf) {
                    encoder->output_buf = new_buf;
                    encoder->output_buf_size = new_cap;
                }
            }

            if (total_written + 5 + rbsp_len * 2 <= encoder->output_buf_size) {
                uint8_t *nal_dst = encoder->output_buf + total_written;
                nal_dst[0] = 0x00;
                nal_dst[1] = 0x00;
                nal_dst[2] = 0x00;
                nal_dst[3] = 0x01;
                nal_dst[4] = is_idr ? ((NAL_REF_IDC_HIGH << 5) | NAL_TYPE_IDR_SLICE)
                                    : ((NAL_REF_IDC_MEDIUM << 5) | NAL_TYPE_SLICE);

                size_t ebsp_len = bs_rbsp_to_ebsp(nal_dst + 5,
                                                  encoder->output_buf_size - total_written - 5,
                                                  slices[s].slice_rbsp,
                                                  rbsp_len);
                total_written += 5 + ebsp_len;
            } else {
                fprintf(stderr, "[bc250-h264] slice %d/%d does not fit output_buf "
                                "(have %zu, used %zu, need %zu) - abandoning frame %u at qp=%d\n",
                        s, num_slices, encoder->output_buf_size, total_written,
                        (size_t)(5 + rbsp_len * 2), encoder->frame_count, qp);
                raw_slice_overflow = true;
            }
        }

        if (slices[s].slice_rbsp) {
            free(slices[s].slice_rbsp);
            slices[s].slice_rbsp = NULL;
        }
    }

    if (raw_slice_overflow) {
        return -1;
    }

    /* CBR filler padding - see maybe_append_filler()'s doc comment. Same
     * placement rationale as h264_encoder_encode_frame: before the
     * output_size guard/memcpy below, so they see the real final (possibly
     * padded) frame size - but rc_update_stats() below is fed the
     * pre-filler size, for the reason documented at the corresponding
     * point in h264_encoder_encode_frame(). */
    size_t real_coded_bytes = total_written;
    total_written = maybe_append_filler(encoder, total_written);

    if (encoder->prev_y_frame) {
        for (uint32_t r = 0; r < encoder->height; r++) {
            memcpy(encoder->prev_y_frame + r * encoder->width,
                   y_plane + r * y_pitch,
                   encoder->width);
        }
        encoder->has_prev_frame = true;
    }

    if (output_size < total_written) {
        return -1;
    }
    memcpy(output_buf, encoder->output_buf, total_written);

    /* Real coded bits only - see the maybe_append_filler() comment above. */
    rc_update_stats(&encoder->rc, (int)(real_coded_bytes * 8));
    manage_dpb(encoder, encoder->frame_num, encoder->poc);

    encoder->frame_num++;
    encoder->poc += 2;
    encoder->frame_count++;

    return (int)total_written;
}

void h264_encoder_destroy(h264_encoder_t *encoder)
{
#ifdef BC250_HAVE_X264
    if (encoder) {
        h264_x264_destroy(encoder->x264);
        free(encoder->x264_y);
        free(encoder->x264_uv);
    }
#endif
    if (!encoder) return;
    if (encoder->output_buf) free(encoder->output_buf);
    if (encoder->prev_y_frame) free(encoder->prev_y_frame);
    if (encoder->nz_luma) free(encoder->nz_luma);
    if (encoder->nz_cb) free(encoder->nz_cb);
    if (encoder->nz_cr) free(encoder->nz_cr);
    if (encoder->dc_cbf_luma) free(encoder->dc_cbf_luma);
    if (encoder->dc_cbf_chroma) free(encoder->dc_cbf_chroma);
    if (encoder->cbp_nb) free(encoder->cbp_nb);
    if (encoder->mvd_x_abs) free(encoder->mvd_x_abs);
    if (encoder->mvd_y_abs) free(encoder->mvd_y_abs);
    if (encoder->skip_flag) free(encoder->skip_flag);
    if (encoder->quant_levels_shadow) free(encoder->quant_levels_shadow);
    if (encoder->dc_coeff_shadow) free(encoder->dc_coeff_shadow);
    if (encoder->pred_modes_shadow) free(encoder->pred_modes_shadow);
    if (encoder->mvs_shadow) free(encoder->mvs_shadow);
    if (encoder->nz_masks_shadow) free(encoder->nz_masks_shadow);
    if (encoder->cpu_mvs) free(encoder->cpu_mvs);
    free(encoder);
}
