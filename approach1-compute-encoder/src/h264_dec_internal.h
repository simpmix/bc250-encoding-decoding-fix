/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * h264_dec_internal.h - the decoder's own state, shared between the picture
 * level in decoder_h264.c and the macroblock layer in h264_mb.c.
 *
 * Nothing here is part of the driver-facing interface; decoder_h264.h is.
 */
#ifndef BC250_H264_DEC_INTERNAL_H
#define BC250_H264_DEC_INTERNAL_H

#include "decoder_h264.h"
#include "h264_cabac_dec.h"
#include "h264_deblock.h"
#include "h264_recon.h"
#include "bitreader.h"

#define H264D_MAX_SLICES 256

/* Context index bases, Rec. ITU-T H.264 Table 9-11 and Tables 9-34 to 9-40.
 * The per-block-category bases for the residual live in the generated
 * tables; these are the ones a single syntax element needs. */
enum {
    CTX_MB_TYPE_I          = 3,
    CTX_MB_SKIP_P          = 11,
    CTX_MB_TYPE_P          = 14,
    CTX_SUB_MB_TYPE_P      = 21,
    CTX_MB_SKIP_B          = 24,
    CTX_MB_TYPE_B          = 27,
    CTX_SUB_MB_TYPE_B      = 36,
    CTX_MVD_X              = 40,
    CTX_MVD_Y              = 47,
    CTX_REF_IDX            = 54,
    CTX_MB_QP_DELTA        = 60,
    CTX_INTRA_CHROMA_PRED  = 64,
    CTX_PREV_INTRA4X4_FLAG = 68,
    CTX_REM_INTRA4X4       = 69,
    CTX_CBP_LUMA           = 73,
    CTX_CBP_CHROMA         = 77,
    CTX_CBF                = 85,      /* + 4 * ctxBlockCat, cats 0..4 */
    CTX_TRANSFORM_8X8      = 399,
    CTX_CBF_8X8            = 1012,    /* cat 5, unused at 4:2:0 - see below */
};

/* Block categories, Table 9-42. */
enum {
    CAT_I16_DC = 0,
    CAT_I16_AC = 1,
    CAT_LUMA_4X4 = 2,
    CAT_CHROMA_DC = 3,
    CAT_CHROMA_AC = 4,
    CAT_LUMA_8X8 = 5,
};

/* One macroblock's residual, as the entropy decoder leaves it: coefficients
 * in raster order inside each block, ready for the inverse transforms.
 *
 * ⚠️ coeff and coeff8 are alternatives - a macroblock uses the 4x4
 * transform or the 8x8 one, never both - but they are kept side by side
 * rather than in a union, because the debug dumps read whichever one the
 * macroblock did not use and a union would make that undefined. */
typedef struct {
    int16_t luma[16][16];         /* [block][coefficient], raster inside */
    int16_t chroma[2][4][16];      /* ⚠️ four blocks a plane at 4:2:0 */
    int16_t coeff8[4][64];        /* the 8x8 transform's luma blocks */
    int16_t dc_luma[16];
    int16_t dc_chroma[2][4];
} h264d_residual_t;

/* One decoded picture's worth of working state. */
struct h264_decoder {
    bc250_gpu_context_t *gpu;
    int width, height;
    int mb_w, mb_h, mb_count;

    h264d_pic_t pic;
    h264d_slice_t slice;

    /* The frame store. Slot H264D_DPB_SIZE - 1 is never a reference; it is
     * where a non-reference picture is decoded. */
    h264d_frame_t dpb[H264D_DPB_SIZE];
    int cur;                      /* slot being decoded */

    h264d_mb_t *mbs;
    uint8_t *slice_of_mb;
    h264d_deblock_params_t deblock[H264D_MAX_SLICES];
    /* ⚠️ Kept by slice number, not just in `slice`: reconstruction reads
     * the weights and the reference lists, and it runs after the entropy
     * decoder has moved on. */
    /* ⚠️ On the heap, not in the structure: a worker copies the decoder
     * to use as a cursor, and 256 slices of parameters would be copied
     * with it. */
    h264d_slice_t *slices;
    int n_slices;

    struct h264d_pool *pool;      /* reconstruction threads, or NULL */

    /* Entropy state for the slice being decoded. */
    h264d_cabac_t cabac;
    br_t br;
    bool cabac_mode;
    uint8_t *rbsp;                /* slice data with emulation bytes removed */
    size_t rbsp_cap;

    /* Rebuilt whenever QP changes, which is rarely. Chroma gets its own two
     * because its QP comes from a different table and each plane has its own
     * offset. */
    h264d_dequant_set_t *dequant;   /* shared, built once a picture */

    int qpy;                      /* running QP through the slice */
    int last_qp_delta_nonzero;    /* the mb_qp_delta context needs it */
    int mb_x, mb_y, mb_idx;
    int prev_mb_skipped;
    int mb_skip_run;              /* CAVLC only */

    /* One residual per macroblock. The entropy decoder fills them in and
     * reconstruction reads them; keeping them apart is what lets the
     * second run behind the first. */
    h264d_residual_t *residuals;
    int n_residuals;                /* a ring of band_rows + 1 rows */
    int band_rows;
    int to_reconstruct;           /* first macroblock not yet reconstructed */
    h264d_residual_t *res;         /* the one being filled in right now */

    /* Cached chroma scratch buffer for NV12 upload */
    uint8_t *uv_buf;
    size_t uv_cap;
};

/* ---- neighbours ------------------------------------------------------- */

/* Whether the macroblock to the left and above exist AND belong to the same
 * slice. A macroblock in another slice is not available for prediction: a
 * slice has to be decodable on its own, which is the whole reason slices
 * exist. The deblocking filter is the one place that may cross the boundary,
 * and it decides separately. */
static inline const h264d_mb_t *h264d_mb_left(const h264_decoder_t *d)
{
    if (d->mb_x == 0) return NULL;
    if (d->slice_of_mb[d->mb_idx - 1] != d->slice_of_mb[d->mb_idx]) return NULL;
    return &d->mbs[d->mb_idx - 1];
}

static inline const h264d_mb_t *h264d_mb_top(const h264_decoder_t *d)
{
    if (d->mb_y == 0) return NULL;
    if (d->slice_of_mb[d->mb_idx - d->mb_w] != d->slice_of_mb[d->mb_idx]) return NULL;
    return &d->mbs[d->mb_idx - d->mb_w];
}

static inline const h264d_mb_t *h264d_mb_top_left(const h264_decoder_t *d)
{
    if (d->mb_x == 0 || d->mb_y == 0) return NULL;
    if (d->slice_of_mb[d->mb_idx - d->mb_w - 1] != d->slice_of_mb[d->mb_idx]) return NULL;
    return &d->mbs[d->mb_idx - d->mb_w - 1];
}

static inline const h264d_mb_t *h264d_mb_top_right(const h264_decoder_t *d)
{
    if (d->mb_x + 1 >= d->mb_w || d->mb_y == 0) return NULL;
    if (d->slice_of_mb[d->mb_idx - d->mb_w + 1] != d->slice_of_mb[d->mb_idx]) return NULL;
    return &d->mbs[d->mb_idx - d->mb_w + 1];
}

/* The residual slot this macroblock fills in. */
static inline void h264d_residual_at(h264_decoder_t *d)
{
    d->res = &d->residuals[d->mb_idx % d->n_residuals];
}

/* ---- the macroblock layer --------------------------------------------- */

/* Decode one macroblock's syntax and reconstruct it. Returns non-zero when
 * the slice must stop. */
int h264d_decode_mb_cabac(h264_decoder_t *d);
int h264d_decode_mb_cavlc(h264_decoder_t *d);

/* One skipped macroblock of a CAVLC slice. CABAC has no equivalent call
 * because there the skip is a flag the macroblock layer reads itself. */
int h264d_cavlc_skip(h264_decoder_t *d);

/* Reconstruct one macroblock. `d` here is a cursor - a copy of the
 * decoder with its own mb_x, mb_y, mb_idx, res and slice - because this
 * reads the decoder and never writes to it, which is what lets several
 * threads run it at once over the same picture. */
void h264d_reconstruct_mb(h264_decoder_t *d);

/* Reconstruct a run of macroblocks of one slice: as a wavefront across
 * the thread pool when there is one and the run is long enough to pay for
 * it, in order on this thread otherwise. Either way the picture is the
 * same to the byte. */
void h264d_reconstruct_range(h264_decoder_t *d, int first, int count,
                             int slice_number);

/* The reconstruction threads. Started on the first picture and kept until
 * the decoder is destroyed; BC250_H264_THREADS sets how many, and 1 turns
 * them off. */
struct h264d_pool;
int h264d_pool_start(h264_decoder_t *d);
void h264d_pool_stop(h264_decoder_t *d);

/* The deblocking filter over a whole picture, threaded when there is a
 * pool. Same result as h264d_deblock_picture(), to the byte. */
void h264d_deblock_wavefront(h264_decoder_t *d, const h264d_deblock_pic_t *dp);

/* One slice on the cursor it is handed. The pool calls this on a copy of
 * the decoder with buffers of its own. */
int h264d_decode_slice(h264_decoder_t *d, const h264d_slice_input_t *in,
                           int number);

/* Several slices at once, one per worker. Returns the first failure. */
int h264d_slices_pool(h264_decoder_t *d, const h264d_slice_input_t *in,
                      int n, int first_number);

/* Motion vector prediction, 8.4.1.3. */
void h264d_predict_mv(h264_decoder_t *d, int list, int blk, int w4, int h4,
                      int ref_idx, int16_t out[2]);

/* The vector of a skipped P macroblock, 8.4.1.1. */
void h264d_skip_mv_p(h264_decoder_t *d, int16_t out[2]);

/* Spatial direct prediction for a B macroblock, 8.4.1.2.2. `mask` says
 * which of the four 8x8 partitions to fill in, because a B_8x8 can be
 * direct in some of them and explicit in the rest. */
int h264d_direct_spatial(h264_decoder_t *d, h264d_mb_t *m, int mask);

/* Temporal direct prediction, 8.4.1.2.2's other half: 8.4.1.2.3. Same
 * arguments, and the co-located block's vector rescaled by picture
 * distance instead of the neighbours' consensus. */
int h264d_direct_temporal(h264_decoder_t *d, h264d_mb_t *m, int mask);

/* Whichever one direct_spatial_mv_pred_flag asked for. */
int h264d_direct(h264_decoder_t *d, h264d_mb_t *m, int mask);

#endif /* BC250_H264_DEC_INTERNAL_H */
