/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * hevc_dec_internal.h - the H.265 decoder's own state.
 *
 * H.265 replaces the macroblock with a coding tree unit that splits down a
 * quadtree, so almost nothing here is indexed per macroblock. What the
 * syntax needs from its neighbours - the depth a coding unit was split to,
 * the intra prediction mode a prediction unit chose - is kept on a grid of
 * the smallest block each of those can be, and read at whatever size the
 * question is asked.
 */
#ifndef BC250_HEVC_DEC_INTERNAL_H
#define BC250_HEVC_DEC_INTERNAL_H

#include "hevc_ps.h"

#include <pthread.h>
#include "hevc_cabac_dec.h"

/* Prediction modes, 7.4.9.5. */
enum { HEVCD_MODE_INTER = 0, HEVCD_MODE_INTRA = 1 };

/* Partition modes, Table 7-10.
 *
 * The four asymmetric ones exist because a moving object's edge rarely
 * falls on a power of two: splitting a block one quarter of the way
 * across costs one more bin and can save a whole partition's worth of
 * residual. */
enum {
    HEVCD_PART_2Nx2N = 0, HEVCD_PART_2NxN = 1, HEVCD_PART_Nx2N = 2,
    HEVCD_PART_NxN = 3,
    HEVCD_PART_2NxnU = 4, HEVCD_PART_2NxnD = 5,
    HEVCD_PART_nLx2N = 6, HEVCD_PART_nRx2N = 7,
};

/* The three intra prediction modes with names; the rest are angles. */
enum { HEVCD_INTRA_PLANAR = 0, HEVCD_INTRA_DC = 1,
       HEVCD_INTRA_ANGULAR_26 = 26, HEVCD_INTRA_ANGULAR_10 = 10 };

/* Coefficient scan orders, 6.5.3. */
enum { HEVCD_SCAN_DIAG = 0, HEVCD_SCAN_HORIZ = 1, HEVCD_SCAN_VERT = 2 };

/* Threads kept for the life of a decoder. See worker_pool.h. */
#include "worker_pool.h"
typedef worker_pool_t hevcd_pool_t;

/* One picture's worth of decoding state. */
/* Which reference lists a prediction unit uses. */
enum { HEVCD_PF_L0 = 1, HEVCD_PF_L1 = 2, HEVCD_PF_BI = 3 };

/* One prediction unit's motion, on the grid of the smallest one.
 *
 * ⚠️ Kept per four samples and not per prediction unit. Every neighbour
 * question in 8.5.3.2 is asked at a sample position - "the block covering
 * (x0 - 1, y0 + nPbH - 1)" - and answering it from a list of units would
 * mean searching. Four samples is the finest a unit can be, so a grid of
 * that size answers by indexing. */
typedef struct {
    int16_t mv[2][2];               /* [list][x, y], in quarter samples */
    int8_t ref_idx[2];
    uint8_t pred_flag;              /* zero for an intra block */
    /* ⚠️ Which picture, and not only which index. Two blocks either side
     * of an edge may have been written by different slices, and an index
     * means nothing outside the slice that wrote it: the deblocking
     * filter asks whether they point at the same picture. */
    int32_t ref_poc[2];
} hevcd_mvf_t;

/* Where pictures announce progress, and where readers wait for it: one
 * per decoder, shared by every picture in its buffer. */
typedef struct hevcd_progress {
    pthread_mutex_t m;
    pthread_cond_t cv;
} hevcd_progress_t;

/* A picture that later ones predict from.
 *
 * ⚠️ It carries its own reference lists as picture order counts and not as
 * indices. A temporal candidate asks what picture the collocated block
 * pointed at, and the index it used means nothing outside the slice that
 * wrote it. */
typedef struct {
    uint8_t *plane[3];
    int stride[3];
    size_t n_planes;
    int poc;
    bool is_valid;
    hevcd_mvf_t *mvf;
    size_t n_mvf;
    /* ⚠️ Per SLICE, and kept with the picture because they are read
     * long after it is finished: a later picture resolves a collocated
     * motion vector's reference through the list of the slice that
     * DECODED that block. One pair per picture handed every block the
     * last slice's. */
    struct hevcd_img_lists {
        int poc_list[2][16];
        int n_list[2];
        /* 8.5.3.2.8 asks whether the collocated block's reference was
         * long-term when THAT picture was decoded. */
        uint8_t is_lt[2][16];
    } *lists;
    size_t n_lists;
    /* Which slice decoded each coding tree block. ⚠️ The picture's own,
     * written as it is decoded - the decoder's map points here - and not
     * a copy taken at the end: a later picture decoding at the same time
     * reads it for its collocated motion before this one has ended. */
    int32_t *slice_of_ctb;
    size_t n_slice_map;
    /* How many coding tree block rows, from the top, are final: decoded,
     * deblocked and offset, never to change again. A picture decoding at
     * the same time as this one reads it only that far - see
     * hevcd_await_rows(). Published with release ordering, so everything
     * written into those rows is visible to whoever sees the count. */
    _Atomic int rows_ready;
    hevcd_progress_t *progress;
    /* How many pictures in flight hold this slot - the one writing it and
     * every one predicting from it. A held slot is not reused even once
     * no longer a reference. Under the decoder's buffer lock. */
    int users;
} hevcd_img_t;

/* One coding tree block's sample adaptive offset, 7.3.8.3.
 *
 * The filter is a lookup table with four entries that the encoder chose
 * and sent: it corrects whatever the rest of the loop got wrong, in
 * whichever direction, which is why it is the last thing to run and why
 * nothing predicts from anything but its output. */
typedef struct {
    uint8_t kind[3];        /* 0 nothing, 1 by band, 2 by edge */
    int8_t off[3][4];       /* signed already: edge offsets have fixed signs */
    uint8_t position[3];   /* which four bands, for the band type */
    uint8_t category[3];      /* which way the edge runs, for the edge type */
} hevcd_sao_t;

/* What one slice says about the loop filters.
 *
 * ⚠️ Per slice and not per picture. Two slices of one picture may
 * disable deblocking differently, carry different beta and tC offsets,
 * and disagree about whether the filters may cross between them - and
 * the filters run once, over the whole picture, after every slice of it
 * has been read. */
typedef struct {
    int16_t beta_offset, tc_offset;
    uint8_t disabled;
    uint8_t across_slices;
} hevcd_slice_filter_t;

typedef struct {
    const hevc_sps_t *sps;
    const hevc_pps_t *pps;
    const hevc_slice_t *slice;
    /* The decoder's, handed to every picture it opens. */
    hevcd_progress_t *progress;

    hevcd_cabac_t cabac;

    /* ⚠️ Per smallest coding block, not per coding unit: split_cu_flag's
     * context asks how deep the neighbour was split, and the neighbour may
     * be any size. */
    uint8_t *ct_depth;              /* [min_cb_height][min_cb_width] */
    /* Per smallest prediction block, which at 4x4 is also the smallest
     * transform block: the intra mode, for the most-probable-mode
     * derivation and for which way the coefficients are scanned. */
    uint8_t *intra_mode;            /* [h >> 2][w >> 2] */
    uint8_t *skip;                  /* per min coding block */
    /* ⚠️ Where each smallest transform block sits in the z-scan order of
     * the whole picture. Whether a neighbour has been decoded yet is a
     * question about that order and not about coordinates: a coding tree
     * unit is a quadtree, so the block above right of a transform block
     * may or may not have come first. */
    int32_t *min_tb_addr_zs;
    /* What that table was built for - it depends on nothing else - so a
     * picture like the one before does not build it again. */
    int zs_w, zs_h, zs_log2_min_tb, zs_log2_ctb;
    int32_t *zs_rs_to_ts;
    size_t n_zs_rs;
    /* 6.5.1. The picture in tile scan and back again, and which tile each
     * unit belongs to - that one indexed by TILE-SCAN address, because
     * the walk asks "has the tile changed since the last unit" and the
     * walk is in tile scan.
     *
     * Built for every picture, tiles or not: without them there is one
     * tile, the two maps are the identity, and nothing downstream needs
     * to ask which case it is in. */
    int32_t *rs_to_ts;
    int32_t *ts_to_rs;
    int32_t *tile_of_ts;
    size_t n_tile_map;
    int n_tiles;
    /* Which SLICE each coding tree block was decoded as part of, by
     * raster address, or -1 for one nothing has reached yet. 6.4.1 wants
     * the same slice as well as the same tile, and a hole left at the
     * end of the picture is a slice that never arrived.
     *
     * ⚠️ Slices, not slice segments: a dependent segment continues the
     * slice before it and carries its number. */
    int32_t *slice_of_ctb;
    size_t n_slice_map;
    int slice_now;
    /* One entry per slice of this picture, indexed by the number in
     * slice_of_ctb. */
    hevcd_slice_filter_t *slice_filter;
    size_t n_slice_filter;
    /* 9.3.1: the context state as the previous slice segment left it. A
     * dependent segment starts from here instead of from the table. */
    uint8_t ctx_at_segment_end[HEVCD_CTX];
    bool have_segment_end;
    /* 9.3.2.3: the state two units into a row, which the row below
     * starts from. ⚠️ Kept here and not in the walk, because with one
     * slice segment per row the walk that takes it is not the walk that
     * needs it. */
    uint8_t wpp_snapshot[HEVCD_CTX];
    bool have_wpp_snapshot;
    /* Which tile the coding tree unit being read belongs to. Set on the
     * way into hevcd_read_ctu(), because the availability tests are
     * asked about a neighbour and know nothing about where "here" is. */
    int tile_now;
    int8_t *qp_y_map;               /* per min coding block */
    /* Which 8x8 cells of the picture have a block boundary on their left
     * edge (bit 0) and on their top edge (bit 1). The deblocking filter
     * only ever looks at that grid, so a transform block boundary at four
     * samples is not one of these: it is a boundary the filter is not
     * allowed to cross. */
    uint8_t *edges;
    int edges_stride;
    size_t n_edges;
    /* Per min coding block: a unit whose samples the loop filters must
     * leave exactly as they are. Lossless coding, today. */
    uint8_t *no_filter;
    /* Per smallest transform block: does it carry a luma residual. The
     * boundary strength asks, and only about luma. */
    uint8_t *cbf_map;
    size_t n_cbf;
    /* Per coding tree block, and the picture as the deblocking filter left
     * it: the offset by an edge asks what the neighbours were before this
     * filter touched them, so it cannot read the plane it is writing. */
    hevcd_sao_t *sao;
    size_t n_sao;
    /* ⚠️ Not the whole deblocked picture, only what an edge offset reads
     * from outside its own block: per plane, the first and the last row of
     * every coding tree block row and the first and the last column of
     * every coding tree block column, as the deblocking filter left them.
     * See sao_copy_row(). */
    uint8_t *sao_lines[3];
    size_t n_sao_lines[3];
    size_t n_ct_depth, n_intra_mode, n_zs, n_qp, n_no_filter, n_skip;
    int min_pu_width, min_pu_height;

    /* The coding unit being read. */
    struct {
        int x, y, log2_size;
        int pred_mode;
        int part_mode;
        int depth;                  /* the coding quadtree depth it sits at */
        bool skip;
        bool transquant_bypass;
        bool intra_split;
        int intra_mode_c;           /* the chroma mode, IntraPredModeC */
    } cu;

    /* Quantisation, 8.6.1. QpY carries across coding units. */
    int qp_y, qp_y_pred;
    /* The last coding unit's QpY, which is qPY_PREV for the next
     * quantisation group, and where that group starts. */
    int qp_y_prev;
    int qg_x, qg_y;
    /* The group takes the slice's parameter rather than the previous
     * group's: first of a slice, of a tile, or of a row under WPP. */
    bool qg_restarts;
    bool cu_qp_delta_coded;
    int cu_qp_delta;

    /* One transform block's coefficients, in raster order inside it. */
    int16_t coeff[32 * 32];
    /* Where the residual reader put a value in coeff, in the order it put
     * them, and the smallest rectangle from the corner that holds them
     * all. The dequantiser touches only those and the transform only the
     * rectangle: each of them used to look for the coefficients again,
     * one branch per position, and those branches are as unpredictable as
     * the coefficients. */
    uint16_t nz_pos[32 * 32];
    int n_nz, nz_max_x, nz_max_y;
    bool transform_skip;            /* of the block just read */

    /* Where the picture is written. The decoder owns these; the harness
     * reads them back. */
    uint8_t *plane[3];
    int stride[3];
    size_t n_planes;

    /* The picture being decoded, its motion field, and what it predicts
     * from. The harness owns the buffer and fills these in per slice. */
    /* The decoded picture buffer, which the caller owns: the decoder
     * writes into it and reads references out of it, and has no opinion
     * about how long anything stays. */
    hevcd_img_t *buf;
    int n_buf;
    hevcd_img_t *current;
    hevcd_mvf_t *mvf;
    /* 7.4.5: the quantisation matrices of the current picture, expanded
     * to one factor per position in each block's raster order, by
     * [matrixId]: intra 0-2 and inter 3-5, one per colour component.
     * scaling_on is scaling_list_enabled_flag. */
    bool scaling_on;
    uint8_t sf4[6][16], sf8[6][64], sf16[6][256], sf32[6][1024];

    const hevcd_img_t *ref_pic[2][16];
    int n_refs[2];
    bool ref_is_lt[2][16];          /* taken from the long-term set */
    const hevcd_img_t *col;         /* the collocated picture, or NULL */

    int ctb_addr;                   /* in the picture's raster order */
    /* The wavefront has already run the loop filters over this picture,
     * and hevc_decoder_end_picture() must not run them again. */
    bool filters_done;
    /* The decoder's threads, made the first time a picture wants more
     * than one and kept until hevc_decoder_destroy(). NULL until then. */
    hevcd_pool_t *pool;
    bool slice_end;
} hevcd_t;

/* Reading one coding tree unit and everything inside it. Returns 0, or
 * non-zero when the slice cannot go on. */
int hevcd_read_ctu(hevcd_t *d, int x0, int y0);

/* Clears one coding tree block's entries in the motion field. Done as each
 * block is decoded, not for the whole picture up front. */
void hevcd_clear_ctb_motion(hevcd_t *d, int rx, int ry);

/* Several coding tree block rows at once, when the stream was written to
 * allow it. Returns the same reasons as the serial walk, or -1 when this
 * slice is not one it can split up. */
int hevcd_wavefront(hevcd_t *d, const hevc_sps_t *sps, const hevc_pps_t *pps,
                    const hevc_slice_t *sl, const uint8_t *base, size_t rest,
                    int init_type);

/* 8.7.2 and 8.7.3, over the whole finished picture, in that order, on as
 * many threads as BC250_HEVC_THREAD or the processor count allows. */
void hevcd_loop_filters(hevcd_t *d);

/* The same, one coding tree block row and one stage at a time: 0 the
 * vertical edges, 1 the horizontal ones, 2 keeping the borders SAO reads,
 * 3 SAO. hevcd_filters_prepare() says which of deblocking and SAO the
 * picture has, and false when neither. For hevc_wpp.c, which runs them
 * while the wavefront is still decoding. */
bool hevcd_filters_prepare(hevcd_t *d, bool *deblock, bool *sao);
void hevcd_filter_stage(hevcd_t *d, int stage, int ry);

/* A pool of worker threads. hevcd_pool_run() calls fn(arg) on the caller
 * and on up to n - 1 of the pool's threads, and returns once every one of
 * them has returned; hevcd_pool_helpers() says how many it would use for
 * n, so a job that must know its team size can know it beforehand.
 * hevcd_pool_for() makes the decoder's pool if there is none yet and
 * returns it, or NULL if threads cannot be had - callers then do the work
 * on the threads of old. */
hevcd_pool_t *hevcd_pool_for(hevcd_t *d);

/* A picture's final rows, announced and waited for. rows_ready only grows;
 * hevcd_await_rows() returns once at least `rows` rows are final. Both
 * accept NULL, for a picture nobody decodes. */
void hevcd_rows_ready(hevcd_img_t *g, int rows);
void hevcd_await_rows(const hevcd_img_t *g, int rows);
static inline int hevcd_pool_helpers(const hevcd_pool_t *p, int n)
{
    return worker_pool_helpers(p, n);
}
static inline void hevcd_pool_run(hevcd_pool_t *p, void *(*fn)(void *), void *arg, int n)
{
    worker_pool_run(p, fn, arg, n);
}
static inline void hevcd_pool_destroy(hevcd_pool_t *p)
{
    worker_pool_destroy(p);
}
void hevcd_free_filters(hevcd_t *d);

/* 8.5.3.2: what motion one prediction unit ended up with, and 8.5.3.3:
 * the samples that motion fetches. */
void hevcd_merge(hevcd_t *d, int x0, int y0, int w, int h, int part_idx,
                 int merge_idx, hevcd_mvf_t *out);
void hevcd_amvp(hevcd_t *d, int x0, int y0, int w, int h, int list_idx,
                int mvp_flag, hevcd_mvf_t *mv);
void hevcd_predict_inter(hevcd_t *d, int x0, int y0, int w, int h,
                         const hevcd_mvf_t *m);

/* How many prediction units a partition mode has, and where the k-th one
 * sits inside a coding block of side `side`. */
int hevcd_pu_count(int part_mode);
void hevcd_pu_rect(int part_mode, int k, int side,
                         int *x, int *y, int *w, int *h);

/* residual_coding(), clause 7.3.8.11. The coefficients land in d->coeff,
 * in raster order inside the transform block. */
void hevcd_read_residual(hevcd_t *d, int x0, int y0, int log2_size, int c_idx);

/* Intra prediction, 8.4.4.2: writes the prediction straight into the
 * picture, where the residual is then added to it. */
void hevcd_predict_intra(hevcd_t *d, int c_idx, int x0, int y0, int log2_size,
                         int mode);

/* 8.6.2 to 8.6.4: the coefficients into a residual, and onto the picture.
 * Told where the coefficients are: dequantise only the n positions in pos
 * (m NULL for the flat matrix), and transform knowing that nothing lies
 * right of max_x or below max_y. hevcd_transform() finds out for itself. */
void hevcd_transform(int16_t *coeff, int log2_size, bool dst, int bd);
void hevcd_dequantize_at(int16_t *coeff, const uint16_t *pos, int n,
                         int log2_size, int qp, int bd, const uint8_t *m);
void hevcd_transform_box(int16_t *coeff, int log2_size, bool dst, int bd,
                         int max_x, int max_y);
void hevcd_skip_transform(int16_t *coeff, int log2_size, int bd);
void hevcd_add(uint8_t *plane, int stride, int x, int y,
               const int16_t *res, int log2_size,
               int bd);

/* 6.5.2: the z-scan address of every smallest transform block. Rebuilt
 * only when the picture size, the block sizes or the tiles change. */
int hevcd_prepare_zscan(hevcd_t *d);
int hevcd_prepare_tiles(hevcd_t *d);
void hevcd_free_tiles(hevcd_t *d);

/* Which tile covers the unit at these LUMA coordinates. Used by the
 * availability rule and by the loop filters, both of which think in
 * samples rather than in unit addresses.
 *
 * ⚠️ Here and inline, with hevcd_slice_at(): the deblocking filter asks
 * both for every four-sample edge segment, and as calls into another file
 * the asking cost more than the lookups. */
static inline int hevcd_tile_at(const hevcd_t *d, int x, int y)
{
    /* The common case by far, and worth one branch: with a single tile
     * every answer is zero and the two map lookups are waste. */
    if (d->n_tiles <= 1 || !d->tile_of_ts) return 0;
    const hevc_sps_t *sps = d->sps;
    const int rs = (y >> sps->log2_ctb) * sps->ctb_width + (x >> sps->log2_ctb);
    if (rs < 0 || rs >= sps->ctb_count) return 0;
    return d->tile_of_ts[d->rs_to_ts[rs]];
}

/* Which slice covers the unit at these LUMA coordinates, or -1 when
 * none has yet. Same shape as hevcd_tile_at() and asked in the same
 * places. */
static inline int hevcd_slice_at(const hevcd_t *d, int x, int y)
{
    if (!d->slice_of_ctb) return 0;
    const hevc_sps_t *sps = d->sps;
    const int rs = (y >> sps->log2_ctb) * sps->ctb_width + (x >> sps->log2_ctb);
    if (rs < 0 || rs >= sps->ctb_count) return -1;
    return d->slice_of_ctb[rs];
}

#endif /* BC250_HEVC_DEC_INTERNAL_H */
