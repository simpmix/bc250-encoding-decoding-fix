/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * hevc_filter_template.c - deblocking (8.7.2) and SAO (8.7.3).
 *
 * Included once per bit depth by hevc_filter.c. See hevc_pixel.h: every
 * stride in here is counted in samples.
 */

static inline int FUNC(clip)(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline int FUNC(clip_pixel)(int v)
{
    return v < 0 ? 0 : (v > PIXEL_MAX ? PIXEL_MAX : v);
}

/* The luma parameter of the coding unit covering a sample. */
static int FUNC(qp_di)(const hevcd_t *d, int x, int y)
{
    const int stride = d->sps->min_cb_width;
    const int l = d->sps->log2_min_cb;
    return d->qp_y_map[(y >> l) * stride + (x >> l)];
}

/* A coding unit coded losslessly keeps its samples exactly as they are:
 * filtering them would be the one thing that made the stream lossy. */
static bool FUNC(untouchable)(const hevcd_t *d, int x, int y)
{
    if (!d->no_filter) return false;
    const int stride = d->sps->min_cb_width;
    const int l = d->sps->log2_min_cb;
    return d->no_filter[(y >> l) * stride + (x >> l)] != 0;
}

/* One four-line segment of a luma edge.
 *
 * `base` points at q0 of the first line, `forward` steps from p to q and
 * `giu` steps from one line to the next: for a vertical edge those are one
 * sample and one row, for a horizontal edge the other way round. Writing
 * it once in these two steps is what keeps the two directions from
 * drifting apart, which is where a hand-unrolled deblocking filter usually
 * goes wrong.
 */
static void FUNC(filter_luma)(pixel *base, int forward, int giu,
                        int beta, int tc, bool keep_p, bool keep_q)
{
#if BIT_DEPTH == 8 && (defined(__x86_64__) || defined(_M_X64))
    if (sao_vector()) {
        filter_luma_ssse3(base, forward, giu, beta, tc, keep_p, keep_q);
        return;
    }
#endif
#define P(k, i) ((int)base[(i) * giu - ((k) + 1) * forward])
#define Q(k, i) ((int)base[(i) * giu + (k) * forward])
#define WRITE_P(k, i, v) \
    do { if (!keep_p) base[(i) * giu - ((k) + 1) * forward] = (v); } while (0)
#define WRITE_Q(k, i, v) \
    do { if (!keep_q) base[(i) * giu + (k) * forward] = (v); } while (0)

    /* 8.7.2.5.3. The decision looks at the first and the last line of the
     * four and at nothing in between: four lines of an eight-sample block
     * edge are alike enough that two of them decide for all four, and
     * halving the work of the decision was worth it to the committee. */
    const int dp0 = abs(P(2, 0) - 2 * P(1, 0) + P(0, 0));
    const int dp3 = abs(P(2, 3) - 2 * P(1, 3) + P(0, 3));
    const int dq0 = abs(Q(2, 0) - 2 * Q(1, 0) + Q(0, 0));
    const int dq3 = abs(Q(2, 3) - 2 * Q(1, 3) + Q(0, 3));
    const int dpq0 = dp0 + dq0, dpq3 = dp3 + dq3;

    if (dpq0 + dpq3 >= beta) return;

    const int dp = dp0 + dp3, dq = dq0 + dq3;
    const int threshold = (5 * tc + 1) >> 1;

    /* 8.7.2.5.6: strong only when the step really is a step - flat on both
     * sides, and the jump across small enough to be an artefact rather
     * than an edge that belongs to the picture. */
    bool forte = true;
    for (int i = 0; i < 4 && forte; i += 3) {
        const int dpq = 2 * (i == 0 ? dpq0 : dpq3);
        forte = dpq < (beta >> 2)
             && abs(P(3, i) - P(0, i)) + abs(Q(0, i) - Q(3, i)) < (beta >> 3)
             && abs(P(0, i) - Q(0, i)) < threshold;
    }

    if (forte) {
        /* 8.7.2.5.7. Three samples each side, every one of them held
         * within two tC of where it started: the filter may smooth a step
         * but it may not invent one. */
        for (int i = 0; i < 4; i++) {
            const int p0 = P(0, i), p1 = P(1, i), p2 = P(2, i), p3 = P(3, i);
            const int q0 = Q(0, i), q1 = Q(1, i), q2 = Q(2, i), q3 = Q(3, i);
            WRITE_P(0, i, (pixel)FUNC(clip)((p2 + 2 * p1 + 2 * p0 + 2 * q0
                                              + q1 + 4) >> 3,
                                             p0 - 2 * tc, p0 + 2 * tc));
            WRITE_P(1, i, (pixel)FUNC(clip)((p2 + p1 + p0 + q0 + 2) >> 2,
                                             p1 - 2 * tc, p1 + 2 * tc));
            WRITE_P(2, i, (pixel)FUNC(clip)((2 * p3 + 3 * p2 + p1 + p0 + q0
                                              + 4) >> 3,
                                             p2 - 2 * tc, p2 + 2 * tc));
            WRITE_Q(0, i, (pixel)FUNC(clip)((p1 + 2 * p0 + 2 * q0 + 2 * q1
                                              + q2 + 4) >> 3,
                                             q0 - 2 * tc, q0 + 2 * tc));
            WRITE_Q(1, i, (pixel)FUNC(clip)((p0 + q0 + q1 + q2 + 2) >> 2,
                                             q1 - 2 * tc, q1 + 2 * tc));
            WRITE_Q(2, i, (pixel)FUNC(clip)((p0 + q0 + q1 + 3 * q2 + 2 * q3
                                              + 4) >> 3,
                                             q2 - 2 * tc, q2 + 2 * tc));
        }
        return;
    }

    /* 8.7.2.5.7, the weak filter. One sample each side always, a second
     * one only on whichever side was flat enough to deserve it. */
    const bool touch_p1 = dp < ((beta + (beta >> 1)) >> 3);
    const bool touch_q1 = dq < ((beta + (beta >> 1)) >> 3);

    for (int i = 0; i < 4; i++) {
        const int p0 = P(0, i), p1 = P(1, i), p2 = P(2, i);
        const int q0 = Q(0, i), q1 = Q(1, i), q2 = Q(2, i);
        int delta = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;
        if (abs(delta) >= tc * 10) continue;
        delta = FUNC(clip)(delta, -tc, tc);
        WRITE_P(0, i, (pixel)FUNC(clip_pixel)(p0 + delta));
        WRITE_Q(0, i, (pixel)FUNC(clip_pixel)(q0 - delta));
        if (touch_p1) {
            const int dp1 = FUNC(clip)((((p2 + p0 + 1) >> 1) - p1 + delta) >> 1,
                                     -(tc >> 1), tc >> 1);
            WRITE_P(1, i, (pixel)FUNC(clip_pixel)(p1 + dp1));
        }
        if (touch_q1) {
            const int dq1 = FUNC(clip)((((q2 + q0 + 1) >> 1) - q1 - delta) >> 1,
                                     -(tc >> 1), tc >> 1);
            WRITE_Q(1, i, (pixel)FUNC(clip_pixel)(q1 + dq1));
        }
    }
#undef P
#undef Q
#undef WRITE_P
#undef WRITE_Q
}

/* 8.7.2.5.5. Chroma gets one sample each side and no decision at all: it
 * is filtered where the boundary strength is two and nowhere else, which
 * in an intra picture means every edge. */
static void FUNC(filter_chroma)(pixel *base, int forward, int giu, int tc,
                         bool keep_p, bool keep_q)
{
#if BIT_DEPTH == 8 && (defined(__x86_64__) || defined(_M_X64))
    if (sao_vector()) {
        filter_chroma_ssse3(base, forward, giu, tc, keep_p, keep_q);
        return;
    }
#endif
    for (int i = 0; i < 4; i++) {
        pixel *p1 = base + i * giu - 2 * forward;
        pixel *p0 = base + i * giu - forward;
        pixel *q0 = base + i * giu;
        pixel *q1 = base + i * giu + forward;
        const int delta = FUNC(clip)(((((int)*q0 - *p0) * 4) + *p1 - *q1 + 4) >> 3,
                                   -tc, tc);
        if (!keep_p) *p0 = (pixel)FUNC(clip_pixel)(*p0 + delta);
        if (!keep_q) *q0 = (pixel)FUNC(clip_pixel)(*q0 - delta);
    }
}

/* Table 8-10 again, as 8.7.2.5.5 asks for it. */
static int FUNC(qp_chroma)(int qp_i)
{
    if (qp_i < 30) return qp_i < 0 ? 0 : qp_i;
    if (qp_i > 43) return qp_i - 6;
    return hevcd_qp_c[qp_i - 30];
}

/* Which slice's loop filter settings govern the sample at these
 * coordinates, or NULL for one no slice ever decoded. */
static const hevcd_slice_filter_t *FUNC(filter_of)(const hevcd_t *d,
                                                   int x, int y)
{
    const int s = hevcd_slice_at(d, x, y);
    if (s < 0 || !d->slice_filter || (size_t)s >= d->n_slice_filter)
        return NULL;
    return &d->slice_filter[s];
}

/* beta and tC for one edge, 8.7.2.5.3. The boundary strength only ever
 * reaches the tables through tC, and only by two quantiser steps.
 *
 * ⚠️ The offsets belong to the slice being filtered, which is the one
 * holding the q side of the edge - not to whichever slice happened to be
 * read last. */
static int FUNC(beta_di)(const hevcd_t *d,
                         const hevcd_slice_filter_t *f, int qp)
{
    const int q = FUNC(clip)(qp + f->beta_offset, 0, 51);
    return hevcd_beta[q] << (d->sps->bit_depth_luma - 8);
}

static int FUNC(tc_di)(const hevcd_t *d, const hevcd_slice_filter_t *f,
                       int qp, int bs)
{
    const int q = FUNC(clip)(qp + 2 * (bs - 1) + f->tc_offset, 0, 53);
    return hevcd_tc[q] << (d->sps->bit_depth_luma - 8);
}

/* 8.7.2.4. Two, one, or nothing at all.
 *
 * Two means an intra block is involved and the step across the edge is
 * whatever the prediction could not reach; that is worth the strong
 * filter and it is the only case where chroma is touched at all. One
 * means two inter blocks that disagree - a coded residual at a transform
 * edge, different reference pictures, or motion a quarter sample apart.
 * Nothing means two blocks that were predicted the same way from the same
 * place, where any step across the edge would be something the filter
 * invented.
 */
static int FUNC(strength)(const hevcd_t *d, int xp, int yp, int xq, int yq,
                 bool transform_edge)
{
    const int stride = d->min_pu_width;
    const hevcd_mvf_t *p = &d->mvf[(yp >> 2) * stride + (xp >> 2)];
    const hevcd_mvf_t *q = &d->mvf[(yq >> 2) * stride + (xq >> 2)];

    if (!p->pred_flag || !q->pred_flag) return 2;

    if (transform_edge && d->cbf_map
        && (d->cbf_map[(yp >> 2) * stride + (xp >> 2)]
            || d->cbf_map[(yq >> 2) * stride + (xq >> 2)]))
        return 1;

    const int np = (p->pred_flag == HEVCD_PF_BI) ? 2 : 1;
    const int nq = (q->pred_flag == HEVCD_PF_BI) ? 2 : 1;
    if (np != nq) return 1;

#define DIFFER(a, la, b, lb)                                   \
    (abs((a)->mv[la][0] - (b)->mv[lb][0]) >= 4                  \
     || abs((a)->mv[la][1] - (b)->mv[lb][1]) >= 4)

    if (np == 1) {
        const int lp = (p->pred_flag & HEVCD_PF_L0) ? 0 : 1;
        const int lq = (q->pred_flag & HEVCD_PF_L0) ? 0 : 1;
        if (p->ref_poc[lp] != q->ref_poc[lq]) return 1;
        return DIFFER(p, lp, q, lq) ? 1 : 0;
    }

    /* Both predict from two places. They agree only if the two pairs of
     * pictures are the same pair - in either order - and the vectors that
     * go with them are close enough. */
    const bool same_ones = (p->ref_poc[0] == q->ref_poc[0]
                         && p->ref_poc[1] == q->ref_poc[1]);
    const bool crossed = (p->ref_poc[0] == q->ref_poc[1]
                             && p->ref_poc[1] == q->ref_poc[0]);
    if (!same_ones && !crossed) return 1;

    if (p->ref_poc[0] != p->ref_poc[1]) {
        /* ⚠️ Two different pictures: there is only one way to pair them
         * up, and it is whichever way makes the pictures match. */
        if (same_ones)
            return (DIFFER(p, 0, q, 0) || DIFFER(p, 1, q, 1)) ? 1 : 0;
        return (DIFFER(p, 0, q, 1) || DIFFER(p, 1, q, 0)) ? 1 : 0;
    }

    /* The same picture twice: either pairing will do, and the edge is
     * quiet if either one is close enough. */
    const bool dritte = !(DIFFER(p, 0, q, 0) || DIFFER(p, 1, q, 1));
    const bool crossed_refs = !(DIFFER(p, 0, q, 1) || DIFFER(p, 1, q, 0));
    return (dritte || crossed_refs) ? 0 : 1;
#undef DIFFER
}

/* One direction over the luma rows [y0, y1). `vertical` says which edges
 * are looked at, not which way the filter reads: a vertical edge is
 * filtered along x and stepped along y.
 *
 * ⚠️ The rows are whole coding tree block rows, so y0 is a multiple of
 * sixteen at least. That is what lets two bands run at once: a horizontal
 * luma edge writes three rows either side of itself and reads four, and
 * the next edge is eight rows on, so no two edges in different bands
 * touch the same row - and the chroma edges, sixteen luma rows apart,
 * keep the same distance on their own grid.
 *
 * Walked one 8x8 cell of the edge map at a time. A cell's edge is two
 * four-sample segments, and almost everything asked about a segment has
 * the same answer for both: the tile, the slice and its filter settings
 * (coding tree blocks are at least 16 and aligned), the quantiser and
 * whether either side is lossless (coding blocks are at least 8). Only the
 * boundary strength is per segment - motion is stored per 4x4. The
 * segments of one pass never touch each other, so the order they are
 * filtered in changes nothing.
 *
 * Always inlined, with `vertical` a constant at both call sites, so each
 * direction gets a loop of its own without the other's arithmetic. */
static inline __attribute__((always_inline))
void FUNC(one_direction)(hevcd_t *d, bool vertical, int y0, int y1)
{
    const hevc_sps_t *sps = d->sps;
    const int forward_l = vertical ? 1 : d->stride[0];
    const int giu_l = vertical ? d->stride[0] : 1;
    const int which = vertical ? 1 : 2;
    const int tu_bit = vertical ? 4 : 8;
    const bool tiles_matter = !d->pps->loop_filter_across_tiles
                              && d->n_tiles > 1;
    pixel *const luma = (pixel *)d->plane[0];
    if (y1 > sps->height) y1 = sps->height;

    for (int cy = y0 >> 3; (cy << 3) < y1; cy++) {
        const uint8_t *erow = d->edges + (size_t)cy * d->edges_stride;
        const int y = cy << 3;
        /* The picture's own top border is never an edge. */
        if (!vertical && y == 0) continue;

        for (int cx = vertical ? 1 : 0; cx < d->edges_stride; cx++) {
            /* Nor is a position the coding tree never put a block boundary
             * at - which is most of them. */
            const int e = erow[cx];
            if (!(e & which)) continue;
            const int x = cx << 3;
            const int xp = vertical ? x - 1 : x;
            const int yp = vertical ? y : y - 1;

            /* ⚠️ 8.7.2: a tile boundary is not filtered across unless the
             * picture parameter set allows it. The two sides were decoded
             * independently and neither knows what the other chose, so
             * smoothing between them invents detail rather than removing
             * it. */
            if (tiles_matter && hevcd_tile_at(d, xp, yp) != hevcd_tile_at(d, x, y))
                continue;

            /* ⚠️ The same for a slice boundary, and the same reason: the
             * two sides were decoded without knowledge of each other. The
             * flag that governs it belongs to the slice holding q. */
            const hevcd_slice_filter_t *fq = FUNC(filter_of)(d, x, y);
            if (!fq || fq->disabled) continue;
            if (!fq->across_slices
                && hevcd_slice_at(d, xp, yp) != hevcd_slice_at(d, x, y))
                continue;

            const bool keep_p = FUNC(untouchable)(d, xp, yp);
            const bool keep_q = FUNC(untouchable)(d, x, y);
            if (keep_p && keep_q) continue;
            const int qp = (FUNC(qp_di)(d, x, y) + FUNC(qp_di)(d, xp, yp) + 1) >> 1;
            const int beta = FUNC(beta_di)(d, fq, qp);
            const bool transform_edge = (e & tu_bit) != 0;

            for (int s = 0; s < 2; s++) {
                const int sx = vertical ? x : x + 4 * s;
                const int sy = vertical ? y + 4 * s : y;
                if (vertical ? sy >= y1 : sx >= sps->width) break;
                const int sxp = vertical ? sx - 1 : sx;
                const int syp = vertical ? sy : sy - 1;

                const int bs = d->mvf
                    ? FUNC(strength)(d, sxp, syp, sx, sy, transform_edge)
                    : 2;
                if (!bs) continue;
                const int tc_l = FUNC(tc_di)(d, fq, qp, bs);

                FUNC(filter_luma)(luma + (size_t)sy * d->stride[0] + sx,
                                  forward_l, giu_l, beta, tc_l,
                                  keep_p, keep_q);

                /* ⚠️ Chroma is filtered on its own grid, which is eight
                 * chroma samples and therefore sixteen luma ones. Filtering
                 * it wherever luma is filtered doubles the edges it touches
                 * and softens the picture in a way no reference decoder
                 * does. */
                if (bs != 2) continue;
                if (vertical ? (sx & 15) : (sy & 15)) continue;
                if (vertical ? (sy & 7) : (sx & 7)) continue;

                for (int c = 1; c < 3; c++) {
                    const int off = c == 1 ? d->pps->cb_qp_offset
                                           : d->pps->cr_qp_offset;
                    const int tc = hevcd_tc[FUNC(clip)(FUNC(qp_chroma)(FUNC(clip)(qp + off,
                                                                       0, 57))
                                                     + 2 + fq->tc_offset,
                                                     0, 53)]
                                   << (d->sps->bit_depth_chroma - 8);
                    if (!tc) continue;
                    const int forward_c = vertical ? 1 : d->stride[c];
                    const int giu_c = vertical ? d->stride[c] : 1;
                    FUNC(filter_chroma)((pixel *)d->plane[c]
                                 + (size_t)(sy / 2) * d->stride[c]
                                 + sx / 2, forward_c, giu_c, tc,
                                 keep_p, keep_q);
                }
            }
        }
    }
}

/* 8.7.2 over one coding tree block row, one direction.
 *
 * The offsets and the on/off switch come from the slice holding each
 * edge, looked up through slice_of_ctb - so a picture whose slices
 * disagree about deblocking gets what each of them asked for.
 */
static void FUNC(deblock_row)(hevcd_t *d, bool vertical, int ry)
{
    const int l = d->sps->log2_ctb;
    /* Two calls with constants, so that each is its own specialised loop. */
    if (vertical) FUNC(one_direction)(d, true, ry << l, (ry + 1) << l);
    else          FUNC(one_direction)(d, false, ry << l, (ry + 1) << l);
}

/* ------------------------------------------------- sample adaptive offset */

/* 8.7.3. The last thing in the decoding loop, and the only part of it the
 * encoder steers directly: four offsets per coding tree block per plane,
 * chosen by measuring the error that everything upstream left behind.
 *
 * ⚠️ It reads the picture the deblocking filter produced and writes a
 * different one. An implementation that reads and writes the same plane
 * gets the band offset right and the edge offset wrong, in a way that is
 * worth a handful of sample values and shows up only where the offsets
 * are large - which is exactly where nobody looks first.
 */
static int FUNC(sign)(int v)
{
    return v > 0 ? 1 : (v < 0 ? -1 : 0);
}

/* Which two neighbours each edge class compares against: horizontal,
 * vertical, and the two diagonals. */
static const int8_t FUNC(sao_dx)[4][2] = { { -1, 1 }, { 0, 0 }, { -1, 1 }, { 1, -1 } };
static const int8_t FUNC(sao_dy)[4][2] = { { 0, 0 }, { -1, 1 }, { -1, 1 }, { -1, 1 } };

/* Does anything in this coding tree block have to be left exactly as it
 * is? Lossless coding units are marked per smallest coding block, so the
 * question is a few dozen bytes per block rather than one per sample. */
static bool FUNC(sao_lossy_only)(const hevcd_t *d, int rx, int ry)
{
    if (!d->no_filter) return true;
    const hevc_sps_t *sps = d->sps;
    const int l = sps->log2_min_cb;
    const int stride = sps->min_cb_width;
    const int x0 = (rx << sps->log2_ctb) >> l, y0 = (ry << sps->log2_ctb) >> l;
    int x1 = ((rx + 1) << sps->log2_ctb) >> l;
    int y1 = ((ry + 1) << sps->log2_ctb) >> l;
    if (x1 > stride) x1 = stride;
    if (y1 > sps->min_cb_height) y1 = sps->min_cb_height;
    for (int y = y0; y < y1; y++)
        for (int x = x0; x < x1; x++)
            if (d->no_filter[y * stride + x]) return false;
    return true;
}

/* May the edge offset of this coding tree block read every neighbour it
 * has? Tiles and slices are laid out in whole coding tree blocks, so the
 * per-sample questions of sao_block() all have the same answer inside one
 * block: yes, whenever the eight blocks around it are in the same tile and
 * slice, or the parameter set and the slice let the filter cross.
 *
 * ⚠️ All eight, whichever direction the class compares along. Asking only
 * about the two it uses would be exact too, but this answers "no" on a
 * handful of blocks at the slice and tile boundaries, and those still go
 * the slow way, which is the reference. */
static bool FUNC(sao_neighbours_free)(const hevcd_t *d, int rx, int ry)
{
    const hevc_sps_t *sps = d->sps;
    const int l = sps->log2_ctb;
    const int x = rx << l, y = ry << l;
    const bool tiles = !d->pps->loop_filter_across_tiles;
    const hevcd_slice_filter_t *f = FUNC(filter_of)(d, x, y);
    const bool slices = f && !f->across_slices;
    if (!tiles && !slices) return true;

    const int tile = hevcd_tile_at(d, x, y);
    const int slice = hevcd_slice_at(d, x, y);
    for (int j = -1; j <= 1; j++)
        for (int i = -1; i <= 1; i++) {
            const int nx = rx + i, ny = ry + j;
            if ((!i && !j) || nx < 0 || ny < 0
                || nx >= sps->ctb_width || ny >= sps->ctb_height)
                continue;
            if (tiles && hevcd_tile_at(d, nx << l, ny << l) != tile)
                return false;
            if (slices && hevcd_slice_at(d, nx << l, ny << l) != slice)
                return false;
        }
    return true;
}

/* Where the kept borders of plane c lie in sao_lines[c]. `which` is 0 for
 * the first row of coding tree block row `index`, 1 for its last row, 2
 * for the first column of block column `index`, 3 for its last column.
 * Rows are indexed by x, columns by y, both in the plane's own samples. */
static pixel *FUNC(sao_line)(hevcd_t *d, int c, int which, int index)
{
    const int giu = c ? 1 : 0;
    const size_t w = (size_t)(d->sps->width >> giu);
    const size_t h = (size_t)(d->sps->height >> giu);
    const size_t rows = (size_t)d->sps->ctb_height;
    const size_t cols = (size_t)d->sps->ctb_width;
    pixel *base = (pixel *)d->sao_lines[c];
    switch (which) {
    case 0:  return base + (size_t)index * w;
    case 1:  return base + rows * w + (size_t)index * w;
    case 2:  return base + 2 * rows * w + (size_t)index * h;
    default: return base + 2 * rows * w + cols * h + (size_t)index * h;
    }
}

/* `plain` says sao_lossy_only() held, `open` that sao_neighbours_free()
 * did too. With them the loops below ask nothing per sample but the
 * picture border, and that is worked out once as the loop bounds. */
static void FUNC(sao_block)(hevcd_t *d, int c, int rx, int ry,
                       const hevcd_sao_t *s, bool plain, bool open)
{
    if (!s->kind[c]) return;

    const int giu = c ? 1 : 0;
    const int log2 = d->sps->log2_ctb - giu;
    const int w = d->sps->width >> giu, h = d->sps->height >> giu;
    const int x0 = rx << log2, y0 = ry << log2;
    const int side = 1 << log2;
    const int x1 = x0 + side < w ? x0 + side : w;
    const int y1 = y0 + side < h ? y0 + side : h;
    const int bd = c ? d->sps->bit_depth_chroma : d->sps->bit_depth_luma;
    const int stride = d->stride[c];
    pixel *plane = (pixel *)d->plane[c];

    if (s->kind[c] == 1) {
        /* By band: the range of a sample is cut into thirty-two bands and
         * four consecutive ones get an offset each. An encoder reaches for
         * this where the error is a shift of level rather than a step -
         * a flat area that came out slightly too dark, say. */
        if (plain) {
            /* One offset per band, zero for the twenty-eight that have
             * none: the same answer as the test below, without a branch. */
            int band[32] = { 0 };
            for (int k = 0; k < 4; k++)
                band[(s->position[c] + k) & 31] = s->off[c][k];
            for (int y = y0; y < y1; y++) {
                pixel *p = plane + (size_t)y * stride;
                for (int x = x0; x < x1; x++)
                    p[x] = (pixel)FUNC(clip_pixel)(p[x] + band[p[x] >> (bd - 5)]);
            }
            return;
        }
        for (int y = y0; y < y1; y++)
            for (int x = x0; x < x1; x++) {
                if (FUNC(untouchable)(d, x << giu, y << giu)) continue;
                pixel *p = plane + (size_t)y * stride + x;
                /* Thirty-two bands over the whole range, so the index
                 * is the sample shifted down by BitDepth - 5. */
                const int k = ((*p >> (bd - 5)) - s->position[c]) & 31;
                if (k < 4) *p = (pixel)FUNC(clip_pixel)(*p + s->off[c][k]);
            }
        return;
    }

    /* By edge: each sample is compared with two neighbours along one of
     * four directions, which sorts it into a valley, a step, or a peak,
     * and each of those gets its own offset. */
    const int cl = s->category[c];
    const int ax = FUNC(sao_dx)[cl][0], ay = FUNC(sao_dy)[cl][0];
    const int bx = FUNC(sao_dx)[cl][1], by = FUNC(sao_dy)[cl][1];

    /* The block as the deblocking filter left it, with a ring of one
     * sample around it: the inside from the plane, which nothing has
     * written since, the ring from the borders sao_copy_row() kept -
     * the neighbouring blocks may have had their own offsets applied
     * already. Only the ring positions inside the picture are filled,
     * and only those are ever read. B(x, y) is the sample at picture
     * coordinates (x, y), for x0 - 1 <= x <= x1 and y0 - 1 <= y <= y1. */
    pixel ring[(64 + 2) * (64 + 2)];
    const int rs = side + 2;
    const pixel *const bb = ring + rs + 1;
#define B(x, y) bb[(ptrdiff_t)((y) - y0) * rs + ((x) - x0)]
    for (int y = y0; y < y1; y++)
        memcpy(ring + (size_t)(y - y0 + 1) * rs + 1, plane + (size_t)y * stride + x0,
               (size_t)(x1 - x0) * sizeof(pixel));
    {
        const int xl = x0 > 0 ? x0 - 1 : 0;
        const int xr = x1 < w ? x1 : w - 1;
        if (y0 > 0)
            memcpy(ring + (xl - x0 + 1), FUNC(sao_line)(d, c, 1, ry - 1) + xl,
                   (size_t)(xr - xl + 1) * sizeof(pixel));
        if (y1 < h)
            memcpy(ring + (size_t)(y1 - y0 + 1) * rs + (xl - x0 + 1),
                   FUNC(sao_line)(d, c, 0, ry + 1) + xl,
                   (size_t)(xr - xl + 1) * sizeof(pixel));
        if (x0 > 0) {
            const pixel *col = FUNC(sao_line)(d, c, 3, rx - 1);
            for (int y = y0; y < y1; y++) ring[(size_t)(y - y0 + 1) * rs] = col[y];
        }
        if (x1 < w) {
            const pixel *col = FUNC(sao_line)(d, c, 2, rx + 1);
            for (int y = y0; y < y1; y++)
                ring[(size_t)(y - y0 + 1) * rs + (x1 - x0 + 1)] = col[y];
        }
    }

    if (plain && open) {
        /* A sample whose neighbour would be outside the picture is left
         * alone, and the neighbours are one sample away: that is the first
         * or last column or row of the picture, nothing else. */
        int xs = x0, xe = x1, ys = y0, ye = y1;
        if ((ax < 0 || bx < 0) && xs == 0) xs = 1;
        if ((ax > 0 || bx > 0) && xe == w) xe = w - 1;
        if ((ay < 0 || by < 0) && ys == 0) ys = 1;
        if ((ay > 0 || by > 0) && ye == h) ye = h - 1;
        if (xs >= xe) return;

        /* Indexed by 2 + the two signs, before Table 8-x renumbers them:
         * a valley, a half valley, flat, a half peak, a peak. Flat gets
         * nothing, which rewrites the sample with the value it had. */
        const int8_t table[16] = { s->off[c][0], s->off[c][1], 0,
                                   s->off[c][2], s->off[c][3] };
        for (int y = ys; y < ye; y++) {
            /* All three from column xs on. */
            const pixel *cur = &B(xs, y);
            const pixel *pa = &B(xs + ax, y + ay);
            const pixel *pb = &B(xs + bx, y + by);
            pixel *out = plane + (size_t)y * stride + xs;
#if defined(__x86_64__) || defined(_M_X64)
#if BIT_DEPTH == 8
            if (sao_vector()) {
                sao_edge_row_ssse3(out, cur, pa, pb, xe - xs, table);
                continue;
            }
#elif BIT_DEPTH == 10
            if (sao_vector()) {
                sao_edge_row_10(out, cur, pa, pb, xe - xs, table);
                continue;
            }
#endif
#endif
            for (int i = 0; i < xe - xs; i++) {
                const int v = cur[i];
                const int idx = 2 + FUNC(sign)(v - pa[i]) + FUNC(sign)(v - pb[i]);
                out[i] = (pixel)FUNC(clip_pixel)(v + table[idx]);
            }
        }
        return;
    }

    for (int y = y0; y < y1; y++)
        for (int x = x0; x < x1; x++) {
            /* A sample whose neighbour would be outside the picture is
             * left alone: there is nothing to compare it with. */
            if (x + ax < 0 || x + ax >= w || x + bx < 0 || x + bx >= w) continue;
            if (y + ay < 0 || y + ay >= h || y + by < 0 || y + by >= h) continue;
            if (FUNC(untouchable)(d, x << giu, y << giu)) continue;
            /* ⚠️ And a neighbour in another tile is outside as far as this
             * sample is concerned, unless the parameter set says the
             * filters may cross. Same rule as the deblocking above, and
             * the same reason. Chroma coordinates are shifted back up to
             * luma to ask, because tiles are laid out in luma units. */
            if (!d->pps->loop_filter_across_tiles) {
                const int here = hevcd_tile_at(d, x << giu, y << giu);
                if (hevcd_tile_at(d, (x + ax) << giu, (y + ay) << giu) != here
                    || hevcd_tile_at(d, (x + bx) << giu, (y + by) << giu) != here)
                    continue;
            }
            /* ⚠️ And the slice boundary, when the slice this sample
             * belongs to says the filters may not cross it. */
            {
                const hevcd_slice_filter_t *f =
                    FUNC(filter_of)(d, x << giu, y << giu);
                if (f && !f->across_slices) {
                    const int here = hevcd_slice_at(d, x << giu, y << giu);
                    if (hevcd_slice_at(d, (x + ax) << giu, (y + ay) << giu) != here
                        || hevcd_slice_at(d, (x + bx) << giu, (y + by) << giu) != here)
                        continue;
                }
            }

            const int v = B(x, y);
            int idx = 2 + FUNC(sign)(v - B(x + ax, y + ay))
                        + FUNC(sign)(v - B(x + bx, y + by));
            /* Table 8-x: the five cases are renumbered so that "neither up
             * nor down" lands on zero, which is the one with no offset. */
            if (idx <= 2) idx = (idx == 2) ? 0 : idx + 1;
            if (!idx) continue;
            plane[(size_t)y * stride + x] = (pixel)
                FUNC(clip_pixel)(v + s->off[c][idx - 1]);
        }
#undef B
}

static bool FUNC(sao_alloc)(hevcd_t *d);

/* Is there any offset to apply in this picture, and somewhere to keep the
 * block borders as the deblocking filter left them. */
static bool FUNC(sao_prepare)(hevcd_t *d)
{
    if (!d->sao || !d->plane[0]) return false;

    bool serve = false;
    for (int i = 0; i < d->sps->ctb_count && !serve; i++)
        serve = d->sao[i].kind[0] || d->sao[i].kind[1] || d->sao[i].kind[2];
    if (!serve) return false;

    return FUNC(sao_alloc)(d);
}

/* Somewhere to keep the block borders, whether or not any offset turns out
 * to be needed - the wavefront has to decide before the picture is read. */
static bool FUNC(sao_alloc)(hevcd_t *d)
{
    for (int c = 0; c < 3; c++) {
        const int giu = c ? 1 : 0;
        const size_t w = (size_t)(d->sps->width >> giu);
        const size_t h = (size_t)(d->sps->height >> giu);
        const size_t need = (2 * (size_t)d->sps->ctb_height * w
                             + 2 * (size_t)d->sps->ctb_width * h) * sizeof(pixel);
        if (!d->sao_lines[c] || d->n_sao_lines[c] < need) {
            free(d->sao_lines[c]);
            d->sao_lines[c] = malloc(need);
            d->n_sao_lines[c] = d->sao_lines[c] ? need : 0;
            if (!d->sao_lines[c]) return false;
        }
    }
    return true;
}

/* The borders of one coding tree block row, in all three planes: its
 * first and last rows whole, and the first and last column of each block
 * in it. Everything an edge offset elsewhere will read of this row. */
static void FUNC(sao_copy_row)(hevcd_t *d, int ry)
{
    const int l = d->sps->log2_ctb;
    for (int c = 0; c < 3; c++) {
        const int giu = c ? 1 : 0;
        const int w = d->sps->width >> giu, h = d->sps->height >> giu;
        const int side = 1 << (l - giu);
        const int y0 = ry * side;
        const int y1 = y0 + side < h ? y0 + side : h;
        if (y0 >= y1) continue;
        const pixel *plane = (const pixel *)d->plane[c];
        const int stride = d->stride[c];
        memcpy(FUNC(sao_line)(d, c, 0, ry), plane + (size_t)y0 * stride,
               (size_t)w * sizeof(pixel));
        memcpy(FUNC(sao_line)(d, c, 1, ry), plane + (size_t)(y1 - 1) * stride,
               (size_t)w * sizeof(pixel));
        for (int rx = 0; rx < d->sps->ctb_width; rx++) {
            const int x0 = rx * side;
            if (x0 >= w) break;
            const int x1 = x0 + side < w ? x0 + side : w;
            pixel *first = FUNC(sao_line)(d, c, 2, rx);
            pixel *last = FUNC(sao_line)(d, c, 3, rx);
            for (int y = y0; y < y1; y++) {
                const pixel *row = plane + (size_t)y * stride;
                first[y] = row[x0];
                last[y] = row[x1 - 1];
            }
        }
    }
}

/* 8.7.3 over one coding tree block row. It reads its own blocks and the
 * kept borders and writes only its own blocks, so rows run in any order
 * once every border is kept. */
static void FUNC(sao_row)(hevcd_t *d, int ry)
{
    for (int rx = 0; rx < d->sps->ctb_width; rx++) {
        const hevcd_sao_t *s = &d->sao[ry * d->sps->ctb_width + rx];
        if (!s->kind[0] && !s->kind[1] && !s->kind[2]) continue;
        /* ⚠️ Asked once per block. Asked per sample, three lookups of
         * which slice a neighbour is in made this filter a quarter of
         * a 4K decode. */
        const bool plain = FUNC(sao_lossy_only)(d, rx, ry);
        const bool open = plain && FUNC(sao_neighbours_free)(d, rx, ry);
        for (int c = 0; c < 3; c++)
            FUNC(sao_block)(d, c, rx, ry, s, plain, open);
    }
}
