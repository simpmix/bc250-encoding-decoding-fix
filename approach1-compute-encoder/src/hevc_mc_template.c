/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * hevc_mc_template.c - the scalar side of motion compensation,
 * Rec. ITU-T H.265 clause 8.5.3.3.
 *
 * Included once per bit depth by hevc_mc.c. The vector paths are not in
 * here: they are eight bit only and are built once, beside this.
 */

#undef HORIZONTAL
#undef VERTICAL


static inline int FUNC(clip_pixel)(int v)
{
    return v < 0 ? 0 : (v > PIXEL_MAX ? PIXEL_MAX : v);
}

/* ----------------------------------------------- and the scalar twins */

/* One pass of the filter across a rectangle, reading whole samples along
 * a row.
 *
 * ⚠️ The tap count is a compile time constant in each instance. With it
 * in a variable the compiler keeps neither the eight coefficients in
 * registers nor any chance of doing several samples at once, and this is
 * three quarters of the decoder's time. */
#define HORIZONTAL(nome, N)                                               \
static void nome(const pixel *src, int sp, int w, int h,                 \
                 const int8_t *f, int16_t *out, int pf, int down)          \
{                                                                          \
    for (int r = 0; r < h; r++) {                                          \
        const pixel *s = src + (size_t)r * sp;                           \
        int16_t *o = out + (size_t)r * pf;                               \
        for (int c = 0; c < w; c++) {                                      \
            int v = 0;                                                     \
            for (int k = 0; k < N; k++) v += f[k] * s[c + k];              \
            o[c] = (int16_t)(v >> down);                                   \
        }                                                                  \
    }                                                                      \
}

/* The same down a column. `TYPE` is whole samples on the way in and the
 * fourteen-bit output of a horizontal pass on the way back, `DOWN` the
 * shift that takes the second pass back to fourteen bits. */
#define VERTICAL(nome, N, TYPE)                                            \
static void nome(const TYPE *src, int sp, int w, int h,                    \
                 const int8_t *f, int16_t *out, int pf, int down)          \
{                                                                          \
    for (int r = 0; r < h; r++) {                                          \
        const TYPE *s = src + (size_t)r * sp;                              \
        int16_t *o = out + (size_t)r * pf;                               \
        for (int c = 0; c < w; c++) {                                      \
            int v = 0;                                                     \
            for (int k = 0; k < N; k++) v += f[k] * s[(size_t)k * sp + c]; \
            o[c] = (int16_t)(v >> down);                                   \
        }                                                                  \
    }                                                                      \
}

HORIZONTAL(FUNC(horiz8), 8)
HORIZONTAL(FUNC(horiz4), 4)
VERTICAL(FUNC(vert8), 8, pixel)
VERTICAL(FUNC(vert4), 4, pixel)
VERTICAL(FUNC(vert8_16), 8, int16_t)
VERTICAL(FUNC(vert4_16), 4, int16_t)

/* ------------------------------------------------ and which one to use */



/* The six that choose. ⚠️ The vector call only exists at eight bits: at
 * ten there is no vector version, because those pack to unsigned bytes
 * and every sample above 255 would come back as 255. */
#if BIT_DEPTH == 8 && (defined(__x86_64__) || defined(_M_X64))
#define TRY_VECTOR(f) if (vec) { f(s, sp, w, h, fl, o, po); return; }
#else
#define TRY_VECTOR(f) (void)vec;
#endif

static void FUNC(horiz8_any)(const pixel *s, int sp, int w, int h,
                             const int8_t *fl, int16_t *o, int po,
                             int down, bool vec)
{
    TRY_VECTOR(horiz8_v)
    FUNC(horiz8)(s, sp, w, h, fl, o, po, down);
}

static void FUNC(horiz4_any)(const pixel *s, int sp, int w, int h,
                             const int8_t *fl, int16_t *o, int po,
                             int down, bool vec)
{
    TRY_VECTOR(horiz4_v)
    FUNC(horiz4)(s, sp, w, h, fl, o, po, down);
}

static void FUNC(vert8_any)(const pixel *s, int sp, int w, int h,
                            const int8_t *fl, int16_t *o, int po,
                            int down, bool vec)
{
    TRY_VECTOR(vert8_v)
    FUNC(vert8)(s, sp, w, h, fl, o, po, down);
}

static void FUNC(vert4_any)(const pixel *s, int sp, int w, int h,
                            const int8_t *fl, int16_t *o, int po,
                            int down, bool vec)
{
    TRY_VECTOR(vert4_v)
    FUNC(vert4)(s, sp, w, h, fl, o, po, down);
}

static void FUNC(vert8_16_any)(const int16_t *s, int sp, int w, int h,
                               const int8_t *fl, int16_t *o, int po,
                               int down, bool vec)
{
    TRY_VECTOR(vert8_16_v)
    FUNC(vert8_16)(s, sp, w, h, fl, o, po, down);
}

static void FUNC(vert4_16_any)(const int16_t *s, int sp, int w, int h,
                               const int8_t *fl, int16_t *o, int po,
                               int down, bool vec)
{
    TRY_VECTOR(vert4_16_v)
    FUNC(vert4_16)(s, sp, w, h, fl, o, po, down);
}
#undef TRY_VECTOR

/* The samples a block reads from a reference picture: where they lie, or a
 * copy of them with the picture's edges repeated outwards.
 *
 * ⚠️ A motion vector may point off the edge of the reference picture, and
 * legitimately: an object entering the frame was not there before. The
 * edge sample is repeated outwards rather than the fetch being refused -
 * but deciding that once per tap, eight times per sample, is what made
 * this the slowest thing in the decoder. Decide it once per block instead:
 * either the whole window is inside the picture and it is read where it
 * lies, or the window is copied out once with its edges repeated and the
 * copy is read. `border` holds bh rows of `border_stride` samples. */
static const pixel *FUNC(window)(const pixel *ref_pic, int stride,
                                 int w_pic, int h_pic,
                                 int bx, int by, int bw, int bh,
                                 pixel *border, int border_stride, int *sp)
{
    if (bx >= 0 && by >= 0 && bx + bw <= w_pic && by + bh <= h_pic) {
        *sp = stride;
        return ref_pic + (size_t)by * stride + bx;
    }
    /* The window hangs over an edge. Clamping every sample by itself is
     * how it was written first and it costs more than the filter that
     * reads the result: the row is the same for a whole span of columns,
     * so each output row is a repeat of one sample, a copy of the middle,
     * and a repeat of the last. */
    *sp = border_stride;
    const int left = bx < 0 ? (-bx > bw ? bw : -bx) : 0;
    const int inside_end = bx + bw > w_pic ? w_pic - bx : bw;
    const int right = inside_end < left ? left : inside_end;
    for (int r = 0; r < bh; r++) {
        int sy = by + r;
        sy = sy < 0 ? 0 : (sy >= h_pic ? h_pic - 1 : sy);
        const pixel *ref_row = ref_pic + (size_t)sy * stride;
        pixel *o = border + (size_t)r * border_stride;
        /* ⚠️ Samples, not bytes: memset would write the low byte of a
         * ten-bit sample over half the span, and the memcpy length has to
         * be multiplied to match. */
        for (int i = 0; i < left; i++) o[i] = ref_row[0];
        if (right > left)
            memcpy(o + left, ref_row + bx + left,
                   (size_t)(right - left) * sizeof(pixel));
        for (int i = right; i < bw; i++) o[i] = ref_row[w_pic - 1];
    }
    return border;
}

/* One rectangle of one plane, at a fractional position, into fourteen-bit
 * intermediate values.
 *
 * `before` is how far back the filter reaches: three samples for the eight
 * taps of luma, one for the four of chroma. */
static void FUNC(interpolate)(const pixel *ref_pic, int stride, int w_pic, int h_pic,
                      int x, int y, int w, int h, int fx, int fy,
                      const int8_t *filter_kind, int count, int before,
                      int16_t *out, int out_stride, int bd)
{
    const int8_t *fh = filter_kind + (size_t)fx * count;
    const int8_t *fv = filter_kind + (size_t)fy * count;

    /* 8.5.3.3.3.2: shift1 after the horizontal pass, shift2 after the
     * vertical one. ⚠️ Only the first moves with the depth; the six is
     * six at every depth the standard defines. */
    const int down = bd - 8;
    const bool vec = use_vectors(bd);

    /* The rectangle of whole samples the passes will read: as far back as
     * the filter reaches, and only in the directions it actually filters. */
    const int px = fx ? before : 0, tx = fx ? count : 1;
    const int py = fy ? before : 0, ty = fy ? count : 1;
    const int bx = x - px, by = y - py;
    const int bw = w + tx - 1, bh = h + ty - 1;

    int sp;
    pixel border[(MAX_SIDE + 7) * (MAX_SIDE + 7)];
    const pixel *src = FUNC(window)(ref_pic, stride, w_pic, h_pic,
                                    bx, by, bw, bh, border, MAX_SIDE + 7, &sp);

    if (!fx && !fy) {
#if BIT_DEPTH == 8 && (defined(__x86_64__) || defined(_M_X64))
        if (vec) { copy14_v(src, sp, w, h, out, out_stride); return; }
#endif
        /* shift3 = 14 - BitDepth: the samples go up to fourteen bits, they
         * do not come down. */
        const int up = 14 - bd;
        for (int r = 0; r < h; r++)
            for (int c = 0; c < w; c++)
                out[r * out_stride + c] =
                    (int16_t)(src[(size_t)r * sp + c] << up);
        return;
    }

    if (!fy) {
        if (count == 8) FUNC(horiz8_any)(src, sp, w, h, fh, out, out_stride, down, vec);
        else            FUNC(horiz4_any)(src, sp, w, h, fh, out, out_stride, down, vec);
        return;
    }

    if (!fx) {
        if (count == 8) FUNC(vert8_any)(src, sp, w, h, fv, out, out_stride, down, vec);
        else            FUNC(vert4_any)(src, sp, w, h, fv, out, out_stride, down, vec);
        return;
    }

    /* Both: horizontally first, over enough extra rows above and below for
     * the vertical pass to have something to stand on. */
    int16_t middle[(MAX_SIDE + 7) * MAX_SIDE];
    const int tall = h + count - 1;
    if (count == 8) {
        FUNC(horiz8_any)(src, sp, w, tall, fh, middle, MAX_SIDE, down, vec);
        FUNC(vert8_16_any)(middle, MAX_SIDE, w, h, fv, out, out_stride, 6, vec);
    } else {
        FUNC(horiz4_any)(src, sp, w, tall, fh, middle, MAX_SIDE, down, vec);
        FUNC(vert4_16_any)(middle, MAX_SIDE, w, h, fv, out, out_stride, 6, vec);
    }
}

/* ------------------------------------- fourteen bits back down to eight */

static void FUNC(one_pred)(pixel *dst, int stride, int w, int h,
                const int16_t *a, int stride_a, int bd)
{
#if BIT_DEPTH == 8 && (defined(__x86_64__) || defined(_M_X64))
    if (use_vectors(bd)) { one_pred_v(dst, stride, w, h, a, stride_a); return; }
#endif
    /* 8.5.3.3.4.2: shift1 = 14 - BitDepth, offset1 = 1 << (shift1 - 1) */
    const int sh = 14 - bd, add = 1 << (13 - bd);
    for (int r = 0; r < h; r++)
        for (int c = 0; c < w; c++)
            dst[r * stride + c] = (pixel)
                FUNC(clip_pixel)((a[r * stride_a + c] + add) >> sh);
}

static void FUNC(two_pred)(pixel *dst, int stride, int w, int h,
                const int16_t *a, const int16_t *b, int stride_p, int bd)
{
#if BIT_DEPTH == 8 && (defined(__x86_64__) || defined(_M_X64))
    if (use_vectors(bd)) {
        two_pred_v(dst, stride, w, h, a, b, stride_p);
        return;
    }
#endif
    const int sh = 15 - bd, add = 1 << (14 - bd);
    for (int r = 0; r < h; r++)
        for (int c = 0; c < w; c++)
            dst[r * stride + c] = (pixel)FUNC(clip_pixel)(
                (a[r * stride_p + c] + b[r * stride_p + c] + add) >> sh);
}

/* Two whole-sample predictions averaged: what two_pred() makes of them
 * after the round trip through fourteen bits, without the round trip. */
static void FUNC(average)(pixel *dst, int stride, const pixel *a, int sa,
                          const pixel *b, int sb, int w, int h, int bd)
{
#if BIT_DEPTH == 8 && (defined(__x86_64__) || defined(_M_X64))
    if (use_vectors(bd)) { average_v(dst, stride, a, sa, b, sb, w, h); return; }
#endif
    (void)bd;
    for (int r = 0; r < h; r++)
        for (int c = 0; c < w; c++)
            dst[(size_t)r * stride + c] = (pixel)
                ((a[(size_t)r * sa + c] + b[(size_t)r * sb + c] + 1) >> 1);
}

/* 8.5.3.3.4.3. ⚠️ Used whenever the slice carries a weight table, even
 * where the weight happens to be neutral: for a neutral weight this is
 * the same arithmetic as the plain path, so there is nothing to gain by
 * deciding per block and something to lose by getting the decision
 * wrong. */
static void FUNC(one_weighted)(pixel *dst, int stride, int w, int h,
                       const int16_t *a, int stride_a,
                       int weight, int off, int den, int bd)
{
    const int log2wd = den + 14 - bd;
    const int o = off * (1 << (bd - 8));
#if BIT_DEPTH == 8 && (defined(__x86_64__) || defined(_M_X64))
    /* log2wd is den + 6 at eight bits and den is never negative, so the
     * other branch is unreachable on any stream the parser accepts. It
     * stays anyway. */
    if (use_vectors(bd) && log2wd >= 1) {
        one_weighted_v(dst, stride, w, h, a, stride_a, weight, off, den);
        return;
    }
#endif
    for (int r = 0; r < h; r++)
        for (int c = 0; c < w; c++) {
            const int v = a[r * stride_a + c];
            dst[r * stride + c] = (pixel)FUNC(clip_pixel)(
                log2wd >= 1 ? (((v * weight + (1 << (log2wd - 1))) >> log2wd) + o)
                            : (v * weight + o));
        }
}

static void FUNC(two_weighted)(pixel *dst, int stride, int w, int h,
                       const int16_t *a, const int16_t *b, int stride_p,
                       int pa, int pb, int oa, int ob, int den, int bd)
{
    const int log2wd = den + 14 - bd;
    const int sa = oa * (1 << (bd - 8)), sb = ob * (1 << (bd - 8));
#if BIT_DEPTH == 8 && (defined(__x86_64__) || defined(_M_X64))
    if (use_vectors(bd)) {
        two_weighted_v(dst, stride, w, h, a, b, stride_p, pa, pb, oa, ob, den);
        return;
    }
#endif
    for (int r = 0; r < h; r++)
        for (int c = 0; c < w; c++)
            dst[r * stride + c] = (pixel)FUNC(clip_pixel)(
                (a[r * stride_p + c] * pa + b[r * stride_p + c] * pb
                 + ((sa + sb + 1) << log2wd)) >> (log2wd + 1));
}

/* Does this slice carry a weight table at all. */
static bool FUNC(has_weights)(const hevcd_t *d)
{
    return (d->pps->weighted_pred && d->slice->type == 1)
        || (d->pps->weighted_bipred && d->slice->type == 0);
}

static void FUNC(predict_inter)(hevcd_t *d, int x0, int y0,
                                int w, int h, const hevcd_mvf_t *m)
{
    const hevc_sps_t *sps = d->sps;
    /* ⚠️ On the stack, not static. Sixteen kilobytes is nothing and one
     * shared buffer would be one prediction handed to whichever row asked
     * last. */
    int16_t p[2][MAX_SIDE * MAX_SIDE];
    const bool usa[2] = { (m->pred_flag & HEVCD_PF_L0) != 0,
                          (m->pred_flag & HEVCD_PF_L1) != 0 };
    const bool weights = FUNC(has_weights)(d);

    /* ⚠️ A reference may be a picture still being decoded at the same
     * time as this one. Wait for the rows this block reads from it: as
     * far down as the vector reaches, plus the filter's reach below - four
     * luma rows, two chroma ones. */
    for (int l = 0; l < 2; l++) {
        if (!usa[l]) continue;
        const int i = m->ref_idx[l];
        if (i < 0 || i >= d->n_refs[l] || !d->ref_pic[l][i]) continue;
        int last = y0 + h - 1 + (m->mv[l][1] >> 2) + 4;
        const int last_c = 2 * ((y0 >> 1) + (h >> 1) - 1 + (m->mv[l][1] >> 3) + 2) + 1;
        if (last_c > last) last = last_c;
        if (last >= sps->height) last = sps->height - 1;
        if (last < 0) last = 0;
        hevcd_await_rows(d->ref_pic[l][i], (last >> sps->log2_ctb) + 1);
    }

    for (int plane = 0; plane < 3; plane++) {
        /* ⚠️ Luma and chroma carry their own depth. Equal in every profile
         * we accept, and reading the wrong one would be invisible until
         * the day they are not. */
        const int bd = plane ? sps->bit_depth_chroma : sps->bit_depth_luma;
        const int giu = plane ? 1 : 0;
        const int pw = w >> giu, ph = h >> giu;
        const int px = x0 >> giu, py = y0 >> giu;
        const int w_pic = sps->width >> giu, h_pic = sps->height >> giu;

        /* ⚠️ Whole-sample motion and no weights: the prediction is the
         * reference samples themselves, or the rounded average of two.
         * 8.5.3.3.4.2 takes them up to fourteen bits and back down again,
         * and that round trip is exact - (a << 6 + 32) >> 6 is a, and two
         * of them give (a + b + 1) >> 1 - so it is skipped. Asked per
         * plane: a luma vector on a whole sample can still land on half a
         * chroma sample. */
        if (!weights) {
            const int frac = plane ? 7 : 3;
            bool whole = true;
            for (int l = 0; l < 2; l++)
                if (usa[l] && ((m->mv[l][0] & frac) || (m->mv[l][1] & frac)))
                    whole = false;
            if (whole) {
                const pixel *src[2] = { NULL, NULL };
                int sp[2] = { 0, 0 };
                pixel border[2][MAX_SIDE * MAX_SIDE];
                for (int l = 0; l < 2; l++) {
                    if (!usa[l]) continue;
                    const int i = m->ref_idx[l];
                    if (i < 0 || i >= d->n_refs[l] || !d->ref_pic[l][i]) return;
                    const hevcd_img_t *r = d->ref_pic[l][i];
                    const int steps = plane ? 3 : 2;
                    src[l] = FUNC(window)((const pixel *)r->plane[plane],
                                          r->stride[plane], w_pic, h_pic,
                                          px + (m->mv[l][0] >> steps),
                                          py + (m->mv[l][1] >> steps),
                                          pw, ph, border[l], MAX_SIDE, &sp[l]);
                }
                pixel *dst = (pixel *)d->plane[plane]
                             + (size_t)py * d->stride[plane] + px;
                if (usa[0] && usa[1])
                    FUNC(average)(dst, d->stride[plane], src[0], sp[0],
                                  src[1], sp[1], pw, ph, bd);
                else {
                    const int l = usa[0] ? 0 : 1;
                    for (int y = 0; y < ph; y++)
                        memcpy(dst + (size_t)y * d->stride[plane],
                               src[l] + (size_t)y * sp[l],
                               (size_t)pw * sizeof(pixel));
                }
                continue;
            }
        }

        for (int l = 0; l < 2; l++) {
            if (!usa[l]) continue;
            const int i = m->ref_idx[l];
            if (i < 0 || i >= d->n_refs[l] || !d->ref_pic[l][i]) return;
            const hevcd_img_t *r = d->ref_pic[l][i];
            const int mvx = m->mv[l][0], mvy = m->mv[l][1];
            /* ⚠️ Luma counts quarters and chroma eighths. At 4:2:0 the
             * chroma plane is half the size, so the same vector lands on a
             * finer grid there, not a coarser one. */
            const int steps = plane ? 3 : 2;
            FUNC(interpolate)((const pixel *)r->plane[plane],
                      r->stride[plane], w_pic, h_pic,
                      px + (mvx >> steps), py + (mvy >> steps), pw, ph,
                      mvx & ((1 << steps) - 1), mvy & ((1 << steps) - 1),
                      plane ? &hevcd_epel[0][0] : &hevcd_qpel[0][0],
                      plane ? 4 : 8, plane ? 1 : 3,
                      p[l], MAX_SIDE, bd);
        }

        pixel *dst = (pixel *)d->plane[plane]
                     + (size_t)py * d->stride[plane] + px;

        if (!weights) {
            if (usa[0] && usa[1])
                FUNC(two_pred)(dst, d->stride[plane], pw, ph, p[0], p[1], MAX_SIDE, bd);
            else
                FUNC(one_pred)(dst, d->stride[plane], pw, ph, p[usa[0] ? 0 : 1],
                         MAX_SIDE, bd);
            continue;
        }

        const int den = plane ? d->slice->chroma_log2_weight_denom
                              : d->slice->luma_log2_weight_denom;
        int weight[2] = { 1 << den, 1 << den }, off[2] = { 0, 0 };
        for (int l = 0; l < 2; l++) {
            if (!usa[l]) continue;
            const int i = m->ref_idx[l];
            weight[l] = plane ? d->slice->chroma_weight[l][i][plane - 1]
                              : d->slice->luma_weight[l][i];
            off[l] = plane ? d->slice->chroma_offset[l][i][plane - 1]
                           : d->slice->luma_offset[l][i];
        }

        if (usa[0] && usa[1])
            FUNC(two_weighted)(dst, d->stride[plane], pw, ph, p[0], p[1], MAX_SIDE,
                       weight[0], weight[1], off[0], off[1], den, bd);
        else {
            const int l = usa[0] ? 0 : 1;
            FUNC(one_weighted)(dst, d->stride[plane], pw, ph, p[l], MAX_SIDE,
                       weight[l], off[l], den, bd);
        }
    }
}

/* The encoder's way in: one block from one reference picture, no weights -
 * the prediction predict_inter() builds for such a block, with the same
 * functions, so that the encoder's reconstruction is the decoder's to the
 * sample. `mvx` and `mvy` count the plane's own fractions: quarters of a
 * luma sample, eighths of a chroma one. `w` and `h` are at most MAX_SIDE. */
void FUNC(hevc_mc_uni)(const pixel *ref, int stride, int w_pic, int h_pic,
                       int x, int y, int w, int h, int mvx, int mvy, int chroma,
                       pixel *dst, int dst_stride)
{
    const int steps = chroma ? 3 : 2;
    const int fx = mvx & ((1 << steps) - 1), fy = mvy & ((1 << steps) - 1);
    const int ix = x + (mvx >> steps), iy = y + (mvy >> steps);
    if (!fx && !fy) {
        int sp;
        pixel border[MAX_SIDE * MAX_SIDE];
        const pixel *src = FUNC(window)(ref, stride, w_pic, h_pic, ix, iy, w, h,
                                        border, MAX_SIDE, &sp);
        for (int r = 0; r < h; r++)
            memcpy(dst + (size_t)r * dst_stride, src + (size_t)r * sp,
                   (size_t)w * sizeof(pixel));
        return;
    }
    int16_t p[MAX_SIDE * MAX_SIDE];
    FUNC(interpolate)(ref, stride, w_pic, h_pic, ix, iy, w, h, fx, fy,
                      chroma ? &hevcd_epel[0][0] : &hevcd_qpel[0][0],
                      chroma ? 4 : 8, chroma ? 1 : 3, p, MAX_SIDE, BIT_DEPTH);
    FUNC(one_pred)(dst, dst_stride, w, h, p, MAX_SIDE, BIT_DEPTH);
}
