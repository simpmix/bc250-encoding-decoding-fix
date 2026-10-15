/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * h264_mc.c - inter prediction, Rec. ITU-T H.264 clause 8.4.2.
 *
 * The standard names the sixteen quarter-sample positions of a luma block
 * with letters, which is worth keeping because the derivation refers to them
 * constantly:
 *
 *        xFrac 0   1   2   3
 *  yFrac 0     G   a   b   c
 *        1     d   e   f   g
 *        2     h   i   j   k
 *        3     n   p   q   r
 *
 * G is the integer sample. b, h and j are the half-sample positions and are
 * the only ones actually filtered: b horizontally, h vertically, j both.
 * Every other position is the average of two of those, or of one of those and
 * an integer sample.
 *
 * ⚠️ j is NOT the six-tap filter applied to already-rounded half samples. It
 * is filtered from the intermediate values before their rounding, and shifted
 * once by ten at the end rather than twice by five. Rounding twice is off by
 * one on a large fraction of samples, which looks like a very slightly soft
 * picture on an I frame and accumulates into a smear over a GOP.
 */
#include "h264_mc.h"

#include <stddef.h>
#include <string.h>

#include "h264_simd.h"

static inline uint8_t clip_uint8(int v)
{
    return (uint8_t)(v & ~255 ? (-v) >> 31 : v);
}

/* The six-tap filter of 8.4.2.2.1, unrounded. */
#define TAP(a, b, c, d, e, f) ((a) - 5 * (b) + 20 * (c) + 20 * (d) - 5 * (e) + (f))

/* -------------------------------------------------------------- fetching */

const uint8_t *h264d_mc_fetch_luma(uint8_t *dst, int *out_stride,
                                   const uint8_t *plane, int stride,
                                   int plane_w, int plane_h,
                                   int x, int y, int w, int h)
{
    /* The filter reads x-2 .. x+w-1+4 and the same vertically. When all of
     * that is inside the plane there is nothing to copy: the caller can
     * read the plane where it stands. */
    if (x >= H264D_MC_PAD_BEFORE && y >= H264D_MC_PAD_BEFORE
        && x + w + H264D_MC_PAD_AFTER <= plane_w
        && y + h + H264D_MC_PAD_AFTER <= plane_h) {
        *out_stride = stride;
        return plane + (size_t)y * stride + x;
    }

    const int pw = w + H264D_MC_PAD_BEFORE + H264D_MC_PAD_AFTER;
    const int ph = h + H264D_MC_PAD_BEFORE + H264D_MC_PAD_AFTER;
    *out_stride = pw;
    for (int j = 0; j < ph; j++) {
        int sy = y - H264D_MC_PAD_BEFORE + j;
        sy = sy < 0 ? 0 : (sy >= plane_h ? plane_h - 1 : sy);
        const uint8_t *row = plane + (size_t)sy * stride;
        uint8_t *out = dst + (size_t)j * pw;
        for (int i = 0; i < pw; i++) {
            int sx = x - H264D_MC_PAD_BEFORE + i;
            sx = sx < 0 ? 0 : (sx >= plane_w ? plane_w - 1 : sx);
            out[i] = row[sx];
        }
    }
    return dst + H264D_MC_PAD_BEFORE * pw + H264D_MC_PAD_BEFORE;
}

const uint8_t *h264d_mc_fetch_chroma(uint8_t *dst, int *out_stride,
                                     const uint8_t *plane, int stride,
                                     int plane_w, int plane_h,
                                     int x, int y, int w, int h)
{
    if (x >= 0 && y >= 0 && x + w < plane_w && y + h < plane_h) {
        *out_stride = stride;
        return plane + (size_t)y * stride + x;
    }

    const int pw = w + 1;
    *out_stride = pw;
    for (int j = 0; j <= h; j++) {
        int sy = y + j;
        sy = sy < 0 ? 0 : (sy >= plane_h ? plane_h - 1 : sy);
        const uint8_t *row = plane + (size_t)sy * stride;
        uint8_t *out = dst + (size_t)j * pw;
        for (int i = 0; i <= w; i++) {
            int sx = x + i;
            sx = sx < 0 ? 0 : (sx >= plane_w ? plane_w - 1 : sx);
            out[i] = row[sx];
        }
    }
    return dst;
}

/* ------------------------------------------------------------------ luma */

/* The half-sample positions, over a (w+1) x (h+1) region: the quarter
 * positions on the right and bottom edges average with the half sample of
 * the next column or row, so one extra of each is needed.
 *
 * `b` is the horizontal half sample and `v` the vertical one, both clipped.
 * j is not built from either of them: it needs the unrounded intermediates
 * over rows -2..h+3, wider than this region, so it is filtered from the
 * source directly.
 */
#define MAXW 17
#define MAXH 17

/* The unrounded horizontal intermediates span the rows the vertical filter
 * of j reaches: two above the block and four below.
 *
 * ⚠️ They fit in sixteen bits: the six-tap filter of a byte plane reaches
 * 255 x 42 = 10710 at most and -2550 at least. That is what lets the
 * vertical pass below use _mm_madd_epi16, which is the whole reason j is
 * affordable. */
#define MAXHT (MAXH + 6)

#if BC250_H264_SSE2
/* Eight bytes, widened to sixteen-bit lanes. */
static inline __m128i otto(const uint8_t *p)
{
    return _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)p),
                             _mm_setzero_si128());
}

/* The coefficient pair _mm_madd_epi16 applies to interleaved rows. */
#define PAIR(lo, hi) _mm_set1_epi32((int)(((uint32_t)(uint16_t)(hi) << 16) \
                                            | (uint32_t)(uint16_t)(lo)))
#endif

/* The six-tap filter across a row, unrounded, n samples from r[0]. */
static void row_filter(const uint8_t *r, int16_t *o, int n)
{
    int x = 0;
#if BC250_H264_SSE2
    const __m128i c20 = _mm_set1_epi16(20), c5 = _mm_set1_epi16(5);
    for (; x + 8 <= n; x += 8) {
        const __m128i extremes = _mm_add_epi16(otto(r + x - 2), otto(r + x + 3));
        const __m128i centre  = _mm_add_epi16(otto(r + x), otto(r + x + 1));
        const __m128i middle   = _mm_add_epi16(otto(r + x - 1), otto(r + x + 2));
        /* Every term stays inside a signed sixteen-bit lane: 20 x 510 is
         * 10200 and the ends add at most 510 more. */
        __m128i t = _mm_add_epi16(extremes, _mm_mullo_epi16(centre, c20));
        t = _mm_sub_epi16(t, _mm_mullo_epi16(middle, c5));
        _mm_storeu_si128((__m128i *)(o + x), t);
    }
#endif
    for (; x < n; x++)
        o[x] = (int16_t)TAP(r[x - 2], r[x - 1], r[x],
                            r[x + 1], r[x + 2], r[x + 3]);
}

/* The six-tap filter down a column of the source, rounded and clipped. */
static void column_filter(const uint8_t *r, int ss, uint8_t *o, int n)
{
    int x = 0;
#if BC250_H264_SSE2
    const __m128i c20 = _mm_set1_epi16(20), c5 = _mm_set1_epi16(5);
    const __m128i c16 = _mm_set1_epi16(16);
    for (; x + 8 <= n; x += 8) {
        const __m128i extremes = _mm_add_epi16(otto(r + x - 2 * ss),
                                              otto(r + x + 3 * ss));
        const __m128i centre  = _mm_add_epi16(otto(r + x), otto(r + x + ss));
        const __m128i middle   = _mm_add_epi16(otto(r + x - ss),
                                              otto(r + x + 2 * ss));
        __m128i t = _mm_add_epi16(extremes, _mm_mullo_epi16(centre, c20));
        t = _mm_sub_epi16(t, _mm_mullo_epi16(middle, c5));
        t = _mm_srai_epi16(_mm_add_epi16(t, c16), 5);
        _mm_storel_epi64((__m128i *)(o + x), _mm_packus_epi16(t, t));
    }
#endif
    for (; x < n; x++)
        o[x] = clip_uint8((TAP(r[x - 2 * ss], r[x - ss], r[x], r[x + ss],
                               r[x + 2 * ss], r[x + 3 * ss]) + 16) >> 5);
}

/* An unrounded strip, rounded and clipped into bytes. */
static void arrotonda(const int16_t *t, uint8_t *o, int n)
{
    int x = 0;
#if BC250_H264_SSE2
    const __m128i c16 = _mm_set1_epi16(16);
    for (; x + 8 <= n; x += 8) {
        __m128i v = _mm_loadu_si128((const __m128i *)(t + x));
        v = _mm_srai_epi16(_mm_add_epi16(v, c16), 5);
        _mm_storel_epi64((__m128i *)(o + x), _mm_packus_epi16(v, v));
    }
#endif
    for (; x < n; x++)
        o[x] = clip_uint8((t[x] + 16) >> 5);
}

/* The vertical six-tap down the strip of horizontal intermediates: j. */
static void strip_filter(const int16_t *t0, const int16_t *t1, const int16_t *t2,
                         const int16_t *t3, const int16_t *t4, const int16_t *t5,
                         uint8_t *o, int n)
{
    int x = 0;
#if BC250_H264_SSE2
    /* ⚠️ The result reaches 450000, so this cannot stay in sixteen bits.
     * _mm_madd_epi16 multiplies interleaved pairs and accumulates into
     * thirty-two, which turns the six taps into three instructions rather
     * than six widenings. */
    const __m128i ca = PAIR(1, -5), cb = PAIR(20, 20), cc = PAIR(-5, 1);
    const __m128i c512 = _mm_set1_epi32(512);
    for (; x + 8 <= n; x += 8) {
        const __m128i a0 = _mm_loadu_si128((const __m128i *)(t0 + x));
        const __m128i a1 = _mm_loadu_si128((const __m128i *)(t1 + x));
        const __m128i a2 = _mm_loadu_si128((const __m128i *)(t2 + x));
        const __m128i a3 = _mm_loadu_si128((const __m128i *)(t3 + x));
        const __m128i a4 = _mm_loadu_si128((const __m128i *)(t4 + x));
        const __m128i a5 = _mm_loadu_si128((const __m128i *)(t5 + x));

        __m128i lo = _mm_add_epi32(
            _mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi16(a0, a1), ca),
                          _mm_madd_epi16(_mm_unpacklo_epi16(a2, a3), cb)),
            _mm_madd_epi16(_mm_unpacklo_epi16(a4, a5), cc));
        __m128i hi = _mm_add_epi32(
            _mm_add_epi32(_mm_madd_epi16(_mm_unpackhi_epi16(a0, a1), ca),
                          _mm_madd_epi16(_mm_unpackhi_epi16(a2, a3), cb)),
            _mm_madd_epi16(_mm_unpackhi_epi16(a4, a5), cc));

        lo = _mm_srai_epi32(_mm_add_epi32(lo, c512), 10);
        hi = _mm_srai_epi32(_mm_add_epi32(hi, c512), 10);
        const __m128i p = _mm_packs_epi32(lo, hi);
        _mm_storel_epi64((__m128i *)(o + x), _mm_packus_epi16(p, p));
    }
#endif
    for (; x < n; x++)
        o[x] = clip_uint8((TAP(t0[x], t1[x], t2[x], t3[x], t4[x], t5[x])
                           + 512) >> 10);
}

/* Which of b, v and j each of the sixteen positions actually reads, as
 * bits 0, 1 and 2. The table is the table below, read backwards. */
static const uint8_t halves_needed[16] = {
    0,  /* G */  1,  /* a */  1,  /* b */  1,  /* c */
    2,  /* d */  3,  /* e */  5,  /* f */  3,  /* g */
    2,  /* h */  6,  /* i */  4,  /* j */  6,  /* k */
    2,  /* n */  3,  /* p */  5,  /* q */  3,  /* r */
};

/* Copy one row, or average two of them. */
static void out_row(uint8_t *o, const uint8_t *s1, const uint8_t *s2, int w)
{
    if (!s2) {
        memcpy(o, s1, (size_t)w);
        return;
    }
    int x = 0;
#if BC250_H264_SSE2
    for (; x + 16 <= w; x += 16)
        _mm_storeu_si128((__m128i *)(o + x),
            _mm_avg_epu8(_mm_loadu_si128((const __m128i *)(s1 + x)),
                         _mm_loadu_si128((const __m128i *)(s2 + x))));
    for (; x + 8 <= w; x += 8)
        _mm_storel_epi64((__m128i *)(o + x),
            _mm_avg_epu8(_mm_loadl_epi64((const __m128i *)(s1 + x)),
                         _mm_loadl_epi64((const __m128i *)(s2 + x))));
    for (; x + 4 <= w; x += 4)
        *(int32_t *)(o + x) = _mm_cvtsi128_si32(
            _mm_avg_epu8(_mm_cvtsi32_si128(*(const int32_t *)(s1 + x)),
                         _mm_cvtsi32_si128(*(const int32_t *)(s2 + x))));
#endif
    for (; x < w; x++)
        o[x] = (uint8_t)((s1[x] + s2[x] + 1) >> 1);
}

void h264d_mc_luma(uint8_t *dst, int ds, const uint8_t *src, int ss,
                   int w, int h, int xfrac, int yfrac)
{
    const int frac_case = yfrac * 4 + xfrac;
    if (!frac_case) {
        for (int y = 0; y < h; y++)
            memcpy(dst + (size_t)y * ds, src + (size_t)y * ss, (size_t)w);
        return;
    }

    uint8_t b[MAXH][MAXW], v[MAXH][MAXW], j[MAXH][MAXW];
    const int which_halves = halves_needed[frac_case];
    const int n = w + 1;

    if (which_halves & 4) {
        /* One pass over the source for the horizontal intermediates, then
         * one down the strip for j. b is the same strip rounded. */
        int16_t ht[MAXHT][MAXW];
        for (int y = -2; y <= h + 3; y++)
            row_filter(src + (ptrdiff_t)y * ss, ht[y + 2], n);
        for (int y = 0; y <= h; y++)
            strip_filter(ht[y], ht[y + 1], ht[y + 2], ht[y + 3],
                         ht[y + 4], ht[y + 5], j[y], n);
        if (which_halves & 1)
            for (int y = 0; y <= h; y++)
                arrotonda(ht[y + 2], b[y], n);
    } else if (which_halves & 1) {
        int16_t row[MAXW];
        for (int y = 0; y <= h; y++) {
            row_filter(src + (size_t)y * ss, row, n);
            arrotonda(row, b[y], n);
        }
    }
    if (which_halves & 2)
        for (int y = 0; y <= h; y++)
            column_filter(src + (size_t)y * ss, ss, v[y], n);

    /* Every one of the fifteen filtered positions is either one of these
     * arrays, or the average of two of them - which is why the whole switch
     * collapses into a choice of pointers. */
    for (int y = 0; y < h; y++) {
        const uint8_t *g = src + (size_t)y * ss;
        const uint8_t *s1, *s2;
        switch (frac_case) {
        case  1: s1 = g;          s2 = b[y];     break; /* a */
        case  2: s1 = b[y];       s2 = NULL;     break; /* b */
        case  3: s1 = g + 1;      s2 = b[y];     break; /* c */
        case  4: s1 = g;          s2 = v[y];     break; /* d */
        case  5: s1 = b[y];       s2 = v[y];     break; /* e */
        case  6: s1 = b[y];       s2 = j[y];     break; /* f */
        case  7: s1 = b[y];       s2 = v[y] + 1; break; /* g */
        case  8: s1 = v[y];       s2 = NULL;     break; /* h */
        case  9: s1 = v[y];       s2 = j[y];     break; /* i */
        case 10: s1 = j[y];       s2 = NULL;     break; /* j */
        case 11: s1 = v[y] + 1;   s2 = j[y];     break; /* k */
        case 12: s1 = g + ss;     s2 = v[y];     break; /* n */
        case 13: s1 = v[y];       s2 = b[y + 1]; break; /* p */
        case 14: s1 = j[y];       s2 = b[y + 1]; break; /* q */
        default: s1 = v[y] + 1;   s2 = b[y + 1]; break; /* r */
        }
        out_row(dst + (size_t)y * ds, s1, s2, w);
    }
}

/* ---------------------------------------------------------------- chroma */

void h264d_mc_chroma(uint8_t *dst, int ds, const uint8_t *src, int ss,
                     int w, int h, int xfrac, int yfrac)
{
    /* ⚠️ The three cases below are the general one with the zero weights
     * dropped, not approximations of it: (8-f)*8 and f*8 divided through by
     * eight give the same result with the same rounding, because the
     * rounding constant divides through too. */
    if (!xfrac && !yfrac) {
        for (int y = 0; y < h; y++)
            memcpy(dst + (size_t)y * ds, src + (size_t)y * ss, (size_t)w);
        return;
    }

    /* ⚠️ The weights sum to sixty-four and the samples reach 255, so the
     * whole sum is at most 16320: it fits a signed sixteen-bit lane with
     * room to spare, and no thirty-two bit widening is needed anywhere in
     * this filter. */
#if BC250_H264_SSE2
    const __m128i zero = _mm_setzero_si128();
#define CHROMA_TWO(pa, pb, ca, cb, rounding, shift)                              \
    do {                                                                      \
        const __m128i va = _mm_unpacklo_epi8(                                 \
            _mm_loadl_epi64((const __m128i *)(pa)), zero);                    \
        const __m128i vb = _mm_unpacklo_epi8(                                 \
            _mm_loadl_epi64((const __m128i *)(pb)), zero);                    \
        __m128i v = _mm_add_epi16(_mm_mullo_epi16(va, (ca)),                  \
                                  _mm_mullo_epi16(vb, (cb)));                 \
        v = _mm_srli_epi16(_mm_add_epi16(v, (rounding)), (shift));              \
        _mm_storel_epi64((__m128i *)(o + x), _mm_packus_epi16(v, v));         \
    } while (0)
#endif

    if (!yfrac) {
        const int a = 8 - xfrac;
#if BC250_H264_SSE2
        const __m128i ca = _mm_set1_epi16((short)a);
        const __m128i cb = _mm_set1_epi16((short)xfrac);
        const __m128i t4 = _mm_set1_epi16(4);
#endif
        for (int y = 0; y < h; y++) {
            const uint8_t *r = src + (size_t)y * ss;
            uint8_t *o = dst + (size_t)y * ds;
            int x = 0;
#if BC250_H264_SSE2
            for (; x + 8 <= w; x += 8) CHROMA_TWO(r + x, r + x + 1, ca, cb, t4, 3);
#endif
            for (; x < w; x++)
                o[x] = (uint8_t)((a * r[x] + xfrac * r[x + 1] + 4) >> 3);
        }
        return;
    }

    if (!xfrac) {
        const int a = 8 - yfrac;
#if BC250_H264_SSE2
        const __m128i ca = _mm_set1_epi16((short)a);
        const __m128i cb = _mm_set1_epi16((short)yfrac);
        const __m128i t4 = _mm_set1_epi16(4);
#endif
        for (int y = 0; y < h; y++) {
            const uint8_t *r0 = src + (size_t)y * ss;
            const uint8_t *r1 = r0 + ss;
            uint8_t *o = dst + (size_t)y * ds;
            int x = 0;
#if BC250_H264_SSE2
            for (; x + 8 <= w; x += 8) CHROMA_TWO(r0 + x, r1 + x, ca, cb, t4, 3);
#endif
            for (; x < w; x++)
                o[x] = (uint8_t)((a * r0[x] + yfrac * r1[x] + 4) >> 3);
        }
        return;
    }

    const int a = (8 - xfrac) * (8 - yfrac);
    const int b = xfrac * (8 - yfrac);
    const int c = (8 - xfrac) * yfrac;
    const int d = xfrac * yfrac;
#if BC250_H264_SSE2
    const __m128i va = _mm_set1_epi16((short)a), vb = _mm_set1_epi16((short)b);
    const __m128i vc = _mm_set1_epi16((short)c), vd = _mm_set1_epi16((short)d);
    const __m128i t32 = _mm_set1_epi16(32);
#endif

    for (int y = 0; y < h; y++) {
        const uint8_t *r0 = src + (size_t)y * ss;
        const uint8_t *r1 = r0 + ss;
        uint8_t *o = dst + (size_t)y * ds;
        int x = 0;
#if BC250_H264_SSE2
        for (; x + 8 <= w; x += 8) {
            const __m128i p00 = _mm_unpacklo_epi8(
                _mm_loadl_epi64((const __m128i *)(r0 + x)), zero);
            const __m128i p01 = _mm_unpacklo_epi8(
                _mm_loadl_epi64((const __m128i *)(r0 + x + 1)), zero);
            const __m128i p10 = _mm_unpacklo_epi8(
                _mm_loadl_epi64((const __m128i *)(r1 + x)), zero);
            const __m128i p11 = _mm_unpacklo_epi8(
                _mm_loadl_epi64((const __m128i *)(r1 + x + 1)), zero);
            __m128i v = _mm_add_epi16(
                _mm_add_epi16(_mm_mullo_epi16(p00, va), _mm_mullo_epi16(p01, vb)),
                _mm_add_epi16(_mm_mullo_epi16(p10, vc), _mm_mullo_epi16(p11, vd)));
            v = _mm_srli_epi16(_mm_add_epi16(v, t32), 6);
            _mm_storel_epi64((__m128i *)(o + x), _mm_packus_epi16(v, v));
        }
#endif
        for (; x < w; x++)
            o[x] = (uint8_t)((a * r0[x] + b * r0[x + 1]
                            + c * r1[x] + d * r1[x + 1] + 32) >> 6);
    }
#if BC250_H264_SSE2
#undef CHROMA_TWO
#endif
}

/* ----------------------------------------------------------- combination */

void h264d_mc_average(uint8_t *dst, int ds, const uint8_t *a, int as,
                      const uint8_t *b, int bs, int w, int h)
{
    for (int y = 0; y < h; y++) {
        const uint8_t *ra = a + (size_t)y * as;
        const uint8_t *rb = b + (size_t)y * bs;
        uint8_t *o = dst + (size_t)y * ds;
        int x = 0;
#if BC250_H264_SSE2
        /* ⚠️ _mm_avg_epu8 is (a + b + 1) >> 1 on each byte, which is what
         * the clause says, not an approximation of it. */
        for (; x + 16 <= w; x += 16)
            _mm_storeu_si128((__m128i *)(o + x),
                _mm_avg_epu8(_mm_loadu_si128((const __m128i *)(ra + x)),
                             _mm_loadu_si128((const __m128i *)(rb + x))));
        for (; x + 8 <= w; x += 8)
            _mm_storel_epi64((__m128i *)(o + x),
                _mm_avg_epu8(_mm_loadl_epi64((const __m128i *)(ra + x)),
                             _mm_loadl_epi64((const __m128i *)(rb + x))));
        for (; x + 4 <= w; x += 4) {
            const __m128i va = _mm_cvtsi32_si128(*(const int32_t *)(ra + x));
            const __m128i vb = _mm_cvtsi32_si128(*(const int32_t *)(rb + x));
            *(int32_t *)(o + x) = _mm_cvtsi128_si32(_mm_avg_epu8(va, vb));
        }
#endif
        for (; x < w; x++)
            o[x] = (uint8_t)((ra[x] + rb[x] + 1) >> 1);
    }
}

void h264d_mc_weight(uint8_t *dst, int ds, const uint8_t *src, int ss,
                     int w, int h, int log2_denom, int weight, int offset)
{
    /* ⚠️ Exactly a copy, not nearly one: with weight = 1 << denom and no
     * offset, ((src << denom) + (1 << (denom - 1))) >> denom is src for
     * every src. x264's weightp leaves most references here. */
    if (weight == (1 << log2_denom) && offset == 0) {
        for (int y = 0; y < h; y++)
            memcpy(dst + (size_t)y * ds, src + (size_t)y * ss, (size_t)w);
        return;
    }

#if BC250_H264_SSE2
    /* One weight times one sample is at most 255 x 127, so the whole thing
     * stays in signed sixteen-bit lanes, and _mm_packus_epi16 saturates to
     * 0..255 - which is exactly the clip. */
    const __m128i zero = _mm_setzero_si128();
    const __m128i vw = _mm_set1_epi16((short)weight);
    const __m128i vt = _mm_set1_epi16((short)(log2_denom ? (1 << (log2_denom - 1)) : 0));
    const __m128i vo = _mm_set1_epi16((short)offset);
#endif
    for (int y = 0; y < h; y++) {
        const uint8_t *r = src + (size_t)y * ss;
        uint8_t *o = dst + (size_t)y * ds;
        int x = 0;
#if BC250_H264_SSE2
        for (; x + 8 <= w; x += 8) {
            __m128i v = _mm_unpacklo_epi8(
                _mm_loadl_epi64((const __m128i *)(r + x)), zero);
            v = _mm_add_epi16(_mm_mullo_epi16(v, vw), vt);
            v = _mm_adds_epi16(_mm_srai_epi16(v, log2_denom), vo);
            _mm_storel_epi64((__m128i *)(o + x), _mm_packus_epi16(v, v));
        }
        if (x + 4 <= w) {
            __m128i v = _mm_unpacklo_epi8(
                _mm_cvtsi32_si128(*(const int32_t *)(r + x)), zero);
            v = _mm_add_epi16(_mm_mullo_epi16(v, vw), vt);
            v = _mm_adds_epi16(_mm_srai_epi16(v, log2_denom), vo);
            *(int32_t *)(o + x) = _mm_cvtsi128_si32(_mm_packus_epi16(v, v));
            x += 4;
        }
#endif
        if (log2_denom)
            for (; x < w; x++)
                o[x] = clip_uint8(((r[x] * weight + (1 << (log2_denom - 1)))
                                   >> log2_denom) + offset);
        else
            for (; x < w; x++)
                o[x] = clip_uint8(r[x] * weight + offset);
    }
}

/* ⚠️ The shift comes first and the offset afterwards. Folding the offset
 * into the rounding term looks equivalent and is not: it rounds the two
 * halves together instead of separately, and bi-predicted blocks come out
 * one low on a good share of samples. */
void h264d_mc_weight_bi(uint8_t *dst, int ds,
                        const uint8_t *a, int as, const uint8_t *b, int bs,
                        int w, int h, int log2_denom, int w0, int o0, int w1, int o1)
{
    /* Likewise exactly the average: two equal weights of 1 << denom with
     * no offset collapse to (a + b + 1) >> 1. Implicit bi-prediction lands
     * here whenever the two references sit symmetrically around this
     * picture, which in a regular B structure is most of the time. */
    /* ⚠️ log2_denom >= 1 is part of the condition, not a detail of it. At
     * zero the clause does not shift, so weights of one mean a + b clipped
     * and not the average of a and b. */
    if (log2_denom >= 1 && w0 == (1 << log2_denom) && w1 == w0
        && o0 == 0 && o1 == 0) {
        h264d_mc_average(dst, ds, a, as, b, bs, w, h);
        return;
    }

    const int off = (o0 + o1 + 1) >> 1;
#if BC250_H264_SSE2
    /* ⚠️ Two weighted samples added together overflow a sixteen-bit lane
     * (255 x 127 x 2), so the pair goes through _mm_madd_epi16, which
     * multiplies and accumulates into thirty-two bits in one step. The
     * samples are interleaved a,b,a,b so the coefficient vector is just
     * w0,w1 repeated. */
    const __m128i zero = _mm_setzero_si128();
    const __m128i vw = _mm_set1_epi32((int)(((uint32_t)(w1 & 0xffff) << 16)
                                            | (uint32_t)(w0 & 0xffff)));
    const int rounding = (log2_denom >= 1) ? (1 << log2_denom) : 0;
    const int shift = (log2_denom >= 1) ? (log2_denom + 1) : 0;
    const __m128i vt = _mm_set1_epi32(rounding);
    const __m128i vo = _mm_set1_epi16((short)off);
#endif
    for (int y = 0; y < h; y++) {
        const uint8_t *ra = a + (size_t)y * as;
        const uint8_t *rb = b + (size_t)y * bs;
        uint8_t *o = dst + (size_t)y * ds;
        int x = 0;
#if BC250_H264_SSE2
        for (; x + 8 <= w; x += 8) {
            const __m128i ia = _mm_loadl_epi64((const __m128i *)(ra + x));
            const __m128i ib = _mm_loadl_epi64((const __m128i *)(rb + x));
            const __m128i mix = _mm_unpacklo_epi8(ia, ib);
            __m128i lo = _mm_madd_epi16(_mm_unpacklo_epi8(mix, zero), vw);
            __m128i hi = _mm_madd_epi16(_mm_unpackhi_epi8(mix, zero), vw);
            lo = _mm_srai_epi32(_mm_add_epi32(lo, vt), shift);
            hi = _mm_srai_epi32(_mm_add_epi32(hi, vt), shift);
            /* ⚠️ Saturating, not wrapping. At log2_denom zero nothing is
             * shifted, the sum can reach 64770, and packing it down clamps
             * it at 32767 - where a wrapping add of the offset would turn
             * the brightest sample into the darkest. */
            __m128i v = _mm_adds_epi16(_mm_packs_epi32(lo, hi), vo);
            _mm_storel_epi64((__m128i *)(o + x), _mm_packus_epi16(v, v));
        }
        if (x + 4 <= w) {
            const __m128i ia = _mm_cvtsi32_si128(*(const int32_t *)(ra + x));
            const __m128i ib = _mm_cvtsi32_si128(*(const int32_t *)(rb + x));
            const __m128i mix = _mm_unpacklo_epi8(ia, ib);
            __m128i lo = _mm_madd_epi16(_mm_unpacklo_epi8(mix, zero), vw);
            lo = _mm_srai_epi32(_mm_add_epi32(lo, vt), shift);
            __m128i v = _mm_adds_epi16(_mm_packs_epi32(lo, lo), vo);
            *(int32_t *)(o + x) = _mm_cvtsi128_si32(_mm_packus_epi16(v, v));
            x += 4;
        }
#endif
        if (log2_denom >= 1) {
            const int t = 1 << log2_denom;
            for (; x < w; x++)
                o[x] = clip_uint8((((ra[x] * w0 + rb[x] * w1 + t)
                                    >> (log2_denom + 1))) + off);
        } else {
            for (; x < w; x++)
                o[x] = clip_uint8(ra[x] * w0 + rb[x] * w1 + off);
        }
    }
}
