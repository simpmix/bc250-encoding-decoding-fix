/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * hevc_filter.c - the deblocking filter, Rec. ITU-T H.265 clause 8.7.2.
 *
 * Block transforms leave steps at the block edges, and the eye finds a
 * straight edge that is not in the picture far more readily than it finds
 * the error that produced it. So the standard smooths them - and does it
 * normatively, in the decoding loop, because the filtered picture is what
 * the next one predicts from. A decoder that skips it does not merely look
 * worse: it drifts.
 *
 * ⚠️ Every vertical edge in the picture is filtered before any horizontal
 * one, and the horizontal pass reads what the vertical pass wrote. Doing
 * it edge by edge, both directions at once, gives a different picture.
 * Two passes over the whole picture is the simplest way to be sure, and
 * the passes are independent inside themselves: the edges of one direction
 * are eight samples apart and reach four, so no two of them touch.
 */
#include "hevc_dec_internal.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* The edge offset at eight bits, sixteen samples at a time. SSSE3 for
 * _mm_shuffle_epi8, which looks the five cases up in one instruction; the
 * decoder asks for it at run time, as hevc_mc.c does, and BC250_HEVC_NOSIMD
 * turns it off to compare with the scalar loop beside it.
 *
 * ⚠️ The samples are compared as signed bytes after flipping the top bit,
 * which orders them the same way as unsigned ones: SSE has no unsigned
 * byte comparison. */
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

/* ⚠️ Atomic, because the first call can come from several filter threads
 * at once. They would all store the same answer, but two plain stores to
 * one int from two threads are a data race all the same, and
 * ThreadSanitizer says so. */
static int sao_vector(void)
{
    static _Atomic int answer = -1;
    int a = atomic_load_explicit(&answer, memory_order_relaxed);
    if (a < 0) {
        a = !getenv("BC250_HEVC_NOSIMD")
            && __builtin_cpu_supports("ssse3") ? 1 : 0;
        atomic_store_explicit(&answer, a, memory_order_relaxed);
    }
    return a;
}

__attribute__((target("ssse3")))
static void sao_edge_row_ssse3(uint8_t *out, const uint8_t *cur,
                               const uint8_t *a, const uint8_t *b, int n,
                               const int8_t table[16])
{
    const __m128i flip = _mm_set1_epi8((char)0x80);
    const __m128i two = _mm_set1_epi8(2);
    const __m128i zero = _mm_setzero_si128();
    const __m128i lut = _mm_loadu_si128((const __m128i *)table);
    int x = 0;
    for (; x + 16 <= n; x += 16) {
        const __m128i v = _mm_loadu_si128((const __m128i *)(cur + x));
        const __m128i vs = _mm_xor_si128(v, flip);
        const __m128i as = _mm_xor_si128(
            _mm_loadu_si128((const __m128i *)(a + x)), flip);
        const __m128i bs = _mm_xor_si128(
            _mm_loadu_si128((const __m128i *)(b + x)), flip);
        /* sign(v - a): the two comparisons are all ones where true. */
        const __m128i sa = _mm_sub_epi8(_mm_cmpgt_epi8(as, vs),
                                        _mm_cmpgt_epi8(vs, as));
        const __m128i sb = _mm_sub_epi8(_mm_cmpgt_epi8(bs, vs),
                                        _mm_cmpgt_epi8(vs, bs));
        const __m128i idx = _mm_add_epi8(_mm_add_epi8(sa, sb), two);
        const __m128i off = _mm_shuffle_epi8(lut, idx);
        const __m128i neg = _mm_cmpgt_epi8(zero, off);
        const __m128i lo = _mm_add_epi16(_mm_unpacklo_epi8(v, zero),
                                         _mm_unpacklo_epi8(off, neg));
        const __m128i hi = _mm_add_epi16(_mm_unpackhi_epi8(v, zero),
                                         _mm_unpackhi_epi8(off, neg));
        _mm_storeu_si128((__m128i *)(out + x), _mm_packus_epi16(lo, hi));
    }
    for (; x < n; x++) {
        const int v = cur[x];
        const int idx = 2 + (v > a[x]) - (v < a[x]) + (v > b[x]) - (v < b[x]);
        const int r = v + table[idx];
        out[x] = (uint8_t)(r < 0 ? 0 : (r > 255 ? 255 : r));
    }
}

#if defined(__x86_64__) || defined(_M_X64)
static void sao_edge_row_10_sse(uint16_t *out, const uint16_t *cur,
                                const uint16_t *a, const uint16_t *b, int n,
                                const int8_t table[16])
{
    const __m128i two = _mm_set1_epi16(2);
    const __m128i zero = _mm_setzero_si128();
    const __m128i max_val = _mm_set1_epi16(1023);
    const int16_t t0 = (int16_t)table[0];
    const int16_t t1 = (int16_t)table[1];
    const int16_t t3 = (int16_t)table[3];
    const int16_t t4 = (int16_t)table[4];
    int x = 0;
    for (; x + 8 <= n; x += 8) {
        __m128i v = _mm_loadu_si128((const __m128i *)(cur + x));
        __m128i as = _mm_loadu_si128((const __m128i *)(a + x));
        __m128i bs = _mm_loadu_si128((const __m128i *)(b + x));
        __m128i sa = _mm_sub_epi16(_mm_cmpgt_epi16(as, v), _mm_cmpgt_epi16(v, as));
        __m128i sb = _mm_sub_epi16(_mm_cmpgt_epi16(bs, v), _mm_cmpgt_epi16(v, bs));
        __m128i idx = _mm_add_epi16(_mm_add_epi16(sa, sb), two);
        __m128i is0 = _mm_cmpeq_epi16(idx, zero);
        __m128i is1 = _mm_cmpeq_epi16(idx, _mm_set1_epi16(1));
        __m128i is3 = _mm_cmpeq_epi16(idx, _mm_set1_epi16(3));
        __m128i is4 = _mm_cmpeq_epi16(idx, _mm_set1_epi16(4));
        __m128i off = _mm_or_si128(
            _mm_or_si128(_mm_and_si128(is0, _mm_set1_epi16(t0)), _mm_and_si128(is1, _mm_set1_epi16(t1))),
            _mm_or_si128(_mm_and_si128(is3, _mm_set1_epi16(t3)), _mm_and_si128(is4, _mm_set1_epi16(t4)))
        );
        __m128i res = _mm_add_epi16(v, off);
        res = _mm_min_epi16(_mm_max_epi16(res, zero), max_val);
        _mm_storeu_si128((__m128i *)(out + x), res);
    }
    for (; x < n; x++) {
        const int v = cur[x];
        const int idx = 2 + (v > a[x]) - (v < a[x]) + (v > b[x]) - (v < b[x]);
        const int r = v + table[idx];
        out[x] = (uint16_t)(r < 0 ? 0 : (r > 1023 ? 1023 : r));
    }
}

__attribute__((target("avx2")))
static void sao_edge_row_10_avx2(uint16_t *out, const uint16_t *cur,
                                 const uint16_t *a, const uint16_t *b, int n,
                                 const int8_t table[16])
{
    const int n16 = n & ~15;
    const __m256i two = _mm256_set1_epi16(2);
    const __m256i zero = _mm256_setzero_si256();
    const __m256i max_val = _mm256_set1_epi16(1023);
    const int16_t t0 = (int16_t)table[0];
    const int16_t t1 = (int16_t)table[1];
    const int16_t t3 = (int16_t)table[3];
    const int16_t t4 = (int16_t)table[4];
    for (int x = 0; x < n16; x += 16) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(cur + x));
        __m256i as = _mm256_loadu_si256((const __m256i *)(a + x));
        __m256i bs = _mm256_loadu_si256((const __m256i *)(b + x));
        __m256i sa = _mm256_sub_epi16(_mm256_cmpgt_epi16(as, v), _mm256_cmpgt_epi16(v, as));
        __m256i sb = _mm256_sub_epi16(_mm256_cmpgt_epi16(bs, v), _mm256_cmpgt_epi16(v, bs));
        __m256i idx = _mm256_add_epi16(_mm256_add_epi16(sa, sb), two);
        __m256i is0 = _mm256_cmpeq_epi16(idx, zero);
        __m256i is1 = _mm256_cmpeq_epi16(idx, _mm256_set1_epi16(1));
        __m256i is3 = _mm256_cmpeq_epi16(idx, _mm256_set1_epi16(3));
        __m256i is4 = _mm256_cmpeq_epi16(idx, _mm256_set1_epi16(4));
        __m256i off = _mm256_or_si256(
            _mm256_or_si256(_mm256_and_si256(is0, _mm256_set1_epi16(t0)), _mm256_and_si256(is1, _mm256_set1_epi16(t1))),
            _mm256_or_si256(_mm256_and_si256(is3, _mm256_set1_epi16(t3)), _mm256_and_si256(is4, _mm256_set1_epi16(t4)))
        );
        __m256i res = _mm256_add_epi16(v, off);
        res = _mm256_min_epi16(_mm256_max_epi16(res, zero), max_val);
        _mm256_storeu_si256((__m256i *)(out + x), res);
    }
    if (n16 < n)
        sao_edge_row_10_sse(out + n16, cur + n16, a + n16, b + n16, n - n16, table);
}

static void sao_edge_row_10(uint16_t *out, const uint16_t *cur,
                            const uint16_t *a, const uint16_t *b, int n,
                            const int8_t table[16])
{
    if (n >= 16 && __builtin_cpu_supports("avx2"))
        sao_edge_row_10_avx2(out, cur, a, b, n, table);
    else
        sao_edge_row_10_sse(out, cur, a, b, n, table);
}
#endif
/* The luma deblocking filter of 8.7.2.5.3 to 8.7.2.5.7, one four-line
 * segment, eight bits.
 *
 * The eight samples across the edge of each of the four lines go into
 * eight vectors, p3 to q3, one line per lane. For a horizontal edge the
 * lines are columns and each vector is four bytes of one row; for a
 * vertical edge the lines are rows, and the four rows of eight bytes are
 * transposed on the way in and back on the way out.
 *
 * The decisions stay scalar and read lanes 0 and 3, exactly the two lines
 * the standard looks at. The filters themselves run on all four lanes,
 * with a mask for the weak filter's per-line "is the step small enough".
 * Every value that goes back is clipped to 0..255 by _mm_packus_epi16,
 * which is clip_pixel() at eight bits.
 *
 * ⚠️ All the arithmetic is sixteen-bit and none of it can overflow: the
 * largest sum, 8 * 255 + 4, and 9 * 255 + 3 * 255 + 8 in the weak delta,
 * are far inside it. The shifts of values that can be negative are
 * arithmetic, as the scalar C is. */
#define LOAD32(p) ({ uint32_t w_; memcpy(&w_, (p), 4); (int)w_; })

__attribute__((target("ssse3")))
static void filter_luma_ssse3(uint8_t *base, int forward, int giu,
                              int beta, int tc, bool keep_p, bool keep_q)
{
    const __m128i zero = _mm_setzero_si128();
    __m128i p3, p2, p1, p0, q0, q1, q2, q3;
    const bool rows = (forward == 1);       /* a vertical edge */

    if (rows) {
        const __m128i r0 = _mm_loadl_epi64((const __m128i *)(base - 4));
        const __m128i r1 = _mm_loadl_epi64((const __m128i *)(base + giu - 4));
        const __m128i r2 = _mm_loadl_epi64((const __m128i *)(base + 2 * giu - 4));
        const __m128i r3 = _mm_loadl_epi64((const __m128i *)(base + 3 * giu - 4));
        /* Columns of four bytes: x - 4 to x - 1 in lo, x to x + 3 in hi. */
        const __m128i a = _mm_unpacklo_epi8(r0, r1);
        const __m128i b = _mm_unpacklo_epi8(r2, r3);
        const __m128i lo = _mm_unpacklo_epi16(a, b);
        const __m128i hi = _mm_unpackhi_epi16(a, b);
        const __m128i c01 = _mm_unpacklo_epi8(lo, zero);
        const __m128i c23 = _mm_unpackhi_epi8(lo, zero);
        const __m128i c45 = _mm_unpacklo_epi8(hi, zero);
        const __m128i c67 = _mm_unpackhi_epi8(hi, zero);
        p3 = c01; p2 = _mm_srli_si128(c01, 8);
        p1 = c23; p0 = _mm_srli_si128(c23, 8);
        q0 = c45; q1 = _mm_srli_si128(c45, 8);
        q2 = c67; q3 = _mm_srli_si128(c67, 8);
    } else {
#define ROW(k) _mm_unpacklo_epi8(_mm_cvtsi32_si128(LOAD32(base + (k) * forward)), zero)
        p3 = ROW(-4); p2 = ROW(-3); p1 = ROW(-2); p0 = ROW(-1);
        q0 = ROW(0);  q1 = ROW(1);  q2 = ROW(2);  q3 = ROW(3);
#undef ROW
    }

    /* 8.7.2.5.3: the decision, from lines 0 and 3. */
    const __m128i dpv = _mm_abs_epi16(_mm_add_epi16(
        _mm_sub_epi16(p2, _mm_add_epi16(p1, p1)), p0));
    const __m128i dqv = _mm_abs_epi16(_mm_add_epi16(
        _mm_sub_epi16(q2, _mm_add_epi16(q1, q1)), q0));
    const int dp0 = _mm_extract_epi16(dpv, 0), dp3 = _mm_extract_epi16(dpv, 3);
    const int dq0 = _mm_extract_epi16(dqv, 0), dq3 = _mm_extract_epi16(dqv, 3);
    const int dpq0 = dp0 + dq0, dpq3 = dp3 + dq3;
    if (dpq0 + dpq3 >= beta) return;

    const int dp = dp0 + dp3, dq = dq0 + dq3;
    const int threshold = (5 * tc + 1) >> 1;
    const __m128i flat = _mm_add_epi16(_mm_abs_epi16(_mm_sub_epi16(p3, p0)),
                                       _mm_abs_epi16(_mm_sub_epi16(q0, q3)));
    const __m128i step = _mm_abs_epi16(_mm_sub_epi16(p0, q0));
    const bool strong =
        2 * dpq0 < (beta >> 2) && _mm_extract_epi16(flat, 0) < (beta >> 3)
        && _mm_extract_epi16(step, 0) < threshold
        && 2 * dpq3 < (beta >> 2) && _mm_extract_epi16(flat, 3) < (beta >> 3)
        && _mm_extract_epi16(step, 3) < threshold;

    __m128i np2 = p2, np1 = p1, np0 = p0, nq0 = q0, nq1 = q1, nq2 = q2;
    if (strong) {
        /* 8.7.2.5.7: three samples each side, each held within 2 tC. */
        const __m128i tc2 = _mm_set1_epi16((int16_t)(2 * tc));
        const __m128i two = _mm_set1_epi16(2), four = _mm_set1_epi16(4);
#define HOLD(v, x) _mm_min_epi16(_mm_max_epi16((v), _mm_sub_epi16((x), tc2)), \
                                 _mm_add_epi16((x), tc2))
        const __m128i p0q0 = _mm_add_epi16(p0, q0);
        if (!keep_p) {
            np0 = HOLD(_mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(
                      _mm_add_epi16(p2, q1), _mm_add_epi16(
                      _mm_add_epi16(p1, p1), _mm_add_epi16(p0q0, p0q0))), four), 3), p0);
            np1 = HOLD(_mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(
                      _mm_add_epi16(p2, p1), p0q0), two), 2), p1);
            np2 = HOLD(_mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(
                      _mm_add_epi16(_mm_add_epi16(p3, p3), _mm_add_epi16(
                      _mm_add_epi16(p2, p2), p2)), _mm_add_epi16(p1, p0q0)), four), 3), p2);
        }
        if (!keep_q) {
            nq0 = HOLD(_mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(
                      _mm_add_epi16(p1, q2), _mm_add_epi16(
                      _mm_add_epi16(q1, q1), _mm_add_epi16(p0q0, p0q0))), four), 3), q0);
            nq1 = HOLD(_mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(
                      _mm_add_epi16(q2, q1), p0q0), two), 2), q1);
            nq2 = HOLD(_mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(
                      _mm_add_epi16(_mm_add_epi16(q3, q3), _mm_add_epi16(
                      _mm_add_epi16(q2, q2), q2)), _mm_add_epi16(q1, p0q0)), four), 3), q2);
        }
#undef HOLD
    } else {
        /* 8.7.2.5.7, the weak filter: one sample each side on every line
         * whose step is small enough, a second on a side flat enough. */
        const bool touch_p1 = dp < ((beta + (beta >> 1)) >> 3);
        const bool touch_q1 = dq < ((beta + (beta >> 1)) >> 3);
        const __m128i tcv = _mm_set1_epi16((int16_t)tc);
        const __m128i ntc = _mm_set1_epi16((int16_t)-tc);
        const __m128i one = _mm_set1_epi16(1);
        __m128i delta = _mm_srai_epi16(_mm_add_epi16(_mm_sub_epi16(
            _mm_mullo_epi16(_mm_sub_epi16(q0, p0), _mm_set1_epi16(9)),
            _mm_mullo_epi16(_mm_sub_epi16(q1, p1), _mm_set1_epi16(3))),
            _mm_set1_epi16(8)), 4);
        const __m128i ok = _mm_cmplt_epi16(_mm_abs_epi16(delta),
                                           _mm_set1_epi16((int16_t)(10 * tc)));
        delta = _mm_min_epi16(_mm_max_epi16(delta, ntc), tcv);
#define PICK_OK(v, x) _mm_or_si128(_mm_and_si128(ok, (v)), _mm_andnot_si128(ok, (x)))
        if (!keep_p) {
            np0 = PICK_OK(_mm_add_epi16(p0, delta), p0);
            if (touch_p1) {
                const __m128i htc = _mm_set1_epi16((int16_t)(tc >> 1));
                const __m128i nhtc = _mm_set1_epi16((int16_t)-(tc >> 1));
                __m128i d1 = _mm_srai_epi16(_mm_add_epi16(_mm_sub_epi16(
                    _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(p2, p0), one), 1),
                    p1), delta), 1);
                d1 = _mm_min_epi16(_mm_max_epi16(d1, nhtc), htc);
                np1 = PICK_OK(_mm_add_epi16(p1, d1), p1);
            }
        }
        if (!keep_q) {
            nq0 = PICK_OK(_mm_sub_epi16(q0, delta), q0);
            if (touch_q1) {
                const __m128i htc = _mm_set1_epi16((int16_t)(tc >> 1));
                const __m128i nhtc = _mm_set1_epi16((int16_t)-(tc >> 1));
                __m128i d1 = _mm_srai_epi16(_mm_sub_epi16(_mm_sub_epi16(
                    _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(q2, q0), one), 1),
                    q1), delta), 1);
                d1 = _mm_min_epi16(_mm_max_epi16(d1, nhtc), htc);
                nq1 = PICK_OK(_mm_add_epi16(q1, d1), q1);
            }
        }
#undef PICK_OK
    }

    if (rows) {
        /* Back into rows: columns 0-3 and 4-7 as bytes, each a 4x4 block
         * stored column by column, turned by one shuffle, and the two
         * halves of every row put side by side. */
        const __m128i left = _mm_packus_epi16(_mm_unpacklo_epi64(p3, np2),
                                              _mm_unpacklo_epi64(np1, np0));
        const __m128i right = _mm_packus_epi16(_mm_unpacklo_epi64(nq0, nq1),
                                               _mm_unpacklo_epi64(nq2, q3));
        const __m128i turn = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13,
                                           2, 6, 10, 14, 3, 7, 11, 15);
        const __m128i tl = _mm_shuffle_epi8(left, turn);
        const __m128i tr = _mm_shuffle_epi8(right, turn);
        const __m128i r01 = _mm_unpacklo_epi32(tl, tr);
        const __m128i r23 = _mm_unpackhi_epi32(tl, tr);
        _mm_storel_epi64((__m128i *)(base - 4), r01);
        _mm_storel_epi64((__m128i *)(base + giu - 4), _mm_srli_si128(r01, 8));
        _mm_storel_epi64((__m128i *)(base + 2 * giu - 4), r23);
        _mm_storel_epi64((__m128i *)(base + 3 * giu - 4), _mm_srli_si128(r23, 8));
        return;
    }
#define STORE_ROW(k, v) do { \
        const int w_ = _mm_cvtsi128_si32(_mm_packus_epi16((v), (v))); \
        memcpy(base + (k) * forward, &w_, 4); } while (0)
    if (!keep_p) {
        STORE_ROW(-3, np2); STORE_ROW(-2, np1); STORE_ROW(-1, np0);
    }
    if (!keep_q) {
        STORE_ROW(0, nq0); STORE_ROW(1, nq1); STORE_ROW(2, nq2);
    }
#undef STORE_ROW
}
/* 8.7.2.5.5, the chroma filter, one four-line segment at eight bits: one
 * sample each side, no decision. p1, p0, q0 and q1 of the four lines go
 * into four vectors, one line per lane - four bytes of one row each for a
 * horizontal edge, a 4x4 block of bytes turned by one shuffle for a
 * vertical one, which turns it back too - and _mm_packus_epi16 does the
 * clip to 0..255. */
__attribute__((target("ssse3")))
static void filter_chroma_ssse3(uint8_t *base, int forward, int giu, int tc,
                                bool keep_p, bool keep_q)
{
    const __m128i zero = _mm_setzero_si128();
    const __m128i turn = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13,
                                       2, 6, 10, 14, 3, 7, 11, 15);
    const bool rows = (forward == 1);       /* a vertical edge */
    __m128i p1, p0, q0, q1;

    if (rows) {
        /* Four rows of x - 2 .. x + 1, then one column per four bytes. */
        const __m128i block = _mm_shuffle_epi8(_mm_setr_epi32(
            LOAD32(base - 2), LOAD32(base + giu - 2),
            LOAD32(base + 2 * giu - 2), LOAD32(base + 3 * giu - 2)), turn);
        const __m128i lo = _mm_unpacklo_epi8(block, zero);
        const __m128i hi = _mm_unpackhi_epi8(block, zero);
        p1 = lo; p0 = _mm_srli_si128(lo, 8);
        q0 = hi; q1 = _mm_srli_si128(hi, 8);
    } else {
#define ROW(k) _mm_unpacklo_epi8(_mm_cvtsi32_si128(LOAD32(base + (k) * forward)), zero)
        p1 = ROW(-2); p0 = ROW(-1); q0 = ROW(0); q1 = ROW(1);
#undef ROW
    }

    __m128i delta = _mm_srai_epi16(_mm_add_epi16(_mm_sub_epi16(_mm_add_epi16(
        _mm_slli_epi16(_mm_sub_epi16(q0, p0), 2), p1), q1), _mm_set1_epi16(4)), 3);
    delta = _mm_min_epi16(_mm_max_epi16(delta, _mm_set1_epi16((int16_t)-tc)),
                          _mm_set1_epi16((int16_t)tc));
    const __m128i np0 = keep_p ? p0 : _mm_add_epi16(p0, delta);
    const __m128i nq0 = keep_q ? q0 : _mm_sub_epi16(q0, delta);

    if (rows) {
        const __m128i out = _mm_shuffle_epi8(
            _mm_packus_epi16(_mm_unpacklo_epi64(p1, np0),
                             _mm_unpacklo_epi64(nq0, q1)), turn);
        /* Row i is bytes 4i .. 4i + 3: p1, p0, q0, q1 of that row. */
        int r_[4];
        _mm_storeu_si128((__m128i *)r_, out);
        for (int i = 0; i < 4; i++) memcpy(base + i * giu - 2, &r_[i], 4);
        return;
    }
    if (!keep_p) {
        const int w_ = _mm_cvtsi128_si32(_mm_packus_epi16(np0, np0));
        memcpy(base - forward, &w_, 4);
    }
    if (!keep_q) {
        const int w_ = _mm_cvtsi128_si32(_mm_packus_epi16(nq0, nq0));
        memcpy(base, &w_, 4);
    }
}
#undef LOAD32
#endif

#define BIT_DEPTH 8
#include "hevc_pixel.h"
#include "hevc_filter_template.c"
#undef BIT_DEPTH

#define BIT_DEPTH 10
#include "hevc_pixel.h"
#include "hevc_filter_template.c"
#undef BIT_DEPTH

/* ------------------------------------------- the filters on many threads
 *
 * Deblocking and SAO used to run on one thread after the wavefront had
 * finished with sixteen, and at 4K they were most of what was left: the
 * picture was decoded in parallel and then filtered in series.
 *
 * Four stages, each split by coding tree block rows, with every thread
 * waiting for the others between two stages:
 *
 *   0. the vertical edges      - 8.7.2 filters all of them first,
 *   1. the horizontal edges    - and these read what stage 0 wrote,
 *   2. what SAO reads across   - the deblocked rows and columns at the
 *                                block borders,
 *   3. SAO                     - which reads those around each block.
 *
 * Inside a stage the rows are independent (see one_direction() for why
 * the horizontal edges are), so the rows are simply handed out in order.
 *
 * ⚠️ One thread runs exactly the same stages: the single-threaded suites
 * exercise this code and not some other serial path beside it. */

#define FILTER_MAX_THREAD 16

typedef struct {
    hevcd_t *d;
    bool ten, deblock, sao;
    int rows;
    _Atomic int next[4];            /* the next row of each stage */

    /* Who takes part, and the wait between two stages. ⚠️ `taking_part`
     * is only known once the threads exist - one may fail to start - so
     * every worker waits for it before anything else. */
    pthread_mutex_t m;
    pthread_cond_t c;
    int taking_part, arrived, generation;
} filter_job_t;

static void stage_done(filter_job_t *j)
{
    pthread_mutex_lock(&j->m);
    const int gen = j->generation;
    if (++j->arrived == j->taking_part) {
        j->arrived = 0;
        j->generation++;
        pthread_cond_broadcast(&j->c);
    } else {
        while (gen == j->generation) pthread_cond_wait(&j->c, &j->m);
    }
    pthread_mutex_unlock(&j->m);
}

static void stage_row(filter_job_t *j, int stage, int ry)
{
    hevcd_t *d = j->d;
    switch (stage) {
    case 0: if (j->ten) deblock_row_10(d, true, ry);  else deblock_row_8(d, true, ry);  break;
    case 1: if (j->ten) deblock_row_10(d, false, ry); else deblock_row_8(d, false, ry); break;
    case 2: if (j->ten) sao_copy_row_10(d, ry);       else sao_copy_row_8(d, ry);       break;
    case 3: if (j->ten) sao_row_10(d, ry);            else sao_row_8(d, ry);            break;
    }
}

static void run_stage(filter_job_t *j, int stage)
{
    for (;;) {
        const int ry = atomic_fetch_add_explicit(&j->next[stage], 1,
                                                 memory_order_relaxed);
        if (ry >= j->rows) break;
        stage_row(j, stage, ry);
    }
}

static void *filter_worker(void *arg)
{
    filter_job_t *j = arg;
    pthread_mutex_lock(&j->m);
    while (!j->taking_part) pthread_cond_wait(&j->c, &j->m);
    pthread_mutex_unlock(&j->m);

    if (j->deblock) {
        run_stage(j, 0);
        stage_done(j);
        run_stage(j, 1);
        stage_done(j);
    }
    if (j->sao) {
        run_stage(j, 2);
        stage_done(j);
        run_stage(j, 3);
    }
    return NULL;
}

static int filter_threads(int rows)
{
    const char *s = getenv("BC250_HEVC_THREAD");
    int n = s ? atoi(s) : (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 1) n = 1;
    if (n > FILTER_MAX_THREAD) n = FILTER_MAX_THREAD;
    if (n > rows) n = rows;
    return n;
}

void hevcd_loop_filters(hevcd_t *d)
{
    if (!d->sps || !d->plane[0]) return;

    filter_job_t j;
    memset(&j, 0, sizeof j);
    j.d = d;
    j.ten = d->sps->bit_depth_luma > 8;
    j.rows = d->sps->ctb_height;
    j.deblock = d->edges != NULL;
    /* ⚠️ After deciding about deblocking, not before, and on this thread:
     * it allocates the copy, which the workers then only fill. */
    j.sao = j.ten ? sao_prepare_10(d) : sao_prepare_8(d);
    if (!j.deblock && !j.sao) return;

    pthread_mutex_init(&j.m, NULL);
    pthread_cond_init(&j.c, NULL);

    const int want = filter_threads(j.rows);
    hevcd_pool_t *pool = want > 1 ? hevcd_pool_for(d) : NULL;
    if (pool) {
        /* The pool says beforehand how many it will use, which is what
         * the waits between stages have to know. */
        j.taking_part = hevcd_pool_helpers(pool, want) + 1;
        hevcd_pool_run(pool, filter_worker, &j, want);
    } else {
        pthread_t t[FILTER_MAX_THREAD];
        int alive = 0;
        for (int i = 1; i < want; i++)
            if (pthread_create(&t[alive], NULL, filter_worker, &j) == 0) alive++;

        pthread_mutex_lock(&j.m);
        j.taking_part = alive + 1;
        pthread_cond_broadcast(&j.c);
        pthread_mutex_unlock(&j.m);

        filter_worker(&j);              /* this thread works too */
        for (int i = 0; i < alive; i++) pthread_join(t[i], NULL);
    }

    pthread_mutex_destroy(&j.m);
    pthread_cond_destroy(&j.c);
}

/* ------------------------------------------ the stages inside the wavefront
 *
 * The same four stages, one coding tree block row at a time, for
 * hevc_wpp.c to run while later rows are still being decoded. It decides
 * when each is allowed; see there. */
bool hevcd_filters_prepare(hevcd_t *d, bool *deblock, bool *sao)
{
    const bool ten = d->sps->bit_depth_luma > 8;
    *deblock = d->edges != NULL;
    /* Whether this picture's slice asked for SAO at all. The wavefront
     * only takes a slice that is the whole picture, so it is the only one
     * that could have. */
    *sao = d->slice && d->sps->sao_enabled
           && (d->slice->sao_luma || d->slice->sao_chroma)
           && (ten ? sao_alloc_10(d) : sao_alloc_8(d));
    return *deblock || *sao;
}

void hevcd_filter_stage(hevcd_t *d, int stage, int ry)
{
    const bool ten = d->sps->bit_depth_luma > 8;
    switch (stage) {
    case 0: if (ten) deblock_row_10(d, true, ry);  else deblock_row_8(d, true, ry);  break;
    case 1: if (ten) deblock_row_10(d, false, ry); else deblock_row_8(d, false, ry); break;
    case 2: if (ten) sao_copy_row_10(d, ry);       else sao_copy_row_8(d, ry);       break;
    default: if (ten) sao_row_10(d, ry);           else sao_row_8(d, ry);            break;
    }
}

void hevcd_free_filters(hevcd_t *d)
{
    free(d->sao);
    d->sao = NULL;
    d->n_sao = 0;
    for (int c = 0; c < 3; c++) {
        free(d->sao_lines[c]);
        d->sao_lines[c] = NULL;
        d->n_sao_lines[c] = 0;
    }
}
