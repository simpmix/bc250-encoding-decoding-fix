/* bc250-encoding-decoding-fix v0.4.3 - https://github.com/simpmix/bc250-encoding-decoding-fix */
/*
 * Copyright (c) 2026 BC-250 Project Contributors
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * hevc_intra.c - see hevc_intra.h for the design rationale. Everything in
 * this file is independently written against the published ITU-T H.265
 * spec text (clauses cited per-function below) - unlike hevc_cabac.c, none
 * of it is adapted from x265's source, though the well-known public 4x4
 * DCT-II integer matrix ({64,64,64,64},{83,36,-36,-83},...) and the
 * standard HEVC dequant scale table ({40,45,51,57,64,72}) were
 * cross-checked against x265's source/common/constants.cpp and
 * source/common/scalinglist.cpp purely to catch transcription mistakes -
 * both are the spec's own normative constants, not x265-original values.
 */
#include "hevc_intra.h"
#include <string.h>
#include <immintrin.h>
#include <stdlib.h>

/* ===================== mode/scan helpers ===================== */

int hevc_scan_idx_for_mode(int mode) {
    if (mode >= 6 && mode <= 14) return 2;  /* SCAN_VER */
    if (mode >= 22 && mode <= 30) return 1; /* SCAN_HOR */
    return 0;                               /* SCAN_DIAG */
}

/* Rec. ITU-T H.265 8.4.2: build the 3-entry candidate mode list from the
 * left/above neighbor PUs' real intra modes. Unavailable neighbors
 * (off-picture, or - not applicable here, one slice per picture - a
 * different slice) are treated as INTRA_DC per the spec's substitution. */
void hevc_derive_mpm(int left_mode, int left_avail, int above_mode, int above_avail,
                      int mpm_out[3]) {
    int cand_a = left_avail ? left_mode : HEVC_MODE_DC;
    int cand_b = above_avail ? above_mode : HEVC_MODE_DC;

    if (cand_a == cand_b) {
        if (cand_a < 2) {
            mpm_out[0] = HEVC_MODE_PLANAR;
            mpm_out[1] = HEVC_MODE_DC;
            mpm_out[2] = HEVC_MODE_VERTICAL;
        } else {
            mpm_out[0] = cand_a;
            mpm_out[1] = 2 + ((cand_a + 29) % 32);
            mpm_out[2] = 2 + ((cand_a - 2 + 1) % 32);
        }
    } else {
        mpm_out[0] = cand_a;
        mpm_out[1] = cand_b;
        if (mpm_out[0] != HEVC_MODE_PLANAR && mpm_out[1] != HEVC_MODE_PLANAR)
            mpm_out[2] = HEVC_MODE_PLANAR;
        else if (mpm_out[0] != HEVC_MODE_DC && mpm_out[1] != HEVC_MODE_DC)
            mpm_out[2] = HEVC_MODE_DC;
        else
            mpm_out[2] = HEVC_MODE_VERTICAL;
    }
}

/* ===================== neighbor gathering (8.4.4.2.2) ===================== */


/*
 * Rec. ITU-T H.265 6.4.1's "z-scan order block availability" is NOT the
 * same thing as "is this position within picture bounds" - a neighbor can
 * be positionally inside the picture and still not yet decoded. This bit
 * this encoder got wrong initially: a CU's bottom-right (z-order index 3)
 * PU's "top-right" reference sample can land inside a SIBLING CU that
 * hasn't been coded yet (e.g. the top-left CU's bottom-right PU reaching
 * into the top-right CU of the same CTU), or inside the next CTU to the
 * right (same picture row, but that CTU is later in raster order) - both
 * pass a naive "x0+4 < width && y0 > 0" check while being genuinely
 * undecoded. Confirmed as the actual remaining bug by an exhaustive
 * bit-for-bit cross-check of this encoder's CABAC output against its own
 * recorded per-CU decisions (which matched perfectly - the bitstream was
 * never wrong), followed by comparing this project's intra prediction/
 * transform code line-by-line against ffmpeg's libavcodec/hevc/
 * pred_template.c and dsp_template.c (both matched exactly) - leaving
 * z-scan availability, which ffmpeg's intra_pred() computes via a real
 * MinTbAddrZs table lookup (cand_up_right &&
 * cur_tb_addr > MIN_TB_ADDR_ZS(...)), as the one remaining candidate, and
 * it was.
 *
 * This encoder's coding order is a FIXED, known shape (CTUs in raster
 * order; each CTU always splits into exactly 4 CUs in z-order - TL,TR,
 * BL,BR; each luma CU always splits into exactly 4 PUs the same way;
 * chroma has no PU sub-split, one block per CU) - which makes an exact
 * z-scan rank computable directly, without needing a real MinTbAddrZs
 * table: rank = ((ctuRow*widthInCtus + ctuCol) * 4 + cuZIndex) * 4 +
 * puZIndex (chroma stops one level early, at the CU). A neighbor is
 * available iff it's in-picture AND its rank is strictly less than the
 * current block's.
 */
/* Shifts, not divisions.
 *
 * Every divisor here is a power of two, but all of them are derived from the
 * is_luma argument, so the compiler cannot prove it and emits real div
 * instructions - six per call, and this is called five times per 4x4 block
 * (once for the block itself, four for its neighbours), which comes to about
 * four million integer divisions per 1080p frame. The profiler put
 * zorder_available() at 5% of the HEVC encode on a BC-250 on its own, before
 * counting what it costs inside gather_neighbors().
 *
 * x and y are never negative at either call site - zorder_available() bounds
 * checks first and the cur_rank call passes the block's own coordinates - so
 * >> and & give exactly what / and % gave. */
static long long zorder_rank(int x, int y, int width, int is_luma) {
    const int sh = is_luma ? 4 : 3;              /* ctu_size = 1 << sh */
    const int ctu_size = 1 << sh;
    const int hs = sh - 1;                       /* CU size  = 1 << hs */
    int width_ctu = (width + ctu_size - 1) >> sh;
    int ctu_col = x >> sh, ctu_row = y >> sh;
    int rx = x & (ctu_size - 1), ry = y & (ctu_size - 1);
    int cu_col = rx >> hs, cu_row = ry >> hs;
    long long rank = ((long long)ctu_row * width_ctu + ctu_col) * 4 + (cu_row * 2 + cu_col);
    if (is_luma) {
        const int qs = hs - 1;                   /* PU size = 1 << qs = 4 */
        int rx2 = rx & ((1 << hs) - 1), ry2 = ry & ((1 << hs) - 1);
        rank = rank * 4 + ((ry2 >> qs) * 2 + (rx2 >> qs));
    }
    return rank;
}

/* y_min is the first pixel row of the current slice: anything above it belongs
 * to another slice and a decoder will not have it, so neither may we. */
static int zorder_available(int nx, int ny, int width, int height, int is_luma,
                            int y_min, long long cur_rank) {
    if (nx < 0 || ny < y_min || nx >= width || ny >= height) return 0;
    return zorder_rank(nx, ny, width, is_luma) < cur_rank;
}

/* The prediction and the mode decision, once per bit depth. See
 * hevc_intra_template.c. */
#define BIT_DEPTH 8
#include "hevc_pixel.h"
#include "hevc_intra_template.c"
#undef BIT_DEPTH

#define BIT_DEPTH 10
#include "hevc_pixel.h"
#include "hevc_intra_template.c"
#undef BIT_DEPTH

/* ===================== transform (8.6.4) ===================== */

/* Public/standard 4x4 integer DCT-II matrix (ITU-T H.265 8.6.4.1's
 * transMatrix for nTbS=4; identical to H.264's and every other MPEG-family
 * codec's 4-point integer DCT approximation). Cross-checked against
 * x265's source/common/constants.cpp g_t4[][] - same standard values. */
static const int16_t DCT4[4][4] = {
    { 64,  64,  64,  64 },
    { 83,  36, -36, -83 },
    { 64, -64, -64,  64 },
    { 36, -83,  83, -36 }
};

/* Public/standard 4x4 DST-VII "alternative transform" matrix, used ONLY
 * for 4x4 luma intra residuals (ITU-T H.265 8.6.4.1: "if cIdx is equal to
 * 0 and predMode is equal to MODE_INTRA and nTbS is equal to 4, the
 * alternative transform... is used"). Cross-checked against x265's
 * primitives.dst4x4 call site (quant.cpp) for when it fires - the matrix
 * values themselves are the spec's own public constant, reproduced in
 * essentially every independent HEVC implementation (HM, libde265, ffmpeg,
 * etc), not x265-original. */
static const int16_t DST4[4][4] = {
    { 29,  55,  74,  84 },
    { 74,  74,   0, -74 },
    { 84, -29, -74,  55 },
    { 55, -84,  74, -29 }
};

static inline int32_t clip_coeff(int32_t v) {
    if (v > 32767) return 32767;
    if (v < -32768) return -32768;
    return v;
}

/* Forward 2D separable transform, matrix M applied directly (not
 * transposed) along both axes. Shift split (1, 8) - chosen so that, paired
 * with the spec-mandated inverse shifts below (7, 12 for 8-bit), a
 * forward-then-inverse round trip with no quantization in between
 * reproduces the original residual exactly for a constant (DC-only) input:
 * forward stage1 shift=1 add=1, stage2 shift=8 add=128 -> for a constant
 * input v, coeff[0][0] = 128*v (verified by hand: row0 of M is {64,64,64,64}
 * so a per-axis DC gain of 4*64=256=2^8; after forward's own /2^1 then
 * /2^8 net divide of 2^9, combined per-axis, the DC coefficient comes out
 * to 128*v); the inverse below then recovers exactly v from that (see its
 * own comment). Non-DC content is NOT expected to be bit-exact through
 * this round trip (that's inherent to any integer DCT/DST approximation,
 * including the real x265/HM ones - see this file's header comment), only
 * well-scaled - forward quantization error is what's supposed to make the
 * picture lossy, not a transform bug. */
/* SSE2 versions of the 4x4 transforms, for both depths - the shifts are
 * parameters.
 *
 * The block is two registers, rows 0-1 and rows 2-3. A pass is a matrix
 * product over the rows: PMADDWD takes rows 0 and 1 interleaved against a
 * pair of matrix entries, rows 2 and 3 against the next pair, and the two
 * 32-bit sums add. Rounding and shift follow, and _mm_packs_epi32
 * saturates to int16 exactly as clip_coeff() does. A transpose between the
 * passes makes the second the same product.
 *
 * The forward transform keeps its intermediate and its output in int16,
 * where the scalar code keeps int32. They fit: a residual of at most 255 at
 * eight bits, first pass shifting by 1, gives at most 4 * 64 * 255 >> 1 =
 * 32640; at ten bits, 1023 and a shift by 3, 32736; and the second pass
 * lands under 32768 too. So the results are the scalar code's - checked
 * over two million pseudo-random blocks, extremes included, when this came
 * in. The SSE4.1 versions this replaces multiplied lane by lane with
 * PMULLD. */
#if defined(__SSE2__)
static inline __m128i pair16(int a0, int a1)
{
    return _mm_set1_epi32((int)(uint16_t)a0 | (int)((uint32_t)(uint16_t)a1 << 16));
}

static inline void mat4_rows(const int16_t A[4][4], int transposed, __m128i x01, __m128i x23,
                             int shift, __m128i *y01, __m128i *y23)
{
    const __m128i p01 = _mm_unpacklo_epi16(x01, _mm_srli_si128(x01, 8));
    const __m128i p23 = _mm_unpacklo_epi16(x23, _mm_srli_si128(x23, 8));
    const __m128i rnd = _mm_set1_epi32(1 << (shift - 1));
    const __m128i sh = _mm_cvtsi32_si128(shift);
    __m128i y[4];
    for (int i = 0; i < 4; i++) {
        const int a0 = transposed ? A[0][i] : A[i][0], a1 = transposed ? A[1][i] : A[i][1];
        const int a2 = transposed ? A[2][i] : A[i][2], a3 = transposed ? A[3][i] : A[i][3];
        const __m128i s = _mm_add_epi32(_mm_madd_epi16(p01, pair16(a0, a1)),
                                        _mm_madd_epi16(p23, pair16(a2, a3)));
        y[i] = _mm_sra_epi32(_mm_add_epi32(s, rnd), sh);
    }
    *y01 = _mm_packs_epi32(y[0], y[1]);
    *y23 = _mm_packs_epi32(y[2], y[3]);
}

static inline void transpose4(__m128i *r01, __m128i *r23)
{
    const __m128i a = _mm_unpacklo_epi16(*r01, _mm_srli_si128(*r01, 8));
    const __m128i b = _mm_unpacklo_epi16(*r23, _mm_srli_si128(*r23, 8));
    *r01 = _mm_unpacklo_epi32(a, b);
    *r23 = _mm_unpackhi_epi32(a, b);
}

/* out[i*4+j] = (sum_c M[j][c] * tmp[i][c] + rnd) >> s2, tmp[i][c] = (sum_r
 * M[i][r] * residual[r*4+c] + rnd) >> s1. */
static void fwd4_sse2(const int16_t residual[16], const int16_t M[4][4], int s1, int s2,
                      int32_t out[16])
{
    __m128i x01 = _mm_loadu_si128((const __m128i *)residual);
    __m128i x23 = _mm_loadu_si128((const __m128i *)(residual + 8));
    __m128i t01, t23;
    mat4_rows(M, 0, x01, x23, s1, &t01, &t23);
    transpose4(&t01, &t23);
    mat4_rows(M, 0, t01, t23, s2, &x01, &x23);
    transpose4(&x01, &x23);
    _mm_storeu_si128((__m128i *)out, _mm_srai_epi32(_mm_unpacklo_epi16(x01, x01), 16));
    _mm_storeu_si128((__m128i *)(out + 4), _mm_srai_epi32(_mm_unpackhi_epi16(x01, x01), 16));
    _mm_storeu_si128((__m128i *)(out + 8), _mm_srai_epi32(_mm_unpacklo_epi16(x23, x23), 16));
    _mm_storeu_si128((__m128i *)(out + 12), _mm_srai_epi32(_mm_unpackhi_epi16(x23, x23), 16));
}

/* 8.6.4.2: the columns with M transposed, (e + 64) >> 7 saturated, then the
 * rows, (g + rnd) >> s2 saturated. */
static void inv4_sse2(const int16_t coeff[16], const int16_t M[4][4], int s2, int16_t out[16])
{
    __m128i x01 = _mm_loadu_si128((const __m128i *)coeff);
    __m128i x23 = _mm_loadu_si128((const __m128i *)(coeff + 8));
    __m128i g01, g23;
    mat4_rows(M, 1, x01, x23, 7, &g01, &g23);
    transpose4(&g01, &g23);
    mat4_rows(M, 1, g01, g23, s2, &x01, &x23);
    transpose4(&x01, &x23);
    _mm_storeu_si128((__m128i *)out, x01);
    _mm_storeu_si128((__m128i *)(out + 8), x23);
}
#endif

#if !defined(__SSE2__)
static void forward_transform_4x4_scalar(const int16_t residual[16], const int16_t M[4][4], int32_t out[16]) {
    int32_t tmp[4][4];
    for (int c = 0; c < 4; c++) {
        for (int i = 0; i < 4; i++) {
            int32_t sum = 0;
            for (int r = 0; r < 4; r++) sum += (int32_t)M[i][r] * residual[r * 4 + c];
            tmp[i][c] = (sum + 1) >> 1;
        }
    }
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int32_t sum = 0;
            for (int c = 0; c < 4; c++) sum += (int32_t)M[j][c] * tmp[i][c];
            out[i * 4 + j] = (sum + 128) >> 8;
        }
    }
}

/* Inverse 2D separable transform, matrix M applied in TRANSPOSED form
 * (M[k][idx], k summed) along both axes, with the spec-mandated shifts for
 * 8-bit content (ITU-T H.265 8.6.4.2): stage1 shift=7/add=64 (fixed,
 * independent of bit depth), stage2 shift = 20-BitDepth = 12/add=2048.
 * This exact process is what a real HEVC decoder performs, and this
 * encoder uses the SAME code for its own reconstruction chaining, so the
 * two are trivially identical by construction. */
static void inverse_transform_4x4_scalar(const int16_t coeff[16], const int16_t M[4][4], int16_t out[16]) {
    int32_t tmp[4][4];
    for (int c = 0; c < 4; c++) {
        for (int r = 0; r < 4; r++) {
            int32_t sum = 0;
            for (int k = 0; k < 4; k++) sum += (int32_t)M[k][r] * coeff[k * 4 + c];
            tmp[r][c] = clip_coeff((sum + 64) >> 7);
        }
    }
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            int32_t sum = 0;
            for (int k = 0; k < 4; k++) sum += (int32_t)M[k][c] * tmp[r][k];
            out[r * 4 + c] = (int16_t)clip_coeff((sum + 2048) >> 12);
        }
    }
}
#endif

/* ===================== quantization (8.6.3) ===================== */

/* levelScale[qp%6] - Rec. ITU-T H.265 Table (8.6.3), the standard HEVC
 * dequant scale, identical to x265's ScalingList::s_invQuantScales. Public
 * spec constant. */
static const int levelScale[6] = { 40, 45, 51, 57, 64, 72 };

/* bdShift = BitDepth(8) + Log2(nTbS=4) - 5 = 5, m = 16 (flat default
 * scaling list, since this encoder's SPS sets scaling_list_enabled_flag=0
 * - ITU-T H.265 7.4.3.2.1's default). Forward quantization is defined here
 * as the algebraic inverse of the (normative) dequant formula below, which
 * guarantees the two are exactly self-consistent regardless of the
 * transform's own internal scale - see hevc_intra.h's top comment. */
#define HEVC_BDSHIFT 5
#define HEVC_FLAT_M  16

/* Levels from forward-transform outputs, for every block size and depth:
 *
 *     level = floor((|v| << shift + offset) / step),  step = 16 * levelScale[q % 6] << (q / 6)
 *
 * with the offset round_q12 twelfths of the step, the sign put back, and
 * the magnitude capped at 32767. The step is levelScale times a power of
 * two, 2^p with p = q / 6 + 4, and floor(floor(x / 2^p) / d) is
 * floor(x / (d * 2^p)); so the division is a shift, then one by d <= 72,
 * done as a multiplication by m = ceil(2^32 / d) and a shift by 32. That
 * is exact while x * (m * d - 2^32) < 2^32: x stays under 2^21 (|v| under
 * 2^16, shifted up by at most 8 and down by at least 4, the offset under
 * the step) and m * d - 2^32 is under d, so it is. Four at a time with SSE2's 32x32 -> 64 multiplication. `n` is a multiple of
 * eight. */
static void quant_levels(const int32_t *v, int n, int shift, int q, int round_q12, int16_t *out)
{
    const int p = q / 6 + 4;
    const uint32_t d = (uint32_t)levelScale[q % 6];
    const uint32_t offset = (uint32_t)(((uint64_t)d << p) * (uint64_t)round_q12 / 12);
    const uint32_t m = (uint32_t)((((uint64_t)1 << 32) + d - 1) / d);
#if defined(__SSE2__)
    const __m128i vm = _mm_set1_epi32((int)m), voff = _mm_set1_epi32((int)offset);
    const __m128i sh_in = _mm_cvtsi32_si128(shift), sh_p = _mm_cvtsi32_si128(p);
    const __m128i odd = _mm_set_epi32(-1, 0, -1, 0);
    for (int i = 0; i < n; i += 8) {
        __m128i lv[2], sg[2];
        for (int h = 0; h < 2; h++) {
            const __m128i x = _mm_loadu_si128((const __m128i *)(v + i + 4 * h));
            const __m128i s = _mm_srai_epi32(x, 31);
            const __m128i mag = _mm_sub_epi32(_mm_xor_si128(x, s), s);
            const __m128i t = _mm_srl_epi32(_mm_add_epi32(_mm_sll_epi32(mag, sh_in), voff), sh_p);
            const __m128i pe = _mm_mul_epu32(t, vm);                       /* lanes 0, 2 */
            const __m128i po = _mm_mul_epu32(_mm_srli_epi64(t, 32), vm);   /* lanes 1, 3 */
            lv[h] = _mm_or_si128(_mm_srli_epi64(pe, 32), _mm_and_si128(po, odd));
            sg[h] = s;
        }
        /* packs caps the (non-negative) levels at 32767; the sign goes back
         * on in sixteen bits, where -32767 is the far end, as in the scalar
         * code. */
        const __m128i l16 = _mm_packs_epi32(lv[0], lv[1]);
        const __m128i s16 = _mm_packs_epi32(sg[0], sg[1]);
        const __m128i r = _mm_sub_epi16(_mm_xor_si128(l16, s16), s16);
        if (n - i >= 8) _mm_storeu_si128((__m128i *)(out + i), r);
        else            _mm_storel_epi64((__m128i *)(out + i), r);
    }
#else
    for (int i = 0; i < n; i++) {
        const int32_t x = v[i];
        const uint32_t mag = (uint32_t)(x < 0 ? -(int64_t)x : x);
        const uint32_t t = ((mag << shift) + offset) >> p;
        uint32_t l = (uint32_t)(((uint64_t)t * m) >> 32);
        if (l > 32767) l = 32767;
        out[i] = (int16_t)(x < 0 ? -(int32_t)l : (int32_t)l);
    }
#endif
}

/* Levels back to scaled coefficients, 8.6.3 with the flat list:
 *
 *     d = clip16((level * 16 * levelScale[q % 6] << (q / 6) + (1 << (bd_shift - 1))) >> bd_shift)
 *
 * With a = q / 6 + 4 the scale is levelScale times 2^a. When a >= bd_shift
 * the rounding term falls below the shift and d is level * levelScale <<
 * (a - bd_shift); otherwise both terms divide by 2^a and d is (level *
 * levelScale + (1 << (bd_shift - a - 1))) >> (bd_shift - a). Either way it
 * fits 32 bits - level * levelScale is under 2^22 and the left shift is at
 * most 9 - so SSE2 does eight at a time, and packs saturates as clip16
 * does. `n` is a multiple of eight. */
static void dequant_levels(const int16_t *c, int n, int q, int bd_shift, int16_t *out)
{
    const int a = q / 6 + 4;
    const int ls = levelScale[q % 6];
#if defined(__SSE2__)
    const __m128i vls = _mm_set1_epi16((int16_t)ls);
    const int up = a >= bd_shift;
    const __m128i sh = _mm_cvtsi32_si128(up ? a - bd_shift : bd_shift - a);
    const __m128i rnd = _mm_set1_epi32(up ? 0 : 1 << (bd_shift - a - 1));
    for (int i = 0; i < n; i += 8) {
        const __m128i x = _mm_loadu_si128((const __m128i *)(c + i));
        const __m128i lo = _mm_mullo_epi16(x, vls), hi = _mm_mulhi_epi16(x, vls);
        __m128i p0 = _mm_unpacklo_epi16(lo, hi), p1 = _mm_unpackhi_epi16(lo, hi);
        if (up) {
            p0 = _mm_sll_epi32(p0, sh);
            p1 = _mm_sll_epi32(p1, sh);
        } else {
            p0 = _mm_sra_epi32(_mm_add_epi32(p0, rnd), sh);
            p1 = _mm_sra_epi32(_mm_add_epi32(p1, rnd), sh);
        }
        _mm_storeu_si128((__m128i *)(out + i), _mm_packs_epi32(p0, p1));
    }
#else
    const int64_t scale = (int64_t)ls << a;
    for (int i = 0; i < n; i++) {
        const int64_t v = ((int64_t)c[i] * scale + (1 << (bd_shift - 1))) >> bd_shift;
        out[i] = (int16_t)(v > 32767 ? 32767 : (v < -32768 ? -32768 : v));
    }
#endif
}

static inline void forward_transform_4x4(const int16_t residual[16], const int16_t M[4][4],
                                         int32_t out[16]) {
#if defined(__SSE2__)
    fwd4_sse2(residual, M, 1, 8, out);
#else
    forward_transform_4x4_scalar(residual, M, out);
#endif
}

static inline void inverse_transform_4x4(const int16_t coeff[16], const int16_t M[4][4],
                                         int16_t out[16]) {
#if defined(__SSE2__)
    inv4_sse2(coeff, M, 12, out);
#else
    inverse_transform_4x4_scalar(coeff, M, out);
#endif
}

int hevc_chroma_qp_from_luma(int qp_luma) {
    /* qPiCb = Clip3(-QpBdOffsetC, 57, QpY + pps_cb_qp_offset +
     * slice_cb_qp_offset). Both PPS chroma offsets are written as 0 and
     * pps_slice_chroma_qp_offsets_present_flag is 0 (see write_pps()), and
     * QpBdOffsetC is 0 at 8-bit, so qPi is just QpY clamped - and Cb and Cr
     * therefore share one value. */
    static const int qpc_30_43[14] = { 29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37 };
    int qpi = qp_luma < 0 ? 0 : (qp_luma > 57 ? 57 : qp_luma);
    if (qpi < 30) return qpi;
    if (qpi > 43) return qpi - 6;
    return qpc_30_43[qpi - 30];
}

void hevc_transform_quant_4x4(const int16_t residual[16], int qp, int use_dst, int round_q12,
                               int16_t coeff_out[16]) {
    /* Fast zero-residual bypass: if residual is all zero, output is all zero */
    const uint64_t *r64 = (const uint64_t *)residual;
    if ((r64[0] | r64[1] | r64[2] | r64[3]) == 0ULL) {
        memset(coeff_out, 0, 16 * sizeof(int16_t));
        return;
    }

    int32_t raw[16];
    forward_transform_4x4(residual, use_dst ? DST4 : DCT4, raw);

    const int clamped_qp = qp < 0 ? 0 : (qp > 51 ? 51 : qp);
    quant_levels(raw, 16, HEVC_BDSHIFT, clamped_qp, round_q12, coeff_out);
}

void hevc_dequant_itransform_4x4(const int16_t coeff[16], int qp, int use_dst,
                                  int16_t residual_out[16]) {
    /* Fast zero-coeff bypass: if quantized coefficients are all zero, residual is all zero */
    const uint64_t *c64 = (const uint64_t *)coeff;
    if ((c64[0] | c64[1] | c64[2] | c64[3]) == 0ULL) {
        memset(residual_out, 0, 16 * sizeof(int16_t));
        return;
    }

    int16_t dq[16];
    dequant_levels(coeff, 16, qp, HEVC_BDSHIFT, dq);
    inverse_transform_4x4(dq, use_dst ? DST4 : DCT4, residual_out);
}

/* ===================== ten bits ===================== */

/* The same transforms and quantizer at ten bits.
 *
 * Three numbers move with the bit depth, and nothing else does:
 *
 *   - the inverse transform's second stage shifts by 20 - BitDepth, so 10
 *     instead of 12 (8.6.4.2);
 *   - the dequantizer's bdShift is BitDepth + Log2(nTbS) - 5, so 7 instead
 *     of 5 (8.6.2 and 8.6.3);
 *   - the QP the scaling uses is Qp'Y = QpY + QpBdOffsetY, twelve more at
 *     ten bits, and the same for chroma. The caller adds that: `qp` here is
 *     the primed value, 0..63.
 *
 * The forward transform is ours to choose, and its first stage shifts by two
 * more than at eight bits so that a ten-bit residual, four times the
 * eight-bit one, comes out of it at the same scale. With the dequantizer's
 * two extra bits and QpBdOffset's factor of four, a given QP then produces
 * the same levels at either depth - which is what makes the rate control and
 * the QP the caller asked for mean the same thing at ten bits.
 *
 * Plain C: the eight-bit vector code has its shifts built in, and this path
 * has to be right before it is fast.
 */
#define HEVC_BDSHIFT_10 7

#if !defined(__SSE2__)
static void forward_transform_4x4_10(const int16_t residual[16], const int16_t M[4][4],
                                     int32_t out[16]) {
    int32_t tmp[4][4];
    for (int c = 0; c < 4; c++) {
        for (int i = 0; i < 4; i++) {
            int32_t sum = 0;
            for (int r = 0; r < 4; r++) sum += (int32_t)M[i][r] * residual[r * 4 + c];
            tmp[i][c] = (sum + 4) >> 3;
        }
    }
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int32_t sum = 0;
            for (int c = 0; c < 4; c++) sum += (int32_t)M[j][c] * tmp[i][c];
            out[i * 4 + j] = (sum + 128) >> 8;
        }
    }
}

static void inverse_transform_4x4_10(const int16_t coeff[16], const int16_t M[4][4],
                                     int16_t out[16]) {
    int32_t tmp[4][4];
    for (int c = 0; c < 4; c++) {
        for (int r = 0; r < 4; r++) {
            int32_t sum = 0;
            for (int k = 0; k < 4; k++) sum += (int32_t)M[k][r] * coeff[k * 4 + c];
            tmp[r][c] = clip_coeff((sum + 64) >> 7);
        }
    }
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            int32_t sum = 0;
            for (int k = 0; k < 4; k++) sum += (int32_t)M[k][c] * tmp[r][k];
            out[r * 4 + c] = (int16_t)clip_coeff((sum + 512) >> 10);
        }
    }
}

#endif

void hevc_transform_quant_4x4_10(const int16_t residual[16], int qp, int use_dst, int round_q12,
                                  int16_t coeff_out[16]) {
    const uint64_t *r64 = (const uint64_t *)residual;
    if ((r64[0] | r64[1] | r64[2] | r64[3]) == 0ULL) {
        memset(coeff_out, 0, 16 * sizeof(int16_t));
        return;
    }

    int32_t raw[16];
#if defined(__SSE2__)
    fwd4_sse2(residual, use_dst ? DST4 : DCT4, 3, 8, raw);
#else
    forward_transform_4x4_10(residual, use_dst ? DST4 : DCT4, raw);
#endif

    /* The algebraic inverse of the dequantizer below, with the caller's
     * rounding. */
    const int q = qp < 0 ? 0 : (qp > 63 ? 63 : qp);
    quant_levels(raw, 16, HEVC_BDSHIFT_10, q, round_q12, coeff_out);
}

void hevc_dequant_itransform_4x4_10(const int16_t coeff[16], int qp, int use_dst,
                                     int16_t residual_out[16]) {
    const uint64_t *c64 = (const uint64_t *)coeff;
    if ((c64[0] | c64[1] | c64[2] | c64[3]) == 0ULL) {
        memset(residual_out, 0, 16 * sizeof(int16_t));
        return;
    }

    const int q = qp < 0 ? 0 : (qp > 63 ? 63 : qp);
    int16_t dq[16];
    dequant_levels(coeff, 16, q, HEVC_BDSHIFT_10, dq);
#if defined(__SSE2__)
    inv4_sse2(dq, use_dst ? DST4 : DCT4, 10, residual_out);
#else
    inverse_transform_4x4_10(dq, use_dst ? DST4 : DCT4, residual_out);
#endif
}

/* ===================== 8x8 (inter luma) ===================== */

/* 8.6.4.2's transMatrix for nTbS = 8: rows 0, 4, 8 ... 28 of the 32-point
 * matrix. */
static const int16_t DCT8[8][8] = {
    { 64,  64,  64,  64,  64,  64,  64,  64 },
    { 89,  75,  50,  18, -18, -50, -75, -89 },
    { 83,  36, -36, -83, -83, -36,  36,  83 },
    { 75, -18, -89, -50,  50,  89,  18, -75 },
    { 64, -64, -64,  64,  64, -64, -64,  64 },
    { 50, -89,  18,  75, -75, -18,  89, -50 },
    { 36, -83,  83, -36, -36,  83, -83,  36 },
    { 18, -50,  75, -89,  89, -75,  50, -18 },
};

/* One 8-point DCT, forward and inverse, by even and odd halves - the same
 * integer products as the matrix, so the same results, with about half of
 * the multiplications. `s` is the stride between the eight inputs. */
static inline void dct8_fwd(const int32_t *x, int s, int32_t y[8])
{
    int32_t e[4], o[4];
    for (int k = 0; k < 4; k++) {
        e[k] = x[k * s] + x[(7 - k) * s];
        o[k] = x[k * s] - x[(7 - k) * s];
    }
    const int32_t ee0 = e[0] + e[3], eo0 = e[0] - e[3];
    const int32_t ee1 = e[1] + e[2], eo1 = e[1] - e[2];
    y[0] = 64 * ee0 + 64 * ee1;
    y[4] = 64 * ee0 - 64 * ee1;
    y[2] = 83 * eo0 + 36 * eo1;
    y[6] = 36 * eo0 - 83 * eo1;
    y[1] = 89 * o[0] + 75 * o[1] + 50 * o[2] + 18 * o[3];
    y[3] = 75 * o[0] - 18 * o[1] - 89 * o[2] - 50 * o[3];
    y[5] = 50 * o[0] - 89 * o[1] + 18 * o[2] + 75 * o[3];
    y[7] = 18 * o[0] - 50 * o[1] + 75 * o[2] - 89 * o[3];
}

static inline void dct8_inv(const int32_t *x, int s, int32_t y[8])
{
    int32_t o[4];
    for (int k = 0; k < 4; k++)
        o[k] = DCT8[1][k] * x[1 * s] + DCT8[3][k] * x[3 * s]
             + DCT8[5][k] * x[5 * s] + DCT8[7][k] * x[7 * s];
    const int32_t eo0 = 83 * x[2 * s] + 36 * x[6 * s];
    const int32_t eo1 = 36 * x[2 * s] - 83 * x[6 * s];
    const int32_t ee0 = 64 * x[0] + 64 * x[4 * s];
    const int32_t ee1 = 64 * x[0] - 64 * x[4 * s];
    const int32_t e[4] = { ee0 + eo0, ee1 + eo1, ee1 - eo1, ee0 - eo0 };
    for (int k = 0; k < 4; k++) {
        y[k] = e[k] + o[k];
        y[k + 4] = e[3 - k] - o[3 - k];
    }
}

/* The same two transforms with SSE2, eight int16 per register.
 *
 * One pass is a matrix product over the rows of the block: Y[i] = sum_r
 * A[i][r] * X[r], each X[r] a row of eight samples. _mm_madd_epi16 takes
 * two rows at a time, interleaved, against a pair of matrix entries, and
 * leaves 32-bit sums; the rounding and the shift follow, and _mm_packs_epi32
 * saturates to int16 exactly as clip_coeff() does. A transpose between the
 * passes turns the second one into the same product.
 *
 * The forward transform keeps its intermediate in int16 here, where the
 * scalar code keeps int32. It fits by construction - HM's shifts are chosen
 * for that: a residual of at most 255 at eight bits, or 1023 at ten, gives
 * at most 64 * 8 * 255 >> 2 = 32640, or 64 * 8 * 1023 >> 4 = 32736, after
 * the first pass, and the second pass lands under 32768 too. So both give
 * the same numbers, and the check in the commit that brought this in ran
 * two million blocks through both. */
#if defined(__SSE2__)
static inline void mat8_rows(const int16_t A[8][8], int transposed, const __m128i X[8], int shift,
                             __m128i Y[8])
{
    __m128i lo[4], hi[4];
    for (int k = 0; k < 4; k++) {
        lo[k] = _mm_unpacklo_epi16(X[2 * k], X[2 * k + 1]);
        hi[k] = _mm_unpackhi_epi16(X[2 * k], X[2 * k + 1]);
    }
    const __m128i rnd = _mm_set1_epi32(1 << (shift - 1));
    const __m128i sh = _mm_cvtsi32_si128(shift);
    for (int i = 0; i < 8; i++) {
        __m128i al = rnd, ah = rnd;
        for (int k = 0; k < 4; k++) {
            const int a0 = transposed ? A[2 * k][i] : A[i][2 * k];
            const int a1 = transposed ? A[2 * k + 1][i] : A[i][2 * k + 1];
            const __m128i c = _mm_set1_epi32((int)(uint16_t)a0 | (int)((uint32_t)(uint16_t)a1 << 16));
            al = _mm_add_epi32(al, _mm_madd_epi16(lo[k], c));
            ah = _mm_add_epi32(ah, _mm_madd_epi16(hi[k], c));
        }
        Y[i] = _mm_packs_epi32(_mm_sra_epi32(al, sh), _mm_sra_epi32(ah, sh));
    }
}

static inline void transpose8(__m128i r[8])
{
    const __m128i a0 = _mm_unpacklo_epi16(r[0], r[1]), a1 = _mm_unpackhi_epi16(r[0], r[1]);
    const __m128i a2 = _mm_unpacklo_epi16(r[2], r[3]), a3 = _mm_unpackhi_epi16(r[2], r[3]);
    const __m128i a4 = _mm_unpacklo_epi16(r[4], r[5]), a5 = _mm_unpackhi_epi16(r[4], r[5]);
    const __m128i a6 = _mm_unpacklo_epi16(r[6], r[7]), a7 = _mm_unpackhi_epi16(r[6], r[7]);
    const __m128i b0 = _mm_unpacklo_epi32(a0, a2), b1 = _mm_unpackhi_epi32(a0, a2);
    const __m128i b2 = _mm_unpacklo_epi32(a1, a3), b3 = _mm_unpackhi_epi32(a1, a3);
    const __m128i b4 = _mm_unpacklo_epi32(a4, a6), b5 = _mm_unpackhi_epi32(a4, a6);
    const __m128i b6 = _mm_unpacklo_epi32(a5, a7), b7 = _mm_unpackhi_epi32(a5, a7);
    r[0] = _mm_unpacklo_epi64(b0, b4); r[1] = _mm_unpackhi_epi64(b0, b4);
    r[2] = _mm_unpacklo_epi64(b1, b5); r[3] = _mm_unpackhi_epi64(b1, b5);
    r[4] = _mm_unpacklo_epi64(b2, b6); r[5] = _mm_unpackhi_epi64(b2, b6);
    r[6] = _mm_unpacklo_epi64(b3, b7); r[7] = _mm_unpackhi_epi64(b3, b7);
}
#endif

/* The forward transform is the encoder's to choose. Its shifts are HM's
 * for 8x8 - log2(8) + BitDepth - 9 after the columns, 9 after the rows -
 * so that the quantizer below, the exact inverse of the dequantizer,
 * gives the same levels a residual would get at 4x4. `qp` is Qp', the
 * QP plus QpBdOffset. */
void hevc_transform_quant_8x8(const int16_t residual[64], int qp, int bit_depth, int round_q12,
                               int16_t coeff_out[64]) {
    int any = 0;
    for (int i = 0; i < 64; i++) any |= residual[i];
    if (!any) {
        memset(coeff_out, 0, 64 * sizeof(int16_t));
        return;
    }
    const int s1 = 3 + bit_depth - 9, s2 = 9;
    const int q = qp < 0 ? 0 : (qp > 63 ? 63 : qp);
    int32_t v[64];
#if defined(__SSE2__)
    __m128i x[8], t[8];
    for (int r = 0; r < 8; r++) x[r] = _mm_loadu_si128((const __m128i *)(residual + 8 * r));
    mat8_rows(DCT8, 0, x, s1, t);   /* t[i][c]: column c, frequency i */
    transpose8(t);
    mat8_rows(DCT8, 0, t, s2, x);   /* x[j][i]: coefficient (i, j) */
    transpose8(x);
    for (int i = 0; i < 8; i++) {
        _mm_storeu_si128((__m128i *)(v + 8 * i), _mm_srai_epi32(_mm_unpacklo_epi16(x[i], x[i]), 16));
        _mm_storeu_si128((__m128i *)(v + 8 * i + 4), _mm_srai_epi32(_mm_unpackhi_epi16(x[i], x[i]), 16));
    }
#else
    int32_t tmp[8][8], in[64], col[8];
    for (int i = 0; i < 64; i++) in[i] = residual[i];
    for (int c = 0; c < 8; c++) {
        dct8_fwd(in + c, 8, col);
        for (int i = 0; i < 8; i++) tmp[i][c] = (col[i] + (1 << (s1 - 1))) >> s1;
    }
    for (int i = 0; i < 8; i++) {
        int32_t row[8];
        dct8_fwd(tmp[i], 1, row);
        for (int j = 0; j < 8; j++) v[i * 8 + j] = (row[j] + (1 << (s2 - 1))) >> s2;
    }
#endif
    quant_levels(v, 64, bit_depth + 3 - 5, q, round_q12, coeff_out);
}

/* The decoder's side, exactly: 8.6.2 scaling with the flat list, then
 * 8.6.4.2 - columns, (e + 64) >> 7 clipped to sixteen bits, rows,
 * (g + rnd) >> (20 - BitDepth). */
void hevc_dequant_itransform_8x8(const int16_t coeff[64], int qp, int bit_depth,
                                  int16_t residual_out[64]) {
    int any = 0;
    for (int i = 0; i < 64; i++) any |= coeff[i];
    if (!any) {
        memset(residual_out, 0, 64 * sizeof(int16_t));
        return;
    }
    const int q = qp < 0 ? 0 : (qp > 63 ? 63 : qp);
    int16_t d[64];
    dequant_levels(coeff, 64, q, bit_depth + 3 - 5, d);
    const int s2 = 20 - bit_depth;
#if defined(__SSE2__)
    __m128i x[8], g[8];
    for (int k = 0; k < 8; k++) x[k] = _mm_loadu_si128((const __m128i *)(d + 8 * k));
    mat8_rows(DCT8, 1, x, 7, g);    /* g[r][c], clipped as 8.6.4.2 wants */
    transpose8(g);
    mat8_rows(DCT8, 1, g, s2, x);   /* x[c][r] */
    transpose8(x);
    for (int r = 0; r < 8; r++) _mm_storeu_si128((__m128i *)(residual_out + 8 * r), x[r]);
#else
    int32_t din[64], g[8][8], col[8];
    for (int i = 0; i < 64; i++) din[i] = d[i];
    for (int c = 0; c < 8; c++) {
        dct8_inv(din + c, 8, col);
        for (int r = 0; r < 8; r++) g[r][c] = clip_coeff((col[r] + 64) >> 7);
    }
    for (int r = 0; r < 8; r++) {
        int32_t row[8];
        dct8_inv(g[r], 1, row);
        for (int c = 0; c < 8; c++)
            residual_out[r * 8 + c] = (int16_t)clip_coeff((row[c] + (1 << (s2 - 1))) >> s2);
    }
#endif
}
