/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * hevc_add_template.c - the residual onto the prediction.
 *
 * Included once per bit depth by hevc_transform.c. See hevc_pixel.h.
 */

#if defined(__x86_64__) || defined(_M_X64)
#if BIT_DEPTH == 10
__attribute__((target("avx2")))
static void add_residual_10_avx2(pixel *dst, int stride, const int16_t *res, int n)
{
    const __m256i zero256 = _mm256_setzero_si256();
    const __m256i max_val256 = _mm256_set1_epi16(1023);
    const __m128i zero = _mm_setzero_si128();
    const __m128i max_val = _mm_set1_epi16(1023);
    for (int y = 0; y < n; y++) {
        pixel *d = dst + (size_t)y * stride;
        const int16_t *r = res + (size_t)y * n;
        int x = 0;
        for (; x + 16 <= n; x += 16) {
            __m256i vd = _mm256_loadu_si256((const __m256i *)(d + x));
            __m256i vr = _mm256_loadu_si256((const __m256i *)(r + x));
            __m256i sum = _mm256_adds_epi16(vd, vr);
            sum = _mm256_min_epi16(_mm256_max_epi16(sum, zero256), max_val256);
            _mm256_storeu_si256((__m256i *)(d + x), sum);
        }
        for (; x + 8 <= n; x += 8) {
            __m128i vd = _mm_loadu_si128((const __m128i *)(d + x));
            __m128i vr = _mm_loadu_si128((const __m128i *)(r + x));
            __m128i sum = _mm_adds_epi16(vd, vr);
            sum = _mm_min_epi16(_mm_max_epi16(sum, zero), max_val);
            _mm_storeu_si128((__m128i *)(d + x), sum);
        }
        for (; x < n; x++) {
            const int v = (int)d[x] + (int)r[x];
            d[x] = (pixel)(v < 0 ? 0 : (v > 1023 ? 1023 : v));
        }
    }
}
#elif BIT_DEPTH == 8
__attribute__((target("avx2")))
static void add_residual_8_avx2(pixel *dst, int stride, const int16_t *res, int n)
{
    for (int y = 0; y < n; y++) {
        pixel *d = dst + (size_t)y * stride;
        const int16_t *r = res + (size_t)y * n;
        int x = 0;
        for (; x + 16 <= n; x += 16) {
            __m128i vd = _mm_loadu_si128((const __m128i *)(d + x));
            __m256i vr = _mm256_loadu_si256((const __m256i *)(r + x));
            __m256i vd256 = _mm256_cvtepu8_epi16(vd);
            __m256i s = _mm256_adds_epi16(vd256, vr);
            __m128i lo = _mm256_castsi256_si128(s);
            __m128i hi = _mm256_extracti128_si256(s, 1);
            __m128i packed = _mm_packus_epi16(lo, hi);
            _mm_storeu_si128((__m128i *)(d + x), packed);
        }
        for (; x + 8 <= n; x += 8) {
            __m128i vd = _mm_loadl_epi64((const __m128i *)(d + x));
            __m128i vr = _mm_loadu_si128((const __m128i *)(r + x));
            __m128i vd0 = _mm_unpacklo_epi8(vd, _mm_setzero_si128());
            __m128i s0 = _mm_adds_epi16(vd0, vr);
            __m128i packed = _mm_packus_epi16(s0, s0);
            _mm_storel_epi64((__m128i *)(d + x), packed);
        }
        for (; x < n; x++) {
            const int v = (int)d[x] + (int)r[x];
            d[x] = (pixel)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
    }
}
#endif
#endif

/* 8.6.5: the residual onto the prediction, clipped back into the
 * picture's own depth. */
static void FUNC(add_residual)(pixel *dst, int stride, const int16_t *res,
                               int log2_size)
{
    const int n = 1 << log2_size;
#if defined(__x86_64__) || defined(_M_X64)
#if BIT_DEPTH == 10
    if (n >= 16 && __builtin_cpu_supports("avx2")) {
        add_residual_10_avx2(dst, stride, res, n);
        return;
    }
    const __m128i zero = _mm_setzero_si128();
    const __m128i max_val = _mm_set1_epi16(1023);
    for (int y = 0; y < n; y++) {
        pixel *d = dst + (size_t)y * stride;
        const int16_t *r = res + (size_t)y * n;
        int x = 0;
        for (; x + 8 <= n; x += 8) {
            __m128i vd = _mm_loadu_si128((const __m128i *)(d + x));
            __m128i vr = _mm_loadu_si128((const __m128i *)(r + x));
            __m128i sum = _mm_adds_epi16(vd, vr);
            sum = _mm_min_epi16(_mm_max_epi16(sum, zero), max_val);
            _mm_storeu_si128((__m128i *)(d + x), sum);
        }
        for (; x < n; x++) {
            const int v = (int)d[x] + (int)r[x];
            d[x] = (pixel)(v < 0 ? 0 : (v > 1023 ? 1023 : v));
        }
    }
    return;
#elif BIT_DEPTH == 8
    if (n >= 16 && __builtin_cpu_supports("avx2")) {
        add_residual_8_avx2(dst, stride, res, n);
        return;
    }
    for (int y = 0; y < n; y++) {
        pixel *d = dst + (size_t)y * stride;
        const int16_t *r = res + (size_t)y * n;
        int x = 0;
        for (; x + 16 <= n; x += 16) {
            __m128i vd = _mm_loadu_si128((const __m128i *)(d + x));
            __m128i vr0 = _mm_loadu_si128((const __m128i *)(r + x));
            __m128i vr1 = _mm_loadu_si128((const __m128i *)(r + x + 8));
            __m128i vd0 = _mm_unpacklo_epi8(vd, _mm_setzero_si128());
            __m128i vd1 = _mm_unpackhi_epi8(vd, _mm_setzero_si128());
            __m128i s0 = _mm_adds_epi16(vd0, vr0);
            __m128i s1 = _mm_adds_epi16(vd1, vr1);
            __m128i packed = _mm_packus_epi16(s0, s1);
            _mm_storeu_si128((__m128i *)(d + x), packed);
        }
        for (; x + 8 <= n; x += 8) {
            __m128i vd = _mm_loadl_epi64((const __m128i *)(d + x));
            __m128i vr = _mm_loadu_si128((const __m128i *)(r + x));
            __m128i vd0 = _mm_unpacklo_epi8(vd, _mm_setzero_si128());
            __m128i s0 = _mm_adds_epi16(vd0, vr);
            __m128i packed = _mm_packus_epi16(s0, s0);
            _mm_storel_epi64((__m128i *)(d + x), packed);
        }
        for (; x < n; x++) {
            const int v = (int)d[x] + (int)r[x];
            d[x] = (pixel)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
    }
    return;
#endif
#endif
    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++) {
            const int v = dst[y * stride + x] + res[y * n + x];
            dst[y * stride + x] = (pixel)(v < 0 ? 0 : (v > PIXEL_MAX
                                                       ? PIXEL_MAX : v));
        }
}
