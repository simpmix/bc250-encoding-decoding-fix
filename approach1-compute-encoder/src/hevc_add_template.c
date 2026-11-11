/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * hevc_add_template.c - the residual onto the prediction.
 *
 * Included once per bit depth by hevc_transform.c. See hevc_pixel.h.
 */

/* 8.6.5: the residual onto the prediction, clipped back into the
 * picture's own depth. */
static void FUNC(add_residual)(pixel *dst, int stride, const int16_t *res,
                               int log2_size)
{
    const int n = 1 << log2_size;
#if defined(__x86_64__) || defined(_M_X64)
#if BIT_DEPTH == 10
    for (int y = 0; y < n; y++) {
        pixel *d = dst + (size_t)y * stride;
        const int16_t *r = res + (size_t)y * n;
        int x = 0;
        for (; x + 8 <= n; x += 8) {
            __m128i vd = _mm_loadu_si128((const __m128i *)(d + x));
            __m128i vr = _mm_loadu_si128((const __m128i *)(r + x));
            __m128i sum = _mm_add_epi16(vd, vr);
            sum = _mm_min_epi16(_mm_max_epi16(sum, _mm_setzero_si128()), _mm_set1_epi16(1023));
            _mm_storeu_si128((__m128i *)(d + x), sum);
        }
        for (; x < n; x++) {
            const int v = (int)d[x] + (int)r[x];
            d[x] = (pixel)(v < 0 ? 0 : (v > 1023 ? 1023 : v));
        }
    }
    return;
#elif BIT_DEPTH == 8
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
            __m128i s0 = _mm_add_epi16(vd0, vr0);
            __m128i s1 = _mm_add_epi16(vd1, vr1);
            __m128i packed = _mm_packus_epi16(s0, s1);
            _mm_storeu_si128((__m128i *)(d + x), packed);
        }
        for (; x + 8 <= n; x += 8) {
            __m128i vd = _mm_loadl_epi64((const __m128i *)(d + x));
            __m128i vr = _mm_loadu_si128((const __m128i *)(r + x));
            __m128i vd0 = _mm_unpacklo_epi8(vd, _mm_setzero_si128());
            __m128i s0 = _mm_add_epi16(vd0, vr);
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
