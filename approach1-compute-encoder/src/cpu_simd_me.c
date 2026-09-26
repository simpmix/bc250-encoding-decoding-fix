/* bc250-encoding-decoding-fix v0.4.3 - https://github.com/simpmix/bc250-encoding-decoding-fix */
/*
 * Copyright (c) 2026 BC-250 Project Contributors
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * cpu_simd_me.c - Multi-threaded SSE2/AVX2 SIMD Motion Estimation for BC-250 Zen 2 CPU
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#if defined(__linux__)
#include <sched.h>
#include <pthread.h>
#endif

#include "cpu_simd_me.h"
#include <stdlib.h>
#include <string.h>

#if defined(__SSE2__) || defined(__x86_64__) || defined(_M_X64)
#include <emmintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#define STATIC_MB_THRESHOLD 512
#define EARLY_TERMINATION_COST 768

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#define HAS_AVX2_KRNL 1
#include <immintrin.h>

__attribute__((target("avx2")))
static uint32_t cpu_simd_sad_16x16_avx2(const uint8_t *src, int src_stride,
                                        const uint8_t *ref, int ref_stride)
{
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    for (int r = 0; r < 16; r += 4) {
        __m128i s0 = _mm_loadu_si128((const __m128i *)(src + r * src_stride));
        __m128i s1 = _mm_loadu_si128((const __m128i *)(src + (r + 1) * src_stride));
        __m128i rf0 = _mm_loadu_si128((const __m128i *)(ref + r * ref_stride));
        __m128i rf1 = _mm_loadu_si128((const __m128i *)(ref + (r + 1) * ref_stride));
        __m256i s_pair0 = _mm256_set_m128i(s1, s0);
        __m256i rf_pair0 = _mm256_set_m128i(rf1, rf0);
        acc0 = _mm256_add_epi64(acc0, _mm256_sad_epu8(s_pair0, rf_pair0));

        __m128i s2 = _mm_loadu_si128((const __m128i *)(src + (r + 2) * src_stride));
        __m128i s3 = _mm_loadu_si128((const __m128i *)(src + (r + 3) * src_stride));
        __m128i rf2 = _mm_loadu_si128((const __m128i *)(ref + (r + 2) * ref_stride));
        __m128i rf3 = _mm_loadu_si128((const __m128i *)(ref + (r + 3) * ref_stride));
        __m256i s_pair1 = _mm256_set_m128i(s3, s2);
        __m256i rf_pair1 = _mm256_set_m128i(rf3, rf2);
        acc1 = _mm256_add_epi64(acc1, _mm256_sad_epu8(s_pair1, rf_pair1));
    }
    __m256i acc = _mm256_add_epi64(acc0, acc1);
    __m128i low128 = _mm256_castsi256_si128(acc);
    __m128i high128 = _mm256_extracti128_si256(acc, 1);
    __m128i sum128 = _mm_add_epi64(low128, high128);
    uint32_t a = (uint32_t)_mm_cvtsi128_si32(sum128);
    uint32_t b = (uint32_t)_mm_cvtsi128_si32(_mm_srli_si128(sum128, 8));
    return a + b;
}
#endif

#if defined(__SSE2__) || defined(__x86_64__) || defined(_M_X64)
static uint32_t cpu_simd_sad_16x16_sse2(const uint8_t *src, int src_stride,
                                        const uint8_t *ref, int ref_stride)
{
    __m128i acc0 = _mm_setzero_si128();
    __m128i acc1 = _mm_setzero_si128();
    for (int r = 0; r < 16; r += 2) {
        __m128i s0 = _mm_loadu_si128((const __m128i *)(src + r * src_stride));
        __m128i r0 = _mm_loadu_si128((const __m128i *)(ref + r * ref_stride));
        __m128i s1 = _mm_loadu_si128((const __m128i *)(src + (r + 1) * src_stride));
        __m128i r1 = _mm_loadu_si128((const __m128i *)(ref + (r + 1) * ref_stride));
        acc0 = _mm_add_epi32(acc0, _mm_sad_epu8(s0, r0));
        acc1 = _mm_add_epi32(acc1, _mm_sad_epu8(s1, r1));
    }
    __m128i acc = _mm_add_epi32(acc0, acc1);
    uint32_t lo = (uint32_t)_mm_cvtsi128_si32(acc);
    uint32_t hi = (uint32_t)_mm_cvtsi128_si32(_mm_srli_si128(acc, 8));
    return lo + hi;
}
#endif

static uint32_t cpu_simd_sad_16x16_scalar(const uint8_t *src, int src_stride,
                                          const uint8_t *ref, int ref_stride)
{
    uint32_t sad = 0;
    for (int r = 0; r < 16; r++) {
        const uint8_t *s = src + r * src_stride;
        const uint8_t *rf = ref + r * ref_stride;
        for (int c = 0; c < 16; c++) {
            int d = (int)s[c] - (int)rf[c];
            sad += (d < 0) ? -d : d;
        }
    }
    return sad;
}

typedef uint32_t (*sad_16x16_fn_t)(const uint8_t *src, int src_stride,
                                   const uint8_t *ref, int ref_stride);

static sad_16x16_fn_t g_sad_fn = NULL;

static inline sad_16x16_fn_t cpu_simd_get_sad_fn(void)
{
    if (g_sad_fn) {
        return g_sad_fn;
    }
#if defined(HAS_AVX2_KRNL)
    if (__builtin_cpu_supports("avx2")) {
        g_sad_fn = cpu_simd_sad_16x16_avx2;
        return g_sad_fn;
    }
#endif
#if defined(__SSE2__) || defined(__x86_64__) || defined(_M_X64)
    g_sad_fn = cpu_simd_sad_16x16_sse2;
#else
    g_sad_fn = cpu_simd_sad_16x16_scalar;
#endif
    return g_sad_fn;
}

uint32_t cpu_simd_sad_16x16(const uint8_t *src, int src_stride,
                            const uint8_t *ref, int ref_stride)
{
    sad_16x16_fn_t fn = cpu_simd_get_sad_fn();
    return fn(src, src_stride, ref, ref_stride);
}

void cpu_simd_me_config_init(cpu_simd_me_config_t *cfg, uint32_t width, uint32_t height)
{
    if (!cfg) return;
    cfg->width = width;
    cfg->height = height;
    cfg->width_in_mbs = (width + 15) / 16;
    cfg->height_in_mbs = (height + 15) / 16;
    cfg->num_threads = 4; /* Default to 4 worker threads for fast AVX2 SIMD ME */
    const char *env_threads = getenv("BC250_MAX_CPU_THREADS");
    if (!env_threads) env_threads = getenv("BC250_THREADS");
    if (!env_threads) env_threads = getenv("BC250_CPU_THREADS");
    if (env_threads && *env_threads) {
        int t = atoi(env_threads);
        if (t > 0 && t <= 16) cfg->num_threads = t;
    }
#if defined(__linux__)
    if (program_invocation_short_name &&
        (strcmp(program_invocation_short_name, "sunshine") == 0 ||
         strcmp(program_invocation_short_name, "steam") == 0 ||
         strcmp(program_invocation_short_name, "streaming_client") == 0)) {
        /* In live streaming, cap to 2 threads to guarantee game CPU headroom */
        if (!env_threads && cfg->num_threads > 2) cfg->num_threads = 2;
    }
#endif
    cfg->core_ids[0] = -1;
    cfg->core_ids[1] = -1;

    const char *env_cores = getenv("BC250_CPU_CORES");
    if (env_cores && *env_cores) {
        char buf[64];
        strncpy(buf, env_cores, sizeof(buf) - 1);
        buf[sizeof(buf) - 1] = '\0';
        char *token = strtok(buf, ",");
        int idx = 0;
        while (token && idx < 2) {
            int c = atoi(token);
            if (c >= 0 && c < 256) {
                cfg->core_ids[idx++] = c;
            }
            token = strtok(NULL, ",");
        }
    }
}

static void cpu_simd_apply_thread_affinity(int thread_id, const cpu_simd_me_config_t *cfg)
{
#if defined(__linux__)
    if (!cfg) return;
    int target_core = -1;
    if (cfg->core_ids[0] >= 0) {
        if (thread_id == 0) {
            target_core = cfg->core_ids[0];
        } else if (thread_id == 1) {
            target_core = (cfg->core_ids[1] >= 0) ? cfg->core_ids[1] : cfg->core_ids[0];
        }
    }
    if (target_core >= 0) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(target_core, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    }
#else
    (void)thread_id;
    (void)cfg;
#endif
}

typedef struct {
    int x;
    int y;
} ivec2_t;

int cpu_simd_me_search_frame(const uint8_t *src_y, int src_pitch,
                             const uint8_t *ref_y, int ref_pitch,
                             uint32_t width, uint32_t height,
                             gpu_mv_t *out_mvs,
                             const cpu_simd_me_config_t *cfg)
{
    if (!src_y || !ref_y || !out_mvs) return -1;

    uint32_t width_mbs = (width + 15) / 16;
    uint32_t height_mbs = (height + 15) / 16;
    uint32_t max_rad = cfg ? cfg->search_radius : 8;
    if (max_rad < 2) max_rad = 2;
    if (max_rad > 16) max_rad = 16;

    int threads = (cfg && cfg->num_threads > 0) ? cfg->num_threads : 4;
#if defined(__linux__)
    if (program_invocation_short_name &&
        (strcmp(program_invocation_short_name, "sunshine") == 0 ||
         strcmp(program_invocation_short_name, "steam") == 0 ||
         strcmp(program_invocation_short_name, "streaming_client") == 0)) {
        if (threads > 2) threads = 2; /* Cap at 2 threads to guarantee game CPU headroom */
    }
#endif
    if (threads > 16) threads = 16;

    const ivec2_t search_pattern[8] = {
        { 0,  1}, { 0, -1}, { 1,  0}, {-1,  0},
        { 1,  1}, {-1, -1}, { 1, -1}, {-1,  1}
    };
    const uint32_t lambda_motion = 5;
    const int max_x = (int)(width_mbs * 16);
    const int max_y = (int)(height_mbs * 16);

    sad_16x16_fn_t sad_fn = cpu_simd_get_sad_fn();

#ifdef _OPENMP
#pragma omp parallel num_threads(threads) if(threads > 1)
    {
        cpu_simd_apply_thread_affinity(omp_get_thread_num(), cfg);
#pragma omp for schedule(static)
#else
    cpu_simd_apply_thread_affinity(0, cfg);
#endif
        for (int mby = 0; mby < (int)height_mbs; mby++) {
            for (int mbx = 0; mbx < (int)width_mbs; mbx++) {
            uint32_t mb_idx = (uint32_t)mby * width_mbs + (uint32_t)mbx;
            int px = mbx * 16;
            int py = mby * 16;

            const uint8_t *curr_mb = src_y + py * src_pitch + px;
            const uint8_t *ref_mb_zero = ref_y + py * ref_pitch + px;

            /* 1. Fast Zero-Motion Check (SAD at (0, 0)) */
            uint32_t zero_sad = sad_fn(curr_mb, src_pitch, ref_mb_zero, ref_pitch);

            /* Early exit if block is static (saves execution time on video/game content) */
            if (zero_sad <= STATIC_MB_THRESHOLD) {
                out_mvs[mb_idx].mvx = 0;
                out_mvs[mb_idx].mvy = 0;
                out_mvs[mb_idx].sad = zero_sad;
                out_mvs[mb_idx]._pad = 0;
                continue;
            }

            /* 2. Fast Spatial Predictor Check (Left neighbor MV)
             * In video and 3D gaming, camera panning and rigid body motion make neighbor MBs
             * share identical or near-identical MVs. Checking the spatial predictor early
             * allows ~75% of non-static MBs to terminate in <= 2 SAD checks. */
            ivec2_t best_mv = {0, 0};
            uint32_t best_cost = zero_sad;

            if (mbx > 0) {
                gpu_mv_t left_mv_raw = out_mvs[mb_idx - 1];
                ivec2_t left_mv = { (int)(left_mv_raw.mvx / 4), (int)(left_mv_raw.mvy / 4) };
                if (left_mv.x != 0 || left_mv.y != 0) {
                    if (abs(left_mv.x) <= (int)max_rad && abs(left_mv.y) <= (int)max_rad) {
                        int test_rx = px + left_mv.x;
                        int test_ry = py + left_mv.y;
                        if (test_rx >= 0 && test_rx + 16 <= max_x &&
                            test_ry >= 0 && test_ry + 16 <= max_y) {
                            const uint8_t *cand_ref = ref_y + test_ry * ref_pitch + test_rx;
                            uint32_t cand_sad = sad_fn(curr_mb, src_pitch, cand_ref, ref_pitch);
                            uint32_t cost = cand_sad + lambda_motion * (uint32_t)(abs(left_mv.x) + abs(left_mv.y));
                            if (cost < best_cost) {
                                best_cost = cost;
                                best_mv = left_mv;
                            }
                        }
                    }
                }
            }

            /* If spatial predictor found a low cost match, early exit immediately */
            if ((best_mv.x != 0 || best_mv.y != 0) && best_cost < EARLY_TERMINATION_COST) {
                out_mvs[mb_idx].mvx = best_mv.x * 4;
                out_mvs[mb_idx].mvy = best_mv.y * 4;
                out_mvs[mb_idx].sad = best_cost;
                out_mvs[mb_idx]._pad = 0;
                continue;
            }

            /* 3. Hierarchical Adaptive Diamond/Square Search */
            for (int step = (int)(max_rad / 2); step >= 1; step /= 2) {
                ivec2_t center = best_mv;
                for (int c = 0; c < 8; c++) {
                    int cand_x = center.x + search_pattern[c].x * step;
                    int cand_y = center.y + search_pattern[c].y * step;

                    if (abs(cand_x) > (int)max_rad || abs(cand_y) > (int)max_rad) continue;

                    int test_ref_x = px + cand_x;
                    int test_ref_y = py + cand_y;

                    /* Bounds check against macroblock-padded buffer edges */
                    if (test_ref_x < 0 || test_ref_x + 16 > max_x ||
                        test_ref_y < 0 || test_ref_y + 16 > max_y) {
                        continue;
                    }

                    const uint8_t *cand_ref = ref_y + test_ref_y * ref_pitch + test_ref_x;
                    uint32_t cand_sad = sad_fn(curr_mb, src_pitch, cand_ref, ref_pitch);
                    uint32_t cost = cand_sad + lambda_motion * (uint32_t)(abs(cand_x) + abs(cand_y));

                    if (cost < best_cost) {
                        best_cost = cost;
                        best_mv.x = cand_x;
                        best_mv.y = cand_y;
                    }
                }

                if (best_cost == 0) break;
            }

            /* Convert integer motion vector to quarter-pel units (multiply by 4) */
            out_mvs[mb_idx].mvx = best_mv.x * 4;
            out_mvs[mb_idx].mvy = best_mv.y * 4;
            out_mvs[mb_idx].sad = best_cost;
            out_mvs[mb_idx]._pad = 0;
        }
    }
#ifdef _OPENMP
    }
#endif

    return 0;
}
