/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * test_hevc_mc.c - Unit tests for HEVC 10-bit and 8-bit motion compensation,
 * residual addition, and SIMD acceleration.
 */

#include "hevc_dec_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

static uint64_t seed = 0x123456789abcdef0ull;
static uint32_t rand_u32(void)
{
    seed ^= seed >> 12;
    seed ^= seed << 25;
    seed ^= seed >> 27;
    return (uint32_t)((seed * 0x2545F4914F6CDD1Dull) >> 32);
}

static void test_residual_addition_10bit(void)
{
    printf("[TEST] Testing HEVC 10-bit residual addition (add_residual_10)...\n");

    /* 1. Test extreme saturation / overflow protection */
    /* If vd = 1000 and res = 32000, sum must clamp to 1023 without 16-bit signed wrap */
    uint16_t plane10[32 * 32];
    int16_t res[32 * 32];

    for (int i = 0; i < 32 * 32; i++) {
        plane10[i] = 1000;
        res[i] = 32000;
    }
    hevcd_add((uint8_t *)plane10, 32, 0, 0, res, 5, 10); /* 32x32 block */

    for (int i = 0; i < 32 * 32; i++) {
        assert(plane10[i] == 1023);
    }

    /* 2. Test extreme negative underflow protection */
    for (int i = 0; i < 32 * 32; i++) {
        plane10[i] = 50;
        res[i] = -32000;
    }
    hevcd_add((uint8_t *)plane10, 32, 0, 0, res, 5, 10);

    for (int i = 0; i < 32 * 32; i++) {
        assert(plane10[i] == 0);
    }

    /* 3. Test various block sizes against scalar reference: 4x4, 8x8, 16x16, 32x32 */
    int log2_sizes[] = { 2, 3, 4, 5 };
    for (size_t s = 0; s < sizeof(log2_sizes)/sizeof(log2_sizes[0]); s++) {
        int log2_sz = log2_sizes[s];
        int n = 1 << log2_sz;
        int stride = n + 8;
        uint16_t sim_dst[64 * 64];
        uint16_t ref_dst[64 * 64];
        int16_t test_res[64 * 64];

        for (int i = 0; i < 64 * 64; i++) {
            uint16_t v = (uint16_t)(rand_u32() % 1024);
            sim_dst[i] = v;
            ref_dst[i] = v;
            test_res[i] = (int16_t)((int)(rand_u32() % 65536) - 32768);
        }

        hevcd_add((uint8_t *)sim_dst, stride, 0, 0, test_res, log2_sz, 10);

        /* Scalar reference */
        for (int y = 0; y < n; y++) {
            for (int x = 0; x < n; x++) {
                int sum = (int)ref_dst[y * stride + x] + (int)test_res[y * n + x];
                ref_dst[y * stride + x] = (uint16_t)(sum < 0 ? 0 : (sum > 1023 ? 1023 : sum));
            }
        }

        for (int y = 0; y < n; y++) {
            for (int x = 0; x < n; x++) {
                assert(sim_dst[y * stride + x] == ref_dst[y * stride + x]);
            }
        }
    }

    printf("  ✓ HEVC 10-bit residual addition verified!\n");
}

static void test_residual_addition_8bit(void)
{
    printf("[TEST] Testing HEVC 8-bit residual addition (add_residual_8)...\n");

    int log2_sizes[] = { 2, 3, 4, 5 };
    for (size_t s = 0; s < sizeof(log2_sizes)/sizeof(log2_sizes[0]); s++) {
        int log2_sz = log2_sizes[s];
        int n = 1 << log2_sz;
        int stride = n + 16;
        uint8_t sim_dst[64 * 64];
        uint8_t ref_dst[64 * 64];
        int16_t test_res[64 * 64];

        for (int i = 0; i < 64 * 64; i++) {
            uint8_t v = (uint8_t)(rand_u32() % 256);
            sim_dst[i] = v;
            ref_dst[i] = v;
            test_res[i] = (int16_t)((int)(rand_u32() % 1024) - 512);
        }

        hevcd_add(sim_dst, stride, 0, 0, test_res, log2_sz, 8);

        /* Scalar reference */
        for (int y = 0; y < n; y++) {
            for (int x = 0; x < n; x++) {
                int sum = (int)ref_dst[y * stride + x] + (int)test_res[y * n + x];
                ref_dst[y * stride + x] = (uint8_t)(sum < 0 ? 0 : (sum > 255 ? 255 : sum));
            }
        }

        for (int y = 0; y < n; y++) {
            for (int x = 0; x < n; x++) {
                assert(sim_dst[y * stride + x] == ref_dst[y * stride + x]);
            }
        }
    }

    printf("  ✓ HEVC 8-bit residual addition verified!\n");
}

static void test_hevc_10bit_inter_prediction(void)
{
    printf("[TEST] Testing HEVC 10-bit motion compensation (hevcd_predict_inter)...\n");

    hevc_sps_t sps;
    memset(&sps, 0, sizeof sps);
    sps.width = 128;
    sps.height = 128;
    sps.bit_depth_luma = 10;
    sps.bit_depth_chroma = 10;
    sps.log2_ctb = 6;

    hevc_pps_t pps;
    memset(&pps, 0, sizeof pps);

    hevcd_t dec;
    memset(&dec, 0, sizeof dec);
    dec.sps = &sps;
    dec.pps = &pps;

    uint16_t curr_y[128 * 128];
    uint16_t curr_cb[64 * 64];
    uint16_t curr_cr[64 * 64];
    hevcd_img_t curr_img;
    memset(&curr_img, 0, sizeof curr_img);
    curr_img.plane[0] = (uint8_t *)curr_y;
    curr_img.plane[1] = (uint8_t *)curr_cb;
    curr_img.plane[2] = (uint8_t *)curr_cr;
    curr_img.stride[0] = 128;
    curr_img.stride[1] = 64;
    curr_img.stride[2] = 64;
    dec.current = &curr_img;

    uint16_t ref0_y[128 * 128], ref0_cb[64 * 64], ref0_cr[64 * 64];
    uint16_t ref1_y[128 * 128], ref1_cb[64 * 64], ref1_cr[64 * 64];
    hevcd_img_t ref0_img, ref1_img;
    memset(&ref0_img, 0, sizeof ref0_img);
    memset(&ref1_img, 0, sizeof ref1_img);

    for (int i = 0; i < 128 * 128; i++) {
        ref0_y[i] = (uint16_t)(rand_u32() % 1024);
        ref1_y[i] = (uint16_t)(rand_u32() % 1024);
    }
    for (int i = 0; i < 64 * 64; i++) {
        ref0_cb[i] = (uint16_t)(rand_u32() % 1024);
        ref0_cr[i] = (uint16_t)(rand_u32() % 1024);
        ref1_cb[i] = (uint16_t)(rand_u32() % 1024);
        ref1_cr[i] = (uint16_t)(rand_u32() % 1024);
    }
    ref0_img.plane[0] = (uint8_t *)ref0_y;
    ref0_img.plane[1] = (uint8_t *)ref0_cb;
    ref0_img.plane[2] = (uint8_t *)ref0_cr;
    ref0_img.stride[0] = 128;
    ref0_img.stride[1] = 64;
    ref0_img.stride[2] = 64;

    ref1_img.plane[0] = (uint8_t *)ref1_y;
    ref1_img.plane[1] = (uint8_t *)ref1_cb;
    ref1_img.plane[2] = (uint8_t *)ref1_cr;
    ref1_img.stride[0] = 128;
    ref1_img.stride[1] = 64;
    ref1_img.stride[2] = 64;

    dec.ref_pic[0][0] = &ref0_img;
    dec.ref_pic[1][0] = &ref1_img;
    dec.n_refs[0] = 1;
    dec.n_refs[1] = 1;

    /* 1. Integer-pel single prediction (copy14_10 vectorization) */
    hevcd_mvf_t mv_int;
    memset(&mv_int, 0, sizeof mv_int);
    mv_int.pred_flag = HEVCD_PF_L0;
    mv_int.ref_idx[0] = 0;
    mv_int.mv[0][0] = 4 * 4; /* dx = 4 whole samples */
    mv_int.mv[0][1] = 2 * 4; /* dy = 2 whole samples */

    hevcd_predict_inter(&dec, 16, 16, 32, 32, &mv_int);

    /* Verify luma destination matches reference at whole-sample offset */
    for (int r = 0; r < 32; r++) {
        for (int c = 0; c < 32; c++) {
            uint16_t expected = ref0_y[(16 + 2 + r) * 128 + (16 + 4 + c)];
            uint16_t actual = curr_y[(16 + r) * 128 + (16 + c)];
            assert(actual == expected);
        }
    }

    /* 2. Fractional sub-pel prediction (one_pred_10 vectorization) */
    hevcd_mvf_t mv_frac;
    memset(&mv_frac, 0, sizeof mv_frac);
    mv_frac.pred_flag = HEVCD_PF_L0;
    mv_frac.ref_idx[0] = 0;
    mv_frac.mv[0][0] = 4 * 4 + 2; /* half-pel x */
    mv_frac.mv[0][1] = 2 * 4 + 2; /* half-pel y */

    hevcd_predict_inter(&dec, 16, 16, 32, 32, &mv_frac);
    for (int r = 0; r < 32; r++) {
        for (int c = 0; c < 32; c++) {
            uint16_t val = curr_y[(16 + r) * 128 + (16 + c)];
            assert(val <= 1023);
        }
    }

    /* 3. Bi-prediction (two_pred_10 vectorization) */
    hevcd_mvf_t mv_bi;
    memset(&mv_bi, 0, sizeof mv_bi);
    mv_bi.pred_flag = HEVCD_PF_BI;
    mv_bi.ref_idx[0] = 0;
    mv_bi.ref_idx[1] = 0;
    mv_bi.mv[0][0] = 2 * 4 + 1; /* quarter-pel */
    mv_bi.mv[0][1] = 1 * 4 + 3;
    mv_bi.mv[1][0] = -1 * 4 + 2;
    mv_bi.mv[1][1] = 2 * 4 + 1;

    hevcd_predict_inter(&dec, 16, 16, 16, 16, &mv_bi);
    for (int r = 0; r < 16; r++) {
        for (int c = 0; c < 16; c++) {
            uint16_t val = curr_y[(16 + r) * 128 + (16 + c)];
            assert(val <= 1023);
        }
    }

    /* 4. Whole-sample bi-prediction (average_10 vectorization) */
    hevcd_mvf_t mv_avg;
    memset(&mv_avg, 0, sizeof mv_avg);
    mv_avg.pred_flag = HEVCD_PF_BI;
    mv_avg.ref_idx[0] = 0;
    mv_avg.ref_idx[1] = 0;
    mv_avg.mv[0][0] = 0;
    mv_avg.mv[0][1] = 0;
    mv_avg.mv[1][0] = 0;
    mv_avg.mv[1][1] = 0;

    hevcd_predict_inter(&dec, 32, 32, 16, 16, &mv_avg);
    for (int r = 0; r < 16; r++) {
        for (int c = 0; c < 16; c++) {
            uint16_t a = ref0_y[(32 + r) * 128 + (32 + c)];
            uint16_t b = ref1_y[(32 + r) * 128 + (32 + c)];
            uint16_t expected = (a + b + 1) >> 1;
            uint16_t actual = curr_y[(32 + r) * 128 + (32 + c)];
            assert(actual == expected);
        }
    }

    printf("  ✓ HEVC 10-bit motion compensation verified!\n");
}

int main(void)
{
    printf("========================================\n");
    printf("BC-250 HEVC Motion Compensation & Residual Test\n");
    printf("========================================\n");

    test_residual_addition_10bit();
    test_residual_addition_8bit();
    test_hevc_10bit_inter_prediction();

    printf("\nALL HEVC MC & RESIDUAL TESTS PASSED!\n");
    return 0;
}
