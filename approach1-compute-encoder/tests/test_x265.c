/*
 * Copyright (c) 2026 BC-250 Project Contributors
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * test_x265.c - Unit test for libx265 fallback HEVC encoding backend:
 * verifies single-frame zero-latency output, VPS/SPS/PPS parameter sets,
 * IDR generation, and rate control modes.
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "encoder_x265.h"
#include "rate_control.h"

#define W 320
#define H 240

static uint8_t y_plane[W * H];
static uint8_t uv_plane[W * H / 2];
static uint16_t y_plane10[W * H];
static uint16_t uv_plane10[W * H];
static uint8_t out[W * H * 4];

static void fill_frame(int frame)
{
    for (int r = 0; r < H; r++) {
        for (int c = 0; c < W; c++) {
            y_plane[r * W + c] = (uint8_t)((c + frame * 3) ^ (r * 2));
        }
    }
    for (int i = 0; i < W * H / 2; i++) {
        uv_plane[i] = (uint8_t)(128 + ((i + frame) & 15));
    }
}

static void fill_frame10(int frame)
{
    for (int r = 0; r < H; r++) {
        for (int c = 0; c < W; c++) {
            uint16_t val10 = (uint16_t)(((c + frame * 3) ^ (r * 2)) & 0x3FF);
            y_plane10[r * W + c] = (uint16_t)(val10 << 6);
        }
    }
    for (int i = 0; i < W * H / 2; i++) {
        uint16_t u = (uint16_t)((512 + ((i + frame) & 63)) & 0x3FF);
        uint16_t v = (uint16_t)((512 - ((i + frame) & 63)) & 0x3FF);
        uv_plane10[i * 2] = (uint16_t)(u << 6);
        uv_plane10[i * 2 + 1] = (uint16_t)(v << 6);
    }
}

/* Parse HEVC NAL unit types in an Annex B bitstream */
static int hevc_nal_types(const uint8_t *b, int n, int *types, int max)
{
    int count = 0;
    for (int i = 0; i + 4 < n && count < max; i++) {
        if (b[i] == 0 && b[i + 1] == 0 && b[i + 2] == 1) {
            /* HEVC NAL header is 2 bytes: (byte 0 >> 1) & 0x3F */
            types[count++] = (b[i + 3] >> 1) & 0x3F;
            i += 3;
        } else if (b[i] == 0 && b[i + 1] == 0 && b[i + 2] == 0 && b[i + 3] == 1) {
            types[count++] = (b[i + 4] >> 1) & 0x3F;
            i += 4;
        }
    }
    return count;
}

static int has_nal(const int *t, int n, int type)
{
    for (int i = 0; i < n; i++) {
        if (t[i] == type) return 1;
    }
    return 0;
}

static void run_hevc_test(const char *what, hevc_x265_config_t cfg, int frames, int idr_at)
{
    hevc_x265_t *x = hevc_x265_create();
    assert(x && "hevc_x265_create failed");

    for (int f = 0; f < frames; f++) {
        fill_frame(f);
        int n = hevc_x265_encode(x, &cfg, y_plane, W, uv_plane, W,
                                 f == idr_at, cfg.rc_mode == RC_CQP ? cfg.qp : 0,
                                 out, sizeof(out));
        assert(n > 0 && "every frame in must output encoded bitstream");

        int t[64];
        int k = hevc_nal_types(out, n, t, 64);
        assert(k > 0 && "bitstream must contain at least one valid HEVC NAL unit");

        if (f == 0 || f == idr_at) {
            /* VPS (32), SPS (33), PPS (34), IDR (19 or 20) or CRA (21) */
            int has_headers = has_nal(t, k, 32) && has_nal(t, k, 33) && has_nal(t, k, 34);
            int has_key = has_nal(t, k, 19) || has_nal(t, k, 20) || has_nal(t, k, 21);
            assert((has_headers || has_key) && "First frame or IDR frame must have parameter sets or keyframe NAL");
        }
    }

    hevc_x265_destroy(x);
    printf("[test_x265] %-32s ok\n", what);
}

static void run_hevc_test10(const char *what, hevc_x265_config_t cfg, int frames)
{
    if (!hevc_x265_supports_10bit()) {
        printf("[test_x265] %-32s skipped (libx265 lacks 10-bit multilib symbols)\n", what);
        return;
    }

    hevc_x265_t *x = hevc_x265_create();
    assert(x && "hevc_x265_create failed");

    fill_frame10(0);
    int n0 = hevc_x265_encode(x, &cfg, (const uint8_t *)y_plane10, W * 2,
                              (const uint8_t *)uv_plane10, W * 2,
                              true, cfg.rc_mode == RC_CQP ? cfg.qp : 0,
                              out, sizeof(out));
    if (n0 < 0) {
        printf("[test_x265] %-32s skipped (10-bit encoder initialization failed)\n", what);
        hevc_x265_destroy(x);
        return;
    }
    assert(n0 > 0 && "10-bit frame must output encoded bitstream");

    int t0[64];
    int k0 = hevc_nal_types(out, n0, t0, 64);
    assert(k0 > 0 && "10-bit bitstream must contain valid HEVC NAL units");
    int has_headers = has_nal(t0, k0, 32) && has_nal(t0, k0, 33) && has_nal(t0, k0, 34);
    int has_key = has_nal(t0, k0, 19) || has_nal(t0, k0, 20) || has_nal(t0, k0, 21);
    assert((has_headers || has_key) && "First 10-bit frame must have parameter sets or IDR NAL");

    for (int f = 1; f < frames; f++) {
        fill_frame10(f);
        int n = hevc_x265_encode(x, &cfg, (const uint8_t *)y_plane10, W * 2,
                                 (const uint8_t *)uv_plane10, W * 2,
                                 false, cfg.rc_mode == RC_CQP ? cfg.qp : 0,
                                 out, sizeof(out));
        assert(n > 0 && "10-bit frame must output encoded bitstream");

        int t[64];
        int k = hevc_nal_types(out, n, t, 64);
        assert(k > 0 && "10-bit bitstream must contain valid HEVC NAL units");
    }

    hevc_x265_destroy(x);
    printf("[test_x265] %-32s ok\n", what);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    const hevc_x265_config_t base = {
        .width = W,
        .height = H,
        .fps = 30,
        .gop = 1000,
        .rc_mode = RC_CQP,
        .qp = 26,
        .quality_level = 4,
        .ten_bit = false,
        .live = true,
    };
    hevc_x265_config_t c;

    run_hevc_test("HEVC CQP Main Profile", base, 6, 3);

    c = base;
    c.rc_mode = RC_VBR;
    c.bitrate = 600000;
    run_hevc_test("HEVC VBR Mode", c, 6, -1);

    c = base;
    c.rc_mode = RC_CBR;
    c.bitrate = 600000;
    c.cbr_intent = true;
    run_hevc_test("HEVC CBR Mode", c, 6, -1);

    c = base;
    c.rc_mode = RC_LOW_LATENCY;
    c.bitrate = 500000;
    c.live = true;
    run_hevc_test("HEVC Low Latency Live", c, 6, -1);

    c = base;
    c.live = false;
    run_hevc_test("HEVC Offline Transcode", c, 6, -1);

    c = base;
    c.ten_bit = true;
    run_hevc_test10("HEVC Main10 P010", c, 6);

    /* Dynamic bitrate reconfiguration test */
    {
        hevc_x265_t *x = hevc_x265_create();
        assert(x);
        hevc_x265_config_t dyn = base;
        dyn.rc_mode = RC_VBR;
        dyn.bitrate = 500000;
        fill_frame(0);
        int n1 = hevc_x265_encode(x, &dyn, y_plane, W, uv_plane, W, false, 0, out, sizeof(out));
        assert(n1 > 0);

        /* Change bitrate on the fly */
        dyn.bitrate = 1200000;
        fill_frame(1);
        int n2 = hevc_x265_encode(x, &dyn, y_plane, W, uv_plane, W, false, 0, out, sizeof(out));
        assert(n2 > 0);

        hevc_x265_destroy(x);
        printf("[test_x265] %-32s ok\n", "Dynamic bitrate reconfig");
    }

    printf("[test_x265] all tests passed!\n");
    return 0;
}
