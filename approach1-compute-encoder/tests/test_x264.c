/*
 * Copyright (c) 2026 BC-250 Project Contributors
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * test_x264.c - the libx264 backend on its own, no GPU: every picture in
 * gives exactly one picture out, with the parameter sets where a decoder
 * needs them, in every rate mode and profile, across a bitrate change and a
 * size change.
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "encoder_x264.h"
#include "rate_control.h"

#define W 320
#define H 240

static uint8_t y_plane[W * H], uv_plane[W * H / 2];
static uint8_t out[W * H * 4];

static void fill(int frame)
{
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++)
            y_plane[r * W + c] = (uint8_t)((c + frame * 3) ^ (r * 2));
    for (int i = 0; i < W * H / 2; i++)
        uv_plane[i] = (uint8_t)(128 + ((i + frame) & 15));
}

/* NAL unit types in an Annex B buffer, in order. */
static int nal_types(const uint8_t *b, int n, int *types, int max)
{
    int count = 0;
    for (int i = 0; i + 3 < n && count < max; i++) {
        if (b[i] == 0 && b[i + 1] == 0 && b[i + 2] == 1) {
            types[count++] = b[i + 3] & 0x1f;
            i += 3;
        }
    }
    return count;
}

static int has(const int *t, int n, int type)
{
    for (int i = 0; i < n; i++) if (t[i] == type) return 1;
    return 0;
}

static void run(const char *what, h264_x264_config_t cfg, int frames, int idr_at)
{
    h264_x264_t *x = h264_x264_create();
    assert(x);
    for (int f = 0; f < frames; f++) {
        fill(f);
        int n = h264_x264_encode(x, &cfg, y_plane, W, uv_plane, W,
                                 f == idr_at, cfg.rc_mode == RC_CQP ? cfg.qp : 0,
                                 out, sizeof out);
        assert(n > 0 && "every picture in must come out at once");
        int t[64];
        int k = nal_types(out, n, t, 64);
        assert(k > 0 && t[0] == 9 && "access unit delimiter first");
        if (f == 0 || f == idr_at) {
            assert(has(t, k, 7) && has(t, k, 8) && has(t, k, 5) && "SPS, PPS and an IDR");
        } else {
            assert(!has(t, k, 5) && has(t, k, 1) && "a P picture");
        }
    }
    h264_x264_destroy(x);
    printf("[test_x264] %-32s ok\n", what);
}

int main(void)
{
    const h264_x264_config_t base = {
        .width = W, .height = H, .fps = 30, .gop = 1000,
        .rc_mode = RC_CQP, .qp = 26, .quality_level = 4, .profile_idc = 100,
    };
    h264_x264_config_t c;

    run("CQP, High", base, 8, 5);
    c = base; c.profile_idc = 66;
    run("CQP, Baseline", c, 8, -1);
    c = base; c.profile_idc = 77;
    run("CQP, Main", c, 8, -1);
    c = base; c.rc_mode = RC_VBR; c.bitrate = 500000;
    run("VBR", c, 8, -1);
    c = base; c.rc_mode = RC_CBR; c.bitrate = 500000; c.cbr_intent = true;
    run("CBR with filler", c, 8, -1);
    c = base; c.rc_mode = RC_LOW_LATENCY; c.bitrate = 500000; c.live = true;
    run("low latency, live", c, 8, -1);
    c = base; c.rc_mode = RC_VBR; c.crf = 23;
    run("ICQ as CRF", c, 8, -1);

    /* A bitrate change reconfigures and carries on with P pictures; a size
     * change reopens, and the next picture is an IDR with new headers. */
    {
        h264_x264_t *x = h264_x264_create();
        c = base; c.rc_mode = RC_CBR; c.bitrate = 400000;
        int t[64];
        for (int f = 0; f < 6; f++) {
            if (f == 3) c.bitrate = 800000;
            fill(f);
            int n = h264_x264_encode(x, &c, y_plane, W, uv_plane, W, false, 0, out, sizeof out);
            assert(n > 0);
            int k = nal_types(out, n, t, 64);
            assert(f == 0 ? has(t, k, 5) : !has(t, k, 5));
        }
        c.width = W / 2; c.height = H / 2;
        fill(9);
        int n = h264_x264_encode(x, &c, y_plane, W, uv_plane, W, false, 0, out, sizeof out);
        assert(n > 0);
        int k = nal_types(out, n, t, 64);
        assert(has(t, k, 7) && has(t, k, 5) && "a new size starts over with an IDR");
        h264_x264_destroy(x);
        printf("[test_x264] %-32s ok\n", "bitrate change, size change");
    }

    /* The preset follows the pixel rate and VA's quality level. */
    c = base; c.width = 1920; c.height = 1080; c.fps = 30;
    assert(strcmp(h264_x264_preset_for(&c), "veryfast") == 0);
    c.fps = 60;
    assert(strcmp(h264_x264_preset_for(&c), "superfast") == 0);
    c.width = 3840; c.height = 2160;
    assert(strcmp(h264_x264_preset_for(&c), "ultrafast") == 0);
    c.quality_level = 1;
    assert(strcmp(h264_x264_preset_for(&c), "superfast") == 0);
    printf("[test_x264] %-32s ok\n", "preset choice");

    printf("[test_x264] all good\n");
    return 0;
}
