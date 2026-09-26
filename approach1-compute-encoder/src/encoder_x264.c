/*
 * Copyright (c) 2026 BC-250 Project Contributors
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * encoder_x264.c - H.264 through libx264. See encoder_x264.h.
 */
#include "encoder_x264.h"
#include "rate_control.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <x264.h>

struct h264_x264 {
    x264_t *h;
    x264_param_t param;
    h264_x264_config_t applied;   /* what h was opened or last reconfigured with */
    const char *preset;
    int threads;
    int64_t pts;
};

/* ultrafast .. faster, in the order x264 trades speed for bits. */
static const char *const presets[] = { "ultrafast", "superfast", "veryfast", "faster" };

static const char *get_cmdline_preset(void)
{
#if defined(__linux__)
    static char cached_preset[32] = {0};
    static bool checked = false;
    if (checked) return cached_preset[0] ? cached_preset : NULL;
    checked = true;

    FILE *f = fopen("/proc/self/cmdline", "rb");
    if (!f) return NULL;

    char buf[4096];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    if (n == 0) return NULL;
    buf[n] = '\0';

    static const char *const valid_presets[] = {
        "ultrafast", "superfast", "veryfast", "faster", "fast",
        "medium", "slow", "slower", "veryslow", "placebo", NULL
    };

    size_t pos = 0;
    while (pos < n) {
        const char *arg = buf + pos;
        size_t len = strlen(arg);

        if ((strcmp(arg, "-preset") == 0 || strcmp(arg, "--preset") == 0) && (pos + len + 1 < n)) {
            const char *val = buf + pos + len + 1;
            for (int i = 0; valid_presets[i]; i++) {
                if (strcmp(val, valid_presets[i]) == 0) {
                    strncpy(cached_preset, val, sizeof(cached_preset) - 1);
                    return cached_preset;
                }
            }
        } else if (strncmp(arg, "-preset=", 8) == 0 || strncmp(arg, "--preset=", 9) == 0) {
            const char *val = strchr(arg, '=') + 1;
            for (int i = 0; valid_presets[i]; i++) {
                if (strcmp(val, valid_presets[i]) == 0) {
                    strncpy(cached_preset, val, sizeof(cached_preset) - 1);
                    return cached_preset;
                }
            }
        }
        pos += len + 1;
    }
#endif
    return NULL;
}

const char *h264_x264_preset_for(const h264_x264_config_t *cfg)
{
    /* 1. Environment variable override takes highest priority:
     * Support BC250_X264_PRESET, BC250_PRESET, and X264_PRESET. */
    const char *env = getenv("BC250_X264_PRESET");
    if (!env || !*env) env = getenv("BC250_PRESET");
    if (!env || !*env) env = getenv("X264_PRESET");
    if (env && *env) return env;

    /* 2. Process command-line argument (-preset <name> / --preset=<name>):
     * Directly honors FFmpeg/application CLI flags even if the application's
     * VA-API wrapper internally dropped or ignored the option. */
    const char *cmd_preset = get_cmdline_preset();
    if (cmd_preset && *cmd_preset) return cmd_preset;

    /* 3. Live streaming server (Sunshine, Steam Link, WiVRn): latency is the
     * primary constraint. Ultrafast executes in 2-4ms per frame on 4 Zen 2
     * threads, keeping total encode latency well under the 16.6ms 60fps budget.
     * Slower presets (superfast) can still be selected via quality_level <= 2
     * or BC250_X264_PRESET. */
    if (cfg->live) {
        if (cfg->quality_level > 0 && cfg->quality_level <= 2) {
            return "superfast";
        }
        return "ultrafast";
    }

    /* 4. By pixel rate first for offline transcodes without explicit preset. */
    const double rate = (double)cfg->width * cfg->height * (cfg->fps ? cfg->fps : 30);
    const double p1080 = 1920.0 * 1080.0;
    int i = rate <= p1080 * 31 ? 2 : (rate <= p1080 * 61 ? 1 : 0);

    /* Then VA's quality level: 1 is "best", 7 "fastest", 4 the default. */
    const uint32_t q = cfg->quality_level ? cfg->quality_level : 4;
    if (q <= 2) i++;
    else if (q >= 6) i = 0;
    else if (q == 5) i--;
    if (i < 0) i = 0;
    if (i > 3) i = 3;
    return presets[i];
}

static int threads_for(const h264_x264_config_t *cfg)
{
    const char *env = getenv("BC250_X264_THREADS");
    if (env && *env) {
        int t = atoi(env);
        if (t >= 0 && t <= 64) return t;
    }
    const char *env_max = getenv("BC250_MAX_CPU_THREADS");
    if (env_max && *env_max) {
        int t = atoi(env_max);
        if (t >= 0 && t <= 64) return t;
    }
    /* A streaming server shares the machine with the game it streams:
     * four threads, which is where x264 stops scaling usefully with sliced
     * threads anyway. For offline transcode, default to 4 threads as well to
     * prevent pinning all 16 Zen 2 cores at 100% CPU. Unconstrained auto-threads
     * can still be requested explicitly via BC250_X264_THREADS=0. */
    (void)cfg;
    return 4;
}

static const char *profile_name(int profile_idc)
{
    if (profile_idc == 100) return "high";
    if (profile_idc == 77) return "main";
    return "baseline";
}

/* The rate part of the parameters, shared by open and reconfigure. */
static void set_rate(x264_param_t *p, const h264_x264_config_t *cfg)
{
    p->rc.i_vbv_max_bitrate = 0;
    p->rc.i_vbv_buffer_size = 0;
    p->i_nal_hrd = X264_NAL_HRD_NONE;
    p->rc.b_filler = 0;

    if (cfg->rc_mode == RC_CQP) {
        p->rc.i_rc_method = X264_RC_CQP;
        p->rc.i_qp_constant = cfg->qp > 0 && cfg->qp <= 51 ? cfg->qp : 26;
        return;
    }
    if (cfg->crf > 0) {
        /* ICQ is constant quality, which is what CRF is. */
        p->rc.i_rc_method = X264_RC_CRF;
        float crf_val = (float)(cfg->crf > 51 ? 51 : cfg->crf);
        const char *env_crf = getenv("BC250_X264_CRF");
        if (env_crf && *env_crf) {
            float env_val = (float)atof(env_crf);
            if (env_val >= 0.0f && env_val <= 51.0f) crf_val = env_val;
        }
        p->rc.f_rf_constant = crf_val;
        return;
    }

    const int kbps = (int)(cfg->bitrate / 1000) > 0 ? (int)(cfg->bitrate / 1000) : 4000;
    const int fps = cfg->fps ? (int)cfg->fps : 30;
    p->rc.i_rc_method = X264_RC_ABR;
    p->rc.i_bitrate = kbps;
    if (cfg->live) {
        /* Live streaming (Sunshine, Steam Link, WiVRn): 1 frame VBV buffer
         * guarantees every frame fits the network link on its own, preventing
         * buffer bloat, frame queueing, and stream latency spikes. */
        p->rc.i_vbv_max_bitrate = kbps;
        p->rc.i_vbv_buffer_size = kbps / fps > 0 ? kbps / fps : 1;
    } else {
        switch (cfg->rc_mode) {
        case RC_LOW_LATENCY:
            p->rc.i_vbv_max_bitrate = kbps;
            p->rc.i_vbv_buffer_size = kbps / fps > 0 ? kbps / fps : 1;
            break;
        case RC_CBR:
            p->rc.i_vbv_max_bitrate = kbps;
            p->rc.i_vbv_buffer_size = kbps;
            break;
        default: /* RC_VBR */
            p->rc.i_vbv_max_bitrate = kbps + kbps / 2;
            p->rc.i_vbv_buffer_size = kbps * 2;
            break;
        }
    }
    if (cfg->cbr_intent && cfg->rc_mode != RC_VBR) {
        p->i_nal_hrd = X264_NAL_HRD_CBR;
        p->rc.b_filler = 1;
    }
}

static bool same_structure(const h264_x264_config_t *a, const h264_x264_config_t *b)
{
    return a->width == b->width && a->height == b->height && a->fps == b->fps
        && a->gop == b->gop && a->profile_idc == b->profile_idc
        && a->quality_level == b->quality_level && a->live == b->live
        /* CQP against a bitrate mode is a different encoder, not a rate. */
        && (a->rc_mode == RC_CQP) == (b->rc_mode == RC_CQP)
        && (a->crf > 0) == (b->crf > 0);
}

static bool same_rate(const h264_x264_config_t *a, const h264_x264_config_t *b)
{
    return a->rc_mode == b->rc_mode && a->bitrate == b->bitrate && a->qp == b->qp
        && a->crf == b->crf && a->cbr_intent == b->cbr_intent;
}

static int open_encoder(h264_x264_t *x, const h264_x264_config_t *cfg)
{
    if (x->h) {
        x264_encoder_close(x->h);
        x->h = NULL;
    }
    x264_param_t *p = &x->param;
    x->preset = h264_x264_preset_for(cfg);
    x->threads = threads_for(cfg);
    /* zerolatency: no B-frames, no lookahead, sliced threads - one picture
     * in, the same picture out, which is what VA-API's one coded buffer per
     * vaEndPicture assumes. */
    if (x264_param_default_preset(p, x->preset, "zerolatency") < 0) {
        fprintf(stderr, "[bc250-x264] unknown preset '%s'\n", x->preset);
        return -1;
    }
    p->i_log_level = X264_LOG_ERROR;
    p->i_width = (int)cfg->width;
    p->i_height = (int)cfg->height;
    p->i_csp = X264_CSP_NV12;
    p->i_threads = x->threads;
    if (cfg->live) {
        /* Sliced threads process each frame in parallel across threads without
         * adding frame delay, keeping latency strictly under 4ms. */
        p->b_sliced_threads = 1;
        p->rc.i_lookahead = 0;
        p->i_sync_lookahead = 0;
    }
    p->i_fps_num = cfg->fps ? cfg->fps : 30;
    p->i_fps_den = 1;
    p->b_vfr_input = 0;
    p->i_keyint_max = cfg->gop ? (int)cfg->gop : (int)p->i_fps_num;
    p->i_scenecut_threshold = 0;    /* IDRs where the caller put them, nowhere else */
    p->b_repeat_headers = 1;        /* SPS and PPS in front of every IDR */
    p->b_annexb = 1;
    p->b_aud = 1;
    set_rate(p, cfg);
    if (x264_param_apply_profile(p, profile_name(cfg->profile_idc)) < 0)
        return -1;

    x->h = x264_encoder_open(p);
    if (!x->h) {
        fprintf(stderr, "[bc250-x264] x264_encoder_open failed (%ux%u, %s)\n",
                cfg->width, cfg->height, x->preset);
        return -1;
    }
    x->applied = *cfg;
    fprintf(stderr, "[bc250-x264] %ux%u @ %u fps, %s, %s profile, threads %d, rc %d%s\n",
            cfg->width, cfg->height, p->i_fps_num, x->preset, profile_name(cfg->profile_idc),
            x->threads, cfg->rc_mode, cfg->crf > 0 ? " (ICQ)" : "");
    return 0;
}

h264_x264_t *h264_x264_create(void)
{
    return calloc(1, sizeof(h264_x264_t));
}

int h264_x264_encode(h264_x264_t *x, const h264_x264_config_t *cfg,
                     const uint8_t *y, int y_stride,
                     const uint8_t *uv, int uv_stride,
                     bool force_idr, int frame_qp,
                     uint8_t *out, size_t out_size)
{
    if (!x || !cfg || !y || !uv || !out || !cfg->width || !cfg->height) return -1;

    if (!x->h || !same_structure(cfg, &x->applied)) {
        if (open_encoder(x, cfg) != 0) return -1;
    } else if (!same_rate(cfg, &x->applied)) {
        set_rate(&x->param, cfg);
        if (x264_encoder_reconfig(x->h, &x->param) < 0) {
            if (open_encoder(x, cfg) != 0) return -1;
        }
        x->applied = *cfg;
    }

    x264_picture_t in, pic_out;
    x264_picture_init(&in);
    in.img.i_csp = X264_CSP_NV12;
    in.img.i_plane = 2;
    in.img.plane[0] = (uint8_t *)y;
    in.img.i_stride[0] = y_stride;
    in.img.plane[1] = (uint8_t *)uv;
    in.img.i_stride[1] = uv_stride;
    in.i_pts = x->pts++;
    in.i_type = force_idr ? X264_TYPE_IDR : X264_TYPE_AUTO;
    if (cfg->rc_mode == RC_CQP && frame_qp > 0 && frame_qp <= 51)
        in.i_qpplus1 = frame_qp + 1;

    x264_nal_t *nal = NULL;
    int n = 0;
    int size = x264_encoder_encode(x->h, &nal, &n, &in, &pic_out);
    /* zerolatency never holds a picture back; should it ever, the caller's
     * coded buffer still has to hold THIS picture, so drain it here. */
    while (size == 0 && x264_encoder_delayed_frames(x->h) > 0)
        size = x264_encoder_encode(x->h, &nal, &n, NULL, &pic_out);
    if (size <= 0) return -1;
    if ((size_t)size > out_size) return -1;
    /* x264 lays a picture's NAL units out back to back in one buffer. */
    memcpy(out, nal[0].p_payload, (size_t)size);
    return size;
}

void h264_x264_destroy(h264_x264_t *x)
{
    if (!x) return;
    if (x->h) x264_encoder_close(x->h);
    free(x);
}
