/*
 * Copyright (c) 2026 BC-250 Project Contributors
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * encoder_x265.c - HEVC/H.265 through libx265. See encoder_x265.h.
 */
#include "encoder_x265.h"
#include "rate_control.h"

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__linux__)
extern char *program_invocation_short_name;
#endif

static const char *const x265_presets[] = {
    "ultrafast", "superfast", "veryfast", "faster", "fast",
    "medium", "slow", "slower", "veryslow", "placebo"
};

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

    size_t pos = 0;
    while (pos < n) {
        const char *arg = buf + pos;
        size_t len = strlen(arg);

        if ((strcmp(arg, "-preset") == 0 || strcmp(arg, "--preset") == 0) && (pos + len + 1 < n)) {
            const char *val = buf + pos + len + 1;
            for (int i = 0; x265_presets[i]; i++) {
                if (strcmp(val, x265_presets[i]) == 0) {
                    strncpy(cached_preset, val, sizeof(cached_preset) - 1);
                    return cached_preset;
                }
            }
        } else if (strncmp(arg, "-preset=", 8) == 0 || strncmp(arg, "--preset=", 9) == 0) {
            const char *val = strchr(arg, '=') + 1;
            for (int i = 0; x265_presets[i]; i++) {
                if (strcmp(val, x265_presets[i]) == 0) {
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

const char *hevc_x265_preset_for(const hevc_x265_config_t *cfg)
{
    const char *env = getenv("BC250_X265_PRESET");
    if (!env || !*env) env = getenv("BC250_HEVC_PRESET");
    if (!env || !*env) env = getenv("BC250_PRESET");
    if (!env || !*env) env = getenv("X265_PRESET");
    if (env && *env) return env;

    const char *cmd_preset = get_cmdline_preset();
    if (cmd_preset && *cmd_preset) return cmd_preset;

    if (cfg->live) {
        if (cfg->quality_level > 0 && cfg->quality_level <= 2) {
            return "superfast";
        }
        return "ultrafast";
    }

    const double rate = (double)cfg->width * cfg->height * (cfg->fps ? cfg->fps : 30);
    const double p1080 = 1920.0 * 1080.0;
    int i = rate <= p1080 * 31 ? 2 : (rate <= p1080 * 61 ? 1 : 0);

    const uint32_t q = cfg->quality_level ? cfg->quality_level : 4;
    if (q <= 2) i++;
    else if (q >= 6) i = 0;
    else if (q == 5) i--;
    if (i < 0) i = 0;
    if (i > 3) i = 3;
    return x265_presets[i];
}

#ifdef BC250_HAVE_X265
#include <dlfcn.h>
#include <pthread.h>
#include <x265.h>

static bool is_steam_caller(void)
{
#if defined(__linux__)
    if (!program_invocation_short_name) return false;
    return (strcmp(program_invocation_short_name, "steam") == 0 ||
            strcmp(program_invocation_short_name, "streaming_client") == 0 ||
            strcmp(program_invocation_short_name, "steamwebhelper") == 0 ||
            strcmp(program_invocation_short_name, "gamescope") == 0 ||
            strstr(program_invocation_short_name, "steam") != NULL ||
            strstr(program_invocation_short_name, "gamescope") != NULL);
#else
    return false;
#endif
}

typedef struct {
    void *handle;
    int (*param_default_preset)(x265_param *param, const char *preset, const char *tune);
    int (*param_apply_profile)(x265_param *param, const char *profile);
    x265_encoder *(*encoder_open)(x265_param *param);
    int (*encoder_reconfig)(x265_encoder *encoder, x265_param *param);
    void (*picture_init)(x265_param *param, x265_picture *pic);
    int (*encoder_encode)(x265_encoder *encoder, x265_nal **pp_nal, uint32_t *pi_nal, x265_picture *pic_in, x265_picture *pic_out);
    void (*encoder_close)(x265_encoder *encoder);
} x265_ops_t;

static pthread_once_t x265_once = PTHREAD_ONCE_INIT;
static x265_ops_t x265_ops;
static bool x265_available = false;

static const char *const x265_candidates[] = {
    "libx265.so",
    "libx265.so.215",
    "libx265.so.212",
    "libx265.so.209",
    "libx265.so.200",
    "libx265.so.199",
    "libx265.so.192",
    "libx265.so.188",
    "libx265.so.179",
    "libx265.so.176",
    "libx265.so.160",
    "libx265.so.151",
    "libx265.so.146",
    "libx265.so.130",
    NULL
};

static void load_x265_ops(void)
{
    void *h = NULL;
    for (int i = 0; x265_candidates[i]; i++) {
        h = dlopen(x265_candidates[i], RTLD_NOW | RTLD_LOCAL);
        if (h) break;
    }
    if (!h) {
        char buf[64];
        for (int ver = 230; ver >= 130; ver--) {
            snprintf(buf, sizeof(buf), "libx265.so.%d", ver);
            h = dlopen(buf, RTLD_NOW | RTLD_LOCAL);
            if (h) break;
        }
    }
    if (!h) {
        static const char *const search_paths[] = {
            "/usr/local/lib64/libx265.so",
            "/usr/local/lib/libx265.so",
            "/usr/lib64/libx265.so",
            "/usr/lib/x86_64-linux-gnu/libx265.so",
            "/usr/lib/libx265.so",
            NULL
        };
        for (int i = 0; search_paths[i]; i++) {
            h = dlopen(search_paths[i], RTLD_NOW | RTLD_LOCAL);
            if (h) break;
        }
    }
    if (!h) {
        fprintf(stderr, "[bc250-x265] libx265 shared library not found via dlopen\n");
        return;
    }

    /* 1. Try resolving through x265_api_get / x265_api_get_### */
    const x265_api *(*p_x265_api_get)(int) = (const x265_api *(*)(int))dlsym(h, "x265_api_get");
#ifdef X265_BUILD
    if (!p_x265_api_get) {
        char sym[64];
        snprintf(sym, sizeof(sym), "x265_api_get_%d", X265_BUILD);
        p_x265_api_get = (const x265_api *(*)(int))dlsym(h, sym);
    }
#endif
    if (!p_x265_api_get) {
        char sym[64];
        for (int b = 230; b >= 130; b--) {
            snprintf(sym, sizeof(sym), "x265_api_get_%d", b);
            p_x265_api_get = (const x265_api *(*)(int))dlsym(h, sym);
            if (p_x265_api_get) break;
        }
    }
    if (p_x265_api_get) {
        const x265_api *api = p_x265_api_get(0);
        if (api) {
            x265_ops.param_default_preset = api->param_default_preset;
            x265_ops.param_apply_profile = api->param_apply_profile;
            x265_ops.encoder_open = api->encoder_open;
            x265_ops.encoder_reconfig = api->encoder_reconfig;
            x265_ops.picture_init = api->picture_init;
            x265_ops.encoder_encode = api->encoder_encode;
            x265_ops.encoder_close = api->encoder_close;
        }
    }

    /* 2. For any symbols not retrieved via x265_api_get, resolve individually */
    if (!x265_ops.param_default_preset)
        x265_ops.param_default_preset = (int (*)(x265_param *, const char *, const char *))dlsym(h, "x265_param_default_preset");
    if (!x265_ops.param_apply_profile)
        x265_ops.param_apply_profile = (int (*)(x265_param *, const char *))dlsym(h, "x265_param_apply_profile");
    if (!x265_ops.encoder_reconfig)
        x265_ops.encoder_reconfig = (int (*)(x265_encoder *, x265_param *))dlsym(h, "x265_encoder_reconfig");
    if (!x265_ops.picture_init)
        x265_ops.picture_init = (void (*)(x265_param *, x265_picture *))dlsym(h, "x265_picture_init");
    if (!x265_ops.encoder_encode)
        x265_ops.encoder_encode = (int (*)(x265_encoder *, x265_nal **, uint32_t *, x265_picture *, x265_picture *))dlsym(h, "x265_encoder_encode");
    if (!x265_ops.encoder_close)
        x265_ops.encoder_close = (void (*)(x265_encoder *))dlsym(h, "x265_encoder_close");

    if (!x265_ops.encoder_open) {
        x265_ops.encoder_open = (x265_encoder *(*)(x265_param *))dlsym(h, "x265_encoder_open");
#ifdef X265_BUILD
        if (!x265_ops.encoder_open) {
            char sym[64];
            snprintf(sym, sizeof(sym), "x265_encoder_open_%d", X265_BUILD);
            x265_ops.encoder_open = (x265_encoder *(*)(x265_param *))dlsym(h, sym);
        }
#endif
        if (!x265_ops.encoder_open) {
            char sym[64];
            for (int b = 230; b >= 130; b--) {
                snprintf(sym, sizeof(sym), "x265_encoder_open_%d", b);
                x265_ops.encoder_open = (x265_encoder *(*)(x265_param *))dlsym(h, sym);
                if (x265_ops.encoder_open) break;
            }
        }
    }

    if (!x265_ops.param_default_preset ||
        !x265_ops.param_apply_profile ||
        !x265_ops.encoder_open ||
        !x265_ops.encoder_reconfig ||
        !x265_ops.picture_init ||
        !x265_ops.encoder_encode ||
        !x265_ops.encoder_close) {
        fprintf(stderr, "[bc250-x265] Failed to resolve required symbols from libx265\n");
        dlclose(h);
        memset(&x265_ops, 0, sizeof(x265_ops));
        return;
    }

    x265_ops.handle = h;
    x265_available = true;
    fprintf(stderr, "[bc250-x265] Successfully loaded libx265 via runtime dlopen\n");
}

struct hevc_x265 {
    x265_encoder *h;
    x265_param param;
    hevc_x265_config_t applied;
    const char *preset;
    int threads;
    int64_t pts;
    uint8_t *i420_u;
    uint8_t *i420_v;
    size_t i420_cap;
};

static int threads_for(const hevc_x265_config_t *cfg)
{
    const char *env = getenv("BC250_X265_THREADS");
    if (!env || !*env) env = getenv("BC250_HEVC_THREADS");
    if (!env || !*env) env = getenv("BC250_THREADS");
    if (env && *env) {
        int t = atoi(env);
        if (t >= 0 && t <= 64) return t;
    }
    (void)cfg;
    return 4;
}

static void set_rate(x265_param *p, const hevc_x265_config_t *cfg)
{
    p->rc.vbvMaxBitrate = 0;
    p->rc.vbvBufferSize = 0;
    p->bEmitHRDSEI = 0;

    if (cfg->rc_mode == RC_CQP) {
        p->rc.rateControlMode = X265_RC_CQP;
        p->rc.qp = cfg->qp > 0 && cfg->qp <= 51 ? cfg->qp : 26;
        return;
    }
    if (cfg->crf > 0) {
        p->rc.rateControlMode = X265_RC_CRF;
        float crf_val = (float)(cfg->crf > 51 ? 51 : cfg->crf);
        const char *env_crf = getenv("BC250_X265_CRF");
        if (env_crf && *env_crf) {
            float env_val = (float)atof(env_crf);
            if (env_val >= 0.0f && env_val <= 51.0f) crf_val = env_val;
        }
        p->rc.rfConstant = (double)crf_val;
        return;
    }

    p->rc.rateControlMode = X265_RC_ABR;
    p->rc.bitrate = cfg->bitrate ? (int)(cfg->bitrate / 1000) : 5000;
    if (cfg->rc_mode == RC_LOW_LATENCY) {
        p->rc.vbvMaxBitrate = p->rc.bitrate;
        p->rc.vbvBufferSize = p->rc.bitrate / (cfg->fps ? (int)cfg->fps : 30);
    } else {
        p->rc.vbvMaxBitrate = p->rc.bitrate * 3 / 2;
        p->rc.vbvBufferSize = p->rc.bitrate;
    }
}

static int open_encoder(hevc_x265_t *x, const hevc_x265_config_t *cfg)
{
    if (x->h) {
        x265_ops.encoder_close(x->h);
        x->h = NULL;
    }
    x->preset = hevc_x265_preset_for(cfg);
    x->threads = threads_for(cfg);

    const char *tune = cfg->live ? "zerolatency" : NULL;
    if (x265_ops.param_default_preset(&x->param, x->preset, tune) < 0) {
        fprintf(stderr, "[bc250-x265] unknown preset '%s'\n", x->preset);
        return -1;
    }

    x->param.logLevel = X265_LOG_ERROR;
    x->param.sourceWidth = (int)cfg->width;
    x->param.sourceHeight = (int)cfg->height;
    x->param.internalCsp = X265_CSP_I420;
    x->param.internalBitDepth = cfg->ten_bit ? 10 : 8;
    x->param.fpsNum = cfg->fps ? cfg->fps : 30;
    x->param.fpsDenom = 1;
    x->param.frameNumThreads = cfg->live ? 1 : x->threads;
    x->param.keyframeMax = cfg->gop ? (int)cfg->gop : (int)x->param.fpsNum;
    x->param.bRepeatHeaders = 1;
    x->param.bAnnexB = 1;

    if (cfg->live) {
        x->param.frameNumThreads = 1;
        x->param.bFrameAdaptive = 0;
        x->param.bframes = 0;
        x->param.rc.cuTree = 0;
        x->param.lookaheadDepth = 0;
        x->param.bEnableWavefront = is_steam_caller() ? 0 : 1;
    } else {
        const char *env_b = getenv("BC250_BFRAMES");
        if (env_b && *env_b) {
            x->param.bframes = atoi(env_b);
        } else {
            x->param.bframes = 2; /* 2 B-frames for offline transcode efficiency */
        }
    }

    set_rate(&x->param, cfg);

    if (cfg->ten_bit) {
        x265_ops.param_apply_profile(&x->param, "main10");
    } else {
        x265_ops.param_apply_profile(&x->param, "main");
    }

    x->h = x265_ops.encoder_open(&x->param);
    if (!x->h) {
        fprintf(stderr, "[bc250-x265] x265_encoder_open failed (%ux%u, %s, %s)\n",
                cfg->width, cfg->height, x->preset, cfg->ten_bit ? "10-bit" : "8-bit");
        return -1;
    }

    x->applied = *cfg;
    x->pts = 0;
    return 0;
}

hevc_x265_t *hevc_x265_create(void)
{
    pthread_once(&x265_once, load_x265_ops);
    if (!x265_available) return NULL;
    hevc_x265_t *x = calloc(1, sizeof(*x));
    return x;
}

static bool same_structure(const hevc_x265_config_t *a, const hevc_x265_config_t *b)
{
    return a->width == b->width && a->height == b->height && a->fps == b->fps
        && a->ten_bit == b->ten_bit && a->live == b->live && a->rc_mode == b->rc_mode;
}

static bool same_rate(const hevc_x265_config_t *a, const hevc_x265_config_t *b)
{
    return a->rc_mode == b->rc_mode && a->bitrate == b->bitrate && a->qp == b->qp
        && a->crf == b->crf && a->cbr_intent == b->cbr_intent;
}

int hevc_x265_encode(hevc_x265_t *x, const hevc_x265_config_t *cfg,
                     const uint8_t *y, int y_stride,
                     const uint8_t *uv, int uv_stride,
                     bool force_idr, int frame_qp,
                     uint8_t *out, size_t out_cap)
{
    if (!x || !cfg || !y || !uv || !out || out_cap == 0) return -1;

    if (!x->h || !same_structure(cfg, &x->applied)) {
        if (open_encoder(x, cfg) < 0) return -1;
    } else if (!same_rate(cfg, &x->applied)) {
        set_rate(&x->param, cfg);
        if (x265_ops.encoder_reconfig(x->h, &x->param) < 0) {
            if (open_encoder(x, cfg) < 0) return -1;
        }
        x->applied = *cfg;
    }

    /* Prepare planar I420 chroma buffers from interleaved UV */
    size_t plane_w = cfg->width / 2;
    size_t plane_h = cfg->height / 2;
    size_t bytes_per_sample = cfg->ten_bit ? 2 : 1;
    size_t plane_sz = plane_w * plane_h * bytes_per_sample;

    if (plane_sz > x->i420_cap) {
        free(x->i420_u);
        free(x->i420_v);
        x->i420_u = malloc(plane_sz);
        x->i420_v = malloc(plane_sz);
        if (!x->i420_u || !x->i420_v) {
            x->i420_cap = 0;
            return -1;
        }
        x->i420_cap = plane_sz;
    }

    if (!cfg->ten_bit) {
        /* De-interleave NV12 to I420 */
        for (size_t r = 0; r < plane_h; r++) {
            const uint8_t *src_uv_row = uv + r * (size_t)uv_stride;
            uint8_t *dst_u_row = x->i420_u + r * plane_w;
            uint8_t *dst_v_row = x->i420_v + r * plane_w;
            for (size_t c = 0; c < plane_w; c++) {
                dst_u_row[c] = src_uv_row[c * 2];
                dst_v_row[c] = src_uv_row[c * 2 + 1];
            }
        }
    } else {
        /* De-interleave P010 (16-bit) to 10-bit I420 */
        for (size_t r = 0; r < plane_h; r++) {
            const uint16_t *src_uv_row = (const uint16_t *)(uv + r * (size_t)uv_stride);
            uint16_t *dst_u_row = (uint16_t *)(x->i420_u + r * plane_w * 2);
            uint16_t *dst_v_row = (uint16_t *)(x->i420_v + r * plane_w * 2);
            for (size_t c = 0; c < plane_w; c++) {
                dst_u_row[c] = src_uv_row[c * 2] >> 6;
                dst_v_row[c] = src_uv_row[c * 2 + 1] >> 6;
            }
        }
    }

    x265_picture pic_in;
    x265_ops.picture_init(&x->param, &pic_in);
    pic_in.colorSpace = X265_CSP_I420;
    pic_in.pts = x->pts++;
    pic_in.planes[0] = (void *)y;
    pic_in.stride[0] = y_stride;
    pic_in.planes[1] = x->i420_u;
    pic_in.stride[1] = (int)(plane_w * bytes_per_sample);
    pic_in.planes[2] = x->i420_v;
    pic_in.stride[2] = (int)(plane_w * bytes_per_sample);

    if (force_idr) {
        pic_in.sliceType = X265_TYPE_IDR;
    }

    if (frame_qp > 0 && cfg->rc_mode == RC_CQP) {
        pic_in.forceqp = frame_qp;
    }

    x265_nal *nal = NULL;
    uint32_t pi_nal = 0;
    int nbytes = x265_ops.encoder_encode(x->h, &nal, &pi_nal, &pic_in, NULL);
    if (nbytes < 0) return -1;

    size_t total = 0;
    for (uint32_t i = 0; i < pi_nal; i++) {
        if (total + nal[i].sizeBytes > out_cap) break;
        memcpy(out + total, nal[i].payload, nal[i].sizeBytes);
        total += nal[i].sizeBytes;
    }

    return (int)total;
}

void hevc_x265_destroy(hevc_x265_t *x)
{
    if (!x) return;
    if (x->h && x265_ops.encoder_close) x265_ops.encoder_close(x->h);
    free(x->i420_u);
    free(x->i420_v);
    free(x);
}

#else /* !BC250_HAVE_X265 */

hevc_x265_t *hevc_x265_create(void) { return NULL; }
int hevc_x265_encode(hevc_x265_t *x, const hevc_x265_config_t *cfg,
                     const uint8_t *y, int y_stride,
                     const uint8_t *uv, int uv_stride,
                     bool force_idr, int frame_qp,
                     uint8_t *out, size_t out_cap) {
    (void)x; (void)cfg; (void)y; (void)y_stride; (void)uv; (void)uv_stride;
    (void)force_idr; (void)frame_qp; (void)out; (void)out_cap;
    return -1;
}
void hevc_x265_destroy(hevc_x265_t *x) { (void)x; }

#endif /* BC250_HAVE_X265 */
