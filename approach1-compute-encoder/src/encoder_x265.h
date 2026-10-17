/*
 * Copyright (c) 2026 BC-250 Project Contributors
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * encoder_x265.h - HEVC/H.265 through libx265, behind the hevc_encoder_* API.
 *
 * Provides CPU-based HEVC encoding for the AMD BC-250 APU, leaving the 40
 * Compute Units 100% free for 3D game rendering. Mirrors the design of
 * encoder_x264.h for H.264.
 */
#ifndef BC250_ENCODER_X265_H
#define BC250_ENCODER_X265_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hevc_x265 hevc_x265_t;

typedef struct {
    uint32_t width, height;   /* picture dimensions */
    uint32_t fps;
    uint32_t gop;             /* IDR/keyframe interval in frames */
    int rc_mode;              /* rc_mode_t: RC_CBR, RC_VBR, RC_LOW_LATENCY, RC_CQP */
    uint32_t bitrate;         /* bits per second */
    int qp;                   /* RC_CQP */
    int crf;                  /* Constant rate factor / ICQ */
    uint32_t quality_level;   /* VA's 1 (best) .. 7 (fastest), 4 = default */
    bool ten_bit;             /* true for Main 10 (P010), false for Main (NV12) */
    bool cbr_intent;          /* true if strict CBR with filler is desired */
    bool live;                /* true for live streaming servers (Sunshine, Steam, WiVRn) */
} hevc_x265_config_t;

/* Creates an x265 encoder wrapper instance. Dynamically loads libx265 if available.
 * Returns NULL if libx265 is not present or initialization fails. */
hevc_x265_t *hevc_x265_create(void);

/* Encodes one picture.
 * For 8-bit (NV12): y and uv point to luma and interleaved chroma.
 * For 10-bit (P010): y and uv point to 16-bit luma and interleaved chroma.
 * Returns bytes written to out, or -1 on error. */
int hevc_x265_encode(hevc_x265_t *x, const hevc_x265_config_t *cfg,
                     const uint8_t *y, int y_stride,
                     const uint8_t *uv, int uv_stride,
                     bool force_idr, int frame_qp,
                     uint8_t *out, size_t out_cap);

void hevc_x265_destroy(hevc_x265_t *x);

/* Checks if runtime libx265 provides 10-bit encoding support. */
bool hevc_x265_supports_10bit(void);

/* Returns the selected preset string for diagnostics. */
const char *hevc_x265_preset_for(const hevc_x265_config_t *cfg);

#ifdef __cplusplus
}
#endif

#endif /* BC250_ENCODER_X265_H */
