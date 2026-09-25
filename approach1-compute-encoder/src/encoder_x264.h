/*
 * Copyright (c) 2026 BC-250 Project Contributors
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * encoder_x264.h - H.264 through libx264, behind the h264_encoder_* API.
 *
 * Measured on a BC-250 through this driver and ffmpeg, 1080p, three natural
 * sequences (park_joy, crowd_run, ducks_take_off), four QPs each; bits are
 * the Bjontegaard rate against x264 veryfast run directly:
 *
 *                              bits      fps
 *   compute encoder (GPU ME)   +64%       28   plus 16 ms of GPU per frame
 *   x264 ultrafast             +90%      151
 *   x264 superfast              +7%      116
 *   x264 veryfast                0%       94
 *
 * The compute encoder was slower, larger, and kept the GPU busy for a whole
 * 60 fps frame time - the GPU the game being streamed needs. So H.264 goes
 * through x264, and the compute encoder stays behind BC250_H264_BACKEND=
 * compute.
 *
 * The encoder takes its settings as a whole on every frame and works out
 * itself what changed: a new rate is applied with x264_encoder_reconfig(),
 * anything else reopens it, which starts with an IDR.
 */
#ifndef BC250_ENCODER_X264_H
#define BC250_ENCODER_X264_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct h264_x264 h264_x264_t;

typedef struct {
    uint32_t width, height;   /* the picture itself, after cropping */
    uint32_t fps;
    uint32_t gop;             /* IDR interval in frames */
    int rc_mode;              /* rc_mode_t */
    uint32_t bitrate;         /* bits per second, bitrate modes */
    int qp;                   /* RC_CQP */
    int crf;                  /* > 0: ICQ, constant quality at this factor */
    uint32_t quality_level;   /* VA's 1 (best) .. 7 (fastest), 4 = default */
    int profile_idc;          /* 66, 77 or 100 */
    bool cbr_intent;          /* the caller wants a real CBR stream, filler included */
    bool live;                /* a streaming server: fewer threads, one-frame VBV */
} h264_x264_config_t;

h264_x264_t *h264_x264_create(void);

/* Encodes one NV12 picture. Returns the bytes written to `out`, or -1.
 * `frame_qp` > 0 forces this frame's QP in RC_CQP. */
int h264_x264_encode(h264_x264_t *x, const h264_x264_config_t *cfg,
                     const uint8_t *y, int y_stride,
                     const uint8_t *uv, int uv_stride,
                     bool force_idr, int frame_qp,
                     uint8_t *out, size_t out_size);

/* The preset the settings above resolve to - exposed for the log and tests. */
const char *h264_x264_preset_for(const h264_x264_config_t *cfg);

void h264_x264_destroy(h264_x264_t *x);

#ifdef __cplusplus
}
#endif

#endif /* BC250_ENCODER_X264_H */
