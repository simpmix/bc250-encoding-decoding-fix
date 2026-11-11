/* bc250-encoding-decoding-fix v0.4.3 - https://github.com/simpmix/bc250-encoding-decoding-fix */
/*
 * Copyright (c) 2026 BC-250 Project Contributors
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * encoder_h265.h - H.265/HEVC Compute Shader Encoder API
 */

#ifndef BC250_ENCODER_H265_H
#define BC250_ENCODER_H265_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "gpu_compute.h"
#include "rate_control.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hevc_encoder hevc_encoder_t;

hevc_encoder_t *hevc_encoder_create(bc250_gpu_context_t *gpu_ctx,
                                    uint32_t width, uint32_t height,
                                    uint32_t fps, uint32_t bitrate);

/* The same at a chosen bit depth: 8 writes Main, 10 writes Main 10 and
 * reads P010. Anything else returns NULL. hevc_encoder_create() is this
 * with 8. */
hevc_encoder_t *hevc_encoder_create_depth(bc250_gpu_context_t *gpu_ctx,
                                          uint32_t width, uint32_t height,
                                          uint32_t fps, uint32_t bitrate,
                                          int bit_depth);
int hevc_encoder_get_bit_depth(const hevc_encoder_t *encoder);

void hevc_encoder_set_force_idr(hevc_encoder_t *encoder);
void hevc_encoder_set_gop_size(hevc_encoder_t *encoder, uint32_t gop_size);
uint32_t hevc_encoder_get_gop_size(const hevc_encoder_t *encoder);
void hevc_encoder_set_qp(hevc_encoder_t *encoder, int qp);
int hevc_encoder_get_qp(const hevc_encoder_t *encoder);
void hevc_encoder_set_bitrate(hevc_encoder_t *encoder, uint32_t bitrate);
uint32_t hevc_encoder_get_bitrate(const hevc_encoder_t *encoder);
void hevc_encoder_set_fps(hevc_encoder_t *encoder, uint32_t fps);
uint32_t hevc_encoder_get_fps(const hevc_encoder_t *encoder);
void hevc_encoder_set_rc_mode(hevc_encoder_t *encoder, rc_mode_t mode);
rc_mode_t hevc_encoder_get_rc_mode(const hevc_encoder_t *encoder);
void hevc_encoder_pace_frame(hevc_encoder_t *encoder);
uint32_t hevc_encoder_get_last_frame_sad(const hevc_encoder_t *encoder);
void hevc_encoder_set_quality_level(hevc_encoder_t *encoder, uint32_t quality_level);
uint32_t hevc_encoder_get_quality_level(const hevc_encoder_t *encoder);
void hevc_encoder_set_max_frame_size(hevc_encoder_t *encoder, uint32_t max_frame_bits);
uint32_t hevc_encoder_get_max_frame_size(const hevc_encoder_t *encoder);
int hevc_encoder_get_governor_tier(const hevc_encoder_t *encoder);
void hevc_encoder_set_cbr_intent(hevc_encoder_t *encoder, bool cbr_intent);

/*
 * @input_memory: backing device memory of `input_surface`, needed for the
 * real gpu_compute_download_nv12() readback this encoder does (see
 * encoder_h265.c's top comment) - unlike the H.264 encoder above, this one
 * needs real host-visible pixel data because HEVC's mandatory 4x4-luma-
 * intra DST-VII transform and per-block reconstruction chaining aren't
 * something the existing GPU shaders compute (see hevc_intra.h).
 */
int hevc_encoder_encode_frame(hevc_encoder_t *encoder,
                              bc250_gpu_context_t *gpu_ctx,
                              gpu_image_t input_surface,
                              gpu_memory_t input_memory,
                              uint8_t *output_buf, size_t output_size);

/**
 * hevc_encoder_encode_raw - Encode a frame straight from host NV12 pixel
 * data, with no GPU context/surface involved at all. Exists for the same
 * reason h264_encoder_encode_raw() does (see encoder_h264.h): a
 * deterministic, GPU-free path for tests and offline validation. This
 * encoder's whole per-4x4-block intra/transform/CABAC pipeline (see
 * encoder_h265.c) already runs on the CPU against host pixel data - the
 * GPU is only ever used, via hevc_encoder_encode_frame() above, to get
 * that pixel data off an already-uploaded VA-API surface - so this
 * entry point is the same core encode logic with that one GPU readback
 * step skipped.
 *
 * A ten-bit encoder takes P010 here: two bytes a sample, the sample in the
 * top ten bits. The pitches are in bytes at either depth.
 */
int hevc_encoder_encode_raw(hevc_encoder_t *encoder,
                            const uint8_t *y_plane, int y_pitch,
                            const uint8_t *uv_plane, int uv_pitch,
                            uint8_t *output_buf, size_t output_size);

void hevc_encoder_destroy(hevc_encoder_t *encoder);

#ifdef __cplusplus
}
#endif

#endif /* BC250_ENCODER_H265_H */
