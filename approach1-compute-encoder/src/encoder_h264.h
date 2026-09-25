/* bc250-encoding-decoding-fix v0.4.3 - https://github.com/simpmix/bc250-encoding-decoding-fix */
/*
 * Copyright (c) 2026 BC-250 Project Contributors
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * encoder_h264.h - H.264/AVC Compute Shader Encoder API
 */

#ifndef BC250_ENCODER_H264_H
#define BC250_ENCODER_H264_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "gpu_compute.h"
#include "rate_control.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct h264_encoder h264_encoder_t;

/**
 * h264_encoder_create - Allocate and configure an H.264 encoder
 * @gpu_ctx: Vulkan compute context
 * @width: Frame width in pixels
 * @height: Frame height in pixels
 * @fps: Framerate (e.g. 30 or 60)
 * @bitrate: Target bitrate in bits per second
 * @profile: VAProfile (Baseline, Main, High)
 */
h264_encoder_t *h264_encoder_create(bc250_gpu_context_t *gpu_ctx,
                                    uint32_t width, uint32_t height,
                                    uint32_t fps, uint32_t bitrate,
                                    int profile);

/**
 * h264_encoder_encode_frame - Encodes a frame to H.264 Annex B byte stream
 * @encoder: Encoder context
 * @gpu_ctx: Vulkan compute context
 * @input_surface: Input GPU surface
 * @output_buf: Destination buffer for NALUs
 * @output_size: Size of output buffer
 *
 * Returns number of bytes written, or -1 on error.
 */
int h264_encoder_encode_frame(h264_encoder_t *encoder,
                              bc250_gpu_context_t *gpu_ctx,
                              gpu_image_t input_surface,
                              uint8_t *output_buf, size_t output_size);

/**
 * Everything h264_encoder_finish_frame() needs to know about a frame whose GPU
 * work h264_encoder_submit_frame() already put in flight. Small on purpose: the
 * bulk per-frame GPU output does NOT live here, it stays in the gpu context's
 * double-buffered staging slot until the matching finish reads it.
 */
typedef struct {
    bool     valid;          /* a submit filled this in and no finish consumed it yet */
    bool     gpu_submitted;  /* dispatch+submit actually succeeded; if false there is
                              * no fence to wait on and no fresh staging data - the
                              * finish must take the same safe all-P_Skip fallback the
                              * synchronous path takes (see encoder_h264.c) */
    bool     is_idr;
    int      qp;
    int      num_slices;
    /* Which GPU double-buffer slot THIS frame was submitted into. Captured at
     * submit time because by finish time another frame may already have been
     * submitted, making "the most recently submitted slot" the wrong one - see
     * gpu_compute_submitted_slot()'s comment for why that mistake is invisible
     * to the mask audit. */
    int      gpu_slot;
} h264_pending_frame_t;

/**
 * h264_encoder_submit_frame - first half of an encode: choose frame type/QP and
 * put this frame's GPU compute work in flight, WITHOUT waiting for it.
 *
 * Writes no bitstream at all; every byte of output is produced by the matching
 * h264_encoder_finish_frame(). Pair each submit with exactly one finish, in
 * order. Splitting the two is what lets frame N+1's GPU work overlap frame N's
 * CPU entropy coding (the CPU is ~8ms of a ~12.5ms frame and the GPU ~4.2ms of
 * it, and they were strictly serial before).
 *
 * NOTE ON RATE CONTROL: QP is chosen here because the quantize shader needs it
 * at dispatch time, so in pipelined use frame N's QP is picked before frame
 * N-1's bits have been accounted. That is ordinary for a pipelined encoder but
 * it IS a behaviour change - pipelined output is therefore not bit-identical to
 * the synchronous path, by construction, and must not be validated with a
 * byte-exactness oracle against it.
 *
 * Returns 0 on success (pending filled in), -1 on bad arguments.
 */
int h264_encoder_encode_frame_ext(h264_encoder_t *encoder,
                                  bc250_gpu_context_t *gpu_ctx,
                                  gpu_image_t input_surface,
                                  gpu_memory_t input_memory,
                                  uint8_t *output_buf, size_t output_size);

int h264_encoder_submit_frame(h264_encoder_t *encoder,
                              bc250_gpu_context_t *gpu_ctx,
                              gpu_image_t input_surface,
                              h264_pending_frame_t *pending);

int h264_encoder_submit_frame_ext(h264_encoder_t *encoder,
                                  bc250_gpu_context_t *gpu_ctx,
                                  gpu_image_t input_surface,
                                  gpu_memory_t input_memory,
                                  h264_pending_frame_t *pending);

int h264_encoder_get_governor_tier(const h264_encoder_t *encoder);

/**
 * h264_encoder_finish_frame - second half: wait for the submitted GPU work,
 * read it back, and produce the whole Annex B frame.
 *
 * @pending: the state filled in by the matching h264_encoder_submit_frame().
 *
 * Returns number of bytes written, or -1 on error.
 */
int h264_encoder_finish_frame(h264_encoder_t *encoder,
                              bc250_gpu_context_t *gpu_ctx,
                              uint8_t *output_buf, size_t output_size,
                              const h264_pending_frame_t *pending);

/**
 * h264_encoder_force_idr - Request next frame to be an instantaneous decoder refresh (IDR)
 */
void h264_encoder_force_idr(h264_encoder_t *encoder);

/**
 * h264_encoder_set_bitrate - Dynamically adjust target bitrate
 */
void h264_encoder_set_bitrate(h264_encoder_t *encoder, uint32_t bitrate_bps);

/**
 * h264_encoder_set_rc_mode - Set rate control mode (RC_CBR, RC_VBR, RC_LOW_LATENCY, RC_CQP)
 */
void h264_encoder_set_rc_mode(h264_encoder_t *encoder, rc_mode_t mode);
rc_mode_t h264_encoder_get_rc_mode(const h264_encoder_t *encoder);

/**
 * h264_encoder_set_gop_size - Configure keyframe (IDR) interval
 */
void h264_encoder_set_gop_size(h264_encoder_t *encoder, uint32_t gop_size);

/* Apply the SPS frame-cropping window the client asked for.
 *
 * This exists because vaCreateContext() does NOT carry the display size for
 * H.264. ffmpeg aligns the context dimensions to a macroblock before calling
 * it - a 1920x1080 encode arrives here as picture=1920x1088 - so the driver
 * cannot derive the crop from the geometry it was handed. It CAN read it from
 * VAEncSequenceParameterBufferH264, which carries frame_cropping_flag and the
 * four offsets, and which ffmpeg fills in correctly; the driver simply was not
 * looking at those fields.
 *
 * That matters here more than in a normal VA-API driver because this one
 * does not advertise packed headers and writes its own SPS. ffmpeg's crop
 * therefore has no other route into the bitstream: unfixed, a 1080p request
 * produces a stream that decodes as 1920x1088.
 *
 * Offsets are in the bitstream's own units (CropUnitX/CropUnitY - for 4:2:0
 * frame-only that is 2 luma samples each), i.e. exactly as the syntax element
 * is coded, so they are passed through unscaled. */
void h264_encoder_set_cropping(h264_encoder_t *encoder, int enable,
                               uint32_t left, uint32_t right,
                               uint32_t top, uint32_t bottom);

/**
 * h264_encoder_set_num_slices - Configure number of slices per frame (1..16)
 *
 * Slices are partitioned into disjoint macroblock ranges. When num_slices > 1,
 * CPU entropy coding (CABAC/CAVLC) is parallelized across slices using OpenMP.
 */
void h264_encoder_set_num_slices(h264_encoder_t *encoder, int num_slices);

/**
 * h264_encoder_get_num_slices - Get configured number of slices per frame
 */
int h264_encoder_get_num_slices(const h264_encoder_t *encoder);

/**
 * h264_encoder_set_cbr_intent - Tell the encoder whether the caller has
 * requested a genuine constant-bitrate contract (as opposed to VBR, where a
 * requested bitrate is a loose ceiling and using fewer bits than that
 * ceiling when content doesn't need them is correct, not a bug - see
 * docs/rate_control_audit.md).
 *
 * Only when this is true, and only for the shortfall between what real
 * coded content used and rate_control.c's per-frame target, does the
 * encoder emit spec-defined filler_data_rbsp() padding NALs
 * (encoder_h264.c's maybe_append_filler()) to actually reach the target.
 * Defaults to false at h264_encoder_create() - a caller that never calls
 * this (or an intermediate layer that doesn't wire it up) gets today's
 * pre-existing behavior: no padding, ever.
 *
 * va_backend.c is the only real caller: it derives this from
 * VAEncMiscParameterRateControl.target_percentage (real CBR requests -
 * VA_RC_CBR mode - are the case ffmpeg's h264_vaapi signals with
 * target_percentage=100 and the recent rate-control-accuracy fix's own
 * board logs confirmed as "RC target: 100% of X bps"; its default VBR
 * invocation sends 50%) and honors
 * VAEncMiscParameterRateControl.rc_flags.bits.disable_bit_stuffing (the
 * VA-API's own explicit "don't pad" signal) when set. This is a narrower,
 * additive signal, not a fix for docs/rate_control_audit.md section 4
 * point 5 (this driver still hardcodes rate_control_t.mode to RC_CBR
 * everywhere and never actually negotiates VA_RC_VBR from the VAConfig) -
 * see that function's own comment for why target_percentage was chosen
 * over plumbing the VAConfig's negotiated rate-control mode through.
 */
void h264_encoder_set_cbr_intent(h264_encoder_t *encoder, bool cbr_intent);

/**
 * h264_encoder_set_fps - Dynamically update framerate
 */
void h264_encoder_set_fps(h264_encoder_t *encoder, uint32_t fps);
uint32_t h264_encoder_get_fps(const h264_encoder_t *encoder);
uint32_t h264_encoder_get_bitrate(const h264_encoder_t *encoder);

/**
 * h264_encoder_set_qp - Set constant/base quantization parameter (0..51)
 */
void h264_encoder_set_qp(h264_encoder_t *encoder, int qp);
int h264_encoder_get_qp(const h264_encoder_t *encoder);

/**
 * h264_encoder_set_quality_level - Set encoding quality/speed preset (1..7)
 * 1 = Highest quality/slowest, 4 = Balanced, 7 = Highest speed/fastest
 */
void h264_encoder_set_quality_level(h264_encoder_t *encoder, uint32_t quality_level);
uint32_t h264_encoder_get_quality_level(const h264_encoder_t *encoder);

/**
 * h264_encoder_set_max_frame_size - Set maximum frame size in bits (0 = unlimited)
 */
void h264_encoder_set_max_frame_size(h264_encoder_t *encoder, uint32_t max_frame_bits);
uint32_t h264_encoder_get_max_frame_size(const h264_encoder_t *encoder);

/**
 * h264_encoder_encode_raw - Encodes a raw NV12 image frame with pattern/content analysis
 * @encoder: Encoder context
 * @y_plane: Host pointer to Y plane data
 * @y_pitch: Row pitch of Y plane in bytes
 * @uv_plane: Host pointer to interleaved UV plane data
 * @uv_pitch: Row pitch of UV plane in bytes
 * @output_buf: Destination buffer for NALUs
 * @output_size: Size of output buffer
 *
 * Returns number of bytes written, or -1 on error.
 */
int h264_encoder_encode_raw(h264_encoder_t *encoder,
                            const uint8_t *y_plane, int y_pitch,
                            const uint8_t *uv_plane, int uv_pitch,
                            uint8_t *output_buf, size_t output_size);

/**
 * h264_encoder_destroy - Teardown and free resources
 */
/* VA's ICQ: constant quality at this factor (1..51), 0 to go back to the
 * bitrate. Only the x264 backend can honour it; the compute encoder keeps
 * treating ICQ as the bitrate va_backend.c derives for it. */
void h264_encoder_set_icq_quality(h264_encoder_t *encoder, int quality);

/* Whether H.264 goes through libx264 - see encoder_x264.h. */
bool h264_encoder_uses_x264(const h264_encoder_t *encoder);

void h264_encoder_destroy(h264_encoder_t *encoder);

/**
 * h264_intra16_luma_dc_transform - forward Hadamard transform, quantize, and
 * transpose the 16 luma DC coefficients of an Intra16x16 macroblock into the
 * row/column layout cavlc_write_4x4_block() (and a real decoder) expect.
 *
 * This is the exact pure-math DC path used by the encoder's Intra16x16
 * macroblock encoding (see encoder_h264.c's encode_mb_i16x16, its only
 * production caller). It is exposed here (rather than kept static) purely
 * so it can be unit-tested in isolation without a GPU/Vulkan context - see
 * tests/test_encode.c's test_intra16_dc_transpose() regression test for the
 * transpose bug fixed in commit d95b840.
 *
 * @param dc_in                16 pre-quant luma DC values (one per luma 4x4
 *                             sub-block of the macroblock), raster
 *                             (row*4+col) order.
 * @param qp                   Quantization parameter for this macroblock.
 * @param dc_out               Output: quantized DC array in the natural
 *                             row/column order CAVLC/a real decoder expect.
 * @param dc_out_pretranspose  Optional (may be NULL): if non-NULL, filled
 *                             with the quantized array BEFORE the transpose
 *                             fix is applied. For regression testing only -
 *                             no production caller needs this.
 */
void h264_intra16_luma_dc_transform(const int dc_in[4][4], int qp,
                                     int dc_out[16], int dc_out_pretranspose[16]);

#ifdef __cplusplus
}
#endif

#endif /* BC250_ENCODER_H264_H */
