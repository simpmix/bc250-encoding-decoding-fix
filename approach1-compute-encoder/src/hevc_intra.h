/* bc250-encoding-decoding-fix v0.4.3 - https://github.com/simpmix/bc250-encoding-decoding-fix */
/*
 * Copyright (c) 2026 BC-250 Project Contributors
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * hevc_intra.h - real HEVC 4x4 intra prediction, transform, quantization
 *                and reconstruction, implemented independently against the
 *                ITU-T H.265 spec (clauses 8.4.2 intra mode derivation,
 *                8.4.4.2 intra sample prediction, 8.6.3 scaling/dequant,
 *                8.6.4 transform).
 *
 * WHY THIS EXISTS INSTEAD OF REUSING THE GPU PIPELINE: the existing Vulkan
 * compute shaders (dct_transform.comp/quantize.comp/residual_predict.comp)
 * are H.264-shaped in two ways that are NOT just block-size differences:
 *
 *   1. HEVC mandates an alternative 4x4 transform (DST-VII) for luma intra
 *      residuals specifically (ITU-T H.265 8.6.4.1/8.6.4.2) - never DCT for
 *      that one case. dct_transform.comp only ever computes DCT. Signaling
 *      DCT-domain coefficients into an HEVC bitstream as a 4x4 luma intra
 *      TU (which every conformant decoder, ffmpeg included, will inverse-
 *      DST) would decode to a real picture that LOOKS plausible but is
 *      systematically wrong - exactly the "syntactically valid, content
 *      garbage" failure mode DEVLOG.md Sec. 1/6 warns about.
 *   2. HEVC's intra prediction is always performed per-transform-block
 *      using that block's own already-reconstructed neighbors, even when
 *      several TUs share one signaled prediction mode - i.e. the actual
 *      per-pixel prediction has to be recomputed and reconstruction-chained
 *      at 4x4 granularity in coding (z-scan) order. The GPU's
 *      residual_predict.comp only ever computes ONE whole-macroblock I16x16
 *      prediction per dispatch; it has no equivalent to H.264's Intra_4x4
 *      chaining at all.
 *   3. HEVC's dequantization scale (ITU-T H.265 8.6.3: m=16, bdShift =
 *      BitDepth+Log2(nTbS)-5, levelScale={40,45,51,57,64,72}) is a
 *      different function of QP than H.264's, so even reusing the GPU's
 *      already-quantized DCT levels verbatim (setting aside point 1) would
 *      hand a decoder numbers scaled for the wrong standard.
 *
 * None of these are solvable by clever re-addressing of the existing
 * buffers - they require new transform/quant math and a new prediction
 * chaining order, so this module implements the whole per-4x4-block intra
 * codec core in portable C, run on the CPU against real picture data
 * (see encoder_h265.c, which downloads it from the GPU surface via the
 * existing, unmodified gpu_compute_download_nv12() - that part IS reused).
 */
#ifndef BC250_HEVC_INTRA_H
#define BC250_HEVC_INTRA_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The only four HEVC intra modes this encoder ever chooses (mirroring the
 * four H.264 Intra16x16 modes the GPU shaders already know how to pick
 * between by SAD, conceptually - DC, Planar, Horizontal, Vertical), applied
 * per-4x4-block with real chained reconstruction rather than GPU-computed
 * whole-macroblock prediction. See hevc_intra.c's mode-decision comment. */
#define HEVC_MODE_PLANAR      0
#define HEVC_MODE_DC          1
#define HEVC_MODE_HORIZONTAL 10
#define HEVC_MODE_VERTICAL   26

/* Rec. ITU-T H.265 Table 8-10 (scan derivation for intra 4x4/8x8 luma, and
 * 4:4:4 chroma - not applicable to our 4:2:0 chroma, which always scans
 * diagonally): dirMode in [6,14] -> SCAN_VER(2), [22,30] -> SCAN_HOR(1),
 * else SCAN_DIAG(0). Matches hevc_cabac.c's scan_idx numbering. */
int hevc_scan_idx_for_mode(int mode);

/* Real intra mode decision (SAD-minimizing among the 4 supported modes,
 * the same "no rate-distortion, SAD-only" criterion the GPU's own I16x16
 * decision uses) for one 4x4 luma block at pixel position (x0,y0). Each
 * candidate mode's prediction is built from `recon_y` (real already-
 * reconstructed neighbor pixels, or the substituted default where
 * unavailable per 8.4.4.2.2 - the same values a real decoder's own
 * prediction will see), then compared against the real SOURCE pixels at
 * (x0,y0) in `src_y` (both planes share `stride`/`width`/`height`) - the
 * comparison target for "which mode is best" is always the true picture
 * content, not the neighbor data used to build the candidate. */
int hevc_choose_luma_mode(int y_min, const uint8_t *src_y, const uint8_t *recon_y, int stride,
                           int width, int height, int x0, int y0, uint8_t pred_out[16]);

/* Derive the 3 most-probable-mode candidates for a 4x4 luma PU at (x0,y0)
 * from its already-decided left/above neighbor block modes, per 8.4.2.
 * left_mode/above_mode are ignored (treated as unavailable) when the
 * neighbor is off-picture OR outside the current slice - this encoder uses
 * one slice per picture, so only the frame edges apply. Pass -1 for an
 * unavailable neighbor's mode (any value works, it is masked out by the
 * avail flags). */
void hevc_derive_mpm(int left_mode, int left_avail, int above_mode, int above_avail,
                      int mpm_out[3]);

/* Predict one 4x4 block (luma if is_luma, else one of Cb/Cr) at pixel
 * position (x0,y0) in a `stride`-wide plane of size width x height, using
 * already-reconstructed neighbor samples (recon_plane) and Rec. ITU-T
 * H.265 8.4.4.2.2's neighbor-substitution + 8.4.4.2.5-7's Planar/DC/
 * angular sample derivation (including the DC/Horizontal/Vertical luma
 * edge-filtering steps - chroma never gets edge-filtered, matching this
 * project's existing H.264 "chroma is simpler" precedent). Writes 16
 * predicted samples, row-major (pred[y*4+x]). */
void hevc_predict_4x4(const uint8_t *recon_plane, int stride, int width, int height,
                      int x0, int y0, int mode, int is_luma, int y_min,
                      uint8_t pred_out[16]);

/* Rec. ITU-T H.265 Table 8-10: the qPi -> QpC mapping for
 * ChromaArrayType == 1 (4:2:0). Chroma is NOT quantized at the luma QP -
 * the mapping is the identity below 30, compresses 30..43, and is qPi - 6
 * above that. Pass the result, not QpY, as the `qp` argument of
 * hevc_transform_quant_4x4()/hevc_dequant_itransform_4x4() for a chroma
 * block.
 *
 * Omitting this causes chroma reconstruction to diverge from the standard
 * when QP >= 30, producing a 6-7 dB chroma PSNR deficit. */
int hevc_chroma_qp_from_luma(int qp_luma);

/* The quantizers' rounding offset, `round_q12`, in twelfths of a step: a
 * level is floor(|c| / step + round_q12 / 12). Six rounds to nearest; less
 * leaves a dead zone around zero, where a coefficient worth little costs
 * more bits than the distortion it saves. HM's offsets are 1/6 for inter
 * and 1/3 for intra; measured here, 1/6 for inter, 1/3 for intra in an I
 * picture and 1/4 for intra in a P picture came out best. */
#define HEVC_QUANT_ROUND_NEAREST 6
#define HEVC_QUANT_ROUND_INTER   2
#define HEVC_QUANT_ROUND_INTRA_I 4
#define HEVC_QUANT_ROUND_INTRA_P 3

/* Forward transform (DST-VII if use_dst, else DCT-II) + real HEVC
 * quantization (8.6.3) of a 4x4 pixel-domain residual (row-major,
 * residual[y*4+x] = source-prediction, may be negative). Writes 16
 * quantized signed coefficient levels, row-major, clipped to int16. */
void hevc_transform_quant_4x4(const int16_t residual[16], int qp, int use_dst, int round_q12,
                               int16_t coeff_out[16]);

/* Real HEVC dequantization (8.6.3) + inverse transform (8.6.4) of 16
 * quantized levels (row-major) back to a pixel-domain residual - the exact
 * same computation a real decoder performs, used here for this encoder's
 * own reconstruction chaining (see hevc_intra.h's top comment, point 2). */
void hevc_dequant_itransform_4x4(const int16_t coeff[16], int qp, int use_dst,
                                  int16_t residual_out[16]);

/* Ten bits.
 *
 * The prediction and the mode decision above take sixteen-bit samples here
 * and otherwise behave the same. The transform pair differs in its shifts,
 * and its `qp` is the PRIMED value, Qp' = QpY + 12 for luma and QpC + 12
 * for chroma (7.4.3.2.1's QpBdOffset): 0..63 instead of 0..51. See the
 * comment above their definitions in hevc_intra.c. */
void hevc_predict_4x4_10(const uint16_t *recon_plane, int stride, int width, int height,
                         int x0, int y0, int mode, int is_luma, int y_min,
                         uint16_t pred_out[16]);
int hevc_choose_luma_mode_10(int y_min, const uint16_t *src_y, const uint16_t *recon_y,
                             int stride, int width, int height, int x0, int y0,
                             uint16_t pred_out[16]);
void hevc_transform_quant_4x4_10(const int16_t residual[16], int qp, int use_dst, int round_q12,
                                  int16_t coeff_out[16]);
void hevc_dequant_itransform_4x4_10(const int16_t coeff[16], int qp, int use_dst,
                                     int16_t residual_out[16]);

/* 8x8, for inter luma: DCT-II only, at either depth - `qp` is Qp' as
 * above, `bit_depth` 8 or 10. */
void hevc_transform_quant_8x8(const int16_t residual[64], int qp, int bit_depth, int round_q12,
                               int16_t coeff_out[64]);
void hevc_dequant_itransform_8x8(const int16_t coeff[64], int qp, int bit_depth,
                                  int16_t residual_out[64]);

/* So that code compiled once per bit depth can name either set with the
 * decoder's FUNC(): FUNC(hevc_predict_4x4) is this at eight bits and the
 * _10 function at ten. */
#define hevc_predict_4x4_8             hevc_predict_4x4
#define hevc_choose_luma_mode_8        hevc_choose_luma_mode
#define hevc_transform_quant_4x4_8     hevc_transform_quant_4x4
#define hevc_dequant_itransform_4x4_8  hevc_dequant_itransform_4x4

#ifdef __cplusplus
}
#endif
#endif /* BC250_HEVC_INTRA_H */
