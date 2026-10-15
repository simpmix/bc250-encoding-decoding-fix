# HEVC/H.265 Dedicated GPU Compute Pipeline — AMD BC-250 (Cyan Skillfish)

## 1. Overview & Architectural Context
On the AMD BC-250 (Cyan Skillfish / RDNA 1.5), the physical Video Core Next (VCN) hardware block is permanently eFused off at the factory.
Previous attempts to reuse H.264 shaders (`motion_estimation.comp`, `residual_predict.comp`, etc.) for HEVC failed due to fundamental codec differences:
- HEVC relies on Coding Tree Units (CTUs) and Coding Units (CUs down to 8x8), whereas H.264 operates on fixed 16x16 macroblocks.
- HEVC specifies DST-VII for 4x4 intra luma residuals (Rec. ITU-T H.265 clause 8.6.4.2) and factored DCT-II for other block sizes.
- Unaligned dispatches previously introduced CPU stalls and fence timeouts (3 to 16 ms) without contributing usable vectors.

This implementation delivers a dedicated Vulkan Compute pipeline tailored for HEVC on the BC-250's 40 Compute Units.

## 2. Dedicated Shaders

1. **`hevc_motion_estimation.comp`**
   - Operates on 8x8 HEVC CUs (`HEVC_CU_SIZE = 8`).
   - Wave64 subgroup architecture matching AMD RDNA execution waves.
   - Cooperative shared memory loads (40x40 search window for +/-16 integer search).
   - Fast zero-motion detection with early exit for static content.
   - Outputs integer motion vectors (MVx, MVy) and SAD into staging buffers.

2. **`hevc_transform.comp`**
   - Canonical 4x4 DST-VII matrix for intra luma.
   - Canonical 4x4 and 8x8 DCT-II matrices with integer fixed-point arithmetic.
   - Two-pass separable 2D forward and inverse transforms conforming to ITU-T Section 8.6.4.

3. **`hevc_quant.comp`**
   - Forward quantization with deadzone scaling.
   - Dequantization with level scale mapping `LEVEL_SCALE[qp % 6] << (qp / 6)` (Section 8.6.3).
   - Computes Coded Block Flags (CBF) and non-zero counts.

4. **`hevc_pred.comp`**
   - Inter prediction with 8-tap luma and 4-tap chroma fractional interpolation filters (Section 8.5.3).
   - Intra prediction modes: Planar (0), DC (1), Horizontal (10), Vertical (26).

5. **`hevc_reconstruct_sao.comp`**
   - Reconstructs pixels: `rec = clip(pred + res)`.
   - Sample Adaptive Offset (SAO) band offset adjustments.
   - Stores directly into GPU device-local reference surfaces.

## 3. Feature Flags & Runtime Control

| Flag | Default | Description |
| :--- | :--- | :--- |
| `BC250_HEVC_GPU_PIPELINE=1` | OFF | Master flag to activate the full HEVC GPU compute pipeline. |
| `BC250_HEVC_GPU_ME=1` | OFF | Activates GPU Motion Estimation. |
| `BC250_HEVC_GPU_TRANSFORM=1` | OFF | Activates GPU Transform pipeline. |
| `BC250_HEVC_GPU_QUANT=1` | OFF | Activates GPU Quantization / Dequantization. |
| `BC250_HEVC_GPU_PRED=1` | OFF | Activates GPU Intra/Inter Prediction. |
| `BC250_HEVC_GPU_FILTER=1` | OFF | Activates GPU Reconstruction, Deblocking, and SAO. |
| `BC250_HEVC_FORCE_CPU=1` | OFF | Forces full CPU fallback across all stages. |
| `BC250_HEVC_GPU_DEBUG=1` | OFF | Enables verbose dispatch and validation logging. |

## 4. Verification Evidence

- **SPIR-V Validation:** All 5 shaders pass `spirv-val --target-env vulkan1.1` with 0 warnings/errors.
- **Unit Tests:** `test_hevc_transform` (10/10 matrices match), `test_hevc_encode` (all structure tests pass).
- **FFmpeg Oracle:** Generated bitstreams decode with `ffmpeg -v error -i ... -f null -` with 0 syntax or bitstream errors.
- **H.264 Regression:** `test_encode` passes all multi-slice and rate-control tests.
- **CPU Fallback:** Confirmed operational with `BC250_HEVC_FORCE_CPU=1`.
