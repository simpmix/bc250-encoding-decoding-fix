/* bc250-encoding-decoding-fix v0.4.3 - https://github.com/simpmix/bc250-encoding-decoding-fix */
/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * gpu_compute.h - Vulkan compute orchestration for AMD BC-250
 */
#ifndef GPU_COMPUTE_H
#define GPU_COMPUTE_H

#include <vulkan/vulkan.h>
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    VkImage y_plane;
    VkImageView y_view;
    VkImage uv_plane;
    VkImageView uv_view;
    uint32_t width;
    uint32_t height;
    /* Tracks the real current VkImageLayout of both planes (they are always
     * transitioned together). Starts at VK_IMAGE_LAYOUT_PREINITIALIZED to match
     * the images' real initialLayout (VA-API uploads pixels via host-mapped
     * memory before the GPU ever touches them) and becomes
     * VK_IMAGE_LAYOUT_GENERAL after the first compute dispatch, where it stays
     * forever since nothing transitions the image back out of GENERAL. */
    VkImageLayout current_layout;
    /* GPU_IMAGE_NV12 or GPU_IMAGE_P010: what the two planes hold, which
     * the dma-buf export has to report and the uploads have to match. */
    int format;
} gpu_image_t;

typedef struct {
    VkDeviceMemory memory;
    VkDeviceSize size;
    void *mapped_ptr;
} gpu_memory_t;

/* Real, Vulkan-derived NV12 plane layout for an already-created+bound
 * gpu_image_t/gpu_memory_t pair - i.e. exactly the addressing that
 * gpu_compute_upload_nv12()/gpu_compute_download_nv12() already use
 * internally (vkGetImageSubresourceLayout() for each plane's real row
 * pitch/offset, plus the real inter-plane bind offset from
 * vkGetImageMemoryRequirements()+alignment). Linear-tiled Vulkan images can
 * have row padding and inter-plane alignment gaps that a naive
 * tightly-packed width/height formula does not account for - any caller
 * that needs to describe this image's memory layout to something outside
 * this file (e.g. a VAImage's pitches/offsets/data_size handed to libva)
 * must use these real values, not a naive formula, whenever that
 * description will be used to address this same memory directly. */
typedef struct {
    uint32_t y_pitch;
    uint64_t y_offset;
    uint32_t uv_pitch;
    uint64_t uv_offset;
    uint64_t total_size;
} gpu_nv12_layout_t;

typedef struct bc250_gpu_context {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue compute_queue;
    uint32_t compute_queue_family;
    VkCommandPool cmd_pool;
    VkDescriptorPool desc_pool;
    
    /* Descriptor set layouts */
    VkDescriptorSetLayout me_desc_layout;
    VkDescriptorSetLayout predict_desc_layout;
    VkDescriptorSetLayout dct_desc_layout;
    VkDescriptorSetLayout quant_desc_layout;
    VkDescriptorSetLayout deblock_desc_layout;
    VkDescriptorSetLayout entropy_desc_layout;
    VkDescriptorSetLayout cc_desc_layout;
    /* reconstruct.comp: quant_levels_buffer + coeff_buffer (readonly) +
     * pred_buffer (readonly) + recon Y/UV images (writeonly). See
     * reconstruct.comp's top-of-file comment. */
    VkDescriptorSetLayout reconstruct_desc_layout;

    /* intra_wavefront.comp: current Y/UV images (readonly), recon Y/UV
     * images (read-write - this frame's in-progress reconstruction, doubles
     * as the next frame's P-reference once the diagonal loop completes),
     * quant_levels_buffer/coeff_buffer/pred_mode_buffer (writeonly). Used
     * ONLY for I-slices, dispatched once per diagonal by
     * gpu_compute_dispatch_encode() - see intra_wavefront.comp's top-of-file
     * comment. */
    VkDescriptorSetLayout intra_wavefront_desc_layout;

    /* Pipeline layouts */
    VkPipelineLayout motion_est_layout;
    VkPipelineLayout predict_layout;
    VkPipelineLayout transform_layout;
    VkPipelineLayout quantize_layout;
    VkPipelineLayout deblock_layout;
    VkPipelineLayout entropy_layout;
    VkPipelineLayout color_convert_layout;
    /* VAEntrypointVideoProc: scale and crop, one pipeline per bit depth.
     * See gpu_compute_video_proc(). */
    VkDescriptorSetLayout vpp_desc_layout;
    VkDescriptorSet vpp_desc_set;
    VkPipelineLayout vpp_layout;
    VkPipelineLayout reconstruct_layout;
    VkPipelineLayout intra_wavefront_layout;

    /* Compute pipelines */
    VkPipeline motion_est_pipeline;
    VkPipeline predict_pipeline;
    VkPipeline transform_pipeline;
    VkPipeline quantize_pipeline;
    VkPipeline deblock_pipeline;
    VkPipeline entropy_pipeline;
    VkPipeline color_convert_pipeline;
    VkPipeline vpp_pipeline;      /* video_proc.comp, eight bit */
    VkPipeline vpp_pipeline10;    /* video_proc10.comp, ten bit */
    VkPipeline reconstruct_pipeline;
    VkPipeline intra_wavefront_pipeline;

    /* Descriptor sets */
    VkDescriptorSet me_desc_set;
    VkDescriptorSet predict_desc_set;
    VkDescriptorSet dct_desc_set;
    VkDescriptorSet quant_desc_set;
    VkDescriptorSet deblock_desc_set;
    VkDescriptorSet entropy_desc_set;
    VkDescriptorSet cc_desc_set;
    VkDescriptorSet reconstruct_desc_set;
    VkDescriptorSet intra_wavefront_desc_set;

    /* Encoding Buffers */
    VkBuffer mv_buffer;
    VkDeviceMemory mv_memory;
    
    VkBuffer residual_buffer;
    VkDeviceMemory residual_memory;
    
    VkBuffer coeff_buffer;
    VkDeviceMemory coeff_memory;
    
    VkBuffer quant_levels_buffer;
    VkDeviceMemory quant_levels_memory;
    
    VkBuffer nz_count_buffer;
    VkDeviceMemory nz_count_memory;

    VkBuffer entropy_buffer;
    VkDeviceMemory entropy_memory;

    /* Prediction value retained by residual_predict.comp (its PredOut,
     * binding 7), same size/indexing as residual_buffer - consumed by
     * reconstruct.comp so it adds the EXACT prediction value back to the
     * reconstructed residual instead of recomputing it. Device-local only;
     * no host readback needed. */
    VkBuffer pred_buffer;
    VkDeviceMemory pred_memory;

    VkBuffer staging_buffers[2];
    VkDeviceMemory staging_memories[2];
    void *staging_mapped[2];
    VkDeviceSize staging_size;

    /* Host-visible readback of the real post-quantization coefficient levels
     * (mirrors quant_levels_buffer) and pre-quantization transform
     * coefficients (mirrors coeff_buffer, needed for the I16x16 luma DC
     * Hadamard). Double-buffered the same way as staging_buffers[]/entropy_buffer. */
    VkBuffer quant_staging_buffers[2];
    VkDeviceMemory quant_staging_memories[2];
    void *quant_staging_mapped[2];
    VkDeviceSize quant_staging_size;

    /* Compact per-4x4-block PRE-quantization DC term (one int per block,
     * num_mbs*24 entries) - the only part of coeff_buffer the CPU ever reads.
     *
     * PERF: coeff_buffer used to be staged to the host in full so
     * encoder_h264.c could do the I16x16 luma DC and chroma DC Hadamards.
     * Every one of its 13 read sites was coeff_block_ptr(coeff, mb, blk)[0] -
     * position 0 only, 24 of the 384 ints per macroblock - so the staging
     * pair, the per-frame vkCmdCopyBuffer and the per-frame shadow_copy() were
     * all moving 16x more data than anything consumed. At 1440p that is 44.2 MB
     * of staging (out of a 2.65 GiB host-visible heap, the smaller half of this
     * APU's ~8 GB GART/GTT aperture - see DEVLOG §21) and 22.1 MB of copy +
     * 22.1 MB of memcpy per frame, for 1.4 MB of actually-used values.
     *
     * dct_transform.comp (P/inter) and intra_wavefront.comp (I) now write this
     * alongside their full coeff output. coeff_buffer itself stays device-local
     * - quantize.comp consumes it as input and reconstruct.comp reads it - it
     * just no longer crosses to the host. */
    VkBuffer dc_coeff_buffer;
    VkDeviceMemory dc_coeff_memory;
    VkBuffer dc_staging_buffers[2];
    VkDeviceMemory dc_staging_memories[2];
    void *dc_staging_mapped[2];
    VkDeviceSize dc_staging_size;

    /* Per-MB chosen I16x16 prediction mode (see residual_predict.comp),
     * device buffer + host-visible readback, same double-buffer contract as
     * quant_staging_buffers/coeff_staging_buffers above. */
    VkBuffer pred_mode_buffer;
    VkDeviceMemory pred_mode_memory;
    VkBuffer pred_mode_staging_buffers[2];
    VkDeviceMemory pred_mode_staging_memories[2];
    void *pred_mode_staging_mapped[2];
    VkDeviceSize pred_mode_staging_size;

    /* Host-visible readback of the real per-MB motion vectors motion_estimation.comp
     * writes to mv_buffer (mv_buffer itself is device-local only and was never
     * readable from the CPU before this). Needed so the CPU CAVLC writer can compute
     * a real spec MVD (median-of-neighbors predictor) instead of a heuristic. Same
     * double-buffer contract as the other staging buffers. */
    VkBuffer mv_staging_buffers[2];
    VkDeviceMemory mv_staging_memories[2];
    void *mv_staging_mapped[2];
    VkDeviceSize mv_staging_size;

    /* Host-visible readback of quantize.comp's per-4x4-block nonzero BITMASK
     * (nz_count_buffer, one uint32 per block, bit i set iff levels[i] != 0).
     *
     * PERF: this is 1/16th the size of quant_staging_buffers and answers every
     * "does this block/MB have a nonzero coefficient" question the CPU asks -
     * questions that previously each rescanned the full 64-byte block out of
     * the 22MB (at 1440p) quant_levels readback. The shader already computed
     * this information per block and threw it away; nothing read
     * nz_count_buffer at all before. Same double-buffer contract as the other
     * staging buffers. */
    VkBuffer nz_staging_buffers[2];
    VkDeviceMemory nz_staging_memories[2];
    void *nz_staging_mapped[2];
    VkDeviceSize nz_staging_size;

    /* Reconstructed frame for DPB */
    gpu_image_t recon_image;
    gpu_memory_t recon_memory;
    bool has_recon_frame;

    /* Double-buffering for pipeline overlap */
    VkCommandBuffer cmd_bufs[2];
    VkFence fences[2];
    VkSemaphore timeline_sem;
    uint64_t timeline_value;
    int current_buf;
    
    /* Frame state */
    uint32_t frame_width;
    uint32_t frame_height;
    
    /* Device properties */
    VkPhysicalDeviceProperties dev_props;
    uint32_t max_workgroup_size;

    /* Opt-in GPU per-stage timing (BC250_PERF_STATS=1) - see gpu_compute.c's
     * BC250_PERF_NUM_TIMESTAMPS comment and gpu_compute_dispatch_encode()/
     * gpu_compute_sync(). One VkQueryPool per double-buffered command
     * buffer, read back (and a "[BC250_PERF_GPU] ..." line printed to
     * stderr) once its frame's fence is known-signaled in gpu_compute_sync().
     * perf_is_intra[] records which prediction path (whole-frame-parallel
     * P-path vs diagonal-wavefront I-path) that buffer's frame took, since
     * the two paths write different subsets of the timestamp slots. */
    bool perf_stats_enabled;
    VkQueryPool timestamp_pools[2];
    double timestamp_period_ns;
    bool perf_is_intra[2];
    uint32_t perf_frame_counter;
    /* gpu_compute_sync() is called from more than one place per real frame
     * (h264_encoder_encode_frame()'s own EndPicture-driven encode, AND
     * va_backend.c's bc250_SyncSurface()) - both calls are cheap/correct
     * (the second just re-waits on an already-signaled fence), but without
     * this flag the perf-stats printer would read+print the same buffer's
     * still-valid query results again on every redundant call. Set true
     * right after gpu_compute_dispatch_encode() resets+writes this buffer's
     * queries; cleared after the first successful readback. */
    bool perf_result_pending[2];

    /* VK_KHR_external_memory_fd's vkGetMemoryFdKHR, resolved once at device
     * creation via vkGetDeviceProcAddr() - see gpu_compute_export_nv12_dmabuf().
     * NULL if the device extension wasn't available (callers must check). */
    PFN_vkGetMemoryFdKHR get_memory_fd_khr;

    /* VK_KHR_external_semaphore_fd's vkImportSemaphoreFdKHR, resolved once at
     * device creation the same way as get_memory_fd_khr above. Lets this
     * driver explicitly wait, before it reads a VA-API render-target surface
     * as encoder input, on whatever GPU work last wrote into that same
     * memory through a *different* API context - e.g. Sunshine's own OpenGL
     * rendering into this surface via its own EGL/GL import of the dma-buf
     * this driver exported for it (see bc250_ExportSurfaceHandle()). Without
     * this, this driver's Vulkan compute dispatch has no explicit ordering
     * relative to that GL write and can start reading the surface before
     * Mesa's radeonsi has actually finished rendering into it - see
     * gpu_compute_wait_for_image_ready()'s doc comment for the full story.
     * NULL if the device extension wasn't available (callers must check). */
    PFN_vkImportSemaphoreFdKHR import_semaphore_fd_khr;
    bool have_external_semaphore_fd;

    /* Persistent semaphore object re-used every frame as the import target
     * for gpu_compute_wait_for_image_ready(). vkImportSemaphoreFdKHR with
     * VK_SEMAPHORE_IMPORT_TEMPORARY_BIT replaces just this semaphore's
     * *payload* each call, so one long-lived handle is all that's needed -
     * no per-frame semaphore creation/destruction. */
    VkSemaphore image_ready_semaphore;

    /* Set by gpu_compute_wait_for_image_ready() when it successfully
     * imported a wait fence; consumed (and cleared) by the next
     * gpu_compute_end_picture() call, which adds image_ready_semaphore to
     * its vkQueueSubmit()'s pWaitSemaphores. */
    bool has_pending_wait_semaphore;

    /* Dynamic CPU/GPU governor latency tracking */
    struct timespec submit_time[2];
    double last_gpu_duration_ms;
} bc250_gpu_context_t;

typedef bc250_gpu_context_t gpu_context_t;

/* Core lifecycle */
int bc250_gpu_init(bc250_gpu_context_t *ctx);
void bc250_gpu_destroy(bc250_gpu_context_t *ctx);

/* What gpu_compute_create_image()'s `format` argument means. NV12 is
 * zero, which is what every caller was already passing when the argument
 * was ignored. */
#define GPU_IMAGE_NV12 0
#define GPU_IMAGE_P010 1

int gpu_compute_init(gpu_context_t *ctx);
void gpu_compute_terminate(gpu_context_t *ctx);

/* Image allocation & transfers */
int gpu_compute_create_image(gpu_context_t *ctx, int width, int height, int format, gpu_image_t *image, gpu_memory_t *memory);
void gpu_compute_destroy_image(gpu_context_t *ctx, gpu_image_t image, gpu_memory_t memory);

/* Queries the real layout described above for `image`/`memory` (both must
 * already be created and bound, e.g. via gpu_compute_create_image()).
 * Returns 0 on success, -1 if ctx/image/layout is NULL or memory is
 * unbound. */
int gpu_compute_get_nv12_layout(gpu_context_t *ctx, gpu_image_t *image, gpu_memory_t memory, gpu_nv12_layout_t *layout);

/* A surface's memory mapped for the CPU to write into, and where its two
 * planes lie in it. NULL when it cannot be mapped. *unmap says whether the
 * mapping was made here, and is handed back to gpu_compute_unmap_surface()
 * when the writing is done - memory the context keeps mapped stays so. */
uint8_t *gpu_compute_map_surface(gpu_context_t *ctx, gpu_image_t *image,
                                 gpu_memory_t memory,
                                 gpu_nv12_layout_t *layout, bool *unmap);
void gpu_compute_unmap_surface(gpu_context_t *ctx, gpu_memory_t memory,
                               bool unmap);

/* ⚠️ Despite the name, these two move whatever the image holds: an
 * image created as GPU_IMAGE_P010 moves sixteen-bit samples, byte for
 * byte, with no shift in either direction. `width` is a sample count.
 * gpu_compute_upload_p010() below is the other thing - it takes the
 * decoder's 0..1023 and makes P010 out of them. */
int gpu_compute_upload_nv12(gpu_context_t *ctx, gpu_image_t *image, gpu_memory_t memory,
                           const uint8_t *y_plane, int y_pitch,
                           const uint8_t *uv_plane, int uv_pitch,
                           int width, int height);

/* The ten-bit twin of the upload above, for a GPU_IMAGE_P010 image.
 *
 * ⚠️ The pitches are in BYTES, like the eight-bit one, but the samples
 * are sixteen-bit words. The decoder hands over values in 0..1023 and
 * P010 wants them in the HIGH ten bits of each word, so the shift by six
 * happens in here - the caller has no business knowing what a surface
 * format is. */
int gpu_compute_upload_p010(gpu_context_t *ctx, gpu_image_t *image,
                            gpu_memory_t memory,
                            const uint16_t *y_plane, int y_pitch,
                            const uint16_t *uv_plane, int uv_pitch,
                            int width, int height);

/* Scale and crop `src` into `dst`, both two-plane YUV 4:2:0 of the same
 * bit depth, bilinear. The rectangles are in LUMA samples; their origins
 * and sizes are rounded to even numbers on the way in, because an odd one
 * cannot be expressed in a 4:2:0 plane.
 *
 * This is the whole of VAEntrypointVideoProc in this driver: no filters,
 * no colour conversion. There is nothing to convert between, since every
 * surface it hands out is NV12 or P010.
 *
 * Submits, waits, and returns 0 or -1. The work is a few hundred
 * microseconds and the descriptor set is shared, so it does not overlap
 * with itself.
 */
int gpu_compute_video_proc(gpu_context_t *ctx,
                           gpu_image_t *src, const int src_rect[4],
                           gpu_image_t *dst, const int dst_rect[4]);

void gpu_compute_copy_from_wc(void *dst, const void *src, size_t n);
int gpu_compute_download_nv12(gpu_context_t *ctx, gpu_image_t *image, gpu_memory_t memory,
                             uint8_t *y_plane, int y_pitch,
                             uint8_t *uv_plane, int uv_pitch,
                             int width, int height);

/* Exports `memory` (the packed Y+UV allocation gpu_compute_create_image()
 * bound both planes into) as a real DMA-BUF file descriptor, for
 * vaExportSurfaceHandle() - see va_backend.c's bc250_ExportSurfaceHandle().
 * Requires gpu_compute_create_image() to have been called on a device where
 * VK_EXT_external_memory_dma_buf was available (bc250_gpu_init() enables it
 * opportunistically; ctx->get_memory_fd_khr is NULL if it wasn't present,
 * and this returns -1 in that case). The returned fd is a new, independent
 * reference each call (the caller owns it and must close() it - VA-API's
 * own contract for vaExportSurfaceHandle() says the same: "backend driver
 * will not close the file descriptor"). Returns 0 on success. */
int gpu_compute_export_nv12_dmabuf(gpu_context_t *ctx, gpu_memory_t memory, int *out_fd);

/* Explicit GPU-side wait for whatever wrote into `memory` last, through
 * *any* API/context - not just this driver's own Vulkan submissions.
 *
 * This driver's VA-API render-target surfaces are shared, zero-copy, with
 * Sunshine's own OpenGL context: bc250_ExportSurfaceHandle() hands out a
 * real DMA-BUF fd for this same `memory`, and Sunshine's EGL/GL code
 * imports it as the render target its color-conversion shader renders NV12
 * output into (see egl::sws_t::convert_nv12() and how va_t::set_frame()
 * wires the export up - both the RAM and VRAM capture paths share this
 * exact mechanism on the *output* side regardless of which one is used for
 * capture, which is why corruption from a race here appears identically
 * under either capture path).
 *
 * Two independent GPU command-submission contexts (Mesa's radeonsi/RADV
 * for GL, and this driver's own Vulkan compute queue) touching the same
 * dma-buf need an explicit hand-off unless implicit kernel-level dma-buf
 * fencing is both engaged and correctly ordered for both sides - which
 * this driver has no way to verify from here, and evidently cannot rely
 * on given the observed corruption. This function makes the dependency
 * explicit instead of assuming implicit sync covers it: it snapshots the
 * dma-buf's current fences via DMA_BUF_IOCTL_EXPORT_SYNC_FILE (kernel,
 * cross-API-agnostic - see <linux/dma-buf.h>) and imports that snapshot as
 * a one-shot Vulkan wait semaphore for the *next* gpu_compute_end_picture()
 * call, so this driver's compute shaders cannot start reading the surface
 * until whatever last wrote to it - our own prior Vulkan work, or a
 * completely separate GL context's render pass - has actually finished on
 * the GPU.
 *
 * Call once per real frame, after the render target's contents are known
 * to be final (i.e. right before this driver would otherwise read it -
 * see bc250_EndPicture()) and before the matching gpu_compute_end_picture()
 * call. A no-op returning -1 if VK_KHR_external_semaphore_fd wasn't
 * available at device creation (have_external_semaphore_fd is false) -
 * callers must tolerate that and proceed without the extra wait, exactly
 * like every other opportunistic capability check in this file. */
int gpu_compute_wait_for_image_ready(gpu_context_t *ctx, gpu_memory_t memory);

/* Creates a diagnostic dump file safely - 0600, no symbolic links
 * followed, in BC250_DUMP_DIR or else $XDG_RUNTIME_DIR/bc250_dump_frames.
 * `what` names the switch that asked, for the messages. Every dump hook
 * below goes through it. */
FILE *bc250_debug_dump_open(const char *name, const char *what);

/* Test-harness instrumentation (tools/quality_test.sh): dumps raw NV12
 * frame bytes to BC250_DUMP_DIR (default $XDG_RUNTIME_DIR/bc250_dump_frames)
 * when BC250_DUMP_INPUT_FRAMES=1 is set in the environment; a no-op
 * otherwise. Shared by every known VA-API upload path so the harness
 * catches whichever one a given libva/ffmpeg build actually uses. See
 * gpu_compute.c. */
void bc250_debug_dump_nv12_frame(const uint8_t *y_plane, int y_pitch,
                                  const uint8_t *uv_plane, int uv_pitch,
                                  int width, int height);

/* Real per-MB motion vectors (see motion_estimation.comp's OutputMV), laid
 * out as num_mbs entries of {int32_t mvx, mvy; uint32_t sad; uint32_t pad;}
 * (16 bytes/entry, matching the GPU's std430 MotionVector struct). Only
 * meaningful for P-slices. Same fence-safe double-buffer contract as above. */
typedef struct {
    int32_t mvx, mvy;
    uint32_t sad;
    uint32_t _pad;
} gpu_mv_t;

/* Picture encoding orchestration */
int gpu_compute_begin_picture(gpu_context_t *ctx, gpu_image_t render_target);
/* num_slices: threaded through to residual_predict.comp so its I16x16
 * neighbor-availability check can correctly treat a different-slice
 * neighbor MB as unavailable - see that shader's SLICE BOUNDARIES comment.
 * Must match the num_slices the caller will actually partition the CAVLC
 * bitstream into (encoder_h264.c's BC250_SLICES_PER_FRAME). */
int gpu_compute_dispatch_encode(gpu_context_t *ctx, gpu_image_t render_target, int width, int height, int qp, int is_intra, int num_slices);
/* Extended dispatch: allows passing dynamic ME mode and CPU-computed motion vectors.
 * If cpu_mvs != NULL, Stage 2 (Vulkan motion estimation) is skipped and CPU MVs are uploaded. */
int gpu_compute_dispatch_encode_ext(gpu_context_t *ctx, gpu_image_t render_target, int width, int height, int qp, int is_intra, int num_slices, int me_mode, const gpu_mv_t *cpu_mvs);
/* Lightweight dispatch running ONLY Stage 2 (subgroup-accelerated Vulkan Motion Estimation)
 * without H.264 transform/quant/entropy stages. Used to accelerate HEVC inter frames on GPU. */
int gpu_compute_dispatch_me_only(gpu_context_t *ctx, gpu_image_t render_target, int width, int height);
double gpu_compute_get_last_latency_ms(const gpu_context_t *ctx);
int gpu_compute_end_picture(gpu_context_t *ctx);
int gpu_compute_sync(gpu_context_t *ctx);

/* Which double-buffer slot (0/1) holds the frame most recently submitted by
 * gpu_compute_end_picture(). The unsuffixed gpu_compute_sync() and
 * gpu_compute_get_*_staging_data() all implicitly mean this slot, which is
 * correct only while exactly one frame is ever in flight.
 *
 * A PIPELINED caller (one that submits frame N+1 before entropy-coding frame
 * N) must capture this at submit time and pass it to the _slot variants below,
 * because by finish time "most recently submitted" is the WRONG frame. That
 * mistake is a silent one-frame data swap: it produces a decodable stream and
 * it is invisible to the BC250_NZ_AUDIT mask audit, since quant_levels and
 * nz_masks would both be read from the same wrong slot and therefore still
 * match exactly. PSNR on moving content is what catches it. */
int gpu_compute_submitted_slot(gpu_context_t *ctx);
int gpu_compute_sync_slot(gpu_context_t *ctx, int slot);
int gpu_compute_get_quant_staging_data_slot(gpu_context_t *ctx, int slot, void **data, size_t *size);
int gpu_compute_get_dc_staging_data_slot(gpu_context_t *ctx, int slot, void **data, size_t *size);
int gpu_compute_get_pred_mode_staging_data_slot(gpu_context_t *ctx, int slot, void **data, size_t *size);
int gpu_compute_get_mv_staging_data_slot(gpu_context_t *ctx, int slot, void **data, size_t *size);
int gpu_compute_get_nz_staging_data_slot(gpu_context_t *ctx, int slot, void **data, size_t *size);
int gpu_compute_get_staging_data(gpu_context_t *ctx, void **data, size_t *size);
int gpu_compute_release_staging_data(gpu_context_t *ctx);

/* Real per-coefficient residual readback (see quant_staging_buffers above).
 * Follows the same double-buffer contract as gpu_compute_get_staging_data():
 * call after gpu_compute_sync(), data points at the buffer that was written by the
 * frame BEFORE the one just submitted (fence-safe to read from the CPU). Layout is
 * num_mbs*24*16 ints, int index = (mb_idx*24+block_idx)*16+pos (raster position within
 * the 4x4 block, NOT zigzag). */
int gpu_compute_get_quant_staging_data(gpu_context_t *ctx, void **data, size_t *size);

/* Compact PRE-quantization DC readback: one int per 4x4 block, num_mbs*24
 * entries, index = mb_idx*24 + block_idx. Replaces the former full-coeff
 * staging buffer, of which only these values were ever read - see
 * dc_coeff_buffer's comment above. Same fence-safe contract. */
int gpu_compute_get_dc_staging_data(gpu_context_t *ctx, void **data, size_t *size);

/* Real per-MB I16x16 prediction mode (see residual_predict.comp), one uint32
 * per MB, values match cavlc.h's H264_I16x16_* constants. Only meaningful for
 * I-slices. Same fence-safe double-buffer contract as above. */
int gpu_compute_get_pred_mode_staging_data(gpu_context_t *ctx, void **data, size_t *size);

int gpu_compute_get_mv_staging_data(gpu_context_t *ctx, void **data, size_t *size);

/* Per-4x4-block nonzero bitmask from quantize.comp, one uint32 per block laid
 * out as num_mbs*24 entries, index = mb_idx*24 + block_idx, with bit p set iff
 * that block's raster position p quantized to a nonzero level. Exactly derived
 * from the same `level != 0` test that produces quant_levels, so a bit test on
 * this is equivalent to - not an approximation of - scanning the block itself.
 * Same fence-safe double-buffer contract as above. */
int gpu_compute_get_nz_staging_data(gpu_context_t *ctx, void **data, size_t *size);

/* TEMPORARY debug instrumentation for Part A (reconstruction) verification -
 * see gpu_compute.c for details. No-op unless BC250_DUMP_RECON_FRAMES=1. */
void gpu_compute_debug_dump_recon(gpu_context_t *ctx, int width, int height);

/* Real-content investigation instrumentation: dumps the ACTUAL surface Y/UV
 * pixel data at encode-dispatch time, reading it back from the Vulkan image
 * itself rather than relying on being called from a known upload path (see
 * bc250_debug_dump_nv12_frame()'s doc comment - that hook only fires for
 * vaPutImage/vaDeriveImage+vaMapBuffer, which real Sunshine sessions never
 * use: Sunshine instead writes directly into the surface's exported DMA-BUF
 * via its own GL blit, bypassing both of those paths entirely). Call right
 * before gpu_compute_dispatch_encode() so it sees exactly what the encoder
 * is about to encode, regardless of how the surface's contents got there.
 * No-op unless BC250_DUMP_REAL_INPUT=1 is set (BC250_DUMP_DIR for the
 * directory, default $XDG_RUNTIME_DIR/bc250_dump_frames, same as the other
 * dump hooks -
 * files are named real_NNNNN.nv12 to disambiguate from frame_/recon_). */
void gpu_compute_debug_dump_real_input(gpu_context_t *ctx, gpu_image_t *image, gpu_memory_t memory, int width, int height);

#ifdef __cplusplus
}
#endif

#endif // GPU_COMPUTE_H
