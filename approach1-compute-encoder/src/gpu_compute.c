/* bc250-encoding-decoding-fix v0.4.3 - https://github.com/simpmix/bc250-encoding-decoding-fix */
/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * gpu_compute.c - Vulkan compute backend for AMD BC-250 encoding
 */
/* Required for dladdr()/Dl_info (glibc guards both behind __USE_GNU), used by
 * driver_install_dir() to find the shaders installed beside this .so. Must
 * precede every include. Safe project-wide-inconsistent because nothing in
 * src/ uses the functions _GNU_SOURCE redefines (strerror_r, basename). */
#define _GNU_SOURCE
#include "gpu_compute.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>
#include <limits.h>
#include <dlfcn.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <linux/dma-buf.h>

#define BC250_DEVICE_ID 0x13FE
#define AMD_VENDOR_ID   0x1002

/* Opt-in GPU per-stage timing (BC250_PERF_STATS=1), added for real-time
 * throughput diagnosis - see gpu_compute_dispatch_encode()'s timestamp
 * writes and gpu_compute_sync()'s readback. One VkQueryPool slot per
 * checkpoint, written in strictly increasing time order every frame
 * regardless of P/I path so the deltas are always well-defined (the path
 * NOT taken just gets a run of zero-duration slots):
 *   0 = frame start (top of command buffer)
 *   1 = after motion estimation
 *   2 = after residual_predict (P only; == 1 on I frames)
 *   3 = after DCT (P only; == 2 on I frames)
 *   4 = after quantize (P only; == 3 on I frames)
 *   5 = after reconstruct (P only; == 4 on I frames)
 *   6 = after diagonal-wavefront intra reconstruction (I only; == 5 on P frames)
 *   7 = after deblock (both paths; == 6 if BC250_FAST_MODE skipped it)
 *   8 = after entropy encode (both paths)
 *   9 = after the GPU->host staging buffer copies (frame end)
 */
#define BC250_PERF_NUM_TIMESTAMPS 10

#define VK_CHECK(x) do { \
    VkResult err = (x); \
    if (err != VK_SUCCESS) { \
        fprintf(stderr, "[bc250-gpu] Vulkan error %d at %s:%d\n", err, __FILE__, __LINE__); \
        return -1; \
    } \
} while(0)

static uint32_t find_memory_type(VkPhysicalDevice physical_device, uint32_t type_filter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) && (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    /* Fallback to any matching type */
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if (type_filter & (1 << i)) {
            return i;
        }
    }
    return 0;
}

/* Like find_memory_type(), but tries `preferred` first (which must be a
 * superset of `required`) and only falls back to a plain `required`-only
 * match if this device exposes no type satisfying `preferred` at all.
 *
 * WHY THIS EXISTS: the GPU-readback staging buffers this driver bulk-copies
 * every frame (quant_staging_buffers/dc_staging_buffers/pred_mode_staging_
 * buffers/mv_staging_buffers - see encoder_h264.c's shadow_copy() doc
 * comment) were being bound to a HOST_VISIBLE|HOST_COHERENT memory type
 * WITHOUT HOST_CACHED, on the reasoning that the GPU's one-shot
 * vkCmdCopyBuffer write into them doesn't care about CPU cacheability. That
 * reasoning only accounted for the write side. On real BC-250 hardware this
 * driver was measured (BC250_PERF_STATS=1, real board run, 1280x720) paying
 * ~81ms/frame - the entire real-time-throughput gap between ~1ms of actual
 * GPU compute + ~1ms of CPU CAVLC and the ~83ms real wall-clock time per
 * frame - inside shadow_copy()'s bulk memcpy() itself, i.e. the CPU
 * *reading* ~10.6MB/frame back out of that same memory. Uncached/
 * write-combined memory has notoriously poor CPU read bandwidth (routinely
 * an order of magnitude or more below normal cached RAM) even for a single
 * fully sequential streaming pass - confirmed by the shadow_copy_ms
 * diagnostic bracket landing within noise of the entire unaccounted gap.
 * vkGetPhysicalDeviceMemoryProperties() on this device confirms a
 * HOST_VISIBLE|HOST_COHERENT|HOST_CACHED type exists on the same heap as the
 * uncached one currently selected (both are system-memory-backed on this
 * APU, not a discrete-GPU BAR), so preferring it costs nothing in
 * portability: find_memory_type()'s original required-only search is kept
 * as the fallback for any device that doesn't expose a cached type at all.
 * This only changes which physical memory type backs these buffers - not
 * their VkBufferUsageFlags, not HOST_COHERENT (still required both passes,
 * so Vulkan still guarantees the CPU sees the GPU's writes after the
 * existing fence wait with no added vkInvalidateMappedMemoryRanges/
 * vkFlushMappedMemoryRanges calls needed), and not a single byte of what
 * either side reads or writes - purely a CPU-read-speed optimization. */
/* Records the memory type index the most recent preferred-search settled on,
 * so init can report what the staging buffers ACTUALLY got rather than what
 * was asked for - a silently-ignored preference and a silently-ignored env
 * toggle look identical in a timing run otherwise. */
static uint32_t g_last_preferred_memtype = UINT32_MAX;

static uint32_t find_memory_type_preferred(VkPhysicalDevice physical_device, uint32_t type_filter,
                                            VkMemoryPropertyFlags preferred, VkMemoryPropertyFlags required) {
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) && (mem_props.memoryTypes[i].propertyFlags & preferred) == preferred) {
            g_last_preferred_memtype = i;
            return i;
        }
    }
    g_last_preferred_memtype = find_memory_type(physical_device, type_filter, required);
    return g_last_preferred_memtype;
}

static int create_buffer_with_memory(gpu_context_t *ctx, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer *buffer, VkDeviceMemory *memory) {
    VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };
    VK_CHECK(vkCreateBuffer(ctx->device, &buffer_info, NULL, buffer));

    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(ctx->device, *buffer, &mem_reqs);

    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mem_reqs.size,
        .memoryTypeIndex = find_memory_type(ctx->physical_device, mem_reqs.memoryTypeBits, properties)
    };
    VK_CHECK(vkAllocateMemory(ctx->device, &alloc_info, NULL, memory));
    VK_CHECK(vkBindBufferMemory(ctx->device, *buffer, *memory, 0));
    return 0;
}

/* Same as create_buffer_with_memory(), but selects the memory type via
 * find_memory_type_preferred() instead of find_memory_type() - see that
 * function's doc comment. Used only for the GPU-readback staging buffers
 * that encoder_h264.c's shadow_copy() bulk-reads every frame, where CPU read
 * bandwidth (not GPU write bandwidth) is what actually matters. */
static int create_buffer_with_memory_preferred(gpu_context_t *ctx, VkDeviceSize size, VkBufferUsageFlags usage,
                                                VkMemoryPropertyFlags preferred, VkMemoryPropertyFlags required,
                                                VkBuffer *buffer, VkDeviceMemory *memory) {
    VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };
    VK_CHECK(vkCreateBuffer(ctx->device, &buffer_info, NULL, buffer));

    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(ctx->device, *buffer, &mem_reqs);

    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mem_reqs.size,
        .memoryTypeIndex = find_memory_type_preferred(ctx->physical_device, mem_reqs.memoryTypeBits, preferred, required)
    };
    VK_CHECK(vkAllocateMemory(ctx->device, &alloc_info, NULL, memory));
    VK_CHECK(vkBindBufferMemory(ctx->device, *buffer, *memory, 0));
    return 0;
}

static void update_storage_buffer_descriptor(VkDevice device, VkDescriptorSet set, uint32_t binding, VkBuffer buffer, VkDeviceSize size) {
    VkDescriptorBufferInfo buf_info = {
        .buffer = buffer,
        .offset = 0,
        .range = size
    };
    VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = set,
        .dstBinding = binding,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buf_info
    };
    vkUpdateDescriptorSets(device, 1, &write, 0, NULL);
}

static void update_storage_image_descriptor(VkDevice device, VkDescriptorSet set, uint32_t binding, VkImageView view) {
    VkDescriptorImageInfo img_info = {
        .sampler = VK_NULL_HANDLE,
        .imageView = view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL
    };
    VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = set,
        .dstBinding = binding,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = &img_info
    };
    vkUpdateDescriptorSets(device, 1, &write, 0, NULL);
}

static int allocate_encoding_buffers(gpu_context_t *ctx, uint32_t width, uint32_t height) {
    if (ctx->mv_buffer) {
        vkDestroyBuffer(ctx->device, ctx->mv_buffer, NULL);
        vkFreeMemory(ctx->device, ctx->mv_memory, NULL);
        ctx->mv_buffer = VK_NULL_HANDLE;
    }
    if (ctx->residual_buffer) {
        vkDestroyBuffer(ctx->device, ctx->residual_buffer, NULL);
        vkFreeMemory(ctx->device, ctx->residual_memory, NULL);
        ctx->residual_buffer = VK_NULL_HANDLE;
    }
    if (ctx->pred_buffer) {
        vkDestroyBuffer(ctx->device, ctx->pred_buffer, NULL);
        vkFreeMemory(ctx->device, ctx->pred_memory, NULL);
        ctx->pred_buffer = VK_NULL_HANDLE;
    }
    if (ctx->coeff_buffer) {
        vkDestroyBuffer(ctx->device, ctx->coeff_buffer, NULL);
        vkFreeMemory(ctx->device, ctx->coeff_memory, NULL);
        ctx->coeff_buffer = VK_NULL_HANDLE;
    }
    if (ctx->dc_coeff_buffer) {
        vkDestroyBuffer(ctx->device, ctx->dc_coeff_buffer, NULL);
        vkFreeMemory(ctx->device, ctx->dc_coeff_memory, NULL);
        ctx->dc_coeff_buffer = VK_NULL_HANDLE;
    }
    if (ctx->quant_levels_buffer) {
        vkDestroyBuffer(ctx->device, ctx->quant_levels_buffer, NULL);
        vkFreeMemory(ctx->device, ctx->quant_levels_memory, NULL);
        ctx->quant_levels_buffer = VK_NULL_HANDLE;
    }
    if (ctx->nz_count_buffer) {
        vkDestroyBuffer(ctx->device, ctx->nz_count_buffer, NULL);
        vkFreeMemory(ctx->device, ctx->nz_count_memory, NULL);
        ctx->nz_count_buffer = VK_NULL_HANDLE;
    }
    if (ctx->pred_mode_buffer) {
        vkDestroyBuffer(ctx->device, ctx->pred_mode_buffer, NULL);
        vkFreeMemory(ctx->device, ctx->pred_mode_memory, NULL);
        ctx->pred_mode_buffer = VK_NULL_HANDLE;
    }
    if (ctx->entropy_buffer) {
        vkDestroyBuffer(ctx->device, ctx->entropy_buffer, NULL);
        vkFreeMemory(ctx->device, ctx->entropy_memory, NULL);
        ctx->entropy_buffer = VK_NULL_HANDLE;
    }
    for (int i = 0; i < 2; i++) {
        if (ctx->staging_mapped[i]) {
            vkUnmapMemory(ctx->device, ctx->staging_memories[i]);
            ctx->staging_mapped[i] = NULL;
        }
        if (ctx->staging_buffers[i]) {
            vkDestroyBuffer(ctx->device, ctx->staging_buffers[i], NULL);
            vkFreeMemory(ctx->device, ctx->staging_memories[i], NULL);
            ctx->staging_buffers[i] = VK_NULL_HANDLE;
            ctx->staging_memories[i] = VK_NULL_HANDLE;
        }
        if (ctx->quant_staging_mapped[i]) {
            vkUnmapMemory(ctx->device, ctx->quant_staging_memories[i]);
            ctx->quant_staging_mapped[i] = NULL;
        }
        if (ctx->quant_staging_buffers[i]) {
            vkDestroyBuffer(ctx->device, ctx->quant_staging_buffers[i], NULL);
            vkFreeMemory(ctx->device, ctx->quant_staging_memories[i], NULL);
            ctx->quant_staging_buffers[i] = VK_NULL_HANDLE;
            ctx->quant_staging_memories[i] = VK_NULL_HANDLE;
        }
        if (ctx->dc_staging_mapped[i]) {
            vkUnmapMemory(ctx->device, ctx->dc_staging_memories[i]);
            ctx->dc_staging_mapped[i] = NULL;
        }
        if (ctx->dc_staging_buffers[i]) {
            vkDestroyBuffer(ctx->device, ctx->dc_staging_buffers[i], NULL);
            vkFreeMemory(ctx->device, ctx->dc_staging_memories[i], NULL);
            ctx->dc_staging_buffers[i] = VK_NULL_HANDLE;
            ctx->dc_staging_memories[i] = VK_NULL_HANDLE;
        }
        if (ctx->pred_mode_staging_mapped[i]) {
            vkUnmapMemory(ctx->device, ctx->pred_mode_staging_memories[i]);
            ctx->pred_mode_staging_mapped[i] = NULL;
        }
        if (ctx->pred_mode_staging_buffers[i]) {
            vkDestroyBuffer(ctx->device, ctx->pred_mode_staging_buffers[i], NULL);
            vkFreeMemory(ctx->device, ctx->pred_mode_staging_memories[i], NULL);
            ctx->pred_mode_staging_buffers[i] = VK_NULL_HANDLE;
            ctx->pred_mode_staging_memories[i] = VK_NULL_HANDLE;
        }
        if (ctx->mv_staging_mapped[i]) {
            vkUnmapMemory(ctx->device, ctx->mv_staging_memories[i]);
            ctx->mv_staging_mapped[i] = NULL;
        }
        if (ctx->mv_staging_buffers[i]) {
            vkDestroyBuffer(ctx->device, ctx->mv_staging_buffers[i], NULL);
            vkFreeMemory(ctx->device, ctx->mv_staging_memories[i], NULL);
            ctx->mv_staging_buffers[i] = VK_NULL_HANDLE;
            ctx->mv_staging_memories[i] = VK_NULL_HANDLE;
        }
        if (ctx->nz_staging_mapped[i]) {
            vkUnmapMemory(ctx->device, ctx->nz_staging_memories[i]);
            ctx->nz_staging_mapped[i] = NULL;
        }
        if (ctx->nz_staging_buffers[i]) {
            vkDestroyBuffer(ctx->device, ctx->nz_staging_buffers[i], NULL);
            vkFreeMemory(ctx->device, ctx->nz_staging_memories[i], NULL);
            ctx->nz_staging_buffers[i] = VK_NULL_HANDLE;
            ctx->nz_staging_memories[i] = VK_NULL_HANDLE;
        }
    }

    ctx->frame_width = width;
    ctx->frame_height = height;

    uint32_t width_in_mbs = (width + 15) / 16;
    uint32_t height_in_mbs = (height + 15) / 16;
    uint32_t num_mbs = width_in_mbs * height_in_mbs;

    VkDeviceSize mv_size = num_mbs * sizeof(uint32_t) * 4;
    VkDeviceSize residual_size = num_mbs * 24 * 16 * sizeof(int);
    VkDeviceSize coeff_size = residual_size;
    /* Half the width of residual/coeff: quantized LEVELS (post-quantization)
     * are stored int16_t, not int32 - verified device support
     * (VK_KHR_16bit_storage / shaderInt16 / storageBuffer16BitAccess, all
     * true on this GPU) and confirmed safe by ITU-T H.264's coefficient
     * magnitude bounds at 8-bit depth, nowhere near +-32767. coeff_buffer and
     * residual_buffer are NOT changed - they hold pre-quantization values,
     * a different (larger-headroom) quantity, out of scope for this. */
    VkDeviceSize quant_levels_size = num_mbs * 24 * 16 * sizeof(int16_t);
    VkDeviceSize nz_count_size = num_mbs * 24 * sizeof(uint32_t);
    VkDeviceSize pred_mode_size = num_mbs * sizeof(uint32_t);
    VkDeviceSize entropy_size = (VkDeviceSize)width * height * 2; /* Generous */

    VkDeviceSize dc_coeff_size = num_mbs * 24 * sizeof(int);

    ctx->staging_size = entropy_size;
    ctx->quant_staging_size = quant_levels_size;
    ctx->dc_staging_size = dc_coeff_size;
    ctx->pred_mode_staging_size = pred_mode_size;
    ctx->mv_staging_size = mv_size;
    ctx->nz_staging_size = nz_count_size;

    /* Every allocation below is checked. create_buffer_with_memory() has
     * always returned -1 on failure, but all ~20 call sites here ignored it,
     * so an out-of-memory left VK_NULL_HANDLE buffers behind and the failure
     * surfaced later as a SEGV in the vkMapMemory()/dispatch path instead of a
     * clean "this encoder is unavailable". That is the same failure shape as
     * the silent slice overflow in DEVLOG §18.2: an ignored return turning a
     * diagnosable error into a crash. On a ~8 GB GART aperture that Sunshine
     * probes 20 times over, OOM here is a genuinely reachable state, not a
     * theoretical one - it was already happening on v0.3.0. */
    int rc = 0;
    rc |= create_buffer_with_memory(ctx, mv_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &ctx->mv_buffer, &ctx->mv_memory);
    rc |= create_buffer_with_memory(ctx, residual_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &ctx->residual_buffer, &ctx->residual_memory);
    rc |= create_buffer_with_memory(ctx, residual_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &ctx->pred_buffer, &ctx->pred_memory);
    /* coeff_buffer no longer needs TRANSFER_SRC: it is consumed on the GPU
     * (quantize.comp's input, reconstruct.comp's DC source) and no longer
     * crosses to the host - dc_coeff_buffer carries the only part the CPU
     * reads. See gpu_compute.h's dc_coeff_buffer comment. */
    rc |= create_buffer_with_memory(ctx, coeff_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &ctx->coeff_buffer, &ctx->coeff_memory);
    rc |= create_buffer_with_memory(ctx, dc_coeff_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &ctx->dc_coeff_buffer, &ctx->dc_coeff_memory);
    rc |= create_buffer_with_memory(ctx, quant_levels_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &ctx->quant_levels_buffer, &ctx->quant_levels_memory);
    /* TRANSFER_SRC added so the per-block nonzero mask can be read back - see
     * gpu_compute.h's nz_staging_buffers and quantize.comp's NonZeroMask. */
    rc |= create_buffer_with_memory(ctx, nz_count_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &ctx->nz_count_buffer, &ctx->nz_count_memory);
    rc |= create_buffer_with_memory(ctx, pred_mode_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &ctx->pred_mode_buffer, &ctx->pred_mode_memory);
    rc |= create_buffer_with_memory(ctx, entropy_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &ctx->entropy_buffer, &ctx->entropy_memory);
    rc |= create_buffer_with_memory(ctx, entropy_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &ctx->staging_buffers[0], &ctx->staging_memories[0]);
    rc |= create_buffer_with_memory(ctx, entropy_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &ctx->staging_buffers[1], &ctx->staging_memories[1]);

    /* These four staging-buffer pairs are the ones encoder_h264.c's
     * shadow_copy() bulk-reads from the CPU every single frame (quant_levels/
     * coeff/pred_modes/mvs) - see find_memory_type_preferred()'s doc comment
     * for why they request HOST_CACHED as a preference, not a requirement.
     * staging_buffers[]/entropy_buffer above are a separate, currently-dead
     * GPU-entropy-coding path (nothing reads gpu_compute_get_staging_data())
     * and are deliberately left on plain create_buffer_with_memory(). */
    VkMemoryPropertyFlags cached_pref = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    VkMemoryPropertyFlags visible_req = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    /* PERF measurement toggle (BC250_STAGING_CACHED=0): drop the HOST_CACHED
     * preference, restoring the pre-fix uncached/write-combined behaviour.
     * Exists so the HOST_CACHED memory type and encoder_h264.c's shadow_copy()
     * - two independently-added fixes for the SAME uncached-read problem - can
     * be A/B'd against each other in one binary instead of being assumed to
     * both still be needed. Memory-type choice cannot change encoded output. */
    {
        const char *sc = getenv("BC250_STAGING_CACHED");
        if (sc && strcmp(sc, "0") == 0) {
            cached_pref = visible_req;
            fprintf(stderr, "[bc250-gpu] BC250_STAGING_CACHED=0: staging buffers forced uncached\n");
        }
    }
    rc |= create_buffer_with_memory_preferred(ctx, quant_levels_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, cached_pref, visible_req, &ctx->quant_staging_buffers[0], &ctx->quant_staging_memories[0]);
    rc |= create_buffer_with_memory_preferred(ctx, quant_levels_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, cached_pref, visible_req, &ctx->quant_staging_buffers[1], &ctx->quant_staging_memories[1]);
    rc |= create_buffer_with_memory_preferred(ctx, dc_coeff_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, cached_pref, visible_req, &ctx->dc_staging_buffers[0], &ctx->dc_staging_memories[0]);
    rc |= create_buffer_with_memory_preferred(ctx, dc_coeff_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, cached_pref, visible_req, &ctx->dc_staging_buffers[1], &ctx->dc_staging_memories[1]);
    rc |= create_buffer_with_memory_preferred(ctx, pred_mode_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, cached_pref, visible_req, &ctx->pred_mode_staging_buffers[0], &ctx->pred_mode_staging_memories[0]);
    rc |= create_buffer_with_memory_preferred(ctx, pred_mode_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, cached_pref, visible_req, &ctx->pred_mode_staging_buffers[1], &ctx->pred_mode_staging_memories[1]);
    rc |= create_buffer_with_memory_preferred(ctx, mv_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, cached_pref, visible_req, &ctx->mv_staging_buffers[0], &ctx->mv_staging_memories[0]);
    rc |= create_buffer_with_memory_preferred(ctx, mv_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, cached_pref, visible_req, &ctx->mv_staging_buffers[1], &ctx->mv_staging_memories[1]);
    rc |= create_buffer_with_memory_preferred(ctx, nz_count_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, cached_pref, visible_req, &ctx->nz_staging_buffers[0], &ctx->nz_staging_memories[0]);
    rc |= create_buffer_with_memory_preferred(ctx, nz_count_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, cached_pref, visible_req, &ctx->nz_staging_buffers[1], &ctx->nz_staging_memories[1]);
    {
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(ctx->physical_device, &mp);
        VkMemoryPropertyFlags f = (g_last_preferred_memtype < mp.memoryTypeCount)
                                  ? mp.memoryTypes[g_last_preferred_memtype].propertyFlags : 0;
        fprintf(stderr, "[bc250-gpu] readback staging memtype[%u] flags=0x%x %s\n",
                g_last_preferred_memtype, f,
                (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? "HOST_CACHED" : "UNCACHED");
    }

    /* Bail BEFORE mapping: vkMapMemory() on a VK_NULL_HANDLE memory from a
     * failed allocation above is undefined behaviour, and in practice this is
     * where the SEGV landed. Returning -1 with frame_width/frame_height left
     * at 0 means the next dispatch retries the allocation (memory pressure is
     * transient - other contexts get destroyed), and until then the staging
     * getters report "no data" and the encoder falls back. */
    if (rc != 0) {
        fprintf(stderr, "[bc250-gpu] allocate_encoding_buffers(%ux%u) FAILED - out of device memory. "
                        "Encoding buffers need ~%.0f MB (%ux%u = %u MBs); on this APU that comes out of "
                        "the ~8 GB GART/GTT aperture (amdgpu.gttsize), whose host-visible heap is the "
                        "smaller half. Encoder unavailable at this resolution.\n",
                width, height,
                (double)(residual_size * 4 + quant_levels_size * 2 + coeff_size * 2 + entropy_size * 3) / (1024.0 * 1024.0),
                width, height, num_mbs);
        ctx->frame_width = 0;
        ctx->frame_height = 0;
        return -1;
    }

    /* Persistently map all staging buffers to eliminate per-frame map/unmap syscall overhead */
    vkMapMemory(ctx->device, ctx->staging_memories[0], 0, entropy_size, 0, &ctx->staging_mapped[0]);
    vkMapMemory(ctx->device, ctx->staging_memories[1], 0, entropy_size, 0, &ctx->staging_mapped[1]);
    vkMapMemory(ctx->device, ctx->quant_staging_memories[0], 0, quant_levels_size, 0, &ctx->quant_staging_mapped[0]);
    vkMapMemory(ctx->device, ctx->quant_staging_memories[1], 0, quant_levels_size, 0, &ctx->quant_staging_mapped[1]);
    vkMapMemory(ctx->device, ctx->dc_staging_memories[0], 0, dc_coeff_size, 0, &ctx->dc_staging_mapped[0]);
    vkMapMemory(ctx->device, ctx->dc_staging_memories[1], 0, dc_coeff_size, 0, &ctx->dc_staging_mapped[1]);
    vkMapMemory(ctx->device, ctx->pred_mode_staging_memories[0], 0, pred_mode_size, 0, &ctx->pred_mode_staging_mapped[0]);
    vkMapMemory(ctx->device, ctx->pred_mode_staging_memories[1], 0, pred_mode_size, 0, &ctx->pred_mode_staging_mapped[1]);
    vkMapMemory(ctx->device, ctx->mv_staging_memories[0], 0, mv_size, 0, &ctx->mv_staging_mapped[0]);
    vkMapMemory(ctx->device, ctx->mv_staging_memories[1], 0, mv_size, 0, &ctx->mv_staging_mapped[1]);
    vkMapMemory(ctx->device, ctx->nz_staging_memories[0], 0, nz_count_size, 0, &ctx->nz_staging_mapped[0]);
    vkMapMemory(ctx->device, ctx->nz_staging_memories[1], 0, nz_count_size, 0, &ctx->nz_staging_mapped[1]);

    /* Update buffer descriptors */
    update_storage_buffer_descriptor(ctx->device, ctx->me_desc_set, 2, ctx->mv_buffer, mv_size);

    /* residual_predict.comp's buffer bindings: mv_buffer (real MVs, for P
     * motion-compensated residual), residual_buffer (its output, consumed by
     * dct_transform.comp) and pred_mode_buffer (its I16x16 mode decision,
     * consumed by encoder_h264.c's CAVLC header writer). Image bindings
     * (current Y/UV, reference Y) are updated per-dispatch in
     * gpu_compute_dispatch_encode() since they change every frame. */
    update_storage_buffer_descriptor(ctx->device, ctx->predict_desc_set, 3, ctx->mv_buffer, mv_size);
    update_storage_buffer_descriptor(ctx->device, ctx->predict_desc_set, 4, ctx->residual_buffer, residual_size);
    update_storage_buffer_descriptor(ctx->device, ctx->predict_desc_set, 5, ctx->pred_mode_buffer, pred_mode_size);
    update_storage_buffer_descriptor(ctx->device, ctx->predict_desc_set, 7, ctx->pred_buffer, residual_size);

    update_storage_buffer_descriptor(ctx->device, ctx->dct_desc_set, 0, ctx->residual_buffer, residual_size);
    update_storage_buffer_descriptor(ctx->device, ctx->dct_desc_set, 1, ctx->coeff_buffer, coeff_size);
    update_storage_buffer_descriptor(ctx->device, ctx->dct_desc_set, 2, ctx->dc_coeff_buffer, dc_coeff_size);

    update_storage_buffer_descriptor(ctx->device, ctx->quant_desc_set, 0, ctx->coeff_buffer, coeff_size);
    update_storage_buffer_descriptor(ctx->device, ctx->quant_desc_set, 1, ctx->quant_levels_buffer, quant_levels_size);
    update_storage_buffer_descriptor(ctx->device, ctx->quant_desc_set, 2, ctx->nz_count_buffer, nz_count_size);

    /* deblock_filter.comp binding 1 is declared "QuantLevels" there (it used
     * to be misleadingly declared "QPMap" while never actually being read -
     * see that shader's top-of-file comment): real per-4x4-block quantized
     * coefficient levels, used for the ITU-T 8.7.2.1 nonzero-coefficient
     * boundary-strength test. Binding 2 is the real per-macroblock motion
     * vectors, used for that section's motion-vector-difference test. */
    update_storage_buffer_descriptor(ctx->device, ctx->deblock_desc_set, 1, ctx->quant_levels_buffer, quant_levels_size);
    update_storage_buffer_descriptor(ctx->device, ctx->deblock_desc_set, 2, ctx->mv_buffer, mv_size);

    update_storage_buffer_descriptor(ctx->device, ctx->entropy_desc_set, 0, ctx->quant_levels_buffer, quant_levels_size);
    update_storage_buffer_descriptor(ctx->device, ctx->entropy_desc_set, 1, ctx->entropy_buffer, entropy_size);

    /* reconstruct.comp's buffer bindings: quant_levels_buffer (post-quant AC
     * levels), coeff_buffer (pre-quant, for the I16x16/chroma DC Hadamard)
     * and pred_buffer (retained prediction). Image bindings (recon Y/UV) are
     * updated per-dispatch in gpu_compute_dispatch_encode() since recon_image
     * can be (re)created there. */
    update_storage_buffer_descriptor(ctx->device, ctx->reconstruct_desc_set, 0, ctx->quant_levels_buffer, quant_levels_size);
    update_storage_buffer_descriptor(ctx->device, ctx->reconstruct_desc_set, 1, ctx->coeff_buffer, coeff_size);
    update_storage_buffer_descriptor(ctx->device, ctx->reconstruct_desc_set, 2, ctx->pred_buffer, residual_size);

    /* intra_wavefront.comp's buffer bindings: quant_levels_buffer/coeff_buffer/
     * pred_mode_buffer (writeonly - same underlying buffers as the whole-frame
     * P-slice path, just written by this shader instead for I-slices). Image
     * bindings (current Y/UV, recon Y/UV) are updated per-dispatch in
     * gpu_compute_dispatch_encode() since recon_image can be (re)created there
     * and render_target changes every frame. */
    update_storage_buffer_descriptor(ctx->device, ctx->intra_wavefront_desc_set, 4, ctx->quant_levels_buffer, quant_levels_size);
    update_storage_buffer_descriptor(ctx->device, ctx->intra_wavefront_desc_set, 5, ctx->coeff_buffer, coeff_size);
    update_storage_buffer_descriptor(ctx->device, ctx->intra_wavefront_desc_set, 6, ctx->pred_mode_buffer, pred_mode_size);
    update_storage_buffer_descriptor(ctx->device, ctx->intra_wavefront_desc_set, 7, ctx->nz_count_buffer, nz_count_size);
    update_storage_buffer_descriptor(ctx->device, ctx->intra_wavefront_desc_set, 8, ctx->dc_coeff_buffer, dc_coeff_size);

    return 0;
}

/* Address anchor: any symbol inside this .so works for dladdr(). */
static const char bc250_module_anchor = 0;

/*
 * Absolute directory this driver .so was actually installed into, resolved
 * once on first use. Empty string if it can't be determined.
 *
 * Every install script ships the .spv shaders next to bc250_drv_video.so,
 * but the fixed search list in load_spirv_shader() can't know where a given
 * system put that - and when none of those fixed paths exist, the driver
 * loads, advertises H.264 encode, and then cannot encode anything. That is
 * not hypothetical: observed on the dev board with the driver at
 * /opt/bc250-driver, all nine shaders sitting right beside it, every fixed
 * search path absent, and BC250_SHADER_DIR unset.
 *
 * realpath() here is load-bearing, not tidiness. libva dlopen()s this
 * driver through /usr/lib64/dri/radeonsi_drv_video.so, which on every
 * install is a SYMLINK to the real install directory, and dladdr() reports
 * the path the object was opened by - not the link target. Using dli_fname
 * unresolved yields /usr/lib64/dri, where the shaders are not, silently
 * defeating the entire point of looking here.
 */
static const char *driver_install_dir(void)
{
    static char dir[PATH_MAX];
    static int resolved;

    if (!resolved) {
        Dl_info info;
        char real_path[PATH_MAX];

        resolved = 1;
        dir[0] = '\0';

        if (dladdr(&bc250_module_anchor, &info) && info.dli_fname &&
            realpath(info.dli_fname, real_path)) {
            char *slash = strrchr(real_path, '/');
            if (slash && slash != real_path) {
                *slash = '\0';
                snprintf(dir, sizeof(dir), "%s", real_path);
            }
        }
    }

    return dir;
}

static VkShaderModule load_spirv_shader(VkDevice device, const char *filename) {
    const char *search_paths[] = {
        "/var/lib/bc250/shaders",
        "/usr/share/bc250/shaders",
        "/usr/local/share/bc250/shaders",
        "/usr/lib64/dri/shaders",
        "/usr/lib/dri/shaders",
        "./shaders",
        "../shaders",
        "../../approach1-compute-encoder/shaders",
        NULL
    };

    FILE *f = NULL;
    char full_path[PATH_MAX];

    const char *env_dir = getenv("BC250_SHADER_DIR");
    if (env_dir && env_dir[0] != '\0') {
        snprintf(full_path, sizeof(full_path), "%s/%s", env_dir, filename);
        f = fopen(full_path, "rb");
    }

    /* Beside the .so itself, before the fixed list: this is where every
     * install script actually puts the shaders. See driver_install_dir(). */
    if (!f) {
        const char *own_dir = driver_install_dir();
        if (own_dir[0] != '\0') {
            snprintf(full_path, sizeof(full_path), "%s/%s", own_dir, filename);
            f = fopen(full_path, "rb");
            if (!f) {
                snprintf(full_path, sizeof(full_path), "%s/shaders/%s", own_dir, filename);
                f = fopen(full_path, "rb");
            }
        }
    }

    if (!f) {
        for (int i = 0; search_paths[i] != NULL; i++) {
            snprintf(full_path, sizeof(full_path), "%s/%s", search_paths[i], filename);
            f = fopen(full_path, "rb");
            if (f) break;
        }
    }

    if (!f) {
        /* Fallback: try raw filename */
        f = fopen(filename, "rb");
    }

    if (!f) {
        fprintf(stderr, "[bc250-gpu] Could not find SPIR-V shader: %s\n", filename);
        return VK_NULL_HANDLE;
    }

    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);

    uint32_t *code = malloc(size);
    if (!code) {
        fclose(f);
        return VK_NULL_HANDLE;
    }
    size_t read_bytes = fread(code, 1, size, f);
    fclose(f);

    if (read_bytes != size || size % 4 != 0) {
        free(code);
        return VK_NULL_HANDLE;
    }

    VkShaderModuleCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = size,
        .pCode = code
    };
    VkShaderModule shader;
    VkResult res = vkCreateShaderModule(device, &create_info, NULL, &shader);
    free(code);

    if (res != VK_SUCCESS) {
        fprintf(stderr, "[bc250-gpu] Failed to create shader module for %s\n", filename);
        return VK_NULL_HANDLE;
    }

    return shader;
}

static VkPipeline create_compute_pipeline(VkDevice device, VkShaderModule shader, VkPipelineLayout layout) {
    if (!shader || !layout) return VK_NULL_HANDLE;

    VkComputePipelineCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main"
        },
        .layout = layout
    };
    VkPipeline pipeline;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, NULL, &pipeline) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return pipeline;
}

int bc250_gpu_init(bc250_gpu_context_t *ctx) {
    memset(ctx, 0, sizeof(*ctx));

    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "BC-250 VCN VA-API Compute Driver",
        .apiVersion = VK_API_VERSION_1_2
    };
    VkInstanceCreateInfo inst_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info
    };
    VK_CHECK(vkCreateInstance(&inst_info, NULL, &ctx->instance));

    uint32_t dev_count = 0;
    vkEnumeratePhysicalDevices(ctx->instance, &dev_count, NULL);
    if (dev_count == 0) {
        fprintf(stderr, "[bc250-gpu] No Vulkan physical devices found!\n");
        return -1;
    }

    VkPhysicalDevice *devices = malloc(dev_count * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(ctx->instance, &dev_count, devices);

    /* 1. Prioritize BC-250 (0x13FE) */
    for (uint32_t i = 0; i < dev_count; i++) {
        vkGetPhysicalDeviceProperties(devices[i], &ctx->dev_props);
        if (ctx->dev_props.deviceID == BC250_DEVICE_ID) {
            ctx->physical_device = devices[i];
            fprintf(stderr, "[bc250-gpu] Found AMD BC-250 APU (0x13FE) - %s\n", ctx->dev_props.deviceName);
            break;
        }
    }

    /* 2. Fallback: Any AMD device */
    if (!ctx->physical_device) {
        for (uint32_t i = 0; i < dev_count; i++) {
            vkGetPhysicalDeviceProperties(devices[i], &ctx->dev_props);
            if (ctx->dev_props.vendorID == AMD_VENDOR_ID) {
                ctx->physical_device = devices[i];
                fprintf(stderr, "[bc250-gpu] BC-250 not found, using AMD GPU: %s\n", ctx->dev_props.deviceName);
                break;
            }
        }
    }

    /* 3. Fallback: Primary compute device */
    if (!ctx->physical_device) {
        ctx->physical_device = devices[0];
        vkGetPhysicalDeviceProperties(devices[0], &ctx->dev_props);
        fprintf(stderr, "[bc250-gpu] Using primary Vulkan device: %s\n", ctx->dev_props.deviceName);
    }
    free(devices);

    /* DIAGNOSTIC ONLY (BC250_PERF_STATS=1), one-time at init: dump every
     * Vulkan memory type this device exposes, to check whether a
     * HOST_VISIBLE|HOST_COHERENT|HOST_CACHED type exists (which would let
     * the GPU-readback staging buffers - see encoder_h264.c's shadow_copy()
     * doc comment - be bulk-read by the CPU at normal cached-RAM speed
     * instead of the current uncached/write-combined type's speed). */
    if (getenv("BC250_PERF_STATS")) {
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(ctx->physical_device, &mp);
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
            VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
            fprintf(stderr, "[bc250-gpu] memtype[%u] heap=%u flags=0x%x%s%s%s%s%s\n",
                    i, mp.memoryTypes[i].heapIndex, f,
                    (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? " DEVICE_LOCAL" : "",
                    (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ? " HOST_VISIBLE" : "",
                    (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ? " HOST_COHERENT" : "",
                    (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? " HOST_CACHED" : "",
                    (f & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) ? " LAZILY_ALLOCATED" : "");
        }
    }

    ctx->max_workgroup_size = ctx->dev_props.limits.maxComputeWorkGroupSize[0];

    /* Find compute queue family */
    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->physical_device, &qf_count, NULL);
    VkQueueFamilyProperties *qf_props = malloc(qf_count * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->physical_device, &qf_count, qf_props);

    /* 1. Prioritize dedicated hardware async compute queue (ACE - Asynchronous
     * Compute Engine, present across GCN and RDNA architectures). */
    ctx->compute_queue_family = (uint32_t)-1;
    for (uint32_t i = 0; i < qf_count; i++) {
        if ((qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            !(qf_props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            ctx->compute_queue_family = i;
            fprintf(stderr, "[bc250-gpu] Using dedicated async compute queue family %u\n", i);
            break;
        }
    }
    /* 2. Fallback to any compute-capable queue */
    if (ctx->compute_queue_family == (uint32_t)-1) {
        for (uint32_t i = 0; i < qf_count; i++) {
            if (qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                ctx->compute_queue_family = i;
                fprintf(stderr, "[bc250-gpu] Using general compute queue family %u\n", i);
                break;
            }
        }
    }
    free(qf_props);
    if (ctx->compute_queue_family == (uint32_t)-1) {
        fprintf(stderr, "[bc250-gpu] No compute queue family available!\n");
        return -1;
    }

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo q_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = ctx->compute_queue_family,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority
    };

    VkPhysicalDeviceVulkan12Features features12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .timelineSemaphore = VK_TRUE
    };

    /* 16-bit has to be ENABLED, not merely supported.
     *
     * deblock_filter, intra_wavefront, quantize and reconstruct declare Int16
     * and StorageBuffer16BitAccess: quantized levels are stored as int16_t,
     * as the buffer sizing above explains. That comment notes the device
     * supports it - but supporting a feature and enabling it are different
     * things in Vulkan, and a shader may only use what vkCreateDevice was
     * asked for. Without these three the SPIR-V is invalid, and the
     * validation layers say so in as many words:
     *
     *   vkCreateShaderModule(): SPIR-V Capability Int16 was declared, but one
     *   of the following requirements is required
     *   (VkPhysicalDeviceFeatures::shaderInt16)
     *
     * What happens when it is used anyway is undefined, and on RADV it is not
     * subtle. Measured on a BC-250: HEVC segfaults inside libvulkan_radeon.so
     * on v0.4.3, and on v0.4.2 it encodes an almost black picture - 5.3 dB
     * PSNR against the source, where libx265 gives 61.8 dB on the same clip.
     * H.264 survives because its hot path happens not to hit those shaders in
     * the same way; that is luck, not design.
     *
     * The device is asked what it actually has rather than assumed at: one
     * that lacks these still gets a working H.264 path and a line in the log
     * saying why, instead of failing somewhere further along.
     */
    VkPhysicalDeviceVulkan11Features have11 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES
    };
    VkPhysicalDeviceFeatures2 have2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &have11
    };
    vkGetPhysicalDeviceFeatures2(ctx->physical_device, &have2);

    VkPhysicalDeviceVulkan11Features features11 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
        .pNext = &features12,
        .storageBuffer16BitAccess = have11.storageBuffer16BitAccess,
        .uniformAndStorageBuffer16BitAccess = have11.uniformAndStorageBuffer16BitAccess
    };
    VkPhysicalDeviceFeatures2 features2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &features11
    };
    features2.features.shaderInt16 = have2.features.shaderInt16;

    if (!have2.features.shaderInt16 || !have11.storageBuffer16BitAccess ||
        !have11.uniformAndStorageBuffer16BitAccess) {
        fprintf(stderr, "[bc250-gpu] warning: this device does not offer 16-bit "
                        "shader support (shaderInt16=%d storageBuffer16=%d "
                        "uniformAndStorageBuffer16=%d) - the shaders that use it "
                        "will misbehave\n",
                (int)have2.features.shaderInt16,
                (int)have11.storageBuffer16BitAccess,
                (int)have11.uniformAndStorageBuffer16BitAccess);
    }

    /* VK_KHR_external_memory_fd (provides vkGetMemoryFdKHR) and
     * VK_EXT_external_memory_dma_buf (adds the DMA_BUF handle type these
     * NV12 images are created/allocated with - see gpu_compute_create_image())
     * are what let gpu_compute_export_nv12_dmabuf() hand out a real DMA-BUF
     * fd for vaExportSurfaceHandle(), needed by real VA-API consumers (e.g.
     * Sunshine's own GL/EGL zero-copy import of the encoder's surfaces) that
     * this driver had no way to satisfy before. Requested opportunistically:
     * if the device doesn't report them (shouldn't happen on RADV, but this
     * driver only ever targets one real piece of silicon - see README - so
     * there's no second real device to have observed this on), device
     * creation still succeeds without them and
     * gpu_compute_export_nv12_dmabuf() simply reports failure via a NULL
     * ctx->get_memory_fd_khr, exactly like every other opportunistic
     * capability check in this file. */
    uint32_t ext_count = 0;
    vkEnumerateDeviceExtensionProperties(ctx->physical_device, NULL, &ext_count, NULL);
    VkExtensionProperties *ext_props = malloc(ext_count * sizeof(VkExtensionProperties));
    vkEnumerateDeviceExtensionProperties(ctx->physical_device, NULL, &ext_count, ext_props);

    bool have_memory_fd = false, have_dma_buf = false, have_semaphore_fd = false;
    bool have_global_priority = false;
    for (uint32_t i = 0; i < ext_count; i++) {
        if (strcmp(ext_props[i].extensionName, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME) == 0) have_memory_fd = true;
        if (strcmp(ext_props[i].extensionName, VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME) == 0) have_dma_buf = true;
        if (strcmp(ext_props[i].extensionName, VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME) == 0) have_semaphore_fd = true;
        if (strcmp(ext_props[i].extensionName, VK_EXT_GLOBAL_PRIORITY_EXTENSION_NAME) == 0) have_global_priority = true;
    }
    free(ext_props);

    const char *device_extensions[4];
    uint32_t device_ext_count = 0;
    if (have_memory_fd && have_dma_buf) {
        device_extensions[device_ext_count++] = VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME;
        device_extensions[device_ext_count++] = VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME;
    } else {
        fprintf(stderr, "[bc250-gpu] VK_KHR_external_memory_fd/VK_EXT_external_memory_dma_buf not available - "
                        "vaExportSurfaceHandle() will report unimplemented\n");
    }
    /* VK_KHR_external_semaphore_fd (provides vkImportSemaphoreFdKHR) - lets
     * gpu_compute_wait_for_image_ready() explicitly wait on whatever GPU
     * work (e.g. Sunshine's own GL rendering into a surface this driver
     * exported) last wrote into a shared render-target surface before this
     * driver's own compute shaders read it. See that function's doc comment
     * in gpu_compute.h. Requested opportunistically, same pattern as
     * have_memory_fd/have_dma_buf above. */
    if (have_semaphore_fd) {
        device_extensions[device_ext_count++] = VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME;
    } else {
        fprintf(stderr, "[bc250-gpu] VK_KHR_external_semaphore_fd not available - "
                        "cannot explicitly wait for cross-context surface writers\n");
    }

    /* BC250_QUEUE_PRIORITY=low|medium|high|realtime asks for a global queue
     * priority instead of the driver-default (MEDIUM).
     *
     * WHY THIS EXISTS: under real GPU contention this encoder's frame is ~675ms
     * of which its shaders only EXECUTE ~2.3ms - the rest is the submission
     * waiting behind the other process's work (DEVLOG 24.4). At 60fps that is
     * only ~14% of the GPU being asked for, yet contention drops the encoder to
     * 1.48fps. Queue *priority*, not more queues and not CPU/GPU overlap, is
     * the mechanism aimed at that wait.
     *
     * PRIVILEGE: measured on this board, HIGH and REALTIME are unavailable to
     * an unprivileged process - vkCreateDevice returns
     * VK_ERROR_NOT_PERMITTED_KHR and the driver's own priority query lists only
     * LOW/MEDIUM. As root all four appear. So this knob does nothing for
     * Sunshine as it currently runs (plain user service, no capabilities)
     * unless CAP_SYS_NICE is granted to it, which is a deliberate system
     * decision, not something this driver should assume.
     *
     * POLICY: raising this above the game does not create GPU time, it
     * reallocates it - the stream gets smoother by making the game wait. That
     * is a reasonable trade on a box whose purpose is streaming, and a bad one
     * elsewhere, which is why there is no default change here.
     *
     * Falls back to the default on NOT_PERMITTED rather than failing init, so
     * an over-optimistic setting degrades instead of breaking the driver. */
    VkDeviceQueueGlobalPriorityCreateInfoEXT gp_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_EXT,
    };
    bool want_priority = false;
    const char *prio_env = getenv("BC250_QUEUE_PRIORITY");
    if (prio_env && have_global_priority) {
        if      (strcmp(prio_env, "low")      == 0) { gp_info.globalPriority = VK_QUEUE_GLOBAL_PRIORITY_LOW_EXT;      want_priority = true; }
        else if (strcmp(prio_env, "medium")   == 0) { gp_info.globalPriority = VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_EXT;   want_priority = true; }
        else if (strcmp(prio_env, "high")     == 0) { gp_info.globalPriority = VK_QUEUE_GLOBAL_PRIORITY_HIGH_EXT;     want_priority = true; }
        else if (strcmp(prio_env, "realtime") == 0) { gp_info.globalPriority = VK_QUEUE_GLOBAL_PRIORITY_REALTIME_EXT; want_priority = true; }
        else fprintf(stderr, "[bc250-gpu] BC250_QUEUE_PRIORITY='%s' not recognised "
                             "(low|medium|high|realtime) - ignoring\n", prio_env);
        if (want_priority) {
            q_info.pNext = &gp_info;
            device_extensions[device_ext_count++] = VK_EXT_GLOBAL_PRIORITY_EXTENSION_NAME;
        }
    } else if (prio_env && !have_global_priority) {
        fprintf(stderr, "[bc250-gpu] BC250_QUEUE_PRIORITY set but "
                        "VK_EXT_global_priority is unavailable - ignoring\n");
    }

    VkDeviceCreateInfo dev_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &features2,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &q_info,
        .enabledExtensionCount = device_ext_count,
        .ppEnabledExtensionNames = device_ext_count > 0 ? device_extensions : NULL
    };
    if (want_priority) {
        VkResult pr = vkCreateDevice(ctx->physical_device, &dev_info, NULL, &ctx->device);
        if (pr == VK_SUCCESS) {
            fprintf(stderr, "[bc250-gpu] queue global priority '%s' GRANTED\n", prio_env);
        } else {
            fprintf(stderr, "[bc250-gpu] queue global priority '%s' REFUSED (VkResult %d%s) - "
                            "falling back to driver default\n", prio_env, (int)pr,
                    pr == VK_ERROR_NOT_PERMITTED_KHR ? " = NOT_PERMITTED, needs CAP_SYS_NICE" : "");
            q_info.pNext = NULL;
            dev_info.enabledExtensionCount = --device_ext_count;
            VK_CHECK(vkCreateDevice(ctx->physical_device, &dev_info, NULL, &ctx->device));
        }
    } else {
        VK_CHECK(vkCreateDevice(ctx->physical_device, &dev_info, NULL, &ctx->device));
    }
    vkGetDeviceQueue(ctx->device, ctx->compute_queue_family, 0, &ctx->compute_queue);

    if (have_memory_fd && have_dma_buf) {
        ctx->get_memory_fd_khr = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(ctx->device, "vkGetMemoryFdKHR");
    }

    ctx->have_external_semaphore_fd = have_semaphore_fd;
    if (have_semaphore_fd) {
        ctx->import_semaphore_fd_khr = (PFN_vkImportSemaphoreFdKHR)vkGetDeviceProcAddr(ctx->device, "vkImportSemaphoreFdKHR");
        if (ctx->import_semaphore_fd_khr) {
            VkSemaphoreCreateInfo sem_info = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
            if (vkCreateSemaphore(ctx->device, &sem_info, NULL, &ctx->image_ready_semaphore) != VK_SUCCESS) {
                fprintf(stderr, "[bc250-gpu] Failed to create image_ready_semaphore\n");
                ctx->have_external_semaphore_fd = false;
            }
        } else {
            ctx->have_external_semaphore_fd = false;
        }
    }

    /* Command Pool */
    VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = ctx->compute_queue_family
    };
    VK_CHECK(vkCreateCommandPool(ctx->device, &pool_info, NULL, &ctx->cmd_pool));

    VkCommandBufferAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = ctx->cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 2
    };
    VK_CHECK(vkAllocateCommandBuffers(ctx->device, &alloc_info, ctx->cmd_bufs));

    /* Fences */
    VkFenceCreateInfo fence_info = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT
    };
    VK_CHECK(vkCreateFence(ctx->device, &fence_info, NULL, &ctx->fences[0]));
    VK_CHECK(vkCreateFence(ctx->device, &fence_info, NULL, &ctx->fences[1]));

    /* Timeline Semaphore */
    VkSemaphoreTypeCreateInfo sem_type_info = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
        .initialValue = 0
    };
    VkSemaphoreCreateInfo sem_info = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = &sem_type_info
    };
    VK_CHECK(vkCreateSemaphore(ctx->device, &sem_info, NULL, &ctx->timeline_sem));
    ctx->timeline_value = 0;

    /* Opt-in GPU per-stage timing (BC250_PERF_STATS=1) - see the
     * BC250_PERF_NUM_TIMESTAMPS comment above and gpu_compute_dispatch_encode()/
     * gpu_compute_sync(). Query pool creation failure or a device that
     * doesn't expose compute-queue timestamps just disables the feature;
     * it is a pure diagnostic and must never affect the encode path. */
    ctx->perf_stats_enabled = false;
    ctx->timestamp_period_ns = 0.0;
    ctx->perf_frame_counter = 0;
    const char *perf_env = getenv("BC250_PERF_STATS");
    if (perf_env && (strcmp(perf_env, "1") == 0 || strcmp(perf_env, "true") == 0)) {
        if (ctx->dev_props.limits.timestampComputeAndGraphics) {
            ctx->timestamp_period_ns = (double)ctx->dev_props.limits.timestampPeriod;
            VkQueryPoolCreateInfo qp_info = {
                .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                .queryType = VK_QUERY_TYPE_TIMESTAMP,
                .queryCount = BC250_PERF_NUM_TIMESTAMPS
            };
            bool pools_ok = true;
            for (int i = 0; i < 2; i++) {
                if (vkCreateQueryPool(ctx->device, &qp_info, NULL, &ctx->timestamp_pools[i]) != VK_SUCCESS) {
                    pools_ok = false;
                    break;
                }
            }
            if (pools_ok) {
                ctx->perf_stats_enabled = true;
                fprintf(stderr, "[bc250-gpu] BC250_PERF_STATS enabled (timestampPeriod=%.4f ns/tick)\n", ctx->timestamp_period_ns);
            } else {
                fprintf(stderr, "[bc250-gpu] BC250_PERF_STATS: failed to create timestamp query pools, disabling\n");
            }
        } else {
            fprintf(stderr, "[bc250-gpu] BC250_PERF_STATS: device does not report timestampComputeAndGraphics support, disabling\n");
        }
    }

    /* Create Descriptor Set Layouts */
    VkDescriptorSetLayoutBinding me_bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}
    };
    VkDescriptorSetLayoutCreateInfo me_layout_info = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 3, .pBindings = me_bindings };
    vkCreateDescriptorSetLayout(ctx->device, &me_layout_info, NULL, &ctx->me_desc_layout);

    /* residual_predict.comp: current Y/UV images, reference Y image, real
     * MVs, its residual_buffer output and its pred_mode_buffer output. See
     * that shader's top-of-file comment. */
    VkDescriptorSetLayoutBinding predict_bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        /* binding 6: referenceUV - previous frame's chroma plane, for real
         * P-slice chroma motion compensation (see residual_predict.comp). */
        {6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        /* binding 7: PredOut - retained prediction value, consumed by
         * reconstruct.comp (see that shader's top-of-file comment). */
        {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}
    };
    VkDescriptorSetLayoutCreateInfo predict_layout_info = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 8, .pBindings = predict_bindings };
    vkCreateDescriptorSetLayout(ctx->device, &predict_layout_info, NULL, &ctx->predict_desc_layout);

    VkDescriptorSetLayoutBinding dct_bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        /* binding 2 = dc_coeff_buffer, the compact per-block DC the CPU reads
         * instead of the full coeff readback (gpu_compute.h's dc_coeff_buffer). */
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}
    };
    VkDescriptorSetLayoutCreateInfo dct_layout_info = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 3, .pBindings = dct_bindings };
    vkCreateDescriptorSetLayout(ctx->device, &dct_layout_info, NULL, &ctx->dct_desc_layout);

    VkDescriptorSetLayoutBinding quant_bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}
    };
    VkDescriptorSetLayoutCreateInfo quant_layout_info = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 3, .pBindings = quant_bindings };
    vkCreateDescriptorSetLayout(ctx->device, &quant_layout_info, NULL, &ctx->quant_desc_layout);

    VkDescriptorSetLayoutBinding deblock_bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}
    };
    VkDescriptorSetLayoutCreateInfo deblock_layout_info = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 3, .pBindings = deblock_bindings };
    vkCreateDescriptorSetLayout(ctx->device, &deblock_layout_info, NULL, &ctx->deblock_desc_layout);

    VkDescriptorSetLayoutBinding entropy_bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}
    };
    VkDescriptorSetLayoutCreateInfo entropy_layout_info = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = entropy_bindings };
    vkCreateDescriptorSetLayout(ctx->device, &entropy_layout_info, NULL, &ctx->entropy_desc_layout);

    VkDescriptorSetLayoutBinding cc_bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}
    };
    VkDescriptorSetLayoutCreateInfo cc_layout_info = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 3, .pBindings = cc_bindings };
    vkCreateDescriptorSetLayout(ctx->device, &cc_layout_info, NULL, &ctx->cc_desc_layout);

    /* reconstruct.comp: quant_levels_buffer + coeff_buffer + pred_buffer
     * (readonly), recon Y/UV images (writeonly) - see that shader's
     * top-of-file comment. */
    VkDescriptorSetLayoutBinding reconstruct_bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}
    };
    VkDescriptorSetLayoutCreateInfo reconstruct_layout_info = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 5, .pBindings = reconstruct_bindings };
    vkCreateDescriptorSetLayout(ctx->device, &reconstruct_layout_info, NULL, &ctx->reconstruct_desc_layout);

    /* intra_wavefront.comp: current Y/UV (readonly), recon Y/UV (read-write -
     * see gpu_compute.h's comment), quant_levels_buffer/coeff_buffer/
     * pred_mode_buffer (writeonly). I-slice-only, diagonal-wavefront
     * dispatch - see that shader's top-of-file comment and
     * gpu_compute_dispatch_encode() below. */
    VkDescriptorSetLayoutBinding intra_wavefront_bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        /* binding 7 = nz_count_buffer (the per-block nonzero mask). This
         * shader writes quant levels for I-slices on its own path, entirely
         * bypassing quantize.comp, so without writing the mask here too the
         * mask would be a stale P-frame's on every I-frame. Measured: a
         * BC250_NZ_AUDIT run caught exactly that - 13443 of 345600 blocks
         * mismatched on frame 0 and zero mismatches on every P frame. */
        {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        /* binding 8 = dc_coeff_buffer. Same reason as binding 7: this shader
         * is the I-slice path and bypasses dct_transform.comp entirely, so it
         * has to maintain the compact DC buffer itself or every I-frame's DC
         * would be a leftover from the previous P-frame. */
        {8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}
    };
    VkDescriptorSetLayoutCreateInfo intra_wavefront_layout_info = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 9, .pBindings = intra_wavefront_bindings };
    vkCreateDescriptorSetLayout(ctx->device, &intra_wavefront_layout_info, NULL, &ctx->intra_wavefront_desc_layout);

    /* VideoProc: source luma, source chroma, destination luma,
     * destination chroma. */
    VkDescriptorSetLayoutBinding vpp_bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}
    };
    VkDescriptorSetLayoutCreateInfo vpp_layout_info = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 4, .pBindings = vpp_bindings };
    vkCreateDescriptorSetLayout(ctx->device, &vpp_layout_info, NULL, &ctx->vpp_desc_layout);

    /* Descriptor Pool */
    /* Headroom, not a tight fit. The nine layouts above bind 24 storage
     * buffers and 14 storage images today; adding the nonzero mask and the
     * compact DC buffer used 3 of the old 32-descriptor margin in one sitting.
     * vkAllocateDescriptorSets()'s result is not checked at its call sites, so
     * exhausting this pool would fail the same silent way an exhausted memory
     * heap did (DEVLOG §19.7) - cheaper to keep the ceiling far away. */
    VkDescriptorPoolSize pool_sizes[] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 64},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64}
    };
    VkDescriptorPoolCreateInfo pool_info_desc = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 32,
        .poolSizeCount = 2,
        .pPoolSizes = pool_sizes
    };
    vkCreateDescriptorPool(ctx->device, &pool_info_desc, NULL, &ctx->desc_pool);

    /* Push constants. 9th word (num_slices) is only read by
     * residual_predict.comp (see its SLICE BOUNDARIES comment) - and is
     * separately repurposed as deblock_filter.comp's `pass` flag
     * (0=vertical edges, 1=horizontal edges - see that shader's
     * PushConstants comment and gpu_compute_dispatch_encode()'s Stage 5);
     * 10th word (diagonal) is only read by intra_wavefront.comp (see its
     * DISPATCH SHAPE comment) - every other shader still only declares the
     * first 8 (or 9) words in its own PushConstants block, which is fine,
     * they just don't read the extra tail byte range this layout now
     * allows. */
    VkPushConstantRange pc_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(uint32_t) * 10
    };

    /* Pipeline Layouts */
    VkPipelineLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pc_range,
        .setLayoutCount = 1
    };

    layout_info.pSetLayouts = &ctx->me_desc_layout;
    vkCreatePipelineLayout(ctx->device, &layout_info, NULL, &ctx->motion_est_layout);

    layout_info.pSetLayouts = &ctx->predict_desc_layout;
    vkCreatePipelineLayout(ctx->device, &layout_info, NULL, &ctx->predict_layout);

    layout_info.pSetLayouts = &ctx->dct_desc_layout;
    vkCreatePipelineLayout(ctx->device, &layout_info, NULL, &ctx->transform_layout);

    layout_info.pSetLayouts = &ctx->quant_desc_layout;
    vkCreatePipelineLayout(ctx->device, &layout_info, NULL, &ctx->quantize_layout);

    layout_info.pSetLayouts = &ctx->deblock_desc_layout;
    vkCreatePipelineLayout(ctx->device, &layout_info, NULL, &ctx->deblock_layout);

    layout_info.pSetLayouts = &ctx->entropy_desc_layout;
    vkCreatePipelineLayout(ctx->device, &layout_info, NULL, &ctx->entropy_layout);

    layout_info.pSetLayouts = &ctx->cc_desc_layout;
    vkCreatePipelineLayout(ctx->device, &layout_info, NULL, &ctx->color_convert_layout);

    layout_info.pSetLayouts = &ctx->reconstruct_desc_layout;
    vkCreatePipelineLayout(ctx->device, &layout_info, NULL, &ctx->reconstruct_layout);

    layout_info.pSetLayouts = &ctx->intra_wavefront_desc_layout;
    vkCreatePipelineLayout(ctx->device, &layout_info, NULL, &ctx->intra_wavefront_layout);

    layout_info.pSetLayouts = &ctx->vpp_desc_layout;
    vkCreatePipelineLayout(ctx->device, &layout_info, NULL, &ctx->vpp_layout);

    /* Allocate Descriptor Sets */
    VkDescriptorSetAllocateInfo alloc_set_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = ctx->desc_pool,
        .descriptorSetCount = 1
    };

    alloc_set_info.pSetLayouts = &ctx->me_desc_layout;
    vkAllocateDescriptorSets(ctx->device, &alloc_set_info, &ctx->me_desc_set);

    alloc_set_info.pSetLayouts = &ctx->predict_desc_layout;
    vkAllocateDescriptorSets(ctx->device, &alloc_set_info, &ctx->predict_desc_set);

    alloc_set_info.pSetLayouts = &ctx->dct_desc_layout;
    vkAllocateDescriptorSets(ctx->device, &alloc_set_info, &ctx->dct_desc_set);

    alloc_set_info.pSetLayouts = &ctx->quant_desc_layout;
    vkAllocateDescriptorSets(ctx->device, &alloc_set_info, &ctx->quant_desc_set);

    alloc_set_info.pSetLayouts = &ctx->deblock_desc_layout;
    vkAllocateDescriptorSets(ctx->device, &alloc_set_info, &ctx->deblock_desc_set);

    alloc_set_info.pSetLayouts = &ctx->entropy_desc_layout;
    vkAllocateDescriptorSets(ctx->device, &alloc_set_info, &ctx->entropy_desc_set);

    alloc_set_info.pSetLayouts = &ctx->cc_desc_layout;
    vkAllocateDescriptorSets(ctx->device, &alloc_set_info, &ctx->cc_desc_set);

    alloc_set_info.pSetLayouts = &ctx->vpp_desc_layout;
    vkAllocateDescriptorSets(ctx->device, &alloc_set_info, &ctx->vpp_desc_set);

    alloc_set_info.pSetLayouts = &ctx->reconstruct_desc_layout;
    vkAllocateDescriptorSets(ctx->device, &alloc_set_info, &ctx->reconstruct_desc_set);

    alloc_set_info.pSetLayouts = &ctx->intra_wavefront_desc_layout;
    vkAllocateDescriptorSets(ctx->device, &alloc_set_info, &ctx->intra_wavefront_desc_set);

    /* Shaders & Pipelines */
    VkShaderModule me_shader = load_spirv_shader(ctx->device, "motion_estimation.comp.spv");
    if (me_shader) {
        ctx->motion_est_pipeline = create_compute_pipeline(ctx->device, me_shader, ctx->motion_est_layout);
        vkDestroyShaderModule(ctx->device, me_shader, NULL);
    }
    VkShaderModule predict_shader = load_spirv_shader(ctx->device, "residual_predict.comp.spv");
    if (predict_shader) {
        ctx->predict_pipeline = create_compute_pipeline(ctx->device, predict_shader, ctx->predict_layout);
        vkDestroyShaderModule(ctx->device, predict_shader, NULL);
    }
    VkShaderModule dct_shader = load_spirv_shader(ctx->device, "dct_transform.comp.spv");
    if (dct_shader) {
        ctx->transform_pipeline = create_compute_pipeline(ctx->device, dct_shader, ctx->transform_layout);
        vkDestroyShaderModule(ctx->device, dct_shader, NULL);
    }
    VkShaderModule quant_shader = load_spirv_shader(ctx->device, "quantize.comp.spv");
    if (quant_shader) {
        ctx->quantize_pipeline = create_compute_pipeline(ctx->device, quant_shader, ctx->quantize_layout);
        vkDestroyShaderModule(ctx->device, quant_shader, NULL);
    }
    VkShaderModule deblock_shader = load_spirv_shader(ctx->device, "deblock_filter.comp.spv");
    if (deblock_shader) {
        ctx->deblock_pipeline = create_compute_pipeline(ctx->device, deblock_shader, ctx->deblock_layout);
        vkDestroyShaderModule(ctx->device, deblock_shader, NULL);
    }
    VkShaderModule entropy_shader = load_spirv_shader(ctx->device, "entropy_encode.comp.spv");
    if (entropy_shader) {
        ctx->entropy_pipeline = create_compute_pipeline(ctx->device, entropy_shader, ctx->entropy_layout);
        vkDestroyShaderModule(ctx->device, entropy_shader, NULL);
    }
    /* ⚠️ Missing shaders are not fatal here, deliberately: the driver
     * still decodes and encodes without them, and gpu_compute_video_proc()
     * checks for the pipeline before using it. A VideoProc context is
     * refused at vaCreateConfig instead, which is a clear error rather
     * than a crash halfway through a frame. */
    VkShaderModule vpp_shader = load_spirv_shader(ctx->device, "video_proc.comp.spv");
    if (vpp_shader != VK_NULL_HANDLE) {
        ctx->vpp_pipeline = create_compute_pipeline(ctx->device, vpp_shader, ctx->vpp_layout);
        vkDestroyShaderModule(ctx->device, vpp_shader, NULL);
    }

    VkShaderModule vpp10_shader = load_spirv_shader(ctx->device, "video_proc10.comp.spv");
    if (vpp10_shader != VK_NULL_HANDLE) {
        ctx->vpp_pipeline10 = create_compute_pipeline(ctx->device, vpp10_shader, ctx->vpp_layout);
        vkDestroyShaderModule(ctx->device, vpp10_shader, NULL);
    }

    VkShaderModule cc_shader = load_spirv_shader(ctx->device, "color_convert.comp.spv");
    if (cc_shader) {
        ctx->color_convert_pipeline = create_compute_pipeline(ctx->device, cc_shader, ctx->color_convert_layout);
        vkDestroyShaderModule(ctx->device, cc_shader, NULL);
    }
    VkShaderModule reconstruct_shader = load_spirv_shader(ctx->device, "reconstruct.comp.spv");
    if (reconstruct_shader) {
        ctx->reconstruct_pipeline = create_compute_pipeline(ctx->device, reconstruct_shader, ctx->reconstruct_layout);
        vkDestroyShaderModule(ctx->device, reconstruct_shader, NULL);
        if (!ctx->reconstruct_pipeline) {
            fprintf(stderr, "[bc250-gpu] FAILED to create reconstruct_pipeline (shader loaded but pipeline creation failed)\n");
        }
    } else {
        fprintf(stderr, "[bc250-gpu] FAILED to load reconstruct.comp.spv shader module\n");
    }
    VkShaderModule intra_wavefront_shader = load_spirv_shader(ctx->device, "intra_wavefront.comp.spv");
    if (intra_wavefront_shader) {
        ctx->intra_wavefront_pipeline = create_compute_pipeline(ctx->device, intra_wavefront_shader, ctx->intra_wavefront_layout);
        vkDestroyShaderModule(ctx->device, intra_wavefront_shader, NULL);
        if (!ctx->intra_wavefront_pipeline) {
            fprintf(stderr, "[bc250-gpu] FAILED to create intra_wavefront_pipeline (shader loaded but pipeline creation failed)\n");
        }
    } else {
        fprintf(stderr, "[bc250-gpu] FAILED to load intra_wavefront.comp.spv shader module\n");
    }

    /* Encoding buffers are allocated LAZILY, on the first
     * gpu_compute_dispatch_encode() at the real resolution (and again on any
     * resolution change) - see that function's staging_buffers[0] ==
     * VK_NULL_HANDLE check.
     *
     * This used to eagerly allocate for 3840x2160 here, on the reasoning that
     * 4K is the worst case so allocating once avoids a reallocation later.
     * That cost roughly 440 MB of VkDeviceMemory per context - at 4K,
     * quant_levels/coeff/residual/pred are 49.8 MB each device-local, and the
     * host-visible quant/coeff staging PAIRS are another ~200 MB - all of it
     * thrown away and reallocated by the first dispatch at the actual
     * resolution.
     *
     * On this hardware that is not affordable. Vulkan exposes ~7.95 GiB here,
     * split as a 2.65 GiB host-visible heap (every HOST_VISIBLE memory type)
     * and a 5.30 GiB DEVICE_LOCAL heap - it is a unified-memory APU, so this
     * is the GART/GTT aperture, NOT the 512 MB mem_info_vram_total carve-out
     * (an earlier version of this comment claimed the latter; see DEVLOG §21).
     * Sunshine's encoder probe calls bc250_gpu_init 20 times, and 20 x ~431 MiB
     * exceeds 7.95 GiB - the host-visible half worst, at 20 x ~222 MiB against
     * 2.65 GiB, which is why the failures clustered on
     * create_buffer_with_memory_preferred(). The shipped v0.3.0 driver was
     * already logging 9 VK_ERROR_OUT_OF_DEVICE_MEMORY failures per probe and
     * surviving only because the allocations that happened to fail were ones
     * nothing dereferenced. Adding two more small staging buffers took it to
     * 18 failures and a SEGV in this function - which is how this was found.
     *
     * Removing it is safe because nothing reads the encoding buffers before
     * the first dispatch: gpu_compute_get_*_staging_data() all return -1 while
     * their mapped pointer is NULL, and encoder_h264.c already treats that as
     * "no GPU output this frame" and falls back. */

    return 0;
}

void bc250_gpu_destroy(bc250_gpu_context_t *ctx) {
    if (!ctx->device) return;

    vkDeviceWaitIdle(ctx->device);

    if (ctx->motion_est_pipeline) vkDestroyPipeline(ctx->device, ctx->motion_est_pipeline, NULL);
    if (ctx->predict_pipeline) vkDestroyPipeline(ctx->device, ctx->predict_pipeline, NULL);
    if (ctx->transform_pipeline) vkDestroyPipeline(ctx->device, ctx->transform_pipeline, NULL);
    if (ctx->quantize_pipeline) vkDestroyPipeline(ctx->device, ctx->quantize_pipeline, NULL);
    if (ctx->deblock_pipeline) vkDestroyPipeline(ctx->device, ctx->deblock_pipeline, NULL);
    if (ctx->entropy_pipeline) vkDestroyPipeline(ctx->device, ctx->entropy_pipeline, NULL);
    if (ctx->color_convert_pipeline) vkDestroyPipeline(ctx->device, ctx->color_convert_pipeline, NULL);
    if (ctx->vpp_pipeline) vkDestroyPipeline(ctx->device, ctx->vpp_pipeline, NULL);
    if (ctx->vpp_pipeline10) vkDestroyPipeline(ctx->device, ctx->vpp_pipeline10, NULL);
    if (ctx->reconstruct_pipeline) vkDestroyPipeline(ctx->device, ctx->reconstruct_pipeline, NULL);
    if (ctx->intra_wavefront_pipeline) vkDestroyPipeline(ctx->device, ctx->intra_wavefront_pipeline, NULL);

    if (ctx->motion_est_layout) vkDestroyPipelineLayout(ctx->device, ctx->motion_est_layout, NULL);
    if (ctx->predict_layout) vkDestroyPipelineLayout(ctx->device, ctx->predict_layout, NULL);
    if (ctx->transform_layout) vkDestroyPipelineLayout(ctx->device, ctx->transform_layout, NULL);
    if (ctx->quantize_layout) vkDestroyPipelineLayout(ctx->device, ctx->quantize_layout, NULL);
    if (ctx->deblock_layout) vkDestroyPipelineLayout(ctx->device, ctx->deblock_layout, NULL);
    if (ctx->entropy_layout) vkDestroyPipelineLayout(ctx->device, ctx->entropy_layout, NULL);
    if (ctx->color_convert_layout) vkDestroyPipelineLayout(ctx->device, ctx->color_convert_layout, NULL);
    if (ctx->vpp_layout) vkDestroyPipelineLayout(ctx->device, ctx->vpp_layout, NULL);
    if (ctx->reconstruct_layout) vkDestroyPipelineLayout(ctx->device, ctx->reconstruct_layout, NULL);
    if (ctx->intra_wavefront_layout) vkDestroyPipelineLayout(ctx->device, ctx->intra_wavefront_layout, NULL);

    if (ctx->me_desc_layout) vkDestroyDescriptorSetLayout(ctx->device, ctx->me_desc_layout, NULL);
    if (ctx->predict_desc_layout) vkDestroyDescriptorSetLayout(ctx->device, ctx->predict_desc_layout, NULL);
    if (ctx->dct_desc_layout) vkDestroyDescriptorSetLayout(ctx->device, ctx->dct_desc_layout, NULL);
    if (ctx->quant_desc_layout) vkDestroyDescriptorSetLayout(ctx->device, ctx->quant_desc_layout, NULL);
    if (ctx->deblock_desc_layout) vkDestroyDescriptorSetLayout(ctx->device, ctx->deblock_desc_layout, NULL);
    if (ctx->entropy_desc_layout) vkDestroyDescriptorSetLayout(ctx->device, ctx->entropy_desc_layout, NULL);
    if (ctx->cc_desc_layout) vkDestroyDescriptorSetLayout(ctx->device, ctx->cc_desc_layout, NULL);
    if (ctx->vpp_desc_layout) vkDestroyDescriptorSetLayout(ctx->device, ctx->vpp_desc_layout, NULL);
    if (ctx->reconstruct_desc_layout) vkDestroyDescriptorSetLayout(ctx->device, ctx->reconstruct_desc_layout, NULL);
    if (ctx->intra_wavefront_desc_layout) vkDestroyDescriptorSetLayout(ctx->device, ctx->intra_wavefront_desc_layout, NULL);

    if (ctx->desc_pool) vkDestroyDescriptorPool(ctx->device, ctx->desc_pool, NULL);

    if (ctx->mv_buffer) { vkDestroyBuffer(ctx->device, ctx->mv_buffer, NULL); vkFreeMemory(ctx->device, ctx->mv_memory, NULL); }
    if (ctx->residual_buffer) { vkDestroyBuffer(ctx->device, ctx->residual_buffer, NULL); vkFreeMemory(ctx->device, ctx->residual_memory, NULL); }
    if (ctx->pred_buffer) { vkDestroyBuffer(ctx->device, ctx->pred_buffer, NULL); vkFreeMemory(ctx->device, ctx->pred_memory, NULL); }
    if (ctx->coeff_buffer) { vkDestroyBuffer(ctx->device, ctx->coeff_buffer, NULL); vkFreeMemory(ctx->device, ctx->coeff_memory, NULL); }
    if (ctx->dc_coeff_buffer) { vkDestroyBuffer(ctx->device, ctx->dc_coeff_buffer, NULL); vkFreeMemory(ctx->device, ctx->dc_coeff_memory, NULL); }
    if (ctx->quant_levels_buffer) { vkDestroyBuffer(ctx->device, ctx->quant_levels_buffer, NULL); vkFreeMemory(ctx->device, ctx->quant_levels_memory, NULL); }
    if (ctx->nz_count_buffer) { vkDestroyBuffer(ctx->device, ctx->nz_count_buffer, NULL); vkFreeMemory(ctx->device, ctx->nz_count_memory, NULL); }
    if (ctx->pred_mode_buffer) { vkDestroyBuffer(ctx->device, ctx->pred_mode_buffer, NULL); vkFreeMemory(ctx->device, ctx->pred_mode_memory, NULL); }
    if (ctx->entropy_buffer) { vkDestroyBuffer(ctx->device, ctx->entropy_buffer, NULL); vkFreeMemory(ctx->device, ctx->entropy_memory, NULL); }
    for (int i = 0; i < 2; i++) {
        if (ctx->staging_mapped[i]) {
            vkUnmapMemory(ctx->device, ctx->staging_memories[i]);
            ctx->staging_mapped[i] = NULL;
        }
        if (ctx->staging_buffers[i]) {
            vkDestroyBuffer(ctx->device, ctx->staging_buffers[i], NULL);
            vkFreeMemory(ctx->device, ctx->staging_memories[i], NULL);
        }
        if (ctx->quant_staging_mapped[i]) {
            vkUnmapMemory(ctx->device, ctx->quant_staging_memories[i]);
            ctx->quant_staging_mapped[i] = NULL;
        }
        if (ctx->quant_staging_buffers[i]) {
            vkDestroyBuffer(ctx->device, ctx->quant_staging_buffers[i], NULL);
            vkFreeMemory(ctx->device, ctx->quant_staging_memories[i], NULL);
        }
        if (ctx->dc_staging_mapped[i]) {
            vkUnmapMemory(ctx->device, ctx->dc_staging_memories[i]);
            ctx->dc_staging_mapped[i] = NULL;
        }
        if (ctx->dc_staging_buffers[i]) {
            vkDestroyBuffer(ctx->device, ctx->dc_staging_buffers[i], NULL);
            vkFreeMemory(ctx->device, ctx->dc_staging_memories[i], NULL);
        }
        if (ctx->pred_mode_staging_mapped[i]) {
            vkUnmapMemory(ctx->device, ctx->pred_mode_staging_memories[i]);
            ctx->pred_mode_staging_mapped[i] = NULL;
        }
        if (ctx->pred_mode_staging_buffers[i]) {
            vkDestroyBuffer(ctx->device, ctx->pred_mode_staging_buffers[i], NULL);
            vkFreeMemory(ctx->device, ctx->pred_mode_staging_memories[i], NULL);
        }
        if (ctx->mv_staging_mapped[i]) {
            vkUnmapMemory(ctx->device, ctx->mv_staging_memories[i]);
            ctx->mv_staging_mapped[i] = NULL;
        }
        if (ctx->mv_staging_buffers[i]) {
            vkDestroyBuffer(ctx->device, ctx->mv_staging_buffers[i], NULL);
            vkFreeMemory(ctx->device, ctx->mv_staging_memories[i], NULL);
        }
        if (ctx->nz_staging_mapped[i]) {
            vkUnmapMemory(ctx->device, ctx->nz_staging_memories[i]);
            ctx->nz_staging_mapped[i] = NULL;
        }
        if (ctx->nz_staging_buffers[i]) {
            vkDestroyBuffer(ctx->device, ctx->nz_staging_buffers[i], NULL);
            vkFreeMemory(ctx->device, ctx->nz_staging_memories[i], NULL);
        }
    }

    if (ctx->recon_image.y_plane) {
        gpu_compute_destroy_image(ctx, ctx->recon_image, ctx->recon_memory);
        ctx->recon_image.y_plane = VK_NULL_HANDLE;
    }

    if (ctx->timestamp_pools[0]) vkDestroyQueryPool(ctx->device, ctx->timestamp_pools[0], NULL);
    if (ctx->timestamp_pools[1]) vkDestroyQueryPool(ctx->device, ctx->timestamp_pools[1], NULL);
    if (ctx->timeline_sem) vkDestroySemaphore(ctx->device, ctx->timeline_sem, NULL);
    if (ctx->image_ready_semaphore) vkDestroySemaphore(ctx->device, ctx->image_ready_semaphore, NULL);
    if (ctx->fences[0]) vkDestroyFence(ctx->device, ctx->fences[0], NULL);
    if (ctx->fences[1]) vkDestroyFence(ctx->device, ctx->fences[1], NULL);
    if (ctx->cmd_pool) vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
    if (ctx->device) vkDestroyDevice(ctx->device, NULL);
    if (ctx->instance) vkDestroyInstance(ctx->instance, NULL);
}

int gpu_compute_init(gpu_context_t *ctx) {
    return bc250_gpu_init(ctx);
}

void gpu_compute_terminate(gpu_context_t *ctx) {
    bc250_gpu_destroy(ctx);
}

/* gpu_compute_create_image()'s allocate+bind step (below) can fail
 * transiently under real concurrent GPU contention - confirmed on-hardware
 * (gdb) that this GPU's non-device-local memory type becomes unstable
 * under repeated back-to-back allocation pressure from multiple real
 * Vulkan/GL clients (e.g. a live desktop compositor): a bind that fails
 * cleanly (VK_ERROR_UNKNOWN) once or twice can, on a later attempt made
 * immediately afterward, segfault *inside* RADV's own
 * radv_BindImageMemory2() instead of returning another clean error. This
 * isn't fixable from our side (it's inside Mesa), but retrying with a
 * short backoff instead of failing (or being retried) immediately is a
 * standard, well-precedented workaround for exactly this class of
 * transient-allocator-failure-under-contention issue - e.g. AMD's own
 * Vulkan Memory Allocator library exists in part because per-resource
 * vkAllocateMemory/vkBindImageMemory calls are known to be fragile under
 * contention on real GPU drivers, and DXVK retries transient allocation
 * failures rather than treating the first one as fatal. Giving the
 * contention a moment to clear between attempts is what actually matters
 * here: an immediate retry is exactly the pattern that reproduced the
 * segfault above. */
#define BC250_ALLOC_MAX_ATTEMPTS 7

int gpu_compute_create_image(gpu_context_t *ctx, int width, int height, int format, gpu_image_t *image, gpu_memory_t *memory) {
    /* One sample per plane component, eight bits or sixteen. Everything
     * else about the allocation - the two separate images, the packed
     * bind, the export flags - is the same either way, so the format
     * reaches only these two VkFormats. */
    const VkFormat vk_y = (format == GPU_IMAGE_P010)
                          ? VK_FORMAT_R16_UNORM : VK_FORMAT_R8_UNORM;
    const VkFormat vk_uv = (format == GPU_IMAGE_P010)
                           ? VK_FORMAT_R16G16_UNORM : VK_FORMAT_R8G8_UNORM;

    image->format = format;
    image->width = width;
    image->height = height;
    /* Matches the real initialLayout used below for both y_plane and uv_plane. */
    image->current_layout = VK_IMAGE_LAYOUT_PREINITIALIZED;

    VkResult result = VK_ERROR_UNKNOWN;
    for (int attempt = 0; attempt < BC250_ALLOC_MAX_ATTEMPTS; attempt++) {
        if (attempt > 0) {
            long backoff_ms = 20L << (attempt - 1); /* 20, 40, 80 ms */
            struct timespec ts = { .tv_sec = backoff_ms / 1000, .tv_nsec = (backoff_ms % 1000) * 1000000L };
            nanosleep(&ts, NULL);
            fprintf(stderr, "[bc250-gpu] Retrying image allocation (attempt %d/%d) after %ldms backoff\n",
                    attempt + 1, BC250_ALLOC_MAX_ATTEMPTS, backoff_ms);
        }

        /* Chained onto both planes' VkImageCreateInfo (only when the device
         * actually enabled the extensions - see bc250_gpu_init()) so the
         * memory they get bound to below is created as DMA_BUF-exportable.
         * Required for gpu_compute_export_nv12_dmabuf()/
         * vaExportSurfaceHandle(): a real external-API consumer (e.g.
         * Sunshine's own GL/EGL import of this surface) needs a real
         * DMA-BUF fd, and Vulkan requires images that will ever be bound to
         * externally-exportable memory to declare that up front here, not
         * just at vkAllocateMemory time. */
        VkExternalMemoryImageCreateInfo ext_image_info = {
            .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
            .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT
        };
        VkImageCreateInfo y_info = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = ctx->get_memory_fd_khr ? &ext_image_info : NULL,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = vk_y,
            .extent = { (uint32_t)width, (uint32_t)height, 1 },
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_LINEAR,
            .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            .initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED
        };
        result = vkCreateImage(ctx->device, &y_info, NULL, &image->y_plane);
        if (result != VK_SUCCESS) {
            fprintf(stderr, "[bc250-gpu] Vulkan error %d at %s:%d\n", result, __FILE__, __LINE__);
            continue;
        }

        VkImageCreateInfo uv_info = y_info;
        uv_info.format = vk_uv;
        uv_info.extent.width = width / 2;
        uv_info.extent.height = height / 2;
        result = vkCreateImage(ctx->device, &uv_info, NULL, &image->uv_plane);
        if (result != VK_SUCCESS) {
            fprintf(stderr, "[bc250-gpu] Vulkan error %d at %s:%d\n", result, __FILE__, __LINE__);
            vkDestroyImage(ctx->device, image->y_plane, NULL);
            image->y_plane = VK_NULL_HANDLE;
            continue;
        }

        VkMemoryRequirements y_req, uv_req;
        vkGetImageMemoryRequirements(ctx->device, image->y_plane, &y_req);
        vkGetImageMemoryRequirements(ctx->device, image->uv_plane, &uv_req);

        VkDeviceSize align = uv_req.alignment > y_req.alignment ? uv_req.alignment : y_req.alignment;
        VkDeviceSize uv_offset = (y_req.size + align - 1) & ~(align - 1);
        VkDeviceSize total_size = uv_offset + uv_req.size;

        memory->size = total_size;
        memory->mapped_ptr = NULL;

        uint32_t mem_bits = y_req.memoryTypeBits & uv_req.memoryTypeBits;
        if (mem_bits == 0) mem_bits = y_req.memoryTypeBits | uv_req.memoryTypeBits;
        if (getenv("BC250_DEBUG_MEMTYPE")) {
            uint32_t chosen = find_memory_type(ctx->physical_device, mem_bits,
                                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            VkPhysicalDeviceMemoryProperties mp;
            vkGetPhysicalDeviceMemoryProperties(ctx->physical_device, &mp);
            fprintf(stderr, "[bc250-gpu] DEBUG mem_bits=0x%x chosen_type=%u heap=%u type_flags=0x%x heap_size=%zu\n",
                    mem_bits, chosen, mp.memoryTypes[chosen].heapIndex, mp.memoryTypes[chosen].propertyFlags,
                    (size_t)mp.memoryHeaps[mp.memoryTypes[chosen].heapIndex].size);
        }

        VkExportMemoryAllocateInfo ext_mem_info = {
            .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
            .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT
        };
        VkMemoryAllocateInfo alloc_info = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = ctx->get_memory_fd_khr ? &ext_mem_info : NULL,
            .allocationSize = total_size,
            .memoryTypeIndex = find_memory_type(ctx->physical_device, mem_bits,
                                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        };
        result = vkAllocateMemory(ctx->device, &alloc_info, NULL, &memory->memory);
        if (result != VK_SUCCESS) {
            fprintf(stderr, "[bc250-gpu] Vulkan error %d at %s:%d\n", result, __FILE__, __LINE__);
            vkDestroyImage(ctx->device, image->y_plane, NULL);
            vkDestroyImage(ctx->device, image->uv_plane, NULL);
            image->y_plane = VK_NULL_HANDLE;
            image->uv_plane = VK_NULL_HANDLE;
            continue;
        }

        result = vkBindImageMemory(ctx->device, image->y_plane, memory->memory, 0);
        if (result == VK_SUCCESS) {
            result = vkBindImageMemory(ctx->device, image->uv_plane, memory->memory, uv_offset);
        }
        if (result != VK_SUCCESS) {
            fprintf(stderr, "[bc250-gpu] Vulkan error %d at %s:%d\n", result, __FILE__, __LINE__);
            vkFreeMemory(ctx->device, memory->memory, NULL);
            vkDestroyImage(ctx->device, image->y_plane, NULL);
            vkDestroyImage(ctx->device, image->uv_plane, NULL);
            memory->memory = VK_NULL_HANDLE;
            image->y_plane = VK_NULL_HANDLE;
            image->uv_plane = VK_NULL_HANDLE;
            continue;
        }

        break; /* success */
    }
    if (result != VK_SUCCESS) {
        return -1;
    }

    VkImageViewCreateInfo view_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = image->y_plane,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = vk_y,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
    };
    VK_CHECK(vkCreateImageView(ctx->device, &view_info, NULL, &image->y_view));

    view_info.image = image->uv_plane;
    view_info.format = vk_uv;
    VK_CHECK(vkCreateImageView(ctx->device, &view_info, NULL, &image->uv_view));

    return 0;
}

void gpu_compute_destroy_image(gpu_context_t *ctx, gpu_image_t image, gpu_memory_t memory) {
    if (image.y_view) vkDestroyImageView(ctx->device, image.y_view, NULL);
    if (image.uv_view) vkDestroyImageView(ctx->device, image.uv_view, NULL);
    if (image.y_plane) vkDestroyImage(ctx->device, image.y_plane, NULL);
    if (image.uv_plane) vkDestroyImage(ctx->device, image.uv_plane, NULL);
    if (memory.memory) vkFreeMemory(ctx->device, memory.memory, NULL);
}

int gpu_compute_export_nv12_dmabuf(gpu_context_t *ctx, gpu_memory_t memory, int *out_fd) {
    if (!ctx || !ctx->get_memory_fd_khr || !memory.memory || !out_fd) return -1;

    VkMemoryGetFdInfoKHR get_fd_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
        .memory = memory.memory,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT
    };
    VkResult result = ctx->get_memory_fd_khr(ctx->device, &get_fd_info, out_fd);
    if (result != VK_SUCCESS) {
        fprintf(stderr, "[bc250-gpu] Vulkan error %d at %s:%d (vkGetMemoryFdKHR)\n", result, __FILE__, __LINE__);
        return -1;
    }
    return 0;
}

/* See gpu_compute.h for the full story on why this exists. Short version:
 * this driver's VA-API render-target surfaces are zero-copy shared with
 * Sunshine's own GL context via the dma-buf bc250_ExportSurfaceHandle()
 * exports, and this driver's Vulkan compute dispatch otherwise has no
 * explicit ordering relative to whatever last wrote into that surface
 * through that *other* API/context. This snapshots the dma-buf's current
 * fences as a sync_file (kernel, API-agnostic) and imports that as a
 * one-shot wait semaphore for the next gpu_compute_end_picture() call. */
/* DMA_BUF_IOCTL_EXPORT_SYNC_FILE (and its struct) are Linux ~6.0 UAPI. This
 * driver only ever runs on one machine, but it is built on others: CI builds
 * on ubuntu-22.04, whose linux/dma-buf.h predates both, and this function
 * broke that build after being verified only on the board's Fedora 43
 * headers. Compile it out where the header can't support it, leaving the
 * same "return -1 and let the caller proceed unsynchronized" behaviour every
 * other opportunistic capability check in this file already uses. */
#ifdef DMA_BUF_IOCTL_EXPORT_SYNC_FILE

int gpu_compute_wait_for_image_ready(gpu_context_t *ctx, gpu_memory_t memory) {
    if (!ctx || !ctx->have_external_semaphore_fd || !ctx->import_semaphore_fd_khr) return -1;

    int dmabuf_fd;
    if (gpu_compute_export_nv12_dmabuf(ctx, memory, &dmabuf_fd) != 0) return -1;

    struct dma_buf_export_sync_file sync_file_info = {
        .flags = DMA_BUF_SYNC_READ,
        .fd = -1
    };
    int ioctl_ret = ioctl(dmabuf_fd, DMA_BUF_IOCTL_EXPORT_SYNC_FILE, &sync_file_info);
    /* This fd is our own independent reference to the buffer object (a
     * fresh vkGetMemoryFdKHR call above) - the sync_file ioctl only reads
     * its attached fences, it doesn't consume or need to keep this fd
     * around afterward. */
    close(dmabuf_fd);
    if (ioctl_ret != 0) {
        if (getenv("BC250_DEBUG_DMABUF")) {
            fprintf(stderr, "[bc250-gpu] DMA_BUF_IOCTL_EXPORT_SYNC_FILE failed: %s\n", strerror(errno));
        }
        return -1;
    }

    VkImportSemaphoreFdInfoKHR import_info = {
        .sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR,
        .semaphore = ctx->image_ready_semaphore,
        .flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT,
        .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT,
        .fd = sync_file_info.fd
    };
    VkResult result = ctx->import_semaphore_fd_khr(ctx->device, &import_info);
    if (result != VK_SUCCESS) {
        if (getenv("BC250_DEBUG_DMABUF")) {
            fprintf(stderr, "[bc250-gpu] vkImportSemaphoreFdKHR failed: %d\n", result);
        }
        /* Import failed - the fd wasn't consumed, so it's still ours to close. */
        close(sync_file_info.fd);
        return -1;
    }
    /* vkImportSemaphoreFdKHR takes ownership of the fd on success (Vulkan
     * spec, "Importing Semaphore Payloads") - must not close it ourselves. */

    ctx->has_pending_wait_semaphore = true;
    return 0;
}

#else  /* !DMA_BUF_IOCTL_EXPORT_SYNC_FILE - pre-6.0 kernel headers */

int gpu_compute_wait_for_image_ready(gpu_context_t *ctx, gpu_memory_t memory) {
    (void)ctx; (void)memory;
    /* No sync_file export available at build time; callers treat -1 as "no
     * explicit wait was queued" and proceed, relying on the kernel's
     * implicit dma-buf fencing exactly as this driver did before the
     * explicit wait was added. */
    return -1;
}

#endif /* DMA_BUF_IOCTL_EXPORT_SYNC_FILE */

/* Test-harness instrumentation (tools/quality_test.sh): dump raw NV12 frame
 * bytes to disk when BC250_DUMP_INPUT_FRAMES=1 is set, building a
 * byte-exact ground-truth reference of what the driver actually received
 * from libva/ffmpeg for later PSNR/SSIM comparison against encoder output.
 * Called from both known upload paths — gpu_compute_upload_nv12()
 * (vaPutImage) and bc250_UnmapBuffer() in va_backend.c (the zero-copy
 * vaDeriveImage+vaMapBuffer path some ffmpeg versions use instead) — so
 * whichever path a given ffmpeg build takes, the frame gets captured.
 * Compiled in unconditionally but a no-op (single getenv check) unless the
 * env var is set, so it costs nothing in normal operation. */
/* Every opt-in diagnostic dump in the driver is created through here.
 *
 * ⚠️ They run inside whatever process loaded the driver - a browser, Steam,
 * a game - and write where they are told. So the file is created 0600, a
 * symbolic link at its name is refused rather than followed, and the
 * default directory is the user's own runtime directory rather than /tmp:
 * a shared directory that someone else created first can hold links
 * pointing at the user's files, and the dump would write through them.
 *
 * `name` is a bare file name. Returns NULL, and says why, when there is
 * nowhere safe to write. */
FILE *bc250_debug_dump_open(const char *name, const char *what)
{
    char dir[512];
    const char *chosen = getenv("BC250_DUMP_DIR");
    if (chosen && chosen[0]) {
        if (snprintf(dir, sizeof dir, "%s", chosen) >= (int)sizeof dir)
            return NULL;
    } else {
        const char *run = getenv("XDG_RUNTIME_DIR");
        if (!run || !run[0]) {
            fprintf(stderr, "[bc250] %s: no XDG_RUNTIME_DIR to put the dump "
                            "in - set BC250_DUMP_DIR\n", what);
            return NULL;
        }
        if (snprintf(dir, sizeof dir, "%s/bc250_dump_frames", run)
            >= (int)sizeof dir)
            return NULL;
        if (mkdir(dir, 0700) != 0 && errno != EEXIST) {
            fprintf(stderr, "[bc250] %s: cannot create %s: %s\n",
                    what, dir, strerror(errno));
            return NULL;
        }
    }
    if (!name || strchr(name, '/')) return NULL;

    char path[768];
    if (snprintf(path, sizeof path, "%s/%s", dir, name) >= (int)sizeof path)
        return NULL;
    const int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW
                              | O_CLOEXEC, 0600);
    if (fd < 0) {
        fprintf(stderr, "[bc250] %s: cannot create %s: %s\n",
                what, path, strerror(errno));
        return NULL;
    }
    FILE *f = fdopen(fd, "wb");
    if (!f) close(fd);
    return f;
}

void bc250_debug_dump_nv12_frame(const uint8_t *y_plane, int y_pitch,
                                  const uint8_t *uv_plane, int uv_pitch,
                                  int width, int height) {
    if (!getenv("BC250_DUMP_INPUT_FRAMES")) return;
    if (!y_plane || !uv_plane || width <= 0 || height <= 0) return;

    static int dump_frame_index = 0;
    char dump_name[64];
    snprintf(dump_name, sizeof(dump_name), "frame_%05d.nv12", dump_frame_index);
    FILE *dumpf = bc250_debug_dump_open(dump_name, "BC250_DUMP_INPUT_FRAMES");
    if (dumpf) {
        for (int r = 0; r < height; r++) {
            fwrite(y_plane + (size_t)r * y_pitch, 1, (size_t)width, dumpf);
        }
        for (int r = 0; r < height / 2; r++) {
            fwrite(uv_plane + (size_t)r * uv_pitch, 1, (size_t)width, dumpf);
        }
        fclose(dumpf);
    }
    dump_frame_index++;
}

int gpu_compute_get_nv12_layout(gpu_context_t *ctx, gpu_image_t *image, gpu_memory_t memory, gpu_nv12_layout_t *layout) {
    if (!ctx || !image || !memory.memory || !layout) return -1;

    /* Same math gpu_compute_upload_nv12()/gpu_compute_download_nv12() use
     * to address this image's real memory: per-plane row pitch + offset via
     * vkGetImageSubresourceLayout(), and the real inter-plane bind offset
     * via vkGetImageMemoryRequirements() + alignment (must match the bind
     * performed in gpu_compute_create_image() exactly, since that's the
     * memory layout actually being described). */
    VkImageSubresource subresource_y = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0 };
    VkSubresourceLayout layout_y;
    vkGetImageSubresourceLayout(ctx->device, image->y_plane, &subresource_y, &layout_y);

    VkMemoryRequirements y_req, uv_req;
    vkGetImageMemoryRequirements(ctx->device, image->y_plane, &y_req);
    vkGetImageMemoryRequirements(ctx->device, image->uv_plane, &uv_req);
    VkDeviceSize align = uv_req.alignment > y_req.alignment ? uv_req.alignment : y_req.alignment;
    VkDeviceSize uv_bind_offset = (y_req.size + align - 1) & ~(align - 1);

    VkImageSubresource subresource_uv = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0 };
    VkSubresourceLayout layout_uv;
    vkGetImageSubresourceLayout(ctx->device, image->uv_plane, &subresource_uv, &layout_uv);

    layout->y_pitch = (uint32_t)layout_y.rowPitch;
    layout->y_offset = (uint64_t)layout_y.offset;
    layout->uv_pitch = (uint32_t)layout_uv.rowPitch;
    layout->uv_offset = (uint64_t)(uv_bind_offset + layout_uv.offset);
    layout->total_size = (uint64_t)memory.size;
    return 0;
}

uint8_t *gpu_compute_map_surface(gpu_context_t *ctx, gpu_image_t *image,
                                 gpu_memory_t memory,
                                 gpu_nv12_layout_t *layout, bool *unmap)
{
    if (!unmap) return NULL;
    *unmap = false;
    if (gpu_compute_get_nv12_layout(ctx, image, memory, layout) != 0)
        return NULL;
    if (memory.mapped_ptr) return (uint8_t *)memory.mapped_ptr;
    void *p = NULL;
    if (vkMapMemory(ctx->device, memory.memory, 0, memory.size, 0, &p)
        != VK_SUCCESS)
        return NULL;
    *unmap = true;
    return (uint8_t *)p;
}

void gpu_compute_unmap_surface(gpu_context_t *ctx, gpu_memory_t memory,
                               bool unmap)
{
    if (unmap && ctx) vkUnmapMemory(ctx->device, memory.memory);
}

/* The ten-bit upload. See gpu_compute.h: the shift into P010's high bits
 * is here, and the plane addressing is whatever Vulkan says it is, asked
 * for through the same query the eight-bit path uses. */
int gpu_compute_upload_p010(gpu_context_t *ctx, gpu_image_t *image,
                            gpu_memory_t memory,
                            const uint16_t *y_plane, int y_pitch,
                            const uint16_t *uv_plane, int uv_pitch,
                            int width, int height) {
    if (!ctx || !image || !memory.memory || !y_plane || !uv_plane) return -1;

    gpu_nv12_layout_t lay;
    if (gpu_compute_get_nv12_layout(ctx, image, memory, &lay) != 0) return -1;

    uint8_t *mapped = (uint8_t *)memory.mapped_ptr;
    int needs_unmap = 0;
    if (!mapped) {
        if (vkMapMemory(ctx->device, memory.memory, 0, memory.size, 0,
                        (void **)&mapped) != VK_SUCCESS) {
            return -1;
        }
        needs_unmap = 1;
    }

    for (int r = 0; r < height; r++) {
        uint16_t *o = (uint16_t *)(mapped + lay.y_offset
                                   + (size_t)r * lay.y_pitch);
        const uint16_t *s = (const uint16_t *)((const uint8_t *)y_plane
                                               + (size_t)r * y_pitch);
        for (int x = 0; x < width; x++) o[x] = (uint16_t)(s[x] << 6);
    }

    /* One row of interleaved Cb/Cr is two samples per luma column. */
    for (int r = 0; r < height / 2; r++) {
        uint16_t *o = (uint16_t *)(mapped + lay.uv_offset
                                   + (size_t)r * lay.uv_pitch);
        const uint16_t *s = (const uint16_t *)((const uint8_t *)uv_plane
                                               + (size_t)r * uv_pitch);
        for (int x = 0; x < width; x++) o[x] = (uint16_t)(s[x] << 6);
    }

    if (needs_unmap) {
        vkUnmapMemory(ctx->device, memory.memory);
    }
    return 0;
}

int gpu_compute_upload_nv12(gpu_context_t *ctx, gpu_image_t *image, gpu_memory_t memory,
                           const uint8_t *y_plane, int y_pitch,
                           const uint8_t *uv_plane, int uv_pitch,
                           int width, int height) {
    if (!ctx || !image || !memory.memory || !y_plane || !uv_plane) return -1;

    /* Test-harness instrumentation (tools/quality_test.sh): capture the
     * exact raw NV12 bytes libva handed us via the vaPutImage upload path,
     * before any GPU work touches them. See bc250_debug_dump_nv12_frame()
     * for the other upload path (zero-copy vaDeriveImage+vaMapBuffer) this
     * doesn't cover. */
    bc250_debug_dump_nv12_frame(y_plane, y_pitch, uv_plane, uv_pitch, width, height);

    VkImageSubresource subresource_y = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0 };
    VkSubresourceLayout layout_y;
    vkGetImageSubresourceLayout(ctx->device, image->y_plane, &subresource_y, &layout_y);

    VkMemoryRequirements y_req, uv_req;
    vkGetImageMemoryRequirements(ctx->device, image->y_plane, &y_req);
    vkGetImageMemoryRequirements(ctx->device, image->uv_plane, &uv_req);
    VkDeviceSize align = uv_req.alignment > y_req.alignment ? uv_req.alignment : y_req.alignment;
    VkDeviceSize uv_offset = (y_req.size + align - 1) & ~(align - 1);

    VkImageSubresource subresource_uv = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0 };
    VkSubresourceLayout layout_uv;
    vkGetImageSubresourceLayout(ctx->device, image->uv_plane, &subresource_uv, &layout_uv);

    uint8_t *mapped = (uint8_t *)memory.mapped_ptr;
    int needs_unmap = 0;
    if (!mapped) {
        if (vkMapMemory(ctx->device, memory.memory, 0, memory.size, 0, (void **)&mapped) != VK_SUCCESS) {
            return -1;
        }
        needs_unmap = 1;
    }

    /* See the download: a row is `width` samples, not `width` bytes. */
    const size_t row = (size_t)width * (image->format == GPU_IMAGE_P010 ? 2 : 1);

    uint8_t *dst_y = mapped + layout_y.offset;
    for (int r = 0; r < height; r++) {
        memcpy(dst_y + (size_t)r * layout_y.rowPitch, y_plane + (size_t)r * y_pitch, row);
    }

    uint8_t *dst_uv = mapped + uv_offset + layout_uv.offset;
    for (int r = 0; r < height / 2; r++) {
        memcpy(dst_uv + (size_t)r * layout_uv.rowPitch, uv_plane + (size_t)r * uv_pitch, row);
    }

    if (needs_unmap) {
        vkUnmapMemory(ctx->device, memory.memory);
    }
    return 0;
}

/* Reading a mapped surface back is a read from WRITE-COMBINING memory.
 *
 * The frame the HEVC encoder works on comes out of the VA surface through
 * gpu_compute_download_nv12(), which used to memcpy row by row. WC memory has
 * no cache line to fill, so an ordinary load fetches a few bytes at a time and
 * the copy crawls: measured on a BC-250, 230 MB/s, which came to 40% of the
 * entire HEVC encode - more than the transform, the quantiser, the intra
 * prediction and CABAC put together.
 *
 * MOVNTDQA is the instruction for this case: it reads a full line into a fill
 * buffer and hands it over in one go. On memory that IS cached it behaves like
 * an ordinary load, so this is safe whichever way the driver ends up mapping
 * the surface, and it does not need to be told which one happened.
 *
 * Dispatched at runtime like cpu_simd_me.c does, so a build that runs
 * somewhere without AVX2 still works.
 */
#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("avx2")))
static void copy_from_wc_avx2(uint8_t *dst, const uint8_t *src, size_t n) {
    size_t i = 0;
    /* MOVNTDQA needs a 32-byte aligned source: walk up to it normally. */
    size_t head = (size_t)((0u - (uintptr_t)src) & 31u);
    if (head > n) head = n;
    if (head) { memcpy(dst, src, head); i = head; }
    for (; i + 128 <= n; i += 128) {
        __m256i a = _mm256_stream_load_si256((const __m256i *)(src + i));
        __m256i b = _mm256_stream_load_si256((const __m256i *)(src + i + 32));
        __m256i c = _mm256_stream_load_si256((const __m256i *)(src + i + 64));
        __m256i d = _mm256_stream_load_si256((const __m256i *)(src + i + 96));
        _mm256_storeu_si256((__m256i *)(dst + i), a);
        _mm256_storeu_si256((__m256i *)(dst + i + 32), b);
        _mm256_storeu_si256((__m256i *)(dst + i + 64), c);
        _mm256_storeu_si256((__m256i *)(dst + i + 96), d);
    }
    for (; i + 32 <= n; i += 32) {
        _mm256_storeu_si256((__m256i *)(dst + i),
                            _mm256_stream_load_si256((const __m256i *)(src + i)));
    }
    if (i < n) memcpy(dst + i, src + i, n - i);
    _mm_sfence();
}
#endif

static void copy_from_wc(uint8_t *dst, const uint8_t *src, size_t n) {
#if defined(__x86_64__) || defined(_M_X64)
    static int ha_avx2 = -1;
    if (ha_avx2 < 0) ha_avx2 = __builtin_cpu_supports("avx2") ? 1 : 0;
    if (ha_avx2) { copy_from_wc_avx2(dst, src, n); return; }
#endif
    memcpy(dst, src, n);
}

/* Public wrapper: the H.264 path's shadow_copy() reads the same kind of
 * write-combining staging memory and was paying the same price. */
void gpu_compute_copy_from_wc(void *dst, const void *src, size_t n) {
    copy_from_wc((uint8_t *)dst, (const uint8_t *)src, n);
}

int gpu_compute_download_nv12(gpu_context_t *ctx, gpu_image_t *image, gpu_memory_t memory,
                             uint8_t *y_plane, int y_pitch,
                             uint8_t *uv_plane, int uv_pitch,
                             int width, int height) {
    if (!ctx || !image || !memory.memory || !y_plane || !uv_plane) return -1;

    VkImageSubresource subresource_y = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0 };
    VkSubresourceLayout layout_y;
    vkGetImageSubresourceLayout(ctx->device, image->y_plane, &subresource_y, &layout_y);

    VkMemoryRequirements y_req, uv_req;
    vkGetImageMemoryRequirements(ctx->device, image->y_plane, &y_req);
    vkGetImageMemoryRequirements(ctx->device, image->uv_plane, &uv_req);
    VkDeviceSize align = uv_req.alignment > y_req.alignment ? uv_req.alignment : y_req.alignment;
    VkDeviceSize uv_offset = (y_req.size + align - 1) & ~(align - 1);

    VkImageSubresource subresource_uv = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0 };
    VkSubresourceLayout layout_uv;
    vkGetImageSubresourceLayout(ctx->device, image->uv_plane, &subresource_uv, &layout_uv);

    uint8_t *mapped = (uint8_t *)memory.mapped_ptr;
    int needs_unmap = 0;
    if (!mapped) {
        if (vkMapMemory(ctx->device, memory.memory, 0, memory.size, 0, (void **)&mapped) != VK_SUCCESS) {
            return -1;
        }
        needs_unmap = 1;
    }

    /* ⚠️ A row is `width` SAMPLES, and a P010 sample is two bytes. Both
      * planes carry the same number of bytes per row here, because the
      * chroma one is half as wide and twice as deep. */
    const size_t row = (size_t)width * (image->format == GPU_IMAGE_P010 ? 2 : 1);

    const uint8_t *src_y = mapped + layout_y.offset;
    for (int r = 0; r < height; r++) {
        copy_from_wc(y_plane + (size_t)r * y_pitch, src_y + (size_t)r * layout_y.rowPitch, row);
    }

    const uint8_t *src_uv = mapped + uv_offset + layout_uv.offset;
    for (int r = 0; r < height / 2; r++) {
        copy_from_wc(uv_plane + (size_t)r * uv_pitch, src_uv + (size_t)r * layout_uv.rowPitch, row);
    }

    if (needs_unmap) {
        vkUnmapMemory(ctx->device, memory.memory);
    }
    return 0;
}

static void transition_image_layout(VkCommandBuffer cmd_buf, VkImage image,
                                    VkImageLayout old_layout, VkImageLayout new_layout) {
    if (!image) return;
    VkImageMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
        }
    };
    VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    if (old_layout == VK_IMAGE_LAYOUT_PREINITIALIZED || old_layout == VK_IMAGE_LAYOUT_UNDEFINED) {
        barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    } else {
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    }
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(cmd_buf, src_stage, dst_stage, 0, 0, NULL, 0, NULL, 1, &barrier);
}

static void insert_compute_barrier(VkCommandBuffer cmd_buf) {
    VkMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT
    };
    vkCmdPipelineBarrier(cmd_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, NULL, 0, NULL);
}

/* Diagnostic-only (BC250_PERF_STATS=1): millisecond delta between two
 * CLOCK_MONOTONIC timespecs. Used below to isolate vkQueueSubmit() and
 * vkWaitForFences() as their own real wall-clock brackets - see the
 * "[BC250_PERF_SUBMIT]"/"[BC250_PERF_WAIT]" lines this enables. This exists
 * because the pre-existing GPU-timestamp-query instrumentation
 * (BC250_PERF_NUM_TIMESTAMPS / "[BC250_PERF_GPU]") only measures GPU
 * *execution* time between the first and last command in a submitted
 * command buffer; it cannot see time spent before the GPU starts executing
 * that command buffer at all (scheduling/dispatch latency between
 * vkQueueSubmit returning and the GPU actually beginning the work), which
 * is exactly the gap this diagnostic was added to find. */
static double bc250_diag_delta_ms(const struct timespec *t0, const struct timespec *t1) {
    return (double)(t1->tv_sec - t0->tv_sec) * 1000.0 +
           (double)(t1->tv_nsec - t0->tv_nsec) / 1e6;
}

int gpu_compute_begin_picture(gpu_context_t *ctx, gpu_image_t render_target) {
    (void)render_target;
    struct timespec w0, w1;
    if (ctx->perf_stats_enabled) clock_gettime(CLOCK_MONOTONIC, &w0);
    /* Unchecked until now - same "assume success" gap already fixed for
     * vkQueueSubmit() in gpu_compute_end_picture() (see that function's doc
     * comment), just on the wait side instead of the submit side. A
     * UINT64_MAX timeout can't return VK_TIMEOUT, but real GPU contention
     * (a concurrently active compositor/cursor-plane update sharing this
     * same hardware queue) can make the underlying driver return
     * VK_ERROR_DEVICE_LOST here without the fence's GPU work having
     * actually finished. Falling through in that case would vkResetFences()
     * and immediately start recording new commands into cmd_bufs[current_buf]
     * while the GPU might still be executing the previous submission into
     * it - undefined behavior that reads exactly like the corruption this
     * was chasing (garbage/partial data landing in scattered blocks of the
     * frame). Bail out instead and let the caller treat this the same as
     * an end_picture() submit failure: no new GPU work this frame. */
    VkResult wait_result = vkWaitForFences(ctx->device, 1, &ctx->fences[ctx->current_buf], VK_TRUE, UINT64_MAX);
    if (ctx->perf_stats_enabled) {
        clock_gettime(CLOCK_MONOTONIC, &w1);
        fprintf(stderr, "[BC250_PERF_WAIT] site=begin_picture buf=%d wait_ms=%.3f\n",
                ctx->current_buf, bc250_diag_delta_ms(&w0, &w1));
    }
    if (wait_result != VK_SUCCESS) {
        fprintf(stderr, "[bc250-gpu] vkWaitForFences failed in begin_picture: %d\n", wait_result);
        return -1;
    }
    vkResetFences(ctx->device, 1, &ctx->fences[ctx->current_buf]);

    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    vkBeginCommandBuffer(ctx->cmd_bufs[ctx->current_buf], &begin_info);

    return 0;
}

int gpu_compute_dispatch_encode_ext(gpu_context_t *ctx, gpu_image_t render_target, int width, int height, int qp, int is_intra, int num_slices, int me_mode, const gpu_mv_t *cpu_mvs) {
    if (!ctx) return -1;
    if (num_slices < 1) num_slices = 1;

    /* Ensure pipeline buffers are allocated for current dimensions. This is
     * now the ONLY place they get allocated (bc250_gpu_init() no longer
     * pre-allocates for 4K - see the comment there), and its failure is
     * checked: proceeding with VK_NULL_HANDLE buffers is what turned an
     * out-of-memory into a SEGV. */
    if (ctx->staging_buffers[0] == VK_NULL_HANDLE || ctx->frame_width != (uint32_t)width || ctx->frame_height != (uint32_t)height) {
        if (allocate_encoding_buffers(ctx, (uint32_t)width, (uint32_t)height) != 0) {
            return -1;
        }
    }

    /* Ensure reconstructed frame buffer is allocated for DPB / reference */
    if (ctx->recon_image.y_plane == VK_NULL_HANDLE || ctx->recon_image.width != (uint32_t)width || ctx->recon_image.height != (uint32_t)height) {
        if (ctx->recon_image.y_plane != VK_NULL_HANDLE) {
            gpu_compute_destroy_image(ctx, ctx->recon_image, ctx->recon_memory);
        }
        gpu_compute_create_image(ctx, width, height, 0, &ctx->recon_image, &ctx->recon_memory);
        ctx->has_recon_frame = false;
    }

    VkCommandBuffer cmd_buf = ctx->cmd_bufs[ctx->current_buf];

    /* Opt-in GPU per-stage timing (BC250_PERF_STATS=1) - see the
     * BC250_PERF_NUM_TIMESTAMPS comment near the top of this file.
     * perf_buf mirrors ctx->current_buf at the moment this frame's commands
     * are being recorded into it (gpu_compute_end_picture() toggles
     * ctx->current_buf AFTER submission, so this is stable for the whole
     * function); is_intra is recorded now since gpu_compute_sync() reads it
     * back later without visibility into this call's parameters. */
    int perf_buf = ctx->current_buf;
    if (ctx->perf_stats_enabled) {
        ctx->perf_is_intra[perf_buf] = is_intra ? true : false;
        ctx->perf_result_pending[perf_buf] = true;
        vkCmdResetQueryPool(cmd_buf, ctx->timestamp_pools[perf_buf], 0, BC250_PERF_NUM_TIMESTAMPS);
        vkCmdWriteTimestamp(cmd_buf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ctx->timestamp_pools[perf_buf], 0);
    }

    uint32_t width_mbs = (width + 15) / 16;
    uint32_t height_mbs = (height + 15) / 16;
    /* qp/is_intra used to be hardcoded (26, 0) here regardless of the actual
     * per-frame rate-control QP or slice type, which silently defeated rate
     * control (the AC quantizer always ran at QP 26, no matter what QP the
     * slice header declared) and always used the inter (/6) quantizer
     * rounding offset even on IDR/I-slices. Now threaded through from the
     * caller so the shader's quantize.comp actually uses the QP/frame-type
     * the encoder decided on - this also keeps the CPU-side luma/chroma DC
     * Hadamard quantization (encoder_h264.c) numerically consistent with
     * what quantize.comp did for the AC coefficients of the same frame. */
    uint32_t pc[9] = { (uint32_t)width, (uint32_t)height, width_mbs, height_mbs,
                       (uint32_t)qp, (uint32_t)(is_intra ? 1 : 0), 0, 5, (uint32_t)num_slices };

    /* Transition image layout to GENERAL for compute storage access.
     *
     * render_target is a copy of the caller's persistent gpu_image_t (e.g.
     * bc250_surface.image in va_backend.c), so render_target.current_layout
     * reflects the image's real layout at the start of this call: PREINITIALIZED
     * on the very first dispatch for this surface (VA-API uploads pixels via
     * host-mapped memory before the GPU touches them), or GENERAL on every
     * dispatch after that, since nothing ever transitions the image back out of
     * GENERAL. Using the tracked real layout here (instead of hardcoding
     * VK_IMAGE_LAYOUT_UNDEFINED) is required by the Vulkan spec: claiming
     * UNDEFINED when the real layout is GENERAL permits the implementation to
     * discard the image's prior contents. Updating render_target.current_layout
     * below only affects this local copy; the caller is responsible for
     * persisting VK_IMAGE_LAYOUT_GENERAL back onto its own stored gpu_image_t
     * (see va_backend.c's bc250_EndPicture) so the next dispatch call passes in
     * the correct real layout. */
    if (render_target.y_plane) {
        transition_image_layout(cmd_buf, render_target.y_plane, render_target.current_layout, VK_IMAGE_LAYOUT_GENERAL);
    }
    if (render_target.uv_plane) {
        transition_image_layout(cmd_buf, render_target.uv_plane, render_target.current_layout, VK_IMAGE_LAYOUT_GENERAL);
    }
    render_target.current_layout = VK_IMAGE_LAYOUT_GENERAL;

    /* Reference image for ME: use recon frame if available, otherwise self */
    VkImageView ref_view = render_target.y_view;
    if (ctx->has_recon_frame && ctx->recon_image.y_view != VK_NULL_HANDLE) {
        ref_view = ctx->recon_image.y_view;
    }
    /* Same idea for chroma - see residual_predict.comp's referenceUV /
     * P-slice chroma motion compensation. */
    VkImageView ref_uv_view = render_target.uv_view;
    if (ctx->has_recon_frame && ctx->recon_image.uv_view != VK_NULL_HANDLE) {
        ref_uv_view = ctx->recon_image.uv_view;
    }

    /* Update image descriptors */
    if (render_target.y_view && render_target.uv_view) {
        update_storage_image_descriptor(ctx->device, ctx->me_desc_set, 0, render_target.y_view);
        update_storage_image_descriptor(ctx->device, ctx->me_desc_set, 1, ref_view);
    }
    /*
     * BUG FIX (reconciled from fix/gradient-boundary-mc-v2's independent
     * finding): deblock_desc_set binding 0 (deblock_filter.comp's
     * `frameImage`) was bound to render_target.y_view - the CURRENT INPUT
     * SURFACE's own pixels, already fully consumed by ME/residual
     * generation earlier in this same dispatch and about to be discarded -
     * instead of ctx->recon_image.y_view, the actual reconstruction buffer
     * Stage 4.5 (reconstruct.comp) / the intra-wavefront loop populates a
     * few lines below and that becomes next frame's ME reference (ref_view
     * above) and what BC250_DUMP_RECON_FRAMES reads back. Net effect: the
     * real ITU-T deblocking filter (see deblock_filter.comp's top-of-file
     * comment) ran on a dead buffer nothing downstream ever reads -
     * provably inert, since recon_image is fully finalized by Stage 4.5/the
     * wavefront loop before Stage 5 (below) even dispatches, and nothing
     * after Stage 5 copies its output anywhere. A real H.264 decoder DOES
     * deblock its reference every frame (disable_deblocking_filter_idc=0
     * here by default), so the encoder's own assumed reference silently
     * diverged from the decoder's actual one from the second frame of every
     * GOP onward - invisible on near-zero-residual (flat/static) content,
     * real and compounding wherever genuine per-frame residual energy
     * exists. frameImage (binding 0) is a plain read-write image2D with no
     * semantic dependency on which buffer backs it, so retargeting is a
     * drop-in change - the real bS/alpha-beta/tc0 algorithm itself is
     * unchanged, it just now actually reaches the reference chain.
     * ctx->recon_image.y_view is a stable handle allocated once at context
     * creation (only its CONTENTS become valid at Stage 4.5/the wavefront
     * loop below), so binding it here, before either has run this frame,
     * is safe - Vulkan descriptor updates only need a valid image view
     * handle, not populated contents, at update time. */
    if (ctx->recon_image.y_view != VK_NULL_HANDLE) {
        update_storage_image_descriptor(ctx->device, ctx->deblock_desc_set, 0, ctx->recon_image.y_view);
    }

    /* Transition recon_image to GENERAL up front, before reconstruct.comp's
     * imageStore writes into it below (Stage 4.5). Like render_target, it
     * starts VK_IMAGE_LAYOUT_PREINITIALIZED (see gpu_compute_create_image())
     * and is never host-written afterwards - only ever written by
     * reconstruct.comp's compute-shader stores, so its tracked layout is
     * updated here using the same real-old-layout pattern render_target
     * uses above. ctx owns recon_image directly (not a by-value copy), so
     * this persists correctly across dispatches. */
    if (ctx->recon_image.y_plane) {
        transition_image_layout(cmd_buf, ctx->recon_image.y_plane, ctx->recon_image.current_layout, VK_IMAGE_LAYOUT_GENERAL);
    }
    if (ctx->recon_image.uv_plane) {
        transition_image_layout(cmd_buf, ctx->recon_image.uv_plane, ctx->recon_image.current_layout, VK_IMAGE_LAYOUT_GENERAL);
    }
    ctx->recon_image.current_layout = VK_IMAGE_LAYOUT_GENERAL;

    /* Stage 1: Color Convert (Skipped: inputs in VA-API are already NV12) */

    /* Stage 2: Motion Estimation */
    if (cpu_mvs != NULL) {
        /* Dynamic Governor Tier 2: CPU SIMD Motion Estimation offload.
         * Copy CPU-computed motion vectors into mv_staging_buffers and transfer
         * to device-local mv_buffer, skipping the expensive GPU compute shader. */
        size_t mv_bytes = (size_t)width_mbs * height_mbs * sizeof(gpu_mv_t);
        if (ctx->mv_staging_mapped[perf_buf]) {
            memcpy(ctx->mv_staging_mapped[perf_buf], cpu_mvs, mv_bytes);
            VkBufferCopy mv_copy = {
                .srcOffset = 0,
                .dstOffset = 0,
                .size = mv_bytes
            };
            vkCmdCopyBuffer(cmd_buf, ctx->mv_staging_buffers[perf_buf], ctx->mv_buffer, 1, &mv_copy);
            insert_compute_barrier(cmd_buf);
        }
    } else if (ctx->motion_est_pipeline) {
        if (me_mode == 1) {
            /* Tier 1: Fast ME mode - higher lambda and fast diamond */
            pc[6] = 1;
            pc[7] = 8;
        }
        vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->motion_est_pipeline);
        vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->motion_est_layout, 0, 1, &ctx->me_desc_set, 0, NULL);
        vkCmdPushConstants(cmd_buf, ctx->motion_est_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
        vkCmdDispatch(cmd_buf, width_mbs, height_mbs, 1);
        insert_compute_barrier(cmd_buf);
    }
    if (ctx->perf_stats_enabled) {
        vkCmdWriteTimestamp(cmd_buf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx->timestamp_pools[perf_buf], 1);
    }

    if (!is_intra) {
        /* P-slice path - UNCHANGED (whole-frame-parallel). See
         * residual_predict.comp's top-of-file comment: P-slices are always
         * inter-coded in this encoder and only depend on the PREVIOUS frame
         * (already fully reconstructed via recon_image by the time this
         * frame starts), so there is no same-frame macroblock-ordering
         * problem here - only I-slices (the `else` branch below) need
         * diagonal-wavefront dispatch. */

        /* Stage 2.5: Real intra/inter prediction + residual generation (see
         * residual_predict.comp) - consumes the real MVs Stage 2 just wrote,
         * for P-slice motion-compensated residual. */
        if (ctx->predict_pipeline && render_target.y_view && render_target.uv_view) {
            update_storage_image_descriptor(ctx->device, ctx->predict_desc_set, 0, render_target.y_view);
            update_storage_image_descriptor(ctx->device, ctx->predict_desc_set, 1, render_target.uv_view);
            update_storage_image_descriptor(ctx->device, ctx->predict_desc_set, 2, ref_view);
            update_storage_image_descriptor(ctx->device, ctx->predict_desc_set, 6, ref_uv_view);

            vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->predict_pipeline);
            vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->predict_layout, 0, 1, &ctx->predict_desc_set, 0, NULL);
            vkCmdPushConstants(cmd_buf, ctx->predict_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
            vkCmdDispatch(cmd_buf, width_mbs, height_mbs, 1);
            insert_compute_barrier(cmd_buf);
        }
        if (ctx->perf_stats_enabled) {
            vkCmdWriteTimestamp(cmd_buf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx->timestamp_pools[perf_buf], 2);
        }

        /* Stage 3: DCT */
        if (ctx->transform_pipeline) {
            vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->transform_pipeline);
            vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->transform_layout, 0, 1, &ctx->dct_desc_set, 0, NULL);
            vkCmdPushConstants(cmd_buf, ctx->transform_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
            vkCmdDispatch(cmd_buf, width_mbs, height_mbs, 1);
            insert_compute_barrier(cmd_buf);
        }
        if (ctx->perf_stats_enabled) {
            vkCmdWriteTimestamp(cmd_buf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx->timestamp_pools[perf_buf], 3);
        }

        /* Stage 4: Quantize */
        if (ctx->quantize_pipeline) {
            vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->quantize_pipeline);
            vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->quantize_layout, 0, 1, &ctx->quant_desc_set, 0, NULL);
            vkCmdPushConstants(cmd_buf, ctx->quantize_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
            vkCmdDispatch(cmd_buf, width_mbs, height_mbs, 1);
            insert_compute_barrier(cmd_buf);
        }
        if (ctx->perf_stats_enabled) {
            vkCmdWriteTimestamp(cmd_buf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx->timestamp_pools[perf_buf], 4);
        }

        /* Stage 4.5: Reconstruct (see reconstruct.comp's top-of-file comment) -
         * dequantizes+inverse-transforms this frame's real quantized residual
         * (quant_levels_buffer, plus coeff_buffer for the chroma DC Hadamard)
         * and adds it back to the retained prediction (pred_buffer, written
         * by Stage 2.5 above), writing the clipped result directly into
         * ctx->recon_image - this REPLACES the old raw vkCmdCopyImage-from-source
         * population of recon_image, so the NEXT frame's P-slice inter
         * prediction (referenceImage/referenceUV, set up via ref_view/ref_uv_view
         * above) sees real reconstructed pixels instead of source pixels. Must
         * run after Stage 4 (quantize) and Stage 2.5 (predict, for pred_buffer);
         * ordering relative to deblock/entropy below doesn't matter since it
         * only needs quantized coefficients + retained prediction. */
        if (ctx->reconstruct_pipeline && ctx->recon_image.y_view != VK_NULL_HANDLE && ctx->recon_image.uv_view != VK_NULL_HANDLE) {
            update_storage_image_descriptor(ctx->device, ctx->reconstruct_desc_set, 3, ctx->recon_image.y_view);
            update_storage_image_descriptor(ctx->device, ctx->reconstruct_desc_set, 4, ctx->recon_image.uv_view);

            vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->reconstruct_pipeline);
            vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->reconstruct_layout, 0, 1, &ctx->reconstruct_desc_set, 0, NULL);
            vkCmdPushConstants(cmd_buf, ctx->reconstruct_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
            vkCmdDispatch(cmd_buf, width_mbs, height_mbs, 1);
            insert_compute_barrier(cmd_buf);
            ctx->has_recon_frame = true;
        }
        if (ctx->perf_stats_enabled) {
            /* Slot 6 (wavefront) is I-only; write it here as a zero-duration
             * duplicate of slot 5 so the downstream delta is well-defined on
             * P frames (see the BC250_PERF_NUM_TIMESTAMPS comment). */
            vkCmdWriteTimestamp(cmd_buf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx->timestamp_pools[perf_buf], 5);
            vkCmdWriteTimestamp(cmd_buf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx->timestamp_pools[perf_buf], 6);
        }
    } else {
        /* I-slice path - diagonal-wavefront intra reconstruction (see
         * intra_wavefront.comp's top-of-file comment for the full
         * rationale). REPLACES, for I-slices only, the whole-frame-parallel
         * predict->dct->quantize->reconstruct chain above: real I16x16 intra
         * prediction has a genuine same-frame spatial dependency (macroblock
         * (mbx,mby) needs macroblocks (mbx-1,mby)/(mbx,mby-1) to be truly
         * reconstructed FIRST), which a single whole-frame-parallel dispatch
         * cannot provide. Dispatched one anti-diagonal (d = mbx+mby) at a
         * time, with an explicit compute memory barrier between diagonals,
         * so every macroblock on diagonal d can safely read diagonal d-1's
         * (and earlier's) already-reconstructed neighbor pixels out of
         * ctx->recon_image (bound as intra_wavefront_desc_set's reconY/
         * reconUV, read-write - see gpu_compute.h's comment on that
         * descriptor set for why reusing recon_image here is safe). */
        if (ctx->perf_stats_enabled) {
            /* Slots 2-5 (predict/dct/quant/reconstruct) are P-only; write
             * them here as zero-duration duplicates of slot 1 so the
             * downstream delta is well-defined on I frames (see the
             * BC250_PERF_NUM_TIMESTAMPS comment). Slot 6 (wavefront) is
             * written below, after the diagonal loop completes. */
            vkCmdWriteTimestamp(cmd_buf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx->timestamp_pools[perf_buf], 2);
            vkCmdWriteTimestamp(cmd_buf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx->timestamp_pools[perf_buf], 3);
            vkCmdWriteTimestamp(cmd_buf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx->timestamp_pools[perf_buf], 4);
            vkCmdWriteTimestamp(cmd_buf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx->timestamp_pools[perf_buf], 5);
        }
        if (ctx->intra_wavefront_pipeline && render_target.y_view && render_target.uv_view &&
            ctx->recon_image.y_view != VK_NULL_HANDLE && ctx->recon_image.uv_view != VK_NULL_HANDLE) {
            update_storage_image_descriptor(ctx->device, ctx->intra_wavefront_desc_set, 0, render_target.y_view);
            update_storage_image_descriptor(ctx->device, ctx->intra_wavefront_desc_set, 1, render_target.uv_view);
            update_storage_image_descriptor(ctx->device, ctx->intra_wavefront_desc_set, 2, ctx->recon_image.y_view);
            update_storage_image_descriptor(ctx->device, ctx->intra_wavefront_desc_set, 3, ctx->recon_image.uv_view);

            vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->intra_wavefront_pipeline);
            vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->intra_wavefront_layout, 0, 1, &ctx->intra_wavefront_desc_set, 0, NULL);

            /* One dispatch per anti-diagonal d in [0, width_mbs+height_mbs-2].
             * Diagonal d's macroblocks are exactly those (mbx,mby) with
             * mbx+mby==d, 0<=mbx<width_mbs, 0<=mby<height_mbs - i.e.
             * mbx in [mbx_start, mbx_end] below. count = mbx_end-mbx_start+1
             * equals min(d+1, width_mbs, height_mbs, width_mbs+height_mbs-1-d),
             * the real number of macroblocks on that diagonal - no more, no
             * fewer. intra_wavefront.comp independently recomputes the same
             * mbx_start (and its own workgroup's mbx/mby) from pcs.diagonal +
             * pcs.width_in_mbs/height_in_mbs - see its DISPATCH SHAPE comment -
             * so nothing else needs to be threaded through push constants
             * beyond the diagonal index itself. */
            uint32_t num_diagonals = width_mbs + height_mbs - 1;
            for (uint32_t d = 0; d < num_diagonals; d++) {
                uint32_t mbx_start = (d + 1 > height_mbs) ? (d + 1 - height_mbs) : 0;
                uint32_t mbx_end = (d < width_mbs) ? d : (width_mbs - 1);
                uint32_t count = mbx_end - mbx_start + 1;

                uint32_t pcw[10] = { (uint32_t)width, (uint32_t)height, width_mbs, height_mbs,
                                      (uint32_t)qp, 1u, 0, 5, (uint32_t)num_slices, d };
                vkCmdPushConstants(cmd_buf, ctx->intra_wavefront_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcw), pcw);
                vkCmdDispatch(cmd_buf, count, 1, 1);
                insert_compute_barrier(cmd_buf);
            }
            ctx->has_recon_frame = true;
        }
        if (ctx->perf_stats_enabled) {
            vkCmdWriteTimestamp(cmd_buf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx->timestamp_pools[perf_buf], 6);
        }
    }

    /* Stage 5: Deblock (Skipped in BC250_FAST_MODE to maximize gaming framerates) */
    const char *fm = getenv("BC250_FAST_MODE");
    int fast_mode = (fm && (strcmp(fm, "1") == 0 || strcmp(fm, "true") == 0)) ? 1 : 0;

    if (!fast_mode && ctx->deblock_pipeline) {
        /* Two whole-frame dispatches - all vertical edges, THEN (after a
         * real vkCmdPipelineBarrier via insert_compute_barrier(), not just
         * an intra-workgroup barrier()) all horizontal edges - matching
         * ITU-T H.264 8.7's required "vertical edges of the whole picture
         * before any horizontal edge" ordering. See deblock_filter.comp's
         * top-of-file comment for the full race-condition rationale: a
         * single dispatch cannot guarantee this ordering across different
         * macroblocks' independently-scheduled workgroups, since the
         * horizontal pass for one macroblock reads pixels a NEIGHBORING
         * macroblock's vertical pass may or may not have written yet.
         * pc[8] (num_slices for other stages, unused by this shader) is
         * repurposed here as pcs.pass (0=vertical, 1=horizontal) - see
         * deblock_filter.comp's PushConstants comment. */
        vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->deblock_pipeline);
        vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->deblock_layout, 0, 1, &ctx->deblock_desc_set, 0, NULL);

        uint32_t pc_deblock[9];
        memcpy(pc_deblock, pc, sizeof(uint32_t) * 8);

        pc_deblock[8] = 0; /* pass 0: vertical edges */
        vkCmdPushConstants(cmd_buf, ctx->deblock_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_deblock), pc_deblock);
        vkCmdDispatch(cmd_buf, width_mbs, height_mbs, 1);
        insert_compute_barrier(cmd_buf);

        pc_deblock[8] = 1; /* pass 1: horizontal edges */
        vkCmdPushConstants(cmd_buf, ctx->deblock_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_deblock), pc_deblock);
        vkCmdDispatch(cmd_buf, width_mbs, height_mbs, 1);
        insert_compute_barrier(cmd_buf);
    }
    if (ctx->perf_stats_enabled) {
        vkCmdWriteTimestamp(cmd_buf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx->timestamp_pools[perf_buf], 7);
    }

    /* Stage 6: Entropy - REMOVED (was: unconditional every-frame dispatch of
     * entropy_encode.comp whenever ctx->entropy_pipeline was non-null, which
     * is just "did the shader load", not "is this needed"). Its output
     * (entropy_buffer / staging_buffers[]) has never been read by the CPU -
     * nothing calls gpu_compute_get_staging_data() - real entropy coding runs
     * on the CPU (cavlc.c/cabac.c). So this was pure waste: a genuine
     * per-frame GPU dispatch (measured ~0.34ms via BC250_PERF_GPU's
     * entropy_ms) that did real work nobody ever used, and every frame
     * lengthens the single contended GPU queue this driver is stuck on
     * under contention (DEVLOG s.21/§ the phase-bracket work) for zero
     * benefit. Found while converting quant_levels_buffer to int16_t
     * storage: this dispatch reads that exact buffer through
     * entropy_desc_set binding 0, and entropy_encode.comp's own SPIR-V was
     * never updated to match (its output is discarded, so there was no
     * reason to) - left running, it would read out of bounds against the
     * now-half-sized buffer. Removing the dispatch fixes that risk and
     * removes a real GPU-time cost in the same stroke; the shader source,
     * pipeline object and descriptor sets are left in place (harmless, never
     * invoked) rather than torn out here, to keep this change to exactly
     * what it needs to be. */
    if (ctx->perf_stats_enabled) {
        vkCmdWriteTimestamp(cmd_buf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx->timestamp_pools[perf_buf], 8);
    }

    /* Copy of entropy_buffer -> staging_buffers[] REMOVED, same reason as the
     * dispatch above: entropy_buffer is never written to now (nothing
     * produces it), and staging_buffers[]'s own readback was already
     * confirmed dead (nothing calls gpu_compute_get_staging_data()) before
     * this change - so this was a GPU-time-costing copy of stale/undefined
     * data to a destination nothing reads, on every frame. Both buffers and
     * their staging/descriptor plumbing are left allocated (harmless) for
     * the same "minimal change" reason as above. */

    /* Copy the REAL post-quantization coefficient levels and pre-quantization
     * transform coefficients to their host-visible staging buffers, full size
     * each time (unlike the entropy copy above, which is deliberately capped
     * to width*height - that cap is specific to entropy_buffer's oversized
     * allocation and must not be carried forward here). */
    VkBufferCopy quant_copy_region = { .srcOffset = 0, .dstOffset = 0, .size = ctx->quant_staging_size };
    vkCmdCopyBuffer(cmd_buf, ctx->quant_levels_buffer, ctx->quant_staging_buffers[ctx->current_buf], 1, &quant_copy_region);
    /* Only the compact per-block DC crosses to the host, not all of
     * coeff_buffer - 1/16th the copy. See gpu_compute.h's dc_coeff_buffer. */
    VkBufferCopy dc_copy_region = { .srcOffset = 0, .dstOffset = 0, .size = ctx->dc_staging_size };
    vkCmdCopyBuffer(cmd_buf, ctx->dc_coeff_buffer, ctx->dc_staging_buffers[ctx->current_buf], 1, &dc_copy_region);

    /* Same for the real per-MB I16x16 pred mode and motion vectors residual_predict.comp
     * / motion_estimation.comp computed this frame - see gpu_compute_get_pred_mode_staging_data()
     * / gpu_compute_get_mv_staging_data(). */
    VkBufferCopy pred_mode_copy_region = { .srcOffset = 0, .dstOffset = 0, .size = ctx->pred_mode_staging_size };
    vkCmdCopyBuffer(cmd_buf, ctx->pred_mode_buffer, ctx->pred_mode_staging_buffers[ctx->current_buf], 1, &pred_mode_copy_region);
    VkBufferCopy mv_copy_region = { .srcOffset = 0, .dstOffset = 0, .size = ctx->mv_staging_size };
    vkCmdCopyBuffer(cmd_buf, ctx->mv_buffer, ctx->mv_staging_buffers[ctx->current_buf], 1, &mv_copy_region);

    /* And quantize.comp's per-block nonzero bitmask - 1/16th the size of the
     * quant_levels copy above, and it lets the CPU answer every
     * "is this block/MB all zero" question without reading quant_levels at
     * all. See gpu_compute_get_nz_staging_data(). */
    VkBufferCopy nz_copy_region = { .srcOffset = 0, .dstOffset = 0, .size = ctx->nz_staging_size };
    vkCmdCopyBuffer(cmd_buf, ctx->nz_count_buffer, ctx->nz_staging_buffers[ctx->current_buf], 1, &nz_copy_region);

    /* ctx->recon_image is now populated directly by Stage 4.5 (Reconstruct)
     * above via real dequant+IDCT+add-back+clip - see that stage's comment.
     * This replaces the old raw vkCmdCopyImage-from-render_target (source
     * pixels) that used to run here; has_recon_frame is set by Stage 4.5. */

    if (ctx->perf_stats_enabled) {
        vkCmdWriteTimestamp(cmd_buf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx->timestamp_pools[perf_buf], 9);
    }

    return 0;
}

int gpu_compute_dispatch_encode(gpu_context_t *ctx, gpu_image_t render_target, int width, int height, int qp, int is_intra, int num_slices) {
    return gpu_compute_dispatch_encode_ext(ctx, render_target, width, height, qp, is_intra, num_slices, 0, NULL);
}


/* See gpu_compute.h. One dispatch over the destination rectangle, then a
 * wait: this runs from vaEndPicture and the caller expects the surface to
 * be finished when it returns. */
int gpu_compute_video_proc(gpu_context_t *ctx,
                           gpu_image_t *src, const int src_rect[4],
                           gpu_image_t *dst, const int dst_rect[4])
{
    if (!ctx || !src || !dst || !src_rect || !dst_rect) return -1;
    if (src->format != dst->format) return -1;

    VkPipeline pipeline = (dst->format == GPU_IMAGE_P010)
                          ? ctx->vpp_pipeline10 : ctx->vpp_pipeline;
    if (pipeline == VK_NULL_HANDLE) return -1;
    if (!src->y_view || !src->uv_view || !dst->y_view || !dst->uv_view) return -1;

    /* ⚠️ Even origins and even sizes. A 4:2:0 plane has one chroma sample
     * per two-by-two luma block, so an odd rectangle has no chroma to
     * write for half of its edge - and the shader's own chroma step
     * assumes the origin is even. Rounded here, once, rather than argued
     * about in the shader. */
    const int sx = src_rect[0] & ~1, sy = src_rect[1] & ~1;
    const int sw = src_rect[2] & ~1, sh = src_rect[3] & ~1;
    const int dx = dst_rect[0] & ~1, dy = dst_rect[1] & ~1;
    const int dw = dst_rect[2] & ~1, dh = dst_rect[3] & ~1;
    if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0) return -1;

    VkCommandBuffer cmd_buf = ctx->cmd_bufs[ctx->current_buf];
    if (gpu_compute_begin_picture(ctx, *dst) != 0) return -1;

    transition_image_layout(cmd_buf, src->y_plane, src->current_layout, VK_IMAGE_LAYOUT_GENERAL);
    transition_image_layout(cmd_buf, src->uv_plane, src->current_layout, VK_IMAGE_LAYOUT_GENERAL);
    src->current_layout = VK_IMAGE_LAYOUT_GENERAL;
    transition_image_layout(cmd_buf, dst->y_plane, dst->current_layout, VK_IMAGE_LAYOUT_GENERAL);
    transition_image_layout(cmd_buf, dst->uv_plane, dst->current_layout, VK_IMAGE_LAYOUT_GENERAL);
    dst->current_layout = VK_IMAGE_LAYOUT_GENERAL;

    update_storage_image_descriptor(ctx->device, ctx->vpp_desc_set, 0, src->y_view);
    update_storage_image_descriptor(ctx->device, ctx->vpp_desc_set, 1, src->uv_view);
    update_storage_image_descriptor(ctx->device, ctx->vpp_desc_set, 2, dst->y_view);
    update_storage_image_descriptor(ctx->device, ctx->vpp_desc_set, 3, dst->uv_view);

    const int32_t pc[10] = {
        sx, sy, sw, sh,
        dx, dy, dw, dh,
        (int32_t)dst->width, (int32_t)dst->height
    };

    vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->vpp_layout, 0, 1, &ctx->vpp_desc_set, 0, NULL);
    vkCmdPushConstants(cmd_buf, ctx->vpp_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
    vkCmdDispatch(cmd_buf, ((uint32_t)dw + 15) / 16, ((uint32_t)dh + 15) / 16, 1);
    insert_compute_barrier(cmd_buf);

    if (gpu_compute_end_picture(ctx) != 0) return -1;
    return gpu_compute_sync(ctx);
}

int gpu_compute_dispatch_me_only(gpu_context_t *ctx, gpu_image_t render_target, int width, int height) {
    if (!ctx || !ctx->motion_est_pipeline) return -1;

    /* The buffers have to be allocated here as well.
     *
     * gpu_compute_dispatch_encode_ext() carries a comment saying its
     * allocation is "the ONLY place they get allocated" and that "proceeding
     * with VK_NULL_HANDLE buffers is what turned an out-of-memory into a
     * SEGV". This function is a second door into the same buffers, and it
     * walked past that check: the only caller is the HEVC encoder, which
     * therefore dispatched motion estimation with an OutputMV descriptor
     * nobody had written and then copied from a null mv_buffer.
     *
     * Measured on a BC-250: SIGSEGV in radv_CmdCopyBuffer2, with the
     * validation layers naming both halves - "the descriptor ... variable
     * OutputMV is being used in dispatch but has never been updated" and
     * "vkCmdCopyBuffer(): srcBuffer is VK_NULL_HANDLE".
     *
     * Same condition and same failure handling as the encode path, so the two
     * doors behave alike.
     */
    if (ctx->staging_buffers[0] == VK_NULL_HANDLE ||
        ctx->frame_width != (uint32_t)width ||
        ctx->frame_height != (uint32_t)height) {
        if (allocate_encoding_buffers(ctx, (uint32_t)width, (uint32_t)height) != 0) {
            return -1;
        }
    }

    VkCommandBuffer cmd_buf = ctx->cmd_bufs[ctx->current_buf];
    uint32_t width_mbs = ((uint32_t)width + 15) / 16;
    uint32_t height_mbs = ((uint32_t)height + 15) / 16;

    if (render_target.y_plane) {
        transition_image_layout(cmd_buf, render_target.y_plane, render_target.current_layout, VK_IMAGE_LAYOUT_GENERAL);
    }
    if (render_target.uv_plane) {
        transition_image_layout(cmd_buf, render_target.uv_plane, render_target.current_layout, VK_IMAGE_LAYOUT_GENERAL);
    }
    render_target.current_layout = VK_IMAGE_LAYOUT_GENERAL;

    VkImageView ref_view = render_target.y_view;
    if (ctx->has_recon_frame && ctx->recon_image.y_view != VK_NULL_HANDLE) {
        ref_view = ctx->recon_image.y_view;
    }

    if (render_target.y_view) {
        update_storage_image_descriptor(ctx->device, ctx->me_desc_set, 0, render_target.y_view);
        update_storage_image_descriptor(ctx->device, ctx->me_desc_set, 1, ref_view);
    }

    uint32_t pc[8] = {
        (uint32_t)width,
        (uint32_t)height,
        width_mbs,
        height_mbs,
        27, /* qp */
        0,  /* is_intra = 0 for ME */
        0,  /* color_space */
        0   /* lambda_motion */
    };

    vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->motion_est_pipeline);
    vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->motion_est_layout, 0, 1, &ctx->me_desc_set, 0, NULL);
    vkCmdPushConstants(cmd_buf, ctx->motion_est_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
    vkCmdDispatch(cmd_buf, width_mbs, height_mbs, 1);
    insert_compute_barrier(cmd_buf);

    VkBufferCopy mv_copy_region = { .srcOffset = 0, .dstOffset = 0, .size = ctx->mv_staging_size };
    vkCmdCopyBuffer(cmd_buf, ctx->mv_buffer, ctx->mv_staging_buffers[ctx->current_buf], 1, &mv_copy_region);

    /* Update recon_image with current frame's pixels for next frame's reference */
    if (ctx->recon_image.y_plane && render_target.y_plane) {
        transition_image_layout(cmd_buf, ctx->recon_image.y_plane, ctx->recon_image.current_layout, VK_IMAGE_LAYOUT_GENERAL);
        ctx->recon_image.current_layout = VK_IMAGE_LAYOUT_GENERAL;

        VkImageCopy copy_region = {
            .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
            .srcOffset = { 0, 0, 0 },
            .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
            .dstOffset = { 0, 0, 0 },
            .extent = { (uint32_t)width, (uint32_t)height, 1 }
        };
        vkCmdCopyImage(cmd_buf, render_target.y_plane, VK_IMAGE_LAYOUT_GENERAL,
                       ctx->recon_image.y_plane, VK_IMAGE_LAYOUT_GENERAL, 1, &copy_region);
        ctx->has_recon_frame = true;
    }

    return 0;
}

int gpu_compute_end_picture(gpu_context_t *ctx) {
    struct timespec e0, e1, s0, s1;
    if (ctx->perf_stats_enabled) clock_gettime(CLOCK_MONOTONIC, &e0);
    vkEndCommandBuffer(ctx->cmd_bufs[ctx->current_buf]);
    if (ctx->perf_stats_enabled) clock_gettime(CLOCK_MONOTONIC, &e1);

    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &ctx->cmd_bufs[ctx->current_buf]
    };
    /* See gpu_compute_wait_for_image_ready()'s doc comment. If it ran for
     * this frame's render target, make this submission explicitly wait on
     * whatever last wrote into that surface (potentially a different GPU
     * API context entirely, e.g. Sunshine's own GL) before the compute
     * shaders below start reading it. VK_SEMAPHORE_IMPORT_TEMPORARY_BIT
     * means this semaphore reverts to its prior (unsignaled, no payload)
     * state once this wait consumes it, so has_pending_wait_semaphore
     * exactly tracks whether a payload is currently imported. */
    VkPipelineStageFlags wait_stage_mask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    if (ctx->has_pending_wait_semaphore) {
        submit_info.waitSemaphoreCount = 1;
        submit_info.pWaitSemaphores = &ctx->image_ready_semaphore;
        submit_info.pWaitDstStageMask = &wait_stage_mask;
        ctx->has_pending_wait_semaphore = false;
    }
    if (ctx->perf_stats_enabled) clock_gettime(CLOCK_MONOTONIC, &s0);
    /* vkQueueSubmit()'s return value was previously discarded entirely.
     * Confirmed on real hardware (see the real-content investigation in
     * docs/DEVLOG.md - "the encoder appears stuck/frozen" report,
     * root-caused via BC250_PERF_STATS + BC250_DUMP_REAL_INPUT showing 40+
     * consecutive real, genuinely-different-content frames all encoding to
     * the exact same byte count): this queue is the same one that already
     * needed a documented retry-with-backoff workaround in
     * gpu_compute_create_image() for real GPU contention (a concurrently
     * running desktop compositor/game) causing VK_ERROR_UNKNOWN - the exact
     * same contention can make a *submission*, not just an allocation,
     * transiently fail. Per the Vulkan spec, a failed vkQueueSubmit leaves
     * fence signaling undefined; on real hardware this fence still read as
     * signaled, so the previous code's unconditional vkWaitForFences()
     * right after (in gpu_compute_sync(), called unconditionally by every
     * caller) returned immediately without the GPU having done any new
     * work - meaning quant_buffer/coeff_buffer/pred_mode_buffer/mv_buffer,
     * and the staging copies vkCmdCopyBuffer'd from them, silently kept
     * whatever the previous *successful* dispatch had left in them. The
     * CPU-side CAVLC/CABAC encoder then deterministically re-emitted that
     * stale residual/MV/mode data - producing bitstream output identical
     * to a previous frame despite genuinely different real input, which is
     * exactly the "frozen" pattern found. This is a systemic transient
     * failure, not a mode-specific one, so this queue itself hasn't
     * changed relative to a synthetic-content encode - it just was never
     * seen because no synthetic test runs an encode alongside a real,
     * concurrently-contending desktop compositor/game.
     *
     * Fixed the same way as the allocation case: check the result, retry
     * with the identical short-backoff schedule (this is the same
     * transient-contention class of failure, so there is no reason for a
     * different recovery policy), and - critically, unlike simply retrying
     * - report failure to the caller if every attempt fails, instead of
     * proceeding to toggle buffers and let the caller wait on a fence that
     * was reset but will now never be signaled. h264_encoder_encode_frame()
     * treats that failure as "no new GPU data this frame" (the same state
     * as gpu_ctx being NULL), which already has real, correct, spec-legal
     * handling: quant_levels/coeff/etc. stay NULL, and the existing
     * skip-decision logic (mb_has_any_luma_nonzero() etc. under a NULL
     * quant_levels) certifies the whole frame P_Skip - a real decoder
     * simply repeats its last reference picture for that frame, which is
     * the correct, safe behavior for "the encoder had nothing new to send
     * this frame," rather than sending fabricated stale data as if it were
     * this frame's real content. */
    VkResult submit_result = VK_ERROR_UNKNOWN;
    for (int attempt = 0; attempt < BC250_ALLOC_MAX_ATTEMPTS; attempt++) {
        if (attempt > 0) {
            long backoff_ms = 20L << (attempt - 1); /* 20, 40, 80... ms - same schedule as gpu_compute_create_image() */
            struct timespec ts = { .tv_sec = backoff_ms / 1000, .tv_nsec = (backoff_ms % 1000) * 1000000L };
            nanosleep(&ts, NULL);
            fprintf(stderr, "[bc250-gpu] Retrying queue submit (attempt %d/%d) after %ldms backoff\n",
                    attempt + 1, BC250_ALLOC_MAX_ATTEMPTS, backoff_ms);
        }
        submit_result = vkQueueSubmit(ctx->compute_queue, 1, &submit_info, ctx->fences[ctx->current_buf]);
        if (submit_result == VK_SUCCESS) {
            clock_gettime(CLOCK_MONOTONIC, &ctx->submit_time[ctx->current_buf]);
            break;
        }
        fprintf(stderr, "[bc250-gpu] vkQueueSubmit failed: %d\n", submit_result);
    }
    if (ctx->perf_stats_enabled) {
        clock_gettime(CLOCK_MONOTONIC, &s1);
        fprintf(stderr, "[BC250_PERF_SUBMIT] buf=%d end_cmdbuf_ms=%.3f queue_submit_ms=%.3f\n",
                ctx->current_buf, bc250_diag_delta_ms(&e0, &e1), bc250_diag_delta_ms(&s0, &s1));
    }

    ctx->current_buf = (ctx->current_buf + 1) % 2;
    return (submit_result == VK_SUCCESS) ? 0 : -1;
}

int gpu_compute_submitted_slot(gpu_context_t *ctx) {
    if (!ctx) return -1;
    /* end_picture() toggles current_buf AFTER submitting, so the slot holding
     * the frame just submitted is the one current_buf now points away from. */
    return (ctx->current_buf + 1) % 2;
}

/* Slot-explicit readback. A pipelined caller MUST use these and pass the slot
 * its own frame was submitted into: once it has submitted frame N+1, "the most
 * recently submitted slot" is no longer the frame it is about to entropy-code.
 *
 * Reading the wrong slot here is a silent one-frame data swap, and note which
 * checks CANNOT see it: the BC250_NZ_AUDIT mask audit compares quant_levels
 * against nz_masks, and in a slot mix-up BOTH come from the same wrong slot, so
 * they still agree exactly and the audit reports 0 mismatches. Byte-exactness
 * is also unavailable (pipelined output legitimately differs). The oracle that
 * does see it is PSNR on MOVING content, plus the tell that gave it away the
 * first time: end_sync_ms failing to collapse, which can only mean the finish
 * is waiting on a frame that was submitted after it. */
int gpu_compute_sync_slot(gpu_context_t *ctx, int slot) {
    if (!ctx || slot < 0 || slot > 1) return -1;
    const int prev_buf = slot;
    struct timespec w0, w1;
    if (ctx->perf_stats_enabled) clock_gettime(CLOCK_MONOTONIC, &w0);
    /* Unchecked until now - this is precisely the call the doc comment in
     * gpu_compute_end_picture() already identified as the vector for stale
     * staging-buffer reads ("the previous code's unconditional
     * vkWaitForFences() right after ... returned immediately without the
     * GPU having done any new work"). That fix only covers the case where
     * the *submit* failed; if this wait itself returns an error (e.g.
     * VK_ERROR_DEVICE_LOST under real GPU contention) even though the
     * submit reported success, the exact same stale-buffer read follows.
     * Report failure so the caller skips the staging-buffer fetch instead
     * of reading quant/coeff/mv data the GPU may still be mid-write on. */
    VkResult wait_result = vkWaitForFences(ctx->device, 1, &ctx->fences[prev_buf], VK_TRUE, UINT64_MAX);
    struct timespec sync_now;
    clock_gettime(CLOCK_MONOTONIC, &sync_now);
    ctx->last_gpu_duration_ms = bc250_diag_delta_ms(&ctx->submit_time[prev_buf], &sync_now);
    if (ctx->perf_stats_enabled) {
        clock_gettime(CLOCK_MONOTONIC, &w1);
        fprintf(stderr, "[BC250_PERF_WAIT] site=sync buf=%d wait_ms=%.3f gpu_total_ms=%.3f\n",
                prev_buf, bc250_diag_delta_ms(&w0, &w1), ctx->last_gpu_duration_ms);
    }
    if (wait_result != VK_SUCCESS) {
        fprintf(stderr, "[bc250-gpu] vkWaitForFences failed in sync: %d\n", wait_result);
        return -1;
    }

    /* Opt-in GPU per-stage timing readback (BC250_PERF_STATS=1). Safe to
     * read now: the fence above just confirmed this exact command buffer's
     * submission (the one gpu_compute_dispatch_encode()/gpu_compute_end_picture()
     * most recently recorded into buffer `prev_buf`) has completed on the
     * GPU, so every vkCmdWriteTimestamp in it is guaranteed available -
     * VK_QUERY_RESULT_WAIT_BIT is added only as defense-in-depth. */
    if (ctx->perf_stats_enabled && ctx->timestamp_pools[prev_buf] && ctx->perf_result_pending[prev_buf]) {
        ctx->perf_result_pending[prev_buf] = false;
        uint64_t ts[BC250_PERF_NUM_TIMESTAMPS];
        VkResult qres = vkGetQueryPoolResults(ctx->device, ctx->timestamp_pools[prev_buf], 0, BC250_PERF_NUM_TIMESTAMPS,
                                               sizeof(ts), ts, sizeof(uint64_t),
                                               VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        if (qres == VK_SUCCESS) {
            double p = ctx->timestamp_period_ns;
            double me_ms        = (double)(ts[1] - ts[0]) * p / 1e6;
            double predict_ms   = (double)(ts[2] - ts[1]) * p / 1e6;
            double dct_ms       = (double)(ts[3] - ts[2]) * p / 1e6;
            double quant_ms     = (double)(ts[4] - ts[3]) * p / 1e6;
            double reconstr_ms  = (double)(ts[5] - ts[4]) * p / 1e6;
            double wavefront_ms = (double)(ts[6] - ts[5]) * p / 1e6;
            double deblock_ms   = (double)(ts[7] - ts[6]) * p / 1e6;
            double entropy_ms   = (double)(ts[8] - ts[7]) * p / 1e6;
            double copy_ms      = (double)(ts[9] - ts[8]) * p / 1e6;
            double total_ms     = (double)(ts[9] - ts[0]) * p / 1e6;
            fprintf(stderr,
                "[BC250_PERF_GPU] frame=%u type=%s total_ms=%.3f me_ms=%.3f predict_ms=%.3f dct_ms=%.3f "
                "quant_ms=%.3f reconstruct_ms=%.3f wavefront_ms=%.3f deblock_ms=%.3f entropy_ms=%.3f copy_ms=%.3f\n",
                ctx->perf_frame_counter, ctx->perf_is_intra[prev_buf] ? "I" : "P",
                total_ms, me_ms, predict_ms, dct_ms, quant_ms, reconstr_ms, wavefront_ms, deblock_ms, entropy_ms, copy_ms);
            ctx->perf_frame_counter++;
        }
    }

    return 0;
}

int gpu_compute_sync(gpu_context_t *ctx) {
    if (!ctx) return -1;
    return gpu_compute_sync_slot(ctx, gpu_compute_submitted_slot(ctx));
}

double gpu_compute_get_last_latency_ms(const gpu_context_t *ctx) {
    return ctx ? ctx->last_gpu_duration_ms : 0.0;
}

int gpu_compute_get_staging_data(gpu_context_t *ctx, void **data, size_t *size) {
    if (!ctx || !data || !size) return -1;
    int prev_buf = (ctx->current_buf + 1) % 2;
    *size = ctx->staging_size;
    *data = ctx->staging_mapped[prev_buf];
    return (*data != NULL) ? 0 : -1;
}

int gpu_compute_release_staging_data(gpu_context_t *ctx) {
    (void)ctx;
    /* Persistently mapped: zero syscall overhead */
    return 0;
}

/* See gpu_compute_sync_slot()'s comment for why a pipelined caller must pass
 * its own frame's slot rather than relying on "most recently submitted". */
int gpu_compute_get_quant_staging_data_slot(gpu_context_t *ctx, int slot, void **data, size_t *size) {
    if (!ctx || !data || !size || slot < 0 || slot > 1) return -1;
    *size = ctx->quant_staging_size;
    *data = ctx->quant_staging_mapped[slot];
    return (*data != NULL) ? 0 : -1;
}

int gpu_compute_get_dc_staging_data_slot(gpu_context_t *ctx, int slot, void **data, size_t *size) {
    if (!ctx || !data || !size || slot < 0 || slot > 1) return -1;
    *size = ctx->dc_staging_size;
    *data = ctx->dc_staging_mapped[slot];
    return (*data != NULL) ? 0 : -1;
}

int gpu_compute_get_pred_mode_staging_data_slot(gpu_context_t *ctx, int slot, void **data, size_t *size) {
    if (!ctx || !data || !size || slot < 0 || slot > 1) return -1;
    *size = ctx->pred_mode_staging_size;
    *data = ctx->pred_mode_staging_mapped[slot];
    return (*data != NULL) ? 0 : -1;
}

int gpu_compute_get_mv_staging_data_slot(gpu_context_t *ctx, int slot, void **data, size_t *size) {
    if (!ctx || !data || !size || slot < 0 || slot > 1) return -1;
    *size = ctx->mv_staging_size;
    *data = ctx->mv_staging_mapped[slot];
    return (*data != NULL) ? 0 : -1;
}

int gpu_compute_get_quant_staging_data(gpu_context_t *ctx, void **data, size_t *size) {
    if (!ctx) return -1;
    return gpu_compute_get_quant_staging_data_slot(ctx, gpu_compute_submitted_slot(ctx), data, size);
}

int gpu_compute_get_dc_staging_data(gpu_context_t *ctx, void **data, size_t *size) {
    if (!ctx) return -1;
    return gpu_compute_get_dc_staging_data_slot(ctx, gpu_compute_submitted_slot(ctx), data, size);
}

int gpu_compute_get_pred_mode_staging_data(gpu_context_t *ctx, void **data, size_t *size) {
    if (!ctx) return -1;
    return gpu_compute_get_pred_mode_staging_data_slot(ctx, gpu_compute_submitted_slot(ctx), data, size);
}

int gpu_compute_get_mv_staging_data(gpu_context_t *ctx, void **data, size_t *size) {
    if (!ctx) return -1;
    return gpu_compute_get_mv_staging_data_slot(ctx, gpu_compute_submitted_slot(ctx), data, size);
}

int gpu_compute_get_nz_staging_data_slot(gpu_context_t *ctx, int slot, void **data, size_t *size) {
    if (!ctx || !data || !size || slot < 0 || slot > 1) return -1;
    *size = ctx->nz_staging_size;
    *data = ctx->nz_staging_mapped[slot];
    return (*data != NULL) ? 0 : -1;
}

int gpu_compute_get_nz_staging_data(gpu_context_t *ctx, void **data, size_t *size) {
    if (!ctx) return -1;
    return gpu_compute_get_nz_staging_data_slot(ctx, gpu_compute_submitted_slot(ctx), data, size);
}

/* Opt-in debug instrumentation, originally added for Part A verification
 * (reconstruction pipeline) and kept as a permanent low-risk diagnostic -
 * dumps ctx->recon_image's current contents as raw NV12 bytes,
 * same file-naming convention as bc250_debug_dump_nv12_frame(), gated by
 * BC250_DUMP_RECON_FRAMES=1 (BC250_DUMP_DIR for the directory, same as that
 * function). Call after gpu_compute_sync() so the frame's GPU writes are
 * guaranteed visible on the host. */
void gpu_compute_debug_dump_recon(gpu_context_t *ctx, int width, int height) {
    if (!getenv("BC250_DUMP_RECON_FRAMES")) return;
    if (!ctx || ctx->recon_image.y_plane == VK_NULL_HANDLE || width <= 0 || height <= 0) return;

    static int dump_frame_index = 0;

    size_t y_size = (size_t)width * height;
    size_t uv_size = (size_t)width * (height / 2);
    uint8_t *y_buf = malloc(y_size);
    uint8_t *uv_buf = malloc(uv_size);
    if (!y_buf || !uv_buf) { free(y_buf); free(uv_buf); return; }

    if (gpu_compute_download_nv12(ctx, &ctx->recon_image, ctx->recon_memory,
                                   y_buf, width, uv_buf, width, width, height) == 0) {
        char dump_name[64];
        snprintf(dump_name, sizeof(dump_name), "recon_%05d.nv12", dump_frame_index);
        FILE *dumpf = bc250_debug_dump_open(dump_name, "BC250_DUMP_RECON_FRAMES");
        if (dumpf) {
            fwrite(y_buf, 1, y_size, dumpf);
            fwrite(uv_buf, 1, uv_size, dumpf);
            fclose(dumpf);
        }
    }
    free(y_buf);
    free(uv_buf);
    dump_frame_index++;
}

/* See gpu_compute.h's doc comment: reads back whatever is ACTUALLY in the
 * surface right before encode, independent of how it got written there -
 * unlike bc250_debug_dump_nv12_frame(), which only fires from the two known
 * upload-path call sites and so never sees real Sunshine sessions (which
 * write into the surface's exported DMA-BUF via their own GL blit). */
void gpu_compute_debug_dump_real_input(gpu_context_t *ctx, gpu_image_t *image, gpu_memory_t memory, int width, int height) {
    if (!getenv("BC250_DUMP_REAL_INPUT")) return;
    if (!ctx || !image || image->y_plane == VK_NULL_HANDLE || width <= 0 || height <= 0) return;

    static int dump_frame_index = 0;

    size_t y_size = (size_t)width * height;
    size_t uv_size = (size_t)width * (height / 2);
    uint8_t *y_buf = malloc(y_size);
    uint8_t *uv_buf = malloc(uv_size);
    if (!y_buf || !uv_buf) { free(y_buf); free(uv_buf); return; }

    if (gpu_compute_download_nv12(ctx, image, memory,
                                   y_buf, width, uv_buf, width, width, height) == 0) {
        char dump_name[64];
        snprintf(dump_name, sizeof(dump_name), "real_%05d.nv12", dump_frame_index);
        FILE *dumpf = bc250_debug_dump_open(dump_name, "BC250_DUMP_REAL_INPUT");
        if (dumpf) {
            fwrite(y_buf, 1, y_size, dumpf);
            fwrite(uv_buf, 1, uv_size, dumpf);
            fclose(dumpf);
        }
    }
    free(y_buf);
    free(uv_buf);
    dump_frame_index++;
}
