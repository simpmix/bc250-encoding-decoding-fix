/* bc250-encoding-decoding-fix v0.4.3 - https://github.com/simpmix/bc250-encoding-decoding-fix */
/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * va_backend.c - Complete VA-API Backend Driver Implementation for AMD BC-250
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include "va_backend.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <math.h>

#ifndef VA_RC_ICQ
#define VA_RC_ICQ 0x00000040
#endif

#define BC250_MAX_WIDTH 4096
#define BC250_MAX_HEIGHT 4096

/* ⚠️ Decoding goes further than encoding. The limits above are the
 * encoder's and post-processing's; the decoder works in system memory and
 * only hands a finished picture to the GPU, so what bounds it is the
 * largest Vulkan image (16384 on either side here) and the largest
 * picture the stream is allowed to be - MaxLumaPs at level 6.2. Shapes
 * like 1056x8440 are legal and in the conformance suite; declaring 4096
 * made ffmpeg refuse to set up the hardware path for them at all. */
#define BC250_MAX_DECODE_SIDE 16384
#define BC250_MAX_DECODE_SAMPLES 35651584

/* Only the HEVC decoder has been taken there - the conformance suite has
 * pictures up to 8440 samples on a side. The H.264 one has not, so it
 * keeps the old limit until something proves it. */
static int big_decode(VAProfile profile, VAEntrypoint entrypoint)
{
    return entrypoint == VAEntrypointVLD
        && (profile == VAProfileHEVCMain || profile == VAProfileHEVCMain10);
}

static bc250_driver_data* get_driver_data(VADriverContextP ctx) {
    return (bc250_driver_data*)ctx->pDriverData;
}

VAStatus bc250_QueryConfigProfiles(VADriverContextP ctx, VAProfile *profile_list, int *num_profiles) {
    if (!ctx || !num_profiles) return VA_STATUS_ERROR_INVALID_PARAMETER;

    if (!profile_list) {
        *num_profiles = 5;
        return VA_STATUS_SUCCESS;
    }

    int i = 0;
    profile_list[i++] = VAProfileH264ConstrainedBaseline;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    profile_list[i++] = VAProfileH264Baseline;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
    profile_list[i++] = VAProfileH264Main;
    profile_list[i++] = VAProfileH264High;
    profile_list[i++] = VAProfileHEVCMain;
    /* Decoded, and encoded from P010 surfaces. */
    profile_list[i++] = VAProfileHEVCMain10;
    /* Post-processing hangs off no codec at all. */
    profile_list[i++] = VAProfileNone;

    *num_profiles = i;
    return VA_STATUS_SUCCESS;
}

VAStatus bc250_QueryConfigEntrypoints(VADriverContextP ctx, VAProfile profile, VAEntrypoint *entrypoint_list, int *num_entrypoints) {
    if (!ctx || !num_entrypoints) return VA_STATUS_ERROR_INVALID_PARAMETER;

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    int is_supported_profile = (profile == VAProfileH264ConstrainedBaseline ||
                                profile == VAProfileH264Baseline ||
                                profile == VAProfileH264Main ||
                                profile == VAProfileH264High ||
                                profile == VAProfileHEVCMain ||
                                profile == VAProfileHEVCMain10 ||
                                profile == VAProfileNone);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
    if (!is_supported_profile) {
        return VA_STATUS_ERROR_UNSUPPORTED_PROFILE;
    }

    /* H.264 and H.265 can be both encoded and decoded, Main 10 included:
     * the HEVC encoder has a ten-bit path that reads P010 surfaces and
     * writes a Main 10 stream. */
    /* VAProfileNone is post-processing and nothing else. */
    if (profile == VAProfileNone) {
        if (!entrypoint_list) {
            *num_entrypoints = 1;
            return VA_STATUS_SUCCESS;
        }
        entrypoint_list[0] = VAEntrypointVideoProc;
        *num_entrypoints = 1;
        return VA_STATUS_SUCCESS;
    }

    const int can_decode = (profile == VAProfileH264ConstrainedBaseline ||
                            profile == VAProfileH264Baseline ||
                            profile == VAProfileH264Main ||
                            profile == VAProfileH264High ||
                            profile == VAProfileHEVCMain ||
                            profile == VAProfileHEVCMain10);
    const int can_encode = 1;
    const int count = (can_decode ? 1 : 0) + (can_encode ? 1 : 0);

    if (!entrypoint_list) {
        *num_entrypoints = count;
        return VA_STATUS_SUCCESS;
    }

    int n = 0;
    if (can_encode) entrypoint_list[n++] = VAEntrypointEncSlice;
    if (can_decode) entrypoint_list[n++] = VAEntrypointVLD;
    *num_entrypoints = n;
    return VA_STATUS_SUCCESS;
}

VAStatus bc250_GetConfigAttributes(VADriverContextP ctx, VAProfile profile, VAEntrypoint entrypoint, VAConfigAttrib *attrib_list, int num_attribs) {
    (void)ctx;
    if (!attrib_list) return VA_STATUS_ERROR_INVALID_PARAMETER;

    for (int i = 0; i < num_attribs; i++) {
        switch (attrib_list[i].type) {
            case VAConfigAttribRTFormat:
                /* Main 10 takes ten-bit render targets and nothing else:
                 * offering eight as well would let an application create
                 * NV12 surfaces for a stream the decoder will write ten
                 * bits into. */
                if (profile == VAProfileHEVCMain10)
                    attrib_list[i].value = VA_RT_FORMAT_YUV420_10;
                else if (profile == VAProfileNone)
                    /* Post-processing takes either, and keeps the depth:
                     * it scales, it does not convert. */
                    attrib_list[i].value = VA_RT_FORMAT_YUV420
                                         | VA_RT_FORMAT_YUV420_10;
                else
                    attrib_list[i].value = VA_RT_FORMAT_YUV420;
                break;
            case VAConfigAttribRateControl:
                /* Meaningless for decoding, and saying so is better than
                 * naming three modes a decode config can never use. */
                if (entrypoint == VAEntrypointVLD) {
                    attrib_list[i].value = VA_ATTRIB_NOT_SUPPORTED;
                } else {
                    /* Advertise CBR, VBR, CQP, and ICQ.
                     * FFmpeg defaults to ICQ (Intelligent Constant Quality) when no
                     * bitrate is specified, matching Intel iGPU behavior (~4 Mbps H.264 /
                     * ~2.2 Mbps HEVC on 1080p) and preventing multi-gigabyte file blowups from
                     * unconstrained CQP defaults. Explicit CQP remains available when
                     * requested via -rc_mode CQP or -qp. */
                    attrib_list[i].value = VA_RC_CBR | VA_RC_VBR | VA_RC_CQP | VA_RC_ICQ;
                }
                break;
            case VAConfigAttribEncPackedHeaders:
                /* Advertise VA_ENC_PACKED_HEADER_SEQUENCE so container muxers (e.g. FFmpeg MP4/MKV)
                 * can extract sequence headers for container extradata (avcC/hvcC) without
                 * logging missing global header warnings. The driver embeds its own conforming
                 * in-band AUD/SPS/PPS/slice headers in the bitstream. */
                attrib_list[i].value = VA_ENC_PACKED_HEADER_SEQUENCE;
                break;
            case VAConfigAttribEncMaxRefFrames:
                attrib_list[i].value = 1;
                break;
            case VAConfigAttribMaxPictureWidth:
                attrib_list[i].value = big_decode(profile, entrypoint)
                                     ? BC250_MAX_DECODE_SIDE : BC250_MAX_WIDTH;
                break;
            case VAConfigAttribMaxPictureHeight:
                attrib_list[i].value = big_decode(profile, entrypoint)
                                     ? BC250_MAX_DECODE_SIDE : BC250_MAX_HEIGHT;
                break;
            case VAConfigAttribEncMaxSlices:
                /* Up to 16 slices per picture supported via multi-threaded OpenMP
                 * CPU entropy coding and BC250_SLICES_PER_FRAME for H.264.
                 * HEVC uses 1 slice per picture. */
                attrib_list[i].value = (profile == VAProfileHEVCMain) ? 1 : 16;
                break;
#ifdef VAConfigAttribEncQualityLevels
            case VAConfigAttribEncQualityLevels:
#endif
            case VAConfigAttribEncQualityRange:
                /* Quality levels 1..7 (1 = High Quality, 4 = Balanced, 7 = High Speed) */
                attrib_list[i].value = 7;
                break;
#if defined(VA_CHECK_VERSION)
#if VA_CHECK_VERSION(1, 13, 0)
            case VAConfigAttribEncHEVCFeatures:
                if (profile == VAProfileHEVCMain) {
                    VAConfigAttribValEncHEVCFeatures features;
                    memset(&features, 0, sizeof(features));
                    attrib_list[i].value = features.value;
                } else {
                    attrib_list[i].value = VA_ATTRIB_NOT_SUPPORTED;
                }
                break;
            case VAConfigAttribEncHEVCBlockSizes:
                if (profile == VAProfileHEVCMain) {
                    VAConfigAttribValEncHEVCBlockSizes bs;
                    memset(&bs, 0, sizeof(bs));
                    /* CTU 16x16: 1 << (1 + 3) = 16 */
                    bs.bits.log2_max_coding_tree_block_size_minus3 = 1;
                    bs.bits.log2_min_coding_tree_block_size_minus3 = 1;
                    /* Min CB 8x8: 1 << (0 + 3) = 8 */
                    bs.bits.log2_min_luma_coding_block_size_minus3 = 0;
                    /* Min/Max TB 4x4: 1 << (0 + 2) = 4 */
                    bs.bits.log2_max_luma_transform_block_size_minus2 = 0;
                    bs.bits.log2_min_luma_transform_block_size_minus2 = 0;
                    attrib_list[i].value = bs.value;
                } else {
                    attrib_list[i].value = VA_ATTRIB_NOT_SUPPORTED;
                }
                break;
#endif
#endif
            default:
                attrib_list[i].value = VA_ATTRIB_NOT_SUPPORTED;
                break;
        }
    }
    return VA_STATUS_SUCCESS;
}

VAStatus bc250_CreateConfig(VADriverContextP ctx, VAProfile profile, VAEntrypoint entrypoint, VAConfigAttrib *attrib_list, int num_attribs, VAConfigID *config_id) {
    bc250_driver_data *data = get_driver_data(ctx);
    if (!data || !config_id) return VA_STATUS_ERROR_INVALID_PARAMETER;

    /* data->configs[i].attribs is a fixed-size array (see bc250_config in
     * va_backend.h). Without this check a caller-supplied num_attribs larger
     * than that capacity would memcpy past the end of the attribs array and
     * corrupt adjacent bc250_config fields / neighboring array entries. */
    if (num_attribs < 0 || (size_t)num_attribs > (sizeof(((bc250_config *)0)->attribs) / sizeof(VAConfigAttrib))) {
        return VA_STATUS_ERROR_MAX_NUM_EXCEEDED;
    }

    DRIVER_LOCK(data);
    for (int i = 0; i < MAX_CONFIGS; i++) {
        if (!data->configs[i].allocated) {
            data->configs[i].allocated = 1;
            data->configs[i].profile = profile;
            data->configs[i].entrypoint = entrypoint;
            data->configs[i].num_attribs = num_attribs;
            if (num_attribs > 0 && attrib_list) {
                memcpy(data->configs[i].attribs, attrib_list, num_attribs * sizeof(VAConfigAttrib));
            }
            *config_id = i;
            DRIVER_UNLOCK(data);
            return VA_STATUS_SUCCESS;
        }
    }
    DRIVER_UNLOCK(data);
    return VA_STATUS_ERROR_MAX_NUM_EXCEEDED;
}

VAStatus bc250_DestroyConfig(VADriverContextP ctx, VAConfigID config_id) {
    bc250_driver_data *data = get_driver_data(ctx);
    if (!data) return VA_STATUS_ERROR_INVALID_CONTEXT;
    DRIVER_LOCK(data);
    if (!VALID_ID(config_id, MAX_CONFIGS) || !data->configs[config_id].allocated) {
        DRIVER_UNLOCK(data);
        return VA_STATUS_ERROR_INVALID_CONFIG;
    }
    data->configs[config_id].allocated = 0;
    DRIVER_UNLOCK(data);
    return VA_STATUS_SUCCESS;
}

VAStatus bc250_QueryConfigAttributes(VADriverContextP ctx, VAConfigID config_id, VAProfile *profile, VAEntrypoint *entrypoint, VAConfigAttrib *attrib_list, int *num_attribs) {
    bc250_driver_data *data = get_driver_data(ctx);
    if (!data) return VA_STATUS_ERROR_INVALID_CONTEXT;
    DRIVER_LOCK(data);
    if (!VALID_ID(config_id, MAX_CONFIGS) || !data->configs[config_id].allocated) {
        DRIVER_UNLOCK(data);
        return VA_STATUS_ERROR_INVALID_CONFIG;
    }

    if (profile) *profile = data->configs[config_id].profile;
    if (entrypoint) *entrypoint = data->configs[config_id].entrypoint;
    if (num_attribs) *num_attribs = data->configs[config_id].num_attribs;
    if (attrib_list && data->configs[config_id].num_attribs > 0) {
        memcpy(attrib_list, data->configs[config_id].attribs, data->configs[config_id].num_attribs * sizeof(VAConfigAttrib));
    }
    DRIVER_UNLOCK(data);
    return VA_STATUS_SUCCESS;
}

VAStatus bc250_QuerySurfaceAttributes(VADriverContextP ctx, VAConfigID config, VASurfaceAttrib *attrib_list, unsigned int *num_attribs) {
    if (!num_attribs) return VA_STATUS_ERROR_INVALID_PARAMETER;

    /* ⚠️ The pixel format depends on the config, so this one cannot
     * ignore it the way it used to. An unknown config answers NV12,
     * which is what every caller got before. */
    bc250_driver_data *data = get_driver_data(ctx);
    int ten_bit = 0, both = 0, decode = 0;
    if (data && VALID_ID(config, MAX_CONFIGS)
        && data->configs[config].allocated) {
        const VAProfile prof = data->configs[config].profile;
        ten_bit = prof == VAProfileHEVCMain10;
        /* Post-processing works at either depth, so it says so. */
        both = prof == VAProfileNone;
        decode = big_decode(prof, data->configs[config].entrypoint);
    }

    if (!attrib_list) {
        *num_attribs = both ? 4 : 3;
        return VA_STATUS_SUCCESS;
    }

    int i = 0;
    attrib_list[i].type = VASurfaceAttribPixelFormat;
    attrib_list[i].flags = VA_SURFACE_ATTRIB_GETTABLE | VA_SURFACE_ATTRIB_SETTABLE;
    attrib_list[i].value.type = VAGenericValueTypeInteger;
    attrib_list[i].value.value.i = ten_bit ? VA_FOURCC_P010 : VA_FOURCC_NV12;
    i++;

    if (both) {
        attrib_list[i].type = VASurfaceAttribPixelFormat;
        attrib_list[i].flags = VA_SURFACE_ATTRIB_GETTABLE | VA_SURFACE_ATTRIB_SETTABLE;
        attrib_list[i].value.type = VAGenericValueTypeInteger;
        attrib_list[i].value.value.i = VA_FOURCC_P010;
        i++;
    }

    attrib_list[i].type = VASurfaceAttribMaxWidth;
    attrib_list[i].flags = VA_SURFACE_ATTRIB_GETTABLE;
    attrib_list[i].value.type = VAGenericValueTypeInteger;
    attrib_list[i].value.value.i = decode ? BC250_MAX_DECODE_SIDE
                                          : BC250_MAX_WIDTH;
    i++;

    attrib_list[i].type = VASurfaceAttribMaxHeight;
    attrib_list[i].flags = VA_SURFACE_ATTRIB_GETTABLE;
    attrib_list[i].value.type = VAGenericValueTypeInteger;
    attrib_list[i].value.value.i = decode ? BC250_MAX_DECODE_SIDE
                                          : BC250_MAX_HEIGHT;
    i++;

    *num_attribs = i;
    return VA_STATUS_SUCCESS;
}

VAStatus bc250_CreateSurfaces(VADriverContextP ctx, int width, int height, int format, int num_surfaces, VASurfaceID *surfaces) {
    bc250_driver_data *data = get_driver_data(ctx);
    if (!data || !surfaces) return VA_STATUS_ERROR_INVALID_PARAMETER;
    if (width > data->max_width || height > data->max_height) return VA_STATUS_ERROR_RESOLUTION_NOT_SUPPORTED;

    DRIVER_LOCK(data);
    int allocated = 0;
    for (int i = 0; i < MAX_SURFACES && allocated < num_surfaces; i++) {
        if (!data->surfaces[i].allocated) {
            bc250_surface *surf = &data->surfaces[i];
            memset(surf, 0, sizeof(*surf));

            /* gpu_compute_create_image() can fail (e.g. Vulkan allocation
             * failure) and returns nonzero in that case (see VK_CHECK in
             * gpu_compute.c). The return value was previously discarded, so
             * a failed surface was still marked allocated and handed back
             * to the caller as a "valid" surface backed by a
             * partially-initialized/invalid gpu_image_t - any later use
             * (encode, GetImage/PutImage, DestroySurfaces) would operate on
             * garbage Vulkan handles. Skip publishing this surface on
             * failure instead.
             *
             * Stop the whole loop here rather than `continue`-ing to the
             * next slot: confirmed on-hardware (gdb) that under real GPU
             * contention (a live desktop compositor also driving this
             * GPU), once one vkBindImageMemory call has already failed
             * with VK_ERROR_UNKNOWN, an immediate retry on the very next
             * surface segfaults *inside* radv_BindImageMemory2() itself -
             * i.e. the failure leaves RADV's own allocator state for this
             * memory type in a condition that a same-loop-iteration retry
             * cannot safely probe further. Bailing out immediately and
             * surfacing VA_STATUS_ERROR_MAX_NUM_EXCEEDED to the caller
             * (via the `allocated < num_surfaces` check below) is the
             * failure this driver can actually recover from; hammering
             * the allocator again cannot be made safe from here. */
            /* ⚠️ Two different vocabularies. `format` here is VA's
             * render-target format, where YUV420 is 1; the image layer
             * wants GPU_IMAGE_NV12 or GPU_IMAGE_P010, where 1 is P010.
             * Handing one straight to the other allocated every
             * eight-bit surface as sixteen. */
            const int gpu_format = (format == VA_RT_FORMAT_YUV420_10)
                                 ? GPU_IMAGE_P010 : GPU_IMAGE_NV12;
            if (gpu_compute_create_image(&data->gpu, width, height, gpu_format, &surf->image, &surf->memory) != 0) {
                memset(surf, 0, sizeof(*surf));
                break;
            }

            void *mapped = NULL;
            if (surf->memory.memory && vkMapMemory(data->gpu.device, surf->memory.memory, 0, surf->memory.size, 0, &mapped) == VK_SUCCESS) {
                surf->mapped_ptr = mapped;
                surf->memory.mapped_ptr = mapped;
            } else {
                surf->mapped_ptr = NULL;
                surf->memory.mapped_ptr = NULL;
            }

            surf->allocated = 1;
            surf->width = width;
            surf->height = height;
            surf->format = format;
            surf->ref_count = 1;
            surf->is_exported = 0;

            surfaces[allocated++] = i;
        }
    }

    if (allocated < num_surfaces) {
        bc250_DestroySurfaces(ctx, surfaces, allocated);
        DRIVER_UNLOCK(data);
        return VA_STATUS_ERROR_MAX_NUM_EXCEEDED;
    }
    DRIVER_UNLOCK(data);
    return VA_STATUS_SUCCESS;
}

VAStatus bc250_CreateSurfaces2(VADriverContextP ctx, unsigned int format, unsigned int width, unsigned int height,
                              VASurfaceID *surfaces, unsigned int num_surfaces,
                              VASurfaceAttrib *attrib_list, unsigned int num_attribs) {
    if (attrib_list && num_attribs > 0) {
        for (unsigned int i = 0; i < num_attribs; i++) {
            if (attrib_list[i].type == VASurfaceAttribMemoryType) {
                int mem_type = attrib_list[i].value.value.i;
                /* This driver allocates internal Vulkan-backed surfaces. If an external caller
                 * requests zero-copy importing of external DMA-BUF memory (such as Gamescope or
                 * screencasting pipelines requesting DRM_PRIME / DRM_PRIME_2), we must return
                 * VA_STATUS_ERROR_UNSUPPORTED_MEMORY_TYPE rather than silently ignoring the attributes
                 * and returning an uninitialized blank surface (which causes solid green or black
                 * screens in Gamescope, Steam Link, and Sunshine). Returning an error here allows
                 * callers to correctly fall back to their working copy or EGL blit paths. */
                if (mem_type != VA_SURFACE_ATTRIB_MEM_TYPE_VA && mem_type != 0) {
                    return VA_STATUS_ERROR_UNSUPPORTED_MEMORY_TYPE;
                }
            }
        }
    }
    return bc250_CreateSurfaces(ctx, width, height, format, num_surfaces, surfaces);
}

/* Drop one reference on surface `id` (data->surfaces[id] must already be
 * known valid/allocated by the caller). The surface's Vulkan image/memory
 * are only actually freed once ref_count reaches zero - i.e. once both the
 * original vaCreateSurfaces() reference AND every vaDeriveImage()-derived
 * image's reference have been released. This is the single place that
 * performs the real Vulkan teardown, shared by bc250_DestroySurfaces()
 * (releasing the app's own reference) and bc250_DestroyBuffer()
 * (releasing a derived image's reference on the surface it aliases). */
static void bc250_surface_unref(bc250_driver_data *data, VASurfaceID id) {
    bc250_surface *surf = &data->surfaces[id];
    if (surf->ref_count > 0) {
        surf->ref_count--;
    }
    if (surf->ref_count <= 0) {
        if (surf->mapped_ptr && surf->memory.memory) {
            vkUnmapMemory(data->gpu.device, surf->memory.memory);
            surf->mapped_ptr = NULL;
            surf->memory.mapped_ptr = NULL;
        }
        gpu_compute_destroy_image(&data->gpu, surf->image, surf->memory);
        surf->allocated = 0;
        surf->pending_destroy = 0;
        surf->is_exported = 0;
    }
}

/* Until no decode into this surface is queued or running. Called with the
 * driver lock held exactly once - the condition variable releases one
 * level of the recursive mutex, and a second level would keep the decode
 * thread from ever finishing. */
static void wait_decoded(bc250_driver_data *data, VASurfaceID id) {
    while (VALID_ID(id, MAX_SURFACES) && data->surfaces[id].allocated
           && data->surfaces[id].decode_pending > 0)
        pthread_cond_wait(&data->idle, &data->lock);
}

void bc250_decode_finished(bc250_driver_data *data, VASurfaceID target,
                           VAStatus st) {
    DRIVER_LOCK(data);
    if (VALID_ID(target, MAX_SURFACES) && data->surfaces[target].allocated) {
        bc250_surface *s = &data->surfaces[target];
        if (s->decode_pending > 0) s->decode_pending--;
        s->decode_status = st;
        s->image.current_layout = VK_IMAGE_LAYOUT_GENERAL;
        bc250_surface_unref(data, target);
    }
    pthread_cond_broadcast(&data->idle);
    DRIVER_UNLOCK(data);
}

VAStatus bc250_DestroySurfaces(VADriverContextP ctx, VASurfaceID *surface_list, int num_surfaces) {
    bc250_driver_data *data = get_driver_data(ctx);
    if (!data || !surface_list) return VA_STATUS_ERROR_INVALID_PARAMETER;

    DRIVER_LOCK(data);
    for (int i = 0; i < num_surfaces; i++) {
        VASurfaceID id = surface_list[i];
        if (!VALID_ID(id, MAX_SURFACES) || !data->surfaces[id].allocated) continue;

        bc250_surface *surf = &data->surfaces[id];
        /* Idempotent: an application that (incorrectly) destroys the same
         * surface twice must not decrement ref_count twice for a single
         * app-held reference - only the first vaDestroySurfaces() call on
         * a given surface releases that reference. */
        if (surf->pending_destroy) continue;
        wait_decoded(data, id);

        /* From here on this VASurfaceID is invalid for the application to
         * use in any other VA call (vaBeginPicture, vaDeriveImage,
         * vaGetImage/vaPutImage, vaSyncSurface, ...), regardless of
         * whether the underlying Vulkan resources are freed immediately
         * below or kept alive a while longer for an outstanding derived
         * image - see bc250_surface.pending_destroy in va_backend.h. */
        surf->pending_destroy = 1;
        bc250_surface_unref(data, id);
    }
    DRIVER_UNLOCK(data);
    return VA_STATUS_SUCCESS;
}

VAStatus bc250_CreateContext(VADriverContextP ctx, VAConfigID config_id, int picture_width, int picture_height, int flag, VASurfaceID *render_targets, int num_render_targets, VAContextID *context) {
    bc250_driver_data *data = get_driver_data(ctx);
    if (!data || !VALID_ID(config_id, MAX_CONFIGS) || !data->configs[config_id].allocated || !context) {
        return VA_STATUS_ERROR_INVALID_CONFIG;
    }
    /* Surfaces are allowed up to the decoder's limit, since a surface does
     * not know what it will be used for; the context does, and the encoder
     * and post-processing stay at their own. */
    if (big_decode(data->configs[config_id].profile,
                   data->configs[config_id].entrypoint)) {
        if ((long)picture_width * picture_height > BC250_MAX_DECODE_SAMPLES)
            return VA_STATUS_ERROR_RESOLUTION_NOT_SUPPORTED;
    } else if (picture_width > BC250_MAX_WIDTH
               || picture_height > BC250_MAX_HEIGHT) {
        return VA_STATUS_ERROR_RESOLUTION_NOT_SUPPORTED;
    }

    DRIVER_LOCK(data);
    for (int i = 0; i < MAX_CONTEXTS; i++) {
        if (!data->contexts[i].allocated) {
            bc250_context *c = &data->contexts[i];
            memset(c, 0, sizeof(*c));
            c->allocated = 1;
            c->config_id = config_id;
            c->width = picture_width;
            c->height = picture_height;
            c->flag = flag;
            c->num_render_targets = num_render_targets;
            c->current_render_target = VA_INVALID_SURFACE;
            c->coded_buf_id = VA_INVALID_ID;

            if (num_render_targets > 0 && render_targets) {
                c->render_targets = malloc(num_render_targets * sizeof(VASurfaceID));
                memcpy(c->render_targets, render_targets, num_render_targets * sizeof(VASurfaceID));
            }

            VAProfile prof = data->configs[config_id].profile;
            VAEntrypoint entry = data->configs[config_id].entrypoint;

            if (entry == VAEntrypointEncSlice) {
                if (prof == VAProfileHEVCMain || prof == VAProfileHEVCMain10) {
                    c->hevc_enc = hevc_encoder_create_depth(&data->gpu, picture_width, picture_height, 30, 2200000,
                                                            prof == VAProfileHEVCMain10 ? 10 : 8);
                    if (c->hevc_enc) {
                        for (int a = 0; a < data->configs[config_id].num_attribs; a++) {
                            if (data->configs[config_id].attribs[a].type == VAConfigAttribRateControl) {
                                unsigned int rc_attrib = data->configs[config_id].attribs[a].value;
                                if (rc_attrib == VA_RC_CQP) {
                                    hevc_encoder_set_rc_mode(c->hevc_enc, RC_CQP);
                                } else if (rc_attrib & VA_RC_CBR) {
#if defined(__linux__)
                                    if (program_invocation_short_name &&
                                        (strcmp(program_invocation_short_name, "sunshine") == 0 ||
                                         strcmp(program_invocation_short_name, "wivrn-server") == 0 ||
                                         strcmp(program_invocation_short_name, "wivrn") == 0 ||
                                         strcmp(program_invocation_short_name, "steam") == 0 ||
                                         strcmp(program_invocation_short_name, "streaming_client") == 0)) {
                                        hevc_encoder_set_rc_mode(c->hevc_enc, RC_LOW_LATENCY);
                                    } else {
                                        hevc_encoder_set_rc_mode(c->hevc_enc, RC_CBR);
                                    }
#else
                                    hevc_encoder_set_rc_mode(c->hevc_enc, RC_CBR);
#endif
                                } else if (rc_attrib & (VA_RC_VBR | VA_RC_ICQ)) {
#if defined(__linux__)
                                    if (program_invocation_short_name &&
                                        (strcmp(program_invocation_short_name, "sunshine") == 0 ||
                                         strcmp(program_invocation_short_name, "wivrn-server") == 0 ||
                                         strcmp(program_invocation_short_name, "wivrn") == 0 ||
                                         strcmp(program_invocation_short_name, "steam") == 0 ||
                                         strcmp(program_invocation_short_name, "streaming_client") == 0)) {
                                        hevc_encoder_set_rc_mode(c->hevc_enc, RC_LOW_LATENCY);
                                    } else {
                                        hevc_encoder_set_rc_mode(c->hevc_enc, RC_VBR);
                                    }
#else
                                    hevc_encoder_set_rc_mode(c->hevc_enc, RC_VBR);
#endif
                                }
                                break;
                            }
                        }
                    }
                } else {
                    c->h264_enc = h264_encoder_create(&data->gpu, picture_width, picture_height, 30, 4000000, prof);
                    if (c->h264_enc) {
                        for (int a = 0; a < data->configs[config_id].num_attribs; a++) {
                            if (data->configs[config_id].attribs[a].type == VAConfigAttribRateControl) {
                                unsigned int rc_attrib = data->configs[config_id].attribs[a].value;
                                if (rc_attrib == VA_RC_CQP) {
                                    h264_encoder_set_rc_mode(c->h264_enc, RC_CQP);
                                } else if (rc_attrib & (VA_RC_VBR | VA_RC_ICQ)) {
                                    h264_encoder_set_rc_mode(c->h264_enc, RC_VBR);
                                } else if (rc_attrib & VA_RC_CBR) {
#if defined(__linux__)
                                    if (program_invocation_short_name &&
                                        (strcmp(program_invocation_short_name, "sunshine") == 0 ||
                                         strcmp(program_invocation_short_name, "wivrn-server") == 0 ||
                                         strcmp(program_invocation_short_name, "wivrn") == 0 ||
                                         strcmp(program_invocation_short_name, "steam") == 0 ||
                                         strcmp(program_invocation_short_name, "streaming_client") == 0)) {
                                        h264_encoder_set_rc_mode(c->h264_enc, RC_LOW_LATENCY);
                                    } else {
                                        h264_encoder_set_rc_mode(c->h264_enc, RC_CBR);
                                    }
#else
                                    h264_encoder_set_rc_mode(c->h264_enc, RC_CBR);
#endif
                                }
                                break;
                            }
                        }
                    }
                }
            } else if (entry == VAEntrypointVideoProc) {
                /* ⚠️ Refused here rather than at vaEndPicture. Without the
                 * shaders there is nothing to scale with, and finding that
                 * out halfway through a frame gives the application an
                 * error it cannot do anything about. */
                if (data->gpu.vpp_pipeline == VK_NULL_HANDLE) {
                    memset(c, 0, sizeof(*c));
                    DRIVER_UNLOCK(data);
                    return VA_STATUS_ERROR_UNSUPPORTED_ENTRYPOINT;
                }
                c->vpp = 1;
                c->vpp_state.source = VA_INVALID_SURFACE;
            } else if (entry == VAEntrypointVLD) {
                if (prof == VAProfileHEVCMain || prof == VAProfileHEVCMain10)
                    c->h265_dec = hevc_decoder_create(&data->gpu, picture_width,
                                                      picture_height);
                else
                    c->h264_dec = h264_decoder_create(&data->gpu, picture_width,
                                                      picture_height);
            }

            *context = i;
            DRIVER_UNLOCK(data);
            return VA_STATUS_SUCCESS;
        }
    }
    DRIVER_UNLOCK(data);
    return VA_STATUS_ERROR_MAX_NUM_EXCEEDED;
}

VAStatus bc250_DestroyContext(VADriverContextP ctx, VAContextID context) {
    bc250_driver_data *data = get_driver_data(ctx);
    if (!data) return VA_STATUS_ERROR_INVALID_CONTEXT;
    DRIVER_LOCK(data);
    if (!VALID_ID(context, MAX_CONTEXTS) || !data->contexts[context].allocated) {
        DRIVER_UNLOCK(data);
        return VA_STATUS_ERROR_INVALID_CONTEXT;
    }
    bc250_context *c = &data->contexts[context];
    /* ⚠️ The decode thread finishes what is queued before it stops, and
     * finishing takes the driver lock - so it is stopped with the lock
     * dropped. */
    if (c->hevc_async) {
        struct bc250_hevc_async *a = c->hevc_async;
        c->hevc_async = NULL;
        DRIVER_UNLOCK(data);
        bc250_hevc_async_stop(a);
        DRIVER_LOCK(data);
    }
    /* Deliberately dropped rather than finished: the client is tearing the
     * context down, so it is never going to read this frame's coded buffer,
     * and finishing would mean entropy-coding a frame nobody wants. Cleared
     * before the encoder is freed so nothing can later try to finish a frame
     * through a dangling h264_enc. */
    c->has_pending_frame = false;
    if (c->h264_enc) {
        h264_encoder_destroy(c->h264_enc);
        c->h264_enc = NULL;
    }
    if (c->hevc_enc) {
        hevc_encoder_destroy(c->hevc_enc);
        c->hevc_enc = NULL;
    }
    if (c->h264_dec) {
        h264_decoder_destroy(c->h264_dec);
        c->h264_dec = NULL;
    }
    if (c->h265_dec) {
        hevc_decoder_destroy(c->h265_dec);
        c->h265_dec = NULL;
    }
    bc250_dec_free(c);
    bc250_hevc_dec_free(c);
    if (c->render_targets) {
        free(c->render_targets);
        c->render_targets = NULL;
    }
    c->allocated = 0;
    DRIVER_UNLOCK(data);
    return VA_STATUS_SUCCESS;
}

VAStatus bc250_CreateBuffer(VADriverContextP ctx, VAContextID context, VABufferType type, unsigned int size, unsigned int num_elements, void *data_ptr, VABufferID *buf_id) {
    bc250_driver_data *data = get_driver_data(ctx);
    if (!data || !buf_id) return VA_STATUS_ERROR_INVALID_PARAMETER;
    (void)context;

    DRIVER_LOCK(data);
    int start_idx = data->next_buffer_hint;
    if (start_idx < 0 || start_idx >= MAX_BUFFERS) start_idx = 0;
    for (int count = 0; count < MAX_BUFFERS; count++) {
        int i = (start_idx + count) % MAX_BUFFERS;
        if (!data->buffers[i].allocated) {
            data->next_buffer_hint = (i + 1) % MAX_BUFFERS;
            bc250_buffer *b = &data->buffers[i];
            b->type = type;
            b->size = size;
            b->num_elements = num_elements;
            b->mapped = 0;
            b->is_derived = 0;
            b->gpu_mem = VK_NULL_HANDLE;
            b->derived_surface = VA_INVALID_SURFACE;

            size_t total_alloc = (size_t)size * num_elements;
            if (type == VAEncCodedBufferType) {
                total_alloc += sizeof(VACodedBufferSegment);
            }

            b->data = calloc(1, total_alloc > 0 ? total_alloc : 1);
            if (!b->data) {
                /* Leave the slot free (allocated stays 0) so this failure
                 * doesn't permanently strand a buffer slot with no backing
                 * memory - a caller that ignored this error and later called
                 * vaMapBuffer/vaDestroyBuffer on buf_id would otherwise
                 * dereference/free a NULL data pointer or operate on a slot
                 * that looks valid but never had memory. */
                DRIVER_UNLOCK(data);
                return VA_STATUS_ERROR_ALLOCATION_FAILED;
            }
            b->allocated = 1;
            if (data_ptr) {
                memcpy(b->data, data_ptr, (size_t)size * num_elements);
            } else if (type == VAEncCodedBufferType) {
                VACodedBufferSegment *seg = (VACodedBufferSegment *)b->data;
                seg->size = 0;
                seg->bit_offset = 0;
                seg->status = 0;
                seg->reserved = 0;
                seg->buf = ((uint8_t *)b->data) + sizeof(VACodedBufferSegment);
                seg->next = NULL;
            }
            *buf_id = i;
            DRIVER_UNLOCK(data);
            return VA_STATUS_SUCCESS;
        }
    }
    DRIVER_UNLOCK(data);
    return VA_STATUS_ERROR_MAX_NUM_EXCEEDED;
}

VAStatus bc250_BufferSetNumElements(VADriverContextP ctx, VABufferID buf_id, unsigned int num_elements) {
    bc250_driver_data *data = get_driver_data(ctx);
    if (!data) return VA_STATUS_ERROR_INVALID_CONTEXT;
    DRIVER_LOCK(data);
    if (!VALID_ID(buf_id, MAX_BUFFERS) || !data->buffers[buf_id].allocated) {
        DRIVER_UNLOCK(data);
        return VA_STATUS_ERROR_INVALID_BUFFER;
    }
    data->buffers[buf_id].num_elements = num_elements;
    DRIVER_UNLOCK(data);
    return VA_STATUS_SUCCESS;
}

/* Defined further down, next to bc250_EndPicture() where the pipeline lives. */
static bool bc250_pipeline_enabled(void);
static void bc250_finish_pending_frame(bc250_driver_data *data, bc250_context *c);

VAStatus bc250_MapBuffer(VADriverContextP ctx, VABufferID buf_id, void **pbuf) {
    bc250_driver_data *data = get_driver_data(ctx);
    if (!data) return VA_STATUS_ERROR_INVALID_CONTEXT;
    DRIVER_LOCK(data);
    if (!VALID_ID(buf_id, MAX_BUFFERS) || !data->buffers[buf_id].allocated || !pbuf) {
        DRIVER_UNLOCK(data);
        return VA_STATUS_ERROR_INVALID_BUFFER;
    }
    /* If this is the coded buffer of a frame still in flight, its bitstream
     * has not been written yet - produce it now, before handing the caller a
     * pointer to what would otherwise be an empty (or previous frame's)
     * buffer. This is the path a client takes when it reads frame N's output
     * without having submitted frame N+1; correct, just with no overlap. */
    if (bc250_pipeline_enabled()) {
        for (int i = 0; i < MAX_CONTEXTS; i++) {
            bc250_context *c = &data->contexts[i];
            if (c->allocated && c->has_pending_frame && c->pending_coded_buf_id == buf_id)
                bc250_finish_pending_frame(data, c);
        }
    }
    data->buffers[buf_id].mapped = 1;
    *pbuf = data->buffers[buf_id].data;
    DRIVER_UNLOCK(data);
    return VA_STATUS_SUCCESS;
}

VAStatus bc250_UnmapBuffer(VADriverContextP ctx, VABufferID buf_id) {
    bc250_driver_data *data = get_driver_data(ctx);
    if (!data) return VA_STATUS_ERROR_INVALID_CONTEXT;
    DRIVER_LOCK(data);
    if (!VALID_ID(buf_id, MAX_BUFFERS) || !data->buffers[buf_id].allocated) {
        DRIVER_UNLOCK(data);
        return VA_STATUS_ERROR_INVALID_BUFFER;
    }

    /* Test-harness instrumentation (tools/quality_test.sh): a derived-image
     * buffer (see bc250_DeriveImage) is a direct mapping of GPU surface
     * memory, so a caller (e.g. ffmpeg's vaapi hwupload doing a zero-copy
     * upload) writes frame data straight into it via vaMapBuffer without
     * ever calling vaPutImage / gpu_compute_upload_nv12(). Capture the
     * frame here, at unmap time, so that upload path is covered too. Only
     * active when BC250_DUMP_INPUT_FRAMES=1 (see bc250_debug_dump_nv12_frame).
     */
    if (data->buffers[buf_id].is_derived && data->buffers[buf_id].data && getenv("BC250_DUMP_INPUT_FRAMES")) {
        for (int i = 0; i < MAX_IMAGES; i++) {
            bc250_image *img = &data->images[i];
            if (img->allocated && img->buffer_id == buf_id) {
                if (VALID_ID(img->surface_id, MAX_SURFACES) && data->surfaces[img->surface_id].allocated) {
                    bc250_surface *surf = &data->surfaces[img->surface_id];
                    const uint8_t *base = (const uint8_t *)data->buffers[buf_id].data;
                    const uint8_t *y_plane = base + img->image.offsets[0];
                    const uint8_t *uv_plane = base + img->image.offsets[1];
                    int y_pitch = img->image.pitches[0] > 0 ? (int)img->image.pitches[0] : surf->width;
                    int uv_pitch = img->image.pitches[1] > 0 ? (int)img->image.pitches[1] : surf->width;
                    bc250_debug_dump_nv12_frame(y_plane, y_pitch, uv_plane, uv_pitch, surf->width, surf->height);
                }
                break;
            }
        }
    }

    data->buffers[buf_id].mapped = 0;
    DRIVER_UNLOCK(data);
    return VA_STATUS_SUCCESS;
}

VAStatus bc250_DestroyBuffer(VADriverContextP ctx, VABufferID buffer_id) {
    bc250_driver_data *data = get_driver_data(ctx);
    if (!data) return VA_STATUS_ERROR_INVALID_BUFFER;
    DRIVER_LOCK(data);
    /* A genuinely out-of-range ID is a real caller bug - keep erroring on
     * that. An in-range ID that's simply not currently allocated (already
     * destroyed, or never allocated) is treated as a harmless no-op instead
     * of an error: observed in practice (ffmpeg's vaapi_encode.c, e.g.
     * "Failed to destroy param buffer 0x1: invalid VABufferID" on the very
     * first frame) calling vaDestroyBuffer a second time on an ID it
     * believes it owns - this driver's own CreateBuffer/RenderPicture/
     * DestroyContext never proactively frees a buffer out from under the
     * caller (checked directly: RenderPicture only reads param data,
     * DestroyContext doesn't touch the buffer table at all), so the double
     * call is on the caller's side, most likely tied to how this driver
     * advertises VA_ENC_PACKED_HEADER_NONE (see bc250_GetConfigAttributes).
     * Several real VA-API drivers (including Mesa's) treat a destroy-again
     * on an already-gone buffer as success for the same reason - the
     * resource the caller wanted gone is, in fact, gone. */
    if (!VALID_ID(buffer_id, MAX_BUFFERS)) {
        DRIVER_UNLOCK(data);
        return VA_STATUS_ERROR_INVALID_BUFFER;
    }
    if (!data->buffers[buffer_id].allocated) {
        DRIVER_UNLOCK(data);
        return VA_STATUS_SUCCESS;
    }
    bc250_buffer *b = &data->buffers[buffer_id];
    if (b->is_derived) {
        if (b->gpu_mem != VK_NULL_HANDLE) {
            /* Unmap while the surface's VkDeviceMemory is still guaranteed
             * alive (it can't have been freed yet: this buffer's own
             * reference, taken in bc250_DeriveImage(), is still held at
             * this point and keeps the surface's ref_count above zero). */
            vkUnmapMemory(data->gpu.device, b->gpu_mem);
        }
        /* Release this derived image's reference on the surface it
         * aliases. If the application already called vaDestroySurfaces()
         * on that surface while this image was still alive, this is what
         * finally lets the surface's Vulkan resources be freed - safely,
         * now that nothing is mapping them anymore. */
        if (VALID_ID(b->derived_surface, MAX_SURFACES) && data->surfaces[b->derived_surface].allocated) {
            bc250_surface_unref(data, b->derived_surface);
        }
        b->derived_surface = VA_INVALID_SURFACE;
    } else {
        free(b->data);
    }
    b->data = NULL;
    b->allocated = 0;
    DRIVER_UNLOCK(data);
    return VA_STATUS_SUCCESS;
}

VAStatus bc250_BeginPicture(VADriverContextP ctx, VAContextID context, VASurfaceID render_target) {
    bc250_driver_data *data = get_driver_data(ctx);
    if (!data) return VA_STATUS_ERROR_INVALID_CONTEXT;
    DRIVER_LOCK(data);
    if (!VALID_ID(context, MAX_CONTEXTS) || !data->contexts[context].allocated) {
        DRIVER_UNLOCK(data);
        return VA_STATUS_ERROR_INVALID_CONTEXT;
    }
    if (!VALID_ID(render_target, MAX_SURFACES) || !data->surfaces[render_target].allocated ||
        data->surfaces[render_target].pending_destroy) {
        DRIVER_UNLOCK(data);
        return VA_STATUS_ERROR_INVALID_SURFACE;
    }

    /* Decoding into a surface still being decoded into would race the
     * decode thread for its contents. */
    wait_decoded(data, render_target);
    bc250_context *c = &data->contexts[context];
    c->current_render_target = render_target;
    c->coded_buf_id = VA_INVALID_ID;

    c->h264_state.has_seq = 0;
    c->h264_state.has_pic = 0;
    c->h264_state.has_slice = 0;

    c->hevc_state.has_seq = 0;
    c->hevc_state.has_pic = 0;
    c->hevc_state.has_slice = 0;

    if (c->h264_dec) bc250_dec_reset(c);
    if (c->h265_dec) bc250_hevc_dec_reset(c);

    DRIVER_UNLOCK(data);
    return VA_STATUS_SUCCESS;
}

VAStatus bc250_RenderPicture(VADriverContextP ctx, VAContextID context, VABufferID *buffers, int num_buffers) {
    bc250_driver_data *data = get_driver_data(ctx);
    if (!data) return VA_STATUS_ERROR_INVALID_CONTEXT;
    DRIVER_LOCK(data);
    if (!VALID_ID(context, MAX_CONTEXTS) || !data->contexts[context].allocated || !buffers) {
        DRIVER_UNLOCK(data);
        return VA_STATUS_ERROR_INVALID_CONTEXT;
    }
    bc250_context *c = &data->contexts[context];

    /* Pre-pass: capture target_percentage from any VAEncMiscParameterType-
     * RateControl buffer in this batch BEFORE processing anything, so the
     * sequence-parameter path below scales its raw bits_per_second by the
     * same percentage regardless of which buffer happens to come first in
     * the array. ffmpeg sends the intended target X as "50% of 2X" in both
     * buffers, and whichever is handled last previously won - so with
     * SeqParam last, rate control was re-initialized at 2X the real target.
     * Making this order-independent is the actual fix; see docs/DEVLOG.md
     * §15 and docs/rate_control_audit.md §2. */
    for (int i = 0; i < num_buffers; i++) {
        VABufferID pid = buffers[i];
        if (!VALID_ID(pid, MAX_BUFFERS) || !data->buffers[pid].allocated) continue;
        bc250_buffer *pb = &data->buffers[pid];
        if (pb->type != VAEncMiscParameterBufferType ||
            pb->size < sizeof(VAEncMiscParameterBuffer) || !pb->data) {
            continue;
        }
        VAEncMiscParameterBuffer *pmisc = (VAEncMiscParameterBuffer *)pb->data;
        if (pmisc->type != VAEncMiscParameterTypeRateControl) continue;
        VAEncMiscParameterRateControl *prc = (VAEncMiscParameterRateControl *)pmisc->data;
        unsigned int pct = prc->target_percentage;
        if (pct == 0 || pct > 100) pct = 100;
        c->h264_state.rc_target_percentage = pct;
    }

    /* A decode context takes a different set of buffers entirely, and
     * nothing below applies to it. The slices are only collected here; the
     * decoding happens in vaEndPicture, with the lock dropped. */
    if (c->h265_dec) {
        for (int i = 0; i < num_buffers; i++) {
            VABufferID buf_id = buffers[i];
            if (!VALID_ID(buf_id, MAX_BUFFERS) || !data->buffers[buf_id].allocated)
                continue;
            VAStatus st = bc250_hevc_dec_render(c, &data->buffers[buf_id]);
            if (st != VA_STATUS_SUCCESS) {
                DRIVER_UNLOCK(data);
                return st;
            }
        }
        DRIVER_UNLOCK(data);
        return VA_STATUS_SUCCESS;
    }

    if (c->h264_dec) {
        for (int i = 0; i < num_buffers; i++) {
            VABufferID buf_id = buffers[i];
            if (!VALID_ID(buf_id, MAX_BUFFERS) || !data->buffers[buf_id].allocated)
                continue;
            VAStatus st = bc250_dec_render(c, &data->buffers[buf_id]);
            if (st != VA_STATUS_SUCCESS) {
                DRIVER_UNLOCK(data);
                return st;
            }
        }
        DRIVER_UNLOCK(data);
        return VA_STATUS_SUCCESS;
    }

    if (c->vpp) {
        for (int i = 0; i < num_buffers; i++) {
            VABufferID buf_id = buffers[i];
            if (!VALID_ID(buf_id, MAX_BUFFERS) || !data->buffers[buf_id].allocated)
                continue;
            bc250_buffer *b = &data->buffers[buf_id];
            if (b->type != VAProcPipelineParameterBufferType) continue;
            if (b->size < sizeof(VAProcPipelineParameterBuffer)) continue;

            const VAProcPipelineParameterBuffer *pp = b->data;

            /* Filters are advertised as none, so a caller asking for one
             * is asking for something this driver said it cannot do. */
            if (pp->num_filters > 0) {
                DRIVER_UNLOCK(data);
                return VA_STATUS_ERROR_UNSUPPORTED_FILTER;
            }
            if (pp->rotation_state != VA_ROTATION_NONE
                || pp->mirror_state != VA_MIRROR_NONE) {
                DRIVER_UNLOCK(data);
                return VA_STATUS_ERROR_UNSUPPORTED_FILTER;
            }

            c->vpp_state.source = pp->surface;
            c->vpp_state.has_source = 1;

            /* ⚠️ Dereferenced HERE. These point into the caller's memory
             * and are only guaranteed to be alive for this call; the
             * buffer holds the pointers, not the rectangles. */
            c->vpp_state.has_src_rect = 0;
            if (pp->surface_region) {
                c->vpp_state.src_rect = *pp->surface_region;
                c->vpp_state.has_src_rect = 1;
            }
            c->vpp_state.has_dst_rect = 0;
            if (pp->output_region) {
                c->vpp_state.dst_rect = *pp->output_region;
                c->vpp_state.has_dst_rect = 1;
            }
        }
        DRIVER_UNLOCK(data);
        return VA_STATUS_SUCCESS;
    }

    for (int i = 0; i < num_buffers; i++) {
        VABufferID buf_id = buffers[i];
        if (!VALID_ID(buf_id, MAX_BUFFERS) || !data->buffers[buf_id].allocated) continue;

        bc250_buffer *b = &data->buffers[buf_id];
        switch (b->type) {
            case VAEncSequenceParameterBufferType:
                if (c->h264_enc && b->size >= sizeof(VAEncSequenceParameterBufferH264)) {
                    memcpy(&c->h264_state.seq_param, b->data, sizeof(VAEncSequenceParameterBufferH264));
                    c->h264_state.has_seq = 1;
                    VAEncSequenceParameterBufferH264 *seq = &c->h264_state.seq_param;
                    if (seq->intra_period > 0) {
                        h264_encoder_set_gop_size(c->h264_enc, seq->intra_period);
                    }
                    if (seq->time_scale > 0 && seq->num_units_in_tick > 0) {
                        uint32_t fps = seq->time_scale / (2 * seq->num_units_in_tick);
                        if (fps > 0) {
                            h264_encoder_set_fps(c->h264_enc, fps);
                        }
                    }
                    /* SPS frame-cropping window: ffmpeg aligns context height to 16 (1080->1088).
                     * Pass sequence crop offsets so the stream carries true display dimensions. */
                    h264_encoder_set_cropping(c->h264_enc,
                                              seq->frame_cropping_flag,
                                              seq->frame_crop_left_offset,
                                              seq->frame_crop_right_offset,
                                              seq->frame_crop_top_offset,
                                              seq->frame_crop_bottom_offset);
                    if (getenv("BC250_DEBUG_RC")) {
                        fprintf(stderr, "[bc250-rc] SeqParam H264: cropping=%u l=%u r=%u t=%u b=%u\n",
                                (unsigned)seq->frame_cropping_flag,
                                seq->frame_crop_left_offset, seq->frame_crop_right_offset,
                                seq->frame_crop_top_offset, seq->frame_crop_bottom_offset);
                    }
                    if (seq->bits_per_second > 0) {
                        unsigned int pct = c->h264_state.rc_target_percentage;
                        if (pct == 0 || pct > 100) pct = 100;
                        uint32_t seq_target = (uint32_t)(((uint64_t)seq->bits_per_second * pct) / 100);
                        if (seq_target == 0) seq_target = seq->bits_per_second;
                        if (getenv("BC250_DEBUG_RC")) {
                            fprintf(stderr, "[bc250-rc] SeqParam H264: bits_per_second=%u pct=%u -> target=%u "
                                            "intra_period=%u\n",
                                    seq->bits_per_second, pct, seq_target, seq->intra_period);
                        }
                        h264_encoder_set_bitrate(c->h264_enc, seq_target);
                    } else if (getenv("BC250_DEBUG_RC")) {
                        fprintf(stderr, "[bc250-rc] SeqParam H264: bits_per_second=0 intra_period=%u\n",
                                seq->intra_period);
                    }
                } else if (c->hevc_enc && b->size >= sizeof(VAEncSequenceParameterBufferHEVC)) {
                    memcpy(&c->hevc_state.seq_param, b->data, sizeof(VAEncSequenceParameterBufferHEVC));
                    c->hevc_state.has_seq = 1;
                    VAEncSequenceParameterBufferHEVC *seq = &c->hevc_state.seq_param;
                    if (seq->intra_period > 0) {
                        hevc_encoder_set_gop_size(c->hevc_enc, seq->intra_period);
                    }
                    if (seq->vui_time_scale > 0 && seq->vui_num_units_in_tick > 0) {
                        uint32_t fps = seq->vui_time_scale / seq->vui_num_units_in_tick;
                        if (fps > 0) {
                            hevc_encoder_set_fps(c->hevc_enc, fps);
                        }
                    }
                    if (seq->bits_per_second > 0) {
                        unsigned int pct = c->h264_state.rc_target_percentage;
                        if (pct == 0 || pct > 100) pct = 100;
                        uint32_t seq_target = (uint32_t)(((uint64_t)seq->bits_per_second * pct) / 100);
                        if (seq_target == 0) seq_target = seq->bits_per_second;
                        if (getenv("BC250_DEBUG_RC")) {
                            fprintf(stderr, "[bc250-rc] SeqParam HEVC: bits_per_second=%u pct=%u -> target=%u "
                                            "intra_period=%u\n",
                                    seq->bits_per_second, pct, seq_target, seq->intra_period);
                        }
                        hevc_encoder_set_bitrate(c->hevc_enc, seq_target);
                    } else if (getenv("BC250_DEBUG_RC")) {
                        fprintf(stderr, "[bc250-rc] SeqParam HEVC: bits_per_second=0 intra_period=%u\n",
                                seq->intra_period);
                    }
                }
                break;
            case VAEncPictureParameterBufferType:
                if (c->h264_enc && b->size >= sizeof(VAEncPictureParameterBufferH264)) {
                    VAEncPictureParameterBufferH264 *pic = (VAEncPictureParameterBufferH264*)b->data;
                    memcpy(&c->h264_state.pic_param, pic, sizeof(VAEncPictureParameterBufferH264));
                    c->h264_state.has_pic = 1;
                    c->coded_buf_id = pic->coded_buf;
                    if (pic->pic_fields.bits.idr_pic_flag) {
                        h264_encoder_force_idr(c->h264_enc);
                    }
                    int qp = 0;
                    const char *cqp_env = getenv("BC250_CQP");
                    if (cqp_env && *cqp_env) {
                        qp = atoi(cqp_env);
                    } else if (h264_encoder_get_rc_mode(c->h264_enc) == RC_CQP) {
                        qp = pic->pic_init_qp;
                    }
                    if (qp > 0) {
                        h264_encoder_set_qp(c->h264_enc, qp);
                    }
                } else if (c->hevc_enc && b->size >= sizeof(VAEncPictureParameterBufferHEVC)) {
                    VAEncPictureParameterBufferHEVC *pic = (VAEncPictureParameterBufferHEVC*)b->data;
                    memcpy(&c->hevc_state.pic_param, pic, sizeof(VAEncPictureParameterBufferHEVC));
                    c->hevc_state.has_pic = 1;
                    c->coded_buf_id = pic->coded_buf;
                    if (pic->pic_fields.bits.idr_pic_flag || pic->nal_unit_type == 19 || pic->nal_unit_type == 20) {
                        hevc_encoder_set_force_idr(c->hevc_enc);
                    }
                    int qp = 0;
                    const char *cqp_env = getenv("BC250_CQP");
                    if (cqp_env && *cqp_env) {
                        qp = atoi(cqp_env);
                    } else if (hevc_encoder_get_rc_mode(c->hevc_enc) == RC_CQP) {
                        qp = pic->pic_init_qp;
                    }
                    if (qp > 0) {
                        hevc_encoder_set_qp(c->hevc_enc, qp);
                    }
                }
                break;
            case VAEncMiscParameterBufferType:
                if (b->size >= sizeof(VAEncMiscParameterBuffer)) {
                    VAEncMiscParameterBuffer *misc = (VAEncMiscParameterBuffer*)b->data;
                    if (getenv("BC250_DEBUG_RC")) {
                        fprintf(stderr, "[bc250-rc] MiscParam: type=%d\n", (int)misc->type);
                    }
                    if (misc->type == VAEncMiscParameterTypeRateControl && (c->h264_enc || c->hevc_enc)) {
                        VAEncMiscParameterRateControl *rc = (VAEncMiscParameterRateControl*)misc->data;
                        if (getenv("BC250_DEBUG_RC")) {
                            fprintf(stderr, "[bc250-rc] RateControl: bits_per_second=%u target_percentage=%u "
                                            "window_size=%u initial_qp=%u min_qp=%u\n",
                                    rc->bits_per_second, rc->target_percentage,
                                    rc->window_size, rc->initial_qp, rc->min_qp);
                        }
                        unsigned int pct = rc->target_percentage;
                        if (pct == 0 || pct > 100) pct = 100;
                        c->h264_state.rc_target_percentage = pct;
                        uint32_t target_bps = (uint32_t)(((uint64_t)rc->bits_per_second * pct) / 100);
                        if (target_bps == 0) target_bps = rc->bits_per_second;

                        /* ICQ mode: bits_per_second is 0 in VAEncMiscParameterRateControl.
                         * Compute standard target bitrate from resolution and ICQ quality factor
                         * (~4.0 Mbps H.264 / ~2.2 Mbps HEVC at 1080p, matching Intel iGPU). */
                        if (target_bps == 0) {
                            uint32_t w = c->width > 0 ? (uint32_t)c->width : 1920;
                            uint32_t h = c->height > 0 ? (uint32_t)c->height : 1080;
                            double pixel_rate = (double)w * (double)h * 30.0;
                            if (c->h264_enc) {
                                double base_bps = pixel_rate * 0.0643004;
                                uint32_t q = 23; /* standard x264 default CRF is 23 (~5.0 Mbps at 1080p) */
#if defined(VA_CHECK_VERSION) && VA_CHECK_VERSION(1, 1, 0)
                                if (rc->ICQ_quality_factor >= 1 && rc->ICQ_quality_factor <= 51) {
                                    q = rc->ICQ_quality_factor;
                                }
#endif
                                target_bps = (uint32_t)(base_bps * pow(2.0, (23.0 - (double)q) / 6.0));
                                /* x264 does real constant quality; the bitrate above
                                 * is only what the compute encoder falls back on. */
                                h264_encoder_set_icq_quality(c->h264_enc, (int)q);
                            } else if (c->hevc_enc) {
                                double base_bps = pixel_rate * 0.0353652;
                                uint32_t q = 25;
#if defined(VA_CHECK_VERSION) && VA_CHECK_VERSION(1, 1, 0)
                                if (rc->ICQ_quality_factor >= 1 && rc->ICQ_quality_factor <= 51) {
                                    q = rc->ICQ_quality_factor;
                                }
#endif
                                target_bps = (uint32_t)(base_bps * pow(2.0, (25.0 - (double)q) / 6.0));
                            }
                        }

                        if (c->h264_enc) {
                            if (rc->bits_per_second > 0)
                                h264_encoder_set_icq_quality(c->h264_enc, 0);
                            if (target_bps > 0) {
                                h264_encoder_set_bitrate(c->h264_enc, target_bps);
                                bool cbr_intent = (rc->bits_per_second > 0) &&
                                                  (rc->target_percentage == 100) &&
                                                  !rc->rc_flags.bits.disable_bit_stuffing;
                                h264_encoder_set_cbr_intent(c->h264_enc, cbr_intent);
                            }
                            if (rc->initial_qp > 0) {
                                h264_encoder_set_qp(c->h264_enc, rc->initial_qp);
                            }
                        } else if (c->hevc_enc) {
                            if (target_bps > 0) {
                                hevc_encoder_set_bitrate(c->hevc_enc, target_bps);
                                bool cbr_intent = (rc->bits_per_second > 0) &&
                                                  (rc->target_percentage == 100) &&
                                                  !rc->rc_flags.bits.disable_bit_stuffing;
                                hevc_encoder_set_cbr_intent(c->hevc_enc, cbr_intent);
                            }
                            if (rc->initial_qp > 0) {
                                hevc_encoder_set_qp(c->hevc_enc, rc->initial_qp);
                            }
                        }
                    } else if (misc->type == VAEncMiscParameterTypeFrameRate && (c->h264_enc || c->hevc_enc)) {
                        VAEncMiscParameterFrameRate *fr = (VAEncMiscParameterFrameRate*)misc->data;
                        uint32_t num = fr->framerate & 0xFFFF;
                        uint32_t den = (fr->framerate >> 16) & 0xFFFF;
                        if (den == 0) den = 1;
                        if (num > 0) {
                            if (c->h264_enc) {
                                h264_encoder_set_fps(c->h264_enc, num / den);
                            } else if (c->hevc_enc) {
                                hevc_encoder_set_fps(c->hevc_enc, num / den);
                            }
                        }
                    } else if (misc->type == VAEncMiscParameterTypeQualityLevel && (c->h264_enc || c->hevc_enc)) {
                        VAEncMiscParameterBufferQualityLevel *ql = (VAEncMiscParameterBufferQualityLevel*)misc->data;
                        uint32_t level = ql->quality_level;
                        if (level < 1) level = 1;
                        if (level > 7) level = 7;
                        if (c->h264_enc) {
                            h264_encoder_set_quality_level(c->h264_enc, level);
                        } else if (c->hevc_enc) {
                            hevc_encoder_set_quality_level(c->hevc_enc, level);
                        }
                    } else if (misc->type == VAEncMiscParameterTypeMaxFrameSize && (c->h264_enc || c->hevc_enc)) {
                        VAEncMiscParameterBufferMaxFrameSize *mfs = (VAEncMiscParameterBufferMaxFrameSize*)misc->data;
                        if (c->h264_enc) {
                            h264_encoder_set_max_frame_size(c->h264_enc, mfs->max_frame_size);
                        } else if (c->hevc_enc) {
                            hevc_encoder_set_max_frame_size(c->hevc_enc, mfs->max_frame_size);
                        }
                    }
                }
                break;
            case VAEncSliceParameterBufferType:
                if (c->h264_enc && b->size >= sizeof(VAEncSliceParameterBufferH264)) {
                    memcpy(&c->h264_state.slice_param, b->data, sizeof(VAEncSliceParameterBufferH264));
                    c->h264_state.has_slice = 1;
                } else if (c->hevc_enc && b->size >= sizeof(VAEncSliceParameterBufferHEVC)) {
                    memcpy(&c->hevc_state.slice_param, b->data, sizeof(VAEncSliceParameterBufferHEVC));
                    c->hevc_state.has_slice = 1;
                    VAEncSliceParameterBufferHEVC *slice = &c->hevc_state.slice_param;
                    if (slice->slice_type == 2) {
                        hevc_encoder_set_force_idr(c->hevc_enc);
                    }
                }
                break;
            case VAEncPackedHeaderParameterBufferType:
            case VAEncPackedHeaderDataBufferType:
                /* Explicitly ignored: see bc250_GetConfigAttributes comment on
                 * VA_ENC_PACKED_HEADER_NONE. We emit our own headers. */
                break;

            default:
                break;
        }
    }
    DRIVER_UNLOCK(data);
    return VA_STATUS_SUCCESS;
}

/* BC250_PIPELINE=1 defers a frame's CPU entropy coding so that the next
 * frame's GPU work can overlap it.
 *
 * 🛑 MEASURED A NET LOSS - see docs/DEVLOG.md 24. Kept only because the
 * submit/finish split it needs is independently useful and is byte-exact when
 * this is off. Do not enable it on the strength of the theory; the theory was
 * measured and it did not hold:
 *
 *   - fps 64.27 -> 65.56 (+2%), BELOW this project's ~2.5% noise floor
 *   - PSNR 42.96 -> 37.01 dB avg, 39.69 -> 28.86 dB on the worst frame
 *   - mean end_sync_ms 4.450 -> 4.463, i.e. UNCHANGED: the overlap this
 *     exists to create provably never happened, so the +2% is not even from
 *     pipelining
 *
 * The overlap requires the VA-API client to have two frames in flight (submit
 * N+1 before reading N's coded buffer). Whether ffmpeg/Sunshine here ever does
 * is unresolved. The quality loss appears even WITHOUT overlap, which points at
 * an unexplained encoder/decoder reference mismatch in the deferred path -
 * root-cause that before trusting this knob for anything. */
static bool bc250_pipeline_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *e = getenv("BC250_PIPELINE");
        enabled = (e && (strcmp(e, "1") == 0 || strcmp(e, "true") == 0)) ? 1 : 0;
        if (enabled)
            fprintf(stderr, "[bc250] WARNING: BC250_PIPELINE=1 is EXPERIMENTAL and was "
                            "measured as a net loss (~6dB PSNR for +2%% fps, which is "
                            "inside the noise floor). See docs/DEVLOG.md 24.\n");
    }
    return enabled == 1;
}

/* Complete the one in-flight frame, if any: wait for its GPU work, read it
 * back, entropy-code it, and fill in ITS coded buffer's segment header (not
 * whatever buffer the context has moved on to). Safe to call when nothing is
 * pending. */
static void bc250_finish_pending_frame(bc250_driver_data *data, bc250_context *c) {
    if (!c->has_pending_frame) return;

    /* Clear the flag first: every exit path below must leave nothing pending,
     * or a later call would wait a second time on an already-consumed fence
     * and re-encode stale staging data as a fresh frame. */
    h264_pending_frame_t pending = c->pending_frame;
    VABufferID buf_id = c->pending_coded_buf_id;
    c->has_pending_frame = false;

    if (!VALID_ID(buf_id, MAX_BUFFERS) || !data->buffers[buf_id].allocated) return;

    bc250_buffer *coded_buf = &data->buffers[buf_id];
    uint8_t *dest = ((uint8_t *)coded_buf->data) + sizeof(VACodedBufferSegment);
    size_t max_payload = (size_t)coded_buf->size * coded_buf->num_elements;

    int written = h264_encoder_finish_frame(c->h264_enc, &data->gpu,
                                            dest, max_payload, &pending);
    if (written > 0) {
        VACodedBufferSegment *seg = (VACodedBufferSegment *)coded_buf->data;
        seg->size = (unsigned int)written;
        seg->bit_offset = 0;
        seg->status = 0;
        uint32_t max_bits = h264_encoder_get_max_frame_size(c->h264_enc);
        if (max_bits > 0 && ((uint64_t)written * 8) > max_bits) {
            seg->status |= VA_CODED_BUF_STATUS_FRAME_SIZE_OVERFLOW;
        }
        seg->reserved = 0;
        seg->buf = dest;
        seg->next = NULL;
    }
}

VAStatus bc250_EndPicture(VADriverContextP ctx, VAContextID context) {
    bc250_driver_data *data = get_driver_data(ctx);
    if (!data) return VA_STATUS_ERROR_INVALID_CONTEXT;
    DRIVER_LOCK(data);
    if (!VALID_ID(context, MAX_CONTEXTS) || !data->contexts[context].allocated) {
        DRIVER_UNLOCK(data);
        return VA_STATUS_ERROR_INVALID_CONTEXT;
    }

    bc250_context *c = &data->contexts[context];
    if (!VALID_ID(c->current_render_target, MAX_SURFACES) || !data->surfaces[c->current_render_target].allocated ||
        data->surfaces[c->current_render_target].pending_destroy) {
        DRIVER_UNLOCK(data);
        return VA_STATUS_ERROR_INVALID_SURFACE;
    }
    bc250_surface *surf = &data->surfaces[c->current_render_target];

    if (c->vpp) {
        if (!c->vpp_state.has_source
            || !VALID_ID(c->vpp_state.source, MAX_SURFACES)
            || !data->surfaces[c->vpp_state.source].allocated
            || data->surfaces[c->vpp_state.source].pending_destroy) {
            DRIVER_UNLOCK(data);
            return VA_STATUS_ERROR_INVALID_SURFACE;
        }
        wait_decoded(data, c->vpp_state.source);
        bc250_surface *from = &data->surfaces[c->vpp_state.source];

        /* Whole surface unless the caller named a rectangle. */
        int sr[4] = { 0, 0, from->width, from->height };
        int dr[4] = { 0, 0, surf->width, surf->height };
        if (c->vpp_state.has_src_rect) {
            sr[0] = c->vpp_state.src_rect.x; sr[1] = c->vpp_state.src_rect.y;
            sr[2] = c->vpp_state.src_rect.width;
            sr[3] = c->vpp_state.src_rect.height;
        }
        if (c->vpp_state.has_dst_rect) {
            dr[0] = c->vpp_state.dst_rect.x; dr[1] = c->vpp_state.dst_rect.y;
            dr[2] = c->vpp_state.dst_rect.width;
            dr[3] = c->vpp_state.dst_rect.height;
        }
        if (sr[0] < 0 || sr[1] < 0 || dr[0] < 0 || dr[1] < 0
            || sr[0] + sr[2] > from->width || sr[1] + sr[3] > from->height
            || dr[0] + dr[2] > surf->width || dr[1] + dr[3] > surf->height) {
            DRIVER_UNLOCK(data);
            return VA_STATUS_ERROR_INVALID_PARAMETER;
        }

        const int rc = gpu_compute_video_proc(&data->gpu, &from->image, sr,
                                              &surf->image, dr);
        c->vpp_state.has_source = 0;
        DRIVER_UNLOCK(data);
        return rc == 0 ? VA_STATUS_SUCCESS : VA_STATUS_ERROR_OPERATION_FAILED;
    }

    if (c->h265_dec) {
        /* What can be refused is refused now, while the application waits
         * for the answer; the decoding itself happens after this returns,
         * on the context's own thread. The surface stays pinned and marked
         * pending until it is done - see bc250_decode_finished(). */
        const VAStatus chk = bc250_hevc_dec_check(c);
        if (chk != VA_STATUS_SUCCESS) {
            DRIVER_UNLOCK(data);
            return chk;
        }
        VASurfaceID target = c->current_render_target;
        struct bc250_hevc_job *job = bc250_hevc_dec_take(c, target, surf->image,
                                                         surf->memory);
        if (!job) {
            DRIVER_UNLOCK(data);
            return VA_STATUS_ERROR_ALLOCATION_FAILED;
        }
        surf->ref_count++;
        surf->decode_pending++;
        surf->decode_status = VA_STATUS_SUCCESS;
        DRIVER_UNLOCK(data);
        return bc250_hevc_dec_submit(data, c, job);
    }

    if (c->h264_dec) {
        /* Same shape as the synchronous encode below: pin the surface, drop
         * the lock for the milliseconds of CPU work, take it back. */
        VASurfaceID target = c->current_render_target;
        gpu_image_t img = surf->image;
        gpu_memory_t memo = surf->memory;
        surf->ref_count++;
        DRIVER_UNLOCK(data);

        VAStatus st = bc250_dec_decode(c, img, memo);

        DRIVER_LOCK(data);
        bc250_surface_unref(data, target);
        if (VALID_ID(target, MAX_SURFACES) && data->surfaces[target].allocated)
            data->surfaces[target].image.current_layout = VK_IMAGE_LAYOUT_GENERAL;
        DRIVER_UNLOCK(data);
        return st;
    }

    if ((c->h264_enc || c->hevc_enc) && VALID_ID(c->coded_buf_id, MAX_BUFFERS) && data->buffers[c->coded_buf_id].allocated) {
        bc250_buffer *coded_buf = &data->buffers[c->coded_buf_id];
        uint8_t *dest = ((uint8_t *)coded_buf->data) + sizeof(VACodedBufferSegment);
        size_t max_payload = (size_t)coded_buf->size * coded_buf->num_elements;

        /* This render target's surface memory may have been written most
         * recently by a completely different GPU API context - e.g.
         * Sunshine's own OpenGL rendering into it via its EGL/GL import of
         * the dma-buf bc250_ExportSurfaceHandle() exported for this same
         * surface (see gpu_compute_wait_for_image_ready()'s doc comment in
         * gpu_compute.h for the full story). Queue an explicit GPU-side wait
         * for that write before the encode dispatch below reads the surface.
         * For surfaces not exported via dma-buf (e.g. standard FFmpeg hwupload),
         * skip this to avoid unnecessary syscall and error overhead. */
        if (surf->is_exported) {
            gpu_compute_wait_for_image_ready(&data->gpu, surf->memory);
        }

        int written = -1;
        gpu_compute_debug_dump_real_input(&data->gpu, &surf->image, surf->memory, surf->width, surf->height);
        if (c->h264_enc && bc250_pipeline_enabled() && !h264_encoder_uses_x264(c->h264_enc)) {
            /* Pipelined: submit THIS frame's GPU work first, then finish the
             * PREVIOUS frame on the CPU. */
            h264_pending_frame_t just_submitted;
            bool submitted = (h264_encoder_submit_frame_ext(c->h264_enc, &data->gpu,
                                                            surf->image, surf->memory, &just_submitted) == 0);

            bc250_finish_pending_frame(data, c);

            if (submitted) {
                c->has_pending_frame = true;
                c->pending_frame = just_submitted;
                c->pending_coded_buf_id = c->coded_buf_id;
            }
        } else if (c->h264_enc) {
            /* Synchronous H.264 encode: drop DRIVER_LOCK so concurrent FFmpeg
             * filter/hwupload threads can derive/map/unmap next frames without
             * stalling on the 10-15ms encoding duration. Pin surface ref_count
             * to guarantee lifetime across the unlocked section. */
            VASurfaceID target_surf_id = c->current_render_target;
            surf->ref_count++;
            DRIVER_UNLOCK(data);

            written = h264_encoder_encode_frame_ext(c->h264_enc, &data->gpu, surf->image, surf->memory, dest, max_payload);

            DRIVER_LOCK(data);
            bc250_surface_unref(data, target_surf_id);
        } else if (c->hevc_enc) {
            /* Synchronous HEVC encode: drop DRIVER_LOCK with surface pinned */
            VASurfaceID target_surf_id = c->current_render_target;
            surf->ref_count++;
            DRIVER_UNLOCK(data);

            written = hevc_encoder_encode_frame(c->hevc_enc, &data->gpu, surf->image, surf->memory, dest, max_payload);

            DRIVER_LOCK(data);
            bc250_surface_unref(data, target_surf_id);
        }

        if (written > 0 && VALID_ID(c->coded_buf_id, MAX_BUFFERS) && data->buffers[c->coded_buf_id].allocated) {
            coded_buf = &data->buffers[c->coded_buf_id];
            VACodedBufferSegment *seg = (VACodedBufferSegment *)coded_buf->data;
            seg->size = (unsigned int)written;
            seg->bit_offset = 0;
            seg->status = 0;
            uint32_t max_bits = 0;
            if (c->h264_enc) max_bits = h264_encoder_get_max_frame_size(c->h264_enc);
            else if (c->hevc_enc) max_bits = hevc_encoder_get_max_frame_size(c->hevc_enc);
            if (max_bits > 0 && ((uint64_t)written * 8) > max_bits) {
                seg->status |= VA_CODED_BUF_STATUS_FRAME_SIZE_OVERFLOW;
            }
            seg->reserved = 0;
            seg->buf = dest;
            seg->next = NULL;
        }
    } else {
        if (surf->is_exported) {
            gpu_compute_wait_for_image_ready(&data->gpu, surf->memory);
        }
        gpu_compute_begin_picture(&data->gpu, surf->image);
        gpu_compute_dispatch_encode(&data->gpu, surf->image, c->width, c->height, 26, 0, 1);
        gpu_compute_end_picture(&data->gpu);
    }

    /* gpu_compute_dispatch_encode() (called above, either directly or via
     * h264_encoder_encode_frame()/hevc_encoder_encode_frame()) always
     * transitions the render target's image layout to VK_IMAGE_LAYOUT_GENERAL.
     * It receives gpu_image_t by value, so that transition only affects its
     * local copy -- surf here is a real pointer into data->surfaces[], so we
     * persist the real post-encode layout onto the surface's stored image
     * state ourselves. */
    if (VALID_ID(c->current_render_target, MAX_SURFACES) && data->surfaces[c->current_render_target].allocated) {
        data->surfaces[c->current_render_target].image.current_layout = VK_IMAGE_LAYOUT_GENERAL;
    }

    DRIVER_UNLOCK(data);
    return VA_STATUS_SUCCESS;
}

VAStatus bc250_SyncSurface(VADriverContextP ctx, VASurfaceID render_target) {
    bc250_driver_data *data = get_driver_data(ctx);
    if (!data) return VA_STATUS_ERROR_INVALID_CONTEXT;

    DRIVER_LOCK(data);
    if (!VALID_ID(render_target, MAX_SURFACES) || !data->surfaces[render_target].allocated ||
        data->surfaces[render_target].pending_destroy) {
        DRIVER_UNLOCK(data);
        return VA_STATUS_ERROR_INVALID_SURFACE;
    }
    if (bc250_pipeline_enabled()) {
        for (int i = 0; i < MAX_CONTEXTS; i++) {
            if (data->contexts[i].allocated && data->contexts[i].has_pending_frame)
                bc250_finish_pending_frame(data, &data->contexts[i]);
        }
    }
    wait_decoded(data, render_target);
    const VAStatus decoded = data->surfaces[render_target].decode_status;
    int slot = gpu_compute_submitted_slot(&data->gpu);
    DRIVER_UNLOCK(data);

    /* Unlocked wait for GPU fence: prevents blocking concurrent encoder_thread
     * actions while filter_thread waits on GPU completion. */
    if (slot >= 0) {
        int sync_res = gpu_compute_sync_slot(&data->gpu, slot);
        if (sync_res != 0) return VA_STATUS_ERROR_OPERATION_FAILED;
    }
    /* A decode that failed after vaEndPicture had already said yes. */
    return decoded == VA_STATUS_SUCCESS ? VA_STATUS_SUCCESS
                                        : VA_STATUS_ERROR_DECODING_ERROR;
}

VAStatus bc250_QuerySurfaceStatus(VADriverContextP ctx, VASurfaceID render_target, VASurfaceStatus *status) {
    if (!status) return VA_STATUS_ERROR_INVALID_PARAMETER;
    *status = VASurfaceReady;
    bc250_driver_data *data = get_driver_data(ctx);
    if (!data) return VA_STATUS_SUCCESS;
    DRIVER_LOCK(data);
    if (VALID_ID(render_target, MAX_SURFACES) && data->surfaces[render_target].allocated
        && data->surfaces[render_target].decode_pending > 0)
        *status = VASurfaceRendering;
    DRIVER_UNLOCK(data);
    return VA_STATUS_SUCCESS;
}

VAStatus bc250_QueryImageFormats(VADriverContextP ctx, VAImageFormat *format_list, int *num_formats) {
    (void)ctx;
    if (!num_formats) return VA_STATUS_ERROR_INVALID_PARAMETER;

    if (!format_list) {
        *num_formats = 3;
        return VA_STATUS_SUCCESS;
    }

    int i = 0;
    format_list[i].fourcc = VA_FOURCC_NV12;
    format_list[i].byte_order = VA_LSB_FIRST;
    format_list[i].bits_per_pixel = 12;
    i++;

    format_list[i].fourcc = VA_FOURCC_P010;
    format_list[i].byte_order = VA_LSB_FIRST;
    format_list[i].bits_per_pixel = 24;
    i++;

    format_list[i].fourcc = VA_FOURCC_RGBA;
    format_list[i].byte_order = VA_LSB_FIRST;
    format_list[i].bits_per_pixel = 32;
    i++;

    *num_formats = i;
    return VA_STATUS_SUCCESS;
}

VAStatus bc250_CreateImage(VADriverContextP ctx, VAImageFormat *format, int width, int height, VAImage *image) {
    bc250_driver_data *data = get_driver_data(ctx);
    if (!data || !format || !image) return VA_STATUS_ERROR_INVALID_PARAMETER;

    DRIVER_LOCK(data);
    for (int i = 0; i < MAX_IMAGES; i++) {
        if (!data->images[i].allocated) {
            bc250_image *img = &data->images[i];
            memset(img, 0, sizeof(*img));
            img->allocated = 1;
            /* Not derived from any surface unless bc250_DeriveImage() below
             * says otherwise. memset() above already zeroed this field, but
             * 0 is a valid VASurfaceID (surface slot 0) - make the "no
             * surface" state unambiguous instead of relying on that. */
            img->surface_id = VA_INVALID_SURFACE;

            image->image_id = i;
            image->format = *format;
            image->width = width;
            image->height = height;

            if (format->fourcc == VA_FOURCC_NV12) {
                image->num_planes = 2;
                image->pitches[0] = width;
                image->offsets[0] = 0;
                image->data_size = width * height * 3 / 2;
                image->pitches[1] = width;
                image->offsets[1] = width * height;
            } else if (format->fourcc == VA_FOURCC_P010) {
                /* The same shape with two bytes a sample. A derived image
                 * overwrites all of this with the real Vulkan layout; a
                 * plain vaCreateImage keeps it. */
                image->num_planes = 2;
                image->pitches[0] = width * 2;
                image->offsets[0] = 0;
                image->data_size = width * height * 3;
                image->pitches[1] = width * 2;
                image->offsets[1] = width * height * 2;
            } else {
                image->num_planes = 1;
                image->pitches[0] = width * 4;
                image->offsets[0] = 0;
                image->data_size = width * height * 4;
            }

            VABufferID buf_id;
            VAStatus buf_status = bc250_CreateBuffer(ctx, 0, VAImageBufferType, image->data_size, 1, NULL, &buf_id);
            if (buf_status != VA_STATUS_SUCCESS) {
                /* Roll back: without this, buf_id is left uninitialized and
                 * gets stored as img->buffer_id / image->buf. A later
                 * vaDestroyImage() would then call bc250_DestroyBuffer() on
                 * that garbage id, which - if it happens to fall in range
                 * and alias a live, unrelated buffer slot - would corrupt or
                 * free memory that belongs to something else entirely. */
                img->allocated = 0;
                DRIVER_UNLOCK(data);
                return buf_status;
            }
            image->buf = buf_id;
            img->image = *image;
            img->buffer_id = buf_id;

            DRIVER_UNLOCK(data);
            return VA_STATUS_SUCCESS;
        }
    }
    DRIVER_UNLOCK(data);
    return VA_STATUS_ERROR_MAX_NUM_EXCEEDED;
}

VAStatus bc250_DestroyImage(VADriverContextP ctx, VAImageID image) {
    bc250_driver_data *data = get_driver_data(ctx);
    if (!data) return VA_STATUS_ERROR_INVALID_IMAGE;
    DRIVER_LOCK(data);
    if (!VALID_ID(image, MAX_IMAGES) || !data->images[image].allocated) {
        DRIVER_UNLOCK(data);
        return VA_STATUS_ERROR_INVALID_IMAGE;
    }

    bc250_DestroyBuffer(ctx, data->images[image].buffer_id);
    data->images[image].allocated = 0;
    DRIVER_UNLOCK(data);
    return VA_STATUS_SUCCESS;
}

VAStatus bc250_DeriveImage(VADriverContextP ctx, VASurfaceID surface, VAImage *image) {
    bc250_driver_data *data = get_driver_data(ctx);
    if (!data) return VA_STATUS_ERROR_INVALID_CONTEXT;
    /* A surface with pending_destroy set has already been handed back to
     * vaDestroySurfaces() by the application - it must be rejected here
     * exactly like any other invalid surface, even if its Vulkan
     * resources happen to still be alive internally pending an earlier
     * derived image's teardown (see bc250_surface.pending_destroy). */
    DRIVER_LOCK(data);
    if (!VALID_ID(surface, MAX_SURFACES) || !data->surfaces[surface].allocated ||
        data->surfaces[surface].pending_destroy || !image) {
        DRIVER_UNLOCK(data);
        return VA_STATUS_ERROR_INVALID_SURFACE;
    }
    wait_decoded(data, surface);
    bc250_surface *surf = &data->surfaces[surface];

    const int ten_bit = surf->image.format == GPU_IMAGE_P010;
    VAImageFormat fmt = {
        .fourcc = ten_bit ? VA_FOURCC_P010 : VA_FOURCC_NV12,
        .byte_order = VA_LSB_FIRST,
        .bits_per_pixel = ten_bit ? 24 : 12
    };
    VAStatus status = bc250_CreateImage(ctx, &fmt, surf->width, surf->height, image);
    if (status != VA_STATUS_SUCCESS) {
        DRIVER_UNLOCK(data);
        return status;
    }

    bc250_image *img = &data->images[image->image_id];
    bc250_buffer *buf = &data->buffers[img->buffer_id];

    /* bc250_CreateImage() above just filled in `image`/`img->image` with a
     * naive, tightly-packed pitches/offsets/data_size (pitches[0]=width,
     * offsets[1]=width*height, data_size=width*height*3/2) - correct for a
     * plain vaCreateImage() whose VAImageBufferType buffer is a private,
     * non-aliased malloc(). But below, this derived image's buffer is about
     * to be replaced with a *direct mapping of the surface's own real
     * Vulkan memory* (buf->data = mapped). A consumer of this VAImage (e.g.
     * ffmpeg's hwupload -> av_frame_copy -> av_image_copy, which does a
     * plain memcpy straight into this buffer using exactly these
     * pitches/offsets/data_size) will address that real memory using the
     * naive numbers, which do not account for the real per-row pitch
     * padding and inter-plane alignment gap that
     * gpu_compute_create_image() actually bound Y/UV to (confirmed
     * on-hardware: 854x480 real data_size=737280B vs naive 614880B;
     * 1920x1080 real=3317760B vs naive=3110400B; the two happen to coincide
     * exactly at 1280x720 because 1280 and 640*2=1280 are already multiples
     * of this hardware's apparent 256-byte row-pitch alignment, so the
     * mismatch is resolution-dependent, not universal). Overwrite the
     * geometry with the real, Vulkan-derived layout (the same
     * vkGetImageSubresourceLayout()/vkGetImageMemoryRequirements() math
     * gpu_compute_upload_nv12()/gpu_compute_download_nv12() already use to
     * address this same memory) before handing it back, so the caller's
     * view of this buffer always matches the real allocation exactly. */
    gpu_nv12_layout_t real_layout;
    if (gpu_compute_get_nv12_layout(&data->gpu, &surf->image, surf->memory, &real_layout) == 0) {
        image->pitches[0] = real_layout.y_pitch;
        image->offsets[0] = (unsigned int)real_layout.y_offset;
        image->pitches[1] = real_layout.uv_pitch;
        image->offsets[1] = (unsigned int)real_layout.uv_offset;
        image->data_size = (unsigned int)real_layout.total_size;
        img->image = *image;
    }

    if (buf && surf->memory.memory) {
        void *mapped = surf->mapped_ptr;
        int needs_unmap = 0;
        if (!mapped) {
            if (vkMapMemory(data->gpu.device, surf->memory.memory, 0, surf->memory.size, 0, &mapped) == VK_SUCCESS) {
                needs_unmap = 1;
            }
        }
        if (mapped) {
            if (buf->data) free(buf->data);
            buf->data = mapped;
            buf->mapped = 1;
            buf->is_derived = 1;
            buf->gpu_mem = needs_unmap ? surf->memory.memory : VK_NULL_HANDLE;
            /* This derived image now aliases the surface's own Vulkan
             * memory directly (buf->data / buf->gpu_mem above). Take a
             * reference on the surface so vaDestroySurfaces() cannot free
             * that memory out from under this still-live mapping - see
             * bc250_surface_unref() / bc250_DestroyBuffer() for the
             * matching release. This is the fix for the use-after-free:
             * previously ref_count was only ever set to 1 at
             * vaCreateSurfaces() and never incremented here, so a
             * vaDestroySurfaces() call while a derived image was still
             * alive would free the surface's VkImage/VkDeviceMemory
             * immediately, and a later vaDestroyImage() -> vkUnmapMemory()
             * on that freed VkDeviceMemory handle would segfault
             * (confirmed on-hardware: radv_UnmapMemory2 SIGSEGV via
             * bc250_DestroyBuffer at va_backend.c, called from
             * bc250_DestroyImage). */
            surf->ref_count++;
            buf->derived_surface = surface;
            img->surface_id = surface;
        }
    }
    DRIVER_UNLOCK(data);
    return VA_STATUS_SUCCESS;
}

VAStatus bc250_GetImage(VADriverContextP ctx, VASurfaceID surface, int x, int y, unsigned int width, unsigned int height, VAImageID image) {
    (void)x; (void)y; (void)width; (void)height;
    bc250_driver_data *data = get_driver_data(ctx);
    if (!data) return VA_STATUS_ERROR_INVALID_CONTEXT;
    DRIVER_LOCK(data);
    if (!VALID_ID(surface, MAX_SURFACES) || !data->surfaces[surface].allocated ||
        data->surfaces[surface].pending_destroy) {
        DRIVER_UNLOCK(data);
        return VA_STATUS_ERROR_INVALID_SURFACE;
    }
    if (!VALID_ID(image, MAX_IMAGES) || !data->images[image].allocated) {
        DRIVER_UNLOCK(data);
        return VA_STATUS_ERROR_INVALID_IMAGE;
    }

    wait_decoded(data, surface);
    bc250_surface *surf = &data->surfaces[surface];
    bc250_image *img = &data->images[image];
    bc250_buffer *buf = &data->buffers[img->buffer_id];

    if (buf && buf->data && surf->memory.memory) {
        uint8_t *dst_y = (uint8_t *)buf->data + img->image.offsets[0];
        uint8_t *dst_uv = (uint8_t *)buf->data + img->image.offsets[1];
        int y_pitch = img->image.pitches[0] > 0 ? (int)img->image.pitches[0] : surf->width;
        int uv_pitch = img->image.pitches[1] > 0 ? (int)img->image.pitches[1] : surf->width;

        /* Copy extent must be the image's OWN allocated width/height
         * (img->image.width/height - what bc250_CreateImage() actually
         * sized buf->data for), not the surface's: surf->width/height is
         * this driver's internal macroblock-padded encode size (e.g. 1088
         * for a 1080-tall frame), which is >= the real display size a
         * plain vaCreateImage()+vaGetImage() caller asked for. Using
         * surf->height here walked this copy past the end of buf->data's
         * real allocation (confirmed on-hardware via gdb: SIGSEGV in the
         * UV-plane memcpy at r=540 for a 1080-tall image, where
         * height/2=544 from surf->height=1088 overran a buffer sized for
         * only 1080/2=540 UV rows). bc250_DeriveImage() is unaffected -
         * there, img->image.width/height are set to surf->width/height by
         * construction (see bc250_DeriveImage() above), so this is the
         * same value in that case, not a behavior change. */
        int copy_width = img->image.width > 0 ? (int)img->image.width : surf->width;
        int copy_height = img->image.height > 0 ? (int)img->image.height : surf->height;
        if (copy_width > surf->width) copy_width = surf->width;
        if (copy_height > surf->height) copy_height = surf->height;

        gpu_compute_download_nv12(&data->gpu, &surf->image, surf->memory,
                                  dst_y, y_pitch,
                                  dst_uv, uv_pitch,
                                  copy_width, copy_height);
    }
    DRIVER_UNLOCK(data);
    return VA_STATUS_SUCCESS;
}

VAStatus bc250_PutImage(VADriverContextP ctx, VASurfaceID surface, VAImageID image, int src_x, int req_src_y, unsigned int src_width, unsigned int src_height, int dest_x, int dest_y, unsigned int dest_width, unsigned int dest_height) {
    (void)src_x; (void)req_src_y; (void)src_width; (void)src_height;
    (void)dest_x; (void)dest_y; (void)dest_width; (void)dest_height;
    bc250_driver_data *data = get_driver_data(ctx);
    if (!data) return VA_STATUS_ERROR_INVALID_CONTEXT;
    DRIVER_LOCK(data);
    if (!VALID_ID(surface, MAX_SURFACES) || !data->surfaces[surface].allocated ||
        data->surfaces[surface].pending_destroy) {
        DRIVER_UNLOCK(data);
        return VA_STATUS_ERROR_INVALID_SURFACE;
    }
    if (!VALID_ID(image, MAX_IMAGES) || !data->images[image].allocated) {
        DRIVER_UNLOCK(data);
        return VA_STATUS_ERROR_INVALID_IMAGE;
    }

    wait_decoded(data, surface);
    bc250_surface *surf = &data->surfaces[surface];
    bc250_image *img = &data->images[image];
    bc250_buffer *buf = &data->buffers[img->buffer_id];

    if (buf && buf->data && surf->memory.memory) {
        const uint8_t *src_y = (const uint8_t *)buf->data + img->image.offsets[0];
        const uint8_t *src_uv = (const uint8_t *)buf->data + img->image.offsets[1];
        int y_pitch = img->image.pitches[0] > 0 ? (int)img->image.pitches[0] : surf->width;
        int uv_pitch = img->image.pitches[1] > 0 ? (int)img->image.pitches[1] : surf->width;

        /* Same fix as bc250_GetImage() above, mirrored: the copy extent
         * must be img->image.width/height (what buf->data was actually
         * allocated for), not surf->width/height (this driver's internal
         * macroblock-padded encode size) - otherwise this reads past the
         * end of buf->data whenever the surface's padded height exceeds
         * the image's real height (e.g. 1088 vs 1080). */
        int copy_width = img->image.width > 0 ? (int)img->image.width : surf->width;
        int copy_height = img->image.height > 0 ? (int)img->image.height : surf->height;
        if (copy_width > surf->width) copy_width = surf->width;
        if (copy_height > surf->height) copy_height = surf->height;

        gpu_compute_upload_nv12(&data->gpu, &surf->image, surf->memory,
                                src_y, y_pitch,
                                src_uv, uv_pitch,
                                copy_width, copy_height);
    }
    DRIVER_UNLOCK(data);
    return VA_STATUS_SUCCESS;
}

/* Exports a surface as a real DRM-PRIME/DMA-BUF handle, for zero-copy
 * sharing with an external API - e.g. Sunshine's own GL/EGL import of this
 * driver's encode surfaces for its cursor-overlay/color-conversion
 * pipeline, confirmed on real hardware to call exactly this
 * (`vaExportSurfaceHandle()` -> "the requested function is not
 * implemented" before this was added; see docs/DEVLOG.md §10.5).
 *
 * Only VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2 (VADRMPRIMESurfaceDescriptor)
 * is supported - the older VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME (single
 * object, VASurfaceAttribExternalBuffers) is a different, legacy
 * descriptor shape this driver doesn't produce.
 *
 * Defaults to composed layers (one layer, two planes - real NV12) unless
 * the caller explicitly asks for VA_EXPORT_SURFACE_SEPARATE_LAYERS,
 * matching the convention other real VA-API drivers (Intel's iHD, Mesa's
 * own radeonsi VAAPI driver) use for this same choice. */
VAStatus bc250_ExportSurfaceHandle(VADriverContextP ctx, VASurfaceID surface_id, uint32_t mem_type, uint32_t flags, void *descriptor) {
    bc250_driver_data *data = get_driver_data(ctx);
    if (!data) return VA_STATUS_ERROR_INVALID_CONTEXT;
    DRIVER_LOCK(data);
    if (!VALID_ID(surface_id, MAX_SURFACES) || !data->surfaces[surface_id].allocated ||
        data->surfaces[surface_id].pending_destroy || !descriptor) {
        DRIVER_UNLOCK(data);
        return VA_STATUS_ERROR_INVALID_SURFACE;
    }
    if (mem_type != VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2) {
        DRIVER_UNLOCK(data);
        return VA_STATUS_ERROR_UNSUPPORTED_MEMORY_TYPE;
    }
    wait_decoded(data, surface_id);
    bc250_surface *surf = &data->surfaces[surface_id];

    gpu_nv12_layout_t layout;
    if (gpu_compute_get_nv12_layout(&data->gpu, &surf->image, surf->memory, &layout) != 0) {
        DRIVER_UNLOCK(data);
        return VA_STATUS_ERROR_OPERATION_FAILED;
    }

    int fd;
    if (gpu_compute_export_nv12_dmabuf(&data->gpu, surf->memory, &fd) != 0) {
        /* Most likely VK_KHR_external_memory_fd/VK_EXT_external_memory_dma_buf
         * weren't available at device creation - see bc250_gpu_init(). */
        DRIVER_UNLOCK(data);
        return VA_STATUS_ERROR_UNIMPLEMENTED;
    }

    /* What the surface actually holds, which the importer on the other
     * side of the fd has no other way to learn. */
    const int ten_bit = surf->image.format == GPU_IMAGE_P010;

    VADRMPRIMESurfaceDescriptor *desc = (VADRMPRIMESurfaceDescriptor *)descriptor;
    memset(desc, 0, sizeof(*desc));
    desc->fourcc = ten_bit ? VA_FOURCC_P010 : VA_FOURCC_NV12;
    desc->width = (uint32_t)surf->width;
    desc->height = (uint32_t)surf->height;
    desc->num_objects = 1;
    desc->objects[0].fd = fd;
    desc->objects[0].size = (uint32_t)layout.total_size;
    desc->objects[0].drm_format_modifier = DRM_FORMAT_MOD_LINEAR;

    if (flags & VA_EXPORT_SURFACE_SEPARATE_LAYERS) {
        desc->num_layers = 2;
        desc->layers[0].drm_format = ten_bit ? DRM_FORMAT_R16 : DRM_FORMAT_R8;
        desc->layers[0].num_planes = 1;
        desc->layers[0].object_index[0] = 0;
        desc->layers[0].offset[0] = (uint32_t)layout.y_offset;
        desc->layers[0].pitch[0] = layout.y_pitch;

        desc->layers[1].drm_format = ten_bit ? DRM_FORMAT_GR1616
                                            : DRM_FORMAT_GR88;
        desc->layers[1].num_planes = 1;
        desc->layers[1].object_index[0] = 0;
        desc->layers[1].offset[0] = (uint32_t)layout.uv_offset;
        desc->layers[1].pitch[0] = layout.uv_pitch;
    } else {
        desc->num_layers = 1;
        desc->layers[0].drm_format = ten_bit ? DRM_FORMAT_P010
                                            : DRM_FORMAT_NV12;
        desc->layers[0].num_planes = 2;
        desc->layers[0].object_index[0] = 0;
        desc->layers[0].object_index[1] = 0;
        desc->layers[0].offset[0] = (uint32_t)layout.y_offset;
        desc->layers[0].offset[1] = (uint32_t)layout.uv_offset;
        desc->layers[0].pitch[0] = layout.y_pitch;
        desc->layers[0].pitch[1] = layout.uv_pitch;
    }

    surf->is_exported = 1;
    DRIVER_UNLOCK(data);
    return VA_STATUS_SUCCESS;
}

VAStatus bc250_SetImagePalette(VADriverContextP ctx, VAImageID image, unsigned char *palette) {
    (void)ctx; (void)image; (void)palette;
    return VA_STATUS_ERROR_UNIMPLEMENTED;
}

/* Subpictures are not supported by this driver. Query returns zero available
 * formats (the standard way to report "unsupported"); the rest are stubs to
 * satisfy libva's vtable completeness check. */
VAStatus bc250_QuerySubpictureFormats(VADriverContextP ctx, VAImageFormat *format_list, unsigned int *flags, unsigned int *num_formats) {
    (void)ctx; (void)format_list; (void)flags;
    if (!num_formats) return VA_STATUS_ERROR_INVALID_PARAMETER;
    *num_formats = 0;
    return VA_STATUS_SUCCESS;
}

VAStatus bc250_CreateSubpicture(VADriverContextP ctx, VAImageID image, VASubpictureID *subpicture) {
    (void)ctx; (void)image; (void)subpicture;
    return VA_STATUS_ERROR_UNIMPLEMENTED;
}

VAStatus bc250_DestroySubpicture(VADriverContextP ctx, VASubpictureID subpicture) {
    (void)ctx; (void)subpicture;
    return VA_STATUS_ERROR_UNIMPLEMENTED;
}

VAStatus bc250_SetSubpictureImage(VADriverContextP ctx, VASubpictureID subpicture, VAImageID image) {
    (void)ctx; (void)subpicture; (void)image;
    return VA_STATUS_ERROR_UNIMPLEMENTED;
}

VAStatus bc250_SetSubpictureChromakey(VADriverContextP ctx, VASubpictureID subpicture, unsigned int chromakey_min, unsigned int chromakey_max, unsigned int chromakey_mask) {
    (void)ctx; (void)subpicture; (void)chromakey_min; (void)chromakey_max; (void)chromakey_mask;
    return VA_STATUS_ERROR_UNIMPLEMENTED;
}

VAStatus bc250_SetSubpictureGlobalAlpha(VADriverContextP ctx, VASubpictureID subpicture, float global_alpha) {
    (void)ctx; (void)subpicture; (void)global_alpha;
    return VA_STATUS_ERROR_UNIMPLEMENTED;
}

VAStatus bc250_AssociateSubpicture(VADriverContextP ctx, VASubpictureID subpicture, VASurfaceID *target_surfaces, int num_surfaces,
                                   short src_x, short src_y, unsigned short src_width, unsigned short src_height,
                                   short dest_x, short dest_y, unsigned short dest_width, unsigned short dest_height,
                                   unsigned int flags) {
    (void)ctx; (void)subpicture; (void)target_surfaces; (void)num_surfaces;
    (void)src_x; (void)src_y; (void)src_width; (void)src_height;
    (void)dest_x; (void)dest_y; (void)dest_width; (void)dest_height; (void)flags;
    return VA_STATUS_ERROR_UNIMPLEMENTED;
}

VAStatus bc250_DeassociateSubpicture(VADriverContextP ctx, VASubpictureID subpicture, VASurfaceID *target_surfaces, int num_surfaces) {
    (void)ctx; (void)subpicture; (void)target_surfaces; (void)num_surfaces;
    return VA_STATUS_ERROR_UNIMPLEMENTED;
}

/* Display attributes are not supported. Query/Get report zero/no-op success
 * (the standard way to report "unsupported"); Set is unimplemented since
 * nothing was ever exposed to set. */
VAStatus bc250_QueryDisplayAttributes(VADriverContextP ctx, VADisplayAttribute *attr_list, int *num_attributes) {
    (void)ctx; (void)attr_list;
    if (!num_attributes) return VA_STATUS_ERROR_INVALID_PARAMETER;
    *num_attributes = 0;
    return VA_STATUS_SUCCESS;
}

VAStatus bc250_GetDisplayAttributes(VADriverContextP ctx, VADisplayAttribute *attr_list, int num_attributes) {
    (void)ctx; (void)attr_list; (void)num_attributes;
    return VA_STATUS_SUCCESS;
}

VAStatus bc250_SetDisplayAttributes(VADriverContextP ctx, VADisplayAttribute *attr_list, int num_attributes) {
    (void)ctx; (void)attr_list; (void)num_attributes;
    return VA_STATUS_ERROR_UNIMPLEMENTED;
}

/* None, and that is the honest answer. This driver's post-processing
 * scales and crops; deinterlacing, denoise, sharpening and colour
 * balance are not implemented, and saying otherwise would get them asked
 * for and then refused a frame later. */
VAStatus bc250_QueryVideoProcFilters(VADriverContextP ctx, VAContextID context, VAProcFilterType *filters, unsigned int *num_filters) {
    (void)ctx; (void)context; (void)filters;
    if (!num_filters) return VA_STATUS_ERROR_INVALID_PARAMETER;
    *num_filters = 0;
    return VA_STATUS_SUCCESS;
}

VAStatus bc250_QueryVideoProcFilterCaps(VADriverContextP ctx, VAContextID context, VAProcFilterType type, void *filter_caps, unsigned int *num_filter_caps) {
    (void)ctx; (void)context; (void)type; (void)filter_caps;
    if (!num_filter_caps) return VA_STATUS_ERROR_INVALID_PARAMETER;
    *num_filter_caps = 0;
    return VA_STATUS_SUCCESS;
}

VAStatus bc250_QueryVideoProcPipelineCaps(VADriverContextP ctx, VAContextID context, VABufferID *filters, unsigned int num_filters, VAProcPipelineCaps *pipeline_caps) {
    (void)ctx; (void)context; (void)filters;
    if (!pipeline_caps) return VA_STATUS_ERROR_INVALID_PARAMETER;

    /* The caller reads these through the pointers below, so they outlive
     * the call: static, like every other driver does it. */
    static uint32_t formats[] = { VA_FOURCC_NV12, VA_FOURCC_P010 };
    static VAProcColorStandardType standards[] = { VAProcColorStandardNone };

    memset(pipeline_caps, 0, sizeof(*pipeline_caps));

    /* A filter asked for is a filter this driver said it does not have. */
    if (num_filters > 0) return VA_STATUS_ERROR_UNSUPPORTED_FILTER;

    pipeline_caps->pipeline_flags = 0;
    pipeline_caps->filter_flags = 0;
    pipeline_caps->num_forward_references = 0;
    pipeline_caps->num_backward_references = 0;
    pipeline_caps->input_color_standards = standards;
    pipeline_caps->num_input_color_standards = 1;
    pipeline_caps->output_color_standards = standards;
    pipeline_caps->num_output_color_standards = 1;
    /* Scaling and cropping only. No rotation, no mirroring, no blending:
     * every one of those would be a shader that does not exist. */
    pipeline_caps->rotation_flags = 1u << VA_ROTATION_NONE;
    pipeline_caps->blend_flags = 0;
    pipeline_caps->mirror_flags = VA_MIRROR_NONE;
    pipeline_caps->num_additional_outputs = 0;
    pipeline_caps->num_input_pixel_formats = 2;
    pipeline_caps->input_pixel_format = formats;
    pipeline_caps->num_output_pixel_formats = 2;
    pipeline_caps->output_pixel_format = formats;
    pipeline_caps->max_input_width = BC250_MAX_WIDTH;
    pipeline_caps->max_input_height = BC250_MAX_HEIGHT;
    pipeline_caps->min_input_width = 2;
    pipeline_caps->min_input_height = 2;
    pipeline_caps->max_output_width = BC250_MAX_WIDTH;
    pipeline_caps->max_output_height = BC250_MAX_HEIGHT;
    pipeline_caps->min_output_width = 2;
    pipeline_caps->min_output_height = 2;
    return VA_STATUS_SUCCESS;
}

VAStatus bc250_Terminate(VADriverContextP ctx) {
    bc250_driver_data *data = get_driver_data(ctx);
    if (data) {
        /* ⚠️ Before the lock: the contexts below are destroyed with it
         * held twice over, and a decode thread still finishing a picture
         * could then never take it. */
        for (int i = 0; i < MAX_CONTEXTS; i++) {
            if (data->contexts[i].allocated && data->contexts[i].hevc_async) {
                struct bc250_hevc_async *a = data->contexts[i].hevc_async;
                data->contexts[i].hevc_async = NULL;
                bc250_hevc_async_stop(a);
            }
        }
        DRIVER_LOCK(data);
        /* The VA-API contract expects callers to have destroyed every
         * config/context/buffer/image/surface before vaTerminate(), but a
         * driver should not silently leak GPU memory and heap allocations
         * if a caller doesn't. Free anything still outstanding here, before
         * tearing down the GPU context, by reusing the existing Destroy*
         * paths. Order matters: contexts (which hold encoder/decoder state
         * and reference surfaces/buffers by id, but don't own them) must go
         * before the buffers/images/surfaces they reference; images (which
         * own a buffer each) before the remaining plain buffers; and
         * surfaces last, since gpu_compute_terminate() below invalidates the
         * VkDevice that surface/derived-image teardown still needs. */
        for (int i = 0; i < MAX_CONTEXTS; i++) {
            if (data->contexts[i].allocated) {
                bc250_DestroyContext(ctx, i);
            }
        }
        for (int i = 0; i < MAX_IMAGES; i++) {
            if (data->images[i].allocated) {
                bc250_DestroyImage(ctx, i);
            }
        }
        for (int i = 0; i < MAX_BUFFERS; i++) {
            if (data->buffers[i].allocated) {
                bc250_DestroyBuffer(ctx, i);
            }
        }
        for (int i = 0; i < MAX_SURFACES; i++) {
            if (data->surfaces[i].allocated) {
                VASurfaceID id = (VASurfaceID)i;
                bc250_DestroySurfaces(ctx, &id, 1);
            }
        }

        gpu_compute_terminate(&data->gpu);
        ctx->pDriverData = NULL;
        DRIVER_UNLOCK(data);
        pthread_cond_destroy(&data->idle);
        pthread_mutex_destroy(&data->lock);
        free(data);
    }
    return VA_STATUS_SUCCESS;
}

VAStatus bc250_Initialize(VADriverContextP ctx, int *major_version, int *minor_version) {
    if (!ctx) return VA_STATUS_ERROR_INVALID_CONTEXT;

    bc250_driver_data *data = calloc(1, sizeof(bc250_driver_data));
    if (!data) return VA_STATUS_ERROR_ALLOCATION_FAILED;

    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&data->lock, &attr);
    pthread_mutexattr_destroy(&attr);
    pthread_cond_init(&data->idle, NULL);

    if (gpu_compute_init(&data->gpu) != 0) {
        fprintf(stderr, "[bc250-drv] Failed to initialize Vulkan compute backend!\n");
        pthread_mutex_destroy(&data->lock);
        free(data);
        return VA_STATUS_ERROR_OPERATION_FAILED;
    }

    data->max_width = BC250_MAX_DECODE_SIDE;
    data->max_height = BC250_MAX_DECODE_SIDE;
    ctx->pDriverData = data;
    ctx->str_vendor = "AMD BC-250 Compute VA-API Driver";

#ifdef _OPENMP
    /* OpenMP Thread Pool & Wait Policy Management:
     * When loaded into a host process like Steam (via 32-bit or 64-bit libva),
     * GCC libgomp default behavior is to spin-wait (ACTIVE) across all host logical cores
     * (16 threads on BC-250), consuming 600%+ host CPU between video frames.
     * Force passive waiting and limit default threads to 1 (preserving maximum Zen 2 headroom).
     * Multi-threading can be explicitly enabled by the user via BC250_MAX_CPU_THREADS. */
    setenv("OMP_WAIT_POLICY", "PASSIVE", 1);
    setenv("GOMP_SPINCOUNT", "0", 1);
    const char *max_t = getenv("BC250_MAX_CPU_THREADS");
    if (!max_t) max_t = getenv("BC250_THREADS");
    if (!max_t) max_t = getenv("BC250_CPU_THREADS");
    int def_threads = 4;
    if (max_t && *max_t) {
        int v = atoi(max_t);
        if (v > 0 && v <= 16) def_threads = v;
    }
#if defined(__linux__)
    else if (program_invocation_short_name) {
        if (strcmp(program_invocation_short_name, "sunshine") == 0 ||
            strcmp(program_invocation_short_name, "steam") == 0 ||
            strcmp(program_invocation_short_name, "streaming_client") == 0) {
            def_threads = 2;
        } else if (strcmp(program_invocation_short_name, "wivrn-server") == 0 ||
                   strcmp(program_invocation_short_name, "wivrn") == 0) {
            def_threads = 4;
        } else if (strcmp(program_invocation_short_name, "ffmpeg") == 0) {
            def_threads = 4;
        }
    }
#endif
    omp_set_num_threads(def_threads);
#endif

    /* libva's core vaInitialize() validates these counts and the vtable
     * completeness before returning control to the driver's caller - both
     * are mandatory, not just documentation. */
    ctx->max_profiles = MAX_PROFILES;
    ctx->max_entrypoints = MAX_ENTRYPOINTS;
    ctx->max_attributes = MAX_CONFIG_ATTRIBUTES;
    ctx->max_image_formats = MAX_IMAGE_FORMATS;
    /* libva requires these positive even though we report zero actual
     * subpicture formats / display attributes at query time - they only
     * size libva's internal arrays, they aren't a "supported" flag. */
    ctx->max_subpic_formats = 1;
    ctx->max_display_attributes = 1;

    /* ⚠️ The second vtable. Post-processing does not go in the main one,
     * and a driver that fills in only the main one has libva answering
     * UNIMPLEMENTED on its behalf - which is what happened here, with all
     * three functions sitting written and unreachable.
     *
     * libva allocates it before calling us; guarded anyway, because a
     * null here would be a segfault at driver load rather than a missing
     * feature. */
    if (ctx->vtable_vpp) {
        ctx->vtable_vpp->version = VA_DRIVER_VTABLE_VPP_VERSION;
        ctx->vtable_vpp->vaQueryVideoProcFilters = bc250_QueryVideoProcFilters;
        ctx->vtable_vpp->vaQueryVideoProcFilterCaps = bc250_QueryVideoProcFilterCaps;
        ctx->vtable_vpp->vaQueryVideoProcPipelineCaps = bc250_QueryVideoProcPipelineCaps;
    }

    /* Wire complete vtable */
    ctx->vtable->vaTerminate = bc250_Terminate;
    ctx->vtable->vaQueryConfigProfiles = bc250_QueryConfigProfiles;
    ctx->vtable->vaQueryConfigEntrypoints = bc250_QueryConfigEntrypoints;
    ctx->vtable->vaGetConfigAttributes = bc250_GetConfigAttributes;
    ctx->vtable->vaCreateConfig = bc250_CreateConfig;
    ctx->vtable->vaDestroyConfig = bc250_DestroyConfig;
    ctx->vtable->vaQueryConfigAttributes = bc250_QueryConfigAttributes;
    ctx->vtable->vaCreateSurfaces = bc250_CreateSurfaces;
    ctx->vtable->vaCreateSurfaces2 = bc250_CreateSurfaces2;
    ctx->vtable->vaDestroySurfaces = bc250_DestroySurfaces;
    ctx->vtable->vaCreateContext = bc250_CreateContext;
    ctx->vtable->vaDestroyContext = bc250_DestroyContext;
    ctx->vtable->vaCreateBuffer = bc250_CreateBuffer;
    ctx->vtable->vaBufferSetNumElements = bc250_BufferSetNumElements;
    ctx->vtable->vaMapBuffer = bc250_MapBuffer;
    ctx->vtable->vaUnmapBuffer = bc250_UnmapBuffer;
    ctx->vtable->vaDestroyBuffer = bc250_DestroyBuffer;
    ctx->vtable->vaBeginPicture = bc250_BeginPicture;
    ctx->vtable->vaRenderPicture = bc250_RenderPicture;
    ctx->vtable->vaEndPicture = bc250_EndPicture;
    ctx->vtable->vaSyncSurface = bc250_SyncSurface;
    ctx->vtable->vaQuerySurfaceStatus = bc250_QuerySurfaceStatus;
    ctx->vtable->vaQueryImageFormats = bc250_QueryImageFormats;
    ctx->vtable->vaQuerySurfaceAttributes = bc250_QuerySurfaceAttributes;
    ctx->vtable->vaCreateImage = bc250_CreateImage;
    ctx->vtable->vaDestroyImage = bc250_DestroyImage;
    ctx->vtable->vaDeriveImage = bc250_DeriveImage;
    ctx->vtable->vaGetImage = bc250_GetImage;
    ctx->vtable->vaPutImage = bc250_PutImage;
    ctx->vtable->vaExportSurfaceHandle = bc250_ExportSurfaceHandle;
    ctx->vtable->vaSetImagePalette = bc250_SetImagePalette;
    ctx->vtable->vaQuerySubpictureFormats = bc250_QuerySubpictureFormats;
    ctx->vtable->vaCreateSubpicture = bc250_CreateSubpicture;
    ctx->vtable->vaDestroySubpicture = bc250_DestroySubpicture;
    ctx->vtable->vaSetSubpictureImage = bc250_SetSubpictureImage;
    ctx->vtable->vaSetSubpictureChromakey = bc250_SetSubpictureChromakey;
    ctx->vtable->vaSetSubpictureGlobalAlpha = bc250_SetSubpictureGlobalAlpha;
    ctx->vtable->vaAssociateSubpicture = bc250_AssociateSubpicture;
    ctx->vtable->vaDeassociateSubpicture = bc250_DeassociateSubpicture;
    ctx->vtable->vaQueryDisplayAttributes = bc250_QueryDisplayAttributes;
    ctx->vtable->vaGetDisplayAttributes = bc250_GetDisplayAttributes;
    ctx->vtable->vaSetDisplayAttributes = bc250_SetDisplayAttributes;

    if (major_version) *major_version = VA_MAJOR_VERSION;
    if (minor_version) *minor_version = VA_MINOR_VERSION;

    return VA_STATUS_SUCCESS;
}

VAStatus __vaDriverInit_1_0(VADriverContextP ctx) {
    int major = VA_MAJOR_VERSION;
    int minor = VA_MINOR_VERSION;
    return bc250_Initialize(ctx, &major, &minor);
}

VAStatus __vaDriverInit_0_32(VADriverContextP ctx) {
    int major = VA_MAJOR_VERSION;
    int minor = VA_MINOR_VERSION;
    return bc250_Initialize(ctx, &major, &minor);
}
