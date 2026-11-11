/* bc250-encoding-decoding-fix v0.4.3 - https://github.com/simpmix/bc250-encoding-decoding-fix */
/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * va_backend.h - VA-API Driver Backend Interface for AMD BC-250
 */
#ifndef BC250_VA_BACKEND_H
#define BC250_VA_BACKEND_H

#include <va/va.h>
#include <va/va_backend.h>
#include <va/va_enc_h264.h>
#include <va/va_enc_hevc.h>
#include <va/va_vpp.h>
/* ⚠️ The post-processing entry points live in a vtable of their own,
 * ctx->vtable_vpp, and this is the header that declares it. Without it
 * the three bc250_QueryVideoProc* below compile fine and are never
 * called. */
#include <va/va_backend_vpp.h>
#include <va/va_drmcommon.h>
#include <drm_fourcc.h>
#include <pthread.h>
#include "gpu_compute.h"
#include "encoder_h264.h"
#include "encoder_h265.h"
#include "decoder_h264.h"
#include "decoder_h265.h"

#ifndef VAConfigAttribEncQualityRange
#define VAConfigAttribEncQualityRange 21
#endif

#ifndef VAEncMiscParameterTypeQualityLevel
#define VAEncMiscParameterTypeQualityLevel 6
#endif

#ifndef VAEncMiscParameterTypeMaxFrameSize
#define VAEncMiscParameterTypeMaxFrameSize 4
#endif

#ifndef VA_CODED_BUF_STATUS_FRAME_SIZE_OVERFLOW
#define VA_CODED_BUF_STATUS_FRAME_SIZE_OVERFLOW (1 << 1)
#endif

#define MAX_PROFILES 16
#define MAX_ENTRYPOINTS 16
#define MAX_CONFIG_ATTRIBUTES 16
#define MAX_IMAGE_FORMATS 8
#define MAX_CONFIGS 256
#define MAX_SURFACES 1024
#define MAX_CONTEXTS 64
#define MAX_BUFFERS 4096
#define MAX_IMAGES 256

#define VALID_ID(id, max) ((unsigned int)(id) < (unsigned int)(max))
#define GET_OBJ(pool, id) (&pool[id])

typedef struct bc250_surface bc250_surface;
typedef struct bc250_config bc250_config;
typedef struct bc250_context bc250_context;
typedef struct bc250_buffer bc250_buffer;
typedef struct bc250_image bc250_image;

struct bc250_surface {
    int allocated;
    int format;
    int width;
    int height;
    gpu_image_t image;
    gpu_memory_t memory;
    void *mapped_ptr;        /* Persistent host-visible memory mapping */
    int is_exported;         /* 1 if exported via vaExportSurfaceHandle (e.g. Sunshine) */
    int ref_count;
    /* Set by bc250_DestroySurfaces() the moment the application asks to
     * destroy this surface. From that point on the VASurfaceID is invalid
     * for any further application-facing VA call (vaBeginPicture,
     * vaDeriveImage, vaGetImage/vaPutImage, vaSyncSurface, ...), even
     * though `allocated` may still be 1 and the underlying Vulkan
     * image/memory may still be alive because a derived VAImage created via
     * vaDeriveImage() is keeping ref_count above zero. This lets the
     * driver honor normal VA-API surface-destroy semantics from the
     * caller's point of view while still deferring the actual Vulkan
     * teardown until the last outstanding derived image is destroyed. */
    int pending_destroy;
    /* How many decodes into this surface are queued or running, and how
     * the last one ended. A surface with one pending is not ready: every
     * call that reads or replaces its contents waits for zero first - see
     * wait_decoded() in va_backend.c. */
    int decode_pending;
    VAStatus decode_status;
    int gpu_work_pending; /* 1 if GPU compute (encode/vpp) commands are pending on this surface */
};

struct bc250_config {
    int allocated;
    VAProfile profile;
    VAEntrypoint entrypoint;
    VAConfigAttrib attribs[64];
    int num_attribs;
};

struct bc250_context {
    int allocated;
    VAConfigID config_id;
    int width;
    int height;
    int flag;
    VASurfaceID *render_targets;
    int num_render_targets;
    VASurfaceID current_render_target;
    VABufferID coded_buf_id;

    /* VAEntrypointVideoProc. No encoder, no decoder: one surface in,
     * one surface out, scaled.
     *
     * ⚠️ The rectangles are COPIES. VAProcPipelineParameterBuffer holds
     * pointers to the caller's own VARectangles, and vaCreateBuffer copies
     * the pointers, not what they point at. They are read while
     * vaRenderPicture is still on the caller's stack and kept here; by
     * vaEndPicture nothing says the originals still exist. */
    int vpp;
    struct {
        VASurfaceID source;
        VARectangle src_rect;
        VARectangle dst_rect;
        int has_source;
        int has_src_rect;
        int has_dst_rect;
    } vpp_state;

    /* Encoders & Decoders */
    h264_encoder_t *h264_enc;
    hevc_encoder_t *hevc_enc;
    h264_decoder_t *h264_dec;
    hevc_decoder_t *h265_dec;
    /* The thread that decodes this context's H.265 pictures after
     * vaEndPicture has returned. NULL until the first one. */
    struct bc250_hevc_async *hevc_async;

    /* BC250_PIPELINE=1 only: one frame whose GPU work is in flight and whose
     * CPU entropy coding has not been done yet. At most one - the pipeline is
     * deliberately bounded to a depth of 1, because the point is to overlap the
     * CPU and GPU halves of adjacent frames, not to buffer a queue of frames
     * (which would add latency to a live stream for no extra overlap).
     *
     * pending_coded_buf_id is the coded buffer the deferred finish must write
     * into: by the time it runs, c->coded_buf_id has already moved on to the
     * next frame's buffer. Getting this wrong would write frame N's bitstream
     * into frame N+1's buffer, which decodes as plausible-looking garbage
     * rather than failing loudly - the same shape of bug as the stale nonzero
     * mask in DEVLOG 19.4, so it is the thing to check first if pipelined
     * output ever looks subtly wrong. */
    bool has_pending_frame;
    h264_pending_frame_t pending_frame;
    VABufferID pending_coded_buf_id;

    /* Codec parameters accumulated during vaRenderPicture */
    struct {
        VAEncSequenceParameterBufferH264 seq_param;
        VAEncPictureParameterBufferH264 pic_param;
        VAEncSliceParameterBufferH264 slice_param;
        int has_seq;
        int has_pic;
        int has_slice;
        /* Last target_percentage seen on a VAEncMiscParameterTypeRateControl
         * buffer, so VAEncSequenceParameterBufferH264's own raw
         * bits_per_second can be scaled the same way. ffmpeg's default
         * h264_vaapi invocation sends the intended target X as "50% of 2X"
         * in BOTH buffers, and the sequence-parameter path used to apply the
         * raw 2X - re-initializing rate control at double the real target
         * and undoing the misc path's correct scaling, since whichever
         * buffer arrives last wins. See docs/DEVLOG.md §15 and
         * docs/rate_control_audit.md §2. 0 means "none seen yet"; treated
         * as 100% (no scaling). */
        unsigned int rc_target_percentage;
    } h264_state;

    /* Everything a decode picture needs, accumulated during
     * vaRenderPicture and consumed by vaEndPicture. */
    struct {
        VAPictureParameterBufferH264 pic;
        VAIQMatrixBufferH264 iq;
        int has_pic;
        int has_iq;
        /* ⚠️ The slices are copied, not pointed at, because they are
         * decoded from vaEndPicture with the driver lock dropped: by then
         * nothing guarantees the application's buffers are still where they
         * were. Copying a slice costs a memcpy; decoding one costs
         * milliseconds. */
        struct bc250_dec_slice {
            VASliceParameterBufferH264 p;
            size_t off;               /* into `data`; (size_t)-1 = no data yet */
            size_t len;
        } *slices;
        int n_slices, cap_slices;
        uint8_t *data;
        size_t n_data, cap_data;
    } dec_state;

    /* The same for H.265. Separate rather than shared: the two carry
     * different structures and nothing is gained by a union that every
     * line then has to ask which half it is looking at. */
    struct {
        VAPictureParameterBufferHEVC pic;
        int has_pic;
        VAIQMatrixBufferHEVC iq;      /* when the stream uses matrices */
        int has_iq;
        struct bc250_hevc_dec_slice {
            VASliceParameterBufferHEVC p;
            size_t off;               /* into `data`; (size_t)-1 = no data yet */
            size_t len;
        } *slices;
        int n_slices, cap_slices;
        uint8_t *data;
        size_t n_data, cap_data;
        /* ⚠️ The slice header is re-parsed here, so the payload needs its
         * emulation prevention bytes taken out somewhere. Kept between
         * slices rather than allocated per slice. */
        uint8_t *rbsp;
        size_t cap_rbsp;
    } hevc_dec_state;

    struct {
        VAEncSequenceParameterBufferHEVC seq_param;
        VAEncPictureParameterBufferHEVC pic_param;
        VAEncSliceParameterBufferHEVC slice_param;
        int has_seq;
        int has_pic;
        int has_slice;
    } hevc_state;
};

struct bc250_buffer {
    int allocated;
    VABufferType type;
    unsigned int size;
    unsigned int num_elements;
    void *data;
    int mapped;
    int is_derived;
    VkDeviceMemory gpu_mem;
    /* Only meaningful when is_derived is set: the surface whose Vulkan
     * memory this buffer aliases (via vaDeriveImage()). Used to release
     * the reference that buffer took on that surface when this buffer is
     * torn down (bc250_DestroyBuffer). VA_INVALID_SURFACE when this slot
     * does not currently back a derived image. */
    VASurfaceID derived_surface;
};

struct bc250_image {
    int allocated;
    VAImage image;
    VASurfaceID surface_id;
    VABufferID buffer_id;
};

typedef struct {
    pthread_mutex_t lock;
    /* Signalled, under `lock`, whenever a surface's decode_pending drops. */
    pthread_cond_t idle;
    gpu_context_t gpu;

    bc250_surface surfaces[MAX_SURFACES];
    bc250_config configs[MAX_CONFIGS];
    bc250_context contexts[MAX_CONTEXTS];
    bc250_buffer buffers[MAX_BUFFERS];
    bc250_image images[MAX_IMAGES];

    int next_buffer_hint;
    int max_width;
    int max_height;
} bc250_driver_data;

#define DRIVER_LOCK(data)   pthread_mutex_lock(&(data)->lock)
#define DRIVER_UNLOCK(data) pthread_mutex_unlock(&(data)->lock)

/* H.264 decoding, VAEntrypointVLD - va_decode.c */
void bc250_dec_reset(bc250_context *c);
void bc250_dec_free(bc250_context *c);
VAStatus bc250_dec_render(bc250_context *c, bc250_buffer *b);
/* ⚠️ Call with the driver lock DROPPED and the target surface pinned: this
 * is where the picture is actually decoded, and it is all CPU. */
VAStatus bc250_dec_decode(bc250_context *c, gpu_image_t out, gpu_memory_t mem);

void bc250_hevc_dec_reset(bc250_context *c);
void bc250_hevc_dec_free(bc250_context *c);
VAStatus bc250_hevc_dec_render(bc250_context *c, bc250_buffer *b);
/* vaEndPicture for H.265, in two halves. check() refuses what can be
 * refused while the application is still waiting for an answer, with the
 * driver lock held. submit() - lock DROPPED, target surface pinned and
 * marked pending - takes the accumulated picture and queues it for the
 * context's decode thread, or with BC250_HEVC_SYNC=1 decodes it there and
 * then. Either way the picture ends in bc250_decode_finished(). */
VAStatus bc250_hevc_dec_check(bc250_context *c);
struct bc250_hevc_job *bc250_hevc_dec_take(bc250_context *c, VASurfaceID target,
                                           gpu_image_t img, gpu_memory_t mem);
VAStatus bc250_hevc_dec_submit(bc250_driver_data *data, bc250_context *c,
                               struct bc250_hevc_job *job);
/* Waits for everything queued, then stops the thread. Lock DROPPED. */
void bc250_hevc_async_stop(struct bc250_hevc_async *a);
/* A decode into `target` is over: unpins it, lowers decode_pending and
 * wakes whoever waits. Takes the driver lock itself. */
void bc250_decode_finished(bc250_driver_data *data, VASurfaceID target,
                           VAStatus st);

/* Core VA-API Driver Functions */
VAStatus __vaDriverInit_1_0(VADriverContextP ctx);
VAStatus __vaDriverInit_0_32(VADriverContextP ctx);
VAStatus bc250_Initialize(VADriverContextP ctx, int *major_version, int *minor_version);
VAStatus bc250_Terminate(VADriverContextP ctx);

VAStatus bc250_QueryConfigProfiles(VADriverContextP ctx, VAProfile *profile_list, int *num_profiles);
VAStatus bc250_QueryConfigEntrypoints(VADriverContextP ctx, VAProfile profile, VAEntrypoint *entrypoint_list, int *num_entrypoints);
VAStatus bc250_GetConfigAttributes(VADriverContextP ctx, VAProfile profile, VAEntrypoint entrypoint, VAConfigAttrib *attrib_list, int num_attribs);
VAStatus bc250_CreateConfig(VADriverContextP ctx, VAProfile profile, VAEntrypoint entrypoint, VAConfigAttrib *attrib_list, int num_attribs, VAConfigID *config_id);
VAStatus bc250_DestroyConfig(VADriverContextP ctx, VAConfigID config_id);
VAStatus bc250_QueryConfigAttributes(VADriverContextP ctx, VAConfigID config_id, VAProfile *profile, VAEntrypoint *entrypoint, VAConfigAttrib *attrib_list, int *num_attribs);

VAStatus bc250_QuerySurfaceAttributes(VADriverContextP ctx, VAConfigID config, VASurfaceAttrib *attrib_list, unsigned int *num_attribs);
VAStatus bc250_CreateSurfaces(VADriverContextP ctx, int width, int height, int format, int num_surfaces, VASurfaceID *surfaces);
VAStatus bc250_CreateSurfaces2(VADriverContextP ctx, unsigned int format, unsigned int width, unsigned int height,
                              VASurfaceID *surfaces, unsigned int num_surfaces,
                              VASurfaceAttrib *attrib_list, unsigned int num_attribs);
VAStatus bc250_DestroySurfaces(VADriverContextP ctx, VASurfaceID *surface_list, int num_surfaces);

VAStatus bc250_CreateContext(VADriverContextP ctx, VAConfigID config_id, int picture_width, int picture_height, int flag, VASurfaceID *render_targets, int num_render_targets, VAContextID *context);
VAStatus bc250_DestroyContext(VADriverContextP ctx, VAContextID context);

VAStatus bc250_CreateBuffer(VADriverContextP ctx, VAContextID context, VABufferType type, unsigned int size, unsigned int num_elements, void *data, VABufferID *buf_id);
VAStatus bc250_BufferSetNumElements(VADriverContextP ctx, VABufferID buf_id, unsigned int num_elements);
VAStatus bc250_MapBuffer(VADriverContextP ctx, VABufferID buf_id, void **pbuf);
VAStatus bc250_UnmapBuffer(VADriverContextP ctx, VABufferID buf_id);
VAStatus bc250_DestroyBuffer(VADriverContextP ctx, VABufferID buffer_id);

VAStatus bc250_BeginPicture(VADriverContextP ctx, VAContextID context, VASurfaceID render_target);
VAStatus bc250_RenderPicture(VADriverContextP ctx, VAContextID context, VABufferID *buffers, int num_buffers);
VAStatus bc250_EndPicture(VADriverContextP ctx, VAContextID context);
VAStatus bc250_SyncSurface(VADriverContextP ctx, VASurfaceID render_target);
VAStatus bc250_QuerySurfaceStatus(VADriverContextP ctx, VASurfaceID render_target, VASurfaceStatus *status);

VAStatus bc250_QueryImageFormats(VADriverContextP ctx, VAImageFormat *format_list, int *num_formats);
VAStatus bc250_CreateImage(VADriverContextP ctx, VAImageFormat *format, int width, int height, VAImage *image);
VAStatus bc250_DestroyImage(VADriverContextP ctx, VAImageID image);
VAStatus bc250_DeriveImage(VADriverContextP ctx, VASurfaceID surface, VAImage *image);
VAStatus bc250_GetImage(VADriverContextP ctx, VASurfaceID surface, int x, int y, unsigned int width, unsigned int height, VAImageID image);
VAStatus bc250_PutImage(VADriverContextP ctx, VASurfaceID surface, VAImageID image, int src_x, int src_y, unsigned int src_width, unsigned int src_height, int dest_x, int dest_y, unsigned int dest_width, unsigned int dest_height);
VAStatus bc250_ExportSurfaceHandle(VADriverContextP ctx, VASurfaceID surface_id, uint32_t mem_type, uint32_t flags, void *descriptor);
VAStatus bc250_SetImagePalette(VADriverContextP ctx, VAImageID image, unsigned char *palette);

/* Subpictures (unsupported - stubs required by the libva driver contract) */
VAStatus bc250_QuerySubpictureFormats(VADriverContextP ctx, VAImageFormat *format_list, unsigned int *flags, unsigned int *num_formats);
VAStatus bc250_CreateSubpicture(VADriverContextP ctx, VAImageID image, VASubpictureID *subpicture);
VAStatus bc250_DestroySubpicture(VADriverContextP ctx, VASubpictureID subpicture);
VAStatus bc250_SetSubpictureImage(VADriverContextP ctx, VASubpictureID subpicture, VAImageID image);
VAStatus bc250_SetSubpictureChromakey(VADriverContextP ctx, VASubpictureID subpicture, unsigned int chromakey_min, unsigned int chromakey_max, unsigned int chromakey_mask);
VAStatus bc250_SetSubpictureGlobalAlpha(VADriverContextP ctx, VASubpictureID subpicture, float global_alpha);
VAStatus bc250_AssociateSubpicture(VADriverContextP ctx, VASubpictureID subpicture, VASurfaceID *target_surfaces, int num_surfaces, short src_x, short src_y, unsigned short src_width, unsigned short src_height, short dest_x, short dest_y, unsigned short dest_width, unsigned short dest_height, unsigned int flags);
VAStatus bc250_DeassociateSubpicture(VADriverContextP ctx, VASubpictureID subpicture, VASurfaceID *target_surfaces, int num_surfaces);

/* Display attributes (unsupported - stubs required by the libva driver contract) */
VAStatus bc250_QueryDisplayAttributes(VADriverContextP ctx, VADisplayAttribute *attr_list, int *num_attributes);
VAStatus bc250_GetDisplayAttributes(VADriverContextP ctx, VADisplayAttribute *attr_list, int num_attributes);
VAStatus bc250_SetDisplayAttributes(VADriverContextP ctx, VADisplayAttribute *attr_list, int num_attributes);

/* Video Processing (VPP) */
VAStatus bc250_QueryVideoProcFilters(VADriverContextP ctx, VAContextID context, VAProcFilterType *filters, unsigned int *num_filters);
VAStatus bc250_QueryVideoProcFilterCaps(VADriverContextP ctx, VAContextID context, VAProcFilterType type, void *filter_caps, unsigned int *num_filter_caps);
VAStatus bc250_QueryVideoProcPipelineCaps(VADriverContextP ctx, VAContextID context, VABufferID *filters, unsigned int num_filters, VAProcPipelineCaps *pipeline_caps);

#endif // BC250_VA_BACKEND_H
