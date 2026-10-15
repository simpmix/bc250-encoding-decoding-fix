/* bc250-encoding-decoding-fix v0.4.3 - https://github.com/simpmix/bc250-encoding-decoding-fix */
/*
 * test_va_api.c - Integration test for BC-250 VA-API Backend Driver
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>
#include "va_backend.h"

#ifndef VA_RC_ICQ
#define VA_RC_ICQ 0x00000040
#endif

struct thread_test_args {
    VADriverContextP ctx;
    VASurfaceID surface;
    VAContextID context_id;
    int iterations;
    int error_count;
};

static void *filter_worker_thread(void *arg) {
    struct thread_test_args *args = (struct thread_test_args *)arg;
    for (int i = 0; i < args->iterations; i++) {
        VAImage derived_img;
        VAStatus s = args->ctx->vtable->vaDeriveImage(args->ctx, args->surface, &derived_img);
        if (s != VA_STATUS_SUCCESS) {
            args->error_count++;
            break;
        }
        void *pbuf = NULL;
        s = args->ctx->vtable->vaMapBuffer(args->ctx, derived_img.buf, &pbuf);
        if (s != VA_STATUS_SUCCESS || !pbuf) {
            args->error_count++;
            args->ctx->vtable->vaDestroyImage(args->ctx, derived_img.image_id);
            break;
        }
        uint8_t *ptr = (uint8_t *)pbuf;
        ptr[0] = (uint8_t)(i & 0xFF);

        s = args->ctx->vtable->vaUnmapBuffer(args->ctx, derived_img.buf);
        if (s != VA_STATUS_SUCCESS) {
            args->error_count++;
        }
        s = args->ctx->vtable->vaDestroyImage(args->ctx, derived_img.image_id);
        if (s != VA_STATUS_SUCCESS) {
            args->error_count++;
        }
    }
    return NULL;
}

static void *encoder_worker_thread(void *arg) {
    struct thread_test_args *args = (struct thread_test_args *)arg;
    for (int i = 0; i < args->iterations; i++) {
        VABufferID buf_id = VA_INVALID_ID;
        VAStatus s = args->ctx->vtable->vaCreateBuffer(args->ctx, args->context_id,
                                                      VAEncCodedBufferType, 64 * 1024, 1, NULL, &buf_id);
        if (s != VA_STATUS_SUCCESS || buf_id == VA_INVALID_ID) {
            args->error_count++;
            break;
        }
        void *pbuf = NULL;
        s = args->ctx->vtable->vaMapBuffer(args->ctx, buf_id, &pbuf);
        if (s == VA_STATUS_SUCCESS && pbuf) {
            uint8_t *ptr = (uint8_t *)pbuf;
            ptr[0] = (uint8_t)(i & 0xFF);
            args->ctx->vtable->vaUnmapBuffer(args->ctx, buf_id);
        } else {
            args->error_count++;
        }
        s = args->ctx->vtable->vaDestroyBuffer(args->ctx, buf_id);
        if (s != VA_STATUS_SUCCESS) {
            args->error_count++;
        }
    }
    return NULL;
}

int main(void) {
    printf("=== Running BC-250 VA-API Driver Tests ===\n");

    struct VADriverContext ctx;
    struct VADriverVTable vtable;
    memset(&ctx, 0, sizeof(ctx));
    memset(&vtable, 0, sizeof(vtable));
    ctx.vtable = &vtable;

    int major = 0, minor = 0;
    VAStatus status = bc250_Initialize(&ctx, &major, &minor);
    if (status != VA_STATUS_SUCCESS) {
        printf("[SKIP] Vulkan initialization skipped (no compatible GPU or display detected in test environment)\n");
        return 0;
    }

    printf("[INFO] Driver initialized: VA-API %d.%d (%s)\n", major, minor, ctx.str_vendor);

    /* 1. Query Profiles (with count-only check and full list check) */
    int num_profiles = 0;
    status = ctx.vtable->vaQueryConfigProfiles(&ctx, NULL, &num_profiles);
    assert(status == VA_STATUS_SUCCESS);
    assert(num_profiles > 0);

    VAProfile profiles[MAX_PROFILES];
    status = ctx.vtable->vaQueryConfigProfiles(&ctx, profiles, &num_profiles);
    assert(status == VA_STATUS_SUCCESS);
    assert(num_profiles >= 5);
    int found_hevc = 0;
    for (int p = 0; p < num_profiles; p++) {
        if (profiles[p] == VAProfileHEVCMain) found_hevc = 1;
    }
    assert(found_hevc == 1 && "VAProfileHEVCMain must be advertised in profile list");
    printf("[PASS] Found %d supported VA profiles (including VAProfileHEVCMain)\n", num_profiles);

    /* 2. Query Entrypoints for H.264 Main and HEVC Main */
    int num_entrypoints = 0;
    status = ctx.vtable->vaQueryConfigEntrypoints(&ctx, VAProfileH264Main, NULL, &num_entrypoints);
    assert(status == VA_STATUS_SUCCESS);
    assert(num_entrypoints >= 1);

    VAEntrypoint entrypoints[MAX_ENTRYPOINTS];
    status = ctx.vtable->vaQueryConfigEntrypoints(&ctx, VAProfileH264Main, entrypoints, &num_entrypoints);
    assert(status == VA_STATUS_SUCCESS);
    assert(num_entrypoints >= 1);
    assert(entrypoints[0] == VAEntrypointEncSlice);

    int num_hevc_entrypoints = 0;
    status = ctx.vtable->vaQueryConfigEntrypoints(&ctx, VAProfileHEVCMain, NULL, &num_hevc_entrypoints);
    assert(status == VA_STATUS_SUCCESS);
    assert(num_hevc_entrypoints >= 1);
    status = ctx.vtable->vaQueryConfigEntrypoints(&ctx, VAProfileHEVCMain, entrypoints, &num_hevc_entrypoints);
    assert(status == VA_STATUS_SUCCESS);
    assert(entrypoints[0] == VAEntrypointEncSlice);
    printf("[PASS] H.264 and HEVC Main support VAEntrypointEncSlice entrypoint\n");

    /* 3. Create Config */
    VAConfigAttrib attribs[2];
    attribs[0].type = VAConfigAttribRTFormat;
    attribs[0].value = VA_RT_FORMAT_YUV420;
    attribs[1].type = VAConfigAttribRateControl;
    attribs[1].value = VA_RC_CBR;

    VAConfigID config_id = VA_INVALID_ID;
    status = ctx.vtable->vaCreateConfig(&ctx, VAProfileH264Main, VAEntrypointEncSlice, attribs, 2, &config_id);
    assert(status == VA_STATUS_SUCCESS);
    assert(config_id != VA_INVALID_ID);
    printf("[PASS] Config created with ID %d\n", config_id);

    /* 4. Query Surface Attributes (contract test for FFmpeg/OBS) */
    unsigned int num_surface_attribs = 0;
    status = ctx.vtable->vaQuerySurfaceAttributes(&ctx, config_id, NULL, &num_surface_attribs);
    assert(status == VA_STATUS_SUCCESS);
    assert(num_surface_attribs > 0);

    VASurfaceAttrib surface_attribs[8];
    status = ctx.vtable->vaQuerySurfaceAttributes(&ctx, config_id, surface_attribs, &num_surface_attribs);
    assert(status == VA_STATUS_SUCCESS);
    printf("[PASS] Surface attributes query passed (%u attributes supported)\n", num_surface_attribs);

    /* 5. Create Surfaces */
    VASurfaceID surfaces[2];
    status = ctx.vtable->vaCreateSurfaces(&ctx, 1920, 1080, VA_RT_FORMAT_YUV420, 2, surfaces);
    assert(status == VA_STATUS_SUCCESS);
    printf("[PASS] Allocated 2 1080p surfaces (IDs %d, %d)\n", surfaces[0], surfaces[1]);

    /* 5b. CreateSurfaces2 external-memory-import rejection (Gamescope/
     * Sunshine/Steam Link screencasting green-black-screen guard).
     * Rejects unsupported external buffer descriptors with
     * VA_STATUS_ERROR_UNSUPPORTED_MEMORY_TYPE while preserving
     * native VA surfaces and no-attribute allocation. */
    {
        VASurfaceID import_surfaces[1];

        /* No attrib_list at all (FFmpeg's own vaCreateSurfaces2 call shape) must succeed */
        status = ctx.vtable->vaCreateSurfaces2(&ctx, VA_RT_FORMAT_YUV420, 64, 64,
                                               import_surfaces, 1, NULL, 0);
        assert(status == VA_STATUS_SUCCESS);
        ctx.vtable->vaDestroySurfaces(&ctx, import_surfaces, 1);

        /* Explicit native memory attribute must succeed */
        VASurfaceAttrib native_attrib;
        memset(&native_attrib, 0, sizeof(native_attrib));
        native_attrib.type = VASurfaceAttribMemoryType;
        native_attrib.flags = VA_SURFACE_ATTRIB_SETTABLE;
        native_attrib.value.type = VAGenericValueTypeInteger;
        native_attrib.value.value.i = VA_SURFACE_ATTRIB_MEM_TYPE_VA;
        status = ctx.vtable->vaCreateSurfaces2(&ctx, VA_RT_FORMAT_YUV420, 64, 64,
                                               import_surfaces, 1, &native_attrib, 1);
        assert(status == VA_STATUS_SUCCESS);
        ctx.vtable->vaDestroySurfaces(&ctx, import_surfaces, 1);

        /* External DRM_PRIME_2 DMA-BUF (Gamescope request) must be rejected without BC250_ENABLE_DMABUF_IMPORT=1 */
        VASurfaceAttrib prime2_attrib;
        memset(&prime2_attrib, 0, sizeof(prime2_attrib));
        prime2_attrib.type = VASurfaceAttribMemoryType;
        prime2_attrib.flags = VA_SURFACE_ATTRIB_SETTABLE;
        prime2_attrib.value.type = VAGenericValueTypeInteger;
        prime2_attrib.value.value.i = VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2;
        status = ctx.vtable->vaCreateSurfaces2(&ctx, VA_RT_FORMAT_YUV420, 64, 64,
                                               import_surfaces, 1, &prime2_attrib, 1);
        assert(status == VA_STATUS_ERROR_UNSUPPORTED_MEMORY_TYPE);

        /* Older single-plane DRM_PRIME must also be rejected */
        VASurfaceAttrib prime1_attrib;
        memset(&prime1_attrib, 0, sizeof(prime1_attrib));
        prime1_attrib.type = VASurfaceAttribMemoryType;
        prime1_attrib.flags = VA_SURFACE_ATTRIB_SETTABLE;
        prime1_attrib.value.type = VAGenericValueTypeInteger;
        prime1_attrib.value.value.i = VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME;
        status = ctx.vtable->vaCreateSurfaces2(&ctx, VA_RT_FORMAT_YUV420, 64, 64,
                                               import_surfaces, 1, &prime1_attrib, 1);
        assert(status == VA_STATUS_ERROR_UNSUPPORTED_MEMORY_TYPE);

        /* USER_PTR must also be rejected */
        VASurfaceAttrib userptr_attrib;
        memset(&userptr_attrib, 0, sizeof(userptr_attrib));
        userptr_attrib.type = VASurfaceAttribMemoryType;
        userptr_attrib.flags = VA_SURFACE_ATTRIB_SETTABLE;
        userptr_attrib.value.type = VAGenericValueTypeInteger;
        userptr_attrib.value.value.i = VA_SURFACE_ATTRIB_MEM_TYPE_USER_PTR;
        status = ctx.vtable->vaCreateSurfaces2(&ctx, VA_RT_FORMAT_YUV420, 64, 64,
                                               import_surfaces, 1, &userptr_attrib, 1);
        assert(status == VA_STATUS_ERROR_UNSUPPORTED_MEMORY_TYPE);

        printf("[PASS] CreateSurfaces2 rejects external DMA-BUF/USER_PTR import, keeps native/no-attrib paths working\n");
    }

    /* 6. Create Context */
    VAContextID context_id = VA_INVALID_ID;
    status = ctx.vtable->vaCreateContext(&ctx, config_id, 1920, 1080, 0, surfaces, 2, &context_id);
    assert(status == VA_STATUS_SUCCESS);
    assert(context_id != VA_INVALID_ID);
    printf("[PASS] Created encode context with ID %d\n", context_id);

    /* 7. Create Coded Buffer and verify VACodedBufferSegment initialization */
    VABufferID coded_buf_id = VA_INVALID_ID;
    status = ctx.vtable->vaCreateBuffer(&ctx, context_id, VAEncCodedBufferType, 1024 * 1024, 1, NULL, &coded_buf_id);
    assert(status == VA_STATUS_SUCCESS);

    void *mapped_data = NULL;
    status = ctx.vtable->vaMapBuffer(&ctx, coded_buf_id, &mapped_data);
    assert(status == VA_STATUS_SUCCESS);
    assert(mapped_data != NULL);

    VACodedBufferSegment *seg = (VACodedBufferSegment *)mapped_data;
    assert(seg->buf != NULL);
    ctx.vtable->vaUnmapBuffer(&ctx, coded_buf_id);
    printf("[PASS] Coded buffer segment initialization validated\n");

    /* 8. Image transfer test (GetImage / PutImage bit-exact round-trip) */
    VAImage image1, image2;
    VAImageFormat fmt = { .fourcc = VA_FOURCC_NV12, .byte_order = VA_LSB_FIRST, .bits_per_pixel = 12 };
    status = ctx.vtable->vaCreateImage(&ctx, &fmt, 1920, 1080, &image1);
    assert(status == VA_STATUS_SUCCESS);
    status = ctx.vtable->vaCreateImage(&ctx, &fmt, 1920, 1080, &image2);
    assert(status == VA_STATUS_SUCCESS);

    /* Fill image1 with a deterministic pixel pattern */
    void *img1_ptr = NULL;
    status = ctx.vtable->vaMapBuffer(&ctx, image1.buf, &img1_ptr);
    assert(status == VA_STATUS_SUCCESS && img1_ptr != NULL);
    uint8_t *y1 = (uint8_t *)img1_ptr + image1.offsets[0];
    uint8_t *uv1 = (uint8_t *)img1_ptr + image1.offsets[1];
    for (int r = 0; r < 1080; r++) {
        for (int c = 0; c < 1920; c++) {
            y1[r * image1.pitches[0] + c] = (uint8_t)((r * 3 + c * 5) & 0xFF);
        }
    }
    for (int r = 0; r < 540; r++) {
        for (int c = 0; c < 1920; c++) {
            uv1[r * image1.pitches[1] + c] = (uint8_t)((r * 7 + c * 11) & 0xFF);
        }
    }
    ctx.vtable->vaUnmapBuffer(&ctx, image1.buf);

    /* Upload pixel data to surface */
    status = ctx.vtable->vaPutImage(&ctx, surfaces[0], image1.image_id, 0, 0, 1920, 1080, 0, 0, 1920, 1080);
    assert(status == VA_STATUS_SUCCESS);

    /* Download pixel data back from surface into image2 */
    status = ctx.vtable->vaGetImage(&ctx, surfaces[0], 0, 0, 1920, 1080, image2.image_id);
    assert(status == VA_STATUS_SUCCESS);

    /* Verify bit-exact match of uploaded vs downloaded pixels */
    void *img2_ptr = NULL;
    status = ctx.vtable->vaMapBuffer(&ctx, image2.buf, &img2_ptr);
    assert(status == VA_STATUS_SUCCESS && img2_ptr != NULL);
    const uint8_t *y2 = (const uint8_t *)img2_ptr + image2.offsets[0];
    const uint8_t *uv2 = (const uint8_t *)img2_ptr + image2.offsets[1];
    int pixel_mismatches = 0;
    for (int r = 0; r < 1080; r++) {
        for (int c = 0; c < 1920; c++) {
            if (y2[r * image2.pitches[0] + c] != (uint8_t)((r * 3 + c * 5) & 0xFF)) {
                pixel_mismatches++;
            }
        }
    }
    for (int r = 0; r < 540; r++) {
        for (int c = 0; c < 1920; c++) {
            if (uv2[r * image2.pitches[1] + c] != (uint8_t)((r * 7 + c * 11) & 0xFF)) {
                pixel_mismatches++;
            }
        }
    }
    assert(pixel_mismatches == 0 && "Pixel data mismatch in PutImage/GetImage round-trip!");
    ctx.vtable->vaUnmapBuffer(&ctx, image2.buf);
    printf("[PASS] Image transfer bit-exact round-trip verified (0 pixel mismatches across 3.1M pixels)\n");

    /* 9. Derive Image test */
    VAImage derived_img;
    status = ctx.vtable->vaDeriveImage(&ctx, surfaces[0], &derived_img);
    assert(status == VA_STATUS_SUCCESS);
    ctx.vtable->vaDestroyImage(&ctx, derived_img.image_id);
    printf("[PASS] Derive image and surface mapping validated\n");

    /* 10. Multi-threaded Concurrency Test (Filter & Encoder threads racing) */
    struct thread_test_args filter_args = {
        .ctx = &ctx,
        .surface = surfaces[0],
        .context_id = context_id,
        .iterations = 100,
        .error_count = 0
    };
    struct thread_test_args encoder_args = {
        .ctx = &ctx,
        .surface = surfaces[1],
        .context_id = context_id,
        .iterations = 100,
        .error_count = 0
    };

    pthread_t th1, th2;
    int r1 = pthread_create(&th1, NULL, filter_worker_thread, &filter_args);
    int r2 = pthread_create(&th2, NULL, encoder_worker_thread, &encoder_args);
    assert(r1 == 0 && r2 == 0);

    pthread_join(th1, NULL);
    pthread_join(th2, NULL);

    assert(filter_args.error_count == 0);
    assert(encoder_args.error_count == 0);
    printf("[PASS] Concurrent multi-threaded execution verified (100 iterations of filter/encoder race without collision)\n");

    /* 11. Test VAConfigAttribRateControl negotiation (VA_RC_CQP, VA_RC_VBR, and VA_RC_ICQ) */
    VAConfigAttrib rc_attrib_query = { .type = VAConfigAttribRateControl, .value = 0 };
    status = ctx.vtable->vaGetConfigAttributes(&ctx, VAProfileH264Main, VAEntrypointEncSlice, &rc_attrib_query, 1);
    assert(status == VA_STATUS_SUCCESS);
    assert((rc_attrib_query.value & (VA_RC_CBR | VA_RC_VBR | VA_RC_CQP | VA_RC_ICQ)) == (VA_RC_CBR | VA_RC_VBR | VA_RC_CQP | VA_RC_ICQ));

    VAConfigAttrib icq_attribs[2] = {
        { .type = VAConfigAttribRTFormat, .value = VA_RT_FORMAT_YUV420 },
        { .type = VAConfigAttribRateControl, .value = VA_RC_ICQ }
    };
    VAConfigID icq_config_id = VA_INVALID_ID;
    status = ctx.vtable->vaCreateConfig(&ctx, VAProfileH264Main, VAEntrypointEncSlice, icq_attribs, 2, &icq_config_id);
    assert(status == VA_STATUS_SUCCESS);
    VAContextID icq_context_id = VA_INVALID_ID;
    status = ctx.vtable->vaCreateContext(&ctx, icq_config_id, 1920, 1080, 0, surfaces, 2, &icq_context_id);
    assert(status == VA_STATUS_SUCCESS);
    bc250_driver_data *icq_drv_data = (bc250_driver_data *)ctx.pDriverData;
    assert(icq_drv_data->contexts[icq_context_id].h264_enc != NULL);
    assert(h264_encoder_get_rc_mode(icq_drv_data->contexts[icq_context_id].h264_enc) == RC_VBR);
    ctx.vtable->vaDestroyContext(&ctx, icq_context_id);
    ctx.vtable->vaDestroyConfig(&ctx, icq_config_id);
    printf("[PASS] Negotiated VA_RC_ICQ config and context creation verified (auto-VBR mapped)\n");

    VAConfigAttrib cqp_attribs[2] = {
        { .type = VAConfigAttribRTFormat, .value = VA_RT_FORMAT_YUV420 },
        { .type = VAConfigAttribRateControl, .value = VA_RC_CQP }
    };
    VAConfigID cqp_config_id = VA_INVALID_ID;
    status = ctx.vtable->vaCreateConfig(&ctx, VAProfileH264Main, VAEntrypointEncSlice, cqp_attribs, 2, &cqp_config_id);
    assert(status == VA_STATUS_SUCCESS);
    VAContextID cqp_context_id = VA_INVALID_ID;
    status = ctx.vtable->vaCreateContext(&ctx, cqp_config_id, 1920, 1080, 0, surfaces, 2, &cqp_context_id);
    assert(status == VA_STATUS_SUCCESS);
    ctx.vtable->vaDestroyContext(&ctx, cqp_context_id);
    ctx.vtable->vaDestroyConfig(&ctx, cqp_config_id);
    printf("[PASS] Negotiated VA_RC_CQP config and context creation verified\n");

    /* 12. Test HEVC VA-API pipeline (Config, Context, Sequence/Picture/Slice/Misc Params) */
    VAConfigAttrib hevc_cfg_attribs[2] = {
        { .type = VAConfigAttribRTFormat, .value = VA_RT_FORMAT_YUV420 },
        { .type = VAConfigAttribRateControl, .value = VA_RC_VBR }
    };
    VAConfigID hevc_config_id = VA_INVALID_ID;
    status = ctx.vtable->vaCreateConfig(&ctx, VAProfileHEVCMain, VAEntrypointEncSlice, hevc_cfg_attribs, 2, &hevc_config_id);
    assert(status == VA_STATUS_SUCCESS && hevc_config_id != VA_INVALID_ID);

    VAContextID hevc_context_id = VA_INVALID_ID;
    status = ctx.vtable->vaCreateContext(&ctx, hevc_config_id, 1920, 1080, 0, surfaces, 2, &hevc_context_id);
    assert(status == VA_STATUS_SUCCESS && hevc_context_id != VA_INVALID_ID);

    /* Verify HEVC context has hevc_enc created and initialized to RC_VBR */
    bc250_driver_data *drv_data = (bc250_driver_data *)ctx.pDriverData;
    bc250_context *hevc_c = &drv_data->contexts[hevc_context_id];
    assert(hevc_c->hevc_enc != NULL);
    assert(hevc_encoder_get_rc_mode(hevc_c->hevc_enc) == RC_VBR);

    /* Create and test VAEncCodedBufferType for HEVC */
    VABufferID hevc_coded_buf_id = VA_INVALID_ID;
    status = ctx.vtable->vaCreateBuffer(&ctx, hevc_context_id, VAEncCodedBufferType, 1024 * 1024, 1, NULL, &hevc_coded_buf_id);
    assert(status == VA_STATUS_SUCCESS && hevc_coded_buf_id != VA_INVALID_ID);

    /* Test vaBeginPicture */
    status = ctx.vtable->vaBeginPicture(&ctx, hevc_context_id, surfaces[0]);
    assert(status == VA_STATUS_SUCCESS);

    /* Render HEVC Sequence, Picture, Slice, and Misc RateControl parameters */
    VAEncSequenceParameterBufferHEVC seq_hevc;
    memset(&seq_hevc, 0, sizeof(seq_hevc));
    seq_hevc.intra_period = 60;
    seq_hevc.bits_per_second = 8000000;
    seq_hevc.pic_width_in_luma_samples = 1920;
    seq_hevc.pic_height_in_luma_samples = 1080;
    seq_hevc.vui_time_scale = 60;
    seq_hevc.vui_num_units_in_tick = 1;

    VABufferID seq_buf_id = VA_INVALID_ID;
    status = ctx.vtable->vaCreateBuffer(&ctx, hevc_context_id, VAEncSequenceParameterBufferType,
                                        sizeof(seq_hevc), 1, &seq_hevc, &seq_buf_id);
    assert(status == VA_STATUS_SUCCESS);

    VAEncPictureParameterBufferHEVC pic_hevc;
    memset(&pic_hevc, 0, sizeof(pic_hevc));
    pic_hevc.coded_buf = hevc_coded_buf_id;
    pic_hevc.pic_init_qp = 24;
    pic_hevc.pic_fields.bits.idr_pic_flag = 1;

    VABufferID pic_buf_id = VA_INVALID_ID;
    status = ctx.vtable->vaCreateBuffer(&ctx, hevc_context_id, VAEncPictureParameterBufferType,
                                        sizeof(pic_hevc), 1, &pic_hevc, &pic_buf_id);
    assert(status == VA_STATUS_SUCCESS);

    VAEncSliceParameterBufferHEVC slice_hevc;
    memset(&slice_hevc, 0, sizeof(slice_hevc));
    slice_hevc.slice_type = 2; /* I-slice */
    slice_hevc.num_ctu_in_slice = (1920 / 16) * (1080 / 16);

    VABufferID slice_buf_id = VA_INVALID_ID;
    status = ctx.vtable->vaCreateBuffer(&ctx, hevc_context_id, VAEncSliceParameterBufferType,
                                        sizeof(slice_hevc), 1, &slice_hevc, &slice_buf_id);
    assert(status == VA_STATUS_SUCCESS);

    /* Misc RateControl parameter buffer */
    uint8_t misc_rc_mem[sizeof(VAEncMiscParameterBuffer) + sizeof(VAEncMiscParameterRateControl)];
    memset(misc_rc_mem, 0, sizeof(misc_rc_mem));
    VAEncMiscParameterBuffer *pmisc = (VAEncMiscParameterBuffer *)misc_rc_mem;
    pmisc->type = VAEncMiscParameterTypeRateControl;
    VAEncMiscParameterRateControl *prc = (VAEncMiscParameterRateControl *)pmisc->data;
    prc->bits_per_second = 10000000;
    prc->target_percentage = 100;
    prc->initial_qp = 22;

    VABufferID misc_buf_id = VA_INVALID_ID;
    status = ctx.vtable->vaCreateBuffer(&ctx, hevc_context_id, VAEncMiscParameterBufferType,
                                        sizeof(misc_rc_mem), 1, misc_rc_mem, &misc_buf_id);
    assert(status == VA_STATUS_SUCCESS);

    VABufferID render_bufs[4] = { seq_buf_id, pic_buf_id, slice_buf_id, misc_buf_id };
    status = ctx.vtable->vaRenderPicture(&ctx, hevc_context_id, render_bufs, 4);
    assert(status == VA_STATUS_SUCCESS);

    /* Verify that sequence, picture, and rate control parameters propagated into hevc_enc */
    assert(hevc_encoder_get_gop_size(hevc_c->hevc_enc) == 60);
    assert(hevc_encoder_get_fps(hevc_c->hevc_enc) == 60);
    assert(hevc_encoder_get_qp(hevc_c->hevc_enc) == 22);
    assert(hevc_encoder_get_bitrate(hevc_c->hevc_enc) == 10000000);
    assert(hevc_c->coded_buf_id == hevc_coded_buf_id);
    printf("[PASS] HEVC VA-API parameter buffer passing and rate control verified\n");

    /* 13. Test Dynamic Bitrate & Framerate Switching (H.264 & HEVC runtime adaptation) */
    bc250_context *h264_c = &drv_data->contexts[context_id];
    assert(h264_c->h264_enc != NULL);

    /* (a) H.264 dynamic bitrate switching via VAEncMiscParameterTypeRateControl */
    {
        uint8_t h264_misc_mem[sizeof(VAEncMiscParameterBuffer) + sizeof(VAEncMiscParameterRateControl)];
        memset(h264_misc_mem, 0, sizeof(h264_misc_mem));
        VAEncMiscParameterBuffer *m = (VAEncMiscParameterBuffer *)h264_misc_mem;
        m->type = VAEncMiscParameterTypeRateControl;
        VAEncMiscParameterRateControl *rc = (VAEncMiscParameterRateControl *)m->data;
        rc->bits_per_second = 12000000;
        rc->target_percentage = 100;
        rc->initial_qp = 28;

        VABufferID h264_rc_buf = VA_INVALID_ID;
        status = ctx.vtable->vaCreateBuffer(&ctx, context_id, VAEncMiscParameterBufferType,
                                            sizeof(h264_misc_mem), 1, h264_misc_mem, &h264_rc_buf);
        assert(status == VA_STATUS_SUCCESS);
        status = ctx.vtable->vaRenderPicture(&ctx, context_id, &h264_rc_buf, 1);
        assert(status == VA_STATUS_SUCCESS);
        assert(h264_encoder_get_bitrate(h264_c->h264_enc) == 12000000);
        assert(h264_encoder_get_qp(h264_c->h264_enc) == 28);
        ctx.vtable->vaDestroyBuffer(&ctx, h264_rc_buf);

        /* Switch bitrate down to 4 Mbps with 75% target percentage -> 3,000,000 bps */
        rc->bits_per_second = 4000000;
        rc->target_percentage = 75;
        rc->initial_qp = 32;
        status = ctx.vtable->vaCreateBuffer(&ctx, context_id, VAEncMiscParameterBufferType,
                                            sizeof(h264_misc_mem), 1, h264_misc_mem, &h264_rc_buf);
        assert(status == VA_STATUS_SUCCESS);
        status = ctx.vtable->vaRenderPicture(&ctx, context_id, &h264_rc_buf, 1);
        assert(status == VA_STATUS_SUCCESS);
        assert(h264_encoder_get_bitrate(h264_c->h264_enc) == 3000000);
        assert(h264_encoder_get_qp(h264_c->h264_enc) == 32);
        ctx.vtable->vaDestroyBuffer(&ctx, h264_rc_buf);
    }

    /* (b) H.264 dynamic framerate switching via VAEncMiscParameterTypeFrameRate */
    {
        uint8_t h264_fps_mem[sizeof(VAEncMiscParameterBuffer) + sizeof(VAEncMiscParameterFrameRate)];
        memset(h264_fps_mem, 0, sizeof(h264_fps_mem));
        VAEncMiscParameterBuffer *m = (VAEncMiscParameterBuffer *)h264_fps_mem;
        m->type = VAEncMiscParameterTypeFrameRate;
        VAEncMiscParameterFrameRate *fr = (VAEncMiscParameterFrameRate *)m->data;
        fr->framerate = (1 << 16) | 120; /* 120 / 1 = 120 fps */

        VABufferID h264_fps_buf = VA_INVALID_ID;
        status = ctx.vtable->vaCreateBuffer(&ctx, context_id, VAEncMiscParameterBufferType,
                                            sizeof(h264_fps_mem), 1, h264_fps_mem, &h264_fps_buf);
        assert(status == VA_STATUS_SUCCESS);
        status = ctx.vtable->vaRenderPicture(&ctx, context_id, &h264_fps_buf, 1);
        assert(status == VA_STATUS_SUCCESS);
        assert(h264_encoder_get_fps(h264_c->h264_enc) == 120);
        ctx.vtable->vaDestroyBuffer(&ctx, h264_fps_buf);

        /* Switch back to 60 fps */
        fr->framerate = (1 << 16) | 60;
        status = ctx.vtable->vaCreateBuffer(&ctx, context_id, VAEncMiscParameterBufferType,
                                            sizeof(h264_fps_mem), 1, h264_fps_mem, &h264_fps_buf);
        assert(status == VA_STATUS_SUCCESS);
        status = ctx.vtable->vaRenderPicture(&ctx, context_id, &h264_fps_buf, 1);
        assert(status == VA_STATUS_SUCCESS);
        assert(h264_encoder_get_fps(h264_c->h264_enc) == 60);
        ctx.vtable->vaDestroyBuffer(&ctx, h264_fps_buf);
    }

    /* (c) HEVC dynamic bitrate switching via VAEncMiscParameterTypeRateControl */
    {
        uint8_t hevc_misc_mem[sizeof(VAEncMiscParameterBuffer) + sizeof(VAEncMiscParameterRateControl)];
        memset(hevc_misc_mem, 0, sizeof(hevc_misc_mem));
        VAEncMiscParameterBuffer *m = (VAEncMiscParameterBuffer *)hevc_misc_mem;
        m->type = VAEncMiscParameterTypeRateControl;
        VAEncMiscParameterRateControl *rc = (VAEncMiscParameterRateControl *)m->data;
        rc->bits_per_second = 15000000;
        rc->target_percentage = 100;
        rc->initial_qp = 20;

        VABufferID hevc_rc_buf = VA_INVALID_ID;
        status = ctx.vtable->vaCreateBuffer(&ctx, hevc_context_id, VAEncMiscParameterBufferType,
                                            sizeof(hevc_misc_mem), 1, hevc_misc_mem, &hevc_rc_buf);
        assert(status == VA_STATUS_SUCCESS);
        status = ctx.vtable->vaRenderPicture(&ctx, hevc_context_id, &hevc_rc_buf, 1);
        assert(status == VA_STATUS_SUCCESS);
        assert(hevc_encoder_get_bitrate(hevc_c->hevc_enc) == 15000000);
        assert(hevc_encoder_get_qp(hevc_c->hevc_enc) == 20);
        ctx.vtable->vaDestroyBuffer(&ctx, hevc_rc_buf);

        /* Switch bitrate to 5 Mbps */
        rc->bits_per_second = 5000000;
        rc->target_percentage = 100;
        rc->initial_qp = 26;
        status = ctx.vtable->vaCreateBuffer(&ctx, hevc_context_id, VAEncMiscParameterBufferType,
                                            sizeof(hevc_misc_mem), 1, hevc_misc_mem, &hevc_rc_buf);
        assert(status == VA_STATUS_SUCCESS);
        status = ctx.vtable->vaRenderPicture(&ctx, hevc_context_id, &hevc_rc_buf, 1);
        assert(status == VA_STATUS_SUCCESS);
        assert(hevc_encoder_get_bitrate(hevc_c->hevc_enc) == 5000000);
        assert(hevc_encoder_get_qp(hevc_c->hevc_enc) == 26);
        ctx.vtable->vaDestroyBuffer(&ctx, hevc_rc_buf);
    }

    /* (d) HEVC dynamic framerate switching via VAEncMiscParameterTypeFrameRate */
    {
        uint8_t hevc_fps_mem[sizeof(VAEncMiscParameterBuffer) + sizeof(VAEncMiscParameterFrameRate)];
        memset(hevc_fps_mem, 0, sizeof(hevc_fps_mem));
        VAEncMiscParameterBuffer *m = (VAEncMiscParameterBuffer *)hevc_fps_mem;
        m->type = VAEncMiscParameterTypeFrameRate;
        VAEncMiscParameterFrameRate *fr = (VAEncMiscParameterFrameRate *)m->data;
        fr->framerate = (1 << 16) | 120;

        VABufferID hevc_fps_buf = VA_INVALID_ID;
        status = ctx.vtable->vaCreateBuffer(&ctx, hevc_context_id, VAEncMiscParameterBufferType,
                                            sizeof(hevc_fps_mem), 1, hevc_fps_mem, &hevc_fps_buf);
        assert(status == VA_STATUS_SUCCESS);
        status = ctx.vtable->vaRenderPicture(&ctx, hevc_context_id, &hevc_fps_buf, 1);
        assert(status == VA_STATUS_SUCCESS);
        assert(hevc_encoder_get_fps(hevc_c->hevc_enc) == 120);
        ctx.vtable->vaDestroyBuffer(&ctx, hevc_fps_buf);

        /* Switch back to 60 fps */
        fr->framerate = (1 << 16) | 60;
        status = ctx.vtable->vaCreateBuffer(&ctx, hevc_context_id, VAEncMiscParameterBufferType,
                                            sizeof(hevc_fps_mem), 1, hevc_fps_mem, &hevc_fps_buf);
        assert(status == VA_STATUS_SUCCESS);
        status = ctx.vtable->vaRenderPicture(&ctx, hevc_context_id, &hevc_fps_buf, 1);
        assert(status == VA_STATUS_SUCCESS);
        assert(hevc_encoder_get_fps(hevc_c->hevc_enc) == 60);
        ctx.vtable->vaDestroyBuffer(&ctx, hevc_fps_buf);
    }
    printf("[PASS] Dynamic bitrate and framerate switching verified for H.264 and HEVC\n");

    /* 14. Test Quality Range / Level Presets and Max Frame Size Constraints */
    {
        VAConfigAttrib q_attrib = { .type = VAConfigAttribEncQualityRange, .value = 0 };
        status = ctx.vtable->vaGetConfigAttributes(&ctx, VAProfileH264Main, VAEntrypointEncSlice, &q_attrib, 1);
        assert(status == VA_STATUS_SUCCESS);
        assert(q_attrib.value == 7 && "Driver must report 7 quality levels (1=Quality, 4=Balanced, 7=Speed)");

        /* (a) H.264 Quality Level (speed preset = 7) */
        uint8_t q_mem[sizeof(VAEncMiscParameterBuffer) + sizeof(VAEncMiscParameterBufferQualityLevel)];
        memset(q_mem, 0, sizeof(q_mem));
        VAEncMiscParameterBuffer *m = (VAEncMiscParameterBuffer *)q_mem;
        m->type = VAEncMiscParameterTypeQualityLevel;
        VAEncMiscParameterBufferQualityLevel *ql = (VAEncMiscParameterBufferQualityLevel *)m->data;
        ql->quality_level = 7;
        VABufferID q_buf = VA_INVALID_ID;
        status = ctx.vtable->vaCreateBuffer(&ctx, context_id, VAEncMiscParameterBufferType,
                                            sizeof(q_mem), 1, q_mem, &q_buf);
        assert(status == VA_STATUS_SUCCESS);
        status = ctx.vtable->vaRenderPicture(&ctx, context_id, &q_buf, 1);
        assert(status == VA_STATUS_SUCCESS);
        assert(h264_encoder_get_quality_level(h264_c->h264_enc) == 7);
        ctx.vtable->vaDestroyBuffer(&ctx, q_buf);

        /* (b) H.264 Max Frame Size */
        uint8_t mfs_mem[sizeof(VAEncMiscParameterBuffer) + sizeof(VAEncMiscParameterBufferMaxFrameSize)];
        memset(mfs_mem, 0, sizeof(mfs_mem));
        m = (VAEncMiscParameterBuffer *)mfs_mem;
        m->type = VAEncMiscParameterTypeMaxFrameSize;
        VAEncMiscParameterBufferMaxFrameSize *mfs = (VAEncMiscParameterBufferMaxFrameSize *)m->data;
        mfs->max_frame_size = 2500000;
        VABufferID mfs_buf = VA_INVALID_ID;
        status = ctx.vtable->vaCreateBuffer(&ctx, context_id, VAEncMiscParameterBufferType,
                                            sizeof(mfs_mem), 1, mfs_mem, &mfs_buf);
        assert(status == VA_STATUS_SUCCESS);
        status = ctx.vtable->vaRenderPicture(&ctx, context_id, &mfs_buf, 1);
        assert(status == VA_STATUS_SUCCESS);
        assert(h264_encoder_get_max_frame_size(h264_c->h264_enc) == 2500000);
        ctx.vtable->vaDestroyBuffer(&ctx, mfs_buf);

        /* (c) HEVC Quality Level (speed preset = 7) */
        memset(q_mem, 0, sizeof(q_mem));
        m = (VAEncMiscParameterBuffer *)q_mem;
        m->type = VAEncMiscParameterTypeQualityLevel;
        ql = (VAEncMiscParameterBufferQualityLevel *)m->data;
        ql->quality_level = 7;
        status = ctx.vtable->vaCreateBuffer(&ctx, hevc_context_id, VAEncMiscParameterBufferType,
                                            sizeof(q_mem), 1, q_mem, &q_buf);
        assert(status == VA_STATUS_SUCCESS);
        status = ctx.vtable->vaRenderPicture(&ctx, hevc_context_id, &q_buf, 1);
        assert(status == VA_STATUS_SUCCESS);
        assert(hevc_encoder_get_quality_level(hevc_c->hevc_enc) == 7);
        ctx.vtable->vaDestroyBuffer(&ctx, q_buf);

        /* (d) HEVC Max Frame Size */
        memset(mfs_mem, 0, sizeof(mfs_mem));
        m = (VAEncMiscParameterBuffer *)mfs_mem;
        m->type = VAEncMiscParameterTypeMaxFrameSize;
        mfs = (VAEncMiscParameterBufferMaxFrameSize *)m->data;
        mfs->max_frame_size = 1800000;
        status = ctx.vtable->vaCreateBuffer(&ctx, hevc_context_id, VAEncMiscParameterBufferType,
                                            sizeof(mfs_mem), 1, mfs_mem, &mfs_buf);
        assert(status == VA_STATUS_SUCCESS);
        status = ctx.vtable->vaRenderPicture(&ctx, hevc_context_id, &mfs_buf, 1);
        assert(status == VA_STATUS_SUCCESS);
        assert(hevc_encoder_get_max_frame_size(hevc_c->hevc_enc) == 1800000);
        ctx.vtable->vaDestroyBuffer(&ctx, mfs_buf);
    }
    printf("[PASS] Quality level presets and max frame size constraints verified for H.264 and HEVC\n");

    /* Destroy HEVC parameter buffers and context */
    ctx.vtable->vaDestroyBuffer(&ctx, seq_buf_id);
    ctx.vtable->vaDestroyBuffer(&ctx, pic_buf_id);
    ctx.vtable->vaDestroyBuffer(&ctx, slice_buf_id);
    ctx.vtable->vaDestroyBuffer(&ctx, misc_buf_id);
    ctx.vtable->vaDestroyBuffer(&ctx, hevc_coded_buf_id);
    ctx.vtable->vaDestroyContext(&ctx, hevc_context_id);
    ctx.vtable->vaDestroyConfig(&ctx, hevc_config_id);

    /* Clean up images */
    ctx.vtable->vaDestroyImage(&ctx, image1.image_id);
    ctx.vtable->vaDestroyImage(&ctx, image2.image_id);
    ctx.vtable->vaDestroyBuffer(&ctx, coded_buf_id);
    ctx.vtable->vaDestroyContext(&ctx, context_id);
    ctx.vtable->vaDestroySurfaces(&ctx, surfaces, 2);
    ctx.vtable->vaDestroyConfig(&ctx, config_id);
    ctx.vtable->vaTerminate(&ctx);

    printf("=== All VA-API Driver Tests Passed Successfully! ===\n");
    return 0;
}
