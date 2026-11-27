/*
 * test_sync_timeout.c - regression test: a GPU fence-wait timeout must not
 * surface from vaSyncSurface() as VA_STATUS_ERROR_OPERATION_FAILED.
 *
 * BC250_GPU_TIMEOUT_MS=1 makes every fence-wait budget shorter than one
 * 1080p frame's GPU time, so the encoder's own wait times out on every frame
 * (and degrades that frame, as designed). vaSyncSurface() used to wait on the
 * same fence again with the same 1 ms budget and report the second timeout as
 * a hard failure, which ffmpeg/OBS turn into "Input/output error" and abort
 * the encode. The surface must instead be released once the GPU has actually
 * finished with it (bounded wait), and the coded buffer must still hold a
 * frame. Needs no GPU load: the short budget alone triggers it.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "va_backend.h"

#define FRAMES 30
#define W 1920
#define H 1088

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0); /* keep the [FAIL] lines if an assert aborts */
    printf("=== BC-250 vaSyncSurface fence-timeout regression test ===\n");

    /* Must be set before the driver reads them (context creation / first wait). */
    setenv("BC250_GPU_TIMEOUT_MS", "1", 1);
    setenv("BC250_H264_BACKEND", "compute", 1);

    struct VADriverContext ctx;
    struct VADriverVTable vtable;
    memset(&ctx, 0, sizeof(ctx));
    memset(&vtable, 0, sizeof(vtable));
    ctx.vtable = &vtable;

    int major = 0, minor = 0;
    if (bc250_Initialize(&ctx, &major, &minor) != VA_STATUS_SUCCESS) {
        printf("[SKIP] Vulkan initialization skipped (no compatible GPU in test environment)\n");
        return 0;
    }

    VAConfigAttrib attrib = { .type = VAConfigAttribRTFormat, .value = VA_RT_FORMAT_YUV420 };
    VAConfigID config = VA_INVALID_ID;
    VAStatus st = vtable.vaCreateConfig(&ctx, VAProfileH264Main, VAEntrypointEncSlice, &attrib, 1, &config);
    assert(st == VA_STATUS_SUCCESS);

    VASurfaceID surface = VA_INVALID_ID;
    st = vtable.vaCreateSurfaces(&ctx, W, H, VA_RT_FORMAT_YUV420, 1, &surface);
    assert(st == VA_STATUS_SUCCESS);

    VAContextID context = VA_INVALID_ID;
    st = vtable.vaCreateContext(&ctx, config, W, H, VA_PROGRESSIVE, &surface, 1, &context);
    assert(st == VA_STATUS_SUCCESS);

    int failed_syncs = 0, empty_frames = 0;
    for (int i = 0; i < FRAMES; i++) {
        VABufferID coded = VA_INVALID_ID, pic_buf = VA_INVALID_ID;
        st = vtable.vaCreateBuffer(&ctx, context, VAEncCodedBufferType, W * H * 3 / 2, 1, NULL, &coded);
        assert(st == VA_STATUS_SUCCESS);

        VAEncPictureParameterBufferH264 pic;
        memset(&pic, 0, sizeof(pic));
        pic.coded_buf = coded;
        pic.pic_init_qp = 26;
        pic.pic_fields.bits.idr_pic_flag = (i == 0);
        st = vtable.vaCreateBuffer(&ctx, context, VAEncPictureParameterBufferType, sizeof(pic), 1, &pic, &pic_buf);
        assert(st == VA_STATUS_SUCCESS);

        assert(vtable.vaBeginPicture(&ctx, context, surface) == VA_STATUS_SUCCESS);
        assert(vtable.vaRenderPicture(&ctx, context, &pic_buf, 1) == VA_STATUS_SUCCESS);
        assert(vtable.vaEndPicture(&ctx, context) == VA_STATUS_SUCCESS);

        st = vtable.vaSyncSurface(&ctx, surface);
        if (st != VA_STATUS_SUCCESS) {
            printf("[FAIL] frame %d: vaSyncSurface returned 0x%x\n", i, st);
            failed_syncs++;
        }

        void *p = NULL;
        assert(vtable.vaMapBuffer(&ctx, coded, &p) == VA_STATUS_SUCCESS && p);
        const VACodedBufferSegment *seg = (const VACodedBufferSegment *)p;
        if (seg->size == 0) empty_frames++;
        vtable.vaUnmapBuffer(&ctx, coded);

        vtable.vaDestroyBuffer(&ctx, pic_buf);
        vtable.vaDestroyBuffer(&ctx, coded);
    }

    vtable.vaDestroyContext(&ctx, context);
    vtable.vaDestroySurfaces(&ctx, &surface, 1);
    vtable.vaDestroyConfig(&ctx, config);
    vtable.vaTerminate(&ctx);

    printf("[INFO] %d frames, %d failed vaSyncSurface, %d empty coded buffers\n",
           FRAMES, failed_syncs, empty_frames);
    assert(failed_syncs == 0 && "fence timeout must not fail vaSyncSurface");
    assert(empty_frames == 0 && "every frame must still produce a bitstream");
    printf("[PASS] fence timeouts degrade frames instead of failing vaSyncSurface\n");
    return 0;
}
