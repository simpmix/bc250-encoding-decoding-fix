/* bc250-encoding-decoding-fix v0.4.3 - https://github.com/simpmix/bc250-encoding-decoding-fix */
/*
 * test_encode.c - End-to-end H.264 bitstream syntax & decode test
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>

#include "encoder_h264.h"
#include "bitstream.h"

/*
 * Regression test for the Intra16x16 luma DC transpose bug fixed in commit
 * d95b840 ("fix(encoder): transpose Intra16x16 luma DC array before CAVLC -
 * closes remaining luma corruption"). This is a pure-math, no-GPU-context
 * test: h264_encoder_create(NULL, ...) below skips encode_mb_i16x16()
 * entirely (falls back to an all-zero-residual MB - see
 * h264_encoder_encode_frame's "No GPU residual data available" branch), so
 * the full end-to-end test in main() never exercises this code path at all;
 * this is why neither this file's original test nor any CAVLC unit test
 * caught the bug.
 *
 * Reproduces the investigation's "DC shuffle" methodology: 16 distinct,
 * known flat DC values (one per luma 4x4 sub-block), spatially varying
 * enough that any misassignment of a value to the wrong (row,col) is
 * immediately detectable. Rather than hardcoding expected Hadamard-domain
 * numbers (which would make this test a tautological re-statement of
 * luma_dc_hadamard()'s own arithmetic), this checks the actual INVARIANT
 * the fix establishes: the array handed to CAVLC (dc_out) must be the
 * transpose of the raw quantized Hadamard output (dc_out_pretranspose).
 * Before commit d95b840, dc_out was a straight copy of the pretranspose
 * array (no transpose at all) - for this asymmetric fixture, that fails
 * the check below exactly the way it failed against a real decoder on
 * real, spatially-varying content.
 */
static void test_intra16_dc_transpose(void) {
    printf("[test_encode] Regression: Intra16x16 luma DC transpose (commit d95b840)...\n");

    int dc_flat[16] = { 16,  28,  40,  52,
                         64,  76,  88, 100,
                        112, 124, 136, 148,
                        160, 172, 184, 196 };
    int dc_in[4][4];
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            dc_in[r][c] = dc_flat[r * 4 + c];

    int dc_out[16], pretranspose[16];
    h264_intra16_luma_dc_transform(dc_in, 26 /* qp */, dc_out, pretranspose);

    /* Sanity check on the fixture itself: confirm the pre-transpose array
     * really is asymmetric, otherwise a missing transpose (the bug) would
     * pass the check below vacuously and this wouldn't be a valid
     * regression test. */
    bool asymmetric = false;
    for (int r = 0; r < 4 && !asymmetric; r++) {
        for (int c = r + 1; c < 4; c++) {
            if (pretranspose[r * 4 + c] != pretranspose[c * 4 + r]) { asymmetric = true; break; }
        }
    }
    assert(asymmetric && "fixture produced a symmetric pre-transpose array - not a valid regression test");

    /* The actual regression check: dc_out must be the transpose of
     * pretranspose, i.e. dc_out[row][col] == pretranspose[col][row]. */
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            assert(dc_out[r * 4 + c] == pretranspose[c * 4 + r]);
        }
    }

    printf("[test_encode] Intra16x16 luma DC transpose verified (dc_out == transpose(pretranspose)).\n");
}

static void test_rate_control_cqp_and_vbr(void) {
    printf("[test_encode] Testing Rate Control CQP and VBR modes...\n");

    rate_control_t rc;
    rc_init(&rc, RC_CQP, 5000000, 60.0, 1920, 1080);
    rc.current_qp = 23;
    assert(rc_get_frame_qp(&rc, 0) == 23);
    assert(rc_get_frame_qp(&rc, 100000) == 23);

    /* In CQP mode, update stats must not change buffer fullness */
    int64_t buf = rc.buffer_fullness;
    rc_update_stats(&rc, 50000);
    assert(rc.buffer_fullness == buf);

    /* Test VBR mode complexity adjustment */
    rc_init(&rc, RC_VBR, 5000000, 60.0, 1920, 1080);
    int qp1 = rc_get_frame_qp(&rc, 1000);
    /* High motion frame (SAD ratio > 1.3): QP should increase */
    int qp2 = rc_get_frame_qp(&rc, 5000);
    assert(qp2 >= qp1);

    /* Test nominal drain: rc_update_stats drains exactly target_bits_per_frame for recording */
    rate_control_t rc_cbr;
    rc_init(&rc_cbr, RC_CBR, 6000000, 60.0, 1920, 1080);
    assert(rc_cbr.target_bits_per_frame == 100000);
    int64_t initial_fullness = rc_cbr.buffer_fullness;
    /* Consuming exactly target_bits_per_frame leaves buffer fullness unchanged */
    rc_update_stats(&rc_cbr, 100000);
    assert(rc_cbr.buffer_fullness == initial_fullness);

    /* Test h264_encoder_set_rc_mode and h264_encoder_set_qp */
    h264_encoder_t *enc = h264_encoder_create(NULL, 1920, 1080, 60, 5000000, PROFILE_BASELINE);
    assert(enc != NULL);
    h264_encoder_set_rc_mode(enc, RC_CQP);
    h264_encoder_set_qp(enc, 28);
    h264_encoder_set_rc_mode(enc, RC_VBR);
    h264_encoder_set_rc_mode(enc, RC_LOW_LATENCY);
    h264_encoder_destroy(enc);

    printf("[test_encode] Rate Control CQP and VBR tests passed.\n");
}

static void test_setenv(const char *name, const char *val) {
#ifdef _WIN32
    char buf[256];
    snprintf(buf, sizeof(buf), "%s=%s", name, val ? val : "");
    _putenv(buf);
#else
    if (val) {
        setenv(name, val, 1);
    } else {
        unsetenv(name);
    }
#endif
}

static void test_multislice_parallel_encoding(void) {
    printf("[test_encode] Testing Multi-threaded Multi-slice Parallel Encoding...\n");

    /* Save host environment variable if present so unit tests run cleanly isolated */
    const char *orig_slice_env = getenv("BC250_SLICES_PER_FRAME");
    char saved_slice_env[64] = {0};
    if (orig_slice_env) {
        strncpy(saved_slice_env, orig_slice_env, sizeof(saved_slice_env) - 1);
    }
    test_setenv("BC250_SLICES_PER_FRAME", NULL);

    const uint32_t width = 320;
    const uint32_t height = 240;
    h264_encoder_t *enc = h264_encoder_create(NULL, width, height, 30, 2000000, PROFILE_BASELINE);
    if (!enc) {
        fprintf(stderr, "[test_encode] ERROR: h264_encoder_create failed in multislice test\n");
    }
    assert(enc != NULL);

    /* Test getter and setter */
    int def_slices = h264_encoder_get_num_slices(enc);
    if (def_slices != 1) {
        fprintf(stderr, "[test_encode] ERROR: expected default num_slices == 1, got %d\n", def_slices);
    }
    assert(def_slices == 1);

    h264_encoder_set_num_slices(enc, 4);
    int set_slices = h264_encoder_get_num_slices(enc);
    if (set_slices != 4) {
        fprintf(stderr, "[test_encode] ERROR: expected set num_slices == 4, got %d\n", set_slices);
    }
    assert(set_slices == 4);

    const size_t y_size = (size_t)width * height;
    const size_t uv_size = (size_t)width * (height / 2);
    uint8_t *y_plane = malloc(y_size);
    uint8_t *uv_plane = malloc(uv_size);
    assert(y_plane && uv_plane);

    memset(y_plane, 128, y_size);
    memset(uv_plane, 128, uv_size);

    const size_t out_cap = width * height * 4;
    uint8_t *out_buf = malloc(out_cap);
    assert(out_buf != NULL);

    /* Frame 0: IDR with 4 slices */
    int written = h264_encoder_encode_raw(enc, y_plane, width, uv_plane, width, out_buf, out_cap);
    if (written <= 0) {
        fprintf(stderr, "[test_encode] ERROR: Frame 0 multi-slice encode failed (written=%d)\n", written);
    }
    assert(written > 0);

    int idr_slice_count = 0;
    int sps_count = 0;
    int pps_count = 0;
    int aud_count = 0;

    for (int p = 0; p < written - 4; p++) {
        if (out_buf[p] == 0x00 && out_buf[p+1] == 0x00 &&
            out_buf[p+2] == 0x00 && out_buf[p+3] == 0x01) {
            uint8_t nal_type = out_buf[p+4] & 0x1F;
            if (nal_type == 9) aud_count++;
            if (nal_type == 7) sps_count++;
            if (nal_type == 8) pps_count++;
            if (nal_type == 5) idr_slice_count++;
        }
    }

    if (aud_count != 1 || sps_count != 1 || pps_count != 1 || idr_slice_count != 4) {
        fprintf(stderr, "[test_encode] ERROR: Frame 0 NAL counts: AUD=%d (exp 1), SPS=%d (exp 1), PPS=%d (exp 1), IDR=%d (exp 4)\n",
                aud_count, sps_count, pps_count, idr_slice_count);
    }
    assert(aud_count == 1);
    assert(sps_count == 1);
    assert(pps_count == 1);
    assert(idr_slice_count == 4 && "Expected exactly 4 IDR slice NALUs for 4-slice frame");
    printf("[test_encode] Frame 0 (IDR): successfully encoded 4 slices in parallel (AUD=%d, SPS=%d, PPS=%d, IDR_slices=%d)\n",
           aud_count, sps_count, pps_count, idr_slice_count);

    /* Frame 1: P-frame with motion */
    for (size_t i = 0; i < y_size; i++) y_plane[i] = (uint8_t)((y_plane[i] + 16) & 0xFF);

    written = h264_encoder_encode_raw(enc, y_plane, width, uv_plane, width, out_buf, out_cap);
    if (written <= 0) {
        fprintf(stderr, "[test_encode] ERROR: Frame 1 multi-slice encode failed (written=%d)\n", written);
    }
    assert(written > 0);

    int p_slice_count = 0;
    for (int p = 0; p < written - 4; p++) {
        if (out_buf[p] == 0x00 && out_buf[p+1] == 0x00 &&
            out_buf[p+2] == 0x00 && out_buf[p+3] == 0x01) {
            uint8_t nal_type = out_buf[p+4] & 0x1F;
            if (nal_type == 1) p_slice_count++;
        }
    }

    if (p_slice_count != 4) {
        fprintf(stderr, "[test_encode] ERROR: Frame 1 NAL counts: P_slices=%d (exp 4)\n", p_slice_count);
    }
    assert(p_slice_count == 4 && "Expected exactly 4 P slice NALUs for 4-slice frame");
    printf("[test_encode] Frame 1 (P): successfully encoded 4 slices in parallel (P_slices=%d)\n",
           p_slice_count);

    /* Test programmatic reduction back to 1 slice */
    h264_encoder_set_num_slices(enc, 1);
    assert(h264_encoder_get_num_slices(enc) == 1);
    for (size_t i = 0; i < y_size; i++) y_plane[i] = (uint8_t)((y_plane[i] + 8) & 0xFF);
    written = h264_encoder_encode_raw(enc, y_plane, width, uv_plane, width, out_buf, out_cap);
    assert(written > 0);
    int p_slice_single = 0;
    for (int p = 0; p < written - 4; p++) {
        if (out_buf[p] == 0x00 && out_buf[p+1] == 0x00 &&
            out_buf[p+2] == 0x00 && out_buf[p+3] == 0x01) {
            uint8_t nal_type = out_buf[p+4] & 0x1F;
            if (nal_type == 1) p_slice_single++;
        }
    }
    assert(p_slice_single == 1 && "Expected exactly 1 P slice NALU after resetting num_slices to 1");

    /* Test BC250_SLICES_PER_FRAME environment variable override on creation */
    test_setenv("BC250_SLICES_PER_FRAME", "2");
    h264_encoder_t *enc_env = h264_encoder_create(NULL, width, height, 30, 2000000, PROFILE_BASELINE);
    assert(enc_env != NULL);
    assert(h264_encoder_get_num_slices(enc_env) == 2);
    h264_encoder_destroy(enc_env);

    /* Restore host environment */
    if (orig_slice_env) {
        test_setenv("BC250_SLICES_PER_FRAME", saved_slice_env);
    } else {
        test_setenv("BC250_SLICES_PER_FRAME", NULL);
    }

    free(y_plane);
    free(uv_plane);
    free(out_buf);
    h264_encoder_destroy(enc);

    printf("[test_encode] Multi-threaded multi-slice parallel encoding verified.\n");
}

/*
 * Regression test for Issue #49: Steam Link green screen with H.264 multi-slice encoding.
 * Verifies that h264_sanitize_i16_mode and h264_sanitize_chroma_mode strictly enforce
 * ITU-T H.264 Section 8.3.3 and 8.3.4 neighbor availability across slice boundaries
 * and image edges, preventing decoders from aborting with:
 *   "top block unavailable for requested intra mode"
 *   "left block unavailable for requested intra mode"
 */
static void test_slice_boundary_intra_sanitization(void) {
    printf("[test_encode] Testing slice boundary intra mode sanitization (Issue #49)...\n");

    /* Case 1: MB 0,0 or first MB in any slice (!top_avail, !left_avail) */
    for (int mode = 0; mode < 4; mode++) {
        assert(h264_sanitize_i16_mode(mode, false, false) == H264_I16x16_DC);
        assert(h264_sanitize_chroma_mode(mode, false, false) == H264_CHROMA_DC);
    }

    /* Case 2: Top row of a slice (!top_avail, left_avail) */
    /* VERT (0) and PLANE (3) are illegal without top neighbor -> fallback to HORIZ (1) */
    assert(h264_sanitize_i16_mode(H264_I16x16_VERT, false, true) == H264_I16x16_HORIZ);
    assert(h264_sanitize_i16_mode(H264_I16x16_PLANE, false, true) == H264_I16x16_HORIZ);
    assert(h264_sanitize_i16_mode(H264_I16x16_HORIZ, false, true) == H264_I16x16_HORIZ);
    assert(h264_sanitize_i16_mode(H264_I16x16_DC, false, true) == H264_I16x16_DC);

    assert(h264_sanitize_chroma_mode(H264_CHROMA_VERT, false, true) == H264_CHROMA_HORIZ);
    assert(h264_sanitize_chroma_mode(H264_CHROMA_PLANE, false, true) == H264_CHROMA_HORIZ);
    assert(h264_sanitize_chroma_mode(H264_CHROMA_HORIZ, false, true) == H264_CHROMA_HORIZ);
    assert(h264_sanitize_chroma_mode(H264_CHROMA_DC, false, true) == H264_CHROMA_DC);

    /* Case 3: Left column of picture/slice (top_avail, !left_avail) */
    /* HORIZ (1) and PLANE (3) are illegal without left neighbor -> fallback to VERT (0) */
    assert(h264_sanitize_i16_mode(H264_I16x16_HORIZ, true, false) == H264_I16x16_VERT);
    assert(h264_sanitize_i16_mode(H264_I16x16_PLANE, true, false) == H264_I16x16_VERT);
    assert(h264_sanitize_i16_mode(H264_I16x16_VERT, true, false) == H264_I16x16_VERT);
    assert(h264_sanitize_i16_mode(H264_I16x16_DC, true, false) == H264_I16x16_DC);

    assert(h264_sanitize_chroma_mode(H264_CHROMA_HORIZ, true, false) == H264_CHROMA_VERT);
    assert(h264_sanitize_chroma_mode(H264_CHROMA_PLANE, true, false) == H264_CHROMA_VERT);
    assert(h264_sanitize_chroma_mode(H264_CHROMA_VERT, true, false) == H264_CHROMA_VERT);
    assert(h264_sanitize_chroma_mode(H264_CHROMA_DC, true, false) == H264_CHROMA_DC);

    /* Case 4: Interior macroblock (top_avail, left_avail) -> all modes preserved */
    for (int mode = 0; mode < 4; mode++) {
        assert(h264_sanitize_i16_mode(mode, true, true) == mode);
        assert(h264_sanitize_chroma_mode(mode, true, true) == mode);
    }

    printf("[test_encode] Slice boundary intra mode sanitization passed.\n");
}

int main(void) {
    /* Isolate test execution from host environment variables */
    const char *orig_slices = getenv("BC250_SLICES_PER_FRAME");
    char saved_slices[64] = {0};
    if (orig_slices) strncpy(saved_slices, orig_slices, sizeof(saved_slices) - 1);
    const char *orig_cabac = getenv("BC250_USE_CABAC");
    char saved_cabac[64] = {0};
    if (orig_cabac) strncpy(saved_cabac, orig_cabac, sizeof(saved_cabac) - 1);

    test_setenv("BC250_SLICES_PER_FRAME", NULL);
    test_setenv("BC250_USE_CABAC", NULL);

    test_intra16_dc_transpose();
    test_rate_control_cqp_and_vbr();
    test_multislice_parallel_encoding();
    test_slice_boundary_intra_sanitization();

    printf("[test_encode] Starting H.264 end-to-end bitstream encoding test...\n");

    const uint32_t width = 1920;
    const uint32_t height = 1080;
    const uint32_t fps = 60;
    const uint32_t bitrate = 5000000; /* 5 Mbps */

    h264_encoder_t *enc = h264_encoder_create(NULL, width, height, fps, bitrate, PROFILE_BASELINE);
    assert(enc != NULL);
    printf("[test_encode] Created 1080p60 H.264 Baseline encoder.\n");

    const size_t out_cap = width * height * 2;
    uint8_t *out_buf = malloc(out_cap);
    assert(out_buf != NULL);

    FILE *f_stream = fopen("bc250_test_stream.h264", "wb");
    if (!f_stream) {
        f_stream = fopen("/tmp/bc250_test_stream.h264", "wb");
    }
    if (!f_stream) {
        fprintf(stderr, "[test_encode] Warning: could not open output stream file for writing, running in-memory checks\n");
    }

    uint8_t *y_plane = malloc(width * height);
    uint8_t *uv_plane = malloc(width * height / 2);
    assert(y_plane != NULL && uv_plane != NULL);

    /* Encode 30 frames (1 GOP): 1 IDR frame followed by 29 P-frames */
    const int num_frames = 30;
    size_t total_bytes = 0;

    for (int i = 0; i < num_frames; i++) {
        /* Generate synthetic animated test pattern: moving gradient bar and cross-hatch */
        int shift = (i * 32) % (int)width;
        for (uint32_t r = 0; r < height; r++) {
            for (uint32_t c = 0; c < width; c++) {
                y_plane[r * width + c] = (uint8_t)(((c + shift) * 255 / width) ^ ((r * 128) / height));
            }
        }
        for (uint32_t r = 0; r < height / 2; r++) {
            for (uint32_t c = 0; c < width; c++) {
                uv_plane[r * width + c] = (uint8_t)(128 + ((r * 64) / (height / 2)));
            }
        }

        int written = h264_encoder_encode_raw(enc, y_plane, width, uv_plane, width, out_buf, out_cap);
        if (written <= 0) {
            fprintf(stderr, "[test_encode] ERROR: Frame %02d failed to encode (written=%d)\n", i, written);
        }
        assert(written > 0);

        /* Verify 4-byte start code at the beginning of each frame (AUD) */
        assert(out_buf[0] == 0x00);
        assert(out_buf[1] == 0x00);
        assert(out_buf[2] == 0x00);
        assert(out_buf[3] == 0x01);
        assert(out_buf[4] == 0x09); /* NAL type 9 (AUD) */

        if (i == 0) {
            /* IDR frame must contain SPS (type 7) and PPS (type 8) */
            bool has_sps = false;
            bool has_pps = false;
            bool has_idr_slice = false;

            for (int p = 0; p < written - 4; p++) {
                if (out_buf[p] == 0x00 && out_buf[p+1] == 0x00 &&
                    out_buf[p+2] == 0x00 && out_buf[p+3] == 0x01) {
                    uint8_t nal_type = out_buf[p+4] & 0x1F;
                    if (nal_type == 7) has_sps = true;
                    if (nal_type == 8) has_pps = true;
                    if (nal_type == 5) has_idr_slice = true;
                }
            }
            if (!has_sps || !has_pps || !has_idr_slice) {
                fprintf(stderr, "[test_encode] ERROR: Frame 0 NAL check failed: SPS=%d, PPS=%d, IDR=%d, written=%d\n",
                        has_sps, has_pps, has_idr_slice, written);
            }
            assert(has_sps && "Frame 0 missing SPS NAL unit");
            assert(has_pps && "Frame 0 missing PPS NAL unit");
            assert(has_idr_slice && "Frame 0 missing IDR Slice NAL unit");
            printf("[test_encode] Frame %02d (IDR): %d bytes (AUD, SPS, PPS, IDR-Slice verified)\n", i, written);
        } else {
            /* P-frame must contain non-IDR slice (type 1) */
            bool has_p_slice = false;
            for (int p = 0; p < written - 4; p++) {
                if (out_buf[p] == 0x00 && out_buf[p+1] == 0x00 &&
                    out_buf[p+2] == 0x00 && out_buf[p+3] == 0x01) {
                    uint8_t nal_type = out_buf[p+4] & 0x1F;
                    if (nal_type == 1) has_p_slice = true;
                }
            }
            if (!has_p_slice) {
                fprintf(stderr, "[test_encode] ERROR: Frame %02d missing P-slice, written=%d\n", i, written);
            }
            assert(has_p_slice && "P-frame missing Slice NAL unit");
            if (i < 5 || i == num_frames - 1) {
                printf("[test_encode] Frame %02d (P):   %d bytes (AUD, P-Slice verified)\n", i, written);
            }
        }

        if (f_stream) {
            fwrite(out_buf, 1, (size_t)written, f_stream);
        }
        total_bytes += (size_t)written;
    }

    if (f_stream) {
        fclose(f_stream);
    }
    free(y_plane);
    free(uv_plane);
    free(out_buf);
    h264_encoder_destroy(enc);

    /* Restore host environment variables */
    if (orig_slices) test_setenv("BC250_SLICES_PER_FRAME", saved_slices);
    if (orig_cabac) test_setenv("BC250_USE_CABAC", saved_cabac);

    printf("[test_encode] Successfully encoded %d frames (%zu total bytes) to bc250_test_stream.h264!\n",
           num_frames, total_bytes);
    printf("[test_encode] ALL TESTS PASSED!\n");

    return 0;
}
