/* bc250-vulkan-encode-stopgap v0.4.0 - https://github.com/Shalasere/bc250-vulkan-encode-stopgap */
/*
 * rc_bench.c - off-board rate-control response harness (docs/backlog.md D1)
 *
 * docs/backlog.md D1: on the board, doubling the requested bitrate (15M ->
 * 31M, 2560x1440, testsrc2, gop=120) moved this encoder's mean QP by only
 * ~0.4 on every path (H.264, HEVC CPU, HEVC GPU), while libx264 spent the
 * extra budget as +6.3 dB. This drives h264_encoder_encode_raw() /
 * hevc_encoder_encode_raw() directly - no VA-API, no ffmpeg, no GPU, no
 * board - so the question "does rate_control.c itself respond to bitrate,
 * or does something stomp it back?" can be answered with nothing but the C
 * library on a dev machine, per docs/performance-measurement.md's "measure
 * with the real path" rule and CLAUDE.md's cavlc_bench precedent.
 *
 * Two things this harness can do, both against the SAME real rate
 * controller (rate_control.c, unmodified):
 *
 *   sweep   - create the encoder at bitrate B, hevc/h264_encoder_set_rc_mode
 *             (RC_VBR), encode N frames. This is exactly what a caller that
 *             only ever resends bitrate (VAEncSequenceParameterBufferH264/
 *             HEVC.bits_per_second, VAEncMiscParameterRateControl.
 *             bits_per_second) produces - rc_estimate_base_qp() runs once at
 *             the real target and rc_get_frame_qp()'s P/I loop walks from
 *             there every frame.
 *
 *   stomp   - identical to sweep, but before EVERY frame this also calls
 *             hevc_encoder_set_qp()/h264_encoder_set_qp() with a fixed QP
 *             (--stomp=N). This is what va_backend.c's bc250_RenderPicture()
 *             does for real when VAEncPictureParameterBufferH264/HEVC.
 *             pic_init_qp or VAEncMiscParameterRateControl.initial_qp is
 *             nonzero: it calls *_encoder_set_qp(), which unconditionally
 *             overwrites BOTH rc.base_qp and rc.current_qp (see that
 *             function's body in encoder_h264.c/encoder_h265.c - unlike
 *             *_encoder_set_bitrate(), it has no "did this actually change"
 *             guard). VAEncPictureParameterBufferType is a per-frame VA-API
 *             buffer by construction (it carries frame_num/poc/reference
 *             lists, so it cannot be reused across frames) - so if a real
 *             caller ever sets a nonzero, roughly bitrate-INsensitive QP
 *             hint in it, that call happens every single frame. `stomp`
 *             reproduces exactly that, so its effect on the bitrate-delta
 *             can be measured without needing ffmpeg or a VA-API stack
 *             off-board.
 *
 * Content: --input=<file.nv12> reads real frames (NV12, WxH, no padding,
 * looping if short) - same convention as tools/hevc_host_repro.c's
 * BC250_HOSTREPRO_INPUT, so `ffmpeg -f lavfi -i testsrc2=size=WxH:rate=N
 * -pix_fmt nv12 -f rawvideo out.nv12` reproduces the EXACT content D1's
 * board run used. Without --input, a synthetic frame-dependent pattern
 * (moving) is used instead - clearly weaker evidence, printed as such.
 *
 * BC250_RC_NOMINAL_DRAIN=1 is set internally (not left to the caller) for
 * every run: rc_update_stats() otherwise drains its leaky bucket by
 * target_bitrate * REAL CLOCK_MONOTONIC elapsed seconds
 * (docs/DEVLOG.md §16), and this harness's frame loop runs far faster than
 * the 60fps the content was generated at - every rc_update_stats() call
 * would see ~0 elapsed time, clamp to the 1ms floor, and drain a tiny
 * fraction of what a real 60fps session would, which would make the
 * buffer-fullness dynamics (and therefore the P/I feedback) an artifact of
 * how fast THIS harness's for-loop runs rather than a property of
 * rate_control.c. The nominal drain (rc_update_stats()'s documented,
 * test-only escape hatch - see tests/test_encode.c's own use of it) pins
 * the drain to the fixed per-frame quota instead, which is exactly the
 * leaky-bucket behaviour a real, perfectly-scheduled 60fps session would
 * produce. Must be set before the first non-CQP rc_update_stats() call -
 * rate_control.c latches the env var into a static on first use.
 */
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "encoder_h264.h"
#include "encoder_h265.h"
#include "bitstream.h"

/* Create an output file with its mode spelled out. fopen(..., "wb") asks for
 * 0666 and leaves the rest to the umask, which is what CodeQL's
 * cpp/world-writable-file-creation reports. */
static FILE *fopen_wb(const char *path)
{
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) return NULL;
    FILE *f = fdopen(fd, "wb");
    if (!f) close(fd);
    return f;
}

/* Same generator shape as tools/hevc_host_repro.c, kept local so this file
 * has no dependency on that tool. Only patterns 2/3 (frame-dependent) are
 * useful here - a still image gives rate control nothing to adapt to. */
static void fill(uint8_t *y, uint8_t *uv, int w, int h, int pat, int frame) {
    for (int j = 0; j < h; j++)
        for (int i = 0; i < w; i++) {
            int v;
            switch (pat) {
            case 0:  v = 128; break;
            case 1:  v = ((i / 8) & 1) ? 200 : 40; break;
            case 2:  v = (i + j + frame * 3) & 0xFF; break;
            default: v = (((i * 73 + j * 151 + frame * 37) >> 3) * 2654435761u) >> 24; break;
            }
            y[j * w + i] = (uint8_t)v;
        }
    for (int j = 0; j < h / 2; j++)
        for (int i = 0; i + 1 < w; i += 2) {
            int v = (pat == 0) ? 128 : (128 + ((i / 16 + j / 16) & 1) * 40);
            uv[j * w + i]     = (uint8_t)v;
            uv[j * w + i + 1] = (uint8_t)(255 - v);
        }
}

typedef struct {
    int      qp;
    long     bytes;
} frame_stat_t;

typedef struct {
    const char *label;
    uint32_t bitrate;
    int      frames;
    double   mean_qp, min_qp, max_qp;
    double   mean_bytes;
    double   implied_mbps;
    long     failed;
} run_result_t;

static void summarize(run_result_t *r, frame_stat_t *fs, int n, double fps) {
    double qsum = 0, bsum = 0;
    r->min_qp = 1e9; r->max_qp = -1e9;
    for (int i = 0; i < n; i++) {
        qsum += fs[i].qp;
        bsum += (double)fs[i].bytes;
        if (fs[i].qp < r->min_qp) r->min_qp = fs[i].qp;
        if (fs[i].qp > r->max_qp) r->max_qp = fs[i].qp;
    }
    r->frames = n;
    r->mean_qp = qsum / n;
    r->mean_bytes = bsum / n;
    r->implied_mbps = (bsum * 8.0 / 1e6) / ((double)n / fps);
}

/* ---- H.264 ------------------------------------------------------------- */

static int run_h264(uint32_t w, uint32_t h, double fps, int frames, int gop,
                     uint32_t bitrate, int pattern, FILE *input_file,
                     int stomp_qp, run_result_t *out, int verbose, FILE *dump) {
    h264_encoder_t *enc = h264_encoder_create(NULL, w, h, (uint32_t)fps, bitrate, PROFILE_BASELINE);
    if (!enc) { fprintf(stderr, "h264_encoder_create failed\n"); return -1; }
    h264_encoder_set_rc_mode(enc, RC_VBR);
    h264_encoder_set_gop_size(enc, (uint32_t)gop);
    /* set_bitrate() no-ops unless the value actually changes vs create()'s -
     * call it anyway so this harness matches what va_backend.c really does
     * (SeqParam/MiscParam bitrate always flows through *_set_bitrate()). */
    h264_encoder_set_bitrate(enc, bitrate);

    uint8_t *y = malloc((size_t)w * h);
    uint8_t *uv = malloc((size_t)w * (h / 2));
    uint8_t *bs = malloc((size_t)w * h * 4 + 65536);
    frame_stat_t *fs = calloc((size_t)frames, sizeof(*fs));
    if (!y || !uv || !bs || !fs) { fprintf(stderr, "oom\n"); return -1; }

    long failed = 0;
    for (int k = 0; k < frames; k++) {
        if (input_file) {
            size_t ysz = (size_t)w * h, uvsz = (size_t)w * (h / 2);
            if (fread(y, 1, ysz, input_file) != ysz || fread(uv, 1, uvsz, input_file) != uvsz) {
                rewind(input_file);
                if (fread(y, 1, ysz, input_file) != ysz || fread(uv, 1, uvsz, input_file) != uvsz) {
                    fprintf(stderr, "input too short for one frame\n"); return -1;
                }
            }
        } else {
            fill(y, uv, (int)w, (int)h, pattern, k);
        }
        if (stomp_qp > 0) {
            /* Simulates a per-frame VAEncPictureParameterBufferH264.
             * pic_init_qp / VAEncMiscParameterRateControl.initial_qp that
             * va_backend.c's bc250_RenderPicture() would forward straight
             * into h264_encoder_set_qp() - see this file's top comment. */
            h264_encoder_set_qp(enc, stomp_qp);
        }
        int n = h264_encoder_encode_raw(enc, y, (int)w, uv, (int)w, bs, (size_t)w * h * 4 + 65536);
        if (n <= 0) { failed++; fs[k].qp = -1; fs[k].bytes = 0; continue; }
        fs[k].qp = h264_encoder_get_qp(enc);
        fs[k].bytes = n;
        if (dump) fwrite(bs, 1, (size_t)n, dump);
        if (verbose) printf("  [h264 %uM] frame %3d: qp=%d bytes=%d\n", bitrate / 1000000, k, fs[k].qp, n);
    }

    /* Drop failed frames (qp=-1 sentinel) from the summary rather than
     * letting them silently drag the mean toward 0. */
    frame_stat_t *ok = malloc((size_t)frames * sizeof(*ok));
    int nok = 0;
    for (int k = 0; k < frames; k++) if (fs[k].qp >= 0) ok[nok++] = fs[k];
    if (nok == 0) { fprintf(stderr, "every frame failed\n"); return -1; }
    summarize(out, ok, nok, fps);
    out->failed = failed;

    free(ok); free(fs); free(y); free(uv); free(bs);
    h264_encoder_destroy(enc);
    return 0;
}

/* ---- HEVC --------------------------------------------------------------*/

static int run_hevc(uint32_t w, uint32_t h, double fps, int frames, int gop,
                     uint32_t bitrate, int pattern, FILE *input_file,
                     int stomp_qp, run_result_t *out, int verbose, FILE *dump) {
    hevc_encoder_t *enc = hevc_encoder_create(NULL, w, h, (uint32_t)fps, bitrate);
    if (!enc) { fprintf(stderr, "hevc_encoder_create failed\n"); return -1; }
    hevc_encoder_set_rc_mode(enc, RC_VBR);
    hevc_encoder_set_gop_size(enc, (uint32_t)gop);
    hevc_encoder_set_bitrate(enc, bitrate);

    uint8_t *y = malloc((size_t)w * h);
    uint8_t *uv = malloc((size_t)w * (h / 2));
    size_t cw = (size_t)((w + 15) / 16 * 16), chh = (size_t)((h + 15) / 16 * 16);
    size_t cap = cw * chh * 3 + (1u << 20);
    uint8_t *bs = malloc(cap);
    frame_stat_t *fs = calloc((size_t)frames, sizeof(*fs));
    if (!y || !uv || !bs || !fs) { fprintf(stderr, "oom\n"); return -1; }

    long failed = 0;
    for (int k = 0; k < frames; k++) {
        if (input_file) {
            size_t ysz = (size_t)w * h, uvsz = (size_t)w * (h / 2);
            if (fread(y, 1, ysz, input_file) != ysz || fread(uv, 1, uvsz, input_file) != uvsz) {
                rewind(input_file);
                if (fread(y, 1, ysz, input_file) != ysz || fread(uv, 1, uvsz, input_file) != uvsz) {
                    fprintf(stderr, "input too short for one frame\n"); return -1;
                }
            }
        } else {
            fill(y, uv, (int)w, (int)h, pattern, k);
        }
        if (stomp_qp > 0) {
            hevc_encoder_set_qp(enc, stomp_qp);
        }
        int n = hevc_encoder_encode_raw(enc, y, (int)w, uv, (int)w, bs, cap);
        if (n <= 0) { failed++; fs[k].qp = -1; fs[k].bytes = 0; continue; }
        fs[k].qp = hevc_encoder_get_qp(enc);
        fs[k].bytes = n;
        if (dump) fwrite(bs, 1, (size_t)n, dump);
        if (verbose) printf("  [hevc %uM] frame %3d: qp=%d bytes=%d\n", bitrate / 1000000, k, fs[k].qp, n);
    }

    frame_stat_t *ok = malloc((size_t)frames * sizeof(*ok));
    int nok = 0;
    for (int k = 0; k < frames; k++) if (fs[k].qp >= 0) ok[nok++] = fs[k];
    if (nok == 0) { fprintf(stderr, "every frame failed\n"); return -1; }
    summarize(out, ok, nok, fps);
    out->failed = failed;

    free(ok); free(fs); free(y); free(uv); free(bs);
    hevc_encoder_destroy(enc);
    return 0;
}

static void print_result(const run_result_t *r) {
    printf("  %-28s qp_mean=%6.2f qp_range=[%.0f,%.0f] bytes/frame=%9.1f implied=%7.2f Mbps%s\n",
           r->label, r->mean_qp, r->min_qp, r->max_qp, r->mean_bytes, r->implied_mbps,
           r->failed ? " (SOME FRAMES FAILED)" : "");
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "usage: %s <h264|hevc> [options]\n"
        "  --res=WxH         default 2560x1440\n"
        "  --fps=N            default 60\n"
        "  --frames=N         default 150\n"
        "  --gop=N            default 120\n"
        "  --bitrates=B1,B2   default 15000000,31000000 (bps)\n"
        "  --pattern=0|1|2|3  default 2 (synthetic; ignored with --input)\n"
        "  --input=file.nv12  real NV12 frames (WxH, no padding), loops if short\n"
        "  --stomp=QP         call *_set_qp(QP) every frame before encoding it -\n"
        "                     simulates a per-frame VA-API pic_init_qp/initial_qp\n"
        "                     override. 0 (default) = off, pure rate control.\n"
        "  --out=PREFIX       dump each bitrate's full bitstream to\n"
        "                     PREFIX_<bps>.264/.hevc, for an external PSNR check.\n"
        "  --verbose          print every frame's qp/bytes\n",
        argv0);
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(argv[0]); return 2; }
    const char *codec = argv[1];
    if (strcmp(codec, "h264") != 0 && strcmp(codec, "hevc") != 0) { usage(argv[0]); return 2; }

    uint32_t w = 2560, h = 1440;
    double fps = 60.0;
    int frames = 150, gop = 120, pattern = 2, stomp_qp = 0, verbose = 0;
    char bitrates_str[256] = "15000000,31000000";
    char input_path[512] = "";
    char out_prefix[400] = "";

    for (int i = 2; i < argc; i++) {
        const char *a = argv[i];
        if (!strncmp(a, "--res=", 6)) {
            unsigned uw, uh;
            if (sscanf(a + 6, "%ux%u", &uw, &uh) == 2) { w = uw; h = uh; }
        } else if (!strncmp(a, "--fps=", 6)) fps = atof(a + 6);
        else if (!strncmp(a, "--frames=", 9)) frames = atoi(a + 9);
        else if (!strncmp(a, "--gop=", 6)) gop = atoi(a + 6);
        else if (!strncmp(a, "--bitrates=", 11)) snprintf(bitrates_str, sizeof(bitrates_str), "%s", a + 11);
        else if (!strncmp(a, "--pattern=", 10)) pattern = atoi(a + 10);
        else if (!strncmp(a, "--input=", 8)) snprintf(input_path, sizeof(input_path), "%s", a + 8);
        else if (!strncmp(a, "--stomp=", 8)) stomp_qp = atoi(a + 8);
        else if (!strncmp(a, "--out=", 6)) snprintf(out_prefix, sizeof(out_prefix), "%s", a + 6);
        else if (!strcmp(a, "--verbose")) verbose = 1;
        else { fprintf(stderr, "unknown option '%s'\n", a); usage(argv[0]); return 2; }
    }

    /* See this file's top comment: must be set before the first non-CQP
     * rc_update_stats() call, which latches it into a static. */
    setenv("BC250_RC_NOMINAL_DRAIN", "1", 1);

    uint32_t bitrates[8]; int nb = 0;
    {
        char *s = strdup(bitrates_str), *tok = strtok(s, ",");
        while (tok && nb < 8) { bitrates[nb++] = (uint32_t)strtoul(tok, NULL, 10); tok = strtok(NULL, ","); }
        free(s);
    }
    if (nb < 2) { fprintf(stderr, "need at least 2 --bitrates values\n"); return 2; }

    FILE *input_file = NULL;
    if (input_path[0]) {
        input_file = fopen(input_path, "rb");
        if (!input_file) { perror(input_path); return 1; }
    }

    printf("# rc_bench: codec=%s res=%ux%u fps=%.0f frames=%d gop=%d stomp=%d content=%s\n",
           codec, w, h, fps, frames, gop, stomp_qp,
           input_file ? input_path : (pattern == 2 ? "synthetic-ramp" : "synthetic"));

    run_result_t results[8];
    for (int i = 0; i < nb; i++) {
        char label[64];
        snprintf(label, sizeof(label), "%uM (%u bps)", bitrates[i] / 1000000, bitrates[i]);
        results[i].bitrate = bitrates[i];
        results[i].label = strdup(label);

        FILE *dump = NULL;
        if (out_prefix[0]) {
            char path[512];
            snprintf(path, sizeof(path), "%s_%u.%s", out_prefix, bitrates[i], !strcmp(codec, "h264") ? "264" : "hevc");
            dump = fopen_wb(path);
            if (!dump) { perror(path); return 1; }
        }

        int rc;
        if (input_file) rewind(input_file);
        if (!strcmp(codec, "h264"))
            rc = run_h264(w, h, fps, frames, gop, bitrates[i], pattern, input_file, stomp_qp, &results[i], verbose, dump);
        else
            rc = run_hevc(w, h, fps, frames, gop, bitrates[i], pattern, input_file, stomp_qp, &results[i], verbose, dump);
        if (dump) fclose(dump);
        if (rc != 0) { fprintf(stderr, "run at bitrate %u failed\n", bitrates[i]); return 1; }
        print_result(&results[i]);
    }

    printf("\n# deltas vs first bitrate:\n");
    for (int i = 1; i < nb; i++) {
        double ratio = (double)bitrates[i] / (double)bitrates[0];
        printf("  %s -> %s (%.2fx bitrate): d_qp=%+.2f  d_bytes/frame=%+.1f  d_dB_est=n/a (no PSNR here)\n",
               results[0].label, results[i].label, ratio,
               results[i].mean_qp - results[0].mean_qp,
               results[i].mean_bytes - results[0].mean_bytes);
    }

    if (input_file) fclose(input_file);
    return 0;
}
