/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * h264dec.c - a standalone driver for the decoder, for testing only.
 *
 *     h264dec <input.264> <output.yuv>
 *
 * Under VA-API the application parses the sequence and picture parameter
 * sets and every slice header, and hands the decoder the results. That is a
 * good deal less code for the driver, but it also means the decoder cannot
 * be run on a file without something doing that parsing - so this does, and
 * only well enough to drive the tests:
 *
 *   - progressive 4:2:0 8-bit, which is what the decoder supports anyway;
 *   - picture order count types 0 and 2;
 *   - the default reference list order, and the modification syntax;
 *   - one slice group.
 *
 * It writes decoded frames in DECODE order, not output order. For an I/P
 * stream the two are the same; when B pictures arrive, this has to grow a
 * reorder buffer, and until then it is the reference decoder's -fps_mode
 * passthrough output that has to match.
 *
 * This file is not part of the driver and is not built into it.
 */
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

#include "decoder_h264.h"
#include "bitreader.h"
#include "h264_dec_tables.h"   /* the two zig-zag scans, for the scaling lists */
#include <time.h>

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

/* Nanoseconds on a clock that does not jump. */
static uint64_t now_seconds(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_nsec;
}
static uint64_t decode_time = 0;

/* The harness runs the decoder with no GPU context, so end_picture keeps the
 * planes instead of uploading them. The upload is still referenced from the
 * object file, so it needs a body to link against; reaching it would mean the
 * null-context path had been lost, which is why it says so rather than
 * returning quietly. */
int gpu_compute_upload_nv12(gpu_context_t *ctx, gpu_image_t *image,
                            gpu_memory_t memory,
                            const uint8_t *y, int ys,
                            const uint8_t *uv, int uvs, int w, int h);
int gpu_compute_upload_nv12(gpu_context_t *ctx, gpu_image_t *image,
                            gpu_memory_t memory,
                            const uint8_t *y, int ys,
                            const uint8_t *uv, int uvs, int w, int h)
{
    (void)ctx; (void)image; (void)memory;
    (void)y; (void)ys; (void)uv; (void)uvs; (void)w; (void)h;
    fprintf(stderr, "h264dec: the decoder tried to upload a surface "
                    "with no GPU context\n");
    abort();
}

/* ------------------------------------------------------- parameter sets */

typedef struct {
    int valid;
    int profile_idc;
    int chroma_format_idc;
    int bit_depth_luma, bit_depth_chroma;
    int log2_max_frame_num;
    int pic_order_cnt_type;
    int log2_max_poc_lsb;
    int delta_pic_order_always_zero;
    int max_num_ref_frames;
    int mb_width, mb_height;
    int crop_left, crop_right, crop_top, crop_bottom;   /* in samples */
    int frame_mbs_only;
    int mb_adaptive;
    int direct_8x8_inference;
    int scaling_present;
    uint8_t scaling4[6][16];
    uint8_t scaling8[6][64];
} sps_t;

typedef struct {
    int valid;
    int sps_id;
    int entropy_coding_mode;
    int pic_order_present;
    int num_ref_idx_default[2];
    int weighted_pred;
    int weighted_bipred_idc;
    int pic_init_qp;
    int chroma_qp_index_offset;
    int second_chroma_qp_index_offset;
    int deblocking_filter_control_present;
    int constrained_intra_pred;
    int redundant_pic_cnt_present;
    int transform_8x8_mode;
    uint8_t scaling4[6][16];
    uint8_t scaling8[6][64];
} pps_t;

static sps_t sps_store[32];
static pps_t pps_store[256];

static void flatten(uint8_t s4[6][16], uint8_t s8[6][64])
{
    for (int i = 0; i < 6; i++) {
        memset(s4[i], 16, 16);
        memset(s8[i], 16, 64);
    }
}

/* Table 7-3 and Table 7-4, in the order they would be sent: zig-zag. */
static const uint8_t def4_intra[16] = {
     6, 13, 13, 20, 20, 20, 28, 28, 28, 28, 32, 32, 32, 37, 37, 42
};
static const uint8_t def4_inter[16] = {
    10, 14, 14, 20, 20, 20, 24, 24, 24, 24, 27, 27, 27, 30, 30, 34
};
static const uint8_t def8_intra[64] = {
     6, 10, 10, 13, 11, 13, 16, 16, 16, 16, 18, 18, 18, 18, 18, 23,
    23, 23, 23, 23, 23, 25, 25, 25, 25, 25, 25, 25, 27, 27, 27, 27,
    27, 27, 27, 27, 29, 29, 29, 29, 29, 29, 29, 31, 31, 31, 31, 31,
    31, 33, 33, 33, 33, 33, 36, 36, 36, 36, 38, 38, 38, 40, 40, 42
};
static const uint8_t def8_inter[64] = {
     9, 13, 13, 15, 13, 15, 17, 17, 17, 17, 19, 19, 19, 19, 19, 21,
    21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 24, 24, 24, 24,
    24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 27, 27, 27, 27, 27,
    27, 28, 28, 28, 28, 28, 30, 30, 30, 30, 32, 32, 32, 33, 33, 35
};

/* ⚠️ Everything below stores a scaling list in RASTER order, because that
 * is how the dequantiser indexes it - and how VA-API hands it over. The
 * bitstream sends it in zig-zag, and so do the tables above. */
static void da_zigzag(const uint8_t *src, uint8_t *dst, int n)
{
    const uint8_t *z = (n == 16) ? h264d_zigzag4 : h264d_zigzag8;
    for (int j = 0; j < n; j++)
        dst[z[j]] = src[j];
}

static void default_list(int i, uint8_t *dst)
{
    if (i < 6)
        da_zigzag(i < 3 ? def4_intra : def4_inter, dst, 16);
    else
        da_zigzag(i == 6 ? def8_intra : def8_inter, dst, 64);
}

/* Clause 7.3.2.1.1.1. Returns 1 when the list asked to be taken from the
 * standard's defaults, 0 when it was sent in full. */
static int scaling_list(br_t *br, uint8_t *list_idx, int n)
{
    uint8_t zz[64];
    int last = 8, next = 8;
    for (int j = 0; j < n; j++) {
        if (next) {
            const int delta = br_read_se(br);
            next = (last + delta + 256) % 256;
            if (j == 0 && next == 0) return 1;
        }
        zz[j] = (uint8_t)(next ? next : last);
        last = zz[j];
    }
    da_zigzag(zz, list_idx, n);
    return 0;
}

/* Table 7-2: what a list that was not sent inherits.
 *
 * Fall-back rule A (a sequence's own matrix, or a picture's when the
 * sequence had none) takes the standard default for the first list of each
 * group and the previous list for the rest. Fall-back rule B (a picture's
 * matrix layered over a sequence's) takes the sequence's list in those
 * same places. `base` is the sequence's set under rule B and NULL under
 * rule A. */
static void scale_missing(int i, uint8_t s4[6][16], uint8_t s8[6][64],
                          const uint8_t (*b4)[16], const uint8_t (*b8)[64])
{
    uint8_t *dst = (i < 6) ? s4[i] : s8[i - 6];
    const int n = (i < 6) ? 16 : 64;
    const int first = (i == 0 || i == 3 || i == 6 || i == 7);

    if (!first) {
        memcpy(dst, (i < 6) ? (const uint8_t *)s4[i - 1]
                            : (const uint8_t *)s8[i - 7], (size_t)n);
        return;
    }
    if (b4)
        memcpy(dst, (i < 6) ? (const uint8_t *)b4[i]
                            : (const uint8_t *)b8[i - 6], (size_t)n);
    else
        default_list(i, dst);
}

static int read_sps(const uint8_t *rbsp, size_t n)
{
    br_t br;
    br_init(&br, rbsp, n);
    br_read(&br, 8);                        /* nal header already stripped? no */

    sps_t s;
    memset(&s, 0, sizeof(s));
    flatten(s.scaling4, s.scaling8);

    s.profile_idc = (int)br_read(&br, 8);
    br_read(&br, 8);                        /* constraint flags + reserved */
    br_read(&br, 8);                        /* level_idc */
    const int id = (int)br_read_ue(&br);
    if (id < 0 || id >= 32) return -1;

    s.chroma_format_idc = 1;
    s.bit_depth_luma = s.bit_depth_chroma = 8;
    if (s.profile_idc == 100 || s.profile_idc == 110 || s.profile_idc == 122
        || s.profile_idc == 244 || s.profile_idc == 44 || s.profile_idc == 83
        || s.profile_idc == 86 || s.profile_idc == 118 || s.profile_idc == 128) {
        s.chroma_format_idc = (int)br_read_ue(&br);
        if (s.chroma_format_idc == 3) br_read1(&br);
        s.bit_depth_luma = 8 + (int)br_read_ue(&br);
        s.bit_depth_chroma = 8 + (int)br_read_ue(&br);
        br_read1(&br);                      /* qpprime_y_zero_transform_bypass */
        s.scaling_present = (int)br_read1(&br);
        if (s.scaling_present) {
            for (int i = 0; i < 8; i++) {
                if (!br_read1(&br)) {
                    scale_missing(i, s.scaling4, s.scaling8, NULL, NULL);
                    continue;
                }
                uint8_t *dst = (i < 6) ? s.scaling4[i] : s.scaling8[i - 6];
                if (scaling_list(&br, dst, (i < 6) ? 16 : 64))
                    default_list(i, dst);
            }
        }
    }

    s.log2_max_frame_num = 4 + (int)br_read_ue(&br);
    s.pic_order_cnt_type = (int)br_read_ue(&br);
    if (s.pic_order_cnt_type == 0) {
        s.log2_max_poc_lsb = 4 + (int)br_read_ue(&br);
    } else if (s.pic_order_cnt_type == 1) {
        s.delta_pic_order_always_zero = (int)br_read1(&br);
        br_read_se(&br);
        br_read_se(&br);
        const int c = (int)br_read_ue(&br);
        for (int i = 0; i < c; i++) br_read_se(&br);
    }
    s.max_num_ref_frames = (int)br_read_ue(&br);
    br_read1(&br);                          /* gaps_in_frame_num_allowed */
    s.mb_width = 1 + (int)br_read_ue(&br);
    const int alt = 1 + (int)br_read_ue(&br);
    s.frame_mbs_only = (int)br_read1(&br);
    if (!s.frame_mbs_only) s.mb_adaptive = (int)br_read1(&br);
    s.mb_height = alt * (2 - s.frame_mbs_only);
    s.direct_8x8_inference = (int)br_read1(&br);

    /* frame_cropping. The decoder works in whole macroblocks; what the
     * picture is actually meant to be is this much smaller, and a reference
     * decoder writes out the cropped picture. Comparing the uncropped one
     * would fail on every size that is not a multiple of sixteen. */
    if (br_read1(&br)) {
        const int sx = 2;                       /* 4:2:0 horizontal units */
        const int sy = 2 * (2 - s.frame_mbs_only);
        s.crop_left = sx * (int)br_read_ue(&br);
        s.crop_right = sx * (int)br_read_ue(&br);
        s.crop_top = sy * (int)br_read_ue(&br);
        s.crop_bottom = sy * (int)br_read_ue(&br);
    }

    s.valid = 1;
    sps_store[id] = s;
    return id;
}

static int read_pps(const uint8_t *rbsp, size_t n)
{
    br_t br;
    br_init(&br, rbsp, n);
    br_read(&br, 8);

    pps_t p;
    memset(&p, 0, sizeof(p));

    const int id = (int)br_read_ue(&br);
    if (id < 0 || id >= 256) return -1;
    p.sps_id = (int)br_read_ue(&br);
    if (p.sps_id < 0 || p.sps_id >= 32 || !sps_store[p.sps_id].valid) return -1;

    memcpy(p.scaling4, sps_store[p.sps_id].scaling4, sizeof(p.scaling4));
    memcpy(p.scaling8, sps_store[p.sps_id].scaling8, sizeof(p.scaling8));

    p.entropy_coding_mode = (int)br_read1(&br);
    p.pic_order_present = (int)br_read1(&br);
    const int groups = 1 + (int)br_read_ue(&br);
    if (groups != 1) { fprintf(stderr, "more than one slice group\n"); return -1; }
    p.num_ref_idx_default[0] = 1 + (int)br_read_ue(&br);
    p.num_ref_idx_default[1] = 1 + (int)br_read_ue(&br);
    p.weighted_pred = (int)br_read1(&br);
    p.weighted_bipred_idc = (int)br_read(&br, 2);
    p.pic_init_qp = 26 + br_read_se(&br);
    br_read_se(&br);                        /* pic_init_qs */
    p.chroma_qp_index_offset = br_read_se(&br);
    p.second_chroma_qp_index_offset = p.chroma_qp_index_offset;
    p.deblocking_filter_control_present = (int)br_read1(&br);
    p.constrained_intra_pred = (int)br_read1(&br);
    p.redundant_pic_cnt_present = (int)br_read1(&br);

    if (br_more_rbsp_data(&br)) {
        p.transform_8x8_mode = (int)br_read1(&br);
        if (br_read1(&br)) {                /* pic_scaling_matrix_present */
            /* Rule B when the sequence sent a matrix of its own, rule A
             * when it did not. The picture's lists start as copies of the
             * sequence's, so a list that is neither sent nor a first-of-
             * group already holds the right thing. */
            const sps_t *sq = &sps_store[p.sps_id];
            const int rule_b = sq->scaling_present;
            const int how_many = 6 + (p.transform_8x8_mode ? 2 : 0);
            for (int i = 0; i < how_many; i++) {
                if (!br_read1(&br)) {
                    scale_missing(i, p.scaling4, p.scaling8,
                                  rule_b ? sq->scaling4 : NULL,
                                  rule_b ? sq->scaling8 : NULL);
                    continue;
                }
                uint8_t *dst = (i < 6) ? p.scaling4[i] : p.scaling8[i - 6];
                if (scaling_list(&br, dst, (i < 6) ? 16 : 64))
                    default_list(i, dst);
            }
        }
        p.second_chroma_qp_index_offset = br_read_se(&br);
    }

    p.valid = 1;
    pps_store[id] = p;
    return id;
}

/* One picture in the reference list. */
typedef struct {
    uint32_t surface;
    int poc;
    int frame_num;
} ref_t;

/* Clause 8.2.5: mark the picture that has just been decoded, then add it.
 *
 * Either the operations its own slice header carried, or - when it carried
 * none - the sliding window, which drops the short-term reference with the
 * smallest PicNum once the store is full. Never both. */
static void mark_refs(ref_t *refs, int *n_refs, int max_rif,
                  uint32_t surface, int poc, int frame_num,
                  int log2_max_frame_num,
                  const int *op, const int *val, int n_op)
{
    const int max_pn = 1 << log2_max_frame_num;

    if (n_op > 0) {
        for (int k = 0; k < n_op; k++) {
            if (op[k] == 5) { *n_refs = 0; continue; }
            if (op[k] != 1) continue;         /* long-term: refused earlier */
            int pn = frame_num - (val[k] + 1);
            if (pn < 0) pn += max_pn;
            for (int c = 0; c < *n_refs; c++) {
                const int its = (refs[c].frame_num > frame_num)
                              ? refs[c].frame_num - max_pn : refs[c].frame_num;
                if (its == pn) {
                    for (int j = c; j + 1 < *n_refs; j++) refs[j] = refs[j + 1];
                    (*n_refs)--;
                    break;
                }
            }
        }
    } else if (*n_refs >= max_rif && *n_refs > 0) {
        /* Sliding window: out goes the smallest PicNum, which is the
         * oldest - the last of a list kept newest first. */
        (*n_refs)--;
    }

    for (int k = 15; k > 0; k--) refs[k] = refs[k - 1];
    refs[0].surface = surface;
    refs[0].poc = poc;
    refs[0].frame_num = frame_num;
    if (*n_refs < 16) (*n_refs)++;
}

/* Pictures held back so they can be written in display order. */
typedef struct {
    int poc;
    int order;                 /* decode order, to break ties stably */
    size_t n;
    uint8_t *data;
} output_t;

static output_t coda[4096];
static int n_coda = 0;

/* Copy one frame out, cropped the way the stream asks for, into the queue. */
static void write_out(FILE *fo, const h264d_frame_t *f, const sps_t *sp)
{
    (void)fo;
    const int w = sp->mb_width * 16 - sp->crop_left - sp->crop_right;
    const int h = sp->mb_height * 16 - sp->crop_top - sp->crop_bottom;
    const size_t n = (size_t)w * h + 2 * (size_t)(w / 2) * (h / 2);
    if (n_coda >= 4096) return;

    uint8_t *d = malloc(n);
    if (!d) return;
    size_t o = 0;
    for (int y = 0; y < h; y++) {
        memcpy(d + o, f->y + (size_t)(y + sp->crop_top) * f->stride_y
                            + sp->crop_left, (size_t)w);
        o += (size_t)w;
    }
    for (int p = 0; p < 2; p++) {
        const uint8_t *pl = p ? f->cr : f->cb;
        for (int y = 0; y < h / 2; y++) {
            memcpy(d + o, pl + (size_t)(y + sp->crop_top / 2) * f->stride_c
                              + sp->crop_left / 2, (size_t)(w / 2));
            o += (size_t)(w / 2);
        }
    }
    coda[n_coda].poc = f->poc;
    coda[n_coda].order = n_coda;
    coda[n_coda].n = n;
    coda[n_coda].data = d;
    n_coda++;
}

/* Display order is picture order count order. */
static void drain(FILE *fo)
{
    for (int a = 0; a + 1 < n_coda; a++)
        for (int b = 0; b + 1 < n_coda - a; b++)
            if (coda[b].poc > coda[b + 1].poc
                || (coda[b].poc == coda[b + 1].poc
                    && coda[b].order > coda[b + 1].order)) {
                output_t t = coda[b]; coda[b] = coda[b + 1]; coda[b + 1] = t;
            }
    for (int k = 0; k < n_coda; k++) {
        fwrite(coda[k].data, 1, coda[k].n, fo);
        free(coda[k].data);
        coda[k].data = NULL;
    }
    n_coda = 0;
}

/* ---------------------------------------------------------- the driver */


#define FLUSH_SLICE()                                                   \
    do {                                                                  \
        if (n_ready > 0) {                                               \
            const uint64_t t0 = now_seconds();                                 \
            const int r = h264_decoder_slices(dec, ready, n_ready);     \
            decode_time += now_seconds() - t0;                            \
            const int mb0 = r ? ready[0].first_mb_diag : 0;              \
            n_ready = 0;                                                 \
            if (r) {                                                      \
                fprintf(stderr, "slice refused (%d) at macroblock %d " \
                                "of picture %d\n", r, mb0, fotogrammi); \
                skipped++;                                                \
                return (r == -2 || r == -3) ? 3 : 4;                      \
            }                                                             \
        }                                                                 \
    } while (0)

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s <input.264> <output.yuv>\n", argv[0]);
        return 2;
    }

    FILE *fi = fopen(argv[1], "rb");
    if (!fi) { perror(argv[1]); return 1; }
    fseek(fi, 0, SEEK_END);
    long len = ftell(fi);
    fseek(fi, 0, SEEK_SET);
    uint8_t *buf = malloc((size_t)len);
    if (!buf || fread(buf, 1, (size_t)len, fi) != (size_t)len) return 1;
    fclose(fi);

    FILE *fo = fopen_wb(argv[2]);
    if (!fo) { perror(argv[2]); return 1; }

    h264_decoder_t *dec = NULL;
    uint8_t *rbsp = malloc((size_t)len);
    int fotogrammi = 0, skipped = 0;

    /* Reference pictures, newest first, as the sliding window keeps them. */
    ref_t refs[16];
    int n_refs = 0;
    uint32_t next_surface = 1;
    int prev_poc_lsb = 0, prev_poc_msb = 0;
    int in_progress = 0;
    uint32_t current_surface = 0;
    int current_reference = 0, current_poc = 0, current_frame_num = 0;
    int n_mmco_expected = 0, mmco_op_expected[32], mmco_val_expected[32];
    /* The slices of the picture being read, decoded together when it ends. */
    h264d_slice_input_t *ready = NULL;
    int n_ready = 0, cap_ready = 0;
    int n_mmco_previous = 0, mmco_op_previous[32], mmco_val_previous[32];

    long i = 0;
    while (i + 3 < len) {
        /* Annex-B start code */
        if (!(buf[i] == 0 && buf[i + 1] == 0
              && (buf[i + 2] == 1 || (buf[i + 2] == 0 && buf[i + 3] == 1)))) {
            i++;
            continue;
        }
        const int sc = (buf[i + 2] == 1) ? 3 : 4;
        long start = i + sc;
        long fine = start;
        while (fine + 3 < len
               && !(buf[fine] == 0 && buf[fine + 1] == 0
                    && (buf[fine + 2] == 1
                        || (buf[fine + 2] == 0 && buf[fine + 3] == 1))))
            fine++;
        if (fine + 3 >= len) fine = len;
        i = fine;

        if (start >= fine) continue;
        const int kind = buf[start] & 0x1f;
        const int nal_ref_idc = (buf[start] >> 5) & 3;

        if (kind == 7) {
            size_t n = br_extract_rbsp(rbsp, (size_t)len, buf + start,
                                       (size_t)(fine - start));
            if (read_sps(rbsp, n) < 0) return 1;
            continue;
        }
        if (kind == 8) {
            size_t n = br_extract_rbsp(rbsp, (size_t)len, buf + start,
                                       (size_t)(fine - start));
            if (read_pps(rbsp, n) < 0) return 1;
            continue;
        }
        if (kind != 1 && kind != 5) continue;

        /* ---- slice header ---- */
        size_t n = br_extract_rbsp(rbsp, (size_t)len, buf + start,
                                   (size_t)(fine - start));
        br_t br;
        br_init(&br, rbsp, n);
        br_read(&br, 8);

        const int first_mb = (int)br_read_ue(&br);
        int st = (int)br_read_ue(&br);
        const int slice_type = st % 5;      /* 0 P, 1 B, 2 I */
        const int pps_id = (int)br_read_ue(&br);
        if (pps_id < 0 || pps_id >= 256 || !pps_store[pps_id].valid) return 1;
        const pps_t *pp = &pps_store[pps_id];
        const sps_t *sp = &sps_store[pp->sps_id];

        if (!h264_decoder_supports(sp->profile_idc, sp->chroma_format_idc,
                                   sp->bit_depth_luma, sp->bit_depth_chroma,
                                   sp->frame_mbs_only, sp->mb_adaptive)) {
            fprintf(stderr, "stream outside what the decoder covers\n");
            return 3;
        }

        const int frame_num = (int)br_read(&br, sp->log2_max_frame_num);
        const int idr = (kind == 5);
        if (idr) br_read_ue(&br);           /* idr_pic_id */

        int poc = 0;
        if (sp->pic_order_cnt_type == 0) {
            const int lsb = (int)br_read(&br, sp->log2_max_poc_lsb);
            const int max = 1 << sp->log2_max_poc_lsb;
            int msb;
            if (idr) { prev_poc_lsb = 0; prev_poc_msb = 0; }
            if (lsb < prev_poc_lsb && prev_poc_lsb - lsb >= max / 2)
                msb = prev_poc_msb + max;
            else if (lsb > prev_poc_lsb && lsb - prev_poc_lsb > max / 2)
                msb = prev_poc_msb - max;
            else
                msb = prev_poc_msb;
            poc = msb + lsb;
            if (nal_ref_idc) { prev_poc_lsb = lsb; prev_poc_msb = msb; }
            if (pp->pic_order_present) br_read_se(&br);
        } else if (sp->pic_order_cnt_type == 2) {
            poc = 2 * frame_num;
        } else {
            fprintf(stderr, "pic_order_cnt_type 1 not implemented\n");
            return 3;
        }

        if (pp->redundant_pic_cnt_present && br_read_ue(&br) != 0) continue;

        h264d_slice_t sl;
        memset(&sl, 0, sizeof(sl));
        sl.type = slice_type;
        sl.first_mb = first_mb;
        sl.num_ref_idx[0] = pp->num_ref_idx_default[0];
        sl.num_ref_idx[1] = pp->num_ref_idx_default[1];

        if (slice_type == 1) sl.direct_spatial_mv_pred = br_read1(&br);
        if (slice_type != 2) {
            if (br_read1(&br)) {            /* num_ref_idx_active_override */
                sl.num_ref_idx[0] = 1 + (int)br_read_ue(&br);
                if (slice_type == 1) sl.num_ref_idx[1] = 1 + (int)br_read_ue(&br);
            }
        }

        /* ---- reference list, newest short-term first (8.2.4.2.1) ---- */
        /* Filled in after begin_picture, which is when the DPB slots the
         * list has to name are settled. */
        for (int k = 0; k < 32; k++) sl.ref_list[0][k] = sl.ref_list[1][k] = 0;

        /* Clause 7.3.3.1, read here and applied once the frame store slots
         * are known. A stream that reorders its list is not exotic: x264
         * does it whenever weighted prediction is on, which is by default. */
        int n_mod[2] = { 0, 0 };
        int mod_idc[2][32], mod_val[2][32];
        const int lists_mod = (slice_type == 1) ? 2 : (slice_type == 0 ? 1 : 0);
        for (int l = 0; l < lists_mod; l++) {
            if (!br_read1(&br)) continue;
            for (;;) {
                const int idc = (int)br_read_ue(&br);
                if (idc == 3 || n_mod[l] >= 32) break;
                const int v = (int)br_read_ue(&br);
                mod_idc[l][n_mod[l]] = idc;
                mod_val[l][n_mod[l]] = v;
                n_mod[l]++;
            }
        }
        /* Clause 7.3.3.2. A reference whose flag is clear takes the neutral
         * weight, which is 1 << denom and not 1: the shift is applied either
         * way, so a weight of 1 would divide the prediction instead of
         * leaving it alone. */
        sl.luma_log2_weight_denom = 0;
        sl.chroma_log2_weight_denom = 0;
        for (int l = 0; l < 2; l++)
            for (int k = 0; k < 32; k++) {
                sl.luma_weight[l][k] = 1;
                sl.luma_offset[l][k] = 0;
                sl.chroma_weight[l][k][0] = 1;
                sl.chroma_weight[l][k][1] = 1;
                sl.chroma_offset[l][k][0] = 0;
                sl.chroma_offset[l][k][1] = 0;
            }

        if ((pp->weighted_pred && slice_type == 0)
            || (pp->weighted_bipred_idc == 1 && slice_type == 1)) {
            sl.luma_log2_weight_denom = (int)br_read_ue(&br);
            sl.chroma_log2_weight_denom = (int)br_read_ue(&br);
            const int ld = sl.luma_log2_weight_denom;
            const int cd = sl.chroma_log2_weight_denom;
            for (int l = 0; l < 2; l++)
                for (int k = 0; k < 32; k++) {
                    sl.luma_weight[l][k] = (int16_t)(1 << ld);
                    sl.chroma_weight[l][k][0] = (int16_t)(1 << cd);
                    sl.chroma_weight[l][k][1] = (int16_t)(1 << cd);
                }
            const int lists = (slice_type == 1) ? 2 : 1;
            for (int l = 0; l < lists; l++) {
                for (int k = 0; k < sl.num_ref_idx[l] && k < 32; k++) {
                    if (br_read1(&br)) {
                        sl.luma_weight[l][k] = (int16_t)br_read_se(&br);
                        sl.luma_offset[l][k] = (int16_t)br_read_se(&br);
                    }
                    if (br_read1(&br)) {
                        for (int j = 0; j < 2; j++) {
                            sl.chroma_weight[l][k][j] = (int16_t)br_read_se(&br);
                            sl.chroma_offset[l][k][j] = (int16_t)br_read_se(&br);
                        }
                    }
                }
            }
        }
        /* Clause 7.3.3.3. The operations are read here and applied below,
         * once the current picture's frame_num is the one they are relative
         * to. */
        int n_mmco = 0;
        int mmco_op[32], mmco_val[32];
        if (nal_ref_idc) {
            if (idr) {
                br_read1(&br);              /* no_output_of_prior_pics */
                br_read1(&br);              /* long_term_reference_flag */
            } else if (br_read1(&br)) {
                for (;;) {
                    const int op = (int)br_read_ue(&br);
                    if (op == 0 || n_mmco >= 32) break;
                    int v = 0;
                    switch (op) {
                    case 1: case 3:
                        v = (int)br_read_ue(&br);       /* difference_of_pic_nums_minus1 */
                        if (op == 3) br_read_ue(&br);   /* long_term_frame_idx */
                        break;
                    case 2:
                        v = (int)br_read_ue(&br);       /* long_term_pic_num */
                        break;
                    case 4:
                        v = (int)br_read_ue(&br);       /* max_long_term_frame_idx_plus1 */
                        break;
                    case 6:
                        v = (int)br_read_ue(&br);       /* long_term_frame_idx */
                        break;
                    case 5:
                        break;                          /* clear everything */
                    default:
                        fprintf(stderr, "marking operation %d "
                                        "unknown\n", op);
                        return 3;
                    }
                    mmco_op[n_mmco] = op;
                    mmco_val[n_mmco] = v;
                    n_mmco++;
                }
            }
        }

        /* ⚠️ Nothing is marked here. These operations belong to the picture
         * this header introduces and run once it has been decoded, in
         * `mark_refs()` below - which is also where the sliding window lives,
         * because a picture uses one or the other and never both. */
        if (first_mb == 0) {
            n_mmco_expected = n_mmco;
            for (int k = 0; k < n_mmco; k++) {
                mmco_op_expected[k] = mmco_op[k];
                mmco_val_expected[k] = mmco_val[k];
            }
        }

        if (pp->entropy_coding_mode && slice_type != 2)
            sl.cabac_init_idc = (int)br_read_ue(&br);
        sl.qpy = pp->pic_init_qp + br_read_se(&br);

        if (pp->deblocking_filter_control_present) {
            sl.disable_deblocking_filter_idc = (int)br_read_ue(&br);
            if (sl.disable_deblocking_filter_idc != 1) {
                sl.alpha_c0_offset = 2 * br_read_se(&br);
                sl.beta_offset = 2 * br_read_se(&br);
            }
        }

        /* ⚠️ In the raw NAL's coordinates, because that is what the
         * decoder is handed - the same thing VA-API's
         * slice_data_bit_offset counts. */
        const int bit_offset = (int)br_raw_offset(buf + start,
                                                  (size_t)(fine - start),
                                                  br.bitpos);

        /* ---- the picture ---- */
        if (!dec) {
            dec = h264_decoder_create(NULL, (uint32_t)(sp->mb_width * 16),
                                      (uint32_t)(sp->mb_height * 16));
            if (!dec) return 1;
        }

        if (first_mb == 0) {
            /* The picture that was being read is complete: decode its
             * slices before anything is said about the picture itself. */
            FLUSH_SLICE();
            if (in_progress && current_reference) {
                mark_refs(refs, &n_refs, sp->max_num_ref_frames, current_surface,
                      current_poc, current_frame_num, sp->log2_max_frame_num,
                      mmco_op_previous, mmco_val_previous, n_mmco_previous);
                current_reference = 0;
            }
            if (in_progress) {
                { const uint64_t te = now_seconds();
                  h264_decoder_end_picture(dec, (gpu_image_t){0}, (gpu_memory_t){0});
                  decode_time += now_seconds() - te; }
                h264d_frame_t *f = h264_decoder_frame_for(dec, current_surface);
                if (f) {
                    write_out(fo, f, sp);
                    fotogrammi++;
                }
            }
            /* ⚠️ An IDR restarts the picture order count at zero, so the
             * queue has to go out before it. Sorting the two groups of
             * pictures together interleaves them: with a group of eight,
             * picture 8 has the same count as picture 0 and lands second.
             * That is also why a nine-picture clip failed where an
             * eight-picture one passed - one group against two. */
            if (idr) {
                drain(fo);
                n_refs = 0;
            }
            current_surface = next_surface++;

            h264d_pic_t pic;
            memset(&pic, 0, sizeof(pic));
            pic.width = sp->mb_width * 16;
            pic.height = sp->mb_height * 16;
            pic.mb_width = sp->mb_width;
            pic.mb_height = sp->mb_height;
            pic.chroma_qp_index_offset = pp->chroma_qp_index_offset;
            pic.second_chroma_qp_index_offset = pp->second_chroma_qp_index_offset;
            pic.pic_init_qp = pp->pic_init_qp;
            pic.num_ref_idx_l0 = pp->num_ref_idx_default[0];
            pic.num_ref_idx_l1 = pp->num_ref_idx_default[1];
            pic.entropy_coding_mode = pp->entropy_coding_mode;
            pic.transform_8x8_mode = pp->transform_8x8_mode;
            pic.constrained_intra_pred = pp->constrained_intra_pred;
            pic.direct_8x8_inference = sp->direct_8x8_inference;
            pic.deblocking_filter_control_present = pp->deblocking_filter_control_present;
            pic.weighted_pred = pp->weighted_pred;
            pic.weighted_bipred_idc = pp->weighted_bipred_idc;
            pic.pic_order_present = pp->pic_order_present;
            memcpy(pic.scaling4, pp->scaling4, sizeof(pic.scaling4));
            memcpy(pic.scaling8, pp->scaling8, sizeof(pic.scaling8));

            h264_decoder_begin_picture(dec, &pic, current_surface, poc, frame_num);

            uint32_t sup[16];
            int pocs[16];
            bool lt[16];
            for (int k = 0; k < n_refs; k++) {
                sup[k] = refs[k].surface;
                pocs[k] = refs[k].poc;
                lt[k] = false;
            }
            h264_decoder_set_references(dec, sup, pocs, lt, n_refs);
            in_progress = 1;
        }

        /* PicNum for a short-term reference, clause 8.2.4.1: a frame_num
         * ahead of the current picture's belongs to the previous cycle, so
         * it counts as negative rather than as a large positive. */
        const int max_pic_num = 1 << sp->log2_max_frame_num;
        int pic_num_di[16];
        for (int k = 0; k < n_refs; k++)
            pic_num_di[k] = (refs[k].frame_num > frame_num)
                          ? refs[k].frame_num - max_pic_num
                          : refs[k].frame_num;

        ref_t list_idx[2][34];
        int n_active[2] = { 0, 0 };

        for (int l = 0; l < (slice_type == 1 ? 2 : 1); l++) {
            int n_ref = sl.num_ref_idx[l];
            if (n_ref > 32) n_ref = 32;
            n_active[l] = n_ref;

            if (slice_type == 0) {
                /* 8.2.4.2.1: descending PicNum, which is the order `refs`
                 * already holds. */
                for (int k = 0; k < n; k++)
                    list_idx[l][k] = refs[k < n_refs ? k : (n_refs > 0 ? n_refs - 1 : 0)];
            } else {
                /* 8.2.4.2.3: by display order. Before this picture, nearest
                 * first; then after it, nearest first. List 1 takes them the
                 * other way round. */
                ref_t before[16], after[16];
                int np = 0, nd = 0;
                for (int k = 0; k < n_refs; k++) {
                    if (refs[k].poc < poc) before[np++] = refs[k];
                    else                   after[nd++] = refs[k];
                }
                for (int a = 0; a + 1 < np; a++)      /* descending POC */
                    for (int b = 0; b + 1 < np - a; b++)
                        if (before[b].poc < before[b + 1].poc) {
                            ref_t t = before[b]; before[b] = before[b + 1]; before[b + 1] = t;
                        }
                for (int a = 0; a + 1 < nd; a++)      /* ascending POC */
                    for (int b = 0; b + 1 < nd - a; b++)
                        if (after[b].poc > after[b + 1].poc) {
                            ref_t t = after[b]; after[b] = after[b + 1]; after[b + 1] = t;
                        }

                ref_t ordered[32];
                int no = 0;
                if (l == 0) {
                    for (int k = 0; k < np && no < 32; k++) ordered[no++] = before[k];
                    for (int k = 0; k < nd && no < 32; k++) ordered[no++] = after[k];
                } else {
                    for (int k = 0; k < nd && no < 32; k++) ordered[no++] = after[k];
                    for (int k = 0; k < np && no < 32; k++) ordered[no++] = before[k];
                }
                for (int k = 0; k < n; k++)
                    list_idx[l][k] = ordered[k < no ? k : (no > 0 ? no - 1 : 0)];
            }

            /* Clause 8.2.4.3.1. Each step names a picture by PicNum, slides
             * the list down from the current index, drops it in, and
             * squeezes out the copy that is now further along.
             *
             * ⚠️ The picture is looked up in the frame store, not in the
             * list being built: the list has just been slid, so searching it
             * finds the copy the slide made. */
            int pred = frame_num;
            int ref_idx = 0;
            for (int k = 0; k < n_mod[l] && ref_idx < n; k++) {
                if (mod_idc[l][k] != 0 && mod_idc[l][k] != 1) {
                    fprintf(stderr, "long-term references, "
                                    "not implemented\n");
                    return 3;
                }
                const int delta = mod_val[l][k] + 1;
                int without_wrap = (mod_idc[l][k] == 0) ? pred - delta : pred + delta;
                if (without_wrap < 0) without_wrap += max_pic_num;
                else if (without_wrap >= max_pic_num) without_wrap -= max_pic_num;
                pred = without_wrap;
                const int pic_num = (without_wrap > frame_num)
                                  ? without_wrap - max_pic_num : without_wrap;

                int found = -1;
                for (int c = 0; c < n_refs; c++)
                    if (pic_num_di[c] == pic_num) { found = c; break; }
                if (found < 0) {
                    fprintf(stderr, "list %d asks for PicNum %d, which is not there\n",
                            l, pic_num);
                    return 3;
                }
                const ref_t chosen = refs[found];

                for (int c = n; c > ref_idx; c--)
                    list_idx[l][c] = list_idx[l][c - 1];
                list_idx[l][ref_idx++] = chosen;
                int n2 = ref_idx;
                for (int c = ref_idx; c <= n; c++)
                    if (list_idx[l][c].surface != chosen.surface)
                        list_idx[l][n2++] = list_idx[l][c];
            }
        }

        /* 8.2.4.2.3: two identical lists of more than one picture would put
         * the same one at index 0 on both sides, and bi-prediction would
         * average a picture with itself. */
        if (slice_type == 1 && n_active[1] > 1 && n_active[0] == n_active[1]) {
            int n_equal = 1;
            for (int k = 0; k < n_active[0]; k++)
                if (list_idx[0][k].surface != list_idx[1][k].surface) { n_equal = 0; break; }
            if (n_equal) {
                ref_t t = list_idx[1][0];
                list_idx[1][0] = list_idx[1][1];
                list_idx[1][1] = t;
            }
        }

        /* The lists name frame store slots, which only exist once
         * begin_picture has run. */
        for (int l = 0; l < 2; l++)
            for (int k = 0; k < n_active[l]; k++) {
                h264d_frame_t *f = h264_decoder_frame_for(dec, list_idx[l][k].surface);
                sl.ref_list[l][k] = f ? (int8_t)h264_decoder_slot_of(dec, f) : 0;
            }

        if (getenv("BC250_H264_W")) {
            fprintf(stderr, "f%d fn%d poc%d type%d nri%d mmco%d nref%d |",
                    fotogrammi, frame_num, poc, slice_type, nal_ref_idc,
                    n_mmco, n_refs);
            fprintf(stderr, " nmod%d/%d nattivo%d/%d |", n_mod[0], n_mod[1],
                    n_active[0], n_active[1]);
            for (int k = 0; k < n_refs; k++)
                fprintf(stderr, " rif(fn%d,poc%d)", refs[k].frame_num, refs[k].poc);
            for (int l = 0; l < 2; l++)
                for (int k = 0; k < n_active[l] && k < 4; k++)
                    fprintf(stderr, " L%d[%d]=slot%d(poc%d)", l, k,
                            sl.ref_list[l][k], list_idx[l][k].poc);
            fprintf(stderr, "\n");
        }

        if (n_ready == cap_ready) {
            const int fresh = cap_ready ? cap_ready * 2 : 16;
            h264d_slice_input_t *p2 = realloc(ready,
                                              (size_t)fresh * sizeof(*p2));
            if (!p2) return 1;
            ready = p2;
            cap_ready = fresh;
        }
        ready[n_ready].slice = sl;
        ready[n_ready].data = buf + start;
        ready[n_ready].size = (size_t)(fine - start);
        ready[n_ready].bit_offset = bit_offset;
        ready[n_ready].first_mb_diag = first_mb;
        n_ready++;

        /* ⚠️ Remembered, not applied: a picture joins the reference list
         * once, when it is finished. Doing it per slice pushed the older
         * references out four times as fast on a picture cut into four. */
        current_reference = nal_ref_idc != 0;
        current_poc = poc;
        current_frame_num = frame_num;
        if (first_mb == 0) {
            n_mmco_previous = n_mmco_expected;
            for (int k = 0; k < n_mmco_expected; k++) {
                mmco_op_previous[k] = mmco_op_expected[k];
                mmco_val_previous[k] = mmco_val_expected[k];
            }
        }
    }

    FLUSH_SLICE();
    if (in_progress && dec) {
        { const uint64_t te = now_seconds();
          h264_decoder_end_picture(dec, (gpu_image_t){0}, (gpu_memory_t){0});
          decode_time += now_seconds() - te; }
        h264d_frame_t *f = h264_decoder_frame_for(dec, current_surface);
        if (f) {
            const sps_t *sp = NULL;
            for (int k = 0; k < 32; k++) if (sps_store[k].valid) { sp = &sps_store[k]; break; }
            if (sp) {
                write_out(fo, f, sp);
                fotogrammi++;
            }
        }
    }

    drain(fo);
    fclose(fo);
    h264_decoder_destroy(dec);
    free(buf);
    free(rbsp);
    printf("%d pictures decoded, %d slices refused\n", fotogrammi, skipped);
    if (fotogrammi > 0) {
        const double ms = (double)decode_time / 1e6;
        fprintf(stderr, "decoding: %.1f ms, %.2f ms/picture, %.1f fps\n",
                ms, ms / fotogrammi, 1000.0 * fotogrammi / ms);
    }
    return skipped ? 5 : 0;
}
