/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * decoder_h265.c - the decoded picture buffer, the reference lists and the
 * slice loop.
 *
 * Everything below the slice header knows nothing about where pictures
 * come from or where they go. This is the piece that does.
 */
#include "decoder_h265.h"
#include "hevc_dec_internal.h"
#include "gpu_compute.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#if defined(__SSE2__) || defined(__x86_64__) || defined(_M_X64)
#include <emmintrin.h>
#include <immintrin.h>
#endif

/* ⚠️ More than the sixteen a stream may keep: a picture being decoded
 * holds its own slot and those it reads until its frame starts the next
 * one, and with several in flight a slot the newest no longer names may
 * still be read by an older one. Four per frame covers its own slot and
 * the references a picture can drop at a time. */
#define IMG_SLOTS (16 + 4 * HEVC_DECODER_FRAMES)

/* One picture being decoded: everything that belongs to a picture while it
 * is decoded - the maps, the parameter sets, the slice header, the threads
 * - so that several can be decoded at once. The pictures themselves, and the
 * references between them, are the decoder's. */
typedef struct {
    hevcd_t d;
    hevc_sps_t sps;
    hevc_pps_t pps;
    hevc_slice_t last_one;
    bool is_open;
    /* The slots this frame holds - the one it writes and those it reads -
     * released when the frame starts its next picture. */
    hevcd_img_t *held[IMG_SLOTS + 16];
    int n_held;
} hevc_frame_t;

struct hevc_decoder {
    hevc_frame_t f[HEVC_DECODER_FRAMES];
    hevcd_progress_t progress;
    /* Which slots are valid, what the caller calls them, and who holds
     * them. Taken by every function that reads or changes any of that. */
    pthread_mutex_t dpb;
    hevcd_img_t buffer[IMG_SLOTS];
    uintptr_t surface_id[IMG_SLOTS];  /* what the caller calls each picture */
    void *gpu;
    int width, height;
};

static void free_img(hevcd_img_t *g)
{
    free(g->lists); g->lists = NULL; g->n_lists = 0;
    free(g->slice_of_ctb); g->slice_of_ctb = NULL; g->n_slice_map = 0;
    for (int i = 0; i < 3; i++) { free(g->plane[i]); g->plane[i] = NULL; }
    free(g->mvf);
    g->mvf = NULL;
    g->n_planes = 0;
    g->n_mvf = 0;
    g->is_valid = false;
}

static hevcd_img_t *find_img(const hevcd_t *d, int poc)
{
    for (int i = 0; i < d->n_buf; i++)
        if (d->buf[i].is_valid && d->buf[i].poc == poc) return &d->buf[i];
    return NULL;
}

/* 8.3.2: which picture a long-term entry names.
 *
 * ⚠️ Without its most significant bits the entry is only the low bits of
 * a count, and it names whichever reference picture has those low bits -
 * the standard forbids there being two. With them it is the whole count,
 * worked out from the current picture's. `skip` is the picture being
 * decoded, which is in the buffer but is nobody's reference yet. */
static const hevcd_img_t *find_lt(const hevcd_t *d, const hevc_slice_t *sl,
                                  int i, const hevcd_img_t *skip)
{
    const int max = 1 << sl->log2_max_poc_lsb;
    int poc = sl->lt_poc_lsb[i];
    if (sl->lt_msb_present[i])
        poc += sl->poc - sl->lt_msb_cycle[i] * max - (sl->poc & (max - 1));
    for (int k = 0; k < d->n_buf; k++) {
        const hevcd_img_t *g = &d->buf[k];
        if (!g->is_valid || g == skip) continue;
        if (sl->lt_msb_present[i] ? g->poc == poc
                                  : (g->poc & (max - 1)) == poc)
            return g;
    }
    return NULL;
}

/* Everything the reference picture set no longer names can go - short-term
 * or long-term, used now or kept for later. */
static void unescape(hevcd_t *d, const hevc_slice_t *sl)
{
    /* 8.3.2: an IRAP that restarts the sequence leaves no reference
     * behind. Every IDR does; a CRA or a BLA does when the caller says it
     * restarts, and its own reference picture set - which a CRA may well
     * carry, for the leading pictures that are not decoded - names
     * pictures that are gone. */
    if (sl->nal_type == HEVC_NAL_IDR_W_RADL || sl->nal_type == HEVC_NAL_IDR_N_LP
        || sl->no_rasl_output_flag) {
        for (int i = 0; i < d->n_buf; i++) d->buf[i].is_valid = false;
        return;
    }
    /* ⚠️ Resolved before anything is dropped: a long-term entry without
     * its high bits matches on the low ones, and dropping first could
     * leave it matching a different picture. */
    const hevcd_img_t *keep[32];
    int n_keep = 0;
    for (int k = 0; k < sl->num_lt && n_keep < 32; k++) {
        const hevcd_img_t *g = find_lt(d, sl, k, NULL);
        if (g) keep[n_keep++] = g;
    }
    const hevc_st_rps_t *r = &sl->st_rps;
    const int count = r->num_negative + r->num_positive;
    for (int i = 0; i < d->n_buf; i++) {
        if (!d->buf[i].is_valid) continue;
        bool serve = false;
        for (int k = 0; k < count && !serve; k++)
            if (d->buf[i].poc == sl->poc + r->delta_poc[k]) serve = true;
        for (int k = 0; k < n_keep && !serve; k++)
            if (keep[k] == &d->buf[i]) serve = true;
        if (!serve) d->buf[i].is_valid = false;
    }
}

/* The three planes of one picture at the coded size, and the motion field
 * a later picture will read for its temporal candidate. The visible size
 * is smaller and is applied when writing out. */
static int open_picture(hevcd_t *d, const hevc_sps_t *sps, int poc)
{
    const int w = sps->width, h = sps->height;
    const size_t n_mvf = (size_t)(w >> 2) * (h >> 2);
    /* ⚠️ Above eight bits a sample is two bytes. n_planes counts BYTES,
     * because that is what the SAO snapshot mallocs and copies, and it
     * doubles as the test below: a sequence that changes depth finds the
     * size different and reallocates, instead of writing ten-bit samples
     * into a buffer sized for eight. Luma and chroma share the figure,
     * which holds while we refuse streams whose two depths differ. */
    const size_t bytes = sps->bit_depth_luma > 8 ? 2 : 1;

    hevcd_img_t *g = NULL;
    for (int i = 0; i < d->n_buf && !g; i++)
        /* ⚠️ Not only unused as a reference: not held either, by a
         * picture still writing or reading it. */
        if (!d->buf[i].is_valid && d->buf[i].users == 0) g = &d->buf[i];
    if (!g) return -1;

    if (g->n_planes != (size_t)w * h * bytes) {
        free_img(g);
        g->plane[0] = malloc((size_t)w * h * bytes);
        g->plane[1] = malloc((size_t)(w / 2) * (h / 2) * bytes);
        g->plane[2] = malloc((size_t)(w / 2) * (h / 2) * bytes);
        g->n_planes = (size_t)w * h * bytes;
    }
    if (g->n_mvf != n_mvf) {
        free(g->mvf);
        g->mvf = malloc(n_mvf * sizeof *g->mvf);
        g->n_mvf = n_mvf;
    }
    if (!g->plane[0] || !g->plane[1] || !g->plane[2] || !g->mvf) return -1;

    /* ⚠️ The motion field is not cleared here any more but one coding tree
     * block at a time, as each is decoded - see hevcd_clear_ctb_motion().
     * Cleared here it was ten megabytes of memset on one thread before
     * every 4K picture, the largest part of what opening a picture cost. */
    /* ⚠️ Sized once for the most slices a picture can have, one per
     * coding tree block, and never grown while it is being decoded: a
     * later picture decoding at the same time reads these, and a realloc
     * would move them from under it. */
    const size_t ctbs = (size_t)sps->ctb_count;
    if (g->n_lists < ctbs) {
        void *p = realloc(g->lists, ctbs * sizeof *g->lists);
        if (!p) return -1;
        g->lists = p;
        g->n_lists = ctbs;
    }
    if (g->n_slice_map < ctbs) {
        free(g->slice_of_ctb);
        g->slice_of_ctb = malloc(ctbs * sizeof *g->slice_of_ctb);
        g->n_slice_map = g->slice_of_ctb ? ctbs : 0;
        if (!g->slice_of_ctb) return -1;
    }

    g->stride[0] = w;
    g->stride[1] = g->stride[2] = w / 2;
    g->poc = poc;
    g->is_valid = true;
    g->progress = d->progress;
    atomic_store_explicit(&g->rows_ready, 0, memory_order_relaxed);

    d->current = g;
    d->slice_of_ctb = g->slice_of_ctb;
    d->n_slice_map = g->n_slice_map;
    d->mvf = g->mvf;
    for (int i = 0; i < 3; i++) { d->plane[i] = g->plane[i]; d->stride[i] = g->stride[i]; }
    d->n_planes = g->n_planes;
    return 0;
}

/* 8.3.4: the lists are the pictures before this one, then the ones after,
 * then the long-term ones, repeated into a temporary list at least as long
 * as the slice header asked for. Each list is then either that, in order,
 * or whatever list_entry picks out of it.
 *
 * ⚠️ List one starts from the other end. That is the whole point of having
 * two: a B picture with one reference each way sends the shorter index
 * for whichever direction it meant. The long-term pictures come last in
 * both.
 *
 * A caller that has the finished lists already - the VA-API bridge -
 * passes them instead, with which entries are long-term. */
static void build_lists(hevcd_t *d, const hevc_slice_t *sl)
{
    if (sl->has_explicit_rpl) {
        for (int l = 0; l < 2; l++) {
            d->n_refs[l] = sl->explicit_n_refs[l];
            for (int i = 0; i < d->n_refs[l]; i++) {
                d->ref_pic[l][i] = (const hevcd_img_t *)sl->explicit_ref_pic[l][i];
                d->ref_is_lt[l][i] = sl->explicit_lt[l][i];
            }
        }
        d->col = (const hevcd_img_t *)sl->explicit_col;
    } else {
        const hevc_st_rps_t *r = &sl->st_rps;
        const hevcd_img_t *before[16], *after[16], *lt[32];
        int np = 0, nd = 0, nl = 0;

        for (int i = 0; i < r->num_negative && np < 16; i++) {
            if (!r->used[i]) continue;
            const hevcd_img_t *g = find_img(d, sl->poc + r->delta_poc[i]);
            if (g) before[np++] = g;
        }
        for (int i = r->num_negative; i < r->num_negative + r->num_positive
                 && nd < 16; i++) {
            if (!r->used[i]) continue;
            const hevcd_img_t *g = find_img(d, sl->poc + r->delta_poc[i]);
            if (g) after[nd++] = g;
        }
        for (int i = 0; i < sl->num_lt && nl < 32; i++) {
            if (!sl->lt_used[i]) continue;
            const hevcd_img_t *g = find_lt(d, sl, i, d->current);
            if (g) lt[nl++] = g;
        }

        for (int l = 0; l < 2; l++) {
            d->n_refs[l] = 0;
            const int how_many = sl->num_ref_idx[l];
            const hevcd_img_t **a = l ? after : before;
            const hevcd_img_t **b = l ? before : after;
            const int na = l ? nd : np, nb = l ? np : nd;
            const int total = na + nb + nl;
            if (!total || !how_many) continue;

            /* RefPicListTemp, NumRpsCurrTempList entries long. */
            const hevcd_img_t *temp[64];
            bool temp_lt[64];
            const int want = how_many > total ? how_many : total;
            int n = 0;
            while (n < want) {
                for (int i = 0; i < na && n < want; i++, n++) {
                    temp[n] = a[i]; temp_lt[n] = false;
                }
                for (int i = 0; i < nb && n < want; i++, n++) {
                    temp[n] = b[i]; temp_lt[n] = false;
                }
                for (int i = 0; i < nl && n < want; i++, n++) {
                    temp[n] = lt[i]; temp_lt[n] = true;
                }
            }
            for (int i = 0; i < how_many; i++) {
                int k = sl->list_mod[l] ? sl->list_entry[l][i] : i;
                if (k >= n) k = n - 1;
                d->ref_pic[l][i] = temp[k];
                d->ref_is_lt[l][i] = temp_lt[k];
            }
            d->n_refs[l] = how_many;
        }

        d->col = NULL;
        if (sl->temporal_mvp_enabled) {
            const int l = sl->collocated_from_l0 ? 0 : 1;
            if (sl->collocated_ref_idx < d->n_refs[l])
                d->col = d->ref_pic[l][sl->collocated_ref_idx];
        }
    }

    /* ⚠️ Never shorter than the header asked for. A reference that
     * could not be found - a picture the stream or the application no
     * longer holds - would leave a hole that every index past it falls
     * into, and an index is a pointer here: the first version of the VA
     * path crashed the application on exactly that. The last picture
     * found stands in for the rest, which is a wrong picture and not a
     * crash; a list with nothing in it at all is refused by the caller. */
    for (int l = 0; l < 2; l++)
        if (d->n_refs[l] > 0)
            while (d->n_refs[l] < sl->num_ref_idx[l] && d->n_refs[l] < 16) {
                d->ref_pic[l][d->n_refs[l]] = d->ref_pic[l][d->n_refs[l] - 1];
                d->ref_is_lt[l][d->n_refs[l]] = d->ref_is_lt[l][d->n_refs[l] - 1];
                d->n_refs[l]++;
            }

    /* What the indices mean, kept with the picture and with the SLICE:
     * a later picture that takes this as its collocated one asks what a
     * stored index pointed at, and an index means nothing outside the
     * slice that wrote it. */
    if (d->slice_now >= 0) {
        hevcd_img_t *g = d->current;
        if ((size_t)d->slice_now >= g->n_lists) {
            const size_t want = (size_t)d->slice_now + 64;
            void *p = realloc(g->lists, want * sizeof *g->lists);
            if (p) {
                g->lists = p;
                g->n_lists = want;
            }
        }
        if ((size_t)d->slice_now < g->n_lists) {
            for (int l = 0; l < 2; l++) {
                g->lists[d->slice_now].n_list[l] = d->n_refs[l];
                for (int i = 0; i < d->n_refs[l]; i++) {
                    g->lists[d->slice_now].poc_list[l][i] =
                        d->ref_pic[l][i] ? d->ref_pic[l][i]->poc : 0;
                    g->lists[d->slice_now].is_lt[l][i] =
                        d->ref_is_lt[l][i];
                }
            }
        }
    }
}

/* 7.4.7.1: the entry point offsets count the NAL unit's bytes, the
 * emulation prevention ones included. The decoder reads the payload with
 * those already removed, so each offset has to lose however many of them
 * it has passed.
 *
 * ⚠️ The rule for which 0x03 is an emulation prevention byte is the one
 * in br_extract_rbsp and not "every 0x03 after two zeros": a 0x03
 * followed by a byte above 0x03 is ordinary payload. Counting them any
 * other way moves the offsets by the wrong amount on exactly the streams
 * where it matters.
 */
static void shift_entry_points(hevc_slice_t *s, const uint8_t *grezzo,
                               size_t n_grezzo, size_t first)
{
    if (s->num_entry_point_offsets <= 0) return;

    size_t i = 0, r = 0, zeros = 0, i0 = 0, r0 = 0;
    bool started = false;
    int k = 0;

    while (i < n_grezzo && k < s->num_entry_point_offsets) {
        if (!started && r == first) {
            i0 = i; r0 = r; started = true;
        }
        if (started && (size_t)(i - i0) == (size_t)s->entry_point[k]) {
            s->entry_point[k] = (uint32_t)(r - r0);
            k++;
            continue;
        }
        const uint8_t c = grezzo[i];
        if (zeros >= 2 && c == 0x03
            && !(i + 1 < n_grezzo && grezzo[i + 1] > 0x03)) {
            zeros = 0;
            i++;
            continue;                 /* removed, so the payload stands still */
        }
        zeros = (c == 0x00) ? zeros + 1 : 0;
        i++;
        r++;
    }
    /* An offset past the end of the NAL is a broken stream; leave the
     * rest where the payload ended and let the row check refuse it. */
    while (k < s->num_entry_point_offsets) {
        s->entry_point[k] = (uint32_t)(r - r0);
        k++;
    }
}

/* Why a slice could not be walked through. */
static const char *slice_reason(int e)
{
    switch (e) {
    case 1: return "ended before the last CTU";
    case 2: return "did not end where it should";
    case 3: return "read past the end of the NAL";
    case 4: return "slice type not walkable yet";
    case 5: return "out of memory";
    case 6: return "end of subset was not one";
    case 7: return "a row is not as long as the header says";
    case 8: return "a reference picture is missing";
    default: return "?";
    }
}

/* Read every bin of a slice's coding tree, and check it lands.
 *
 * ⚠️ Nothing is reconstructed. What this proves is that the syntax was
 * read correctly, which CABAC makes checkable without any samples: a slice
 * read correctly ends with end_of_slice_segment_flag set exactly after the
 * last coding tree unit, and with the arithmetic decoder at the end of the
 * NAL. One bin read against the wrong context almost never lands there. */
/* 6.4.1, for the one unit the wavefront synchronisation reads: the one
 * above and to the right of where a row starts.
 *
 * ⚠️ It has to belong to the same SLICE, which is stronger than the same
 * segment: a dependent segment continues its slice, but the row above it
 * may have been decoded by a different one. And a picture one unit wide
 * has no such neighbour at all, anywhere.
 */
static bool wpp_above_right(const hevcd_t *d, const hevc_sps_t *sps, int addr)
{
    const int col = addr % sps->ctb_width;
    const int row = addr / sps->ctb_width;
    if (row == 0 || col + 1 >= sps->ctb_width) return false;
    const int nb = (row - 1) * sps->ctb_width + col + 1;
    return d->slice_of_ctb && nb < sps->ctb_count
        && d->slice_of_ctb[nb] == d->slice_now;
}

/* Everything one picture needs before any of its slices is read.
 *
 * ⚠️ Called once per PICTURE. It used to run at the top of walk_slice(),
 * which is once per SLICE, and the difference did not show while every
 * picture had exactly one. A second slice arrived to find the first
 * slice's work erased - and then the loop filters ran over the whole
 * picture using boundary strengths that only the last slice had written.
 */
static int prepare_picture(hevcd_t *d, const hevc_sps_t *sps,
                           const hevc_pps_t *pps)
{
    const size_t serve_cb = (size_t)sps->min_cb_width * sps->min_cb_height;
    const size_t serve_pu = (size_t)(sps->width >> 2) * (sps->height >> 2);
    if (!d->ct_depth || d->n_ct_depth < serve_cb) {
        free(d->ct_depth);
        d->ct_depth = calloc(serve_cb, 1);
        d->n_ct_depth = serve_cb;
    }
    if (!d->intra_mode || d->n_intra_mode < serve_pu) {
        free(d->intra_mode);
        d->intra_mode = calloc(serve_pu, 1);
        d->n_intra_mode = serve_pu;
    }
    const size_t edges_needed = (size_t)((sps->width + 7) >> 3)
                            * (size_t)((sps->height + 7) >> 3);
    if (!d->edges || d->n_edges < edges_needed) {
        free(d->edges);
        d->edges = calloc(edges_needed, 1);
        d->n_edges = edges_needed;
    }
    if (!d->no_filter || d->n_no_filter < serve_cb) {
        free(d->no_filter);
        d->no_filter = calloc(serve_cb, 1);
        d->n_no_filter = serve_cb;
    }
    if (!d->sao || d->n_sao < (size_t)sps->ctb_count) {
        free(d->sao);
        d->sao = calloc((size_t)sps->ctb_count, sizeof *d->sao);
        d->n_sao = (size_t)sps->ctb_count;
    }
    if (!d->cbf_map || d->n_cbf < serve_pu) {
        free(d->cbf_map);
        d->cbf_map = calloc(serve_pu, 1);
        d->n_cbf = serve_pu;
    }
    if (!d->cbf_map) return 5;
    memset(d->cbf_map, 0, serve_pu);
    if (!d->skip || d->n_skip < serve_cb) {
        free(d->skip);
        d->skip = calloc(serve_cb, 1);
        d->n_skip = serve_cb;
    }
    if (!d->edges || !d->no_filter || !d->sao || !d->skip) return 5;
    d->edges_stride = (sps->width + 7) >> 3;
    if (!d->qp_y_map || d->n_qp < serve_cb) {
        free(d->qp_y_map);
        d->qp_y_map = calloc(serve_cb, 1);
        d->n_qp = serve_cb;
    }
    if (!d->ct_depth || !d->intra_mode || !d->qp_y_map) return 5;
    memset(d->ct_depth, 0, serve_cb);
    memset(d->edges, 0, edges_needed);
    memset(d->no_filter, 0, serve_cb);
    memset(d->skip, 0, serve_cb);
    memset(d->intra_mode, HEVCD_INTRA_DC, serve_pu);
    /* ⚠️ A slice that switches SAO off for both planes sends no
     * parameters at all, and read_sao() - which clears the entry it is
     * about to fill - is never called for its coding tree units. Without
     * this the previous picture's offsets stay in the map and 8.7.3
     * applies them to a picture whose slice header said not to. */
    memset(d->sao, 0, (size_t)sps->ctb_count * sizeof *d->sao);

    d->sps = sps;
    d->pps = pps;
    if (!d->current) return 5;

    /* 7.4.5, once per picture: the PPS's lists when it sends them, the
     * SPS's otherwise - which are the defaults unless it sent its own. The
     * 16x16 and 32x32 factors are the 8x8 list upsampled, with the DC
     * position taken from its own value. */
    d->scaling_on = sps->scaling_list_enabled;
    if (d->scaling_on) {
        const hevc_scaling_t *sl = pps->pps_scaling_list_present
                                   ? &pps->scaling : &sps->scaling;
        for (int m = 0; m < 6; m++) {
            for (int i = 0; i < 16; i++)
                d->sf4[m][hevcd_diag4_y[i] * 4 + hevcd_diag4_x[i]] =
                    sl->list[0][m][i];
            for (int i = 0; i < 64; i++) {
                const int x = hevcd_diag8_x[i], y = hevcd_diag8_y[i];
                d->sf8[m][y * 8 + x] = sl->list[1][m][i];
                for (int j = 0; j < 2; j++)
                    for (int k = 0; k < 2; k++)
                        d->sf16[m][(y * 2 + j) * 16 + x * 2 + k] =
                            sl->list[2][m][i];
                for (int j = 0; j < 4; j++)
                    for (int k = 0; k < 4; k++)
                        d->sf32[m][(y * 4 + j) * 32 + x * 4 + k] =
                            sl->list[3][m][i];
            }
            d->sf16[m][0] = sl->dc[2][m];
            d->sf32[m][0] = sl->dc[3][m];
        }
    }
    /* ⚠️ Before the z-scan, which is built out of these. */
    if (hevcd_prepare_tiles(d)) return 5;
    if (hevcd_prepare_zscan(d)) return 5;

    /* The picture's own map, allocated when it was opened. */
    if (!d->slice_of_ctb || d->n_slice_map < (size_t)sps->ctb_count) return 5;
    /* ⚠️ Minus one, not zero: zero is a real slice number. A unit still
     * holding minus one when the picture ends is one no slice ever
     * covered. */
    for (int i = 0; i < sps->ctb_count; i++) d->slice_of_ctb[i] = -1;
    d->slice_now = -1;
    d->filters_done = false;
    /* Nothing carries across a picture boundary. */
    d->have_segment_end = false;
    d->have_wpp_snapshot = false;
    return 0;
}

static int walk_slice(hevcd_t *d, const hevc_sps_t *sps,
                          const hevc_pps_t *pps, const hevc_slice_t *sl,
                          const uint8_t *rbsp, size_t n)
{
    /* ⚠️ P and B slices are read through, not reconstructed. Their
     * samples are meaningless until motion compensation exists; what the
     * walk proves is that every bin of their syntax was read against the
     * right context. */

    /* ⚠️ The maps belong to the picture and were built by
     * prepare_picture() before the first slice arrived. Nothing is
     * allocated or wiped here: a slice that did that would erase the
     * slices before it. */
    if (!d->ct_depth || !d->intra_mode || !d->edges || !d->no_filter
        || !d->sao || !d->cbf_map || !d->skip || !d->qp_y_map
        || !d->rs_to_ts || !d->min_tb_addr_zs)
        return 5;
    if (!d->current) return 5;
    d->slice = sl;

    /* ⚠️ Kept because the filters run after every slice has been read,
     * by which time d->slice is the last one. Reading the offsets from
     * there applied one slice's settings to the whole picture. */
    if (d->slice_now >= 0) {
        if ((size_t)d->slice_now >= d->n_slice_filter) {
            const size_t want = (size_t)d->slice_now + 64;
            hevcd_slice_filter_t *p = realloc(d->slice_filter,
                                              want * sizeof *p);
            if (!p) return 5;
            d->slice_filter = p;
            d->n_slice_filter = want;
        }
        hevcd_slice_filter_t *f = &d->slice_filter[d->slice_now];
        f->beta_offset = (int16_t)sl->beta_offset;
        f->tc_offset = (int16_t)sl->tc_offset;
        f->disabled = sl->deblocking_filter_disabled ? 1 : 0;
        f->across_slices = sl->loop_filter_across_slices ? 1 : 0;
    }

    d->min_pu_width = sps->width >> 2;
    d->min_pu_height = sps->height >> 2;
    /* Does this segment begin a tile, or a coding tree row under
     * wavefront? Both 8.6.1 and 9.3.1 ask, and both let the answer
     * outrank the dependent-segment rule. */
    const bool wpp = pps->entropy_coding_sync_enabled;
    const int first_ts = d->rs_to_ts[sl->segment_address];
    const bool starts_tile = pps->tiles_enabled
        && (first_ts == 0
            || d->tile_of_ts[first_ts - 1] != d->tile_of_ts[first_ts]);
    const bool starts_row = wpp
        && (sl->segment_address % sps->ctb_width) == 0;

    /* ⚠️ 8.6.1: a dependent segment continues the quantisation parameter
     * prediction of the segment before it. Only an independent one
     * restarts from the slice's own parameter - or a segment that starts
     * a tile or a wavefront row, which restart it whatever the segment
     * is. A tile or a row that is exactly one dependent segment carries
     * no entry point, so the substream boundary further down never sees
     * it. */
    const bool dependent = sl->dependent_slice_segment && d->have_segment_end;
    if (!dependent || starts_tile || starts_row) {
        d->qp_y = sl->qp;
        d->qp_y_pred = sl->qp;
        d->qp_y_prev = sl->qp;
        d->qg_restarts = true;
    }
    d->slice_end = false;

    const size_t first = sl->data_bit_offset >> 3;
    if (first >= n) return 3;
    const uint8_t *base = rbsp + first;
    size_t rest = n - first;
    hevcd_cabac_init(&d->cabac, base, rest,
                     sl->type, sl->cabac_init_flag, sl->qp);

    const int init_type = hevcd_init_type(sl->type, sl->cabac_init_flag);

    /* 9.3.1, in the order the clause gives them. The engine is always
     * restarted - that part is per segment - and only the context state
     * is in question.
     *
     * ⚠️ The wavefront rule outranks the dependent-segment one. When
     * every row is its own dependent segment, as kvazaar writes with
     * --wpp --slices wpp, both apply and taking the second is wrong by
     * half a picture. */
    {
        if (starts_tile) {
            /* already initialised above */
        } else if (starts_row) {
            /* ⚠️ The decision is made INSIDE this branch and never falls
             * out of it. When the unit above right is not available the
             * clause says to initialise, which is what the header did
             * already - it does not say to go on to the next rule. A
             * dependent segment that starts a row therefore does NOT
             * continue from the segment before it. */
            if (d->have_wpp_snapshot
                && wpp_above_right(d, sps, sl->segment_address))
                memcpy(d->cabac.state, d->wpp_snapshot, HEVCD_CTX);
        } else if (dependent) {
            memcpy(d->cabac.state, d->ctx_at_segment_end, HEVCD_CTX);
        }
    }

    /* ⚠️ Only a slice that starts at the first unit and carries one entry
     * point per row: anything else - a slice segment starting mid picture,
     * tiles, a header that does not say where the rows are - is walked one
     * unit at a time rather than guessed at. */
    if (wpp && sl->segment_address == 0 && sps->ctb_height >= 2
        && sl->num_entry_point_offsets == sps->ctb_height - 1) {
        const int e = hevcd_wavefront(d, sps, pps, sl, base, rest, init_type);
        if (e >= 0) return e;
    }

    const int count = sps->ctb_count;
    const bool tiles = pps->tiles_enabled;
    int progress = 0;
    int substream = 0;      /* how many substream boundaries have passed */
    /* Asked once, not once per unit: getenv walks the whole environment. */
    const bool trace = getenv("HEVC_TRACE") != NULL;

    /* ⚠️ Tile scan. Without tiles the map is the identity and this is the
     * raster walk it always was. */
    for (int ts = d->rs_to_ts[sl->segment_address]; ts < count; ts++) {
        const int addr = d->ts_to_rs[ts];
        const int cx = addr % sps->ctb_width;
        const int x = cx << sps->log2_ctb;
        const int y = (addr / sps->ctb_width) << sps->log2_ctb;
        if (hevcd_read_ctu(d, x, y)) return 4;   /* refused inside */
        progress++;

        /* 9.3.2.3: after the second unit of a row, so the row below can
         * start from here. */
        if (wpp && cx == 1) {
            memcpy(d->wpp_snapshot, d->cabac.state, HEVCD_CTX);
            d->have_wpp_snapshot = true;
        }
        /* HEVC_TRACE: how far into the NAL each coding tree unit got.
         * When a slice does not land, this says where it stopped being
         * right - a unit that consumed implausibly little is where to
         * look, not the one that ran out of data. */
        if (trace) {
            const long n_read = (long)((d->cabac.ptr - d->cabac.start) * 8
                                      - d->cabac.cache_bits);
            fprintf(stderr, "ctu ts %d rs %d (%d,%d): %ld bits of %ld" "\n",
                    ts, addr, x, y, n_read, (long)(n - first) * 8);
        }
        if (hevcd_overrun(&d->cabac)) return 3;

        const int fine = hevcd_terminate(&d->cabac);
        if (fine) {
            /* end_of_slice_segment_flag. ⚠️ Wherever it lands, this
             * segment is over and that is legitimate: another one
             * carries on from the next address. Whether the picture
             * ended up whole is asked at the end of the picture, where
             * the question belongs - see hevc_decoder_end_picture(). */
            memcpy(d->ctx_at_segment_end, d->cabac.state, HEVCD_CTX);
            d->have_segment_end = true;
            return 0;
        }

        /* 7.3.8.1: the substream ends when the next unit starts a new
         * tile, or under wavefront a new row. */
        const bool tile_break = tiles && ts + 1 < count
                                && d->tile_of_ts[ts + 1] != d->tile_of_ts[ts];
        const bool row_break = wpp && addr + 1 < count
                               && (addr + 1) % sps->ctb_width == 0;
        if (tile_break || row_break) {
            if (!hevcd_terminate(&d->cabac))
                return 6;                  /* the bit is defined to be one */

            const size_t n_used = h264d_cabac_byte_pos(&d->cabac);

            /* Where the header says this substream ends. One entry point
             * per substream, counted in the order they are walked, which
             * is why this is a running index and not a row number: under
             * tiles the substreams are tiles. */
            size_t step = n_used;
            if (substream < sl->num_entry_point_offsets) {
                const uint32_t fin = sl->entry_point[substream];
                const uint32_t ini = substream > 0
                                     ? sl->entry_point[substream - 1] : 0;
                step = (size_t)(fin - ini);
                /* A substream may stop short of what the header allows -
                 * the bytes left over are the engine's own look-ahead. It
                 * may not run past it: that is one read wrongly. */
                if (n_used > step) return 7;
            }
            substream++;
            if (step >= rest) return 3;
            base += step;
            rest -= step;
            h264d_cabac_init_engine(&d->cabac, base, rest);
            /* 8.6.1: a new substream predicts its first quantisation
             * group from the slice's own parameter, not from where the
             * previous one happened to end. */
            d->qg_restarts = true;

            /* 9.3.1. A tile starts from nothing: no other tile's state
             * carries into it, which is the whole point of tiles.
             * ⚠️ A wavefront row is the opposite - it continues from the
             * snapshot taken two units into the row above - so the two
             * cases cannot share this line. */
            if (tile_break)
                hevcd_cabac_ctx_init(d->cabac.state, init_type, sl->qp);
            else if (d->have_wpp_snapshot
                     && wpp_above_right(d, sps, d->ts_to_rs[ts + 1]))
                memcpy(d->cabac.state, d->wpp_snapshot, HEVCD_CTX);
            else
                hevcd_cabac_ctx_init(d->cabac.state, init_type, sl->qp);
        }
    }
    (void)progress;
    return 2;                                 /* ran out of CTUs first */
}

/* ------------------------------------------------------------- the API */

hevc_decoder_t *hevc_decoder_create(void *gpu, int width, int height)
{
    hevc_decoder_t *h = calloc(1, sizeof *h);
    if (!h) return NULL;
    h->gpu = gpu;
    h->width = width;
    h->height = height;
    pthread_mutex_init(&h->progress.m, NULL);
    pthread_cond_init(&h->progress.cv, NULL);
    pthread_mutex_init(&h->dpb, NULL);
    for (int k = 0; k < HEVC_DECODER_FRAMES; k++) {
        h->f[k].d.buf = h->buffer;
        h->f[k].d.n_buf = IMG_SLOTS;
        h->f[k].d.progress = &h->progress;
    }
    return h;
}

static void free_frame(hevc_frame_t *f)
{
    hevcd_t *d = &f->d;
    hevcd_free_tiles(d);
    free(d->slice_filter);
    free(d->ct_depth); free(d->intra_mode); free(d->min_tb_addr_zs);
    free(d->zs_rs_to_ts);
    free(d->qp_y_map); free(d->edges); free(d->no_filter);
    free(d->skip); free(d->cbf_map);
    hevcd_free_filters(d);
    hevcd_pool_destroy(d->pool);
}

void hevc_decoder_destroy(hevc_decoder_t *h)
{
    if (!h) return;
    for (int k = 0; k < HEVC_DECODER_FRAMES; k++) free_frame(&h->f[k]);
    for (int i = 0; i < IMG_SLOTS; i++) free_img(&h->buffer[i]);
    pthread_mutex_destroy(&h->dpb);
    pthread_mutex_destroy(&h->progress.m);
    pthread_cond_destroy(&h->progress.cv);
    free(h);
}

/* The buffer lock, from functions that only read and take a const decoder:
 * the lock is not part of what they promise to leave alone. */
static pthread_mutex_t *dpb_lock(const hevc_decoder_t *h)
{
    return (pthread_mutex_t *)&h->dpb;
}

void hevc_decoder_set_references(hevc_decoder_t *h, const uintptr_t *id,
                                 const int *poc, int n)
{
    pthread_mutex_lock(&h->dpb);
    for (int i = 0; i < IMG_SLOTS; i++) {
        if (!h->buffer[i].is_valid) continue;
        bool serve = false;
        for (int k = 0; k < n && !serve; k++) {
            if ((h->surface_id[i] == id[k] && h->buffer[i].poc == poc[k])
                || (h->buffer[i].poc == poc[k])) {
                serve = true;
                h->surface_id[i] = id[k];
            }
        }
        if (!serve) h->buffer[i].is_valid = false;
    }
    pthread_mutex_unlock(&h->dpb);
}

static const void *find_ref_locked(const hevc_decoder_t *h, uintptr_t id, int poc)
{
    /* 1. Exact match on both surface ID and POC */
    for (int i = 0; i < IMG_SLOTS; i++) {
        if (h->buffer[i].is_valid && h->surface_id[i] == id && h->buffer[i].poc == poc)
            return &h->buffer[i];
    }
    /* 2. Match on POC (surface ID may have been rebound by player) */
    for (int i = 0; i < IMG_SLOTS; i++) {
        if (h->buffer[i].is_valid && h->buffer[i].poc == poc)
            return &h->buffer[i];
    }
    /* 3. Match on surface ID */
    for (int i = 0; i < IMG_SLOTS; i++) {
        if (h->buffer[i].is_valid && h->surface_id[i] == id)
            return &h->buffer[i];
    }
    return NULL;
}

const void *hevc_decoder_find_ref(const hevc_decoder_t *h, uintptr_t id, int poc)
{
    pthread_mutex_lock(dpb_lock(h));
    const void *g = find_ref_locked(h, id, poc);
    pthread_mutex_unlock(dpb_lock(h));
    return g;
}

const void *hevc_decoder_find_closest(const hevc_decoder_t *h, int poc)
{
    pthread_mutex_lock(dpb_lock(h));
    const hevcd_img_t *best = NULL;
    int min_diff = 0x7fffffff;
    for (int i = 0; i < IMG_SLOTS; i++) {
        if (!h->buffer[i].is_valid) continue;
        int diff = abs(h->buffer[i].poc - poc);
        if (diff < min_diff) {
            min_diff = diff;
            best = &h->buffer[i];
        }
    }
    pthread_mutex_unlock(dpb_lock(h));
    return best;
}

int hevc_decoder_begin_frame(hevc_decoder_t *h, int k, const hevc_sps_t *sps,
                             const hevc_pps_t *pps, uintptr_t id, int poc,
                             const void *const *refs, int n_refs)
{
    if (k < 0 || k >= HEVC_DECODER_FRAMES) return -1;
    hevc_frame_t *f = &h->f[k];
    f->sps = *sps;
    f->pps = *pps;

    pthread_mutex_lock(&h->dpb);
    /* What this frame held for its last picture, which is done. */
    for (int i = 0; i < f->n_held; i++) f->held[i]->users--;
    f->n_held = 0;
    const int e = open_picture(&f->d, &f->sps, poc);
    if (!e) {
        /* Held until this frame's next picture: the slot it writes, and
         * every slot it reads, so that a later picture's reference set
         * cannot free one of them from under it. */
        f->held[f->n_held++] = f->d.current;
        f->d.current->users++;
        for (int i = 0; i < n_refs && f->n_held < IMG_SLOTS + 16; i++) {
            hevcd_img_t *g = (hevcd_img_t *)refs[i];
            if (!g) continue;
            f->held[f->n_held++] = g;
            g->users++;
        }
        for (int i = 0; i < IMG_SLOTS; i++)
            if (&h->buffer[i] == f->d.current) h->surface_id[i] = id;
    }
    pthread_mutex_unlock(&h->dpb);
    if (e) return -1;

    /* ⚠️ Here and not in walk_slice(): once per picture, not once per
     * slice. See prepare_picture(). */
    if (prepare_picture(&f->d, &f->sps, &f->pps)) return -1;
    f->is_open = true;
    return 0;
}

int hevc_decoder_begin_picture(hevc_decoder_t *h, const hevc_sps_t *sps,
                               const hevc_pps_t *pps, uintptr_t id, int poc)
{
    return hevc_decoder_begin_frame(h, 0, sps, pps, id, poc, NULL, 0);
}

int hevc_decoder_frame_slice(hevc_decoder_t *h, int k, const hevc_slice_t *sl,
                             const uint8_t *rbsp, size_t n)
{
    if (k < 0 || k >= HEVC_DECODER_FRAMES) return 5;
    hevc_frame_t *f = &h->f[k];
    if (!f->is_open || !f->d.current) return 5;

    if (sl->dependent_slice_segment && f->d.slice_now >= 0) {
        /* ⚠️ Everything except the few fields a dependent segment really
         * does carry. last_one already holds the header this one
         * continues - which may itself be a merged dependent segment, so
         * a run of them chains correctly. */
        const int addr = sl->segment_address;
        const int n_ep = sl->num_entry_point_offsets;
        const size_t off = sl->data_bit_offset;
        const int nal = sl->nal_type;
        uint32_t ep[600];
        if (n_ep > 0) memcpy(ep, sl->entry_point, (size_t)n_ep * sizeof ep[0]);

        f->last_one.dependent_slice_segment = true;
        f->last_one.first_slice_in_pic = false;
        f->last_one.segment_address = addr;
        f->last_one.num_entry_point_offsets = n_ep;
        if (n_ep > 0) memcpy(f->last_one.entry_point, ep,
                             (size_t)n_ep * sizeof ep[0]);
        f->last_one.data_bit_offset = off;
        f->last_one.nal_type = nal;
    } else {
        f->last_one = *sl;
    }
    /* ⚠️ The picture order count comes from the picture, not from the
     * slice segment header. Only the first segment of a picture carries
     * what it is derived from, so every caller fills it in for that one
     * and leaves the rest at zero - and build_lists() then looks up
     * `poc + delta_poc` in the buffer, finds nothing, and hands the slice
     * an empty reference list without a word. */
    f->last_one.poc = f->d.current->poc;
    f->d.slice = &f->last_one;
    /* ⚠️ A dependent segment is not a new slice, it is the rest of the
     * one before it, so it keeps that number. An independent one starts
     * a new slice. */
    if (!sl->dependent_slice_segment || f->d.slice_now < 0)
        f->d.slice_now++;
    /* ⚠️ Lists built from the slice header look pictures up in the
     * buffer, so they take its lock; lists handed over ready-made (the VA
     * path) were resolved before the picture was begun. */
    if (!f->last_one.has_explicit_rpl) pthread_mutex_lock(&h->dpb);
    build_lists(&f->d, &f->last_one);
    if (!f->last_one.has_explicit_rpl) pthread_mutex_unlock(&h->dpb);
    if (f->last_one.type != 2) {
        for (int l = 0; l < (f->last_one.type == 0 ? 2 : 1); l++)
            if (!f->d.n_refs[l]) return 8;
    }
    return walk_slice(&f->d, &f->sps, &f->pps, &f->last_one, rbsp, n);
}

int hevc_decoder_slice(hevc_decoder_t *h, const hevc_slice_t *sl,
                       const uint8_t *rbsp, size_t n)
{
    return hevc_decoder_frame_slice(h, 0, sl, rbsp, n);
}

void hevc_decoder_end_frame(hevc_decoder_t *h, int k)
{
    if (k < 0 || k >= HEVC_DECODER_FRAMES) return;
    hevc_frame_t *f = &h->f[k];
    if (!f->is_open || !f->d.current) return;
    /* ⚠️ Units no slice reached - a slice lost or refused - were never
     * cleared, and their motion field still holds whatever picture used
     * this buffer before. A later picture reads it as its collocated
     * motion, so it is cleared now. */
    if (f->d.slice_of_ctb && f->d.mvf) {
        const hevc_sps_t *sps = f->d.sps;
        for (int rs = 0; rs < sps->ctb_count; rs++)
            if (f->d.slice_of_ctb[rs] < 0)
                hevcd_clear_ctb_motion(&f->d, rs % sps->ctb_width,
                                       rs / sps->ctb_width);
    }
    if (f->d.slice && !f->d.filters_done) hevcd_loop_filters(&f->d);
    /* ⚠️ Every row final now, whatever happened on the way - a picture
     * that failed half way is still a picture someone may be waiting to
     * read, and a wait that is never answered is a hang. */
    hevcd_rows_ready(f->d.current, f->sps.ctb_height);
    f->is_open = false;
}

void hevc_decoder_end_picture(hevc_decoder_t *h)
{
    hevc_decoder_end_frame(h, 0);
}

const uint8_t *hevc_decoder_plane(const hevc_decoder_t *h, int plane,
                                  int *stride)
{
    const hevcd_img_t *g = h->f[0].d.current;
    if (!g || plane < 0 || plane > 2) return NULL;
    if (stride) *stride = g->stride[plane];
    return g->plane[plane];
}

bool hevc_decoder_holds(const hevc_decoder_t *h, uintptr_t id)
{
    bool held = false;
    pthread_mutex_lock(dpb_lock(h));
    for (int i = 0; i < IMG_SLOTS && !held; i++)
        if (h->buffer[i].is_valid && h->surface_id[i] == id) held = true;
    pthread_mutex_unlock(dpb_lock(h));
    return held;
}

void hevc_decoder_unescape(hevc_decoder_t *h, const hevc_slice_t *sl)
{
    pthread_mutex_lock(&h->dpb);
    unescape(&h->f[0].d, sl);
    pthread_mutex_unlock(&h->dpb);
}

void hevc_decoder_shift_entry_points(hevc_slice_t *s, const uint8_t *grezzo,
                                     size_t n_grezzo, size_t first)
{
    shift_entry_points(s, grezzo, n_grezzo, first);
}

/* ------------------------------------------ the picture into the surface
 *
 * ⚠️ Written once, straight into the mapped surface, and never read back:
 * surface memory is write-combining, fast to write straight through and
 * very slow to read or revisit. The chroma planes are interleaved on the
 * way, where they used to go through a buffer of their own first - four
 * megabytes read and written again for every 4K picture.
 *
 * And on several threads. At 4K this is twelve megabytes a picture, and on
 * one thread it came to a quarter of what the driver took over the
 * decoder alone. */

#define LOAD_BAND 32                /* luma rows per band, sixteen of chroma */
#define LOAD_MAX_THREAD 16

typedef struct {
    const hevcd_img_t *g;
    uint8_t *y, *uv;
    size_t y_pitch, uv_pitch;
    int width, height;
    bool ten;
    int bands;
    _Atomic int next;
} load_job_t;

#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("avx2")))
static void interleave8_avx2(uint8_t *o, const uint8_t *a, const uint8_t *b, int n, int *out_x)
{
    int x = 0;
    for (; x + 32 <= n; x += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + x));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + x));
        __m256i lo = _mm256_unpacklo_epi8(va, vb);
        __m256i hi = _mm256_unpackhi_epi8(va, vb);
        __m256i out0 = _mm256_permute2x128_si256(lo, hi, 0x20);
        __m256i out1 = _mm256_permute2x128_si256(lo, hi, 0x31);
        _mm256_storeu_si256((__m256i *)(o + 2 * x), out0);
        _mm256_storeu_si256((__m256i *)(o + 2 * x + 32), out1);
    }
    *out_x = x;
}
#endif

static void interleave8(uint8_t *o, const uint8_t *a, const uint8_t *b, int n)
{
    int x = 0;
#if defined(__x86_64__) || defined(_M_X64)
    if (__builtin_cpu_supports("avx2")) {
        interleave8_avx2(o, a, b, n, &x);
    }
    for (; x + 16 <= n; x += 16) {
        const __m128i va = _mm_loadu_si128((const __m128i *)(a + x));
        const __m128i vb = _mm_loadu_si128((const __m128i *)(b + x));
        _mm_storeu_si128((__m128i *)(o + 2 * x), _mm_unpacklo_epi8(va, vb));
        _mm_storeu_si128((__m128i *)(o + 2 * x + 16), _mm_unpackhi_epi8(va, vb));
    }
#elif defined(__SSE2__)
    for (; x + 16 <= n; x += 16) {
        const __m128i va = _mm_loadu_si128((const __m128i *)(a + x));
        const __m128i vb = _mm_loadu_si128((const __m128i *)(b + x));
        _mm_storeu_si128((__m128i *)(o + 2 * x), _mm_unpacklo_epi8(va, vb));
        _mm_storeu_si128((__m128i *)(o + 2 * x + 16), _mm_unpackhi_epi8(va, vb));
    }
#endif
    for (; x < n; x++) { o[2 * x] = a[x]; o[2 * x + 1] = b[x]; }
}

#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("avx2")))
static void shift10_avx2(uint16_t *o, const uint16_t *s, int n, int *out_x)
{
    int x = 0;
    for (; x + 16 <= n; x += 16) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(s + x));
        __m256i shifted = _mm256_slli_epi16(v, 6);
        _mm256_storeu_si256((__m256i *)(o + x), shifted);
    }
    *out_x = x;
}
#endif

/* P010 keeps its ten bits at the top of each sixteen. */
static void shift10(uint16_t *o, const uint16_t *s, int n)
{
    int x = 0;
#if defined(__x86_64__) || defined(_M_X64)
    if (__builtin_cpu_supports("avx2")) {
        shift10_avx2(o, s, n, &x);
    }
    for (; x + 8 <= n; x += 8)
        _mm_storeu_si128((__m128i *)(o + x),
            _mm_slli_epi16(_mm_loadu_si128((const __m128i *)(s + x)), 6));
#elif defined(__SSE2__)
    for (; x + 8 <= n; x += 8)
        _mm_storeu_si128((__m128i *)(o + x),
            _mm_slli_epi16(_mm_loadu_si128((const __m128i *)(s + x)), 6));
#endif
    for (; x < n; x++) o[x] = (uint16_t)(s[x] << 6);
}

#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("avx2")))
static void interleave10_avx2(uint16_t *o, const uint16_t *a, const uint16_t *b,
                             int n, int *out_x)
{
    int x = 0;
    for (; x + 16 <= n; x += 16) {
        __m256i va = _mm256_slli_epi16(_mm256_loadu_si256((const __m256i *)(a + x)), 6);
        __m256i vb = _mm256_slli_epi16(_mm256_loadu_si256((const __m256i *)(b + x)), 6);
        __m256i lo = _mm256_unpacklo_epi16(va, vb);
        __m256i hi = _mm256_unpackhi_epi16(va, vb);
        __m256i out0 = _mm256_permute2x128_si256(lo, hi, 0x20);
        __m256i out1 = _mm256_permute2x128_si256(lo, hi, 0x31);
        _mm256_storeu_si256((__m256i *)(o + 2 * x), out0);
        _mm256_storeu_si256((__m256i *)(o + 2 * x + 16), out1);
    }
    *out_x = x;
}
#endif

static void interleave10(uint16_t *o, const uint16_t *a, const uint16_t *b,
                         int n)
{
    int x = 0;
#if defined(__x86_64__) || defined(_M_X64)
    if (__builtin_cpu_supports("avx2")) {
        interleave10_avx2(o, a, b, n, &x);
    }
    for (; x + 8 <= n; x += 8) {
        const __m128i va = _mm_slli_epi16(
            _mm_loadu_si128((const __m128i *)(a + x)), 6);
        const __m128i vb = _mm_slli_epi16(
            _mm_loadu_si128((const __m128i *)(b + x)), 6);
        _mm_storeu_si128((__m128i *)(o + 2 * x), _mm_unpacklo_epi16(va, vb));
        _mm_storeu_si128((__m128i *)(o + 2 * x + 8), _mm_unpackhi_epi16(va, vb));
    }
#elif defined(__SSE2__)
    for (; x + 8 <= n; x += 8) {
        const __m128i va = _mm_slli_epi16(
            _mm_loadu_si128((const __m128i *)(a + x)), 6);
        const __m128i vb = _mm_slli_epi16(
            _mm_loadu_si128((const __m128i *)(b + x)), 6);
        _mm_storeu_si128((__m128i *)(o + 2 * x), _mm_unpacklo_epi16(va, vb));
        _mm_storeu_si128((__m128i *)(o + 2 * x + 8), _mm_unpackhi_epi16(va, vb));
    }
#endif
    for (; x < n; x++) {
        o[2 * x] = (uint16_t)(a[x] << 6);
        o[2 * x + 1] = (uint16_t)(b[x] << 6);
    }
}

static void load_band(load_job_t *j, int band)
{
    const hevcd_img_t *g = j->g;
    const int y0 = band * LOAD_BAND;
    const int y1 = y0 + LOAD_BAND < j->height ? y0 + LOAD_BAND : j->height;
    const int cw = j->width / 2;

    if (!j->ten) {
        for (int r = y0; r < y1; r++)
            memcpy(j->y + (size_t)r * j->y_pitch,
                   g->plane[0] + (size_t)r * g->stride[0], (size_t)j->width);
        for (int r = y0 / 2; r < y1 / 2; r++)
            interleave8(j->uv + (size_t)r * j->uv_pitch,
                        g->plane[1] + (size_t)r * g->stride[1],
                        g->plane[2] + (size_t)r * g->stride[2], cw);
        return;
    }

    /* ⚠️ The strides are sample counts, the pitches byte counts. */
    for (int r = y0; r < y1; r++)
        shift10((uint16_t *)(j->y + (size_t)r * j->y_pitch),
                (const uint16_t *)g->plane[0] + (size_t)r * g->stride[0],
                j->width);
    for (int r = y0 / 2; r < y1 / 2; r++)
        interleave10((uint16_t *)(j->uv + (size_t)r * j->uv_pitch),
                     (const uint16_t *)g->plane[1] + (size_t)r * g->stride[1],
                     (const uint16_t *)g->plane[2] + (size_t)r * g->stride[2],
                     cw);
}

static void *load_worker(void *arg)
{
    load_job_t *j = arg;
    for (;;) {
        const int band = atomic_fetch_add_explicit(&j->next, 1,
                                                   memory_order_relaxed);
        if (band >= j->bands) break;
        load_band(j, band);
    }
    return NULL;
}

int hevc_decoder_load(hevc_decoder_t *h, gpu_image_t out, gpu_memory_t mem)
{
    return hevc_decoder_load_frame(h, 0, out, mem);
}

int hevc_decoder_load_frame(hevc_decoder_t *h, int k, gpu_image_t out,
                            gpu_memory_t mem)
{
    if (!h->gpu) return 0;                 /* the harness keeps the planes */
    if (k < 0 || k >= HEVC_DECODER_FRAMES) return -1;
    hevc_frame_t *f = &h->f[k];
    const hevcd_img_t *g = f->d.current;
    if (!g || !g->plane[0]) return -1;

    gpu_context_t *gpu = h->gpu;
    gpu_nv12_layout_t lay;
    bool unmap = false;
    uint8_t *mapped = gpu_compute_map_surface(gpu, &out, mem, &lay, &unmap);
    if (!mapped) return -1;

    load_job_t j;
    memset(&j, 0, sizeof j);
    j.g = g;
    j.width = h->width;
    j.height = h->height;
    j.ten = f->sps.bit_depth_luma > 8;
    j.y_pitch = lay.y_pitch;
    j.uv_pitch = lay.uv_pitch;
    j.bands = (h->height + LOAD_BAND - 1) / LOAD_BAND;

    /* ⚠️ Both planes have to fit inside the memory the surface was given.
     * The copy this replaced trusted the layout; a surface smaller than
     * the stream would have been written past its end. */
    const size_t row = (size_t)h->width * (j.ten ? 2 : 1);
    if (h->width <= 0 || h->height <= 1
        || lay.y_offset + (uint64_t)(h->height - 1) * lay.y_pitch + row
               > lay.total_size
        || lay.uv_offset + (uint64_t)(h->height / 2 - 1) * lay.uv_pitch + row
               > lay.total_size) {
        gpu_compute_unmap_surface(gpu, mem, unmap);
        return -1;
    }
    j.y = mapped + lay.y_offset;
    j.uv = mapped + lay.uv_offset;

    const char *s = getenv("BC250_HEVC_THREAD");
    int want = s ? atoi(s) : (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (want > LOAD_MAX_THREAD) want = LOAD_MAX_THREAD;
    if (want > j.bands) want = j.bands;

    hevcd_pool_t *pool = want > 1 ? hevcd_pool_for(&f->d) : NULL;
    if (pool) {
        hevcd_pool_run(pool, load_worker, &j, want);
    } else {
        pthread_t t[LOAD_MAX_THREAD];
        int alive = 0;
        for (int i = 1; i < want; i++)
            if (pthread_create(&t[alive], NULL, load_worker, &j) == 0) alive++;
        load_worker(&j);
        for (int i = 0; i < alive; i++) pthread_join(t[i], NULL);
    }

    gpu_compute_unmap_surface(gpu, mem, unmap);
    return 0;
}

const char *hevc_decoder_reason(int e)
{
    return slice_reason(e);
}
