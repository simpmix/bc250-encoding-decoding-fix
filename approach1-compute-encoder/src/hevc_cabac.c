/* bc250-encoding-decoding-fix v0.4.3 - https://github.com/simpmix/bc250-encoding-decoding-fix */
/*
 * Copyright (c) 2026 BC-250 Project Contributors
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * hevc_cabac.c - see hevc_cabac.h for the full provenance note (this file's
 * arithmetic-coder core and residual_coding() control flow is adapted from
 * x265's source/encoder/entropy.cpp + source/common/contexts.h +
 * source/common/constants.cpp, Copyright (C) 2013-2020 MulticoreWare, Inc,
 * licensed GPL-2.0-or-later - combination-compatible with this project's
 * GPL-3.0-only license via the "or any later version" grant).
 */
#include "hevc_cabac.h"
#include <string.h>

/* ===================== spec/x265 numeric tables ===================== */

/* CABAC state-transition table: g_hevc_next_state[state<<1|mps][bin].
 * Identical to x265's g_nextState[128][2] / the HEVC spec's Table 9-47. */
static const uint8_t g_hevc_next_state[128][2] = {
    { 2, 1 }, { 0, 3 }, { 4, 0 }, { 1, 5 }, { 6, 2 }, { 3, 7 }, { 8, 4 }, { 5, 9 },
    { 10, 4 }, { 5, 11 }, { 12, 8 }, { 9, 13 }, { 14, 8 }, { 9, 15 }, { 16, 10 }, { 11, 17 },
    { 18, 12 }, { 13, 19 }, { 20, 14 }, { 15, 21 }, { 22, 16 }, { 17, 23 }, { 24, 18 }, { 19, 25 },
    { 26, 18 }, { 19, 27 }, { 28, 22 }, { 23, 29 }, { 30, 22 }, { 23, 31 }, { 32, 24 }, { 25, 33 },
    { 34, 26 }, { 27, 35 }, { 36, 26 }, { 27, 37 }, { 38, 30 }, { 31, 39 }, { 40, 30 }, { 31, 41 },
    { 42, 32 }, { 33, 43 }, { 44, 32 }, { 33, 45 }, { 46, 36 }, { 37, 47 }, { 48, 36 }, { 37, 49 },
    { 50, 38 }, { 39, 51 }, { 52, 38 }, { 39, 53 }, { 54, 42 }, { 43, 55 }, { 56, 42 }, { 43, 57 },
    { 58, 44 }, { 45, 59 }, { 60, 44 }, { 45, 61 }, { 62, 46 }, { 47, 63 }, { 64, 48 }, { 49, 65 },
    { 66, 48 }, { 49, 67 }, { 68, 50 }, { 51, 69 }, { 70, 52 }, { 53, 71 }, { 72, 52 }, { 53, 73 },
    { 74, 54 }, { 55, 75 }, { 76, 54 }, { 55, 77 }, { 78, 56 }, { 57, 79 }, { 80, 58 }, { 59, 81 },
    { 82, 58 }, { 59, 83 }, { 84, 60 }, { 61, 85 }, { 86, 60 }, { 61, 87 }, { 88, 60 }, { 61, 89 },
    { 90, 62 }, { 63, 91 }, { 92, 64 }, { 65, 93 }, { 94, 64 }, { 65, 95 }, { 96, 66 }, { 67, 97 },
    { 98, 66 }, { 67, 99 }, { 100, 66 }, { 67, 101 }, { 102, 68 }, { 69, 103 }, { 104, 68 }, { 69, 105 },
    { 106, 70 }, { 71, 107 }, { 108, 70 }, { 71, 109 }, { 110, 70 }, { 71, 111 }, { 112, 72 }, { 73, 113 },
    { 114, 72 }, { 73, 115 }, { 116, 72 }, { 73, 117 }, { 118, 74 }, { 75, 119 }, { 120, 74 }, { 75, 121 },
    { 122, 74 }, { 75, 123 }, { 124, 76 }, { 77, 125 }, { 124, 76 }, { 77, 125 }, { 126, 126 }, { 127, 127 }
};

/* LPS range table: g_hevc_lps_range[state][(range>>6)&3]. */
static const uint8_t g_hevc_lps_range[64][4] = {
    { 128, 176, 208, 240 }, { 128, 167, 197, 227 }, { 128, 158, 187, 216 }, { 123, 150, 178, 205 },
    { 116, 142, 169, 195 }, { 111, 135, 160, 185 }, { 105, 128, 152, 175 }, { 100, 122, 144, 166 },
    {  95, 116, 137, 158 }, {  90, 110, 130, 150 }, {  85, 104, 123, 142 }, {  81,  99, 117, 135 },
    {  77,  94, 111, 128 }, {  73,  89, 105, 122 }, {  69,  85, 100, 116 }, {  66,  80,  95, 110 },
    {  62,  76,  90, 104 }, {  59,  72,  86,  99 }, {  56,  69,  81,  94 }, {  53,  65,  77,  89 },
    {  51,  62,  73,  85 }, {  48,  59,  69,  80 }, {  46,  56,  66,  76 }, {  43,  53,  63,  72 },
    {  41,  50,  59,  69 }, {  39,  48,  56,  65 }, {  37,  45,  54,  62 }, {  35,  43,  51,  59 },
    {  33,  41,  48,  56 }, {  32,  39,  46,  53 }, {  30,  37,  43,  50 }, {  29,  35,  41,  48 },
    {  27,  33,  39,  45 }, {  26,  31,  37,  43 }, {  24,  30,  35,  41 }, {  23,  28,  33,  39 },
    {  22,  27,  32,  37 }, {  21,  26,  30,  35 }, {  20,  24,  29,  33 }, {  19,  23,  27,  31 },
    {  18,  22,  26,  30 }, {  17,  21,  25,  28 }, {  16,  20,  23,  27 }, {  15,  19,  22,  25 },
    {  14,  18,  21,  24 }, {  14,  17,  20,  23 }, {  13,  16,  19,  22 }, {  12,  15,  18,  21 },
    {  12,  14,  17,  20 }, {  11,  14,  16,  19 }, {  11,  13,  15,  18 }, {  10,  12,  15,  17 },
    {  10,  12,  14,  16 }, {   9,  11,  13,  15 }, {   9,  11,  12,  14 }, {   8,  10,  12,  14 },
    {   8,   9,  11,  13 }, {   7,   9,  11,  12 }, {   7,   9,  10,  12 }, {   7,   8,  10,  11 },
    {   6,   8,   9,  11 }, {   6,   7,   9,  10 }, {   6,   7,   8,   9 }, {   2,   2,   2,   2 }
};

/* 4x4 scan tables: scan position -> raster index (row*4+col). Index 0 =
 * diagonal (up-right), 1 = horizontal, 2 = vertical - matches x265's
 * SCAN_DIAG/SCAN_HOR/SCAN_VER numbering, which hevc_intra.c's
 * hevc_scan_idx_for_mode() also follows. */
static const uint8_t g_hevc_scan4x4[3][16] = {
    { 0,  4,  1,  8,  5,  2, 12,  9,  6,  3, 13, 10,  7, 14, 11, 15 },
    { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
    { 0,  4,  8, 12,  1,  5,  9, 13,  2,  6, 10, 14,  3,  7, 11, 15 }
};

/* last_sig_coeff_{x,y}_prefix per-position (prefixOnes | suffixLen<<4)
 * table, restricted to the 4 positions a 4x4 block can have (values 0-3
 * never need a suffix - see x265's g_lastCoeffTable, first 4 entries). */
static const uint8_t g_hevc_last_ctx4[4] = { 0x00, 0x01, 0x02, 0x03 };

/* sig_coeff_flag context increment for a lone 4x4 coefficient group,
 * indexed directly by raster block position (0-15) - x265's
 * table_cnt[4][...] (the "4x4" special case of its 5-entry pattern table). */
static const uint8_t g_hevc_sig_ctx4[16] = {
    0, 1, 4, 5,
    2, 3, 4, 5,
    6, 6, 8, 8,
    7, 7, 8, 8
};

#define COEF_REMAIN_BIN_REDUCTION 3
#define C1FLAG_NUMBER             8

/* What coding a bin costs, in 1/32768 of a bit, by context state:
 * [state << 1] for the most probable symbol, [state << 1 | 1] for the
 * least. From the probability the state stands for, 9.3.4.3.2: pLPS =
 * 0.5 * a^state with a = (0.01875 / 0.5)^(1/63). */
static const uint32_t g_hevc_entropy_bits[128] = {
     32768,  32768,  30426,  35232,  28306,  37696,  26377,  40159,
     24617,  42623,  23005,  45087,  21523,  47551,  20159,  50015,
     18899,  52479,  17734,  54942,  16653,  57406,  15650,  59870,
     14717,  62334,  13849,  64798,  13038,  67262,  12282,  69725,
     11575,  72189,  10914,  74653,  10294,  77117,   9714,  79581,
      9169,  82044,   8658,  84508,   8178,  86972,   7727,  89436,
      7303,  91900,   6903,  94364,   6527,  96827,   6173,  99291,
      5840, 101755,   5525, 104219,   5228, 106683,   4948, 109147,
      4684, 111610,   4435, 114074,   4199, 116538,   3977, 119002,
      3767, 121466,   3568, 123929,   3380, 126393,   3202, 128857,
      3034, 131321,   2876, 133785,   2725, 136249,   2583, 138712,
      2448, 141176,   2321, 143640,   2200, 146104,   2086, 148568,
      1978, 151032,   1875, 153495,   1778, 155959,   1686, 158423,
      1599, 160887,   1517, 163351,   1439, 165814,   1364, 168278,
      1294, 170742,   1228, 173206,   1164, 175670,   1105, 178134,
      1048, 180597,    994, 183061,    943, 185525,    943, 185525,
};

/* ===================== Context init values (Rec. ITU-T H.265 9.3.2.2) ==== */
/* Row index 0 = P-slice (initType 1), Row index 1 = I-slice (initType 2).
 * Values match ITU-T H.265 Tables 9-5 through 9-30 and x265 entropy.cpp. */

static const uint8_t INIT_SPLIT_FLAG[2][3] = {
    { 107, 139, 126 }, /* P-slice */
    { 139, 141, 157 }, /* I-slice */
};
static const uint8_t INIT_PART_SIZE[2]           = { 154, 184 }; /* ctx index 0 */
static const uint8_t INIT_INTRA_PRED_MODE[2]     = { 154, 184 };
static const uint8_t INIT_CHROMA_PRED_MODE[2][2] = {
    { 152, 139 }, /* P-slice */
    {  63, 139 }, /* I-slice */
};
static const uint8_t INIT_QT_CBF[2][7] = {
    { 153, 111, 149, 107, 167, 154, 154 }, /* P-slice */
    { 111, 141,  94, 138, 182, 154, 154 }, /* I-slice */
};
static const uint8_t INIT_SIG_FLAG[2][42] = {
    { 155, 154, 139, 153, 139, 123, 123,  63, 153, 166, 183, 140, 136, 153, 154,
      166, 183, 140, 136, 153, 154, 166, 183, 140, 136, 153, 154, 170, 153, 123,
      123, 107, 121, 107, 121, 167, 151, 183, 140, 151, 183, 140 }, /* P-slice */
    { 111, 111, 125, 110, 110,  94, 124, 108, 124, 107, 125, 141, 179, 153, 125,
      107, 125, 141, 179, 153, 125, 107, 125, 141, 179, 153, 125, 140, 139, 182,
      182, 152, 136, 152, 136, 153, 136, 139, 111, 136, 139, 111 }, /* I-slice */
};
static const uint8_t INIT_LAST[2][18] = {
    { 125, 110,  94, 110,  95,  79, 125, 111, 110,  78, 110, 111, 111,  95,  94, 108, 123, 108 }, /* P-slice */
    { 110, 110, 124, 125, 140, 153, 125, 127, 140, 109, 111, 143, 127, 111,  79, 108, 123,  63 }, /* I-slice */
};
static const uint8_t INIT_ONE_FLAG[2][24] = {
    { 154, 196, 196, 167, 154, 152, 167, 182, 182, 134, 149, 136,
      153, 121, 136, 137, 169, 194, 166, 167, 154, 167, 137, 182 }, /* P-slice */
    { 140,  92, 137, 138, 140, 152, 138, 139, 153,  74, 149,  92,
      139, 107, 122, 152, 140, 179, 166, 182, 140, 227, 122, 197 }, /* I-slice */
};
static const uint8_t INIT_ABS_FLAG[2][6] = {
    { 107, 167,  91, 122, 107, 167 }, /* P-slice */
    { 138, 153, 136, 167, 152, 152 }, /* I-slice */
};

/* Inter/Skip syntax elements (used only in P-slices, initType 1) */
static const uint8_t INIT_SKIP_FLAG[3] = { 197, 185, 201 };
static const uint8_t INIT_PRED_MODE    = 149;
static const uint8_t INIT_MERGE_FLAG   = 110;
static const uint8_t INIT_MERGE_IDX    = 122;
/* Tables 9-20, 9-21 and 9-32, initType 1. */
static const uint8_t INIT_MVD[2]       = { 140, 198 };
static const uint8_t INIT_MVP_IDX      = 168;
static const uint8_t INIT_ROOT_CBF     = 79;
/* Tables 9-14 and 9-26: split_transform_flag and coded_sub_block_flag,
 * rows as the others here - P-slice first. */
static const uint8_t INIT_TRANS_SUBDIV[2][3] = { { 124, 138, 94 }, { 153, 138, 138 } };
static const uint8_t INIT_SIG_CG[2][4] = { { 121, 140, 61, 154 }, { 91, 171, 134, 141 } };

/* ===================== context init formula (Rec. ITU-T H.265 9.3.2.2) === */

static uint8_t hevc_sbac_init_state(int qp, int init_value) {
    if (qp < 0) qp = 0;
    if (qp > 51) qp = 51;
    int slope = (init_value >> 4) * 5 - 45;
    int offset = ((init_value & 15) << 3) - 16;
    int init_state = ((slope * qp) >> 4) + offset;
    if (init_state < 1) init_state = 1;
    if (init_state > 126) init_state = 126;
    uint32_t mps = (uint32_t)(init_state >= 64);
    uint32_t state = ((mps ? (uint32_t)(init_state - 64) : (uint32_t)(63 - init_state)) << 1) + mps;
    return (uint8_t)state;
}

static void init_bank(uint8_t *ctx, const uint8_t *init_values, int count, int qp) {
    for (int i = 0; i < count; i++) ctx[i] = hevc_sbac_init_state(qp, init_values[i]);
}

void hevc_cabac_reset_contexts(hevc_cabac_t *cb, int slice_qp, int slice_type) {
    int type_idx = (slice_type == 1) ? 0 : 1;

    init_bank(&cb->ctx[HEVC_CTX_SPLIT_FLAG], INIT_SPLIT_FLAG[type_idx], 3, slice_qp);
    cb->ctx[HEVC_CTX_PART_SIZE] = hevc_sbac_init_state(slice_qp, INIT_PART_SIZE[type_idx]);
    cb->ctx[HEVC_CTX_INTRA_PRED] = hevc_sbac_init_state(slice_qp, INIT_INTRA_PRED_MODE[type_idx]);
    init_bank(&cb->ctx[HEVC_CTX_CHROMA_PRED], INIT_CHROMA_PRED_MODE[type_idx], 2, slice_qp);
    init_bank(&cb->ctx[HEVC_CTX_QT_CBF], INIT_QT_CBF[type_idx], 7, slice_qp);
    init_bank(&cb->ctx[HEVC_CTX_SIG_FLAG], INIT_SIG_FLAG[type_idx], 42, slice_qp);
    init_bank(&cb->ctx[HEVC_CTX_LAST_X], INIT_LAST[type_idx], 18, slice_qp);
    init_bank(&cb->ctx[HEVC_CTX_LAST_Y], INIT_LAST[type_idx], 18, slice_qp);
    init_bank(&cb->ctx[HEVC_CTX_ONE_FLAG], INIT_ONE_FLAG[type_idx], 24, slice_qp);
    init_bank(&cb->ctx[HEVC_CTX_ABS_FLAG], INIT_ABS_FLAG[type_idx], 6, slice_qp);
    init_bank(&cb->ctx[HEVC_CTX_TRANS_SUBDIV], INIT_TRANS_SUBDIV[type_idx], 3, slice_qp);
    init_bank(&cb->ctx[HEVC_CTX_SIG_CG], INIT_SIG_CG[type_idx], 4, slice_qp);

    if (slice_type == 1) {
        init_bank(&cb->ctx[HEVC_CTX_SKIP_FLAG], INIT_SKIP_FLAG, 3, slice_qp);
        cb->ctx[HEVC_CTX_PRED_MODE] = hevc_sbac_init_state(slice_qp, INIT_PRED_MODE);
        cb->ctx[HEVC_CTX_MERGE_FLAG] = hevc_sbac_init_state(slice_qp, INIT_MERGE_FLAG);
        cb->ctx[HEVC_CTX_MERGE_IDX] = hevc_sbac_init_state(slice_qp, INIT_MERGE_IDX);
        init_bank(&cb->ctx[HEVC_CTX_MVD], INIT_MVD, 2, slice_qp);
        cb->ctx[HEVC_CTX_MVP_IDX] = hevc_sbac_init_state(slice_qp, INIT_MVP_IDX);
        cb->ctx[HEVC_CTX_ROOT_CBF] = hevc_sbac_init_state(slice_qp, INIT_ROOT_CBF);
    }
}

/* ===================== arithmetic coder core ===================== */

void hevc_cabac_init(hevc_cabac_t *cb, bitstream_t *bs) {
    memset(cb, 0, sizeof(*cb));
    cb->bs = bs;
}

void hevc_cabac_start(hevc_cabac_t *cb) {
    cb->low = 0;
    cb->range = 510;
    cb->bits_left = -12;
    cb->num_buffered_bytes = 0;
    cb->buffered_byte = 0xff;
}

static void cabac_put_byte(hevc_cabac_t *cb, uint8_t b) {
    bs_write_u(cb->bs, 8, b);
}

static void cabac_write_out(hevc_cabac_t *cb) {
    uint32_t lead_byte = cb->low >> (13 + cb->bits_left);
    uint32_t low_mask = (uint32_t)(~0u) >> (11 + 8 - cb->bits_left);

    cb->bits_left -= 8;
    cb->low &= low_mask;

    if (lead_byte == 0xff) {
        cb->num_buffered_bytes++;
    } else {
        if (cb->num_buffered_bytes > 0) {
            uint32_t carry = lead_byte >> 8;
            uint32_t byte_to_write = cb->buffered_byte + carry;
            cabac_put_byte(cb, (uint8_t)byte_to_write);

            byte_to_write = (0xff + carry) & 0xff;
            while (cb->num_buffered_bytes > 1) {
                cabac_put_byte(cb, (uint8_t)byte_to_write);
                cb->num_buffered_bytes--;
            }
        }
        cb->num_buffered_bytes = 1;
        cb->buffered_byte = lead_byte;
    }
}

/* The arithmetic coding of one bin whose context has already moved on
 * from `mstate`. */
static void encode_bin_coded(hevc_cabac_t *cb, uint32_t mstate, uint32_t bin) {
    uint32_t range = cb->range;
    uint32_t state = mstate >> 1;
    uint32_t lps = g_hevc_lps_range[state][(range >> 6) & 3];
    range -= lps;

    int num_bits = (int)(((uint32_t)(range - 256)) >> 31);
    uint32_t low = cb->low;

    if ((bin ^ mstate) & 1u) {
        unsigned idx = 31u - (unsigned)__builtin_clz(lps);
        num_bits = (int)(8 - idx);
        if (state >= 63) num_bits = 6;
        low += range;
        range = lps;
    }
    cb->low = low << num_bits;
    cb->range = range << num_bits;
    cb->bits_left += num_bits;
    while (cb->bits_left >= 0) cabac_write_out(cb);
}

/* Every bin of the syntax below comes through here. Estimating is most of
 * what the encoder does with them - each candidate of each CU runs its
 * syntax through an estimator - so that path is inline in every caller,
 * and only real coding pays for a call. Called through the exported
 * function it was 9% of the encoder's time. */
static inline __attribute__((always_inline)) void encode_bin(hevc_cabac_t *cb, int ctx_idx, uint32_t bin) {
    const uint32_t mstate = cb->ctx[ctx_idx];
    cb->ctx[ctx_idx] = g_hevc_next_state[mstate][bin & 1];
    if (cb->est) {
        cb->est_bits += g_hevc_entropy_bits[mstate ^ (bin & 1)];
        return;
    }
    encode_bin_coded(cb, mstate, bin);
}

void hevc_cabac_encode_bin(hevc_cabac_t *cb, int ctx_idx, uint32_t bin) {
    encode_bin(cb, ctx_idx, bin);
}

static void encode_bypass_coded(hevc_cabac_t *cb, uint32_t bin) {
    cb->low <<= 1;
    if (bin) cb->low += cb->range;
    cb->bits_left++;
    while (cb->bits_left >= 0) cabac_write_out(cb);
}

static inline __attribute__((always_inline)) void encode_bypass(hevc_cabac_t *cb, uint32_t bin) {
    if (cb->est) { cb->est_bits += 32768; return; }
    encode_bypass_coded(cb, bin);
}

void hevc_cabac_encode_bypass(hevc_cabac_t *cb, uint32_t bin) {
    encode_bypass(cb, bin);
}

static void encode_bypass_bins_coded(hevc_cabac_t *cb, uint32_t value, int num_bins) {
    while (num_bins > 8) {
        num_bins -= 8;
        uint32_t pattern = value >> num_bins;
        cb->low <<= 8;
        cb->low += cb->range * pattern;
        value -= pattern << num_bins;
        cb->bits_left += 8;
        while (cb->bits_left >= 0) cabac_write_out(cb);
    }
    cb->low <<= num_bins;
    cb->low += cb->range * value;
    cb->bits_left += num_bins;
    while (cb->bits_left >= 0) cabac_write_out(cb);
}

static inline __attribute__((always_inline)) void encode_bypass_bins(hevc_cabac_t *cb, uint32_t value, int num_bins) {
    if (cb->est) { cb->est_bits += 32768u * (uint32_t)num_bins; return; }
    encode_bypass_bins_coded(cb, value, num_bins);
}

void hevc_cabac_encode_bypass_bins(hevc_cabac_t *cb, uint32_t value, int num_bins) {
    encode_bypass_bins(cb, value, num_bins);
}

void hevc_cabac_encode_terminate(hevc_cabac_t *cb, uint32_t bin) {
    if (cb->est) return;
    cb->range -= 2;
    if (bin) {
        cb->low += cb->range;
        cb->low <<= 7;
        cb->range = 2 << 7;
        cb->bits_left += 7;
    } else if (cb->range >= 256) {
        return;
    } else {
        cb->low <<= 1;
        cb->range <<= 1;
        cb->bits_left++;
    }
    while (cb->bits_left >= 0) cabac_write_out(cb);
}

void hevc_cabac_finish(hevc_cabac_t *cb) {
    if (cb->low >> (21 + cb->bits_left)) {
        cabac_put_byte(cb, (uint8_t)(cb->buffered_byte + 1));
        while (cb->num_buffered_bytes > 1) {
            cabac_put_byte(cb, 0x00);
            cb->num_buffered_bytes--;
        }
        cb->low -= 1u << (21 + cb->bits_left);
    } else {
        if (cb->num_buffered_bytes > 0)
            cabac_put_byte(cb, (uint8_t)cb->buffered_byte);
        while (cb->num_buffered_bytes > 1) {
            cabac_put_byte(cb, 0xff);
            cb->num_buffered_bytes--;
        }
    }
    /* Emit the remaining (13 + bits_left) bits of low, MSB-first - same as
     * x265's m_bitIf->write(m_low>>8, 13+m_bitsLeft). This is generally
     * NOT a whole number of bytes, so it goes through bs_write_u() (the
     * same bit-level accumulator CAVLC uses) rather than cabac_put_byte();
     * the caller finishes byte-alignment with bs_rbsp_trailing_bits() on
     * the same bitstream_t right after this call (see encoder_h265.c). */
    int nbits = 13 + cb->bits_left;
    if (nbits > 0)
        bs_write_u(cb->bs, nbits, cb->low >> 8);
}

/* ===================== syntax element wrappers ===================== */

void hevc_cabac_code_cu_skip_flag(hevc_cabac_t *cb, int skip, int ctx_inc) {
    encode_bin(cb, HEVC_CTX_SKIP_FLAG + ctx_inc, (uint32_t)(skip ? 1 : 0));
}

void hevc_cabac_code_pred_mode_flag(hevc_cabac_t *cb, int pred_mode) {
    encode_bin(cb, HEVC_CTX_PRED_MODE, (uint32_t)(pred_mode ? 1 : 0));
}

void hevc_cabac_code_merge_idx(hevc_cabac_t *cb, int merge_idx) {
    /* Truncated Unary (TU) binarization for merge_idx with cMax = 4:
     * - bin 0 is context-coded with HEVC_CTX_MERGE_IDX (0 if merge_idx == 0, else 1)
     * - bins 1..3 (if merge_idx > 0) are bypass-coded:
     *   (merge_idx - 1) bypass 1s, followed by terminating bypass 0 if merge_idx < 4. */
    if (merge_idx <= 0) {
        encode_bin(cb, HEVC_CTX_MERGE_IDX, 0);
        return;
    }
    encode_bin(cb, HEVC_CTX_MERGE_IDX, 1);
    for (int i = 0; i < merge_idx - 1; i++) {
        encode_bypass(cb, 1);
    }
    if (merge_idx < 4) {
        encode_bypass(cb, 0);
    }
}

void hevc_cabac_code_merge_flag(hevc_cabac_t *cb, int merge) {
    encode_bin(cb, HEVC_CTX_MERGE_FLAG, (uint32_t)(merge ? 1 : 0));
}

/* k-th order Exp-Golomb in bypass bins (9.3.3.6), for abs_mvd_minus2 with
 * k = 1. */
static void write_ep_exp_golomb(hevc_cabac_t *cb, uint32_t symbol, uint32_t k) {
    uint32_t bins = 0;
    int num_bins = 0;
    while (symbol >= (1u << k)) {
        bins = 2 * bins + 1;
        num_bins++;
        symbol -= 1u << k;
        k++;
    }
    bins = 2 * bins;
    num_bins++;
    bins = (bins << k) | symbol;
    num_bins += (int)k;
    encode_bypass_bins(cb, bins, num_bins);
}

/* 7.3.8.9 mvd_coding(): both greater0 flags, both greater1 flags, then per
 * component the remainder and the sign. */
void hevc_cabac_code_mvd(hevc_cabac_t *cb, int mvd_x, int mvd_y) {
    const uint32_t ax = (uint32_t)(mvd_x < 0 ? -mvd_x : mvd_x);
    const uint32_t ay = (uint32_t)(mvd_y < 0 ? -mvd_y : mvd_y);
    encode_bin(cb, HEVC_CTX_MVD, ax > 0);
    encode_bin(cb, HEVC_CTX_MVD, ay > 0);
    if (ax > 0) encode_bin(cb, HEVC_CTX_MVD + 1, ax > 1);
    if (ay > 0) encode_bin(cb, HEVC_CTX_MVD + 1, ay > 1);
    if (ax > 0) {
        if (ax > 1) write_ep_exp_golomb(cb, ax - 2, 1);
        encode_bypass(cb, mvd_x < 0);
    }
    if (ay > 0) {
        if (ay > 1) write_ep_exp_golomb(cb, ay - 2, 1);
        encode_bypass(cb, mvd_y < 0);
    }
}

void hevc_cabac_code_mvp_idx(hevc_cabac_t *cb, int idx) {
    encode_bin(cb, HEVC_CTX_MVP_IDX, (uint32_t)(idx ? 1 : 0));
}

void hevc_cabac_code_split_transform_flag(hevc_cabac_t *cb, int split, int log2_size) {
    encode_bin(cb, HEVC_CTX_TRANS_SUBDIV + 5 - log2_size, (uint32_t)(split ? 1 : 0));
}

void hevc_cabac_code_rqt_root_cbf(hevc_cabac_t *cb, int cbf) {
    encode_bin(cb, HEVC_CTX_ROOT_CBF, (uint32_t)(cbf ? 1 : 0));
}

void hevc_cabac_code_split_cu_flag(hevc_cabac_t *cb, int bin, int ctx_inc) {
    encode_bin(cb, HEVC_CTX_SPLIT_FLAG + ctx_inc, (uint32_t)(bin ? 1 : 0));
}

void hevc_cabac_code_part_mode_intra(hevc_cabac_t *cb, int is_2nx2n) {
    encode_bin(cb, HEVC_CTX_PART_SIZE, (uint32_t)(is_2nx2n ? 1 : 0));
}

int hevc_cabac_code_intra_luma_flag(hevc_cabac_t *cb, int mode, const int mpm[3]) {
    int pred_idx = -1;
    for (int i = 0; i < 3; i++) {
        if (mode == mpm[i]) { pred_idx = i; break; }
    }
    encode_bin(cb, HEVC_CTX_INTRA_PRED, (uint32_t)(pred_idx != -1 ? 1 : 0));
    return pred_idx;
}

void hevc_cabac_code_intra_luma_data(hevc_cabac_t *cb, int mode, int pred_idx, const int mpm_in[3]) {
    if (pred_idx != -1) {
        int nonzero = (pred_idx != 0);
        encode_bypass_bins(cb, (uint32_t)(pred_idx + nonzero), 1 + nonzero);
    } else {
        int mpm[3] = { mpm_in[0], mpm_in[1], mpm_in[2] };
        if (mpm[0] > mpm[1]) { int t = mpm[0]; mpm[0] = mpm[1]; mpm[1] = t; }
        if (mpm[0] > mpm[2]) { int t = mpm[0]; mpm[0] = mpm[2]; mpm[2] = t; }
        if (mpm[1] > mpm[2]) { int t = mpm[1]; mpm[1] = mpm[2]; mpm[2] = t; }
        int dir = mode;
        dir += (dir > mpm[2]) ? -1 : 0;
        dir += (dir > mpm[1]) ? -1 : 0;
        dir += (dir > mpm[0]) ? -1 : 0;
        encode_bypass_bins(cb, (uint32_t)dir, 5);
    }
}

void hevc_cabac_code_intra_chroma_pred_mode(hevc_cabac_t *cb, int luma_mode_pu0) {
    if (luma_mode_pu0 == 1) {
        /* DM_CHROMA (derived == luma): luma is already DC, so chroma == DC. */
        encode_bin(cb, HEVC_CTX_CHROMA_PRED, 0);
    } else {
        /* Candidate list {Planar,Vertical,Horizontal,DC} has DC untouched
         * at index 3 whenever luma mode isn't DC itself, so index 3 always
         * yields chroma==DC here. */
        encode_bin(cb, HEVC_CTX_CHROMA_PRED, 1);
        encode_bypass_bins(cb, 3, 2);
    }
}

void hevc_cabac_code_cbf_luma(hevc_cabac_t *cb, int cbf, int trafo_depth) {
    int ctx = (trafo_depth == 0) ? 1 : 0;
    encode_bin(cb, HEVC_CTX_QT_CBF + ctx, (uint32_t)(cbf ? 1 : 0));
}

void hevc_cabac_code_cbf_chroma(hevc_cabac_t *cb, int cbf, int trafo_depth) {
    int ctx = 2 + trafo_depth;
    encode_bin(cb, HEVC_CTX_QT_CBF + ctx, (uint32_t)(cbf ? 1 : 0));
}

/* ===================== residual_coding() for one 4x4 TU ===================== */

static void write_coef_remain_exp_golomb(hevc_cabac_t *cb, uint32_t code_number, uint32_t rice) {
    uint32_t code_remain = code_number & ((1u << rice) - 1);
    if ((code_number >> rice) < COEF_REMAIN_BIN_REDUCTION) {
        uint32_t length = code_number >> rice;
        encode_bypass_bins(cb, (((1u << (length + 1)) - 2) << rice) + code_remain,
                                       (int)(length + 1 + rice));
    } else {
        uint32_t cn = (code_number >> rice) - COEF_REMAIN_BIN_REDUCTION;
        unsigned idx = 31u - (unsigned)__builtin_clz(cn + 1);
        uint32_t length = idx;
        cn -= (1u << idx) - 1;
        cn = (cn << rice) + code_remain;
        encode_bypass_bins(cb, (1u << (COEF_REMAIN_BIN_REDUCTION + length + 1)) - 2,
                                       (int)(COEF_REMAIN_BIN_REDUCTION + length + 1));
        encode_bypass_bins(cb, cn, (int)(length + rice));
    }
}

void hevc_cabac_code_residual_4x4(hevc_cabac_t *cb, const int16_t coeff[16],
                                   int is_luma, int scan_idx) {
    const uint8_t *scan = g_hevc_scan4x4[scan_idx];

    /* Find last significant scan position. */
    int scan_pos_last = -1;
    for (int sp = 15; sp >= 0; sp--) {
        if (coeff[scan[sp]] != 0) { scan_pos_last = sp; break; }
    }
    if (scan_pos_last < 0) return; /* caller must not call this when cbf==0 */

    int pos_raster = scan[scan_pos_last];
    int pos[2] = { pos_raster & 3, pos_raster >> 2 }; /* [0]=x, [1]=y */
    if (scan_idx == 2) { int t = pos[0]; pos[0] = pos[1]; pos[1] = t; }

    /* last_sig_coeff_{x,y}_prefix (+ suffix, always length 0 for 4x4). */
    int ctx_base = is_luma ? 0 : 15;
    for (int i = 0; i < 2; i++) {
        int bank = (i == 0) ? HEVC_CTX_LAST_X : HEVC_CTX_LAST_Y;
        uint8_t temp = g_hevc_last_ctx4[pos[i]];
        int prefix_ones = temp & 15;
        for (int c = 0; c < prefix_ones; c++)
            encode_bin(cb, bank + ctx_base + c, 1);
        if (prefix_ones < 3)
            encode_bin(cb, bank + ctx_base + prefix_ones, 0);
    }

    /* Significance map + gather absolute levels (scan order, decreasing
     * from scan_pos_last down to 0 - absCoeff[0] is always the last-scan-
     * position coefficient itself, inferred significant, never coded). */
    int16_t abs_coeff[16];
    uint32_t sign_bits;
    int num_nonzero = 1;
    int last_val = coeff[pos_raster];
    abs_coeff[0] = (int16_t)(last_val < 0 ? -last_val : last_val);
    sign_bits = (uint32_t)(last_val < 0);

    int sig_base = is_luma ? 0 : 27;
    for (int sp = scan_pos_last - 1; sp >= 0; sp--) {
        int raster = scan[sp];
        int val = coeff[raster];
        int sig = (val != 0);
        int ctx_sig = g_hevc_sig_ctx4[raster];
        encode_bin(cb, HEVC_CTX_SIG_FLAG + sig_base + ctx_sig, (uint32_t)sig);
        if (sig) {
            abs_coeff[num_nonzero] = (int16_t)(val < 0 ? -val : val);
            sign_bits = (sign_bits << 1) | (uint32_t)(val < 0);
            num_nonzero++;
        }
    }

    /* Greater-than-1 flags. ctxSet is always 0 here: this encoder's 4x4 TUs
     * are always exactly one coefficient group, so x265's
     * ctxSet=(((subSet>0)+bIsLuma)&2)+!(c1&3) always reduces to
     * !(c1&3) with c1's initial value of 1, i.e. 0. */
    int one_base = HEVC_CTX_ONE_FLAG + (is_luma ? 0 : 16);
    int abs_base = HEVC_CTX_ABS_FLAG + (is_luma ? 0 : 4);

    uint32_t c1 = 1, c1_next = 0xFFFFFFFEu;
    int first_c2_idx = 8, first_c2_flag = 2;
    int num_c1_flag = num_nonzero < C1FLAG_NUMBER ? num_nonzero : C1FLAG_NUMBER;
    for (int idx = 0; idx < num_c1_flag; idx++) {
        int symbol1 = abs_coeff[idx] > 1;
        int symbol2 = abs_coeff[idx] > 2;
        encode_bin(cb, one_base + (int)c1, (uint32_t)symbol1);
        if (symbol1) c1_next = 0;
        if (symbol1 + first_c2_flag == 3) first_c2_flag = symbol2;
        if (symbol1 + first_c2_idx == 9) first_c2_idx = idx;
        c1 = c1_next & 3;
        c1_next >>= 2;
    }
    if (!c1) {
        encode_bin(cb, abs_base, (uint32_t)first_c2_flag);
    }

    /* Sign bits (bypass), decreasing-scan-position order, no sign hiding
     * (this project's PPS sets sign_data_hiding_flag=0). Batched into a
     * single bypass emission call, saving num_nonzero-1 branch/write loops. */
    encode_bypass_bins(cb, sign_bits, num_nonzero);

    /* coeff_abs_level_remaining. */
    if (!c1 || num_nonzero > C1FLAG_NUMBER) {
        uint32_t go_rice = 0;
        int base_level = 3;
        uint32_t threshold = COEF_REMAIN_BIN_REDUCTION;
        int idx = first_c2_idx;
        do {
            if (idx >= C1FLAG_NUMBER) base_level = 1;
            if ((uint32_t)abs_coeff[idx] >= (uint32_t)base_level) {
                write_coef_remain_exp_golomb(cb, (uint32_t)(abs_coeff[idx] - base_level), go_rice);
                int adjust = (abs_coeff[idx] > (int)threshold) && (go_rice <= 3);
                if (adjust) { go_rice++; threshold += threshold; }
            }
            base_level = 2;
            idx++;
        } while (idx < num_nonzero);
    }
}

/* ===================== residual_coding() for one 8x8 TU ===================== */

/* The four 4x4 sub-blocks of an 8x8 block in diagonal scan order, as
 * yS * 2 + xS: (0,0), (0,1), (1,0), (1,1) - 6.5.3 for a 2x2 array. */
static const uint8_t g_hevc_sb_scan2x2[4] = { 0, 2, 1, 3 };

/* last_sig_coeff_*_prefix group and the first position of each group,
 * positions 0..7 (Table 9-4's binarization, 9.3.3.2). */
static const uint8_t g_hevc_last_group8[8] = { 0, 1, 2, 3, 4, 4, 5, 5 };
static const uint8_t g_hevc_last_min8[6] = { 0, 1, 2, 3, 4, 6 };

/* One 8x8 transform block, diagonal scan - every inter luma block at this
 * size. `coeff` is raster, coeff[y * 8 + x]. Written from 7.3.8.11 and
 * 9.3.4.2.4-9.3.4.2.7 directly. */
void hevc_cabac_code_residual_8x8(hevc_cabac_t *cb, const int16_t coeff[64], int is_luma) {
    const uint8_t *scan = g_hevc_scan4x4[0];
    int last_n = -1;
    for (int n = 63; n >= 0; n--) {
        const int sb = g_hevc_sb_scan2x2[n >> 4], p = scan[n & 15];
        const int x = (sb & 1) * 4 + (p & 3), y = (sb >> 1) * 4 + (p >> 2);
        if (coeff[y * 8 + x]) { last_n = n; break; }
    }
    if (last_n < 0) return;

    /* last_sig_coeff_x/y: both prefixes, then both suffixes. For 8x8 the
     * prefix contexts are offset 3 (luma) or 15 (chroma), shifted by 1. */
    {
        const int sb = g_hevc_sb_scan2x2[last_n >> 4], p = scan[last_n & 15];
        const int pos[2] = { (sb & 1) * 4 + (p & 3), (sb >> 1) * 4 + (p >> 2) };
        const int off = is_luma ? 3 : 15;
        for (int i = 0; i < 2; i++) {
            const int bank = i == 0 ? HEVC_CTX_LAST_X : HEVC_CTX_LAST_Y;
            const int prefix = g_hevc_last_group8[pos[i]];
            for (int b = 0; b < prefix; b++) encode_bin(cb, bank + off + (b >> 1), 1);
            if (prefix < 5) encode_bin(cb, bank + off + (prefix >> 1), 0);
        }
        for (int i = 0; i < 2; i++) {
            const int prefix = g_hevc_last_group8[pos[i]];
            if (prefix > 3)
                encode_bypass_bins(cb, (uint32_t)(pos[i] - g_hevc_last_min8[prefix]),
                                              (prefix >> 1) - 1);
        }
    }

    const int last_sb = last_n >> 4;
    uint8_t csbf[2][2] = { { 0, 0 }, { 0, 0 } };   /* [yS][xS] */
    const int one_base = HEVC_CTX_ONE_FLAG + (is_luma ? 0 : 16);
    const int abs_base = HEVC_CTX_ABS_FLAG + (is_luma ? 0 : 4);
    const int sig_base = HEVC_CTX_SIG_FLAG + (is_luma ? 0 : 27);
    const int cg_base = HEVC_CTX_SIG_CG + (is_luma ? 0 : 2);
    int c1_prev = 1;   /* greater1Ctx left by the previous sub-block */

    for (int i = last_sb; i >= 0; i--) {
        const int sb = g_hevc_sb_scan2x2[i], xs = sb & 1, ys = sb >> 1;
        int16_t lv[16];
        int any = 0;
        for (int n = 0; n < 16; n++) {
            const int p = scan[n];
            lv[n] = coeff[(ys * 4 + (p >> 2)) * 8 + xs * 4 + (p & 3)];
            any |= lv[n];
        }
        const int right = xs < 1 ? csbf[ys][xs + 1] : 0;
        const int below = ys < 1 ? csbf[ys + 1][xs] : 0;

        /* coded_sub_block_flag: coded between the last and the first. */
        int infer_dc = 0;
        if (i < last_sb && i > 0) {
            encode_bin(cb, cg_base + ((right | below) ? 1 : 0), any ? 1 : 0);
            csbf[ys][xs] = (uint8_t)(any ? 1 : 0);
            if (!any) continue;
            infer_dc = 1;
        } else {
            csbf[ys][xs] = 1;
        }

        /* sig_coeff_flag, from the position before the last one down. */
        const int prev_csbf = right | (below << 1);
        const int start = i == last_sb ? (last_n & 15) - 1 : 15;
        int16_t absv[16];
        uint32_t signs = 0;
        int num = 0;
        if (i == last_sb) {
            const int v = lv[last_n & 15];
            absv[num++] = (int16_t)(v < 0 ? -v : v);
            signs = (uint32_t)(v < 0);
        }
        for (int n = start; n >= 0; n--) {
            const int v = lv[n];
            if (n == 0 && infer_dc) {
                /* inferred 1: this sub-block is coded and nothing else in
                 * it was. */
                absv[num++] = (int16_t)(v < 0 ? -v : v);
                signs = (signs << 1) | (uint32_t)(v < 0);
                break;
            }
            const int p = scan[n], xp = p & 3, yp = p >> 2;
            int sig;
            if (sb == 0 && p == 0) {
                sig = 0;
            } else {
                switch (prev_csbf) {
                case 0: sig = (xp + yp == 0) ? 2 : (xp + yp < 3) ? 1 : 0; break;
                case 1: sig = yp == 0 ? 2 : (yp == 1 ? 1 : 0); break;
                case 2: sig = xp == 0 ? 2 : (xp == 1 ? 1 : 0); break;
                default: sig = 2; break;
                }
                if (is_luma) {
                    if (sb != 0) sig += 3;
                    sig += 9;              /* 8x8, diagonal scan */
                } else {
                    sig += 9;              /* 8x8 chroma */
                }
            }
            encode_bin(cb, sig_base + sig, v != 0);
            if (v) {
                absv[num++] = (int16_t)(v < 0 ? -v : v);
                signs = (signs << 1) | (uint32_t)(v < 0);
                infer_dc = 0;
            }
        }
        if (!num) continue;

        /* coeff_abs_level_greater1_flag / greater2_flag. */
        int ctx_set = (i == 0 || !is_luma) ? 0 : 2;
        if (c1_prev == 0) ctx_set++;
        int c1 = 1, first_c2 = -1;
        const int n1 = num < 8 ? num : 8;
        for (int k = 0; k < n1; k++) {
            const int g1 = absv[k] > 1;
            encode_bin(cb, one_base + ctx_set * 4 + c1, (uint32_t)g1);
            if (g1) {
                c1 = 0;
                if (first_c2 < 0) first_c2 = k;
            } else if (c1 > 0 && c1 < 3) {
                c1++;
            }
        }
        c1_prev = c1;
        if (first_c2 >= 0)
            encode_bin(cb, abs_base + ctx_set, absv[first_c2] > 2);

        encode_bypass_bins(cb, signs, num);

        /* coeff_abs_level_remaining. */
        uint32_t rice = 0;
        int first_coeff2 = 1;
        for (int k = 0; k < num; k++) {
            const int base = (k < 8 ? 2 : 1) | first_coeff2;
            if (absv[k] >= base) {
                write_coef_remain_exp_golomb(cb, (uint32_t)(absv[k] - base), rice);
                if (absv[k] > (3 << rice) && rice < 4) rice++;
            }
            if (absv[k] >= 2) first_coeff2 = 0;
        }
    }
}
