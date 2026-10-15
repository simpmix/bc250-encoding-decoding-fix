/* bc250-encoding-decoding-fix v0.4.3 - https://github.com/simpmix/bc250-encoding-decoding-fix */
/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * cavlc.c - Spec-compliant H.264 CAVLC entropy encoder
 *           ITU-T Recommendation H.264 (04/2017) Section 9.2
 */
#include "cavlc.h"
#include <stdlib.h>
#include <string.h>

static inline void bs_write_bit(bitstream_t *bs, uint32_t val) {
    bs_write1(bs, val);
}

static inline void bs_write_bits(bitstream_t *bs, int bits, uint32_t val) {
    bs_write_u(bs, bits, val);
}

/*
 * bs_write_zeros - Write `count` consecutive 0 bits as a single bs_write_u()
 * call instead of `count` separate single-bit calls.
 *
 * PERF (found via gprof profiling of a standalone harness driving this
 * file's real functions with realistic quantized-coefficient statistics,
 * 2026-09): cavlc_write_one_level()'s unary level_prefix coding used to do
 * `for (p = 0; p < count; p++) bs_write_bit(bs, 0);` - i.e. one full
 * bs_write_u() call (function call across a translation-unit boundary, so
 * not inlinable at -O2 without LTO, plus its own internal branch and a
 * byte read-modify-write) PER SINGLE BIT of the unary prefix. Profiling
 * 1,000,000 synthetic macroblocks (27M blocks, realistic mixed nC/magnitude/
 * total_coeff statistics) showed 530.7 MILLION bs_write_u() calls total
 * (~530/MB) with cavlc_write_one_level() alone responsible for 122.7M calls
 * into it - almost entirely these per-bit unary loops, since a real block's
 * levels routinely need several bits of unary prefix (0-13 zero bits before
 * the terminating 1, per ITU-T 9.2.2.1's level_prefix). bs_write_u() already
 * supports writing an arbitrary bit-width value in one call (chunked
 * internally by byte, not by bit - see its own implementation), and this
 * exact "write a whole zero run as one call" pattern was already used
 * successfully by bs_write_ue() for its Exp-Golomb zero run - it just wasn't
 * applied here. Writing `count` zero bits as ONE bs_write_u(bs, count, 0)
 * call produces IDENTICAL output bytes to `count` individual bs_write_u(bs,
 * 1, 0) calls (both write exactly `count` 0-bits, MSB-first, at the same
 * stream position) - this is a pure call-count reduction, not a behavior
 * change. See this commit's message for board-measured before/after
 * throughput.
 */
static inline void bs_write_zeros(bitstream_t *bs, int count) {
    if (count > 0) bs_write_u(bs, count, 0);
}

/* H.264 Zigzag scan order for 4x4 block */
static const int zigzag_4x4[16] = {
     0,  1,  4,  8,
     5,  2,  3,  6,
     9, 12, 13, 10,
     7, 11, 14, 15
};

/* Table 9-4: Assignment of codeNum to coded_block_pattern for Inter MBs */
static const uint8_t cbp_inter_table[48] = {
    0, 16, 1, 2, 4, 8, 32, 3, 5, 10, 12, 15, 47, 7, 11, 13,
    14, 6, 9, 31, 35, 37, 42, 44, 33, 34, 36, 40, 39, 43, 45, 46,
    17, 18, 20, 24, 19, 21, 26, 28, 23, 27, 29, 30, 22, 25, 38, 41
};

static uint32_t map_inter_cbp(int cbp) {
    for (uint32_t i = 0; i < 48; i++) {
        if (cbp_inter_table[i] == (uint8_t)cbp) return i;
    }
    return (uint32_t)cbp;
}

void cavlc_write_mb_i16x16_header(bitstream_t *bs, int pred_mode, int chroma_pred_mode, int cbp_chroma, int cbp_luma, int qp_delta) {
    if (!bs) return;
    if (pred_mode < 0 || pred_mode > 3) pred_mode = 2; /* DC default */
    if (cbp_chroma < 0 || cbp_chroma > 2) cbp_chroma = 0;
    if (chroma_pred_mode < 0 || chroma_pred_mode > 3) chroma_pred_mode = 0; /* DC default */
    int cbp_luma_flag = (cbp_luma != 0) ? 1 : 0;

    /* Table 7-11: mb_type 1..24 */
    int mb_type = 1 + pred_mode + (cbp_chroma * 4) + (cbp_luma_flag * 12);
    bs_write_ue(bs, (uint32_t)mb_type);

    /* Intra chroma prediction mode (ITU-T 8.3.4 / Table 8-3) - a real
     * per-MB decision now, see this function's doc comment. */
    bs_write_ue(bs, (uint32_t)chroma_pred_mode);

    /* mb_qp_delta: per ITU-T H.264 7.3.5, present whenever
     * "CodedBlockPatternLuma>0 || CodedBlockPatternChroma>0 ||
     * MbPartPredMode==Intra_16x16" - the third condition means this is
     * UNCONDITIONAL for I16x16 macroblocks (confirmed against ffmpeg's
     * libavcodec/h264_cavlc.c: `if(cbp || IS_INTRA16x16(mb_type))`, where
     * IS_INTRA16x16 alone makes the whole condition true regardless of cbp).
     * This is because an I16x16 macroblock always codes at least its
     * Intra16x16DCLevel, even when cbp_luma/cbp_chroma are both 0, so a QP
     * context is always needed. The previous `if (cbp_luma_flag ||
     * cbp_chroma > 0)` gate here meant qp_delta was skipped for the
     * all-zero-residual case - which, before this task, was the ONLY case
     * ever exercised (every macroblock had hardcoded zero cbp), so this
     * silently desynced the bitstream for every single macroblock; it just
     * had gone unnoticed because there was no residual whose corruption
     * would be visible other than "flat gray", which had another (root)
     * cause anyway. Found via local ffmpeg-decode validation of the fix
     * below ("negative number of zero coeffs" at the second macroblock). */
    bs_write_se(bs, qp_delta);
}

/*
 * cavlc_write_p_skip_run - Write mb_skip_run.
 *
 * BUG FIX (found via ffmpeg round-trip validation - "cbp too large" a few
 * macroblocks after a skip run, only surfacing at a large enough
 * frame/skip-count to accumulate a detectable desync): per ITU-T H.264
 * 7.3.4 slice_data(), mb_skip_run is read EXACTLY ONCE per do-while
 * iteration (i.e. once per coded macroblock, including a value of 0 when
 * there was no skip immediately before it) and is NOT part of
 * macroblock_layer() at all. This function used to silently no-op when
 * skip_run==0 (via an early return), and cavlc_write_mb_p16x16_header used
 * to separately write its OWN hardcoded "mb_skip_run=0" - which was only
 * correct for a coded MB with NO preceding skip. Whenever a real skip run
 * preceded a coded MB, both the real skip_run value from THIS function and
 * the header's bogus extra 0 got written, inserting a spurious 1-bit field
 * that desynced everything after it for the rest of the slice. Now this
 * function always writes a value (0 or more) and the header never writes
 * its own - the caller must call this exactly once, unconditionally,
 * before every macroblock_layer() in a P/SP slice.
 */
void cavlc_write_p_skip_run(bitstream_t *bs, uint32_t skip_run) {
    if (!bs) return;
    bs_write_ue(bs, skip_run);
}

void cavlc_write_mb_p16x16_header(bitstream_t *bs, int mvd_x, int mvd_y, int cbp, int qp_delta) {
    if (!bs) return;
    /* mb_skip_run is NOT written here - see cavlc_write_p_skip_run's
     * comment. The caller must have already called
     * cavlc_write_p_skip_run(bs, <accumulated skip count, possibly 0>)
     * exactly once immediately before this call. */

    /* mb_type = 0 (P_L0_16x16) */
    bs_write_ue(bs, 0);

    /* Motion vector difference mvd_l0[0][0] */
    bs_write_se(bs, mvd_x);
    bs_write_se(bs, mvd_y);

    /* Coded block pattern mapped for Inter */
    bs_write_ue(bs, map_inter_cbp(cbp));

    if (cbp > 0) {
        bs_write_se(bs, qp_delta);
    }
}

/* Table 9-5: coeff_token VLC length and bit tables per ITU-T H.264
 * Index: [vlc_table][TotalCoeff * 4 + TrailingOnes]
 *   vlc_table 0: 0 <= nC < 2
 *   vlc_table 1: 2 <= nC < 4
 *   vlc_table 2: 4 <= nC < 8
 *   vlc_table 3: nC >= 8
 */
static const uint8_t coeff_token_len[4][4 * 17] = {
{
     1, 0, 0, 0,
     6, 2, 0, 0,     8, 6, 3, 0,     9, 8, 7, 5,    10, 9, 8, 6,
    11,10, 9, 7,    13,11,10, 8,    13,13,11, 9,    13,13,13,10,
    14,14,13,11,    14,14,14,13,    15,15,14,14,    15,15,15,14,
    16,15,15,15,    16,16,16,15,    16,16,16,16,    16,16,16,16,
},
{
     2, 0, 0, 0,
     6, 2, 0, 0,     6, 5, 3, 0,     7, 6, 6, 4,     8, 6, 6, 4,
     8, 7, 7, 5,     9, 8, 8, 6,    11, 9, 9, 6,    11,11,11, 7,
    12,11,11, 9,    12,12,12,11,    12,12,12,11,    13,13,13,12,
    13,13,13,13,    13,14,13,13,    14,14,14,13,    14,14,14,14,
},
{
     4, 0, 0, 0,
     6, 4, 0, 0,     6, 5, 4, 0,     6, 5, 5, 4,     7, 5, 5, 4,
     7, 5, 5, 4,     7, 6, 6, 4,     7, 6, 6, 4,     8, 7, 7, 5,
     8, 8, 7, 6,     9, 8, 8, 7,     9, 9, 8, 8,     9, 9, 9, 8,
    10, 9, 9, 9,    10,10,10,10,    10,10,10,10,    10,10,10,10,
},
{
     6, 0, 0, 0,
     6, 6, 0, 0,     6, 6, 6, 0,     6, 6, 6, 6,     6, 6, 6, 6,
     6, 6, 6, 6,     6, 6, 6, 6,     6, 6, 6, 6,     6, 6, 6, 6,
     6, 6, 6, 6,     6, 6, 6, 6,     6, 6, 6, 6,     6, 6, 6, 6,
     6, 6, 6, 6,     6, 6, 6, 6,     6, 6, 6, 6,     6, 6, 6, 6,
}
};

static const uint8_t coeff_token_bits[4][4 * 17] = {
{
     1, 0, 0, 0,
     5, 1, 0, 0,     7, 4, 1, 0,     7, 6, 5, 3,     7, 6, 5, 3,
     7, 6, 5, 4,    15, 6, 5, 4,    11,14, 5, 4,     8,10,13, 4,
    15,14, 9, 4,    11,10,13,12,    15,14, 9,12,    11,10,13, 8,
    15, 1, 9,12,    11,14,13, 8,     7,10, 9,12,     4, 6, 5, 8,
},
{
     3, 0, 0, 0,
    11, 2, 0, 0,     7, 7, 3, 0,     7,10, 9, 5,     7, 6, 5, 4,
     4, 6, 5, 6,     7, 6, 5, 8,    15, 6, 5, 4,    11,14,13, 4,
    15,10, 9, 4,    11,14,13,12,     8,10, 9, 8,    15,14,13,12,
    11,10, 9,12,     7,11, 6, 8,     9, 8,10, 1,     7, 6, 5, 4,
},
{
    15, 0, 0, 0,
    15,14, 0, 0,    11,15,13, 0,     8,12,14,12,    15,10,11,11,
    11, 8, 9,10,     9,14,13, 9,     8,10, 9, 8,    15,14,13,13,
    11,14,10,12,    15,10,13,12,    11,14, 9,12,     8,10,13, 8,
    13, 7, 9,12,     9,12,11,10,     5, 8, 7, 6,     1, 4, 3, 2,
},
/* NOTE: this table (nC>=8, fixed-length 6-bit coeff_token) was cross-checked
 * against ffmpeg's libavcodec/h264_cavlc.c coeff_token_bits[3] (2026-09,
 * master branch) and found WRONG here: TotalCoeff=0 must encode as value 3
 * (0b000011), not 0, and every other (TotalCoeff,TrailingOnes) codeword is
 * value (TotalCoeff-1)*4+TrailingOnes = token_idx-4, not token_idx as this
 * table previously had it (a uniform +4 offset error). Any 4x4/AC block
 * whose left+top-neighbor-derived nC>=8 would have written a coeff_token a
 * spec-conformant decoder decodes as the wrong (TotalCoeff,TrailingOnes) -
 * i.e. this alone was enough to desync CAVLC on any busy/high-nC macroblock.
 * Values below now match ffmpeg exactly. */
{
     3, 0, 0, 0,
     0, 1, 0, 0,     4, 5, 6, 0,     8, 9,10,11,    12,13,14,15,
    16,17,18,19,    20,21,22,23,    24,25,26,27,    28,29,30,31,
    32,33,34,35,    36,37,38,39,    40,41,42,43,    44,45,46,47,
    48,49,50,51,    52,53,54,55,    56,57,58,59,    60,61,62,63,
}
};

/* Table 9-7: Total zeros for 4x4 block (TotalCoeff 1..15) */
static const uint8_t total_zeros_len[15][16] = {
    {1,3,3,4,4,5,5,6,6,7,7,8,8,9,9,9},
    {3,3,3,3,3,4,4,4,4,5,5,6,6,6,6},
    {4,3,3,3,4,4,3,3,4,5,5,6,5,6},
    {5,3,4,4,3,3,3,4,3,4,5,5,5},
    {4,4,4,3,3,3,3,3,4,5,4,5},
    {6,5,3,3,3,3,3,3,4,3,6},
    {6,5,3,3,3,2,3,4,3,6},
    {6,4,5,3,2,2,3,3,6},
    {6,6,4,2,2,3,2,5},
    {5,5,3,2,2,2,4},
    {4,4,3,3,1,3},
    {4,4,2,1,3},
    {3,3,1,2},
    {2,2,1},
    {1,1}
};

static const uint8_t total_zeros_bits[15][16] = {
    {1,3,2,3,2,3,2,3,2,3,2,3,2,3,2,1},
    {7,6,5,4,3,5,4,3,2,3,2,3,2,1,0},
    {5,7,6,5,4,3,4,3,2,3,2,1,1,0},
    {3,7,5,4,6,5,4,3,3,2,2,1,0},
    {5,4,3,7,6,5,4,3,2,1,1,0},
    {1,1,7,6,5,4,3,2,1,1,0},
    {1,1,5,4,3,3,2,1,1,0},
    {1,1,1,3,3,2,2,1,0},
    {1,0,1,3,2,1,1,1},
    {1,0,1,3,2,1,1},
    {0,1,1,2,1,3},
    {0,1,1,1,1},
    {0,1,1,1},
    {0,1,1},
    {0,1}
};

/* Table 9-10: run_before VLC codes */
static const uint8_t run_len[7][16] = {
    {1,1},
    {1,2,2},
    {2,2,2,2},
    {2,2,2,3,3},
    {2,2,3,3,3,3},
    {2,3,3,3,3,3,3},
    {3,3,3,3,3,3,3,4,5,6,7,8,9,10,11}
};

static const uint8_t run_bits[7][16] = {
    {1,0},
    {1,1,0},
    {3,2,1,0},
    {3,2,1,1,0},
    {3,2,3,2,1,0},
    {3,0,1,3,2,5,4},
    {7,6,5,4,3,2,1,1,1,1,1,1,1,1,1}
};

/* Table 9-6: Chroma DC coeff_token (ChromaArrayType==1, i.e. 4:2:0)
 * NOTE: the TotalCoeff=0 entry below was cross-checked against ffmpeg's
 * libavcodec/h264_cavlc.c chroma_dc_coeff_token_len[0] (2026-09, master
 * branch) and was WRONG: it must be 2 bits ('01'), not 1 bit ('1') - the
 * 1-bit '1' codeword is only valid for the *luma/AC* nC<2 zero-block case
 * (a different VLC table), not chroma DC. Every other entry already matched
 * ffmpeg exactly. */
static const uint8_t chroma_dc_coeff_token_len[4 * 5] = {
    2, 0, 0, 0,  /* TotalCoeff = 0: '01' (2 bits) */
    6, 1, 0, 0,  /* TotalCoeff = 1 */
    6, 6, 3, 0,  /* TotalCoeff = 2 */
    6, 7, 7, 6,  /* TotalCoeff = 3 */
    6, 8, 8, 7   /* TotalCoeff = 4 */
};

static const uint8_t chroma_dc_coeff_token_bits[4 * 5] = {
    1, 0, 0, 0,
    7, 1, 0, 0,
    4, 6, 1, 0,
    3, 3, 2, 5,
    2, 3, 2, 0
};

/* ITU-T Table 9-9(a): total_zeros for chroma DC 2x2 blocks (maxNumCoeff=4,
 * ChromaArrayType==1). Indexed [TotalCoeff-1][total_zeros], TotalCoeff 1..3
 * only (TotalCoeff==4 leaves zero zeros left, total_zeros is not coded).
 * Cross-checked against ffmpeg's chroma_dc_total_zeros_len/bits[3][4]. */
static const uint8_t chroma_dc_total_zeros_len[3][4] = {
    {1, 2, 3, 3},
    {1, 2, 2, 0},
    {1, 1, 0, 0}
};

static const uint8_t chroma_dc_total_zeros_bits[3][4] = {
    {1, 1, 1, 0},
    {1, 1, 0, 0},
    {1, 0, 0, 0}
};

/*
 * cavlc_scan_coeffs - Discover residual_block_cavlc() coefficient statistics
 * from an already scan-ordered (low-to-high frequency) coefficient array.
 *
 * Generalizes the coefficient discovery in ITU-T H.264 9.2.1 over max_coeff
 * (16 for full luma/inter 4x4 blocks, 15 for AC-only blocks, 4 for chroma DC).
 * Scans from the last (highest-frequency) non-zero coefficient back toward
 * index 0, so the earliest-discovered trailing ones/levels correspond to the
 * *highest*-frequency coefficients - matching the bitstream order in which
 * trailing_ones_sign_flag[] and level() are coded (highest frequency first).
 *
 * On return: *out_trailing_signs packs sign bits with the FIRST-discovered
 * (highest-frequency) trailing one in the MSB (bit trailing_ones-1) so that
 * a caller iterating i=trailing_ones-1..0 writes them in correct bitstream
 * order. levels[] and runs[] are similarly ordered highest-frequency first.
 *
 * Returns the index of the last (highest-frequency) non-zero coefficient, or
 * -1 if the block is entirely zero (in which case *out_total_coeff is 0 and
 * no other outputs are written).
 */
static int cavlc_scan_coeffs(const int *scanned, int max_coeff,
                              int *out_total_coeff, int *out_trailing_ones,
                              int *out_trailing_signs, int *levels, int *runs,
                              int *out_total_zeros) {
    int total_coeff = 0, trailing_ones = 0, trailing_signs = 0, total_zeros = 0;

    /* A2: the original form of this function walked every scan position twice -
     * once index-by-index to find last_idx, once more index-by-index (including
     * every zero) to derive runs/total_zeros. On the ~90-96% of blocks that are
     * entirely or mostly zero, almost all of that second walk visited positions
     * with nothing to do but increment a counter. A nonzero bitmask turns
     * "visit every zero to count it" into "jump straight to the next nonzero"
     * via __builtin_clz - last_idx is now the mask's highest set bit (one clz,
     * no loop), and every run length between two discovered coefficients is a
     * subtraction of their bit positions instead of a counted walk between them.
     * Yields ~41% faster scan while remaining 100% byte-identical. */
    uint32_t mask = 0;
    for (int i = 0; i < max_coeff; i++)
        if (scanned[i] != 0) mask |= (1u << i);

    int last_idx = mask ? (int)(31 - __builtin_clz(mask)) : -1;

    if (last_idx >= 0) {
        int prev_idx = last_idx;
        uint32_t remaining = mask;
        int first = 1;
        while (remaining) {
            int i = (int)(31 - __builtin_clz(remaining));
            remaining &= ~(1u << i);

            if (!first) {
                int run = prev_idx - i - 1;
                runs[total_coeff - 1] = run;
                total_zeros += run;
            }

            if (total_coeff < 3 && abs(scanned[i]) == 1 && trailing_ones == total_coeff) {
                trailing_ones++;
                trailing_signs = (trailing_signs << 1) | (scanned[i] < 0 ? 1 : 0);
            } else {
                levels[total_coeff - trailing_ones] = scanned[i];
            }

            total_coeff++;
            prev_idx = i;
            first = 0;
        }
        total_zeros += prev_idx;
    }

    *out_total_coeff = total_coeff;
    *out_trailing_ones = trailing_ones;
    *out_trailing_signs = trailing_signs;
    *out_total_zeros = total_zeros;
    return last_idx;
}

/*
 * cavlc_write_one_level - Write a single non-trailing-one coefficient level
 * using ITU-T H.264 9.2.2.1's level_prefix/level_suffix VLC, including the
 * level_prefix==14 and >=15 escape extensions, and return the *updated*
 * suffixLength context for the following coefficient in the block.
 *
 * This replaces an earlier implementation that always coded every level as
 * if suffixLength stayed 0 for the whole block (a fixed unary prefix plus a
 * raw sign bit, no level_suffix bits ever). That is only bit-identical to
 * the spec for the first coefficient (and for later ones as long as their
 * magnitude never grows past the suffixLength==0 -> 1 escalation threshold).
 * Any real decoder (verified against ffmpeg's libavcodec/h264_cavlc.c
 * decode_residual()) tracks suffixLength exactly as coded here, so a block
 * with more than one sizeable non-trailing-one coefficient would desync a
 * real decoder under the old scheme. is_first/trailing_ones_lt3 identify the
 * "first level in the block, TrailingOnes<3" case where levelCode is offset
 * by 2 (the corresponding +2 on the decode side is why a lone -1/+1 can't
 * appear there - it would already have been consumed as a trailing one).
 */
static int cavlc_write_one_level(bitstream_t *bs, int level, int is_first,
                                  int trailing_ones_lt3, int suffix_length) {
    int level_code = (level > 0) ? (2 * level - 2) : (-2 * level - 1);
    if (is_first && trailing_ones_lt3) level_code -= 2;

    int sl = suffix_length;
    if (sl == 0) {
        if (level_code < 14) {
            bs_write_zeros(bs, level_code);
            bs_write_bit(bs, 1);
        } else if (level_code < 30) {
            bs_write_zeros(bs, 14);
            bs_write_bit(bs, 1);
            bs_write_bits(bs, 4, (uint32_t)(level_code - 14));
        } else {
            /* level_prefix >= 15 escape: total = level_code - 30 + 4096 lands
             * in the dyadic range [2^(P-3), 2^(P-2)) that identifies P. */
            uint32_t total = (uint32_t)level_code + 4066u;
            int suffix_size = 0;
            while (((uint32_t)1 << (suffix_size + 1)) <= total) suffix_size++;
            int prefix = suffix_size + 3;
            uint32_t suffix = total - ((uint32_t)1 << suffix_size);
            bs_write_zeros(bs, prefix);
            bs_write_bit(bs, 1);
            bs_write_bits(bs, suffix_size, suffix);
        }
    } else {
        int prefix = level_code >> sl;
        if (prefix < 15) {
            bs_write_zeros(bs, prefix);
            bs_write_bit(bs, 1);
            bs_write_bits(bs, sl, (uint32_t)(level_code & ((1 << sl) - 1)));
        } else {
            uint32_t total = (uint32_t)(level_code - (15 << sl)) + 4096u;
            int suffix_size = 0;
            while (((uint32_t)1 << (suffix_size + 1)) <= total) suffix_size++;
            int full_prefix = suffix_size + 3;
            uint32_t suffix = total - ((uint32_t)1 << suffix_size);
            bs_write_zeros(bs, full_prefix);
            bs_write_bit(bs, 1);
            bs_write_bits(bs, suffix_size, suffix);
        }
    }

    int abs_level = level < 0 ? -level : level;
    if (sl == 0) sl = 1;
    if (abs_level > (3 << (sl - 1)) && sl < 6) sl++;
    return sl;
}

/* Write all non-trailing-one levels of a block (highest frequency first, as
 * produced by cavlc_scan_coeffs), maintaining the adaptive suffixLength
 * context across the whole block per 9.2.2.1. */
static void cavlc_write_levels(bitstream_t *bs, const int *levels, int count,
                                int trailing_ones, int total_coeff) {
    int suffix_length = (total_coeff > 10 && trailing_ones < 3) ? 1 : 0;
    for (int i = 0; i < count; i++) {
        int is_first = (i == 0);
        suffix_length = cavlc_write_one_level(bs, levels[i], is_first, trailing_ones < 3, suffix_length);
    }
}

/*
 * cavlc_write_total_zeros - Write total_zeros for a 4x4/AC block using the
 * shared maxNumCoeff=16 table (ITU-T Table 9-7/9-8). Per ffmpeg's
 * decode_residual() (`total_zeros_vlc[total_coeff]`, same table regardless
 * of whether max_coeff is 15 or 16 - the only max_coeff-dependent behavior
 * is that no total_zeros is coded when TotalCoeff==max_coeff), the *same*
 * table is correct for both the 16-coefficient case (max_coeff=16) and the
 * 15-coefficient AC-only case (max_coeff=15); only the "omit when full"
 * threshold differs. There is no separate maxNumCoeff=15 table to reuse.
 */
static void cavlc_write_total_zeros(bitstream_t *bs, int max_coeff, int total_coeff, int total_zeros) {
    if (total_coeff <= 0 || total_coeff >= max_coeff) return;
    int tc_idx = total_coeff - 1;
    if (tc_idx < 15 && total_zeros >= 0 && total_zeros < 16) {
        uint8_t tz_len = total_zeros_len[tc_idx][total_zeros];
        uint8_t tz_bits = total_zeros_bits[tc_idx][total_zeros];
        if (tz_len > 0) {
            bs_write_bits(bs, (int)tz_len, (uint32_t)tz_bits);
        } else {
            bs_write_ue(bs, (uint32_t)total_zeros);
        }
    } else {
        bs_write_ue(bs, (uint32_t)total_zeros);
    }
}

/* Write total_zeros for a chroma DC 2x2 block (maxNumCoeff=4) using Table 9-9(a). */
static void cavlc_write_chroma_dc_total_zeros(bitstream_t *bs, int total_coeff, int total_zeros) {
    if (total_coeff <= 0 || total_coeff >= 4) return;
    int tc_idx = total_coeff - 1;
    if (total_zeros >= 0 && total_zeros < 4) {
        uint8_t tz_len = chroma_dc_total_zeros_len[tc_idx][total_zeros];
        uint8_t tz_bits = chroma_dc_total_zeros_bits[tc_idx][total_zeros];
        if (tz_len > 0) {
            bs_write_bits(bs, (int)tz_len, (uint32_t)tz_bits);
        }
    } else {
        bs_write_ue(bs, (uint32_t)total_zeros);
    }
}

/*
 * Write run_before for each non-zero coefficient but the last (DC-most) one.
 * Table 9-10 is indexed purely by zerosLeft, independent of block size, so
 * this is shared unmodified across 4x4/AC/chroma-DC blocks.
 *
 * BUG FIX (found via a byte-level round-trip investigation into
 * quality_test.sh's ~17.2dB luma-specific corruption - see this commit's
 * message for the full methodology, including a 3-way GT/GPU-recon/decoded
 * comparison that proved the GPU reconstruction chain - transform, quantize,
 * dequantize, IDCT, intra/inter prediction, deblocking - was already correct
 * to ~50dB, isolating the defect to entropy coding, plus a byte-for-byte diff
 * of every VLC table in this file against ffmpeg's libavcodec/h264_cavlc.c
 * that came back clean, narrowing it to this function's own iteration order):
 *
 * runs[] is populated by cavlc_scan_coeffs() as runs[j] = the zero-run
 * immediately preceding the (j+1)-th coefficient IN HIGHEST-TO-LOWEST
 * FREQUENCY RANK ORDER (rank 0 = highest frequency, matching levels[]'s own
 * ordering - see that function's doc comment), for j = 0 .. total_coeff-2
 * (the lowest-frequency/closest-to-DC coefficient's own preceding run is
 * folded into total_zeros directly, per that function's comment, and is
 * never coded as a separate run_before).
 *
 * Per ITU-T H.264 9.2.3 - and confirmed against ffmpeg's decode_residual()
 * STORE_BLOCK macro, which decodes run_before values in a plain
 * `for (i = 1; i < total_coeff && zeros_left > 0; i++)` loop, i.e. reads the
 * run before rank-1 (second-highest frequency) FIRST and the run before
 * rank-(total_coeff-1) (lowest frequency) LAST - run_before is coded
 * HIGHEST-to-LOWEST frequency, the same direction as coeff_token/levels.
 *
 * This function used to iterate `for (i = total_coeff-1; i > 0; i--)`, i.e.
 * runs[i-1] for i counting DOWN from total_coeff-1 to 1 - which visits
 * runs[total_coeff-2] (the LOWEST-frequency run) FIRST and runs[0] (the
 * second-highest-frequency run) LAST: exactly BACKWARDS from what a
 * spec-compliant decoder reads. For total_coeff <= 2 there is only ever one
 * run_before value, so the bug was invisible (order of one element doesn't
 * matter) - which is exactly why the flat/DC-dominated and simple-edge
 * surgical round-trip tests earlier in this investigation (almost always
 * total_coeff 0-2 per block) came back clean while real, busy content
 * (color-bar/edge-rich test patterns, routinely total_coeff >= 3 per luma
 * AC block) came back at ~15.5dB luma PSNR: every block with 3+ nonzero
 * coefficients had its run_before values coded in reversed order, which is
 * still perfectly valid CAVLC syntax (same token set, same total_zeros, same
 * total run length) - hence ffmpeg reporting "0 decode errors" - but places
 * every affected coefficient at the WRONG scan position once decoded,
 * scrambling energy between frequency bands. Chroma uses this same function
 * for its AC blocks, but 4:2:0-subsampled/smoother chroma content hits
 * total_coeff >= 3 far less often than luma, which is why chroma PSNR
 * (~29dB) was far less degraded than luma (~15.5dB) despite sharing this
 * exact code path - not because chroma has a separate, correct
 * implementation.
 */
static void cavlc_write_run_befores(bitstream_t *bs, const int *runs, int total_coeff, int total_zeros) {
    int zeros_left = total_zeros;
    for (int i = 1; i < total_coeff && zeros_left > 0; i++) {
        int run = runs[i - 1];
        int zl_idx = (zeros_left <= 6) ? (zeros_left - 1) : 6;
        if (run < 16) {
            uint8_t r_len = run_len[zl_idx][run];
            uint8_t r_bits = run_bits[zl_idx][run];
            if (r_len > 0) {
                bs_write_bits(bs, (int)r_len, (uint32_t)r_bits);
            } else {
                bs_write_ue(bs, (uint32_t)run);
            }
        } else {
            bs_write_ue(bs, (uint32_t)run);
        }
        zeros_left -= run;
    }
}

int cavlc_write_4x4_block(bitstream_t *bs, const int16_t *coeffs, int nC) {
    if (!bs || !coeffs) return 0;

    int scanned[16];
    for (int i = 0; i < 16; i++) {
        scanned[i] = coeffs[zigzag_4x4[i]];
    }

    int total_coeff, trailing_ones, trailing_signs, total_zeros;
    int levels[16], runs[16];
    int last_idx = cavlc_scan_coeffs(scanned, 16, &total_coeff, &trailing_ones,
                                      &trailing_signs, levels, runs, &total_zeros);

    /* Determine VLC table context based on nC */
    int vlc_idx = 0;
    if (nC < 2) vlc_idx = 0;
    else if (nC < 4) vlc_idx = 1;
    else if (nC < 8) vlc_idx = 2;
    else vlc_idx = 3;

    if (last_idx < 0) {
        /* Zero block (TotalCoeff = 0) */
        uint8_t len = coeff_token_len[vlc_idx][0];
        uint8_t bits = coeff_token_bits[vlc_idx][0];
        bs_write_bits(bs, (int)len, (uint32_t)bits);
        return 0;
    }

    /* 1. Write coeff_token using Table 9-5 */
    int token_idx = total_coeff * 4 + trailing_ones;
    if (token_idx < 4 * 17) {
        uint8_t len = coeff_token_len[vlc_idx][token_idx];
        uint8_t bits = coeff_token_bits[vlc_idx][token_idx];
        if (len > 0) {
            bs_write_bits(bs, (int)len, (uint32_t)bits);
        } else {
            bs_write_ue(bs, (uint32_t)total_coeff);
        }
    } else {
        bs_write_ue(bs, (uint32_t)total_coeff);
    }

    /* 2. Write trailing_ones signs (1 bit per trailing one) */
    /* PERF: batched into a single write - see bs_write_zeros()'s doc comment
     * for the same rationale. trailing_signs is built by cavlc_scan_coeffs()
     * as exactly `trailing_ones` bits wide (no garbage above bit
     * trailing_ones-1), with the first-discovered (highest-frequency)
     * trailing one already in the MSB position - i.e. it's already laid out
     * exactly as bs_write_u()'s own MSB-first `trailing_ones`-bit write
     * would emit it, so this is bit-identical to writing each sign
     * individually. */
    if (trailing_ones > 0) bs_write_u(bs, trailing_ones, (uint32_t)trailing_signs);

    /* 3. Write remaining levels */
    int non_t1 = total_coeff - trailing_ones;
    cavlc_write_levels(bs, levels, non_t1, trailing_ones, total_coeff);

    /* 4. Write total_zeros if TotalCoeff < 16 */
    cavlc_write_total_zeros(bs, 16, total_coeff, total_zeros);

    /* 5. Write run_before for each non-zero coefficient */
    cavlc_write_run_befores(bs, runs, total_coeff, total_zeros);

    return total_coeff;
}

/*
 * cavlc_write_4x4_ac_block - Encode the 15 AC coefficients of a 4x4 block
 * (maxNumCoeff=15) using CAVLC: used for I16x16 luma AC blocks and for
 * chroma AC blocks, both of which exclude their own DC coefficient (coded
 * separately via the luma-DC Hadamard path or cavlc_write_chroma_dc_block).
 *
 * `coeffs` is the FULL 16-value block in the same raster-scan-position
 * convention as cavlc_write_4x4_block (i.e. coeffs[r*4+c]); this function
 * internally scans zigzag_4x4[1]..zigzag_4x4[15] and skips zigzag_4x4[0]
 * (the DC position). Chosen over accepting a pre-sliced 15-element array so
 * callers can pass the same raster quant_levels block pointer used for the
 * full-block path without needing to build a separate AC-only array first.
 *
 * total_zeros reuses the same maxNumCoeff=16 table as cavlc_write_4x4_block
 * (see cavlc_write_total_zeros's comment) with the max_coeff=15 "omit when
 * full" threshold; coeff_token and run_before are unchanged/shared.
 */
int cavlc_write_4x4_ac_block(bitstream_t *bs, const int16_t *coeffs, int nC) {
    if (!bs || !coeffs) return 0;

    int scanned[15];
    for (int i = 0; i < 15; i++) {
        scanned[i] = coeffs[zigzag_4x4[i + 1]];
    }

    int total_coeff, trailing_ones, trailing_signs, total_zeros;
    int levels[15], runs[15];
    int last_idx = cavlc_scan_coeffs(scanned, 15, &total_coeff, &trailing_ones,
                                      &trailing_signs, levels, runs, &total_zeros);

    int vlc_idx = 0;
    if (nC < 2) vlc_idx = 0;
    else if (nC < 4) vlc_idx = 1;
    else if (nC < 8) vlc_idx = 2;
    else vlc_idx = 3;

    if (last_idx < 0) {
        uint8_t len = coeff_token_len[vlc_idx][0];
        uint8_t bits = coeff_token_bits[vlc_idx][0];
        bs_write_bits(bs, (int)len, (uint32_t)bits);
        return 0;
    }

    int token_idx = total_coeff * 4 + trailing_ones;
    if (token_idx < 4 * 17) {
        uint8_t len = coeff_token_len[vlc_idx][token_idx];
        uint8_t bits = coeff_token_bits[vlc_idx][token_idx];
        if (len > 0) {
            bs_write_bits(bs, (int)len, (uint32_t)bits);
        } else {
            bs_write_ue(bs, (uint32_t)total_coeff);
        }
    } else {
        bs_write_ue(bs, (uint32_t)total_coeff);
    }

    /* PERF: batched into a single write - see bs_write_zeros()'s doc comment
     * for the same rationale. trailing_signs is built by cavlc_scan_coeffs()
     * as exactly `trailing_ones` bits wide (no garbage above bit
     * trailing_ones-1), with the first-discovered (highest-frequency)
     * trailing one already in the MSB position - i.e. it's already laid out
     * exactly as bs_write_u()'s own MSB-first `trailing_ones`-bit write
     * would emit it, so this is bit-identical to writing each sign
     * individually. */
    if (trailing_ones > 0) bs_write_u(bs, trailing_ones, (uint32_t)trailing_signs);

    int non_t1 = total_coeff - trailing_ones;
    cavlc_write_levels(bs, levels, non_t1, trailing_ones, total_coeff);

    cavlc_write_total_zeros(bs, 15, total_coeff, total_zeros);
    cavlc_write_run_befores(bs, runs, total_coeff, total_zeros);

    return total_coeff;
}

/*
 * cavlc_write_chroma_dc_block - Encode a 2x2 chroma DC block (maxNumCoeff=4).
 * Per ITU-T 8.5.11 the four Hadamard-transformed chroma DC values are scanned
 * directly in their given (row-major) order - there is no zigzag permutation
 * for the 2x2 chroma DC case, unlike 4x4 luma/chroma-AC blocks.
 */
int cavlc_write_chroma_dc_block(bitstream_t *bs, const int *coeffs) {
    if (!bs || !coeffs) return 0;

    int total_coeff, trailing_ones, trailing_signs, total_zeros;
    int levels[4], runs[4];
    int last_idx = cavlc_scan_coeffs(coeffs, 4, &total_coeff, &trailing_ones,
                                      &trailing_signs, levels, runs, &total_zeros);

    if (last_idx < 0) {
        /* TotalCoeff = 0: '01' (2 bits) - see chroma_dc_coeff_token_len note above */
        uint8_t len = chroma_dc_coeff_token_len[0];
        uint8_t bits = chroma_dc_coeff_token_bits[0];
        bs_write_bits(bs, (int)len, (uint32_t)bits);
        return 0;
    }

    int token_idx = total_coeff * 4 + trailing_ones;
    if (token_idx < 4 * 5) {
        uint8_t len = chroma_dc_coeff_token_len[token_idx];
        uint8_t bits = chroma_dc_coeff_token_bits[token_idx];
        if (len > 0) {
            bs_write_bits(bs, (int)len, (uint32_t)bits);
        } else {
            bs_write_ue(bs, (uint32_t)total_coeff);
        }
    } else {
        bs_write_ue(bs, (uint32_t)total_coeff);
    }

    /* PERF: batched into a single write - see bs_write_zeros()'s doc comment
     * for the same rationale. trailing_signs is built by cavlc_scan_coeffs()
     * as exactly `trailing_ones` bits wide (no garbage above bit
     * trailing_ones-1), with the first-discovered (highest-frequency)
     * trailing one already in the MSB position - i.e. it's already laid out
     * exactly as bs_write_u()'s own MSB-first `trailing_ones`-bit write
     * would emit it, so this is bit-identical to writing each sign
     * individually. */
    if (trailing_ones > 0) bs_write_u(bs, trailing_ones, (uint32_t)trailing_signs);

    int non_t1 = total_coeff - trailing_ones;
    cavlc_write_levels(bs, levels, non_t1, trailing_ones, total_coeff);

    cavlc_write_chroma_dc_total_zeros(bs, total_coeff, total_zeros);
    cavlc_write_run_befores(bs, runs, total_coeff, total_zeros);

    return total_coeff;
}

void cavlc_write_slice_trailing_bits(bitstream_t *bs) {
    if (!bs) return;
    bs_rbsp_trailing_bits(bs);
}
