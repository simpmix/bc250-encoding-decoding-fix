/* bc250-encoding-decoding-fix v0.4.3 - https://github.com/simpmix/bc250-encoding-decoding-fix */
/*
 * Copyright (c) 2026 BC-250 Project Contributors
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * hevc_cabac.h - HEVC CABAC (context-adaptive binary arithmetic coding)
 *                entropy engine.
 *
 * ITU-T H.265 always uses CABAC (there is no CAVLC option, unlike H.264) -
 * see docs/hevc_scope_note.md and DEVLOG.md Sec. 6/8 for why this is a real,
 * separate implementation rather than a reuse of cavlc.c.
 *
 * PROVENANCE: the core binary arithmetic coder (the encodeBin/encodeBinEP/
 * encodeBinTrm state machine, the writeOut byte-emission-with-carry logic,
 * the g_hevc_next_state / g_hevc_lps_range transition tables, the
 * context-initialization formula, and the residual_coding() coefficient
 * syntax algorithm including its scan tables, greater1/greater2 context-set
 * state machine, and Golomb-Rice remainder coding) is ADAPTED from the x265
 * HEVC encoder (https://github.com/videolan/x265), specifically
 * source/encoder/entropy.cpp, source/common/contexts.h and
 * source/common/constants.cpp.
 *
 * x265 is dual-licensed: GNU GPL version 2 or (at your option) any later
 * version, OR a commercial proprietary license (see its COPYING file and
 * per-file license headers, e.g. source/encoder/entropy.cpp: "either
 * version 2 of the License, or (at your option) any later version"). The
 * "or any later version" grant makes x265's GPL-licensed code usable under
 * GPL-3.0 terms, which is combination-compatible with this project's own
 * GPL-3.0-only license (this project relicensed from MIT to GPL-3.0-only
 * specifically to enable this adaptation - see the top-level LICENSE file
 * and the commit that changed it).
 *
 *   Copyright (C) 2013-2020 MulticoreWare, Inc
 *   Authors: Steve Borho <steve@borho.org>; Min Chen <chenm003@163.com>
 *   Licensed GPL-2.0-or-later (see above).
 *
 * The context initialization VALUE TABLES (INIT_SPLIT_FLAG, INIT_PART_SIZE,
 * INIT_QT_CBF, INIT_SIG_FLAG, INIT_LAST, INIT_ONE_FLAG, INIT_ABS_FLAG, etc.)
 * and the g_lpsTable/g_nextState/g_scan4x4/g_lastCoeffTable numeric constant
 * tables are themselves the ITU-T H.265 spec's own normative/informative
 * tables (Rec. ITU-T H.265 clauses 9.3.2.2 and 9.3.4) as reproduced in
 * x265's source - i.e. standard values every conformant HEVC implementation
 * uses, not x265-original IP, but the concrete table LAYOUT and the C
 * encodeBin()/residual-coding control flow below is a close adaptation of
 * x265's specific implementation of that spec.
 *
 * SCOPE: this is deliberately NOT a general HEVC entropy coder. It only
 * implements the syntax elements this project's intra-only, always-4x4-TU
 * encoder actually emits (see encoder_h265.c's top-of-file design comment):
 * split_cu_flag, part_mode (NxN-vs-2Nx2N single bin), intra_luma_pred_mode
 * (prev_flag + mpm_idx/rem_mode), intra_chroma_pred_mode, cbf_luma/cbf_cb/
 * cbf_cr, and residual_coding() for a single always-4x4, always-one-
 * coefficient-group transform block. It has no sig_coeff_group_flag path
 * (a lone 4x4 TU is exactly one coefficient group, so that flag is always
 * spec-inferred rather than coded - see x265's own codeCoeffNxN, which
 * takes the same shortcut for log2TrSize==2), no >4x4 transform support, no
 * sign-data-hiding (this project's PPS disables it), and no RDO/fast-path
 * (m_bitIf==NULL) branch, since this encoder always has a real output
 * buffer.
 */
#ifndef BC250_HEVC_CABAC_H
#define BC250_HEVC_CABAC_H

#include <stdint.h>
#include <stddef.h>
#include "bitstream.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Context-model offsets. Kept as a single flat array, laid out in the same
 * relative order as x265's contexts.h OFF_* macros (though several banks
 * x265 defines - skip/merge/inter/mvd/ref-idx/sao/etc - are omitted here
 * since an intra-only all-I-slice encoder never touches them). */
#define HEVC_CTX_SPLIT_FLAG   0   /* 3 contexts */
#define HEVC_CTX_PART_SIZE    3   /* 1 context used (index 0 only) */
#define HEVC_CTX_INTRA_PRED   4   /* 1 context  (ADI / prev_intra_luma_pred_flag) */
#define HEVC_CTX_CHROMA_PRED  5   /* 2 contexts */
#define HEVC_CTX_QT_CBF       7   /* 7 contexts: 0-1 luma (tuDepth!=0 / ==0), 2-6 chroma (trafoDepth 0-4) */
#define HEVC_CTX_SIG_FLAG    14   /* 42 contexts: 0-26 luma, 27-41 chroma */
#define HEVC_CTX_LAST_X      56   /* 18 contexts: 0-14 luma, 15-17 chroma */
#define HEVC_CTX_LAST_Y      74   /* 18 contexts */
#define HEVC_CTX_ONE_FLAG    92   /* 24 contexts: 0-15 luma, 16-23 chroma */
#define HEVC_CTX_ABS_FLAG   116   /* 6 contexts: 0-3 luma, 4-5 chroma */
#define HEVC_CTX_SKIP_FLAG  122   /* 3 contexts: cond_l + cond_a (0..2) */
#define HEVC_CTX_PRED_MODE  125   /* 1 context: inter (0) vs intra (1) */
#define HEVC_CTX_MERGE_FLAG 126   /* 1 context: merge_flag */
#define HEVC_CTX_MERGE_IDX  127   /* 1 context: merge_idx bin 0 */
#define HEVC_CTX_MVD        128   /* 2 contexts: abs_mvd_greater0_flag, abs_mvd_greater1_flag */
#define HEVC_CTX_MVP_IDX    130   /* 1 context: mvp_l0_flag */
#define HEVC_CTX_ROOT_CBF   131   /* 1 context: rqt_root_cbf */
#define HEVC_CTX_TRANS_SUBDIV 132 /* 3 contexts: split_transform_flag, 5 - log2TrafoSize */
#define HEVC_CTX_SIG_CG     135   /* 4 contexts: coded_sub_block_flag, 0-1 luma, 2-3 chroma */
#define HEVC_NUM_CTX        139

typedef struct {
    /* Output sink: a plain bit-level bitstream_t (bitstream.h/.c, the same
     * writer H.264's CAVLC path uses) rather than a private byte buffer -
     * this lets hevc_cabac_finish()'s final partial-byte tail (13+bitsLeft
     * bits, not generally a multiple of 8) go through the exact same
     * bs_write_u() accumulator as everything else, with the caller doing
     * the final bs_rbsp_trailing_bits() byte-alignment/stop-bit exactly
     * like every other RBSP this project emits (see encoder_h265.c). */
    bitstream_t *bs;

    /* Arithmetic coder register state - names/roles match x265's Entropy
     * (m_low/m_range/m_bitsLeft/m_bufferedByte/m_numBufferedBytes). */
    uint32_t low;
    uint32_t range;
    int      bits_left;
    uint32_t buffered_byte;
    int      num_buffered_bytes;

    uint8_t  ctx[HEVC_NUM_CTX];

    /* Estimation: when set, nothing is written - every bin adds what it
     * would cost to est_bits, in 1/32768 of a bit, and the context states
     * move exactly as they would. See hevc_cabac_estimator(). */
    int      est;
    uint32_t est_bits;
} hevc_cabac_t;

/* A coder that counts instead of writing, starting from `from`'s context
 * states. The encoder runs a candidate's syntax through one to learn what
 * it costs - the same functions, the same order, the same contexts. */
static inline void hevc_cabac_estimator(hevc_cabac_t *est, const hevc_cabac_t *from)
{
    *est = *from;
    est->bs = NULL;
    est->est = 1;
    est->est_bits = 0;
}

/* Bind the coder to an output bit-writer (does not reset arithmetic/context
 * state - call hevc_cabac_reset_contexts() once, then hevc_cabac_start()
 * once per slice; see encoder_h265.c). */
void hevc_cabac_init(hevc_cabac_t *cb, bitstream_t *bs);

/* (Re)initialize every context model this encoder uses from the real HEVC
 * init-value tables, per Rec. ITU-T H.265 9.3.2.2, for the given
 * slice QP (0-51, 8-bit, no QP offset) and slice type (1 = P-slice, 2 = I-slice). */
void hevc_cabac_reset_contexts(hevc_cabac_t *cb, int slice_qp, int slice_type);

/* Reset the arithmetic coder's low/range/carry state for a new slice
 * (does NOT touch context models - call reset_contexts() separately, once,
 * before the first slice of the sequence, or per-slice if you want fresh
 * contexts every slice; this project's design always uses one slice per
 * picture, so this is called once per frame right after reset_contexts()). */
void hevc_cabac_start(hevc_cabac_t *cb);

void hevc_cabac_encode_bin(hevc_cabac_t *cb, int ctx_idx, uint32_t bin);
void hevc_cabac_encode_bypass(hevc_cabac_t *cb, uint32_t bin);
void hevc_cabac_encode_bypass_bins(hevc_cabac_t *cb, uint32_t value, int num_bins);
void hevc_cabac_encode_terminate(hevc_cabac_t *cb, uint32_t bin);

/* Flush the arithmetic coder (writes the final closing bytes) and
 * byte-align via rbsp_slice_segment_trailing_bits (the CABAC end-of-slice
 * "1" bit is emitted by hevc_cabac_encode_terminate(cb, 1) before calling
 * this - see encoder_h265.c). */
void hevc_cabac_finish(hevc_cabac_t *cb);

/* cu_skip_flag: ctx_inc = condL + condA (0, 1, or 2), per ITU-T H.265 9.3.4.2.2.
 * In a P-slice, 1 indicates the entire 8x8 CU is skipped (reproduced from
 * reference frame with zero motion). */
void hevc_cabac_code_cu_skip_flag(hevc_cabac_t *cb, int skip, int ctx_inc);

/* pred_mode_flag for non-skip CU in P-slice (ITU-T H.265 7.3.8.5):
 * 1 = MODE_INTRA, 0 = MODE_INTER. */
void hevc_cabac_code_pred_mode_flag(hevc_cabac_t *cb, int pred_mode);

/* merge_idx for skip CU (ITU-T H.265 7.3.8.6): bin 0 coded with context 0. */
void hevc_cabac_code_merge_idx(hevc_cabac_t *cb, int merge_idx);

/* The inter prediction unit of a P-slice CU (7.3.8.6 and 7.3.8.9):
 * merge_flag, the motion vector difference (in quarter samples) and
 * mvp_l0_flag. There is one reference picture, so no ref_idx_l0, and no
 * inter_pred_idc in a P-slice. */
void hevc_cabac_code_merge_flag(hevc_cabac_t *cb, int merge);
void hevc_cabac_code_mvd(hevc_cabac_t *cb, int mvd_x, int mvd_y);
void hevc_cabac_code_mvp_idx(hevc_cabac_t *cb, int idx);

/* split_transform_flag of a transform tree node of size 1 << log2_size. */
void hevc_cabac_code_split_transform_flag(hevc_cabac_t *cb, int split, int log2_size);

/* residual_coding() for one 8x8 transform block in diagonal scan, with at
 * least one nonzero coefficient: coeff[y * 8 + x]. */
void hevc_cabac_code_residual_8x8(hevc_cabac_t *cb, const int16_t coeff[64], int is_luma);

/* rqt_root_cbf of an inter CU that is not a 2Nx2N merge (7.3.8.5): whether
 * any residual follows at all. */
void hevc_cabac_code_rqt_root_cbf(hevc_cabac_t *cb, int cbf);

/* split_cu_flag: ctx_inc = (left neighbor CU coded at a depth greater than
 * `depth`) + (above neighbor CU coded at a depth greater than `depth`),
 * per ITU-T H.265 9.3.4.2.2. This encoder's CTU quadtree always splits
 * exactly one level (16x16 CTU -> four 8x8 CUs, see encoder_h265.c), so
 * `depth` is always 0 and left/above availability is a simple frame-edge
 * check the caller (encoder_h265.c) already has to do for other reasons. */
void hevc_cabac_code_split_cu_flag(hevc_cabac_t *cb, int bin, int ctx_inc);

/* part_mode for an intra CU at minimum CU size (the only case this encoder
 * ever hits): a single context-coded bin, 1 = PART_2Nx2N, 0 = PART_NxN. */
void hevc_cabac_code_part_mode_intra(hevc_cabac_t *cb, int is_2nx2n);

/* intra_luma_pred_mode is coded as TWO separate passes over all 4 PUs of a
 * CU (ITU-T H.265 7.3.8.5 coding_unit()): every PU's prev_intra_luma_pred_
 * flag first, THEN every PU's mpm_idx/rem_intra_luma_pred_mode - NOT
 * interleaved per PU (x265's Entropy::codeIntraDirLumaAng() does the same
 * two-loop split for exactly this reason). Split into two calls so the
 * caller (encoder_h265.c) can enforce that order:
 *
 *   for each PU: pred_idx[pu] = hevc_cabac_code_intra_luma_flag(...)
 *   for each PU: hevc_cabac_code_intra_luma_data(..., pred_idx[pu], ...)
 *
 * `mode` is the real HEVC intra mode (0=Planar, 1=DC, 10=Horizontal,
 * 26=Vertical - the only four this encoder ever chooses, see hevc_intra.c)
 * and mpm[3] are this PU's three most-probable-mode candidates from its
 * left/above neighbors (8.4.2). hevc_cabac_code_intra_luma_flag() returns
 * the MPM list index (0-2) if `mode` is one of the 3 candidates, else -1 -
 * pass that value back in as `pred_idx` to the second call. */
int hevc_cabac_code_intra_luma_flag(hevc_cabac_t *cb, int mode, const int mpm[3]);
void hevc_cabac_code_intra_luma_data(hevc_cabac_t *cb, int mode, int pred_idx, const int mpm[3]);

/* intra_chroma_pred_mode for one CU, given that CU's representative luma
 * mode (IntraPredModeY of PU 0, per 8.4.3) and the always-DC(1) chroma mode
 * this encoder always signals (see hevc_intra.c's chroma path). */
void hevc_cabac_code_intra_chroma_pred_mode(hevc_cabac_t *cb, int luma_mode_pu0);

/* cbf_luma / cbf_cb / cbf_cr. `trafo_depth` matches the ctx formulas used
 * throughout this file's .c (luma: ctx = (trafo_depth==0) ? 1 : 0; chroma:
 * ctx = 2 + trafo_depth). */
void hevc_cabac_code_cbf_luma(hevc_cabac_t *cb, int cbf, int trafo_depth);
void hevc_cabac_code_cbf_chroma(hevc_cabac_t *cb, int cbf, int trafo_depth);

/* residual_coding() for exactly one 4x4 transform block (16 signed
 * quantized coefficient levels, raster order coeff[y*4+x]) that is known to
 * have at least one nonzero coefficient (caller checks cbf==0 and skips the
 * call entirely in that case, matching HEVC's own transform_tree() control
 * flow). `is_luma` selects the luma/chroma context banks; `scan_idx` is
 * 0=diagonal, 1=horizontal, 2=vertical, per Rec. ITU-T H.265 Table 8-10's
 * intra-mode-to-scan derivation (hevc_intra.c computes this from the
 * block's real intra mode and calls in here - see its
 * hevc_scan_idx_for_mode()). */
void hevc_cabac_code_residual_4x4(hevc_cabac_t *cb, const int16_t coeff[16],
                                   int is_luma, int scan_idx);

#ifdef __cplusplus
}
#endif
#endif /* BC250_HEVC_CABAC_H */
