/*
 * Copyright (c) 2026 BC-250 Project Contributors
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * hevc_inter.h - what the HEVC encoder borrows from the decoder for inter
 * prediction.
 *
 * The encoder has to predict a block exactly as every decoder will, or its
 * reconstruction and theirs part the moment a vector is not a whole sample.
 * So it does not have an interpolation of its own: it calls the decoder's,
 * through this one function per bit depth (see hevc_mc_template.c).
 */
#ifndef BC250_HEVC_INTER_H
#define BC250_HEVC_INTER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* One w x h block of one plane, predicted from `ref` (a w_pic x h_pic plane,
 * `stride` samples a row) at (x, y) displaced by (mvx, mvy): quarter samples
 * for luma, eighth samples for chroma. Samples outside the picture repeat
 * its edge, as 8.5.3.3.3 has it. w, h <= 64. */
void hevc_mc_uni_8(const uint8_t *ref, int stride, int w_pic, int h_pic,
                   int x, int y, int w, int h, int mvx, int mvy, int chroma,
                   uint8_t *dst, int dst_stride);
void hevc_mc_uni_10(const uint16_t *ref, int stride, int w_pic, int h_pic,
                    int x, int y, int w, int h, int mvx, int mvy, int chroma,
                    uint16_t *dst, int dst_stride);

#ifdef __cplusplus
}
#endif
#endif /* BC250_HEVC_INTER_H */
