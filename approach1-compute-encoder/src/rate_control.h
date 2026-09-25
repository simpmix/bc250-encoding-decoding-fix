/* bc250-encoding-decoding-fix v0.4.3 - https://github.com/simpmix/bc250-encoding-decoding-fix */
/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * rate_control.h - CBR/VBR/Low-Latency rate control header
 */
#ifndef RATE_CONTROL_H
#define RATE_CONTROL_H

#include <stdint.h>

typedef enum {
    RC_CBR,
    RC_VBR,
    RC_LOW_LATENCY, /* Low latency mode for Sunshine / Moonlight streaming */
    RC_CQP          /* Constant QP mode */
} rc_mode_t;

typedef struct {
    rc_mode_t mode;
    uint32_t target_bitrate;
    uint32_t max_bitrate;
    int qp_min;
    int qp_max;

    /* Buffer model variables */
    double framerate;
    uint32_t target_bits_per_frame;
    int64_t buffer_fullness;
    int64_t buffer_size;
    int base_qp;
    int current_qp;
    uint64_t prev_frame_sad;
    int64_t error_integral;

    /* Wall-clock drain (see rc_update_stats): the leaky bucket used to drain
     * exactly target_bits_per_frame every frame, which silently enforces the
     * *negotiated* frame rate rather than the achieved one. On this hardware
     * a 1440p session negotiates 60 fps but the compute encoder sustains
     * ~40 fps, so a 60fps-sized per-frame budget delivered only ~2/3 of the
     * requested bitrate (measured: 20.91 Mbps against a 30.99 Mbps request,
     * with bits/frame matching target_bits_per_frame to 0.01% - the
     * controller was hitting its target perfectly, the target was just
     * sized for a frame rate that never arrives). Draining by real elapsed
     * time instead makes the bucket rate-correct at any achieved fps.
     * See docs/DEVLOG.md §16. */
    uint64_t last_frame_ns;   /* CLOCK_MONOTONIC of previous rc_update_stats; 0 = none yet */
    double   measured_fps;    /* EMA of achieved frame rate, diagnostics only */

    /* Max frame size constraint (bits) and Quality/Speed preset level (1..7) */
    uint32_t max_frame_bits;  /* 0 = unconstrained */
    uint32_t quality_level;   /* 1 = Highest quality, 4 = Balanced, 7 = Highest speed */

    /* Model-based QP (rc_model_frame_qp), for an encoder that turns it on
     * and reports what every picture cost. rc_init() leaves `model` and
     * the complexities alone - a new bitrate does not make the content
     * any easier - and starts the debt over. */
    int      model;
    double   cplx[2];         /* bits * 2^((qp - 12) / 5) of P [0] and I [1] pictures, 0 = not seen */
    double   debt;            /* bits produced minus bits allowed, since rc_init() */
    int      model_last_p_qp; /* 0 = no P picture yet */
    double   pixels;          /* per picture, for the first guess */
} rate_control_t;

void rc_init(rate_control_t *rc, rc_mode_t mode, uint32_t bitrate, double fps,
             uint32_t width, uint32_t height);
int rc_get_frame_qp(rate_control_t *rc, uint64_t est_sad);
void rc_update_stats(rate_control_t *rc, int bits_used);

/* The model: the QP for the next picture, and what a picture coded at
 * `qp` cost. rc_update_stats() still takes every picture's bits. */
int rc_model_frame_qp(rate_control_t *rc, int intra);
void rc_model_frame_coded(rate_control_t *rc, int intra, int qp, int bits);

void rc_set_max_frame_size(rate_control_t *rc, uint32_t max_frame_bits);
uint32_t rc_get_max_frame_size(const rate_control_t *rc);
void rc_set_quality_level(rate_control_t *rc, uint32_t quality_level);
uint32_t rc_get_quality_level(const rate_control_t *rc);

#endif
