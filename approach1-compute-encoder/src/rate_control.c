/* bc250-encoding-decoding-fix v0.4.3 - https://github.com/simpmix/bc250-encoding-decoding-fix */
/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * rate_control.c - Proportional-Integral CBR/VBR/Low-Latency rate controller
 */
#include "rate_control.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <time.h>

#if defined(__linux__)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <errno.h>
extern char *program_invocation_short_name;
#endif

/*
 * rc_estimate_base_qp - derive a starting QP from the requested bitrate and
 * resolution instead of a hardcoded constant.
 *
 * See docs/rate_control_audit.md sections 1 and 4 point 1: the old code set
 * base_qp = 26 unconditionally, so every QP the feedback loop could ever
 * reach (base_qp +- a handful of steps) was completely independent of what
 * bitrate was requested. This uses the standard, well-understood
 * bits-per-pixel <-> QP relationship real encoders (x264/x265's first-pass
 * QP guess included) rely on: H.264's quantization step size doubles every
 * 6 QP steps, so bits-per-pixel is roughly log-linear in QP - doubling the
 * bitrate at fixed resolution/framerate costs about -6 QP, and vice versa.
 *
 * The curve needs one calibration point to anchor it. Rather than invent
 * one, this reuses a real board measurement already on record
 * (docs/rate_control_audit.md SS3 Part C): a genuine CBR encode of 1280x720
 * @30fps synthetic "testsrc" content targeting 1 Mbps, under the OLD
 * hardcoded base_qp=26, converged to within -8.1% of that target - i.e.
 * QP ~26 empirically was already about right for ~0.036 bits/pixel of this
 * kind of content (see audit section 3, Part C). Anchoring here means the
 * one combination that was already known-good is left unchanged, and every
 * other resolution/bitrate combination scales off of a real measurement
 * instead of a guess.
 */
#define RC_QP_REF          26.0
#define RC_BPP_REF         0.036169   /* 1,000,000 / (1280*720*30) bits/pixel */
#define RC_QP_PER_DOUBLING 6.0        /* QP cost of one bitrate doubling */

static int get_configured_qp_min(void) {
    const char *env_qp_min = getenv("BC250_QP_MIN");
    if (env_qp_min && *env_qp_min) {
        int q = atoi(env_qp_min);
        if (q >= 1 && q <= 51) return q;
    }
    return 12;
}

static int rc_estimate_base_qp(uint32_t bitrate, double fps, uint32_t width, uint32_t height) {
    double pixels_per_sec = (double)width * (double)height * (fps > 0 ? fps : 60.0);
    if (pixels_per_sec <= 0.0) return (int)RC_QP_REF;

    double bpp = (double)bitrate / pixels_per_sec;
    if (bpp <= 0.0) return 51;

    double qp = RC_QP_REF - RC_QP_PER_DOUBLING * (log(bpp / RC_BPP_REF) / log(2.0));
    int qp_i = (int)lround(qp);
    int qp_min = get_configured_qp_min();
    if (qp_i < qp_min) qp_i = qp_min;
    if (qp_i > 51) qp_i = 51;
    return qp_i;
}

void rc_init(rate_control_t *rc, rc_mode_t mode, uint32_t bitrate, double fps,
             uint32_t width, uint32_t height) {
    if (!rc) return;
    rc->mode = mode;
    rc->target_bitrate = bitrate > 0 ? bitrate : 5000000;
    rc->max_bitrate = rc->target_bitrate * 3 / 2;
    /* Default stays at 12 to preserve high encode throughput on standard desktop sessions.
     * Can be overridden down to 1 via BC250_QP_MIN for high-bitrate VR streaming (>150 Mbps). */
    rc->qp_min = get_configured_qp_min();
    rc->qp_max = 51;
    rc->framerate = fps > 0 ? fps : 60.0;

    rc->target_bits_per_frame = (uint32_t)(rc->target_bitrate / rc->framerate);
    if (rc->target_bits_per_frame < 100) rc->target_bits_per_frame = 100;

    if (mode == RC_LOW_LATENCY) {
        /* 2-frame buffer for instant game streaming feedback */
        rc->buffer_size = rc->target_bits_per_frame * 2;
    } else {
        /* Standard 1-second leaky bucket buffer */
        rc->buffer_size = rc->target_bitrate;
    }
    if (rc->buffer_size < 1000) rc->buffer_size = 1000;

    rc->buffer_fullness = rc->buffer_size / 2;

    int base_qp = rc_estimate_base_qp(rc->target_bitrate, rc->framerate, width, height);
    if (mode == RC_CQP && rc->current_qp >= 12 && rc->current_qp <= 51) {
        base_qp = rc->current_qp;
    }
    rc->base_qp = base_qp;
    rc->current_qp = base_qp;
    rc->prev_frame_sad = 0;
    rc->error_integral = 0;
    rc->max_frame_bits = 0;
    rc->quality_level = 4; /* Default: balanced */
    /* Wall-clock drain state: a re-init is a fresh bucket, so forget the
     * previous frame's timestamp rather than charging this frame for the
     * gap across the re-init (see rc_update_stats). */
    rc->last_frame_ns = 0;
    rc->measured_fps = 0.0;
    rc->debt = 0.0;
    rc->pixels = (double)width * (double)height;

    /* Diagnostic (BC250_DEBUG_RC=1): every rc_init with the target it was
     * actually handed and the base QP that fell out of it. Added while
     * root-causing "requested bitrate has no effect on output" - see
     * docs/DEVLOG.md §15. */
    if (getenv("BC250_DEBUG_RC")) {
        fprintf(stderr, "[bc250-rc] rc_init: mode=%d target=%u bps fps=%.1f %ux%u "
                        "-> base_qp=%d target_bits_per_frame=%u\n",
                (int)mode, rc->target_bitrate, rc->framerate, width, height,
                base_qp, rc->target_bits_per_frame);
    }
}

/*
 * Integral-term time constant and gain. This file's header comment has
 * always described a "Proportional-Integral" controller, and the struct
 * has always carried an error_integral field - but no code ever wrote or
 * read it, so in practice the loop was proportional-only, and (per
 * docs/rate_control_audit.md sections 1 and 4 point 2) p_term's normalized
 * error/target_level ratio is mathematically bounded to +-1, capping
 * qp_adjust to a fixed +-6 around base_qp no matter how large or
 * persistent the buffer error is.
 *
 * Wiring the integral term up lets *sustained* error - buffer error that
 * doesn't clear on its own within a handful of frames - keep pushing
 * current_qp further, toward the real qp_min/qp_max bounds (12/51), the
 * way a real CBR/VBR controller's long-term correction works, instead of
 * saturating at a fixed +-6 window forever. RC_INTEGRAL_WINDOW_FRAMES is
 * the number of frames of continuously-saturated error needed for the
 * integral term to reach its full RC_INTEGRAL_QP_RANGE contribution -
 * chosen as a multi-second settling window (independent of fps, since the
 * error unit here is already a per-frame buffer delta), similar in spirit
 * to the multi-second VBV windows real encoders use.
 */
#define RC_INTEGRAL_WINDOW_FRAMES 30.0
#define RC_INTEGRAL_QP_RANGE      40.0

int rc_get_frame_qp(rate_control_t *rc, uint64_t est_sad) {
    if (!rc) return 26;

    /* Constant QP mode: no buffer accounting or rate adjustment */
    if (rc->mode == RC_CQP) {
        return rc->current_qp;
    }

    /* Compute buffer fullness deviation from 50% target */
    int64_t target_level = rc->buffer_size / 2;
    if (target_level <= 0) target_level = 1;
    int64_t error = rc->buffer_fullness - target_level;

    /* Proportional feedback: map buffer error to QP adjustments */
    double p_term = (double)error / (double)target_level * 6.0;

    /* Integral feedback: accumulate buffer error over time so a target that
     * is persistently unreachable within the proportional term's +-6 band
     * keeps walking current_qp further, instead of the loop giving up at a
     * fixed offset from base_qp forever. Anti-windup: stop accumulating in
     * a direction that's already saturated current_qp at qp_min/qp_max, so
     * the integral doesn't overshoot once the error eventually reverses. */
    int64_t integral_cap = (int64_t)(target_level * RC_INTEGRAL_WINDOW_FRAMES);
    if (integral_cap < 1) integral_cap = 1;
    bool saturated_high = (rc->current_qp >= rc->qp_max && error > 0);
    bool saturated_low  = (rc->current_qp <= rc->qp_min && error < 0);
    if (!saturated_high && !saturated_low) {
        rc->error_integral += error;
        if (rc->error_integral > integral_cap) rc->error_integral = integral_cap;
        if (rc->error_integral < -integral_cap) rc->error_integral = -integral_cap;
    }
    double i_term = ((double)rc->error_integral / (double)integral_cap) * RC_INTEGRAL_QP_RANGE;

    int qp_adjust = (int)round(p_term + i_term);

    /* VBR: Adjust for temporal complexity */
    if (rc->mode == RC_VBR && rc->prev_frame_sad > 0) {
        double complexity_ratio = (double)est_sad / (double)rc->prev_frame_sad;
        if (complexity_ratio > 1.3) qp_adjust += 2;
        else if (complexity_ratio < 0.7) qp_adjust -= 2;
    }

    /* Max frame size constraint: if per-frame budget is close to or exceeds max_frame_bits, bias QP higher */
    if (rc->max_frame_bits > 0 && rc->target_bits_per_frame > 0) {
        if (rc->target_bits_per_frame > rc->max_frame_bits) {
            qp_adjust += 2;
        } else if (rc->max_frame_bits < (rc->target_bits_per_frame * 3 / 2)) {
            qp_adjust += 1;
        }
    }

    /* Clamp maximum single-frame QP delta to prevent visual pulsation.
     * In low latency mode or high-speed preset (quality_level >= 5), allow step of 3
     * so rate control adapts promptly to high-motion scene bursts. */
    int max_step = (rc->mode == RC_LOW_LATENCY || rc->quality_level >= 5) ? 3 : 2;
    int delta = (rc->base_qp + qp_adjust) - rc->current_qp;
    if (abs(delta) >= 8) max_step += 2;
    else if (abs(delta) >= 4) max_step += 1;
    if (delta > max_step) delta = max_step;
    if (delta < -max_step) delta = -max_step;

    rc->current_qp += delta;

    if (rc->current_qp < rc->qp_min) rc->current_qp = rc->qp_min;
    if (rc->current_qp > rc->qp_max) rc->current_qp = rc->qp_max;

    rc->prev_frame_sad = est_sad;
    return rc->current_qp;
}

void rc_update_stats(rate_control_t *rc, int bits_used) {
    if (!rc) return;
    if (rc->mode == RC_CQP) return;

    rc->buffer_fullness += bits_used;

    /* Drain by REAL elapsed time, not a fixed per-frame quota.
     *
     * The bucket used to drain exactly target_bits_per_frame each call,
     * which makes the controller enforce target_bitrate ONLY if frames
     * actually arrive at the framerate rc_init() was given. They don't:
     * a 1440p Sunshine session negotiates 60 fps, this compute encoder
     * sustains ~40 fps, and the result was a measured 20.91 Mbps against
     * a 30.99 Mbps request - with per-frame output matching
     * target_bits_per_frame to 0.01%, i.e. rate control was tracking its
     * target faithfully and the target itself was a third too small.
     *
     * Draining target_bitrate * elapsed_seconds instead is correct at any
     * achieved frame rate: slower frames each get a proportionally larger
     * share, so the long-run output rate converges on target_bitrate
     * rather than on target_bitrate * (achieved_fps / negotiated_fps).
     *
     * The elapsed clamp keeps the first frame (no previous timestamp) and
     * any pathological gap (a stall, a paused stream, a suspended session)
     * from injecting a huge one-shot drain that would slam QP to qp_min;
     * outside those cases it is a no-op. Falls back to the old fixed quota
     * when no timestamp is available yet. See docs/DEVLOG.md §16. */
    /* Wall-clock drain was introduced specifically for live network streaming
     * (Sunshine / WiVRn) where video packets are delivered across the network
     * in real-time, and if the compute encoder achieves ~40 fps instead of 60 fps,
     * network transmission rate matches target_bitrate.
     *
     * However, for ANY file recording or offline encoding (OBS Studio recording to disk,
     * FFmpeg transcoding, GStreamer recording, SimpleScreenRecorder, etc.):
     * The output media container (MP4, MKV) plays back at the stream's nominal framerate.
     * If the encoder runs at e.g. 37.5 fps on a 60 fps stream, wall-clock elapsed time
     * is 26.6ms instead of 16.6ms (1.60x longer). Draining by wall-clock time causes the
     * leaky bucket to drain 60% faster, making the encoder output 60% larger frames.
     * When played back at 60 fps, the video file's bitrate balloons by 60% (e.g. 9.0 Mbps
     * request balloons to 14.4 Mbps in the recorded file!).
     *
     * Therefore:
     * - By default, ALL file recording and video encoding (FFmpeg, OBS, etc.) uses NOMINAL
     *   per-frame drain (target_bits_per_frame = target_bitrate / framerate), guaranteeing
     *   exact bitrate conformance and preventing bitrate ballooning.
     * - Wall-clock drain is ONLY used for live streaming servers (Sunshine, WiVRn) or when
     *   explicitly requested via BC250_RC_WALLCLOCK_DRAIN=1.
     * - BC250_RC_NOMINAL_DRAIN=1 can be set to force nominal drain anywhere.
     */
    static int wallclock_drain_mode = -1;
    if (wallclock_drain_mode < 0) {
        const char *e_nom = getenv("BC250_RC_NOMINAL_DRAIN");
        if (e_nom && strcmp(e_nom, "1") == 0) {
            wallclock_drain_mode = 0;
        } else {
            const char *e_wc = getenv("BC250_RC_WALLCLOCK_DRAIN");
            if (e_wc && strcmp(e_wc, "1") == 0) {
                wallclock_drain_mode = 1;
            } else {
#if defined(__linux__)
                if (program_invocation_short_name &&
                    (strcmp(program_invocation_short_name, "sunshine") == 0 ||
                     strcmp(program_invocation_short_name, "wivrn-server") == 0 ||
                     strcmp(program_invocation_short_name, "wivrn") == 0)) {
                    wallclock_drain_mode = 1;
                } else {
                    wallclock_drain_mode = 0;
                }
#else
                wallclock_drain_mode = 0;
#endif
            }
        }
    }

    int64_t drain = rc->target_bits_per_frame;   /* exact per-frame quota for video recording */
    if (wallclock_drain_mode == 1) {
        struct timespec now;
        uint64_t now_ns = 0;
        if (clock_gettime(CLOCK_MONOTONIC, &now) == 0) {
            now_ns = (uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec;
        }

        if (now_ns != 0 && rc->last_frame_ns != 0 && now_ns > rc->last_frame_ns) {
            double elapsed = (double)(now_ns - rc->last_frame_ns) / 1e9;
            /* Clamp to a sane inter-frame window: 1ms (1000fps) .. 250ms (4fps). */
            if (elapsed < 0.001) elapsed = 0.001;
            if (elapsed > 0.250) elapsed = 0.250;
            drain = (int64_t)((double)rc->target_bitrate * elapsed);

            /* Diagnostics only: EMA of achieved frame rate. */
            double inst_fps = 1.0 / elapsed;
            rc->measured_fps = (rc->measured_fps > 0.0)
                                 ? (rc->measured_fps * 0.95 + inst_fps * 0.05)
                                 : inst_fps;
        }
        if (now_ns != 0) rc->last_frame_ns = now_ns;
    }

    rc->buffer_fullness -= drain;

    /* The model's account is not clamped at the top: what was overspent is
     * owed. At the bottom, CBR and low latency cannot bank unspent bits (a
     * CBR stream pads them out), VBR may bank half a second's worth. */
    if (rc->model) {
        rc->debt += (double)bits_used - (double)drain;
        const double floor = rc->mode == RC_VBR ? -(double)rc->target_bitrate / 2.0 : 0.0;
        if (rc->debt < floor) rc->debt = floor;
    }

    if (rc->buffer_fullness < 0) {
        rc->buffer_fullness = 0;
    } else if (rc->buffer_fullness > rc->buffer_size) {
        rc->buffer_fullness = rc->buffer_size;
    }
}

void rc_set_max_frame_size(rate_control_t *rc, uint32_t max_frame_bits) {
    if (rc) {
        rc->max_frame_bits = max_frame_bits;
    }
}

uint32_t rc_get_max_frame_size(const rate_control_t *rc) {
    return rc ? rc->max_frame_bits : 0;
}

void rc_set_quality_level(rate_control_t *rc, uint32_t quality_level) {
    if (rc) {
        if (quality_level < 1) quality_level = 1;
        if (quality_level > 7) quality_level = 7;
        rc->quality_level = quality_level;
    }
}

uint32_t rc_get_quality_level(const rate_control_t *rc) {
    return rc ? rc->quality_level : 4;
}

/* ------------------------------------------------ model-based QP */

/* The PI loop above steers QP from how full a clamped buffer is, which
 * says nothing about what a picture will cost at a given QP. On a hard
 * clip it opens far too low (the bits-per-pixel guess is anchored on
 * testsrc), fills the buffer in a few pictures, winds up its integral,
 * and then holds QP at 50-51 for a second while the debt drains: crowd_run
 * at 10 Mbit/s came out at 23 dB, where QP 38 gives the same bitrate at
 * about 29.
 *
 * The model instead keeps, for P and for I pictures, a complexity: the
 * bits a picture cost, scaled to QP 12 by the rule that every
 * RC_MODEL_QP_HALF steps of QP halve the bits. The next picture gets the
 * QP at which that complexity costs the bits it is allowed - one
 * picture's share of the bitrate, less a part of whatever the stream owes
 * so far. Nothing saturates and nothing winds up; a wrong guess costs one
 * picture. */
#define RC_MODEL_QP_HALF 5.0   /* the HEVC encoder with its dead zone, crowd_run QP 22-37 */
#define RC_MODEL_MAX_STEP 2    /* QP change from one P picture to the next */
#define RC_MODEL_I_BOOST 2     /* I pictures this much finer: everything after predicts from them */
#define RC_MODEL_I_OVER_P 4.0  /* I against P bits at one QP, until a P picture has been seen */

static double rc_model_qp_exact(double cplx, double bits)
{
    return 12.0 + RC_MODEL_QP_HALF * log2(cplx / bits);
}

static int rc_model_qp_for(double cplx, double bits)
{
    return (int)lround(rc_model_qp_exact(cplx, bits));
}

static double rc_model_bits(double cplx, int qp)
{
    return cplx * pow(2.0, -(qp - 12) / RC_MODEL_QP_HALF);
}

/* Over how many pictures a debt is paid back: a second for VBR, half of
 * one for CBR, a quarter for low latency.
 *
 * ⚠️ Not the low-latency mode's two-picture buffer. An I picture costs
 * several pictures' budget, and paying that back within two pictures held
 * QP at 49-51 for the ten after it - a fifth of a second of mush after every
 * IDR, which Moonlight asks for after each lost packet. Over a quarter of a
 * second the pictures after it lose one or two QP instead. */
static double rc_model_horizon(const rate_control_t *rc)
{
    double h = rc->mode == RC_LOW_LATENCY ? rc->framerate / 4.0
             : rc->mode == RC_CBR ? rc->framerate / 2.0 : rc->framerate;
    return h < 2.0 ? 2.0 : h;
}

int rc_model_frame_qp(rate_control_t *rc, int intra)
{
    if (!rc) return 26;
    if (rc->mode == RC_CQP) return rc->current_qp;

    const double bpf = (double)rc->target_bitrate / rc->framerate;
    double want = bpf - rc->debt / rc_model_horizon(rc);
    if (want < bpf / 8.0) want = bpf / 8.0;
    if (want > bpf * 4.0) want = bpf * 4.0;

    int qp;
    double cplx;   /* this picture's, for the size limit */
    if (rc->cplx[0] > 0.0) {
        const double exact = rc_model_qp_exact(rc->cplx[0], want);
        qp = (int)lround(exact);
        /* Hold the QP of the last P picture until the model moves a whole
         * step away from it: a value near a rounding edge otherwise flips
         * between two QPs picture after picture, and the picture flickers. */
        if (rc->model_last_p_qp > 0 && fabs(exact - rc->model_last_p_qp) < 1.0) qp = rc->model_last_p_qp;
        if (rc->model_last_p_qp > 0) {
            if (qp > rc->model_last_p_qp + RC_MODEL_MAX_STEP) qp = rc->model_last_p_qp + RC_MODEL_MAX_STEP;
            if (qp < rc->model_last_p_qp - RC_MODEL_MAX_STEP) qp = rc->model_last_p_qp - RC_MODEL_MAX_STEP;
        }
        if (intra) qp -= RC_MODEL_I_BOOST;
        cplx = intra ? (rc->cplx[1] > 0.0 ? rc->cplx[1] : rc->cplx[0] * RC_MODEL_I_OVER_P) : rc->cplx[0];
    } else if (rc->cplx[1] > 0.0) {
        /* An I picture seen, no P yet: guess P from it. */
        qp = rc_model_qp_for(rc->cplx[1] / RC_MODEL_I_OVER_P, want);
        if (intra) qp -= RC_MODEL_I_BOOST;
        cplx = intra ? rc->cplx[1] : rc->cplx[1] / RC_MODEL_I_OVER_P;
    } else {
        /* Nothing seen: bits per pixel against what the HEVC encoder
         * spends on natural 1080p video - QP 36 at 0.1 bit per pixel.
         * Rather high than low: the hardest derf clips need about that,
         * and a picture guessed too coarse costs only itself, where one
         * guessed too fine leaves a debt for the pictures after it. */
        const double bpp = rc->pixels > 0.0 ? bpf / rc->pixels : 0.1;
        qp = (int)lround(36.0 - RC_MODEL_QP_HALF * log2(bpp / 0.1)) - (intra ? RC_MODEL_I_BOOST : 0);
        cplx = 0.0;
    }

    /* Never above the largest picture the caller allows. */
    if (rc->max_frame_bits > 0 && cplx > 0.0)
        while (qp < rc->qp_max && rc_model_bits(cplx, qp) > (double)rc->max_frame_bits) qp++;

    if (qp < rc->qp_min) qp = rc->qp_min;
    if (qp > rc->qp_max) qp = rc->qp_max;
    rc->current_qp = qp;
    return qp;
}

void rc_model_frame_coded(rate_control_t *rc, int intra, int qp, int bits)
{
    if (!rc || bits <= 0) return;
    const double c = (double)bits * pow(2.0, (qp - 12) / RC_MODEL_QP_HALF);
    double *k = &rc->cplx[intra ? 1 : 0];
    /* Follow the content over several pictures, averaging log(complexity)
     * with a quarter's weight on the newest.
     *
     * ⚠️ Not the last picture alone. A P picture at a high QP right after
     * one at a lower QP is nearly all skips and costs a tenth of it, so the
     * last picture alone said "cheap", the next QP went down, that picture
     * was dear, and QP see-sawed by four every picture (ducks_take_off at
     * 5 Mbit/s: 40, 44, 40, 44 ...). Only a jump of 16 times, a cut, starts
     * the average over. */
    if (*k > 0.0 && c < *k * 16.0 && c > *k / 16.0) *k = exp(0.75 * log(*k) + 0.25 * log(c));
    else                                            *k = c;
    if (!intra) rc->model_last_p_qp = qp;
}

