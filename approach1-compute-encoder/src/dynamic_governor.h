/* bc250-encoding-decoding-fix v0.4.3 - https://github.com/simpmix/bc250-encoding-decoding-fix */
/*
 * Copyright (c) 2026 BC-250 Project Contributors
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * dynamic_governor.h - Real-time dynamic CPU/GPU load governor for BC-250
 */

#ifndef BC250_DYNAMIC_GOVERNOR_H
#define BC250_DYNAMIC_GOVERNOR_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GOV_TIER_0_GPU_FULL = 0,    /* Normal: GPU full ME (search radius 16, subpel) */
    GOV_TIER_1_GPU_FAST = 1,    /* Mild contention: GPU fast ME (reduced search window) */
    GOV_TIER_2_CPU_OFFLOAD = 2, /* Heavy contention: CPU SIMD ME offload, GPU skips ME stage */
    GOV_TIER_3_FAILOVER = 3     /* Emergency: P_Skip failover */
} governor_tier_t;

typedef struct {
    uint32_t total_frames;
    uint32_t total_tier0_frames;
    uint32_t total_tier1_frames;
    uint32_t total_offload_frames;
    uint32_t total_failover_frames;
    double ema_latency_ms;
    double last_latency_ms;
    governor_tier_t current_tier;
    double frame_deadline_ms;      /* Real-time frame deadline: 16.67ms (60fps), 8.33ms (120fps), etc. */
    uint32_t dropped_frames_prevented;
} governor_stats_t;

typedef struct {
    double ema_latency_ms;
    double last_latency_ms;
    governor_tier_t current_tier;
    uint32_t stable_frames_count;
    uint32_t total_frames;
    uint32_t total_tier0_frames;
    uint32_t total_tier1_frames;
    uint32_t total_offload_frames;
    uint32_t total_failover_frames;
    int stats_log_interval;        /* 0: disabled, >0: log every N frames */
    double frame_deadline_ms;      /* Real-time frame deadline: 16.67ms (60fps), 8.33ms (120fps), etc. */
    double tier1_threshold_ms;     /* Default: 8.0 ms */
    double tier2_threshold_ms;     /* Default: 12.0 ms */
    double tier3_threshold_ms;     /* Default: 15.5 ms */
    bool tier1_overridden;
    bool tier2_overridden;
    bool tier3_overridden;
    bool predictive_pacing_enabled; /* Predictive velocity-aware contention pacing */
    uint32_t dropped_frames_prevented;
    uint32_t step_down_hysteresis; /* Default: 4 frames */
    /* Minimum frames a tier must have been held before the governor may move
     * *up* into the CPU offload tier. Default 0 (no dwell) generally, 8 for a
     * live streaming caller.
     *
     * WHY, given that there is already a step_down_hysteresis: that one
     * governs how long the governor must see a healthy EMA before it *leaves*
     * the offload, and it says nothing about entering it. Entering happened on
     * the first frame whose EMA crossed the tier-2 threshold, and on the frame
     * after any emergency failover. On a live stream that produced a
     * repeating one-frame pattern that is very expensive and, as reported,
     * visible: a game spiking past the emergency threshold drops one frame,
     * the next frame is handed to the CPU ME offload - a full-frame search
     * over memory the CPU cannot cache (see me_src_stage in encoder_h264.c) -
     * and the frame after that goes back to the GPU. The work the encoder
     * does per frame changed by a large factor several times a second, and
     * the frame time changed with it. See docs/DEVLOG.md. */
    uint32_t min_dwell_frames;
    uint32_t frames_in_tier;     /* Frames spent in current_tier; the dwell clock. */
    bool enabled;
    int forced_tier;               /* -1: auto, 0..3: forced via BC250_FORCE_TIER */
    bool cpu_offload_enabled;      /* Default: false; enabled via BC250_ENABLE_CPU_ME=1 */
    bool allow_failover;           /* False for offline transcode to avoid dropping frames */
} dynamic_governor_t;

/**
 * Initializes the dynamic governor with default thresholds and hysteresis.
 */
void dynamic_governor_init(dynamic_governor_t *gov);

/**
 * Sets stream framerate and dynamically scales Governor v2 thresholds
 * to match the real-time frame budget (e.g. 16.6ms @ 60fps, 8.3ms @ 120fps).
 */
void dynamic_governor_set_framerate(dynamic_governor_t *gov, double fps);

/**
 * Sets explicit frame deadline in milliseconds.
 */
void dynamic_governor_set_deadline_ms(dynamic_governor_t *gov, double deadline_ms);

/**
 * Gets the active frame deadline in milliseconds.
 */
double dynamic_governor_get_deadline_ms(const dynamic_governor_t *gov);

/**
 * Returns total count of dropped frames prevented by governor pacing.
 */
uint32_t dynamic_governor_get_dropped_frames_prevented(const dynamic_governor_t *gov);

/**
 * Updates governor with the measured GPU frame completion latency.
 * Returns the active operational tier for the NEXT frame.
 */
governor_tier_t dynamic_governor_update(dynamic_governor_t *gov, double gpu_latency_ms);

/**
 * Gets the current active tier without updating metrics.
 */
governor_tier_t dynamic_governor_get_tier(const dynamic_governor_t *gov);

/**
 * Resets governor state.
 */
void dynamic_governor_reset(dynamic_governor_t *gov);

/**
 * Notifies governor that an emergency Tier 3 failover frame was handled
 * and immediately transitions to Tier 2 (CPU offload) for the subsequent frame.
 */
void dynamic_governor_notify_failover_handled(dynamic_governor_t *gov);

/**
 * Returns human-readable name of a governor tier.
 */
const char *dynamic_governor_tier_name(governor_tier_t tier);

/**
 * Queries cumulative telemetry statistics from the governor.
 */
void dynamic_governor_get_stats(const dynamic_governor_t *gov, governor_stats_t *out_stats);

#ifdef __cplusplus
}
#endif

#endif /* BC250_DYNAMIC_GOVERNOR_H */
