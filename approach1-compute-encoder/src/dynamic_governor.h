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
    double tier1_threshold_ms;     /* Default: 8.0 ms */
    double tier2_threshold_ms;     /* Default: 12.0 ms */
    double tier3_threshold_ms;     /* Default: 15.5 ms */
    uint32_t step_down_hysteresis; /* Default: 4 frames */
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
