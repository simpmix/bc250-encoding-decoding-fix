/* bc250-encoding-decoding-fix v0.4.3 - https://github.com/simpmix/bc250-encoding-decoding-fix */
/*
 * Copyright (c) 2026 BC-250 Project Contributors
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * dynamic_governor.c - Real-time dynamic CPU/GPU load governor for BC-250
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <errno.h>
#include "dynamic_governor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

void dynamic_governor_init(dynamic_governor_t *gov)
{
    if (!gov) return;
    memset(gov, 0, sizeof(*gov));
    gov->tier1_threshold_ms = 8.0;
    gov->tier2_threshold_ms = 12.0;
    gov->tier3_threshold_ms = 15.5;
    gov->step_down_hysteresis = 4;
    gov->current_tier = GOV_TIER_0_GPU_FULL;
    /* ⚠️ Only for a live stream. Every tier trades the picture for time:
     * tier 1 searches motion less carefully, tier 3 drops the frame and
     * repeats the last one. That is the right trade when a client is
     * waiting for the frame, and the wrong one when a file is being
     * written, where there is no deadline at all. With the thresholds
     * below, ffmpeg encoding real 1080p50 through the compute H.264
     * encoder spent 16.3 ms of GPU per frame, tripped tier 3 on every
     * one, and so repeated every second frame: 24 dB where 37 was there to
     * be had. BC250_GOVERNOR_ENABLE=1 turns it on for anyone. */
    gov->enabled = false;
    gov->forced_tier = -1;
    gov->cpu_offload_enabled = false;

#if defined(__linux__)
    if (program_invocation_short_name &&
        (strcmp(program_invocation_short_name, "sunshine") == 0 ||
         strcmp(program_invocation_short_name, "wivrn-server") == 0 ||
         strcmp(program_invocation_short_name, "wivrn") == 0 ||
         strcmp(program_invocation_short_name, "steam") == 0 ||
         strcmp(program_invocation_short_name, "streaming_client") == 0)) {
        gov->enabled = true;
    }
    if (program_invocation_short_name &&
        (strcmp(program_invocation_short_name, "sunshine") == 0 ||
         strcmp(program_invocation_short_name, "steam") == 0 ||
         strcmp(program_invocation_short_name, "streaming_client") == 0)) {
        /* In Sunshine and Steam Link, enable hybrid CPU SIMD ME offload by default
         * and tune thresholds to protect 60fps streaming deadlines (<16.6ms).
         * Tier 0: < 7.0ms (GPU Full ME)
         * Tier 1: 7.0 - 10.5ms (GPU Fast ME)
         * Tier 2: 10.5 - 15.5ms (CPU SIMD Offload, relieves GPU CUs for games)
         * Tier 3: > 15.5ms (Emergency Failover P_Skip) */
        gov->cpu_offload_enabled = true;
        gov->tier1_threshold_ms = 7.0;
        gov->tier2_threshold_ms = 10.5;
        gov->tier3_threshold_ms = 15.5;
    }
#endif

    const char *env_t1 = getenv("BC250_GOVERNOR_TIER1_MS");
    if (env_t1) {
        double v = atof(env_t1);
        if (v > 0) gov->tier1_threshold_ms = v;
    }
    const char *env_t2 = getenv("BC250_GOVERNOR_TIER2_MS");
    if (env_t2) {
        double v = atof(env_t2);
        if (v > 0) gov->tier2_threshold_ms = v;
    }
    const char *env_t3 = getenv("BC250_GOVERNOR_TIER3_MS");
    if (env_t3) {
        double v = atof(env_t3);
        if (v > 0) gov->tier3_threshold_ms = v;
    }

    const char *env_enable = getenv("BC250_GOVERNOR_ENABLE");
    if (env_enable && (strcmp(env_enable, "0") == 0 || strcmp(env_enable, "false") == 0)) {
        gov->enabled = false;
    } else if (env_enable && (strcmp(env_enable, "1") == 0 || strcmp(env_enable, "true") == 0)) {
        gov->enabled = true;
    }

    const char *env_cpu_me = getenv("BC250_ENABLE_CPU_ME");
    if (!env_cpu_me) env_cpu_me = getenv("BC250_TIER2_ENABLE");
    if (env_cpu_me && (strcmp(env_cpu_me, "1") == 0 || strcmp(env_cpu_me, "true") == 0)) {
        gov->cpu_offload_enabled = true;
    } else if (env_cpu_me && (strcmp(env_cpu_me, "0") == 0 || strcmp(env_cpu_me, "false") == 0)) {
        gov->cpu_offload_enabled = false;
    }

    const char *env_hyst = getenv("BC250_GOVERNOR_HYSTERESIS");
    if (env_hyst) {
        int h = atoi(env_hyst);
        if (h > 0) gov->step_down_hysteresis = (uint32_t)h;
    }

    const char *env_force = getenv("BC250_FORCE_TIER");
    if (env_force) {
        int ft = atoi(env_force);
        if (ft >= 0 && ft <= 3) {
            gov->forced_tier = ft;
            gov->current_tier = (governor_tier_t)ft;
        }
    }

    const char *env_stats = getenv("BC250_GOVERNOR_STATS");
    if (env_stats) {
        if (strcmp(env_stats, "1") == 0 || strcmp(env_stats, "true") == 0) {
            gov->stats_log_interval = 60; /* Log telemetry every 60 frames (~1 sec at 60fps) */
        } else {
            int interval = atoi(env_stats);
            if (interval > 0) gov->stats_log_interval = interval;
        }
    }
}

governor_tier_t dynamic_governor_update(dynamic_governor_t *gov, double gpu_latency_ms)
{
    if (!gov) return GOV_TIER_0_GPU_FULL;

    if (gov->forced_tier >= 0) {
        return (governor_tier_t)gov->forced_tier;
    }

    if (!gov->enabled) {
        return GOV_TIER_0_GPU_FULL;
    }

    if (gpu_latency_ms < 0.0) {
        gpu_latency_ms = 0.0;
    }

    gov->last_latency_ms = gpu_latency_ms;
    gov->total_frames++;

    /* Initialize or update Exponential Moving Average (EMA) with alpha=0.25 */
    if (gov->ema_latency_ms <= 0.0) {
        gov->ema_latency_ms = gpu_latency_ms;
    } else {
        gov->ema_latency_ms = 0.75 * gov->ema_latency_ms + 0.25 * gpu_latency_ms;
    }

    double metric = gov->ema_latency_ms;

    /* Emergency spike trip-wire: single frame over threshold triggers failover */
    if (gpu_latency_ms >= gov->tier3_threshold_ms) {
        gov->current_tier = GOV_TIER_3_FAILOVER;
        gov->stable_frames_count = 0;
    } else if (gov->current_tier == GOV_TIER_3_FAILOVER) {
        /* Drop from Tier 3 immediately after the emergency frame */
        gov->current_tier = gov->cpu_offload_enabled ? GOV_TIER_2_CPU_OFFLOAD : GOV_TIER_1_GPU_FAST;
        gov->stable_frames_count = 0;
    } else if (metric >= gov->tier2_threshold_ms) {
        /* If CPU offload is enabled, transition to Tier 2; otherwise clamp to Tier 1 GPU Fast ME */
        governor_tier_t target_tier = gov->cpu_offload_enabled ? GOV_TIER_2_CPU_OFFLOAD : GOV_TIER_1_GPU_FAST;
        if (gov->current_tier < target_tier) {
            gov->current_tier = target_tier;
            gov->stable_frames_count = 0;
        } else {
            gov->stable_frames_count = 0;
        }
    } else if (gov->current_tier == GOV_TIER_2_CPU_OFFLOAD) {
        /* Downward tier transitions from Tier 2 require hysteresis to avoid fluttering */
        gov->stable_frames_count++;
        if (gov->stable_frames_count >= gov->step_down_hysteresis) {
            gov->current_tier = (metric >= gov->tier1_threshold_ms) ? GOV_TIER_1_GPU_FAST : GOV_TIER_0_GPU_FULL;
            gov->stable_frames_count = 0;
        }
    } else if (metric >= gov->tier1_threshold_ms) {
        /* Upward tier transition to Tier 1 */
        if (gov->current_tier < GOV_TIER_1_GPU_FAST) {
            gov->current_tier = GOV_TIER_1_GPU_FAST;
            gov->stable_frames_count = 0;
        } else {
            gov->stable_frames_count = 0;
        }
    } else if (gov->current_tier == GOV_TIER_1_GPU_FAST) {
        /* Downward tier transitions from Tier 1 require hysteresis */
        gov->stable_frames_count++;
        if (gov->stable_frames_count >= gov->step_down_hysteresis) {
            gov->current_tier = GOV_TIER_0_GPU_FULL;
            gov->stable_frames_count = 0;
        }
    } else {
        gov->stable_frames_count = 0;
    }

    if (gov->current_tier == GOV_TIER_0_GPU_FULL) gov->total_tier0_frames++;
    else if (gov->current_tier == GOV_TIER_1_GPU_FAST) gov->total_tier1_frames++;
    else if (gov->current_tier == GOV_TIER_2_CPU_OFFLOAD) gov->total_offload_frames++;
    else if (gov->current_tier == GOV_TIER_3_FAILOVER) gov->total_failover_frames++;

    if (gov->stats_log_interval > 0 && (gov->total_frames % (uint32_t)gov->stats_log_interval == 0)) {
        fprintf(stderr, "[bc250-gov] Frame %u: Tier %d (%s) | GPU: %.2f ms | EMA: %.2f ms | Offload: %u | Failover: %u\n",
                gov->total_frames,
                (int)gov->current_tier,
                dynamic_governor_tier_name(gov->current_tier),
                gov->last_latency_ms,
                gov->ema_latency_ms,
                gov->total_offload_frames,
                gov->total_failover_frames);
    }

    return gov->current_tier;
}

governor_tier_t dynamic_governor_get_tier(const dynamic_governor_t *gov)
{
    if (!gov) return GOV_TIER_0_GPU_FULL;
    if (gov->forced_tier >= 0) return (governor_tier_t)gov->forced_tier;
    if (!gov->enabled) return GOV_TIER_0_GPU_FULL;
    return gov->current_tier;
}

void dynamic_governor_reset(dynamic_governor_t *gov)
{
    if (!gov) return;
    gov->ema_latency_ms = 0.0;
    gov->last_latency_ms = 0.0;
    gov->stable_frames_count = 0;
    gov->current_tier = (gov->forced_tier >= 0) ? (governor_tier_t)gov->forced_tier : GOV_TIER_0_GPU_FULL;
}

void dynamic_governor_notify_failover_handled(dynamic_governor_t *gov)
{
    if (!gov) return;
    if (gov->current_tier == GOV_TIER_3_FAILOVER) {
        gov->current_tier = gov->cpu_offload_enabled ? GOV_TIER_2_CPU_OFFLOAD : GOV_TIER_1_GPU_FAST;
        gov->stable_frames_count = 0;
    }
}

const char *dynamic_governor_tier_name(governor_tier_t tier)
{
    switch (tier) {
        case GOV_TIER_0_GPU_FULL: return "GPU Full ME";
        case GOV_TIER_1_GPU_FAST: return "GPU Fast ME";
        case GOV_TIER_2_CPU_OFFLOAD: return "CPU SIMD Offload";
        case GOV_TIER_3_FAILOVER: return "Emergency Failover";
        default: return "Unknown";
    }
}

void dynamic_governor_get_stats(const dynamic_governor_t *gov, governor_stats_t *out_stats)
{
    if (!gov || !out_stats) return;
    out_stats->total_frames = gov->total_frames;
    out_stats->total_tier0_frames = gov->total_tier0_frames;
    out_stats->total_tier1_frames = gov->total_tier1_frames;
    out_stats->total_offload_frames = gov->total_offload_frames;
    out_stats->total_failover_frames = gov->total_failover_frames;
    out_stats->ema_latency_ms = gov->ema_latency_ms;
    out_stats->last_latency_ms = gov->last_latency_ms;
    out_stats->current_tier = gov->current_tier;
}
