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

#if defined(__linux__)
extern char *program_invocation_short_name;
#endif

void dynamic_governor_init(dynamic_governor_t *gov)
{
    if (!gov) return;
    memset(gov, 0, sizeof(*gov));
    gov->tier1_threshold_ms = 8.0;
    gov->tier2_threshold_ms = 12.0;
    gov->tier3_threshold_ms = 15.5;
    gov->frame_deadline_ms = 16.666667;
    gov->tier1_overridden = false;
    gov->tier2_overridden = false;
    gov->tier3_overridden = false;
    gov->predictive_pacing_enabled = true;
    gov->dropped_frames_prevented = 0;
    gov->step_down_hysteresis = 4;
    gov->min_dwell_frames = 0;
    gov->frames_in_tier = 0;
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
    gov->allow_failover = true;

#if defined(__linux__)
    if (program_invocation_short_name &&
        (strcmp(program_invocation_short_name, "sunshine") == 0 ||
         strcmp(program_invocation_short_name, "wivrn-server") == 0 ||
         strcmp(program_invocation_short_name, "wivrn") == 0 ||
         strcmp(program_invocation_short_name, "steam") == 0 ||
         strcmp(program_invocation_short_name, "streaming_client") == 0 ||
         strcmp(program_invocation_short_name, "gamescope") == 0 ||
         strstr(program_invocation_short_name, "gamescope") != NULL)) {
        gov->enabled = true;
    }
    if (program_invocation_short_name &&
        (strcmp(program_invocation_short_name, "sunshine") == 0 ||
         strcmp(program_invocation_short_name, "steam") == 0 ||
         strcmp(program_invocation_short_name, "streaming_client") == 0 ||
         strcmp(program_invocation_short_name, "steamwebhelper") == 0 ||
         strcmp(program_invocation_short_name, "gamescope") == 0 ||
         strstr(program_invocation_short_name, "steam") != NULL ||
         strstr(program_invocation_short_name, "gamescope") != NULL)) {
        /* In Sunshine and Steam Link, enable hybrid CPU SIMD ME offload by default
         * and tune thresholds to protect 60fps streaming deadlines (<16.6ms).
         * Tier 0: < 7.0ms (GPU Full ME)
         * Tier 1: 7.0 - 10.5ms (GPU Fast ME)
         * Tier 2: 10.5 - 15.5ms (CPU SIMD Offload, relieves GPU CUs for games)
         * Tier 3: > 15.5ms (Emergency Failover P_Skip)
         * Increase step_down_hysteresis to 8 to avoid fluttering under heavy game contention.
         *
         * min_dwell_frames = 8 (133ms at 60fps) is the other half of that:
         * step_down_hysteresis governs leaving the offload tier, this governs
         * entering it, and without it a single bad frame could cross into a
         * full CPU ME and back. Together they bound how often the per-frame
         * work mix can change, which is what the ">200ms spikes" on a live
         * stream were. */
        gov->cpu_offload_enabled = true;
        gov->tier1_threshold_ms = 7.0;
        gov->tier2_threshold_ms = 10.5;
        gov->tier3_threshold_ms = 15.5;
        gov->step_down_hysteresis = 8;
        gov->min_dwell_frames = 8;
    }
    if (program_invocation_short_name && strcmp(program_invocation_short_name, "ffmpeg") == 0) {
        /* For FFmpeg transcoding in compute/hybrid mode, enable CPU ME offload
         * but disable Tier 3 P_Skip failover so no video frames are dropped. */
        gov->allow_failover = false;
        gov->cpu_offload_enabled = true;
    }
#endif

    const char *env_t1 = getenv("BC250_GOVERNOR_TIER1_MS");
    if (env_t1) {
        double v = atof(env_t1);
        if (v > 0) { gov->tier1_threshold_ms = v; gov->tier1_overridden = true; }
    }
    const char *env_t2 = getenv("BC250_GOVERNOR_TIER2_MS");
    if (env_t2) {
        double v = atof(env_t2);
        if (v > 0) { gov->tier2_threshold_ms = v; gov->tier2_overridden = true; }
    }
    const char *env_t3 = getenv("BC250_GOVERNOR_TIER3_MS");
    if (env_t3) {
        double v = atof(env_t3);
        if (v > 0) { gov->tier3_threshold_ms = v; gov->tier3_overridden = true; }
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

    /* Dwell before entering the CPU offload tier. 0 disables the gate, which
     * is the right default everywhere but a live stream: it exists to stop a
     * single bad frame changing the per-frame work mix, and an offline
     * transcode has no such thing as a bad frame. */
    const char *env_dwell = getenv("BC250_GOVERNOR_DWELL_FRAMES");
    if (env_dwell) {
        int d = atoi(env_dwell);
        if (d >= 0) gov->min_dwell_frames = (uint32_t)d;
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

void dynamic_governor_set_deadline_ms(dynamic_governor_t *gov, double deadline_ms)
{
    if (!gov || deadline_ms <= 0.0) return;
    gov->frame_deadline_ms = deadline_ms;

    /* Governor v2 Deadline-Aware Pacing:
     * Gear-shifting thresholds scale proportionally with target frame budget:
     * - Tier 1 (GPU Fast ME / Downhill): 45% of frame deadline (e.g. 7.5ms @ 60fps, 3.75ms @ 120fps)
     * - Tier 2 (CPU SIMD ME Offload): 65% of frame deadline (e.g. 10.8ms @ 60fps, 5.4ms @ 120fps)
     * - Tier 3 (Emergency Failover P_Skip): 92% of frame deadline (e.g. 15.3ms @ 60fps, 7.67ms @ 120fps)
     * Explicit environment overrides (BC250_GOVERNOR_TIERx_MS) take precedence. */
    if (!gov->tier1_overridden) {
        gov->tier1_threshold_ms = deadline_ms * 0.45;
    }
    if (!gov->tier2_overridden) {
        gov->tier2_threshold_ms = deadline_ms * 0.65;
    }
    if (!gov->tier3_overridden) {
        gov->tier3_threshold_ms = deadline_ms * 0.92;
    }
}

void dynamic_governor_set_framerate(dynamic_governor_t *gov, double fps)
{
    if (!gov || fps <= 0.0) return;
    dynamic_governor_set_deadline_ms(gov, 1000.0 / fps);
}

double dynamic_governor_get_deadline_ms(const dynamic_governor_t *gov)
{
    return gov ? gov->frame_deadline_ms : 16.666667;
}

uint32_t dynamic_governor_get_dropped_frames_prevented(const dynamic_governor_t *gov)
{
    return gov ? gov->dropped_frames_prevented : 0;
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

    double latency_velocity = (gov->last_latency_ms > 0.0) ? (gpu_latency_ms - gov->last_latency_ms) : 0.0;
    gov->last_latency_ms = gpu_latency_ms;
    gov->total_frames++;
    gov->frames_in_tier++;

    /* Initialize or update Exponential Moving Average (EMA) with alpha=0.25 */
    if (gov->ema_latency_ms <= 0.0) {
        gov->ema_latency_ms = gpu_latency_ms;
    } else {
        gov->ema_latency_ms = 0.75 * gov->ema_latency_ms + 0.25 * gpu_latency_ms;
    }

    double metric = gov->ema_latency_ms;

    /* Predictive velocity pacing: if latency is accelerating steeply under 3D game contention,
     * project impending deadline breach and shift gears before the frame deadline is missed */
    if (gov->predictive_pacing_enabled && latency_velocity > 1.5 && gpu_latency_ms >= gov->tier1_threshold_ms) {
        double predicted = gov->ema_latency_ms + 0.5 * latency_velocity;
        if (predicted > metric) metric = predicted;
    }

    /* Emergency spike trip-wire: single frame over threshold triggers failover.
     * Deliberately NOT subject to the dwell gate below: dropping a frame
     * removes work rather than moving it elsewhere, so it cannot make a
     * deadline worse, and it is the one response that must always be
     * available. */
    if (gpu_latency_ms >= gov->tier3_threshold_ms) {
        if (gov->allow_failover) {
            gov->current_tier = GOV_TIER_3_FAILOVER;
        } else {
            gov->current_tier = gov->cpu_offload_enabled ? GOV_TIER_2_CPU_OFFLOAD : GOV_TIER_1_GPU_FAST;
        }
        gov->dropped_frames_prevented++;
        gov->stable_frames_count = 0;
        gov->frames_in_tier = 1;
    } else if (gov->current_tier == GOV_TIER_3_FAILOVER) {
        /* Drop from Tier 3 immediately after the emergency frame, and land on
         * the CHEAPEST tier rather than the CPU offload.
         *
         * The frame before this one was dropped because the GPU was late
         * *once*. Handing the very next frame to a full CPU motion search
         * treats one late frame as proof that the GPU is unusable, and on a
         * live stream that frame is the most expensive frame in the cycle -
         * the search is a whole-frame pass over memory the CPU cannot cache
         * (see me_src_stage in encoder_h264.c). Tier 1 is the cheapest thing
         * that still uses the GPU, and the EMA above - which is an average
         * over 4 frames, so one spike barely moves it - will still promote
         * this stream to the offload tier if the pressure is real and
         * sustained. A spike costs one frame; sustained pressure costs the
         * tier it deserves. */
        gov->current_tier = GOV_TIER_1_GPU_FAST;
        gov->stable_frames_count = 0;
        gov->frames_in_tier = 1;
    } else if (metric >= gov->tier2_threshold_ms) {
        /* If CPU offload is enabled, transition to Tier 2; otherwise clamp to Tier 1 GPU Fast ME */
        governor_tier_t target_tier = gov->cpu_offload_enabled ? GOV_TIER_2_CPU_OFFLOAD : GOV_TIER_1_GPU_FAST;
        /* The dwell gate, and it only ever gates the expensive tier: crossing
         * into the CPU offload requires the current tier to have been held
         * for min_dwell_frames, so a frame that lands the EMA over the
         * threshold right after a tier change is answered with Tier 1 (a
         * reduced search window on the GPU, i.e. strictly less work than
         * Tier 2) and the offload is only entered once the pressure has been
         * there for a while. Leaving the offload is governed by
         * step_down_hysteresis below, as before. */
        if (target_tier == GOV_TIER_2_CPU_OFFLOAD &&
            gov->frames_in_tier < gov->min_dwell_frames) {
            target_tier = GOV_TIER_1_GPU_FAST;
        }
        if (gov->current_tier < target_tier) {
            gov->current_tier = target_tier;
            gov->dropped_frames_prevented++;
            gov->stable_frames_count = 0;
            gov->frames_in_tier = 1;
        } else {
            gov->stable_frames_count = 0;
        }
    } else if (gov->current_tier == GOV_TIER_2_CPU_OFFLOAD) {
        /* Downward tier transitions from Tier 2 require hysteresis to avoid fluttering */
        gov->stable_frames_count++;
        if (gov->stable_frames_count >= gov->step_down_hysteresis) {
            gov->current_tier = (metric >= gov->tier1_threshold_ms) ? GOV_TIER_1_GPU_FAST : GOV_TIER_0_GPU_FULL;
            gov->stable_frames_count = 0;
            gov->frames_in_tier = 1;
        }
    } else if (metric >= gov->tier1_threshold_ms) {
        /* Upward tier transition to Tier 1 */
        if (gov->current_tier < GOV_TIER_1_GPU_FAST) {
            gov->current_tier = GOV_TIER_1_GPU_FAST;
            gov->dropped_frames_prevented++;
            gov->stable_frames_count = 0;
            gov->frames_in_tier = 1;
        } else {
            gov->stable_frames_count = 0;
        }
    } else if (gov->current_tier == GOV_TIER_1_GPU_FAST) {
        /* Downward tier transitions from Tier 1 require hysteresis */
        gov->stable_frames_count++;
        if (gov->stable_frames_count >= gov->step_down_hysteresis) {
            gov->current_tier = GOV_TIER_0_GPU_FULL;
            gov->stable_frames_count = 0;
            gov->frames_in_tier = 1;
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
    gov->frames_in_tier = 0;
    gov->dropped_frames_prevented = 0;
    gov->current_tier = (gov->forced_tier >= 0) ? (governor_tier_t)gov->forced_tier : GOV_TIER_0_GPU_FULL;
}

void dynamic_governor_notify_failover_handled(dynamic_governor_t *gov)
{
    if (!gov) return;
    if (gov->current_tier == GOV_TIER_3_FAILOVER) {
        /* Same reasoning as the Tier 3 step-down inside dynamic_governor_update():
         * the frame that failed over proves the GPU was late once, not that
         * the GPU is unusable, so the frame after it gets the cheapest tier
         * and the EMA decides whether the offload is really warranted. */
        gov->current_tier = GOV_TIER_1_GPU_FAST;
        gov->stable_frames_count = 0;
        gov->frames_in_tier = 1;
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
    out_stats->frame_deadline_ms = gov->frame_deadline_ms;
    out_stats->dropped_frames_prevented = gov->dropped_frames_prevented;
}
