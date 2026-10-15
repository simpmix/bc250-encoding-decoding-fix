/* bc250-encoding-decoding-fix v0.4.3 - https://github.com/simpmix/bc250-encoding-decoding-fix */
/*
 * Copyright (c) 2026 BC-250 Project Contributors
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * test_dynamic_governor.c - Unit test for Dynamic Asymmetric Governor
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "dynamic_governor.h"
#include "telemetry.h"

static void test_governor_transitions(void)
{
    printf("[TEST] Testing dynamic governor tier transitions & hysteresis...\n");

    dynamic_governor_t gov;
    dynamic_governor_init(&gov);
    gov.cpu_offload_enabled = true;
    gov.step_down_hysteresis = 15;
    assert(dynamic_governor_get_tier(&gov) == GOV_TIER_0_GPU_FULL);

    /* 1. Low latency: stays Tier 0 */
    for (int i = 0; i < 10; i++) {
        governor_tier_t t = dynamic_governor_update(&gov, 4.5);
        assert(t == GOV_TIER_0_GPU_FULL);
    }

    /* 2. Moderate latency spike (>8ms): transitions to Tier 1 */
    for (int i = 0; i < 5; i++) {
        dynamic_governor_update(&gov, 9.5);
    }
    assert(dynamic_governor_get_tier(&gov) == GOV_TIER_1_GPU_FAST);

    /* 3. Severe latency spike (>12ms): transitions to Tier 2 (CPU offload) */
    for (int i = 0; i < 5; i++) {
        dynamic_governor_update(&gov, 13.5);
    }
    assert(dynamic_governor_get_tier(&gov) == GOV_TIER_2_CPU_OFFLOAD);

    /* 4. Single-frame emergency spike (>15.5ms): trips Tier 3 (Failover) */
    governor_tier_t emergency = dynamic_governor_update(&gov, 16.5);
    assert(emergency == GOV_TIER_3_FAILOVER);

    /* 5. Next frame drops from Tier 3 to the CHEAPEST tier, not to the CPU
     * offload. One late frame is not evidence that the GPU is unusable, and
     * the offload is the most expensive frame in the cycle on a live stream
     * (a whole-frame CPU motion search) - see dynamic_governor_update(). */
    governor_tier_t post_emergency = dynamic_governor_update(&gov, 13.0);
    assert(post_emergency == GOV_TIER_1_GPU_FAST);

    /* 6. ...but sustained pressure still gets the offload: the EMA is an
     * average over 4 frames, so it stays over the tier-2 threshold and the
     * next update promotes it. (Dwell is 0 in this test - see
     * test_governor_dwell_gate() for the live-stream value.) */
    dynamic_governor_update(&gov, 13.5);
    assert(dynamic_governor_get_tier(&gov) == GOV_TIER_2_CPU_OFFLOAD);

    /* 7. Hysteresis test: lower latency (4.0ms) requires 15 stable frames to drop back to Tier 0 */
    for (int i = 0; i < 14; i++) {
        dynamic_governor_update(&gov, 4.0);
        /* Should NOT have dropped back to Tier 0 yet due to hysteresis */
        assert(dynamic_governor_get_tier(&gov) >= GOV_TIER_1_GPU_FAST);
    }
    /* 15th frame triggers step down */
    dynamic_governor_update(&gov, 4.0);
    assert(dynamic_governor_get_tier(&gov) == GOV_TIER_0_GPU_FULL);

    printf("  ✓ Governor transitions and hysteresis verified!\n");
}

/* The dwell gate is the fix for the reported ">200ms spikes": it stops a
 * single frame's worth of pressure from changing the per-frame work mix.
 * With min_dwell_frames = 8 (the live-stream default), the tier the governor
 * just moved to must be held for 8 frames before the CPU offload can be
 * entered - and while it is being held, an over-threshold EMA is answered
 * with Tier 1 (less GPU work than Tier 0) rather than with a full CPU ME. */
static void test_governor_dwell_gate(void)
{
    printf("[TEST] Testing the dwell gate before entering the CPU offload...\n");

    dynamic_governor_t gov;
    dynamic_governor_init(&gov);
    gov.cpu_offload_enabled = true;
    gov.min_dwell_frames = 8;

    /* Establish a healthy Tier 0 baseline, so the pressure below is the only
     * thing that can move the tier. */
    for (int i = 0; i < 10; i++) {
        dynamic_governor_update(&gov, 4.0);
    }
    assert(dynamic_governor_get_tier(&gov) == GOV_TIER_0_GPU_FULL);

    /* Now hold 14ms of GPU latency - over the tier-2 threshold, under the
     * emergency one - and record when the governor reached each tier. The
     * EMA is an average over 4 frames, so it takes a few updates to climb;
     * the point of interest is the gap between the two transitions. */
    int tier1_at = -1, offload_at = -1;
    for (int i = 0; i < 60; i++) {
        dynamic_governor_update(&gov, 14.0);
        governor_tier_t t = dynamic_governor_get_tier(&gov);
        if (t == GOV_TIER_1_GPU_FAST && tier1_at < 0) tier1_at = i;
        if (t == GOV_TIER_2_CPU_OFFLOAD) { offload_at = i; break; }
    }
    assert(tier1_at >= 0);
    assert(offload_at >= 0);
    /* The offload must not be entered until the tier it is being entered from
     * has been held for the full dwell, counting the crossing update (the
     * frame that changed tier counts as the first frame of the new tier). */
    assert(offload_at - tier1_at + 1 >= (int)gov.min_dwell_frames);

    /* Control, and the reason the gate is off by default: with no dwell the
     * very same input crosses into the offload as soon as the EMA gets there,
     * several frames earlier. An offline transcode wants exactly that. */
    dynamic_governor_t gov0;
    dynamic_governor_init(&gov0);
    gov0.cpu_offload_enabled = true;
    gov0.min_dwell_frames = 0;
    for (int i = 0; i < 10; i++) dynamic_governor_update(&gov0, 4.0);
    int tier1_at0 = -1, offload_at0 = -1;
    for (int i = 0; i < 60; i++) {
        dynamic_governor_update(&gov0, 14.0);
        governor_tier_t t = dynamic_governor_get_tier(&gov0);
        if (t == GOV_TIER_1_GPU_FAST && tier1_at0 < 0) tier1_at0 = i;
        if (t == GOV_TIER_2_CPU_OFFLOAD) { offload_at0 = i; break; }
    }
    assert(offload_at0 >= 0);
    assert(offload_at0 - tier1_at0 + 1 < (int)gov.min_dwell_frames);

    printf("  ✓ Dwell gate verified!\n");
}

static void test_governor_gpu_only_mode(void)
{
    printf("[TEST] Testing default GPU-only governor mode (Tier 2 CPU ME offload disabled)...\n");

    dynamic_governor_t gov;
    dynamic_governor_init(&gov);
    assert(!gov.cpu_offload_enabled);
    assert(gov.step_down_hysteresis == 4);

    /* 1. Low latency: stays Tier 0 */
    for (int i = 0; i < 5; i++) {
        governor_tier_t t = dynamic_governor_update(&gov, 4.0);
        assert(t == GOV_TIER_0_GPU_FULL);
    }

    /* 2. Severe latency spike (>12ms): caps at Tier 1 (GPU Fast ME), never enters Tier 2 */
    for (int i = 0; i < 5; i++) {
        dynamic_governor_update(&gov, 14.0);
    }
    assert(dynamic_governor_get_tier(&gov) == GOV_TIER_1_GPU_FAST);

    /* 3. Emergency spike (>15.5ms): trips Tier 3 (Failover) for 1 frame */
    governor_tier_t emergency = dynamic_governor_update(&gov, 16.5);
    assert(emergency == GOV_TIER_3_FAILOVER);

    /* 4. Unlatch drops to Tier 1 GPU Fast ME, not Tier 2 */
    dynamic_governor_notify_failover_handled(&gov);
    assert(dynamic_governor_get_tier(&gov) == GOV_TIER_1_GPU_FAST);

    /* 5. Fast recovery in stable frames */
    for (int i = 0; i < 10; i++) {
        dynamic_governor_update(&gov, 3.5);
    }
    assert(dynamic_governor_get_tier(&gov) == GOV_TIER_0_GPU_FULL);

    printf("  ✓ Default GPU-only governor mode verified!\n");
}

static void test_governor_failover_handled(void)
{
    printf("[TEST] Testing failover notification and unlatch...\n");

    dynamic_governor_t gov;
    dynamic_governor_init(&gov);
    gov.cpu_offload_enabled = true;

    /* Trip Tier 3 emergency failover */
    dynamic_governor_update(&gov, 17.0);
    assert(dynamic_governor_get_tier(&gov) == GOV_TIER_3_FAILOVER);

    /* In Tier 3, no GPU work is submitted, so dynamic_governor_update() is not called.
     * notify_failover_handled must transition immediately down to the cheapest tier -
     * Tier 1, not the CPU offload. The frame that failed over proves the GPU was late
     * once, and the offload is the most expensive thing this encoder can do with a
     * frame; sustained pressure promotes to it through the EMA instead. */
    dynamic_governor_notify_failover_handled(&gov);
    assert(dynamic_governor_get_tier(&gov) == GOV_TIER_1_GPU_FAST);

    /* Calling it again while already unlatched should be a safe no-op */
    dynamic_governor_notify_failover_handled(&gov);
    assert(dynamic_governor_get_tier(&gov) == GOV_TIER_1_GPU_FAST);

    /* Calling with NULL should be safe */
    dynamic_governor_notify_failover_handled(NULL);

    printf("  ✓ Governor failover unlatch verified!\n");
}

static void test_governor_negative_latency(void)
{
    printf("[TEST] Testing negative latency guard...\n");

    dynamic_governor_t gov;
    dynamic_governor_init(&gov);

    /* Passing a negative latency (e.g. clock anomaly) must be clamped to 0.0 */
    governor_tier_t t = dynamic_governor_update(&gov, -10.0);
    assert(t == GOV_TIER_0_GPU_FULL);
    assert(gov.last_latency_ms == 0.0);

    printf("  ✓ Negative latency guard verified!\n");
}

static void test_governor_telemetry_stats(void)
{
    printf("[TEST] Testing governor telemetry stats & tier names...\n");

    /* Test tier names */
    assert(strcmp(dynamic_governor_tier_name(GOV_TIER_0_GPU_FULL), "GPU Full ME") == 0);
    assert(strcmp(dynamic_governor_tier_name(GOV_TIER_1_GPU_FAST), "GPU Fast ME") == 0);
    assert(strcmp(dynamic_governor_tier_name(GOV_TIER_2_CPU_OFFLOAD), "CPU SIMD Offload") == 0);
    assert(strcmp(dynamic_governor_tier_name(GOV_TIER_3_FAILOVER), "Emergency Failover") == 0);

    dynamic_governor_t gov;
    dynamic_governor_init(&gov);
    gov.cpu_offload_enabled = true;
    gov.stats_log_interval = 5; /* trigger logging every 5 frames for test */

    for (int i = 0; i < 5; i++) {
        dynamic_governor_update(&gov, 5.0);
    }
    for (int i = 0; i < 10; i++) {
        dynamic_governor_update(&gov, 14.0);
    }

    governor_stats_t stats;
    dynamic_governor_get_stats(&gov, &stats);
    assert(stats.total_frames == 15);
    assert(stats.total_tier0_frames >= 4);
    assert(stats.total_offload_frames >= 1);
    assert(stats.last_latency_ms == 14.0);

    /* Test NULL safety */
    dynamic_governor_get_stats(NULL, &stats);
    dynamic_governor_get_stats(&gov, NULL);

    printf("  ✓ Governor telemetry stats & tier names verified!\n");
}

static void test_governor_deadline_and_predictive_pacing(void)
{
    printf("[TEST] Testing Governor v2 deadline scaling & predictive velocity pacing...\n");

    dynamic_governor_t gov;
    dynamic_governor_init(&gov);

    /* 1. Test 60 FPS deadline scaling */
    dynamic_governor_set_framerate(&gov, 60.0);
    assert(fabs(dynamic_governor_get_deadline_ms(&gov) - 16.666667) < 0.01);
    assert(fabs(gov.tier1_threshold_ms - 7.50) < 0.05);
    assert(fabs(gov.tier2_threshold_ms - 10.83) < 0.05);
    assert(fabs(gov.tier3_threshold_ms - 15.33) < 0.05);

    /* 2. Test 120 FPS deadline scaling */
    dynamic_governor_set_framerate(&gov, 120.0);
    assert(fabs(dynamic_governor_get_deadline_ms(&gov) - 8.333333) < 0.01);
    assert(fabs(gov.tier1_threshold_ms - 3.75) < 0.05);
    assert(fabs(gov.tier2_threshold_ms - 5.42) < 0.05);
    assert(fabs(gov.tier3_threshold_ms - 7.67) < 0.05);

    /* 3. Test 90 FPS deadline scaling */
    dynamic_governor_set_framerate(&gov, 90.0);
    assert(fabs(dynamic_governor_get_deadline_ms(&gov) - 11.111111) < 0.01);
    assert(fabs(gov.tier1_threshold_ms - 5.00) < 0.05);

    /* 4. Test predictive velocity pacing at 60 FPS */
    dynamic_governor_reset(&gov);
    dynamic_governor_set_framerate(&gov, 60.0);
    assert(gov.dropped_frames_prevented == 0);

    /* Baseline frame at 6.0ms */
    governor_tier_t t0 = dynamic_governor_update(&gov, 6.0);
    assert(t0 == GOV_TIER_0_GPU_FULL);
    assert(gov.dropped_frames_prevented == 0);

    /* Sudden contention jump: 6.0ms -> 8.5ms (velocity = +2.5ms).
     * Standard EMA is 0.75*6.0 + 0.25*8.5 = 6.625ms (< 7.5ms Tier 1 threshold).
     * Predictive pacing projects EMA + 0.5*velocity = 6.625 + 1.25 = 7.875ms,
     * immediately shifting to Tier 1 and recording dropped frame prevented. */
    governor_tier_t t1 = dynamic_governor_update(&gov, 8.5);
    assert(t1 == GOV_TIER_1_GPU_FAST);
    assert(dynamic_governor_get_dropped_frames_prevented(&gov) == 1);

    /* 5. Test predictive velocity pacing at 120 FPS (deadline = 8.33ms, vel_thresh = 0.75ms) */
    dynamic_governor_reset(&gov);
    dynamic_governor_set_framerate(&gov, 120.0);
    assert(gov.dropped_frames_prevented == 0);
    governor_tier_t t120_0 = dynamic_governor_update(&gov, 3.0);
    assert(t120_0 == GOV_TIER_0_GPU_FULL);
    /* Jump: 3.0ms -> 4.2ms (velocity = +1.2ms > 0.75ms vel_thresh).
     * Standard EMA is 0.75*3.0 + 0.25*4.2 = 3.3ms (< 3.75ms Tier 1 threshold).
     * Predictive pacing projects EMA + 0.5*1.2 = 3.9ms (> 3.75ms),
     * immediately shifting to Tier 1 at 120 FPS. */
    governor_tier_t t120_1 = dynamic_governor_update(&gov, 4.2);
    assert(t120_1 == GOV_TIER_1_GPU_FAST);
    assert(dynamic_governor_get_dropped_frames_prevented(&gov) == 1);

    printf("  ✓ Governor v2 deadline scaling & predictive velocity pacing verified!\n");
}

static void test_governor_direct_tier0_spike_dwell(void)
{
    printf("[TEST] Testing that high latency spike from Tier 0 dwells in Tier 1 before entering CPU offload...\n");

    dynamic_governor_t gov;
    dynamic_governor_init(&gov);
    gov.cpu_offload_enabled = true;
    gov.min_dwell_frames = 6;

    /* Long-running Tier 0 baseline (100 frames) */
    for (int i = 0; i < 100; i++) {
        dynamic_governor_update(&gov, 3.5);
    }
    assert(dynamic_governor_get_tier(&gov) == GOV_TIER_0_GPU_FULL);
    assert(gov.frames_in_tier >= 100);

    /* Immediate severe latency spike (13.5ms > tier2 threshold 10.5ms).
     * Even though frames_in_tier was 100 in Tier 0, the governor MUST NOT
     * jump directly to Tier 2 CPU offload; it must enter Tier 1 GPU Fast ME first! */
    for (int i = 0; i < 4; i++) {
        dynamic_governor_update(&gov, 13.5);
    }
    /* Metric is now > 12ms, but dwell in Tier 1 is still < min_dwell_frames (4 < 6) */
    assert(dynamic_governor_get_tier(&gov) == GOV_TIER_1_GPU_FAST);

    /* After holding Tier 1 for min_dwell_frames under sustained pressure, it safely promotes to Tier 2 */
    for (int i = 0; i < 5; i++) {
        dynamic_governor_update(&gov, 13.5);
    }
    assert(dynamic_governor_get_tier(&gov) == GOV_TIER_2_CPU_OFFLOAD);

    printf("  ✓ Direct Tier 0 spike dwell gate verified!\n");
}

static void test_telemetry_subsystem(void)
{
    printf("[TEST] Testing shared memory telemetry subsystem...\n");

    int init_rc = bc250_telemetry_init();
    assert(init_rc == 0);
    assert(bc250_telemetry_get_path() != NULL);

    bc250_telemetry_data_t snapshot;
    memset(&snapshot, 0, sizeof(snapshot));
    snapshot.magic = BC250_TELEMETRY_MAGIC;
    snapshot.version = BC250_TELEMETRY_VERSION;
    snapshot.pid = 4242;
    strncpy(snapshot.process_name, "sunshine", sizeof(snapshot.process_name) - 1);
    snapshot.last_gpu_latency_ms = 5.25;
    snapshot.ema_gpu_latency_ms = 5.10;
    snapshot.frame_deadline_ms = 16.67;
    snapshot.is_async_compute = 1;
    snapshot.compute_queue_family = 1;
    snapshot.current_tier = (uint32_t)GOV_TIER_0_GPU_FULL;
    strncpy(snapshot.tier_name, "GPU Full ME", sizeof(snapshot.tier_name) - 1);
    snapshot.total_frames = 120;
    snapshot.tier0_frames = 120;
    snapshot.dropped_frames_prevented = 3;
    snapshot.target_fps = 60.0;
    snapshot.measured_fps = 59.95;
    snapshot.target_bitrate = 25000000;
    snapshot.current_qp = 22;
    snapshot.jitter_smoothing_active = 1;

    bc250_telemetry_update(&snapshot);

    bc250_telemetry_data_t read_back;
    memset(&read_back, 0, sizeof(read_back));
    int read_rc = bc250_telemetry_read(&read_back);
    assert(read_rc == 0);
    assert(read_back.magic == BC250_TELEMETRY_MAGIC);
    assert(read_back.version == BC250_TELEMETRY_VERSION);
    assert(read_back.pid == 4242);
    assert(strcmp(read_back.process_name, "sunshine") == 0);
    assert(read_back.is_async_compute == 1);
    assert(read_back.compute_queue_family == 1);
    assert(read_back.dropped_frames_prevented == 3);
    assert(read_back.current_qp == 22);
    assert(read_back.jitter_smoothing_active == 1);
    assert(fabs(read_back.last_gpu_latency_ms - 5.25) < 0.001);
    assert((read_back.sequence & 1) == 0); /* Even sequence indicates stable seqlock */

    /* Verify ASCII human-readable header in byte 0..255 */
    assert(strstr(read_back.text_summary, "[BC-250]") != NULL);
    assert(strstr(read_back.text_summary, "sunshine") != NULL);
    assert(strstr(read_back.text_summary, "Async Compute") != NULL);

    bc250_telemetry_close();
    printf("  ✓ Telemetry subsystem verified!\n");
}

int main(void)
{
    /* The governor is off unless a live streaming server runs it; these
     * tests are about what it does once on. */
    setenv("BC250_GOVERNOR_ENABLE", "1", 1);

    printf("========================================\n");
    printf("BC-250 Dynamic Governor Unit Test\n");
    printf("========================================\n");

    test_governor_transitions();
    test_governor_dwell_gate();
    test_governor_direct_tier0_spike_dwell();
    test_governor_gpu_only_mode();
    test_governor_failover_handled();
    test_governor_negative_latency();
    test_governor_telemetry_stats();
    test_governor_deadline_and_predictive_pacing();
    test_telemetry_subsystem();

    printf("\nALL DYNAMIC GOVERNOR TESTS PASSED!\n");
    return 0;
}
