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
#include "dynamic_governor.h"

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

    /* 5. Next frame drops from Tier 3 back to Tier 2 */
    governor_tier_t post_emergency = dynamic_governor_update(&gov, 13.0);
    assert(post_emergency == GOV_TIER_2_CPU_OFFLOAD);

    /* 6. Hysteresis test: lower latency (4.0ms) requires 15 stable frames to drop back to Tier 0 */
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
     * notify_failover_handled must transition immediately down to Tier 2 CPU offload when enabled. */
    dynamic_governor_notify_failover_handled(&gov);
    assert(dynamic_governor_get_tier(&gov) == GOV_TIER_2_CPU_OFFLOAD);

    /* Calling it again while in Tier 2 should be a safe no-op */
    dynamic_governor_notify_failover_handled(&gov);
    assert(dynamic_governor_get_tier(&gov) == GOV_TIER_2_CPU_OFFLOAD);

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

int main(void)
{
    /* The governor is off unless a live streaming server runs it; these
     * tests are about what it does once on. */
    setenv("BC250_GOVERNOR_ENABLE", "1", 1);

    printf("========================================\n");
    printf("BC-250 Dynamic Governor Unit Test\n");
    printf("========================================\n");

    test_governor_transitions();
    test_governor_gpu_only_mode();
    test_governor_failover_handled();
    test_governor_negative_latency();
    test_governor_telemetry_stats();

    printf("\nALL DYNAMIC GOVERNOR TESTS PASSED!\n");
    return 0;
}
