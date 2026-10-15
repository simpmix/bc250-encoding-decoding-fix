/* bc250-encoding-decoding-fix v0.5.2 - https://github.com/simpmix/bc250-encoding-decoding-fix */
/*
 * Copyright (c) 2026 BC-250 Project Contributors
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * telemetry.h - Real-time observability and shared-memory telemetry for AMD BC-250
 */

#ifndef BC250_TELEMETRY_H
#define BC250_TELEMETRY_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BC250_TELEMETRY_MAGIC 0x30353242 /* "B250" in little-endian */
#define BC250_TELEMETRY_VERSION 2

#define BC250_SHM_PATH_DEFAULT "/dev/shm/bc250_stats"

/*
 * Dual-format telemetry structure:
 * 1. Byte 0..255: Null-terminated human-readable ASCII summary string.
 *    Any tool or human running "cat /dev/shm/bc250_stats" or "head -n 1"
 *    instantly sees a live formatted telemetry line without binary noise.
 * 2. Binary fields from offset 256 onward:
 *    Structured metrics with atomic seqlock for lockless, zero-overhead
 *    consumption by monitoring overlays (MangoHud, SteamVR, Sunshine Web UI,
 *    and tools/bc250_stats).
 */
typedef struct {
    /* Human-readable ASCII header (offset 0..255) */
    char text_summary[256];

    /* Binary protocol identification */
    uint32_t magic;                 /* 0x30353242 ("B250") */
    uint32_t version;               /* BC250_TELEMETRY_VERSION = 2 */
    uint32_t struct_size;           /* sizeof(bc250_telemetry_data_t) */
    uint32_t pid;                   /* Process ID */
    char process_name[32];          /* "sunshine", "obs", "ffmpeg", etc. */

    /* Atomic sequence counter (odd = writer updating, even = stable) */
    uint32_t sequence;
    uint64_t timestamp_ns;          /* Monotonic timestamp in nanoseconds */

    /* Real-time encode latencies (milliseconds) */
    double last_gpu_latency_ms;
    double ema_gpu_latency_ms;
    double last_cpu_latency_ms;
    double total_frame_latency_ms;
    double frame_deadline_ms;       /* e.g. 16.67 for 60fps, 8.33 for 120fps */

    /* Hardware queue & pipelining */
    uint32_t is_async_compute;      /* 1 if dedicated ACE hardware async compute queue active */
    uint32_t compute_queue_family;
    uint32_t double_buffered_slot;  /* 0 or 1 */

    /* Governor v2 telemetry */
    uint32_t current_tier;          /* 0: GPU Full, 1: GPU Fast, 2: CPU Offload, 3: Failover */
    char tier_name[24];             /* "GPU Full ME", "GPU Fast ME", etc. */
    uint64_t total_frames;
    uint64_t tier0_frames;
    uint64_t tier1_frames;
    uint64_t offload_frames;
    uint64_t failover_frames;
    uint64_t dropped_frames_prevented;

    /* Rate control & Network metrics */
    double target_fps;
    double measured_fps;
    uint32_t target_bitrate;
    uint32_t current_qp;
    int64_t buffer_fullness;
    uint32_t jitter_smoothing_active;
} bc250_telemetry_data_t;

/**
 * Initializes telemetry shared-memory segment (/dev/shm/bc250_stats).
 * Safe to call multiple times (idempotent within the process).
 */
int bc250_telemetry_init(void);

/**
 * Updates telemetry shared-memory with latest frame metrics.
 * Uses atomic seqlock sequencing for lockless, non-blocking readers.
 */
void bc250_telemetry_update(const bc250_telemetry_data_t *snapshot);

/**
 * Reads telemetry data from shared memory.
 * Returns 0 on success, -1 on failure or uninitialized.
 */
int bc250_telemetry_read(bc250_telemetry_data_t *out_data);

/**
 * Closes telemetry shared memory mapping.
 */
void bc250_telemetry_close(void);

/**
 * Returns the active shared-memory file path.
 */
const char *bc250_telemetry_get_path(void);

#ifdef __cplusplus
}
#endif

#endif /* BC250_TELEMETRY_H */
