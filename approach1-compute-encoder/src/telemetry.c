/* bc250-encoding-decoding-fix v0.5.4 - https://github.com/simpmix/bc250-encoding-decoding-fix */
/*
 * Copyright (c) 2026 BC-250 Project Contributors
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * telemetry.c - Real-time observability and shared-memory telemetry for AMD BC-250
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "telemetry.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <errno.h>

#if defined(__linux__) || defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
extern char *program_invocation_short_name;
#define HAVE_POSIX_SHM 1
#else
#define HAVE_POSIX_SHM 0
#endif

#include <stdatomic.h>

static char g_shm_path[256] = {0};
static int g_shm_fd = -1;
static bc250_telemetry_data_t *g_shm_ptr = NULL;
static bool g_initialized = false;

static void resolve_shm_path(void) {
    if (g_shm_path[0] != '\0') return;

    const char *env_path = getenv("BC250_STATS_PATH");
    if (env_path && *env_path) {
        strncpy(g_shm_path, env_path, sizeof(g_shm_path) - 1);
        return;
    }

#if HAVE_POSIX_SHM
    /* Check if /dev/shm exists and is a directory */
    struct stat st;
    if (stat("/dev/shm", &st) == 0 && S_ISDIR(st.st_mode)) {
        strncpy(g_shm_path, BC250_SHM_PATH_DEFAULT, sizeof(g_shm_path) - 1);
        return;
    }

    const char *xdg = getenv("XDG_RUNTIME_DIR");
    if (xdg && *xdg) {
        snprintf(g_shm_path, sizeof(g_shm_path), "%s/bc250_stats", xdg);
        return;
    }

    strncpy(g_shm_path, "/tmp/bc250_stats", sizeof(g_shm_path) - 1);
#else
    const char *tmp = getenv("TEMP");
    if (!tmp) tmp = getenv("TMP");
    if (tmp && *tmp) {
        snprintf(g_shm_path, sizeof(g_shm_path), "%s/bc250_stats", tmp);
    } else {
        strncpy(g_shm_path, "bc250_stats", sizeof(g_shm_path) - 1);
    }
#endif
}

const char *bc250_telemetry_get_path(void) {
    resolve_shm_path();
    return g_shm_path;
}

int bc250_telemetry_init(void) {
    if (g_initialized && g_shm_ptr) return 0;

    resolve_shm_path();

#if HAVE_POSIX_SHM
    /* Open or create the shared memory file with 0666 permissions so user tools can read it */
    g_shm_fd = open(g_shm_path, O_RDWR | O_CREAT | O_CLOEXEC, 0666);
    if (g_shm_fd < 0 && (errno == EACCES || errno == EPERM)) {
        /* Fallback to /tmp if /dev/shm permission was denied */
        const char *fallback = "/tmp/bc250_stats";
        strncpy(g_shm_path, fallback, sizeof(g_shm_path) - 1);
        g_shm_fd = open(g_shm_path, O_RDWR | O_CREAT | O_CLOEXEC, 0666);
    }
    if (g_shm_fd < 0 && (errno == EACCES || errno == EPERM)) {
        /* Multi-user fallback: user-specific /tmp/bc250_stats_<uid> */
        snprintf(g_shm_path, sizeof(g_shm_path), "/tmp/bc250_stats_%u", (unsigned int)getuid());
        g_shm_fd = open(g_shm_path, O_RDWR | O_CREAT | O_CLOEXEC, 0666);
    }

    if (g_shm_fd < 0) {
        return -1;
    }

    /* Explicitly ensure 0666 permissions so user-level monitoring tools can read it regardless of umask */
    fchmod(g_shm_fd, 0666);

    /* Size the file to fit the telemetry struct */
    size_t target_size = sizeof(bc250_telemetry_data_t);
    if (ftruncate(g_shm_fd, (off_t)target_size) != 0) {
        close(g_shm_fd);
        g_shm_fd = -1;
        return -1;
    }

    void *mapped = mmap(NULL, target_size, PROT_READ | PROT_WRITE, MAP_SHARED, g_shm_fd, 0);
    if (mapped == MAP_FAILED) {
        close(g_shm_fd);
        g_shm_fd = -1;
        return -1;
    }

    g_shm_ptr = (bc250_telemetry_data_t *)mapped;
#else
    static bc250_telemetry_data_t local_fallback;
    g_shm_ptr = &local_fallback;
#endif

    g_initialized = true;
    return 0;
}

void bc250_telemetry_update(const bc250_telemetry_data_t *snapshot) {
    if (!snapshot) return;

    if (!g_initialized || !g_shm_ptr) {
        if (bc250_telemetry_init() != 0) return;
    }

    /* Seqlock update: increment sequence to odd value before writing */
    uint32_t seq = atomic_load_explicit((_Atomic uint32_t *)&g_shm_ptr->sequence, memory_order_relaxed);
    seq = (seq + 1) | 1;
    atomic_store_explicit((_Atomic uint32_t *)&g_shm_ptr->sequence, seq, memory_order_release);
    atomic_thread_fence(memory_order_release);

    /* Format human-readable ASCII summary into the initial text header buffer */
    const char *pname = snapshot->process_name[0] ? snapshot->process_name : "bc250";
#if defined(__linux__)
    if (snapshot->process_name[0] == '\0' && program_invocation_short_name) {
        pname = program_invocation_short_name;
    }
#endif

    char summary_line[256];
    snprintf(summary_line, sizeof(summary_line),
             "[BC-250] %s (PID %u) | FPS: %.1f | GPU: %.2fms (EMA: %.2fms) | Tier: %u (%s) | %s | Dropped Prev: %llu\n",
             pname,
             snapshot->pid > 0 ? snapshot->pid : (uint32_t)getpid(),
             snapshot->measured_fps > 0.0 ? snapshot->measured_fps : snapshot->target_fps,
             snapshot->last_gpu_latency_ms,
             snapshot->ema_gpu_latency_ms,
             snapshot->current_tier,
             snapshot->tier_name[0] ? snapshot->tier_name : "Active",
             snapshot->is_async_compute ? "Async Compute" : "General Queue",
             (unsigned long long)snapshot->dropped_frames_prevented);

    /* Copy ASCII summary into byte offset 0..255 */
    memset(g_shm_ptr->text_summary, 0, sizeof(g_shm_ptr->text_summary));
    strncpy(g_shm_ptr->text_summary, summary_line, sizeof(g_shm_ptr->text_summary) - 1);

    /* Copy structured metrics */
    g_shm_ptr->magic = BC250_TELEMETRY_MAGIC;
    g_shm_ptr->version = BC250_TELEMETRY_VERSION;
    g_shm_ptr->struct_size = (uint32_t)sizeof(bc250_telemetry_data_t);
    g_shm_ptr->pid = snapshot->pid > 0 ? snapshot->pid : (uint32_t)getpid();
    strncpy(g_shm_ptr->process_name, pname, sizeof(g_shm_ptr->process_name) - 1);

    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        g_shm_ptr->timestamp_ns = (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
    } else {
        g_shm_ptr->timestamp_ns = snapshot->timestamp_ns;
    }

    g_shm_ptr->last_gpu_latency_ms = snapshot->last_gpu_latency_ms;
    g_shm_ptr->ema_gpu_latency_ms = snapshot->ema_gpu_latency_ms;
    g_shm_ptr->last_cpu_latency_ms = snapshot->last_cpu_latency_ms;
    g_shm_ptr->total_frame_latency_ms = snapshot->total_frame_latency_ms;
    g_shm_ptr->frame_deadline_ms = snapshot->frame_deadline_ms;

    g_shm_ptr->is_async_compute = snapshot->is_async_compute;
    g_shm_ptr->compute_queue_family = snapshot->compute_queue_family;
    g_shm_ptr->double_buffered_slot = snapshot->double_buffered_slot;

    g_shm_ptr->current_tier = snapshot->current_tier;
    strncpy(g_shm_ptr->tier_name, snapshot->tier_name, sizeof(g_shm_ptr->tier_name) - 1);
    g_shm_ptr->total_frames = snapshot->total_frames;
    g_shm_ptr->tier0_frames = snapshot->tier0_frames;
    g_shm_ptr->tier1_frames = snapshot->tier1_frames;
    g_shm_ptr->offload_frames = snapshot->offload_frames;
    g_shm_ptr->failover_frames = snapshot->failover_frames;
    g_shm_ptr->dropped_frames_prevented = snapshot->dropped_frames_prevented;

    g_shm_ptr->target_fps = snapshot->target_fps;
    g_shm_ptr->measured_fps = snapshot->measured_fps;
    g_shm_ptr->target_bitrate = snapshot->target_bitrate;
    g_shm_ptr->current_qp = snapshot->current_qp;
    g_shm_ptr->buffer_fullness = snapshot->buffer_fullness;
    g_shm_ptr->jitter_smoothing_active = snapshot->jitter_smoothing_active;

    /* End seqlock update: increment sequence to even value with release barrier */
    atomic_thread_fence(memory_order_release);
    atomic_store_explicit((_Atomic uint32_t *)&g_shm_ptr->sequence, seq + 1, memory_order_release);
}

int bc250_telemetry_read(bc250_telemetry_data_t *out_data) {
    if (!out_data) return -1;

    resolve_shm_path();

#if HAVE_POSIX_SHM
    int fd = open(g_shm_path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return -1;

    size_t sz = sizeof(bc250_telemetry_data_t);
    void *map = mmap(NULL, sz, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        close(fd);
        return -1;
    }

    const bc250_telemetry_data_t *shm = (const bc250_telemetry_data_t *)map;

    /* Seqlock read with retries in case writer is actively updating */
    for (int retry = 0; retry < 10; retry++) {
        uint32_t s1 = atomic_load_explicit((_Atomic uint32_t *)&shm->sequence, memory_order_acquire);
        if (s1 & 1) {
            /* Writer is in the middle of writing; brief pause */
            struct timespec pause_ts = { .tv_sec = 0, .tv_nsec = 100000 }; /* 0.1ms */
            nanosleep(&pause_ts, NULL);
            continue;
        }

        atomic_thread_fence(memory_order_acquire);
        memcpy(out_data, shm, sz);
        atomic_thread_fence(memory_order_acquire);

        uint32_t s2 = atomic_load_explicit((_Atomic uint32_t *)&shm->sequence, memory_order_acquire);
        if (s1 == s2 && !(s1 & 1)) {
            munmap(map, sz);
            close(fd);
            return (out_data->magic == BC250_TELEMETRY_MAGIC) ? 0 : -1;
        }
    }

    munmap(map, sz);
    close(fd);
    return -1;
#else
    if (g_shm_ptr) {
        memcpy(out_data, g_shm_ptr, sizeof(bc250_telemetry_data_t));
        return (out_data->magic == BC250_TELEMETRY_MAGIC) ? 0 : -1;
    }
    return -1;
#endif
}

void bc250_telemetry_close(void) {
#if HAVE_POSIX_SHM
    if (g_shm_ptr && g_shm_ptr != MAP_FAILED) {
        munmap(g_shm_ptr, sizeof(bc250_telemetry_data_t));
        g_shm_ptr = NULL;
    }
    if (g_shm_fd >= 0) {
        close(g_shm_fd);
        g_shm_fd = -1;
    }
#endif
    g_initialized = false;
}
