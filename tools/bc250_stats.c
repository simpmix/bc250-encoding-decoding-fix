/* bc250-encoding-decoding-fix v0.5.2 - https://github.com/simpmix/bc250-encoding-decoding-fix */
/*
 * Copyright (c) 2026 BC-250 Project Contributors
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * bc250_stats.c - Real-time telemetry monitor for AMD BC-250 VA-API driver
 *
 * Reads /dev/shm/bc250_stats (or $BC250_STATS_PATH) updated by the compute encoder.
 *
 * Usage:
 *   bc250_stats             # Print current live stats snapshot
 *   bc250_stats --watch     # Live updating stats dashboard (default 1s interval)
 *   bc250_stats --json      # Output structured JSON for monitoring scripts/overlays
 *   bc250_stats --raw       # Output raw ASCII header (same as cat /dev/shm/bc250_stats)
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include <time.h>

#if defined(__linux__) || defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#define HAVE_POSIX_SHM 1
#else
#include <windows.h>
#define HAVE_POSIX_SHM 0
#endif

#define BC250_TELEMETRY_MAGIC 0x30353242 /* "B250" */
#define BC250_TELEMETRY_VERSION 2
#define BC250_SHM_PATH_DEFAULT "/dev/shm/bc250_stats"

typedef struct {
    char text_summary[256];
    uint32_t magic;
    uint32_t version;
    uint32_t struct_size;
    uint32_t pid;
    char process_name[32];
    uint32_t sequence;
    uint64_t timestamp_ns;
    double last_gpu_latency_ms;
    double ema_gpu_latency_ms;
    double last_cpu_latency_ms;
    double total_frame_latency_ms;
    double frame_deadline_ms;
    uint32_t is_async_compute;
    uint32_t compute_queue_family;
    uint32_t double_buffered_slot;
    uint32_t current_tier;
    char tier_name[24];
    uint64_t total_frames;
    uint64_t tier0_frames;
    uint64_t tier1_frames;
    uint64_t offload_frames;
    uint64_t failover_frames;
    uint64_t dropped_frames_prevented;
    double target_fps;
    double measured_fps;
    uint32_t target_bitrate;
    uint32_t current_qp;
    int64_t buffer_fullness;
    uint32_t jitter_smoothing_active;
} bc250_telemetry_layout_t;

static const char *get_stats_path(void) {
    const char *env = getenv("BC250_STATS_PATH");
    if (env && *env) return env;
#if HAVE_POSIX_SHM
    struct stat st;
    if (stat("/dev/shm", &st) == 0 && S_ISDIR(st.st_mode)) {
        return BC250_SHM_PATH_DEFAULT;
    }
    const char *xdg = getenv("XDG_RUNTIME_DIR");
    if (xdg && *xdg) {
        static char xdg_path[256];
        snprintf(xdg_path, sizeof(xdg_path), "%s/bc250_stats", xdg);
        return xdg_path;
    }
    return "/tmp/bc250_stats";
#else
    const char *tmp = getenv("TEMP");
    if (!tmp) tmp = getenv("TMP");
    if (tmp && *tmp) {
        static char win_path[256];
        snprintf(win_path, sizeof(win_path), "%s/bc250_stats", tmp);
        return win_path;
    }
    return "bc250_stats";
#endif
}

static int read_telemetry_snapshot(const char *path, bc250_telemetry_layout_t *out) {
    if (!out) return -1;
#if HAVE_POSIX_SHM
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return -1;

    size_t sz = sizeof(bc250_telemetry_layout_t);
    void *map = mmap(NULL, sz, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        close(fd);
        return -1;
    }

    const bc250_telemetry_layout_t *shm = (const bc250_telemetry_layout_t *)map;
    for (int retry = 0; retry < 5; retry++) {
        uint32_t s1 = shm->sequence;
        if (s1 & 1) {
            struct timespec pause_ts = { .tv_sec = 0, .tv_nsec = 500000 };
            nanosleep(&pause_ts, NULL);
            continue;
        }

        memcpy(out, shm, sz);

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
        __asm__ __volatile__("" ::: "memory");
#endif

        uint32_t s2 = shm->sequence;
        if (s1 == s2) {
            munmap(map, sz);
            close(fd);
            return (out->magic == BC250_TELEMETRY_MAGIC) ? 0 : -1;
        }
    }

    munmap(map, sz);
    close(fd);
    return -1;
#else
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    size_t read_bytes = fread(out, 1, sizeof(*out), f);
    fclose(f);
    if (read_bytes < sizeof(*out) || out->magic != BC250_TELEMETRY_MAGIC) return -1;
    return 0;
#endif
}

static void print_formatted(const bc250_telemetry_layout_t *t) {
    printf("==================================================================\n");
    printf("   AMD BC-250 Cyan Skillfish Live Telemetry & Governor v2 Stats\n");
    printf("==================================================================\n");
    printf(" Application:       %s (PID: %u)\n", t->process_name[0] ? t->process_name : "active", t->pid);
    printf(" Framerate Target:  %.1f FPS (Measured: %.1f FPS)\n", t->target_fps, t->measured_fps);
    printf(" Frame Deadline:    %.2f ms (Budget for 0-drop streaming)\n", t->frame_deadline_ms);
    printf(" GPU Latency:       %.2f ms (EMA Contention: %.2f ms)\n", t->last_gpu_latency_ms, t->ema_gpu_latency_ms);
    printf(" Active Governor:   Tier %u [%s]\n", t->current_tier, t->tier_name[0] ? t->tier_name : "Active");
    printf(" Compute Engine:    %s (Queue Family %u, Slot %u)\n",
           t->is_async_compute ? "Dedicated Async Compute (ACE)" : "General Graphics/Compute Queue",
           t->compute_queue_family, t->double_buffered_slot);
    printf(" Dropped Prevented: %llu frames preserved by predictive pacing\n", (unsigned long long)t->dropped_frames_prevented);
    printf(" Total Encoded:     %llu frames (T0: %llu, T1: %llu, CPU: %llu, Skip: %llu)\n",
           (unsigned long long)t->total_frames,
           (unsigned long long)t->tier0_frames,
           (unsigned long long)t->tier1_frames,
           (unsigned long long)t->offload_frames,
           (unsigned long long)t->failover_frames);
    printf(" Rate Controller:   Bitrate %u kbps | QP %u | Buffer %lld bits\n",
           t->target_bitrate / 1000, t->current_qp, (long long)t->buffer_fullness);
    printf(" Jitter Smoothing:  %s\n", t->jitter_smoothing_active ? "Active (Anti-burst damping)" : "Disabled");
    printf("==================================================================\n");
}

static void print_json(const bc250_telemetry_layout_t *t) {
    printf("{\n");
    printf("  \"magic\": \"0x%08X\",\n", t->magic);
    printf("  \"version\": %u,\n", t->version);
    printf("  \"pid\": %u,\n", t->pid);
    printf("  \"process_name\": \"%s\",\n", t->process_name);
    printf("  \"sequence\": %u,\n", t->sequence);
    printf("  \"timestamp_ns\": %llu,\n", (unsigned long long)t->timestamp_ns);
    printf("  \"last_gpu_latency_ms\": %.3f,\n", t->last_gpu_latency_ms);
    printf("  \"ema_gpu_latency_ms\": %.3f,\n", t->ema_gpu_latency_ms);
    printf("  \"frame_deadline_ms\": %.3f,\n", t->frame_deadline_ms);
    printf("  \"is_async_compute\": %u,\n", t->is_async_compute);
    printf("  \"compute_queue_family\": %u,\n", t->compute_queue_family);
    printf("  \"double_buffered_slot\": %u,\n", t->double_buffered_slot);
    printf("  \"current_tier\": %u,\n", t->current_tier);
    printf("  \"tier_name\": \"%s\",\n", t->tier_name);
    printf("  \"total_frames\": %llu,\n", (unsigned long long)t->total_frames);
    printf("  \"tier0_frames\": %llu,\n", (unsigned long long)t->tier0_frames);
    printf("  \"tier1_frames\": %llu,\n", (unsigned long long)t->tier1_frames);
    printf("  \"offload_frames\": %llu,\n", (unsigned long long)t->offload_frames);
    printf("  \"failover_frames\": %llu,\n", (unsigned long long)t->failover_frames);
    printf("  \"dropped_frames_prevented\": %llu,\n", (unsigned long long)t->dropped_frames_prevented);
    printf("  \"target_fps\": %.2f,\n", t->target_fps);
    printf("  \"measured_fps\": %.2f,\n", t->measured_fps);
    printf("  \"target_bitrate\": %u,\n", t->target_bitrate);
    printf("  \"current_qp\": %u,\n", t->current_qp);
    printf("  \"buffer_fullness\": %lld,\n", (long long)t->buffer_fullness);
    printf("  \"jitter_smoothing_active\": %u\n", t->jitter_smoothing_active);
    printf("}\n");
}

int main(int argc, char **argv) {
    bool opt_watch = false;
    bool opt_json = false;
    bool opt_raw = false;
    double opt_interval = 1.0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-w") == 0 || strcmp(argv[i], "--watch") == 0) {
            opt_watch = true;
        } else if (strcmp(argv[i], "-j") == 0 || strcmp(argv[i], "--json") == 0) {
            opt_json = true;
        } else if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--raw") == 0) {
            opt_raw = true;
        } else if ((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--interval") == 0) && i + 1 < argc) {
            opt_interval = atof(argv[++i]);
            if (opt_interval < 0.1) opt_interval = 0.1;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: bc250_stats [options]\n");
            printf("Options:\n");
            printf("  -w, --watch            Continuously refresh display (live monitor)\n");
            printf("  -i, --interval <sec>   Watch refresh interval in seconds (default: 1.0)\n");
            printf("  -j, --json             Output metrics as structured JSON\n");
            printf("  -r, --raw              Output raw ASCII summary string\n");
            printf("  -h, --help             Show this help message\n");
            return 0;
        }
    }

    const char *shm_path = get_stats_path();

    do {
        bc250_telemetry_layout_t data;
        int rc = read_telemetry_snapshot(shm_path, &data);

        if (rc != 0) {
            if (opt_watch) {
                printf("\033[H\033[J");
            }
            fprintf(stderr, "[bc250_stats] No active BC-250 encode telemetry found at %s\n", shm_path);
            fprintf(stderr, "Note: Start Sunshine, Moonlight, OBS, or FFmpeg with bc250_drv_video to produce stats.\n");
            if (!opt_watch) return 1;
        } else {
            if (opt_raw) {
                printf("%s", data.text_summary);
            } else if (opt_json) {
                print_json(&data);
            } else {
                if (opt_watch) {
                    printf("\033[H\033[J");
                }
                print_formatted(&data);
            }
        }

        if (opt_watch) {
#if HAVE_POSIX_SHM
            struct timespec req = {
                .tv_sec = (time_t)opt_interval,
                .tv_nsec = (long)((opt_interval - (time_t)opt_interval) * 1e9)
            };
            nanosleep(&req, NULL);
#else
            Sleep((DWORD)(opt_interval * 1000.0));
#endif
        }
    } while (opt_watch);

    return 0;
}
