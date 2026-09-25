# BC-250 Sunshine & Moonlight Game Streaming Guide

## Overview

The AMD BC-250 mining card features an 8-core AMD Zen 2 CPU, 40 Compute Units (CUs, Oberon / Cyan Skillfish semi-custom architecture), and 16 GB of high-speed unified GDDR6 memory. However, the fixed-function Video Core Next (VCN) ASIC is disabled/fused off on these salvage dies.

The **bc250-encoding-decoding-fix** driver restores ultra-low-latency game streaming by combining Vulkan compute shaders with a **Dynamic 4-Tier Hybrid Load Governor** and **AVX2 SIMD CPU offloading**.

### Real-World Benchmark Results

In verified gaming benchmarks on the BC-250:
- **Baseline (No Streaming)**: ~5,410 points
- **Active Game Streaming (Sunshine + bc250 driver)**: ~5,165 points
- **Performance Impact**: **Only ~4.5% overhead!**

By dynamically shifting motion estimation between GPU CUs and Zen 2 CPU cores, 3D graphics rendering maintains full frame pacing without dropped stream frames.

---

## Quick Start Configuration

### 1. Automated Preset Application

You can apply the pre-tuned Sunshine streaming configuration with one command:
```bash
# Standard 60 FPS streaming (~8ms host latency)
./tools/sunshine_preset/apply_sunshine_preset.sh

# High-Refresh 120 FPS Ultra-Low-Latency streaming (~3-5ms host latency)
./tools/sunshine_preset/apply_sunshine_preset.sh --120fps
```

### 2. Manual Driver Installation & Environment Setup

Ensure the driver library is in your library path (e.g., `/usr/lib/x86_64-linux-gnu/dri/bc250_drv_video.so`).

To configure Sunshine to use the BC-250 driver, launch Sunshine with the following environment variables:

```bash
export LIBVA_DRIVER_NAME=bc250
export LIBVA_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri

# Recommended: Enable 4 slices per frame for 60fps (or 2 slices for 120fps high refresh)
export BC250_SLICES_PER_FRAME=4

# Recommended: Pin the 2 encoder worker threads to CPU cores 6 & 7 (leaving cores 0-5 100% free for 3D games)
export BC250_CPU_CORES=6,7

# Optional: Live telemetry logging in console/stderr every 60 frames (~1 sec)
export BC250_GOVERNOR_STATS=60
```

---

## Recommended Sunshine Web UI Settings

Open the Sunshine Web Configuration (`https://localhost:47990`) and set:

| Setting | Recommended Value | Why |
| :--- | :--- | :--- |
| **Encoder** | `VA-API` | Selects our high-performance Vulkan compute + SIMD driver. |
| **Video Codec** | `H.264` or `HEVC` | Both codecs are fully accelerated. H.264 has the lowest latency client decode. |
| **Bitrate** | `20 Mbps` (1080p60) / `35 Mbps` (1440p60) | Controlled by the low-latency Proportional-Integral rate controller. |
| **P-Frame Slices** | Auto (or driven via `BC250_SLICES_PER_FRAME=4`) | Allows Moonlight clients to decode macroblock slice stripes in parallel. |
| **Keyframe Interval (GOP)** | `60` (or `fps * 1s`) | Matches standard 1-second recovery keyframe interval. |
| **Frame Rate** | `60 fps` or `120 fps` | The governor dynamically tracks frame deadlines ($<16.6$ms for 60fps). |

---

## Dynamic Governor Tuning

The driver monitors Vulkan encode compute latency in real-time. When intense 3D scenes cause GPU contention, it shifts gears automatically:

```
+------------------------------------------------------------------------------------------------+
|  Tier 0: GPU Full ME   (< 14 ms in Sunshine / < 8 ms default) - Full diamond search on 40 CUs  |
|  Tier 1: GPU Fast ME   (>= 14 ms Sunshine / >= 8 ms default)  - Scaled search radius on GPU    |
|  Tier 2: CPU SIMD ME   (Optional) - Opt-in via BC250_ENABLE_CPU_ME=1                          |
|  Tier 3: Failover P-Skip (> 45 ms Sunshine / > 15.5 ms default) - Emergency bypass trip-wire   |
+------------------------------------------------------------------------------------------------+
```

> [!NOTE]
> **Sunshine Zero-Stutter Auto-Tuning (`v0.4.3`+)**: When Sunshine launches the encoder, the driver automatically auto-tunes the Tier 3 failover trip-wire from 15.5ms to **45.0ms** and grants 2 CPU worker threads for parallel slice entropy coding. This completely eliminates the `0.5 / 32 / 22 ms` frame latency oscillation caused by false-positive failover trips during complex frame encoding.

### Environment Variable Reference

| Variable | Values | Default | Purpose |
| :--- | :--- | :--- | :--- |
| `BC250_GOVERNOR_ENABLE` | `1` / `0` | `1` (Enabled) | Enable or disable dynamic CPU/GPU load balancing. |
| `BC250_GOVERNOR_TIER1_MS` | Float (`ms`) | `14.0` (Sunshine) / `8.0` | GPU latency threshold to transition into Tier 1 Fast ME. |
| `BC250_GOVERNOR_TIER2_MS` | Float (`ms`) | `22.0` (Sunshine) / `12.0` | GPU latency threshold to transition into Tier 2 CPU SIMD offload. |
| `BC250_GOVERNOR_TIER3_MS` | Float (`ms`) | `45.0` (Sunshine) / `15.5` | Emergency trip-wire threshold to emit failover P-Skip frame. |
| `BC250_ENABLE_CPU_ME` | `1` / `0` | `0` (Disabled) | Opt-in to Tier 2 CPU SIMD ME offload. Keep `0` (default) for smooth zero-stutter GPU ME. |
| `BC250_GOVERNOR_HYSTERESIS` | `1` to `30` | `4` | Number of stable frames required to step down tier (fast recovery from scene changes). |
| `BC250_GOVERNOR_STATS` | `1` or `N` | `0` (Disabled) | Print live telemetry stats every `N` frames to `stderr` (1 = every 60 frames). |
| `BC250_CPU_CORES` | `6,7` or `c1,c2` | Unpinned | Pin encoder worker threads to specific CPU cores. |
| `BC250_MAX_CPU_THREADS` | `1` to `8` | `2` (Sunshine) / `1` | Cap maximum OpenMP worker threads for slice entropy coding (protects host game headroom). |
| `BC250_DISABLE_OPENMP` | `1` / `0` | `0` | Force strictly single-threaded slice encoding without OpenMP thread pool overhead. |
| `BC250_SLICES_PER_FRAME` | `1` to `16` | `1` (or `4` recommended) | Divide frame into independent slices for multi-threaded decoding. |
| `BC250_FORCE_TIER` | `0`, `1`, `2`, `3` | `-1` (Auto) | Force a specific governor tier for benchmarking/debugging. |
| `BC250_HEVC_QP` | `1` to `51` | `27` | Base quantization parameter for HEVC encoder. |

---

## Live Telemetry Example

When `BC250_GOVERNOR_STATS=60` is set, Sunshine logs show the governor adapting in real-time:

```
[bc250-gov] Frame 60: Tier 0 (GPU Full ME) | GPU: 4.82 ms | EMA: 5.10 ms | Offload: 0 | Failover: 0
[bc250-gov] Frame 120: Tier 0 (GPU Full ME) | GPU: 5.15 ms | EMA: 5.08 ms | Offload: 0 | Failover: 0
[bc250-gov] Frame 180: Tier 1 (GPU Fast ME) | GPU: 14.42 ms | EMA: 14.11 ms | Offload: 0 | Failover: 0
[bc250-gov] Frame 240: Tier 0 (GPU Full ME) | GPU: 4.90 ms | EMA: 5.25 ms | Offload: 0 | Failover: 0
```

Notice that as GPU contention rises, the encoder scales search radius smoothly without dropping a single frame, and steps down smoothly with 4-frame hysteresis once GPU contention subsides.

---

## Troubleshooting

### 1. `vainfo` does not list BC-250
Run:
```bash
LIBVA_DRIVER_NAME=bc250 vainfo --display drm --device /dev/dri/renderD128
```
Ensure your user is part of the `video` and `render` groups:
```bash
sudo usermod -a -G video,render $USER
```

### 2. Sunshine reports `Could not open VA display`
Verify `/dev/dri/renderD128` exists and permissions allow read/write:
```bash
ls -l /dev/dri/render*
```
If multiple GPUs exist in the system, set `DRI_PRIME` or point Sunshine to the BC-250 render node.

### 3. Steam Link & Steam Remote Play Setup
Steam client runs as a 32-bit process on Linux and requires the companion 32-bit driver installed in the 32-bit DRI directory:
```bash
sudo install -Dm755 bc250_drv_video.so /usr/lib32/dri/bc250_drv_video.so
# Debian/Ubuntu multiarch:
sudo install -Dm755 bc250_drv_video.so /usr/lib/i386-linux-gnu/dri/bc250_drv_video.so
```
In `streaming_log.txt` (or Steam console output), verify that Steam loads `/usr/lib32/dri/bc250_drv_video.so`. The driver automatically enforces passive OpenMP thread waiting (`OMP_WAIT_POLICY=PASSIVE`, `GOMP_SPINCOUNT=0`) and caps slice worker threads to 2 (`BC250_MAX_CPU_THREADS=2`) to guarantee low host CPU usage.
