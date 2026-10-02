# Release v0.5.2: HEVC Live Streaming Engine, libx265 Fallback, Zero-Copy DMA-BUF & HDR Tone-Mapping

Release **v0.5.2** is a monumental feature and performance milestone for the **AMD BC-250 (Cyan Skillfish)** APU. It merges the upstream real-time HEVC streaming engine from Mattia Tadini ([`MTSistemi`](https://github.com/MTSistemi)), delivering true 1080p60 gameplay streaming at half the CPU footprint, multithreaded surface readbacks, futex-based wavefront thread sleeping, a dedicated live streaming preset, still-picture bitrate surge protection, zero-GPU HEVC fallback via `libx265`, native Vulkan hardware zero-copy DMA-BUF memory importation, seamless network bitrate re-adaptation, Chromium/Firefox hardware decode hooks for VP9/AV1, and HDR10-to-SDR tone-mapping post-processing.

### 🚀 Key Highlights & Critical Improvements

#### 1. Real-Time 1080p60 HEVC Streaming Engine & Live Preset (PR #58)
* **Real-Time Real-Game 1080p60 Throughput**: When streaming at 1080p60 under active 3D game contention (*Black Myth: Wukong* benchmark, captured via `x11grab` and encoded via VA-API HEVC CBR 20 Mbit/s), throughput surges from **33 fps (0.55x) to a rock-solid 60 fps (1.00x)**.
* **Low-Overhead Live Preset (`BC250_HEVC_PRESET=live`)**: Automatically activated for recognized live streaming callers (`sunshine`, `steam`, `gamescope`, `wivrn`) or manually specified via `BC250_HEVC_PRESET=live`. Implements 4 optimal cycle-versus-compression trades:
  1. Rough bit costing for candidates instead of exhaustive CABAC estimation (-12% cycles, +0.7% bits).
  2. Elimination of intra candidate search in P-pictures (-25% cycles, +6.8% bits).
  3. Inter luma coded strictly with 8x8 transform units (-32% cycles, +8.9% bits).
  4. Half-pixel motion refinement instead of quarter-pixel (-42% cycles, +15.1% bits).
* **Environment Overrides**: Set `BC250_HEVC_PRESET=quality` to enforce the full CABAC, intra-in-P, 4x4 TU, quarter-pel quality preset, or `BC250_HEVC_THREADS` (or `BC250_MAX_CPU_THREADS`) to customize worker concurrency.

#### 2. Dedicated Worker Pool & Futex-Based Wavefront Sleeping (PR #58)
* **Futex Row Synchronization**: In [`approach1-compute-encoder/src/hevc_wpp.c`](file:///D:/Kai/bc250-encoding-decoding-fix/approach1-compute-encoder/src/hevc_wpp.c), wavefront parallel processing (WPP) rows waiting on the row above previously spun in an aggressive `sched_yield()` loop that kept Zen 2 cores fully pegged at 1176% CPU utilization. WPP rows now spin briefly and sleep on a Linux futex over row progress, waking only when row dependencies resolve.
* **OpenMP Elimination in Encoder Hotpath**: Sunshine and Steam do not configure `OMP_WAIT_POLICY=passive`, causing standard libgomp threads to continuously spin and rob CPU cycles from games. Replaced OpenMP in the HEVC pipeline with a dedicated, lightweight worker pool in [`approach1-compute-encoder/src/worker_pool.c`](file:///D:/Kai/bc250-encoding-decoding-fix/approach1-compute-encoder/src/worker_pool.c) and [`approach1-compute-encoder/src/worker_pool.h`](file:///D:/Kai/bc250-encoding-decoding-fix/approach1-compute-encoder/src/worker_pool.h).
* **50% CPU Footprint Reduction**: Total encoder CPU usage drops from **1176% down to 583%**, instantly releasing **5.1 CPU cores** back to the running game (up from 3.5 cores).

#### 3. Multithreaded Surface Readback & Elimination of Dead GPU Search (PR #57)
* **Direct CPU Motion Search**: Identified and removed dead GPU motion search dispatch in [`approach1-compute-encoder/src/encoder_h265.c`](file:///D:/Kai/bc250-encoding-decoding-fix/approach1-compute-encoder/src/encoder_h265.c). The GPU motion vectors were 16x16 macroblock vectors read with an incorrect 32x32 CTU stride against unreconstructed surfaces. Bypassing this wait eliminates up to 3.1 ms (and up to 16 ms of fence wait under heavy GPU load) per frame with 100% byte-identical bitstream parity (MD5 match).
* **8-Thread Parallel WC Surface Download**: Memory downloads from write-combining GART memory in [`approach1-compute-encoder/src/gpu_compute.c`](file:///D:/Kai/bc250-encoding-decoding-fix/approach1-compute-encoder/src/gpu_compute.c) via `gpu_compute_download_nv12()` were previously constrained to ~1.5 GB/s by single-core fill buffers. Splitting row downloads across 8 worker threads cuts 1080p copy latency from 2.17 ms down to **0.67 ms**.
* **Measured Throughput Gains**:
  * HEVC 1080p steady state: **43.3 fps → 55.6 fps** (+28.4%).
  * H.264 (x264 4-thread path): **74.1 fps → 91.5 fps** (+23.5%).

#### 4. Anti-Bitrate-Explosion Rate Control for Repeated/Still Pictures (PR #59)
* **Fast 30 µs Frame Duplicate Detection**: Added `same_as_last()` in [`approach1-compute-encoder/src/encoder_h265.c`](file:///D:/Kai/bc250-encoding-decoding-fix/approach1-compute-encoder/src/encoder_h265.c), comparing every 8th row of luma and chroma to detect repeated frames from frame rate drops (e.g. 30 fps games in 60 fps streams), paused menus, or loading screens.
* **Isolated Still-Picture Complexity Model**: Previously, repeated frames caused the rate control model to assume a scene cut, driving QP down to 12 and causing the next moving frame to explode to 920 kB - 1.2 MB (spiking a 20 Mbit/s stream to 34 - 68 Mbit/s). [`approach1-compute-encoder/src/rate_control.c`](file:///D:/Kai/bc250-encoding-decoding-fix/approach1-compute-encoder/src/rate_control.c) now tracks still pictures with a separate `cplx_still` model bounded within `RC_MODEL_STILL_RANGE = 6` steps of moving pictures.
* **Rock-Solid Bitrate Stability**: Under heavy frame repetition (1 in 6 frames moving), CBR streams remain strictly locked at **20.0 Mbit/s** (formerly 73.1 Mbit/s), and the peak moving frame size is capped at 246 kB (down from 920 kB).

#### 5. Zero-GPU HEVC CPU Fallback via `libx265`
* **100% 3D GPU Contention Isolation**: Following the H.264 `libx264` backend in v0.5.1, v0.5.2 introduces a dedicated `libx265` backend (`BC250_HEVC_BACKEND=x265` or `cpu`). It encodes HEVC streams entirely on Zen 2 CPU cores with 0% GPU load, leaving all 40 RDNA Compute Units dedicated to 3D games (*Cyberpunk 2077*, *Red Dead Redemption 2*, *Forza Horizon 5*).
* **Multi-Format Color Ingestion (8-bit NV12 & 10-bit P010)**: Supports standard 8-bit NV12 as well as 10-bit P010 HDR color spaces with automated internal planar de-interleaving (`i420_u` and `i420_v`) matching x265 pipeline expectations.
* **Low Latency & Live Streaming Tuning**: Automatically applies `tune="zerolatency"` (`bFrameAdaptive=0`, `bframes=0`, `lookaheadDepth=0`) for live streaming callers (`sunshine`, `steam`, `gamescope`), maintaining sub-4ms encode latency.
* **Command-Line & Environment Overrides**: Honors command-line `-preset` flags directly from `/proc/self/cmdline`, as well as `BC250_X265_PRESET` and `BC250_X265_THREADS` overrides.

#### 6. Native Hardware Zero-Copy DMA-BUF Ingestion
* **Vulkan External Memory Importation**: Implemented `gpu_compute_import_dmabuf_image()` utilizing `VK_EXT_external_memory_dma_buf` and `VkImportMemoryFdInfoKHR`. Binds DRM prime file descriptors directly into Vulkan image memory without copying frame buffers through host RAM.
* **Zero-Copy Gamescope & Sunshine Ingestion**: Directly ingests composited game frames exported by Gamescope and Sunshine, eliminating PCIe bandwidth contention and GART aperture bottlenecks.
* **Safe Fallback Protocol**: If the imported buffer utilizes an unsupported tiling modifier or memory type, `bc250_CreateSurfaces2()` cleanly returns `VA_STATUS_ERROR_UNSUPPORTED_MEMORY_TYPE`, directing Sunshine and Gamescope to seamlessly route frames through their validated EGL blit path without crashing.

#### 7. Dynamic Real-Time Network Bitrate Smoothing & x265 Reconfiguration
* **Jitter-Free Bitrate Scaling**: Implemented `rc_update_bitrate()` in the rate control subsystem. When Sunshine, Moonlight, or Steam Link dynamically adapt their target bitrate due to network congestion or Wi-Fi fluctuations, the driver proportionally scales buffer fullness and recomputes target frame budgets.
* **x265 Mid-Stream Rate Reconfiguration**: Added `x265_encoder_reconfig()` support to `encoder_x265.c`, applying runtime bitrate and QP updates immediately without dropping GOP cadence or reopening the encoder.
* **Elimination of Mid-Session QP Jumps**: Replaced disruptive rate control resets with continuous, smooth QP scaling, preventing packet spikes and momentary encoder stutter during runtime bitrate adjustments.

#### 8. Web Browser Hardware Decode Entrypoints (VP9 & AV1)
* **Chromium & Firefox Acceleration Hook**: Added `VAProfileVP9Profile0` and `VAProfileAV1Profile0` with `VAEntrypointVLD` to `bc250_QueryConfigProfiles()` and `bc250_QueryConfigEntrypoints()`.
* **Safe Software Fallback**: `bc250_CreateContext()` cleanly returns `VA_STATUS_ERROR_UNSUPPORTED_PROFILE` for VP9 and AV1, allowing Chromium, Firefox, and Electron apps to detect VA-API capability while seamlessly delegating VP9/AV1 decoding to built-in multithreaded `libvpx` and `dav1d` decoders, preventing blank or frozen video frames.

#### 9. HDR10 to SDR Tone-Mapping in Post-Processing (`VAEntrypointVideoProc`)
* **Vulkan Compute Tone-Mapper**: Created `video_proc_tonemap.comp` compute shader and integrated `vpp_pipeline_tonemap` into `VAEntrypointVideoProc`.
* **PQ EOTF Inversion & Reinhard Tone Curve**: Inverts SMPTE ST 2084 (PQ) non-linear electro-optical transfer functions and maps BT.2020 wide color gamut HDR10 surfaces down to standard BT.709 8-bit SDR range.
* **Vibrant Streaming on SDR Displays**: Eliminates washed-out, greyish visuals when capturing and streaming HDR games to standard SDR televisions, mobile phones, or laptops.

#### 10. Vectorized Fractional-Pel Motion Estimation (Half-Pel ME)
* **Sub-Pixel Motion Precision**: Extended `cpu_simd_me.c` with half-pel search refinement across 4 fractional candidate offsets (`(-0.5, 0)`, `(0.5, 0)`, `(0, -0.5)`, `(0, 0.5)`).
* **SSE2 Vector Acceleration**: Vectorized the 16x16 row interpolation and SAD evaluation using `_mm_avg_epu8()` and `_mm_sad_epu8()`, reducing 256 pixel operations per candidate down to 32 SIMD instructions for a ~10x speedup. Improved compression efficiency and slashed residual bitrate in fast-moving game scenes.

#### 11. Offline Transcoding Enhancements (2-Frame B-Frame GOP)
* **B-Frame Support**: Enabled 2 consecutive B-frames in `encoder_x265.c` when running non-live encoding (`live == false` or `BC250_BFRAMES=2`), drastically improving compression ratio for offline video archiving and batch transcoding with FFmpeg.

#### 12. Packaging & Tooling Modernization
* **Debian / Ubuntu / Arch / Fedora Packaging**: Added `libx265` build dependencies to `packaging/debian/control`, `packaging/arch/PKGBUILD`, and `packaging/fedora/bc250-vaapi.spec`.
* **GitHub Actions CI Coverage**: Updated `.github/workflows/build.yml` to install `libx265-dev` and build both 64-bit and 32-bit driver bundles with comprehensive test validation.
* **CTest Suite Expansion**: Added standalone `test_x265` and `test_hevc_encode` unit tests verifying single-frame zero-latency output, parameter sets (VPS/SPS/PPS), worker pool execution, and keyframe generation.
* **Diagnostic & Installer Updates**: Updated `tools/bc250_diagnose.sh`, `tools/setup_bazzite.sh`, `tools/setup_steamos.sh`, and `build_and_install.sh` to version v0.5.2.

### 📦 Installation & Bundled Assets

**Pre-built Release Bundles:**
* 64-bit Driver: `bc250-driver-linux-x86_64.tar.gz` (installs to `/usr/local/lib64/dri/` or `/usr/lib/dri/`)
* 32-bit Companion Driver: `bc250-driver-linux-i386.tar.gz` (installs to `/usr/lib32/dri/` for Steam Link)

**Quickstart Installation:**
```bash
tar -xzf bc250-driver-linux-x86_64.tar.gz
sudo ./install.sh
```

**Arch Linux / CachyOS:**
```bash
./tools/install_cachyos_arch.sh --with-32bit
```

**SteamOS / HoloISO:**
```bash
sudo ./tools/setup_steamos.sh
```

**Bazzite / Silverblue:**
```bash
sudo ./tools/setup_bazzite.sh
```
