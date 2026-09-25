# AMD BC-250 VA-API Driver & Video Acceleration Suite (`bc250-encoding-decoding-fix`)

[![Build & Release BC-250 Drivers](https://github.com/simpmix/bc250-encoding-decoding-fix/actions/workflows/build.yml/badge.svg)](https://github.com/simpmix/bc250-encoding-decoding-fix/actions/workflows/build.yml)
[![GitHub Release](https://img.shields.io/github/v/release/simpmix/bc250-encoding-decoding-fix?color=blue&logo=github)](https://github.com/simpmix/bc250-encoding-decoding-fix/releases/latest)
[![Driver License: GPL-3.0](https://img.shields.io/badge/Driver%20License-GPL--3.0-blue.svg)](LICENSE)
[![Kernel Module: GPL-2.0](https://img.shields.io/badge/Audio%20Module-GPL--2.0-green.svg)](audio-fix/README.md)

A high-performance, spec-compliant VA-API driver (`bc250_drv_video.so`) engineered specifically for the **AMD BC-250 (Cyan Skillfish)** APU on Linux. 

The BC-250 is a repurposed PS5 APU (Zen 2 8-core/16-thread CPU, up to 40 unlocked RDNA 2 Compute Units) whose physical VCN (Video Core Next) hardware engine was permanently unprovisioned and eFused off at the factory. Without a working VCN block, Linux applications fail to initialize hardware video acceleration. 

This project solves this by delivering:
1. **GPU Compute Video Encoders**: Real-time H.264 and H.265/HEVC encoding executed across the APU's 40 RDNA 2 Compute Units using custom Vulkan compute shaders with asynchronous pipelining and AVX2 CPU SIMD offloading.
2. **Bit-Exact VA-API Video Decoders (`VAEntrypointVLD`)**: Threaded H.264 and HEVC decoding running on the Zen 2 CPU, verified bit-exact against reference decoders across all 302 conformance tests.
3. **Low-Latency Game & VR Streaming**: Pre-tuned presets and passive thread policies for Sunshine / Moonlight (1080p60/1440p) and WiVRn wireless VR streaming (~36ms motion-to-photon latency, ~190 Mbps throughput).
4. **Hardware Audio Clock Fix**: DKMS kernel module repairing the missing DisplayPort/HDMI audio clock.

---

## Performance & Conformance Highlights

### 1. Encoding Benchmarks (Measured on BC-250 Silicon)

* **H.264 (Vulkan Compute)**:
  * **640x480**: 267 fps
  * **720p**: 179 fps
  * **1080p**: 100–134 fps
  * **1440p**: 67–80 fps
  * **Game Streaming Overhead**: Only **~4.5%** total GPU impact during active 60 FPS gaming with Sunshine/Moonlight.
* **H.265 / HEVC (Multi-Slice Sliced Compute)**:
  * **1080p**: **111+ fps** (with default `BC250_HEVC_SLICES=4`, SIMD 4x4 transforms, and `MOVNTDQA` streaming readback).
  * **Dynamic Base QP Rate Control**: Full dynamic QP targeting (12..51) matches requested bitrates and eliminates blurry over-quantization.
  * **HEVC CBR Filler NALs**: Conforms bit-exact to target bitrates in CBR mode.
  * **Fine Detail Preservation**: Tuned skip decision threshold preserves delicate textures, hair, grain, and high-frequency motion.
  * **Chroma Fidelity**: Bit-exact non-linear Table 8-10 QP mapping eliminates the standard chroma PSNR deficit.
* **WiVRn VR Streaming**:
  * **Motion-to-Photon Latency**: **~36 ms** (down from 145 ms).
  * **Headset Download Throughput**: **~190 Mbits/s** (surpassing software encode).
  * **Host CPU Utilization**: **~350%** (slashed from 1300% lockup by enforcing passive OpenMP thread waiting).

### 2. Decoding Benchmarks & JCT-VC Conformance (`VAEntrypointVLD`)

Bit-exact conformance against official ITU JCT-VC test streams and reference decoders across both standalone execution and hardware VA-API on the BC-250:

| Codec | Resolution | Threads / Topology | Throughput (FPS) | Conformance Status |
| :--- | :--- | :--- | :--- | :--- |
| **H.264** | 1080p (CRF 23) | 1 thread | **68.4 fps** | 100% Bit-Exact (90/90 pass) |
| **H.264** | 1080p (CRF 23) | 8 threads (multi-slice) | **156.2 – 181.5 fps** | 100% Bit-Exact (90/90 pass) |
| **H.265 / HEVC** | 1080p | Multi-threaded (Wavefront + In-Flight) | **323+ fps** (3.3x speedup) | **146 of 147 Bit-Exact (99.3%)** |
| **H.265 / HEVC** | 4K (2160p) | Multi-threaded (Wavefront + In-Flight) | **164+ fps** (5.8x speedup) | **146 of 147 Bit-Exact (99.3%)** |

*(Note: The sole unmapped test vector, `TSUNEQBD_A_MAIN10`, specifies differing bit depths for luma and chroma, which FFmpeg itself does not support).*

---

## Supported Codec Matrix

| Profile | Entrypoint | Acceleration | Max Resolution |
| :--- | :--- | :--- | :--- |
| `VAProfileH264Baseline` | `VAEntrypointEncSlice` | libx264 (compute encoder as fallback) | 4096x2160 (4K) |
| `VAProfileH264Main` | `VAEntrypointEncSlice` | libx264 (compute encoder as fallback) | 4096x2160 (4K) |
| `VAProfileH264High` | `VAEntrypointEncSlice` | libx264 (compute encoder as fallback) | 4096x2160 (4K) |
| `VAProfileHEVCMain` | `VAEntrypointEncSlice` | Vulkan Compute ME + Host Slices | 4096x2160 (4K) |
| `VAProfileHEVCMain10` | `VAEntrypointEncSlice` | Host Slices, 10-Bit (P010 in) | 4096x2160 (4K) |
| `VAProfileH264*` | `VAEntrypointVLD` (Decode) | Multi-Threaded CPU Wavefront | 4096x2160 (4K) |
| `VAProfileHEVCMain` | `VAEntrypointVLD` (Decode) | Multi-Threaded CPU Wavefront (WPP) | **16384x16384 (16K)** |
| `VAProfileHEVCMain10` | `VAEntrypointVLD` (Decode) | 10-Bit CPU Wavefront (P010) | **16384x16384 (16K)** |
| `VAProfileNone` | `VAEntrypointVideoProc` | Vulkan Compute Scaler & Cropping | 4096x2160 (4K) |

---

## Quick Installation

### Option A: Automated Distribution Installers

We provide native package manifests with automatic Cyan Skillfish PCI (`0x1002:0x13fe`) device detection:

* **Arch Linux / CachyOS**:
  ```bash
  ./tools/install_cachyos_arch.sh
  ```
  *(Or use `cd packaging/arch && makepkg -si`)*
* **SteamOS / HoloISO**:
  ```bash
  sudo ./tools/setup_steamos.sh
  ```
* **Bazzite / Fedora / Silverblue**:
  ```bash
  sudo ./tools/setup_bazzite.sh
  ```
  *(Or build the RPM spec: `rpmbuild -ba packaging/fedora/bc250-vaapi.spec`)*
* **Debian / Ubuntu**:
  Package sources available under `packaging/debian/`.

### Option B: Pre-Compiled GitHub Release Tarballs

Download the latest release bundles directly from [GitHub Releases](https://github.com/simpmix/bc250-encoding-decoding-fix/releases/latest):
```bash
tar -xzf bc250-driver-linux-x86_64.tar.gz
sudo ./install.sh
```

---

## Streaming Presets & Tuning

### Sunshine / Moonlight (Game Streaming)
To configure Sunshine for zero-stutter 60/120 FPS game streaming with minimal GPU latency:
```bash
./tools/sunshine_preset/apply_sunshine_preset.sh
```
*See [`docs/sunshine-guide.md`](docs/sunshine-guide.md) for full details.*

### WiVRn (Wireless VR Streaming)
To configure WiVRn for ~36ms motion-to-photon latency and ~190 Mbps throughput:
```bash
./tools/wivrn_preset/apply_wivrn_preset.sh
```
*Recommended Topology*: 2 Hardware VA-API streams (`left_eye` and `right_eye`) + 1 Software stream (alpha/foveation). See [`docs/wivrn-guide.md`](docs/wivrn-guide.md).

---

## Key Environment Variables

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `LIBVA_DRIVER_NAME` | *(unset)* | Set to `bc250` to activate this driver. Handled automatically on BC-250 by systemd generator. |
| `BC250_FAST_MODE` | `1` | Restricts GPU compute overhead to <3–5%, preventing GPU starvation in heavy 3D games. |
| `BC250_SLICES_PER_FRAME` | `4` | Number of slices per H.264 frame. Use `2` for multi-stream VR to prevent CPU thread congestion. |
| `BC250_HEVC_SLICES` | `4` | Number of concurrent slices for HEVC encode (1..16). Yields 111+ fps at default 4. |
| `OMP_WAIT_POLICY` | `PASSIVE` | Critical: enforces passive wait in `libgomp`, cutting CPU usage from 1300% to ~350%. |
| `GOMP_SPINCOUNT` | `0` | Disables CPU busy-wait spin loops in worker threads. |
| `BC250_USE_CABAC` | `1` (Main/High) | Toggles CABAC (10–13% smaller bitrate) vs CAVLC for H.264 encode. |

---

## Verification & Diagnostics

1. Check VA-API profiles and entrypoints:
   ```bash
   LIBVA_DRIVER_NAME=bc250 vainfo
   ```
2. Run the automated hardware diagnostic suite:
   ```bash
   ./tools/bc250_diagnose.sh
   ```
3. Test encode and decode pipelines with FFmpeg:
   ```bash
   # Encode test (H.264 VA-API)
   ffmpeg -vaapi_device /dev/dri/renderD128 -f lavfi -i testsrc=size=1920x1080:rate=60 \
     -vf 'format=nv12,hwupload' -c:v h264_vaapi -b:v 15M -frames:v 300 test_enc.mp4

   # Decode test (Hardware VA-API decode)
   ffmpeg -hwaccel vaapi -vaapi_device /dev/dri/renderD128 -i test_enc.mp4 -f null -
   ```
4. Run official JCT-VC HEVC conformance test suite (146 of 147 bitstreams):
   ```bash
   # Test all 147 vectors directly through the driver via VA-API:
   ./tools/test_vaapi_conformance.sh

   # Test standalone HEVC decoder:
   ./tools/test_hevc_conformance.sh
   ```

---

## Building from Source

```bash
cd approach1-compute-encoder
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
sudo cmake --install build
```

---

## License & Credits

* **Driver & Shader Code**: Licensed under [GNU General Public License v3.0](LICENSE).
* **Audio Fix Kernel Module**: Licensed under [GNU General Public License v2.0](audio-fix/README.md).
* **Normative Tables**: Extracted mechanically from FFmpeg n8.1.2 under LGPL-2.1-or-later clause 3.
* **Special Thanks**:
  * **Mattia Tadini (@MTSistemi)** for authoring the complete bit-exact H.264 & HEVC decoders (`VAEntrypointVLD`), multi-slice HEVC, 16-bit Vulkan features, and distro packaging.
  * **Shalasere** for SPS crop research, Table 8-10 chroma QP mapping, and CABAC residual optimizations.
  * **Community Testers**: `oblique99`, `Cosmos`, `land_and_air`
