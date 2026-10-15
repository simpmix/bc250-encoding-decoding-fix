# Testing & Verification Guide on BC-250 Hardware

This guide covers safety procedures, automated test runners, and real-world test pipelines for validating the BC-250 driver (`bc250_drv_video.so`) on Linux.

## 1. Safety Guidelines on Hardware

When working with development drivers or testing extreme workloads on the BC-250:

1. **Use a secondary machine via SSH:** If a GPU hang occurs, the desktop display will freeze. An SSH session allows you to monitor kernel messages (`dmesg -w`), capture backtraces, or reboot safely (`sudo reboot`).
2. **Monitor temperatures:** Monitor APU temperature and power with:
   ```bash
   watch -n 1 sensors
   # or
   amdgpu_top
   ```
3. **Module Loading:** When testing audio fix kernel updates, unload the previous version cleanly:
   ```bash
   sudo rmmod bc250_audio_fix
   sudo insmod bc250_audio_fix.ko
   ```
4. **Recovery:** If an unrecoverable hard lock occurs, hold the physical power button or use the kernel SysRq keys (`Alt + SysRq + R-E-I-S-U-B`) to trigger a clean disk sync and restart.

## 2. Automated Diagnostic Tool

Run the self-contained diagnostic script to verify device detection, active compute units, and run an automated encoding benchmark:

```bash
./tools/bc250_diagnose.sh
```

This script verifies:
* PCI Device ID matches Cyan Skillfish (`0x1002:0x13fe`).
* Vulkan compute queues and memory heaps.
* DKMS audio fix status (`bc250_audio_fix`).
* VA-API entrypoints and profiles with `vainfo`.
* Performs a 100-frame hardware encode test.

## 3. Unit Test Suite (11 Test Suites)

Run the internal CTest test suite covering bitstream writers, CABAC/CAVLC entropy, motion estimation kernels, dynamic governor, and VA-API entrypoints:

```bash
cd approach1-compute-encoder
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

All 11 test suites pass cleanly headless:
1. `BitstreamTest`
2. `CavlcUnitTest`
3. `VaApiDriverTest`
4. `EncodeBitstreamTest`
5. `HevcEncodeBitstreamTest`
6. `CpuSimdMeTest`
7. `DynamicGovernorTest`
8. `CabacRoundtripTest`
9. `McUnitTest`
10. `IntraPredUnitTest`
11. `HevcTransformUnitTest`

## 4. Official ITU JCT-VC HEVC Conformance Harness

The driver includes an automated test harness validating conformance against official ITU-T H.265 / ISO/IEC 23008-2 JCT-VC bitstreams (146 of 147 vectors passing bit-exact):

### Hardware VA-API Driver Conformance (146 of 147 Bit-Exact)
Tests the bitstream directly through the driver via `ffmpeg -hwaccel vaapi` against reference decoders on the BC-250 board:
```bash
./tools/test_vaapi_conformance.sh
```

### Standalone Decoder Direct Path Conformance (146 of 147 Bit-Exact)
Tests the standalone HEVC decoder directly:
```bash
./tools/test_hevc_conformance.sh
```

*(Note: `TSUNEQBD_A_MAIN10` specifies differing bit depths for luma and chroma, which is unsupported by FFmpeg).*

## 5. Real-World Encoding Tests with FFmpeg

### (a) High-Quality VBR Video Encoding (H.264 & HEVC)
```bash
# H.264 VBR (1080p60 @ 8 Mbps)
ffmpeg -vaapi_device /dev/dri/renderD128 -f lavfi -i testsrc=size=1920x1080:rate=60 \
  -vf 'format=nv12,hwupload' -c:v h264_vaapi -b:v 8M -frames:v 300 test_h264.mp4

# HEVC VBR (1080p60 @ 4 Mbps)
ffmpeg -vaapi_device /dev/dri/renderD128 -f lavfi -i testsrc=size=1920x1080:rate=60 \
  -vf 'format=nv12,hwupload' -c:v hevc_vaapi -b:v 4M -frames:v 300 test_hevc.mp4
```

### (b) Constant Bitrate (CBR) Streaming Test
Verify exact target bitrate conformance with filler NALs enabled:
```bash
ffmpeg -vaapi_device /dev/dri/renderD128 -f lavfi -i testsrc=size=1920x1080:rate=60 \
  -vf 'format=nv12,hwupload' -c:v hevc_vaapi -b:v 6M -maxrate 6M -minrate 6M -bufsize 6M -frames:v 300 test_cbr.mp4
```

### (c) Constant Quality (CQP) Archival Encoding
```bash
ffmpeg -vaapi_device /dev/dri/renderD128 -i input.mkv \
  -vf 'format=nv12,hwupload' -c:v hevc_vaapi -qp 18 -frames:v 300 output_cqp18.mp4
```

## 6. Real-World Playback & Decoding Tests

### (a) mpv Hardware Decoding Verification
Verify smooth playback without dropped frames or A/V desync:
```bash
LIBVA_DRIVER_NAME=bc250 mpv --hwdec=vaapi video.mp4
```

### (b) Testing mpv with Display Clock Resampling
Test audio/video sync stability with multi-threaded wavefront parallel processing active on frame 0:
```bash
LIBVA_DRIVER_NAME=bc250 mpv --hwdec=vaapi --video-sync=display-resample video.mp4
```

## 7. Game & VR Streaming Tests

### Sunshine Game Streaming
Apply the optimized Sunshine preset:
```bash
./tools/sunshine_preset/apply_sunshine_preset.sh
```
Verify that GPU encode latency is under 3–5 ms in Sunshine's Web UI statistics.

### WiVRn Wireless VR Streaming
Apply the WiVRn 2-hardware + 1-software stream topology:
```bash
./tools/wivrn_preset/apply_wivrn_preset.sh
```
Verify that motion-to-photon latency is ~36 ms and host CPU utilization remains under 350%.
