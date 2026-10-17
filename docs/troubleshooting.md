# AMD BC-250 Driver Troubleshooting & Diagnostic Guide

This guide covers solutions to all known issues when using the AMD BC-250 (Cyan Skillfish) custom VA-API compute driver and audio fix on Linux distributions (Bazzite, Fedora, Ubuntu, Arch, CachyOS, ChimeraOS).

---

## Quick Diagnostic Check

Before troubleshooting individual issues, run the built-in diagnostic tool from your terminal:

```bash
cd /path/to/bc250-encoding-decoding-fix
./tools/bc250_diagnose.sh
```

This automated script checks your APU identification, active Compute Units (CUs), audio driver status, VA-API profile queries, and runs a live 100-frame encode benchmark.

---

## 1. DisplayPort / HDMI Audio Issues ("Drunk" or Stuttering Audio)

### Symptoms
* Audio over DisplayPort or HDMI sounds distorted, robotic, pitched down, or stutters ("drunk audio").
* Audio drops out completely when switching resolutions.

### Cause
The Cyan Skillfish display controller uses a non-standard clock divisor in the `amdgpu` display engine (`dc`), calculating the wrong sample clock for standard 48kHz audio.

### Solution
Install the DKMS audio fix module, which dynamically applies the correct clock configuration:

```bash
cd audio-fix
sudo ./install_dkms.sh
```

**Verify the fix is active:**
```bash
lsmod | grep bc250_audio_fix
```
If active, you will see `bc250_audio_fix` listed. Because this is registered with **DKMS**, it will automatically rebuild and persist whenever you update your Linux kernel!

---

## 2. "vaInitialize failed: driver bc250 not found"

### Symptoms
* Running `vainfo` or launching OBS reports:
  ```
  vaInitialize failed with error code -1 (unknown libva error),exit
  ```
  or
  ```
  vaGetDriverNameByIndex() failed with unknown libva error, driver_name = (null)
  ```

### Cause
The VA-API loader (`libva`) looks for driver libraries in specific DRI directories depending on your Linux distribution (`/usr/lib64/dri/`, `/usr/lib/x86_64-linux-gnu/dri/`, or `/usr/lib/dri/`). If `bc250_drv_video.so` is missing from the directory expected by your distro, it will fail to load.

### Solution
1. Ensure `LIBVA_DRIVER_NAME=bc250` is set in your environment:
   ```bash
   export LIBVA_DRIVER_NAME=bc250
   ```
2. Re-link the driver into all common system DRI paths:
   ```bash
   sudo mkdir -p /usr/lib/x86_64-linux-gnu/dri /usr/lib64/dri /usr/lib/dri
   sudo cp -f approach1-compute-encoder/build/bc250_drv_video.so /usr/lib/x86_64-linux-gnu/dri/
   sudo cp -f approach1-compute-encoder/build/bc250_drv_video.so /usr/lib64/dri/
   sudo cp -f approach1-compute-encoder/build/bc250_drv_video.so /usr/lib/dri/
   ```
3. Test again:
   ```bash
   LIBVA_DRIVER_NAME=bc250 vainfo
   ```

---

## 3. Permission Denied on `/dev/dri/renderD128`

### Symptoms
* `vainfo` or FFmpeg fails with:
  ```
  Failed to open /dev/dri/renderD128: Permission denied
  ```
* Sunshine reports encoder initialization error when launched as a non-root user.

### Cause
Your user account does not have read/write access to the GPU render node.

### Solution
Add your user account to the `video` and `render` groups:
```bash
sudo usermod -a -G video,render $USER
```
Then log out and log back in (or reboot) for the permissions to take effect. Verify with:
```bash
groups
```
Ensure `render` is listed in the output.

---

## 4. Sunshine / Moonlight Streaming Issues

### Symptom A: Moonlight shows black screen or immediate disconnect
* **Solution 1:** In the Sunshine Web UI (**Configuration -> Audio/Video**):
  * Set **Video Encoder** to `VA-API`.
  * Ensure **Resolution** matches a standard 16:9 ratio (1280x720, 1920x1080, or 2560x1440).
* **Solution 2:** Enable the low-overhead gaming mode in your environment:
  ```bash
  export BC250_FAST_MODE=1
  ```
  This bypasses the deblock filter and uses 2:1 subsampled motion estimation, keeping frame latency below 5ms!

### Symptom B: Stream drops frames when game graphics are demanding
* The BC-250 driver automatically prioritizes the GPU's dedicated **Async Compute Engine (ACE)** queues so encoding does not stall graphics rendering.
* Ensure you are running in fast mode:
  ```bash
  export BC250_FAST_MODE=1
  ```
  This keeps encoder GPU load under **3–5% of the 40 CUs**, leaving the rest of the APU for the game.

---

## 5. OBS Studio: "Failed to open video codec"

### Symptoms
* Clicking **Start Recording** or **Start Streaming** in OBS results in:
  ```
  Starting the output failed. Please check the log for details.
  Error: Failed to open video codec: Function not implemented (-40)
  ```

### Solution
1. Launch OBS from the terminal with the driver specified:
   ```bash
   LIBVA_DRIVER_NAME=bc250 obs
   ```
2. In OBS:
   * Go to **Settings -> Output -> Output Mode: Advanced**.
   * Under the **Streaming** or **Recording** tab, select **FFmpeg VAAPI**.
   * Set **VAAPI Device** to `/dev/dri/renderD128`.
   * Under **Profile**, select **Main** or **High**.

---

## 6. Verifying 40 Compute Units (CUs) vs 24 CUs

### Check Status
Run:
```bash
cat /sys/class/drm/card0/device/current_compute_units 2>/dev/null || dmesg | grep -i "compute units"
```
* **Expected:** 40 active CUs.
* **If it reports 24 CUs:** Your kernel or BIOS is limiting the APU to its stock crypto-mining default (24 CUs / 12 WGPs). This project does *not* bundle a kernel unlock patch because modifying compute unit allocation requires an `amdgpu` kernel driver patch. To unlock all 40 CUs (20 WGPs):
  1. Install an APU-optimized distribution like **Bazzite** (BC-250 / Deck image) or **SkillFishOS**, which pre-integrates the 40 CU unlock out of the box.
  2. Or apply the community kernel patch from **[duggasco/bc250-40cu-unlock](https://github.com/duggasco/bc250-40cu-unlock)** using module parameter `amdgpu.bc250_cc_write_mode=3`.
  * Note: Once unlocked, this driver's compute shaders automatically dispatch across all 40 CUs with <3–5% overhead!

---

## 7. SteamOS & Bazzite (Immutable Distribution) Considerations

### SteamOS / HoloISO
* **Why `/var/lib/bc250`?** SteamOS uses an A/B partition layout where the root partition (`/` and `/usr`) is completely overwritten during system upgrades (e.g. SteamOS 3.5 → 3.6). The `/var` and `/etc` partitions are persistent state partitions. By installing the driver to `/var/lib/bc250/dri` and configuring `/etc/environment.d/99-bc250.conf`, your driver survives all future OS updates.
* **Kernel headers for Audio Fix:** If `setup_steamos.sh` notes missing kernel headers for DKMS, run:
  ```bash
  sudo steamos-readonly disable
  sudo pacman -S --needed linux-neptune-headers dkms
  sudo ./tools/setup_steamos.sh
  ```

### Bazzite / Fedora Silverblue
* **SELinux context denials:** On Fedora Silverblue and Bazzite, SELinux is set to Enforcing. If Gamescope or Sunshine fails to load the driver from `/usr/local/lib64/dri/`, restore SELinux file contexts:
  ```bash
  sudo restorecon -Rv /usr/local/lib64/dri /usr/local/share/bc250
  ```
* **Kernel headers on Bazzite:** If DKMS fails to compile the audio fix:
  ```bash
  ujust install-kernel-headers
  sudo ./tools/setup_bazzite.sh
  ```

---

## 8. FFmpeg Warning: "Driver does not support some wanted packed headers" / Container Global Headers

### Symptoms
When encoding with FFmpeg's `h264_vaapi` or `hevc_vaapi` to MP4 containers, the console previously printed:
```text
[h264_vaapi @ 0x560ddcc95dc0] Driver does not support some wanted packed headers (wanted 0xd, found 0).
[h264_vaapi @ 0x560ddcc95dc0] Driver does not support packed sequence headers, but a global header is requested.
[h264_vaapi @ 0x560ddcc95dc0] No global header will be written: this may result in a stream which is not usable for some purposes (e.g. not muxable to some containers).
```

### Resolution in v0.5.0+
* The driver advertises `VA_ENC_PACKED_HEADER_SEQUENCE` (`0x1`), allowing FFmpeg's MP4/MKV container muxers to cleanly extract sequence parameters for container global headers (`avcC` / `hvcC` atom) without warnings.
* Full in-band AUD, SPS, PPS, and Slice NAL units are maintained in the bitstream for standard streaming and player compatibility.

### Default Rate Control & File Size Efficiency
* When running standard FFmpeg commands without explicit bitrate constraints (`ffmpeg -i input.avi -c:v h264_vaapi out.mp4`), the driver advertises `VA_RC_VBR | VA_RC_CBR` so FFmpeg automatically selects **VBR** with standard resolution/framerate-derived target bitrates (~4 Mbps on 1080p24 H.264, ~2.2 Mbps on HEVC), matching Intel iGPU and software encoder defaults (~280 MiB H.264 / ~157 MiB HEVC on Big Buck Bunny 1080p).
* Constant QP (CQP) mode can be explicitly selected if desired via `-rc_mode CQP` or `BC250_ENABLE_CQP=1`. Target QP can be customized via `-qp <val>` or `BC250_CQP=<val>`.

---

## 9. 32-bit VA-API Clients (Steam Link)

### Symptoms
* Steam Link silently falls back to software encoding or fails to initialize hardware acceleration.
* Log reports that `bc250_drv_video.so` cannot be loaded or is the wrong ELF class.

### Cause
* Steam Link's client runtime is a 32-bit process and cannot load 64-bit `.so` drivers from `/usr/lib64/dri` or `/usr/lib/x86_64-linux-gnu/dri`.

### Solution
1. Download the companion 32-bit driver archive `bc250-driver-linux-i386.tar.gz` from Releases.
2. Install it to your system's 32-bit DRI directory (leaving the 64-bit driver in place for 64-bit apps):
   ```bash
   tar -xzvf bc250-driver-linux-i386.tar.gz
   sudo install -Dm755 bc250-driver-i386/bc250_drv_video.so /usr/lib32/dri/bc250_drv_video.so
   # Or on Debian/Ubuntu multiarch:
   # sudo install -Dm755 bc250-driver-i386/bc250_drv_video.so /usr/lib/i386-linux-gnu/dri/bc250_drv_video.so
   ```
3. Shaders are shared at `/usr/share/bc250/shaders` from the 64-bit install — no extra shaders needed.

---

## 10. How to Completely Uninstall the Driver

Use the automated uninstaller script:

```bash
# Preview what will be removed without changing anything:
sudo ./tools/bc250_uninstall.sh --dry-run

# Perform full cleanup:
sudo ./tools/bc250_uninstall.sh

# Remove the DKMS audio fix:
cd audio-fix && sudo ./uninstall_dkms.sh
```

---

## 11. HEVC Hardware Decoding Artifacts / Ghosting in mpv or FFmpeg

### Symptoms
When playing HEVC/H.265 files (such as *Big Buck Bunny* or MP4/MKV video streams) in `mpv --hwdec=vaapi` or FFmpeg (`-hwaccel vaapi`), video playback previously exhibited:
* Severe macroblock displacements, blocky pixelation, or cross-frame smearing.
* Previous scenes or title cards "burned into" subsequent frames as ghost images.
* Audio/video desync and dropped frames reported by player logs.

### Root Cause
1. **Container SPS RPS vs Slice Headers**: In standard HEVC MP4/MKV containers, sequence parameters and the short-term Reference Picture Set (RPS) table are stored globally in container headers (`hvcC` atom) rather than repeated in-band. In slice headers, `short_term_ref_pic_set_sps_flag` is set to `1` to reference SPS tables.
2. **Missing In-Band Sets**: When VA-API is invoked, the application does not transmit the raw SPS RPS table. When the driver attempted to re-parse the slice header from raw bits without the SPS table, it failed to parse the RPS and could not synchronize the bitstream reader, leading to dropped slices and zero populated reference frames (`d->n_refs[0] = 0, d->n_refs[1] = 0`). As a result, motion compensation was skipped, leaving older pixels ghosting across the display.

### Resolution in v0.5.0+
* **Direct VA-API RPL Derivation**: `va_decode_hevc.c` now derives reference picture lists (L0 and L1) and the temporal collocated picture directly from VA-API's pre-resolved `VASliceParameterBufferHEVC.RefPicList[2][15]`, eliminating the fragile dependency on in-band RPS bitstream re-parsing.
* **Exact Slice Parameter Mapping**: Slice type, QP deltas, SAO, deblocking, and slice data byte offsets are mapped directly from VA-API buffers.
* **Fault-Tolerant Concealment**: Added DPB closest-POC concealment fallback to smoothly hide any missing reference frames caused by seek operations or network packet drops.

---

## 12. Video Quality Loss, Texture Smearing, or Incorrect File Size (Encoder)

### Symptoms
* High-bitrate encodes (e.g. 15–20 Mbps) appear soft, blurry, or washed out compared to equivalent encodes from Intel/AMD hardware encoders.
* Fine textures, grain, hair, or subtle camera motion get smoothed away into flat blocks.
* Constant Bitrate (CBR) encodes significantly undershoot requested file size and target bitrate.

### Root Cause
1. **FFmpeg `pic_init_qp = 26` Stomping**: FFmpeg fills `pic_init_qp = 26` in `VAEncPictureParameterBuffer` on every picture. The driver was unconditionally resetting the rate controller's `base_qp` to 26 on every frame, wiping out the bitrate-derived QP estimate (`rc_estimate_base_qp`). As a result, the encoder remained pinned at QP 26–27 even when the user requested high bitrates.
2. **Aggressive CU Skip Threshold**: In HEVC, the decision threshold for skipping residual coding in an 8x8 block was set to `128 * (2 + qp/4)` (~1024), treating an average per-pixel difference of up to 10.6 as identical background and skipping residual transforms.
3. **Missing CBR Filler NALs**: HEVC lacked filler NAL padding, causing low-complexity scenes to output fewer bits than requested without reaching target bitrates.

### Resolution in v0.5.0+
* **Guarded `pic_init_qp`**: `pic_init_qp` is now only applied when explicitly encoding in Constant QP (CQP) mode (`-qp` or `BC250_CQP`). In VBR, CBR, and low-latency streaming modes, the rate controller has full authority to modulate QP between 12 and 51 based on target bitrate and frame complexity.
* **Tuned CU Skip Threshold**: Skip thresholds are scaled based on quality levels: 48–64 for quality presets (`quality_level <= 4`), ensuring fine skin texture, hair, and film grain are preserved with full residual transforms.
* **HEVC CBR Filler NALs**: Implemented `maybe_append_filler_hevc` so CBR streams accurately maintain constant target bitrates and file sizes.

---

## 13. Startup Stutter or A/V Desync Warning in mpv (`--video-sync=display-resample`)

### Symptoms
* Playing HEVC media in mpv with `--video-sync=display-resample` triggers noticeable stutter or frame drops near the beginning of playback:
  ```text
  [ao/pulse] The audio device is reporting an inaccurate playback position.
  [playback] A/V desync: audio ahead of video, dropping video frames
  ```
* Without `--video-sync=display-resample`, video playback is smooth but an initial A/V desync warning might appear in terminal logs.

### Root Cause
1. **Container RPS Index Bit Misalignment**: In container-muxed files where `sps->num_st_rps == 0` in VA-API, re-reading the slice header previously failed to locate slice entry points. Without entry points, Wavefront Parallel Processing (WPP) multi-threading could not launch, causing the decoder to fall back to single-threaded CPU execution (~40 ms per frame).
2. **Audio/Video Display Lock**: When `--video-sync=display-resample` is enabled, mpv forces video presentation to lockstep to the system audio clock. Because the first few frames took 40 ms each (exceeding the 16.6 ms 60fps frame budget), mpv dropped 20–30 frames in rapid succession to catch up to the PipeWire/PulseAudio clock.

### Resolution in v0.5.0+
* **Exact Bitstream Stepping via `st_rps_bits`**: `va_decode_hevc.c` now passes `num_short_term_ref_pic_sets` and `st_rps_bits` directly from VA-API, ensuring slice entry points and byte alignment are parsed bit-exact on frame 0.
* **Multi-Threaded WPP on Frame Start**: Multi-threaded wavefront parallel processing launches immediately from the very first frame, keeping frame decode times well under 10 ms.
* **Persistent Buffers & SSE2 Vectorization**: Eliminated 1 MB per-frame dynamic allocations and accelerated UV plane interleaving using SSE2 SIMD unpack instructions (`_mm_unpacklo_epi8` / `_mm_unpackhi_epi8`).

---

## 14. Official 147 JCT-VC HEVC Conformance Vectors & Long-Term Reference Frames

### Enhancements in v0.5.0+
* **146 of 147 Vectors Passing**: JCT-VC conformance increased from 111/147 to **146/147 (99.3%)** bit-exact across both standalone testing and hardware VA-API on the BC-250 (`tools/test_vaapi_conformance.sh`).
* **Long-Term References & List Modification**: Implemented normative ITU-T H.265 sections 7.3.6.1 and 7.3.6.2. Motion vectors referencing long-term frames (`VA_PICTURE_HEVC_LONG_TERM_REFERENCE`) are correctly protected from scaling.
* **Quantization Matrices**: Full scaling list support from `VAIQMatrixBufferHEVC` and Table 7-6 defaults.
* **PCM Coding Units**: Supported and decoded bit-exact.
* **16K Decode Resolution**: HEVC decoding supports up to 16384x16384 on a side.

---

## 15. Gaming Mode (Gamescope Session) Black or Green Screen (Sunshine & Steam Link)

### Symptoms
* Connecting via Steam Link or Moonlight while the BC-250 console is in **Gaming Mode** (Gamescope compositor on Bazzite, SteamOS, or CachyOS) results in working game audio but a **solid bright green screen** or **blank black screen**.
* The exact same games stream with perfect video when running in Desktop Mode (KDE Plasma / GNOME).

### Root Causes
1. **Unimplemented DMA-BUF Import in VA-API**: In Gaming Mode, Gamescope passes composited video frames as external DMA-BUF handles (`DRM_PRIME_2`) directly into `vaCreateSurfaces2()`. In earlier driver versions, `bc250_CreateSurfaces2()` ignored `attrib_list` and silently returned `VA_STATUS_SUCCESS` with an empty Vulkan surface. Believing zero-copy import succeeded, the caller skipped its EGL/GL blit and encoded the blank buffer. In YUV (NV12), an all-zeros buffer decodes to $RGB(0, 135, 0)$—a solid bright green screen.
2. **KMS File Capabilities & `AT_SECURE` Environment Stripping**: Sunshine inside Gamescope cannot use portal or Wayland capture and must use direct KMS capture (`capture = kms`). Granting `cap_sys_admin` puts Linux into secure-execution mode (`AT_SECURE`), which strips user environment variables like `LIBVA_DRIVER_NAME=bc250`, causing `libva` to fall back to the disabled `radeonsi` driver.
3. **Incorrect Display Card Node (`card0` vs `card1`)**: On Cyan Skillfish boards, the active display connector (`DP-1`) is often driven by `/dev/dri/card1` rather than `/dev/dri/card0`. If Sunshine targets `card0`, KMS captures an unattached blank connector.

### Resolution Steps
1. **Update to Driver v0.5.1 or Newer**: Driver v0.5.1 updates `bc250_CreateSurfaces2()` to explicitly reject external DMA-BUF imports with `VA_STATUS_ERROR_UNSUPPORTED_MEMORY_TYPE`, forcing Sunshine onto its working EGL blit pathway (`vaExportSurfaceHandle`) and Steam Link onto its frame copy path.
2. **Grant Sunshine KMS Permissions on Canonical Binary**:
   On Arch / CachyOS / Bazzite, `/usr/bin/sunshine` is often a wrapper script or symlink. Apply capabilities directly to the real binary:
   ```bash
   sudo setcap cap_sys_admin,cap_sys_nice+p $(readlink -f $(which sunshine))
   ```
3. **If Running Sunshine via systemd User Service**:
   Add ambient capabilities under `[Service]` in `~/.config/systemd/user/sunshine.service`:
   ```ini
   AmbientCapabilities=CAP_SYS_ADMIN CAP_SYS_NICE
   ```
   Then reload and restart the service:
   ```bash
   systemctl --user daemon-reload
   systemctl --user restart sunshine
   ```
4. **Enable "Force Composite" in Gamescope**:
   When Gamescope uses direct scanout, secondary KMS plane lookups fail with permission errors. In Steam Game Mode, go to **Settings -> System -> Developer Mode** and enable **"Force Composite"**.
5. **Install the VA-API Boot Redirect**:
   ```bash
   sudo ./tools/install_vaapi_boot_redirect.sh
   ```
   This ensures `libva` maps `radeonsi_drv_video.so` directly to `bc250_drv_video.so`, bypassing `AT_SECURE` variable filtering.
6. **Configure Active Adapter in Sunshine**:
   Run `./tools/bc250_diagnose.sh` to check which card node has the active monitor. In Sunshine Web Configuration (**Configuration -> Audio/Video**), set **adapter_name** to your active node (typically `/dev/dri/card1`).
7. **Install 32-bit Driver for Steam Link**:
   Ensure `/usr/lib32/dri/bc250_drv_video.so` is present on the host:
   ```bash
   ./tools/build_32bit.sh
   ```

---

## 16. Steam Link Won't Launch Select Games (Red Dead Redemption 2, GTA V, EA/Ubisoft Launchers)

### Symptoms
* Steam Link connects and streams the Steam library or Big Picture UI normally.
* Launching games with secondary launchers (such as *Red Dead Redemption 2*, *Grand Theft Auto V*, *Cyberpunk 2077*, or EA App titles) causes Steam Link to immediately abort the stream, freeze on a loading screen, or report *"Cannot launch game while streaming"*, even though the game starts on the host.

### Root Causes
1. **Secondary Launcher Process Detachment**: Games like RDR 2 launch a bootstrap process (`PlayRDR2.exe`), which starts the Rockstar Games Launcher in Proton, which in turn spawns `RDR2.exe`. Steam Remote Play hooks the process Steam initially launched. When the launcher closes or minimizes to spawn the game executable, Steam assumes the application exited and terminates the streaming session.
2. **GPU Compute Queue Contention on 32-bit Driver**: Red Dead Redemption 2 saturates 100% of the APU's 40 Compute Units and large amounts of unified GDDR6. When Steam Link ran the 32-bit compute encoder without `libx264`, motion estimation compute shaders contended with RDR 2's Vulkan/DX12 rendering, causing frame budget overruns and launcher timeouts.
3. **Proton Exclusive Fullscreen Capture**: By default, RDR 2 may attempt exclusive fullscreen DirectX 12 via `vkd3d-proton`, which fails window capture hooks in Steam Remote Play.

### Resolution Steps
1. **Enable Desktop Capture in Steam**:
   On the host machine (or in Steam Big Picture):
   * Go to **Settings -> Remote Play -> Advanced Host Options**.
   * Check **"Enable Desktop Capture"** (and uncheck "Direct capture only" if enabled).
   * This allows Steam Link to stream the active display continuously, preventing stream termination when secondary launchers detach.
2. **Install 32-bit Driver with `libx264` CPU Backend**:
   Update to the v0.5.1 companion bundle or rebuild using:
   ```bash
   ./tools/build_32bit.sh
   ```
   The 32-bit driver now incorporates native `libx264` support (`BC250_H264_BACKEND=x264`), encoding frames on the Zen 2 CPU cores and leaving the 40 CUs 100% free for RDR 2.
3. **Set Proton Launch Options for RDR 2**:
   In Steam, right-click **Red Dead Redemption 2 -> Properties -> General -> Launch Options**, and enter:
   ```text
   -vulkan -windowed -noborder
   ```
   This ensures RDR 2 runs via native Vulkan in a borderless window, ensuring seamless capture by Steam Remote Play.

---

## 17. Low GPU Usage on `nvtop` / `amdgpu_top` during H.264 Encoding (`backend=x264`)

### Symptoms
* During FFmpeg H.264 VA-API encoding (`-c:v h264_vaapi`), `nvtop` or `amdgpu_top` shows minimal GPU load while CPU usage is high.
* Terminal logs output:
  ```text
  [bc250-x264] 1920x1080 @ 24 fps, veryfast, high profile, threads 4, rc 1 (ICQ)
  ```

### Explanation & Backend Architecture
1. **Is the driver having problems using GPU and CPU simultaneously?**:
   **No.** There is no failure, error, or fallback occurring. In driver builds with `libx264`, the default H.264 pipeline intentionally runs on the 8-core Zen 2 CPU (`backend=x264`). The BC-250 has no hardware VCN encoder, and running the experimental GPU compute encoder consumed ~16ms of GPU time per frame (starving running 3D games of GPU compute and capping throughput at ~28 fps). `libx264` achieves 94–151 fps while leaving the 40 CUs 100% free for gaming.
2. **GPU Compute Encoder Opt-In (`backend=compute`)**:
   If you want the **GPU to encode H.264** using Vulkan compute Motion Estimation on the 40 CUs along with the **Dynamic Hybrid Governor** and Zen 2 SIMD entropy coding, simply launch with:
   ```bash
   export BC250_H264_BACKEND=compute
   ```
   When `BC250_H264_BACKEND=compute` is active, `amdgpu_top` and `nvtop` will register active GPU compute load.
3. **HEVC Encoding**:
   HEVC (`-c:v hevc_vaapi`) **always** runs on the GPU compute engine and registers active compute load on `nvtop` / `amdgpu_top`.
### FFmpeg Logs & Parameter Explanations (from Terminal Output)

1. **`Codec AVOption preset (Encoding preset) has not been used for any stream`**:
   * **Why it happens**: In FFmpeg, the `-preset` option belongs to software `libx264`. Hardware encoders like `h264_vaapi` do NOT register `-preset` in their private AVOption table, so FFmpeg's CLI parser prints this warning.
   * **Direct Command-Line Honor in v0.5.1+**: The BC-250 driver now directly parses `/proc/self/cmdline` for `-preset <val>` and `--preset=<val>`. Even though FFmpeg prints the warning, the BC-250 driver **directly intercepts and applies your chosen preset** (`medium`, `slow`, `faster`, etc.)!
   * **Environment Variable Support**: You can also set `export BC250_PRESET=medium` (or `export BC250_X264_PRESET=medium` or `export X264_PRESET=medium`).
   * **VA-API Option**: You can also use `-compression_level <1-7>` (1 = highest quality, 4 = balanced, 7 = fastest).

2. **`[h264_vaapi] No quality level set; using default (20)`**:
   * **Why it happens**: This is a standard informational warning from FFmpeg when no target bitrate (`-b:v`) or QP (`-qp`) is specified on the command line.
   * **How the driver responds**: The driver cleanly handles this by engaging Intelligent Constant Quality (ICQ) mode at default quality level 23 (~5.0 Mbps for 1080p).
   * **How to set quality explicitly**: Pass `-qp <1-51>` or `-b:v <bitrate>` (e.g. `-b:v 8M`), or override via `export BC250_X264_CRF=20`.

3. **`[h264_vaapi] Driver does not support some wanted packed headers (wanted 0xd, found 0x1)`**:
   * **Why it happens**: FFmpeg asks if the driver wants external packed slice and picture headers (`0xd = SEQUENCE | PICTURE | SLICE`).
   * **Why it is harmless**: The BC-250 driver deliberately advertises `0x1` (`VA_ENC_PACKED_HEADER_SEQUENCE`) so container muxers extract MP4 global headers (`avcC`), while the driver generates conformant in-band SPS, PPS, AUD, and slice headers directly. This log is purely informational and not an error.

4. **`threads 0` vs `threads 4` (CPU Utilization)**:
   * In driver versions prior to commit `83fbca8`, offline transcoding defaulted to `threads 0` (auto-detect all cores), which spawned 24 worker threads across all 16 Zen 2 threads and pinned CPU at 100%.
   * In commit `83fbca8` (v0.5.1+), offline transcoding is automatically capped to **4 worker threads** (`threads 4`), reducing CPU load to ~250% and keeping cooling fans quiet.
   * You can explicitly adjust thread counts at runtime: `export BC250_X264_THREADS=4` or `export BC250_MAX_CPU_THREADS=4`.

5. **`backend=x264` vs `backend=compute` (GPU Load)**:
   * By default, H.264 uses the high-performance CPU offload (`backend=x264`), ensuring 100% of the 40 GPU Compute Units are kept free for running 3D games.
   * To encode using the GPU's 40 Compute Units with OpenCL Motion Estimation and the Dynamic Hybrid Governor, set:
     ```bash
     export BC250_H264_BACKEND=compute
     ```
   * When `BC250_H264_BACKEND=compute` is set, `nvtop` and `amdgpu_top` will reflect active GPU compute load. HEVC (`-c:v hevc_vaapi`) always runs on the GPU.

### Fine-Tuning `libx264` Backend Parameters
* **Target CRF Quality**:
  ```bash
  export BC250_X264_CRF=23   # Default is 23 (~5.0 Mbps for 1080p). Lower = higher bitrate/quality.
  ```
* **Encoding Preset**:
  ```bash
  export BC250_X264_PRESET=medium   # slow, medium, fast, faster, veryfast, superfast, ultrafast
  ```
* **Thread Count**:
  ```bash
  export BC250_X264_THREADS=4       # Default is 4 threads. Set to 0 for unconstrained 16-thread saturation.
  ```

---

## 18. Arch Linux / CachyOS 32-bit Driver Build: `lib32-x264` in AUR

### Symptoms
* Running `./tools/build_32bit.sh` on Arch Linux or CachyOS failed with:
  ```text
  error: target not found: lib32-x264
  ```

### Cause
In Arch Linux, `lib32-x264` is located in the **Arch User Repository (AUR)** rather than the official `[multilib]` repository.

### Solution
1. Install `lib32-x264` via your AUR helper:
   ```bash
   paru -S lib32-x264
   # or
   yay -S lib32-x264
   ```
2. Re-run `./tools/build_32bit.sh`:
   `tools/build_32bit.sh` now automatically detects `paru` / `yay` or falls back gracefully to the GPU compute encoder if `lib32-x264` is omitted, ensuring 32-bit driver compilation always succeeds.

---

## 19. Streaming Latency & Dynamic Hybrid Governor vs Software Encode

### Symptoms & Feedback
* Community tester `oblique99` observed:
  > *"Not much impact on GPU performance but latency is not good. Point being the governor doesn't appear to work. It is behaving like an expensive rate controller for SW encode rather than a hybrid encode."*
  > *"cpu still giving 100% cpu encode, compute giving 100% gpu encode."*

### Root Causes
1. **Why "cpu" Mode was Giving 100% CPU Encode (0% GPU)**:
   * In driver builds with `libx264`, the default H.264 backend intentionally offloads encoding to the CPU (`backend=x264`).
   * Because it runs entirely in software on the Zen 2 CPU cores, GPU usage is near zero, and the Dynamic Hybrid Governor is completely bypassed.
   * In versions prior to commit `83fbca8`, offline transcoding defaulted to `threads 0`, which engaged all 16 Zen 2 cores at 100% CPU.
2. **Why "compute" Mode was Giving 100% GPU Encode (0% CPU)**:
   * In earlier builds, `gov->enabled` was `false` by default for any process other than Sunshine/Steam.
   * When running FFmpeg or benchmarks with `BC250_H264_BACKEND=compute`, the driver logged `hybrid_governor=enabled` but the governor was actually inactive (`gov->enabled = false`), permanently locking the pipeline into Tier 0 (GPU Full ME with radius 8).
   * Because Tier 0 ran 2.3 million block searches per frame on the 40 CUs and never transitioned into Tier 2 CPU SIMD offload, GPU compute usage registered at 100% while the CPU remained mostly idle.

### Resolutions Applied in v0.5.1+
1. **Real Dynamic Hybrid Governor (`backend=compute` or `backend=hybrid`)**:
   * The Dynamic Hybrid Governor and Tier 2 CPU SIMD offload are now **enabled by default** whenever `BC250_H264_BACKEND=compute` (or `hybrid`) is selected.
   * For live streaming callers (Sunshine, Steam Link, WiVRn):
     * **Tier 0 (< 7.0 ms)**: Full GPU Motion Estimation (`motion_estimation.comp`).
     * **Tier 1 (7.0 – 10.5 ms)**: Fast GPU Motion Estimation (reduced radius).
     * **Tier 2 (10.5 – 15.5 ms)**: **CPU SIMD Offload**. Motion estimation is shifted to 2 Zen 2 worker threads using SSE2/AVX2 intrinsics (`_mm_sad_epu8`), relieving the 40 GPU CUs so 3D games maintain maximum FPS.
     * **Tier 3 (> 15.5 ms)**: Emergency Failover (emits P_Skip to prevent network packet drops).
   * For FFmpeg offline transcoding:
     * Tier 3 frame-dropping is automatically disabled (`allow_failover = false`). Under GPU load, the pipeline automatically engages Tier 2 CPU SIMD offload, splitting motion estimation onto the CPU while the GPU handles DCT/quantization—delivering a true 50/50 hybrid encode without dropping frames.
2. **Low-Latency CPU Streaming Mode (`backend=x264`)**:
   * Live streaming callers now automatically use the **`"ultrafast"`** preset with strict sliced threading, zero lookahead, and a **single-frame VBV buffer** (`i_vbv_buffer_size = 1 frame`). This drops frame encode latency from ~15 ms down to **2–4 ms**.
   * Thread utilization can be customized via `export BC250_THREADS=2` (or `BC250_MAX_CPU_THREADS=2`) to prevent CPU saturation.
3. **Command-Line Preset Honor**:
   * Passing `-preset <name>` on the FFmpeg command line is now intercepted and directly applied by the driver, bypassing FFmpeg's internal option filter. Universal environment variable aliases (`BC250_PRESET`, `BC250_X264_PRESET`, `X264_PRESET`) are also supported.
4. **Persistent Surface Mapping Guard (SIGSEGV / Address Boundary Error Fix)**:
   * In earlier builds, entering Dynamic Governor Tier 2 (CPU SIMD Motion Estimation) called raw `vkMapMemory` and `vkUnmapMemory` on the input surface device memory.
   * Because VA-API surfaces are persistently mapped by the driver at creation time, unmapping the surface memory during Tier 2 caused FFmpeg's `AVHWFramesContext` surface pool to access unmapped virtual memory when recycling the surface on subsequent frames (`Map surface 0x2`), triggering a `SIGSEGV (Address boundary error)` crash.
   * Resolved by routing all Tier 2 CPU SIMD searches through [`gpu_compute_map_surface()`](file:///D:/Kai/bc250-encoding-decoding-fix/approach1-compute-encoder/src/gpu_compute.c) and [`gpu_compute_unmap_surface()`](file:///D:/Kai/bc250-encoding-decoding-fix/approach1-compute-encoder/src/gpu_compute.c), which preserves persistently mapped surface pointers, and persistently mapping `ctx->recon_memory` to avoid per-frame mapping overhead.
5. **High-Throughput GPU Compute Mode & Multi-Threaded SIMD Offload**:
   * **Governor IDR Latency Poisoning Fix**: IDR frames are intra-coded and do not use motion estimation; their ~14 ms execution time previously caused the governor EMA to falsely trip Tier 2 CPU offload on uncontended P-frames, dropping encode speed below 1x realtime and starving the GPU (causing intermittent GPU usage). The governor now only updates on P-frames (`!is_idr`), maintaining full-speed GPU compute (>150 fps, >6x realtime) in Tier 0.
   * **Multi-Threaded CPU SIMD Motion Estimation**: When Tier 2 offload is engaged (via `BC250_H264_BACKEND=hybrid` or under genuine GPU contention), the SIMD motion search now defaults to 4 parallel OpenMP worker threads (honoring `BC250_THREADS`, `BC250_CPU_THREADS`, and `BC250_MAX_CPU_THREADS` up to 16 threads), eliminating the single-thread bottleneck.
   * **Explicit GPU Mode**: Setting `BC250_H264_BACKEND=gpu` completely bypasses the governor for 100% dedicated GPU compute encoding.
6. **Multi-Slice OpenMP Parallel Entropy Coding (<1x Realtime / Single-Thread Bottleneck Fix)**:
   * **Root Cause**: In H.264, entropy coding (CABAC/CAVLC) cannot be parallelized within a single slice because macroblock context models and arithmetic coder states sequentially depend on the prior macroblock. When `num_slices == 1`, all 8,160 macroblocks at 1080p were coded serially on a single CPU core, taking ~45–60 ms per frame and capping encoding throughput at ~16–22 fps (<1x realtime) with exactly 1 CPU core pegged at 100%.
   * **Parallel Multi-Slice Architecture**: The compute encoder now defaults to 4 slices per frame for FFmpeg and GPU-backed HD/FHD/4K encodes (`total_mbs >= 1000` or resolution >= 720p). Slices are encoded concurrently across 4 OpenMP worker threads, slashing CPU entropy coding latency from ~50 ms down to ~10–12 ms and scaling throughput from <1x to >4x realtime (>100 fps).
   * **CLI & Environment Control**: Automatically honors `-threads <N>` / `-slices <N>` from the FFmpeg command line, as well as `BC250_SLICES_PER_FRAME`, `BC250_THREADS`, `BC250_CPU_THREADS`, and `BC250_MAX_CPU_THREADS` up to 16 threads.
7. **Steam Link Desktop Mode Grey/Black Screen (Single-Slice Requirement)**:
   * **Root Cause**: Many hardware decoders on smart TVs, Android TV devices, and Steam Link client hardware do not support multi-slice H.264 streams or discard subsequent slices within a frame. When multi-slice parallel encoding was enabled globally (`def_slices = 4`), Steam Remote Play (`steam`, `streaming_client`, `steamwebhelper`) received 4 slices per frame, causing the client decoders to drop macroblocks or abort, resulting in a solid grey or black screen during desktop streaming.
   * **Single-Slice Enforcement**: The driver automatically detects Steam Remote Play callers and forces single-slice encoding (`num_slices = 1` and `b_sliced_threads = 0`) across both [`va_backend.c`](file:///D:/Kai/bc250-encoding-decoding-fix/approach1-compute-encoder/src/va_backend.c), [`encoder_h264.c`](file:///D:/Kai/bc250-encoding-decoding-fix/approach1-compute-encoder/src/encoder_h264.c), and [`encoder_x264.c`](file:///D:/Kai/bc250-encoding-decoding-fix/approach1-compute-encoder/src/encoder_x264.c), even if `BC250_SLICES_PER_FRAME=4` is present in system environment configs. Slices can still be manually forced with `BC250_FORCE_SLICES=1`.
8. **Heavy 3D Game / RDR 2 Benchmark 200ms Latency Spike Elimination**:
   * **Root Cause**: Titles that saturate 100% of the 40 Compute Units (such as Red Dead Redemption 2 running on DirectX 12 / Vulkan via Proton) submit massive rendering command buffers to the GPU. In the compute encoder, [`gpu_compute_sync_slot()`](file:///D:/Kai/bc250-encoding-decoding-fix/approach1-compute-encoder/src/gpu_compute.c) previously waited with an unbounded fence timeout (`UINT64_MAX`), blocking the streaming thread for up to 200 ms while waiting for GPU compute passes to schedule behind RDR 2's graphics queue. Additionally, the governor oscillated back to GPU compute every other frame, causing severe latency spikes to recur frequently.
   * **16ms Fence Timeout & Immediate CPU Failover**: [`gpu_compute_sync_slot()`](approach1-compute-encoder/src/gpu_compute.c) and [`gpu_compute_begin_picture()`](approach1-compute-encoder/src/gpu_compute.c) now bound fence waits to 16 ms (1 frame interval at 60 fps) for live streaming sessions (overrideable via `BC250_GPU_TIMEOUT_MS`). If the GPU is blocked behind a game, the wait times out and the frame completes as a safe P_Skip.
    > ⚠️ **Correction:** the "Total latency is strictly bounded under 16 ms" claim in the original text of this item was wrong, and this is where it stopped being true. Bounding the *fence wait* bounds one thing; a frame can still be blown by an unbounded `nanosleep` elsewhere in the same encode path, or by doing much more work than the tier it was on. Both happened - see §20, which is the full account.

   * **Recommended Zero-GPU Mode for 100% Saturated Titles**: For games that fully saturate the 40 CUs, using the CPU encoder (`backend=x264`, the default H.264 mode when `BC250_H264_BACKEND` is unset) ensures 0% GPU compute overhead and steady 2–4 ms encode latency on the Zen 2 CPU cores.

## 20. Latency Spikes and Garbled Video That Survived the 16 ms Fix (§19 item 8)

### Symptoms
A live stream is fine on the desktop and degrades sharply during fast action in a GPU-saturating game. Two separate things are reported, and they have two separate causes:

* **Host processing latency spikes past 200 ms**, with the Sunshine statistics showing a min/max/avg like `5.3/218.5/141.6 ms` and the frame rate collapsing to single digits. The stream's own numbers look "very close to running without Sunshine", which is the part that makes it look like a measurement mistake.
* **The picture is garbled** - macroblock-level corruption, not just a soft frame. Much worse during fast action; a lesser version of the same thing appears during game benchmarks.

### Why the 16 ms fence bound did not catch either

The bound covers the `vkWaitForFences` in `gpu_compute_sync_slot()`. Neither of these spikes is a fence wait:

1. **An unbounded sleep in the same encode path.** The transient-failure retries in `gpu_compute_create_image()` and `gpu_compute_end_picture()` slept 20, 40, 80, 160, 320, 640 ms between up to 7 attempts. A live caller hitting an allocation or a submit retry once had a *half-second* stall sitting in the frame, and a 160 ms one is by itself a ">200 ms spike" once the rest of the frame is counted. Those retries exist for real GPU contention, so a game that causes them is exactly the game that triggers them.

2. **Work that got dramatically more expensive when it was least affordable.** Governor Tier 2 ("CPU SIMD Offload") is the tier that engages when the GPU is slow, and it moved the whole motion search to the CPU. The cost of that move was not counted anywhere: the SIMD search runs over a surface allocated `HOST_VISIBLE|HOST_COHERENT`, which on this APU is the GART aperture rather than a cached host mapping, so its many scattered 16-byte loads cannot be served from CPU cache at all. The search's own early-exits (static blocks, spatial predictor) are what keep it cheap on ordinary content - and they are exactly what stops working on fast action. So the tier meant to absorb GPU pressure was itself paying tens of milliseconds, precisely when the frame had least budget for it.

### The garbling: a genuine read/write race, not a quality problem

The same CPU ME path is the only place in this encoder that reads the input frame and the reconstruction **from the CPU** rather than handing them to a compute shader, and both of those reads were unordered:

* The input surface is written by a **foreign GPU context** - Sunshine's own GL pass, through a dma-buf this driver exported. The existing protection was `gpu_compute_wait_for_image_ready()`, which snapshots the dma-buf read fence and hands it to `vkQueueSubmit()` as a Vulkan semaphore. That orders the *GPU* dispatch. It orders nothing on a `vkMapMemory()` + CPU load, which happens in this thread before that submit has even been recorded.
* The reconstruction is written by **this driver's own GPU**, and in the synchronous path that is finished by construction. In the pipelined path it is not, and the CPU read raced it.

Both produce a torn frame, and a torn frame produces exactly the reported symptom: corruption that scales with how busy the GPU is, because a busy GPU is what makes the foreign render pass slow *and* what makes the governor choose the tier that reads it. Garbling that tracks GPU load is a race, not a bitrate.

### Fixes applied

* **Live callers get a short retry schedule** - 3 attempts over 6 ms instead of 7 over 1260 ms - and then report failure to a caller that already has the correct answer (certify the frame P_Skip, which is what a decoder repeats for a frame it gets no new information for). An offline transcode keeps the long schedule, where spending two seconds to avoid a re-encode is right.
* **The CPU ME path waits before it reads.** `gpu_compute_wait_for_image_ready_host()` blocks on the same dma-buf read fence the GPU was going to be given microseconds later, so it costs the wait that was going to happen anyway and additionally orders the CPU. `gpu_compute_cpu_read_safe()` separately requires this driver's own last submission to be complete, which covers the pipelined case; failing either gate falls back to GPU ME, i.e. the behaviour from before Tier 2 existed.
* **The CPU ME path stages through cached host RAM.** One streaming row-by-row copy of each luma plane per frame, then the search runs out of ordinary cached memory instead of out of the GART. This is the difference between ~2 ms and tens of ms for the same search.
* **The governor holds a tier before entering the offload** (`BC250_GOVERNOR_DWELL_FRAMES`, default 8 for live callers, 0 offline). The existing 8-frame hysteresis governed *leaving* the offload and said nothing about entering it, so a single bad frame could cross into a full CPU ME and back.
* **A Tier 3 failover now lands on Tier 1, not Tier 2.** One late frame is not evidence that the GPU is unusable, and Tier 2 is the most expensive frame in the cycle. Sustained pressure still reaches Tier 2 through the EMA, which is an average and is unmoved by one spike.

### Ring-fencing the CUs

The related question - can the game be stopped from taking every CU - has an answer, with a granularity caveat, and it is not a driver setting. See **[`docs/streaming-ringfence.md`](streaming-ringfence.md)** and `tools/sunshine_preset/apply_gpu_ringfence.sh`.

Short version: `VK_AMD_shader_core_policy` is the extension that would do this from inside a Vulkan app and **RADV does not expose it**. Mesa's per-process `AMD_CU_MASK` (Mesa ≥ 22.0) does work, but it is a mask *within* a shader array applied to *every* array, so "2 CUs" means 2 per array, and Mesa rejects a mask that would leave its hardware late-allocation constraints unsatisfied. `BC250_CU_REPORT=1` prints the topology and the two masks to use.

Be clear-eyed about what a fence buys: it stops the game from dispatching on those CUs, which shortens how long the encoder's submission waits its turn. It does not create GPU time, and it does not touch the CPU-side cost, which DEVLOG §24.6 measures going from 7.1 ms to 24.0 ms under the same load with no code change.

## 21. Bazzite / Fedora / Immutable OS "libx264.so.163 / libx265.so.199 not found" (Issue #62)

### Symptoms
When running `./tools/bc250_diagnose.sh` or launching Sunshine on modern immutable distributions (such as Bazzite 40/44, Fedora Silverblue/Kinoite, Arch Linux, or Debian 13), `libva` errors out with:
```text
libva error: dlopen of /usr/local/lib64/dri/bc250_drv_video.so failed: libx264.so.163: cannot open shared object file: No such file or directory
```
or `libx265.so.199: cannot open shared object file`.

### Cause
In pre-v0.5.4 releases, binaries were built on an Ubuntu 22.04 LTS host that hard-coded link-time ELF dependencies (`DT_NEEDED`) on Ubuntu's exact library SONAMEs (`libx264.so.163` and `libx265.so.199`). Modern distributions package newer versions (`libx264.so.164+` and `libx265.so.209+` or `libx265.so.215+`), which caused dynamic link failures upon driver loading.

### Solution
1. **Update to Driver Release v0.5.4+ (Permanent Fix)**:
   Release v0.5.4 converts both `encoder_x264.c` and `encoder_x265.c` to use dynamic runtime symbol probing via `dlopen()` and `dlsym()`, completely eliminating compile-time `DT_NEEDED` dependencies on specific library versions. The driver now automatically probes and resolves whichever version of `libx264` or `libx265` is installed on your distribution.
   Download the latest release from [GitHub Releases](https://github.com/simpmix/bc250-encoding-decoding-fix/releases/latest) and run `./install.sh`.

2. **Immediate Symlink Workaround (for legacy builds)**:
   If running an older driver build without updating:
   ```bash
   sudo ln -sf /usr/lib64/libx264.so.* /usr/local/lib64/libx264.so.163
   sudo ln -sf /usr/lib64/libx265.so.* /usr/local/lib64/libx265.so.199
   sudo ldconfig
   ```

## 22. 10-Bit P010 HDR Encoding & Single-Depth vs. Multilib x265 Packages

### Symptoms
Enabling 10-bit encoding (`VAProfileHEVCMain10` with P010 surfaces) causes a crash with `corrupted double-linked list` or aborts during encoder initialization on distributions with 8-bit-only `libx265.so`.

### Cause
Distributions package `libx265` in two distinct formats:
* **Multilib builds** (Debian, Ubuntu, Fedora) which compile 8-bit, 10-bit, and 12-bit depth libraries and route requests via `x265_api_get(10)`.
* **Single-depth builds** (8-bit only). On these systems, `x265_api_get(10)` returns the 8-bit API table where parameter structure sizes and planar pointer offsets do not match 16-bit P010 structures.

### Solution
Release v0.5.4 inspects the exported `x265_max_bit_depth` symbol at runtime via `dlsym`. If the host library does not support 10-bit encoding (`*x265_ops.max_bit_depth < 10`), the driver gracefully rejects the 10-bit request and falls back to the Vulkan GPU compute backend, preventing memory corruption.
To verify 10-bit support on your host system:
```bash
readelf -s /usr/lib64/libx265.so* | grep x265_max_bit_depth
```

## 23. 4K H.264 Playback & Wavefront Decoding Stuttering

### Symptoms
Playing high-bitrate 4K (3840x2160) H.264 videos in `mpv` or streaming at 4K60 experiences severe micro-stuttering and CPU thread starvation despite low overall CPU percentage.

### Cause
At 4K resolution, a picture contains 32,400 macroblocks across 135 macroblock rows. Older decoder implementations throttled wavefront reconstruction cache bands to 4 rows (`band_rows = 4`), requiring 34 separate condition variable sleep/wait barriers per 4K frame, causing severe thread mutex contention. Additionally, dynamic heap allocation of chroma buffers on every frame added memory allocator pressure.

### Solution
Update to release v0.5.4+. The H.264 decoder now scales `band_rows` to the full frame height (`d->mb_h`), eliminating 34 condition variable barriers per 4K frame. Persistent scratch buffers are maintained in `struct h264_decoder`, and chroma interleaving uses SSE2 128-bit unpack instructions (`_mm_unpacklo_epi8`, `_mm_unpackhi_epi8`) for zero-allocation reconstruction.

## 24. Shared-Memory Telemetry Permission Denied (`/dev/shm/bc250_stats`)

### Symptoms
Running `cat /dev/shm/bc250_stats` or `./tools/bc250_stats` as a non-root user outputs `Permission denied` when Sunshine or Gamescope was started under a different UID or as root.

### Cause
In earlier versions, `/dev/shm/bc250_stats` was created with default umask permissions (`0644` or `0600`), preventing access across user accounts.

### Solution
The driver now explicitly sets `fchmod(g_shm_fd, 0666)` upon creating shared memory segments and includes an automatic per-UID fallback (`/tmp/bc250_stats_<uid>`).
The `bc250_stats` monitor tool automatically checks the current user's UID, `SUDO_UID`, and global fallback paths:
```bash
./tools/bc250_stats --watch
```
