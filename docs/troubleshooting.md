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
* During FFmpeg H.264 VA-API encoding (`-c:v h264_vaapi`) or live streaming, `nvtop` or `amdgpu_top` shows minimal/zero GPU load.
* Terminal logs output:
  ```text
  [bc250-h264] Encoder initialized: 1920x1080, profile 100, backend=x264 (CPU libx264; set BC250_H264_BACKEND=compute for GPU)
  ```
* CPU usage shows 4–6 active worker threads with 5.5x transcoding speeds.

### Explanation & Backend Architecture
By default in driver versions built with `libx264`:
1. **H.264 CPU Offload (`backend=x264`)**: The full H.264 encoding pipeline (motion estimation, transform, quantization, and CABAC/CAVLC entropy coding) is handled by `libx264` using Zen 2 CPU cores. The GPU is only used for downloading the raw NV12 surface. This was adopted because the experimental GPU compute encoder took ~16ms of GPU time per frame (limiting throughput to ~28 fps and starving active games of GPU compute). `libx264` achieves 94–151 fps while leaving the GPU 100% dedicated to 3D rendering.
2. **GPU Compute Encoder Opt-In (`backend=compute`)**: If you explicitly want the Vulkan compute pipeline with the **Dynamic Hybrid Governor** (GPU Motion Estimation + Zen 2 CPU entropy coding), launch your application with:
   ```bash
   export BC250_H264_BACKEND=compute
   ```
3. **HEVC Encoding**: HEVC (`-c:v hevc_vaapi`) **always** runs on the GPU compute engine with Vulkan compute shaders and registers active compute load on `nvtop` / `amdgpu_top`.

### Fine-Tuning `libx264` Backend Parameters
You can configure the CPU encoder behavior through environment variables:
* **Target CRF Quality**:
  ```bash
  export BC250_X264_CRF=23   # Default is 23 (~5.0 Mbps for 1080p). Lower = higher bitrate/quality.
  ```
* **Encoding Preset**:
  ```bash
  export BC250_X264_PRESET=veryfast   # ultrafast, superfast, veryfast, faster
  ```
* **Thread Count**:
  ```bash
  export BC250_X264_THREADS=4        # Default: 4 threads for live streaming (Sunshine, WiVRn, Steam), auto for FFmpeg.
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
