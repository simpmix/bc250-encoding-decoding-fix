# Sunshine Game Streaming Preset for AMD BC-250

Pre-tuned configuration for **Sunshine** to enable smooth, low-latency game streaming from your AMD BC-250 console to handheld devices (Steam Deck, ROG Ally, Android phones) or smart TVs via **Moonlight**.

## Quick Apply

Run the automatic installer script from your terminal:

```bash
chmod +x apply_sunshine_preset.sh
./apply_sunshine_preset.sh
```

This will safely back up your existing configuration and apply the BC-250 VA-API encoder settings to `~/.config/sunshine/sunshine.conf`.

## Ring-fencing the GPU

If a game saturates all 40 CUs and the stream degrades with it:

```bash
chmod +x apply_gpu_ringfence.sh
./apply_gpu_ringfence.sh
```

This reads the CU topology from the driver, then limits the game to the lower CUs of each shader array (via Mesa's `AMD_CU_MASK`) and Sunshine to the top ones, so the encoder has CUs the game cannot dispatch on. It also grants Sunshine `CAP_SYS_NICE` so the existing `BC250_QUEUE_PRIORITY=high` knob is actually accepted by Vulkan.

`--dry-run` shows the plan, `--remove` undoes it, and **[`docs/streaming-ringfence.md`](../../docs/streaming-ringfence.md)** explains the granularity limits, what a fence genuinely buys, and what it does not.

## Recommended Moonlight Client Settings

For the best visual quality and lowest latency on your handheld or client device:

| Setting | Recommended Value | Notes |
|---|---|---|
| **Resolution** | `1920x1080` (1080p) or `1280x720` (720p) | Matches standard handheld and TV resolutions |
| **Framerate** | `60 FPS` | Smooth frame pacing |
| **Bitrate** | `20 - 30 Mbps` | Clean stream with no compression artifacts |
| **Video Codec** | `H.264` | Lowest decode latency across Android, Linux, & Windows |
| **Frame Pacing** | `Balanced with lowest latency` | Prevents stuttering |

## Why These Settings?
* **VA-API Backend:** Directly leverages the APU's 40 unlocked Compute Units (Cyan Skillfish / Oberon architecture).
* **FEC 20%:** Forward Error Correction ensures dropped Wi-Fi packets do not cause video stuttering or freezes.
* **Low-Latency CBR:** Dynamic rate control maintains consistent bandwidth utilization without packet bursts.
