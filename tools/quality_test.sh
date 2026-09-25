#!/usr/bin/env bash
# bc250-encoding-decoding-fix v0.4.0 - https://github.com/simpmix/bc250-encoding-decoding-fix
#
# quality_test.sh - Correctness test harness for the BC-250 Vulkan-compute
# VA-API H.264 encoder.
#
# WHY THIS EXISTS
# ----------------
# The existing CI (.github/workflows/build.yml) only checks that ffmpeg can
# decode the produced bitstream without a hard error, and that the decoded
# byte count is roughly the expected size. That check is blind to an
# encoder that emits *syntactically valid but content-free* H.264: every
# macroblock coded as header/skip-only with zero residual data. Such a
# stream has correct SPS/PPS, the right frame count, decodes with zero
# errors, and is the right size on disk -- and decodes to a flat gray image
# with no relation whatsoever to the source frames. The old CI check passes
# on that broken output.
#
# This script closes that gap by checking *pixel content*, not just
# "did the decoder not crash":
#
#   1. Capture ground truth: run the real encode pipeline with
#      BC250_DUMP_INPUT_FRAMES=1, which makes the driver dump the exact raw
#      NV12 bytes it receives from libva (see bc250_debug_dump_nv12_frame()
#      in approach1-compute-encoder/src/gpu_compute.c, called from both
#      known VA-API upload paths) to disk before any GPU work touches them.
#      This is the real mid-pipeline data, not a parallel file that might
#      not match byte-for-byte. If neither upload path fires (e.g. some
#      future ffmpeg/libva build uses a dma-buf import path this driver
#      doesn't instrument), the script falls back to an ffmpeg-generated
#      raw reference built from the identical, deterministic lavfi source.
#   2. Encode those frames through the real, full pipeline (ffmpeg + real
#      libva driver + Vulkan compute shaders on the BC-250's Compute Units).
#   3. Decode the result independently with ffmpeg's software H.264
#      decoder as an oracle (no VA-API involved on the decode side).
#   4. Compare decoded pixels against the ground-truth reference with
#      ffmpeg's own psnr/ssim filters -- real numeric scores.
#   5. Pass/fail against a PSNR floor (see PSNR_THRESHOLD below).
#
# A content-free (all-skip, zero-residual) encode is expected to fail this
# script with a very low PSNR against a textured/moving test pattern; a
# real encode should score comfortably above the threshold.
#
# INVESTIGATION NOTE (RESOLVED 2026-09-08) -- luma-specific corruption
# --------------------------------------------------------------------------
# Between the intra/inter-prediction fixes landing and the two commits
# below, this script reported FAIL (~12-17 dB average PSNR) on real content.
# Before looking at the encoder itself, two harness-side theories were
# considered and both ruled out by direct measurement: a fixed-frame-lag
# misalignment between the dumped ground-truth input and decoded output
# (a per-offset PSNR sweep showed no peak at any offset, ruling out a lag),
# and a color-range mismatch between limited/TV and full range (a control
# encode through software libx264 via the same decode/extract commands
# preserved the input range faithfully, and pre-converting the test source
# to full range didn't change the score either) -- this script's frame
# pairing and range handling were never the problem.
#
# Root cause: two independent CAVLC/entropy-coding bugs, both in the
# encoder, neither in this script -- fixed in commit 8feddef
# (cavlc_write_run_befores() coded run_before values in the wrong
# frequency order per ITU-T 9.2.3) and commit d95b840 (Intra16x16 luma DC
# array was not transposed before being handed to CAVLC). Both left the
# GPU transform/reconstruction pipeline itself untouched and correct
# (confirmed via a ground-truth-vs-GPU-reconstruction comparison at
# ~50dB), which is why a manually-extracted decoded frame could look
# plausible at a glance while differing from the source pixel-for-pixel.
# Current board-validated result with both fixes applied: PSNR avg
# 37.7 dB / SSIM 0.99, PASS -- see those two commits' messages for full
# methodology and numbers.
#
# USAGE
#   ./tools/quality_test.sh
#
# ENVIRONMENT OVERRIDES (all optional)
#   BUILD_DIR       cmake build directory
#                     (default: <repo>/approach1-compute-encoder/build)
#   BC250_ENV_SCRIPT board build-toolchain env script to source if present
#                     (default: $HOME/build-deps/env.sh)
#   WORK_DIR        scratch directory for frames/streams/logs
#                     (default: /tmp/bc250_quality_test)
#   WIDTH, HEIGHT   test resolution (default: 640x480)
#   FRAMERATE       test framerate (default: 25)
#   DURATION        test clip length in seconds (default: 2)
#   BITRATE         encoder target bitrate (default: 4M)
#   PSNR_THRESHOLD  minimum acceptable average PSNR in dB (default: 25)
#   RENDER_DEVICE   VA-API render node (default: /dev/dri/renderD128)
#   SKIP_BUILD      if set to 1, don't (re)build; require an existing
#                     bc250_drv_video.so in BUILD_DIR
#
# This script never touches a system-wide driver install path -- it always
# runs the driver straight out of BUILD_DIR via LIBVA_DRIVERS_PATH, so it is
# safe to run alongside other work using the shared system install.
#
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/approach1-compute-encoder/build}"
BC250_ENV_SCRIPT="${BC250_ENV_SCRIPT:-$HOME/build-deps/env.sh}"
WORK_DIR="${WORK_DIR:-/tmp/bc250_quality_test}"

# Probe for modern -fps_mode passthrough (FFmpeg 5.1+) vs legacy -vsync 0 (FFmpeg 4.x)
if ffmpeg -hide_banner -loglevel quiet -f lavfi -i "color=s=2x2:d=0.04" \
        -fps_mode passthrough -f null - </dev/null >/dev/null 2>&1; then
    FFMPEG_FPSMODE=(-fps_mode passthrough)
else
    FFMPEG_FPSMODE=(-vsync 0)
fi

WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-480}"
FRAMERATE="${FRAMERATE:-25}"
DURATION="${DURATION:-2}"
BITRATE="${BITRATE:-4M}"
PSNR_THRESHOLD="${PSNR_THRESHOLD:-25}"
RENDER_DEVICE="${RENDER_DEVICE:-/dev/dri/renderD128}"
SKIP_BUILD="${SKIP_BUILD:-0}"

FRAME_SIZE=$(( WIDTH * HEIGHT * 3 / 2 ))   # NV12 4:2:0, 8-bit

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}${BOLD}   BC-250 Encoder Correctness / Quality Test          ${NC}"
echo -e "${BLUE}======================================================${NC}"
echo -e "  Resolution     : ${WIDTH}x${HEIGHT} @ ${FRAMERATE}fps, ${DURATION}s"
echo -e "  Build dir      : ${BUILD_DIR}"
echo -e "  Work dir       : ${WORK_DIR}"
echo -e "  PSNR threshold : ${PSNR_THRESHOLD} dB"

# ------------------------------------------------------------------
# [1/6] Build (or verify) a private driver build. Never touches the
# shared system install path -- ffmpeg is pointed at BUILD_DIR directly
# via LIBVA_DRIVERS_PATH / BC250_SHADER_DIR below.
# ------------------------------------------------------------------
echo -e "\n${BOLD}[1/6] Preparing private driver build...${NC}"
if [ -f "$BC250_ENV_SCRIPT" ]; then
    echo -e "  -> Sourcing build toolchain: $BC250_ENV_SCRIPT"
    # Third-party env scripts aren't guaranteed to be `set -u`-safe (e.g.
    # they may reference PKG_CONFIG_PATH etc. without a default). Relax
    # nounset just for the source, then restore our strict mode.
    set +u
    # shellcheck disable=SC1090
    source "$BC250_ENV_SCRIPT"
    set -u
fi

DRIVER_SO="$BUILD_DIR/bc250_drv_video.so"
if [ "$SKIP_BUILD" != "1" ] || [ ! -f "$DRIVER_SO" ]; then
    mkdir -p "$BUILD_DIR"
    (
        cd "$BUILD_DIR"
        cmake "$REPO_ROOT/approach1-compute-encoder" \
              -DCMAKE_BUILD_TYPE=Release \
              -DBUILD_TESTS=OFF \
              -DCMAKE_INSTALL_PREFIX="$WORK_DIR/fake_prefix" \
              -DLIBVA_DRIVERS_PATH="$BUILD_DIR"
        make -j"$(nproc)"
    )
fi

if [ ! -f "$DRIVER_SO" ]; then
    echo -e "${RED}✗ bc250_drv_video.so not found at $DRIVER_SO after build. Aborting.${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓ Driver ready: $DRIVER_SO${NC}"

SPV_COUNT=$(find "$BUILD_DIR" -maxdepth 1 -name '*.spv' | wc -l)
if [ "$SPV_COUNT" -eq 0 ]; then
    echo -e "  ${RED}✗ No compiled .spv shaders found in $BUILD_DIR. Aborting.${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓ Found $SPV_COUNT compiled shader(s) in $BUILD_DIR${NC}"

# Point libva at *our* private build directory only. Do not fall back to
# any system search path, so this run can never pick up the shared install.
export LIBVA_DRIVER_NAME=bc250
export LIBVA_DRIVERS_PATH="$BUILD_DIR"
export BC250_SHADER_DIR="$BUILD_DIR"

# ------------------------------------------------------------------
# [2/6] Fresh scratch space.
# ------------------------------------------------------------------
echo -e "\n${BOLD}[2/6] Setting up scratch workspace...${NC}"
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR/dump_frames"
echo -e "  ${GREEN}✓ $WORK_DIR${NC}"

# ------------------------------------------------------------------
# [3/6] Capture ground truth + encode through the real pipeline.
# BC250_DUMP_INPUT_FRAMES=1 makes the driver write every raw NV12 frame it
# actually receives from libva to $WORK_DIR/dump_frames, from whichever
# upload path ffmpeg's vaapi hwupload happens to take.
# ------------------------------------------------------------------
echo -e "\n${BOLD}[3/6] Encoding test pattern through the real VA-API pipeline...${NC}"
ENCODED="$WORK_DIR/encoded.mp4"
ENCODE_LOG="$WORK_DIR/ffmpeg_encode.log"
FRAME_COUNT=$(( FRAMERATE * DURATION ))

# NOTE: drive the frame count with -frames:v rather than testsrc's own
# duration=... option. Both should be equivalent, but on-hardware testing
# found that letting the lavfi source terminate itself via duration= causes
# a hard SIGSEGV partway through the vaapi encode on this driver (isolated
# empirically: identical crash with and without BC250_DUMP_INPUT_FRAMES, so
# it is a pre-existing EOF/flush issue in the encode path, not something
# this harness's instrumentation introduced). -frames:v completed cleanly
# in every trial from 3 to 50 frames.
set +e
BC250_DUMP_INPUT_FRAMES=1 BC250_DUMP_DIR="$WORK_DIR/dump_frames" \
ffmpeg -y -v info -f lavfi -i "testsrc=size=${WIDTH}x${HEIGHT}:rate=${FRAMERATE}" -frames:v "$FRAME_COUNT" \
    -vaapi_device "$RENDER_DEVICE" \
    -vf 'format=nv12,hwupload' \
    -c:v h264_vaapi -b:v "$BITRATE" \
    "$ENCODED" > "$ENCODE_LOG" 2>&1
ENCODE_RC=$?
set -e

if [ $ENCODE_RC -ne 0 ] || [ ! -s "$ENCODED" ]; then
    echo -e "  ${RED}✗ Encode failed (exit $ENCODE_RC) or produced an empty file. Log:${NC}"
    tail -n 30 "$ENCODE_LOG" | sed 's/^/    /'
    exit 1
fi
ENCODED_BYTES=$(wc -c < "$ENCODED")
echo -e "  ${GREEN}✓ Encoded $ENCODED_BYTES bytes -> $ENCODED${NC}"

# ------------------------------------------------------------------
# [4/6] Assemble the ground-truth reference: prefer the driver's own
# instrumented dump (real mid-pipeline bytes); fall back to an
# ffmpeg-generated raw file from the identical deterministic lavfi source
# if the dump directory came up empty (e.g. a future ffmpeg/libva upload
# path this driver doesn't instrument).
# ------------------------------------------------------------------
echo -e "\n${BOLD}[4/6] Building ground-truth reference...${NC}"
REFERENCE="$WORK_DIR/raw_reference.yuv"
DUMP_COUNT=$(find "$WORK_DIR/dump_frames" -maxdepth 1 -name 'frame_*.nv12' | wc -l)

if [ "$DUMP_COUNT" -gt 0 ]; then
    echo -e "  ${GREEN}✓ Driver captured $DUMP_COUNT real input frame(s) via instrumentation${NC}"
    # frame_00000.nv12, frame_00001.nv12, ... sort correctly as plain text.
    find "$WORK_DIR/dump_frames" -maxdepth 1 -name 'frame_*.nv12' | sort | xargs cat > "$REFERENCE"
    REFERENCE_SOURCE="driver instrumentation (BC250_DUMP_INPUT_FRAMES)"
else
    echo -e "  ${YELLOW}! No frames captured by driver instrumentation -- falling back to${NC}"
    echo -e "  ${YELLOW}  an ffmpeg-generated raw reference from the identical lavfi source.${NC}"
    ffmpeg -y -v error -f lavfi -i "testsrc=size=${WIDTH}x${HEIGHT}:rate=${FRAMERATE}" -frames:v "$FRAME_COUNT" \
        -pix_fmt nv12 -f rawvideo "$REFERENCE"
    DUMP_COUNT=$(( $(wc -c < "$REFERENCE") / FRAME_SIZE ))
    REFERENCE_SOURCE="ffmpeg lavfi fallback (testsrc, identical params)"
fi

REFERENCE_BYTES=$(wc -c < "$REFERENCE")
REFERENCE_FRAMES=$(( REFERENCE_BYTES / FRAME_SIZE ))
echo -e "  Reference source : $REFERENCE_SOURCE"
echo -e "  Reference frames : $REFERENCE_FRAMES ($REFERENCE_BYTES bytes)"

if [ "$REFERENCE_FRAMES" -lt 1 ]; then
    echo -e "  ${RED}✗ Ground-truth reference has zero usable frames. Aborting.${NC}"
    exit 1
fi

# ------------------------------------------------------------------
# [5/6] Decode independently. Plain software decode -- no VA-API, no
# involvement of this driver at all -- so it's a trustworthy oracle for
# "what does a standards-compliant player actually see."
# ------------------------------------------------------------------
echo -e "\n${BOLD}[5/6] Decoding with ffmpeg's software H.264 decoder (oracle)...${NC}"
DECODED="$WORK_DIR/decoded.yuv"
ffmpeg -y -v error "${FFMPEG_FPSMODE[@]}" -i "$ENCODED" -f rawvideo -pix_fmt nv12 "$DECODED"
DECODED_BYTES=$(wc -c < "$DECODED")
DECODED_FRAMES=$(( DECODED_BYTES / FRAME_SIZE ))
echo -e "  ${GREEN}✓ Decoded $DECODED_FRAMES frame(s) ($DECODED_BYTES bytes)${NC}"

if [ "$DECODED_FRAMES" -lt 1 ]; then
    echo -e "  ${RED}✗ Decoder produced zero frames. Encoder output is unreadable. FAIL.${NC}"
    exit 1
fi

# Trim both raw streams to the frame count they have in common so the
# comparison filters see matched, aligned buffers.
COMMON_FRAMES=$(( DECODED_FRAMES < REFERENCE_FRAMES ? DECODED_FRAMES : REFERENCE_FRAMES ))
COMMON_BYTES=$(( COMMON_FRAMES * FRAME_SIZE ))
echo -e "  Comparing ${COMMON_FRAMES} frame(s) common to both streams"

DECODED_TRIM="$WORK_DIR/decoded_trim.yuv"
REFERENCE_TRIM="$WORK_DIR/reference_trim.yuv"
head -c "$COMMON_BYTES" "$DECODED" > "$DECODED_TRIM"
head -c "$COMMON_BYTES" "$REFERENCE" > "$REFERENCE_TRIM"

# ------------------------------------------------------------------
# [6/6] Score decoded pixels against ground truth with real PSNR/SSIM,
# and gate pass/fail on the PSNR floor.
#
# Threshold reasoning: this driver is an unoptimized-but-functional
# software-shader encoder, not a quality-tuned one. PSNR > 25 dB is not a
# high bar -- it is roughly the point below which a human immediately
# recognizes an image is unrelated to (not just "compressed from") the
# source, e.g. heavy blocking/color-shift/structural collapse. A flat
# gray, content-free frame compared to a moving/colorful test pattern
# scores far below this (typically single digits to low teens), while
# any encoder that is actually carrying the picture's structure through
# comfortably clears it even at a modest bitrate. 25 dB is therefore a
# floor for "this is recognizably the source content," not a
# high-quality bar -- exactly what's needed to catch the
# all-skip/zero-residual class of bug without being a strict codec
# quality gate.
# ------------------------------------------------------------------
echo -e "\n${BOLD}[6/6] Scoring decoded output against ground truth...${NC}"
COMPARE_LOG="$WORK_DIR/ffmpeg_compare.log"
ffmpeg -y -hide_banner -loglevel info -nostats \
    -f rawvideo -pix_fmt nv12 -s "${WIDTH}x${HEIGHT}" -r "$FRAMERATE" -i "$DECODED_TRIM" \
    -f rawvideo -pix_fmt nv12 -s "${WIDTH}x${HEIGHT}" -r "$FRAMERATE" -i "$REFERENCE_TRIM" \
    -lavfi "[0:v]split=2[d1][d2];[1:v]split=2[r1][r2];[d1][r1]psnr=stats_file=$WORK_DIR/psnr_per_frame.log;[d2][r2]ssim=stats_file=$WORK_DIR/ssim_per_frame.log" \
    -f null - > "$COMPARE_LOG" 2>&1

PSNR_LINE=$(grep -o 'PSNR[^$]*' "$COMPARE_LOG" | tail -n 1 || true)
SSIM_LINE=$(grep -o 'SSIM[^$]*' "$COMPARE_LOG" | tail -n 1 || true)

if [ -z "$PSNR_LINE" ]; then
    echo -e "  ${RED}✗ Could not find a PSNR result in ffmpeg output. Log:${NC}"
    tail -n 30 "$COMPARE_LOG" | sed 's/^/    /'
    exit 1
fi

PSNR_AVG_RAW=$(echo "$PSNR_LINE" | grep -oP 'average:\K[0-9.]+|average:\Kinf' || true)
SSIM_ALL_RAW=$(echo "$SSIM_LINE" | grep -oP 'All:\K[0-9.]+' || true)

echo -e "  $PSNR_LINE"
[ -n "$SSIM_LINE" ] && echo -e "  $SSIM_LINE"

echo -e "\n${BLUE}======================================================${NC}"
if [ "$PSNR_AVG_RAW" = "inf" ]; then
    echo -e "  ${GREEN}${BOLD}PASS${NC} -- decoded output is bit-exact with the reference (PSNR: inf dB)"
    RESULT=0
elif [ -n "$PSNR_AVG_RAW" ] && awk "BEGIN{exit !($PSNR_AVG_RAW >= $PSNR_THRESHOLD)}"; then
    echo -e "  ${GREEN}${BOLD}PASS${NC} -- average PSNR ${PSNR_AVG_RAW} dB >= threshold ${PSNR_THRESHOLD} dB"
    [ -n "$SSIM_ALL_RAW" ] && echo -e "  SSIM (All): ${SSIM_ALL_RAW}"
    RESULT=0
else
    echo -e "  ${RED}${BOLD}FAIL${NC} -- average PSNR ${PSNR_AVG_RAW:-<unparsed>} dB < threshold ${PSNR_THRESHOLD} dB"
    [ -n "$SSIM_ALL_RAW" ] && echo -e "  SSIM (All): ${SSIM_ALL_RAW}"
    echo -e "  ${YELLOW}This means the decoded frames do not resemble the encoder's own${NC}"
    echo -e "  ${YELLOW}input -- consistent with a content-free / all-skip encode bug.${NC}"
    RESULT=1
fi
echo -e "${BLUE}======================================================${NC}"

{
    echo "timestamp=$(date -u +%FT%TZ)"
    echo "resolution=${WIDTH}x${HEIGHT}"
    echo "framerate=${FRAMERATE}"
    echo "duration_s=${DURATION}"
    echo "bitrate=${BITRATE}"
    echo "reference_source=${REFERENCE_SOURCE}"
    echo "reference_frames=${REFERENCE_FRAMES}"
    echo "decoded_frames=${DECODED_FRAMES}"
    echo "common_frames_compared=${COMMON_FRAMES}"
    echo "psnr_line=${PSNR_LINE}"
    echo "ssim_line=${SSIM_LINE}"
    echo "psnr_threshold_db=${PSNR_THRESHOLD}"
    echo "result=$([ $RESULT -eq 0 ] && echo PASS || echo FAIL)"
} > "$WORK_DIR/summary.txt"
echo -e "\nFull summary written to: $WORK_DIR/summary.txt"
echo -e "Per-frame PSNR/SSIM logs: $WORK_DIR/psnr_per_frame.log, $WORK_DIR/ssim_per_frame.log"

exit $RESULT
