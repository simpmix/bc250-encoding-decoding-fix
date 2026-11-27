#!/usr/bin/env bash
# repro_sync_timeout.sh - deterministic reproduction of
#   "Error encoding a frame: Input/output error"
# caused by a GPU fence timeout (VK_TIMEOUT) being reported to the VA-API
# client as a hard failure, instead of the P_Skip failover the encoder uses
# internally for the same condition.
#
# No GPU load is needed: BC250_GPU_TIMEOUT_MS=1 makes the fence-wait budget
# smaller than one frame's GPU time, so the timeout happens on every frame.
# Safe to run (one encode, a few seconds); it does NOT stress the GPU.
#
# Usage: tools/repro_sync_timeout.sh [seconds]      (default 10)
#
# Exit codes: 0 = control passes AND timeout case passes (bug fixed),
#             1 = timeout case fails (bug present),
#             2 = setup problem or the control run itself failed.

set -u
SECS=${1:-10}
DEVICE=$(ls /dev/dri/by-path/*-render 2>/dev/null | head -n1)
[ -n "$DEVICE" ] || DEVICE=/dev/dri/renderD128
command -v ffmpeg >/dev/null || { echo "ffmpeg not found" >&2; exit 2; }
[ -e "$DEVICE" ] || { echo "no render node" >&2; exit 2; }

export LIBVA_DRIVER_NAME=${LIBVA_DRIVER_NAME:-bc250}
export BC250_H264_BACKEND=compute

run_case() { # $1 = BC250_GPU_TIMEOUT_MS, $2 = log file
    BC250_GPU_TIMEOUT_MS=$1 ffmpeg -hide_banner -y \
        -init_hw_device "vaapi=va:$DEVICE" -filter_hw_device va \
        -re -f lavfi -i "testsrc2=size=1080x1920:rate=30" -t "$SECS" \
        -vf "format=nv12,hwupload" \
        -c:v h264_vaapi -profile:v main -rc_mode CBR -b:v 6000k -maxrate 6000k -g 120 -bf 0 \
        -f null - >"$2" 2>&1
}

LOGDIR=$(mktemp -d)
echo "== control: BC250_GPU_TIMEOUT_MS=0 (unbounded) =="
run_case 0 "$LOGDIR/control.log"; RC0=$?
echo "   ffmpeg exit $RC0"
if [ "$RC0" -ne 0 ]; then
    echo "control run failed; environment problem, not the bug. See $LOGDIR/control.log" >&2
    exit 2
fi

echo "== case: BC250_GPU_TIMEOUT_MS=1 (forces VK_TIMEOUT on every frame) =="
run_case 1 "$LOGDIR/timeout.log"; RC1=$?
echo "   ffmpeg exit $RC1"
echo "   timeouts logged by the driver: $(grep -c 'engaging failover' "$LOGDIR/timeout.log")"
if [ "$RC1" -ne 0 ] || grep -q -i "Input/output error" "$LOGDIR/timeout.log"; then
    echo "RESULT: BUG PRESENT - fence timeout surfaced as an encode error"
    grep -i -E "error|failed" "$LOGDIR/timeout.log" | tail -n 3
    echo "logs: $LOGDIR"
    exit 1
fi
echo "RESULT: OK - fence timeouts degrade frames instead of failing the encode"
rm -rf "$LOGDIR"
exit 0
