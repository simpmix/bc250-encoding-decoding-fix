#!/bin/bash
# Build the standalone H.265 harness.
#
# Same shape as the H.264 one: the decoder reconstructs into plain system
# memory, so the harness needs no GPU and no libva and builds anywhere.
set -eu
# ⚠️ The tree this script sits in, not a fixed path. With a fixed path a
# copy of the repository - an older commit checked out elsewhere to
# compare with, a throwaway tree with a diagnostic patch - silently built
# the main checkout instead, and the comparison compared it with itself.
D="$(cd "$(dirname "$0")/.." && pwd)/approach1-compute-encoder"
OUT="${1:-/tmp/hevcps}"

gcc -O2 -g -Wall -Wextra -std=gnu11 \
    -I"$D/src" -I"$D/include" \
    -o "$OUT" \
    "$D/tools/hevcps.c" \
    "$D/src/hevc_ps.c" \
    "$D/src/hevc_cu.c" \
    "$D/src/hevc_residual.c" \
    "$D/src/hevc_transform.c" \
    "$D/src/hevc_pred.c" \
    "$D/src/hevc_filter.c" \
    "$D/src/decoder_h265.c" \
    "$D/src/hevc_wpp.c" \
    "$D/src/worker_pool.c" \
    "$D/src/hevc_mv.c" \
    "$D/src/hevc_mc.c" \
    "$D/src/hevc_tiles.c" \
    "$D/src/hevc_dec_tables.c" \
    "$D/src/cabac.c" \
    -pthread -lm
echo "done: $OUT"
