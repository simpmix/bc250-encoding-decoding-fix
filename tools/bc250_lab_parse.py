#!/usr/bin/env python3
"""
bc250_lab_parse.py - the ONE parser for this driver's instrumentation logs.

WHY THIS EXISTS:
The per-stage numbers in docs/DEVLOG.md §19-§21 were produced by about a
dozen throwaway awk one-liners, each re-deriving the same parse. That went
wrong in ways that cost real time: one of them sorted its input and destroyed
chronological order (making a fix look like it worked when it had not), and
several reported a mean without any measure of spread, which is how a 0.4%
"win" gets written down next to a 6% run-to-run variance.

So: one parser, one place to fix a bug, and it always reports spread
alongside the mean.

INPUT   ffmpeg stderr from a run with BC250_PERF_STATS=1, which carries the
        driver's [BC250_PERF_FRAME] / [BC250_PERF_CPU] / [BC250_PERF_SHADOW] /
        [BC250_PERF_GPU] / [BC250_NZ_AUDIT] lines.
OUTPUT  a single flat JSON object of metrics (default) or one TSV row.

Every timing metric emits three keys: <name> (mean), <name>_sd (sample
standard deviation) and <name>_n (sample count). A mean with n=1 and no sd is
reported as sd=0.0, which is honest but should not be mistaken for precision -
use `bc250_lab.sh noise` to establish the floor before believing any delta.
"""

import argparse
import json
import math
import re
import sys

# [TAG] key=value key=value ...  - the driver's own format throughout.
LINE_RE = re.compile(r'\[(BC250_[A-Z_]+)\]\s+(.*)')
KV_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)=(-?[0-9]+\.?[0-9]*|[A-Za-z]+)')

# GPU stage keys, in pipeline order, as emitted by gpu_compute.c.
GPU_STAGES = ["me", "predict", "dct", "quant", "reconstruct",
              "wavefront", "deblock", "entropy", "copy", "total"]


def stats(values):
    """mean, sample stddev, n. Sample (n-1) stddev, so a single sample
    reports 0.0 rather than pretending to a population figure."""
    n = len(values)
    if n == 0:
        return None, None, 0
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0, n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var), n


def parse(text):
    frames = []      # [BC250_PERF_FRAME]
    cpu = []         # [BC250_PERF_CPU]
    shadow = []      # [BC250_PERF_SHADOW]
    gpu = []         # [BC250_PERF_GPU]
    audit = []       # [BC250_NZ_AUDIT]
    waits = []       # [BC250_PERF_WAIT]   site=sync|begin_picture
    submits = []     # [BC250_PERF_SUBMIT]
    phases = []      # [BC250_PERF_PHASE]

    for raw in text.splitlines():
        m = LINE_RE.search(raw)
        if not m:
            continue
        tag, rest = m.group(1), m.group(2)
        rec = {}
        for k, v in KV_RE.findall(rest):
            try:
                rec[k] = float(v) if ('.' in v or v.lstrip('-').isdigit()) else v
            except ValueError:
                rec[k] = v
        if tag == "BC250_PERF_FRAME":
            frames.append(rec)
        elif tag == "BC250_PERF_CPU":
            cpu.append(rec)
        elif tag == "BC250_PERF_SHADOW":
            shadow.append(rec)
        elif tag == "BC250_PERF_GPU":
            gpu.append(rec)
        elif tag == "BC250_NZ_AUDIT":
            audit.append(rec)
        elif tag == "BC250_PERF_WAIT":
            waits.append(rec)
        elif tag == "BC250_PERF_SUBMIT":
            submits.append(rec)
        elif tag == "BC250_PERF_PHASE":
            phases.append(rec)

    out = {}

    def emit(name, values):
        mean, sd, n = stats(values)
        if n == 0:
            return
        out[name] = round(mean, 4)
        out[name + "_sd"] = round(sd, 4)
        out[name + "_n"] = n

    # --- EVERYTHING is split by frame type --------------------------------
    # I-frames are architecturally far more expensive than P-frames: whole-
    # frame intra via the diagonal wavefront on the GPU, and several times the
    # entropy-coding work on the CPU. Pooling the two makes any such metric
    # track GOP LENGTH rather than anything about the encoder, so a change of
    # `-g` looks like a performance change.
    #
    # This bit me while writing this parser: wall_ms was split by type but
    # cavlc_ms was not, which pooled an 8 ms I-frame with 3.5 ms P-frames and
    # produced a negative `unaccounted_ms`. Hence the rule, applied to every
    # metric without exception: the bare name is P-frames only, `_i` is
    # I-frames only, and anything pooled must say `all_` in its name.
    def pick(records, key, ftype):
        return [r[key] for r in records
                if key in r and (ftype is None or r.get("type") == ftype)]

    def emit_by_type(name, records, key):
        emit(name, pick(records, key, "P"))
        emit(name + "_i", pick(records, key, "I"))

    emit_by_type("p_wall_ms", frames, "wall_ms")   # historical name, P-only
    emit("all_wall_ms", pick(frames, "wall_ms", None))

    # The headline number, P-frames only for the reason above.
    p_wall = pick(frames, "wall_ms", "P")
    if p_wall:
        mean = sum(p_wall) / len(p_wall)
        if mean > 0:
            out["p_fps_ceiling"] = round(1000.0 / mean, 2)

    emit("bytes_p", pick(frames, "bytes", "P"))
    emit("bytes_i", pick(frames, "bytes", "I"))
    emit("qp", pick(frames, "qp", "P"))
    emit("qp_i", pick(frames, "qp", "I"))
    out["frames_total"] = len(frames)
    out["frames_i"] = sum(1 for f in frames if f.get("type") == "I")
    out["frames_p"] = sum(1 for f in frames if f.get("type") == "P")

    # --- CPU side ---------------------------------------------------------
    emit_by_type("cavlc_ms", cpu, "cavlc_ms")
    emit_by_type("shadow_ms", shadow, "shadow_copy_ms")
    for k in ("quant_bytes", "dc_bytes", "coeff_bytes", "pred_mode_bytes",
              "mv_bytes", "nz_bytes", "total_bytes"):
        vals = [s[k] for s in shadow if k in s]
        if vals:
            out["shadow_" + k] = int(vals[0])   # constant per resolution

    # --- GPU stages -------------------------------------------------------
    # Also type-split: `wavefront` only runs on I-frames and `me` only on
    # P-frames, so a pooled mean of either is a fraction with GOP length in
    # the denominator rather than a stage cost.
    for st in GPU_STAGES:
        emit_by_type("gpu_%s_ms" % st, gpu, st + "_ms")

    # --- fence waits and submits -----------------------------------------
    # `wait_sync_ms` is the host blocked in vkWaitForFences. It is the term
    # that explodes under GPU contention (1.9 ms idle -> 321 ms with a GPU
    # load), and it matters because this device exposes exactly ONE queue
    # family with ONE queue (graphics+compute+transfer): there is no
    # async-compute ring, so encode is ordered against everything else on the
    # machine rather than running alongside it.
    for site in ("sync", "begin_picture"):
        emit("wait_%s_ms" % site,
             [w["wait_ms"] for w in waits
              if w.get("site") == site and "wait_ms" in w])
    emit("submit_ms", [s["submit_ms"] for s in submits if "submit_ms" in s])

    # Phase brackets from encoder_h264.c. `dispatch_ms` covers command-buffer
    # recording, per-frame descriptor updates and image-layout transitions -
    # all CPU work that was never timed, and the first place to look for the
    # ~327 ms that no other bracket accounted for under GPU load.
    emit_by_type("phase_begin_ms", phases, "begin_ms")
    emit_by_type("phase_dispatch_ms", phases, "dispatch_ms")
    emit_by_type("phase_end_sync_ms", phases, "end_sync_ms")

    # DELIBERATELY NOT DERIVED: an earlier version computed
    # queue_wait = wait_sync - gpu_total and reported it as "time spent
    # queued behind other work". That is wrong, and measurably so - it came
    # out NEGATIVE (-2.9 ms) on an idle board. The pipeline is
    # double-buffered: gpu_compute_sync() waits on prev_buf, the PREVIOUS
    # frame's fence, so GPU execution overlaps the next frame's CPU work and
    # wait_sync can legitimately be far smaller than execution time.
    #
    # wait_sync_ms is therefore reported raw, and means "host blocked waiting
    # for the previous frame's submission to complete". Under GPU contention
    # that is dominated by queue backlog, but the two are not separable from
    # these counters alone, so no such number is invented here.
    if out.get("wait_sync_ms") is not None and out.get("p_wall_ms"):
        out["wait_sync_pct"] = round(
            100.0 * out["wait_sync_ms"] / out["p_wall_ms"], 2)

    # --- accounting check -------------------------------------------------
    # The frame is synchronous, so the host-side terms should sum to p_wall:
    #
    #   p_wall ~= wait(begin_picture) + submit + wait(sync) + shadow + cavlc
    #
    # NOTE gpu_total_ms is NOT a term here. It comes from GPU timestamp
    # queries and measures execution, which happens *inside* wait(sync) - so
    # adding it would double-count. An earlier version of this parser summed
    # gpu_total + cavlc + shadow, which is only right when the GPU is idle
    # (submit-then-immediately-wait makes wait(sync) ~= execution). Under a
    # GPU load that approximation reported ~96% of the frame as
    # "unaccounted", which read like missing instrumentation when in fact the
    # model was wrong and the time was plainly in a bracket already being
    # logged. Prefer the wait-based sum whenever the WAIT lines are present.
    have_waits = out.get("wait_sync_ms") is not None
    if have_waits:
        # phase_end_sync_ms already contains submit + wait(sync), so those
        # are not added again; phase_begin_ms contains wait(begin_picture).
        if out.get("phase_dispatch_ms") is not None:
            parts = [out.get("phase_begin_ms") or 0.0,
                     out.get("phase_dispatch_ms") or 0.0,
                     out.get("phase_end_sync_ms") or 0.0,
                     out.get("shadow_ms") or 0.0,
                     out.get("cavlc_ms") or 0.0]
            out["accounted_model"] = "phase-based"
        else:
            parts = [out.get("wait_begin_picture_ms") or 0.0,
                     out.get("submit_ms") or 0.0,
                     out.get("wait_sync_ms") or 0.0,
                     out.get("shadow_ms") or 0.0,
                     out.get("cavlc_ms") or 0.0]
            out["accounted_model"] = "wait-based"
    else:
        parts = [out.get("gpu_total_ms"), out.get("cavlc_ms"), out.get("shadow_ms")]
        out["accounted_model"] = "gpu-timestamp-based (approximate; valid only when idle)"
    if out.get("p_wall_ms") and all(p is not None for p in parts):
        accounted = sum(parts)
        out["accounted_ms"] = round(accounted, 4)
        out["unaccounted_ms"] = round(out["p_wall_ms"] - accounted, 4)
        if out["p_wall_ms"] > 0:
            out["unaccounted_pct"] = round(
                100.0 * (out["p_wall_ms"] - accounted) / out["p_wall_ms"], 2)

    # --- nonzero-mask audit ----------------------------------------------
    if audit:
        out["audit_frames"] = len(audit)
        out["audit_mismatch_frames"] = sum(
            1 for a in audit if a.get("mismatches", 0) > 0)
        out["audit_mismatches_total"] = int(sum(a.get("mismatches", 0) for a in audit))
        emit("audit_all_zero_pct", [
            100.0 * a["all_zero"] / a["blocks"]
            for a in audit if a.get("blocks")])
        emit("audit_ac_zero_pct", [
            100.0 * a["ac_zero"] / a["blocks"]
            for a in audit if a.get("blocks")])

    # --- failures the harness must never silently pass over ---------------
    # Counted here rather than grepped by each caller, so a run that produced
    # numbers *and* errors cannot be reported as clean.
    out["err_vk_oom"] = len(re.findall(r'Vulkan error -2', text))
    out["err_alloc_failed"] = len(re.findall(r'allocate_encoding_buffers\(', text))
    out["err_slice_overflow"] = len(re.findall(r'abandoning frame', text))
    out["err_shader_load"] = len(re.findall(r'FAILED to load', text))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile", nargs="?", default="-")
    ap.add_argument("--tsv", action="store_true", help="one TSV row instead of JSON")
    ap.add_argument("--header", action="store_true", help="with --tsv, print a header row first")
    ap.add_argument("--fields", default="", help="comma-separated subset, in order")
    ap.add_argument("--tag", default="", help="opaque label echoed back as the 'tag' field")
    args = ap.parse_args()

    if args.logfile == "-":
        text = sys.stdin.read()
    else:
        with open(args.logfile, "r", errors="replace") as fh:
            text = fh.read()

    out = parse(text)
    if args.tag:
        out = dict([("tag", args.tag)] + list(out.items()))

    if not args.tsv:
        print(json.dumps(out, indent=2, sort_keys=False))
        return

    keys = [k.strip() for k in args.fields.split(",") if k.strip()] or list(out.keys())
    if args.header:
        print("\t".join(keys))
    print("\t".join(str(out.get(k, "")) for k in keys))


if __name__ == "__main__":
    main()
