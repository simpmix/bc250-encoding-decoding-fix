# Multi-core CAVLC / slice-parallel encoding — design & feasibility

Status: **design-only research**, no encoding-logic changes made. Written against
`main@65081df` (forked as `perf/multicore-cavlc-design`). This document does not
touch `cavlc.c` or `encoder_h264.c` encoding logic — a separate, active effort is
fixing the per-macroblock CAVLC algorithmic bottleneck in those exact files.
This is the orthogonal, additive lever: once that fix lands, can the (now-faster)
per-macroblock CAVLC work be spread across the BC-250's 16 logical cores?

**Bottom line up front:** the codebase already has a slice partitioning feature
(`BC250_SLICES_PER_FRAME`) that is genuinely spec-independent per-slice today —
that is the correct unit to thread. The blocking issue is not slice independence
(that's already solid); it's that the current per-slice loop places each slice's
EBSP bytes in-place at a *serially-computed* running offset into the shared
output buffer. That one thing has to change for parallel execution, and it's a
small, mechanical change. Real measurements below show CAVLC is currently
97–99% of frame time (confirming the ~45 µs/MB figure in the task brief almost
exactly), which makes an N-way parallel CAVLC look very attractive *today* — but
Amdahl's law says that attractiveness is entirely contingent on how big the
CAVLC share still is *after* the algorithmic fix lands. That's the open variable
a follow-up implementer needs to re-check, not assume.

---

## 1. Is slice structure already the right parallelism unit, and is it wired up correctly?

Read: `approach1-compute-encoder/src/encoder_h264.c` (the `for (int s = 0; s < num_slices; s++)`
loop in `h264_encoder_encode_frame()`, ~line 1107, and its `h264_encoder_encode_raw()`
twin at ~line 1366), plus `luma_nc()`/`chroma_nc()` (~line 468/491) and
`gpu_compute.h`'s `gpu_compute_dispatch_encode()` doc comment (~line 276).

**Yes — slices are already a correct, independent unit, on both the CPU and GPU side:**

- `BC250_SLICES_PER_FRAME` (env var, 1–16, default 1) partitions `total_mbs` into
  `num_slices` **contiguous mb-address ranges** (`start_mb = s*total_mbs/num_slices`,
  `end_mb = (s+1)*total_mbs/num_slices` — not necessarily row-aligned).
- Each slice gets its own slice header (`first_mb_in_slice = start_mb`, its own
  `bitstream_t bs` over its own `malloc`'d `slice_rbsp` buffer) — **already
  physically separated per slice**, not shared mutable bitstream state.
- CAVLC neighbor availability (`nC`, ITU-T 9.2.1) is derived by `luma_nc()`/
  `chroma_nc()` via an explicit `(mb - 1) >= nc->start_mb` / `(mb - width_in_mbs)
  >= nc->start_mb` check — i.e. **any neighbor mb address outside `[start_mb,
  end_mb)` is treated as unavailable**, regardless of whether the shared
  `nz_luma`/`nz_cb`/`nz_cr` arrays actually hold valid (in-memory) data from an
  earlier slice. This matches ITU-T H.264 6.4.9 exactly (a neighbouring
  macroblock belonging to a different slice than the current macroblock is
  defined as **not available**, independent of physical adjacency) — it is not
  a simplification or approximation, it's spec-correct slice independence by
  construction. The `mb_skip_run` counter (`current_skip_run`) is also declared
  fresh inside the per-slice loop body and never carried across slices, so
  P-slice skip-run state is likewise already slice-scoped.
- **This is corroborated on the GPU side too**, not just CPU: `num_slices` is
  passed into `gpu_compute_dispatch_encode()` *before* CAVLC runs, specifically
  so `residual_predict.comp`'s I16x16 mode-decision/neighbor-pixel logic can
  also treat a different-slice neighbor MB as unavailable for intra prediction
  (see `gpu_compute.h` line 276's doc comment and that shader's own "SLICE
  BOUNDARIES" section). So slice independence is enforced consistently at both
  the GPU residual-generation stage and the CPU entropy-coding stage — no
  hidden asymmetry where one stage assumes independence the other doesn't honor.

**No blockers found in the "is it really independent" sense.** The only shared
*mutable* state touched inside the per-slice loop is `encoder->nz_luma` /
`nz_cb` / `nz_cr` (persistent `[total_mbs]`-sized arrays, see `nc_ctx_t` and the
encoder struct, ~line 87–136) — but each slice's thread only ever **writes**
indices in its own `[start_mb, end_mb)` range, and (per the availability check
above) only ever **reads** indices already inside that same range. Ranges are
disjoint across slices by construction (`s*total_mbs/num_slices` is a strict
partition), so there is no data race in the C11 memory-model sense — this is
N threads writing disjoint array elements, not any form of shared mutable
counter. (There *is* a minor false-sharing perf consideration — adjacent
slices' boundary mb's `nz_luma[mb]` entries, 16 bytes each, can land in the
same cache line — noted in §2, not a blocker.)

`cavlc.c` and `bitstream.c` were also audited for hidden shared state: every
function in both files is a pure function of its explicit arguments (`bitstream_t
*bs` passed by the caller) plus `static const` lookup tables (Exp-Golomb/CAVLC
code tables — read-only, never written after compile time). There is no
`static` mutable variable anywhere in either file. **Both files are already
thread-safe as written**, provided each thread owns its own `bitstream_t`
instance and its own `nc_ctx_t` view into its own mb range (both already true
today, per-slice, in the serial loop).

`rate_control.c`'s mutable state (`rc->current_qp`, `buffer_fullness`,
`prev_frame_sad`) is read once *before* the per-slice loop (`rc_get_frame_qp`,
producing one `qp` shared read-only by every slice — this already matches the
existing single-QP-per-frame design, `BC250_SLICES_PER_FRAME` was never
per-slice-adaptive-QP) and updated once *after* the loop (`rc_update_stats`,
fed the frame's total byte count). Neither call is inside the parallel region.
**No rate-control synchronization is needed.**

## 2. Threading model

### What data does each thread need?

- **Read-only, shared, no copy needed:** `quant_levels`, `coeff`, `pred_modes`,
  `mvs` — these are the whole-frame GPU staging-buffer readback pointers
  (`gpu_compute_get_{quant,coeff,pred_mode,mv}_staging_data()`), indexed by
  absolute mb address across the *entire* frame, not per-slice buffers. Nothing
  in `encode_mb_i16x16()`/`encode_mb_p16x16()` writes through these pointers.
  Each thread reads only its own slice's disjoint mb-index sub-range. **No
  per-slice copy of GPU readback data is needed or beneficial** — passing the
  same base pointer + this slice's `(start_mb, end_mb)` to each thread is
  sufficient and avoids a real memcpy of a potentially multi-MB buffer
  (`total_mbs * 24 * 16 * sizeof(int)`, e.g. ~28 MB at 1080p for `quant_levels`
  alone) per slice per frame.
- **Read+write, shared, disjoint ranges:** `encoder->nz_luma/nz_cb/nz_cr` — safe
  as-is (§1), no locking. Minor tuning opportunity: since these are read
  immediately after being written (same-slice, same or next mb), the current
  layout (`uint8_t[total_mbs][16]`, 16 bytes/mb) already places each mb's data
  contiguously, so false sharing is limited to the ~1-2 mb's worth of bytes
  exactly at each slice boundary — not worth restructuring for this.
- **Private per thread (already true today, unchanged):** `slice_rbsp` buffer,
  `bitstream_t bs`, `nc_ctx_t nc` (just holds pointers + this slice's
  `start_mb`, cheap to construct per-thread).

### The real blocker: output placement

Today, *after* each slice's RBSP is finished, `bs_rbsp_to_ebsp()` writes EBSP
bytes directly into `encoder->output_buf` at `nal_dst = encoder->output_buf +
total_written`, where `total_written` is a plain local variable incremented
serially as each slice completes in program order (encoder_h264.c ~line
1259–1273). This is **the one piece of real, necessary serialization** in the
current design, and it does not survive parallel execution unmodified: with N
worker threads, slice completion order is not guaranteed to match slice index
order, and the final byte size of each slice's EBSP is not known until that
slice's CAVLC finishes (variable-length coding) — so no thread can know its
correct `output_buf` offset in advance.

**Fix (mechanical, not conceptual):** give each slice thread its own private
EBSP output buffer (or just keep working in RBSP and do the RBSP→EBSP pass
during the join step below — either is fine; EBSP-per-thread is slightly
simpler since it reuses `bs_rbsp_to_ebsp()` unchanged). After
`pthread_join`-ing all N threads, a short **serial join loop, in slice index
order 0..num_slices-1** (not completion order — see §4), copies each thread's
NAL bytes (4-byte start code + NAL header + that slice's EBSP, exactly what the
current code assembles inline) into `encoder->output_buf` at a running offset —
i.e. the same assembly logic that exists today, just performed once, serially,
after all CAVLC work is done, instead of interleaved with it. This join is pure
`memcpy` work over already-encoded bytes (typically tens of KB/frame) — trivial
compared to the CAVLC time it's unlocking, and it does not need to be
parallelized itself.

### Thread pool lifetime

Create the pool **once, at `h264_encoder_create()`**, sized `min(num_slices,
N_CORES)` workers (see §3 for what N to actually pick), not per-frame. Two
reasons, one obvious and one worth stating explicitly:

1. Per-frame `pthread_create`/`pthread_join` overhead (typically low tens of µs
   per thread on Linux, but real and non-zero) is currently negligible against
   a 45–165 ms/frame CAVLC budget — but that's exactly the budget the
   concurrent algorithmic fix is trying to shrink. If CAVLC per-macroblock cost
   drops by, say, 10–50×, a *few hundred µs* of thread-spawn overhead per frame
   could become a measurable fraction of the new, much smaller frame budget.
   Design for the post-fix world, not the current one.
2. `num_slices` can change frame-to-frame only via re-reading the env var
   (today it's read fresh every `h264_encoder_encode_frame()` call, so in
   principle a caller could vary it — though no current caller does). A
   persistent pool sized for the *maximum* slice count (16, the codebase's own
   validated ceiling) and simply using a subset of workers when `num_slices` is
   smaller handles this without recreating the pool.

`pthread.h`/`Threads::Threads` is **already linked into `bc250_drv_video`**
(`CMakeLists.txt` line 85, `find_package(Threads REQUIRED)` +
`${CMAKE_THREAD_LIBS_INIT}` in `target_link_libraries`) — a follow-up
implementer needs **zero build-system changes** to add `pthread_create`/
`pthread_join`/a simple work-queue.

### How many threads (N) relative to 16 real cores?

Board-confirmed: `nproc --all` = 16 logical cores (Zen 2, this is the BC-250's
CPU side, separate from its Oberon / Cyan Skillfish GPU). `BC250_SLICES_PER_FRAME` already
validates to the range `[1, 16]` — so the existing env var ceiling happens to
line up with the real core count, but that's the *parallelism* ceiling, not the
*quality-sensible* ceiling — see the measured curve below. More slices always
means more independent parallel work, but each additional slice boundary is
also an additional point where CAVLC context sharing, intra prediction, and
deblocking are cut off — a genuine quality cost, independent of and additional
to the threading question.

**Real measured quality-vs-slice-count data**, from `tools/quality_test.sh`
(640×480, 25 fps, 2 s / 50 frames, default 4 Mbps, same driver build for every
point so the comparison is apples-to-apples):

| `BC250_SLICES_PER_FRAME` | PSNR avg (dB) | SSIM (All) | Δ PSNR vs. N=1 |
|---:|---:|---:|---:|
| 1  | 37.66 | 0.9918 | — (baseline) |
| 2  | 34.00 | 0.9886 | −3.66 dB |
| 4  | 29.27 | 0.9777 | −8.39 dB |
| 8  | 29.67 | 0.9783 | −8.00 dB |
| 16 | 28.91 | 0.9729 | −8.75 dB |

(All five pass the script's 25 dB floor, but the floor is a "not
content-free/corrupt" gate, not a quality target — see the script's own
threshold-reasoning comment. Note the non-monotonic 4→8 blip (29.27→29.67 dB)
is very likely sampling noise from a single 50-frame test clip and a
resolution where slice boundaries land differently relative to scene content
at different N — not a claim that 8 slices measurably beats 4.)

The curve says most of the real quality cost is paid going from 1→2→4 slices
(−8.4 dB by N=4), and it **plateaus** from 4→8→16 (−8.0 to −8.75 dB) — squeezing
out that far more parallelism buys almost no additional quality cost beyond
N≈4, but also delivers comparatively little *additional* parallel speedup once
Amdahl's law (§3) is accounted for. **This makes N=4 a reasonable starting
point to implement and validate against**, not N=16: it captures most of the
achievable parallel speedup for the CAVLC-dominant regime while the PSNR hit
(−8.4 dB at this resolution/bitrate) is a real, visible cost that should be
weighed by whoever owns the quality/perf tradeoff for this project — this
document is not asserting that tradeoff is automatically worth it, only
reporting the real curve. A follow-up implementer/reviewer may reasonably pick
N=2 (smaller quality hit, less speedup) or re-run this sweep at the actual
target bitrate/resolution before committing to a number.

## 3. Realistic speedup estimate (Amdahl's law, grounded in real measurements)

**Real measured proportions**, `tools/perf_test.sh` with `BC250_PERF_STATS=1`,
60 frames, against an existing on-board build (pre-algorithmic-fix, i.e.
today's ~45 µs/MB CAVLC cost — this driver build predates the concurrent
CAVLC fix, so these numbers are the "before" baseline, not a claim about the
"after" state):

| Resolution | MBs/frame | GPU total (mean) | CPU CAVLC (mean) | Frame wall (mean) | CAVLC / wall |
|---|---:|---:|---:|---:|---:|
| 640×480  | 1200 | 0.459 ms | 54.534 ms | 55.465 ms | **98.3%** |
| 1280×720 | 3600 | 0.993 ms | 164.357 ms | 165.865 ms | **99.1%** |

Dividing CAVLC time by MB count: 54.534 ms / 1200 MB = **45.4 µs/MB** (640×480),
164.357 ms / 3600 MB = **45.7 µs/MB** (720p) — a flat per-MB cost independent of
resolution, exactly matching the task brief's "~45 microseconds/macroblock"
figure and confirming it's a per-MB algorithmic cost, not something that scales
with frame dimensions in a way that would already partially explain itself
(e.g. cache effects at larger working sets) — good corroboration that this is
the same bug the concurrent effort is chasing. GPU compute is 1.7–0.9% of frame
time, consistent with the task brief's "<1.5%" figure. This is today's reality:
**CAVLC is essentially the whole frame budget.**

**Amdahl's law**, with parallel fraction *p* = CAVLC's share of frame time, N =
worker threads, infinite-core ceiling = 1/(1−p):

- **Today (p ≈ 0.98, before the algorithmic fix):**
  - N=4: speedup ≈ 1/(0.02 + 0.98/4) ≈ **3.8×**
  - N=8: speedup ≈ 1/(0.02 + 0.98/8) ≈ **7.0×**
  - N=16: speedup ≈ 1/(0.02 + 0.98/16) ≈ **12.3×**
  - Ceiling (N→∞): 1/0.02 = **50×** (meaningless in practice — 16 real cores
    cap it at ~12×, and thread/join/false-sharing overhead will erode some of
    that further)

  This looks dramatic, but **it's the wrong number to plan around** — it's
  measuring parallelism against a CAVLC implementation that the other
  concurrent effort is specifically trying to make much faster per-MB. Once
  that fix lands, CAVLC's *absolute* time shrinks a lot, but so does its
  *share* of frame time only if the other, currently-tiny costs (GPU sync,
  slice header writing, NAL assembly/join, rate-control/DPB bookkeeping,
  malloc/free) stay fixed in absolute terms while CAVLC drops — which is
  exactly what will happen, since none of those are touched by the CAVLC fix.

- **Illustrative post-fix scenarios** (p = CAVLC share *after* the fix — these
  are deliberately hypothetical bracketing scenarios, not a prediction; a
  follow-up implementer should re-run `perf_test.sh` with `BC250_PERF_STATS=1`
  once the fix lands and plug the *real* post-fix p into this formula before
  deciding whether to build the threading layer at all):

  | Post-fix CAVLC share (p) | Ceiling (N→∞) | N=4 | N=8 | N=16 |
  |---:|---:|---:|---:|---:|
  | 0.98 (no change) | 50× | 3.8× | 7.0× | 12.3× |
  | 0.90 | 10× | 3.3× | 5.0× | 6.4× |
  | 0.75 | 4× | 2.5× | 3.1× | 3.5× |
  | 0.60 | 2.5× | 1.8× | 2.1× | 2.3× |
  | 0.40 | 1.67× | 1.4× | 1.5× | 1.6× |

  **The lesson**: if the algorithmic fix is very effective and drives CAVLC's
  share down toward 40–60% of frame time (i.e. GPU sync + NAL assembly + other
  fixed per-frame overhead becomes comparably sized to the now-much-smaller
  CAVLC cost), multicore CAVLC alone caps out around **1.5–2.5×** even with all
  16 cores thrown at it — a real, additive win, but nowhere near "16×", and on
  its own may not be enough to close the real-time gap at 1080p (the task
  brief's ~3 fps baseline needs ~10–20× to reach 30–60 fps; if the algorithmic
  fix delivers most of that already, multicore is the icing, not the cake). If
  instead the fix is more modest and CAVLC stays the 80–95%+ dominant cost,
  multicore remains a large, first-order lever (4–10× plausible at N=4–8).
  **This is the single most important open question for a follow-up
  implementer to resolate empirically before writing threading code** — the
  design in §2 is valid either way, but whether it's *worth building* depends
  entirely on where post-fix p actually lands.

- Quality-adjusted N: combining this with §2's quality curve, N=4 sits at a
  reasonable point on both curves (~85–90% of the N→∞ Amdahl ceiling already
  reached in every row above, and past most of the PSNR cliff) — reinforcing
  N=4 as the sensible first target over N=16, independent of which p scenario
  turns out to be real.

## 4. Correctness/determinism risks

This session's dominant lesson elsewhere in this project has been "verify
bit-exactness, don't assume" — the same discipline applies here:

1. **Slice-order-in-bitstream MUST match slice-order-in-frame, regardless of
   thread completion order.** This is the classic parallel-encoder bug the
   task brief calls out, and it is a real risk with the join design in §2: if
   the join loop iterates by *completion order* (e.g. a naive "whichever
   thread's output is ready first, append it next") instead of *slice index
   order* (`s = 0..num_slices-1`, always, deterministically), the resulting
   NAL sequence is a real correctness bug — silently wrong (a decoder may even
   fail to notice depending on `first_mb_in_slice` values, but the picture
   will be visibly scrambled: slice 2's macroblocks would end up decoded into
   slice 1's mb address range or vice versa). **Required property: the join
   step must index into an array of N per-slice result buffers by slice
   number, not consume a completion-ordered queue.** This is a one-line
   discipline (`results[s] = ...` inside each worker, not
   `results[next_free_slot++] = ...`), but it's exactly the kind of thing that
   looks fine in a quick implementation and is wrong.
2. **Determinism of the encode itself is unaffected** by threading, given the
   design in §2: each slice's CAVLC output depends only on that slice's own
   `(start_mb, end_mb)`, the shared read-only GPU staging buffers, and the
   shared `nz_luma/nz_cb/nz_cr` arrays restricted to that slice's own index
   range (never read cross-slice, per §1) — none of that is a function of
   thread scheduling, wall-clock timing, or which core a thread lands on. If
   the join order is correct (point 1), the byte-for-byte output should be
   identical to today's serial output for the same `num_slices` value, on
   every run.
3. **Required validation for whoever implements this**: a bit-exact diff
   against the current single-threaded output is the correct acceptance test,
   not just another `quality_test.sh` PSNR pass (PSNR passing only proves "not
   badly broken," not "identical to serial"). Concretely: run the same input
   clip through the pre-threading serial build and the post-threading parallel
   build at the *same* `BC250_SLICES_PER_FRAME` value, and `cmp` (or
   `diff <(xxd ...)`) the two `.mp4`/raw-NAL outputs byte-for-byte. They should
   be **identical** — any difference at a fixed slice count is a bug in the
   threading layer, not an expected/tolerable variance, since nothing about
   going parallel should change what bits get written, only how fast they get
   written. (`tests/test_encode.c`'s existing `EncodeBitstreamTest` CTest
   target is the natural place to add this as an automated regression check —
   run the encode twice, once forcing 1 worker thread and once forcing the
   real worker count at the same slice count, assert the outputs match.)
4. **Minor secondary risk, not a blocker**: `slice_rbsp = malloc(...)` /
   `free(slice_rbsp)` per slice per frame, called concurrently from N threads.
   glibc's malloc is thread-safe (per-thread arenas), so this is correct as-is,
   but N concurrent mallocs/frees of similarly-sized buffers every frame is a
   plausible (unmeasured) contention point at high frame rates once CAVLC
   itself is fast — worth profiling post-implementation, and an easy fix if it
   shows up (pre-allocate one `slice_rbsp`-sized buffer per pool worker, reused
   frame to frame, sized once at pool-creation time using the same `(end_mb -
   start_mb) * 64 + 4096` formula the code already uses).

## 5. Minimal first implementation step (for a follow-up coding agent)

**Preconditions before starting**: the concurrent CAVLC algorithmic fix (in
`cavlc.c`/`encoder_h264.c`) has landed and merged to `main`. Re-run
`tools/perf_test.sh` with `BC250_PERF_STATS=1` first and recompute real p
(CAVLC share of frame time) per §3's table — this determines whether the work
below is worth doing at all, or whether the realistic ceiling (e.g. ~1.5–2×) is
judged not worth the added complexity by whoever owns that tradeoff.

**Files to touch** (only these — `cavlc.c` and `bitstream.c` need zero changes,
per §1's "already pure functions" finding):

- `approach1-compute-encoder/src/encoder_h264.c`:
  - Add a small thread-pool struct (N `pthread_t` workers + a simple
    per-worker job/result slot — no need for a general work-queue since the
    job shape is fixed: "encode slice s of frame f") to `struct h264_encoder`,
    created in `h264_encoder_create()` sized `min(16, nproc)` or a
    `BC250_CAVLC_THREADS` env var (mirroring the existing `BC250_*` debug-var
    convention already used throughout this file) defaulting to something
    like 4, and torn down in an `h264_encoder_destroy()` (check whether that
    function exists yet — if not, this is also the natural place to add clean
    shutdown, since a persistent pool needs one).
  - Extract the per-slice body of the existing `for (int s = 0; s < num_slices;
    s++)` loop (both in `h264_encoder_encode_frame()` and its
    `h264_encoder_encode_raw()` twin — the two loops are near-duplicates today,
    worth extracting to one shared `encode_one_slice(...)` function as part of
    this change regardless of threading, since it removes drift risk between
    the two copies) into a function taking `(encoder, s, start_mb, end_mb,
    quant_levels, coeff, pred_modes, mvs, qp, is_idr, deblock_idc, slice_type,
    poc_bits, slice_qp_delta, out_ebsp_buf, out_ebsp_len)` — everything the
    body currently closes over, made explicit.
  - Dispatch `min(num_slices, pool_size)` of these to worker threads (if
    `num_slices` > pool size, workers pick up more than one slice each — simple
    static assignment, e.g. `slice s` always goes to `worker s % pool_size`, is
    sufficient given slices are already-equal-sized-ish work units; a dynamic
    queue is unnecessary complexity for N ≤ 16 fixed-shape jobs).
  - Replace the current inline "compute `nal_dst` at running `total_written`,
    call `bs_rbsp_to_ebsp` straight into `output_buf`" with: each worker writes
    into its **own** pre-sized EBSP buffer (`results[s].buf`,
    `results[s].len`), then after `pthread_join`, a serial loop `for (s = 0; s
    < num_slices; s++)` (index order, **not** completion order — see §4 point
    1) appends `results[s]`'s NAL bytes to `encoder->output_buf` exactly as the
    current inline code does, advancing `total_written`.
- `approach1-compute-encoder/tests/test_encode.c` (or a new
  `test_encode_threaded.c` added to `tests/CMakeLists.txt`): the bit-exactness
  regression described in §4 point 3 — encode the same deterministic input at
  a fixed `BC250_SLICES_PER_FRAME` once with the pool forced to 1 worker and
  once with the real worker count, assert the output buffers are byte-identical.

**Threading primitive**: plain **POSIX `pthread_create`/`pthread_join`**
(already linked, per §2) — no need for a heavier abstraction
(`std::thread`/a thread-pool library) given this is C, the job count is small
and fixed-shape per frame (≤16), and the pool is long-lived (created once, not
per-frame). A simple array of `pthread_t` + a per-worker `struct { int slice;
... } job` passed via `pthread_create`'s `arg`, with `pthread_join` called on
exactly the workers that were dispatched this frame, is sufficient — no
condvar/queue machinery needed unless a later iteration wants to avoid the
(cheap, but real) per-frame `pthread_create` cost by keeping workers alive
across frames and signaling new work via a condvar; that's a reasonable *second*
step once the basic per-frame-spawn version is validated bit-exact and its
actual overhead is measured, not something to build speculatively up front.

**Validation checklist for the follow-up implementer**:
1. `tests/test_encode.c`'s new bit-exactness check passes (1 worker vs. N
   workers, identical output, at every `BC250_SLICES_PER_FRAME` value 1/2/4/8/16).
2. `tools/quality_test.sh` still reports the same PSNR/SSIM numbers as this
   document's §2 table at each slice count (threading must not change quality —
   only speed — so any PSNR delta vs. this document's baseline numbers means
   something in the threading change altered actual encoding behavior, which
   would itself be a bug to chase down, not a threading-perf question anymore).
3. `tools/perf_test.sh` with `BC250_PERF_STATS=1` re-measures CAVLC ms/frame
   and total wall ms/frame at `BC250_SLICES_PER_FRAME=4` (or whatever N was
   chosen) vs. `=1`, and the resulting real speedup is compared against this
   document's §3 Amdahl prediction for the actual post-fix p measured at that
   time — if real speedup is well below the Amdahl ceiling for that p, that's
   a signal of thread/join/false-sharing overhead worth profiling before
   declaring the lever exhausted.

---

## Appendix: raw measurement logs

Measurements were taken against pre-existing, already-built driver binaries on
the board (`/var/home/user/bc250-perf-test/.../build/bc250_drv_video.so` for
§3's `perf_test.sh` run and §2's `quality_test.sh` slice sweep — this build's
`slices=1` result, 37.66 dB / SSIM 0.9918, matches the `quality_test.sh`
top-of-file comment's own recorded board-validated baseline of "37.7 dB / SSIM
0.99, PASS," confirming this build is representative of main's
correctness-fixed state, not a stale/broken snapshot). Nothing in any
`bc250-*` worktree was modified to take these measurements — `SKIP_BUILD=1`
was used throughout, pointing `tools/perf_test.sh`/`tools/quality_test.sh`
(copied from this branch, unmodified) at those existing build directories as
read-only inputs. Full stdout is preserved in this repo's commit history for
this file's authoring session if a re-check is needed; the summarized numbers
above are the complete relevant content.
