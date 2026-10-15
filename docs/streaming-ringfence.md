# Ring-fencing the GPU for the encoder

This answers two questions that both come up the first time someone streams a
game on the BC-250 with this driver:

1. *Can I reserve some Compute Units for the encoder, so the game can't take
   the whole GPU?* Yes, with a caveat about granularity that matters more than
   the caveat usually gets.
2. *Why does the stream still spike under load, even with that?* Because the
   encoder's problem under a saturating game is mostly **waiting to be
   scheduled**, not a shortage of CUs. See "What this will not fix" below.

Everything here is measured against the driver's own log
(`BC250_PERF_STATS=1`, `BC250_GOVERNOR_STATS=1`) and its DEVLOG, which is
where the numbers come from. Nothing on this page is a guess.

## 1. What there is, and what there isn't

| Mechanism | Available here? |
| :--- | :--- |
| `VK_AMD_shader_core_policy` (`vkCmdSetShaderCorePolicyAMD`, `AMD_SHADER_CORE_POLICY_FLAG_CU_MASK`) | **No.** This is the vendor extension that would let a Vulkan app pin its own dispatches to a CU subset, and RADV does not expose it: Mesa's `radv_physical_device_get_supported_extensions()` lists `AMD_shader_core_properties` and `AMD_shader_core_properties2`, and no `shader_core_policy`. Calling it would be code that can never run. |
| `VK_EXT_global_priority` / `VK_KHR_global_priority` | Yes, already implemented here as `BC250_QUEUE_PRIORITY=`, but HIGH and REALTIME are refused to an unprivileged process (`VK_ERROR_NOT_PERMITTED_KHR`) - it needs `CAP_SYS_NICE`. And it reorders service; it does not partition resources. |
| Mesa `AMD_CU_MASK` | **Yes. This is the mechanism.** Per-process, since Mesa 22.0, for both radeonsi and RADV. Parsed by `set_custom_cu_en_mask()` in `src/amd/common/ac_gpu_info.c`. |
| `amdgpu.disable_cu=se.sh.cu,...` | Kernel module parameter; disables those CUs **for every process on the machine**, including the encoder. Not a fence, a sacrifice. |
| gamescope `--amdgpu-cus` | Does not exist. (Checked against gamescope's current `gamescope_options` table; it has `--rt` and `CAP_SYS_NICE`, but no CU mask option. It is worth knowing that `--rt` *raises the game's* priority, which is the opposite of what you want here.) |

So: two processes, two environment variables, applied before either starts.

## 2. The granularity trap

`AMD_CU_MASK` is a mask **within one shader array**, and Mesa applies the same
mask to **every** array on the part:

```
 *   ID = [0-9][0-9]*                         ex. base 10 numbers
 *   ID_list = (ID | ID-ID)[, (ID | ID-ID)]*  ex. 0,2-4,7
 *   CU_list = 0x[0-F]* | ID_list             ex. 0x337F OR 0,2-4,7
 *   AMD_CU_MASK = CU_list
 *
 * It's a CU mask within a shader array. It's applied to all shader arrays.
```

It is also a list of CUs to **enable**, not to disable.

The consequence: "ring-fence 2 CUs for the encoder" means *two CUs in each
shader array*. On a part with 8 arrays that is 16 CUs, not 2. There is no way
to name two specific CUs out of forty through this interface, and a doc that
implies otherwise is wrong.

### Mesa will reject some masks, silently

`set_custom_cu_en_mask()` refuses a mask that:

* enables no CU in an array at all;
* enables none of CU0/CU1/CU3/CU4 - the SPI late-alloc hardware constraint;
* enables none of CU2/CU3 - the PS late-alloc hardware constraint.

A rejected mask is reported on the driver's stderr and then **ignored** - the
process gets all CUs. So a typo does not degrade, it does nothing while
looking like it worked. Both `apply_gpu_ringfence.sh` and the driver's own
`BC250_CU_REPORT=1` check the mask against these rules before recommending it.

**This rules out the obvious approach.** CU2 and CU3 are required in *every*
mask, and a contiguous range can only put them in one half. So:

* a contiguous split is **never** legal, on either side;
* neither half can be smaller than **2 CUs**, since each needs at least one of
  CU2/CU3 and at least one of the others;
* on a part with fewer than 4 CUs per array, no legal split exists at all.

The masks are therefore found by *search* over CU sets, not by taking "the top
N" - which is why both the driver and the script contain a small
`bc250_pick_cu_split()` / `find_split()` rather than some arithmetic. On a
5-CU array the answer is `0,2` for the encoder and `1,3-4` for the game, not
`3,4` and `0,2`. On a part where no legal split exists, both say so plainly
instead of emitting a mask Mesa would throw away.

## 3. Doing it

```bash
# One-time. Reports the topology, writes both halves, restarts Sunshine.
./tools/sunshine_preset/apply_gpu_ringfence.sh

# See the plan first, change nothing:
./tools/sunshine_preset/apply_gpu_ringfence.sh --dry-run

# Fence two CUs per array instead of the default, or hand back a different
# number: --cus-per-array N
#
# Skip the queue-priority half if you do not want CAP_SYS_NICE on Sunshine:
#   --no-priority
#
# Undo everything:
./tools/sunshine_preset/apply_gpu_ringfence.sh --remove
```

It writes:

* `~/.config/systemd/user/sunshine.service.d/bc250-gpu-ringfence.conf` -
  `Environment=AMD_CU_MASK=<top K per array>`, plus `BC250_QUEUE_PRIORITY=high`
  and `AmbientCapabilities=CAP_SYS_NICE` (and the matching
  `sudo setcap cap_sys_nice+ep /usr/bin/sunshine`).
* `~/.local/bin/bc250-cu-masked-game` - a wrapper that sets the *complementary*
  mask and execs the game. Put it in the game's Steam launch options:
  `bc250-cu-masked-game %command%`.

Mesa reads the variable when it creates a screen, which is why the drop-in has
to be in place before Sunshine starts and cannot be set from inside the driver.

### Reading the topology yourself

```bash
BC250_CU_REPORT=1 vainfo        # any VA-API/Vulkan client will do
```

```
[bc250-gpu] CU topology: 8 SE x 1 SA x 5 CU/SA = 40 CUs (wave64)
[bc250-gpu]   ring-fence: encode-> AMD_CU_MASK=0,2   (2 of 5 CU/SA, 16 CUs total)
[bc250-gpu]   ring-fence: game  -> AMD_CU_MASK=1,3-4 (3 of 5 CU/SA, 24 CUs total)
[bc250-gpu]   Mesa requires in EVERY mask: at least 1 CU, at least one of ...
```

(`BC250_RINGFENCE_CUS_PER_SA=N` changes the K the report aims for; the search
takes the nearest legal size at or above it.)

The 40-CU BC-250 works out at 16 CUs for the encoder, not 2 - and that is the
interface's granularity talking, not a driver choice. "Two CUs" through this
interface is always "two per shader array".

## 4. What this will not fix

Two of the three things a live stream loses under game load are not CU
shortage, and no amount of fencing touches them. This is the part worth being
straight about, because the numbers say so.

**The submission wait.** Under `ffmpeg --load=gpu`, a 675 ms frame of which
this encoder's shaders *execute* for 2.34 ms: about 646 ms is the submission
waiting its turn behind the other process's work (DEVLOG §24.6). The GPU
timestamps measure only our command buffer running, which is why "GPU time"
*appears* to drop under load - it is being serviced less, not working less.
Fencing the game narrows how much of the part it can occupy at once, which
shortens that wait. It cannot remove it, and it cannot make 646 ms of queueing
become zero.

**The CPU side.** DEVLOG §24.6 measures `cavlc` alone going 7.1 ms → 24.0 ms
with no code change, purely because a saturating game is saturating the shared
memory bus this encoder is cache-bound on (§20.4). Thread priority and CU
fencing do not change that. The 7 ms of entropy coding is the other thing a
frame's budget is spent on, and it is the largest single component of an
idle-GPU frame.

**What is actually helped:** the encoder no longer has to share every CU with
the game, so its own dispatch stops being preempted mid-wave and stops losing
races for CU slots. That is a real effect, it is the effect you asked for, and
on a part with 40 CUs it is worth having. It is just worth much less than
"the game can no longer affect the stream at all".

### Do not use `--rt`

`gamescope --rt` sets realtime scheduling for the compositor and the game
process, and gamescope lowers its own nice value to -20 when it has
`CAP_SYS_NICE`. That is the wrong direction for this problem: it makes the
game win every scheduling contest it is in, on both CPU and GPU. If you are
using it for other reasons, take `--cus-per-array` over it.

## 5. Verifying it took effect

```bash
# Sunshine's environment:
systemctl --user show sunshine -p Environment

# The game's:
#   launch the game through the wrapper, then in the game:
cat /proc/$(pgrep -n <game>)/environ | tr '\0' '\n' | grep AMD_CU_MASK

# The encoder's own view, once per start:
BC250_CU_REPORT=1 BC250_GOVERNOR_STATS=1 sunshine
```

If Mesa rejected the mask you will see it on the game's stderr:

```
amd: invalid AMD_CU_MASK: at least 1 CU from 0x1f per SA must be enabled (SPI limitation)
```

and the process will be running on all CUs.
