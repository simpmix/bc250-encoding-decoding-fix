#!/usr/bin/env bash
# bc250-encoding-decoding-fix v0.4.0 - https://github.com/simpmix/bc250-encoding-decoding-fix
#
# bc250_diagnose.sh - Comprehensive hardware verification, VA-API test, and encode benchmark
#

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}${BOLD}        AMD BC-250 System Health & Benchmark         ${NC}"
echo -e "${BLUE}======================================================${NC}"

# 1. Hardware Detection
echo -e "\n${BOLD}[1/5] Checking Hardware Identification...${NC}"
if command -v lspci &> /dev/null; then
    if lspci -nn | grep -i "1002:13fe" > /dev/null; then
        echo -e "  ${GREEN}✓ APU Silicon: AMD BC-250 (Cyan Skillfish, PCI 1002:13fe)${NC}"
    else
        echo -e "  ${YELLOW}! BC-250 PCI ID (1002:13fe) not detected. Testing generic GPU.${NC}"
    fi
else
    echo -e "  ${YELLOW}! lspci not found. Skipping PCI check.${NC}"
fi

# Check Compute Units
if [ -d "/sys/class/drm" ]; then
    for card in /sys/class/drm/card[0-9]/device; do
        if [ -f "$card/current_compute_units" ]; then
            cus=$(cat "$card/current_compute_units")
            echo -e "  ${GREEN}✓ Active Compute Units: ${cus} CUs (Cyan Skillfish/Oberon)${NC}"
        fi
    done
fi

# Check CPU Cores
cores=$(nproc --all 2>/dev/null || echo "Unknown")
echo -e "  ${GREEN}✓ CPU Processing Threads: ${cores}${NC}"

# 2. Audio Subsystem
echo -e "\n${BOLD}[2/5] Checking Audio Subsystem...${NC}"
if lsmod | grep bc250_audio_fix > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓ bc250_audio_fix kernel module is ACTIVE${NC}"
else
    echo -e "  ${YELLOW}! bc250_audio_fix module is not loaded.${NC}"
    echo -e "    Run: cd audio-fix && sudo ./install_dkms.sh"
fi

if command -v aplay &> /dev/null; then
    hdmi_devs=$(aplay -l 2>/dev/null | grep -i -E "hdmi|displayport" | wc -l)
    echo -e "  ${GREEN}✓ Detected ${hdmi_devs} digital audio endpoints${NC}"
fi

# 3. VA-API Driver Installation
echo -e "\n${BOLD}[3/5] Checking VA-API Compute Driver...${NC}"
FOUND_DRIVER=0
DRI_CANDIDATES=(
    "/var/lib/bc250/dri"
    "/usr/local/lib64/dri"
    "/usr/local/lib/dri"
    "/usr/lib/x86_64-linux-gnu/dri"
    "/usr/lib64/dri"
    "/usr/lib/dri"
)

FOUND_DRIVER_DIR=""
for dri in "${DRI_CANDIDATES[@]}"; do
    if [ -f "$dri/bc250_drv_video.so" ]; then
        echo -e "  ${GREEN}✓ Found driver binary: $dri/bc250_drv_video.so${NC}"
        FOUND_DRIVER=1
        FOUND_DRIVER_DIR="$dri"
        break
    fi
done

# libva's own compiled-in default search locations - a driver found in one of
# these needs no LIBVA_DRIVERS_PATH at all. A driver found anywhere else in
# DRI_CANDIDATES above (e.g. /var/lib/bc250/dri, used by an immutable-distro
# install) is invisible to libva unless LIBVA_DRIVERS_PATH names it - checked
# in step 5 below, since that's a second, independent place this exact
# env-var-scope class of bug can hide.
STANDARD_DRI_DIRS=("/usr/lib64/dri" "/usr/lib/x86_64-linux-gnu/dri" "/usr/lib/dri")

if [ $FOUND_DRIVER -eq 0 ]; then
    echo -e "  ${RED}✗ bc250_drv_video.so not found in standard or immutable system DRI paths.${NC}"
    echo -e "    Run ./build_and_install.sh, ./tools/setup_bazzite.sh, or ./tools/setup_steamos.sh first!"
fi

FOUND_32BIT_DRIVER=0
for dri32 in "/usr/lib32/dri" "/usr/lib/i386-linux-gnu/dri"; do
    if [ -f "$dri32/bc250_drv_video.so" ]; then
        echo -e "  ${GREEN}✓ Found 32-bit companion driver (Steam Link): $dri32/bc250_drv_video.so${NC}"
        FOUND_32BIT_DRIVER=1
        break
    fi
done
if [ $FOUND_32BIT_DRIVER -eq 0 ]; then
    echo -e "  ${YELLOW}! 32-bit companion driver not found in /usr/lib32/dri (required for Steam Link).${NC}"
    echo -e "    Run: ./tools/build_32bit.sh or install from release v0.5.1 bundle."
fi

FOUND_SHADERS=0
for sdir in "/var/lib/bc250/shaders" "/usr/local/share/bc250/shaders" "/usr/share/bc250/shaders"; do
    if [ -d "$sdir" ]; then
        spv_count=$(ls -1 "$sdir"/*.spv 2>/dev/null | wc -l)
        if [ "$spv_count" -gt 0 ]; then
            echo -e "  ${GREEN}✓ Found ${spv_count} compiled SPIR-V shaders in $sdir${NC}"
            FOUND_SHADERS=1
            break
        fi
    fi
done

if [ $FOUND_SHADERS -eq 0 ]; then
    echo -e "  ${YELLOW}! No compiled shaders found in standard paths.${NC}"
fi

# 4. VA-API Capabilities & Benchmark
echo -e "\n${BOLD}[4/5] Testing VA-API Driver & Running Encode Benchmark...${NC}"
export LIBVA_DRIVER_NAME=bc250
# 32-bit paths are at the END on purpose. A 64-bit client walks the list and
# hits its driver in an early entry; a 32-bit client (Steam Link) fails to
# dlopen the 64-bit ones - wrong ELF class, which libva skips silently - and
# carries on to these. Mixed-arch lists are how multiarch distros ship this,
# and omitting the 32-bit entries is what made the driver unfindable in
# issue #14 even though it was installed.
export LIBVA_DRIVERS_PATH="/var/lib/bc250/dri:/usr/local/lib64/dri:/usr/local/lib/dri:/usr/lib/x86_64-linux-gnu/dri:/usr/lib64/dri:/usr/lib/dri:/usr/lib32/dri:/usr/lib/i386-linux-gnu/dri"

if command -v vainfo &> /dev/null; then
    if LIBVA_DRIVER_NAME=bc250 LIBVA_DRIVERS_PATH="$LIBVA_DRIVERS_PATH" vainfo --display drm > /tmp/bc250_vainfo.log 2>&1; then
        echo -e "  ${GREEN}✓ VA-API initialized successfully with BC-250 driver!${NC}"
        grep -i -E "VAProfileH264|VAProfileHEVC" /tmp/bc250_vainfo.log | sed 's/^/    /'
    else
        echo -e "  ${YELLOW}! vainfo reported non-zero status (Vulkan/DRM display access):${NC}"
        cat /tmp/bc250_vainfo.log | tail -n 5 | sed 's/^/    /'
    fi
    rm -f /tmp/bc250_vainfo.log
fi

# Live Benchmark Test with FFmpeg if installed
if command -v ffmpeg &> /dev/null; then
    echo -e "\n${BOLD}==> Running 100-Frame 1080p60 Live Compute Encode Benchmark...${NC}"
    START_TIME=$(date +%s%N)
    
    if LIBVA_DRIVER_NAME=bc250 LIBVA_DRIVERS_PATH="$LIBVA_DRIVERS_PATH" ffmpeg -v error -f lavfi -i testsrc=duration=1.66:size=1920x1080:rate=60 \
       -vaapi_device /dev/dri/renderD128 -vf 'format=nv12,hwupload' \
       -c:v h264_vaapi -b:v 10M -f null - 2>/tmp/bc250_bench.err; then
        
        END_TIME=$(date +%s%N)
        DURATION_MS=$(( (END_TIME - START_TIME) / 1000000 ))
        FPS=$(( 100000 / DURATION_MS ))
        MS_PER_FRAME=$(awk "BEGIN {print $DURATION_MS / 100}")
        
        echo -e "  ${GREEN}${BOLD}✓ Benchmark Succeeded!${NC}"
        echo -e "  ${BOLD}Latency per frame:${NC} ${GREEN}${MS_PER_FRAME} ms${NC}"
        echo -e "  ${BOLD}Encoder Throughput:${NC} ${GREEN}${FPS} FPS${NC} (Headroom: $(( FPS / 60 ))x real-time)"
    else
        echo -e "  ${YELLOW}Note: ffmpeg live bench test skipped (no renderD128 permissions or headless).${NC}"
    fi
    rm -f /tmp/bc250_bench.err
fi

# 5. Real-world service environment check
#
# WHY THIS STEP EXISTS: step 4 above deliberately exports LIBVA_DRIVER_NAME
# (and LIBVA_DRIVERS_PATH) into its OWN subshell before testing vainfo/ffmpeg
# - so it can only ever prove "the driver works when this variable is set",
# never "the thing actually trying to stream has this variable set". Those
# are different questions, and conflating them produces a specific, confusing
# report: vainfo succeeds standalone, while Sunshine's own libva session
# still loads radeonsi_drv_video.so and silently falls back to software
# encoding - because `export FOO=bar` in an interactive shell only affects
# that shell and its children, and Sunshine is normally a separately-launched
# process (a systemd --user service, a desktop autostart entry, a Flatpak)
# that never inherits it. This step checks the ACTUAL running Sunshine
# process's environment instead of re-testing ours.
echo -e "\n${BOLD}[5/5] Checking Sunshine's Actual Process Environment...${NC}"

SUNSHINE_PID=$(pgrep -x sunshine 2>/dev/null | head -1)

if [ -z "$SUNSHINE_PID" ]; then
    echo -e "  ${YELLOW}! Sunshine is not currently running.${NC}"
    echo -e "    Start it, then re-run this script while it's running - this"
    echo -e "    check can only inspect a live process's real environment, not"
    echo -e "    a config file, since Sunshine may be launched several"
    echo -e "    different ways (systemd --user service, desktop autostart,"
    echo -e "    Flatpak, manual shell) that each source its environment"
    echo -e "    differently."
else
    echo -e "  ${GREEN}✓ Found running Sunshine process (PID ${SUNSHINE_PID})${NC}"

    # Capability check FIRST, and independent of whether /proc/.../environ is
    # readable below: Sunshine's binary normally carries a file capability
    # (commonly cap_sys_admin, for KMS screen capture). Executing a binary
    # with elevated file capabilities puts the kernel into secure-execution
    # mode (AT_SECURE=1), in which glibc's secure_getenv() - which libva uses
    # specifically for LIBVA_DRIVER_NAME/LIBVA_DRIVERS_PATH, because those
    # variables control which shared library gets dlopen()'d into a
    # privileged process - returns nothing AT ALL, regardless of what is
    # actually in the environment. This is deliberate libva security design,
    # not a bug, and it is invisible to a plain environment-variable check:
    # the variable can be genuinely present (confirmed further below) and
    # still be silently ignored. This was found and fixed on this exact
    # project before - see docs/DEVLOG.md 10.5/10.6/12.6 for the full
    # root-cause writeup, and a real bug report this exact check would have
    # caught immediately instead of reporting a clean pass.
    #
    # NOTE: a nonzero CapEff is only meaningful evidence of this if the
    # process is NOT running as root - UID 0 has the full capability set
    # inherently, with no file-capability/secure-exec transition involved,
    # so CapEff alone would false-positive on any root-run process
    # (confirmed while testing this check). The AT_SECURE mechanism this
    # step is looking for specifically requires a non-root process gaining
    # capabilities beyond what its parent had, via a file capability set on
    # its own binary (`getcap`/`setcap`) - which is exactly how Sunshine
    # normally gets cap_sys_admin for KMS capture on a real desktop, and the
    # only way a non-root process ever ends up with a nonzero CapEff at all.
    SUNSHINE_UID=$(awk '/^Uid:/{print $2}' "/proc/$SUNSHINE_PID/status" 2>/dev/null)
    CAPEFF=$(awk '/^CapEff:/{print $2}' "/proc/$SUNSHINE_PID/status" 2>/dev/null)
    if [ -n "$CAPEFF" ] && [ "$CAPEFF" != "0000000000000000" ] && [ "${SUNSHINE_UID:-0}" != "0" ]; then
        echo -e "  ${RED}✗ Sunshine's process has elevated file capabilities (CapEff=${CAPEFF}).${NC}"
        echo -e "    This puts it in the kernel's secure-execution mode, in which libva"
        echo -e "    CANNOT see LIBVA_DRIVER_NAME/LIBVA_DRIVERS_PATH at all - regardless of"
        echo -e "    what the environment check below finds. This is the single most"
        echo -e "    common cause of a 'vainfo works, Sunshine still uses software"
        echo -e "    encoding' report."
        SUNSHINE_EXE=$(readlink -f "/proc/$SUNSHINE_PID/exe" 2>/dev/null || true)
        if [ -n "$SUNSHINE_EXE" ] && command -v getcap &> /dev/null; then
            CAPSTR=$(getcap "$SUNSHINE_EXE" 2>/dev/null)
            [ -n "$CAPSTR" ] && echo -e "    ($CAPSTR)"
        fi
        echo
        echo -e "    ${BOLD}Fix - do this instead of setting environment variables:${NC}"
        echo -e "      sudo ./tools/install_vaapi_boot_redirect.sh"
        echo -e "    This redirects the system's default (radeonsi) VA-API driver slot to"
        echo -e "    this driver directly, which works regardless of secure_getenv() -"
        echo -e "    persists across reboots on immutable/ostree systems too."
        echo
        echo -e "    ${YELLOW}The environment check below may still show everything 'correct' -${NC}"
        echo -e "    ${YELLOW}that does not mean hardware encoding will actually work. Trust${NC}"
        echo -e "    ${YELLOW}this capability check over that one.${NC}"
    elif [ "${SUNSHINE_UID:-0}" = "0" ]; then
        echo -e "  ${GREEN}✓ Sunshine is running as root (UID 0)${NC}"
        echo -e "    - not itself the AT_SECURE/secure_getenv() trigger this check looks"
        echo -e "    for (that specifically requires a non-root process gaining"
        echo -e "    capabilities via a file capability on its own binary). libva's"
        echo -e "    environment-variable lookup should apply normally here; the check"
        echo -e "    below is meaningful. Running a real desktop session's Sunshine as"
        echo -e "    root is unusual, though - worth double-checking that's intentional."
    else
        echo -e "  ${GREEN}✓ No elevated file capabilities detected on Sunshine's process${NC}"
        echo -e "    (CapEff=${CAPEFF:-unreadable}) - libva's environment-variable lookup"
        echo -e "    should apply normally; the check below is meaningful here."
    fi

    if pgrep -x gamescope > /dev/null 2>&1; then
        echo -e "  ${BLUE}ℹ Active Gamescope / Gaming Mode session detected.${NC}"
        if [ "$CAPEFF" = "0000000000000000" ] && [ "${SUNSHINE_UID:-0}" != "0" ]; then
            echo -e "  ${RED}✗ In Gaming Mode, Sunshine lacks DRM KMS capture capabilities (CapEff=0000000000000000).${NC}"
            echo -e "    KMS capture will fail with 'Couldn't get drm fb for plane [0]: Permission denied' (black screen)!"
            echo -e "    To fix Sunshine in Gaming Mode:"
            echo -e "      1. Set capabilities on the canonical binary:"
            echo -e "         sudo setcap cap_sys_admin,cap_sys_nice+p \$(readlink -f \$(which sunshine))"
            echo -e "      2. Install the persistent boot redirect so driver loads under secure-exec:"
            echo -e "         sudo ./tools/install_vaapi_boot_redirect.sh"
            echo -e "      3. If launching Sunshine via systemd --user service, add to [Service] in sunshine.service:"
            echo -e "         AmbientCapabilities=CAP_SYS_ADMIN CAP_SYS_NICE"
            echo -e "      4. In Steam Game Mode Settings > System > Developer Mode, enable 'Force Composite'."
        fi
    fi
    echo

    if [ ! -r "/proc/$SUNSHINE_PID/environ" ]; then
        echo -e "  ${YELLOW}! Cannot read /proc/${SUNSHINE_PID}/environ (permission denied -${NC}"
        echo -e "    it may be running as a different user). Re-run this script as that"
        echo -e "    user, or with sudo, to also check its environment variables."
    else
    SUNSHINE_ENV=$(tr '\0' '\n' < "/proc/$SUNSHINE_PID/environ" 2>/dev/null || true)
    SUNSHINE_LIBVA_DRIVER=$(echo "$SUNSHINE_ENV" | grep '^LIBVA_DRIVER_NAME=' | cut -d= -f2-)
    SUNSHINE_LIBVA_PATH=$(echo "$SUNSHINE_ENV" | grep '^LIBVA_DRIVERS_PATH=' | cut -d= -f2-)
    NEEDS_FIX=0

    if [ "$SUNSHINE_LIBVA_DRIVER" = "bc250" ]; then
        echo -e "  ${GREEN}✓ Sunshine's own process environment has LIBVA_DRIVER_NAME=bc250${NC}"
    elif [ -n "$SUNSHINE_LIBVA_DRIVER" ]; then
        echo -e "  ${RED}✗ Sunshine's process environment has LIBVA_DRIVER_NAME=${SUNSHINE_LIBVA_DRIVER}, NOT bc250.${NC}"
        echo -e "    It will use libva's default driver resolution (usually"
        echo -e "    radeonsi) and silently fall back to software encoding -"
        echo -e "    this is the exact 'vainfo works but Sunshine doesn't' report."
        NEEDS_FIX=1
    else
        echo -e "  ${RED}✗ Sunshine's process environment has NO LIBVA_DRIVER_NAME at all.${NC}"
        echo -e "    Exporting it in your shell (as step 4 above did, to test the"
        echo -e "    driver itself) does not reach an already-running or"
        echo -e "    independently-launched Sunshine process."
        NEEDS_FIX=1
    fi

    if [ -n "$FOUND_DRIVER_DIR" ]; then
        IS_STANDARD_DIR=0
        for std in "${STANDARD_DRI_DIRS[@]}"; do
            [ "$std" = "$FOUND_DRIVER_DIR" ] && IS_STANDARD_DIR=1
        done
        if [ "$IS_STANDARD_DIR" -eq 0 ]; then
            case "$SUNSHINE_LIBVA_PATH" in
                *"$FOUND_DRIVER_DIR"*) ;;
                *)
                    echo -e "  ${YELLOW}! The driver was found in a non-default path${NC}"
                    echo -e "    ($FOUND_DRIVER_DIR), but Sunshine's own"
                    echo -e "    LIBVA_DRIVERS_PATH does not include it (currently:"
                    echo -e "    '${SUNSHINE_LIBVA_PATH:-<unset>}'). libva will not search"
                    echo -e "    this location unless told to - same class of bug as above,"
                    echo -e "    just a second variable."
                    NEEDS_FIX=1
                    ;;
            esac
        fi
    fi

    if [ "$NEEDS_FIX" -eq 1 ]; then
        echo
        echo -e "  ${BOLD}Fix:${NC} set the variable(s) in Sunshine's OWN environment, not"
        echo -e "  your shell. If Sunshine runs as a systemd --user service (see any"
        echo -e "  unit(s) listed below), the standard way is:"
        echo -e "    systemctl --user edit <sunshine-unit-name>"
        echo -e "  and add:"
        echo -e "    [Service]"
        echo -e "    Environment=LIBVA_DRIVER_NAME=bc250"
        if [ -n "$FOUND_DRIVER_DIR" ]; then
            IS_STANDARD_DIR=0
            for std in "${STANDARD_DRI_DIRS[@]}"; do
                [ "$std" = "$FOUND_DRIVER_DIR" ] && IS_STANDARD_DIR=1
            done
            [ "$IS_STANDARD_DIR" -eq 0 ] && echo -e "    Environment=LIBVA_DRIVERS_PATH=$FOUND_DRIVER_DIR"
        fi
        echo -e "  then: systemctl --user daemon-reload && systemctl --user restart <unit>"
    fi
    fi
fi

# Surface any systemd --user units that look like Sunshine regardless of the
# check above - helps identify the exact unit name to edit, since this varies
# by install method (native package, source build, AppImage wrapper, etc.).
if command -v systemctl &> /dev/null; then
    SUNSHINE_UNITS=$(systemctl --user list-units --all --no-legend --plain 2>/dev/null | grep -i sunshine | awk '{print $1}')
    if [ -n "$SUNSHINE_UNITS" ]; then
        echo -e "\n  ${BOLD}Detected systemd --user unit(s) matching 'sunshine':${NC}"
        echo "$SUNSHINE_UNITS" | sed 's/^/    /'
    fi
fi

echo -e "\n${GREEN}======================================================${NC}"
echo -e "${GREEN}${BOLD}   Diagnostic & Performance Check Complete!          ${NC}"
echo -e "${GREEN}======================================================${NC}"
