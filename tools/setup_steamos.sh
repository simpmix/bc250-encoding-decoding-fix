#!/usr/bin/env bash
# bc250-encoding-decoding-fix v0.4.0 - https://github.com/simpmix/bc250-encoding-decoding-fix
#
# setup_steamos.sh - Automated installer tailored specifically for Valve SteamOS & HoloISO
#
# Designed to survive SteamOS A/B system updates by staging to persistent storage (/var/lib/bc250)
# and integrating seamlessly with Gamescope and Sunshine game streaming.
#

set -e

WITH_AUDIO_FIX=0
for arg in "$@"; do
    case "$arg" in
        --with-audio-fix)
            WITH_AUDIO_FIX=1
            ;;
        --without-audio-fix)
            WITH_AUDIO_FIX=0
            ;;
        -h|--help)
            echo "Usage: ./tools/setup_steamos.sh [options]"
            echo "Options:"
            echo "  --with-audio-fix     Install legacy DKMS audio fix (only for older kernels; deprecated on modern/CachyOS kernels)"
            echo "  --without-audio-fix  Skip DKMS audio fix (default)"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
    esac
done

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}${BOLD}     AMD BC-250 Installer for SteamOS / HoloISO       ${NC}"
echo -e "${BLUE}======================================================${NC}"

# Check for root / sudo
SUDO=""
if [ "$EUID" -ne 0 ]; then
    if command -v sudo &> /dev/null; then
        SUDO="sudo"
    else
        echo -e "${RED}Error: Please run with sudo (sudo ./tools/setup_steamos.sh).${NC}"
        exit 1
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Safe trap to ensure steamos-readonly is never left disabled on exit
CLEANUP_READONLY=0
cleanup() {
    if [ "$CLEANUP_READONLY" -eq 1 ] && command -v steamos-readonly &> /dev/null; then
        echo -e "\n  -> Restoring SteamOS rootfs lock (steamos-readonly enable)..."
        $SUDO steamos-readonly enable 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo -e "\n${BOLD}[1/5] Detecting SteamOS Immutable Environment...${NC}"
IS_STEAMOS=0
if command -v steamos-readonly &> /dev/null || ( [ -f /etc/os-release ] && grep -qi "steamos" /etc/os-release ); then
    IS_STEAMOS=1
    echo -e "  ${GREEN}✓ Detected Valve SteamOS / HoloISO immutable environment${NC}"
else
    echo -e "  ${YELLOW}! Non-SteamOS system detected; proceeding with persistent console paths.${NC}"
fi

# SteamOS uses A/B root partitions where /usr and rootfs are wiped during OS updates.
# Staging to /var/lib/bc250 guarantees driver and shaders permanently survive OS updates!
INSTALL_LIB_DIR="/var/lib/bc250/dri"
INSTALL_SHADER_DIR="/var/lib/bc250/shaders"

echo -e "\n${BOLD}[2/5] Locating VA-API Driver & Shaders...${NC}"
$SUDO mkdir -p "$INSTALL_LIB_DIR"
$SUDO mkdir -p "$INSTALL_SHADER_DIR"

# Locate driver binary (pre-built release package vs local compilation)
DRIVER_BIN=""
if [ -f "$REPO_ROOT/bc250_drv_video.so" ]; then
    DRIVER_BIN="$REPO_ROOT/bc250_drv_video.so"
    echo -e "  ${GREEN}✓ Found pre-compiled release package in repository root${NC}"
elif [ -f "$SCRIPT_DIR/bc250_drv_video.so" ]; then
    DRIVER_BIN="$SCRIPT_DIR/bc250_drv_video.so"
    echo -e "  ${GREEN}✓ Found pre-compiled release package in tools directory${NC}"
elif [ -f "$REPO_ROOT/approach1-compute-encoder/build/bc250_drv_video.so" ]; then
    DRIVER_BIN="$REPO_ROOT/approach1-compute-encoder/build/bc250_drv_video.so"
    echo -e "  ${GREEN}✓ Found locally compiled driver binary in build directory${NC}"
elif [ -f "$REPO_ROOT/approach1-compute-encoder/build/libbc250_drv_video.so" ]; then
    DRIVER_BIN="$REPO_ROOT/approach1-compute-encoder/build/libbc250_drv_video.so"
    echo -e "  ${GREEN}✓ Found locally compiled driver binary in build directory${NC}"
else
    BUILD_DIR="$REPO_ROOT/approach1-compute-encoder/build"
    echo -e "  -> No pre-built driver binary found; building from source..."
    if ! command -v cmake &> /dev/null; then
        if [ "$IS_STEAMOS" -eq 1 ]; then
            echo -e "${YELLOW}Notice: Build tools not found on SteamOS.${NC}"
            echo -e "Attempting to temporarily unlock pacman to install dependencies..."
            if command -v steamos-readonly &> /dev/null; then
                $SUDO steamos-readonly disable || true
                CLEANUP_READONLY=1
            fi
            if [ ! -d /etc/pacman.d/gnupg ]; then
                $SUDO pacman-key --init || true
                $SUDO pacman-key --populate archlinux holo || true
            fi
            $SUDO pacman -Sy --noconfirm base-devel cmake libva libdrm vulkan-devel glslang libva-utils || true
        else
            echo -e "${RED}Error: 'cmake' not found and pre-built binary was not provided.${NC}"
            echo -e "Download bc250_drv_video.so and the shaders/ directory of compiled"
            echo -e "shaders from:"
            echo -e "  ${BOLD}https://github.com/Shalasere/bc250-vulkan-encode-stopgap/releases${NC}"
            echo -e "and place both at the repository root before re-running this script,"
            echo -e "or install build tools (cmake, gcc, libva, libdrm, vulkan-devel, glslang)"
            echo -e "to build from source."
            exit 1
        fi
    fi
    mkdir -p "$BUILD_DIR"
    (cd "$BUILD_DIR" && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j"$(nproc)")
    DRIVER_BIN="$BUILD_DIR/bc250_drv_video.so"
fi

if [ ! -f "$DRIVER_BIN" ]; then
    echo -e "${RED}Error: Could not locate or build bc250_drv_video.so.${NC}"
    exit 1
fi

echo -e "\n${BOLD}[3/5] Installing Driver & Shaders (OS-Update Persistent)...${NC}"
echo -e "  -> Installing driver to $INSTALL_LIB_DIR/bc250_drv_video.so"
$SUDO cp -f "$DRIVER_BIN" "$INSTALL_LIB_DIR/bc250_drv_video.so"

# Also install to /usr/local/lib/dri if accessible
$SUDO mkdir -p /usr/local/lib/dri 2>/dev/null || true
$SUDO cp -f "$DRIVER_BIN" /usr/local/lib/dri/bc250_drv_video.so 2>/dev/null || true

# Copy compute shaders. The driver loads compiled SPIR-V (.spv) at runtime -
# the raw .comp GLSL sources under approach1-compute-encoder/shaders/ are
# build-time input only and are useless to the running driver on their own.
# A pre-built release package (matching build_and_install.sh's own
# convention) ships its compiled .spv files in a top-level shaders/
# directory alongside bc250_drv_video.so; a local from-source build instead
# produces them in approach1-compute-encoder/build/. Check both - checking
# only the local build/ directory (as this script previously did) meant
# installing from a pre-built .so with no local build/ present copied *only*
# the non-functional .comp sources and still reported success.
echo -e "  -> Installing compiled shaders to $INSTALL_SHADER_DIR/"
if [ -d "$REPO_ROOT/shaders" ]; then
    $SUDO cp -f "$REPO_ROOT/shaders"/*.spv "$INSTALL_SHADER_DIR/" 2>/dev/null || true
fi
if [ -d "$REPO_ROOT/approach1-compute-encoder/build" ]; then
    $SUDO cp -f "$REPO_ROOT/approach1-compute-encoder/build"/*.spv "$INSTALL_SHADER_DIR/" 2>/dev/null || true
fi

# Also mirror shaders to /usr/local/share if writable
$SUDO mkdir -p /usr/local/share/bc250/shaders 2>/dev/null || true
$SUDO cp -f "$INSTALL_SHADER_DIR"/*.spv /usr/local/share/bc250/shaders/ 2>/dev/null || true

# Set world-readable permissions so unprivileged Steam / Gamescope / Sunshine processes can access them
$SUDO chmod 755 /var/lib/bc250 "$INSTALL_LIB_DIR" "$INSTALL_SHADER_DIR" 2>/dev/null || true
$SUDO chmod 755 "$INSTALL_LIB_DIR/bc250_drv_video.so" 2>/dev/null || true
$SUDO chmod 644 "$INSTALL_SHADER_DIR"/* 2>/dev/null || true

SPV_COUNT=$(find "$INSTALL_SHADER_DIR" -maxdepth 1 -name '*.spv' 2>/dev/null | wc -l)
if [ "$SPV_COUNT" -eq 0 ]; then
    echo -e "\n${RED}Error: no compiled .spv shaders were found to install.${NC}"
    echo -e "${RED}A driver install with no compiled shaders will fail at runtime.${NC}"
    echo -e "Expected compiled .spv files in one of:"
    echo -e "  ${BOLD}$REPO_ROOT/shaders/${NC}          (pre-built release package)"
    echo -e "  ${BOLD}$REPO_ROOT/approach1-compute-encoder/build/${NC}  (local from-source build)"
    echo -e "If you only have bc250_drv_video.so, you also need its matching compiled"
    echo -e "shaders/ directory from the same build - get both from:"
    echo -e "  ${BOLD}https://github.com/Shalasere/bc250-vulkan-encode-stopgap/releases${NC}"
    echo -e "or build from source (requires cmake, gcc, libva, libdrm, vulkan-devel, glslang)."
    exit 1
fi
echo -e "  ${GREEN}✓ Installed $SPV_COUNT compiled shader(s)${NC}"

echo -e "\n${BOLD}[4/5] Configuring SteamOS Gamescope & Session Environment...${NC}"
# /etc/environment.d is persistent across SteamOS updates and read by Gamescope / systemd
$SUDO mkdir -p /etc/environment.d
cat << 'EOF' | $SUDO tee /etc/environment.d/99-bc250.conf > /dev/null
# AMD BC-250 VA-API Compute Driver Configuration (SteamOS)
LIBVA_DRIVER_NAME=bc250
LIBVA_DRIVERS_PATH=/var/lib/bc250/dri:/usr/local/lib/dri:/usr/local/lib64/dri:/usr/lib/dri:/usr/lib64/dri:/usr/lib32/dri:/usr/lib/i386-linux-gnu/dri
BC250_FAST_MODE=1
BC250_SLICES_PER_FRAME=4
BC250_SHADER_DIR=/var/lib/bc250/shaders
OMP_WAIT_POLICY=PASSIVE
GOMP_SPINCOUNT=0
OMP_NUM_THREADS=2
OMP_DYNAMIC=FALSE
EOF
$SUDO chmod 644 /etc/environment.d/99-bc250.conf
echo -e "  ${GREEN}✓ Configured persistent environment in /etc/environment.d/99-bc250.conf${NC}"

# Configure /etc/profile.d for interactive shells and desktop sessions
$SUDO mkdir -p /etc/profile.d
cat << 'EOF' | $SUDO tee /etc/profile.d/bc250.sh > /dev/null
# AMD BC-250 Driver Profile Settings
export LIBVA_DRIVER_NAME=bc250
export LIBVA_DRIVERS_PATH=/var/lib/bc250/dri:/usr/local/lib/dri:/usr/local/lib64/dri:/usr/lib/dri:/usr/lib64/dri:/usr/lib32/dri:/usr/lib/i386-linux-gnu/dri
export BC250_FAST_MODE=1
export BC250_SLICES_PER_FRAME=4
export BC250_SHADER_DIR=/var/lib/bc250/shaders
export OMP_WAIT_POLICY=PASSIVE
export GOMP_SPINCOUNT=0
export OMP_NUM_THREADS=2
export OMP_DYNAMIC=FALSE
EOF
$SUDO chmod 644 /etc/profile.d/bc250.sh
echo -e "  ${GREEN}✓ Configured persistent environment in /etc/profile.d/bc250.sh${NC}"

# Ensure user 'deck' or current sudo user has render permissions
CURRENT_USER="${SUDO_USER:-$USER}"
if [ -n "$CURRENT_USER" ] && [ "$CURRENT_USER" != "root" ]; then
    $SUDO usermod -a -G video,render "$CURRENT_USER" 2>/dev/null || true
    echo -e "  ${GREEN}✓ Granted GPU render permissions to user '$CURRENT_USER'${NC}"
fi

echo -e "\n${BOLD}[5/5] Setting Up Audio Clock Fix...${NC}"
if [ "$WITH_AUDIO_FIX" -eq 1 ]; then
    if [ -d "$REPO_ROOT/audio-fix" ]; then
        cd "$REPO_ROOT/audio-fix"
        if command -v dkms &> /dev/null; then
            if $SUDO bash ./install_dkms.sh; then
                echo -e "  ${GREEN}✓ Audio fix installed via DKMS (auto-rebuilds on kernel updates).${NC}"
            else
                echo -e "  ${YELLOW}! DKMS setup encountered an issue (kernel headers may be needed).${NC}"
                echo -e "    On SteamOS, install headers via:"
                echo -e "      ${BOLD}sudo steamos-readonly disable && sudo pacman -S --needed linux-neptune-headers dkms${NC}"
            fi
        else
            echo -e "  ${YELLOW}! DKMS not installed.${NC}"
            echo -e "  -> Attempting direct module compilation..."
            if make && $SUDO make install && $SUDO modprobe bc250_audio_fix 2>/dev/null; then
                echo -e "  ${GREEN}✓ Audio module compiled and loaded.${NC}"
            else
                echo -e "  ${YELLOW}! Direct module compilation skipped. To enable HDMI/DP audio, install headers:${NC}"
                echo -e "      ${BOLD}sudo steamos-readonly disable && sudo pacman -S --needed linux-neptune-headers dkms${NC}"
            fi
        fi
        cd "$REPO_ROOT"
    fi
else
    echo -e "  ${BLUE}[*] Audio Fix skipped by default (modern kernels/CachyOS support audio natively).${NC}"
    echo -e "      Pass ${BOLD}--with-audio-fix${NC} if running on an older kernel that requires the legacy module."
fi

echo -e "\n${GREEN}======================================================${NC}"
echo -e "${GREEN}${BOLD}    SteamOS / HoloISO Setup Completed Successfully!   ${NC}"
echo -e "${GREEN}======================================================${NC}"
echo -e "\nSummary:"
echo -e "  * Persistent driver path: ${YELLOW}$INSTALL_LIB_DIR/bc250_drv_video.so${NC} (Survives OS updates!)"
echo -e "  * Compute shaders at:     ${YELLOW}$INSTALL_SHADER_DIR/${NC}"
echo -e "  * Multi-Slice mode:       ${GREEN}4 slices per frame${NC} (smooth Moonlight game streaming)"
echo -e "  * Audio Fix:              ${GREEN}Active + Sleep/Wake auto-resume${NC}"
echo -e "\nNext steps:"
echo -e "  1. Switch back to Gaming Mode or reboot your console."
echo -e "  2. ${BOLD}If you use Sunshine:${NC} ${YELLOW}sudo ./tools/install_vaapi_boot_redirect.sh${NC}"
echo -e "     ${YELLOW}(one more step, required)${NC} - Sunshine's binary carries a file"
echo -e "     capability for KMS capture that puts it in a kernel mode where libva"
echo -e "     CANNOT see the LIBVA_DRIVER_NAME/LIBVA_DRIVERS_PATH set above at all,"
echo -e "     no matter how they're configured. This redirects around that instead,"
echo -e "     and persists across reboots/OS updates."
echo -e "  3. Test the driver with: ${YELLOW}./tools/bc250_diagnose.sh${NC}"
echo -e "     (it will tell you explicitly if a running Sunshine still needs step 2)"
echo -e "  4. In Sunshine Web UI: set Video Encoder to ${GREEN}VA-API${NC}."

echo
echo -e "${BOLD}This changed your system in a few places.${NC} What, and how to undo it:"
echo -e "  ${BOLD}sudo ./tools/bc250_uninstall.sh --dry-run${NC}   # list it, change nothing"
echo -e "  ${BOLD}sudo ./tools/bc250_uninstall.sh${NC}             # remove it"
echo -e "Note: LIBVA_DRIVER_NAME=bc250 was set ${BOLD}system-wide${NC}, and this driver is"
echo -e "encode-only - so other apps lose hardware video ${BOLD}decode${NC} until it is removed."
echo -e "To avoid that, delete /etc/environment.d/99-bc250.conf and set the variable"
echo -e "only in the environment of the one app you want encoding. See README.md"
echo -e "(\"What the installers change, and how to undo it\")."

echo
echo -e "${YELLOW}${BOLD}Note: this installed the 64-bit driver only.${NC}"
echo -e "${YELLOW}Steam Link's runtime is 32-bit and dlopen()s a 32-bit VA-API driver, so it${NC}"
echo -e "${YELLOW}cannot see the driver installed above and will fall back to software.${NC}"
echo -e "If you use Steam Link, also run:"
echo -e "  ${BOLD}./tools/build_32bit.sh${NC}"
echo -e "(installs an i386 driver alongside this one; nothing above is replaced.)"
