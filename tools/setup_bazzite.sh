#!/usr/bin/env bash
# bc250-encoding-decoding-fix v0.4.0 - https://github.com/simpmix/bc250-encoding-decoding-fix
#
# setup_bazzite.sh - Automated installer tailored specifically for Bazzite / Silverblue / Kinoite (rpm-ostree)
#
# Handles rpm-ostree immutable filesystem constraints, SELinux security contexts,
# Gamescope user sessions, and Sunshine game streaming integration.
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
            echo "Usage: ./tools/setup_bazzite.sh [options]"
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
echo -e "${BLUE}${BOLD}     AMD BC-250 Installer for Bazzite (rpm-ostree)    ${NC}"
echo -e "${BLUE}======================================================${NC}"

# Check for root / sudo
SUDO=""
if [ "$EUID" -ne 0 ]; then
    if command -v sudo &> /dev/null; then
        SUDO="sudo"
    else
        echo -e "${RED}Error: Please run with sudo (sudo ./tools/setup_bazzite.sh).${NC}"
        exit 1
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "\n${BOLD}[1/5] Detecting Immutable OS Environment...${NC}"
IS_OSTREE=0
if [ -f "/run/ostree-booted" ] || command -v rpm-ostree &> /dev/null; then
    IS_OSTREE=1
    echo -e "  ${GREEN}✓ Detected rpm-ostree immutable filesystem (Bazzite / Fedora Silverblue)${NC}"
else
    echo -e "  ${YELLOW}! Standard writable filesystem detected; installing to /usr/local.${NC}"
fi

# On rpm-ostree, /usr is read-only; /usr/local and /etc are writable
INSTALL_LIB_DIR="/usr/local/lib64/dri"
INSTALL_LIB_FALLBACK="/usr/local/lib/dri"
INSTALL_SHADER_DIR="/usr/local/share/bc250/shaders"

echo -e "\n${BOLD}[2/5] Locating VA-API Driver & Shaders...${NC}"
$SUDO mkdir -p "$INSTALL_LIB_DIR"
$SUDO mkdir -p "$INSTALL_LIB_FALLBACK"
$SUDO mkdir -p "$INSTALL_SHADER_DIR"

# Locate driver binary (pre-built release package vs local compilation)
DRIVER_BIN=""
if [ -f "$REPO_ROOT/bc250_drv_video.so" ]; then
    DRIVER_BIN="$REPO_ROOT/bc250_drv_video.so"
    echo -e "  ${GREEN}✓ Found pre-compiled driver package in repository root${NC}"
elif [ -f "$SCRIPT_DIR/bc250_drv_video.so" ]; then
    DRIVER_BIN="$SCRIPT_DIR/bc250_drv_video.so"
    echo -e "  ${GREEN}✓ Found pre-compiled driver package in tools directory${NC}"
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
        echo -e "${RED}Error: 'cmake' not found and pre-built binary was not provided.${NC}"
        echo -e "To resolve this:"
        echo -e "  Option A (Recommended): Download bc250_drv_video.so and the shaders/"
        echo -e "    directory of compiled shaders from:"
        echo -e "    ${BOLD}https://github.com/Shalasere/bc250-vulkan-encode-stopgap/releases${NC}"
        echo -e "    and place both at the repository root before re-running this script."
        echo -e "  Option B: Install development tools on Bazzite via:"
        echo -e "    ${BOLD}ujust dev-tools${NC} (or layered via rpm-ostree install cmake gcc"
        echo -e "    libva-devel libdrm-devel vulkan-loader-devel glslang) - a reboot is"
        echo -e "    required after layering packages on rpm-ostree."
        exit 1
    fi
    mkdir -p "$BUILD_DIR"
    (cd "$BUILD_DIR" && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j"$(nproc)")
    DRIVER_BIN="$BUILD_DIR/bc250_drv_video.so"
fi

if [ ! -f "$DRIVER_BIN" ]; then
    echo -e "${RED}Error: Could not locate or build bc250_drv_video.so.${NC}"
    exit 1
fi

echo -e "\n${BOLD}[3/5] Installing Driver, Shaders & Setting SELinux Contexts...${NC}"
echo -e "  -> Installing driver to $INSTALL_LIB_DIR/bc250_drv_video.so"
$SUDO cp -f "$DRIVER_BIN" "$INSTALL_LIB_DIR/bc250_drv_video.so"
$SUDO cp -f "$DRIVER_BIN" "$INSTALL_LIB_FALLBACK/bc250_drv_video.so"
$SUDO chmod 755 "$INSTALL_LIB_DIR" "$INSTALL_LIB_FALLBACK"
$SUDO chmod 755 "$INSTALL_LIB_DIR/bc250_drv_video.so" "$INSTALL_LIB_FALLBACK/bc250_drv_video.so"

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
$SUDO chmod 755 "$INSTALL_SHADER_DIR"
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
    echo -e "or build from source (requires cmake, gcc, libva-devel, libdrm-devel,"
    echo -e "vulkan-loader-devel, glslang - see [2/5] above)."
    exit 1
fi
echo -e "  ${GREEN}✓ Installed $SPV_COUNT compiled shader(s)${NC}"

# Restore SELinux security contexts on Bazzite / Fedora Silverblue
if command -v restorecon &> /dev/null; then
    echo -e "  -> Applying SELinux file contexts on /usr/local..."
    $SUDO restorecon -Rv "$INSTALL_LIB_DIR" "$INSTALL_LIB_FALLBACK" "$INSTALL_SHADER_DIR" 2>/dev/null || true
fi

echo -e "\n${BOLD}[4/5] Configuring Gamescope & User Environment...${NC}"
# Configure /etc/environment.d for systemd and Gamescope user session
$SUDO mkdir -p /etc/environment.d
cat << 'EOF' | $SUDO tee /etc/environment.d/99-bc250.conf > /dev/null
# AMD BC-250 VA-API Compute Driver Configuration (Bazzite)
LIBVA_DRIVER_NAME=bc250
LIBVA_DRIVERS_PATH=/usr/local/lib64/dri:/usr/local/lib/dri:/usr/lib64/dri:/usr/lib/dri:/usr/lib32/dri:/usr/lib/i386-linux-gnu/dri
BC250_FAST_MODE=1
BC250_SLICES_PER_FRAME=4
BC250_SHADER_DIR=/usr/local/share/bc250/shaders
OMP_WAIT_POLICY=PASSIVE
GOMP_SPINCOUNT=0
OMP_NUM_THREADS=2
OMP_DYNAMIC=FALSE
EOF
$SUDO chmod 644 /etc/environment.d/99-bc250.conf
echo -e "  ${GREEN}✓ Configured /etc/environment.d/99-bc250.conf${NC}"

# Also configure /etc/profile.d for login/desktop/interactive shells
$SUDO mkdir -p /etc/profile.d
cat << 'EOF' | $SUDO tee /etc/profile.d/bc250.sh > /dev/null
# AMD BC-250 Driver Profile Settings
export LIBVA_DRIVER_NAME=bc250
export LIBVA_DRIVERS_PATH=/usr/local/lib64/dri:/usr/local/lib/dri:/usr/lib64/dri:/usr/lib/dri:/usr/lib32/dri:/usr/lib/i386-linux-gnu/dri
export BC250_FAST_MODE=1
export BC250_SLICES_PER_FRAME=4
export BC250_SHADER_DIR=/usr/local/share/bc250/shaders
export OMP_WAIT_POLICY=PASSIVE
export GOMP_SPINCOUNT=0
export OMP_NUM_THREADS=2
export OMP_DYNAMIC=FALSE
EOF
$SUDO chmod 644 /etc/profile.d/bc250.sh
echo -e "  ${GREEN}✓ Configured /etc/profile.d/bc250.sh${NC}"

# Add user to video and render groups
CURRENT_USER="${SUDO_USER:-$USER}"
if [ -n "$CURRENT_USER" ] && [ "$CURRENT_USER" != "root" ]; then
    $SUDO usermod -a -G video,render "$CURRENT_USER" 2>/dev/null || true
    echo -e "  ${GREEN}✓ Granted GPU render permissions to user '$CURRENT_USER'${NC}"
fi

echo -e "\n${BOLD}[5/5] Setting Up DisplayPort Audio Fix...${NC}"
if [ "$WITH_AUDIO_FIX" -eq 1 ]; then
    if [ -d "$REPO_ROOT/audio-fix" ]; then
        cd "$REPO_ROOT/audio-fix"
        if command -v dkms &> /dev/null; then
            if $SUDO bash ./install_dkms.sh; then
                echo -e "  ${GREEN}✓ Audio fix installed via DKMS (persists across kernel updates).${NC}"
            else
                echo -e "  ${YELLOW}! DKMS build encountered an issue (kernel headers may be missing).${NC}"
                echo -e "    To install kernel headers on Bazzite:"
                echo -e "      ${BOLD}ujust install-kernel-headers${NC} or ${BOLD}rpm-ostree install kernel-devel-$(uname -r)${NC}"
            fi
        else
            echo -e "  ${YELLOW}! DKMS not installed.${NC}"
            if [ "$IS_OSTREE" -eq 1 ]; then
                echo -e "    Note: On rpm-ostree, manual module installation to /usr/lib/modules is read-only."
                echo -e "    To enable DKMS on Bazzite, run:"
                echo -e "      ${BOLD}ujust install-kernel-headers && rpm-ostree install dkms${NC}"
            else
                echo -e "  -> Building module locally..."
                if make && $SUDO make install && $SUDO modprobe bc250_audio_fix 2>/dev/null; then
                    echo -e "  ${GREEN}✓ Audio module compiled and loaded.${NC}"
                else
                    echo -e "  ${YELLOW}! Local compilation skipped. Install kernel headers if audio is needed.${NC}"
                fi
            fi
        fi
        cd "$REPO_ROOT"
    fi
else
    echo -e "  ${BLUE}[*] Audio Fix skipped by default (modern kernels/CachyOS support audio natively).${NC}"
    echo -e "      Pass ${BOLD}--with-audio-fix${NC} if running on an older kernel that requires the legacy module."
fi

echo -e "\n${GREEN}======================================================${NC}"
echo -e "${GREEN}${BOLD}     Bazzite Setup Completed Successfully!           ${NC}"
echo -e "${GREEN}======================================================${NC}"
echo -e "\nSummary:"
echo -e "  * Driver installed:       ${YELLOW}$INSTALL_LIB_DIR/bc250_drv_video.so${NC}"
echo -e "  * Compute shaders at:     ${YELLOW}$INSTALL_SHADER_DIR/${NC}"
echo -e "  * SELinux labels:         ${GREEN}Applied${NC}"
echo -e "  * Multi-Slice mode:       ${GREEN}4 slices per frame${NC} (low-latency streaming)"
echo -e "\nNext steps:"
echo -e "  1. Restart your Gamescope session or reboot your console."
echo -e "  2. ${BOLD}If you use Sunshine:${NC} ${YELLOW}sudo ./tools/install_vaapi_boot_redirect.sh${NC}"
echo -e "     ${YELLOW}(one more step, required)${NC} - Sunshine's binary carries a file"
echo -e "     capability for KMS capture that puts it in a kernel mode where libva"
echo -e "     CANNOT see the LIBVA_DRIVER_NAME/LIBVA_DRIVERS_PATH set above at all,"
echo -e "     no matter how they're configured. This redirects around that instead,"
echo -e "     and persists across reboots on this ostree system."
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
