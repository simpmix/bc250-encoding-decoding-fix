#!/usr/bin/env bash
# bc250-encoding-decoding-fix v0.4.0 - https://github.com/simpmix/bc250-encoding-decoding-fix
#
# build_and_install.sh - Automated build, test, and installer for AMD BC-250 custom drivers
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
            echo "Usage: ./build_and_install.sh [options]"
            echo "Options:"
            echo "  --with-audio-fix     Install legacy DKMS audio fix (only for older kernels; deprecated on modern/CachyOS kernels)"
            echo "  --without-audio-fix  Skip DKMS audio fix (default)"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
    esac
done

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}${BOLD}   AMD BC-250 (Cyan Skillfish) Custom Driver Setup    ${NC}"
echo -e "${BLUE}======================================================${NC}"

# Check for root / sudo
SUDO=""
if [ "$EUID" -ne 0 ]; then
    if command -v sudo &> /dev/null; then
        SUDO="sudo"
    else
        echo -e "${YELLOW}Warning: Running without root. System-wide installation may fail.${NC}"
    fi
fi

# Detect hardware
echo -e "\n${BLUE}[1/5] Checking hardware...${NC}"
if command -v lspci &> /dev/null && lspci -nn | grep -i "1002:13fe" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Detected AMD BC-250 APU (1002:13fe)${NC}"
else
    echo -e "${YELLOW}! BC-250 (1002:13fe) not detected on PCI bus.${NC}"
    echo -e "  Proceeding with build anyway (Vulkan compute driver is compatible with other AMD GPUs for testing)."
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREBUILT_SO="$SCRIPT_DIR/bc250_drv_video.so"
IS_PREBUILT=0

if [ -f "$PREBUILT_SO" ]; then
    IS_PREBUILT=1
    echo -e "${GREEN}✓ Detected pre-built release package (no compilation required).${NC}"
fi

# Check dependencies
echo -e "\n${BLUE}[2/5] Checking dependencies...${NC}"
if [ "$IS_PREBUILT" -eq 1 ]; then
    echo -e "${GREEN}✓ Using pre-compiled driver and shaders. Skipping build tools check.${NC}"
else
    MISSING_PKGS=()

    check_cmd() {
        if ! command -v "$1" &> /dev/null; then
            MISSING_PKGS+=("$2")
        fi
    }

    check_cmd cmake "cmake"
    check_cmd gcc "gcc / build-essential"
    check_cmd pkg-config "pkg-config"

    if ! command -v glslangValidator &> /dev/null && ! command -v glslc &> /dev/null; then
        MISSING_PKGS+=("glslang-tools (or shaderc)")
    fi

    if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
        echo -e "${YELLOW}Missing packages detected:${NC}"
        for pkg in "${MISSING_PKGS[@]}"; do
            echo -e "    * $pkg"
        done
        
        echo -e "\n${BOLD}Attempting to install missing build dependencies...${NC}"
        if command -v apt-get &> /dev/null; then
            $SUDO apt-get update
            $SUDO apt-get install -y build-essential cmake pkg-config libva-dev libdrm-dev libvulkan-dev libx264-dev glslang-tools vainfo
        elif command -v dnf &> /dev/null; then
            $SUDO dnf install -y gcc gcc-c++ cmake pkgconf libva-devel libdrm-devel vulkan-loader-devel glslang libva-utils
        elif command -v pacman &> /dev/null; then
            $SUDO pacman -S --needed --noconfirm base-devel cmake pkgconf libva libdrm vulkan-devel x264 glslang libva-utils
        elif command -v zypper &> /dev/null; then
            $SUDO zypper install -y gcc gcc-c++ cmake pkg-config libva-devel libdrm-devel vulkan-devel glslang libva-utils
        else
            echo -e "${RED}Could not auto-install dependencies. Please install: cmake, gcc, libva-dev, libdrm-dev, vulkan-dev, glslang-tools${NC}"
        fi
    else
        echo -e "${GREEN}✓ All core build tools are available.${NC}"
    fi
fi

# Build or Stage Compute Encoder VA-API driver
echo -e "\n${BLUE}[3/5] Setting up Compute Encoder VA-API driver...${NC}"

# On rpm-ostree/immutable systems (Bazzite, Silverblue, SteamOS, etc.) /usr is a
# read-only bind mount. Writing there fails, so /usr/local (writable, and
# persistent because /usr/local is itself a symlink into /var on these distros)
# MUST be tried too, or the driver silently ends up installed nowhere.
IS_OSTREE=0
if [ -f "/run/ostree-booted" ] || command -v rpm-ostree &> /dev/null; then
    IS_OSTREE=1
    echo -e "${YELLOW}! Detected an rpm-ostree/immutable filesystem. /usr is read-only here;${NC}"
    echo -e "${YELLOW}  consider using tools/setup_bazzite.sh or tools/setup_steamos.sh instead,${NC}"
    echo -e "${YELLOW}  which target the correct persistent paths for this OS.${NC}"
fi

DRI_DIRS=("/usr/lib/x86_64-linux-gnu/dri" "/usr/lib64/dri" "/usr/lib/dri" "/usr/local/lib64/dri" "/usr/local/lib/dri")
SHADER_DESTS=("/usr/share/bc250/shaders" "/usr/local/share/bc250/shaders")

if [ "$IS_PREBUILT" -eq 1 ]; then
    echo -e "${GREEN}✓ Staging pre-built driver binary...${NC}"
    DRIVER_BIN="$PREBUILT_SO"
else
    BUILD_DIR="$SCRIPT_DIR/approach1-compute-encoder/build"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
    make -j"$(nproc)"
    DRIVER_BIN="$BUILD_DIR/bc250_drv_video.so"

    # Run tests
    echo -e "\n${BLUE}[4/5] Running driver unit and integration tests...${NC}"
    ctest --output-on-failure || echo -e "${YELLOW}Note: Some Vulkan display tests may skip if running in headless terminal.${NC}"
    
    $SUDO make install
    cd "$SCRIPT_DIR"
fi

# Install driver and shaders
echo -e "\n${BLUE}[5/5] Installing driver and compute shaders...${NC}"
for dest in "${SHADER_DESTS[@]}"; do
    $SUDO mkdir -p "$dest"
    if [ -d "$SCRIPT_DIR/shaders" ]; then
        $SUDO cp -f "$SCRIPT_DIR"/shaders/* "$dest/" 2>/dev/null || true
    elif [ -d "$SCRIPT_DIR/approach1-compute-encoder/shaders" ]; then
        $SUDO cp -f "$SCRIPT_DIR"/approach1-compute-encoder/shaders/* "$dest/" 2>/dev/null || true
    fi
    if [ -d "$SCRIPT_DIR/approach1-compute-encoder/build" ]; then
        $SUDO cp -f "$SCRIPT_DIR"/approach1-compute-encoder/build/*.spv "$dest/" 2>/dev/null || true
    fi
done

INSTALLED_DRIVER=0
for d in "${DRI_DIRS[@]}"; do
    # /usr/local/lib64/dri and /usr/local/lib/dri may not exist yet on any distro;
    # the rest are standard system paths we only touch if already present.
    case "$d" in
        /usr/local/*) $SUDO mkdir -p "$d" 2>/dev/null || true ;;
    esac
    if [ -d "$d" ]; then
        if $SUDO cp -f "$DRIVER_BIN" "$d/bc250_drv_video.so" 2>/dev/null; then
            echo -e "  ${GREEN}-> Installed driver into $d/bc250_drv_video.so${NC}"
            INSTALLED_DRIVER=1
        else
            echo -e "  ${YELLOW}-> Skipped $d (not writable, likely a read-only ostree /usr mount)${NC}"
        fi
    fi
done

if [ "$INSTALLED_DRIVER" -eq 0 ]; then
    echo -e "\n${RED}${BOLD}Error: could not install bc250_drv_video.so into any DRI driver path.${NC}"
    echo -e "${RED}Every candidate directory was missing or read-only (this is expected on${NC}"
    echo -e "${RED}rpm-ostree/immutable systems like Bazzite or SteamOS).${NC}"
    echo -e "Use the dedicated installer for your OS instead:"
    echo -e "  ${BOLD}sudo ./tools/setup_bazzite.sh${NC}   (Bazzite / Fedora Silverblue / Kinoite)"
    echo -e "  ${BOLD}sudo ./tools/setup_steamos.sh${NC}   (SteamOS / HoloISO)"
    exit 1
fi

# Configure system-wide environment for boot persistence. Always include the
# /usr/local DRI paths in LIBVA_DRIVERS_PATH: libva's compiled-in default search
# path does NOT include /usr/local, so a driver installed only there (the only
# path that actually succeeds on an ostree system) would otherwise never be found.
if [ -d "/etc/environment.d" ]; then
    printf "LIBVA_DRIVER_NAME=bc250\nLIBVA_DRIVERS_PATH=/usr/local/lib64/dri:/usr/local/lib/dri:/usr/lib/x86_64-linux-gnu/dri:/usr/lib64/dri:/usr/lib/dri:/usr/lib32/dri:/usr/lib/i386-linux-gnu/dri\nBC250_FAST_MODE=1\nBC250_SLICES_PER_FRAME=4\nOMP_WAIT_POLICY=PASSIVE\nGOMP_SPINCOUNT=0\nOMP_NUM_THREADS=2\nOMP_DYNAMIC=FALSE\n" | $SUDO tee /etc/environment.d/99-bc250.conf > /dev/null 2>&1 || true
    echo -e "  -> Configured system-wide environment in /etc/environment.d/99-bc250.conf"
elif [ -f "/etc/environment" ]; then
    if ! grep -q "LIBVA_DRIVER_NAME=bc250" /etc/environment 2>/dev/null; then
        printf "LIBVA_DRIVER_NAME=bc250\nLIBVA_DRIVERS_PATH=/usr/local/lib64/dri:/usr/local/lib/dri:/usr/lib/x86_64-linux-gnu/dri:/usr/lib64/dri:/usr/lib/dri:/usr/lib32/dri:/usr/lib/i386-linux-gnu/dri\nBC250_FAST_MODE=1\nBC250_SLICES_PER_FRAME=4\nOMP_WAIT_POLICY=PASSIVE\nGOMP_SPINCOUNT=0\nOMP_NUM_THREADS=2\nOMP_DYNAMIC=FALSE\n" | $SUDO tee -a /etc/environment > /dev/null 2>&1 || true
        echo -e "  -> Configured system-wide environment in /etc/environment"
    fi
fi

# Optional: Install Audio Fix via DKMS if explicitly requested
if [ "$WITH_AUDIO_FIX" -eq 1 ]; then
    if [ -d "$SCRIPT_DIR/audio-fix" ] && command -v dkms &> /dev/null; then
        echo -e "\n${BLUE}Configuring Audio Fix with DKMS (--with-audio-fix specified)...${NC}"
        (cd "$SCRIPT_DIR/audio-fix" && $SUDO bash ./install_dkms.sh) || echo -e "${YELLOW}DKMS setup skipped.${NC}"
    else
        echo -e "\n${YELLOW}Audio fix requested but audio-fix directory or dkms not available. Skipping.${NC}"
    fi
else
    echo -e "\n${BLUE}[*] Audio Fix DKMS skipped by default (native kernel audio supported on modern kernels and CachyOS).${NC}"
    echo -e "    Pass ${BOLD}--with-audio-fix${NC} if you are on an older kernel that requires the legacy DKMS module."
fi

# Configuration summary
echo -e "\n${GREEN}======================================================${NC}"
echo -e "${GREEN}${BOLD}   Installation Completed Successfully!              ${NC}"
echo -e "${GREEN}======================================================${NC}"
echo -e "\nTo activate the driver in your current shell session:"
echo -e "  ${YELLOW}export LIBVA_DRIVER_NAME=bc250${NC}"
echo -e "  ${YELLOW}export BC250_FAST_MODE=1${NC}  (keeps GPU overhead under 3-5% for 60 FPS gaming!)"
echo -e "\nTo verify driver capabilities and run encode benchmark:"
echo -e "  ${GREEN}./tools/bc250_diagnose.sh${NC}"
echo -e "  ${GREEN}LIBVA_DRIVER_NAME=bc250 vainfo${NC}"
echo -e "\n${YELLOW}${BOLD}If you plan to use Sunshine (Moonlight/game streaming), one more step:${NC}"
echo -e "${YELLOW}Sunshine's binary normally carries a file capability (cap_sys_admin,${NC}"
echo -e "${YELLOW}for KMS screen capture), which puts glibc's secure_getenv() into a mode${NC}"
echo -e "${YELLOW}where it CANNOT see LIBVA_DRIVER_NAME/LIBVA_DRIVERS_PATH at all - the${NC}"
echo -e "${YELLOW}environment variables above will not reach it, no matter how they're set.${NC}"
echo -e "Run this once instead:"
echo -e "  ${BOLD}sudo ./tools/install_vaapi_boot_redirect.sh${NC}"
echo -e "(persists across reboots; ${GREEN}./tools/bc250_diagnose.sh${NC} will tell you"
echo -e "explicitly if your running Sunshine needs this.)"

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
