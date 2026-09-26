#!/usr/bin/env bash
# build_32bit.sh - build and install the 32-bit (i386) driver on this machine.
#
# WHY THIS EXISTS SEPARATELY: build_and_install.sh, setup_bazzite.sh and
# setup_steamos.sh only ever build the 64-bit driver. That is the right default
# - the 32-bit one is needed by exactly one kind of client (a 32-bit VA-API
# consumer, in practice Steam Link) and pulling a whole multilib toolchain onto
# every install to serve that would be rude. So the 32-bit build is opt-in, and
# until now "opt-in" meant hand-typing a cmake invocation, which is why an
# install could look complete and still have no 32-bit driver anywhere.
#
# It installs ALONGSIDE the 64-bit driver, into the 32-bit DRI directory. It
# does not replace anything, and 64-bit clients are unaffected.
#
# Usage:
#   ./tools/build_32bit.sh                 # install deps, build, verify, install
#   ./tools/build_32bit.sh --skip-deps     # I already have the 32-bit libraries
#   ./tools/build_32bit.sh --deps-only     # just install the libraries
#   ./tools/build_32bit.sh --no-install    # build and verify, don't install
#   ./tools/build_32bit.sh --dry-run       # print what would happen
set -uo pipefail

SKIP_DEPS=0; DEPS_ONLY=0; NO_INSTALL=0; DRY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-deps)  SKIP_DEPS=1 ;;
        --deps-only)  DEPS_ONLY=1 ;;
        --no-install) NO_INSTALL=1 ;;
        --dry-run)    DRY=1 ;;
        -h|--help)    sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

GREEN=$'\033[0;32m'; RED=$'\033[0;31m'; YELLOW=$'\033[1;33m'
BLUE=$'\033[0;34m'; BOLD=$'\033[1m'; NC=$'\033[0m'
ok()   { echo -e "  ${GREEN}✓ $*${NC}"; }
warn() { echo -e "  ${YELLOW}! $*${NC}"; }
die()  { echo -e "  ${RED}✗ $*${NC}" >&2; exit 1; }
step() { echo -e "\n${BOLD}$*${NC}"; }
run()  { if [ "$DRY" -eq 1 ]; then echo "  [dry-run] $*"; else "$@"; fi; }

SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
SRC="$REPO_ROOT/approach1-compute-encoder"
BUILD="$REPO_ROOT/build32"

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}${BOLD}   BC-250 32-bit (i386) driver — for Steam Link       ${NC}"
echo -e "${BLUE}======================================================${NC}"

[ -f "$SRC/CMakeLists.txt" ] || die "can't find $SRC/CMakeLists.txt - run this from the repo"
grep -q "BUILD_32BIT" "$SRC/CMakeLists.txt" \
    || die "this checkout's CMakeLists.txt has no BUILD_32BIT option - git pull first"

# ------------------------------------------------------------------ 1. deps
step "[1/5] 32-bit build dependencies"
if [ "$SKIP_DEPS" -eq 1 ]; then
    ok "skipped (--skip-deps)"
elif command -v pacman >/dev/null 2>&1; then
    # Arch/CachyOS: needs the [multilib] repo enabled in /etc/pacman.conf, which
    # is NOT on by default. lib32-gcc-libs is what provides the shared 32-bit
    # libgomp - without it the link fails on -Wl,-z,text by design (see
    # CMakeLists.txt), rather than silently shipping an unloadable driver.
    if ! pacman-conf --repo-list 2>/dev/null | grep -qx multilib; then
        warn "the [multilib] repository does not look enabled in /etc/pacman.conf."
        warn "Uncomment these two lines, run 'sudo pacman -Sy', then re-run this:"
        echo "      [multilib]"
        echo "      Include = /etc/pacman.d/mirrorlist"
        die "multilib repo required for the 32-bit libraries"
    fi
    run $SUDO pacman -S --needed --noconfirm \
        lib32-libva lib32-libdrm lib32-vulkan-icd-loader lib32-gcc-libs \
        || die "pacman failed to install the 32-bit libraries"

    # lib32-x264 is an AUR package on standard Arch Linux (or found in custom repos like CachyOS).
    # Try installing it if available, but do not fail the build if absent since CMake will
    # gracefully fall back to the compute encoder if lib32-x264 is omitted.
    if pacman -Si lib32-x264 >/dev/null 2>&1; then
        run $SUDO pacman -S --needed --noconfirm lib32-x264 2>/dev/null || true
    elif command -v paru >/dev/null 2>&1; then
        paru -S --needed --noconfirm lib32-x264 2>/dev/null || true
    elif command -v yay >/dev/null 2>&1; then
        yay -S --needed --noconfirm lib32-x264 2>/dev/null || true
    fi

    if pacman -Qi lib32-x264 >/dev/null 2>&1 || PKG_CONFIG_LIBDIR=/usr/lib32/pkgconfig pkg-config --exists x264 2>/dev/null; then
        ok "Arch/CachyOS 32-bit libraries present (including lib32-x264)"
    else
        ok "Arch/CachyOS base 32-bit libraries present"
        warn "lib32-x264 not installed. The 32-bit driver will build with the GPU compute encoder."
        warn "For x264 software offload in 32-bit Steam Link, install: 'paru -S lib32-x264' or 'yay -S lib32-x264'"
    fi
elif command -v dnf >/dev/null 2>&1; then
    run $SUDO dnf install -y \
        glibc-devel.i686 libgcc.i686 libgomp.i686 \
        libva-devel.i686 libdrm-devel.i686 vulkan-loader-devel.i686 x264-devel.i686 \
        || die "dnf failed to install the i686 libraries"
    ok "Fedora i686 libraries present"
elif command -v apt-get >/dev/null 2>&1; then
    run $SUDO dpkg --add-architecture i386
    run $SUDO apt-get update
    run $SUDO apt-get install -y gcc-multilib \
        libva-dev:i386 libdrm-dev:i386 libvulkan-dev:i386 libgomp1:i386 libx264-dev:i386 \
        || die "apt failed to install the i386 libraries"
    ok "Debian/Ubuntu i386 libraries present"
elif command -v zypper >/dev/null 2>&1; then
    run $SUDO zypper install -y \
        glibc-devel-32bit libgomp1-32bit libva-devel-32bit \
        libdrm-devel-32bit vulkan-devel-32bit \
        || die "zypper failed to install the 32-bit libraries"
    ok "openSUSE 32-bit libraries present"
else
    warn "unrecognised package manager - install these yourself, then --skip-deps:"
    echo "      32-bit libva, libdrm, vulkan-loader, AND a shared 32-bit libgomp"
    die "cannot install dependencies automatically"
fi

[ "$DEPS_ONLY" -eq 1 ] && { echo; ok "--deps-only: stopping here"; exit 0; }

# --------------------------------------------------------------- 2. configure
step "[2/5] Configuring (BUILD_32BIT=ON)"
command -v cmake >/dev/null 2>&1 || die "cmake not found"
# Fresh build dir every time. Reusing one whose cache was configured for the
# other architecture is exactly how a 64-bit .so ends up installed into a
# 32-bit path (issue #14), and the cache is cheap to discard.
run rm -rf "$BUILD"
if ! run cmake -B "$BUILD" -S "$SRC" \
        -DBUILD_32BIT=ON -DBUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release; then
    echo
    die "configure failed - almost always a missing 32-bit library above"
fi
ok "configured"

# ------------------------------------------------------------------ 3. build
step "[3/5] Building"
if ! run cmake --build "$BUILD" --parallel 4; then
    echo
    warn "If this failed with 'read-only segment has dynamic relocations', the"
    warn "shared 32-bit libgomp is missing (lib32-gcc-libs / libgomp.i686 /"
    warn "libgomp1:i386). That guard is deliberate - see CMakeLists.txt."
    die "build failed"
fi
ok "built"

# ----------------------------------------------------------------- 4. verify
step "[4/5] Verifying the artifact"
if [ "$DRY" -eq 1 ]; then
    echo "  [dry-run] would verify ELF32 / i386 / no DT_TEXTREL"
else
    SO=$(find "$BUILD" -name 'bc250_drv_video.so' | head -n 1)
    [ -n "$SO" ] || die "no bc250_drv_video.so was produced"
    if command -v readelf >/dev/null 2>&1; then
        readelf -h "$SO" | grep -qE 'ELF32'  || die "not ELF32 - this would be useless to Steam Link"
        readelf -h "$SO" | grep -qi '80386'  || die "not i386"
        if readelf -d "$SO" | grep -q TEXTREL; then
            die "DT_TEXTREL present - SELinux deny_execmod would refuse to dlopen() this"
        fi
        ok "ELF32, i386, no text relocations"
    else
        warn "readelf not available - skipping the architecture check"
        warn "(the install step below verifies the ELF class regardless)"
    fi
fi

# ---------------------------------------------------------------- 5. install
if [ "$NO_INSTALL" -eq 1 ]; then
    step "[5/5] Install"
    ok "skipped (--no-install). Built driver is under $BUILD"
    exit 0
fi

step "[5/5] Installing"
if ! run $SUDO cmake --install "$BUILD"; then
    die "install failed (the CMake install step also re-checks the ELF class)"
fi

if [ "$DRY" -eq 0 ]; then
    FOUND=""
    for d in /usr/lib32/dri /usr/lib/i386-linux-gnu/dri; do
        [ -f "$d/bc250_drv_video.so" ] && FOUND="$d/bc250_drv_video.so"
    done
    if [ -n "$FOUND" ]; then
        ok "installed: $FOUND"
        command -v file >/dev/null 2>&1 && echo "    $(file -b "$FOUND")"
    else
        warn "install reported success but no driver found in a 32-bit DRI dir"
    fi
fi

# ----------------------------------------------------------------- epilogue
echo
echo -e "${BOLD}Done.${NC} A few things that decide whether Steam Link actually picks it up:"
echo
echo -e "  1. ${BOLD}LIBVA_DRIVERS_PATH must include the 32-bit directory.${NC}"
echo -e "     The installers write it for you, but an install from before that fix"
echo -e "     will not have it. Check:"
echo -e "       ${BOLD}grep LIBVA_DRIVERS_PATH /etc/environment.d/99-bc250.conf${NC}"
echo -e "     and confirm ${BOLD}/usr/lib32/dri${NC} appears. If not, re-run your installer."
echo
echo -e "  2. ${BOLD}Environment changes only reach newly started processes.${NC}"
echo -e "     Fully quit Steam (not just the stream) and log out/in, or reboot."
echo -e "     This is the single most common reason a correct install looks broken."
echo
echo -e "  3. Shaders are shared with the 64-bit install - nothing extra to do."
echo
echo -e "  To remove: ${BOLD}sudo ./tools/bc250_uninstall.sh${NC} (covers both architectures)"
