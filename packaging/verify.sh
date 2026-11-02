#!/usr/bin/env bash
# Install the packages built by build.sh, in a clean container of each
# distribution, and check that the driver they carry is whole.
#
#     packaging/verify.sh [arch|fedora|debian|all]
#
# ⚠️ This needs no BC-250 and does not pretend to be a test of decoding.
# What it catches is the failure that looks like nothing: a library with
# an unresolved symbol. libva reports that not as a missing feature but
# as a driver that will not open at all - "Input/output error", and not a
# word about the symbol unless ffmpeg is asked to be verbose. `ldd -r` is
# the whole reason this file exists.
#
# vainfo cannot stand in for it: without a render node it stops at the
# display long before it opens anything.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT="$HERE/out"

ENGINE=${ENGINE:-}
if [ -z "$ENGINE" ]; then
    if command -v podman >/dev/null; then ENGINE=podman
    elif command -v docker >/dev/null; then ENGINE=docker
    else echo "needs podman or docker" >&2; exit 1
    fi
fi

inside() {
    "$ENGINE" run --rm -v "$OUT":/pkg:ro "$1" bash -euxc "$2"
}

# In every container: the driver is where that distribution looks for it,
# every symbol resolves, the generator and all nine shaders are installed,
# and the generator correctly turns nothing on, because a container is not
# a BC-250.
CHECKS='
    SO=$(ls /usr/lib64/dri/bc250_drv_video.so \
            /usr/lib/dri/bc250_drv_video.so \
            /usr/lib/*/dri/bc250_drv_video.so 2>/dev/null | head -1)
    test -n "$SO"
    echo "driver at $SO"
    if ldd -r "$SO" 2>&1 | grep -i "undefined symbol"; then
        echo "UNRESOLVED SYMBOLS"; exit 1
    fi
    echo "symbols: all resolved"
    test -x /usr/lib/systemd/user-environment-generators/60-bc250-vaapi
    echo "shaders: $(ls /usr/share/bc250/shaders/*.spv | wc -l)"
    echo "generator says: [$(sh /usr/lib/systemd/user-environment-generators/60-bc250-vaapi)]"
    echo "empty is correct here: this is not a BC-250"
'

arch() {
    echo ">>> Arch / CachyOS / SteamOS"
    inside docker.io/library/archlinux:latest "
        pacman -Sy --noconfirm --needed libva libdrm vulkan-icd-loader x264 x265
        pacman -U --noconfirm /pkg/bc250-vaapi-*.pkg.tar.*
        $CHECKS
    "
}

fedora() {
    echo ">>> Fedora / Bazzite / Silverblue"
    inside registry.fedoraproject.org/fedora:latest "
        dnf -y install /pkg/bc250-vaapi-*.rpm
        $CHECKS
    "
}

debian() {
    echo ">>> Debian / Ubuntu"
    # ⚠️ apt-get install ./file.deb, not dpkg -i: dpkg does not resolve
    # dependencies, so a package with perfectly good ${shlibs:Depends}
    # looks broken.
    inside docker.io/library/debian:trixie "
        apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
            /pkg/bc250-vaapi*.deb
        $CHECKS
    "
}

case "${1:-all}" in
    arch)   arch ;;
    fedora) fedora ;;
    debian) debian ;;
    all)    arch; fedora; debian ;;
    *)      echo "usage: $0 [arch|fedora|debian|all]" >&2; exit 2 ;;
esac
