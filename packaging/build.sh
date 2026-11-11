#!/usr/bin/env bash
# Build the distribution packages, each inside a container of its own
# distribution, from any machine that has podman or docker.
#
#     packaging/build.sh arch      -> bc250-vaapi-*.pkg.tar.zst
#     packaging/build.sh fedora    -> bc250-vaapi-*.rpm
#     packaging/build.sh debian    -> bc250-vaapi_*.deb
#     packaging/build.sh all
#
# ⚠️ In a container of the target distribution, not on the build machine,
# because a package is only as good as the libraries it was linked
# against. An .rpm produced on Debian against Debian's libva is an .rpm
# that installs on Fedora and does not run there.
#
# Each container builds the working tree, not whatever is on GitHub: the
# tarball is placed where makepkg and rpmbuild expect their download, so
# the packaging here is tested against the code beside it.
#
# ⚠️ And the tarball excludes every build directory. One left behind by
# another distribution's run carries an absolute CMakeCache path, and the
# next cmake stops on it instead of overwriting it.
#
# The finished packages land in packaging/out/.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(dirname "$HERE")
OUT="$HERE/out"

ENGINE=${ENGINE:-}
if [ -z "$ENGINE" ]; then
    if command -v podman >/dev/null; then ENGINE=podman
    elif command -v docker >/dev/null; then ENGINE=docker
    else echo "needs podman or docker" >&2; exit 1
    fi
fi

mkdir -p "$OUT"

# ⚠️ No -w: podman will not create a working directory that is not in the
# image, and fails the run rather than making one.
inside() {
    # inside <image> <script>
    "$ENGINE" run --rm \
        -v "$ROOT":/src:ro \
        -v "$OUT":/out \
        "$1" bash -euxc "mkdir -p /work && cd /work && $2"
}

arch() {
    echo ">>> Arch / CachyOS / SteamOS"
    inside docker.io/library/archlinux:latest '
        pacman -Syu --noconfirm --needed base-devel cmake glslang \
            vulkan-headers vulkan-icd-loader libva libdrm x264 x265 pkgconf
        useradd -m builder
        cp -a /src /work/src
        VERSION=$(sed -n "s/^pkgver=//p" /work/src/packaging/arch/PKGBUILD | tr -d "[:space:]")
        VERSION=${VERSION:-0.5.5}
        tar czf "/work/bc250-vaapi-${VERSION}.tar.gz" \
            --exclude=src/.git --exclude=src/debian \
            --exclude=src/packaging/out \
            --exclude=src/approach1-compute-encoder/build \
            --exclude=src/approach1-compute-encoder/redhat-linux-build \
            --exclude=*.tar.gz --exclude=*.deb --exclude=*.rpm \
            -C /work --transform "s,^src,bc250-vaapi-main," src
        mv "/work/bc250-vaapi-${VERSION}.tar.gz" /work/src/packaging/arch/
        chown -R builder /work
        su builder -c "cd /work/src/packaging/arch && \
            makepkg --noconfirm --skipinteg --nocheck"
        cp /work/src/packaging/arch/*.pkg.tar.* /out/
    '
}

fedora() {
    echo ">>> Fedora / Bazzite / Silverblue"
    inside registry.fedoraproject.org/fedora:latest '
        dnf -y install --nogpgcheck https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm || true
        dnf -y install rpm-build rpmdevtools cmake gcc gcc-c++ make \
            pkgconfig libva-devel libdrm-devel vulkan-headers \
            vulkan-loader-devel glslang tar x264-devel x265-devel || \
        dnf -y install rpm-build rpmdevtools cmake gcc gcc-c++ make \
            pkgconfig libva-devel libdrm-devel vulkan-headers \
            vulkan-loader-devel glslang tar
        rpmdev-setuptree
        cp -a /src /work/src
        VERSION=$(rpm -q --qf "%{version}" --specfile /work/src/packaging/fedora/bc250-vaapi.spec 2>/dev/null | head -1)
        VERSION=${VERSION:-0.5.5}
        tar czf ~/rpmbuild/SOURCES/bc250-vaapi-${VERSION}.tar.gz \
            --exclude=src/.git --exclude=src/debian \
            --exclude=src/packaging/out \
            --exclude=src/approach1-compute-encoder/build \
            --exclude=src/approach1-compute-encoder/redhat-linux-build \
            --exclude=*.tar.gz --exclude=*.deb --exclude=*.rpm \
            -C /work --transform "s,^src,bc250-vaapi-main," src
        RPMBUILD_FLAGS=""
        if ! pkg-config --exists x264 2>/dev/null; then RPMBUILD_FLAGS="$RPMBUILD_FLAGS --without x264"; fi
        if ! pkg-config --exists x265 2>/dev/null; then RPMBUILD_FLAGS="$RPMBUILD_FLAGS --without x265"; fi
        rpmbuild -bb $RPMBUILD_FLAGS /work/src/packaging/fedora/bc250-vaapi.spec
        cp ~/rpmbuild/RPMS/x86_64/*.rpm /out/
    '
}

debian() {
    echo ">>> Debian / Ubuntu"
    inside docker.io/library/debian:trixie '
        apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
            build-essential debhelper cmake pkgconf libva-dev libdrm-dev \
            libvulkan-dev glslang-tools ca-certificates libx264-dev libx265-dev
        cp -a /src /work/src
        cd /work/src
        rm -rf debian approach1-compute-encoder/build approach1-compute-encoder/redhat-linux-build
        cp -a packaging/debian debian
        dpkg-buildpackage -us -uc -b
        cp /work/*.deb /out/
    '
}

case "${1:-all}" in
    arch)   arch ;;
    fedora) fedora ;;
    debian) debian ;;
    all)    arch; fedora; debian ;;
    *)      echo "usage: $0 [arch|fedora|debian|all]" >&2; exit 2 ;;
esac

echo
echo ">>> in $OUT:"
ls -la "$OUT"
