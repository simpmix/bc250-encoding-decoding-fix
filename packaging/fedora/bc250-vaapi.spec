# VA-API video encoding and decoding for the AMD BC-250 (Cyan Skillfish).
# For Bazzite, Silverblue, Kinoite, Fedora and anything else rpm-based.
#
# ⚠️ On an image-based system (Bazzite and the other rpm-ostree ones) this
# is layered, not installed:
#
#     rpm-ostree install ./bc250-vaapi-26.09.2-1.fc*.x86_64.rpm
#     systemctl reboot
#
# ⚠️ Installing it does not switch anything on by itself. The driver is
# only used on a machine that really is a BC-250, decided at session start
# by the environment generator. See that file for why.

%global _commit main
%global debug_package %{nil}

Name:           bc250-vaapi
Version:        26.09.2
Release:        1%{?dist}
Summary:        VA-API video encoding and decoding for the AMD BC-250

License:        GPL-3.0-only
URL:            https://github.com/MTSistemi/bc250-vaapi
Source0:        %{url}/archive/refs/heads/%{_commit}.tar.gz#/%{name}-%{version}.tar.gz

ExclusiveArch:  x86_64

BuildRequires:  gcc
BuildRequires:  gcc-c++
BuildRequires:  cmake
BuildRequires:  pkgconfig
BuildRequires:  libva-devel
BuildRequires:  libdrm-devel
BuildRequires:  vulkan-headers
BuildRequires:  vulkan-loader-devel
BuildRequires:  glslang
# H.264 through libx264, which Fedora itself does not ship: it comes from
# RPM Fusion. Build --without x264 where that is not enabled, and H.264
# falls back to the compute encoder.
%bcond_without x264
%if %{with x264}
BuildRequires:  pkgconfig(x264)
%endif

Requires:       libva
Requires:       libdrm
Requires:       vulkan-loader

%description
The AMD BC-250 has no usable video engine, so recording, streaming and
playback all fall back to the processor. This is a VA-API driver that does
the work itself: it encodes H.264 and HEVC with Vulkan compute shaders on
the board's 40 compute units, and decodes H.264 and H.265 on the
processor. Any application that speaks VA-API finds it, where otherwise it
would find nothing at all - no other VA-API driver initialises on this
hardware.

Measured on the board at 1920x1080: encoding reaches 129 frames per second
with H.264 and 113 with HEVC, decoding 119 with H.264 and 96 with H.265.
Every picture comes out sample for sample as the reference decoder makes
it.

Faster than nothing, not faster than everything: a threaded software
decoder reaches around 360 frames per second on the same processor. This
driver is for an application that asks for VA-API and would otherwise be
told there is none.

%prep
%autosetup -n %{name}-%{_commit}

%build
# ⚠️ THE PREFIX IS NOT A DETAIL. The shaders are looked for at a path
# compiled into the library, which CMake derives from
# CMAKE_INSTALL_PREFIX. With the default they would land under
# /usr/local/share, where a package may not write.
#
# ⚠️ And the tests are deliberately not built: on some versions they are
# missing includes and fail the whole build, while the library itself is
# perfectly well.
cd approach1-compute-encoder
%cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=%{_prefix} \
    -DLIBVA_DRIVERS_PATH=%{_libdir}/dri \
    -DBC250_WITH_X264=%{?with_x264:ON}%{!?with_x264:OFF} \
    -DBUILD_TESTS=OFF
%cmake_build --target bc250_drv_video compile_shaders

%install
cd approach1-compute-encoder
install -Dpm 0644 %{_vpath_builddir}/bc250_drv_video.so \
    %{buildroot}%{_libdir}/dri/bc250_drv_video.so

install -d %{buildroot}%{_datadir}/bc250/shaders
install -pm 0644 %{_vpath_builddir}/*.spv \
    %{buildroot}%{_datadir}/bc250/shaders/

cd ..
install -Dpm 0755 packaging/common/60-bc250-vaapi \
    %{buildroot}%{_prefix}/lib/systemd/user-environment-generators/60-bc250-vaapi

%files
%license LICENSE
%doc README.md
%{_libdir}/dri/bc250_drv_video.so
%dir %{_datadir}/bc250
%dir %{_datadir}/bc250/shaders
%{_datadir}/bc250/shaders/*.spv
%{_prefix}/lib/systemd/user-environment-generators/60-bc250-vaapi

%changelog
* Tue Sep 22 2026 SkillFishOS <info@mtsistemi.it> - 26.09.2-1
- H.264 and H.265 decoding through VAEntrypointVLD, bit-exact against the
  reference decoder
- H.265 decoding reaches 96 frames per second at 1920x1080 on the board
