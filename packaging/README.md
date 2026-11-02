# Distribution packages

The BC-250 VA-API driver, packaged for the distributions people actually
run on the board.

| | package | driver lands in |
|---|---|---|
| CachyOS, Arch, SteamOS, HoloISO | `arch/PKGBUILD` | `/usr/lib/dri` |
| Bazzite, Silverblue, Kinoite, Fedora | `fedora/bc250-vaapi.spec` | `/usr/lib64/dri` |
| Debian, Ubuntu, SkillFishOS | `debian/` | `/usr/lib/<triplet>/dri` |

All three install the driver, the compiled shaders under
`/usr/share/bc250/shaders`, the `bc250_stats` real-time telemetry monitor,
and one systemd user environment generator.

## Installing

Arch and its family:

    sudo pacman -U bc250-vaapi-*.pkg.tar.zst

Fedora:

    sudo dnf install ./bc250-vaapi-*.rpm

Bazzite, Silverblue, Kinoite and the other image-based ones layer it
instead, and need a reboot for the layer to appear:

    sudo rpm-ostree install ./bc250-vaapi-*.rpm
    systemctl reboot

Debian and Ubuntu:

    sudo apt install ./bc250-vaapi_*_amd64.deb

## It does not switch anything on by itself

Installing the package does not point the system at the driver. That
decision is made at session start, on the machine itself, by
`/usr/lib/systemd/user-environment-generators/60-bc250-vaapi`: it reads
the PCI identifier of the Cyan Skillfish APU straight out of sysfs and
prints `LIBVA_DRIVER_NAME=bc250` only if it finds one.

⚠️ **This is not fussiness.** A fixed `LIBVA_DRIVER_NAME=bc250` in
`/etc/environment.d` would follow a disk image onto an ordinary PC, where
radeonsi or iHD provide a VA-API that a graphics chip actually runs. This
driver decodes on the processor, so there it would take their place and
give back less. On a BC-250 there is nothing to lose: no other VA-API
driver initialises on that hardware at all.

To try it without installing anything, point one command at it:

    LIBVA_DRIVER_NAME=bc250 vainfo

## Building them

Each package is built inside a container of its own distribution, from
any machine with podman or docker:

    packaging/build.sh arch
    packaging/build.sh fedora
    packaging/build.sh debian
    packaging/build.sh all

⚠️ In a container of the target distribution and not on the build
machine, because a package is only as good as the libraries it was
linked against. An `.rpm` produced on Debian against Debian's libva is an
`.rpm` that installs on Fedora and does not run there.

The containers build the working tree, not whatever is on GitHub, so the
packaging here is always tested against the code beside it. Finished
packages land in `packaging/out/`.

## Checking them

    packaging/verify.sh all

This installs each package in a clean container of its distribution and
checks that the driver is where that distribution looks for it, that
every symbol resolves, that all compiled shaders arrived, and that the
generator correctly turns nothing on.

⚠️ **`ldd -r` is the reason this script exists.** A library with an
unresolved symbol is the failure that looks like nothing: libva reports
it not as a missing feature but as a driver that will not open at all,
"Input/output error", without a word about the symbol unless ffmpeg is
asked to be verbose. It has happened here once already.

It needs no BC-250 and is not a test of decoding. For that, run
`tools/test_vaapi_decode.sh` and `tools/test_vaapi_hevc.sh` on the board.
