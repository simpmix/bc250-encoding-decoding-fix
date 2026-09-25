#!/usr/bin/env bash
# bc250-encoding-decoding-fix - https://github.com/simpmix/bc250-encoding-decoding-fix
#
# install_cachyos_arch.sh - Automated Arch Linux & CachyOS Package Installer
#

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}${BOLD}   BC-250 Driver Installer for Arch / CachyOS       ${NC}"
echo -e "${BLUE}======================================================${NC}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARCH_DIR="$REPO_ROOT/packaging/arch"

if [ ! -f "$ARCH_DIR/PKGBUILD" ]; then
    echo -e "${YELLOW}Error: PKGBUILD not found in $ARCH_DIR${NC}"
    exit 1
fi

echo -e "  -> Installing required build dependencies via pacman..."
sudo pacman -S --needed --noconfirm base-devel git cmake meson ninja libva vulkan-headers glslang x264

echo -e "  -> Building and installing bc250-vaapi package..."
cd "$ARCH_DIR"
makepkg -si --noconfirm

echo -e "\n${GREEN}======================================================${NC}"
echo -e "${GREEN}${BOLD}   Package Successfully Installed!                   ${NC}"
echo -e "${GREEN}======================================================${NC}"
echo -e "The driver is now registered system-wide with automatic PCI detection."
echo -e "To verify driver functionality:"
echo -e "  ${YELLOW}LIBVA_DRIVER_NAME=bc250 vainfo${NC}"
