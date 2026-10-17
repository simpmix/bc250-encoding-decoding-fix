#!/usr/bin/env bash
# bc250-encoding-decoding-fix v0.5.4 - https://github.com/simpmix/bc250-encoding-decoding-fix
#
# apply_mangohud_preset.sh - Configures MangoHud to display BC-250 VA-API encode telemetry on-screen in games

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}${BOLD}   Apply BC-250 MangoHud Performance Overlay Preset   ${NC}"
echo -e "${BLUE}======================================================${NC}"

TARGET_DIR="$HOME/.config/MangoHud"
TARGET_CONF="$TARGET_DIR/MangoHud.conf"

if [[ "$1" == "--remove" || "$1" == "--uninstall" ]]; then
    echo -e "  -> Removing BC-250 overlay hook from MangoHud configuration..."
    if [ -f "$TARGET_CONF" ]; then
        sed -i '/custom_text=BC-250/d' "$TARGET_CONF"
        sed -i '/exec=bc250_stats --hud/d' "$TARGET_CONF"
        sed -i '/exec=\/usr\/local\/bin\/bc250_stats --hud/d' "$TARGET_CONF"
        sed -i '/exec=\/usr\/bin\/bc250_stats --hud/d' "$TARGET_CONF"
        echo -e "  ${GREEN}✓ BC-250 metrics cleanly removed from $TARGET_CONF${NC}"
    fi
    exit 0
fi

mkdir -p "$TARGET_DIR"

# Check if bc250_stats is available in PATH or system binaries
STATS_BIN="bc250_stats"
if ! command -v bc250_stats &> /dev/null; then
    if [ -x "/usr/local/bin/bc250_stats" ]; then
        STATS_BIN="/usr/local/bin/bc250_stats"
    elif [ -x "/usr/bin/bc250_stats" ]; then
        STATS_BIN="/usr/bin/bc250_stats"
    elif [ -x "$(dirname "${BASH_SOURCE[0]}")/bc250_stats" ]; then
        STATS_BIN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/bc250_stats"
    fi
fi

if [ -f "$TARGET_CONF" ]; then
    BACKUP_FILE="${TARGET_CONF}.backup.$(date +%s)"
    echo -e "  -> Existing configuration detected. Creating backup: $BACKUP_FILE"
    cp "$TARGET_CONF" "$BACKUP_FILE"

    # Remove any existing hook lines before appending
    sed -i '/custom_text=BC-250/d' "$TARGET_CONF"
    sed -i '/exec=.*bc250_stats --hud/d' "$TARGET_CONF"

    echo -e "  -> Appending BC-250 telemetry hook to: $TARGET_CONF"
    cat >> "$TARGET_CONF" <<EOF

# AMD BC-250 VA-API Compute Encoder Telemetry Hook
custom_text=BC-250
exec=${STATS_BIN} --hud
EOF
else
    echo -e "  -> Creating new MangoHud configuration at: $TARGET_CONF"
    cat > "$TARGET_CONF" <<EOF
### AMD BC-250 Living Room Console Overlay Configuration
legacy_layout=false
gpu_stats
gpu_temp
cpu_stats
cpu_temp
fps
frametime
frame_timing

# AMD BC-250 VA-API Compute Encoder Telemetry Hook
custom_text=BC-250
exec=${STATS_BIN} --hud
EOF
fi

echo -e "\n${GREEN}======================================================${NC}"
echo -e "${GREEN}${BOLD}   MangoHud Preset Applied Successfully!             ${NC}"
echo -e "${GREEN}======================================================${NC}"
echo -e "\nHow to see BC-250 encoder stats in games:"
echo -e "  1. ${BOLD}Steam Gaming Mode (Bazzite, SteamOS, Gamescope)${NC}:"
echo -e "     Press the Quick Access button (${BOLD}...${NC} or ${BOLD}Guide + A${NC}), go to the ${BOLD}Performance${NC} tab,"
echo -e "     and set the ${BOLD}Performance Overlay Level${NC} slider to 2, 3, or 4."
echo -e "  2. ${BOLD}Desktop / Flatpak / Heroic / Lutris${NC}:"
echo -e "     Launch any game with ${BOLD}MANGOHUD=1 %command%${NC} or enable MangoHud in Steam game properties."
echo -e "\nBehavior:"
echo -e "  * While streaming with Sunshine / Steam Link / OBS, MangoHud will display:"
echo -e "    ${YELLOW}BC250: [T0 GPU] 2.8ms | 60fps | 20.0M${NC}"
echo -e "  * When no stream is active, the metric stays completely silent and hidden."
echo -e "  * To uninstall: ${BOLD}./tools/apply_mangohud_preset.sh --remove${NC}\n"
