#!/usr/bin/env bash
# Quick convenience alias for build_and_install.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/build_and_install.sh" "$@"
