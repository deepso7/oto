#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Developer tool: builds OtoMicrophone.driver and oto-hal-helper as you, then has the helper (run
# as root) install the driver into /Library/Audio/Plug-Ins/HAL and restart coreaudiod if no audio
# is playing or recording (otherwise the driver loads on your next restart), and waits until
# "Oto Microphone" appears. The app does not use this script; it installs its bundled driver
# itself through the same helper.
#
#   scripts/macos/install-driver.sh            build + install
#   scripts/macos/install-driver.sh --no-build install driver/macos/build as it is
#   SIGN_ID=<identity>                          passed on to the driver Makefile (default: ad-hoc)
#   OTO_TEAM_ID=<team>                          pin the helper to this Developer ID team
#   OTO_SUDO_PASSWORD_FILE=<file>               feed sudo from a file instead of prompting
#
# Run it as yourself, not with sudo: the build must never run as root. Only the helper runs as
# root. It takes a kernel lock, swaps the driver in atomically (the old one never goes missing),
# and allows a downgrade, since this is a developer tool.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$ROOT/driver/macos"
BUNDLE="OtoMicrophone.driver"
DEST_DIR="/Library/Audio/Plug-Ins/HAL"

die() {
    echo "error: $*" >&2
    exit 1
}

as_root() {
    if [ -n "${OTO_SUDO_PASSWORD_FILE:-}" ]; then
        sudo -S -p '' "$@" < "$OTO_SUDO_PASSWORD_FILE"
    else
        sudo "$@"
    fi
}

BUILD=1
case "${1:-}" in
    "") ;;
    --no-build) BUILD=0 ;;
    *) die "unknown argument '$1' (usage: $0 [--no-build])" ;;
esac

if [ "$(id -u)" -eq 0 ]; then
    die "run this as yourself, not as root or with sudo: the driver is never built as root. The script asks for sudo when it installs."
fi

if [ "$BUILD" -eq 1 ]; then
    make -C "$SRC"
fi

BUILT="$SRC/build/$BUNDLE"
HELPER="$SRC/build/oto-hal-helper"
[ -d "$BUILT" ] || die "$BUILT is missing. Run without --no-build."
[ -x "$HELPER" ] || die "$HELPER is missing. Run without --no-build."
codesign --verify --strict "$BUILT" || die "$BUILT is not validly signed."

# The helper is pinned by identifier (and, for Developer ID builds, team) and run from a
# root-owned copy verified after copying, so what runs as root is what was checked.
if [ -n "${OTO_TEAM_ID:-}" ]; then
    HELPER_REQ="anchor apple generic and identifier \"dev.oto.hal-helper\" and certificate leaf[subject.OU] = \"$OTO_TEAM_ID\""
else
    HELPER_REQ='identifier "dev.oto.hal-helper"'
fi

STAGE_DIR=""
cleanup() {
    if [ -n "$STAGE_DIR" ]; then
        as_root rm -rf "$STAGE_DIR" || true
    fi
}
trap cleanup EXIT

STAGE_DIR="$(as_root mktemp -d /private/tmp/oto-driver-install.XXXXXX)"
as_root chmod 0755 "$STAGE_DIR"
as_root cp "$HELPER" "$STAGE_DIR/oto-hal-helper"
as_root cp -R "$BUILT" "$STAGE_DIR/$BUNDLE"
as_root chown -R root:wheel "$STAGE_DIR"
as_root chmod -R go-w "$STAGE_DIR"
as_root codesign --verify --strict -R="$HELPER_REQ" "$STAGE_DIR/oto-hal-helper" \
    || die "the staged oto-hal-helper failed signature verification."
as_root codesign --verify --strict "$STAGE_DIR/$BUNDLE" || die "the staged driver failed signature verification."

echo "Installing $BUNDLE to $DEST_DIR"
as_root mkdir -p "$DEST_DIR"
OUT="$(as_root "$STAGE_DIR/oto-hal-helper" install "$STAGE_DIR/$BUNDLE" "$DEST_DIR" "$BUNDLE" "$(id -u)" \
    --allow-downgrade)" || {
    printf '%s\n' "$OUT"
    die "install failed; the previously installed driver (if any) is unchanged."
}
printf '%s\n' "$OUT" | grep -v '^OTO_' || true

case "$(printf '%s\n' "$OUT" | sed -n 's/^OTO_RESULT=//p' | tail -1)" in
    restarted) ;;
    reboot)
        echo "The driver is installed, but audio is in use (or coreaudiod couldn't be restarted)."
        echo "Restart your Mac when your audio is idle to finish, or rerun this script."
        exit 0
        ;;
    *) die "oto-hal-helper did not report a result." ;;
esac

echo -n "Waiting for Oto Microphone"
for _ in $(seq 1 30); do
    if system_profiler SPAudioDataType 2>/dev/null | grep -q "Oto Microphone"; then
        echo " - present."
        exit 0
    fi
    echo -n "."
    sleep 1
done
echo
echo "The driver is installed, but Oto Microphone did not appear within 30 s." >&2
echo "Restart your Mac, or check: log show --last 2m --predicate 'process == \"coreaudiod\"'" >&2
exit 1
