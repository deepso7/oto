#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Developer tool: removes OtoMicrophone.driver through oto-hal-helper (run as root), which
# restarts coreaudiod only if no audio is playing or recording; otherwise the driver disappears
# on your next restart. Run it as yourself; it uses sudo for the helper.
#
#   scripts/macos/uninstall-driver.sh
#   OTO_TEAM_ID=<team>              pin the helper to this Developer ID team
#   OTO_SUDO_PASSWORD_FILE=<file>   feed sudo from a file instead of prompting
#
# The helper comes from driver/macos/build (built with `make -C driver/macos` if missing).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$ROOT/driver/macos"
BUNDLE="OtoMicrophone.driver"
DEST_DIR="/Library/Audio/Plug-Ins/HAL"
DEST="$DEST_DIR/$BUNDLE"
HELPER="$SRC/build/oto-hal-helper"

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

if [ "$(id -u)" -eq 0 ]; then
    die "run this as yourself, not as root or with sudo."
fi

if [ ! -e "$DEST" ] && [ ! -L "$DEST" ]; then
    echo "Not installed."
    exit 0
fi

if [ ! -x "$HELPER" ]; then
    make -C "$SRC" sign-helper
fi

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

STAGE_DIR="$(as_root mktemp -d /private/tmp/oto-driver-uninstall.XXXXXX)"
as_root chmod 0755 "$STAGE_DIR"
as_root cp "$HELPER" "$STAGE_DIR/oto-hal-helper"
as_root chown -R root:wheel "$STAGE_DIR"
as_root chmod -R go-w "$STAGE_DIR"
as_root codesign --verify --strict -R="$HELPER_REQ" "$STAGE_DIR/oto-hal-helper" \
    || die "the staged oto-hal-helper failed signature verification."

OUT="$(as_root "$STAGE_DIR/oto-hal-helper" uninstall "$DEST_DIR" "$BUNDLE" "$(id -u)")" || {
    printf '%s\n' "$OUT"
    die "couldn't remove $DEST; nothing was changed."
}
printf '%s\n' "$OUT" | grep -v '^OTO_' || true

case "$(printf '%s\n' "$OUT" | sed -n 's/^OTO_RESULT=//p' | tail -1)" in
    absent)
        echo "Not installed."
        exit 0
        ;;
    restarted) ;;
    reboot)
        echo "The driver is removed, but audio is in use (or coreaudiod couldn't be restarted)."
        echo "Restart your Mac to finish: Oto Microphone disappears after the restart."
        exit 0
        ;;
    *) die "oto-hal-helper did not report a result." ;;
esac

for _ in $(seq 1 30); do
    if ! system_profiler SPAudioDataType 2>/dev/null | grep -q "Oto Microphone"; then
        echo "Oto Microphone is gone."
        exit 0
    fi
    sleep 1
done
echo "The driver is removed, but Oto Microphone is still listed after 30 s. Restart your Mac." >&2
exit 1
