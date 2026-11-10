#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Developer tool: builds OtoMicrophone.driver as you, installs it into /Library/Audio/Plug-Ins/HAL
# as root, restarts coreaudiod (briefly interrupts all audio) and waits until "Oto Microphone"
# appears. The app does not use this script; it installs its bundled driver itself.
#
#   scripts/macos/install-driver.sh            build + install
#   scripts/macos/install-driver.sh --no-build install driver/macos/build as it is
#   SIGN_ID=<identity>                          passed on to the driver Makefile (default: ad-hoc)
#   OTO_SUDO_PASSWORD_FILE=<file>               feed sudo from a file instead of prompting
#
# Run it as yourself, not with sudo: the build must never run as root. Only the copy into
# /Library and the coreaudiod restart use sudo.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$ROOT/driver/macos"
BUNDLE="OtoMicrophone.driver"
DEST_DIR="/Library/Audio/Plug-Ins/HAL"
DEST="$DEST_DIR/$BUNDLE"
STAGE="$DEST_DIR/.$BUNDLE.staging-$$"
OLD="$DEST_DIR/.$BUNDLE.old-$$"

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
[ -d "$BUILT" ] || die "$BUILT is missing. Run without --no-build."
codesign --verify --strict "$BUILT" || die "$BUILT is not validly signed."

# Stage a root-owned copy next to the destination, then swap it in with a rename, so
# coreaudiod never sees a half-copied bundle and a failure leaves the old driver in place.
STAGED=0
SWAPPED=0
cleanup() {
    status=$?
    if [ "$SWAPPED" -eq 0 ]; then
        if [ "$STAGED" -eq 1 ]; then
            as_root rm -rf "$STAGE" || true
        fi
        if [ "$status" -ne 0 ]; then
            echo "Install failed; the previously installed driver (if any) is unchanged." >&2
        fi
    fi
    exit "$status"
}
trap cleanup EXIT

echo "Installing $BUNDLE to $DEST_DIR"
as_root mkdir -p "$DEST_DIR"
STAGED=1
as_root cp -R "$BUILT" "$STAGE"
as_root chown -R root:wheel "$STAGE"
as_root chmod -R go-w "$STAGE"
as_root xattr -dr com.apple.quarantine "$STAGE" 2>/dev/null || true
as_root codesign --verify --strict "$STAGE" || die "the staged copy failed signature verification."

HAD_OLD=0
if [ -e "$DEST" ] || [ -L "$DEST" ]; then
    as_root mv "$DEST" "$OLD" || die "couldn't move the installed driver aside."
    HAD_OLD=1
fi
if ! as_root mv "$STAGE" "$DEST"; then
    if [ "$HAD_OLD" -eq 1 ]; then
        if as_root mv "$OLD" "$DEST"; then
            echo "Restored the previous driver." >&2
        else
            echo "Couldn't restore the previous driver; it is at $OLD" >&2
        fi
    fi
    die "couldn't move the new driver into place."
fi
SWAPPED=1
if [ "$HAD_OLD" -eq 1 ]; then
    as_root rm -rf "$OLD" || echo "warning: couldn't remove $OLD" >&2
fi
echo "Installed $DEST"

echo "Restarting coreaudiod"
# launchctl kickstart is refused under SIP on macOS 27; coreaudiod is relaunched by launchd.
if ! as_root killall coreaudiod; then
    echo "The driver is installed, but coreaudiod could not be restarted." >&2
    echo "Restart your Mac to finish: Oto Microphone appears after the restart." >&2
    exit 0
fi

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
