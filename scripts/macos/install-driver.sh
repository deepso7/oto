#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Builds and installs OtoMicrophone.driver into /Library/Audio/Plug-Ins/HAL, restarts coreaudiod
# (briefly interrupts all audio) and waits until "Oto Microphone" appears.
#
#   scripts/macos/install-driver.sh            build + install
#   Privileged steps use sudo (interactive prompt by default).
#   OTO_SUDO_PASSWORD_FILE=<file>               feed sudo from a file instead (sudo -S)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$ROOT/driver/macos"
BUNDLE="OtoMicrophone.driver"
DEST="/Library/Audio/Plug-Ins/HAL"

as_root() {
    if [ -n "${OTO_SUDO_PASSWORD_FILE:-}" ]; then
        sudo -S -p '' "$@" < "$OTO_SUDO_PASSWORD_FILE"
    else
        sudo "$@"
    fi
}

make -C "$SRC"

echo "Installing $BUNDLE to $DEST"
as_root rm -rf "$DEST/$BUNDLE"
as_root cp -R "$SRC/build/$BUNDLE" "$DEST/$BUNDLE"
as_root chown -R root:wheel "$DEST/$BUNDLE"

echo "Restarting coreaudiod"
# launchctl kickstart is refused under SIP on macOS 27; coreaudiod is relaunched by launchd.
as_root killall coreaudiod

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
echo "Oto Microphone did not appear. Check: log show --last 2m --predicate 'process == \"coreaudiod\"'" >&2
exit 1
