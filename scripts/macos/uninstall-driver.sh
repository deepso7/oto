#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Developer tool: removes OtoMicrophone.driver and restarts coreaudiod (briefly interrupts all
# audio). Run it as yourself; it uses sudo for the removal and the restart.
#   OTO_SUDO_PASSWORD_FILE=<file>   feed sudo from a file instead of prompting
set -euo pipefail

BUNDLE="OtoMicrophone.driver"
DEST_DIR="/Library/Audio/Plug-Ins/HAL"
DEST="$DEST_DIR/$BUNDLE"
OLD="$DEST_DIR/.$BUNDLE.old-$$"

as_root() {
    if [ -n "${OTO_SUDO_PASSWORD_FILE:-}" ]; then
        sudo -S -p '' "$@" < "$OTO_SUDO_PASSWORD_FILE"
    else
        sudo "$@"
    fi
}

if [ ! -e "$DEST" ] && [ ! -L "$DEST" ]; then
    echo "Not installed."
    exit 0
fi

# Rename first: the bundle leaves its place in one step, whatever happens to the delete.
if ! as_root mv "$DEST" "$OLD"; then
    echo "error: couldn't remove $DEST; nothing was changed." >&2
    exit 1
fi
as_root rm -rf "$OLD" || echo "warning: couldn't delete $OLD" >&2
echo "Removed $DEST"

echo "Restarting coreaudiod"
# launchctl kickstart is refused under SIP on macOS 27; coreaudiod is relaunched by launchd.
if ! as_root killall coreaudiod; then
    echo "The driver is removed, but coreaudiod could not be restarted." >&2
    echo "Restart your Mac to finish: Oto Microphone disappears after the restart." >&2
    exit 0
fi

for _ in $(seq 1 30); do
    if ! system_profiler SPAudioDataType 2>/dev/null | grep -q "Oto Microphone"; then
        echo "Oto Microphone is gone."
        exit 0
    fi
    sleep 1
done
echo "The driver is removed, but Oto Microphone is still listed after 30 s. Restart your Mac." >&2
exit 1
