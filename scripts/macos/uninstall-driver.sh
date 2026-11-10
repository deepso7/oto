#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Removes OtoMicrophone.driver and restarts coreaudiod (briefly interrupts all audio).
#   OTO_SUDO_PASSWORD_FILE=<file>   feed sudo from a file instead of prompting
set -euo pipefail

TARGET="/Library/Audio/Plug-Ins/HAL/OtoMicrophone.driver"

as_root() {
    if [ -n "${OTO_SUDO_PASSWORD_FILE:-}" ]; then
        sudo -S -p '' "$@" < "$OTO_SUDO_PASSWORD_FILE"
    else
        sudo "$@"
    fi
}

if [ ! -e "$TARGET" ]; then
    echo "Not installed."
    exit 0
fi
as_root rm -rf "$TARGET"
echo "Restarting coreaudiod"
# launchctl kickstart is refused under SIP on macOS 27; coreaudiod is relaunched by launchd.
as_root killall coreaudiod

for _ in $(seq 1 30); do
    if ! system_profiler SPAudioDataType 2>/dev/null | grep -q "Oto Microphone"; then
        echo "Removed."
        exit 0
    fi
    sleep 1
done
echo "Oto Microphone still listed after 30 s." >&2
exit 1
