#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Exercises oto-hal-helper's lock, naming and swap logic against a scratch directory as an
# ordinary user (--no-restart --no-verify: no root, no codesign, no coreaudiod).
#   tools/test-hal-helper.sh build/oto-hal-helper
set -euo pipefail

HELPER="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/oto-hal-helper-test.XXXXXX")"
trap 'rm -rf "$SCRATCH"' EXIT
HAL="$SCRATCH/HAL"
NAME="OtoMicrophone.driver"
mkdir -p "$HAL"
FAILS=0

pass() { echo "  [PASS] $*"; }
fail() { echo "  [FAIL] $*"; FAILS=$((FAILS + 1)); }
expect() { if eval "$1"; then pass "$2"; else fail "$2"; fi; }

# make_driver <dir> <revision|legacy> <marker>
make_driver() {
    mkdir -p "$1/Contents/MacOS"
    echo "$3" > "$1/Contents/MacOS/OtoMicrophone"
    if [ "$2" = legacy ]; then
        plutil -create xml1 "$1/Contents/Info.plist"
        plutil -insert CFBundleVersion -string 0.0.3 "$1/Contents/Info.plist"
    else
        plutil -create xml1 "$1/Contents/Info.plist"
        plutil -insert OtoDriverVersion -integer "$2" "$1/Contents/Info.plist"
    fi
}

run() { "$HELPER" "$@" --no-restart --no-verify; }
leftovers() { find "$HAL" -maxdepth 1 -name ".$NAME.*" ! -name ".$NAME.lock" | wc -l | tr -d ' '; }

echo "fresh install"
make_driver "$SCRATCH/v1" 1 one
OUT="$(run install "$SCRATCH/v1" "$HAL" "$NAME" "$(id -u)")"
expect '[ "$(cat "$HAL/$NAME/Contents/MacOS/OtoMicrophone")" = one ]' "driver is in place"
expect 'grep -qx OTO_FROM=absent <<<"$OUT"' "reports OTO_FROM=absent"
expect 'grep -qx OTO_TO=1 <<<"$OUT"' "reports OTO_TO=1"
expect 'grep -qx OTO_RESULT=installed <<<"$OUT"' "reports OTO_RESULT=installed"
expect '[ -f "$HAL/.$NAME.lock" ]' "lock file persists"
expect '[ "$(stat -f %Lp "$HAL/.$NAME.lock")" = 600 ]' "lock file is 0600"
expect '[ "$(leftovers)" = 0 ]' "no leftovers"
expect '[ -d "$SCRATCH/v1" ]' "source is untouched"

echo "upgrade over legacy, with stale leftovers"
rm -rf "${HAL:?}/$NAME"
make_driver "$HAL/$NAME" legacy legacy
make_driver "$HAL/.$NAME.old-123" legacy stale-old
make_driver "$HAL/.$NAME.swap-9-deadbeef" 1 stale-swap
make_driver "$HAL/.$NAME.removing-9-deadbeef" 1 stale-removing
make_driver "$HAL/.$NAME.staging-77" 1 stale-staging
make_driver "$SCRATCH/v2" 2 two
OUT="$(run install "$SCRATCH/v2" "$HAL" "$NAME" "$(id -u)")"
expect 'grep -qx OTO_FROM=0 <<<"$OUT"' "legacy driver reads as revision 0"
expect '[ "$(cat "$HAL/$NAME/Contents/MacOS/OtoMicrophone")" = two ]' "new driver swapped in"
expect '[ "$(leftovers)" = 0 ]' "swapped-out copy and stale leftovers removed"

echo "downgrade refused"
set +e
OUT="$(run install "$SCRATCH/v1" "$HAL" "$NAME" "$(id -u)" 2>&1)"
STATUS=$?
set -e
expect '[ "$STATUS" = 3 ]' "exit status 3"
expect '[ "$(cat "$HAL/$NAME/Contents/MacOS/OtoMicrophone")" = two ]' "installed driver unchanged"
expect '[ "$(leftovers)" = 0 ]' "no staging left behind"
OUT="$(run install "$SCRATCH/v1" "$HAL" "$NAME" "$(id -u)" --allow-downgrade)"
expect '[ "$(cat "$HAL/$NAME/Contents/MacOS/OtoMicrophone")" = one ]' "--allow-downgrade installs"

echo "lock serializes callers"
/usr/bin/python3 -c 'import fcntl, sys, time
f = open(sys.argv[1], "a"); fcntl.flock(f, fcntl.LOCK_EX)
open(sys.argv[2], "w").close(); time.sleep(1.5)' "$HAL/.$NAME.lock" "$SCRATCH/locked" &
HOLDER=$!
for _ in $(seq 1 100); do [ -e "$SCRATCH/locked" ] && break; sleep 0.05; done
START=$(date +%s)
OUT="$(run install "$SCRATCH/v2" "$HAL" "$NAME" "$(id -u)" 2>&1)"
END=$(date +%s)
wait "$HOLDER"
expect 'grep -q "waiting for another driver install" <<<"$OUT"' "second caller waits"
expect '[ $((END - START)) -ge 1 ]' "and proceeds only after the holder exits"
expect '[ "$(cat "$HAL/$NAME/Contents/MacOS/OtoMicrophone")" = two ]' "then installs"

echo "uninstall"
make_driver "$HAL/.$NAME.old-5" legacy stale-old
OUT="$(run uninstall "$HAL" "$NAME" "$(id -u)")"
expect 'grep -qx OTO_RESULT=removed <<<"$OUT"' "reports OTO_RESULT=removed"
expect '[ ! -e "$HAL/$NAME" ]' "driver removed"
expect '[ "$(leftovers)" = 0 ]' "no .removing-* or .old-* left"
OUT="$(run uninstall "$HAL" "$NAME" "$(id -u)")"
expect 'grep -qx OTO_RESULT=absent <<<"$OUT"' "second uninstall reports OTO_RESULT=absent"

echo "bad input"
set +e
"$HELPER" install "$SCRATCH/v1" "$HAL" "../evil.driver" "$(id -u)" --no-restart >/dev/null 2>&1
expect '[ $? = 2 ]' "rejects a driver name with a slash"
"$HELPER" install "$SCRATCH/v1" "$HAL" "$NAME" "$(id -u)" >/dev/null 2>&1
expect '[ $? = 2 ]' "refuses to restart coreaudiod when not root"
"$HELPER" install "$SCRATCH/missing" "$HAL" "$NAME" "$(id -u)" --no-restart >/dev/null 2>&1
expect '[ $? = 1 ]' "fails on a missing stage"
set -e
expect '[ ! -e "$HAL/$NAME" ]' "nothing installed by failed calls"

echo
if [ "$FAILS" -eq 0 ]; then echo "RESULT: PASS"; else echo "RESULT: FAIL ($FAILS)"; exit 1; fi
