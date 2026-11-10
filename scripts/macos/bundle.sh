#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Builds target/bundle/Oto.app: the release oto-app binary, its Info.plist, the licence and
# third-party notice files, and the prebuilt driver the app installs
# (Contents/Resources/OtoMicrophone.driver). See packaging/macos/README.md.
#
#   scripts/macos/bundle.sh                                         ad-hoc signed, for this Mac
#   SIGN_ID="Developer ID Application: Name (TEAMID)" scripts/macos/bundle.sh
#                                                                   hardened runtime + timestamp
#
# SIGN_ID is a certificate name or SHA-1 hash from `security find-identity -v -p codesigning`.
# Run as yourself, never as root. Notarization is NOT done here.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PKG="$ROOT/packaging/macos"
TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
OUT="$TARGET_DIR/bundle"
APP="$OUT/Oto.app"
DRIVER="OtoMicrophone.driver"
SIGN_ID="${SIGN_ID:--}"

die() {
    echo "error: $*" >&2
    exit 1
}

[ "$(id -u)" -ne 0 ] || die "run this as yourself, not as root."

VERSION="$(awk '/^\[workspace\.package\]/ { in_pkg = 1; next } /^\[/ { in_pkg = 0 }
    in_pkg && $1 == "version" { gsub(/"/, "", $3); print $3; exit }' "$ROOT/Cargo.toml")"
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "couldn't read the version from Cargo.toml (got '$VERSION')."

# Work out what kind of identity SIGN_ID is. Only a Developer ID build gets the hardened
# runtime, a secure timestamp, entitlements, and a team id baked into the app (which then
# refuses to install a driver signed by anyone else).
RELEASE=0
TEAM_ID=""
if [ "$SIGN_ID" != "-" ]; then
    IDENTITY="$(security find-identity -v -p codesigning | grep -F -- "$SIGN_ID" | head -n 1 || true)"
    [ -n "$IDENTITY" ] || die "SIGN_ID '$SIGN_ID' is not a code-signing identity in the keychain."
    if [[ "$IDENTITY" == *"Developer ID Application"* ]]; then
        RELEASE=1
        TEAM_ID="$(sed -n -E 's/.*\(([A-Z0-9]{10})\)"[[:space:]]*$/\1/p' <<< "$IDENTITY")"
        [ -n "$TEAM_ID" ] || die "couldn't read the team id from '$IDENTITY'."
    fi
fi

echo "==> Driver"
if [ "$RELEASE" -eq 1 ]; then
    make -C "$ROOT/driver/macos" release SIGN_ID="$SIGN_ID"
else
    make -C "$ROOT/driver/macos" SIGN_ID="$SIGN_ID"
fi

echo "==> oto-app $VERSION (release)"
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-12.0}"
if [ "$RELEASE" -eq 1 ]; then
    (cd "$ROOT" && OTO_TEAM_ID="$TEAM_ID" cargo build --release -p oto-app)
else
    (cd "$ROOT" && env -u OTO_TEAM_ID cargo build --release -p oto-app)
fi

echo "==> $APP"
mkdir -p "$OUT"
rm -rf "$OUT/Oto.app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$TARGET_DIR/release/oto-app" "$APP/Contents/MacOS/oto-app"
sed "s/@VERSION@/$VERSION/g" "$PKG/Info.plist.in" > "$APP/Contents/Info.plist"
plutil -lint -s "$APP/Contents/Info.plist" || die "the generated Info.plist is invalid."
printf 'APPL????' > "$APP/Contents/PkgInfo"
cp "$ROOT/LICENSE-MIT" "$ROOT/LICENSE-APACHE" "$ROOT/THIRD_PARTY_NOTICES.md" "$APP/Contents/Resources/"
# The driver is already signed by its Makefile; cp -R keeps that signature intact.
cp -R "$ROOT/driver/macos/build/$DRIVER" "$APP/Contents/Resources/$DRIVER"
xattr -cr "$APP"

# Inside-out: the driver's signature (made above) is verified first, then the app is signed
# around it. No --deep when signing: each piece is signed with its own options.
echo "==> Signing"
codesign --verify --strict "$APP/Contents/Resources/$DRIVER"
if [ "$RELEASE" -eq 1 ]; then
    codesign --force --options runtime --timestamp \
        --entitlements "$PKG/Oto.entitlements" -s "$SIGN_ID" "$APP"
elif [ "$SIGN_ID" = "-" ]; then
    echo "signing ad-hoc"
    codesign --force --timestamp=none -s - "$APP"
else
    codesign --force --timestamp=none -s "$SIGN_ID" "$APP"
fi

codesign --verify --deep --strict --verbose=2 "$APP"
# --deep does not descend into Resources, so check the embedded driver survived the copy.
codesign --verify --strict "$APP/Contents/Resources/$DRIVER"
echo
echo "Built $APP ($(du -sh "$APP" | cut -f1 | tr -d " "), version $VERSION, $(lipo -archs "$APP/Contents/MacOS/oto-app"))"
if [ "$RELEASE" -eq 1 ]; then
    echo "Signed with Developer ID, team $TEAM_ID. NOT notarized: see packaging/macos/README.md."
else
    echo "Not a distributable build (no Developer ID signature, not notarized)."
fi
