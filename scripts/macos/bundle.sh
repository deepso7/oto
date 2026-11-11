#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Builds target/bundle/Oto.app: the release oto-app binary, its Info.plist, the licence and
# third-party notice files, and the prebuilt driver the app installs
# (Contents/Resources/OtoMicrophone.driver). See packaging/macos/README.md.
#
#   scripts/macos/bundle.sh                                         ad-hoc signed, for this Mac
#   SIGN_ID="Developer ID Application: Name (TEAMID)" scripts/macos/bundle.sh
#                                                                   hardened runtime + timestamp
#   UNIVERSAL=1 scripts/macos/bundle.sh                             arm64 + x86_64 app binary
#
# UNIVERSAL defaults to 1 for Developer ID builds and 0 otherwise (host architecture only). It
# needs both Rust targets: rustup target add aarch64-apple-darwin x86_64-apple-darwin
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

UNIVERSAL="${UNIVERSAL:-$RELEASE}"
if [ "$UNIVERSAL" = 1 ]; then
    TARGETS=(aarch64-apple-darwin x86_64-apple-darwin)
    for t in "${TARGETS[@]}"; do
        (cd "$ROOT" && rustup target list --installed) | grep -qx "$t" ||
            die "UNIVERSAL=1 needs the $t Rust target: rustup target add $t"
    done
else
    TARGETS=()
fi

echo "==> Driver"
if [ "$RELEASE" -eq 1 ]; then
    make -C "$ROOT/driver/macos" release SIGN_ID="$SIGN_ID" VERSION="$VERSION"
else
    make -C "$ROOT/driver/macos" SIGN_ID="$SIGN_ID" VERSION="$VERSION"
fi

export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-12.0}"
# OTO_FEATURES replaces the app's default cargo features, e.g. "dfn3,onnx" for a release.
FEATURE_ARGS=()
if [ -n "${OTO_FEATURES:-}" ]; then
    FEATURE_ARGS=(--no-default-features --features "$OTO_FEATURES")
fi
build_app() {
    if [ "$RELEASE" -eq 1 ]; then
        (cd "$ROOT" && OTO_TEAM_ID="$TEAM_ID" cargo build --release -p oto-app ${FEATURE_ARGS[@]+"${FEATURE_ARGS[@]}"} "$@")
    else
        (cd "$ROOT" && env -u OTO_TEAM_ID cargo build --release -p oto-app ${FEATURE_ARGS[@]+"${FEATURE_ARGS[@]}"} "$@")
    fi
}
if [ "$UNIVERSAL" = 1 ]; then
    BINS=()
    for t in "${TARGETS[@]}"; do
        echo "==> oto-app $VERSION (release, $t)"
        if [ "$t" = x86_64-apple-darwin ]; then
            # ONNX Runtime (`ort`) has no prebuilt x86_64 macOS library, so Intel Macs get
            # DeepFilterNet3 and RNNoise only; the app offers just the models in its slice.
            (FEATURE_ARGS=(--no-default-features --features "${OTO_FEATURES_X86_64:-dfn3}")
                build_app --target "$t")
        else
            build_app --target "$t"
        fi
        BINS+=("$TARGET_DIR/$t/release/oto-app")
    done
    APP_BIN="$OUT/oto-app.universal"
    mkdir -p "$OUT"
    lipo -create -output "$APP_BIN" "${BINS[@]}"
else
    echo "==> oto-app $VERSION (release)"
    build_app
    APP_BIN="$TARGET_DIR/release/oto-app"
fi

echo "==> $APP"
mkdir -p "$OUT"
rm -rf "$OUT/Oto.app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$APP_BIN" "$APP/Contents/MacOS/oto-app"
sed "s/@VERSION@/$VERSION/g" "$PKG/Info.plist.in" > "$APP/Contents/Info.plist"
# The app icon is optional until there is artwork: packaging/macos/AppIcon.icns, if present.
if [ -f "$PKG/AppIcon.icns" ]; then
    cp "$PKG/AppIcon.icns" "$APP/Contents/Resources/AppIcon.icns"
    plutil -insert CFBundleIconFile -string AppIcon "$APP/Contents/Info.plist"
fi
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
    echo "Signed with Developer ID, team $TEAM_ID. NOT notarized: scripts/macos/release.sh does that."
else
    echo "Not a distributable build (no Developer ID signature, not notarized)."
fi
