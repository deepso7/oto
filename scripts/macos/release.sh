#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Builds a downloadable Oto release: a universal, Developer ID signed, notarized and stapled
# Oto.app inside a signed, notarized and stapled DMG, the same Oto.app zipped for the in-app
# updater, their SHA-256s, and the unsigned update manifest. See packaging/macos/README.md.
#
#   SIGN_ID="Developer ID Application: Name (TEAMID)" scripts/macos/release.sh
#
#   SIGN_ID         Developer ID Application identity (name or SHA-1 from
#                   `security find-identity -v -p codesigning`). Required.
#   NOTARY_PROFILE  notarytool keychain profile (default: oto), made once with
#                   `xcrun notarytool store-credentials oto --apple-id … --team-id …`
#   NOTARY_KEYCHAIN  keychain file holding that profile, if not the default keychain (CI sets
#                   this; see scripts/macos/ci-keychain.sh)
#   ALLOW_DIRTY=1   build from a working tree with uncommitted changes (not for real releases)
#
#   MIN_DRIVER_VERSION  oldest loaded driver revision the new engine accepts (default: read
#                   MIN_COMPATIBLE_DRIVER_VERSION from crates/oto-engine/src)
#
# Writes to target/dist:
#   Oto-<version>.dmg, .dmg.sha256            for people downloading Oto
#   Oto-<version>-macos.zip, .zip.sha256      for the in-app updater
#   update-payload.json                       the unsigned manifest; CI signs it into update.json
# The zip is unpacked again with the updater's own extractor (oto-sign extract-verify) and the
# result checked, so a release ships only if the updater can install it. Publishing is a
# separate, deliberate step: the script prints the `gh release create` command and does not
# run it.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
APP="$TARGET_DIR/bundle/Oto.app"
DIST="$TARGET_DIR/dist"
NOTARY_PROFILE="${NOTARY_PROFILE:-oto}"
NOTARY_AUTH=(--keychain-profile "$NOTARY_PROFILE")
[ -z "${NOTARY_KEYCHAIN:-}" ] || NOTARY_AUTH+=(--keychain "$NOTARY_KEYCHAIN")

die() {
    echo "error: $*" >&2
    exit 1
}

[ "$(id -u)" -ne 0 ] || die "run this as yourself, not as root."
[ -n "${SIGN_ID:-}" ] || die "set SIGN_ID to your Developer ID Application identity."
DRIVER_VERSION="$(tr -d '[:space:]' < "$ROOT/driver/macos/DRIVER_VERSION")"
[[ "$DRIVER_VERSION" =~ ^[1-9][0-9]*$ ]] || die "driver/macos/DRIVER_VERSION is not a positive integer."
if [ -z "${MIN_DRIVER_VERSION:-}" ]; then
    MIN_DRIVER_VERSION="$(grep -rhoE 'MIN_COMPATIBLE_DRIVER_VERSION[^=;]*=[[:space:]]*[0-9]+' "$ROOT/crates/oto-engine/src" |
        sed -E 's/.*=[[:space:]]*//' | sort -u)"
fi
[[ "$MIN_DRIVER_VERSION" =~ ^[0-9]+$ ]] ||
    die "couldn't read MIN_COMPATIBLE_DRIVER_VERSION from crates/oto-engine/src (got '$MIN_DRIVER_VERSION'); set MIN_DRIVER_VERSION."
[ "$MIN_DRIVER_VERSION" -le "$DRIVER_VERSION" ] ||
    die "MIN_DRIVER_VERSION ($MIN_DRIVER_VERSION) is above DRIVER_VERSION ($DRIVER_VERSION)."
security find-identity -v -p codesigning | grep -F -- "$SIGN_ID" | grep -q "Developer ID Application" ||
    die "SIGN_ID '$SIGN_ID' is not a Developer ID Application identity in the keychain."
xcrun notarytool history "${NOTARY_AUTH[@]}" > /dev/null 2>&1 ||
    die "no working notarytool profile '$NOTARY_PROFILE'. Create it with:
    xcrun notarytool store-credentials $NOTARY_PROFILE --apple-id <apple-id> --team-id <TEAMID>"

if [ "${ALLOW_DIRTY:-0}" != 1 ] && [ -n "$(git -C "$ROOT" status --porcelain)" ]; then
    die "the working tree has uncommitted changes; commit them or set ALLOW_DIRTY=1."
fi

# Submits a file for notarization and waits. On rejection, prints Apple's log and fails.
notarize() {
    local file="$1" result id status
    echo "Submitting $(basename "$file") for notarization (usually a few minutes)"
    result="$(xcrun notarytool submit "$file" "${NOTARY_AUTH[@]}" \
        --wait --output-format json)" || true
    id="$(plutil -extract id raw - <<< "$result" 2>/dev/null || true)"
    status="$(plutil -extract status raw - <<< "$result" 2>/dev/null || true)"
    if [ "$status" != "Accepted" ]; then
        echo "$result" >&2
        if [ -n "$id" ]; then
            xcrun notarytool log "$id" "${NOTARY_AUTH[@]}" >&2 || true
        fi
        die "notarization of $(basename "$file") failed (status: ${status:-unknown})."
    fi
    echo "Accepted ($id)"
}

echo "==> Bundle"
# FastEnhancer's weights licence is unstated upstream: releases leave the `fastenhancer`
# feature out and ship DeepFilterNet3, DPDFNet-2 and RNNoise (bundle.sh builds the Intel slice
# without `onnx`, which has no prebuilt x86_64 macOS runtime).
OTO_FEATURES=dfn3,onnx UNIVERSAL=1 SIGN_ID="$SIGN_ID" "$ROOT/scripts/macos/bundle.sh"

VERSION="$(plutil -extract CFBundleShortVersionString raw "$APP/Contents/Info.plist")"
ARCHS="$(lipo -archs "$APP/Contents/MacOS/oto-app")"
[[ "$ARCHS" == *arm64* && "$ARCHS" == *x86_64* ]] || die "the app binary is not universal ($ARCHS)."

rm -rf "$DIST"
mkdir -p "$DIST"
WORK="$(mktemp -d "$DIST/.work.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

# 1. The app itself, so it carries its own ticket once copied out of the DMG (and can open
#    offline the first time).
echo "==> Notarize Oto.app"
ditto -c -k --keepParent "$APP" "$WORK/Oto.zip"
notarize "$WORK/Oto.zip"
xcrun stapler staple "$APP"
xcrun stapler validate "$APP"
spctl --assess --type execute -vv "$APP"

# 2. The update archive: the stapled app, zipped without resource forks or extended
#    attributes (the extractor writes plain files and directories only).
ZIP="$DIST/Oto-$VERSION-macos.zip"
echo "==> $ZIP"
ditto -c -k --keepParent --norsrc --noextattr "$APP" "$ZIP"
(cd "$DIST" && shasum -a 256 "$(basename "$ZIP")" > "$(basename "$ZIP").sha256")
ZIP_SHA256="$(cut -d ' ' -f 1 < "$ZIP.sha256")"
ZIP_SIZE="$(stat -f %z "$ZIP")"

echo "==> Round trip through the updater's extractor"
(cd "$ROOT" && cargo build --release --locked -p oto-sign)
OTO_SIGN="$TARGET_DIR/release/oto-sign"
RT_APP="$("$OTO_SIGN" extract-verify --zip "$ZIP" --dest "$WORK/roundtrip" \
    --expected-sha256 "$ZIP_SHA256" --expected-size "$ZIP_SIZE")"
[ "$RT_APP" = "$WORK/roundtrip/Oto.app" ] || die "extract-verify returned '$RT_APP'."
TEAM_ID="$(codesign -dv "$APP" 2>&1 | sed -n 's/^TeamIdentifier=//p')"
[[ "$TEAM_ID" =~ ^[A-Z0-9]{10}$ ]] || die "couldn't read the team id from the app's signature."
# Developer ID Application only (not Apple Development or Mac App Store certs of the same team).
devid_req() {
    echo "identifier \"$1\" and anchor apple generic and certificate 1[field.1.2.840.113635.100.6.2.6] exists and certificate leaf[field.1.2.840.113635.100.6.1.13] exists and certificate leaf[subject.OU] = \"$TEAM_ID\""
}
RT_DRIVER="$RT_APP/Contents/Resources/OtoMicrophone.driver"
RT_HELPER="$RT_APP/Contents/Helpers/oto-hal-helper"
codesign --verify --deep --strict --verbose=2 -R="$(devid_req dev.oto.app)" "$RT_APP"
codesign --verify --strict -R="$(devid_req "$(plutil -extract CFBundleIdentifier raw "$RT_DRIVER/Contents/Info.plist")")" "$RT_DRIVER"
codesign --verify --strict -R="$(devid_req dev.oto.hal-helper)" "$RT_HELPER"
spctl --assess --type execute -vv "$RT_APP"
xcrun stapler validate "$RT_APP"
for bin in "$RT_APP/Contents/MacOS/oto-app" "$RT_HELPER" "$RT_DRIVER/Contents/MacOS/OtoMicrophone"; do
    archs="$(lipo -archs "$bin")"
    [[ "$archs" == *arm64* && "$archs" == *x86_64* ]] || die "$(basename "$bin") is not universal ($archs)."
done
plist() { plutil -extract "$2" raw "$1/Contents/Info.plist"; }
[ "$(plist "$RT_APP" CFBundleIdentifier)" = dev.oto.app ] || die "unexpected CFBundleIdentifier in the zip."
[ "$(plist "$RT_APP" CFBundleExecutable)" = oto-app ] || die "unexpected CFBundleExecutable in the zip."
[ "$(plist "$RT_APP" CFBundleShortVersionString)" = "$VERSION" ] || die "the zipped app is not version $VERSION."
MIN_OS="$(plist "$RT_APP" LSMinimumSystemVersion)"
[[ "$MIN_OS" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]] || die "LSMinimumSystemVersion '$MIN_OS' is not X.Y[.Z]."
BUNDLED_DRIVER="$(plist "$RT_DRIVER" OtoDriverVersion)"
[ "$BUNDLED_DRIVER" = "$DRIVER_VERSION" ] ||
    die "the bundled driver is revision '$BUNDLED_DRIVER' but driver/macos/DRIVER_VERSION is $DRIVER_VERSION."

echo "==> Update manifest"
"$OTO_SIGN" make-payload --version "$VERSION" --zip "$ZIP" --driver-version "$DRIVER_VERSION" \
    --min-driver-version "$MIN_DRIVER_VERSION" --min-os "$MIN_OS" --repo deepso7/oto \
    --out "$DIST/update-payload.json" > /dev/null
PAYLOAD_CHECK="$(plutil -extract platforms.macos-universal.sha256 raw "$DIST/update-payload.json"),$(plutil -extract driver_version raw "$DIST/update-payload.json")"
[ "$PAYLOAD_CHECK" = "$ZIP_SHA256,$DRIVER_VERSION" ] || die "update-payload.json disagrees with the zip or DRIVER_VERSION."

# 3. The DMG around it: Oto.app and a link to /Applications to drag it onto.
DMG="$DIST/Oto-$VERSION.dmg"
echo "==> $DMG"
mkdir "$WORK/dmg"
ditto "$APP" "$WORK/dmg/Oto.app"
ln -s /Applications "$WORK/dmg/Applications"
hdiutil create -volname "Oto $VERSION" -srcfolder "$WORK/dmg" -fs HFS+ -format UDZO \
    -ov "$DMG" > /dev/null
codesign --force --timestamp -s "$SIGN_ID" "$DMG"
codesign --verify --strict "$DMG"

echo "==> Notarize the DMG"
notarize "$DMG"
xcrun stapler staple "$DMG"
xcrun stapler validate "$DMG"
spctl --assess --type open --context context:primary-signature -vv "$DMG"

(cd "$DIST" && shasum -a 256 "$(basename "$DMG")" > "$(basename "$DMG").sha256")

echo
echo "Built $DMG ($(du -h "$DMG" | cut -f1 | tr -d ' '), version $VERSION, $ARCHS)"
echo "      $ZIP ($(du -h "$ZIP" | cut -f1 | tr -d ' '), driver revision $DRIVER_VERSION, macOS $MIN_OS+)"
cat "$DMG.sha256" "$ZIP.sha256"
echo
echo "CI signs $DIST/update-payload.json into update.json and publishes everything. By hand"
echo "(tag the commit first if you haven't; needs OTO_UPDATE_SIGNING_KEY):"
echo "  git tag v$VERSION && git push origin v$VERSION"
echo "  $OTO_SIGN sign --payload '$DIST/update-payload.json' --out '$DIST/update.json'"
echo "  gh release create v$VERSION '$DMG' '$DMG.sha256' '$ZIP' '$ZIP.sha256' '$DIST/update.json' --title 'Oto $VERSION' --generate-notes"
