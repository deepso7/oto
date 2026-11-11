#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Builds a downloadable Oto release: a universal, Developer ID signed, notarized and stapled
# Oto.app inside a signed, notarized and stapled DMG, plus its SHA-256. See
# packaging/macos/README.md.
#
#   SIGN_ID="Developer ID Application: Name (TEAMID)" scripts/macos/release.sh
#
#   SIGN_ID         Developer ID Application identity (name or SHA-1 from
#                   `security find-identity -v -p codesigning`). Required.
#   NOTARY_PROFILE  notarytool keychain profile (default: oto), made once with
#                   `xcrun notarytool store-credentials oto --apple-id … --team-id …`
#   ALLOW_DIRTY=1   build from a working tree with uncommitted changes (not for real releases)
#
# Writes target/dist/Oto-<version>.dmg and Oto-<version>.dmg.sha256. Publishing is a separate,
# deliberate step: the script prints the `gh release create` command and does not run it.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
APP="$TARGET_DIR/bundle/Oto.app"
DIST="$TARGET_DIR/dist"
NOTARY_PROFILE="${NOTARY_PROFILE:-oto}"

die() {
    echo "error: $*" >&2
    exit 1
}

[ "$(id -u)" -ne 0 ] || die "run this as yourself, not as root."
[ -n "${SIGN_ID:-}" ] || die "set SIGN_ID to your Developer ID Application identity."
security find-identity -v -p codesigning | grep -F -- "$SIGN_ID" | grep -q "Developer ID Application" ||
    die "SIGN_ID '$SIGN_ID' is not a Developer ID Application identity in the keychain."
xcrun notarytool history --keychain-profile "$NOTARY_PROFILE" > /dev/null 2>&1 ||
    die "no working notarytool profile '$NOTARY_PROFILE'. Create it with:
    xcrun notarytool store-credentials $NOTARY_PROFILE --apple-id <apple-id> --team-id <TEAMID>"

if [ "${ALLOW_DIRTY:-0}" != 1 ] && [ -n "$(git -C "$ROOT" status --porcelain)" ]; then
    die "the working tree has uncommitted changes; commit them or set ALLOW_DIRTY=1."
fi

# Submits a file for notarization and waits. On rejection, prints Apple's log and fails.
notarize() {
    local file="$1" result id status
    echo "Submitting $(basename "$file") for notarization (usually a few minutes)"
    result="$(xcrun notarytool submit "$file" --keychain-profile "$NOTARY_PROFILE" \
        --wait --output-format json)" || true
    id="$(plutil -extract id raw - <<< "$result" 2>/dev/null || true)"
    status="$(plutil -extract status raw - <<< "$result" 2>/dev/null || true)"
    if [ "$status" != "Accepted" ]; then
        echo "$result" >&2
        if [ -n "$id" ]; then
            xcrun notarytool log "$id" --keychain-profile "$NOTARY_PROFILE" >&2 || true
        fi
        die "notarization of $(basename "$file") failed (status: ${status:-unknown})."
    fi
    echo "Accepted ($id)"
}

echo "==> Bundle"
UNIVERSAL=1 SIGN_ID="$SIGN_ID" "$ROOT/scripts/macos/bundle.sh"

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

# 2. The DMG around it: Oto.app and a link to /Applications to drag it onto.
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
cat "$DMG.sha256"
echo
echo "To publish (tag the commit first if you haven't):"
echo "  git tag v$VERSION && git push origin v$VERSION"
echo "  gh release create v$VERSION '$DMG' '$DMG.sha256' --title 'Oto $VERSION' --generate-notes"
