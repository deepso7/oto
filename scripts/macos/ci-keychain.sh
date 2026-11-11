#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Signing and notarization credentials for a release build in GitHub Actions
# (.github/workflows/macos-release.yml). CI only: it refuses to run anywhere else, so it can
# never touch a developer's own keychain. See packaging/macos/README.md, "Releasing from CI".
#
#   scripts/macos/ci-keychain.sh setup     temporary keychain + Developer ID identity +
#                                          notarytool profile; exports SIGN_ID, NOTARY_PROFILE
#                                          and NOTARY_KEYCHAIN to later steps ($GITHUB_ENV)
#   scripts/macos/ci-keychain.sh cleanup   deletes the keychain and the decoded key files
#
# setup reads (secrets):
#   MACOS_CERTIFICATE_P12_BASE64, MACOS_CERTIFICATE_PASSWORD    Developer ID Application .p12
#   and one notarization credential set (the API key wins if both are complete):
#   APPLE_API_KEY_P8_BASE64, APPLE_API_KEY_ID, APPLE_API_ISSUER_ID   App Store Connect API key
#   APPLE_ID, APPLE_TEAM_ID, APPLE_APP_SPECIFIC_PASSWORD             Apple ID fallback
# and optionally SIGN_ID (which identity to use if the .p12 holds more than one) and
# NOTARY_PROFILE (default: oto).
set -euo pipefail

die() {
    echo "::error::$*" >&2
    exit 1
}

[ "${GITHUB_ACTIONS:-}" = true ] || die "ci-keychain.sh only runs in GitHub Actions."
: "${RUNNER_TEMP:?RUNNER_TEMP is not set}"

KEYCHAIN="$RUNNER_TEMP/oto-signing.keychain-db"
WORK="$RUNNER_TEMP/oto-signing"
NOTARY_PROFILE="${NOTARY_PROFILE:-oto}"

# The user keychain search list, one path per line, without the quotes `security` adds.
search_list() {
    security list-keychains -d user | sed -E 's/^[[:space:]]*"(.*)"[[:space:]]*$/\1/'
}

# Decodes base64 from a variable into a file, tolerating line breaks in the secret.
decode() {
    printf '%s' "$1" | tr -d '[:space:]' | base64 --decode > "$2"
    [ -s "$2" ] || die "$3 did not decode to anything; is it base64?"
}

setup() {
    [ -n "${MACOS_CERTIFICATE_P12_BASE64:-}" ] || die "the MACOS_CERTIFICATE_P12_BASE64 secret is not set."
    [ -n "${MACOS_CERTIFICATE_PASSWORD:-}" ] || die "the MACOS_CERTIFICATE_PASSWORD secret is not set."
    [ -n "${GITHUB_ENV:-}" ] || die "GITHUB_ENV is not set."

    # Pick the notarization credentials before changing anything. A half-filled set is an
    # error, not a reason to quietly use the other one.
    local api=0 appleid=0
    if [ -n "${APPLE_API_KEY_P8_BASE64:-}${APPLE_API_KEY_ID:-}${APPLE_API_ISSUER_ID:-}" ]; then
        [ -n "${APPLE_API_KEY_P8_BASE64:-}" ] && [ -n "${APPLE_API_KEY_ID:-}" ] && [ -n "${APPLE_API_ISSUER_ID:-}" ] ||
            die "set all of APPLE_API_KEY_P8_BASE64, APPLE_API_KEY_ID and APPLE_API_ISSUER_ID, or none."
        api=1
    fi
    if [ "$api" -eq 0 ] && [ -n "${APPLE_ID:-}${APPLE_TEAM_ID:-}${APPLE_APP_SPECIFIC_PASSWORD:-}" ]; then
        [ -n "${APPLE_ID:-}" ] && [ -n "${APPLE_TEAM_ID:-}" ] && [ -n "${APPLE_APP_SPECIFIC_PASSWORD:-}" ] ||
            die "set all of APPLE_ID, APPLE_TEAM_ID and APPLE_APP_SPECIFIC_PASSWORD, or none."
        appleid=1
    fi
    [ "$api" -eq 1 ] || [ "$appleid" -eq 1 ] ||
        die "no notarization credentials: set the APPLE_API_KEY_* secrets (preferred) or APPLE_ID, APPLE_TEAM_ID and APPLE_APP_SPECIFIC_PASSWORD."

    rm -rf "$WORK"
    mkdir -m 700 "$WORK"

    local kc_pass
    kc_pass="$(openssl rand -base64 32)"
    echo "::add-mask::$kc_pass"

    echo "Creating $KEYCHAIN"
    [ ! -e "$KEYCHAIN" ] || security delete-keychain "$KEYCHAIN" || rm -f "$KEYCHAIN"
    security create-keychain -p "$kc_pass" "$KEYCHAIN"
    security set-keychain-settings -lut 21600 "$KEYCHAIN" # lock after 6 h idle, and on sleep
    security unlock-keychain -p "$kc_pass" "$KEYCHAIN"

    # Put it first in the search list, keeping what was there (codesign and find-identity only
    # look at the search list).
    local existing=() k
    while IFS= read -r k; do
        [ -n "$k" ] && [ "$k" != "$KEYCHAIN" ] && existing+=("$k")
    done < <(search_list)
    security list-keychains -d user -s "$KEYCHAIN" ${existing[@]+"${existing[@]}"}

    echo "Importing the Developer ID certificate"
    decode "$MACOS_CERTIFICATE_P12_BASE64" "$WORK/cert.p12" MACOS_CERTIFICATE_P12_BASE64
    security import "$WORK/cert.p12" -k "$KEYCHAIN" -f pkcs12 -P "$MACOS_CERTIFICATE_PASSWORD" \
        -T /usr/bin/codesign -T /usr/bin/security > /dev/null ||
        die "couldn't import the .p12 (wrong MACOS_CERTIFICATE_PASSWORD?)."
    rm -f "$WORK/cert.p12"
    # Lets codesign use the key without a (headless, unanswerable) access prompt.
    security set-key-partition-list -S apple-tool:,apple:,codesign: -s -k "$kc_pass" "$KEYCHAIN" > /dev/null

    # The Developer ID intermediates (G2 for current certificates, the original for older ones),
    # in case neither the .p12 nor the runner image has them: without the right one the identity
    # doesn't count as valid. Importing one that's already present is harmless.
    local ca
    for ca in DeveloperIDG2CA DeveloperIDCA; do
        if curl -fsSL --retry 3 -o "$WORK/$ca.cer" "https://www.apple.com/certificateauthority/$ca.cer"; then
            security import "$WORK/$ca.cer" -k "$KEYCHAIN" > /dev/null 2>&1 || true
        else
            echo "::warning::couldn't download Apple's $ca intermediate certificate."
        fi
    done

    # Which identity to sign with: SIGN_ID if given, else the only Developer ID Application
    # identity in the .p12.
    local ids count
    ids="$(security find-identity -v -p codesigning "$KEYCHAIN" |
        sed -n -E 's/^.*"(Developer ID Application: .*)"[[:space:]]*$/\1/p')"
    count="$(grep -c . <<< "$ids" || true)"
    if [ -n "${SIGN_ID:-}" ]; then
        grep -qxF -- "$SIGN_ID" <<< "$ids" ||
            die "SIGN_ID '$SIGN_ID' is not a valid Developer ID Application identity in the .p12 (found: ${ids:-none})."
    elif [ "$count" -eq 1 ]; then
        SIGN_ID="$ids"
    elif [ "$count" -eq 0 ]; then
        security find-identity -p codesigning "$KEYCHAIN" >&2 || true
        die "the .p12 holds no valid Developer ID Application identity (certificate and private key)."
    else
        die "the .p12 holds $count Developer ID Application identities; set the MACOS_SIGN_ID repository variable to one of them."
    fi
    echo "Signing identity: $SIGN_ID"

    echo "Storing notarytool profile '$NOTARY_PROFILE' (credentials are validated with Apple)"
    if [ "$api" -eq 1 ]; then
        local p8="$WORK/AuthKey_$APPLE_API_KEY_ID.p8"
        decode "$APPLE_API_KEY_P8_BASE64" "$p8" APPLE_API_KEY_P8_BASE64
        xcrun notarytool store-credentials "$NOTARY_PROFILE" --key "$p8" \
            --key-id "$APPLE_API_KEY_ID" --issuer "$APPLE_API_ISSUER_ID" --keychain "$KEYCHAIN"
    else
        xcrun notarytool store-credentials "$NOTARY_PROFILE" --apple-id "$APPLE_ID" \
            --team-id "$APPLE_TEAM_ID" --password "$APPLE_APP_SPECIFIC_PASSWORD" --keychain "$KEYCHAIN"
    fi

    {
        echo "SIGN_ID=$SIGN_ID"
        echo "NOTARY_PROFILE=$NOTARY_PROFILE"
        echo "NOTARY_KEYCHAIN=$KEYCHAIN"
    } >> "$GITHUB_ENV"
}

cleanup() {
    if [ -e "$KEYCHAIN" ]; then
        local rest=() k
        while IFS= read -r k; do
            [ -n "$k" ] && [ "$k" != "$KEYCHAIN" ] && rest+=("$k")
        done < <(search_list)
        security list-keychains -d user -s ${rest[@]+"${rest[@]}"} || true
        security delete-keychain "$KEYCHAIN" || rm -f "$KEYCHAIN"
        echo "Deleted $KEYCHAIN"
    fi
    rm -rf "$WORK"
}

case "${1:-}" in
    setup) setup ;;
    cleanup) cleanup ;;
    *) die "usage: $0 setup|cleanup" ;;
esac
