#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# The `publish` job of .github/workflows/macos-release.yml. Runs under the global
# `oto-release-publish` concurrency lock, on Linux, with the `release` environment's secrets.
# See packaging/macos/README.md, "Releasing from CI".
#
#   TAG                     vX.Y.Z
#   SHA                     the commit validate-tag verified the tag points to
#   DMG_RESULT              result of the `dmg` job (success / skipped / ...)
#   ARTIFACTS               directory holding the `dmg` job's artifact (draft releases only)
#   OTO_SIGN                  path to a built oto-sign
#   OTO_UPDATE_SIGNING_KEY    <key_id>:<base64 seed>[,...] (required for drafts; optional on re-runs)
#   OTO_UPDATE_VERIFY_PUBKEYS <key_id>:<base64 pub>[,...] (optional; used to verify published releases,
#                             including historical releases signed by retired keys, without private seeds)
#   HOMEBREW_TAP_TOKEN        token that can push to $HOMEBREW_TAP (optional; no tap update without)
#   HOMEBREW_TAP            owner/name of the tap (default deepso7/homebrew-oto)
#   GITHUB_REPOSITORY, GH_TOKEN  as set by Actions
#
# If the release is still a draft: signs update-payload.json into update.json, uploads all
# assets, verifies them through the API, publishes (marked latest only if it is the newest
# version), then updates the tap. If it is already published (immutable): uploads nothing,
# re-verifies the published assets and envelope, and reconciles the tap. Both paths are safe
# to re-run.
set -euo pipefail

: "${TAG:?}" "${SHA:?}" "${OTO_SIGN:?}" "${GITHUB_REPOSITORY:?}"
REPO="$GITHUB_REPOSITORY"
HOMEBREW_TAP="${HOMEBREW_TAP:-deepso7/homebrew-oto}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VERSION="${TAG#v}"
DMG="Oto-$VERSION.dmg"
ZIP="Oto-$VERSION-macos.zip"
ASSETS=("$DMG" "$DMG.sha256" "$ZIP" "$ZIP.sha256" update.json)

die() {
    echo "::error::$*" >&2
    exit 1
}

[[ "$TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "tag '$TAG' is not vX.Y.Z"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

sha256() { sha256sum "$1" | cut -d ' ' -f 1; }
size() { stat -c %s "$1"; }

# --- Under the lock: the tag still points at the verified commit, and the draft state now. ---
tag_commit() {
    local ref type sha
    ref="$(gh api "repos/$REPO/git/ref/tags/$TAG" --jq '.object.type + " " + .object.sha')" || return 1
    read -r type sha <<< "$ref"
    # Annotated tags point at a tag object; peel to the commit.
    while [ "$type" = tag ]; do
        ref="$(gh api "repos/$REPO/git/tags/$sha" --jq '.object.type + " " + .object.sha')" || return 1
        read -r type sha <<< "$ref"
    done
    [ "$type" = commit ] || return 1
    echo "$sha"
}
REMOTE_SHA="$(tag_commit)" || die "couldn't resolve refs/tags/$TAG on $REPO"
[ "$REMOTE_SHA" = "$SHA" ] || die "refs/tags/$TAG now points to $REMOTE_SHA, not the verified $SHA; refusing to publish"
IS_DRAFT="$(gh release view "$TAG" --repo "$REPO" --json isDraft --jq .isDraft)" ||
    die "there is no GitHub release for $TAG"
echo "$TAG → $SHA, draft: $IS_DRAFT"

# --- The asset set to check: local files (draft) or freshly downloaded ones (published). ---
LOCAL="$WORK/assets"
mkdir -p "$LOCAL"
if [ "$IS_DRAFT" = true ]; then
    [ -n "${OTO_UPDATE_SIGNING_KEY:-}" ] ||
        die "OTO_UPDATE_SIGNING_KEY is not set in the release environment"
    [ "${DMG_RESULT:-}" = success ] ||
        die "$TAG is still a draft but the build job did not succeed (${DMG_RESULT:-unknown}); re-run: gh workflow run macos-release.yml --ref main -f tag=$TAG"
    : "${ARTIFACTS:?}"
    for f in "$DMG" "$DMG.sha256" "$ZIP" "$ZIP.sha256" update-payload.json; do
        [ -f "$ARTIFACTS/$f" ] || die "the build artifact has no $f"
        cp "$ARTIFACTS/$f" "$LOCAL/$f"
    done
    "$OTO_SIGN" sign --payload "$LOCAL/update-payload.json" --out "$LOCAL/update.json"
    rm "$LOCAL/update-payload.json"
else
    [ "$IS_DRAFT" = false ] || die "unexpected isDraft '$IS_DRAFT'"
    echo "$TAG is already published: verifying its assets, uploading nothing."
    gh release download "$TAG" --repo "$REPO" --dir "$LOCAL" \
        --pattern "$DMG" --pattern "$DMG.sha256" --pattern "$ZIP" --pattern "$ZIP.sha256" --pattern update.json ||
        die "couldn't download the published assets of $TAG"
fi

# --- Content checks: checksums, envelope signature, and manifest agree with the files. ---
for f in "${ASSETS[@]}"; do [ -f "$LOCAL/$f" ] || die "missing $f"; done
DMG_SHA="$(sha256 "$LOCAL/$DMG")"
ZIP_SHA="$(sha256 "$LOCAL/$ZIP")"
[ "$(cut -d ' ' -f 1 < "$LOCAL/$DMG.sha256")" = "$DMG_SHA" ] || die "$DMG.sha256 does not match $DMG"
[ "$(cut -d ' ' -f 1 < "$LOCAL/$ZIP.sha256")" = "$ZIP_SHA" ] || die "$ZIP.sha256 does not match $ZIP"
VERIFY_ARGS=()
if [ "$IS_DRAFT" = true ]; then
    VERIFY_ARGS+=(--key-env OTO_UPDATE_SIGNING_KEY)
else
    if [ -n "${OTO_UPDATE_VERIFY_PUBKEYS:-}" ]; then
        IFS=',' read -r -a PUBKEYS <<< "$OTO_UPDATE_VERIFY_PUBKEYS"
        for pk in "${PUBKEYS[@]}"; do
            pk="${pk#"${pk%%[![:space:]]*}"}"
            pk="${pk%"${pk##*[![:space:]]}"}"
            [ -n "$pk" ] && VERIFY_ARGS+=(--pubkey "$pk")
        done
    fi
    if [ -n "${OTO_UPDATE_SIGNING_KEY:-}" ]; then
        VERIFY_ARGS+=(--key-env OTO_UPDATE_SIGNING_KEY)
    fi
    [ "${#VERIFY_ARGS[@]}" -gt 0 ] ||
        die "set OTO_UPDATE_VERIFY_PUBKEYS (or OTO_UPDATE_SIGNING_KEY) to verify published update.json"
fi
"$OTO_SIGN" verify --envelope "$LOCAL/update.json" --repo "$REPO" "${VERIFY_ARGS[@]}" > "$WORK/manifest.json" ||
    die "update.json does not verify against the configured release keys"
jq -e --arg v "$VERSION" --arg sha "$ZIP_SHA" --argjson size "$(size "$LOCAL/$ZIP")" \
    --arg url "https://github.com/$REPO/releases/download/$TAG/$ZIP" \
    '.version == $v and .platforms["macos-universal"].sha256 == $sha
     and .platforms["macos-universal"].size == $size and .platforms["macos-universal"].url == $url' \
    "$WORK/manifest.json" > /dev/null || die "update.json's manifest does not describe $ZIP for $TAG"

# --- Upload (drafts only; --clobber so a re-run replaces a half-finished upload). ---
if [ "$IS_DRAFT" = true ]; then
    (cd "$LOCAL" && gh release upload "$TAG" "${ASSETS[@]}" --clobber --repo "$REPO")
fi

# --- Every asset as GitHub stores it: uploaded, exact size, and the same SHA-256. ---
API_URL="$(gh release view "$TAG" --repo "$REPO" --json apiUrl --jq .apiUrl)"
gh api --paginate "${API_URL#https://api.github.com/}/assets" --jq '.[]' > "$WORK/assets.jsonl"
for f in "${ASSETS[@]}"; do
    rec="$(jq -c --arg n "$f" 'select(.name == $n)' "$WORK/assets.jsonl")"
    [ "$(wc -l <<< "$rec" | tr -d ' ')" = 1 ] && [ -n "$rec" ] || die "release $TAG has no single asset named $f"
    [ "$(jq -r .state <<< "$rec")" = uploaded ] || die "$f is not fully uploaded (state $(jq -r .state <<< "$rec"))"
    [ "$(jq -r .size <<< "$rec")" = "$(size "$LOCAL/$f")" ] || die "$f is $(jq -r .size <<< "$rec") bytes on GitHub, expected $(size "$LOCAL/$f")"
    want="$(sha256 "$LOCAL/$f")"
    digest="$(jq -r '.digest // empty' <<< "$rec")"
    if [ -n "$digest" ]; then
        [ "$digest" = "sha256:$want" ] || die "$f digest on GitHub is $digest, expected sha256:$want"
    else
        # No digest from the API: download it again and hash what GitHub serves.
        mkdir -p "$WORK/redownload"
        gh release download "$TAG" --repo "$REPO" --dir "$WORK/redownload" --pattern "$f" --clobber
        [ "$(sha256 "$WORK/redownload/$f")" = "$want" ] || die "$f as served by GitHub does not match"
    fi
    echo "ok: $f ($(size "$LOCAL/$f") bytes, sha256 $want)"
done

# --- Publish, marking it latest only if it is newer than the current latest. ---
if [ "$IS_DRAFT" = true ]; then
    # Only a confirmed 404 means no published release exists yet; any API/network failure must
    # abort so an older backfill can never be promoted to latest.
    LATEST_ERR="$WORK/latest.err"
    CURRENT=""
    if ! CURRENT="$(gh api "repos/$REPO/releases/latest" --jq .tag_name 2>"$LATEST_ERR")"; then
        if grep -q 'HTTP 404' "$LATEST_ERR"; then
            CURRENT=""
        else
            die "couldn't query the current latest release: $(cat "$LATEST_ERR")"
        fi
    fi
    LATEST_FLAG=--latest
    if [ -n "$CURRENT" ] && ! "$OTO_SIGN" check-version-gt --candidate "$TAG" --current "$CURRENT"; then
        LATEST_FLAG=--latest=false
    fi
    echo "Publishing $TAG ($LATEST_FLAG; current latest: ${CURRENT:-none})"
    gh release edit "$TAG" --repo "$REPO" --draft=false "$LATEST_FLAG"
    [ "$(gh release view "$TAG" --repo "$REPO" --json isDraft --jq .isDraft)" = false ] || die "$TAG is still a draft"
fi

# --- Homebrew tap: monotonic and idempotent; only moves forward, and fixes a wrong checksum. ---
if [ -z "${HOMEBREW_TAP_TOKEN:-}" ]; then
    echo "::notice::HOMEBREW_TAP_TOKEN is not set; skipping the Homebrew tap"
    exit 0
fi
TAP="$WORK/tap"
git clone --quiet --depth 1 "https://x-access-token:$HOMEBREW_TAP_TOKEN@github.com/$HOMEBREW_TAP.git" "$TAP" ||
    die "couldn't clone $HOMEBREW_TAP"
CASK="$TAP/Casks/oto.rb"
TAP_VERSION=""
TAP_SHA=""
if [ -f "$CASK" ]; then
    TAP_VERSION="$(sed -nE 's/^[[:space:]]*version "([^"]*)".*/\1/p' "$CASK" | head -n 1)"
    TAP_SHA="$(sed -nE 's/^[[:space:]]*sha256 "([^"]*)".*/\1/p' "$CASK" | head -n 1)"
fi
if [ -n "$TAP_VERSION" ] && [ "$TAP_VERSION" != "$VERSION" ] &&
    ! "$OTO_SIGN" check-version-gt --candidate "$VERSION" --current "$TAP_VERSION" > /dev/null; then
    echo "The tap already has $TAP_VERSION, newer than $VERSION; leaving it."
    exit 0
fi
if [ "$TAP_VERSION" = "$VERSION" ] && [ "$TAP_SHA" = "$DMG_SHA" ]; then
    echo "The tap already has $VERSION with the published DMG checksum."
    exit 0
fi
mkdir -p "$TAP/Casks"
sed -e "s/@VERSION@/$VERSION/g" -e "s/@SHA256@/$DMG_SHA/g" "$ROOT/packaging/homebrew/oto.rb" > "$CASK"
grep -q '@' <<< "$(grep -E '^[[:space:]]*(version|sha256) ' "$CASK")" && die "unrendered placeholder in the cask"
git -C "$TAP" add Casks/oto.rb
git -C "$TAP" -c user.name="oto-release" -c user.email="oto-release@users.noreply.github.com" \
    commit --quiet -m "oto $VERSION"
git -C "$TAP" push --quiet origin HEAD || die "couldn't push to $HOMEBREW_TAP; re-run: gh workflow run macos-release.yml --ref main -f tag=$TAG"
echo "Updated $HOMEBREW_TAP to $VERSION (${TAP_VERSION:-new cask} → $VERSION)"
