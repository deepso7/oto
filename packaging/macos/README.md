# Packaging Oto for macOS

`scripts/macos/bundle.sh` builds `target/bundle/Oto.app`:

```
Oto.app/Contents/
  Info.plist                      from Info.plist.in (version from the workspace Cargo.toml)
  Resources/AppIcon.icns          from packaging/macos/AppIcon.icns
  MacOS/oto-app                   release build of crates/oto-app
  Resources/OtoMicrophone.driver  the prebuilt, signed HAL driver the app installs
  Helpers/oto-hal-helper          the signed tool that installs/removes it (dev.oto.hal-helper)
  Resources/LICENSE-MIT, LICENSE-APACHE, THIRD_PARTY_NOTICES.md
```

The app installs the driver from its own `Resources` and nowhere else, so the driver has to be
in the bundle. The driver and `oto-hal-helper` are built and signed by `driver/macos/Makefile`;
`bundle.sh` checks both signatures before and after signing the app around them. The driver's
revision is `driver/macos/DRIVER_VERSION` (an integer), not the app version: app-only releases
ship the same driver and never ask for a password or restart `coreaudiod`. Run every command here as yourself, never with `sudo`.

## Local build (ad-hoc signed)

```sh
scripts/macos/bundle.sh
open target/bundle/Oto.app
```

The app and driver are ad-hoc signed. That is enough to run on the Mac that built it, and
gives macOS an Info.plist to show the microphone prompt for. It is not distributable:
Gatekeeper rejects it on other Macs.

The app binary is built for the host architecture only, unless `UNIVERSAL=1` (the default for
Developer ID builds). The driver is always universal.

## Developer ID build

```sh
security find-identity -v -p codesigning        # find your identity
SIGN_ID="Developer ID Application: Your Name (TEAMID)" scripts/macos/bundle.sh
```

With a Developer ID identity the script:

- builds the driver with `make release` (hardened runtime, secure timestamp; it fails rather
  than fall back to ad-hoc if the identity is missing);
- builds `oto-app` with `OTO_TEAM_ID=TEAMID`, so the app only installs a driver signed by that
  team;
- signs the app with the hardened runtime, a secure timestamp and `Oto.entitlements`
  (`com.apple.security.device.audio-input`);
- verifies the result with `codesign --verify --deep --strict`.

No signing identity is stored in the repository; `SIGN_ID` always comes from your environment.

## Releasing

### One-time setup

1. **Developer ID Application certificate.** Requires the paid Apple Developer Program, and only
   the team's Account Holder can create one. Xcode → Settings → Accounts → your team → Manage
   Certificates → **+** → *Developer ID Application*. It then appears in
   `security find-identity -v -p codesigning`. Back it up (Keychain Access → export as `.p12`):
   Apple allows only a few per account.
2. **Notarization credentials** in the keychain, under the profile name `oto`. Make an
   app-specific password at <https://account.apple.com> (Sign-In and Security → App-Specific
   Passwords), then:

   ```sh
   xcrun notarytool store-credentials oto --apple-id <apple-id> --team-id <TEAMID>
   ```
3. **Both Rust targets** for the universal binary:
   `rustup target add aarch64-apple-darwin x86_64-apple-darwin`.

### Each release

1. Bump `version` under `[workspace.package]` in `Cargo.toml` and commit (release-plz does
   this in CI). If anything under `driver/macos/src/`, `driver/macos/Info.plist` or
   `driver/macos/Makefile` changed since the last release, bump `driver/macos/DRIVER_VERSION`
   too; `macos-pr-check.yml` fails the PR otherwise.
2. Build:

   ```sh
   SIGN_ID="Developer ID Application: Your Name (TEAMID)" scripts/macos/release.sh
   ```

   This runs `bundle.sh` with `UNIVERSAL=1`, notarizes and staples `Oto.app`, wraps it in a
   DMG (the app plus an Applications link), signs, notarizes and staples the DMG, checks both
   with `spctl`, and writes `target/dist/Oto-<version>.dmg` and its `.sha256`. It also zips
   the stapled app as `Oto-<version>-macos.zip` (+ `.sha256`) for the in-app updater, unpacks
   that zip again with the updater's own extractor (`oto-sign extract-verify`) and checks the
   result (Developer ID signatures of the app, driver and `oto-hal-helper`, `spctl`, the
   stapled ticket, both architectures, `LSMinimumSystemVersion`, and that the bundled
   `OtoDriverVersion` equals `DRIVER_VERSION`), then writes the unsigned manifest
   `update-payload.json`. It refuses a dirty working tree unless `ALLOW_DIRTY=1`. If Apple
   rejects a submission, the script prints the notarization log.
3. Before publishing, test the DMG on a Mac (or a fresh user account) that has never run Oto:
   download it through a browser so it's quarantined, drag Oto to Applications, open it, and go
   through onboarding (driver install, microphone prompt).
4. Publish: the script prints the `git tag`, `oto-sign sign` and `gh release create` commands.
   It never runs them itself. Normally CI publishes (below).

The app is notarized on its own as well as inside the DMG, so it carries its own ticket after
it's dragged out and opens offline the first time. The embedded driver is covered by the app's
notarization, but a ticket can only be stapled to the app, not to the driver copied out of it.

## Releasing from CI

`.github/workflows/macos-release.yml` turns a tag into a published release. release-plz
(`release-plz.yml`) calls it after merging a release PR, once it has tagged `vX.Y.Z` and created
a **draft** release (`git_release_draft = true` in `release-plz.toml`). Because the call comes
from a push to `main`, the workflow runs on `refs/heads/main`, and so do manual runs. Four jobs:

1. **validate-tag** (no secrets, read-only): the tag is `vX.Y.Z`, it is an ancestor of
   `origin/main`, and `Cargo.toml` at the tagged commit has that version. Outputs the commit.
2. **check-release** (no secrets): reads whether the release is still a draft.
3. **dmg** (drafts only, `release` environment): checks out exactly that commit, sets up a
   temporary keychain (`scripts/macos/ci-keychain.sh`), runs `scripts/macos/release.sh`, and
   keeps the DMG, the update zip, their `.sha256` files and `update-payload.json` as a
   workflow artifact (7 days). The keychain is always deleted.
4. **publish** (`release` environment, one at a time for all tags: concurrency group
   `oto-release-publish`, never cancelled) runs `scripts/macos/publish-release.sh`:
   - re-checks, under that lock, that the remote tag still points at the validated commit and
     re-reads the draft state;
   - **draft:** requires a successful `dmg` job, signs `update-payload.json` into `update.json`
     (`oto-sign sign`, key from `OTO_UPDATE_SIGNING_KEY`), checks the checksums and the
     manifest against the files, uploads `Oto-<v>.dmg`, `.dmg.sha256`, `Oto-<v>-macos.zip`,
     `.zip.sha256` and `update.json`, checks each through the release assets API
     (`state == uploaded`, exact size, `digest`, or a re-download if there is no digest),
     publishes it, marked **latest** only if its version is above the current latest release,
     then updates the Homebrew tap;
   - **already published:** uploads nothing; downloads and re-verifies the published assets
     and envelope the same way, then reconciles the Homebrew tap. So a re-run is always safe.

Pull requests touching `scripts/macos/`, `packaging/macos/`, `driver/macos/`,
`crates/oto-sign/` or `.github/workflows/macos-*.yml` run `macos-pr-check.yml` instead: no
secrets, read-only. It checks the `DRIVER_VERSION` bump rule, runs `cargo test -p oto-sign`,
builds an ad-hoc `Oto.app` with `bundle.sh`, zips it like `release.sh` and unpacks it with
`oto-sign extract-verify`.

### One-time repository setup

1. **`release` environment.** Settings → Environments → New environment `release`. Under
   *Deployment branches and tags*, choose *Selected branches and tags* and add `main` only (and
   protect `main` with a branch ruleset). Put every secret below in this environment, not in
   repository secrets, so only jobs that run on `main` with `environment: release` see them.
2. **Immutable releases.** Settings → General → Releases → enable *Immutable releases*. Once a
   release is published, its assets and tag can't change; drafts stay editable, which is what
   the pipeline uploads to.
3. **Tag ruleset.** Settings → Rules → Rulesets → New tag ruleset, target `v*`, enforcement
   *Active*, rules *Restrict updates* and *Restrict deletions* (bypass: nobody, or repository
   admins only). A tag can then never move while its draft is open, and `publish` re-checks it
   anyway.
4. **Actions → General:** "Allow GitHub Actions to create and approve pull requests" (release-plz
   opens the release PR).
5. **Update signing key** and, optionally, **Homebrew tap token** (below).

### Secrets

Set these as secrets of the `release` environment (Settings → Environments → release →
Environment secrets), not as repository secrets.

| Secret | What it is |
|---|---|
| `OTO_UPDATE_SIGNING_KEY` | Update signing key(s), `<key_id>:<base64 seed>[,…]` (see *Update signing key*) |
| `HOMEBREW_TAP_TOKEN` | Optional: token that can push to the Homebrew tap (see *Homebrew tap*) |
| `MACOS_CERTIFICATE_P12_BASE64` | The Developer ID Application certificate and its private key, as a base64 `.p12` |
| `MACOS_CERTIFICATE_PASSWORD` | The password the `.p12` was exported with |
| `APPLE_API_KEY_P8_BASE64` | App Store Connect API key (`AuthKey_<id>.p8`), base64 |
| `APPLE_API_KEY_ID` | That key's ID |
| `APPLE_API_ISSUER_ID` | The team's issuer ID (a UUID) |
| `APPLE_ID` | Fallback: Apple ID email |
| `APPLE_TEAM_ID` | Fallback: `T9DPKXJBUS` |
| `APPLE_APP_SPECIFIC_PASSWORD` | Fallback: an app-specific password for that Apple ID |

Notarization needs one of the two sets: the API key (preferred: it isn't tied to a personal
Apple ID and survives password changes) or the three `APPLE_ID*` secrets. If the API key set is
complete it is used; a partly filled set is an error. The signing identity defaults to
`Developer ID Application: Deepanshu Singh (T9DPKXJBUS)`; to use another, set the repository
**variable** `MACOS_SIGN_ID`.

How to produce them:

- **Certificate.** Keychain Access → login → My Certificates → right-click *Developer ID
  Application: …* (expand it to check the private key is under it) → Export → `.p12`, with a
  strong password. Then:

  ```sh
  base64 -i DeveloperID-T9DPKXJBUS.p12 | pbcopy    # paste as MACOS_CERTIFICATE_P12_BASE64
  ```

  The password goes in `MACOS_CERTIFICATE_PASSWORD`. Delete the `.p12` from disk afterwards
  (or keep it only in your password manager).
- **App Store Connect API key.** <https://appstoreconnect.apple.com> → Users and Access →
  Integrations → Team Keys → **+**, name it (e.g. *Oto CI*), access **Developer**. Download the
  `.p8` (Apple lets you download it only once), note the **Key ID** next to it and the
  **Issuer ID** above the list. Then:

  ```sh
  base64 -i AuthKey_XXXXXXXXXX.p8 | pbcopy         # paste as APPLE_API_KEY_P8_BASE64
  ```

  Revoke the key on the same page if it leaks.
- **Apple ID fallback.** The same app-specific password as the local profile works
  (<https://account.apple.com> → Sign-In and Security → App-Specific Passwords).

With `gh`, from the repository: `gh secret set MACOS_CERTIFICATE_P12_BASE64 --env release < cert.b64`,
or `gh secret set APPLE_API_KEY_ID --env release` to be prompted.

### Update signing key

`update.json` is an Ed25519-signed envelope (format in docs/plans/0003-app-updates.md,
Decision 2). The public keys the app trusts are compiled into
`crates/oto-app/src/updater/keys.rs`; the private key exists only offline and in the `release`
environment.

1. On a trusted Mac, offline if you can: `cargo run --release -p oto-sign -- keygen`
   (`--key-id 2026-11`; the default is the current year and month). It prints the key id, the
   secret value `<key_id>:<base64 seed>`, the public key, and a `TrustedKey { … }` line.
2. `gh secret set OTO_UPDATE_SIGNING_KEY --env release` and paste the secret value.
3. Back the secret value up offline (password manager plus a second, offline copy). Losing it
   means shipping a release that existing users can't verify (they get the "check GitHub
   Releases" fallback and must update by hand once).
4. Commit the `TrustedKey` line to `keys.rs`. Never commit the secret value.

Check a signed envelope: `oto-sign verify --envelope update.json --pubkey <key_id>:<base64 pub>`.

#### Rotating the key (dual signing)

`OTO_UPDATE_SIGNING_KEY` takes several comma-separated keys, and every one signs each release:

1. `keygen` a new key. Add its `TrustedKey` to `keys.rs` **next to** the old one and release
   that build (the bridge release).
2. Set `OTO_UPDATE_SIGNING_KEY` to `<old_id>:<old seed>,<new_id>:<new seed>`. Every release now
   carries both signatures, so builds that only know the old key still update.
3. Keep dual signing for at least **6 months and 3 releases**, so people who skip releases
   still reach the bridge.
4. Then remove the old key from `keys.rs`, set `OTO_UPDATE_SIGNING_KEY` to the new key alone,
   keep the retired public key in the `OTO_UPDATE_VERIFY_PUBKEYS` variable
   (`<key_id>:<base64 pub>[,…]`) so re-running `publish` on historical published releases still
   verifies them without private seeds, and destroy the old seed.

If a key leaks: rotate immediately (steps 1–2), and shorten step 3 to what you can accept.
Builds that trust only the leaked key keep trusting it, but every update still has to pass
the Developer ID signature, Team ID, notarization and Gatekeeper checks, which the key alone
can't forge.

### Homebrew tap

`packaging/homebrew/oto.rb` is the cask: `auto_updates true` (Oto updates itself, so
`brew upgrade` skips it unless `--greedy`), `uninstall quit: "dev.oto.app"`, and no
`uninstall delete:` for the HAL driver, which is machine-wide and only removed through Oto's
audio-checked uninstall (the caveats say so). `publish` renders it with the version and the
published DMG's SHA-256 and pushes it to `deepso7/homebrew-oto` as `Casks/oto.rb`, but only
forward: an older tag never replaces a newer cask, and the same version is rewritten only if
its checksum is wrong.

To enable it, create the `deepso7/homebrew-oto` repository, make a fine-grained personal
access token (or GitHub App token) with *Contents: read and write* on that repository only,
and `gh secret set HOMEBREW_TAP_TOKEN --env release`. Without the secret the step is skipped.
Another tap: set the repository variable `HOMEBREW_TAP` to `owner/name`.

### Recovery

Every case is the same command, run on `main`; the workflow works out what is left to do:

```sh
gh workflow run macos-release.yml --ref main -f tag=vX.Y.Z
gh run watch
```

- **Build or notarization failed**, or **publish failed** (upload, API check, signing): the
  release is still a draft, nothing is public. Fix the cause if needed (a fix means a new
  version; the tag is fixed) and re-run.
- **Publish run displaced.** GitHub keeps only one *pending* run per concurrency group, so if
  several releases queue behind a running publish, older pending ones are cancelled and their
  drafts stay unpublished. Re-run each such tag, oldest first.
- **Tap push failed** after publishing: re-run. The release is already published, so nothing
  is uploaded; the assets are re-verified and the tap updated.
- **Tag pushed but no release** (release-plz failed between the two):
  `gh release create vX.Y.Z --draft --verify-tag --title vX.Y.Z --generate-notes`, then re-run.
- **A draft for a version you won't ship:** delete the draft (`gh release delete vX.Y.Z`);
  the tag stays (the ruleset protects it), and the next version supersedes it.

Old tags made before `crates/oto-sign` existed can't be built by this workflow.

## Recovering an interrupted update on a Mac

The in-app updater swaps `Oto.app` atomically and keeps the previous version until the new one
has started; if anything goes wrong it rolls back by itself, and the next launch of Oto
finishes or undoes any update that was cut short. If Oto won't start or seems stuck after an
update:

1. **Force Quit, then reopen.** Apple menu → Force Quit… → Oto (or
   `killall oto-app`), then open Oto again. Startup runs the recovery.
2. **Run the recovery directly** with the copy of the previous Oto kept for that purpose:

   ```sh
   ls ~/Library/Application\ Support/oto/update-helper-*
   ~/Library/Application\ Support/oto/update-helper-<nonce> --recover-update
   ```

   or, with the command-line tool, `oto recover-update`. Either one finishes or rolls back the
   interrupted update and relaunches Oto.
3. The log is `~/Library/Logs/Oto/updater.log`.

## App icon

The artwork is SVG in `packaging/brand`: `oto-icon.svg` (the app icon), `oto-icon-small.svg`
(a heavier version for 16 and 32 px) and `oto-mark.svg` (the mark alone, in `currentColor`).
After changing them, rebuild the committed `AppIcon.icns` with `scripts/macos/make-icon.sh`.
