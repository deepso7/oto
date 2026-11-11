# Packaging Oto for macOS

`scripts/macos/bundle.sh` builds `target/bundle/Oto.app`:

```
Oto.app/Contents/
  Info.plist                      from Info.plist.in (version from the workspace Cargo.toml)
  MacOS/oto-app                   release build of crates/oto-app
  Resources/OtoMicrophone.driver  the prebuilt, signed HAL driver the app installs
  Resources/LICENSE-MIT, LICENSE-APACHE, THIRD_PARTY_NOTICES.md
```

The app installs the driver from its own `Resources` and nowhere else, so the driver has to be
in the bundle. Run every command here as yourself, never with `sudo`.

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

1. Bump `version` under `[workspace.package]` in `Cargo.toml` (the app and the driver both take
   it from there) and commit.
2. Build:

   ```sh
   SIGN_ID="Developer ID Application: Your Name (TEAMID)" scripts/macos/release.sh
   ```

   This runs `bundle.sh` with `UNIVERSAL=1`, notarizes and staples `Oto.app`, wraps it in a
   DMG (the app plus an Applications link), signs, notarizes and staples the DMG, checks both
   with `spctl`, and writes `target/dist/Oto-<version>.dmg` and its `.sha256`. It refuses a
   dirty working tree unless `ALLOW_DIRTY=1`. If Apple rejects a submission, the script prints
   the notarization log.
3. Before publishing, test the DMG on a Mac (or a fresh user account) that has never run Oto:
   download it through a browser so it's quarantined, drag Oto to Applications, open it, and go
   through onboarding (driver install, microphone prompt).
4. Publish: the script prints the `git tag` and `gh release create` commands. It never runs
   them itself.

The app is notarized on its own as well as inside the DMG, so it carries its own ticket after
it's dragged out and opens offline the first time. The embedded driver is covered by the app's
notarization, but a ticket can only be stapled to the app, not to the driver copied out of it.

## Releasing from CI

`.github/workflows/macos-release.yml` runs the same `release.sh` on a GitHub-hosted macOS
runner and attaches `Oto-<version>.dmg` and its `.sha256` to an existing GitHub release. The
release-plz workflow calls it after it tags `vX.Y.Z` and creates the release. The workflow:

1. checks the tag is `vX.Y.Z` and its GitHub release exists;
2. checks out exactly that tag and fails unless it matches `version` under
   `[workspace.package]` in `Cargo.toml`;
3. installs the toolchain from `rust-toolchain.toml` plus both darwin targets;
4. runs `scripts/macos/ci-keychain.sh setup`: a temporary keychain with a random password,
   the Developer ID identity imported into it, and the notarytool profile stored in it;
5. runs `scripts/macos/release.sh` (with `NOTARY_KEYCHAIN` pointing at that keychain);
6. uploads the DMG as a workflow artifact (kept 7 days) and attaches the DMG and `.sha256` to
   the release (`gh release upload --clobber`, so re-runs replace them);
7. always deletes the temporary keychain and the decoded key files.

The build scripts come from the tag, so only tags made after this pipeline existed can be built.

Pull requests that change `scripts/macos/`, `packaging/macos/`, `driver/macos/` or this
workflow get a build-only run: steps 3–5 and 7 on the PR's code, with the signed, notarized
DMG kept as a workflow artifact and nothing attached to a release. PRs from forks skip it,
since they get no secrets.

### Secrets

Set these under the repository's Settings → Secrets and variables → Actions → Secrets.

| Secret | What it is |
|---|---|
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

With `gh`, from the repository: `gh secret set MACOS_CERTIFICATE_P12_BASE64 < cert.b64`, or
`gh secret set APPLE_API_KEY_ID` to be prompted.

### Running it by hand

To rebuild the DMG for a release, or backfill one, the release for the tag must already exist:

```sh
gh workflow run macos-release.yml -f tag=v0.1.0
gh run watch
```

or Actions → *macOS release* → Run workflow, with the tag. It replaces the release's DMG and
`.sha256` if they are already attached.

## Still to do

- An app icon: put artwork at `packaging/macos/AppIcon.icns` and `bundle.sh` picks it up.
- A Homebrew cask, and an updater.
