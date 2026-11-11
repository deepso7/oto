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

## Still to do

- An app icon: put artwork at `packaging/macos/AppIcon.icns` and `bundle.sh` picks it up.
- Release from CI on a tag push (GitHub Actions on a macOS runner, with the certificate as a
  `.p12` and an App Store Connect API key as repository secrets).
- A Homebrew cask, and an updater.
