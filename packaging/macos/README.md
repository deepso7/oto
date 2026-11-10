# Packaging Oto for macOS

`scripts/macos/bundle.sh` builds `target/bundle/Oto.app`:

```
Oto.app/Contents/
  Info.plist                      from Info.plist.in (version from the workspace Cargo.toml)
  MacOS/oto-app                   release build of crates/oto-app
  Resources/OtoMicrophone.driver  the prebuilt, signed HAL driver the app installs
  Resources/LICENSE-MIT, LICENSE-APACHE
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

The binary is built for the host architecture only (the driver is universal).

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

## Not automated: what a release still needs

**None of the following is done by any script here.** A Developer ID build that skips them is
still blocked by Gatekeeper when downloaded.

1. Notarize and staple the app:

   ```sh
   ditto -c -k --keepParent target/bundle/Oto.app target/bundle/Oto.zip
   xcrun notarytool submit target/bundle/Oto.zip --keychain-profile <profile> --wait
   xcrun stapler staple target/bundle/Oto.app
   spctl --assess --type execute -vv target/bundle/Oto.app
   ```

   The embedded driver is notarized as part of the app, but a ticket can only be stapled to
   the app, not to the driver copied out of it.

2. Ship it in a container (a notarized `.dmg`, or eventually a notarized `.pkg` signed with a
   "Developer ID Installer" identity that installs the app and the driver in one step). Neither
   exists yet.

3. A universal (arm64 + x86_64) app binary, an app icon, and keeping the driver's
   `CFBundleShortVersionString` (driver/macos/Info.plist) in step with the workspace version
   are also still to do.

The Developer ID path above has not been exercised yet: it was written without a Developer ID
certificate available.
