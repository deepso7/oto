# 0003: End-to-end app and driver update flow

**Status:** Implemented (code & automated tests complete; awaiting offline Ed25519 key setup, GitHub environment/rulesets, Homebrew tap repo, and live Developer ID E2E run) · 2026-11-27 (council-approved: GPT-6.2-Sol med 9/10, Claude Opus 5.5 med 9/10)

[← Docs index](../README.md)

## Goal

Give Oto a safe, private, low-friction update flow end to end:

1. **Automated, atomic, monotonic releases** from CI that validate tag ancestry without secrets, build/sign/notarize/verify every artifact in a draft release, and publish `/releases/latest` atomically under a global publish lock.
2. **In-app update check, explicit download, cryptographic verification, and recoverable atomic swap/relaunch** in `oto-app` on macOS, preserving microphone permission, preferences, and login items, deferring restarts while calls are active, and rolling back automatically if the new build fails to start.
3. **Decoupled HAL driver versioning and engine-enforced compatibility** so app-only updates never ask for an admin password or restart `coreaudiod`, while genuine driver changes are detected reliably and upgraded safely.
4. **Forward-compatible distribution** with a multi-platform signed manifest envelope, Homebrew Cask support (`auto_updates true`), and package-manager-aware notify-only behavior on Linux.

## Background

Today ([Status](../status.md), [packaging/macos/README.md](../../packaging/macos/README.md)):

- **Release publishing is automated** (`release-plz.yml` → `macos-release.yml` → `scripts/macos/release.sh`), producing a universal (`arm64` + `x86_64`), Developer ID–signed, notarized, and stapled `Oto.app` inside `Oto-<version>.dmg` attached to GitHub Releases.
- **There is no in-app updater.** Users must manually notice a new GitHub Release, download the DMG, and drag `Oto.app` over `/Applications/Oto.app`.
- **Releases are not atomic.** `release-plz` publishes the GitHub Release immediately on merge, then `macos-release.yml` builds and attaches the DMG ~15 minutes later. During that window `/releases/latest` has no downloadable artifacts.
- **Driver versioning is coupled to the workspace version.** `driver/macos/Makefile` stamps `Cargo.toml`'s workspace version into `OtoMicrophone.driver/Contents/Info.plist`, and `OtoDriver.c` exposes no version property to Core Audio. `oto-app` only probes whether the driver is loaded/present (`Engine::driver_installed` + `driver::bundle_present`), so users cannot tell when a driver upgrade is needed or avoid unnecessary `coreaudiod` restarts.
- **Oto has specific runtime constraints** that any updater must respect:
  - **Audio continuity:** Restarting `oto-app` drops `Oto Microphone`; running `killall coreaudiod` for a HAL driver upgrade interrupts **all** audio on the Mac (not just Oto Microphone).
  - **Single-instance lock & teardown:** `oto-engine` holds an exclusive flock on `engine.lock` (`crates/oto-engine/src/state.rs`). A second `oto-app` launched before the first exits gets `ALREADY_RUNNING` and silently exits (`crates/oto-app/src/main.rs:452-457`). On shutdown, `Engine::drop` joins the control thread and restores the previous default input synchronously via `RestoreTarget` (`crates/oto-engine/src/lib.rs:307-314`), but discards the `Result` while `main.rs` bounds the wait to 2 s.
  - **LaunchAgent process-group teardown:** `crates/oto-app/src/login.rs` registers a LaunchAgent plist (`RunAtLoad = true`, no `AbandonProcessGroup`). When a login-started Oto exits, `launchd` kills its entire process group—including any child updater helper that did not escape into its own session.
  - **No telemetry:** [Product](../product.md) commits to *"update checks only; no account, no telemetry by default."*
  - **Security posture:** `crates/oto-app/src/driver.rs` pins `OTO_TEAM_ID` and verifies strict `codesign` requirements.
  - **Install contexts:** Users may run `Oto.app` from a user-owned, writable folder (`/Applications` when owned by the current user, or `~/Applications`), a bundle owned by another user or `root`, an `AppTranslocation` path or mounted DMG (read-only), Homebrew Cask, or a dev/ad-hoc build.

## Decisions

### 1. Pure-Rust writable-only updater in `oto-app` for v1 (not Sparkle 2)

We build a self-contained updater module (`crates/oto-app/src/updater/`) rather than embedding `Sparkle.framework`, and restrict in-place self-replacement in v1 to **user-owned, user-writable** app bundles (`st_uid == geteuid()` in a writable parent directory such as `/Applications` for the installing admin user or `~/Applications`), falling back to **notify-only ("Download…")** for bundles owned by `root`/another user, read-only volumes, or translocated launches.

Why:
- **Tradeoff vs. Sparkle 2:** Because Oto is unsandboxed, Sparkle's sandboxing XPC services could be omitted, and Sparkle's `SPUUserDriver` and `updater:shouldPostponeRelaunchForUpdate:untilInvokingBlock:` address UI and call-deferral concerns. However, Sparkle still adds an external Objective-C framework to sign/notarize/maintain, an Objective-C `SPUUserDriver` bridge on top of Rune, and an external termination lifecycle around Oto's audio restoration handoff.
- **Eliminating privileged & cross-user app-swap complexity for v1:** Requiring `st_uid == geteuid()` covers the standard single-owner macOS setup (dragging `Oto.app` into `/Applications` or `~/Applications`) while preventing cross-user lock/recovery spoofing on multi-user Macs, avoiding `osascript` TOCTOU/ownership pitfalls, and making the zero-admin-prompt guarantee for app-only updates unconditional.
- **Ownership of Sparkle-equivalent invariants:** Choosing a custom updater means Oto explicitly owns and tests: LaunchAgent session escape (`setsid`), atomic APFS bundle exchange (`renamex_np`), persisted transaction recovery, single-writer startup acknowledgment with automatic rollback, `AppTranslocation` and read-only mount detection, macOS 13+ App Management permission, and Gatekeeper assessment.

### 2. Single signed manifest envelope & defense-in-depth verification

Updates use **one** bounded file (`update.json`, ≤ 64 KiB) that wraps the raw manifest bytes and one or more Ed25519 signatures:

```json
{
  "format": 1,
  "signatures": [
    { "key_id": "2026-11", "sig": "<base64-64-byte-ed25519-sig-over-raw-payload-bytes>" }
  ],
  "payload": "<base64-encoded-utf8-manifest-json>"
}
```

Decoded `payload` schema:

```json
{
  "schema": 1,
  "version": "0.1.0",
  "published_at": "2026-11-27T00:00:00Z",
  "notes_url": "https://github.com/deepso7/oto/releases/tag/v0.1.0",
  "driver_version": 1,
  "min_driver_version": 1,
  "platforms": {
    "macos-universal": {
      "url": "https://github.com/deepso7/oto/releases/download/v0.1.0/Oto-0.1.0-macos.zip",
      "sha256": "<64-hex>",
      "size": 45123456,
      "min_os": "12.0"
    }
  }
}
```

Why a single envelope with a `signatures` array:
- Eliminates split-file mismatch races between a separate manifest and `.sig` file (especially during `/releases/latest` publication).
- Signs the exact decoded `payload` bytes, avoiding JSON canonicalization ambiguity.
- Supports **dual-signature key rotation**: during rotation, CI signs the same payload with both the old and new keys for at least 6 months / 3 releases so users who skip the bridge release can still verify with their embedded old key. If no signature matches a trusted key (whether rotated or tampered), the updater refuses to parse the payload and shows neutral copy (*"Oto couldn't verify this update's signature — check GitHub Releases for the latest version."*) with a fixed link to `https://github.com/deepso7/oto/releases`.

An update is accepted only if **all** of the following checks pass:

1. **HTTPS & redirect validation:** Initial manifest URL is pinned to `https://github.com/deepso7/oto/releases/latest/download/update.json` (an `OTO_UPDATE_URL` override is compiled in **only** under `cfg(debug_assertions)` or when `OTO_TEAM_ID` is absent—never honored in Developer ID release builds). Every redirect hop is parsed and validated before following: `https://` only, standard port 443, no userinfo/credentials, max 5 hops, bounded connect/read/total timeouts. The manifest's `platforms.*.url` must begin with `https://github.com/deepso7/oto/releases/download/v<manifest.version>/` (redirects from GitHub to its CDN—e.g. `release-assets.githubusercontent.com`—are allowed over HTTPS without pinning the CDN hostname). `notes_url` must equal `https://github.com/deepso7/oto/releases/tag/v<manifest.version>` before opening in a browser.
2. **Ed25519 signature verification:** At least one entry in `signatures` matches a compiled-in `(key_id, pubkey)` pair in `crates/oto-app/src/updater/keys.rs` and verifies the exact decoded `payload` bytes.
3. **OS compatibility & strict version monotonicity:** Current macOS version (`NSProcessInfo.operatingSystemVersion`) ≥ `min_os`, and `semver(manifest.version) > semver(current_version)`. (Static releases do not carry a hard expiration timestamp, because a long quiet period without releases must not brick updating; monotonic version checks stop downgrades, and key compromise is handled via dual-key rotation + Apple Developer ID as an independent second trust anchor.)
4. **Disk space, exact byte size & SHA-256:** Check free disk space ≥ 3× `size` on the cache volume before downloading. Stream into a `0700` private cache file (`~/Library/Caches/dev.oto.app/updates/<version>.zip.part`) with an incremental byte cap of `min(size, 150 MiB)` enforced inside the data delegate, verifying exact `size` and `sha256` before extraction.
5. **Single bounded Rust ZIP extractor (no two-tool parser split):** Extract directly with one bounded Rust ZIP extractor (`zip` crate, minimal features) into a fresh empty `0700` directory:
   - First scan the central directory to compute `uncompressed_total` and verify free disk space ≥ `uncompressed_total` on the cache volume and ≥ `uncompressed_total` on the target volume.
   - During extraction, enforce: ≤ 5,000 entries, ≤ 400 MiB **actual bytes written** (defeating zip bombs with spoofed headers), reject `..` / absolute paths / duplicate case-folded (APFS/HFS+) paths, reject symlinks and all special file types (regular files and directories only—verified from `scripts/macos/bundle.sh:110`, which puts no symlinks inside `Oto.app`), require single top-level `Oto.app/`, and write clean modes (`0755` for dirs/executables, `0644` for regular files, stripping setuid/setgid/world-write).
   - `scripts/macos/release.sh` runs this **exact same extractor** (via `oto-sign extract-verify`) against `Oto-<version>-macos.zip` and validates the stapled notarization ticket (`xcrun stapler validate`), `codesign`, and `spctl` on the extracted `Oto.app`.
6. **Bundle metadata, Developer ID signature & Gatekeeper:** Verify `Oto.app/Contents/Info.plist` has `CFBundleIdentifier == "dev.oto.app"`, `CFBundleExecutable == "oto-app"`, and `CFBundleShortVersionString == manifest.version > current_version`. Verify code signatures with an explicit **Developer ID Application** requirement (excluding Apple Development and Mac App Store certs from the same team) on **both** the app and its embedded driver:
   ```text
   identifier "<BUNDLE_ID>" and anchor apple generic and certificate 1[field.1.2.840.113635.100.6.2.6] exists and certificate leaf[field.1.2.840.113635.100.6.1.13] exists and certificate leaf[subject.OU] = "<OTO_TEAM_ID>"
   ```
   Apply this same tightened requirement to `driver::requirement()` (`crates/oto-app/src/driver.rs:117`, updating the exact-string unit test at `driver.rs:471`), expose `driver::validate_with` as `pub(crate)`, and run `/usr/sbin/spctl --assess --type execute --verbose <cached_Oto.app>` on the `.app`-named extracted copy in the cache (offline Gatekeeper check against the stapled notarization ticket, since `NSURLSession` downloads carry no quarantine xattr), followed by `codesign --verify --deep --strict` + `driver::validate_with` on the non-`.app` sibling copy before swapping.

### 3. Transport & privacy: ephemeral `NSURLSession` on macOS, `ureq` later on Linux

Define a small synchronous `Transport` trait (`crates/oto-app/src/updater/http.rs`, two methods: `get_bytes` and `download_file`):
- **macOS backend:** Ephemeral `NSURLSession` via `objc2-foundation` + `block2` (enabling the `NSURL`, `NSURLRequest`, `NSURLResponse`, `NSHTTPURLResponse`, `NSURLSession`, `NSData`, `NSError`, `NSOperationQueue`, and `NSProcessInfo` features in `crates/oto-app/Cargo.toml:34`), with `HTTPCookieStorage = None` and `URLCredentialStorage = None` explicitly disabled on `ephemeralSessionConfiguration()`, and a `define_class!` delegate that validates every redirect hop before following, enforces the byte cap incrementally on incoming chunks, reports progress, and supports cancellation.
  - Minimal binary delta (measured in universal builds before merge; avoids linking a second TLS stack).
  - Automatically honors macOS System Settings proxies, PAC files, and system/enterprise TLS roots—which GUI apps launched from Finder or LaunchAgent do **not** get from `HTTPS_PROXY` env vars.
- **Linux backend (future, Phase 1 Track B):** `ureq` 3 with `rustls` (`ring` crypto provider, not `aws-lc-rs`) and `RootCerts::PlatformVerifier`.
- **Privacy guarantees:**
  - `User-Agent: Oto/<version>`; no machine ID, install ID, or telemetry headers.
  - Background checks fetch **only** `update.json` (≤ 64 KiB) at most once per 24 h (starting 20 s after launch), silently ignore offline errors, never open or focus the popover, and can be disabled in Settings (`check_for_updates = false`).
  - **Explicit download only:** The archive zip is downloaded **only** after the user clicks **Update… / Download**. Never silently download ~25–45 MB in the background.
  - *Public download-count note:* GitHub Releases exposes aggregate `download_count` per asset, so checking `/releases/latest/download/update.json` makes total check counts public (still zero per-user data). We accept this for v1 for zero extra infrastructure; because `update.json` is self-signed, mirroring it to GitHub Pages later is a single URL change.

### 4. Atomic, monotonic CI releases & secret isolation

- **Draft-then-publish, tag rulesets & repository immutability:** Set `git_release_draft = true` on `oto-app` in `release-plz.toml`. Add `[[package]] name = "oto-sign" release = false` so the workspace signing CLI is never versioned or tagged. As explicit one-time repository setup steps in `packaging/macos/README.md`: (a) enable GitHub's **Immutable Releases** setting (locking release assets and tag upon publication), and (b) add a GitHub **Tag Ruleset** protecting `v*` tags against force-update and deletion while drafts are open.
  - Verify on a test run that the pinned `release-plz/action` (`v0.5.148`, bundling `release-plz` CLI `0.4.3`) still emits `tag` for draft releases so `release-plz.yml`'s `macos` job runs as expected.
- **Secret isolation & real ancestry validation (compatible with `workflow_call` on `main`):**
  - Because `release-plz.yml` runs on `push: branches: [main]` and invokes `macos-release.yml` via `workflow_call` (using `GITHUB_TOKEN`, whose tag pushes cannot trigger separate workflows), `macos-release.yml`'s `github.ref` is **`refs/heads/main`**, NOT `refs/tags/v*`.
  - Remove `pull_request` and the existing workflow-level `concurrency` block from `macos-release.yml` (replacing it with per-job concurrency groups below); move PR verification to a separate read-only workflow (`.github/workflows/macos-pr-check.yml`) with **no secrets** and **no `contents: write`**, running an ad-hoc `scripts/macos/bundle.sh` build + zip extraction check.
  - Structure `macos-release.yml` into four cleanly separated jobs:
    1. **`validate-tag` (ubuntu, `contents: read` only, no secrets, no `environment`):** Verifies `github.ref == 'refs/heads/main'` and `inputs.tag` matches `^v[0-9]+\.[0-9]+\.[0-9]+$`. Checks out with `fetch-depth: 0`, fetches `origin/main` and `refs/tags/$TAG`, resolves `TAG_SHA="$(git rev-parse "refs/tags/$TAG^{commit}")"` and `MAIN_SHA="$(git rev-parse refs/remotes/origin/main)"`, enforces **`git merge-base --is-ancestor "$TAG_SHA" "$MAIN_SHA"`**, verifies `Cargo.toml` at `$TAG_SHA` matches `$TAG`, and outputs `sha=$TAG_SHA`.
    2. **`check-release` (ubuntu, `needs: validate-tag`, `contents: write`, NO `environment` / NO signing secrets):** Queries `gh release view "$TAG" --json isDraft` (requires `contents: write` on `GITHUB_TOKEN` to see draft releases) and outputs `is_draft`.
    3. **`dmg` (macos-26, `needs: [validate-tag, check-release]`, `if: needs.check-release.outputs.is_draft == 'true'`, `environment: release`, per-tag concurrency `macos-release-build-${{ inputs.tag }}`):** Checks out the **exact verified `needs.validate-tag.outputs.sha`**, runs `scripts/macos/release.sh`, and uploads the built DMG, zip, checksums, and unsigned `update-payload.json` as workflow artifacts.
    4. **`publish` (ubuntu, `needs: [validate-tag, check-release, dmg]`, `if: always() && needs.validate-tag.result == 'success' && needs.check-release.result == 'success' && (needs.dmg.result == 'success' || needs.dmg.result == 'skipped')`, `environment: release`, global concurrency `group: oto-release-publish`, `cancel-in-progress: false`):**
       - **Under the global publish lock**, re-verifies that remote `refs/tags/$TAG` still points to `needs.validate-tag.outputs.sha` and **re-reads `isDraft`** so overlapping dispatches cannot race past a stale `is_draft`.
       - If `isDraft == true`: requires `needs.dmg.result == 'success'` (fails if `dmg` was skipped or failed—never publishes an empty draft), downloads artifacts, signs `update-payload.json` → `update.json`, uploads all assets, verifies every asset via the GitHub release assets API (`state == "uploaded"`, exact `size`, and `digest` with re-download SHA-256 fallback if `digest` is omitted), compares `semver($TAG)` against the current published latest release, publishes with `gh release edit "$TAG" --draft=false --latest` **only** if strictly newer (otherwise `--draft=false --latest=false`), and runs monotonic Homebrew tap reconciliation.
       - If `isDraft == false` (immutable published release): skips uploads, verifies the existing published signed envelope and assets (API state/size/digest + re-download fallback), and runs **idempotent monotonic Homebrew tap reconciliation** using the verified published DMG digest before exiting 0—so a failed tap push is recoverable without ever clobbering live assets.
  - All signing secrets (`OTO_UPDATE_SIGNING_KEY` and Apple signing/notarization secrets) live inside GitHub Environment `release` restricted to protected branch `main` only. Manual recovery dispatches run against `main`: `gh workflow run macos-release.yml --ref main -f tag=vX.Y.Z`.
- **Build & round-trip verification in `scripts/macos/release.sh`:**
  - After `xcrun stapler staple "$APP"`, package `target/dist/Oto-<version>-macos.zip` with `ditto -c -k --keepParent --norsrc --noextattr "$APP"` (plus `.sha256`).
  - Round-trip extract with the **exact same Rust extractor** (`oto-sign extract-verify`) and verify `codesign --verify --deep --strict`, embedded driver & `oto-hal-helper` signatures, `spctl --assess --type execute`, `xcrun stapler validate`, universal architectures (`lipo -archs`), `LSMinimumSystemVersion`, and agreement between `driver/macos/DRIVER_VERSION`, bundled `OtoDriverVersion`, and `update-payload.json`.
- **Recovery runbook in `packaging/macos/README.md`:** Document (a) failed build, displaced publish queue (GitHub keeps one pending run per concurrency group), or failed tap push → `gh workflow run macos-release.yml --ref main -f tag=vX.Y.Z`; (b) tag pushed before draft creation succeeded → `gh release create "$TAG" --draft` then dispatch `macos-release.yml --ref main -f tag=$TAG`.

### 5. Decoupled HAL driver versioning & engine-enforced compatibility

Separate the HAL driver ABI/implementation revision from the app release version so routine app updates need zero admin prompts and zero `coreaudiod` restarts:

- **Single source of truth:** `driver/macos/DRIVER_VERSION` (plain integer, starting at `1`).
  - `driver/macos/Makefile` makes `DRIVER_VERSION` an explicit prerequisite of both `$(BIN)` and `$(PLIST)`, passes `-DOTO_DRIVER_VERSION=$(DRIVER_VERSION)` to `clang`, and writes `OtoDriverVersion` (integer), `CFBundleVersion`, and `CFBundleShortVersionString` in `driver/macos/Info.plist` from `DRIVER_VERSION`. Remove the stale `VERSION="$VERSION"` overrides in `scripts/macos/bundle.sh:65,67`. The driver's revision (and behavior) stays unchanged across app-only releases.
  - Add a CI check that fails if `driver/macos/src/**`, `driver/macos/Info.plist`, or `driver/macos/Makefile` changed since the last tag without bumping `driver/macos/DRIVER_VERSION`.
  - **Forward ABI policy:** Newer driver revisions (`loaded > bundled`) are required to remain backward-compatible with already-shipped engines; if a future breaking driver revision is ever needed, an explicit compatibility range (`max_driver_version`) will be introduced before shipping it.
- **Core Audio IPC-safe version property:**
  - `AudioServerPlugIn.h` only transports custom properties across process boundaries when declared in `kAudioObjectPropertyCustomPropertyInfoList` (`'cust'`) as `CFStringRef` or `CFPropertyListRef` (raw `UInt32` does not cross IPC).
  - Expose `'otvr'` on **`kObj_Mic`** (resolved by UID `dev.oto.microphone` in clients—never hardcode driver-internal `kObj_PlugIn`/`kObj_Mic` IDs in `oto-engine`) declared in `'cust'` with `mPropertyDataType = kAudioServerPlugInCustomPropertyDataTypeCFString` and `mQualifierDataType = kAudioServerPlugInCustomPropertyDataTypeNone`.
  - Implement `'otvr'` in `HasProperty`, `IsPropertySettable` (`false`), `GetPropertyDataSize`, and `GetPropertyData` returning an immortal `CFSTR(...)` macro string (zero allocations in all four entry points). Client callers must `CFRelease` the returned `CFStringRef`.
  - Add a property round-trip check to `driver/macos/tools/oto-verify.c` and verify against a loaded driver on macOS before relying on it.
- **Engine-enforced compatibility (`crates/oto-engine`):**
  - Add `crates/oto-engine/build.rs` reading `../../driver/macos/DRIVER_VERSION` (emitting `cargo:rerun-if-changed=../../driver/macos/DRIVER_VERSION`) to define `BUNDLED_DRIVER_VERSION` alongside `MIN_COMPATIBLE_DRIVER_VERSION` (`1`: legacy v0.0.1–v0.0.3 drivers predate `'otvr'`, read as `Loaded(0)`, and are reported as incompatible so the app offers **Update Driver…**).
  - Preserve Core Audio errors in `crates/oto-engine/src/platform/macos/mod.rs`: UID lookup returns `Result<Option<AudioObjectID>, OSStatus>`, `default_input_uid()` (`macos/mod.rs:484`) returns `Result<Option<String>, String>`, and activity probes return `Result<Option<bool>, OSStatus>` instead of folding errors to `false`.
  - Update **all** `RestoreTarget` methods (`open`, `switch_to_oto`, `restore`, `default_changed` in `crates/oto-engine/src/state.rs:84-166`, updating `restore` and `default_changed`'s `current` closure from `FnOnce() -> Option<String>` to `FnOnce() -> Result<Option<String>, String>`) so a transient Core Audio query error **never** erases the saved restore target.
  - Add `driver_status(&self) -> DriverStatus` to the engine `Platform` trait (implemented in fakes in `control.rs` and `fault_check.rs`), keeping `clients_active` independent of driver compatibility:
    - `Absent`: both `dev.oto.microphone` and `dev.oto.sink` return `Ok(None)`.
    - `Loaded(rev)`: both UIDs return `Ok(Some(_))`, and either `'otvr'` parses cleanly as `u32`, or `AudioObjectHasProperty(..., 'otvr') == false` (specifically identifying legacy v0.0.1–v0.0.3 drivers as `rev = 0`).
    - `Unknown`: partial enumeration (only one UID resolves), transient Core Audio error (e.g. `coreaudiod` restarting), or malformed `'otvr'` string. Never fold `Unknown` into legacy `0` or incompatible.
  - Skip the version gate when `EngineOptions::mic_uid` or `sink_uid` is overridden (dev/test paths).
  - Compatibility behavior in `control.rs` and `set_default_input_to_oto`:
    - Require `Loaded(rev)` with `rev >= MIN_COMPATIBLE_DRIVER_VERSION` before opening streams or setting Oto as the system default input.
    - If `Loaded(rev)` with `rev < MIN_COMPATIBLE_DRIVER_VERSION`: refuse streams and system-mic selection, and emit `Event::DriverIncompatible { loaded, min_required }` (not an `Event::Error`), which the app shows as the urgent `DriverUiState::Incompatible` banner.
    - If `Unknown` at startup: do not permanently refuse; retry on `DevicesChanged` and reconcile ticks. If `Unknown` occurs transiently while streams are already running, do not tear down running streams—let device-alive/stall detection handle real `coreaudiod` restarts.
- **Complete driver state table (UI + `Probe`):**

  | Loaded (Core Audio) | On disk (`Info.plist`) | State & UI action |
  |---|---|---|
  | `Absent` | absent | Not installed → onboarding / **Install…** |
  | `Absent` | present (any rev) | Installed on disk, not loaded → *"Restart your Mac to finish."* |
  | `Unknown` | any | Transient/rechecking → keep current state, do not flag incompatible |
  | `Loaded(r == bundled)` | `== bundled` or `> bundled` | Up to date (no prompt, no banner) |
  | `Loaded(r == bundled)` | absent or `< bundled` | Loaded copy works now, on-disk copy missing/outdated → offer **Reinstall…** in Settings |
  | `Loaded(r > bundled)` (`r >= min_compat`) | any | Newer compatible driver already loaded (e.g. older app opened) → allow, never offer downgrade |
  | `Loaded(r < bundled)` | `>= bundled` | New driver already on disk, `coreaudiod` still holds old one → *"Restart your Mac to finish updating the audio driver."* |
  | `Loaded(r < bundled)` | absent | Driver removed on disk while still loaded → *"Restart your Mac to finish removing the driver."* (or **Reinstall…**) |
  | `Loaded(min_compat <= r < bundled)` | `< bundled` | Compatible older driver → audio keeps working; show non-blocking *"Audio driver update available"* banner with explicit **Update Driver…** button |
  | `Loaded(r < min_compat)` | `< bundled` | Incompatible driver → engine refuses streams/system-mic; show urgent **Update Driver…** banner |

  *(Legacy `Info.plist` files with `CFBundleVersion = "0.0.3"` and no `OtoDriverVersion` key parse as on-disk version `0`.)*
- **Kernel-locked, atomic HAL driver install/uninstall & explicit execution boundary:**
  - Add a tiny universal C tool `driver/macos/tools/oto-hal-helper.c` (built by `driver/macos/Makefile` alongside `oto-verify`, signed with the same `SIGN_ID` / `SIGN_FLAGS` as the driver and identifier `"dev.oto.hal-helper"`—so Developer ID builds get hardened runtime + timestamp and ad-hoc local builds via `install-driver.sh` work seamlessly without a Developer ID cert—bundled at `Oto.app/Contents/Helpers/oto-hal-helper`, and shared by **both** `driver.rs` and `scripts/macos/{install,uninstall}-driver.sh`):
    1. **Kernel-held lock (no stale-timeout races):** Opens persistent root-owned `/Library/Audio/Plug-Ins/HAL/.OtoMicrophone.driver.lock` (`0600 root:wheel`, never unlinked) and takes an exclusive kernel `flock()`, which the OS releases automatically on exit, `SIGKILL`, or power loss.
    2. **Atomic HAL bundle replacement & unambiguous temp naming:**
       - Treat legacy `.OtoMicrophone.driver.old-*` leftovers as **ambiguous** (the pre-plan installer and uninstaller both used `.old-*`, so a crash with `DEST` missing cannot distinguish a failed install from a failed uninstall; never auto-restore `.old-*`, and clean up stale `.old-*` only after an explicit install or uninstall succeeds).
       - New installs/upgrades stage into `.OtoMicrophone.driver.swap-<nonce>` and replace an existing `DEST` in one atomic step via `renamex_np(STAGE, DEST, RENAME_SWAP)` (or single `rename` on fresh install), then delete the swapped-out copy—so `DEST` is **never** absent mid-upgrade!
       - New uninstalls rename `DEST` to `.OtoMicrophone.driver.removing-<nonce>` (distinct prefix, never restored by an installer) before `rm -rf`.
    3. **Post-authorization non-root audio check:** Before running `oto-hal-helper`, the root script copies `oto-hal-helper` into the root-owned `$STAGE_DIR` (`0755 root:wheel`) and verifies `codesign --verify --strict -R="$HELPER_REQ"` on the root-owned copy (pinning Developer ID + `OTO_TEAM_ID` in release builds, or identifier `"dev.oto.hal-helper"` in ad-hoc dev builds; zero TOCTOU). Immediately before `killall coreaudiod`, it invokes that root-owned verified binary **as the invoking non-root user** (`launchctl asuser "$INVOKING_UID" /usr/bin/sudo -u "#$INVOKING_UID" "$STAGE_HELPER" --check-audio-idle`) under a 3 s timeout to query `kAudioDevicePropertyDeviceIsRunningSomewhere` across all Core Audio devices (`kAudioHardwarePropertyDevices`). Note: Oto's own streams are idle (gated by `clients_active`), and always-running devices (e.g. Bluetooth headsets or browsers holding output open) will trigger the safe reboot deferral. If any device is running—or the check errors or times out—it **skips `killall coreaudiod`**, leaves the newly verified driver installed on disk, and emits `OTO_RESULT=reboot` (`Outcome::InstalledNeedsReboot`: *"Installed. Restart your Mac when your audio is idle to finish."*). *(Residual race: Core Audio has no atomic admission lock on `coreaudiod`, so a new stream starting in the milliseconds between the final check and `killall` can still be interrupted; the copy and acceptance criteria reflect this honestly.)*
  - Update `driver::install()`'s wait loop (`driver.rs:345`) to wait until `Engine::driver_status() == DriverStatus::Loaded(BUNDLED_DRIVER_VERSION)`, explicitly **skipping** this wait when the privileged step already returned `OTO_RESULT=reboot` (`InstalledNeedsReboot`). Update `driver::uninstall()` (`driver.rs:380`) to wait for explicit `DriverStatus::Absent` (treating `Unknown` as "not yet").
  - Driver upgrades are **always** triggered by an explicit user click with clear copy (*"Asks for your password. Audio will pause briefly in all apps (or finish on your next restart if audio is playing)."*), serialized with app update apply in `State` (routing Settings and onboarding through the same guard), and pre-checked off the UI thread before prompting.

### 6. Recoverable, call-safe bundle swap & relaunch transaction

- **Storage locations (all durable transaction & recovery artifacts live in non-purgeable storage):**
  - **Non-purgeable transaction state** lives in `~/Library/Application Support/oto/` (alongside `engine.lock` and `restore-input`): `updating.lock`, `update-tx.json`, `tx-<nonce>-state`, `ack-<nonce>-started`, `ack-<nonce>-ready`, **and `update-helper-<nonce>`** (so macOS cache pressure can never purge the recovery executable while a transaction is unresolved; deleted only after terminal cleanup and any required relaunch complete).
  - **Purgeable download/staging cache** lives in `~/Library/Caches/dev.oto.app/updates/`: `<version>.zip.part` and extracted `Oto.app`.
- **Install-mode classification** (computed off the UI thread from canonical `current_exe`):
  1. Not inside an `.app` bundle, or `OTO_TEAM_ID` unset (dev/ad-hoc build) → `Unsupported(DevBuild)`.
  2. Engine running without `engine.lock` (`!Engine::has_instance_lock()`) → `NotifyOnly(NoInstanceLock)`.
  3. `login::registrable_bundle(exe)` is `None` (`AppTranslocation`) or `statfs` has `MNT_RDONLY` (mounted DMG) → `NotifyOnly(NotInApplications)` (*"Move Oto to Applications for automatic updates."*).
  4. Bundle is owned by the current user (`st_uid == geteuid()`), parent dir is writable (`access(parent, W_OK)`), bundle is writable (`access(bundle, W_OK)`), target is a real directory (not a symlink), and target volume supports atomic exchange (`getattrlist` `VOL_CAP_INT_RENAME_SWAP`, or `f_fstypename == "apfs"`) → `SelfUpdatable`.
  5. Otherwise (bundle owned by `root` or another macOS user, standard non-admin user in `/Applications`, or filesystem lacking `RENAME_SWAP`) → `NotifyOnly(RequiresManualInstall)` (*"Download the installer to update Oto."*). *(Because only the bundle's owning user can ever self-update it, per-user locks and transaction journals are never shared across untrusted local accounts; another logged-in user on the same Mac is `NotifyOnly` and continues running cleanly from the swapped-out inode until restarted.)*
- **Pre-shutdown staging (while old `Oto.app` is still running and untouched):**
  1. Copy verified `Oto.app` onto the target's parent directory under a non-`.app` sibling name (`.Oto-update-<nonce>`, so LaunchServices never indexes a second bundle) using `clonefile`, falling back to a metadata-preserving recursive copy on `EXDEV` or `ENOTSUP`.
  2. Re-verify `Info.plist` (`CFBundleIdentifier`, `CFBundleExecutable`, `CFBundleShortVersionString == manifest.version > target_version`), Developer ID `codesign --verify --deep --strict` on the sibling copy, and `driver::validate_with` on its embedded driver (with `spctl --assess --type execute` already verified on the `.app`-named cache copy).
  3. Copy the currently running (trusted) `oto-app` executable to non-purgeable `~/Library/Application Support/oto/update-helper-<nonce>` (mode `0700`, outside both staged and target bundles; safe because `oto-app` has no `@rpath` frameworks—ONNX Runtime is statically linked per `crates/oto-dsp/Cargo.toml:13`, revisit if bundled frameworks are ever added).
- **Helper handoff, serialized journal creation, pipe commit protocol & acknowledged shutdown:**
  1. Create an inherited pipe `(commit_rx, commit_tx)` (`commit_tx` stays `O_CLOEXEC` in the old app) and a startup-ready pipe from helper → old app.
  2. Spawn the helper (`--finish-update --nonce <nonce> --target <target> --staging <staging> --old-ver <old> --new-ver <new> --old-pid <pid> --commit-fd <fd>`) in its own session (`setsid()` in `pre_exec`, surviving LaunchAgent process-group teardown!), logging to `~/Library/Logs/Oto/updater.log`.
  3. **Helper acquires `updating.lock` FIRST** (bounded retry up to 5 s so a racing caller's momentary `try_lock` never causes a spurious abort). **Under `updating.lock`, check if `update-tx.json` already exists:** if so, run shared recovery first (and abort starting the new transaction if an unresolved transaction cannot be cleared) so a stale journal/backup is never overwritten! Otherwise atomically write `~/Library/Application Support/oto/update-tx.json` (`{ nonce, phase: "prepared", started_at, stage1_deadline: null, stage2_deadline: null, commit_deadline: null, target_path, staging_path, old_version, new_version, helper_path, candidate_pid: null }`) while holding `updating.lock`, and signal "ready" back to the old app. If helper fails to start or acquire `updating.lock`, old app cleans up `staging` and `update-helper-<nonce>` and stays running.
  4. Old app sends `Request::CommitShutdown(ack_tx)` to the host thread:
     - Host re-probes Oto Microphone activity (`is_running_somewhere_checked() -> Result<Option<bool>, OSStatus>`): `Ok(None)` (no Oto Microphone installed) and `Ok(Some(false))` allow the update; `Ok(Some(true))` (call active) or `Err(_)` (lookup/activity error) aborts with `"End your call to finish updating."` (or retry message).
     - Host stops streams and restores the previous default input via checked `RestoreTarget::restore()`.
     - If restoration fails, host restarts/keeps itself running with its previous settings and returns `Err(msg)` so old app cancels the helper (closes `commit_tx` with 0 bytes) and surfaces the error.
     - On `Ok(())`, host drops `Engine` (releasing `engine.lock`) and replies `Ok(())`.
  5. Old app writes the 1-byte **commit** signal to `commit_tx`. **If writing fails (`EPIPE`, helper already died):** old app restarts its engine with its previous settings, surfaces a recoverable update error, and stays running instead of quitting! On success, old app marks shutdown complete (making `on_quit`'s `State::shutdown()` idempotent), calls `tx.quit()`, and exits. (If old app cancels before writing that byte, helper sees EOF with 0 bytes, deletes `update-tx.json`, `staging`, and `update-helper-<nonce>` while holding `updating.lock`, releases `updating.lock`, and exits cleanly without touching `<target>`.)
- **Race-free admission & strictly inert candidate startup in shared `oto-engine` + `oto-app`:**
  - Add `EngineOptions::update_candidate: Option<CandidateToken>` and put update-interlock admission directly in `Engine::start_with` (`crates/oto-engine/src/lib.rs:202`, `crates/oto-engine/src/state.rs`) so **every** caller (`oto-app`, `oto run`, `oto bench --live`, `oto latency`) enforces the exact same rule:
    1. **Non-candidate callers** check whether `updating.lock` is held **OR** an unresolved `update-tx.json` exists (`phase` in `swapping | verifying | rolling_back`) *before* acquiring `engine.lock`, AND **re-check both immediately after acquiring `engine.lock`**; if either holds, they immediately drop `engine.lock` and return `Err(UPDATING)`! *(Prevents `oto run` / `oto bench --live` / `oto latency` from claiming `engine.lock` or mutating `RestoreTarget` after a helper crash before recovery runs! `oto-app` runs startup recovery BEFORE calling `Engine::start_with`, and treats `Err(UPDATING)` as a silent exit; `oto-cli` exits cleanly advising `oto recover-update` or waiting for the update.)*
    2. **Designated candidate** (`update_candidate: Some(token)`): checks `tx-<nonce>-state == "active"` (`current_exe` inside `target`, `CARGO_PKG_VERSION == expected_version`) *before* acquiring `engine.lock`, **requires a real acquired `InstanceLock`** (`Err` if lockless), writes its PID to `engine.lock` (so recovery can verify lock-holder identity), AND **re-checks `tx-<nonce>-state == "active"` immediately after acquiring `engine.lock`** (dropping `engine.lock` and returning `Err(UPDATING)` if revoked/rolled back in between!).
    3. **Strictly inert until durable commit:** While `update_candidate` is uncommitted:
       - `Engine::start_with` acquires `engine.lock` and starts the control thread in **`candidate_inert = true`** mode: `RestoreTarget` is **NOT** opened or mutated, and `control.rs::reconcile()` does **NOT** open capture/sink audio streams even if a client connects to `Oto Microphone`!
       - `oto-app` gates **all** mutating UI hooks (`save`, `set_login`, `install`, `uninstall`, `request_mic`, `sync_system_mic`), **"Hear the Difference"** (`State::hear_blocked()` returns `Some("Finishing update…")` and `start_hear_test()` rejects calls through both tray and fallback-window entry points), and automatic onboarding popup until commit!
       - Calling `Engine::commit_candidate()` flips `candidate_inert = false`, opens/adopts `RestoreTarget`, immediately runs `reconcile()` to admit any waiting audio clients, and unlocks all UI hooks and "Hear the Difference".
    4. Meanwhile, once the helper observes EOF on `commit_rx` (proving full old-process exit), the helper acquires and holds `engine.lock` (bounded retry up to 5 s, while already holding `updating.lock`) across `renamex_np`, `LSRegisterURL`, and `tx-<nonce>-state = "active"`, releasing `engine.lock` **only** immediately before `NSWorkspace` launches the designated candidate (keeping `updating.lock` held through commit or rollback!).
- **Single-writer control/ack files, persisted deadlines, atomic swap, `NSWorkspace` launch, durable commit & rollback:**
  - **One writer per file (stored in non-purgeable `~/Library/Application Support/oto/`):**
    - Helper-only control file: `tx-<nonce>-state` (`0600`), written atomically by the helper (or recovery holding `updating.lock`): `"active"` → `"committed"` **or** `"revoked"`.
    - Candidate-only ack files: `ack-<nonce>-started` (`{ nonce, pid }`) and `ack-<nonce>-ready` (`{ nonce, pid, ready_at }`), written atomically by the candidate only.
  1. Helper reads the 1-byte commit signal, then waits for **EOF on `commit_rx`** (proving full old-process exit with zero PID-reuse risk), bounded at 10 s. If EOF does not arrive in 10 s, helper deletes `update-tx.json` + `staging` + `update-helper-<nonce>` while holding `updating.lock`, releases `updating.lock`, and only launches `<target>` (`createsNewApplicationInstance = true`) if `old_pid` has actually exited (never activating a dying process).
  2. Helper acquires `engine.lock`, updates `update-tx.json` to `phase: "swapping"`, re-verifies target bundle identity/version, and executes a **single atomic exchange**: `renamex_np(staging, target, RENAME_SWAP)`. Now `target` is the new `Oto.app` and `staging` is the verified old backup. Calls `LSRegisterURL(target, true)` (linked via `CoreServices`) so LaunchServices refreshes bundle metadata.
  3. Helper writes `tx-<nonce>-state = "active"` and updates `update-tx.json` with `phase: "verifying"` and **persisted deadlines** (`stage1_deadline = now + 15 s`, `stage2_deadline = now + 35 s`, `commit_deadline = now + 50 s`) so both the live helper and dead-helper recovery enforce identical timeouts!
  4. Helper releases `engine.lock` (keeping `updating.lock` held!) and launches `<target>` via **`NSWorkspace.sharedWorkspace.openApplicationAtURL:configuration:completionHandler:`** with `createsNewApplicationInstance = true` (macOS 11+) and `arguments = ["--update-ack", nonce]`, retaining the returned `NSRunningApplication` and tracking whether the async launch callback has completed.
  5. **Designated candidate & two-stage ack (independent of `DriverStatus`):**
     - First, candidate checks `--update-ack <nonce>` against `tx-<nonce>-state == "active"` (`current_exe` inside `target`, `CARGO_PKG_VERSION == expected_version`). If `tx-<nonce>-state` is `"revoked"` or missing, candidate exits immediately with zero side effects.
     - **Stage 1 (`started`):** Immediately after passing that admission check (before initializing Rune or the engine), candidate writes `ack-<nonce>-started` (`{ nonce, pid }`); helper records `candidate_pid` in `update-tx.json`.
     - **Stage 2 (`ready`):** Written to `ack-<nonce>-ready` (`{ nonce, pid, ready_at }`) once `Engine::start_with` acquires a real `engine.lock` (with post-lock `tx-<nonce>-state == "active"` re-check) AND tray initialization (or fallback window) succeeds—explicitly **independent** of `DriverStatus`, stream opening, or mic permission so a release raising `MIN_COMPATIBLE_DRIVER_VERSION` or encountering a pending reboot commits cleanly!
     - Candidate waits for commit by polling `tx-<nonce>-state`:
       - On **`"committed"`**: calls `Engine::commit_candidate()`, unlocks all UI hooks and "Hear the Difference", and **the candidate itself** deletes `tx-<nonce>-state` and `ack-<nonce>-*`.
       - On **`"revoked"`** or **missing** (whether `updating.lock` is held or free, unless self-recovering below): exits immediately with zero side effects.
  6. **Strict helper commit order (all journal cleanup under `updating.lock`, never deleting `tx-<nonce>-state`):**
     - On observing `ack-<nonce>-ready` (matching `nonce` and expected candidate PID), while **still holding `updating.lock`**, the helper:
       1. Writes durable terminal `phase: "committed"` to `update-tx.json`.
       2. Atomically writes `tx-<nonce>-state = "committed"` (never deleting `tx-<nonce>-state` or `ack-<nonce>-*`—the candidate owns deleting them after reading `"committed"`, and future startups clean any stale terminal files for older nonces).
       3. Best-effort deletes `staging` (old backup) and zip (partial `rm -rf` failure is logged and never fails the update).
       4. Deletes `update-tx.json` **before** releasing `updating.lock` (so the next transaction cannot race on `update-tx.json`).
       5. Releases `updating.lock` (helper binary `update-helper-<nonce>` is cleaned up on next startup once no unresolved journal references it).
  7. **Rollback on failure or hang (using the persisted `stage1_deadline` / `stage2_deadline`):**
     - If `NSWorkspace` fails, the retained `NSRunningApplication` terminates before `ack-<nonce>-ready`, `now > stage1_deadline` without `ack-<nonce>-started`, or `now > stage2_deadline` without `ack-<nonce>-ready`:
       1. **Revoke admission positively:** atomically write `tx-<nonce>-state = "revoked"` (serialized under `updating.lock`).
       2. Wait up to 3 s for any still-in-flight `NSWorkspace` completion handler to resolve so we capture any late `NSRunningApplication`.
       3. Stop the exact candidate (`NSRunningApplication.terminate()`, wait 3 s, then `forceTerminate()`, and check any PID in `update-tx.json` / `ack-<nonce>-started` with `proc_pidpath` matching `<target>`'s executable), waiting until terminated **and** re-acquiring `engine.lock`.
       4. Write `phase: "rolling_back"` to `update-tx.json`, execute `renamex_np(staging, target, RENAME_SWAP)`, call `LSRegisterURL(target, true)`, write terminal `phase: "rolled_back"` (`relaunch_pending: true`) to `update-tx.json`, best-effort delete the failed candidate (`staging`), delete `tx-<nonce>-state` + ack files, release `engine.lock`, launch the restored `<target>` via `NSWorkspace` (`createsNewApplicationInstance = true`, `arguments = ["--wait-update-release"]`), and only after `NSWorkspace` confirms launch started delete `update-tx.json` and immediately release `updating.lock`. *(When launched with `--wait-update-release`, `oto-app` blocks up to 5 s waiting for `updating.lock` to clear before checking engine admission so it never exits prematurely with `UPDATING`; `update-helper-<nonce>` is preserved until the restored app cleans up old helpers on normal startup.)*
- **Complete dead-helper, hung-candidate & independent recovery state machine:**
  - **Independent recovery entry point:** Shared recovery logic runs automatically on the next `oto-app` launch (before `Engine::start_with`) and is also exposed independently via `~/Library/Application Support/oto/update-helper-<nonce> --recover-update` (preserved copy of the old working binary in non-purgeable storage) and `oto recover-update` (in `oto-cli`). Document both manual recovery commands (and *"Force Quit, then reopen"* if a candidate hangs after helper death) in `packaging/macos/README.md`.
  - **Stuck candidate (helper died after Stage 2):** While awaiting `"committed"`, the healthy candidate polls `updating.lock` (every 500 ms). If `updating.lock` becomes free while `tx-<nonce>-state == "active"` (and the candidate holds `engine.lock` and wrote `ack-<nonce>-ready`), the candidate acquires `updating.lock` and **re-reads both `update-tx.json` and `tx-<nonce>-state` under the lock**:
    - If `tx-<nonce>-state == "committed"` already → if `update-tx.json` is still present for this nonce, best-effort delete `staging_path` first, then delete `update-tx.json`, release `updating.lock`, unlock deferred side effects, and delete `tx-<nonce>-state` + `ack-<nonce>-*`.
    - If `tx-<nonce>-state == "revoked"` → release `updating.lock` and exit immediately with zero side effects.
    - If `tx-<nonce>-state == "active"` and `update-tx.json` matches this nonce → write `phase: "committed"` to `update-tx.json`, write `tx-<nonce>-state = "committed"`, best-effort delete `staging`, delete `update-tx.json`, release `updating.lock`, unlock deferred side effects, and delete `tx-<nonce>-state` + `ack-<nonce>-*`.
  - **Dead-helper recovery (triggered on next `oto-app` startup or `--recover-update` whenever `update-tx.json` exists and `updating.lock` is NOT held):**
    - Acquire `updating.lock` first, then dispatch strictly by `phase`:
      - **`prepared`** (no swap occurred—does **NOT** require `engine.lock`, since old app may legitimately still be running!): delete `staging_path`, `tx-<nonce>-state`, ack files, `update-tx.json`, and `helper_path`; release `updating.lock`.
      - **`committed`** (terminal, idempotent): **first reconcile the candidate signal under `updating.lock`**—if `tx-<nonce>-state` exists and is still `"active"` (helper was killed between writing `phase: "committed"` and publishing `tx-<nonce>-state = "committed"`), atomically write `tx-<nonce>-state = "committed"` (never deleting `tx-<nonce>-state` or `ack-<nonce>-*` while the candidate may still read it!). Then best-effort delete `staging_path`, delete `update-tx.json`, and release `updating.lock` (cleaning `helper_path` and stale terminal files for older nonces once no matching live candidate exists).
      - **`rolled_back`** (terminal): best-effort delete any leftover `staging_path`, `tx-<nonce>-state`, and ack files (keep `helper_path` until relaunch succeeds).
        - If this process **IS** the restored `<target>` starting normally → delete `update-tx.json` + `helper_path`, release `updating.lock`, and continue normal startup.
        - Else if `relaunch_pending == true`: check `engine.lock` with `try_lock`:
          - If `engine.lock` is held: inspect the lock-holder PID via `proc_pidpath`; **only** if it matches `<target>`'s executable (restored app is already running), delete `update-tx.json` + `helper_path` and release `updating.lock`. Otherwise (e.g. an unrelated `oto` CLI holds `engine.lock`), preserve `update-tx.json` and `helper_path` and release `updating.lock` so the relaunch obligation is not lost.
          - If `engine.lock` is free: release `engine.lock`, launch restored `<target>` via `NSWorkspace` (`createsNewApplicationInstance = true`, `arguments = ["--wait-update-release"]`), delete `update-tx.json` only after `NSWorkspace` confirms launch started (leaving `helper_path` for the restored app's normal startup cleanup), and immediately release `updating.lock`.
        - Else (`relaunch_pending == false`): delete `update-tx.json` + `helper_path` and release `updating.lock`.
      - **`swapping`, `verifying`, or `rolling_back`:** Check `engine.lock` with `try_lock`:
        - **If `engine.lock` IS held:**
          - Verify lock-holder identity (`engine.lock` PID matches recorded `candidate_pid` / `ack-<nonce>-started` PID AND `proc_pidpath` matches `<target>`'s executable). If it is NOT a verified candidate in `phase == "verifying"`, release `updating.lock` and skip recovery!
          - When `phase == "verifying"` and lock holder IS the verified candidate:
            - If `tx-<nonce>-state == "committed"` (commit already durable): best-effort delete `staging_path`, delete `update-tx.json`, release `updating.lock` (leaving live committed candidate untouched!).
            - Else if `ack-<nonce>-ready` exists and `now <= commit_deadline`: healthy candidate is about to self-commit → release `updating.lock` and skip.
            - Else if (`ack-<nonce>-ready` exists and `now > commit_deadline`, i.e., candidate hung after `ready` without self-committing) **OR** (`ack-<nonce>-ready` does NOT exist and `now > stage2_deadline`, i.e., candidate hung before `ready`): write `tx-<nonce>-state = "revoked"`, terminate verified `candidate_pid` (`SIGTERM` → 3 s → `SIGKILL`), acquire `engine.lock`, and fall through to rollback below!
            - Else (`now <= stage2_deadline`): candidate is still starting → release `updating.lock` and skip.
        - **Once both `updating.lock` and `engine.lock` are held:**
          - **`swapping` or `verifying`**: inspect `Info.plist` versions at `target_path` and `staging_path`:
            - `target == old_version` → swap never occurred → delete `staging_path`, `tx-<nonce>-state`, ack files, `update-tx.json`, `helper_path`; release locks.
            - `target == new_version` and `staging == old_version`:
              - If `tx-<nonce>-state == "committed"` → finalize cleanup of `staging_path`, `update-tx.json`, stale ack/state files, `helper_path`; release locks.
              - Otherwise: write `tx-<nonce>-state = "revoked"`, if any recorded `candidate_pid` is still alive and passes `proc_pidpath` terminate it, write `phase: "rolling_back"`, swap `old_version` back via `renamex_np(staging, target, RENAME_SWAP)` + `LSRegisterURL(target, true)`, write `phase: "rolled_back"` (`relaunch_pending: true`), best-effort delete `staging_path`, delete `tx-<nonce>-state` + ack files, release `engine.lock`, launch restored `<target>` via `NSWorkspace` (`createsNewApplicationInstance = true`, `arguments = ["--wait-update-release"]`), delete `update-tx.json` after launch starts, release `updating.lock`, and exit immediately if running from the rolled-back bundle or `--recover-update` (`helper_path` is cleaned up once the restored app starts).
          - **`rolling_back`**: inspect `Info.plist` at `target_path` vs `staging_path`:
            - If `target == new_version` and `staging == old_version` → complete `renamex_np(staging, target, RENAME_SWAP)` + `LSRegisterURL(target, true)`.
            - Either way (including `target == old_version`, where the helper crashed after swapping back but before relaunching): write `phase: "rolled_back"` (`relaunch_pending: true`), best-effort delete `staging_path`, delete `tx-<nonce>-state` + ack files, release `engine.lock`; if running via `--recover-update` or from the rolled-back bundle (i.e. no healthy restored Oto is already starting in this process), launch restored `<target>` via `NSWorkspace` (`createsNewApplicationInstance = true`, `arguments = ["--wait-update-release"]`), then delete `update-tx.json` after launch starts, release `updating.lock`, and exit (preserving `helper_path` until the restored app cleans it on normal startup); if this process IS the restored `<target>` already starting normally, delete `update-tx.json` + `helper_path`, release `updating.lock`, and continue normal startup!

### 7. Homebrew Cask & Linux forward compatibility

- **Homebrew Cask:** Maintain a tap (`deepso7/homebrew-oto`) with `auto_updates true` (so Oto owns in-app updates and `brew upgrade` skips it unless `--greedy`) and `uninstall quit: "dev.oto.app"`. **Do not** add `uninstall delete:` for `/Library/Audio/Plug-Ins/HAL/OtoMicrophone.driver`—the HAL driver is machine-wide, may be shared across users/copies, requires `sudo` + `coreaudiod` restart, and must stay under Oto's explicit audio-checked uninstall (mention driver removal in cask `caveats`). Update the cask version and literal DMG SHA-256 in an idempotent, monotonic post-publish CI job using a dedicated fine-grained PAT / GitHub App token. Test `brew upgrade --greedy` explicitly.
- **Linux:** The manifest's `platforms` map is multi-platform from day 1 (`linux-x86_64-appimage`, `linux-aarch64-appimage`). Package-managed installs (`/usr`, `/nix/store`, `FLATPAK_ID`) classify as `NotifyOnly` (or disabled via build flag) and never self-modify.

## Work

### Phase 1: Decoupled HAL driver versioning & engine compatibility

1. Create `driver/macos/DRIVER_VERSION` (`1`).
2. Update `driver/macos/Makefile` to depend on `DRIVER_VERSION` for both `$(BIN)` and `$(PLIST)`, pass `-DOTO_DRIVER_VERSION=$(DRIVER_VERSION)`, and set `OtoDriverVersion` (integer), `CFBundleVersion`, and `CFBundleShortVersionString` from `DRIVER_VERSION`. Remove `VERSION="$VERSION"` from `scripts/macos/bundle.sh:65,67`.
3. Update `driver/macos/src/OtoDriver.c` to declare `'otvr'` in `kAudioObjectPropertyCustomPropertyInfoList` (`'cust'`) on `kObj_Mic` as `kAudioServerPlugInCustomPropertyDataTypeCFString` and serve an immortal `CFSTR` (zero allocations) in `HasProperty`, `IsPropertySettable`, `GetPropertyDataSize`, and `GetPropertyData`.
4. Add `driver/macos/tools/oto-hal-helper.c` (universal, signed with the same `SIGN_ID`/`SIGN_FLAGS` as the driver and identifier `"dev.oto.hal-helper"`—working in both Developer ID and ad-hoc local builds—bundled at `Oto.app/Contents/Helpers/oto-hal-helper` in `bundle.sh`) providing: kernel `flock` on persistent `/Library/Audio/Plug-Ins/HAL/.OtoMicrophone.driver.lock`, atomic `renamex_np(STAGE, DEST, RENAME_SWAP)` upgrade with `.swap-<nonce>` / `.removing-<nonce>` naming (treating legacy `.old-*` as ambiguous), and `--check-audio-idle` (all-device `kAudioDevicePropertyDeviceIsRunningSomewhere`).
5. Add a property round-trip check to `driver/macos/tools/oto-verify.c` and verify on macOS against a loaded driver.
6. Add `crates/oto-engine/build.rs` (reading `../../driver/macos/DRIVER_VERSION` with `cargo:rerun-if-changed`), `DriverStatus` (`Absent`, `Loaded(u32)`, `Unknown`), `Engine::driver_status()`, `Engine::has_instance_lock()`, `EngineOptions::update_candidate`, `Engine::commit_candidate()`, `BUNDLED_DRIVER_VERSION`, and `MIN_COMPATIBLE_DRIVER_VERSION` in `crates/oto-engine`:
   - Preserve `OSStatus` in UID lookup (`Result<Option<AudioObjectID>, OSStatus>`), `default_input_uid()` (`Result<Option<String>, String>`), and activity checks (`Result<Option<bool>, OSStatus>`), detect legacy `0` via `AudioObjectHasProperty == false` with both UIDs present, `CFRelease` returned `CFStringRef`s, and add `driver_status()` to the `Platform` trait (updating fakes in `control.rs` and `fault_check.rs`).
   - Enforce `rev >= MIN_COMPATIBLE_DRIVER_VERSION` in `control.rs` and `set_default_input_to_oto` (skipped when `mic_uid`/`sink_uid` overrides are set), retry `Unknown` on `DevicesChanged`, and do not tear down running streams on transient `Unknown`.
   - Update ALL `RestoreTarget` methods (`open`, `switch_to_oto`, `restore`, `default_changed` in `crates/oto-engine/src/state.rs:84-166`, changing `restore`/`default_changed`'s closure to `FnOnce() -> Result<Option<String>, String>`) to preserve the saved target on `Err`.
   - Implement `candidate_inert` mode in `Engine::start_with` + `control.rs` (requiring a real `InstanceLock`, recording holder PID, deferring `RestoreTarget`, and refusing to open streams until `Engine::commit_candidate()`).
7. Update `crates/oto-app/src/driver.rs`, `state.rs`, `view.rs`, `main.rs`, and `scripts/macos/{install,uninstall}-driver.sh`:
   - Tighten `driver::requirement()` (`driver.rs:117`, updating test at `driver.rs:471`) with Developer ID Application OIDs (`certificate 1[field.1.2.840.113635.100.6.2.6] exists and certificate leaf[field.1.2.840.113635.100.6.1.13] exists`); make `validate_with` `pub(crate)`.
   - Read on-disk `OtoDriverVersion` (legacy `Info.plist` without the key → `0`) into `Probe`.
   - Wire `INSTALL_BODY`, `UNINSTALL_BODY`, `install-driver.sh`, and `uninstall-driver.sh` through root-staged, `codesign`-verified `oto-hal-helper` (kernel `flock`, destination revision check, atomic `renamex_np`, post-auth non-root `--check-audio-idle` via `launchctl asuser` / `sudo -u "#$INVOKING_UID"` under 3 s timeout, deferring to `OTO_RESULT=reboot` if audio is active/uncertain).
   - Wait for `DriverStatus::Loaded(BUNDLED_DRIVER_VERSION)` in `driver::install()` (`driver.rs:345`, skipping wait on `OTO_RESULT=reboot`), and explicit `DriverStatus::Absent` in `driver::uninstall()` (`driver.rs:380`).
   - Implement the complete driver state table, serialize with app updates, and show *"Asks for your password. Audio will pause briefly in all apps (or finish on your next restart if audio is playing)."* + `driver-update-available` snapshot scene.
8. Add a CI check verifying `DRIVER_VERSION` was bumped whenever `driver/macos/src/**`, `driver/macos/Info.plist`, or `driver/macos/Makefile` changed since the last tag.

### Phase 2: Atomic, monotonic CI release pipeline & signed manifest envelope

1. Add a shared/workspace signing & extraction tool (`crates/oto-sign`, `publish = false`, `release = false` in `release-plz.toml`) using `ed25519-dalek` and the bounded Rust ZIP extractor so CI and `oto-app` share the exact same envelope format and extraction code.
2. Generate the initial Ed25519 keypair offline; create GitHub Environment `release` restricted to protected branch `main` holding `OTO_UPDATE_SIGNING_KEY` and Apple signing/notarization secrets; enable GitHub **Immutable Releases** and a **Tag Ruleset** protecting `v*` tags from force-update/deletion. Commit the public key + `key_id` in `crates/oto-app/src/updater/keys.rs` and document setup, key backup, 6-month/3-release dual-key rotation, and recovery runbooks in `packaging/macos/README.md`.
3. Set `git_release_draft = true` on `oto-app` in `release-plz.toml`. Verify `release-plz/action` (`v0.5.148` / CLI `0.4.3`) emits `tag` for drafts on a test run.
4. Split PR verification into `.github/workflows/macos-pr-check.yml` (read-only, no secrets, ad-hoc `bundle.sh` + zip extraction check) and remove `pull_request` and the top-level `concurrency` block from `.github/workflows/macos-release.yml`.
5. Update `scripts/macos/release.sh` to:
   - Package stapled `Oto.app` into `target/dist/Oto-<version>-macos.zip` (`ditto -c -k --keepParent --norsrc --noextattr`) + `.sha256`.
   - Round-trip extract with `oto-sign extract-verify` and verify `codesign --verify --deep --strict`, embedded driver & `oto-hal-helper` signatures, `spctl --assess --type execute`, `xcrun stapler validate`, `lipo -archs`, `LSMinimumSystemVersion`, and `OtoDriverVersion` consistency.
   - Emit unsigned `target/dist/update-payload.json`.
6. Update `.github/workflows/macos-release.yml` with the 4-job graph (`validate-tag` → `check-release` → `dmg` → `publish`):
   - `validate-tag` (`contents: read`, no secrets, no `environment`): `fetch-depth: 0`, fetch `origin/main` + `refs/tags/$TAG`, enforce `git merge-base --is-ancestor "$TAG_SHA" "$MAIN_SHA"`, verify `Cargo.toml` at `$TAG_SHA`, output `sha`.
   - `check-release` (`contents: write`, NO `environment`, no signing secrets): query `is_draft`.
   - `dmg` (`if: needs.check-release.outputs.is_draft == 'true'`, `environment: release`, per-tag concurrency): check out exact `sha`, run `release.sh`, upload artifacts.
   - `publish` (`if: always() && needs.validate-tag.result == 'success' && needs.check-release.result == 'success' && (needs.dmg.result == 'success' || needs.dmg.result == 'skipped')`, `environment: release`, global `oto-release-publish` concurrency with `cancel-in-progress: false`): **under the global lock**, re-verify remote `refs/tags/$TAG` still points to `sha` and **re-read `isDraft`**; if draft, require `needs.dmg.result == 'success'`, sign, upload, verify assets, publish monotonically (`--latest` iff newest), and reconcile Homebrew tap; if already published, verify published envelope/assets and run idempotent monotonic Homebrew tap reconciliation using the verified DMG digest.

### Phase 3: Updater core, transport & recoverable swap (`crates/oto-app/src/updater/`)

1. Update `crates/oto-app/Cargo.toml`: enable `NSURLSession` / `NSProcessInfo` features on `objc2-foundation`, enable `NSWorkspace` (which also exports `NSWorkspaceOpenConfiguration` in 0.3.2) and `NSRunningApplication` on `objc2-app-kit`, link `CoreServices` (for `extern "C"` `LSRegisterURL`), declare/verify `renamex_np`/`RENAME_SWAP`/`clonefile`, and add `ed25519-dalek`, `sha2`, `serde_json`, `base64`, and minimal `zip` crate. Measure and record universal binary size delta.
2. Implement `http.rs`: `Transport` trait + macOS ephemeral `NSURLSession` (`HTTPCookieStorage = None`, `URLCredentialStorage = None`, per-hop HTTPS redirect delegate, incremental byte-cap enforcement, progress callback, cancellation).
3. Implement `verify.rs`: signed envelope parsing + multi-key Ed25519 verification (`keys.rs`, neutral copy + fixed releases URL on unrecognized key), manifest validation, strict URL/`notes_url` allowlists, `min_os` check, strict semver `new > current`, single bounded Rust ZIP extractor + cache/target disk-space checks, `Info.plist` check, Developer ID `codesign` on app + `driver::validate_with` on driver, and `spctl --assess --type execute` on the `.app`-named cache copy.
4. Implement `apply.rs` + `oto recover-update` in `crates/oto-cli`:
   - Install-mode classifier (`SelfUpdatable`, `NotifyOnly(reason)`, `Unsupported(reason)`), checking `OTO_TEAM_ID`, `Engine::has_instance_lock()`, `login::registrable_bundle`, `statfs` `MNT_RDONLY`, `st_uid == geteuid()`, `access(parent, W_OK)`, `access(bundle, W_OK)`, non-symlink target, and `VOL_CAP_INT_RENAME_SWAP` / `apfs`.
   - Shared admission in `oto-engine`'s `Engine::start_with`: non-candidates check `updating.lock` OR unresolved `update-tx.json` (`swapping`/`verifying`/`rolling_back`) before + after `engine.lock`; candidate requires real `InstanceLock`, records PID, and checks `tx-<nonce>-state == "active"` before + after `engine.lock`.
   - `Request::CommitShutdown(ack_tx)` in `host.rs` (commit-time Oto Mic activity recheck allowing `Ok(None)`/`Ok(Some(false))` and deferring on `Ok(Some(true))`/`Err(_)`, plus acknowledged `RestoreTarget` teardown; restarts host on restore failure or `EPIPE` on commit byte; idempotent `State::shutdown()`).
   - Detached helper (`--finish-update`) copied to non-purgeable `~/Library/Application Support/oto/update-helper-<nonce>`, launched with `setsid()` in `pre_exec`, logging to `~/Library/Logs/Oto/updater.log`, acquiring `updating.lock` before checking/writing `update-tx.json`, commit-pipe EOF exit wait, bounded retry acquisition of `engine.lock` through `renamex_np(..., RENAME_SWAP)` + `LSRegisterURL` + `tx-<nonce>-state = "active"` + persisted deadlines (`stage1_deadline`, `stage2_deadline`, `commit_deadline`), `NSWorkspace` launch (`createsNewApplicationInstance = true`, `--update-ack <nonce>`) with retained `NSRunningApplication`, single-writer `ack-<nonce>-started` / `ack-<nonce>-ready` in `~/Library/Application Support/oto/` (independent of `DriverStatus`), strict commit order (`committed` → `tx-<nonce>-state = "committed"` → cleanup `staging` → delete `update-tx.json` → release `updating.lock`, leaving `tx-<nonce>-state` for candidate deletion), and complete phase-gated dead-helper / hung-candidate (before AND after `ready`) / rollback-relaunch / independent `--recover-update` recovery.

### Phase 4: Preferences, state machine, UI & tray

1. Extend `Prefs` (`crates/oto-app/src/prefs.rs`, field-by-field loader, writes serialized through `State`): `check_for_updates` (default `true`), `skipped_version`, `last_update_check`.
2. Extend `State` and `Hooks` (`crates/oto-app/src/state.rs`):
   - Single active update operation with generation/cancel token (discarding stale completions).
   - `UpdateState`: `Idle`, `Checking { manual }`, `UpToDate`, `Available(UpdateInfo)`, `Downloading { info, pct }`, `Verifying(UpdateInfo)`, `Ready(StagedUpdate)`, `Applying`, `Failed(String)`.
   - Manual check overrides `skipped_version`; background offline errors return silently to `Idle`; background results never open or focus the popover; cancelled driver password prompts return cleanly to `Install::Idle` via `after_prompt` without showing an error banner.
   - Serialize `Applying` with `Install`; block both while `clients_active == true`.
   - Use `manifest.min_driver_version` / `manifest.driver_version` in the Available banner (*"Requires an audio driver update after restarting (asks for your password)"* vs. *"An audio driver update may be offered separately after restarting"*).
   - Gate `Engine::commit_candidate()`, **all** mutating UI hooks (`save`, `set_login`, `install`, `uninstall`, `request_mic`, `sync_system_mic`), **"Hear the Difference"** (`hear_blocked()` + `start_hear_test()` across both tray and fallback window), and onboarding auto-open while awaiting positive commit (`tx-<nonce>-state == "committed"` or self-recovery); exit immediately with zero side effects if `"revoked"` or missing.
3. Update `view.rs`, `tray.rs`, and `main.rs`:
   - Background check 20 s after launch (if `check_for_updates` and ≥ 24 h since last check), then every 24 h.
   - Main popover: compact `notice()` banner when available/ready with explicit **Update…** / **Restart to Update** / **Download…** and **Skip** / **Dismiss**.
   - Settings page: **Updates** card (`Check for updates automatically` switch, version + check/progress/restart button, **Release Notes** link).
   - Right-click tray menu: `Check for Updates…` (or `Update to vX.Y.Z…` when available).
   - Add headless snapshot scenes: `update-available`, `update-downloading`, `update-ready`, `driver-update-available`.

### Phase 5: Homebrew Cask & end-to-end verification

1. Create/wire `deepso7/homebrew-oto` cask (`auto_updates true`, `uninstall quit: "dev.oto.app"`, no HAL `uninstall delete:`; driver removal noted in `caveats`) updated by an idempotent, monotonic post-publish CI job using a dedicated tap token and literal DMG SHA-256.
2. Automated tests:
   - `verify.rs`: bad signature, unrecognized key (neutral copy + fixed URL), dual-key rotation, tampered payload, downgraded/equal version, mismatched plist version, disallowed URL/redirect/`notes_url`, `min_os` too high, zip-slip / case-folded duplicate / symlink / special-file / zip-bomb / missing `Oto.app`.
   - `apply.rs`: `O_CLOEXEC` check on `engine.lock`, check-then-recheck `updating.lock` + unresolved `update-tx.json` admission across `oto-app` and `oto-engine` callers, candidate real-lock requirement + post-lock `tx-<nonce>-state` recheck, bounded retry lock acquisition under brief `try_lock` contention, commit-pipe abort (0 bytes → no swap) and `EPIPE` engine restart, existing stale `update-tx.json` check before starting a new transaction, no-driver vs error commit-time audio check, `clonefile` fallback on `EXDEV`/`ENOTSUP`, `renamex_np(RENAME_SWAP)` fresh swap, single-writer `tx-<nonce>-state` positive commit vs `"revoked"` / missing abort, client connecting between `started` and `ready` (inert engine opens zero streams until `commit_candidate()`), "Hear the Difference" blocked before commit in both tray and fallback-window paths, hung-candidate rollback both before `ready` (`> stage2_deadline`) and after `ready` (`> commit_deadline`) using persisted deadlines, helper `SIGKILL`ed at every boundary (`prepared` while old app holds `engine.lock`, `swapping` pre/post-exchange, `verifying`, `rolling_back` pre/post-exchange ensuring relaunch, `committed`, `rolled_back`), pre-`main` candidate crash + dead helper recovered via `update-helper-<nonce> --recover-update` and `oto recover-update`, stuck-candidate self-commit, healthy ack when `DriverStatus` is incompatible.
   - `driver` & `engine`: `oto-verify` `'otvr'` round-trip, complete `DriverStatus` state table, engine refusing streams below `MIN_COMPATIBLE_DRIVER_VERSION`, `RestoreTarget` (`open`, `switch_to_oto`, `restore`, `default_changed`) preserving target on query failure, `oto-hal-helper` kernel `flock` + `renamex_np` + `.swap-*`/`.removing-*` isolation + ambiguous legacy `.old-*` non-restoration.
   - `state.rs` + `view.rs`: cancellation/stale-worker discard, cancelled password prompt handling, active-call blocking, all snapshot scenes.
3. Manual E2E verification on clean macOS accounts (Apple silicon & Intel, Finder-launched + LaunchAgent-started):
   - Exported Developer ID `v0.X.0` → grant mic permission → install driver → start from LaunchAgent → test direct, PAC/proxy, and offline checks → verify restart blocked mid-call → end call → **Restart to Update** → confirm helper survives LaunchAgent exit, candidate acks Stage 1 + Stage 2, TCC mic permission is **not** re-prompted, `start_at_login` and `system_microphone` keep working, and zero driver prompts appear when `DRIVER_VERSION` is unchanged.
   - Test bumped `DRIVER_VERSION` → verify post-update driver banner, all-device post-auth non-root audio check + reboot deferral when audio is playing, and loaded-revision confirmation after restart.
   - Test non-owner user in `/Applications` (standard/second account), mounted DMG / translocated launch, `brew upgrade --greedy`, and macOS 13+ App Management behavior for the copied bare helper binary.

## Done when

- Merging a `release-plz` PR validates tag ancestry against fetched `origin/main` in a secretless `contents: read` job, checks draft state without signing secrets, builds/notarizes/round-trip-verifies DMG + zip + signed `update.json` from that exact SHA inside the `main`-protected `release` environment, re-verifies tag SHA and draft state under the global publish lock, verifies uploaded assets, publishes monotonically with `--latest` when newest, and reconciles the Homebrew tap idempotently (with Immutable Releases and `v*` tag rulesets enabled).
- A running Developer ID `Oto.app` owned by the current user in a writable location checks silently, downloads only on explicit click, verifies Ed25519 + SHA-256 + single bounded Rust ZIP extraction + Developer ID `codesign` + Gatekeeper + semver, shuts down cleanly with acknowledged default-mic restoration, swaps atomically via `renamex_np` under held `updating.lock` + `engine.lock`, keeps the candidate inert (no streams, no `RestoreTarget`, no mutating UI hooks, no "Hear the Difference") until durable positive commit, automatically rolls back and relaunches on any candidate crash/hang while the helper is alive, and reconciles cleanly on the next launch (or immediately via `update-helper-<nonce> --recover-update` / `oto recover-update`) if the helper itself is `SIGKILL`ed at any phase (including between commit writes or before relaunch).
- Updating from a LaunchAgent-started `Oto.app` completes cleanly, preserving TCC microphone permission, preferences, and login registration.
- Updating when `DRIVER_VERSION` is unchanged requires zero admin prompts and zero `coreaudiod` restarts; when `DRIVER_VERSION` bumps, `oto-engine` enforces compatibility and `oto-app` offers an explicit, audio-checked driver upgrade (kernel-locked, atomic `renamex_np`, running post-auth checks as the invoking user with reboot deferral if audio is active/uncertain) verified by loaded revision.
- Non-owner, protected, translocated, lockless, and read-only installs gracefully present notify-only download actions.
- All automated tests, `oto-verify` property tests, and Rune snapshot scenes pass, and measured binary size delta is recorded in [Status](../status.md).

## Risks

| Risk | Mitigation |
|---|---|
| `kAudioObjectPropertyCustomPropertyInfoList` + `CFStringRef` `'otvr'` behaves differently across macOS 12–27 | Prototype and test via `oto-verify` on target macOS versions first in Phase 1 before wiring engine gating |
| macOS 13+ App Management prompt appears or blocks in-place `/Applications/Oto.app` swap | The copied helper runs as a bare same-Team-ID signed binary outside a bundle after its parent exits; test exported Developer ID builds empirically on clean macOS 13+ accounts |
| LaunchAgent kills detached helper despite `setsid()` | Test updating from a real LaunchAgent-started instance; if needed, also add `AbandonProcessGroup` to `login.rs` |
| `NSURLSession` delegate boilerplate in `objc2-foundation` is subtle | Keep `Transport` tiny (`get_bytes` + `download_file`) and test redirects, cancellation, offline, and PAC proxy behavior directly |
| Second logged-in user runs a shared `/Applications/Oto.app` while its owner updates it | Non-owners classify as `NotifyOnly` (never mutating the shared bundle); atomic `renamex_np` keeps the second user's running instance alive on its existing inode until restarted |
| Residual race between post-auth audio check and `killall coreaudiod` | Recheck all audio devices as the invoking user immediately before `killall`, defer to reboot on active/uncertain audio (including always-on Bluetooth/browser streams), warn clearly, and require explicit click |
| `release-plz`, publish queue, or tap push fails partway | Document and test `workflow_dispatch --ref main -f tag=vX.Y.Z` recovery for draft-present, tag-only, displaced-queue, and published-tap-retry states in `packaging/macos/README.md` |
| Signing key rotation or compromise | Back up `OTO_UPDATE_SIGNING_KEY` offline; dual-sign with `signatures[]` for ≥ 6 months / 3 releases; neutral manual-download fallback on unrecognized keys; Apple Developer ID + Team ID + Gatekeeper remain an independent trust anchor |
| `brew upgrade --greedy` replaces a running app outside `updating.lock` | Set `auto_updates true` and `uninstall quit: "dev.oto.app"` (never `uninstall delete:` on the shared HAL driver), and test `--greedy` explicitly |

## Out of scope

- In-place privileged or cross-user replacement for `/Applications/Oto.app` owned by `root` or another account (v1 uses notify-only Download…; revisit only if users need it).
- Delta / binary-diff updates (full zipped app is ~22–30 MB).
- Silent background binary downloads or automatic restarts without user confirmation.
- Beta / prerelease update channels (single stable channel for v1).
- Linux AppImage self-replacement (manifest supports it; implementation belongs with Phase 1 Track B Linux packaging).
- Windows updater.
