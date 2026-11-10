# Roadmap

[← Docs index](README.md)

## Phases

Each phase ends with something that can be shown and a criterion that can fail.

**Phase 0 — spikes (de-risk before building).** Throwaway code is fine.

| Spike | Passes when |
|---|---|
| Linux passthrough virtual mic with `pipewire` | "Oto Microphone" is selectable and audible in Chromium, Firefox, Discord and GNOME/KDE settings |
| macOS NullAudio loopback, locally signed | Same in Zoom, Meet (Chrome/Safari), Discord; hidden sink stays hidden; latency and drift measured over an hour |
| Model bake-off: DeepFilterNet3 (and Plus fork), FastEnhancer-S, DPDFNet-2, RNNoise 0.2 | Per-frame p50/p99/max under realtime pacing on an M-series Mac and an older x86 laptop; quality scores from the harness; a default is chosen |
| Rune shell | Menu-bar icon plus popover on macOS, tray plus window on Linux, meter at 60 fps from another thread |

The bake-off harness: `oto bench` for timing in Rust; `tools/eval` (VERSA: PESQ, DNSMOS, SI-SDR, plus Whisper word error rate to catch eaten speech) for quality. Measure true latency by sending an impulse through the real virtual-mic path.

**Phase 1 — Linux alpha.** Engine with capture, chosen model, PipeWire backend, device hot-plug and suspend handling, idle-at-zero. `oto` CLI runs it headless. Passes when it survives a day of calls and unplugging the mic.

**Phase 2 — UI.** Rune app: tray, switch, device picker, strength, meter, test recording, start at login. Passes when a new user gets a clean mic in a call without reading anything.

**Phase 3 — macOS.** HAL driver, drift control, onboarding with driver install, signed and notarized installer, clean uninstall. Passes on clean macOS 26 and 27 machines, Apple silicon and Intel.

**Phase 4 — release.** Linux packages (deb, rpm/COPR, AUR, AppImage, Nix; Flatpak second, since it needs a flagged PipeWire socket permission), macOS installer and Homebrew cask, auto-update, published CPU and latency measurements, docs.

**After v1.** Voice shaping presets → virtual speaker → Max tier download → own model training for lower latency and background voices → Windows.

## Risks

| Risk | Mitigation |
|---|---|
| Default model misses the 50 ms target | Decided in phase 0 with measurements; FastEnhancer retrain as the fallback; shared-memory driver path saves about 10 ms on macOS |
| macOS driver loading or signing breaks on a new OS release (a BlackHole signature failure on 26.5 is unexplained) | Test on betas; keep the driver tiny and close to Apple's sample; reboot fallback if coreaudiod restart fails |
| Rune API churn or a blocking bug | Pin revisions; keep UI thin and all logic in `oto-engine`, so swapping the UI layer stays a contained job. Rune takes no pull requests, so we cannot upstream fixes |
| Rune + third-party tray on macOS do not coexist | Spike in phase 0; fall back to raw `NSStatusItem` on Rune's main thread |
| DeepFilterNet upstream is unmaintained | Pinned to `tract` 0.21.4 and kept off the audio thread on a worker (plan 0001); vendor it if the pin becomes a problem |
| "Oto" is hard to search for and shared with other apps (a tinnitus app, an iOS voice companion) | Name is settled as Oto; check trademarks before a public launch |

## Decisions

Settled 2026-11-10, except where a later date is given.

1. **Apple Developer Program:** already have a membership, so the driver can be Developer ID signed and notarized.
2. **Licence:** `MIT OR Apache-2.0`.
3. **Open source and free to use.** No paid tier. Models still need weights we are allowed to redistribute under a permissive licence, so the DeepFilterNet3 weights-licence check and FastEnhancer's unstated weights licence remain phase 0 checks; being free does not waive them.
4. **Name:** Oto. The virtual device is "Oto Microphone".
5. **Default model (2026-11-10):** DeepFilterNet3 is the default and RNNoise is the light option, decided by ear after RNNoise let a background TV through. The measured bake-off is deferred. See [plan 0001](plans/0001-deepfilternet3-default.md).
