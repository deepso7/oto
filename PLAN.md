# Oto: plan

Oto is a lightweight, realtime noise-cancelling virtual microphone for Linux and macOS (Windows later), written in Rust with a [Rune](https://github.com/anomalyco/rune) UI. Any app that can pick a microphone picks "Oto Microphone" and gets a clean voice.

Status: macOS v1 path built (driver, engine, CLI, menu-bar app); Linux backend next. Research date: 2026-11-10.

## 1. Product

**What it does in v1**

- Removes background noise from the mic, system-wide, in every app.
- One switch, in the menu bar or tray. Strength is Light / Balanced / Strong.
- Costs nothing when idle: no inference unless an app is actually recording from Oto Microphone.
- A 10-second "hear the difference" test that plays your voice back raw and cleaned.

**Later, in order**

1. Voice shaping: Warm / Neutral / Bright presets, then manual EQ.
2. Virtual speaker, to clean the audio you hear from others.
3. Background-voice removal (other people talking near you).
4. Windows.

**Why anyone would use it**

- No one ships a polished app for both Linux and macOS. The closest projects are single-OS and add about 80 ms of delay (HushMic on Linux, Stillmic on macOS). NoiseTorch has been broken on PipeWire 1.6.3 since April 2026.
- Krisp has no Linux build, no free plan, and a 250 MB+ installer.
- On macOS the real competitor is Apple's Voice Isolation. It is per-app, cannot be tuned, and is missing in many apps (Audacity, DAWs, Firefox, some Electron apps). Oto's pitch there is "works in every app, tunable, plus voice shaping", not "denoises better than Apple".

**Targets we hold ourselves to**

| Measure | Target |
|---|---|
| CPU, no app recording | 0% |
| CPU in a call, default model, 2020+ laptop | under 10% of one core |
| Added latency, end to end | 50 ms or less (stretch: 35 ms) |
| RAM | under 50 MB, excluding the UI's GPU surfaces |
| Download | under 25 MB |
| Network use | update checks only; no account, no telemetry by default |

The latency target is the hard one; see section 4.

## 2. Architecture

```
            ┌──────────────── oto-app (Rune UI) ───────────────┐
            │ tray/menu bar · popover · onboarding · settings  │
            └──────┬─────────────────────────────▲─────────────┘
          Command  │                             │ Event, Live<f32> levels
            ┌──────▼─────────────────────────────┴─────────────┐
            │                   oto-engine                      │
            │ device watch · state machine · drift control      │
            │                                                   │
 real mic ──► capture ─► resample ─► oto-dsp chain ─► virtual mic backend ──► Zoom, Meet,
            │            to 48k      HPF → denoise →   (per OS, below)    │    Discord, OBS…
            │                        shaping → limiter                    │
            └───────────────────────────────────────────────────┘
```

**One process, three kinds of thread.** The realtime audio thread runs the DSP chain and never allocates, locks or blocks. A control thread owns devices and reacts to hot-plug, sleep and default-device changes. The UI thread is Rune's. They talk through lock-free ring buffers (`rtrb`) for audio and channels for control.

**The engine is a library with a message boundary.** The UI and the `oto` CLI only send `Command`s and receive `Event`s. That keeps a headless mode cheap now (useful on GNOME, where tray icons need an extension) and lets the engine move into its own daemon later without touching callers. We do not build the daemon in v1.

**Workspace**

| Path | Role |
|---|---|
| `crates/oto-dsp` | Signal chain. `Denoiser` trait, frame adapter, filters, model backends behind features. No device code. |
| `crates/oto-engine` | Capture, virtual mic backends, drift control, device watching, `Command`/`Event`. |
| `crates/oto-app` | Rune UI. |
| `crates/oto-cli` | `oto` binary: headless run, device list, `bench`, `latency`. |
| `driver/macos` | The CoreAudio HAL plug-in. |
| `tools/eval` | Python quality harness (not shipped). |

## 3. Virtual microphone, per OS

### Linux: PipeWire, in-process

One PipeWire client with two streams, using the `pipewire` crate (0.10.x) directly:

- A capture stream pinned to the physical mic with `target.object`, so making Oto the default input cannot create a feedback loop.
- An output-direction stream with `media.class=Audio/Source`, `node.description="Oto Microphone"`, `node.virtual=true`. This is the documented way to present a virtual mic.
- `node.passive` on capture, so the real mic is released when nobody is recording.
- Both streams share the graph clock, so no drift correction is needed.
- The graph quantum can change at runtime; `FrameAdapter` in `oto-dsp` already absorbs that.

No root, no `setcap`, no config files to edit. PulseAudio-only and ALSA-only systems are out of scope: Ubuntu, Fedora and Debian all default to PipeWire.

`pipewire` types are not `Send`, so the PipeWire loop gets its own thread.

### macOS: HAL plug-in as a loopback

The only API that can present a virtual input device is an AudioServerPlugIn in `/Library/Audio/Plug-Ins/HAL`. DriverKit is not permitted for virtual devices and Core Audio taps only capture output.

- **v1 design: loopback, no custom IPC.** The driver exposes a visible "Oto Microphone" and a hidden sink. The app captures the real mic, processes it, and plays into the hidden sink as an ordinary Core Audio client; the driver loops that to the microphone side. A known implementation of this pattern measures about 10.7 ms of loopback delay.
- **Driver language.** The driver holds no DSP, only a ring buffer, so start from Apple's NullAudio sample in C (about 1.5k lines). A pure-Rust driver is proven possible but buys little here. Revisit if the C becomes a burden.
- **Clock drift.** The mic and the virtual device run on different clocks. Keep a small reserve in the ring and correct with an adaptive resampler (`rubato`) driven by fill level.
- **When the app is not running**, the device outputs silence, never stale audio.
- **Later optimisation:** replace the loopback hop with a shared-memory ring through a Mach service declared in the driver's Info.plist. Lower latency, more moving parts. Only if the latency budget demands it.

Install needs an admin prompt and a coreaudiod restart, which interrupts all audio for a moment. The driver ships Developer ID signed inside a notarized installer. Only the Oto app needs microphone permission.

### Windows: later

A virtual mic needs a kernel-mode audio driver (SysVAD-derived, C++) signed through a Partner Center account with an EV certificate, which requires a legal entity. Several open projects stalled exactly there. The interim option is to support an existing virtual cable such as VB-Cable. Nothing in v1 should make this harder: the engine's backend trait is the only seam needed.

## 4. Models

All run on one CPU thread. No GPU: it costs battery and wakes hardware for a job a single core can do.

| Tier | Model | Rate | Model delay | Cost | Licence | Rust path |
|---|---|---|---|---|---|---|
| Standard (default) | DeepFilterNet3 | 48 kHz | 40 ms | about 3–8% of a recent core; 19% of a 2017 U-series core | MIT/Apache, weights included | native `libDF` + `tract`, vendored |
| Light | RNNoise 0.2 | 48 kHz | 10 ms frames | about 3% of a core | BSD-3 | C library or an audited port |
| Max | DPDFNet-2 / -8 (48 kHz) | 48 kHz | unpublished | 4–20× DeepFilterNet3's compute | Apache-2.0 | ONNX via `ort` |
| Challenger | FastEnhancer-S (48 kHz) | 48 kHz | about 21 ms | about 4% of a core (M5) | MIT code; weights licence unstated | ONNX, or hand-written inference |

**Why DeepFilterNet3 first.** It is the only fullband model with a native Rust streaming implementation, clearly permissive weights, and years of field use. Two real problems:

- **Latency.** 40 ms of model delay plus one 10 ms frame of buffering plus device I/O lands around 55–65 ms, over our 50 ms target. Phase 0 must either find a lower-lookahead configuration that still sounds good, or the default becomes FastEnhancer.
- **Maintenance.** Upstream is near-dormant and pinned to an old `tract`. We vendor `libDF` and own it. Compare against the DeepFilterNetPlus fork, which fixes NaN-on-silence and per-frame allocations.

**FastEnhancer is the one to beat it.** Half the delay and less CPU, but no published 48 kHz quality numbers, an unstated weights licence, and training data that includes export-restricted corpora. Shipping it means retraining it ourselves on clean data.

**Second sweep for 2026 releases (10 Nov 2026).** Nothing new displaces the table above. What changed:

- **FastEnhancer now has first-party 48 kHz weights** in five sizes (Tiny to Large), plus a dependency-free Rust port, `faster_enhancer_rs` (x86-64 only, published Oct 2026). Tiny (28k parameters, about 1% of an M5 core) also becomes a challenger for the Light tier. Quality at 48 kHz is still unpublished and the training-data problem stands, so the bake-off has to measure it and shipping still means retraining.
- **No successor to DeepFilterNet3 or RNNoise 0.2.** DPDFNet is at v0.6.0 with the same top model.
- **One licence flag to clear before committing:** a third-party repo claims the DeepFilterNet3 weights are non-commercial, against the upstream MIT/Apache labelling. Read the upstream licence files and issue #700 ourselves in phase 0.
- **Not usable:** several 2026 papers (CoFi-Lite, TF-Refiner, HALO) have no released weights; LaCo-SENet and RT-SEMamba are 16 kHz with missing or ambiguous licences; Clearline's licence is unset because of non-commercial training data.

Bake-off list: DeepFilterNet3 (and the Plus fork), FastEnhancer 48 kHz Tiny / Base / Small, DPDFNet-2 48 kHz, RNNoise 0.2.

**Runtime.** Prefer `tract` (pure Rust, no native library, small). Use `ort` only for the Max tier, where it adds roughly 4–18 MB of native runtime; make that tier an optional download so the base install stays small.

**Strength** is an attenuation limit on one model, not three models.

**Background-voice removal** needs our own training. The open starting point is Hush (16 kHz, DeepFilterNet3 with a separation head used in training); the path is applying that recipe to a 48 kHz model. Usable data: DNS5, LibriTTS, MLS, Common Voice, MUSAN, WHAM. Avoid or check: ESC-50 (non-commercial), FSD50K, AudioSet, DEMAND. This is a post-v1 project of its own.

Two newer open models are worth measuring when we get there, both 16 kHz and both with self-reported numbers only: **alkove Focus** (1.1 MB, Apache-2.0 code and weights, Rust engine, removes noise and other voices; latency and CPU unstated) and **VoxLock** (Apache-2.0, claims to beat Krisp on its author's benchmark, but uses about a full CPU core, so it is a quality ceiling rather than something to ship).

## 5. Signal chain

```
high-pass (80 Hz) → denoise → [gate] → [EQ] → [de-esser] → [compressor] → [auto-gain] → limiter (−1 dBFS)
```

Bracketed stages are the voice-shaping roadmap and are off in v1. "Warm / Bright" presets are a low shelf near 150 Hz (±3–4 dB), a presence peak at 3–5 kHz and a high shelf at 8–10 kHz. Filters come from `fundsp` or hand-written biquads; there is no Rust de-esser crate, so build one from a band-pass sidechain and a compressor. Pitch and formant shifting add about 43 ms and stay optional.

Skip inference on digital silence, and stop the whole chain when the virtual mic has no listeners.

## 6. UI with Rune

Rune is a native GPU-rendered Rust UI framework: winit + wgpu, plain-Rust builder functions, one app-owned state struct. It is three weeks old and pre-release, so we pin a `dev` revision and bump on purpose.

**What fits well**

- `Live<f32>` is a lock-free latest-wins value that redraws without re-running the view. It is made for level meters. Publish from a metering thread fed by a ring buffer, not from the audio callback.
- `AppHandle` lets the engine's control thread update UI state.
- `App::windowless().keep_running().accessory()` gives a background app with no Dock icon.
- Global shortcuts are built in, including Wayland.
- `rune-kit` has Switch, Select, Combobox, Popover, Menu, Tooltip, Toast, SegmentedControl.
- Every app renders headless snapshots (`--snap`), and views are unit-testable.

**Gaps we fill**

| Gap | Fill |
|---|---|
| No macOS menu-bar icon | `tray-icon` crate or `NSStatusItem` via `objc2`. Coexistence with Rune's event loop is untested: spike it. |
| No Slider in `rune-kit` | Hand-roll with `.on_drag`; Rune's examples have three to copy from. |
| Wayland cannot position windows or keep them on top | No tray-anchored popover on Wayland; open a normal small window there. |
| Linux tray menu is flat, GNOME needs an extension | Keep the tray menu to toggle / open / quit. Window and CLI work without a tray. |
| No autostart | `auto-launch` crate. |
| No packaging, notarization, updater | `cargo-packager` or scripts; `sparkle-updater` on macOS; distro packages on Linux. |
| No Linux notifications | `notify-rust`. |

**Screens**

1. **Popover / main window:** big switch, input device picker, strength, live meter (raw vs cleaned), which apps are using Oto Microphone, current CPU cost.
2. **Onboarding:** mic permission → install driver (macOS) → offer to make Oto Microphone the system default → test recording.
3. **Settings:** model tier, start at login, shortcut, updates.

On quit, restore the previous default input.

## 7. Phases

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

## 8. Risks

| Risk | Mitigation |
|---|---|
| Default model misses the 50 ms target | Decided in phase 0 with measurements; FastEnhancer retrain as the fallback; shared-memory driver path saves about 10 ms on macOS |
| macOS driver loading or signing breaks on a new OS release (a BlackHole signature failure on 26.5 is unexplained) | Test on betas; keep the driver tiny and close to Apple's sample; reboot fallback if coreaudiod restart fails |
| Rune API churn or a blocking bug | Pin revisions; keep UI thin and all logic in `oto-engine`, so swapping the UI layer stays a contained job. Rune takes no pull requests, so we cannot upstream fixes |
| Rune + third-party tray on macOS do not coexist | Spike in phase 0; fall back to raw `NSStatusItem` on Rune's main thread |
| DeepFilterNet upstream is unmaintained | Vendor it; it is small and we need to own the realtime path anyway |
| "Oto" is hard to search for and shared with other apps (a tinnitus app, an iOS voice companion) | Name is settled as Oto; check trademarks before a public launch |

## 9. Decisions (settled 2026-11-10)

1. **Apple Developer Program:** already have a membership, so the driver can be Developer ID signed and notarized.
2. **Licence:** `MIT OR Apache-2.0`.
3. **Open source and free to use.** No paid tier. Models still need weights we are allowed to redistribute under a permissive licence, so the DeepFilterNet3 weights-licence check and FastEnhancer's unstated weights licence remain phase 0 checks; being free does not waive them.
4. **Name:** Oto. The virtual device is "Oto Microphone".

## 10. What exists today

- **`oto-dsp`**: the v1 chain (80 Hz high-pass → denoise → −1 dBFS limiter), allocation-free on the audio thread. RNNoise (`nnnoiseless`) is the default model: 20 ms chain latency, about 0.5% of an M2 Pro core. DeepFilterNet3 builds behind the `dfn3` feature (40 ms, 6.5% of a core, allocates per frame) and is bench-only for now (`oto bench --model dfn3`) because it allocates on the audio thread. `oto bench` times either.
- **DeepFilterNet3 weights licence**: cleared. Upstream commit `7b14b89` (closing issue #700) states that code and weights are MIT/Apache-2.0.
- **`driver/macos`**: the C HAL plug-in. A visible "Oto Microphone" and a hidden "Oto Sink" share one clock; the sink loops to the mic through time-tagged slots, so it plays silence (never stale audio) when nothing is playing. `scripts/macos/install-driver.sh` installs it, and `driver/macos/build/oto-verify` checks it.
- **`oto-engine`**: the Core Audio backend behind a `Platform` trait (Linux PipeWire comes next). It captures the chosen mic, runs the chain, and plays into Oto Sink with fill-driven `rubato` drift control. It runs only while an app records from Oto Microphone, and it handles hot-plug and default-device changes.
- **`oto-cli`**: `oto run | devices | status | latency | bench`.
- **`oto-app`**: Rune menu-bar app with a status item via `tray-icon`, a popover, onboarding with driver install, strength and device controls, meters, settings, and start at login.

Measured on an M2 Pro running macOS 27.0.1:
- **Latency**: 35 ms end to end (mic IO buffer → chain → sink → driver → recorder), measured with `oto latency`.
- **CPU**: 0% when idle.

Not done yet:
- Developer ID signing and notarized installer.
- "Hear the difference" test.
- Live checks for unplug and sleep.
- Linux.

```bash
cargo test
make -C driver/macos && scripts/macos/install-driver.sh
cargo run --release -p oto-cli -- run
cargo run --release -p oto-app
```

## Sources

Rune: [repo](https://github.com/anomalyco/rune). Models: [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet), [DPDFNet](https://github.com/ceva-ip/DPDFNet), [FastEnhancer](https://github.com/aask1357/fastenhancer), [RNNoise 0.2](https://github.com/xiph/rnnoise/issues/221), [Hush](https://github.com/pulp-vision/Hush), [VERSA](https://github.com/ftshijt/speech_evaluation). macOS: [WWDC21 on audio drivers](https://developer.apple.com/videos/play/wwdc2021/10190/), [libASPL](https://github.com/gavv/libASPL), [snapcaster](https://github.com/tsirysndr/snapcaster). Linux: [PipeWire streams](https://docs.pipewire.org/page_streams.html), [EasyEffects](https://github.com/wwmm/easyeffects). Prior art: [HushMic](https://github.com/Fovty/hushmic), [Stillmic](https://github.com/robin-liquidium/stillmic), [NoiseTorch](https://github.com/noisetorch/NoiseTorch), [Krisp](https://krisp.ai).

Not verified by the research and worth checking as we go: Krisp's internals; DPDFNet's CPU cost and true latency; whether ad-hoc signed HAL drivers load on macOS 26+; `Audio/Source` versus `Audio/Source/Virtual` visibility across desktops; PipeWire version on Ubuntu 26.04.
