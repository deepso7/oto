# Architecture

[← Docs index](README.md)

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

## Virtual microphone, per OS

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

## Signal chain

```
high-pass (80 Hz) → denoise → [gate] → [EQ] → [de-esser] → [compressor] → [auto-gain] → limiter (−1 dBFS)
```

Bracketed stages are the voice-shaping roadmap and are off in v1. "Warm / Bright" presets are a low shelf near 150 Hz (±3–4 dB), a presence peak at 3–5 kHz and a high shelf at 8–10 kHz. Filters come from `fundsp` or hand-written biquads; there is no Rust de-esser crate, so build one from a band-pass sidechain and a compressor. Pitch and formant shifting add about 43 ms and stay optional.

Skip inference on digital silence, and stop the whole chain when the virtual mic has no listeners.

## UI with Rune

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
