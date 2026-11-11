# Status: what exists today

[← Docs index](README.md)

- **`oto-dsp`**: the v1 chain (80 Hz high-pass → denoise → −1 dBFS limiter). Non-finite input is zeroed at the chain's entry. Two models ([plan 0001](plans/0001-deepfilternet3-default.md)): **DeepFilterNet3** is the default (40 ms chain latency, 6.7% of an M2 Pro core in `oto bench`) and the original **RNNoise** (2018 weights, through `nnnoiseless`) runs in Low CPU mode (20 ms, 0.44%). DeepFilterNet3 is the `dfn3` cargo feature, on by default; `--no-default-features` builds an RNNoise-only app with no `tract`. Debug builds load it too: `tract-*` are built without debug assertions (one of them rejects the model's graph) and the model crates with `opt-level = 3`.
- **Realtime path**: RNNoise runs inside the audio callback, which does not allocate after its first callback (`no_alloc` tests in `oto-dsp` and `oto-engine`). DeepFilterNet3 allocates about 1,700 times per frame inside `tract`, so it runs on a realtime worker thread fed through two rings with a 7.3 ms reserve; the callback side is allocation-free and falls back to the dry signal if the worker is late. If DeepFilterNet3 cannot load, the engine runs RNNoise, says so once and does not retry until the mode changes.
- **DeepFilterNet3 weights licence**: cleared. Upstream commit `7b14b89` (closing issue #700) states that code and weights are MIT/Apache-2.0.
- **`driver/macos`**: the C HAL plug-in. A visible "Oto Microphone" and a hidden "Oto Sink" share one clock; the sink loops to the mic through time-tagged slots, so it plays silence (never stale audio) when nothing is playing. Signing comes from `SIGN_ID` in the environment and defaults to ad-hoc; `make release` refuses anything but a Developer ID identity.
- **`oto-engine`**: the Core Audio backend behind a `Platform` trait (Linux PipeWire comes next). It captures the chosen mic, runs the chain, and plays into Oto Sink with fill-driven `rubato` drift control. It runs only while an app records from Oto Microphone, handles hot-plug and default-device changes, allows one engine at a time, and remembers the previous default input on disk so a crash cannot strand the system on a silent microphone. Inputs with more than two channels use the first channel.
- **More models** ([models.md](models.md)): DPDFNet-2 (Max tier) and FastEnhancer-S (Fast tier) through ONNX Runtime, behind the `onnx` feature (on by default). The app's Model picker replaces the Low CPU switch. Apple silicon only (no prebuilt x86_64 ONNX Runtime). FastEnhancer's weights licence is not cleared: evaluation only, left out of releases.
- **`oto-cli`**: `oto run | devices | status | latency | bench`.
- **`oto-app`**: Rune menu-bar app with a status item via `tray-icon`, a popover, strength and device controls, meters and settings. The status line shows the running model and its latency. Onboarding has three real steps: install the driver, allow the microphone, and optionally use Oto as the system microphone. The popover has a Model picker (DeepFilterNet3, DPDFNet-2, FastEnhancer-S, RNNoise), shown only when the build has more than one model; a switch during a call fades out for 10 ms and reloads the model before stopping the streams.
- **Driver install from the app**: the app installs only the prebuilt driver inside its own bundle (or the repo build in debug builds), checks its signature, and runs a fixed command sequence as root. It never runs a script or a build as root. Settings can uninstall it.
- **App bundle**: `scripts/macos/bundle.sh` builds `target/bundle/Oto.app` with the microphone usage description, the driver and `THIRD_PARTY_NOTICES.md` inside. 38 MB unpacked, 22.5 MB zipped (20 MB before DeepFilterNet3); a clean release build of the app takes 231 s (160 s without `dfn3`). Start at login registers the bundle and is unavailable when running outside one.

Measured on an M2 Pro running macOS 27.0.1, 11 Nov 2026:
- **Latency** (`oto latency --no-denoise`, 48 kHz mics): DeepFilterNet3 62.6 ms on a USB mic (Elgato Wave:1) and 60.8 ms on the built-in mic; Low CPU mode 35.4 ms and 34.3 ms.
- **Noise removed**: a pink-noise-and-hum clip through `oto run --input-raw` at Balanced drops from about −6 to −30 dBFS with DeepFilterNet3 and to −14 dBFS with RNNoise.
- **CPU in a call**: 3% of a core with DeepFilterNet3 (worker p50 0.2 ms, p99 1.3 ms per 5.3 ms block), 0.6% in Low CPU mode. 0% when idle.
- **Soak**: 30 minutes of DeepFilterNet3 with a forced client: no underruns, resyncs or starvations; worker max 2.0 ms. A first run that overlapped a full workspace build had one underrun and no late callbacks.

Not done yet:
- A published release. `scripts/macos/release.sh` builds a universal (arm64 + x86_64), Developer ID signed, notarized and stapled app inside a notarized DMG (0.0.1: 48 MB, accepted by Apple and by `spctl` on 11 Nov 2026), but the DMG hasn't been installed on a Mac that never ran Oto. See `packaging/macos/README.md`.
- A real install, the admin prompt and the microphone prompt have not been exercised from the bundled app.
- "Hear the difference" test, and a listening comparison of the two modes with a real TV recording.
- Which apps are using Oto Microphone (only "in use" is shown).
- Live checks for unplug, sleep and Bluetooth mics.
- Shortcut, updater, Linux.

```bash
cargo test
make -C driver/macos && scripts/macos/install-driver.sh
cargo run --release -p oto-cli -- run
scripts/macos/bundle.sh && open target/bundle/Oto.app
```
