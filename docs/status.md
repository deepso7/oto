# Status: what exists today

[← Docs index](README.md)

- **`oto-dsp`**: the v1 chain (80 Hz high-pass → denoise → −1 dBFS limiter), allocation-free on the audio thread. Non-finite input is zeroed at the chain's entry. The model that ships is the original RNNoise (2018 weights, through `nnnoiseless`), not RNNoise 0.2 and not DeepFilterNet3: 20 ms chain latency, about 0.5% of an M2 Pro core. [Models](models.md) describes the intended tiers. [Plan 0001](plans/0001-deepfilternet3-default.md) makes DeepFilterNet3 the default and keeps RNNoise as the light option.
- **DeepFilterNet3** builds behind the `dfn3` feature (40 ms, 6.5% of a core). With the feature on, Standard and Max use it and Light uses RNNoise. It is off by default because it allocates on the audio thread and fails to load in debug builds.
- **DeepFilterNet3 weights licence**: cleared. Upstream commit `7b14b89` (closing issue #700) states that code and weights are MIT/Apache-2.0.
- **`driver/macos`**: the C HAL plug-in. A visible "Oto Microphone" and a hidden "Oto Sink" share one clock; the sink loops to the mic through time-tagged slots, so it plays silence (never stale audio) when nothing is playing. Signing comes from `SIGN_ID` in the environment and defaults to ad-hoc; `make release` refuses anything but a Developer ID identity.
- **`oto-engine`**: the Core Audio backend behind a `Platform` trait (Linux PipeWire comes next). It captures the chosen mic, runs the chain, and plays into Oto Sink with fill-driven `rubato` drift control. It runs only while an app records from Oto Microphone, handles hot-plug and default-device changes, allows one engine at a time, and remembers the previous default input on disk so a crash cannot strand the system on a silent microphone. Inputs with more than two channels use the first channel.
- **`oto-cli`**: `oto run | devices | status | latency | bench`.
- **`oto-app`**: Rune menu-bar app with a status item via `tray-icon`, a popover, strength and device controls, meters and settings. Onboarding has three real steps: install the driver, allow the microphone, and optionally use Oto as the system microphone. The model picker appears only when the build has more than one model.
- **Driver install from the app**: the app installs only the prebuilt driver inside its own bundle (or the repo build in debug builds), checks its signature, and runs a fixed command sequence as root. It never runs a script or a build as root. Settings can uninstall it.
- **App bundle**: `scripts/macos/bundle.sh` builds `target/bundle/Oto.app` with the microphone usage description and the driver inside. Start at login registers the bundle and is unavailable when running outside one.

Measured on an M2 Pro running macOS 27.0.1:
- **Latency**: 35 ms end to end with a 48 kHz mic (mic IO buffer → chain → sink → driver → recorder), measured with `oto latency` before the review fixes. Not re-measured since.
- **CPU**: 0% when idle.

Not done yet:
- Developer ID signing, notarization and a notarized installer. `packaging/macos/README.md` lists the manual steps.
- A real install, the admin prompt and the microphone prompt have not been exercised from the bundled app.
- "Hear the difference" test.
- Which apps are using Oto Microphone (only "in use" is shown).
- Live checks for unplug, sleep and Bluetooth mics.
- Shortcut, updater, Linux.

```bash
cargo test
make -C driver/macos && scripts/macos/install-driver.sh
cargo run --release -p oto-cli -- run
scripts/macos/bundle.sh && open target/bundle/Oto.app
```
