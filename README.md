# Oto

A realtime noise-cancelling virtual microphone for macOS and Linux, written in Rust. Any app that can pick a microphone picks "Oto Microphone" and gets a clean voice. Open source and free.

**Status:** works on macOS from a source checkout. Not yet signed or packaged for other machines; Linux is next. See [docs/status.md](docs/status.md).

## Try it (macOS)

```bash
scripts/macos/install-driver.sh
```

```bash
scripts/macos/bundle.sh && open target/bundle/Oto.app
```

The first command builds the audio driver and installs it, which needs your password and briefly interrupts all audio. The headless alternative to the app is `cargo run --release -p oto-cli -- run`.

## Layout

| Path | What |
|---|---|
| `crates/oto-dsp` | Signal chain and models |
| `crates/oto-engine` | Audio devices, the realtime path, the virtual microphone backends |
| `crates/oto-app` | Menu-bar app, built on [Rune](https://github.com/anomalyco/rune) |
| `crates/oto-cli` | `oto`: headless run, status, latency and benchmark tools |
| `driver/macos` | The Core Audio driver that provides "Oto Microphone" |
| `docs/` | [Design, roadmap, status and plans](docs/README.md) |

## Licence

MIT or Apache-2.0, at your option.
