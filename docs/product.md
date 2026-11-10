# Product

[← Docs index](README.md)

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
| Added latency, end to end | 65 ms or less in the default mode (DeepFilterNet3); 40 ms or less in Low CPU mode (RNNoise). Measured: 61–63 ms and 34–35 ms |
| RAM | under 50 MB, excluding the UI's GPU surfaces |
| Download | under 25 MB (measured: 22.5 MB zipped; the bundle is 38 MB unpacked) |
| Network use | update checks only; no account, no telemetry by default |

The latency target is the hard one; see [Models](models.md). It was 50 ms until [plan 0001](plans/0001-deepfilternet3-default.md) accepted DeepFilterNet3's 30 ms of model delay for the default mode.
