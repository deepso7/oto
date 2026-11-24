# 0002: End-to-end roadmap: evaluation, Linux, Voice Isolation, our own model

**Status:** Accepted · 2026-11-24 (council-reviewed, 4 rounds: Claude Opus 5.5 9.5/10, Codex GPT 6.2 Sol 9.7/10). Phase 1 Track A in progress.

[← Docs index](../README.md)

## Goal

Take Oto from a macOS noise suppressor to a measured, cross-platform product: a repeatable evaluation harness with frozen gates, a Linux PipeWire build, opt-in background-voice removal (Voice Isolation) that never leaks the raw mic, echo-safe operation in call apps, and finally a model of our own with permissive weights.

## Background

[Plan 0001](0001-deepfilternet3-default.md) made DeepFilterNet3 the default and moved its inference to a worker thread. Since then DPDFNet-2 and FastEnhancer-S landed behind a Model picker ([Status](../status.md)). Three things are missing before the next model or platform can be judged:

- **No offline evaluation.** Models are compared by `oto bench` (CPU) and by ear. There is no `oto process`, no fixture suite, no scorecard, and no gate that a new model must pass.
- **No Linux.** The engine has a `Platform` trait with one Core Audio backend.
- **No background-voice removal.** Every shipped model is trained to keep speech, including the TV's and other people's ([Models](../models.md)).

Latency is the constraint that shapes every decision below. The figures are built from components so a new model's cost can be read off before it is measured.

**Component-derived latency.** Chain delay = model delay + adapter frame. End to end = chain + platform overhead (device I/O, ring reserve, and for worker models the worker's reserve).

| Platform overhead | Inline (in the callback) | Worker thread |
|---|---|---|
| macOS (Core Audio + HAL plug-in) | ~14–15 ms | ~21–23 ms |
| Linux PipeWire (provisional, `node.latency = 480/48000`) | ~10–15 ms | ~17–22 ms |

| Model | Placement | Chain | macOS end to end | Linux end to end |
|---|---|---|---|---|
| RNNoise | inline | 10 + 10 = 20 ms | ~35 ms (measured 34–35 ms) | ~30–35 ms (provisional) |
| DeepFilterNet3 (default) | worker | 30 + 10 = 40 ms | ≤ 65 ms (measured 61–63 ms) | ~57–62 ms (provisional) |
| DPDFNet-2 (Max, 48 kHz) | worker | 50 + 10 = 60 ms | ≤ 85 ms (~81–83 ms) | ~77–82 ms (provisional) |
| Nearfield v2.3 `fast` (opt-in Voice Isolation, `frame_size = 384`) | worker | 48 ms model + 8 ms adapter + ~2–3 ms 48↔16 kHz resamplers ≈ 58–59 ms | ~79–82 ms (provisional); ~87–90 ms behind a 10 ms clock + 6 ms FIFO in Phase 3 | ~75–81 ms (provisional); ~83–89 ms behind a 10 ms clock |

Provisional figures are replaced by measured values before `tools/eval/gates.toml` is frozen.

## Decisions

Accepted on 2026-11-24 after four council rounds.

1. **Measure before shipping.** Every model, mode and platform passes the same harness (`tools/eval/`) and the gates in `tools/eval/gates.toml`. Thresholds are frozen per phase; a gate is never loosened to let a candidate through in the same change.
2. **Provenance is machine-checked.** [`MODELS.toml`](../../MODELS.toml) lists every model with its source, pinned version or hash, licences and whether it may ship. The harness verifies the hashes; release builds may only embed `ship_in_release = true` entries.
3. **Voice Isolation is opt-in and fails closed.** When it cannot run correctly it outputs silence or a held fade, never the raw mic and never a different model. Background-voice removal that leaks the TV some of the time is worse than none.
4. **DeepFilterNet3 stays the default** until a candidate beats it on the default-promotion gates, on both platforms, at an equal or lower latency class.
5. **Linux first on PipeWire only.** No PulseAudio or JACK backend. PipeWire libraries are linked from the system, never bundled.
6. **Training data follows [the audit](../training-data-audit.md).** Non-commercial and share-alike corpora may be used for local evaluation only, never to train shipped weights and never committed to the repo.
7. **Phases run in parallel tracks where they do not share files**, so Linux and evaluation work do not block each other.

## Work

### Phase 1: measurement and Linux (parallel tracks A and B)

**Track A: evaluation (macOS, in progress).**

- `oto process IN.wav OUT.wav`: offline, deterministic file-to-file processing through the real chain (`--model`, `--strength`, `--no-denoise`, `--no-delay-comp`, `--json`; WAV at any rate, resampled to 48 kHz and back), so every model is judged on identical input.
- `oto bench --live`: realtime-paced bench reporting p50, p99, p99.9 and max per block, deadline misses and worker starvations, plus `--json`.
- `oto latency` measured on speech onsets, not only tone bursts, and reporting the missed-onset rate.
- `tools/eval/`: stdlib-only CI suite on synthetic fixtures, an external manifest mode for local datasets, same-ASR head-to-head, Markdown and JSON scorecards checked against `gates.toml`.
- `MODELS.toml` and the Phase 5 [training-data audit](../training-data-audit.md).

**Track B: Linux PipeWire (developed and tested on Linux).**

- `pipewire` crate 0.10.x; capture and virtual source streams with `StreamFlags::RT_PROCESS`.
- Pin to the chosen mic: `node.dont-fallback`, `node.dont-reconnect`, `node.dont-move`. Check the peer before any audio flows, and invalidate the gate on any runtime retarget, so Oto never silently records from a different device.
- `node.group` / `node.link-group` to keep capture and source in one driver group; `node.latency = 480/48000`; `node.passive = true` so Oto does not keep the graph awake when no app records.
- Default source via `default.configured.audio.source`, restored on exit or crash with a tagged sentinel, `Restore::{Uid, Unset}`, so "there was no configured default" is restored as unset rather than to a guess.
- Keep `rubato` for drift control initially.
- Tray via `rune::Tray`.
- Packages: `.deb` and AppImage, excluding `libpipewire-0.3` and SPA plugins (always the system's).

### Phase 2: opt-in Voice Isolation

- **Nearfield v2.3 `fast`** (`frame_size = 384`) behind an opt-in mode. Provenance in `MODELS.toml` (`candidate-pending-phase2-gates`): upstream `NOTICE` reproduced, training data checked against the audit.
  - Known healthy-inference limitation (upstream v2.3 README): its proximity rule ("nearest voice heard so far") deliberately passes a lone background voice before the caller speaks, and upstream synthetic tests report losing the caller's first word in ~25% of equal-talker clips on `fast`. Fault-injection guards do not alter this behavior; a **background-first startup case** in the disjoint holdout is required before opt-in promotion (and evaluated separately from fault-injection).
- **Fail closed on every leak path:**
  - worker starvation (no dry fallback in this mode);
  - worker overflow or flush restart: wet output whose origin `o0 < wet_base` is discarded, not played;
  - `Stage` warmup, and strength semantics where non-maximum strength must not mix in dry signal that carries other voices;
  - `control.rs` mode switches: mute while the new model preloads; a load failure raises `Event::Degraded` and stays muted instead of falling back to RNNoise.
- **Two kinds of pass criteria, kept apart:** fault-injection tests (hard zero leak, `gates.toml` `[voice_isolation.fault_injection]`) and healthy-inference quality (`[voice_isolation.opt_in]`).
- DPDFNet-2 parity check against Ceva's reference.
- **"Hear the Difference"**: level-matched A/B of raw and cleaned voice (`HEAR_TEST = true`), disabled during active calls.
- Rune UI simplified to two modes: Noise removal and Voice Isolation.
- A 48 kHz single-path gain-mask hybrid is explicitly **not** part of Phase 2's completion criteria.

### Phase 3: Linux daemon and echo safety (independent tracks A and B)

**Track A: Linux `otod`.** `systemd --user` service; control socket `$XDG_RUNTIME_DIR/oto.sock` with mode `0600`; models loaded lazily when an app records and unloaded when idle.

**Track B: echo cancellation and the clock.**

- **Gate 0:** does Oto break Zoom, Meet or Discord AEC? Measured before anything else in this track.
- HAL version and upgrade gate: compare the on-disk `CFBundleVersion` with the loaded driver's property; never restart `coreaudiod` mid-call.
- A single 10 ms frame clock across capture, model and sink.
- Stereo AEC prototype: `sonora` against `webrtc-audio-processing`.
- Opt-in **Oto Speaker**: a post-processing render feed; on macOS 14.2+ a Process Tap opt-in, runtime-gated, with fallback.

### Phase 4: performance and reach

- DeepFilterNet3 on ONNX Runtime (`IoBinding`), gated on ≤ −59 dB parity with `tract`. Intel Macs keep `tract` DeepFilterNet3 and RNNoise until/unless Intel is dropped on macOS 27+.
- Opt-in streaming/OBS voice shaping, off by default for calls.
- Rune global shortcuts, a Bluetooth HFP soak, a Homebrew Cask, and expanded Linux packages.

### Phase 5: our own model

A custom Apache-2.0, causal 32/48 kHz model covering background-voice cancellation (BVC), dereverberation and denoising, under 30 ms algorithmic delay, trained only on the permissive set in the [training-data audit](../training-data-audit.md) and evaluated on speaker- and session-disjoint holdouts.

## Done when

- **Phase 1:** `oto process`, `oto bench --live --json` and onset-based `oto latency` exist; `python3 tools/eval/eval.py --suite ci` passes for every shipped model; Linux runs DeepFilterNet3 and RNNoise end to end from a `.deb` and an AppImage; measured latency replaces every provisional figure in the table above and in `gates.toml`.
- **Phase 2:** Voice Isolation passes every fault-injection case with zero leak, meets the frozen opt-in gates, and Nearfield's `NOTICE` and data provenance are cleared.
- **Phase 3:** Gate 0 answered with measurements; `otod` runs as a user service; the AEC prototype has a go/no-go.
- **Phase 4:** each item either ships or is recorded as dropped with its reason.
- **Phase 5:** our model meets the default-promotion gates and replaces third-party weights where it beats them.

## Risks

| Risk | What we do |
|---|---|
| Provisional Linux and Nearfield latencies are optimistic | Measure in Phase 1/2 before freezing `gates.toml`; the table is updated, not the gate loosened |
| Voice Isolation leaks the raw mic on an unforeseen path | Fail-closed design plus fault injection on every listed path; any leak blocks the phase |
| Nearfield's training data turns out to be non-commercial or restricted | It stays `ship_in_release = false`; Phase 5 replaces it |
| Oto's processing breaks call-app AEC | Gate 0 in Phase 3 measures it before more work is built on top |
| PipeWire behaviour differs across distributions | Link the system's libraries; test on at least two distributions before packaging |
| ASR-based metrics are biased by the ASR model | Same-ASR head-to-head only: compare candidates with one ASR model and one configuration, never absolute WER across tools |

## Out of scope

- Windows.
- PulseAudio-only and JACK-only Linux systems.
- GPU inference.
- Shipping FastEnhancer weights (evaluation only, see [`MODELS.toml`](../../MODELS.toml)).
- Telemetry of any kind, including from the evaluation harness.
