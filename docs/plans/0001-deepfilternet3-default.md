# 0001: DeepFilterNet3 as the default model

**Status:** Accepted · 2026-11-10

[← Docs index](../README.md)

## Goal

Oto ships with DeepFilterNet3 as its noise-suppression model. RNNoise stays available behind one "Low CPU mode" switch for weak or battery-bound machines.

## Background

The app runs the original RNNoise (2018 weights) for every tier today. In real use it lets too much through: with a TV on in the room, the TV is clearly audible on the cleaned microphone.

DeepFilterNet3 was the planned default from the start ([Models](../models.md)). It is already in the codebase behind the `dfn3` cargo feature, and with that feature on the tier mapping already does what we want: Standard and Max run DeepFilterNet3, Light runs RNNoise. It was left off for these reasons, which this plan has to resolve:

| Blocker | Where | State |
|---|---|---|
| Allocates on the audio thread on every frame | `libDF` clones tensors; `tract` allocates per run (`crates/oto-dsp/src/dfn3.rs`) | Known, not measured |
| Fails to load in debug builds | `tract` errors with "running pass codegen" | Cause unknown; works in release |
| No fallback | With the feature on in a build where it cannot load, the engine reports an error and retries every 2 s | Known |
| Latency | 30 ms of model delay against RNNoise's 10 ms, so about 20 ms more end to end | Inherent to the model |
| Cost | About 6.5% of an M2 Pro core against 0.5% | Measured with `oto bench` |
| Size and build time | Adds `tract`, `ndarray` and embedded weights | Not measured |

What the numbers mean for the user: RNNoise measures about 36 ms end to end today. DeepFilterNet3 should land near 56 ms, which is over the 50 ms product target.

**What DeepFilterNet3 will and will not fix.** It removes music, effects, fans, keyboards and general room noise far better than RNNoise. It will not reliably remove TV dialogue or other people talking: like every ordinary noise suppressor it is trained to keep speech. Removing background voices needs a different kind of model and is its own later plan. This plan should be judged on noise, not on other voices.

## Decisions

Accepted on 2026-11-10.

1. **Two modes, one switch.** The user sees a single "Low CPU mode" switch in Settings, off by default. Off is DeepFilterNet3; on is RNNoise. Internally these stay `Tier::Standard` and `Tier::Light`. `Tier::Max` stays in the enum for a later model and is not offered.
2. **Accept a higher delay in the default mode.** The 50 ms target in [Product](../product.md) becomes: 65 ms or less in the default mode, 40 ms or less in Low CPU mode. Expected is about 56 ms if inference stays in the audio callback and about 60 ms or a little over if it moves to a worker thread (step 1). We do not try to cut DeepFilterNet3's lookahead; that needs retraining.
3. **The audio callback does not allocate once a stream is running.** That rule is not relaxed for this model. Today there is one known exception, which this plan makes explicit: the first callback of a stream primes the model on the audio thread, because RNNoise keeps per-thread state. It stays as the single documented exception for any model that runs inside the callback.
4. **RNNoise is always compiled in** and is the fallback if DeepFilterNet3 cannot load.
5. **No bake-off first.** The measured comparison with FastEnhancer and DPDFNet stays on the roadmap and does not block this change.

## Work

Until step 4 turns the feature on by default, every command in steps 1 to 3 passes `--features oto-engine/dfn3` (or `oto-dsp/dfn3`) and checks which model is actually running. A run that silently used RNNoise does not count.

### 1. Measure, then make inference realtime-safe

**Add the instruments first.** The engine only accumulates total callback time today (`busy_ns`), so it cannot report a worst case, and `oto bench` runs unpaced. Add to the shared stats, with fixed-size atomics only: the maximum callback time and a small fixed histogram for p99, per stream. Show them in `oto run`, which owns the engine and can read its stats. `oto status` is a separate process with no channel to a running engine; reporting across processes is deferred.

**Fix the allocation test.** `crates/oto-dsp/tests/no_alloc.rs` primes before it starts counting, while the engine primes inside the first callback (`CaptureStage::process` calls `Chain::prime`). Extend the test to run the real capture path and the setters (strength, bypass), and to count from the second callback on, so it checks the contract in decision 3 as written.

**Measure DeepFilterNet3**: allocations and bytes per frame, and callback time at p50, p99 and max under realtime pacing in a live run.

**Then pick one of two designs.**

- **A. Remove the allocations.** Vendor `libDF` into the repo and make it reuse its buffers; check whether `tract`'s per-run allocations can be avoided with a preallocated state. Inference stays inside the audio callback and nothing else changes. No added latency. Choose this if the extended allocation test passes for DeepFilterNet3.
- **B. Move the chain to a worker thread.** The boundary is the whole chain, not the model: `FrameAdapter` calls the model synchronously and expects its output in the same call, so an asynchronous model behind the existing adapter would add a full 10 ms frame. Instead the capture callback pushes its block into a lock-free ring (`rtrb`, already a dependency) and pulls processed audio from a second one; a dedicated high-priority thread owns the adapter, the model and the limiter, and may allocate. Priming also moves to that thread. **B applies to DeepFilterNet3 only.** RNNoise does not allocate after priming, so it stays in the callback as it is today, keeps its 36 ms, and keeps the one priming exception from decision 3. Low CPU mode is therefore unaffected by B. What B must define and test:
  - **Delay.** The reserve between the two rings, in addition to the adapter's existing 10 ms. Expected about one capture block (5.3 ms at 48 kHz) plus inference time. Measured before committing to B.
  - **Starvation.** When the worker has not delivered in time, the callback outputs the dry signal at the same delay, cross-faded. When the worker catches up, wet output older than the current position is discarded before the fade back, so audio never jumps backwards.
  - **Counters.** Worker time per frame (max and p99), starvation count and ring overflow count, next to the existing underrun and resync counts. The CPU load figure includes the worker.

Try A first with a time limit of about two days. If `tract` cannot be made allocation-free in that time, do B.

### 2. Make it load in debug builds

Find out why `tract` fails with "running pass codegen" in debug builds. A likely cause is that the dependency is compiled without optimisation; the usual fix is a per-package `opt-level` for `tract-*`, `ndarray` and `deep_filter` in the dev profile, which also makes debug builds usable in real time. Confirm rather than assume. Done when `cargo run -p oto-cli --features oto-engine/dfn3 -- run --force-clients` reports DeepFilterNet3 as the running model in a debug build.

### 3. Fall back instead of failing

The control thread compares the running model with the one the tier asks for on every tick, so simply starting RNNoise when DeepFilterNet3 fails would restart the streams forever. The engine therefore keeps two things apart: the **requested** model (from the tier) and the **effective** model (what is running).

- A failed load is remembered. While it is remembered, a request for DeepFilterNet3 resolves to RNNoise without retrying.
- The memory is cleared when the user changes the tier or the app restarts. Nothing retries on a timer.
- The engine sends one event naming the effective model and the reason, and `Stats` reports the effective model. The app shows a notice, not an error.
- Tests: force the load to fail, then check that repeated reconcile ticks do not restart the streams, and that a client stopping and starting again still gets RNNoise without another load attempt.

### 4. Turn it on by default

Changing the default feature list of the two library crates is not enough: the app and the CLI depend on `oto-engine` with its defaults on and expose no features of their own.

- `oto-dsp` and `oto-engine` get `default = ["dfn3"]`. `oto-engine` depends on `oto-dsp` with `default-features = false` and forwards the feature.
- `oto-cli` and `oto-app` each get a `dfn3` feature, on by default, that forwards to `oto-engine`, and depend on it with `default-features = false`.
- `Model::default()` is DeepFilterNet3 when the feature is on and RNNoise when it is off. The tier mapping follows: Standard is DeepFilterNet3, Light is RNNoise, Max resolves to Standard. `available_tiers()` returns Light and Standard.
- These must both work, and the second must contain no `tract`: `cargo build --workspace` and `cargo build --workspace --no-default-features`.
- Measure the release binary, the app bundle and a clean build time before and after. The bundle is 20 MB today against a 25 MB download target; if DeepFilterNet3 pushes it over, record the number and decide whether to raise the target or compress the weights.

### 5. App and CLI

- Replace the tier picker in Settings with the "Low CPU mode" switch and one line of explanation: lower quality, less CPU, lower delay.
- **Switching during a call.** A tier change stops and restarts the streams. Today the stop is abrupt and only the restart fades in. Add a short bounded fade-out (about 10 ms) before the streams stop, load the new model on the control thread before the stop so the silent gap is short, and keep the existing fade-in after priming. Target: no click and under half a second of silence. Test both directions, since the two models have different delays.
- Existing saved preferences keep working: a saved Standard now means DeepFilterNet3.
- The latency shown in the app and `oto run` reflects the effective model.
- `oto run --tier light` selects RNNoise; `oto bench` keeps timing both.

### 6. Verify on real audio

- **Latency.** `oto latency` detects a 10 ms tone above a fixed threshold, and a denoiser can suppress the tone, so it can under-count bursts and still exit 0. Measure each tier with `--no-denoise`, which keeps that model's delay with processing bypassed, and make the command fail when it detects fewer than most of the bursts it sent. Run it on the USB mic and the built-in mic.
- **The model is really running.** Separately from latency: play a recorded noise clip through `oto run --input-raw` and check that the cleaned level drops well below the raw level, with the effective model reported as DeepFilterNet3.
- **Load.** CPU while in a call in both modes, including the worker if design B was chosen, and 0% when no app is recording.
- **Soak.** A 30-minute run with a forced client: no underruns, no resyncs, no starvation, and callback time p99 well inside the buffer period, read from the instruments added in step 1.
- **Listening.** Record a clip with the TV on, loop it with `--input-raw`, and record Oto Microphone in any recorder at each tier and strength. Compare by ear.
- **Switching.** Change modes during a real call.

### 7. Housekeeping

- A third-party notices file covering DeepFilterNet (code and weights), `tract` and `nnnoiseless`, included in the app bundle.
- Update [Status](../status.md), [Models](../models.md) and the targets in [Product](../product.md) to match what was measured.

## Done when

- A fresh install denoises with DeepFilterNet3 without any configuration, and the app reports it as the running model.
- Low CPU mode switches to RNNoise and back during a call without a click or a hang.
- The extended allocation test passes for both models under the contract in decision 3.
- Measured end to end with `--no-denoise`: 65 ms or less in the default mode, 40 ms or less in Low CPU mode.
- A 30-minute run shows no underruns, resyncs or starvation.
- Debug and release builds both run DeepFilterNet3; `--no-default-features` builds an RNNoise-only app with no `tract`.
- If DeepFilterNet3 cannot load, the app runs on RNNoise, says so once, and does not restart audio in a loop.
- The numbers in the docs are the measured ones.

## Risks

| Risk | What we do |
|---|---|
| `tract` cannot be made allocation-free | Design B, the worker thread |
| The worker thread adds more delay than expected | Measured in step 1 before committing. If the total passes 65 ms, halve the mic IO buffer (saves about 2.7 ms) before raising the target |
| DeepFilterNet3 is too heavy on older Intel laptops (about 19% of a 2017 U-series core) | Low CPU mode exists for this; consider suggesting it when callback load stays high |
| Upstream `libDF` is dormant and pinned to `tract` 0.21.4 | Vendoring in design A makes it ours; otherwise keep the pin and note it |
| The bundle goes over 25 MB | Measured in step 4; decide then |
| Users expect background voices to disappear | Say plainly in the app and docs what the model removes; background-voice removal is a separate plan |

## Out of scope

- Removing background voices (TV dialogue, other people). Candidates are in [Models](../models.md).
- The measured bake-off against FastEnhancer and DPDFNet, and RNNoise 0.2 as a better light model.
- Offline `oto process` and `oto record` commands. They belong with the bake-off; this plan uses the existing `--input-raw` flag.
- A third "Max" tier.
- Linux.
