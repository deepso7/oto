# 0004: Oto-1 — Custom 48 kHz pure-Rust & WASM model (Clean & Isolate)

**Status:** Proposed (merged with [fable review](fable/0004-review.md) and [fable draft](fable/0004-fable-plan.md); research notes in [fable/research/](fable/research/)) · 2026-11-27

[← Docs index](../README.md)

## Goal

Ship Oto’s own permissive-licence (`MIT OR Apache-2.0`), fullband 48 kHz realtime speech model (**Oto-1**), trained in PyTorch on Colab / rented GPUs and executed **100% in pure Rust** natively (Apple Silicon, Intel Mac, Linux) and inside a browser **`AudioWorklet`** (single-threaded WASM, no `SharedArrayBuffer`, no `ort`, no `tract`) — beating Krisp on concrete gates frozen before training begins.

Two checkpoints sharing one zero-allocation Rust kernel, built in order:

1. **Oto-1 Clean:** 48 kHz noise cancellation + dereverberation + crisp voice clarity trained directly into the model. Replaces DeepFilterNet3 as Oto’s default.
2. **Oto-1 Isolate:** 48 kHz background-voice removal (BVC), fine-tuned from Clean. Blind near-field mode for headsets; enrolled mode for built-in laptop mics (where Krisp’s own BVC does not claim support).

## Background

Today ([Status](../status.md), [Models](../models.md), [MODELS.toml](../../MODELS.toml)):

- Every third-party open model fails at least one requirement:
  - **DeepFilterNet3:** 48 kHz, permissive, no BVC, 40 ms chain (~62 ms macOS e2e), allocates ~1,700×/frame inside pinned `tract 0.21.4`.
  - **DPDFNet-2:** 48 kHz, strong denoise, no BVC, 60 ms chain (~82 ms e2e), Apple Silicon only (`ort` lacks prebuilt x86_64 macOS binaries).
  - **FastEnhancer-S:** 48 kHz, 21.3 ms chain, MIT code, no BVC, **cannot ship** (unstated weights licence; trained on export-restricted AI-Hub plus DEMAND/WHAM/TUT).
  - **Nearfield v2.3 `fast`:** has BVC, but **16 kHz only**, 50 ms chain (~80 ms e2e), tuned on the Krisp test set, passes lone background voices before the caller speaks, loses ~25% of first words in equal-talker clips, and has undisclosed training data.
- `ort` adds ~25–30 MB of C++ runtime to `Oto.app` and cannot run inside a browser `AudioWorklet`; `tract` in WASM is 9–15 MB before weights.
- **What Krisp actually is** ([fable/research/krisp.md](fable/research/krisp.md)): three distinct products, **none at 48 kHz**:
  - **Krisp NC** (on-device noise): up to 32 kHz, ~25 ms algorithmic latency (10 ms frames at 16 kHz), 2.8–5.8 MB `.kef`, no public benchmark outputs.
  - **Krisp BVC** (on-device isolation): 32 kHz, 15 ms algorithmic latency, 8.6–14.7 MB `.kef`, 5–13% CPU, enrollment-free, **gated to an allow-list of ~291 headsets** (built-in laptop mics only *"might demonstrate acceptable performance"*).
  - **Krisp VI 2.5** (server isolation): 16 kHz (one 32 kHz HD variant), 15 ms algorithmic latency, **only Krisp product with public scoreable outputs** on the 265-recording Krisp Voice Isolation Benchmark (CC BY-NC 4.0, 32 kHz, no clean references). With Parakeet TDT 0.6B v2: **5.30% WER / 1.52 leaked words/min** (raw: 23.21% / 66.48; DeepFilterNet3: 13.30% / 22.77).
  - **Krisp Browser SDK:** 12 MB package, ~100 MB RAM, 1.5–2 ms per 10 ms frame in a Worker.

## What "beat Krisp" means

These gates are written into `tools/eval/gates.toml` and frozen in Phase 0 before training starts.

| Axis | Krisp reference | Oto-1 gate |
|---|---|---|
| Audio bandwidth | ≤ 32 kHz (16 kHz on VI 2.5 Default); no 48 kHz model | **48 kHz**, with measured spectral content > 16 kHz on fullband input |
| Algorithmic latency | NC ~25 ms; BVC & VI 15 ms | **Clean ≤ 21.3 ms** (`1024/512`, 0 lookahead). **Isolate ≤ 21.3 ms**, plus a `≤ 16 ms` asymmetric-window experiment (Phase 2.5) |
| Native macOS e2e latency | — | **≤ 40 ms** inline in the Core Audio callback (vs. ~62 ms DeepFilterNet3 today) |
| Browser download | 12 MB package | **≤ 1.5 MB** brotli (`wasm` + `f16` weights) for Clean |
| Browser CPU (`AudioWorklet`) | 1.5–2 ms per 10 ms frame (Worker) | **Paced p99 ≤ 2.0 ms** per `10.67 ms` hop on a 2020-class laptop (hard quantum ceiling `2.67 ms`); racing mean `≤ 0.5 ms` |
| Native CPU | BVC 5–13% | **≤ 5%** of one M-series core, inline, **0 heap allocations/frame** |
| Noise quality (vs. Krisp NC) | No public outputs | DNSMOS `SIG` and `OVRL` not lower than Krisp NC at 95% confidence on identical files; `BAK` within `0.1`; clean-speech WER regression `≤ 0.2 pp` |
| Isolation (vs. Krisp VI 2.5 Default) | 5.30% WER, 1.52 leaked wpm | WER not significantly worse, leak significantly better (speaker-bootstrap over the 47 speakers, strict streaming mode, never tuned on this set) |
| Do no harm (phone-call subset) | Krisp worsens clean phone calls (`3.82% → 4.15%`) | Phone-call subset WER **≤ 4.15%** (stretch: `≤ raw`) |
| Built-in MacBook mics | Unsupported / hedged by Krisp | Enrolled Isolate passes `[voice_isolation.opt_in]` on self-recorded MacBook clips |

Until Krisp NC reference outputs are collected in Phase 0, Clean’s interim gate is **"beat DeepFilterNet3 on the manifest suite at ≤ 40 ms macOS e2e and lower CPU"** (matching `[voice_isolation.default_promotion]`’s latency/quality bar).

**Honest expectation:** Clean matching or beating Krisp NC at 48 kHz is realistic. A first Isolate attempt at this size and latency will likely land around **6–8% WER** on the Krisp set (between Krisp Lite at 6.21% and Hush at 12.09%); tying or beating 5.30% is a stretch goal expected to take multiple recipe iterations and possibly teacher distillation.

## Decisions

### 1. Clean first, Isolate second — separate checkpoints sharing one kernel

- Every surveyed background-voice isolator slightly degrades lone-talker speech. Default noise cancellation (**Oto-1 Clean**) must not carry that regression.
- **Oto-1 Isolate** starts from the trained Clean weights and shares the exact same Rust inference kernel. Merging both into one checkpoint is only revisited later if Clean’s gates do not move.

### 2. Bake off two proven 48 kHz `1024/512` architectures on identical data

Rather than inventing an unproven custom stack on run 1, train two proven streaming templates on the exact same permissive dataset and keep the winner on quality + measured browser timing:

- **Candidate A — DeepFilterNet-style hybrid (`1024/512`, 0 lookahead):** `32` ERB gains over the full 48 kHz band + complex deep filtering on the lowest `96` bins (`0–4.5 kHz`, `46.875 Hz/bin`) + causal grouped GRU temporal core.
- **Candidate B — FastEnhancer-style (`1024/512`, 0 lookahead, MIT code):** FastEnhancer-B/S layout (frequency-only conv kernels, foldable `BatchNorm`, time-axis GRU).

### 3. STFT geometry & clock alignment (`hop = 512`)

- **Sample rate:** `48,000 Hz`.
- **STFT:** `N_fft = 1024` (`513` complex bins, `46.875 Hz/bin`), **hop `H = 512` (`10.667 ms`)**, symmetric `sqrt(Hann)` analysis/synthesis, **0 frames lookahead** → **`21.33 ms` chain delay** (`512` adapter + `512` overlap-add).
- **Resolution against plan 0002 Phase 3’s 10 ms AEC clock:** Keep **hop `512`** for Oto-1. Web Audio’s `AudioWorklet` is hard-locked to `128`-sample quanta (`512 = 4 × 128`, zero reframing jitter), and Core Audio favors `128`/`256`. When desktop Phase 3 AEC (`480`-sample WebRTC frames) lands later, `oto-engine` bridges `480 ↔ 512` via a small ring (~0.7–1.3 ms average offset) rather than breaking browser quantum alignment.
- **Latency experiment:** Phase 2.5 tests an asymmetric synthesis window (`1024` analysis, shorter synthesis) to push algorithmic delay toward `13–16 ms` at the same `512` hop, kept only if quality holds.

### 4. Budget by measured paced wall-clock time, not MAC counts

- Literature and WASM benchmarks ([fable/research/architectures.md](fable/research/architectures.md), [fable/research/wasm-runtime.md](fable/research/wasm-runtime.md)) show MACs do not predict latency: time-axis conv kernels > 1 and `LayerNorm` hurt wall-clock speed disproportionately, while foldable `BatchNorm` + time-axis GRU run far faster than MAC counts suggest.
- **Browser ceiling reality check:** On a 2020 Intel laptop in single-threaded WASM `simd128`, ~`1 GMAC/s` already lands at 2–5 ms racing (and paced execution can add 2–4× due to cold caches and clock scaling). Because a `512`-sample hop runs inside **one `2.67 ms` render quantum**, single-threaded `AudioWorklet` realistically targets **`~0.35–0.8 GMAC/s`** (DeepFilterNet-style / FastEnhancer-B-class), not `1.8+ GMAC/s`.
- Therefore **Phase 2 benchmarks random-weight candidates in real Chrome, Safari, and Firefox (racing and paced) *before* locking model width for full training.**

### 5. Hand-written pure-Rust kernel (`oto-dsp`) + `wasm-bindgen`-free worklet (`oto-wasm`)

- **Native & WASM kernel (`crates/oto-dsp`):** Minimal layer set (linear, frequency conv with folded `BatchNorm`, GRU with subnormal flushing, activations, fixed `1024`-pt real FFT). Each op has a scalar reference plus `core::arch` SIMD for `aarch64` NEON, `x86_64` AVX2/SSE2, and `wasm32 +simd128` (`std::simd` and `f16` remain unstable on stable Rust). Enforced **0 allocations/frame** via `no_alloc` tests.
- **Browser crate (`crates/oto-wasm`):** Plain `cdylib` with `extern "C"` exports, **zero imports**, `panic = "abort"`, **no `wasm-bindgen`** (`TextDecoder`/`TextEncoder` still break `AudioWorkletGlobalScope` registration in 2026). Ship a `+simd128` build and a scalar fallback chosen on the main thread via `WebAssembly.validate`, passed to the worklet via `processorOptions` (with bytes fallback). Run warm-up frames in the processor constructor to trigger V8 TurboFan tiering before audio flows, and include an overload monitor that falls back cleanly to matched-delay bypass.
- **Honest browser latency:** `21.3 ms` is STFT delay only; real added delay in a browser `AudioContext` is ~`30–45 ms` and will be measured per browser in Phase 2.
- Keep `tract` and `ort` as test oracles and for existing models until Oto-1 passes its promotion gates.

### 6. Weight format (`.otow` `f16`) & layered parity

- **Format:** Export from PyTorch as `.safetensors` (inspectable, loaded by parity tests), compiled at build time into a flat 64-byte-aligned `.otow` binary with architecture hash, checksum, folded `BatchNorm`, and **`f16` weights** (unpacked to `f32` once at load). **No `int8` in v1** (avoids the `0.08–0.3` PESQ degradation on GRUs).
- **Layered parity gates (replacing the contradictory "bit-for-bit to −60 dB"):**
  - Scalar `f32` Rust vs. PyTorch reference: **≤ −90 dB** relative RMS error (and max abs diff ~`1e-6`–`1e-7`).
  - SIMD builds vs. scalar Rust: bit-identical where reduction order matches, otherwise **≤ −90 dB**.
  - Any future quantised (`int8`) build is gated on perceptual/WER metrics rather than waveform dB.

### 7. Voice Isolation: blind for headsets, enrollment for built-in mics (no VoxCeleb)

- Single-mic blind proximity isolation fails reliably on arm’s-length built-in laptop mics (Krisp itself restricts BVC to headsets). Oto selects **blind mode on headsets** and **enrolled mode on built-in mics**, matching device class.
- **Clean provenance:** Every standard pretrained speaker encoder (ECAPA-TDNN, TitaNet, WeSpeaker) is trained on VoxCeleb (YouTube audio) and fails Oto’s data audit. We will **train our own enrollment encoder jointly on permissive speakers** (4,600+ speakers in HiFiTTS-2 plus wideband Common Voice / GLOBE / VCTK). Because it runs once offline during enrollment, it has zero realtime cost; only its size in the download matters.
- **Enrollment UX:** Evaluate **10 s vs. 20–30 s** of enrollment speech in Phase 3 (balancing onboarding friction against separation quality; 5 s is likely too short).
- **Fail-closed contract:** Isolate remains fail-closed (`silence`, never raw mic or fallback model) and must pass all `[voice_isolation.fault_injection]` checks.

### 8. Data provenance, training recipe & compute plan

- **Audit updates ([docs/training-data-audit.md](../training-data-audit.md)):**
  - **Fullband clean speech:** **HiFiTTS-2** (CC-BY-4.0 metadata over PD LibriVox, filtered to `≥ 16 kHz` bandwidth, single-speaker, quality-scored, capped per speaker, spot-checked for MP3 top-band artefacts before bulk download) + **VCTK 0.92** (CC-BY-4.0, oversampled). Wideband speaker diversity / interferers: Common Voice (CC0), GLOBE (CC0), LibriTTS/MLS (CC-BY-4.0), with **per-clip effective-bandwidth loss masking** (URGENT methodology).
  - **Excluded from training:** **DAPS** (CC BY-NC 4.0 — eval-only), **ODbL** subsets in DNS5 (CREMA-D, PTDB-TUG), DNS5 German/AudioSet/VoxCeleb2, **real SLR28 RIRs** (RWCP research-only).
  - **Permissive noise & real crosstalk:** **FSD50K** filtered per clip to CC0 + CC-BY (~85% of clips, human-voice classes routed to interferers), **DNS5 `freesound_*`** (CC0, DEMAND removed), **FMA** CC0 + CC-BY tracks (~59 h music), **ACE** (CC BY 4.0, 48 kHz room noise + RIRs), **AMI headset crosstalk** (CC BY 4.0, real crosstalk, 16 kHz band), **DiPCo** (CDLA-Permissive 1.0, allowed as attribution-only/permissive).
  - **48 kHz RIRs:** Simulate paired near (`0.15–0.6 m`) and far (`1.2–6 m`) banks from the same rooms at **48 kHz** using `pyroomacoustics` (hybrid ISM + ray tracing, air absorption, source/mic directivity) plus verified real 48 kHz RIRs (ACE, Arni, AIR, BUT ReverbDB).
- **Training recipe (proven elements, no GAN in v1):**
  - Multi-resolution compressed STFT (magnitude + complex) + waveform L1 + **over-attenuation / asymmetric loss** (prevents swallowing the primary speaker) + **scale-aware silent-target loss** (replaces SI-SDR on lone-background-talker frames) + auxiliary target-presence BCE head.
  - Dereverberation target: direct sound + short early reflections (time-aligned).
  - BVC data recipe: interferers through the far-field RIR bank + low-pass + loudspeaker/codec coloration, **user distance decoupled from interferer distance**, background-speaks-first and background-only clips, device EQ/HFP augmentation, and **30–60 s long-sequence fine-tuning** (prevents continuous-talker fade).
  - GAN fine-tune is deferred as an optional last step only if perceptual quality lags after discriminative convergence.
- **Storage & compute:**
  - Pack shards on **`mito`** (macOS disk is nearly full) and host them in **Cloudflare R2** or a **private Hugging Face dataset repo** — **not Google Drive** (avoids FUSE bottlenecks and download-quota lockouts).
  - Plan for **5–10 training runs** (~100–250+ A100-hours across Clean + Isolate) with GPU-side mixing/RIR convolution and automatic 15–30 min checkpoint resume.
  - Use **Google AI Ultra Colab** monthly compute units (~95–185 A100-h/month) + **`rem`’s local NVIDIA GPU** for fast prototyping; allow an optional **$100–350 rented RTX 4090** overflow ceiling for uninterrupted multi-day runs without Colab’s 24 h cap.

## Work

### Phase 0: Measure the target (no training yet)
1. Complete the `tools/eval` manifest suite and reproduce Krisp’s published benchmark rows with Parakeet TDT 0.6B v2 (raw `23.21% / 66.48`, Krisp VI 2.5 Default `5.30% / 1.52`) within `0.2` WER points.
2. Score DeepFilterNet3, DPDFNet-2, and Nearfield v2.3 on the same 265-clip benchmark.
3. Capture Krisp NC reference outputs on a fixed 48 kHz eval list (via LiveKit Cloud’s free Krisp NC filter or Krisp macOS desktop app trial) and save locally outside the repo.
4. Record a ~30-minute private validation set on real target hardware (MacBook built-in mic + headset, with TV, second talker, background-speaks-first, and long silences) for Isolate checkpoint selection — **never select checkpoints on the Krisp benchmark**.
5. Write the gates above into `tools/eval/gates.toml` and freeze them.

### Phase 1: Data pipeline & Oto-1 Clean
1. Update `docs/training-data-audit.md` with the verified corpus list above.
2. Build the shard packer on `mito`: filter HiFiTTS-2 (`≥ 16 kHz` bandwidth, single speaker, DNSMOS/SNR-filtered, per-speaker cap), spot-check 50 chapters for MP3 top-band artefacts before bulk download, enforce speaker- and session-disjoint splits, generate an attribution manifest, and upload shards to R2 / private HF repo.
3. Generate paired 48 kHz near/far `pyroomacoustics` RIR banks.
4. Implement PyTorch trainer (`tools/train/`) with GPU-side mixing, effective-bandwidth loss masking, over-attenuation loss, and auto-resuming checkpoints.
5. Run a smoke test, then train **Candidate A (DeepFilterNet-style)** and **Candidate B (FastEnhancer-style)** on identical data; pick the winner on frozen gates + Phase 2 browser timings.
6. Fine-tune the winning Clean checkpoint on 30–60 s sequences.

### Phase 2: Pure-Rust kernel & browser WASM (parallel with Phase 1)
1. Implement the minimal pure-Rust layer library, `.otow` `f16` loader, and SIMD kernels (`aarch64` NEON, `x86_64` AVX2/SSE2, `wasm32 +simd128`) in `crates/oto-dsp`, with `no_alloc` tests and layered parity tests (`≤ −90 dB`) against PyTorch on random weights.
2. Build `crates/oto-wasm` (no `wasm-bindgen`, `extern "C"` exports, dual SIMD/scalar builds, V8 warm-up, overload bypass) and a static `web/demo/` `AudioWorklet` test page.
3. Benchmark both random-weight candidates in real **Chrome, Safari, and Firefox** (racing and paced) and lock model width inside the `≤ 2.0 ms` paced p99 quantum budget.
4. Run Oto-1 Clean inline in Oto’s Core Audio callback; measure end-to-end latency (`≤ 40 ms` target) and run a 30-minute soak.
5. Register in `MODELS.toml` and promote to default once `default_promotion` passes.
6. Run the asymmetric-window experiment (`1024` analysis, shorter synthesis) targeting `13–16 ms` algorithmic latency at `H = 512`; keep only if quality holds.

### Phase 3: Oto-1 Isolate
1. Fine-tune **blind BVC** from the Clean checkpoint: far-field rendered + low-passed interferers, decoupled user/interferer distance, background-first & background-only clips, auxiliary background-mask head, over-attenuation + scale-aware silent-target losses, 30–60 s sequences.
2. Fine-tune on **AMI headset crosstalk** (CC BY 4.0, 16 kHz band-masked) to close sim-to-real transfer.
3. Add **enrolled mode**: FiLM conditioning with embedding dropout + jointly trained permissive enrollment encoder; ablate **10 s vs. 20–30 s** enrollment duration.
4. Wire fail-closed enforcement, device-class selection (blind on headsets, enrollment prompt on built-in mics), and enrollment UI in `oto-app` and `web/demo/`.
5. Only if Krisp gates are still missed: teacher distillation (auditing teacher provenance first), and GAN polish last.

## Done when

- **Phase 0:** `tools/eval` reproduces Krisp’s published benchmark rows within `0.2` WER points, baseline models are scored, and `gates.toml` is frozen.
- **Clean:** passes `[ci]` and `default_promotion`, beats DeepFilterNet3 on the manifest suite (and meets Krisp NC gates once captured), runs inline on macOS at `≤ 40 ms` e2e and `≤ 5%` M-series CPU with `0` allocations/frame, and meets browser gates (`≤ 1.5 MB` brotli, paced p99 `≤ 2.0 ms`) in Chrome, Safari, and Firefox.
- **Isolate:** passes `[voice_isolation.fault_injection]` with zero leaks, passes `[voice_isolation.opt_in]` on built-in MacBook mics with enrollment, and clears the Krisp benchmark gates with speaker-bootstrap confidence intervals across at least two seeds.
- Release builds drop `ort` (and drop `tract` once DeepFilterNet3 is retired).

## Risks

| Risk | Mitigation |
|---|---|
| Neither candidate beats DeepFilterNet3 at 0 lookahead | Spend spare native/WASM headroom on width; allow 1 frame lookahead (`32 ms` chain, still faster than DFN3’s `40 ms`) |
| Single-threaded WASM paced p99 exceeds `2.0 ms` on 2020 laptops | Benchmark random weights in real browsers *before* training; keep Clean in the `0.35–0.8 GMAC/s` band, or split inference across the 4 quanta |
| Simulated near/far cues do not transfer to real rooms | Private real validation set from Phase 0, AMI headset fine-tune, and voice enrollment as the robust mechanism for built-in mics |
| HiFiTTS-2 MP3 artefacts degrade the >16 kHz band | Inspect 50 chapters before bulk download; oversample VCTK 0.92; use bandwidth-masked losses for sub-48 kHz speakers |
| Isolate plateaus at 6–8% WER | Ship opt-in on `[voice_isolation.opt_in]` gates for headsets + enrolled mics; next lever is provenance-clean teacher distillation |
| Safari / iOS `AudioWorklet` rate & `getUserMedia` quirks | Create `AudioContext` after `getUserMedia`, verify live `sampleRate` (resample or bypass if ≠ 48 kHz), test in real WebKit |
| Maintaining a private Rust inference engine | Keep the op set tiny (~6 primitives), maintain scalar references, enforce `≤ −90 dB` layered parity on every weight export |

## Out of scope

- Training in Rust (`burn` / `candle`), WebGPU, threaded WASM (`SharedArrayBuffer` / COOP / COEP), and Windows.
- Bolted-on post-model Speech Boost / EQ / compressor chains on top of the current third-party models (voice crispness and clarity come from Oto-1’s own 48 kHz dereverb + clean-speech training).
- Accent conversion and generative voice synthesis.
