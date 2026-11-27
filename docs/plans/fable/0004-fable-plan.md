# 0004 (Fable draft): Oto-1, our own 48 kHz model in pure Rust and WASM

**Status:** Draft for merging with [0004](../0004-oto-v1-model-and-wasm.md) · 2026-11-27 · Reasons for every change are in [0004-review.md](0004-review.md); sources in [research/](research/)

## Goal

Ship Oto's own permissive-licence 48 kHz speech model, run by a hand-written Rust kernel natively and in a browser `AudioWorklet`, and beat Krisp on gates that are written down before training starts.

Two checkpoints sharing one kernel, built in this order:

1. **Oto-1 Clean**: denoise plus light dereverberation. Replaces DeepFilterNet3 as the default.
2. **Oto-1 Isolate**: background-voice removal. Blind on headsets, enrolled on built-in mics.

## What "beat Krisp" means

Krisp's numbers are from its own docs and benchmark ([research/krisp.md](research/krisp.md)).

| Axis | Krisp | Oto-1 gate |
|---|---|---|
| Audio bandwidth | 16 kHz (32 kHz models); none at 48 kHz | 48 kHz, with measured content above 16 kHz on fullband input |
| Algorithmic latency | NC about 25 ms; BVC and VI 15 ms | Clean ≤ 21.3 ms. Isolate ≤ 21.3 ms, with a ≤ 16 ms variant tested (Work, step 2.5) |
| Browser download | 12 MB package | ≤ 1.5 MB (wasm plus weights, brotli) for Clean |
| Browser CPU | 1.5–2 ms per 10 ms frame | paced p99 ≤ 2 ms per 10.67 ms hop on a 2020 laptop; racing mean ≤ 0.5 ms |
| Native CPU | BVC 5–13% | ≤ 5% of one M-series core, inside the audio callback |
| Noise quality (vs Krisp NC) | no public outputs | DNSMOS SIG and OVRL not lower than Krisp NC at 95% confidence on the same files; no WER regression above 0.2 points on clean speech |
| Isolation (vs Krisp VI 2.5 Default) | 5.30% WER, 1.52 leaked words/min | WER not significantly worse, leak significantly better, bootstrap over the 47 speakers, streaming mode, never tuned on this set |
| Do no harm | Krisp makes clean phone calls worse (3.82% raw to 4.15%) | phone-call subset WER ≤ 4.15%; stretch ≤ raw |
| Built-in laptop mics | "might demonstrate acceptable performance" | Isolate with enrollment passes `[voice_isolation.opt_in]` on self-recorded MacBook clips |

Until Krisp NC outputs are obtained, the Clean gate is "beat DeepFilterNet3 at lower latency and CPU". That is also what the `default_promotion` gates require.

Expectation, stated honestly: Clean matching Krisp NC is realistic. A first Isolate attempt at this size and latency will probably land at 6–8% WER on Krisp's set (my estimate, between Krisp Lite and the open small models); a tie is a stretch that needs several training iterations.

## Decisions

1. **Clean first, Isolate second, separate checkpoints.** Isolation models damage a lone talker's speech in every system surveyed, so the default mode must not carry that risk. Isolate starts from the Clean weights. Merge into one checkpoint later only if Clean's gates do not move.
2. **Start from proven architectures, not a new one.** Train two candidates on the same data and keep the winner:
   - **A, DeepFilterNet-style**: 32 ERB gains plus deep filtering on the lowest 96 bins, with 1024/512 and no lookahead. This is what 0004 describes.
   - **B, FastEnhancer-style** (MIT code): the B and S sizes at 48 kHz already use 1024/512.
   Neither has published 48 kHz quality at zero lookahead on permissive data, which is why both are trained.
3. **Budget by measured time, not MAC counts.** The gates above allow roughly 0.5–2 GMAC/s, about ten times what 0004 assumed. Spend it on quality.
4. **Hand-written Rust kernel.** Small layer set (linear, conv, GRU, activations, fixed 1024 FFT), a scalar reference for each, `core::arch` SIMD for NEON, AVX2 and simd128. Keep `tract` and `ort` as test oracles and for the existing models until Oto-1 clears its gates.
5. **Browser: inference inside the worklet, no `wasm-bindgen`.** `extern "C"` exports with zero imports, `panic = "abort"`, a simd128 build and a scalar build chosen with `WebAssembly.validate`, one hop every fourth 128-sample quantum, an overload monitor that bypasses.
6. **Weights: f16 flat file, exported from safetensors at build time.** No int8 in v1.
7. **Parity in layers.** Scalar Rust against PyTorch at about −90 dB; SIMD builds bit-identical to scalar where the operation order allows, otherwise ≤ −90 dB; any quantised build gated on quality metrics instead.
8. **Enrollment is the answer for built-in mics; blind mode is for headsets.** Oto picks by device class, as Krisp does with its allow list. No VoxCeleb-trained encoder: train the enrollment encoder jointly on permissive speakers (4,600 in HiFiTTS-2, many more wideband in Common Voice), sized freely since it runs once, with 20–30 s of enrollment speech.
9. **Data follows the audit, with the corrections in Work step 1.2.**
10. **Compute: many short runs.** Shards live in Cloudflare R2 or a private Hugging Face repo, not Drive. Colab units cover roughly 95–185 A100-hours a month; a rented RTX 4090 covers the rest. Checkpoint every 15–30 minutes and resume automatically. Mixing, RIR convolution and EQ run on the GPU.
11. **Speech Boost moves to its own plan.** It does not depend on the model.

## Work

### Phase 0: measure the target (no training)

1. Finish the manifest suite in `tools/eval` and reproduce the published Krisp benchmark rows: unprocessed 23.21% / 66.48 and Krisp Default 5.30% / 1.52 with Parakeet TDT 0.6B v2. This proves the harness.
2. Score DeepFilterNet3, DPDFNet-2 and Nearfield on the same set.
3. Get Krisp NC outputs for a fixed file list (SDK trial, desktop app virtual mic, or LiveKit Cloud).
4. Record a private validation set on the real devices: MacBook mic and a headset, with a TV and a second talker, including background-speaks-first and long silences. About 30 minutes. This is the checkpoint-selection set for Isolate.
5. Write the gates above into `tools/eval/gates.toml` and freeze them.

### Phase 1: data and Oto-1 Clean

1. Update `docs/training-data-audit.md`: add HiFiTTS-2, VCTK, FSD50K (CC0 and CC-BY clips only), FMA (CC-BY and CC0 tracks), ACE, Arni, BUT ReverbDB, AMI; restrict DNS5 to its LibriVox, VCTK and Freesound parts; restrict SLR28 to simulated RIRs; record that DAPS is non-commercial.
2. Build the packer on `mito` (the Mac's disk is too full): filter HiFiTTS-2 to ≥ 16 kHz bandwidth, single speaker, quality-scored, capped per speaker; inspect 50 files for MP3 artefacts before the bulk download. Speaker- and session-disjoint splits, with a generated attribution file.
3. Simulate 48 kHz RIRs with `pyroomacoustics` in paired near (0.15–0.6 m) and far (1.2–6 m) banks from the same rooms.
4. Trainer with GPU-side mixing, bandwidth-masked losses and the dereverberation target set to direct sound plus early reflections.
5. Smoke run, then train candidates A and B. Pick one on the frozen gates and the browser timing from Phase 2.
6. Fine-tune the winner on 30–60 s sequences.

### Phase 2: kernel and browser (parallel with Phase 1)

1. Layer library and `.otow` loader in `oto-dsp`, with the `no_alloc` test and layered parity tests against PyTorch on random weights.
2. `crates/oto-wasm` and a worklet demo. Time both candidates with random weights in Chrome, Safari and Firefox, racing and paced.
3. Run Oto-1 Clean inline in the Core Audio callback; measure end-to-end latency (expected about 36 ms) and a 30-minute soak.
4. Register in `MODELS.toml`; promote to default when `default_promotion` passes.
5. Try an asymmetric window (1024 analysis, shorter synthesis) to cut latency toward 13–16 ms at the same hop. Keep it only if quality holds.

### Phase 3: Oto-1 Isolate

1. Blind mode, from the Clean weights: interferers rendered through the far bank, low-passed and levelled; user distance varied independently; background-first and background-only clips; an auxiliary background-mask head; a loss term against over-suppressing the user; a scale-aware loss for silent-target frames; 30–60 s sequences.
2. Fine-tune on AMI headset crosstalk (CC BY 4.0, real, 16 kHz band only).
3. Enrolled mode: FiLM conditioning with embedding dropout and the jointly trained enrollment encoder.
4. Wire fail-closed behaviour, device-class selection and the enrollment UI.
5. Only if the gates are still missed: distil from a large teacher, checking the teacher's own training data against the audit first. GAN fine-tune last.

## Done when

- Phase 0: the harness reproduces Krisp's published rows within 0.2 WER points and the gates are frozen.
- Clean: passes `[ci]` and `default_promotion`, beats DeepFilterNet3 on the manifest suite, runs inline at ≤ 40 ms end to end on macOS, and meets the browser gates in Chrome, Safari and Firefox.
- Isolate: passes `[voice_isolation.fault_injection]` and `[voice_isolation.opt_in]`, and meets the Krisp gates above with confidence intervals from at least two seeds.
- Release builds no longer need `ort`; `tract` goes when DeepFilterNet3 is retired.

## Risks

| Risk | What we do |
|---|---|
| Neither candidate beats DeepFilterNet3 at zero lookahead | Spend the spare compute budget on width; allow one frame of lookahead (32 ms chain, still under DeepFilterNet3's 40 ms) |
| Simulated distance cues do not transfer to real rooms | Private real validation set from Phase 0; AMI fine-tune; enrollment as the fallback mechanism |
| Too little clean fullband speech after filtering | Bandwidth-masked loss lets wideband speakers train everything below their cutoff; oversample VCTK |
| Isolate lands at 6–8% WER and stays there | Ship it opt-in for headsets and enrolled users on the `opt_in` gates; keep Nearfield as the comparison; distillation is the next lever |
| Tail latency in the worklet on older laptops | Paced p99 gate, overload bypass, and the smaller candidate size as a fallback tier |
| Safari and iOS audio quirks | Create the context after `getUserMedia`, verify the rate, test in real WebKit |
| A private inference engine is ours to maintain | Small layer set, scalar reference, parity tests on every weight file |

## Open questions for the merge

- Krisp NC access: which route, and who runs it.
- Hop 512 against plan 0002's 10 ms frame clock for echo cancellation.
- Whether ODbL and CDLA-Permissive data pass the audit.
- Budget ceiling for rented GPUs (research estimate: $100–350 for the project).

## Out of scope

- Training in Rust, WebGPU, threaded WASM, Windows.
- Speech Boost (separate plan).
- Accent conversion and other generative processing.
