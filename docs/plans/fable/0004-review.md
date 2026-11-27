# Review of plan 0004 (Oto-1 model and WASM)

**Written:** 2026-11-27 · Reviews [0004](../0004-oto-v1-model-and-wasm.md) · Proposed replacement: [0004-fable-plan.md](0004-fable-plan.md)

Sources for every claim below are in [research/](research/): [krisp.md](research/krisp.md), [architectures.md](research/architectures.md), [wasm-runtime.md](research/wasm-runtime.md), [data-and-compute.md](research/data-and-compute.md). Those four files are raw notes from web research; anything they mark UNVERIFIED was not confirmed from a primary source. I re-read six pages myself: Krisp's SDK docs, the Krisp benchmark card, the DAPS and HiFiTTS-2 pages, and the FastEnhancer README.

## Verdict

Keep the direction, rewrite the plan. The idea (own permissive weights, 48 kHz, hand-written Rust inference that also runs in a browser worklet, small download) is sound and the research supports it. The plan as written fails on four points:

1. **"Beat Krisp" is never defined**, and the plan's latency target (21.3 ms) is behind Krisp's voice-isolation latency (15 ms).
2. **It builds everything at once.** A new architecture, a new speaker encoder, blind and enrolled isolation, dereverberation, a GAN stage, a new inference kernel and a new weight format all land in one training run. No published small model does all of that, and the projects that got isolation working needed many training iterations.
3. **Several facts are wrong**: a proposed dataset is non-commercial, two approved RIR sets are 16 kHz, the compute figure does not follow from the layer list, and the browser budget and latency are miscounted.
4. **"One overnight run" is not a training plan.** The one documented 48 kHz retrain of a comparable model took about 240 A100-hours.

## What Krisp is

The plan treats Krisp as one thing. It is three, and only one has public outputs to score against.

| | Krisp NC (noise) | Krisp BVC (on-device isolation) | Krisp VI 2.5 (server isolation) |
|---|---|---|---|
| Model rate | 32 kHz at most | 32 kHz | 16 kHz (one 32 kHz variant) |
| Algorithmic latency | about 25 ms (10 ms frames, 16 kHz) | 15 ms | 15 ms |
| Model file | 2.8–5.8 MB | 8.6–14.7 MB | not published |
| CPU | not published in text | 5–13% | about 1.5–3 ms per 20 ms frame |
| Enrollment | none | none | none |
| Devices | near field, under 50 cm | headsets on an allow list; built-in laptop mics "might demonstrate acceptable performance" | n/a |
| Public outputs to score | none | none | yes, on Krisp's own benchmark |

Three facts follow from this table:

- **Krisp has no 48 kHz model.** Its docs say rates above 32 kHz are downsampled. Fullband is a spec Oto can win, though the audible gain over 32 kHz is small; the audible gain is over the 16 kHz open models.
- **Krisp's browser SDK is a 12 MB package** using about 100 MB of memory and 1.5–2 ms per 10 ms frame. A sub-1 MB download at lower CPU is a clear, measurable win.
- **Krisp's isolation is headset-only and enrollment-free.** Oto runs mostly on built-in MacBook mics, which is the case Krisp does not support. Enrollment is the one mechanism shown to work there (see point 5 below).

The Krisp Voice Isolation Benchmark is 265 real recordings, English, 32 kHz, CC BY-NC 4.0, with no clean references. Scores are word error rate and leaked background words per minute. Krisp VI 2.5 Default scores 5.30% and 1.52 with the Parakeet recogniser; unprocessed audio scores 23.21% and 66.48; DeepFilterNet3 scores 13.30% and 22.77. Open models claiming a tie are either 27M parameters and most of a CPU core (VoxLock), or tuned on the benchmark itself at 48–80 ms latency (Nearfield, which Oto already carries).

There is no public way to score against Krisp NC. That comparison needs Krisp access: an SDK trial, the desktop app's virtual mic, or LiveKit Cloud's Krisp filter.

## What the plan gets right

| Plan item | Evidence |
|---|---|
| Hand-written Rust inference instead of `ort` or `tract` | Every browser denoiser under 3 MB is hand-written. `tract` in wasm is 9–15 MB and ONNX Runtime Web 11–14 MB |
| STFT 1024 / hop 512, no lookahead | Identical to FastEnhancer's 48 kHz models. Lookahead buys little for denoising (+0.01 PESQ from 12.5 to 25 ms in one ablation) |
| Complex processing in the low band, band gains above | This is DeepFilterNet's layout, confirmed by three later papers |
| One checkpoint with a mode flag and a zeroed embedding | Works in two published systems |
| Loss masking above each clip's real bandwidth | The URGENT challenges do this. Almost all permissive speech has less bandwidth than its container |
| Benchmark random weights in the browser before training | Correct order; the compute budget should be set by measurement |
| Allocation-free kernel that can run in the audio callback | Removes the worker's 7.3 ms reserve, about 36 ms end to end against 62 ms today |
| Training stays in PyTorch | No reason to change |

## What is wrong or unsupported

**1. The target is undefined.** "Competitive with Krisp on the local benchmark" names no metric, no threshold and no Krisp product. See the gate table in the [proposed plan](0004-fable-plan.md).

**2. The latency does not beat Krisp's.** 21.3 ms against 15 ms for Krisp BVC and VI. It does beat Krisp NC's 25 ms. Reaching 15 ms at the same hop needs a shorter synthesis window than analysis window. I know that technique from the literature but did not research it in this pass, so the proposed plan treats it as an experiment.

**3. The compute figure is inconsistent, in the harmless direction.** By my count the listed layers cost at least 150 MMAC/s under either reading of the GRU, not 50–100. But the real budget is far looser: FastEnhancer-S at 48 kHz costs 1.82 GMAC/s and runs at 4.2% of one M5 thread, and DeepFilterNet3 (0.34 GMAC/s) runs in wasm at 0.72 ms mean per frame on a 2020 i7. The plan under-spends compute by roughly 5–10x, which costs quality.

**4. Blind isolation at 48 kHz with about 1M parameters has not been shown by anyone.** Every open isolation model is 16 kHz. Nearfield started at 48 kHz and dropped to 16 kHz. All of them report the same failures: a background voice that speaks first passes through, equal-level talkers cannot be separated, a user far from the mic gets muted, and a lone user's speech is damaged. Krisp's own models make clean phone calls slightly worse than raw.

**5. The speaker encoder is the least supported part.** Small published systems use a frozen pretrained 192-dimension encoder, not a tiny jointly trained 64-dimension one. Every pretrained encoder found was trained on VoxCeleb, which fails Oto's data audit. Five seconds of enrollment is also short. Enrollment is still worth doing: one open model suppresses an overlapping voice by 1.4 dB blind and 33.8 dB with a voiceprint.

**6. Data errors.**

| Plan says | Finding |
|---|---|
| "VCTK / DAPS" as fullband sources | DAPS is CC BY-NC 4.0. VCTK is fine (CC BY 4.0, 110 speakers) |
| OpenSLR SLR26 and SLR28 RIRs | Both are 8/16 kHz. SLR28's real RIRs come from sources with research-only or unstated licences |
| Licence-filtered DNS5 subsets | No per-file licence manifest was found for the noise; clean speech includes ODbL and VoxCeleb2 material. Only coarse filtering is possible |
| LibriTTS-R, MLS, Common Voice as main speech | None is fullband. The main fullband permissive source is HiFiTTS-2 (LibriVox audio, 31.7k hours at 13 kHz bandwidth or more, per-chapter bandwidth labels), which the plan does not mention |
| MUSAN noise | 16 kHz. FSD50K filtered to CC0 and CC-BY clips (about 85% of clips) is a better fullband source |

FastEnhancer's 48 kHz weights are also worse off than Oto's notes say: besides AI-Hub, they trained on DEMAND (share-alike), WHAM (non-commercial) and TUT (non-commercial). Shipping that architecture means a retrain.

**7. Training cost.** One overnight run covers a denoise-only baseline or a fine-tune. Projects that got isolation working went through four to ten recipe changes, and long training sequences (30–60 s) are needed to stop models fading continuous talkers. Plan for 5–10 runs. Google Drive as the shard store hits download-quota lockouts; Colab sessions end at 24 hours and GPUs are not guaranteed.

**8. Browser numbers.**

- "≤ 3 ms per hop" is wrong: one hop's work lands inside a single 2.67 ms render quantum.
- "About 21 ms in the browser" counts only the STFT. The extra AudioContext hop adds roughly 30–45 ms (the 30–45 figure is the research agent's estimate; one project measured about 21 ms).
- `wasm-bindgen` glue still breaks in worklets. Plain `extern "C"` exports with no imports is the robust pattern.
- `std::simd` and `f16` are unstable on Rust 1.99. Use `core::arch` intrinsics.
- Benchmarks that race through a file understate paced cost; one port measured 4.2x.

**9. The parity rule contradicts itself.** "Bit-for-bit to ≤ −60 dB" is neither. Float ports reach about 1e-7 maximum difference; int8 ports sit at −38 to −53 dB.

**10. Missing from the training recipe:** a loss term against over-suppressing the user, a scale-aware loss for frames where the target is silent (SI-SDR is undefined there), far-field rendering of interferers, user distance varied independently of interferer distance, and real near-field recordings for validation. No paper found shows a single-mic distance model trained only on simulated rooms transferring to real ones.

**11. Smaller points.**

- The GAN stage gains about +0.05 PESQ in the closest ablation and is fragile. Make it optional and last.
- int8 weights cost 0.08–0.3 PESQ on GRU models unless done carefully. Ship f16 first.
- The 512-sample hop conflicts with the "single 10 ms frame clock" in plan 0002 Phase 3. One of them has to give.
- Speech Boost is independent of the model. It should be its own plan so neither blocks the other.

## What I did not verify

- Krisp's CPU tables are images and were not read.
- The mapping between Krisp's blog model names and its model file names is partly inferred.
- How many HiFiTTS-2 hours survive a 16 kHz bandwidth and quality filter is an estimate (500–1,000 hours), and MP3 artefacts in the top band are unchecked.
- Current Colab unit burn rates conflict across sources (5.4–10.6 units per A100-hour).
- My MMAC recount of the 0004 layer list is a back-of-envelope figure, not a profiler run.
