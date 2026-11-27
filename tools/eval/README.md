# tools/eval: Oto's evaluation harness

Runs Oto's real DSP chain offline (`oto process`), scores the output and checks it against
[`gates.toml`](gates.toml). Part of Phase 1 Track A in
[plan 0002](../../docs/plans/0002-e2e-roadmap.md). Every run also checks the SHA-256 hashes
in [`MODELS.toml`](../../MODELS.toml) and writes `scorecard.md` and `report.json` to `--out`
(default `tools/eval/out/`, git-ignored). It exits 0 when every gate passes, 1 when a gate
fails and 2 on a usage or environment error.

The CI suite needs only Python 3.11+ (stdlib). No network access and no downloaded data.

```bash
cargo build --release -p oto-cli                     # or a debug build; OTO_BIN / --oto pick another
python3 tools/eval/eval.py --suite ci                # rnnoise, dfn3, dpdfnet2 × light/balanced/strong
python3 tools/eval/eval.py --suite ci --models fastenhancer --strengths balanced
python3 tools/eval/eval.py --suite ci --bench        # adds `oto bench --json` (p50/p99/p99.9/RTF/CPU)
python3 tools/eval/eval.py --bench --live --models dfn3   # real engine (needs the driver)
python3 -m unittest discover -s tools/eval -v        # the harness's own tests
```

With `uv`, `uv run --project tools/eval tools/eval/eval.py …` works the same. Extras:
`--extra metrics` (numpy, scipy), `--extra perceptual` (PESQ, onnxruntime for DNSMOS),
`--extra asr` (faster-whisper), `--extra parakeet` (Parakeet TDT 0.6B v2 via onnx-asr; numpy +
onnxruntime, no torch), `--extra nemo-tn` (NeMo text normalisation, Linux-first).

## CI suite

[`fixtures.py`](fixtures.py) writes deterministic synthetic WAVs (no recorded audio, so they
can be redistributed): a speech-like voice (clean, and mixed with pink noise and hum at 15 and
5 dB SNR), noise only, an impulse-and-step alignment file, digital silence, and a 16 kHz
phone-band (HFP) variant. For each model, the harness:

1. Runs the alignment file with `--no-denoise --no-delay-comp` and checks that the measured
   delay equals the chain latency in `MODELS.toml` (`frame_samples + latency_samples`).
2. Runs every fixture bypassed once, then at each strength.
3. Measures NaN/inf, clipping and peak, output length, silence in → silence out, the level
   drop in noise-only windows (what the listener hears, `*_attenuation_db`, and the model alone
   against the bypassed chain, `*_model_attenuation_db`), speech-level change, SI-SDR before
   and after, and a coarse delay from the energy envelope.
4. Checks the `[ci]`, `[ci.<strength>]`, `[models.<m>.ci]` and `[models.<m>.ci.<strength>]`
   gates, in that order of precedence. The strength caps (Light 12 dB, Balanced 24 dB) are
   checked from above on the model's own attenuation.

The fixture voice is not speech, so the CI bars are sanity bars, not quality scores. Quality
comes from the manifest suite on real recordings.

`oto process`'s command line comes from a template, so a CLI change needs no code change:

```bash
--process-template '{oto} process {input} {output} --model {model} --strength {strength} {bypass} --no-delay-comp'
```

(also `$OTO_PROCESS_TEMPLATE`). `--processor passthrough` (a bypassed chain) and
`--processor oracle` (a perfect denoiser on inputs with a clean reference) test the harness
without a binary.

## Manifest suite: local datasets

```bash
python3 tools/eval/eval.py --suite manifest --manifest ~/data/oto-holdout.jsonl \
    --gates-profile voice_isolation.opt_in [--asr] [--dnsmos-model ~/models/sig_bak_ovr.onnx]
```

One JSON object per line. Relative paths are relative to the manifest file. Lines starting
with `#` are comments.

```json
{"id": "vi-0001", "noisy": "mix/0001.wav", "clean": "clean/0001.wav", "interferer": "bg/0001.wav", "transcript": "please call me back today", "interferer_transcript": "turn on the tv", "kind": "speech", "noise_windows": [[0.0, 0.8]], "dataset": "permissive-holdout", "license": "CC-BY-4.0"}
{"id": "noise-0001", "noisy": "noise/fan.wav", "kind": "noise_only"}
{"id": "bg-0001", "noisy": "bg/tv-only.wav", "kind": "background_only"}
```

Only `noisy` is required. What each field enables:

| Field | Metrics |
|---|---|
| `clean` (same rate) | SI-SDR in, out and delta; SNR; PESQ-WB with `perceptual` |
| `noise_windows`, or `kind` `noise_only` / `background_only` | quiet-part attenuation |
| `interferer` (same rate) | overlap leakage: interferer energy kept, relative to the input |
| `transcript` + `--asr` | WER, WER delta against the unprocessed input, first-word deletion |
| `interferer_transcript` + `--asr` | wrong-speaker rate |
| `kind: background_only` + `--asr` | leaked words per minute |
| `kind: noise_only` + `--asr` | hallucinated words per minute, next to the ASR's own rate on the raw input |
| `--dnsmos-model` | DNSMOS P.835 SIG / BAK / OVR |
| `segments` + `speaker_id` + `--asr` | Krisp scoring: pooled WER, missed caller words, secondary-segment leak and noise-segment hallucinations per minute, speaker-bootstrap 95% CIs (`krisp.py`) |
| `baselines` (`{name: wav}`) + `--asr` | the same Krisp metrics for each baseline, and paired `*_delta_vs_<name>` intervals; with `--dnsmos-model`, `dnsmos_*_delta_vs_<name>` |
| fullband input (≥ 44.1 kHz, content > 16 kHz) | `high_band_16k_db` and `high_band_16k_delta_db`: the 48 kHz fullband check |

### Same-ASR head-to-head

`--asr` loads one faster-whisper model (`--asr-model`, default `small.en`) with one fixed
configuration (beam 5, temperature 0, VAD off so hallucinations are counted, no conditioning
on previous text). It transcribes the unprocessed input and every model × strength output with
that configuration, which is recorded in the report. Compare rows within one report only.
Never compare absolute WER across ASR systems, models or configurations. The noise-only rows
calibrate the ASR: its hallucination rate on the raw noise is the floor no denoiser can be
blamed for.

`speech_onset_miss_rate` and end-to-end latency ceilings come from saved `oto latency --stimulus speech --json` files passed via `--latency-report PATH` (repeatable):

```bash
oto latency --stimulus speech --tier standard --json > latency-dfn3.json
python3 tools/eval/eval.py --latency-report latency-dfn3.json [--gates-profile voice_isolation.opt_in]
```

`--asr-backend parakeet` uses NVIDIA Parakeet TDT 0.6B v2 (the Krisp benchmark's recogniser)
through onnx-asr and returns word timestamps. Use the default fp32 weights for anything compared
with published numbers: the int8 export drops quiet background talkers (on one Krisp clip it
transcribed none of the 38 background words the fp32 model and NeMo transcribe).

### Krisp Voice Isolation Benchmark (plan 0004, Phase 0)

[`krisp_bench.py`](krisp_bench.py) fetches the benchmark and Krisp's published VI 2.5 outputs
from Hugging Face (stdlib HTTP, `$HF_TOKEN` optional) into `--data-dir` (default
`~/.cache/oto-eval/krisp`; paths inside the repo are refused), checks free disk first
(`--reserve-gb`, default 20), builds an eval.py manifest, transcribes with Parakeet, and scores.

```bash
python3 tools/eval/krisp_bench.py download --dry-run        # 1.94 GB: raw + VI 2.5 Default
uv run --project tools/eval --extra parakeet tools/eval/krisp_bench.py reproduce
uv run --project tools/eval --extra parakeet tools/eval/krisp_bench.py eval --models dfn3,dpdfnet,nearfield
python3 tools/eval/krisp_bench.py score --clips krisp.clips.jsonl --check   # re-pool published per-clip counts
```

`reproduce` exits 1 unless pooled WER is within 0.2 points of the published rows (raw 23.21%,
Krisp VI 2.5 Default 5.30%). `eval` runs eval.py with `--gates-profile oto1.isolate`; Krisp's
output is the `krisp_default` baseline, so the "not significantly worse WER / significantly lower
leak" gates are paired speaker-bootstrap intervals over the 47 speakers. Scoring follows Krisp's
methodology (regex rules + NeMo TN + jiwer transform, pooled WER) and the VoxLock scorer's leak
and missed definitions; see the docstring of [`krisp.py`](krisp.py). Without `nemo-tn` a stdlib
verbaliser stands in for NeMo TN (reported as `krisp_normalizer` in the report).

### Fault injection

`--fault-report engine-faults.json` checks the engine's Voice Isolation fault tests against
`[voice_isolation.fault_injection]`: every case × strength must be present with zero leaked
samples and frames and no fallback to another model.

```json
{"model": "nearfield_v2_3_fast", "cases": [{"case": "starvation", "strength": "balanced",
  "leaked_samples": 0, "leaked_frames": 0, "fallback_to_other_model": 0}]}
```

## Licence boundaries

From [the training-data audit](../../docs/training-data-audit.md):

- **In the repo and in CI:** synthetic fixtures from `fixtures.py`, or CC0 audio. Nothing
  else. `*.wav` is git-ignored at the root.
- **Permissive holdouts** (LibriTTS-R, LibriSpeech, MLS, Common Voice, MUSAN, licence-filtered
  DNS5, OpenSLR RIRs) can be used for development and holdout splits. Keep the splits
  speaker-disjoint and session-disjoint, and from training data too.
- **Non-commercial benchmarks** (Krisp Voice Isolation Benchmark, EARS, WHAM!, ESC-50) are for
  local evaluation only. Keep their audio and their manifests out of the repo. Publishing the
  resulting numbers is fine. Never train shipped weights on them.
- **Avoid:** DEMAND (share-alike), AudioSet, unfiltered FSD50K, export-restricted corpora.

## Files

| File | What it is |
|---|---|
| `eval.py` | CLI: suites, processors, gate checks, reports |
| `fixtures.py` | Synthetic CI fixtures (`python3 tools/eval/fixtures.py OUT_DIR`) |
| `metrics.py` | RMS, peak, SNR, SI-SDR, windowed attenuation, delay estimation |
| `asr.py` | WER alignment, first-word deletion, words per minute, faster-whisper and Parakeet (onnx-asr) backends |
| `krisp.py` | Krisp benchmark scoring: normalisation, pooled WER, missed, leak, speaker bootstrap |
| `krisp_bench.py` | Krisp benchmark CLI: download, manifest, ASR cache, score, reproduce, eval |
| `wavio.py` | Stdlib WAV reader and writer (PCM 8/16/24/32, float 32/64, extensible) |
| `gates.toml` | Gates for the CI suite, per model, Voice Isolation, and Oto-1 (`[oto1.clean]`, `[oto1.isolate]`) |
| `test_eval.py` | Tests, including the `MODELS.toml` hash check |
