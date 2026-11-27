# Training-data and evaluation-data audit

[← Docs index](README.md)

Which datasets Oto may train shipped weights on, which it may only evaluate on locally, and which it avoids. Written for Phase 5 of [plan 0002](plans/0002-e2e-roadmap.md) and [plan 0004](plans/0004-oto-v1-model-and-wasm.md) (research detail in [fable/research/data-and-compute.md](plans/fable/research/data-and-compute.md)), and binding on [`tools/eval/`](../tools/eval/README.md).

Licences are as published by each dataset at the time of writing. Re-check the licence file of the exact version downloaded and record it next to the data; if it differs from this page, the stricter reading wins until this page is updated.

## Permissive: trainable, and weights may be redistributed

| Dataset | Licence | Rate / bandwidth | Use | Notes |
|---|---|---|---|---|
| HiFiTTS-2 | CC-BY-4.0 metadata; PD LibriVox audio | 44.1 kHz (`≥ 13–16 kHz` filtered) | Clean fullband speech & speaker diversity | Main permissive fullband speech pool (4,631 speakers). Filter to `≥ 16 kHz` bandwidth, single speaker, DNSMOS/SNR-checked, capped per speaker; spot-check 50 chapters for MP3 top-band artefacts before bulk download |
| VCTK 0.92 | CC-BY-4.0 | 48 kHz true fullband | Studio fullband speech (110 speakers) | Download from Edinburgh directly; oversample for top-band quality; high-pass infrasound rumble |
| LibriTTS-R / LibriTTS | CC-BY-4.0 | 24 kHz (`≤ 12 kHz` bw) | Wideband speech / interferers | Use with effective-bandwidth loss masking above cutoff; watch for restoration artefacts in `-R` |
| LibriSpeech | CC-BY-4.0 | 16 kHz | Wideband speech / ASR WER | Bandwidth-masked only |
| Multilingual LibriSpeech (MLS) | CC-BY-4.0 | 16 kHz (OpenSLR) | Multilingual wideband speech | For fullband English, prefer HiFiTTS-2 (same LibriVox books at source rate) |
| Common Voice | CC0-1.0 (no re-identification) | Band-limited consumer mics | Wideband speaker diversity & interferers | Filter by validation votes & DNSMOS; do not count as fullband targets; do not attempt speaker identification |
| GLOBE / GLOBE_V2 | CC0-1.0 | Wideband (V2 is supersampled) | Accent/speaker diversity | Derived from Common Voice; not true fullband |
| AMI (headset channels) | CC-BY-4.0 | 16 kHz | Real near-field headset crosstalk | Critical for Isolate sim-to-real fine-tuning (16 kHz bandwidth-masked) |
| DiPCo | CDLA-Permissive-1.0 | 16 kHz | Real close-talk + far-field crosstalk | Attribution-only permissive licence |
| FSD50K (filtered) | Per-clip CC0 / CC-BY only | Fullband | Noise (~85% of clips pass filter) | Filter strictly via `*_clips_info_FSD50K.json`; drop CC-BY-NC and Sampling+; route human-voice classes to interferers; ship CC-BY attribution list |
| FMA (filtered) | Per-track CC0 / CC-BY only | Fullband | Music (~59 h passes filter) | Filter strictly via `tracks.csv` `license` column; drop SA, NC, ND |
| MUSAN (filtered) | CC-BY-4.0 / per-dir LICENSE | 16 kHz | Wideband noise / speech / babble | Check per-directory LICENSE files; drop NC/ND music tracks; not fullband |
| DNS Challenge 5 (restricted subsets only) | Per source | Mixed | `read_speech` (LibriVox PD), `vctk_wav48`, `VocalSet` (CC-BY), `freesound_*` noise (CC0, verify no DEMAND) | **Exclude** `audioset_*`, `emotional_speech` (CREMA-D ODbL), `PTDB-TUG` (ODbL), German (contains Spoken Wikipedia CC-BY-SA), Spanish (unconfirmed), and VoxCeleb2 |
| ACE Challenge | CC-BY-4.0 | 48 kHz | Real 48 kHz RIRs & room noise | Small (7 rooms) but true 48 kHz |
| Arni / dEchorate / AIR | CC-BY-4.0 / MIT | 44.1–48 kHz | Real fullband RIRs | Verify licence file on download; cap single-room sampling weight (Arni) |
| BUT ReverbDB | CC-BY-4.0 | Likely 16 kHz | Real room RIRs & retransmitted speech | Verify sample rate on download; bandwidth-mask if 16 kHz |
| OpenSLR SLR26 / SLR28 (`simulated_rirs/` only) | Apache-2.0 | **8 / 16 kHz only** | Simulated wideband RIRs | **Exclude `SLR28/real_rirs_isotropic_noises/`** (wraps research-only RWCP and unstated-licence REVERB). For 48 kHz, simulate RIRs directly with `pyroomacoustics` (MIT) |

Attribution for every dataset used goes in the model card and in `THIRD_PARTY_NOTICES.md` for the weights.

## Evaluation only: non-commercial

**Never** used to train shipped weights, and **never** committed to the repo or redistributed in any form (including derived mixtures and fixtures). They may be downloaded by a developer and used locally through `tools/eval/eval.py --suite manifest`.

| Dataset | Licence | Why we want it |
|---|---|---|
| Krisp Voice Isolation Benchmark | CC BY-NC 4.0 | 265 real 32 kHz recordings with Krisp outputs; primary background-voice leakage benchmark |
| EARS | CC BY-NC 4.0 | Fullband 48 kHz, expressive, anechoic speech |
| WHAM! / WHAMR! | CC BY-NC 4.0 | Realistic urban noise and speech-in-noise |
| DAPS / DDS | CC BY-NC 4.0 | Real consumer-device recordings in real rooms (44.1 kHz) |
| ESC-50 | CC BY-NC 3.0 | Labelled environmental sounds |

## Avoid or restricted

| Dataset / Pretrained Asset | Licence / Issue | Problem |
|---|---|---|
| DEMAND / QUT-NOISE / MicIRP | CC BY-SA | Share-alike: weights trained on it may inherit SA terms |
| CREMA-D / PTDB-TUG (in DNS5) | ODbL 1.0 | Database share-alike terms; not needed |
| AudioSet / VoxCeleb1 & 2 / YODAS | YouTube audio | Copyright stays with video uploaders; no clean audio licence. **Also excludes VoxCeleb-trained speaker encoders** (SpeechBrain ECAPA, WeSpeaker, NVIDIA TitaNet) as frozen teachers or shipped components |
| Unfiltered FSD50K / FMA | Mixed per clip | Only CC0 + CC-BY filtered subsets are allowed |
| SLR28 real RIRs (RWCP, REVERB) | Research-only / unstated | Upstream licences are narrower than the SLR28 bundle’s Apache-2.0 tag |
| MIT IR Survey / TAU Urban 2019 / GWA | Unstated / NC | Excluded for training |
| Korean AI-Hub corpora | Export-restricted | May not leave South Korea; one reason FastEnhancer’s weights cannot ship |

Anything not on this page is **avoid** until it is added here with its licence.

## Rules

**CI fixtures.** Everything in the repo and everything CI generates is synthetic (deterministic generators in `tools/eval/fixtures.py`) or CC0. No recorded speech under any other licence is committed, even short clips.

**Development and holdout splits.** Built only from the permissive set. Splits are **speaker-disjoint and session-disjoint** from each other and from training: no speaker, recording session or source recording appears on both sides. Noise and RIR files are split the same way. The split lists (file IDs, not audio) are versioned so results are reproducible.

**Effective-bandwidth masking.** Container sample rate is unreliable across permissive corpora. Estimate each clip’s effective bandwidth (URGENT / Hi-Fi TTS method) during shard packing, store `f_max` in shard metadata, and mask spectral losses above `0.9 × f_max` so sub-48 kHz clips never teach the model to muffle 48 kHz audio.

**External local benchmarks.** Non-commercial sets (Krisp VI, EARS, WHAM!, DAPS) are referenced by a local JSONL manifest pointing at files on the developer’s disk. Neither the audio nor the manifest’s paths are committed. Scorecards from them may be published as numbers. **Never select training checkpoints on the Krisp benchmark.**

**Model provenance.** For Oto-trained weights ([plan 0004](plans/0004-oto-v1-model-and-wasm.md)) and any newly adopted third-party model, `ship_in_release = true` in [`MODELS.toml`](../MODELS.toml) requires documented training data drawn only from the permissive table. Newly evaluated third-party weights with unknown or restricted training data (FastEnhancer today; Nearfield until checked) stay `ship_in_release = false`. Existing Apache-2.0 / MIT third-party weights already shipped before this audit (specifically Ceva’s DPDFNet-2 checkpoint, whose paper cites FSD50K without per-clip licence metadata) are grandfathered under their upstream Apache-2.0 weights licence until either Ceva’s FSD50K subset is cleared or Oto-1 replaces them.
