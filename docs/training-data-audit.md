# Training-data and evaluation-data audit

[← Docs index](README.md)

Which datasets Oto may train shipped weights on, which it may only evaluate on locally, and which it avoids. Written for Phase 5 (our own model) of [plan 0002](plans/0002-e2e-roadmap.md), and binding on the evaluation harness in [`tools/eval/`](../tools/eval/README.md) now.

Licences are as published by each dataset at the time of writing. Re-check the licence file of the exact version downloaded and record it next to the data; if it differs from this page, the stricter reading wins until this page is updated.

## Permissive: trainable, and weights may be redistributed

| Dataset | Licence | Use | Notes |
|---|---|---|---|
| LibriTTS-R | CC-BY-4.0 | Clean speech, 24 kHz | Attribution in the model card. Upsampled audio has no content above 12 kHz; mix with fullband sources for 48 kHz |
| LibriSpeech | CC-BY-4.0 | Clean speech, 16 kHz | Read speech; ASR transcripts for WER |
| Multilingual LibriSpeech (MLS) | CC-BY-4.0 | Multilingual clean speech | |
| Common Voice | CC0 | Varied speakers and microphones | Quality varies; filter by validation votes. Do not attempt to identify speakers |
| MUSAN | CC-BY-4.0 (check the per-part notes) | Noise, music, babble | Some music tracks carry their own licences; keep only parts confirmed CC-BY or more permissive |
| DNS Challenge 5 (permissive subsets only) | Per source | Speech, noise, RIRs | DNS5 is an aggregate. **Filter every file by its source dataset's licence**; drop any source that is non-commercial, share-alike or unstated |
| OpenSLR RIRs (e.g. SLR26, SLR28) | Apache-2.0 / CC-BY | Room impulse responses for reverb and dereverb | Check the licence on each OpenSLR resource page |

Attribution for every dataset used goes in the model card and in `THIRD_PARTY_NOTICES.md` for the weights.

## Evaluation only: non-commercial

**Never** used to train shipped weights, and **never** committed to the repo or redistributed in any form (including derived mixtures and fixtures). They may be downloaded by a developer and used locally through `tools/eval/eval.py --suite manifest`.

| Dataset | Licence | Why we want it |
|---|---|---|
| WHAM! | CC BY-NC 4.0 | Realistic urban noise and speech-in-noise |
| Krisp Voice Isolation Benchmark | CC BY-NC 4.0 | Background-voice leakage, directly comparable with commercial products |
| EARS | Non-commercial | Fullband, expressive, anechoic speech |
| ESC-50 | CC BY-NC 3.0 | Labelled environmental sounds |

## Avoid or restricted

| Dataset | Licence | Problem |
|---|---|---|
| DEMAND | CC BY-SA 3.0 | Share-alike: weights trained on it may be argued to be derivative and inherit SA terms. Do not train on it; prefer not to evaluate on it either |
| AudioSet | Mixed (YouTube, per clip) | Per-clip rights unknown and clips disappear; not reproducible |
| FSD50K | Mixed per clip (CC0, CC-BY, CC-BY-NC, Sampling+) | Usable only after filtering to CC0/CC-BY clips with the attribution kept; until that filter exists, treat as avoid |
| Korean AI-Hub corpora | Export-restricted | May not leave South Korea; one reason FastEnhancer's weights are not shipped |

Anything not on this page is **avoid** until it is added here with its licence.

## Rules

**CI fixtures.** Everything in the repo and everything CI generates is synthetic (deterministic generators in `tools/eval/fixtures.py`) or CC0. No recorded speech under any other licence is committed, even short clips.

**Development and holdout splits.** Built only from the permissive set. Splits are **speaker-disjoint and session-disjoint** from each other and from training: no speaker, recording session or source recording appears on both sides. Noise and RIR files are split the same way. The split lists (file IDs, not audio) are versioned so results are reproducible.

**External local benchmarks.** Non-commercial sets (Krisp VI, EARS, WHAM!) are referenced by a local JSONL manifest that points at files on the developer's disk. Neither the audio nor the manifest's paths are committed. Scorecards from them may be published as numbers.

**Model provenance.** For Oto-trained weights (Phase 5) and any newly adopted third-party model, `ship_in_release = true` in [`MODELS.toml`](../MODELS.toml) requires documented training data drawn only from the permissive table. Newly evaluated third-party weights with unknown or restricted training data (FastEnhancer today; Nearfield until checked) stay `ship_in_release = false`. Existing Apache-2.0 / MIT third-party weights already shipped before this audit (specifically Ceva's DPDFNet-2 checkpoint, whose paper cites FSD50K without per-clip licence metadata) are grandfathered under their upstream Apache-2.0 weights licence until either Ceva's FSD50K subset is cleared or Phase 5 replaces them.
