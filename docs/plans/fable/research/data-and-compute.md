Raw research notes, 2026-11-27; claims marked UNVERIFIED were not confirmed from a primary source

# Data and compute research: causal 48 kHz denoise + dereverb + background-voice cancellation

Scope: which training data can be used when the weights will be redistributed under MIT OR Apache-2.0 (rule: no NonCommercial, no ShareAlike, no unstated licences, no export-restricted corpora), and what compute/storage setup is realistic.

Conventions:

- "Primary" = I read the licence page, dataset card, Zenodo record or official repo in this session.
- "Secondary" = web search summary of third-party pages; treat as a lead, not as fact.
- UNVERIFIED = not confirmed from a primary source (or the primary source was silent).
- "Fullband" here means real spectral content above 16 kHz, not just a 44.1/48 kHz container.

---

## 0. Headline findings

1. **The draft's "VCTK / DAPS" line is half wrong.** VCTK 0.92 is CC BY 4.0 (fine). DAPS is **CC BY-NC 4.0** (not allowed). DDS (device-degraded DAPS/VCTK re-recordings) is also CC BY-NC 4.0.
2. **True-fullband permissive speech is not scarce if LibriVox is re-derived at source quality.** HiFiTTS-2 (NVIDIA, CC BY 4.0 metadata over public-domain LibriVox audio) gives 31.7k h / 4,631 speakers at >= 13 kHz estimated bandwidth; a stricter >= 16 kHz filter on the same source gave 12.8k h / 2,776 speakers in a published codec paper. The caveat is quality: MP3 source, amateur rooms, variable noise floor. Studio-grade permissive fullband is small: VCTK (110 speakers, ~44 h per mic) and Hi-Fi TTS (10 speakers, 292 h).
3. **Several "approved" items are 16/24 kHz, not fullband:** LibriTTS-R (24 kHz), LibriSpeech (16 kHz), MLS as distributed on OpenSLR (16 kHz, UNVERIFIED rate; from memory), SLR26 and SLR28 RIRs (8/16 kHz, primary). They are fine for speaker diversity and low-band learning but cannot teach the 8 to 24 kHz band.
4. **SLR28 is a real licence risk.** Its "Apache 2.0" label covers a bundle whose real RIRs come from RWCP (research-and-development-only per NII/OpenSLR SLR13, secondary), the REVERB challenge (no licence found) and AIR (MIT). Use only the `simulated_rirs` part, or better, simulate at 48 kHz yourself.
5. **DNS5 is a mixed bag.** Noise = AudioSet (YouTube audio; only labels are CC BY 4.0) + Freesound CC0 selection + DEMAND (CC BY-SA 3.0). The archives are split `audioset_*` / `freesound_*` (primary: download script), so a coarse filter is possible, but where DEMAND sits and whether per-file metadata exists is UNVERIFIED. Clean speech includes ODbL parts (CREMA-D, PTDB-TUG), a German set built partly from Spoken Wikipedia (CC BY-SA, UNVERIFIED licence), and M-AILABS material that is 16 kHz at source.
6. **Every off-the-shelf speaker encoder checked was trained on VoxCeleb** (YouTube audio, copyright with the video owners; official page now says URLs/audio are no longer available there). Under the stated rule this is a provenance problem. Cheapest clean route: no external encoder, i.e. condition directly on the enrollment clip (embedding-free TSE) or train a small encoder jointly on permissive speech.
7. **Colab on Google AI Ultra is workable but tight and unreliable for 100 to 300 GPU-hours.** 1,000 (Ultra 5x) or 2,000 (Ultra 20x) compute units per month; one VM is hard-capped at 24 h; GPU type is not guaranteed; A100 burn-rate reports range from 5.4 to ~10.6 units/h. A rented RTX 4090 (about $0.35 to $0.90/h) is cheaper per hour, has no 24 h wall, and is enough for a model of this size.
8. **Do not use Google Drive as the shard store.** Use Cloudflare R2 (zero egress, about $15/TB-month) or a Hugging Face private dataset repo (100 GB free, 1 TB on the $9 PRO plan).

---

## 1. Fullband clean speech

### 1.1 Per-corpus table

| Corpus | Licence (exact string where read) | Rate / real bandwidth | Hours / speakers | Verdict | Source |
|---|---|---|---|---|---|
| VCTK 0.92 | "Creative Commons License: Attribution 4.0 International" (license_text.txt, primary) | Recorded 96 kHz/24-bit, released 48 kHz/16-bit; hemi-anechoic chamber; 2 mics (DPA 4035 omni, Sennheiser MKH 800) | 110 speakers, ~400 sentences each; ~44 h per mic (~80 h both mics per URGENT); p280/p315 have MKH 800 issues | OK. True fullband. Known issues: low-frequency rumble/room tone (URGENT applies a high-pass by default because "some sources have high-energy infrasound noise"), and URGENT 2024 found noisy samples in VCTK | https://datashare.ed.ac.uk/handle/10283/3443 , https://datashare.ed.ac.uk/bitstream/handle/10283/3443/license_text.txt |
| DAPS | "Creative Commons Attribution Non Commercial 4.0 International" (primary) | 44.1 kHz (UNVERIFIED on page) | 20 speakers, ~4.5 h per version, 15 versions | **Excluded (NC).** Eval only | https://zenodo.org/records/4660670 |
| Hi-Fi TTS (SLR109) | "CC BY 4.0" (primary) | 44.1 kHz; selected for bandwidth >= 13 kHz and SNR >= 32 dB | 291.6 h, 10 speakers (>= 17 h each) | OK but adds almost no speaker diversity. LibriVox MP3 source | https://www.openslr.org/109/ , https://arxiv.org/abs/2104.01497 |
| HiFiTTS-2 | "Creative Commons Attribution 4.0 International License (CC BY 4.0)" for the metadata repo; audio is LibriVox public domain, **not redistributed**, fetched from archive.org by the NeMo SDP script (primary) | 44.1 kHz subset: estimated bandwidth >= 13 kHz ("removes all audio recorded at a sampling rate of 24 kHz or lower"); per-utterance bandwidth in the manifest | 22.05 kHz set: ~36.7k h, 5,013 speakers. 44.1 kHz set: ~31.7k h, 4,631 speakers. ~104 h flagged multi-speaker by diarization (tagged, not removed). Full 44 kHz set needs ~4.0 TB | **Best single source for speaker diversity at high bandwidth.** Take a subset: filter bandwidth >= 16 kHz (or >= 20 kHz), single-speaker, then SNR/DNSMOS filter, cap hours per speaker | https://huggingface.co/datasets/nvidia/hifitts-2 , https://arxiv.org/html/2506.04152 |
| LibriVox at >= 16 kHz bandwidth (same source, different filter) | LibriVox public domain | 44.1 kHz, bandwidth >= 16 kHz by the Hi-Fi TTS estimator | 12.8k h, 3.1M utterances, 2,776 speakers | Evidence for how much LibriVox is really fullband | https://arxiv.org/html/2406.05298v2 |
| DNS5 `read_speech` (LibriVox) | README: public domain, https://librivox.org/pages/public-domain/ (URGENT table lists it as CC BY 4.0) | Nominal 48 kHz; URGENT 2024: ~100% of DNS5 LibriVox samples had effective bandwidth below Nyquist | 299 GB unpacked; ~350 h after URGENT 2025 filtering, ~150 h in the URGENT 2026 700 h selection | OK licence. Must be bandwidth-filtered. Largely superseded by HiFiTTS-2, which has speaker IDs | https://github.com/microsoft/DNS-Challenge , https://urgent-challenge.github.io/urgent2025/data/ , https://arxiv.org/html/2506.01611v1 |
| DNS5 `vctk_wav48_silence_trimmed` | README says "Open Data Commons Attribution License (ODC-By) v1.0" (the VCTK 0.80 licence); the directory name matches VCTK 0.92, which is CC BY 4.0 | 48 kHz, true fullband | 27 GB | OK either way (both attribution-only). Prefer downloading 0.92 from Edinburgh directly | https://github.com/microsoft/DNS-Challenge |
| DNS5 `emotional_speech` (CREMA-D) | ODbL 1.0 for the database + DbCL 1.0 for contents (official repo, secondary); DNS README cites DbCL | 16 kHz source (secondary), 91 actors, 7,442 clips | 2.4 GB | **Exclude:** ODbL has share-alike terms for derivative databases and is not fullband anyway | https://github.com/CheyneyComputerScience/CREMA-D |
| DNS5 `VocalSet_48kHz_mono` | "Creative Commons Attribution 4.0 International" | 44.1 kHz singing | ~10 h, 20 singers | OK, optional (singing) | https://zenodo.org/record/1193957 |
| DNS5 PTDB-TUG | ODbL 1.0 (per DNS README) | 48 kHz, 20 speakers (UNVERIFIED) | small | **Exclude** (ODbL share-alike) unless a lawyer says ODbL "produced work" terms are acceptable | https://github.com/microsoft/DNS-Challenge |
| DNS5 French / Italian / Russian (M-AILABS) | BSD-style: "Redistribution and use in any form, including any commercial use, with or without modification are permitted" with notice retention and no-endorsement clauses (secondary quote of the licence text) | M-AILABS is 16 kHz per the Hi-Fi TTS comparison table, so these "fullband" folders are upsampled (UNVERIFIED for the DNS copies) | FR 190 h, IT 128 h, RU 47 h (Interspeech 2021 DNS paper, secondary) | Licence OK, **not fullband**. Useful only as multilingual wideband speech | https://github.com/i-celeste-aurora/m-ailabs-dataset , https://arxiv.org/pdf/2101.01902 |
| DNS5 German | Four sources: Spoken Wikipedia Corpora, Telecooperation German Corpus for Kinect, M-AILABS, zamia-speech forschergeist (secondary). Spoken Wikipedia is CC BY-SA (UNVERIFIED) | mixed | 636 h (secondary) | **Risky**: likely contains ShareAlike material that cannot be separated without per-file provenance. Exclude | https://arxiv.org/pdf/2101.01902 |
| DNS5 Spanish | Source not confirmed in this session | UNVERIFIED | 65 GB | Unknown provenance: exclude until confirmed | - |
| DNS5 VoxCeleb2 (listed as a clean speech source in the README) | "CC BY 4.0 ... The copyright remains with the original owners of the video" | 16 kHz | - | **Exclude** (YouTube audio) | https://github.com/microsoft/DNS-Challenge |
| Common Voice (Scripted Speech 26.0 English) | CC0-1.0 on Mozilla Data Collective, with a no-re-identification condition; some MDC bundles forbid re-hosting outside MDC (secondary) | 48 kHz MP3 (rate not stated on MDC; UNVERIFIED), lossy; URGENT 2024: ~100% of CV 11 English clips had effective bandwidth below Nyquist | English: 3,780.95 h (2,784.88 validated), 100,172 speakers | OK licence. Excellent speaker diversity and real consumer mics, but noisy and band-limited. Use as **wideband speaker-diversity and interferer** material, filtered with DNSMOS; do not count as fullband targets | https://mozilladatacollective.com/datasets/cmqim2hn800ssnr07gvmpcnwu |
| MLS | "CC BY 4.0" (primary; URGENT table says CC0) | OpenSLR FLAC/Opus, 16 kHz (UNVERIFIED rate; page silent). URGENT re-downloaded "a less compressed version from LibriVox" | English 2.4 TB FLAC; EN ~44.7k h / 5,574 speakers (via HiFiTTS-2 paper) | OK, but the distributed files are wideband only. For fullband use HiFiTTS-2 (same books, source-rate audio) | https://www.openslr.org/94/ |
| LibriTTS-R / LibriTTS | CC BY 4.0 | 24 kHz (12 kHz bandwidth max); URGENT 2024: ~25% of LibriTTS samples have less bandwidth than the 24 kHz container, some only 4 kHz | 585 h, ~2.4k speakers | OK, not fullband. LibriTTS-R is model-restored speech, so its "clean" targets carry generative artefacts (UNVERIFIED how audible) | https://arxiv.org/html/2506.01611v1 |
| LibriSpeech | CC BY 4.0 | 16 kHz | 960 h, ~2.4k speakers | OK, wideband only | - |
| GLOBE / GLOBE_V2 | "cc0-1.0" (primary card) | V1 24 kHz; V2 "44.1kHz sample rate. (Supersampling)" i.e. upsampled/super-resolved, not native | 535 h, 23,519 speakers, 164 accents (V1) | Licence OK. **Not real fullband** (derived from Common Voice with enhancement). Good for accent/speaker diversity at wideband | https://huggingface.co/datasets/MushanW/GLOBE_V2 |
| HUI-Audio-Corpus-German | CC0 (secondary: paper table) | 44.1 kHz, LibriVox source | 326 h, 122 speakers (5 speakers hold most hours; other 117 share 97 h) | Probably OK; verify licence file at download. German fullband | https://arxiv.org/pdf/2106.06309 , https://github.com/iisys-hof/HUI-Audio-Corpus-German |
| Thorsten-Voice | CC0 (secondary) | 22.05 kHz releases; "TV-44kHz-Full" ~40 h at 44 kHz | 1 speaker | OK, negligible diversity | https://huggingface.co/datasets/Thorsten-Voice/TV-44kHz-Full |
| AISHELL-3 (SLR93) | Apache License v2.0 (secondary: OpenSLR + paper) | Recorded 44.1 kHz/16-bit; distributed rate UNVERIFIED | ~85 h, 218 speakers, Mandarin | Probably OK; verify rate and that it is not export-restricted | https://www.openslr.org/93/ |
| NNCES (children) | CC0 on Kaggle (secondary) | 44.1 kHz stereo | "20 h" on Kaggle vs ~3 h in the paper's tables; 20 children | Small, optional; verify | https://www.kaggle.com/datasets/kodaliradha20phd7093/nonnative-children-english-speech-nnces-corpus |
| YODAS v3 | "CC BY 3.0" in the arXiv abstract; `cc-by-4.0` tag on the HF card (conflict) | 48 kHz stereo; stored from WebM/Opus so real bandwidth varies (secondary) | >1.1M h, 147 languages, weak labels, no speaker labels mentioned | **Risky**: YouTube audio whose CC licence is self-declared by uploaders; no speaker IDs; quality unknown. Not recommended for clean targets | https://arxiv.org/abs/2609.29448 , https://huggingface.co/datasets/espnet/yodas3 |
| Emilia | Original Emilia CC BY-NC 4.0; Emilia-YODAS CC BY 4.0 (secondary) | 24 kHz | 101k h + 114k h | Exclude original (NC); YODAS part has the same provenance concern as above | https://huggingface.co/datasets/amphion/Emilia-Dataset |
| EARS | CC BY-NC 4.0 (secondary: paper + repo) | 48 kHz anechoic | 100 h, 107 speakers | **Excluded (NC).** Eval only | https://github.com/facebookresearch/ears_dataset |
| Expresso | CC BY-NC 4.0 (secondary: project page) | 48 kHz | ~40 h, 4 speakers | Excluded (NC) | https://speechbot.github.io/expresso/ |
| JSUT / JVS | Audio: academic research, non-commercial research, personal use only; redistribution prohibited (secondary: official pages) | 48 kHz / 24 kHz | - | Excluded | https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus |
| LJSpeech | Public domain | 22.05 kHz, 1 speaker | 24 h | OK, not fullband, no diversity | - |
| Libriheavy | Apache-2.0 on the repo vs CC BY 4.0 in third-party reports (conflict) | 16 kHz | 50k h | OK-ish, wideband only, redundant with LibriVox | https://github.com/k2-fsa/libriheavy |
| People's Speech | CC-BY and CC-BY-SA subsets (HF configs `cc-by-*` and `cc-by-sa-*`) | distributed 16 kHz (UNVERIFIED) | ~27.7k h CC-BY/PD + ~3.1k h CC-BY-SA | CC-BY configs only; wideband; noisy | https://huggingface.co/datasets/MLCommons/peoples_speech |
| VoxPopuli | CC0 (secondary) | 16 kHz (UNVERIFIED) | 400k h unlabelled | OK, wideband, far-field parliament audio (not clean) | - |
| FLEURS | CC BY 4.0 (secondary) | 16 kHz | ~12 h x 102 languages | OK, tiny | - |
| GigaSpeech, SPGISpeech, MSP-Podcast, NCSSD, SeniorTalk, ESD | custom / NC / gated | - | - | Excluded | - |
| ParCzech4Speech | CC-BY (secondary) | rate unknown | 2,695 h, 587 speakers | Not clean studio speech; skip | https://arxiv.org/pdf/2509.06675 |

No new 2025/2026 corpus was found that is studio-recorded, 48 kHz, many-speaker and permissive at once. The 2025/2026 additions of note are HiFiTTS-2 (useful) and YODAS v3 (risky).

### 1.2 What the challenges and reference models actually used

- **URGENT 2025 (track 1)**: DNS5 LibriVox (~350 h), LibriTTS (~200 h), VCTK (~80 h), WSJ (LDC), EARS (NC, ~100 h), MLS de/en/es/fr (~450 h), Common Voice 19 (~1300 h); noise from DNS5, WHAM! (NC), FSD50K, FMA; RIRs from DNS5. Total ~2.5k h speech, ~500 h noise. Not licence-clean (WSJ, EARS, WHAM!, AudioSet). https://urgent-challenge.github.io/urgent2025/data/
- **URGENT 2026 (track 1)**: a 700 h selection of the 2025 data plus NNCES, SeniorTalk (NC-SA), VocalSet, ESD (NC). https://urgent-challenge.github.io/urgent2026/track1/
- **Bandwidth tooling (reusable, Apache-2.0 repo):** URGENT estimates the effective bandwidth of every speech and noise file with an energy-threshold algorithm (`utils/estimate_audio_bandwidth.py`), then resamples each file to "the lowest SF among {8, 16, 22.05, 24, 32, 44.1, 48} kHz that can fully cover the estimated effective bandwidth", then applies VAD and DNSMOS thresholds. The estimator comes from the Hi-Fi TTS paper: highest frequency whose mean power spectrum is within -50 dB of the peak. https://github.com/urgent-challenge/urgent2025_challenge , https://huggingface.co/datasets/nvidia/hifitts-2
- **URGENT 2024 lessons:** nominal rate is unreliable (DNS5 LibriVox and CV English ~100% mismatched, LibriTTS ~25%); supposedly clean corpora (WSJ, VCTK, LibriTTS) still contain noisy samples after VAD + DNSMOS filtering; about 2,000 training labels had negative WADA-SNR; "a model may be misguided to preserve the noise floor". The hours-per-rate breakdown exists only in the paper's Figure 1 and was not readable here (UNVERIFIED numbers). https://arxiv.org/html/2506.01611v1
- **DeepFilterNet (1/2/3):** full multilingual DNS4 fullband set (>750 h speech, ~180 h noise), with the high-quality VCTK and PTDB subsets oversampled 10x; the authors simulated their own 48 kHz RIRs because the provided ones were 16 kHz (secondary reading of arXiv 2110.05588 / 2305.08227). Code and weights are "dual-licensed" MIT / Apache-2.0, but the training data includes AudioSet noise, DEMAND (SA) and ODbL speech, so DeepFilterNet's own weights would not pass this project's rule. https://github.com/Rikorose/DeepFilterNet
- **FastEnhancer 48 kHz** (repo README Table 3): VCTK-0.92 (81 h) plus three AI-Hub Korean sets (1,200 h subsampled; "Exporting these datasets outside of South Korea is prohibited"), Arabic speech (3.8 h); noise from DEMAND, DNS noise, FMA, MSRBench, WHAM!, TUT Urban 2018, simulated wind. Files without "active segments in the 22.05~24kHz band" were dropped. Not licence-clean. No licence stated for the weights on the page read. https://github.com/aask1357/fastenhancer
- **DPDFNet:** trained at **16 kHz** on DNS4 (760 h, 3,230 speakers, downsampled) + 4,000 h MLS English, noise from DNS4 + MUSAN + FSD50K, all speech pre-cleaned with DPCRN; 48 kHz variants exist but how they were trained is not described. Model card licence: Apache-2.0. https://arxiv.org/html/2512.16420v2 , https://huggingface.co/Ceva-IP/DPDFNet
- **Treble / DeepFilterNet3 study (Aug 2026):** see section 4.

### 1.3 How much true-fullband permissive speech is realistically available?

| Tier | Source | Hours | Speakers | Notes |
|---|---|---|---|---|
| Studio, true fullband | VCTK 0.92 | ~44 (one mic) to ~80 (both) | 110 | Best quality; small |
| LibriVox, curated | Hi-Fi TTS | 292 | 10 | >= 13 kHz bandwidth |
| LibriVox, large | HiFiTTS-2 at >= 13 kHz | ~31.7k | 4,631 | MP3 source, amateur recordings |
| LibriVox, strict | same source at >= 16 kHz | ~12.8k | 2,776 | From arXiv 2406.05298 |
| German | HUI | 326 | 122 | CC0 (verify) |
| Mandarin | AISHELL-3 | 85 | 218 | Apache-2.0 (verify rate) |

Realistic working set after a >= 16 kHz bandwidth filter, single-speaker filter, DNSMOS/SNR filter and a per-speaker cap of 15 to 30 minutes: **roughly 500 to 1,000 h from about 2,500 to 3,000 speakers**, plus VCTK. The hour count is a planning estimate, not a measured number (UNVERIFIED until the manifest is filtered). Open question: LibriVox MP3s are 128 kbps mono at 44.1 kHz (LibriVox wiki, secondary); what the encoder's low-pass leaves above 16 kHz, and how audible MP3 artefacts are in the top band, was not confirmed. The existence of 12.8k h passing a 16 kHz energy test says the band is not empty.

Wideband (<= 12 kHz bandwidth) permissive speaker pool for interferers and enrollment diversity: Common Voice (100k+ English client IDs), GLOBE (23.5k speakers), LibriTTS/LibriSpeech (~2.4k), MLS.

---

## 2. Speaker-labelled data and speaker encoders

### 2.1 Speaker-labelled corpora

| Corpus | Speakers | Licence | Usable? |
|---|---|---|---|
| HiFiTTS-2 (44 kHz subset) | 4,631 | CC BY 4.0 metadata, PD audio | Yes. Best: fullband, many utterances per speaker |
| LibriSpeech / LibriTTS | ~2.4k | CC BY 4.0 | Yes (16/24 kHz) |
| MLS | 5,574 English speakers (+ other languages) | CC BY 4.0 | Yes (wideband as distributed) |
| Common Voice | 100,172 English `client_id`s in v26 | CC0-1.0 + no re-identification condition | Yes. `client_id` is per browser/app session, so one person can have several IDs and one ID can be shared; treat as noisy labels. Training a speaker encoder is not "determining identity", but that condition should be read before training anything that looks like speaker recognition (UNVERIFIED legal reading) |
| GLOBE | 23,519 | CC0-1.0 | Yes (derived from Common Voice, cleaned) |
| VCTK | 110 | CC BY 4.0 | Yes |
| VoxCeleb1/2 | 7k+ | vox1 page: "The provided VoxCeleb metadata is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License."; DNS README quotes "CC BY 4.0" and "The copyright remains with the original owners of the video"; the vox1 page states URLs, timestamps and audio are no longer available from the site, while the landing page still describes a request form for audio (conflict between pages) | **No.** Licence covers metadata only; audio is YouTube content |
| CN-Celeb (SLR82) | 3,000 | "Attribution-ShareAlike 4.0 International" (primary) | **No** (SA), 16 kHz |

Sources: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html , https://www.robots.ox.ac.uk/~vgg/data/voxceleb/ , https://www.openslr.org/82/

### 2.2 Pretrained speaker encoders

| Model | Weights licence | Training data | Passes the rule? |
|---|---|---|---|
| SpeechBrain `spkrec-ecapa-voxceleb` | Apache-2.0 (card tag, secondary) | VoxCeleb1 + VoxCeleb2 | No (VoxCeleb provenance) |
| WeSpeaker ResNet34 / ECAPA / CAM++ (VoxCeleb) | Inconsistent: CC BY 4.0 in `docs/pretrained.md` vs Apache-2.0 on some HF cards; open issue #471 asks for clarification (secondary) | VoxCeleb2 dev (5,994 speakers) | No |
| NVIDIA TitaNet-Large | "License to use this model is covered by the CC-BY-4.0" (primary) | "Voxceleb-1", "Voxceleb-2", "Fisher", "Switchboard", "Librispeech", "SRE (2004-2010)" (primary) | No (VoxCeleb plus LDC corpora) |
| 3D-Speaker CAM++ / ERes2Net | Apache-2.0 repo (secondary) | VoxCeleb variants, or an unnamed 200k-speaker Mandarin corpus | No / unknowable |
| Resemblyzer | licence not confirmed | VoxCeleb1/2 + LibriSpeech-other (third-party statement) | No |

Sources: https://huggingface.co/nvidia/speakerverification_en_titanet_large , https://github.com/wenet-e2e/wespeaker/issues/471 , https://github.com/modelscope/3D-Speaker

Is "weights trained on VoxCeleb" a provenance problem under the rule? **Yes, if the rule is applied to everything shipped.** The weights' own licence (Apache-2.0 / CC BY 4.0) is permissive, and most of the industry ships such models, but the training audio has no licence from its owners ("unstated licence" in the project's terms), and TitaNet adds LDC corpora. This is a policy call, not a settled legal one. Options, cleanest first:

1. **No external encoder.** Condition on the enrollment waveform/spectrum directly (embedding-free target-speaker extraction, e.g. cross-attention over the enrollment clip's features) and train it end to end on permissive data. This also avoids a second model at inference.
2. **Train a small speaker encoder** on HiFiTTS-2 + LibriTTS + Common Voice/GLOBE + VCTK (about 4.6k clean speakers plus tens of thousands of noisy ones). Expect worse verification EER than VoxCeleb models, but BVC needs "same vs different talker in this room", not open-set verification.
3. Use a VoxCeleb-trained encoder only as an optional, separately downloaded component, and document it as an exception. Do not use it as a frozen teacher for shipped weights if the rule is meant to be strict.

A recent open system worth reading for the BVC task itself: VoxLock (arXiv 2610.40727, Oct 2026). 16 kHz causal BSRNN, 27M parameters, 24 ms algorithmic latency. It takes 3 s of the caller cut from the call itself instead of a separate enrollment, trains on LibriSpeech simulations (near caller plus 1 to 2 background talkers 0 to 15 dB quieter, half at the same distance, half 1.5 to 5 m away in simulated rooms) and then on **real AMI headset crosstalk**, and reports 4.95% WER on the Krisp benchmark (unprocessed 23.21%, Krisp VI 2.5 5.30%). Its stage 4 exists because the model first learned the shortcut "keep whoever is closer". It builds on an ECAPA branch and a WeSep base model, so its own provenance is not clean by this project's rule. https://arxiv.org/html/2610.40727

---

## 3. Noise

### 3.1 Per-corpus table

| Corpus | Licence | Notes | Verdict |
|---|---|---|---|
| MUSAN (SLR17) | Corpus label: "Attribution 4.0 International (CC BY 4.0)" (primary). Per-file licences are in a LICENSE file in each subdirectory (secondary: paper) | Noise: ~6 h, 929 files; Freesound-derived files are public domain, Sound Bible files carry CC licences (variant not confirmed). Music: Jamendo, FMA, Incompetech, HD Classical; the VOiCES authors had to drop MUSAN music files with ND or NC restrictions. Speech: `us-gov` part is public domain. 16 kHz | Noise part OK after reading the per-directory LICENSE files. Music part must be filtered per file. **16 kHz, so not fullband** |
| DNS5 `noise_fullband` | README: Audioset "CC BY 4.0"; Freesound "Only files with CC0 licenses were selected"; DEMAND "CC BY-SA 3.0" | 58 GB, ~181 h, ~62k clips, ~150 classes; ~60k AudioSet clips + ~10k Freesound/DEMAND clips (secondary). Archives are `audioset_000..006` and `freesound_000..001` (primary: download script) | Use **only the `freesound_*` archives**, then remove DEMAND if it is in there (location UNVERIFIED; no per-file licence manifest was found). AudioSet part excluded |
| AudioSet | Labels/segments CC BY 4.0, ontology CC BY-SA 4.0; audio is YouTube content under per-video terms (secondary) | - | Excluded |
| FSD50K | Dataset-level: "FSD50K is released under CC-BY"; per-clip licences in `dev_clips_info_FSD50K.json` / `eval_clips_info_FSD50K.json` (primary). Zenodo asks commercial users to contact the authors | 51,197 clips, 108.3 h. Dev (40,966): CC0 14,959, CC-BY 20,017, CC-BY-NC 4,616, Sampling+ 1,374. Eval (10,231): CC0 4,914, CC-BY 3,489, CC-BY-NC 1,425, Sampling+ 403 | **OK after per-clip filter.** CC0 + CC-BY = 43,379 clips (84.7%), about 90 h if clip lengths are licence-independent (estimate). Also drop human-voice classes for the noise set (URGENT does), or route them to the "interferer" set. CC-BY clips need an attribution list shipped with the weights' data card |
| FSDnoisy18k | Per-clip, only CC-BY and CC0 occur (secondary) | ~42 h | OK, overlaps Freesound |
| FSDKaggle2018 / 2019 | Dataset CC-BY; per-clip `license` column (2018, secondary); 2019 not confirmed | - | 2018 OK after filter |
| Clotho | Per-clip Freesound licences in metadata CSVs; some forbid commercial use; captions under a separate Tampere licence (secondary) | 15 to 30 s field recordings | Audio OK after per-clip filter; ignore captions |
| ARCA23K | All clips CC-licensed; variants and per-clip metadata not confirmed | - | UNVERIFIED |
| Freesound direct | Per-sound CC0 / CC-BY / CC-BY-NC / Sampling+; the Freesound FAQ allows bulk download for model training if each licence is respected (secondary) | API gives licence per sound | Best route to targeted classes (keyboard, fans, kitchens, TV rooms) with CC0/CC-BY only |
| FMA | Metadata CC BY 4.0; audio per-track, artist-chosen; `license` field in `tracks.csv` | fma_large ~883 h of 30 s clips; by licence (HF mirror table, secondary): CC-BY (all versions) ~58 h, CC0 1.3 h, CC-BY-SA ~23 h; the rest is NC and/or ND | Use only CC-BY + CC0 tracks: ~59 h of music. Enough |
| MTG-Jamendo, MusicCaps | per-track CC (Jamendo); MusicCaps audio is YouTube | not confirmed | Jamendo UNVERIFIED; MusicCaps excluded |
| TAU Urban Acoustic Scenes 2019 | Zenodo: "Other (Non-Commercial)" (primary) | 40 h, 10 scenes | **Excluded (NC)** |
| DEMAND | CC BY-SA 3.0 | - | Excluded (SA) |
| WHAM! / WHAMR! | CC BY-NC 4.0 | - | Excluded (NC) |
| QUT-NOISE | CC BY-SA 3.0 Australia (secondary: Research Data Australia) | - | Excluded (SA) |
| ESC-50 | CC BY-NC 3.0 | - | Excluded |
| NOISEX-92, CHiME noise, CochlScene, EPIC-Sounds, "Sound Events for Surveillance", MS-SNSD | not confirmed in this session | - | UNVERIFIED: treat as excluded until read |
| ACE Challenge corpus | CC BY 4.0 (primary) | 48 kHz ambient, fan and babble noise recorded in 7 rooms, plus RIRs | OK. Small but real room noise at 48 kHz |
| Simulated wind | generated (SC-Wind-Noise-Generator / FFmpeg per URGENT) | - | OK |

Sources: https://www.openslr.org/17/ , https://github.com/microsoft/DNS-Challenge/blob/master/download-dns-challenge-4.sh , https://zenodo.org/records/4060432 , https://zenodo.org/records/2589280 , https://zenodo.org/records/6257551 , https://arxiv.org/pdf/1612.01840 , https://huggingface.co/datasets/benjamin-paine/free-music-archive-large

### 3.2 TV / podcast / media-playback background speech

No permissive corpus of "TV in the next room" at 48 kHz was found. Options:

- **VOiCES** (SRI / Lab41): "CC BY 4.0" (arXiv abstract + AWS Open Data registry, secondary). LibriSpeech replayed in furnished rooms with television, music and babble distractors played from loudspeakers, recorded on 12+ far-field mics. Caveats: 16 kHz; the TV distractor audio itself came from "movies and television", so the distractor content is third-party copyrighted material inside a CC BY corpus. Use it for evaluation and for studying the acoustics, be cautious about training on the TV-distractor channel. https://arxiv.org/abs/1804.05053 , https://registry.opendata.aws/lab41-sri-voices/
- **BUT ReverbDB retransmitted LibriSpeech**: "The database has CC-BY 4.0 license" (primary). 117 GB of LibriSpeech test-clean replayed and re-recorded in real rooms on 31 mics. Real far-field speech with a permissive licence; 16 kHz (UNVERIFIED rate). https://speech.fit.vut.cz/software/but-speech-fit-reverb-database
- **Simulate (recommended main route):** permissive speech/music/podcast-like material (LibriVox read speech, Common Voice, CC-BY FMA music, speech+music mixes) -> loudspeaker colouration (small-speaker band-pass ~150 Hz to 12 kHz, mild compression/limiting, optional lossy codec) -> far-field RIR (1.5 to 5 m, strong reverb, often non-line-of-sight) -> mix at -5 to -25 dB relative to the near talker. Dynamic range compression and codec artefacts are what make TV speech sound unlike a live talker.
- **Record a small real set yourself** (a phone/laptop playing LibriVox and CC-BY content in a few rooms, captured on a laptop mic and a headset). A few hours with full rights is worth more for validation than any licensed corpus found here.

---

## 4. Room impulse responses

### 4.1 Per-source table

| Source | Licence | Rate / content | Verdict |
|---|---|---|---|
| OpenSLR SLR26 | "Apache 2.0" (primary) | Simulated only; `sim_rir_8k.zip` and `sim_rir_16k.zip` | Licence clean, **8/16 kHz only**: not usable for a 48 kHz model without accepting band-limited reverb |
| OpenSLR SLR28 (RIRS_NOISES) | "License: Apache 2.0" on the page, nothing else (primary). "all audio is in 16k sampling rate and 16-bit precision". Real RIRs and isotropic noises from "the RWCP sound scene database", "the 2014 REVERB challenge database" and "the Aachen impulse response database (AIR)"; point-source noises from MUSAN | Upstream: RWCP = research and development use only (NII SRC + OpenSLR SLR13, secondary); REVERB = no licence found, built on LDC corpora; AIR = MIT licence file in v1.4 (RWTH page, secondary), "Not stated in the download" on OpenSLR SLR20 | **Risky.** The Apache label cannot grant more than the upstream owners did. Use only `simulated_rirs/`; drop `real_rirs_isotropic_noises/`. And it is 16 kHz |
| DNS5 `impulse_responses` | README: OpenSLR26 and OpenSLR28, "Apache 2.0" | 5.9 GB. Described as 248 real + 60,000 synthetic (3 room sizes x 200 rooms x 100 RIRs). Rate conflict: URGENT and the Treble paper call them 48 kHz; the DeepFilterNet authors said the provided RIRs were 16 kHz and simulated their own (secondary) | Synthetic part OK (check the actual sample rate and whether they are upsampled); real part inherits the SLR28 problem |
| BUT ReverbDB | "The database has CC-BY 4.0 license" (primary) | 9 rooms (40 to 2000 m^3), 31 mics, ~5 loudspeaker positions each, about 1,426 RIRs (my sum), ESS method; rate not stated (UNVERIFIED, likely 16 kHz) | OK |
| ACE Challenge | "Creative Commons Attribution 4.0 International" (primary) | 48 kHz, 7 rooms, 1/2/3/5/8/32-channel arrays, with T60/DRR | OK. True 48 kHz real RIRs |
| Arni (Aalto variable acoustics) | CC BY 4.0 (Zenodo 6985104, secondary) | 132,037 RIRs, 5,342 absorption configurations, one room, 1 source x 5 receivers; rate UNVERIFIED (44.1 kHz from memory) | OK. Huge T60 variety but a single geometry: cap its sampling weight |
| dEchorate | CC BY 4.0 (Zenodo 6576203, secondary); code MIT | one cuboid room, 11 wall configurations, 48 kHz (UNVERIFIED rate) | OK |
| MeshRIR | CC BY 4.0 (Zenodo 10852693, secondary) | one room, dense grid, 48 kHz (UNVERIFIED rate) | OK, low diversity |
| AIR (Aachen) | MIT in v1.4 download (secondary) | 48 kHz, binaural + phone-style positions | OK if the licence file is in the copy you download from RWTH |
| MIT IR Survey | **No licence stated** on the lab page (primary). Third-party mirrors label it CC-BY 4.0 or "unknown" | 271 IRs, 1.5 m source-mic spacing, 32 kHz (secondary) | Excluded for training (unstated licence, not fullband). Fine for local eval |
| OpenAIR | per-IR licence; mix of CC BY and CC BY-SA (secondary) | large spaces, churches | Per-IR filter needed; mostly irrelevant acoustics for desk use |
| C4DM RIRs | CC BY-NC-SA (secondary: paper) | - | Excluded |
| GWA (Geometric-Wave Acoustic) | "Creative Commons Attribution-NonCommercial 4.0 International License" for the dataset (primary: repo) | 2M simulated IRs | **Excluded (NC)** |
| SoundSpaces | not checked | - | UNVERIFIED |
| Surrey multi-angle multi-distance microphone IRs | CC BY 4.0 (Zenodo 4633508, secondary) | 25 studio microphones | OK, for mic colouration |

Sources: https://www.openslr.org/26/ , https://www.openslr.org/28/ , https://www.openslr.org/13/ , https://www.openslr.org/20/ , https://zenodo.org/records/6985104 , https://zenodo.org/records/6576203 , https://zenodo.org/records/10852693 , https://mcdermottlab.mit.edu/Reverb/IR_Survey.html , https://github.com/RoyJames/GWA

### 4.2 Simulation tools

| Tool | Licence | Notes |
|---|---|---|
| pyroomacoustics | MIT (PyPI) | ISM, ray tracing and hybrid (recommended `max_order=3` for hybrid), `air_absorption=True`, source and mic `directivity` objects, octave-band wall absorption 125 Hz to 8 kHz. CPU. **Default choice** |
| rir-generator (audiolabs) | MIT | Classic ISM with simple mic patterns. Other Python ports of the same name are GPL-3.0 |
| FRA-RIR / FRAM-RIR (tencent-ailab) | Apache-2.0 on the repo (secondary) | Fast random approximation of ISM in torch; cheap enough for on-the-fly generation |
| gpuRIR | **AGPL-3.0** | Running it to produce data does not relicense the outputs (UNVERIFIED legal reading), but do not vendor it into an MIT/Apache repo |
| FAST-RIR | AGPL-3.0 | Neural generator, shoebox rooms; skip |
| Treble SDK | commercial | Hybrid wave-based + geometrical; see below |

### 4.3 Near-field vs far-field cues, and sim-to-real

- The origin paper for the task, "Distance-Based Sound Separation" (Patterson, Wilson et al., arXiv 2207.00562), trained and tested on **synthetic reverberant mixtures from a room simulator** with a near/far threshold; real-room generalisation was left as future work. It notes its simulator lacked proximity effect and air absorption.
- "Single-Channel Target Speech Extraction Utilizing Distance and Room Clues" (arXiv 2505.14433): distance-only models trained on simulation reached about 3 dB SDR on 784 measured RIRs; the authors state fine-tuning on real data is needed because distance cues are highly environment-dependent; after fine-tuning, 7.34 dB on real recorded speech.
- VoxLock (arXiv 2610.40727): simulation alone taught "keep the closer talker"; real AMI headset crosstalk and a stage that decorrelates caller distance from background distance were needed.
- Treble / DeepFilterNet3 (arXiv 2608.20971, vendor-authored): DFN3 trained on shoebox ISM RIRs (the DNS4 60k set) vs hybrid wave+geometric RIRs in furnished rooms with frequency-dependent materials and directive sources, tested on 284 **measured** RIRs (ACE + MIT). Hybrid minus ISM: PESQ +0.166, SI-SDRi +0.11 dB, STOI +0.013, SRMR +0.27; ASR WER improvement over noisy input 12.0 to 23.8% (hybrid) vs 0.5 to 13.1% (ISM). Modest on signal metrics, larger on ASR; the paper says the design cannot attribute the gain to one factor.
- Far-field keyword spotting study (arXiv 2006.02774): hybrid ISM + ray tracing with air absorption and real material coefficients gave up to 35.8% relative improvement over single-coefficient ISM; the code is in pyroomacoustics.

Reading: **nobody has shown zero-shot sim-to-real for single-mic distance-based separation.** Simulated training transfers partially; real near-field data with real crosstalk is what closes the gap.

What the near/far cue physically is, and what to simulate:

- At 15 to 50 cm the direct path dominates: direct-to-reverberant ratio (DRR) is high (roughly +10 to +25 dB in a living room or office), level is high, and spectrum is close to the talker's on-axis voice plus proximity effect on directional mics. At 1.5 to 4.5 m the talker is at or beyond critical distance: DRR around 0 dB or negative, early reflections dense, high frequencies reduced by off-axis mouth directivity and absorption.
- Shoebox ISM gets DRR-vs-distance roughly right but gets late reverb too regular (flutter, sweeping echoes) and ignores furniture, so add ray tracing/diffuse tail, randomised per-wall and per-band absorption, and jittered image positions.
- Add **source directivity** (a talker is cardioid-like above ~1 kHz; a background talker is usually not facing the mic) and **mic directivity/colouration** (laptop mics are omni MEMS behind a port; headset booms are noise-cancelling directional with strong proximity effect).
- Do not let level be the only cue: randomise near/far level ratio widely, including cases where the far talker is as loud or louder (a TV), otherwise the model learns "louder = target".
- Include same-distance interferers as an explicit, labelled "do not remove / ambiguous" or enrollment-resolved case, as VoxLock did.
- For mouth-to-headset distances (2 to 10 cm) point-source ISM is not physically valid; treat it as "nearly dry + device EQ + breath/pop noise" rather than simulating it.

---

## 5. Device / microphone realism

| Resource | Licence | Verdict |
|---|---|---|
| MicIRP | CC BY-SA 4.0 (project About page, secondary) | **Excluded (SA).** Also vintage studio mics, not laptops |
| DAPS device recordings | CC BY-NC 4.0 | Excluded for training; good local eval (tablet/phone in real rooms) |
| DDS (device-degraded speech) | "Creative Commons Attribution Non Commercial 4.0 International" for the DAPS portion (primary); VCTK portions in two other records not checked | Excluded for training. ~2,000 h, 3 mics x 9 rooms x 6 positions, 16 kHz file names |
| VoiceBank-DEMAND | Speech CC BY 4.0, noise DEMAND CC BY-SA 3.0 | Excluded for training; standard eval |
| MS-SNSD | not confirmed | UNVERIFIED |
| Surrey multi-angle multi-distance microphone IRs | CC BY 4.0 (secondary) | OK; 25 studio mics |
| Illinois wearable microphone IRs | CC BY (secondary) | OK; body-worn positions |
| POLIPHONE (20 smartphones, per-model IRs) | dataset licence not confirmed | UNVERIFIED |
| Common Voice itself | CC0 | The largest permissive collection of real consumer laptop/phone/headset mic recordings. Not paired, but it teaches the input distribution if used as "target at wideband" or for device-colouration statistics |
| AMI headset mics | CC BY 4.0 (official corpus page, secondary) | Real close-talk headset recordings with real crosstalk, 16 kHz |
| Krisp benchmark | CC BY-NC 4.0 | Eval only; 24 + 13 real headsets/earbuds |

No permissive corpus of laptop or headset microphone impulse responses was found. Practical substitute:

- Random parametric device EQ: high-pass 60 to 300 Hz, 1 to 3 peaking filters of +-6 dB between 1 and 10 kHz, low-pass 7 to 20 kHz, plus occasional comb filtering (laptop lid reflection, 0.3 to 2 ms).
- Non-linearities: soft/hard clipping, AGC pumping, noise gate, OS-level noise suppression artefacts (spectral gating), self-noise floor of MEMS mics (pink-ish, -70 to -50 dBFS).
- **Codec / Bluetooth HFP:** when a Bluetooth headset opens its mic the link drops to HFP: CVSD at 8 kHz (narrowband) or mSBC at 16 kHz (wideband); newer stacks use LC3 / LC3-SWB (super-wideband, up to ~14 kHz audio bandwidth) (from memory, UNVERIFIED). Simulate by resampling to 8/16/32 kHz, encoding with FFmpeg codecs (`libopus`, `sbc`, `aac`, `libmp3lame`, AMR via opencore, G.711/G.722) or Google's `liblc3`, random packet loss with concealment, then resampling back to 48 kHz. URGENT's simulation scripts already implement bandwidth limitation, codec distortion, packet loss and clipping and are Apache-2.0. https://github.com/urgent-challenge/urgent2026_challenge_track1
- Decide explicitly whether the model should bandwidth-extend HFP input or pass it through; if targets are fullband and inputs are 8 kHz, the model will hallucinate highs.

---

## 6. Evaluation sets (local use; NC is acceptable here)

| Set | Licence | Real or simulated | Near-field + real background talker? | Notes |
|---|---|---|---|---|
| Krisp Voice Isolation Benchmark | `cc-by-nc-4.0` (primary: HF card) | **Real**, "No synthetic mixing" | **Yes.** Main talker on a headset/earbud plus a second voice in the room | 265 recordings (Work 96, Call Center 106, Phone Calls 63), 47 speakers, 24 + 13 devices, segments labelled primary / mix / secondary, transcripts. No clean references: score with WER, leakage words/min and DNSMOS. Released Sept 2026. https://huggingface.co/datasets/Krisp-AI/VoiceIsolation-Benchmark-Dataset |
| DNS5 blind test set | licence not stated (README: datasets "provided under the original terms that Microsoft received such datasets") | **Real**, crowd-sourced | **Yes.** Headset track 389 clips, speakerphone track 331; 220 clips per track have interfering talkers; enrollment clips (10 to 30 s) provided | No clean references; needs DNSMOS/P.835-style scoring. Exact download path not confirmed. https://arxiv.org/pdf/2303.11510 |
| AMI (headset channels) | CC BY 4.0 (secondary) | **Real** | **Yes**: real crosstalk on close-talk headsets | 16 kHz. Also legally usable for **training**. VoxLock's blind AMI protocol: 400 windows, 16 people. https://groups.inf.ed.ac.uk/ami/corpus/ |
| DiPCo | CDLA-Permissive 1.0 (secondary: paper + Amazon blog) | **Real** dinner-party, close-talk + far-field arrays | Yes | 16 kHz. Usable for training too |
| RealMAN | CC BY 4.0 (secondary: repo + NeurIPS supplement) | **Real** array recordings with direct-path targets | Far-field mostly | 48 kHz. Usable for training (real noise and real reverberant speech with clean targets) |
| VOiCES | CC BY 4.0 (secondary) | Re-recorded (real acoustics, played-back sources) | Far-field target with TV/music/babble distractors | 16 kHz |
| LibriCSS | CC BY 4.0 (secondary only) | Re-recorded LibriSpeech, overlapping talkers, array | No near-field | 16 kHz |
| CHiME-5/6/7 | restricted (not checked) | Real | Yes | Check terms |
| EARS-WHAM / EARS-Reverb | CC BY-NC 4.0 (EARS speech, WHAM! noise) | Simulated | No | The standard 48 kHz paired benchmark |
| VoiceBank-DEMAND | CC BY 4.0 + CC BY-SA 3.0 | Simulated | No | 48 kHz, 824 test utterances, legacy comparability |
| URGENT 2025/2026 test sets | not checked; 2026 blind set: 360 simulated + 480 real | Mixed | No | Multi-distortion, multi-rate |
| DAPS | CC BY-NC 4.0 | Real device recordings, aligned to clean | No | Good for device/room realism at 44.1 kHz |
| DPDFNet EvalSet | not stated | Simulated, long clips with speech-free gaps | No | 12 languages, 9 noise environments |

Only Krisp, the DNS5 blind set, AMI and DiPCo give **real** near-field recordings with real background talkers. Krisp and DNS5 are the closest to the product scenario and both lack clean references, so plan for reference-free metrics (ASR WER on the primary talker, leaked-word rate on secondary-only segments, DNSMOS/NISQA) alongside intrusive metrics on simulated sets.

---

## 7. Compute platform reality check (late 2026)

All Colab numbers below are from third-party write-ups and Google's plan pages as summarised by search; Google does not publish machine specs. Treat as indicative.

### 7.1 Colab / Google AI Ultra

- Since 22 Sept 2026 Google AI plans include Colab compute units: AI Plus 50, AI Pro 200, **Ultra 5x 1,000, Ultra 20x 2,000** per month (Google One help page via search; one blog claims 2,000 for both Ultra tiers). Ultra adds background execution and premium GPU access. Units expire after 90 days. Standalone: Pro $9.99 (100 units), Pro+ $49.99 (600 units), extra units $9.99 per 100. https://support.google.com/googleone/answer/16286513 , https://one.google.com/about/google-ai-plans/
- **Session cap: 24 h per VM, hard**, including on Ultra. If units run out mid-run the session drops to free-tier limits (12 h, 90 min idle).
- **GPU availability is not guaranteed.** H100 and G4 are reported as restricted to Pro+/premium tiers since Aug 2026. The CLI's `--gpu A100` returned the 40 GB card while the web UI's high-RAM A100 gave the 80 GB card.
- **Burn rates (conflicting):** A100 40 GB 5.40 units/h and 80 GB 7.52 units/h (March 2026 measurement); 10.59 units/h (Oct 2024 measurement); one comparison claims ~15. H100: about 5 to 6 h per 100 units, i.e. ~17 to 20 units/h (derived).
  - 1,000 units = roughly 95 to 185 A100-hours or 50 to 60 H100-hours per month.
  - 2,000 units = roughly 190 to 370 A100-hours.
- **Machine:** A100 runtime measured at 12 vCPUs, ~84 GB RAM, ~236 GB disk (2024); high-RAM A100 167 GB RAM (2026); pay-as-you-go A100 reported with only ~113 GB disk. vCPU count on the 2026 high-RAM shape not confirmed.
- **Drive:** mounting Drive in Colab is a FUSE layer over the Drive API. Known failure modes: per-file download quota lockouts for ~24 h after repeated reads (colabtools issue #1020; surfaces as 403 `downloadQuotaExceeded`), very slow many-small-file access, a 750 GB/day per-user upload/copy cap, 400,000 item limit per shared drive. No official download quota number is published. Copying a few hundred GB of tar shards once per session is exactly the pattern that trips this. https://github.com/googlecolab/colabtools/issues/1020
- **Disconnects:** background execution keeps the VM alive after the tab closes, up to the 24 h cap; nothing protects against preemption, runtime recycling or unit exhaustion. The training script must checkpoint every 15 to 30 minutes to remote storage and resume automatically.

Verdict on the draft plan ("one 12 to 24 h run on Colab with shards on Drive"): the single-run framing is fragile (hard 24 h wall, no GPU guarantee), the Drive part is the weakest link, and one run is not a project. Budget for 5 to 15 runs (data bugs, ablations, the real run, a fine-tune).

### 7.2 Alternatives (November 2026 snapshots, third-party trackers)

| Option | Price | Reliability / fit |
|---|---|---|
| Vast.ai RTX 4090 | ~$0.22 to $0.54/h, median ~$0.47 | Cheapest. Marketplace hosts: variable disk/network, can be interrupted. Fine with checkpoint + resume |
| RunPod RTX 4090 | $0.34 spot/community, $0.89 on-demand | Good middle ground; persistent volumes |
| Vast.ai A100 80 GB / H100 | median ~$0.84 / ~$1.73 | Good value if VRAM is needed |
| RunPod A100 80 GB / H100 | $1.39 to $2.19 / $1.99 spot to $4.39 list | Reliable |
| Lambda H100 | $3.99 to $4.59 | Most reliable, most expensive; prices rose during 2026 |
| Modal | A100 40 GB ~$2.10/h, A100 80 GB ~$2.50/h, L40S ~$1.95/h; $30/month free credit | Per-second billing, good for preprocessing fan-out and short jobs |
| Lightning AI | free credits (figures conflict: 15 to 30 credits, ~50 to 80 interruptible GPU-hours) | 4 h restart on free studios; fine for smoke tests |
| Kaggle | free ~30 GPU-h/week, T4 x2, ~12 h sessions; P100 reportedly retired Sept 2026 (disputed) | Too slow and too little disk for 48 kHz training; fine for eval |
| Hugging Face | Spaces/Jobs GPUs not priced here | UNVERIFIED |
| Local NVIDIA laptop GPU (the user's Ubuntu laptop) | electricity | Good for debugging the pipeline and overfitting tests; thermals and 6 to 8 GB VRAM (UNVERIFIED spec) make multi-day runs unpleasant |

Cost for 100 to 300 GPU-hours:

- RTX 4090 at $0.35 to $0.90/h: **$35 to $270**.
- A100 80 GB at $0.85 to $2.20/h: $85 to $660.
- H100 at $1.75 to $4.40/h: $175 to $1,320.
- Colab on an already-paid Ultra plan: "free" for ~95 to 185 A100-hours per month on Ultra 5x, with the reliability caveats above.

A causal 48 kHz enhancement model in the 2 to 10 M parameter range trains comfortably on a 24 GB card; DeepFilterNet-class models were trained on single consumer GPUs. An A100/H100 mainly buys larger batches, not a different outcome. Recommendation: debug locally, run sweeps and the main runs on a rented 4090 (or A100 80 GB if batch size demands), and use the Ultra Colab allowance as overflow for evaluation and short ablations.

### 7.3 Storage

Size arithmetic (48 kHz, 16-bit, mono): PCM is 345.6 MB/h.

- 500 h WAV = **~173 GB**.
- FLAC for speech typically lands at 40 to 60% of PCM: **~70 to 105 GB for 500 h**. Data point: VCTK 0.92 is 10.94 GB zipped for ~88 h of 48 kHz FLAC (both mics), i.e. ~124 MB/h or 36% of PCM (very clean, lots of silence).
- 1,000 h speech FLAC ~ 150 to 200 GB; 150 h noise ~ 30 GB; RIRs < 10 GB. Whole training set: **~200 to 300 GB**, which fits a Colab 236 GB disk only barely and fits any rented box with a 500 GB volume easily.
- Opus at 96 to 128 kbps would cut speech to ~45 to 55 MB/h but is lossy above ~20 kHz and adds decode cost; keep targets lossless.

| Store | Cost | Notes |
|---|---|---|
| Google Drive | bundled with the AI plan | Quota lockouts, slow FUSE, no S3 API. **Avoid as the primary store** |
| Cloudflare R2 | $0.015/GB-month, **zero egress**; Class B reads $0.36/M; 10 GB free | 300 GB ~ $4.50/month. S3 API, works from Colab, RunPod, Vast, Modal alike. **Recommended** |
| Backblaze B2 | $6.95/TB-month; free egress up to 3x stored volume, then $0.01/GB | 300 GB ~ $2.10/month; repeated full pulls from many machines can exceed the 3x allowance |
| Hugging Face private dataset repo | Free: 100 GB private. PRO $9/month: 1 TB private. Overage quoted as $18 or $25/TB-month (conflict) | Fast Xet/CDN downloads, `hf_hub_download` / `datasets` streaming, per-file cap 500 GB, <= 10k files per folder. Good second choice; keep it private (Common Voice terms discourage re-hosting; HiFiTTS-2 itself declines to rehost LibriVox audio) |

Shard layout: 1 to 2 GB tar shards (150 to 300 files total), one prefix per source corpus, a manifest (parquet/JSON) with path, speaker, duration, estimated bandwidth, DNSMOS, licence and attribution string per item. At session start pull with 16 to 32 parallel streams (`s5cmd`, `rclone`, or `hf download`) to local NVMe; expect 100 to 500 MB/s, i.e. 10 to 40 minutes for 250 GB.

---

## 8. Data loading practice

What others do:

- **DeepFilterNet:** HDF5 files per split and type (`speech`, `noise`, `rir`), built by `df/scripts/prepare_data.py`; mixing and augmentation on CPU in a Rust loader (`libDF`) with a PyTorch wrapper; up to five noises per utterance, SNR -5 to 40 dB, RIRs applied to speech and noise, random resampling for pitch. https://github.com/Rikorose/DeepFilterNet
- **URGENT / ESPnet:** plain audio files + `wav.scp`-style index; "Dynamic simulation: Recommended during training"; needs > 8 CPU cores for preparation.
- **DPDFNet:** 3 s chunks, SNR -5 to 40 dB, online augmentation (random second-order filters, gain, reverb with early-reflection targets), 1.6M iterations at batch 32.
- A multichannel Wave-U-Net team reported on-the-fly augmentation was too slow and pre-generated data instead (secondary).

Format choice:

| Format | For | Against |
|---|---|---|
| WebDataset tar shards | Sequential reads from object storage, trivial to sync, works with streaming | No random access to a specific speaker; BVC/enrollment needs "another utterance of the same speaker" and "a different speaker", which is awkward with pure sequential shards (workaround: shard by speaker, or keep a shuffle buffer of speaker pools) |
| Plain FLAC + parquet index on local NVMe | Random access by speaker, simple, debuggable; trivial per-item metadata filters | Many small files are slow to sync (ship as tars, unpack locally) |
| HDF5 | Random access in one file; proven by DeepFilterNet | Multi-worker reads need care; no streaming from object storage; opaque |
| lhotse (cuts + Shar) | Manifests carry speaker/supervision metadata; Shar gives sequential tar shards with indexed fields; built-in reverb/mix transforms | Another framework to learn; mixing logic for 3-source BVC still custom |

Recommendation: **tar shards for transport, unpacked FLAC + one parquet index for training.** Sample by speaker from the index (target utterance, enrollment utterance from the same speaker, 0 to 3 interferers from other speakers, 0 to 3 noises, RIRs), read random 3 to 6 s crops with `soundfile` seek (no full-file decode).

Is CPU-side 48 kHz mixing feasible on ~12 vCPUs? Back-of-envelope (my estimate, not a benchmark; UNVERIFIED):

- Demand: a small causal model at batch 32 x 4 s and ~10 to 15 steps/s consumes ~1,300 to 1,900 s of mixture audio per second. Each mixture needs ~4 to 6 source crops (target, enrollment, interferers, noises): ~6,000 to 10,000 s of decoded audio per second.
- Supply: FLAC decode of a seeked crop runs at several hundred times real time per core; 10 workers give ~3,000 to 6,000x. Decode alone is already near the limit.
- RIR convolution on CPU (FFT convolve, 4 s x 1 s RIR at 48 kHz) costs a few ms per source; several sources x resampling x codec simulation x EQ pushes the CPU well past 12 cores.

So: **decode and crop on CPU workers, do everything else on the GPU in batch.** Concretely:

- Workers return raw float crops plus sampled parameters (which RIR index, SNRs, gains, EQ coefficients).
- On GPU: batched FFT convolution with RIRs (`torchaudio.functional.fftconvolve` or a manual rFFT multiply; RIRs kept in a GPU tensor bank or generated on the fly with an FRA-RIR-style approximation), early-reflection/direct-path target construction (convolve the target with the RIR truncated to ~0 to 50 ms for the dereverb target), gain/SNR scaling, biquad EQ via `torchaudio.functional.lfilter` or frequency-domain masks, clipping, resample-based band-limiting, summation.
- Keep true codec simulation (Opus/SBC/MP3 via FFmpeg) on CPU and apply it to a minority of items (10 to 20%), or pre-render a codec-degraded copy of a subset offline.
- Use `torch-audiomentations` only if its transforms fit; a 100-line custom GPU mixer is easier to make deterministic.
- Pre-compute a fixed validation set once (no on-the-fly randomness) and store it with the shards.

This also makes the pipeline GPU-agnostic: a rented 4090 box with 16 to 32 vCPUs behaves the same as Colab's 12.

---

## 9. Recommendation

### 9.1 Proposed approved-data list

Speech (targets, enrollment, interferers):

| Source | Use | Hours (working set) | Speakers | True bandwidth | Licence |
|---|---|---|---|---|---|
| VCTK 0.92 (mic1; hold out ~10 speakers) | Fullband target anchor; oversample | ~40 | ~100 | 24 kHz (48 kHz native) | CC BY 4.0 |
| HiFiTTS-2 subset: bandwidth >= 16 kHz, single-speaker, DNSMOS/SNR-filtered, per-speaker cap | Main fullband targets + enrollment pairs | 500 to 1,000 (estimate) | ~2,500 to 2,800 | 16 to 22 kHz | CC BY 4.0 metadata; LibriVox public domain audio |
| Hi-Fi TTS | Extra high-SNR fullband | up to 292 (cap per speaker) | 10 | >= 13 kHz | CC BY 4.0 |
| HUI-Audio-Corpus-German (verify licence file) | Non-English fullband | ~100 to 300 | 122 | 44.1 kHz | CC0 |
| AISHELL-3 (verify rate/licence) | Tonal-language fullband | ~85 | 218 | 44.1 kHz recorded | Apache-2.0 |
| LibriTTS-R / LibriTTS | Wideband targets (bandwidth-labelled), speaker pool | 200 to 585 | ~2.4k | <= 12 kHz | CC BY 4.0 |
| Common Voice (DNSMOS-filtered) and/or GLOBE | Interferer voices, enrollment diversity, real device colouration | 300 to 1,000 | 20k to 100k (noisy IDs) | mostly <= 8 to 12 kHz | CC0-1.0 |
| MLS (non-English, OpenSLR copy) | Multilingual interferers | 100 to 300 | thousands | 8 kHz bandwidth | CC BY 4.0 |
| AMI headset channels | Real near-field + real crosstalk fine-tuning (16 kHz band) | ~100 | ~190 (UNVERIFIED) | 8 kHz | CC BY 4.0 |
| VocalSet | Optional singing | 10 | 20 | 44.1 kHz | CC BY 4.0 |

Because most speakers are wideband, train with **per-item bandwidth labels**: the loss above an item's effective bandwidth is masked (or the input is band-limited to match), so wideband material teaches separation and identity while only true-fullband material supervises the top band. This is what makes the large wideband speaker pool usable in a 48 kHz model.

Noise:

| Source | Hours | Bandwidth | Licence |
|---|---|---|---|
| FSD50K, CC0 + CC-BY clips only, minus speech-like classes | ~80 to 90 (estimate) | mixed, much is 44.1 kHz | per-clip CC0 / CC-BY (attribution list required) |
| DNS5 `freesound_*` archives, after removing DEMAND | part of ~10k clips | 48 kHz | CC0 (per README) |
| Freesound API pull, CC0 (+ CC-BY) for target classes: keyboards, fans/HVAC, kitchens, traffic through windows, cafes, dogs, doors | 50 to 200 | 44.1/48 kHz | per-sound |
| MUSAN noise (after reading per-directory LICENSE files) | ~6 | 16 kHz | PD / CC |
| FMA, CC-BY + CC0 tracks only | ~59 | 44.1 kHz | per-track |
| ACE corpus noise | few | 48 kHz | CC BY 4.0 |
| Synthetic: wind, hum (50/60 Hz + harmonics), coloured stationary noise, mic self-noise, clicks | unlimited | 48 kHz | n/a |
| Simulated media playback (section 3.2) from permissive speech + CC-BY music | unlimited | 48 kHz | inherits sources |

RIRs:

| Source | Count | Rate | Licence |
|---|---|---|---|
| Own simulation, pyroomacoustics hybrid ISM + ray tracing, directivity, air absorption, 48 kHz; near bank (0.15 to 0.6 m) and far bank (1.2 to 6 m) from the same rooms | 50k to 200k | 48 kHz | n/a (tool MIT) |
| ACE | hundreds | 48 kHz | CC BY 4.0 |
| Arni (down-weighted) | 132k | verify | CC BY 4.0 |
| BUT ReverbDB | ~1.4k | verify | CC BY 4.0 |
| dEchorate, MeshRIR, AIR (MIT copy) | thousands | 48 kHz (verify) | CC BY 4.0 / MIT |
| SLR26, SLR28 `simulated_rirs` only | 60k | 16 kHz | Apache-2.0: keep only if a 16 kHz RIR bank is useful; otherwise drop |

### 9.2 Items on the current approved list that are actually risky

| Item | Problem | Action |
|---|---|---|
| OpenSLR SLR28 | Real RIRs/noises from RWCP (R&D only), REVERB (no licence), AIR; 16 kHz | Use `simulated_rirs` only, or drop |
| OpenSLR SLR26 | Clean licence but 8/16 kHz | Drop for a 48 kHz model |
| "Licence-filtered DNS5 subsets" | Noise: AudioSet + DEMAND cannot be filtered per file without a manifest (only by archive name). Speech: ODbL (CREMA-D, PTDB), likely CC BY-SA in German, unconfirmed Spanish source, VoxCeleb2 listed; non-English "fullband" is upsampled 16 kHz | Keep only `read_speech` (prefer HiFiTTS-2 instead), VCTK, VocalSet, and `freesound_*` noise minus DEMAND |
| MUSAN "(CC-BY parts)" | Music has NC/ND tracks; licences are per file; all 16 kHz | Noise + filtered music only; low value at 48 kHz |
| Common Voice | CC0 but MDC download terms add no-re-identification and discourage re-hosting; not fullband | Fine; keep shards private; treat speaker IDs as noisy |
| MLS | Fine, but 16 kHz as distributed | Use HiFiTTS-2 for fullband |
| LibriTTS-R | Restored (model-generated) targets, 24 kHz | Prefer original LibriTTS for targets or mix both |
| Draft: "VCTK / DAPS" | DAPS is CC BY-NC 4.0 | Remove DAPS from training; keep for eval |
| Any VoxCeleb-trained speaker encoder | Training audio has no licence from owners | Avoid; condition on enrollment audio directly |

### 9.3 Gaps

1. **Studio-quality fullband speaker diversity.** 110 VCTK speakers is all there is at studio quality. LibriVox fills the count (thousands) at amateur quality with MP3 artefacts. Mitigation: bandwidth-masked loss, strict quality filtering, oversample VCTK, and consider recording a small in-house set.
2. **Real near-field + background-talker training data at 48 kHz.** None found with a permissive licence. AMI (16 kHz) is the only real crosstalk data that is licence-clean. Sim-to-real for distance cues is unproven; plan a real-data fine-tune and a small self-recorded validation set from day one.
3. **Licence-clean fullband noise volume.** ~150 to 300 h is achievable (FSD50K filtered + Freesound CC0 + FMA CC-BY + synthetic), which is less than DNS5's 181 h + extras but adequate with on-the-fly mixing.
4. **Device/mic impulse responses.** No permissive laptop/headset IR corpus. Use parametric EQ + Common Voice statistics.
5. **TV/media playback.** Simulate; validate on self-recorded clips and VOiCES.
6. **Speaker encoder provenance.** Solved only by not using one.
7. **Attribution bookkeeping.** CC-BY sources (FSD50K clips, FMA tracks, VCTK, HiFiTTS-2 metadata, ACE, Arni, BUT) require an attribution file shipped with the weights; generate it from the manifest.

### 9.4 Compute + storage setup and budget

- **Storage:** Cloudflare R2 bucket, ~250 to 300 GB of FLAC tar shards + parquet manifest: ~$4 to $5/month. Alternative: Hugging Face PRO ($9/month, 1 TB private).
- **Preprocessing** (download HiFiTTS-2 subset from archive.org, bandwidth estimation, DNSMOS, resample, shard): CPU-bound; 1 to 3 days on a 16 to 32 core rented box or the Ubuntu dev machine; DNSMOS/quality scoring benefits from a GPU. Budget $20 to $50. Download only the filtered manifest subset, not the 4 TB full set.
- **Training:** rented RTX 4090 (RunPod/Vast), 500 GB local NVMe, 16+ vCPUs, GPU-side mixing, checkpoint to R2 every 15 to 30 min with auto-resume. Plan 150 to 300 GPU-hours across ~10 runs: **$60 to $270**. If batch size needs it, A100 80 GB at ~$0.85 to $2.20/h: $130 to $660.
- **Colab (Ultra):** use the included 1,000 to 2,000 units for evaluation, ablations and overflow; never depend on a single 24 h session; pull shards from R2/HF, not Drive.
- **Total realistic budget:** about **$100 to $350** for the project excluding the existing Google subscription; under $100 if most hours come from the Ultra allowance and things go right the first time (they will not).

### 9.5 Items to verify before committing

- Sample rates of MLS (OpenSLR), Arni, BUT ReverbDB, dEchorate, MeshRIR, AISHELL-3 distribution, DNS5 `impulse_responses`.
- HUI and AISHELL-3 licence files inside the downloads.
- Whether DEMAND files are inside the DNS5 `freesound_*` archives and how to identify them.
- MUSAN per-directory LICENSE contents.
- How much of the HiFiTTS-2 >= 16 kHz subset survives a DNSMOS/SNR filter, and what the MP3 encoder leaves above 16 kHz (inspect spectrograms of 50 random files before downloading hundreds of GB).
- Common Voice / Mozilla Data Collective full terms (re-hosting, speaker-recognition-adjacent use).
- AMI speaker count and licence scope for headset signals.
- Colab's current A100/H100 unit rates and disk size on the user's own account (run `nvidia-smi`, `nproc`, `df -h` and watch the unit counter for an hour).
- Whether the project's rule should treat ODbL (CREMA-D, PTDB) and CDLA-Permissive (DiPCo) as acceptable.

---

## Source index

Speech corpora
- VCTK 0.92: https://datashare.ed.ac.uk/handle/10283/3443 ; licence text https://datashare.ed.ac.uk/bitstream/handle/10283/3443/license_text.txt
- DAPS: https://zenodo.org/records/4660670
- DDS: https://zenodo.org/records/5464104 ; https://arxiv.org/abs/2109.07931
- Hi-Fi TTS: https://www.openslr.org/109/ ; https://arxiv.org/abs/2104.01497
- HiFiTTS-2: https://huggingface.co/datasets/nvidia/hifitts-2 ; https://arxiv.org/html/2506.04152
- Spectral codecs (12.8k h >= 16 kHz LibriVox): https://arxiv.org/html/2406.05298v2
- MLS: https://www.openslr.org/94/
- Common Voice 26 English: https://mozilladatacollective.com/datasets/cmqim2hn800ssnr07gvmpcnwu
- GLOBE_V2: https://huggingface.co/datasets/MushanW/GLOBE_V2 ; https://arxiv.org/abs/2406.14875
- YODAS v3: https://arxiv.org/abs/2609.29448 ; https://huggingface.co/datasets/espnet/yodas3
- M-AILABS licence mirror: https://github.com/i-celeste-aurora/m-ailabs-dataset
- HUI: https://arxiv.org/pdf/2106.06309 ; https://github.com/iisys-hof/HUI-Audio-Corpus-German
- AISHELL-3: https://www.openslr.org/93/
- JVS / JSUT: https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus ; https://sites.google.com/site/shinnosuketakamichi/publication/jsut
- EARS: https://github.com/facebookresearch/ears_dataset ; Expresso: https://speechbot.github.io/expresso/ ; Emilia: https://huggingface.co/datasets/amphion/Emilia-Dataset
- People's Speech: https://huggingface.co/datasets/MLCommons/peoples_speech
- CREMA-D: https://github.com/CheyneyComputerScience/CREMA-D

Challenges and models
- DNS Challenge repo: https://github.com/microsoft/DNS-Challenge ; DNS4 script https://github.com/microsoft/DNS-Challenge/blob/master/download-dns-challenge-4.sh ; DNS5 headset script https://github.com/microsoft/DNS-Challenge/blob/master/download-dns-challenge-5-headset-training.sh
- DNS papers: https://arxiv.org/pdf/2009.06122 ; https://arxiv.org/pdf/2101.01902 ; https://arxiv.org/pdf/2202.13288 ; https://arxiv.org/pdf/2303.11510
- URGENT 2025 data: https://urgent-challenge.github.io/urgent2025/data/ ; repo https://github.com/urgent-challenge/urgent2025_challenge
- URGENT 2026 track 1: https://urgent-challenge.github.io/urgent2026/track1/ ; repo https://github.com/urgent-challenge/urgent2026_challenge_track1 ; paper https://arxiv.org/pdf/2601.13531
- URGENT 2024 lessons: https://arxiv.org/html/2506.01611v1
- DeepFilterNet: https://github.com/Rikorose/DeepFilterNet ; https://arxiv.org/pdf/2110.05588 ; https://arxiv.org/abs/2305.08227
- FastEnhancer: https://github.com/aask1357/fastenhancer ; https://arxiv.org/pdf/2509.21867
- DPDFNet: https://arxiv.org/html/2512.16420v2 ; https://huggingface.co/Ceva-IP/DPDFNet
- VoxLock: https://arxiv.org/html/2610.40727

Speaker data and encoders
- VoxCeleb: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/ ; https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html
- CN-Celeb: https://www.openslr.org/82/
- TitaNet: https://huggingface.co/nvidia/speakerverification_en_titanet_large
- SpeechBrain ECAPA: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
- WeSpeaker licence issue: https://github.com/wenet-e2e/wespeaker/issues/471
- 3D-Speaker: https://github.com/modelscope/3D-Speaker

Noise
- MUSAN: https://www.openslr.org/17/ ; https://arxiv.org/abs/1510.08484
- FSD50K: https://zenodo.org/records/4060432
- TAU Urban 2019: https://zenodo.org/records/2589280
- FMA: https://arxiv.org/pdf/1612.01840 ; https://github.com/mdeff/fma ; https://huggingface.co/datasets/benjamin-paine/free-music-archive-large
- Clotho: https://zenodo.org/records/4743815 ; FSDKaggle2018: https://zenodo.org/records/2552860
- QUT-NOISE: https://researchdata.edu.au/qut-noise-databases-protocols/674966
- VOiCES: https://arxiv.org/abs/1804.05053 ; https://registry.opendata.aws/lab41-sri-voices/

RIRs and simulation
- SLR26: https://www.openslr.org/26/ ; SLR28: https://www.openslr.org/28/ ; RWCP: https://www.openslr.org/13/ ; AIR: https://www.openslr.org/20/ and https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database/
- BUT ReverbDB: https://speech.fit.vut.cz/software/but-speech-fit-reverb-database
- ACE: https://zenodo.org/records/6257551
- Arni: https://zenodo.org/records/6985104
- dEchorate: https://zenodo.org/records/6576203 ; MeshRIR: https://zenodo.org/records/10852693
- MIT IR Survey: https://mcdermottlab.mit.edu/Reverb/IR_Survey.html
- GWA: https://github.com/RoyJames/GWA
- RIR dataset list: https://github.com/RoyJames/room-impulse-responses
- pyroomacoustics: https://pyroomacoustics.readthedocs.io/ ; gpuRIR: https://github.com/DavidDiazGuerra/gpuRIR ; FRA-RIR: https://github.com/tencent-ailab/FRA-RIR ; rir-generator: https://github.com/audiolabs/rir-generator ; FAST-RIR: https://github.com/anton-jeran/FAST-RIR
- Distance-based separation: https://arxiv.org/pdf/2207.00562 ; https://arxiv.org/pdf/2505.14433 ; distance estimation https://arxiv.org/html/2403.17514
- Treble / DFN3 accurate room simulation: https://arxiv.org/html/2608.20971v1
- Realistic room simulation for far-field KWS: https://arxiv.org/pdf/2006.02774

Devices and evaluation
- MicIRP: https://micirp.blogspot.com/p/about-micirp.html
- Surrey microphone IRs: https://zenodo.org/records/4633508
- Krisp benchmark: https://huggingface.co/datasets/Krisp-AI/VoiceIsolation-Benchmark-Dataset ; https://krisp.ai/blog/voice-isolation-benchmark/
- AMI: https://groups.inf.ed.ac.uk/ami/corpus/ ; RealMAN: https://github.com/Audio-WestlakeU/RealMAN ; DiPCo: https://arxiv.org/pdf/1909.13447

Compute and storage
- Google AI plans / Colab benefits: https://one.google.com/about/google-ai-plans/ ; https://support.google.com/googleone/answer/16286513
- Colab limits and unit rates (third-party): https://fastero.com/blog/google-colab-pro ; https://fastero.com/blog/google-colab-limits ; https://varlog.info/colab-credit-consumption-comparison-2024/
- Drive quota issue: https://github.com/googlecolab/colabtools/issues/1020
- GPU prices (third-party trackers): https://altstreet.investments/tools/gpu/prices ; https://tech-insider.org/runpod-vs-lambda-vs-vast-ai-2026/
- Hugging Face storage: https://huggingface.co/docs/hub/en/storage-limits ; https://huggingface.co/pricing
- R2 / B2 pricing (third-party): https://egresscost.com/cloudflare/ ; https://egresscost.com/backblaze-b2/
- Modal / Lightning / Kaggle (third-party): https://lightning.ai/pricing ; https://fastero.com/blog/kaggle-notebooks
