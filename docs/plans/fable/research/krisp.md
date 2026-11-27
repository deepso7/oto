Raw research notes, 2026-11-27; claims marked UNVERIFIED were not confirmed from a primary source

# Krisp as a benchmark target: facts, comparisons, and a measurable gate set

Conventions used below:

- "Primary" means Krisp's own docs/blog/Hugging Face, or the project's own repo/paper.
- "Self-reported" means the number comes from the party that benefits from it.
- UNVERIFIED means I only found it in a secondary source, could not read it (image-only tables), or inferred it.
- All pages were read on 2026-11-27.

## 0. The ten facts that matter most

1. Krisp has no fullband (48 kHz) model. Every outbound model works internally at 8, 16 or 32 kHz and resamples. Krisp's docs say "Sampling rates above 32kHz will be downsampled and will not be noticeable by the human ear." Voice Isolation 2.5 (server) lists 16 kHz bandwidth; only the "HD"/"pro" variant is 32 kHz.
2. Krisp's stated algorithmic latency is 15 ms for BVC / Voice Isolation, and about 25 ms for NC with 10 ms frames at 16 kHz. A third party measured 25 to 28 ms end to end on VIVA.
3. BVC / Voice Isolation needs no enrollment (primary, quoted below).
4. BVC is gated to headsets through a device allow list (about 291 entries in the consumer app list). Built-in laptop mics only "might demonstrate acceptable performance". Unlisted devices fall back to NC.
5. BVC is available in the browser (Krisp JS SDK and LiveKit's web package). Native mobile LiveKit SDKs get NC only.
6. Model files: NC 2.8 to 5.8 MB, BVC 8.6 to 14.7 MB. Browser SDK: 12 MB package, about 100 MB memory, 1.5 to 2 ms per 10 ms frame.
7. The Krisp Voice Isolation Benchmark is real: 265 real recordings, 32 kHz mono WAV, English, CC BY-NC 4.0, on Hugging Face, with Krisp's own model outputs published. Released September 2026.
8. On that benchmark with one fixed recogniser (Parakeet TDT 0.6B v2), Krisp VI 2.5 Default scores 5.30% WER and 1.52 leaked background words per minute. Unprocessed is 23.21% / 66.48.
9. At least two open models already claim a statistical tie or small win there: VoxLock (27M params, 4.95%) and Nearfield (0.63M params, 4.99 to 5.01%, but tuned on the benchmark, 48 to 80 ms latency). Both are 16 kHz only.
10. There is no public, reproducible benchmark with Krisp NC (noise-only) outputs. Any "beats Krisp NC" claim needs a Krisp licence or a recording route through a product that embeds Krisp.

## 1. Krisp product and model lineup (outbound mic)

### 1.1 Naming

Krisp uses three product lines whose names overlap:

- RTC SDK (on-device, calls and apps): "NC" (noise cancellation) and "BVC" (background voice cancellation).
- VIVA SDK and Krisp API (server side, voice agents): "Voice Isolation" (VI), plus turn-taking, interrupt prediction and VAD.
- Consumer desktop app: the BVC feature is now labelled "Voice Isolation" in the help centre.

Sources:
- https://developers-docs.krisp.ai/docs/models-and-versions
- https://help.krisp.ai/hc/en-us/articles/15856897420444-Background-Voice-Cancellation-with-Krisp-AI-Meeting-Assistant

### 1.2 RTC SDK models (primary)

Source: https://developers-docs.krisp.ai/docs/rtc-model-guide-bvc-nc (model files use a `.kef` extension).

| Model | Stream | Working rate | Size | Krisp's note |
|---|---|---|---|---|
| krisp-nc-o-pro-v1 | outbound NC | 32 kHz | 5.8 MB | "Outstanding quality with balanced CPU usage" |
| krisp-nc-o-med-v7 | outbound NC | 32 kHz | 5.7 MB | mobile and power-limited systems |
| krisp-nc-o-lite-v1 | outbound NC | 16 kHz | 2.8 MB | only when low CPU is the absolute priority |
| krisp-nc-o-nb-v2 | outbound NC | 8 kHz | 2.8 MB | narrowband, on demand |
| krisp-bvc-o-pro-v3 | outbound BVC | 32 kHz | 14.7 MB | "Preferred choice for headset and earbuds." |
| krisp-bvc-o-lite-v2 | outbound BVC | 16 kHz | 8.6 MB | "Must be used with USB headsets or headsets with jack connectors only." |
| krisp-nc-i-wb-pro-v3 | inbound NC | 16 kHz | 13 MB | conference and telephony inbound |
| krisp-nc-i-nb-pro-v1 | inbound NC | 8 kHz | 3.6 MB | telephony inbound |
| krisp-nc-i-nb-lite-v2 | inbound NC | 8 kHz | 4.1 MB | telephony inbound, low CPU |

Also: `krisp-nc-o-max-v1` and `krisp-bvc-o-max-v3` are large models "available for evaluation on request" (https://developers-docs.krisp.ai/docs/models-and-versions). No sizes given.

Other statements from the same guide:

- "Audio is resampled to the model's working rate and back"; "Sampling rates above 32kHz will be downsampled and will not be noticeable by the human ear."
- The pro NC model "consumes significantly more CPU due to its different architecture" than the medium one despite similar size.
- Outbound models are "Trained to handle single-speaker" and expect raw, unmodified mic audio.
- "You should never use two Krisp NC/BVC models on the same audio stream." BVC includes NC.

Inference (UNVERIFIED): if the `.kef` files hold float32 weights, 5.8 MB is about 1.4M parameters and 14.7 MB about 3.7M. If they are int8 the counts are 4x higher. Krisp publishes no parameter counts.

### 1.3 NC behaviour (primary)

Source: https://developers-docs.krisp.ai/docs/krisp-rtc-bvc-nc

- Sample rates: "The models work natively at 32 kHz, 16 kHz and 8 kHz." Higher rates are downsampled, processed, upsampled back.
- Latency: depends on rate and frame duration; "with 10 ms frames at 16 kHz it is about 25 ms."
- Sizes: "Small" is the SDK default and "runs about 7x faster than Big"; "Big" is on demand.
- Near field: designed for "the mouth less than 50 cm from the microphone."
- De-reverb: "NC also removes room echo (reverberation) from the user's voice automatically." There is no separate de-reverb model.
- AEC placement: before Krisp NC on the capture path.
- CPU: no numbers in text for NC. Per-platform CPU tables are images (UNVERIFIED, could not read). Benchmark machines are a Dell Inspiron i5-1135G7, a 2019 Intel MacBook Pro and an M1 MacBook Pro, Desktop SDK v9.0.0 (https://developers-docs.krisp.ai/docs/desktop-sdk-bvc).

### 1.4 BVC behaviour (primary)

Source: https://developers-docs.krisp.ai/docs/krisp-rtc-bvc-nc

- Enrollment: "The technology does not require user voice enrollment or training on user voice data." Confirmed enrollment-free.
- Rate: "The BVC model works at the 32KHz sampling rate." Input above 8 kHz is resampled to 32 kHz. Not usable at 8 kHz or below. So BVC is super-wideband, not fullband.
- CPU: "varies depending on the platform within the range of 5-13%."
- Removes nearby voices plus noise plus reverberation. Outbound only.
- Devices: designed "to work with any headset and earbud"; "It works best with wired USB headsets with a boom microphone"; compatible with most Bluetooth headsets including AirPods.
- Built-in mics: the model "might demonstrate acceptable performance" with high-quality built-in or external mics such as MacBooks. Hedged, not supported.
- Block list: some devices report a high rate but capture narrowband; then "BVC quality can drop and the user's voice may be suppressed."
- BVC latency: not stated on this page. The server blog says 15 ms, "the same as Krisp's on-device models" (https://krisp.ai/blog/improving-turn-taking-of-ai-voice-agents-with-background-voice-cancellation/).

Device gating details:

- Device SDK: integration "requires detecting the connected audio device, matching against the allow/block lists tested by Krisp" plus an optional local allow list (https://developers-docs.krisp.ai/docs/desktop-sdk-bvc).
- Consumer app: Voice Isolation requires a compatible headset and an "enclosed area"; it activates automatically in a meeting. Listed devices "are selected automatically"; unlisted headsets "may still work with reduced voice or Noise Cancellation quality, and can often be activated manually." The list has about 291 entries (my count), all headsets and their dongles, updated 2026-09-23 (https://help.krisp.ai/hc/en-us/articles/30534957581596-Devices-compatible-with-Voice-Isolation-in-Krisp-AI-Meeting-Assistant).
- Third-party integrator (Gather): on unapproved devices BVC "will fall back to regular Enhanced Background Noise Reduction" (https://support.gather.town/hc/en-us/articles/15999534644372-Background-Noise-Reduction).

Supported frame sizes and rates in the Device SDK: frames 10, 15, 20, 30, 32 ms; rates 8, 16, 24, 32, 44.1, 48, 88.2, 96 kHz (https://developers-docs.krisp.ai/docs/desktop-sdk-bvc). These are I/O rates; the model rate stays at 32 kHz or lower.

### 1.5 Server-side Voice Isolation (VIVA)

Model list (https://developers-docs.krisp.ai/docs/models-and-versions):

| Model | Rate | Note |
|---|---|---|
| krisp-viva-vi-tel-v2.5.1 | 8 to 16 kHz | telephony, "Best noise cancellation." |
| krisp-viva-vi-tel-lite-v2.5 | 8 to 16 kHz | "Lowest CPU usage." |
| krisp-viva-vi-tel-v2.7 | 8 to 16 kHz | "The most ASR-friendly model." |
| krisp-viva-vi-pro-v1 | above 16 kHz | WebRTC |
| krisp-viva-nc-ss-v1 | not stated | smart-speaker / far-field NC, keeps background voices |

Voice Isolation 2.5 (https://krisp.ai/blog/voice-isolation-2-5/, August 2026, updated September):

- Algorithmic latency 15 ms. CPU only. Audio bandwidth 16 kHz. Handles G.711, G.729, G.722, Opus.
- "VI 2.5 lite" is about 3.5x smaller in parameters.
- No enrollment or primary-speaker mechanism is described.
- Krisp's own internal benchmark (1,685 recordings, about 7 h, 10 STT systems): competing speech WER 35.92% raw to 10.90% with VI 2.5; clean 2.10% raw to 2.15%; "extra challenging acoustics" 8.02% raw to 11.61% with VI 2.5 (worse than raw).

The benchmark post names four variants: Default, Balanced, Lite, HD (32 kHz). The CPU post calls Balanced "v2.7" once. Mapping between blog names and `.kef` names is partly my inference (UNVERIFIED): Default = vi-tel-v2.5.1, Lite = vi-tel-lite-v2.5, Balanced = vi-tel-v2.7, HD = vi-pro-v1.

Older server names: BVC-tel (up to 16 kHz) and BVC-app (up to 32 kHz), 15 ms algorithmic latency, Linux x64/ARM64 and Windows x64 (https://krisp.ai/blog/improving-turn-taking-of-ai-voice-agents-with-background-voice-cancellation/).

Server CPU cost (primary, https://krisp.ai/blog/voice-isolation-cpu-optimization/, 20 ms frames, August 2026, Python SDK 1.12.0):

| CPU (16 vCPU) | Model | FP16 | Single-stream RTF p95 | ms per 20 ms frame | Max streams |
|---|---|---|---|---|---|
| Graviton3 | Default | off | 0.149 | 2.98 | 76 |
| Graviton3 | Default | on | 0.074 | 1.48 | 138 |
| Graviton3 | Lite | off | 0.051 | 1.01 | 209 |
| Graviton3 | Lite | on | 0.027 | 0.54 | 358 |
| Sapphire Rapids | Default | off | 0.119 | 2.39 | 45 |
| Sapphire Rapids | Lite | off | 0.041 | 0.82 | 127 |
| Ice Lake (no FP16) | Default | n/a | not recorded | n/a | 39 |

So the full VI model costs roughly 7 to 15% of one modern server core per stream, and Lite 2.5 to 5%.

Third-party CPU and delay measurement (Hamsa, a competitor; single thread AMD EPYC 7R32, 10 ms frames, Krisp SDK 1.4.0; https://github.com/hamsa-ai/hamsa-safi-v1-fast):

| System | CPU ms per 10 ms frame | Measured delay |
|---|---|---|
| Krisp VIVA Tel v2 | 0.95 | about 25 ms |
| Krisp VIVA Pro v1 | 0.98 | about 28 ms |
| Hush | 0.84 | about 10 ms |
| RNNoise | 0.25 | about 20 ms |

### 1.6 Accent conversion

Source: https://developers-docs.krisp.ai/docs/accent-conversion

- Converts Indian, Philippine, African and Latin American English to North American English, preserving the voice.
- "Low latency (~180ms)". "AC requires no user enrollment".
- Platforms: Windows, Mac, Linux, Chrome, iOS, Android. A browser SDK exists.
- CPU, model size, sample rate: not stated. Not a relevant target for an enhancement model.

### 1.7 Quality evidence Krisp publishes for NC

Source: https://developers-docs.krisp.ai/docs/krisp-audio-quality-testings

- 3QUEST (ETSI TS 103 281) N-MOS and S-MOS, tested by SigmaConnectivity in an ETSI room against Zoom, MS Teams, WebEx and Google Meet, Q4 2022. Scores are in an image only (UNVERIFIED numbers). Krisp claims a win on both.
- No DNSMOS, PESQ, POLQA or DNS Challenge numbers for NC anywhere I found. I found no Krisp entry in any DNS Challenge (UNVERIFIED absence; not exhaustively searched).

## 2. SDKs, with focus on browser

### 2.1 Krisp Web Browser (JS) SDK (primary)

Sources:
- https://developers-docs.krisp.ai/docs/web-browser-sdk.md (updated 2026-10-09)
- https://developers-docs.krisp.ai/docs/getting-started-browser-sdk-nc-bvc
- https://developers-docs.krisp.ai/docs/background-voice-cancellation-integration.md

Facts:

- Runs "in the browser with WebAssembly"; "The noise-canceling neural network is built on WASM for high performance."
- `createNoiseFilter` "returns `AudioWorkletNode`". Processing happens "in a dedicated worker to offload the main thread", so the worklet exchanges audio with a worker thread.
- `useSharedArrayBuffer` (default false) "Enables using SharedArrayBuffer between worker and worklet threads to exchange audio data improving CPU performance." So SAB is optional, not required.
- SIMD: not mentioned anywhere (UNVERIFIED either way).
- "Package Size: 12MB". "Memory Footprint: ~100 MB". "Frame Processing: 1.5-2ms". Processing "is performed using 10ms frames". That implies roughly 15 to 20% of one core in the browser; CPU% charts are images (UNVERIFIED).
- "CPU% is generally higher in JS SDK compared to native SDK."
- Accepted AudioContext rates: 8000, 12000, 16000, 24000, 32000, 44100, 48000, 88200, 96000.
- Browsers: "Chrome, Firefox, and Safari version v.17.4.1 and above." "8K audio stream is not supported by the Safari browser".
- Models in the browser: `model_8.kef` (8 kHz NC), `model_nc_mq.kef` (NC above 8 kHz), `model_bvc.kef` (BVC), inbound 8k/16k, ringtone model. Per-file sizes are not stated. "BVC model is bigger than NC models", so preload is advised. The memory figure for BVC "is measured for the small model exclusively" (so the browser likely ships the smaller BVC; UNVERIFIED which one).
- BVC in the browser: `useBVC: true`, and `bvc.allowedDevices` (a bundled `bvc-allowed.txt`) is mandatory. Devices are matched by name; `KrispSDK.BVC.add(name, force)` extends the list. `isBVCEnabled()` reports whether BVC is active.
- Buffering: `bufferOverflowMS` default 200 ms; audio is dropped at 500 ms. Under load "the JS SDK may malfunction and result in choppy voice output."
- Known issues listed by Krisp: Chrome echo cancellation "muffles speech"; "voice degradation (sounding cartoon-like) for about 10 seconds when switching between devices"; "Randomly, after switching between built-in device microphones and headsets, there may be BVC quality degradation with secondary voices passing."
- Distribution: only via the Developers Portal, not public npm.
- End-to-end browser latency: not stated.

Historical (2020): the first Chrome extension used a 10x smaller DNN, 30 ms DNN frames, XNNPACK via WASM, and split inference across 3 ms callbacks (https://krisp.ai/blog/how-we-shrunk-dnn-to-run-inside-chrome/).

### 2.2 Integrations

- LiveKit (https://docs.livekit.io/transport/media/noise-cancellation/): Web frontend gets NC and BVC via `@livekit/krisp-noise-filter`; "The BVC model is available in the JavaScript frontend; other frontend SDKs support the NC model only." Models download at runtime. Agent side: Krisp NC (free on LiveKit Cloud), Krisp VIVA and VIVA telephony (paid). Safari below 17.4 unsupported.
- npm check (registry, 2026-11-27): `@livekit/krisp-noise-filter` 0.4.6, unpacked 12.5 MB, two JS bundles of about 6.1 and 6.4 MB (the WASM is embedded in the JS). `@twilio/krisp-audio-plugin` 2.0.0, unpacked 13.8 MB.
- Twilio Video (https://www.twilio.com/docs/video/noise-cancellation.md): needs `Worklet.addModule`; Chrome 66+, Edge 79+, Firefox 76+; desktop Safari and all iOS browsers unsupported; CSP needs `unsafe-eval` with `default-src self`; one audio track per participant.
- GetStream's web audio filters lazy-load a `krispsdk.js` chunk of 4,965,228 bytes (https://github.com/GetStream/stream-video-js/pull/2588).
- Daily: partnered in 2023; `updateInputSettings()` with processor type `noise-cancellation`; at launch desktop Chromium and Firefox only (https://dev.to/trydaily/daily-partners-with-krisp-to-provide-world-class-noise-cancellation-capabilities-l8d). Current details UNVERIFIED.
- Discord: uses Krisp on desktop and mobile (Wikipedia, secondary). Model details UNVERIFIED.
- Agora: found nothing tying Agora's AI noise suppression to Krisp. UNVERIFIED.
- Pricing: one marketing blog says "Krisp BVC went paid May 1 2026" for SDK consumers (https://callsphere.ai/blog/vw5e-webrtc-ai-noise-suppression-krisp-rnnoise-patterns-2026). Single unofficial source, UNVERIFIED. Krisp API pricing per a third party: $0.18/h Starter, $0.12/h on a $400/month plan; on-prem SDK is enterprise only (https://dev.to/kevin_menesesgonzlez/6-best-voice-isolation-apis-and-models-for-ai-voice-agents-2026-tested-4g7e).

## 3. Krisp Voice Isolation Benchmark

### 3.1 What it is (primary)

Sources:
- Dataset: https://huggingface.co/datasets/Krisp-AI/VoiceIsolation-Benchmark-Dataset
- Collection (includes model outputs): https://huggingface.co/collections/Krisp-AI/krisp-voice-isolation-benchmark-suite
- Space with results: https://huggingface.co/spaces/Krisp-AI/VoiceIsolation-Benchmark
- Results post (2026-09-23, updated October): https://krisp.ai/blog/voice-isolation-benchmark/
- Dataset story (2026-09-28): https://krisp.ai/blog/speech-recognition-dataset-background-voices/

Facts:

- Licence: `cc-by-nc-4.0` in the dataset card. A search snippet suggested the Space metadata says cc-by-nc-nd-4.0 (UNVERIFIED; the Space page did not render).
- 265 real recordings, 47 speakers, English only. "No synthetic mixing, no noise added in post."
- Audio: "uncompressed WAV files (PCM 16-bit signed little-endian), mono channel, sampled at 32 kHz." So it cannot show any advantage above 16 kHz audio bandwidth.
- Clip length 61 to 149 s in the call-centre subset. Total 337.5 min, of which 85.3 min is background-only speech (figures from third-party papers, see 5.1 and 5.2).
- Subsets: Work 96 files (29 speakers, 24 devices, 4 room types), Call center 106 files (7 speakers, 13 to 18 headset models, 4 real floors), Phone calls 63 files (18 phone models; almost no second voice, acts as a "do no harm" control).
- Script: primary speaks, secondary joins, primary goes silent while secondary continues, primary returns.
- Annotation: verbatim hand transcript plus segments labelled `primary`, `mix`, `secondary`, `noise`. Metadata has device, environment, gender, speaker id.
- The results post mentions a private slice held back as a hidden test set (stated in the blog, not in the dataset story).
- No clean reference signal exists, so intrusive metrics (PESQ, SI-SDR, STOI) cannot be computed on it.
- Evaluation harness: said to be published, but I found no repository URL (UNVERIFIED location).

### 3.2 Metrics

- WER: `jiwer` 4.0.0, pooled at corpus level; normalisation is a regex pass, NVIDIA NeMo text processing, then lowercasing, contraction expansion, punctuation and filler removal.
- Perceptual: DNSMOS-C (https://github.com/Hope-Liang/DNSMOS-C), reference-free, on primary-speaker segments, 10 s windows.
- 11 STT configurations (9 engines), 5 runs per file.
- Third parties added "leak" (background words transcribed per minute of background-only speech) and "missed" (caller words lost). Those are not Krisp's metrics but are the most informative ones.

### 3.3 Krisp's published scores

Pooled WER (%), mean over 11 STT configurations:

| Scenario | Raw | Default | Balanced | Lite | HD |
|---|---|---|---|---|---|
| Work | 31.84 | 6.30 | 6.92 | 7.43 | 7.34 |
| Call center | 23.83 | 6.86 | 6.56 | 8.35 | 6.98 |
| Phone calls | 3.48 | 3.91 | 3.53 | 3.81 | 3.89 |
| All 265 | 23.24 | 6.24 | 6.21 | 7.37 | 6.62 |

Parakeet TDT 0.6B v2 row (the recogniser third parties use): raw 22.88, Default 5.31, Balanced 5.31, Lite 6.21, HD 5.77.

DNSMOS-C:

| Scenario | Raw | Default | Balanced | Lite | HD |
|---|---|---|---|---|---|
| Work | 3.43 | 3.96 | 3.78 | 3.96 | 3.92 |
| Call center | 3.65 | 4.02 | 3.94 | 4.01 | 3.97 |
| Phone calls | 3.11 | 3.45 | 3.42 | 3.42 | 3.22 |

Notes:

- Every Krisp model makes phone calls slightly worse than raw. One file goes from 9.42% to 21.65%.
- Krisp publishes no competitor rows. It says the set is "not a leaderboard".
- Over-isolation shows up as deletions (missing clauses), not substitutions.

### 3.4 Other Krisp numbers on public data

- Turn-taking post (AMI headset recordings plus ITU-T P.501 noise mixes, Silero VAD, Whisper v3): VAD false triggers down 3.5x on AMI; WER more than 2x better on AMI; on P.501 noise, WER "increased by about 2x" when Krisp-processed audio was fed to Whisper. Absolute values are in images (UNVERIFIED). Source: https://krisp.ai/blog/improving-turn-taking-of-ai-voice-agents-with-background-voice-cancellation/
- Developer page headline: 46% average WER reduction, 70% with background speech (https://krisp.ai/developers/).

## 4. Third-party comparisons with numbers

Quality of evidence is poor across the board: almost every comparison was run by a competitor. I found no neutral lab test with Krisp NC against DeepFilterNet3, NVIDIA, Apple, Dolby, Hance or DTLN.

### 4.1 Hamsa (competitor, closed model) vs Krisp VIVA, RNNoise, Hush

Source: https://github.com/hamsa-ai/hamsa-safi-v1-fast. FLEURS speech plus MS-SNSD noise at -5 to +15 dB SNR, streaming, 10 ms frames, Krisp SDK 1.4.0 at NS level 100. Self-reported.

English wideband 16 kHz (n = 188):

| System | PESQ | STOI | SI-SDR | SIG | BAK | OVRL | WER (Whisper large-v3) |
|---|---|---|---|---|---|---|---|
| Hamsa Safi v1 | 1.74 | 0.763 | 9.21 | 3.37 | 4.07 | 3.11 | 20.5% |
| Krisp Tel v2 | 1.54 | 0.703 | 6.34 | 2.94 | 3.95 | 2.68 | 30.2% |
| Krisp Pro v1 | 1.56 | 0.737 | 7.12 | 3.01 | 3.99 | 2.75 | 25.0% |
| RNNoise | 1.44 | 0.701 | 5.45 | 3.20 | 3.92 | 2.89 | 34.8% |
| Hush | 1.43 | 0.678 | 3.56 | 3.08 | 3.52 | 2.61 | 31.2% |
| Unprocessed | 1.34 | 0.708 | 4.04 | 2.63 | 1.96 | 1.90 | n/a |

Takeaways: in this test Krisp VIVA's DNSMOS SIG (speech quality) is below RNNoise, and its PESQ and SI-SDR at low SNR are modest. Krisp is strong on BAK (background removal) and on telephone-band audio. This is Krisp's voice-isolation model, not Krisp NC.

### 4.2 ai-coustics (Quail, Voice Focus)

- LiveKit docs, one sample, Deepgram Nova-3: original 117.6% WER; Voice Focus S 7.1%; Voice Focus L 14.3%; Krisp VIVA 11.8% (https://docs.livekit.io/transport/media/noise-cancellation/). One clip is not a benchmark.
- SLNG benchmark (FLEURS, 50 clips per language, Deepgram Nova-3, synthetic mixes), as quoted on ai-coustics' page: Quail Voice Focus "reduced WER by 4.23pp at full intensity"; "Krisp NC raised WER by over 10pp" on competing speech; on background noise "NVIDIA Maxine and Krisp NC increased WER instead, by 2.40-4.29pp." Sources: https://ai-coustics.com/benchmarks-quantitative and https://slng.ai/slng-noise-cancellation-stt-benchmark. Caveats: SLNG shipped ai-coustics models, the full tables are in a PDF I did not read, and SLNG ceased trading on 2026-10-21. Krisp NC is a noise-only model, so losing on competing speech is expected.
- ai-coustics specs: Voice Focus 2.1 has 30 ms end-to-end latency, 8 and 16 kHz, runs on CPU; S is 10x smaller than 2.0 (https://ai-coustics.com/blog/quail-voice-focus-2.1). A third-party harness measured a 30 ms output shift for every ai-coustics SDK model and lists a 48 kHz "Rook" model (https://github.com/finaea/de-denoiser).
- On Krisp's own benchmark (run by the VoxLock author): Voice Focus 2.2 L 6.84% WER / 0.66 leak; 2.2 S 6.91% / 1.00. Krisp Default is ahead on WER, ai-coustics ahead on leak.

### 4.3 Small independent test on a Krisp-benchmark subset

Source: https://dev.to/kevin_menesesgonzlez/6-best-voice-isolation-apis-and-models-for-ai-voice-agents-2026-tested-4g7e (freelance content writer, 24 files, faster-whisper base.en).

| Tool | WER | Leaked words |
|---|---|---|
| Raw | 35.2% | 548 |
| Krisp VIVA-tel | 18.5% | 63 |
| Krisp VIVA-pro (32 kHz) | 21.5% | 130 |
| DeepFilterNet3 | 29.5% | 282 |
| DPDFNet | 32.6% | 498 |

Notes: DeepFilterNet3 doubled main-speaker deletions (88 to 184). The 32 kHz Krisp model did worse than the 16 kHz one. Weak evidence.

### 4.4 Others

- Picovoice Koala: Picovoice's own benchmark covers Koala vs RNNoise (STOI only) and left Krisp out for SDK-access reasons (https://picovoice.ai/blog/best-noise-suppression-software-for-streaming/). No Krisp numbers.
- Sanas SE 1.0 on Krisp's benchmark: 9.38% WER, 13.76 leak (VoxLock paper).
- ElevenLabs Voice Isolator on 7 Krisp files: 43.79% WER (VoxLock repo). Not a realtime product.
- NVIDIA Broadcast/Maxine: latency not published; needs an RTX GPU (https://www.forasoft.com/learn/audio-for-video/articles-audio/noise-suppression-rnnoise-krisp-rtx-voice). No head-to-head numbers with Krisp found beyond the SLNG line above.
- Apple Voice Isolation, Dolby, Hance, DTLN vs Krisp: no numeric comparison found. UNVERIFIED / absent.
- Reference latencies quoted in "The Voice on the Line": DeepFilterNet3 40 ms, VoxLock 24 ms.

## 5. Open projects that claim to match or beat Krisp at background-voice removal

Summary table. All are 16 kHz. None is 48 kHz.

| Project | Params | Size | Algorithmic latency | CPU | Licence | Krisp-benchmark WER / leak | Credibility of Krisp claim |
|---|---|---|---|---|---|---|---|
| VoxLock | 27M | 47 to 94 MB | 24 ms plus chunk (practical 64 to 104 ms) | 61 to 110% of a core | Apache-2.0 | 4.95% / 0.11 (live: 5.15% / 0.12) | Medium to high. Reproduced independently. |
| Nearfield v2.3 | about 628k | 2.5 MB | 48 ms (fast), 80 ms (best) | about 20% of a core | Apache-2.0 | 5.01% / 0.70 and 4.99% / 0.36 | Low to medium. Tuned on the test set, no error bars, 4 stars. |
| Scicom "Voice on the Line" C22 | 22.1M | n/a | 20 ms | 33% of a core at 20 ms frames | unclear | 5.66% | Makes no "beats Krisp" claim. |
| alkove Focus 1 | not stated (preview about 0.8M) | 2.3 MB | 16 ms plus 16 ms hop | RTF 0.05 to 0.2 | Apache-2.0 | none published | No Krisp claim found. |
| Hush (Weya) | about 1.8M | 8 MB | about 20 ms | under 1 ms per 10 ms frame | Apache-2.0 | 12.09% / 19.58 (run by VoxLock author) | Makes no Krisp claim; clearly behind. |
| Krisp VI 2.5 Default (reference) | unknown | unknown | 15 ms | about 10% of a core | proprietary | 5.30% / 1.52 | n/a |

### 5.1 VoxLock

Sources: https://arxiv.org/abs/2610.40727 (29 Oct 2026), https://github.com/arbazsiddiqui/voxlock, https://huggingface.co/arbazsiddiqui/voxlock

- Architecture: causal band-split RNN target-speaker extractor (REAL-TSE baseline from WeSep), 512-sample STFT, 128 hop, six band-split blocks, ECAPA-TDNN conditioning. 27M parameters. 16 kHz.
- Enrollment-free trick: it takes 3 s of the caller from the call itself (loudest 3 s in the first 15 s for whole files; in streaming it passes audio through unprocessed until 3 s of speech is heard, then stalls 2 to 5 s to warm the state).
- CPU (int8, one thread, Xeon 2.2 GHz): 110% at 20 ms chunks, 77% at 40 ms, 61% at 80 ms.
- Training: one NVIDIA L4, about 9,200 fine-tuning steps in four stages; LibriSpeech simulated calls plus AMI headset crosstalk. Loss is negative SI-SDR.
- Browser: demo uses the fp16 model, 47 MB download. No browser CPU numbers.
- Krisp comparison: the author scored Krisp's published outputs, did not run Krisp. Parakeet TDT 0.6B v2.

| System | WER % | Missed % | Leak /min |
|---|---|---|---|
| VoxLock whole file | 4.95 | 4.22 | 0.11 |
| VoxLock live() | 5.15 | 4.41 | 0.12 |
| Krisp VI 2.5 Default | 5.30 | 4.21 | 1.52 |
| Krisp VI 2.5 Balanced | 5.32 | 4.06 | 2.02 |
| Krisp VI 2.5 HD | 5.79 | 4.37 | 2.74 |
| Krisp VI 2.5 Lite | 6.21 | 4.26 | 4.52 |
| ai-coustics VF 2.2 L | 6.84 | 6.03 | 0.66 |
| Sanas SE 1.0 | 9.38 | 4.90 | 13.76 |
| Hush | 12.09 | 6.27 | 19.58 |
| DeepFilterNet3 | 13.30 | 6.46 | 22.77 |
| No processing | 23.21 | 4.28 | 66.48 |

- Significance (bootstrap over speakers, uncorrected): not significantly different from Krisp Default or Balanced on WER; significantly better than Lite; borderline vs HD.
- Per scenario: on phone calls VoxLock raises WER from 3.82% to 5.36%, worse than Krisp Default's 4.15%.
- Stated weaknesses: locks onto the wrong speaker if a background talker dominates the opening seconds; the Krisp set was seen during development; headline row uses 15 s of lookahead for the sample.
- Independent reproduction (Scicom paper): 4.99% with the PyTorch script, 5.19% with the streaming ONNX package, 5.63% in true live mode. So in a fair causal comparison VoxLock is slightly behind Krisp Default (5.30%).

### 5.2 "The Voice on the Line" (Scicom)

Source: https://arxiv.org/html/2610.35272v1 (26 Oct 2026), https://huggingface.co/collections/Scicom-intl/voice-isolator

- Release model: 555M params (243M causal transformer plus causal WavLM-Large), GPU. CPU student C22: 22.1M params, 33% of one core at 20 ms frames (22% int8), 20 ms algorithmic latency, 16 kHz.
- Training data: 2,761 h clean speech, 51 h noise, 20,000 simulated rooms, 620 h real crosstalk from AMI, NOTSOFAR and AliMeeting.
- Krisp-benchmark WER: release 5.74%, C22 5.66%, Krisp Default 5.30%, DeepFilterNet3 13.15%, raw 23.19%.
- Model licence unclear; results are "for internal study until a release is cleared".
- Useful critique of the Krisp benchmark: scripted calls; almost no digital silence, so it hides cold-start and lone-caller damage; 47 callers; being level on Krisp hides large differences on long meetings (AMI 60 s: 16.13% vs VoxLock live 23.66%) and on clean calls (5.58% vs 2.92%).
- Run-to-run noise: two runs of the same recipe differ by 0.1 to 0.2 WER points. Gaps of that size "rank runs rather than recipes".
- A noise-only reference point: GTCRN costs 5% of a core.

### 5.3 Nearfield (Ground-State-Foundry/nearfield)

Source: https://github.com/Ground-State-Foundry/nearfield (v1.0.0 on 10 Nov 2026; v2.3 current)

- Architecture: 32 ms window, 8 ms hop, 64 bands, four recurrent blocks (bidirectional across frequency, causal in time), complex mask. About 628,000 weights, 2.5 MB ONNX. Trained from scratch (v1 was built on DeepFilterNet3 and worked at 48 kHz; v2 dropped to 16 kHz).
- Lookahead 16 ms (fast) or 48 ms (best); total latency 48 ms or 80 ms.
- CPU: about 0.2 s per second of audio on one Xeon 8488C thread.
- Heavy post-processing: levelling, pause attenuation up to 18 dB, -5 dB tilt above 5 kHz.
- Training data: simulated calls from public speech and noise; datasets unnamed; training code not released.
- Krisp-benchmark claim: "Beats Krisp and Sanas on Krisp's own benchmark", 4.99% / 0.36 (best) and 5.01% / 0.70 (fast) vs Krisp Default 5.30% / 1.52.
- Their own caveats: "Gaps of a few tenths of a point are within noise. We have no error bars for this set." "v2.3 was tuned against this benchmark"; `best` is the snapshot that scored best on it. v2.2 scored 7.83% and 8.37% on the same set.
- On their synthetic test v2.3 got worse than v2.2, and `fast` loses the caller's first word in 25% of equal-talker clips.
- Credibility: low to medium. The model is far smaller than Krisp's and the latency is 3 to 5x Krisp's 15 ms. The result is selected on the test set. 4 GitHub stars, no independent reproduction found.

### 5.4 alkove Focus

Sources: https://github.com/alkove-ai/alkove, https://huggingface.co/alkove-ai/alkove-focus-1

- Initialised from the GTCRN DNS3 checkpoint; conditioned on an optional 192-dim voiceprint (zeros means "keep the closest voice"). 2.3 MB with int8 and 4-bit weights. Parameter count not stated; the preview was about 0.8M (search snippet, UNVERIFIED).
- 16 kHz only. Hop 256 samples; output is 16 ms late on top of the 16 ms hop.
- CPU: 46 to 57 ms per second of audio natively (M1 Max, Galaxy Fold), 170 to 195 ms with the user-sample path. Own Rust engine, 185 KB brotli WASM, single thread.
- Apache-2.0 code and weights. Training data undisclosed.
- No Krisp comparison and no Krisp-benchmark result. On AMI (VoxLock protocol): WER 16.8% vs VoxLock 15.1%; leak 2.3 vs 0.4 words/min.
- Without a voiceprint it barely suppresses an overlapping voice in their synthetic test (-1.4 dB vs -33.8 dB with a print). So its enrollment-free mode is weak.
- Honest card: says DeepFilterNet3 and GTCRN are better for plain denoising (VoiceBank+DEMAND DNSMOS 3.07 vs 3.21 for DeepFilterNet3; WER 5.6% vs 2.7% for the noisy input).

### 5.5 Hush (pulp-vision/Hush, also weya-ai/hush)

Source: https://github.com/pulp-vision/Hush

- DeepFilterNet3 architecture retrained at 16 kHz with an auxiliary separation head used only in training. About 1.8M params, 8 MB. FFT 320, hop 160, 32 ERB bands. About 20 ms algorithmic latency, under 1 ms compute per 10 ms frame.
- Training: 10,000+ hours of mixtures; LibriSpeech, VCTK, Common Voice speech; DNS Challenge, FreeSound, ESC-50 noise; 60% of samples include a competing speaker at 12 to 24 dB SIR. 100 epochs.
- Apache-2.0. No published metrics of its own and no Krisp claim.
- Third-party results: 12.09% WER / 19.58 leak on Krisp's set; 33.00% on AMI (no better than raw). It only handles quiet background speakers.
- Note: `alshell7/hush-ai` is an unrelated desktop app wrapping DeepFilterNet3.

### 5.6 Others seen

- Hamsa Safi v1: closed, claims wins over Krisp VIVA on DNSMOS and wideband CER (section 4.1).
- Arctan Eigen: closed, vendor chart against Krisp, unverifiable.
- FastEnhancer, GTCRN, UL-UNAS, DPDFNet: open noise-only models, no Krisp comparison.

## 6. Known weaknesses and complaints

From Krisp's own material (strongest evidence):

- Bandwidth capped at 32 kHz; the best voice-isolation models are 16 kHz.
- Phone-call audio with nothing to remove gets slightly worse with every VI model (3.48% to 3.53-3.91%).
- "Extra challenging acoustics" (reverb, echo, heavy noise): VI 2.5 is worse than raw (11.61% vs 8.02%).
- Over-suppression drops whole clauses of the primary speaker.
- BVC needs a headset and an allow list; built-in mics are hedged; device switching can let secondary voices through.
- Near-field only (under 50 cm).
- Browser: about 100 MB memory, 12 MB package, choppy output under load, "cartoon-like" voice for about 10 s on device switch in Chrome, no Safari in Twilio's build.
- Feeding processed audio to an STT model that never saw it can double WER on plain noise (P.501 test).

From third parties (weaker evidence):

- Lower DNSMOS SIG than RNNoise at low SNR in a competitor's test (section 4.1).
- User-review themes: robotic, muffled or choppy outgoing voice; clipped speech after long silence; quiet talkers and sentence endings cut; 10 to 20% CPU on weak laptops (https://www.bluedothq.com/blog/top-krisp-alternatives, a competitor; https://forum.finepix-x100.com/t/can-anyone-share-an-honest-krisp-ai-user-review/2201). A favourable review claims 3 to 5% CPU and under 10 ms latency (https://aumiqx.com/ai-tools/krisp-ai-review-noise-cancellation-meetings/); the latency figure contradicts Krisp's own 15 ms. All UNVERIFIED anecdotes. I found no Reddit threads in this pass.
- Headsets with built-in noise cancellation can sound muffled when Krisp is stacked on top (Gather support page).
- Closed, licensed, per-minute or enterprise pricing; no public npm package.

## 7. Judgement: what "beating Krisp" should mean

### 7.1 Decide which Krisp first

"Krisp" is three different targets:

- A. Krisp NC (noise only, 32 kHz, 5.8 MB, about 25 ms). This is the closest match to a 48 kHz causal enhancement model. There are no public Krisp NC outputs, so the comparison needs Krisp access.
- B. Krisp BVC (on-device voice isolation, 32 kHz, 14.7 MB, headsets only).
- C. Krisp VI 2.5 (server, 16 kHz, 15 ms). This is the only one with public, scoreable outputs.

A plan that says "beat Krisp" without choosing will default to C, because that is where the public numbers are, and C is a 16 kHz speech-to-text benchmark. It says nothing about 48 kHz listening quality.

### 7.2 Proposed gate set

Constraints (hard gates, measured, not claimed):

- Algorithmic latency at most 20 ms including lookahead (Krisp: 15 ms VI, about 25 ms NC). Report STFT window minus hop plus lookahead, and confirm by cross-correlation.
- Native CPU: at most 1.0 ms per 10 ms frame on one x86 server thread (Krisp VIVA measured 0.95 ms; Krisp Lite about 0.3 ms).
- Browser: at most 2 ms per 10 ms frame in single-thread WASM in Chrome on an M1-class laptop (Krisp: 1.5 to 2 ms), no SharedArrayBuffer required, total download under 12 MB, memory under 100 MB.
- True 48 kHz output with content above 16 kHz. Krisp cannot match this by design.

Track 1, voice isolation, reproducible today (vs Krisp VI 2.5 Default):

- Dataset: Krisp Voice Isolation Benchmark, all 265 files, used for evaluation only. Do not select checkpoints on it; use AMI headset windows or a self-built set for selection.
- Recogniser and scoring: Parakeet TDT 0.6B v2, Krisp's normaliser, `jiwer` pooled WER. Score in true streaming mode.
- Gates: pooled WER at most 5.30%; leak at most 1.52 words/min; phone-call subset at most 4.15% (stretch: at most raw, 3.82%); missed-caller rate at most 4.3%.
- Statistics: bootstrap over the 47 speakers. The existing results show that 4.95 to 5.32 are all statistically tied, so "beat" realistically means "not significantly worse on WER and significantly better on leak".
- Second set to avoid a Krisp-only claim: AMI headset windows (VoxLock protocol) and a 60 s long-call set, where Krisp has no public outputs but ai-coustics and open models do.
- Licence: CC BY-NC 4.0 permits evaluation and publishing numbers non-commercially. It rules out training on it and commercial use.

Track 2, noise suppression at 48 kHz (vs Krisp NC), needs Krisp access:

- Obtain Krisp NC output by SDK trial, or by routing files through the Krisp desktop app's virtual mic, or through LiveKit Cloud's free Krisp NC filter (the de-denoiser project does this).
- Datasets: DNS Challenge 5 fullband blind set (real recordings), VoiceBank+DEMAND (48 kHz, with references), plus a self-recorded laptop-mic and headset set.
- Metrics: DNSMOS P.835 SIG/BAK/OVRL; PESQ-WB, ESTOI, SI-SDR where a reference exists; WER delta against the unprocessed input on a clean-speech set (gate: no WER regression greater than 0.2 points on clean speech); a first-word test (speech onset after 30 s of silence).
- Gates: OVRL and SIG not lower than Krisp NC with 95% confidence; BAK within 0.1.
- Without Krisp access, the honest fallback gate is "beat DeepFilterNet3 at lower latency and CPU", since DeepFilterNet3 is the common reference in every third-party table here.

Things to stop claiming: "beats Krisp" from a single LiveKit sample, from DNSMOS alone, or from any set the model was tuned on.

### 7.3 How hard is it for a 1 to 3M parameter causal model, one person, rented GPUs

Noise-only parity at 48 kHz (Track 2): realistic.

- Krisp NC's file is 5.8 MB, so it is plausibly a low-millions-parameter model itself (UNVERIFIED inference).
- Open models in this size class already match or beat Krisp's voice-isolation model on noise metrics in one competitor test, and RNNoise beats it on SIG.
- The main risk is evaluation access, not modelling.

Background-voice removal without enrollment (Track 1): possible in near-field, but the evidence is thin.

- For: Nearfield reports a tie with Krisp Default using about 0.63M parameters. VoxLock's training ran on one L4 GPU in about 9,200 steps, with most of the gain coming from simulated near-field calls. Krisp's own "do no harm" numbers are beatable, since every Krisp model regresses on phone calls.
- Against: Nearfield got there with 48 to 80 ms latency (3 to 5x Krisp), was tuned on the test set, and its previous version scored 7.8 to 8.4%. The 1.8M-parameter Hush, with 10,000 hours of data, only reaches 12.09%. alkove's small model barely removes a second voice without a voiceprint. The independently verified open results (VoxLock, Scicom C22) use 22 to 27M parameters and 33 to 110% of a core. The one small-model success is unreplicated.
- At 15 to 20 ms latency and under 3M parameters, I would expect roughly 6 to 8% WER on Krisp's set from a first serious attempt (between Krisp Lite at 6.21% and Hush), with a statistical tie as a stretch goal needing several training iterations. This is an estimate, not a measured fact.
- The near-field loudness cue does most of the work. The model will inherit Krisp's limitation: headsets and close mics only, and failures when a background talker starts first.

Doing both at 48 kHz in one model: the hard part.

- Every open voice-isolation model found is 16 kHz, and Nearfield moved from 48 kHz down to 16 kHz between v1 and v2.
- Krisp itself splits this across separate models and caps at 32 kHz.
- A practical design is a 16 kHz (or ERB-band) decision path with a fullband gain or deep-filter path on top, as DeepFilterNet does, and separate gates for the two tracks.

Compute budget pointers (from the projects above, not a quote):

- VoxLock: one L4, single run, fine-tune from a pretrained checkpoint.
- Hush: 100 epochs on 10,000+ hours of on-the-fly mixtures, batch size 16.
- Scicom reports run-to-run noise of 0.1 to 0.2 WER points, so every gate needs at least two seeds or a confidence interval.

## 8. Not found or not verified

- Krisp NC and BVC CPU% per platform (image-only tables).
- Per-file sizes of browser `.kef` models, and which BVC model ships in the browser.
- Whether the browser WASM uses SIMD.
- BVC algorithmic latency in the RTC docs (15 ms comes from the server blog's statement that it equals the on-device models).
- Krisp parameter counts and architecture.
- The claim that BVC became paid for SDK users on 2026-05-01.
- The Krisp evaluation harness repository URL and the Space's licence tag.
- Krisp numbers on DNS Challenge sets, PESQ or POLQA.
- Any neutral comparison of Krisp with Apple Voice Isolation, NVIDIA Broadcast, Dolby, Hance or DTLN.
- Current Daily, Agora and Discord integration details.
- The SLNG PDF tables and the Krisp BVC one-pager PDF (could not extract text).
