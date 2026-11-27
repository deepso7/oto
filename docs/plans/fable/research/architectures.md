Raw research notes, 2026-11-27; claims marked UNVERIFIED were not confirmed from a primary source

# Architectures for a tiny realtime 48 kHz denoise + dereverb + background-voice-cancellation model

Scope: literature and open implementations (2023 to Nov 2026) needed to judge the proposed design
(48 kHz, STFT 1024/512, 0.8-1.5M params, 50-100 MMAC/s, hybrid complex-low-band + ERB-high-band, GRU core,
FiLM-conditioned blind/enrolled BVC, tiny joint speaker encoder, GAN fine-tune, one overnight GPU run).

Conventions used below
- "Primary" = read from the paper (arXiv HTML), the repo README or the model card. Numbers from search-result
  summaries only are marked UNVERIFIED.
- Self-reported vendor/README numbers are marked "self-reported".
- MACs are per second of audio unless stated.
- Latency conventions differ between sources: some report window length (+ lookahead), some report
  window minus hop. For STFT 1024/hop 512 at 48 kHz that is 21.3 ms vs 10.7 ms. Compare with care.

Not researched / nothing found in this pass: DCCRN-light, TF-GridNet-lite causal variants, BSRNN-lite beyond the
BSRNN row in the FastEnhancer table, DNS low-complexity-track winners, URGENT 2024/2025 winners. URGENT systems are
large and non-causal, so they matter here only as possible teachers.

---

## A. Low-latency, low-complexity causal enhancement: state of the art

### A.1 Reference table (primary unless marked)

| Model | Rate | Params | MACs/s | Window / hop | Latency | CPU speed | Quality (as published) | Licence | Source |
|---|---|---|---|---|---|---|---|---|---|
| DeepFilterNet2 | 48k | 2.31M | 0.36G | 20 / 10 ms, 2-frame lookahead | 40 ms | RTF 0.04 (i5-8250U), 0.42 (RPi4) | VB+D PESQ 3.08; DNS4 blind DNSMOS OVL 3.88 | MIT or Apache-2.0, code and weights | https://ar5iv.labs.arxiv.org/html/2205.05474 , https://github.com/Rikorose/DeepFilterNet |
| DeepFilterNet3 | 48k | 2.13-2.14M (from other papers' tables) | 0.35G (DPDFNet table) | 20 / 10 ms, 2-frame lookahead | 40 ms | RTF 0.19 (i5-8250U, Rust realtime loop, tract) | VB+D PESQ 3.17, STOI 0.944 | same | https://ar5iv.labs.arxiv.org/html/2305.08227 |
| DPDFNet baseline / -2 / -4 / -8 | 16k | 2.31 / 2.49 / 2.84 / 3.54M | 0.36 / 1.35 / 2.36 / 4.37G | 20 / 10 ms, DF lookahead 2 | not stated (GigaEdge paper measured 40 ms output delay for the 48k model) | no CPU RTF published; DPDFNet-4 RTF 0.97 on Ceva NPN32 | low-SNR set PESQ 2.85 / 3.14 / 3.18 / 3.20 vs DFN3 2.76 | code Apache-2.0; weight licence not stated on repo; no training code | https://arxiv.org/html/2512.16420v3 , https://github.com/ceva-ip/DPDFNet |
| DPDFNet2 / 8 48 kHz HR | 48k | 2.58 / 3.63M | 2.42 / 7.17G | 20 / 10 ms | as above | sherpa-onnx: 17 parallel realtime streams on Core Ultra 5 125H (vs 151 for a FullSubNet-style 48k model) | EARS-WHAM v2: PESQ-WB 2.21, SI-SDR 14.69, DNSMOS OVRL 2.99 (DPDFNet2) | as above | https://github.com/ceva-ip/DPDFNet , https://arxiv.org/html/2611.13243 |
| FastEnhancer T / B / S / M / L | 16k | 22K / 91K / 194K / 492K / 1105K | 60M / 262M / 664M / 2.9G / 12G | n_fft 512; hop 256 / 256 / 256 / 160 / 100 | window 32 ms, no lookahead | RTF M5: 0.0058 / 0.011 / 0.019 / 0.039 / 0.105; Xeon 6248R: 0.012 / 0.022 / 0.034 / 0.101 / 0.313 | VB+D PESQ 2.99 / 3.13 / 3.19 / 3.24 / 3.26; SI-SDR 18.6-19.6 | MIT (repo) | https://github.com/aask1357/fastenhancer , https://arxiv.org/html/2509.21867v1 |
| FastEnhancer T / B / S / M / L | 48k | 28K / 101K / 207K / 512K / 1132K | 177M / 750M / 1.82G / 8.0G / 32.4G | n_fft 1024; hop 512 / 512 / 512 / 320 / 200 | 21.3 ms window, no lookahead | RTF M5 single thread: 0.012 / 0.026 / 0.042 / 0.135 / 0.424 | 48k Medium fp32 on VB+D: PESQ 3.06, SNR 19.4 dB (third-party measurement) | MIT repo; 48k weights released (ckpt + ONNX, all five sizes per release notes) | repo README Table 4; https://arxiv.org/html/2607.25350 |
| GTCRN | 16k | 24K (48K counting ERB module per repo, UNVERIFIED) | 40M (33M per repo, UNVERIFIED) | 32 / 16 ms | 32 ms | RTF 0.060 Xeon, 0.026 M5 | VB+D PESQ 2.87; DNS1 PESQ 2.26 | MIT | FastEnhancer README table |
| LiSenNet (streamable retrain) | 16k | 37K | 56M | 32 / 16 ms | 32 ms | RTF 0.034 Xeon | VB+D PESQ 2.98 | - | FastEnhancer README table |
| FSPEN (reimpl.) | 16k | 79K | 64M | - | - | RTF 0.046 Xeon | VB+D PESQ 3.00 | - | FastEnhancer README table |
| BSRNN small (reimpl.) | 16k | 334K | 245M | - | - | RTF 0.059 Xeon | VB+D PESQ 3.06 | - | FastEnhancer README table |
| ULCNet | 16k | 688K | 98M | 32 / 16 ms | not stated | RTF 0.02 (i5-8250U), 0.127 (Cortex-A53) | VB+D PESQ 2.87 (MS) / 2.54 (Freq); MUSHRA close to DFN2 | - | https://arxiv.org/html/2312.08132 |
| CoFi-Lite / Large | 16k | 83K / 221K | 12.9M / 32.9M | 32 / 16 ms sqrt-Hann | 32 ms, no lookahead | RTF 0.033 / 0.036 (i5-14600KF, excl. STFT) | DNS3 sim PESQ 2.16 / 2.30 vs GTCRN 2.07 | repo exists | https://arxiv.org/html/2607.10142 |
| HC-AINet | 16k | 0.27M | 0.66-1.08G | 32 / 12.5 ms | not measured; streaming unvalidated | none | VB+D PESQ 3.18-3.20 | - | https://arxiv.org/html/2610.30818 |
| LCT / LCT-GAN | 16k | 0.14M | 0.35G | 32 / 16 ms | 32 ms (48 ms with 1-frame lookahead) | - | VB+D PESQ 3.01 / 3.06 | - | https://arxiv.org/html/2505.21057 |
| TF-Refiner T / S / B (no-lookahead variants) | 16k+48k, one weight set | 0.09 / 0.19 / 0.45M | 48k: 1.27 / 2.03 / 3.85G | 40 / 20 ms sqrt-Hann | 40 ms (80 ms for default B) | not measured | 48k VB+D PESQ 3.10 / 3.36 / 3.36 | - | https://arxiv.org/html/2609.29463v1 |
| LaCo-SENet | 16k | 1.37M | not reported | 25 / 6.25 ms | 12.5-75 ms configurable | RTF 4.59 per-frame on Xeon Silver (not realtime at chunk 1); 0.30 at 64-frame chunks | VB+D PESQ 3.35 (12.5 ms) to 3.43 (75 ms) | repo + weights | https://arxiv.org/html/2606.19688 |
| RT-SEMamba 1 / 8 layer | 16k | 1.05 / 2.74M | 20.6G / 47.4G | 25 / 6.25 ms | 25 ms | RTF 0.11 / 0.29 on an RTX 5090 GPU | VB+D PESQ 3.18 (distilled) / 3.32 | - | https://arxiv.org/html/2608.12099 |
| aTENNuate | 16k (raw waveform) | 0.84M | 0.33G | - | 46.5 ms | - | VB+D PESQ 3.27 | - | UNVERIFIED (search summary + LaCo table); https://arxiv.org/abs/2409.03377 |
| RNNoise 0.2 | 48k | not checked | 107M sparse (284M dense equivalent) | 20 / 10 ms | 20 ms output delay | 1089 parallel realtime streams on Core Ultra 5 125H | EARS-WHAM: PESQ-WB 1.53, SI-SDR 9.21, NISQA MOS 2.17 (same as noisy) | BSD (UNVERIFIED) | https://arxiv.org/html/2611.13243 |
| "One model, many latencies" (causal RE-USE) | 8-48k SFI | 2.0-3.7M | 19-32G | 40 / 20 ms | 40-80 ms | GPU only | URGENT25 non-blind DNSMOS 3.05-3.15 | NVIDIA weights on HF | https://arxiv.org/html/2606.25621v2 |
| HALO | 16k | plug-in for GTCRN-class models | halves backbone frame rate | - | unchanged | - | +0.1 PESQ, +0.5 dB SI-SNR, +0.04 OVRL over GTCRN | - | UNVERIFIED (abstract via search); https://arxiv.org/abs/2606.12328 |

### A.2 Which families win in the 0.5-3M param / under 5% of a laptop core regime

1. Deployed-speed-optimised STFT encoder-decoders with a GRU time core (FastEnhancer, DeepFilterNet family, ULCNet,
   GTCRN/CoFi-Lite, CRUSE). The recurring finding of 2025-2026 is that MACs and parameter counts do not predict speed:
   - FastEnhancer paper: GTCRN has 40M MACs but RTF 0.060 on a Xeon; FastEnhancer-B has 262M MACs and RTF 0.022.
     Named costs that MACs miss: time-axis conv kernels > 1 (state buffering; kernel 3 raised RTF 0.022 -> 0.028 with
     no quality gain), LayerNorm (cannot be folded like BatchNorm), time-axis attention (key/value caches),
     grouped/shuffled/sub-band structures. https://arxiv.org/html/2509.21867v1
   - "Beyond Model Size" (GN, LiSenNet on STM32N6): same conclusion on NPUs; GRU along time is fine and cheap in state,
     GRU along frequency is the problem; BatchNorm folding and bounded activations help int8.
     https://arxiv.org/html/2609.29866
   - GigaEdge SDK paper: "arithmetic alone cannot describe deployment efficiency"; a FullSubNet-style 48k model served
     8.7x the streams of DPDFNet2 48k. https://arxiv.org/html/2611.13243
2. Dual-path (frequency RNN/attention + time GRU) inside a compressed frequency grid gives the best quality per
   parameter but costs MACs: DPDFNet (+1-4G), Nearfield v2 (628K params but ~0.2 s CPU per audio second), TF-Refiner
   (1.3-3.9G at 48k), FastEnhancer M/L (8-32G at 48k).
3. Mamba/state-space and conformer-style causal models reach the highest PESQ (3.3+) but are not CPU-realtime as
   published (RT-SEMamba RTF is on an RTX 5090; LaCo-SENet needs 7-12-frame chunks on a Xeon).

Laptop budget, concretely: on one Apple M5 thread at 48 kHz, FastEnhancer-B (101K params, 750 MMAC/s) runs at 2.6% and
FastEnhancer-S (207K params, 1.82 GMAC/s) at 4.2% of realtime in fp32 ONNX Runtime. An int8 C/Rust runtime ran
FastEnhancer-M (512K params, 8 GMAC/s, hop 320) at RTF 0.069 on an M2 core, 3.3x faster than fp32 ORT
(https://arxiv.org/html/2607.25350). DeepFilterNet2 (360 MMAC/s) is RTF 0.04 on a 2017 i5-8250U. So "under 5% of a
laptop core" corresponds to roughly 0.5-2 GMAC/s for a speed-friendly architecture, not 50-100 MMAC/s. Web/WASM costs
1.8-3.3x native in the GigaEdge measurements.

Nothing published at 48 kHz sits at 50-100 MMAC/s except RNNoise 0.2 (107 MMAC/s sparse), which is far weaker than
the DFN class (table above). The closest neural points: FastEnhancer-T 48k at 177 MMAC/s with 28K params; ULCNet at
98 MMAC/s but 16 kHz and denoise-only.

### A.3 Frame and hop sizes in use, and what shorter hops cost

- 48 kHz: DeepFilterNet/DPDFNet/RNNoise use 20 ms / 10 ms (960/480). FastEnhancer 48k T/B/S use n_fft 1024 / hop 512
  (identical to the proposed design); M uses hop 320, L hop 200. GigaEdge uses 32/16 ms (1536/768). TF-Refiner 40/20
  ms. Krisp BVC runs at 32 kHz internally (docs). No source directly compares 1024/512 against 960/480 at 48 kHz.
- 16 kHz tiny models almost all use 32 ms / 16 ms.
- Cost of a shorter hop, controlled study (Wu and Braun, Microsoft, ICASSP 2025, CRUSE 0.625M params,
  https://arxiv.org/html/2409.10358v1): compute scales inversely with hop (230 MMAC/s at 20 ms latency, 461M at 10 ms,
  921M at 5 ms). With model size fixed, DNSMOS OVR is flat from 20 ms to 10 ms (2.79 both) and drops at 5 ms (2.74).
  With compute held constant by shrinking the model, quality falls much faster (OVR 2.76 at 10 ms with 0.55M params,
  2.68 at 5 ms with 0.16M params).
- FastEnhancer itself buys quality with smaller hops only in the big sizes (M: hop 160, L: hop 100 at 16k), at 4-18x
  the MACs of S.

### A.4 Lookahead

- LaCo-SENet (same 1.37M weights budget, VB+D): 12.5 ms -> 25 ms latency +0.01 PESQ; -> 50 ms +0.05; -> 75 ms +0.08;
  -> 200 ms +0.12. https://arxiv.org/html/2606.19688
- Causal RE-USE: one 20 ms lookahead frame beat adding four more Mamba layers on every metric; the authors advise
  compute-limited users to add lookahead rather than depth. https://arxiv.org/html/2606.25621v2
- DeepFilterNet keeps a 2-frame (20 ms) lookahead; Hush runs the DFN3 architecture with zero lookahead at 16 kHz
  (20 ms latency, self-reported, no metrics published).

---

## B. The hybrid frontend and the STFT choice

Supported by several independent lines:
- DeepFilterNet 2/3: 32 ERB gains over the full band plus order-5 complex deep filtering on the lowest 96 bins
  (4.8-5 kHz), DF output replaces the ERB-gained signal in the low band. This is the template for the proposal.
- DPDFNet branch ablation: adding the dual-path block to the complex (low-band) branch alone gives PESQ 3.12 vs 3.04
  for the ERB branch alone and 3.14 for both, i.e. most of the gain comes from the low-band complex branch.
- TF-Refiner: deep encoder only below a configurable cutoff (default 5 kHz); the observed high band enters a shallow
  decoder. Raising the cutoff to 16 kHz doubled cost with no PESQ gain at 48 kHz; lowering 5 -> 3 kHz saved 16% at
  48 kHz for "a small PESQ decrease". Cost grows 2.0x from 16 to 48 kHz while bins triple.
- CoFi-Lite: coarse ERB-compressed full-band path plus a fine low-band path (complex-derived input) with cross-path
  fusion. Coarse only PESQ 1.97, fine only 1.53, both without fusion 2.02-2.06, with fusion 2.16. Cutoff sweep at
  16 kHz: 17/33/65/97 bins -> 2.08/2.11/2.16/2.15 (65 bins = ~2 kHz).
- HC-AINet band study: oracle 32-band ERB gains reach PESQ 3.58 (above Bark/Mel/rectangular); learned 32-band gains
  reach 2.87-2.94 vs 3.10 full-band, so band gains alone cap quality and need a fine-resolution path.

Caveats for the proposal as written
- The low-band path in DFN is deep filtering (order 5, with lookahead 2), not a complex mask plus filter. A 3-tap
  causal filter is in line with the TF-Refiner no-lookahead variants (3 same-frame taps), but there is no direct
  evidence on "complex mask + 3-tap DF" versus DFN's order 5.
- 1024/512 at 48 kHz gives 46.9 Hz bins; 4.5 kHz = 96 bins. That matches DFN's 96 low bins numerically, at 46.9 Hz
  instead of 50 Hz spacing. FastEnhancer's released 48k T/B/S models use exactly 1024/512, so the STFT choice has a
  shipped precedent with Rust and C runtimes.
- Zero lookahead at 21 ms: reasonable for denoise (small measured loss). For background-voice removal the open
  systems that work best use more: Nearfield v2.3 added 16 ms (fast) or 48 ms (best) lookahead over a 32 ms window;
  VoxLock is 24 ms; the "Voice on the Line" models are 20 ms but 10-555M params.

Production systems
- Krisp BVC (docs, primary): no enrollment; runs natively at 32 kHz; designed for headsets and earbuds, best with
  wired USB boom mics, "acceptable performance" on good built-in mics such as MacBooks; maintains a device
  allow/block list; warns that "the user's voice may be suppressed" on devices that capture narrowband; CPU 5-13%
  by platform. NC models: about 25 ms latency with 10 ms frames at 16 kHz; near-field guidance of mouth under 50 cm.
  Architecture and model size are not published. https://developers-docs.krisp.ai/docs/krisp-rtc-bvc-nc
- GigaEdge Enhance: modified FullSubNet, 32/16 ms at 48 kHz; its discriminative 48k model costs 4% of a core at
  16 ms chunks (Core Ultra 5 125H); the GAN-trained generative add-on recovers speech-quality scores at a 63%
  capacity cost.

---

## C. Dereverberation jointly with denoising in tiny causal models

What is done
- DeepFilterNet2: reverb treated as a distortion on the speech; target keeps "a smaller amount of reverberation by
  decaying the room transfer function" (partial dereverb). Later work describes the original DFN3 target as (5, 500):
  5 ms offset, decay to T60 500 ms.
- DPDFNet: reverberant inputs with early-reflection targets; no dereverb-specific evaluation published.
- CoFi-Lite / GTCRN recipe: target keeps the first 100 ms of the RIR.
- URGENT challenges: 50 ms early-reflection reference.

Evidence on target choice
- "How much to dereverberate?" (DFN3 and HS-TasNet-small, DNSMOS only, https://arxiv.org/html/2505.01338):
  - Close mic (0.1-0.5 m): the direct-sound target is as good as anything; a hard 30 ms window was significantly
    worse (BAK).
  - Distant mic (up to 2 m / 10 m): keeping early energy or a decay helps: OVRL 2.56 (direct only) -> 2.78 (50 ms)
    -> 2.86 (decay to 300 ms). Mic distance, not room size, decides the best target.
  - Pitfalls: hard truncation of the RIR is unnatural; too much retained reverb hurts BAK because late reverb is
    noise-like; naive T60/room sampling gives unrealistic RIRs (volume-aware T60 sampling raised PESQ 2.09 -> 2.17);
    intrusive metrics are not comparable across models trained with different targets.
- "Rethinking training targets" (NVIDIA, large non-causal USEMamba, https://arxiv.org/html/2603.02641): time-shifted
  anechoic target beat the 50 ms early-reflection target (listening MOS 3.68 vs 3.00); the unshifted anechoic target
  was worst because of input/target misalignment. Models trained on early-reflected targets stay "noticeably
  reverberant". Not shown for tiny causal models.
- DFN-based dereverb study (https://pmc.ncbi.nlm.nih.gov/articles/PMC11820353/, UNVERIFIED, search summary): 50 ms
  split; complex ratio masks struggle with spectral notches from early reflections (observed bin near zero, target
  not), which is an argument for multi-frame filtering in the low band; a second decoder was judged not worth it.
- Li et al. 2021 (UNVERIFIED, search summary): over-aggressive dereverb by a capacity-limited network causes
  discontinuous speech.

Interaction with BVC: single-mic proximity selection relies on the level and reverberation of the interferer. A
separate dereverb/denoise stage in front erases those cues (FVAD paper, D below), so doing dereverb and BVC in one
network, as proposed, is the right structure.

---

## D. Background voice cancellation without enrollment

### D.1 Research literature
- ICASSP 2023 DNS Challenge (DNS-5, https://ar5iv.labs.arxiv.org/html/2303.11510): joint denoise + dereverb +
  interfering-talker suppression at 48 kHz, headset and speakerphone tracks. The headset track exists because
  headset "DRR, SNR" cues "make it possible to suppress neighboring talkers even without enrollment data". Still,
  "almost all models in both tracks were personalized"; SIG degraded for most models, partly because personalized
  models "end up suppressing the primary talker". Winner TEA-PSE 3.0: 22.24M params, 19.66G MACs, RTF 0.46
  (https://arxiv.org/html/2303.07704); its subjective SIG (3.52) and word accuracy (0.761) were below the noisy
  input (3.76, 0.843). A later paper states the DNS 2023 latency cap as 20 ms (via arXiv 2610.35272).
- Distance-based sound separation (Patterson et al., Google, Interspeech 2022): near/far split from one mic with a
  fixed threshold, trained on synthetic reverberant mixtures; SI-SNR improvement 4.4 dB (near) and 6.8 dB (far) with
  one near and four far talkers; one real demo at 0.5 m vs 2 m. Non-causal research model (UNVERIFIED details).
  https://google-research.github.io/sound-separation/papers/distance-based-separation-interspeech2022/
- Distance-based single-channel TSE (https://arxiv.org/html/2412.20144): 1.25M-param BLSTM (non-causal) model,
  query-by-distance. SDRi 8.8 dB in one fixed room, 3.4 dB with random rooms and mics, 3.7 dB on real RIRs;
  distance MAE 0.26 m -> 1.09 m. The cue is DRR: direct and early energy fall with distance, late reverb does not.
  SI-SDR was rejected as a loss because scale matters for target-absent output.
- Foreground VAD (https://arxiv.org/html/2609.19856): a 0.6M-param streaming model learns "foreground" purely from
  the data recipe: interferers at 0-15 dB below the target, convolved with RIRs and low-passed at 1-4 kHz, covering
  10-50% of the utterance. Far-field rendering of interferers was needed for transfer to real far-field data.
  Failure: when the target is not dominant the cue inverts; before a foreground is established it behaves as a
  plain VAD.

Single-mic cues, summarised: level difference, direct-to-reverberant ratio / early-to-late energy, high-frequency
roll-off of distant or loudspeaker sources, temporal continuity of one voice. A 48 kHz model keeps the
high-frequency cue that 16 kHz models lose, but no source measures how much that is worth.

### D.2 Open implementations (all 16 kHz unless noted; numbers self-reported unless noted)

| System | Size | Mechanism | Latency / CPU | Evidence | Licence | Source |
|---|---|---|---|---|---|---|
| Hush | 1.8M params, 8 MB; DeepFilterNet3 architecture | no enrollment; primary = dominant talker by training recipe (interferers in 60% of samples at 12-24 dB SIR); training-only auxiliary head predicting the background-talker ERB mask (L1, weight 0.1) | 20 ms, zero lookahead; under 1 ms per 10 ms frame | no metrics in repo. Third-party: Krisp benchmark WER 12.09% vs 13.30% DFN3 vs 5.30% Krisp; leak 19.6 words/min vs 22.8 DFN3 vs 1.5 Krisp; on AMI headset no better than unprocessed (VoxLock paper) | Apache-2.0, training code + weights; 100 epochs, batch 16, "10,000+ hours" | https://github.com/pulp-vision/Hush |
| Nearfield v1 | DeepFilterNet3-based, 48 kHz, 11 MB | 32 band gains + frame gate presets | 30 ms; 0.5-2 ms per 10 ms frame | against an equal-level talker lost 31-47% of caller words (unfiltered 41%); babble worse than unfiltered | Apache-2.0 | https://github.com/Ground-State-Foundry/nearfield/releases/tag/v1.1.0 |
| Nearfield v2.3 | 628K params, 2.5 MB, from scratch; 64 bands, 4 dual-path blocks (freq bidirectional, time causal), mask + phase | "the caller is the nearest voice heard so far"; earlier versions had an implicit distance cutoff that muted speakerphone/car callers | 32 ms window, 8 ms hop; 48 ms (fast) or 80 ms (best); ~0.2 s CPU per audio second on a Xeon 8488C | Krisp benchmark WER 4.99-5.01%, leak 0.36-0.70/min (but tuned against that set); equal-level talker: 17-20% of caller words lost; babble 35% lost; a voice before the caller passes through | Apache-2.0 weights; training code not released | https://github.com/Ground-State-Foundry/nearfield |
| alkove Focus 1 (preview: ~0.8M params, 0.93 MB int8) | 2.3 MB; fine-tuned from GTCRN (MIT) | zeros voiceprint = "keeps the voice closest to the microphone"; optional 192-d voiceprint from a ~1 MB companion model; optional 3 s sample via key/value attention | 16 ms hop, 32 ms total; 57 ms CPU per audio second on M1 Max | With no voiceprint on synthetic equal-ish mixtures: other voice only -1.4 dB (basically no suppression); with voiceprint -33.8 dB, SI-SDR 10.7 dB. On AMI headset both modes cut leaked words from 17.0 to ~2.3/min. Hurts clean speech: VB+D STOI 0.892 vs 0.921 noisy, WER 5.6% vs 2.7% | Apache-2.0 code and weights | https://huggingface.co/alkove-ai/alkove-focus-1 |
| VoxLock | 27M params, causal BSRNN + fine-tuned ECAPA-TDNN frame features | self-enrollment: first 3 s of speech in the stream (assumes loudest, earliest talker is the user) | 24 ms; 61% of a Xeon thread at 80 ms chunks, not realtime at 20 ms chunks | Krisp benchmark WER 4.95%, leak 0.11/min (Krisp Default 5.30%, 1.52). Found and fixed: a "keep whichever talker is closer" shortcut; 20 dB fade of a continuous talker within 40 s when trained on 6 s windows | open (base Apache-2.0) | https://arxiv.org/html/2610.40727 |
| "Voice on the Line" students | 9.8M / 22.1M (teacher 555M with WavLM-Large) | null speaker vector, then self-enrollment on the loudest 3 s; FiLM in every layer | 20 ms; 8.5-17% of a core | mean WER over nine sets 15.5% vs 18.0% VoxLock; "nearness alone is a weak cue" | open | https://arxiv.org/html/2610.35272 |

### D.3 Is single-channel proximity BVC at 48 kHz with ~1M params credible?

- Credible for the headset / close-mic case with a clearly quieter background talker. Hush (1.8M), Nearfield (0.63M)
  and alkove Focus (~0.8M) all demonstrate real suppression of quieter background speech at 16 kHz, and the DNS-5
  organisers and Krisp both state the headset case works without enrollment.
- Not demonstrated by anyone: 48 kHz, ~1M params, 50-100 MMAC/s. Nearfield dropped from 48 kHz (v1, DFN3 band gains)
  to 16 kHz when it moved to a real separator. Krisp BVC is 32 kHz at 5-13% CPU. The small models that work spend
  far more compute per parameter (Nearfield: ~0.2x realtime on a Xeon thread).
- Failure modes reported consistently:
  - Lone background talker (TV, colleague) before the user speaks: Nearfield v2.3 passes it through until the
    caller's first word; FVAD falls back to plain VAD; stream-learned voiceprints can lock onto the TV (alkove,
    VoxLock live mode, Voice on the Line).
  - Equal-level talkers: blind models have no basis to choose. alkove blind mode -1.4 dB; Nearfield loses 17-20% of
    caller words; Hush documents that loud competing speakers are not suppressed.
  - User far from the mic (laptop at arm's length, speakerphone, car): treated as background. Nearfield <= v2.2
    muted such callers for the rest of the call; alkove falls back to raw mic; Krisp steers users to headsets.
    The 50 cm laptop case therefore sits at the edge of what vendors support.
  - Quiet callers: below -45 dBFS every model in the Voice on the Line study removed most of the caller.
  - Lone-caller damage: isolation models cost WER when there is nothing to remove (VoxLock phone calls 3.82% ->
    5.36%; alkove VB+D 2.7% -> 5.6%).
  - Long-session drift: models trained on 3-6 s clips fade or mute continuous talkers; VoxLock, Nearfield (60 s
    calls with 40 s pauses) and DPDFNet (30-40 s fine-tune segments) all had to add long-sequence training.
  - Evaluation trap: short test windows hide both the forgetting and the lone-caller failures (median
    background-only stretch 16.9 s in the Krisp set vs under 4 s in short windows).
- Benchmark to use: Krisp Voice Isolation benchmark (265 real recordings; CC BY-NC 4.0, evaluation only), AMI
  headset windows, WER + leaked words per minute, plus long windows.

---

## E. Enrolled target-speaker extraction in small causal models

### E.1 What published systems use

| System | Enhancer size | Speaker encoder | Embedding | Conditioning | Source |
|---|---|---|---|---|---|
| VoiceFilter-Lite (2020) | 2.2 MB quantised, LSTM | pretrained d-vector | UNVERIFIED dim | concatenated to features per frame | https://arxiv.org/abs/2009.04323 (UNVERIFIED, search summary) |
| pDCCRN | 3.96-4.5M, RTF 0.195 | pretrained d-vector | - | concatenation | https://arxiv.org/html/2204.00771 |
| E3Net | 6.61M (student 4.5M), 20/10 ms, RTF 0.065 | pretrained (model not named in paper) | 256 | concat to encoder features + linear projection, once | same |
| Personalized PercepNet / Unified PercepNet | 26.5M, 10 ms frames, 30 ms lookahead, 17.2% of a mobile x86 core | ECAPA-TDNN (1024 ch), frozen, EER 0.39% | 192 | concat [embedding, mode flag] at GRU input | https://arxiv.org/html/2302.11768v1 |
| TEA-PSE 3.0 | 22.24M | pretrained ECAPA-TDNN + a jointly trained BLSTM "local-global" enrollment encoder | - | multiplicative fusion + concat | https://arxiv.org/html/2303.07704 |
| pDeepFilterNet2 | 2.31M, 0.33G MACs, RTF 0.025 (i7-10870H), 48 kHz | ECAPA-TDNN, frozen, EER 0.96% | 192 | plain concat where ERB and DF encoder branches join | https://arxiv.org/html/2404.08022 |
| pDFN2 + TinyECAPA | 2.38M, 0.374G | enrollment still by full ECAPA; a 150K-param distilled encoder (EER 4.26%) runs on the live mixture as a similarity cue | 192 | concat | https://arxiv.org/html/2601.16235 |
| alkove Focus | ~0.8M-class, 16 kHz | separate ~1 MB "Timbre" model, >= 1.5 s speech | 192 | not documented | model card |
| VoxLock | 27M | ECAPA-TDNN, fine-tuned jointly, frame-level features + cross attention | - | attention | arXiv 2610.40727 |
| Voice on the Line | 10M-555M | internal, self-enrolled | - | FiLM in every layer | arXiv 2610.35272 |
| DeepVQE-based (Parnamaa and Saabas, Interspeech 2024) | not on abstract page | none: the PSE model's own internal representation is the embedding | - | - | https://www.isca-archive.org/interspeech_2024/parnamaa24_interspeech.html |
| SEF-PNet / USEF-PNet | - | none: enrollment spectrogram interacts with the mixture | - | interactive adaptation module | https://arxiv.org/abs/2501.11274 , https://arxiv.org/abs/2505.12288 (UNVERIFIED, search summaries) |

Performance reference for the tiny end (DNS-5 blind set, PDNSMOS SIG/BAK/OVRL, headset track; from arXiv 2601.16235):
noisy 4.15/2.37/2.71; E3Net 3.82/3.47/3.10; pDCCRN 3.76/3.29/2.99; pDeepFilterNet2 3.77/3.47/3.08;
pDFN2 + TinyECAPA 3.81/3.49/3.13; TEA-PSE 3.0 4.11/4.05/3.65. Every small model lowers SIG well below the noisy
input; the 22M model is the only one that does not.

### E.2 Is a tiny jointly trained 64-d encoder from 5 s credible?

- For: the Interspeech 2024 DeepVQE paper reports that an internally learned embedding is "equally well or better"
  than a pretrained speaker model and beat the DNS-5 winner by 0.15 MOS; speaker-encoder-free systems (SEF-PNet)
  exist; TEA-PSE 3.0 gained from a jointly trained enrollment encoder on top of ECAPA; alkove gets -33 dB with a
  ~1 MB voiceprint model and 1.5 s of speech.
- Against: none of the positive results is at sub-1M total size with a 64-d vector trained from scratch in a short
  run. The systems that publish tiny results all use a frozen pretrained 192-d ECAPA for enrollment. The one tiny
  encoder with published numbers (TinyECAPA, 150K params) was distilled from ECAPA, reached only 4.26% EER, and was
  used as an auxiliary cue, not as the enrollment encoder. alkove does not disclose how Timbre was trained.
- Practical point: the enrollment embedding is computed once, offline. A large encoder adds no realtime cost, only
  download size. The realtime budget argument for a tiny encoder applies only if the embedding is refreshed from the
  live stream.
- Verdict: plausible but the least-evidenced component. Lower-risk options: (a) distil a small encoder from a
  pretrained one, then fine-tune jointly; (b) use the big pretrained encoder offline for enrollment only.

### E.3 Licences of pretrained speaker encoders
- SpeechBrain spkrec-ecapa-voxceleb: Apache-2.0 on the model card (UNVERIFIED, search summary).
- WeSpeaker: toolkit Apache-2.0; VoxCeleb-trained weights stated as CC-BY-4.0 in project docs and on the pyannote
  wrapper card, Apache-2.0 on another card (UNVERIFIED, conflicting secondary sources; docs page returned 503).
- NVIDIA TitaNet-Large: CC-BY-4.0, ~23M params, EER 0.66% (primary:
  https://huggingface.co/nvidia/speakerverification_en_titanet_large ).
- 3D-Speaker CAM++: repo Apache-2.0 (UNVERIFIED for individual weights).
- Underlying data: the Oxford VoxCeleb pages license the metadata under CC BY-SA 4.0 and distribute YouTube URLs,
  not audio; whether that constrains trained weights is unsettled, and VoxBlink2's authors chose CC BY-NC-SA for
  their models for this reason (UNVERIFIED, search summary). Treat VoxCeleb-trained encoders as a provenance risk
  if clean licensing matters.

### E.4 One checkpoint for blind and enrolled modes
- Unified PercepNet (Amazon, ICASSP 2023): one network, per-frame flag appended to the embedding; flag 0 with a
  zero embedding = non-personalized. Matches the specialised models (personalized OVRL 2.877 vs 2.880; subjective MOS
  3.11 vs 3.10 and 3.40 vs 3.43). Caveat: 26.5M parameters; its non-personalized mode keeps all talkers, i.e. it is
  denoise, not blind BVC.
- USEF-PNet: mixed batches of personalized samples and plain-enhancement samples with a zero dummy enrollment
  (UNVERIFIED, abstract summary).
- alkove Focus: zeros voiceprint selects the closest-voice behaviour in the same weights, at ~0.8M-class size. Its
  blind mode is weak on equal-level mixtures (-1.4 dB).
- Nothing found that combines three behaviours (denoise-only, blind near-field isolate, enrolled isolate) in one
  sub-2M checkpoint with published numbers. The mechanism is supported; the capacity is not.

### E.5 Target-speaker over-suppression
- Metric: TSOS (Eskimez et al., ICASSP 2022), reported as suppressed target seconds per half hour.
- Mitigations with evidence:
  - Asymmetric / over-suppression loss (VoiceFilter-Lite asymmetric L2; pDFN2 lambda_OS = 5e2; DPDFNet
    over-attenuation loss raised PESQ 2.85 -> 2.92 and SIG 2.94 -> 3.19 on its low-SNR set).
  - VoiceFilter-Lite adaptive suppression strength from an auxiliary overlap predictor (UNVERIFIED).
  - VAD-weighted loss (UPN): weighting speech-active frames 0.9 vs 0.5 raised personalized SIG 3.285 -> 3.346.
  - Enrollment-embedding augmentation with noise and reverb (UPN): SIG +3.2%, OVRL +2.9%.
  - Knowledge distillation (E3Net paper): pDCCRN student TSOS 51.9 (supervised) -> 4.8 (KD) -> 0.6 (multi-task +
    KD on unlabeled data); E3Net student 3.57 -> 0.50.
  - Live similarity cue from a tiny encoder (pDFN2 + TinyECAPA), +0.05 OVRL.
  - Presence gate that outputs digital silence, and falling back to raw mic when the user looks far (alkove,
    Voice on the Line).

---

## F. Losses and GANs

What the best small models use
- DeepFilterNet2/3, pDFN2, DPDFNet, Hush: compressed (c = 0.3) spectral loss plus multi-resolution compressed
  magnitude + complex STFT loss at windows {5, 10, 20, 40} ms (lambda 1e3 / 5e2); DPDFNet and pDFN2 add an
  over-attenuation term; Hush adds a local-SNR loss and the auxiliary separation loss. No GAN.
- FastEnhancer (and TF-Refiner reuses it): 0.3 magnitude + 0.2 complex + 0.3 consistency (all on power-compressed
  spectra) + 0.2 waveform L1 + 0.001 differentiable PESQ. PESQ loss was dropped for the DNS and 48 kHz models.
  AdamP, lr 2e-3, batch 64, 2 s segments, cosine schedule. No GAN.
- GTCRN / CoFi-Lite: hybrid compressed spectral loss (details not re-read).
- MP-SENet lineage (LaCo-SENet, HC-AINet, CMGAN): magnitude + phase + complex + consistency + a small
  MetricGAN-style PESQ discriminator term (weight 0.05).
- VoxLock: negative SI-SDR only.
- TEA-PSE: SI-SNR + magnitude + asymmetric + multi-resolution STFT.

Is a GAN fine-tune worth it for a tiny mask-based model?
- LCT-GAN (0.14M params, causal magnitude mask): HiFi-GAN multi-period + multi-scale waveform discriminators gave
  +0.05 PESQ on VB+D and +0.06 DNSMOS SIG / +0.06 OVL on DNS3 real recordings; STOI and SI-SDR unchanged; no
  inference cost. The discriminators needed extremely small learning rates (1e-7 and 1e-9) to stay stable.
- MetricGAN-style discriminators mostly move PESQ (MP-SENet ablations, UNVERIFIED search summary).
- Large gains from adversarial training are reported for big or generative models (NVIDIA: listening MOS 3.68 ->
  3.97 with a GAN generator stage, non-causal 30-layer model; GigaEdge hybrid).
- Verdict: a modest, optional polish step (about +0.05 on PESQ/SIG scales) with real instability risk. It does not
  address the hard parts (speaker selection, over-suppression). Do it last or skip it.

Notes on the proposed loss list
- SI-SDR is scale-invariant and undefined for silent targets; target-absent frames (lone background talker) are
  central to BVC. The distance-TSE paper replaced SI-SDR for this reason. Use a scale-aware term (thresholded SDR,
  waveform L1, compressed spectral) for those frames.
- An over-attenuation / asymmetric term and long-segment fine-tuning have direct evidence and are missing from the
  proposal.
- The per-frame target-speech probability (BCE) has precedent as an auxiliary or gating head (PercepNet VAD score,
  VoiceFilter-Lite strength predictor, presence gates in Voice on the Line, DFN3 local-SNR gating).

---

## G. Training cost

| Model | Reported training | Source |
|---|---|---|
| DeepFilterNet2 | 100 epochs, AdamW, batch 8 -> 96; >500 h clean speech, ~150 h noise | arXiv 2205.05474 |
| DeepFilterNet3 retrains | "Training all the six DeepFilterNet3 models took 30 days on two A100s", 117-120 epochs each on 1315 h speech + 177 h noise at 48 kHz, i.e. about 5 days on 2 A100s per model (~240 A100-hours) | https://arxiv.org/html/2507.09750 |
| DeepFilterNet3, small dataset | 120 epochs in 15 h 8 min, batch 32 (GPU and data size unknown) | UNVERIFIED (search summary) |
| DPDFNet | 1.6M iterations, batch 32, 3 s chunks ("42,667 hours of speech" seen) on 760 h DNS4 + 4000 h MLS; then 5000 iterations on 30-40 s segments; GPU/time not stated | arXiv 2512.16420 |
| FastEnhancer | 16k VB+D: T/B/S/M on one RTX 3090, L on two; DNS and 48k: T/B/S 1 GPU, M 2, L 4; batch 64; step counts not published; 48k data ~1285 h clean + ~75 h noise | repo releases and README |
| GTCRN | 40,000 pairs of 8 s per epoch drawn from 2000 h; epochs/GPU not found | UNVERIFIED |
| CoFi-Lite | 250K iterations, batch 8, 200 h | arXiv 2607.10142 |
| LaCo-SENet | 400K steps, batch 8, VB+D only | arXiv 2606.19688 |
| TF-Refiner | 400K steps, batch 2 | arXiv 2609.29463 |
| Hush | 100 epochs (patience 25), batch 16, "10,000+ hours" mixed audio; hardware not stated | repo |
| pDeepFilterNet2 | 950 h generated, batch 8 -> 128, early stopping patience 15 epochs | arXiv 2404.08022 |
| Unified PercepNet | batch 256 of 20 s inputs on 8 V100s | arXiv 2302.11768 |
| E3Net | 2000 h simulated; 209K updates supervised, then 200-500K for KD | arXiv 2204.00771 |
| VoxLock | 9,200 fine-tuning steps on one L4, starting from a pretrained 27M TSE model, in four hand-designed stages | arXiv 2610.40727 |

Reading
- A from-scratch 48 kHz DFN-class model on DNS-scale data is on the order of 100-250 A100-hours per run in the one
  documented case; several others are 250K-1.6M iterations.
- Overnight on one A100/H100 is realistic for: a denoise-only baseline of a FastEnhancer-S-class model (they train
  on a single GPU), a fine-tune of an existing checkpoint (VoxLock, alkove from GTCRN, Hush/Nearfield v1 from the
  DFN3 design), or a smoke-test of the full recipe.
- It is not supported for a converged joint denoise + dereverb + blind BVC + enrolled BVC model from scratch. The
  projects that got BVC working needed multiple recipe iterations (VoxLock four stages; Nearfield v1 -> v2.3 with
  changed architecture, 60 s training calls, device-processing augmentation; Voice on the Line long-call and
  lone-caller fixes). Long-sequence training (30-60 s) multiplies cost per step.
- Realistic planning figure (my estimate, not from a source): 3-7 single-GPU days per full run, and 5-10 runs to
  reach a shippable recipe, plus CPU-bound 48 kHz on-the-fly mixing (RIR convolution, device simulation) that often
  limits throughput before the GPU does.

---

## H. Knowledge distillation into tiny streaming students

- Two-step KD (Bose, ICASSP 2024, causal CRUSE, teacher 1.9M): gains shrink with student size: +1.10 dB SDR / +0.11
  PESQ at 30K params, +0.43 dB at 62K, +0.36 dB at 240K, +0.03 dB at 350K; larger gains at low SNR. Plain
  output-matching KD gave essentially nothing for tiny causal students. https://arxiv.org/html/2309.08144
- RT-SEMamba: 8-layer causal teacher -> 1-layer student (1.05M): PESQ 3.06 -> 3.18 (teacher 3.32); 2-layer student
  +0.03. Student initialised from teacher weights; output + feature KD with ramp-up.
- E3Net / pDCCRN (personalized): KD and KD on unlabeled real data mainly fix over-suppression (TSOS 51.9 -> 0.6 for
  the 1.5M pDCCRN student) and WER (TS1 44.4 -> 37.2), with small DNSMOS gains.
- Voice on the Line: 10M and 22M CPU students distilled from a 555M causal WavLM-based teacher land within ~0.7 WER
  points of it (15.5 vs 14.8 mean over nine sets); the authors conclude distillation matters more than student
  size.
- Distil-DCCRN, LaMi-KD, FT-JNF KD comparison: positive but for larger students or other tasks (UNVERIFIED, search
  summaries).
- Reading: for plain denoising at 0.3M+ params, KD gains are small. For the selection problem (which talker to
  keep, not over-suppressing the target) KD from a strong separator/TSE teacher is one of the few interventions with
  large measured effect, and it also lets real unlabeled recordings be used via pseudo-labels.

---

## Critical assessment of the proposed design

### Well supported
- Hybrid frontend: complex processing below ~5 kHz with ERB gains above is the DeepFilterNet template and is
  independently supported by TF-Refiner, CoFi-Lite and the DPDFNet branch ablation.
- STFT 1024/512 at 48 kHz, sqrt-Hann, zero lookahead: identical to the shipped FastEnhancer 48k T/B/S configuration;
  measured cost of zero lookahead for denoising is small.
- GRU along time as the temporal core; convolutions with frequency-only kernels; BatchNorm (foldable).
- Joint denoise + dereverb + BVC in one network (not cascaded), since proximity cues are level and reverberation.
- Early-reflection or direct-sound target for a near-field product.
- Mode flag + zero embedding for a multi-mode checkpoint (UPN, USEF-PNet, alkove Focus); FiLM conditioning (used
  per layer by the strongest open model; small published models use concat, with no evidence that FiLM is worse).
- Multi-resolution compressed STFT loss + waveform L1; a per-frame presence head.

### Risky or unsupported
1. Compute target. 50-100 MMAC/s at 48 kHz has no precedent for this task; the only 48 kHz system there is RNNoise.
   Working sub-1M BVC models spend 5-20x more compute per second at 16 kHz. The constraint is also unnecessary:
   under 5% of a laptop core allows roughly 0.5-2 GMAC/s with a speed-oriented design. Also, 0.8-1.5M parameters at
   50-100 MMAC/s and 93.75 frames/s means nearly every weight is used about once per frame (dense/GRU-dominated),
   leaving almost nothing for the Conv2d encoder that the design describes; the two numbers are in tension.
2. Blind proximity BVC as a headline feature on laptop mics. Vendors scope it to headsets. Every open system
   documents the same failures: background talker first, equal-level talkers, far user, quiet user, lone-caller
   damage.
3. Tiny jointly trained 64-d speaker encoder: least-evidenced component at this size and budget.
4. Three behaviours in one ~1M checkpoint: mechanism supported, capacity not demonstrated. Small personalized models
   all degrade SIG well below the unprocessed input.
5. "One 12-24 h run": not supported for the full task from scratch (see G).
6. Missing from the recipe: over-attenuation/asymmetric loss, long-sequence (30-60 s) training, far-field rendering
   and low-pass of interferers, decoupling user distance from interferer distance, device-processing augmentation,
   target-absent handling (SI-SDR is the wrong tool there).
7. GAN fine-tune: small expected gain, stability risk, orthogonal to the hard problems.
8. Custom encoder with Conv2d over time context: time kernels > 1 and non-foldable norms are the documented
   speed traps; depthwise-separable/grouped structure lowers MACs more than it lowers wall-clock time.

### Lower-risk roadmap
1. Baseline: take an existing permissively licensed, proven 48 kHz streaming architecture and retrain it on data with
   clean provenance.
   - FastEnhancer B or S at 48 kHz (MIT; 101K / 207K params; 0.75 / 1.82 GMAC/s; RTF 0.026 / 0.042 on one M5
     thread; 1024/512; existing Rust and C int8 runtimes for the Medium size). The released 48k weights were trained
     partly on AI-Hub data whose export outside South Korea is prohibited, so retraining is needed for clean
     provenance regardless.
   - Or DeepFilterNet3 (MIT/Apache code and weights, 2.1M params, 0.35 GMAC/s, 40 ms; can be run with zero lookahead
     as Hush does). This is the architecture the proposed frontend already imitates, with a public training
     pipeline and the best-documented dereverb behaviour.
   - Exit criterion: match the published checkpoint on VB+D / DNS blind / EARS-WHAM before changing anything.
2. Add dereverb via the target (direct sound or short early window for near-field), an over-attenuation loss, and a
   long-segment fine-tune. Evaluate with non-intrusive metrics and WER.
3. Add blind BVC by data recipe first, architecture second: interferers at 0-15 dB below the user, far-field
   rendered and low-passed, intermittent, 30-60 s sequences, user distance varied independently, device-processing
   augmentation; an auxiliary background-mask head as in Hush. Position it as headset / close-mic and gate it by
   device. Measure on the Krisp benchmark and AMI with WER + leaked words per minute on long windows, and on
   lone-user speech to quantify damage.
4. Add enrollment conditioning (FiLM, zero vector for blind mode) with embeddings from a pretrained or distilled
   encoder; enrollment-embedding augmentation and VAD-weighted loss against over-suppression; a presence gate.
   Consider self-enrollment from the stream as the default, as the strongest open systems do.
5. Distil from a large teacher (offline separator or a big causal TSE model), including pseudo-labels on real
   recordings. Leave any GAN stage for last.

Rough targets: main network 0.2-1.0M params; 0.5-2 GMAC/s at 48 kHz (RTF <= 0.05 on one recent laptop thread in
fp32, less in int8); weights 1-4 MB; latency 21 ms for denoise, with an optional 1-2 frame lookahead mode for
isolation if blind BVC quality demands it; speaker encoder 150-300K params if it must ship in the bundle (distilled),
otherwise any size offline. Budget several single-GPU days per full run and multiple recipe iterations.

---

## Source list

- FastEnhancer: https://github.com/aask1357/fastenhancer , https://arxiv.org/abs/2509.21867 , https://aask1357.github.io/fastenhancer/train/fastenhancer/ , https://github.com/aask1357/fastenhancer/releases
- faster-enhancer.c: https://arxiv.org/abs/2607.25350 , https://github.com/kdrkdrkdr/faster-enhancer.c ; Rust port: https://github.com/TanixLu/faster_enhancer_rs (x86-64 AVX2 only, FastEnhancer-Medium 48 kHz, 320-sample frames, 14.67 ms stated delay, weights embedded; per search summary, UNVERIFIED licence)
- DeepFilterNet: https://github.com/Rikorose/DeepFilterNet , https://arxiv.org/abs/2205.05474 , https://arxiv.org/abs/2305.08227
- DPDFNet: https://arxiv.org/abs/2512.16420 , https://github.com/ceva-ip/DPDFNet , https://huggingface.co/Ceva-IP/DPDFNet
- GigaEdge Enhance SDK: https://arxiv.org/abs/2611.13243
- ULCNet: https://arxiv.org/abs/2312.08132 ; CoFi-Lite: https://arxiv.org/abs/2607.10142 ; HALO: https://arxiv.org/abs/2606.12328 ; HC-AINet: https://arxiv.org/abs/2610.30818 ; LiSenNet-NPU: https://arxiv.org/abs/2609.29866
- LaCo-SENet: https://arxiv.org/abs/2606.19688 ; causal RE-USE: https://arxiv.org/abs/2606.25621 ; RT-SEMamba: https://arxiv.org/abs/2608.12099 ; TF-Refiner: https://arxiv.org/abs/2609.29463 ; aTENNuate: https://arxiv.org/abs/2409.03377
- Latency study: https://arxiv.org/abs/2409.10358
- Dereverb targets: https://arxiv.org/abs/2505.01338 , https://arxiv.org/abs/2603.02641 , https://arxiv.org/abs/2106.12743 , https://pmc.ncbi.nlm.nih.gov/articles/PMC11820353/
- DNS-5: https://arxiv.org/abs/2303.11510 ; TEA-PSE 3.0: https://arxiv.org/abs/2303.07704
- Distance-based: https://www.isca-archive.org/interspeech_2022/patterson22_interspeech.html , https://arxiv.org/abs/2412.20144 ; FVAD: https://arxiv.org/abs/2609.19856
- Open BVC: https://github.com/pulp-vision/Hush , https://github.com/Ground-State-Foundry/nearfield , https://huggingface.co/alkove-ai/alkove-focus-1 , https://github.com/alkove-ai/alkove , https://arxiv.org/abs/2610.40727 (VoxLock), https://arxiv.org/abs/2610.35272 (The Voice on the Line)
- Krisp: https://developers-docs.krisp.ai/docs/krisp-rtc-bvc-nc , https://krisp.ai/blog/improving-turn-taking-of-ai-voice-agents-with-background-voice-cancellation/
- PSE: https://arxiv.org/abs/2204.00771 (E3Net), https://arxiv.org/abs/2302.11768 (Unified PercepNet), https://arxiv.org/abs/2404.08022 (pDeepFilterNet2), https://arxiv.org/abs/2601.16235 (TinyECAPA), https://www.isca-archive.org/interspeech_2024/parnamaa24_interspeech.html , https://arxiv.org/abs/2501.11274 , https://arxiv.org/abs/2505.12288 , https://arxiv.org/abs/2009.04323
- Speaker encoders: https://huggingface.co/nvidia/speakerverification_en_titanet_large , https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb , https://github.com/wenet-e2e/wespeaker , https://github.com/modelscope/3D-Speaker , https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html
- GAN: https://arxiv.org/abs/2505.21057 (LCT-GAN)
- KD: https://arxiv.org/abs/2309.08144 , https://arxiv.org/abs/2408.04267 , https://arxiv.org/abs/2507.19208
- Training cost: https://arxiv.org/abs/2507.09750 (MB-RIRs, DFN3 retraining)
