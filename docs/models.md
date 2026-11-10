# Models

[← Docs index](README.md)

> **Today the app runs the original RNNoise for every tier.** [Plan 0001](plans/0001-deepfilternet3-default.md) switches the default to DeepFilterNet3. The rest of this page is the research behind the model choices.

All run on one CPU thread. No GPU: it costs battery and wakes hardware for a job a single core can do.

| Tier | Model | Rate | Model delay | Cost | Licence | Rust path |
|---|---|---|---|---|---|---|
| Standard (default) | DeepFilterNet3 | 48 kHz | 40 ms | about 3–8% of a recent core; 19% of a 2017 U-series core | MIT/Apache, weights included | native `libDF` + `tract`, vendored |
| Light | RNNoise 0.2 | 48 kHz | 10 ms frames | about 3% of a core | BSD-3 | C library or an audited port (`nnnoiseless`, used today, is the older 2018 model) |
| Max | DPDFNet-2 / -8 (48 kHz) | 48 kHz | unpublished | 4–20× DeepFilterNet3's compute | Apache-2.0 | ONNX via `ort` |
| Challenger | FastEnhancer-S (48 kHz) | 48 kHz | about 21 ms | about 4% of a core (M5) | MIT code; weights licence unstated | ONNX, or hand-written inference |

**Why DeepFilterNet3 first.** It is the only fullband model with a native Rust streaming implementation, clearly permissive weights, and years of field use. Two real problems:

- **Latency.** 40 ms of model delay plus one 10 ms frame of buffering plus device I/O lands around 55–65 ms, over our 50 ms target. Phase 0 must either find a lower-lookahead configuration that still sounds good, or the default becomes FastEnhancer.
- **Maintenance.** Upstream is near-dormant and pinned to an old `tract`. We vendor `libDF` and own it. Compare against the DeepFilterNetPlus fork, which fixes NaN-on-silence and per-frame allocations.

**FastEnhancer is the one to beat it.** Half the delay and less CPU, but no published 48 kHz quality numbers, an unstated weights licence, and training data that includes export-restricted corpora. Shipping it means retraining it ourselves on clean data.

**Second sweep for 2026 releases (10 Nov 2026).** Nothing new displaces the table above. What changed:

- **FastEnhancer now has first-party 48 kHz weights** in five sizes (Tiny to Large), plus a dependency-free Rust port, `faster_enhancer_rs` (x86-64 only, published Oct 2026). Tiny (28k parameters, about 1% of an M5 core) also becomes a challenger for the Light tier. Quality at 48 kHz is still unpublished and the training-data problem stands, so the bake-off has to measure it and shipping still means retraining.
- **No successor to DeepFilterNet3 or RNNoise 0.2.** DPDFNet is at v0.6.0 with the same top model.
- **One licence flag to clear before committing:** a third-party repo claims the DeepFilterNet3 weights are non-commercial, against the upstream MIT/Apache labelling. Read the upstream licence files and issue #700 ourselves in phase 0.
- **Not usable:** several 2026 papers (CoFi-Lite, TF-Refiner, HALO) have no released weights; LaCo-SENet and RT-SEMamba are 16 kHz with missing or ambiguous licences; Clearline's licence is unset because of non-commercial training data.

Bake-off list: DeepFilterNet3 (and the Plus fork), FastEnhancer 48 kHz Tiny / Base / Small, DPDFNet-2 48 kHz, RNNoise 0.2.

**Runtime.** Prefer `tract` (pure Rust, no native library, small). Use `ort` only for the Max tier, where it adds roughly 4–18 MB of native runtime; make that tier an optional download so the base install stays small.

**Strength** is an attenuation limit on one model, not three models.

**Background-voice removal** needs our own training. The open starting point is Hush (16 kHz, DeepFilterNet3 with a separation head used in training); the path is applying that recipe to a 48 kHz model. Usable data: DNS5, LibriTTS, MLS, Common Voice, MUSAN, WHAM. Avoid or check: ESC-50 (non-commercial), FSD50K, AudioSet, DEMAND. This is a post-v1 project of its own.

Two newer open models are worth measuring when we get there, both 16 kHz and both with self-reported numbers only: **alkove Focus** (1.1 MB, Apache-2.0 code and weights, Rust engine, removes noise and other voices; latency and CPU unstated) and **VoxLock** (Apache-2.0, claims to beat Krisp on its author's benchmark, but uses about a full CPU core, so it is a quality ceiling rather than something to ship).
