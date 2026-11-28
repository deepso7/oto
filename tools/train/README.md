# tools/train: Oto-1 training, data packing and `.otow` export

PyTorch side of [plan 0004](../../docs/plans/0004-oto-v1-model-and-wasm.md), Phase 1: the two candidate models (batch and exact streaming), losses, data pipeline, trainer, Colab notebook and the `.otow` v1 exporter read by `crates/oto-dsp`. Data use is bound by [docs/training-data-audit.md](../../docs/training-data-audit.md).

```sh
cd tools/train
uv sync --extra rir --group dev                 # project-local .venv (torch, numpy, scipy, ...)
uv run pytest                                   # parity, export, losses, data, smoke-train tests
uv run python -m oto_train.models               # param and MAC counts
uv run python -m oto_train.train --smoke --arch a --ckpt-dir /tmp/oto-smoke   # no datasets needed
```

Extras: `rir` (pyroomacoustics), `upload` (boto3 for R2, huggingface_hub), `plots` (spectrogram PNGs for spot checks).

## Layout

| Path | What |
|---|---|
| `oto_train/models/stft.py` | Shared STFT contract: 48 kHz, `1024/512`, periodic sqrt-Hann, causal 512-sample left pad; batch `stft`/`istft`, `StreamingSTFT`; the pinned 32-band ERB edges |
| `oto_train/models/layers.py` | The op set the Rust kernel mirrors: freq-only `FreqConv` (+foldable BN, SiLU), grouped `GRULayer`, `group_shuffle`, sub-pixel upsample, `comp` |
| `oto_train/models/candidate_a.py` | `arch_id = 1` (`oto1-a`), DeepFilterNet-style: 32 ERB gains + 3-tap complex DF on bins `0..96`, 2 grouped GRUs 192/4; 574 328 params |
| `oto_train/models/candidate_b.py` | `arch_id = 2` (`oto1-b`), FastEnhancer-style: freq-only conv U-Net `512->128->64->32->16`, 2 GRUs 128, bounded complex mask; 373 817 params |
| `oto_train/export_otow.py` | BN folding, `.safetensors` + `.otow` v1 writer, verifying `read_otow`, `load_otow_into_model` (also loads Rust-written files), parity fixtures |
| `oto_train/losses.py` | Bandwidth-masked multi-res compressed STFT, over-attenuation, band-limited L1, scale-aware silent-target, presence BCE |
| `oto_train/data/bandwidth.py` | Effective bandwidth (`-50 dB` from peak, Hi-Fi TTS / URGENT method) |
| `oto_train/data/pack_shards.py` | Licence-enforcing shard packer (run on `mito`), spot checks, splits, `ATTRIBUTION.md`, R2/HF upload |
| `oto_train/data/rir_gen.py` | Paired near (`0.15-0.6 m`) / far (`1.2-6 m`) 48 kHz RIRs per room (pyroomacoustics hybrid ISM + ray tracing) |
| `oto_train/data/mixer.py` | GPU mixer: FFT RIR conv, early-reflection dereverb target, noise, EQ / clipping / band-limit, Phase 3 interferer hooks |
| `oto_train/data/stream.py` | Shard reader and synthetic (`--smoke`) sample streams |
| `oto_train/train.py` | Trainer: bf16, `--compile`, EMA, NVMe staging, auto-resume, long-seq and isolate modes, `.otow` export of the best model |
| `colab_train.ipynb` | Colab driver for the trainer |

## Streaming contract

Frame `f` covers input samples `[(f-1)*512, (f+1)*512)`; frame 0 is `[zeros(512), x[0..512]]`. Each streaming step takes 512 new samples and returns 512: the overlap-add of the previous frame's second half and the current frame's first half, i.e. input block `f-1`. The first 512 output samples are a pre-roll (exactly zero for an identity model). Parity is checked as `stream_out[512:] == batch_out[:-512]`. Total algorithmic latency is 1024 samples (21.33 ms).

Convolutions only look along frequency, so the only temporal state is the GRU hidden state (`(layers, B, hidden)`) and, for Candidate A, the last two frames of the 96 low bins for deep filtering.

## `.otow` v1 and the Rust kernel

The contract is [`crates/oto-dsp/src/oto1/SPEC.md`](../../crates/oto-dsp/src/oto1/SPEC.md): file layout, flat config JSON (`{"arch":"oto1-b","channels":24,"gru_hidden":128}`, plus `meta_*` keys the kernel ignores), tensor names and shapes, features and equations. The models here follow it exactly, so `export_tensors()` is the file's tensor list, and `load_otow_into_model()` loads any v1 file, including the kernel's own random models.

`tests/test_rust_parity.py` checks both directions at <= -90 dB (needs `cargo`; the wasm part also needs `node` and `web/demo/wasm/`): the kernel's `target/oto1/random-{a,b}.otow` run in PyTorch (streaming and batch) against `ref-{a,b}-out.f32`, and PyTorch exports (f32, f16, folded random BatchNorm, a `--smoke` checkpoint) run through every native backend with `oto1_bench --eval <model.otow> <in.f32> <out-prefix>` and through both wasm builds.

```sh
# random-weight files for Rust parity tests and browser benchmarks, plus a streaming fixture
python -m oto_train.export_otow --arch a --random --dtype f32 --out out/a.otow \
    --safetensors out/a.safetensors --parity-fixture out/a_parity.safetensors
# trained checkpoint
python -m oto_train.export_otow --ckpt runs/a/best.pt --out out/oto1_clean.otow --dtype f16
```

The parity fixture holds `input`, `stream_out` (pre-roll included), `p_logit` per frame, and `spec_in`/`spec_out` for frames 0..8, computed with the weights as stored in the `.otow` (after f16 rounding).

## Data pipeline (on `mito`)

1. HiFiTTS-2 spot check before bulk download: `python -m oto_train.data.pack_shards spot-check --hifitts2-manifest ... --hifitts2-audio ... --spot-check 50 --out spot/`. Look at `spot_check_summary.json` and the PNGs.
2. RIRs: `python -m oto_train.data.rir_gen --out rirs/ --rooms 20000`. Room `% 50 == 0` is dev, `% 50 == 1` is test.
3. Pack: `python -m oto_train.data.pack_shards pack --out shards/ --hifitts2-manifest ... --vctk ... --fsd50k ... --fma-tracks ... --fma-audio ...` (`--dry-run` writes the manifest, split lists and `ATTRIBUTION.md` without audio). Unknown or denied sources abort; FSD50K and FMA are filtered per clip to CC0 / CC-BY; HiFiTTS-2 rows with no bandwidth or no single-speaker information are dropped (`--hifitts2-allow-unknown-multispk` overrides the latter). The HiFiTTS-2 manifest field names are flags, so check them against the downloaded manifest.
4. Upload: `python -m oto_train.data.pack_shards upload --src shards/ --r2-bucket oto-train --prefix shards/v1/` (env `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`) or `--hf-repo user/repo` (private dataset repo).

## Training

`python -m oto_train.train --arch a --data r2://oto-train/shards/v1/ --rir-data r2://oto-train/rirs/v1/ --ckpt-dir /content/ckpt --sync r2://oto-train/runs/a1/ --compile`. Re-running the same command resumes from `latest.pt` (local or synced). `--long-seq` switches to 30 s+ segments at a lower learning rate; `--isolate` turns on the Phase 3 mixing recipe; `--init-from` starts a fine-tune from another checkpoint.
