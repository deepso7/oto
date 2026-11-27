# Oto-1 kernel contract (`.otow` v1, STFT, Candidates A and B)

What the PyTorch exporter must produce and what the Rust kernel (`crates/oto-dsp/src/oto1/`) computes. If the two disagree, this file and the Rust tests decide. Anything not stated here is not part of the model.

Executable references:

- `format.rs`: `OtowWriter` writes a valid file. `random_otow(arch, seed)` in `mod.rs` builds one per candidate.
- `cargo run --release -p oto-dsp --no-default-features --features oto1 --example oto1_bench -- --write` writes `target/oto1/random-{a,b}.otow`, `ref-input.f32` and `ref-{a,b}-out.f32` (raw little-endian f32, 200 hops, scalar backend) for parity checks.

## 1. Signal frame

| | |
|---|---|
| Sample rate | 48 000 Hz, mono |
| FFT | `N_FFT = 1024`, `N_BINS = 513` (bin spacing 46.875 Hz) |
| Hop | `HOP = 512` (10.667 ms) |
| Window | periodic `sqrt(Hann)` for analysis and synthesis: `w[n] = sqrt(0.5 * (1 - cos(2π n / 1024)))`, `n = 0..1024`, computed in f64 and rounded once to f32. `w[n]² + w[n+512]² = 1` |
| Forward | unnormalised real FFT (`torch.fft.rfft(x * w)`) |
| Inverse | `irfft` scaled by 1/1024 (`torch.fft.irfft`). The imaginary parts of DC and Nyquist are ignored |
| Lookahead | none |

Streaming, per hop:

1. Shift `analysis[1024]` (zero at start) left by 512, then append the 512 new samples. Inputs with `|x| < 1e-15` become 0.
2. `X = rfft(analysis * w)`, then the network gives `Y`.
3. `frame = irfft(Y) * w`. Values with `|v| < 1e-15` become 0.
4. `out = frame[0..512] + ola`, then `ola = frame[512..1024]`.

The model delays its input by 512 samples (`Denoiser::latency`). `FrameAdapter`, or the worklet's 128→512 ring, adds another 512. The chain delay is therefore **1024 samples (21.33 ms)**, which the tests check with an impulse.

Training side: use `torch.stft(x, 1024, 512, window=sqrt_hann, center=False)` on a signal left-padded with 512 zeros, or reproduce the loop above. The default `center=True` would put frames on the wrong grid.

## 2. `.otow` v1 layout (little-endian)

The header is 64 bytes:

| Bytes | Field |
|---|---|
| `0..4` | magic `b"OTOW"` |
| `4..8` | `u32` version = 1 |
| `8..12` | `u32` `arch_id`: 1 = Candidate A, 2 = Candidate B |
| `12..16` | `u32` file dtype: 1 = f16, 2 = f32 |
| `16..20` | `u32` flags: bit 0 = isolate (run fail-closed), bit 1 = enrollment (rejected by this kernel version). Other bits must be 0 |
| `20..24` | `u32` `config_len` |
| `24..28` | `u32` `num_tensors` |
| `28..32` | `u32` `table_offset`, a multiple of 64 and at least `64 + config_len` |
| `32..64` | SHA-256 of bytes `[64..EOF]` |

The rest of the file:

- **Config:** UTF-8 JSON at `[64 .. 64 + config_len]`, zero-padded up to `table_offset`. It is flat and canonical: `json.dumps(cfg, sort_keys=True, separators=(",", ":"))`. Integer values are unsigned. The kernel reads only the keys listed per candidate below, uses the defaults when a key is missing, and rejects a value it cannot run.
- **Directory:** at `table_offset`, `num_tensors` entries of 96 bytes each:
  - `[0..64]` name, UTF-8, non-empty, zero-padded.
  - `[64..68]` `u32` ndim, from 1 to 4.
  - `[68..84]` `[u32; 4]` shape, row-major. Unused dimensions are 0, used ones are non-zero.
  - `[84..88]` `u32` dtype: 1 = f16, 2 = f32.
  - `[88..96]` `u64` absolute byte offset, a multiple of 64, placed after the directory.
- **Payloads:** contiguous, row-major, little-endian. Each is zero-padded to a multiple of 64. Payloads must not overlap, and all weights must be finite.
- **f16:** IEEE binary16, written with round-to-nearest-even (`tensor.half()`). It is widened to f32 exactly at load.

Strictness:

- Every tensor the architecture needs must be present with exactly the listed shape.
- Any tensor the architecture does not need is an error.
- Order in the directory is free.
- Conv weights may be stored as `[Co, Ci, K]` (Conv1d) or `[Co, Ci, 1, K]` (Conv2d with a `(1, K)` kernel). Everything else must be stored exactly as listed.

## 3. Primitives

These are the only operations. Activations are stored frequency-major, `[F][C]` (index `f * C + c`). In PyTorch terms, a `[B, C, T, F]` tensor at one time step `t` is `x[b, :, t, :].T` (`F` rows of `C`).

- **`linear(W [out, in], b [out])`:** `y = W x + b`, i.e. `nn.Linear`.
- **`conv(Co, Ci, K, stride, pad)`:** frequency-only, time kernel 1.
  - Equivalent to `nn.Conv2d(Ci, Co, (1, K), stride=(1, stride), padding=(0, pad))` on `[B, Ci, T, F]`.
  - Output size is `F_out = floor((F_in + 2·pad − K) / stride) + 1`, with zero padding on both sides.
  - Fold BatchNorm into weight and bias before export: `W' = W·γ/√(σ²+ε)`, `b' = (b−μ)·γ/√(σ²+ε) + β`.
- **`gru(I, H, G)`:** PyTorch `nn.GRU` equations with gate order `(r, z, n)`:

  ```
  r = σ(W_ir x + b_ir + W_hr h + b_hr)
  z = σ(W_iz x + b_iz + W_hz h + b_hz)
  n = tanh(W_in x + b_in + r ⊙ (W_hn h + b_hn))
  h = (1 − z) ⊙ n + z ⊙ h,   then every |h| < 1e-15 is set to 0
  ```

  - With `G` groups, group `g` reads `x[g·I/G : (g+1)·I/G]` and owns `h[g·H/G : (g+1)·H/G]`, using its own weights (block-diagonal).
  - Tensors: `weight_ih [G, 3H/G, I/G]`, `weight_hh [G, 3H/G, H/G]`, `bias_ih [G, 3H/G]`, `bias_hh [G, 3H/G]`.
  - When `G = 1` there is no leading `G` dimension, which is plain `nn.GRU` (`weight_ih_l0` etc. renamed).
  - Hidden state starts at zero.
- **`group_shuffle(G)`**, between grouped GRU layers (DeepFilterNet's `GroupedGRU` shuffle): `h.view(H/G, G).transpose(0, 1).reshape(H)`, i.e. `out[b·(H/G) + a] = h[a·G + b]`.
- **Sub-pixel upsample by `r`:** a conv with `r·C'` output channels, then output channel `j·C' + c` at frequency `f` becomes channel `c` at frequency `f·r + j`. In PyTorch: `y.view(B, r, C', T, F).permute(0, 2, 3, 4, 1).reshape(B, C', T, F·r)`. In `[F][C]` memory this is a no-op.
- **Skip concat:** `torch.cat([decoder, skip], dim=1)`, decoder channels first.
- **Activations:**
  - `silu(x) = x·σ(x)`, `σ` is the sigmoid, and `tanh`.
  - The kernel evaluates them with a Cephes `exp` polynomial, within ~3e-7 of libm. It clamps the `exp` argument to `[−87, 80]`, so `σ(x)` never returns a subnormal. For `x < −80`, `σ(x)` is ~1.8e-35, against a true value even smaller.
- **Compressed spectrum:** `comp(X) = X · (|X|² + 1e-12)^(−0.35)`, so its magnitude is `|X|^0.3` and its phase is kept. Stored as channels `(re, im)`.

## 4. Candidate A (`arch_id = 1`): DeepFilterNet-style hybrid

Config defaults: `{"arch":"oto1-a","channels":48,"df_order":3,"gru_groups":4,"gru_hidden":192,"n_df_bins":96,"n_erb":32}`.

- `channels` (C), `gru_hidden` (H) and `gru_groups` (G) are free, as long as `G | H` and `G | 8C`.
- `n_erb`, `n_df_bins` and `df_order` are fixed at 32, 96 and 3.

**ERB bands.** There are 32 rectangular bands over all 513 bins, with edges in bins computed in f64:

- `erb(f) = 21.4·log10(1 + 0.00437 f)`.
- For `0 < k < 32`: `edge[k] = max(floor(erb⁻¹(k·erb(24000)/32) / 46.875 + 0.5), edge[k−1] + 2)`.
- `edge[0] = 0` and `edge[32] = 513`.

Pinned by test:

```
[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 33, 39, 45, 53, 62, 73, 85, 99, 116, 134, 156, 182, 211, 245, 284, 329, 381, 442, 513]
```

**Features** for frame `t`, with `P[k] = |X[k]|²`:

- `erb_feat[b] = log10(mean_{k ∈ band b} P[k] + 1e-10)`, shape `[32][1]`.
- `df_feat[k] = comp(X[k])` for `k < 96`, shape `[96][2]`.

**Network.** SiLU after every conv and linear unless noted:

```
e0 = enc.erb0(erb_feat)   conv Co=C  Ci=1  K=3 s=1 p=1   [32][C]
e1 = enc.erb1(e0)         conv C→C        K=3 s=2 p=1   [16][C]
e2 = enc.erb2(e1)         conv C→C        K=3 s=2 p=1   [ 8][C]
d0 = enc.df0(df_feat)     conv Co=C  Ci=2  K=3 s=1 p=1   [96][C]
d1 = enc.df1(d0)          conv C→C        K=3 s=2 p=1   [48][C]
d2 = enc.df2(d1)          conv C→C        K=3 s=2 p=1   [24][C]
d3 = enc.df3(d2)          conv C→C        K=3 s=3 p=0   [ 8][C]
emb = e2 + d3                                            [8C]   (flattened f·C + c)
h0 = gru0(emb)            grouped GRU I=8C, H, G     (state)
h1 = gru1(group_shuffle(h0))  grouped GRU I=H, H, G   (state)

t  = silu(erb_dec.fc(h1)) + e2                           [8][C]
u  = silu(erb_dec.up2(t))  conv C→2C K=3 s=1 p=1, sub-pixel r=2 → [16][C];  u += e1
v  = silu(erb_dec.up1(u))  conv C→2C K=3 s=1 p=1, sub-pixel r=2 → [32][C];  v += e0
gains = σ(erb_dec.out(v))  conv C→1  K=3 s=1 p=1          [32]

q    = silu(df_dec.fc0(h1))                               [8C]
coef = tanh(df_dec.fc1(q) + df_dec.convp(d0))             [96][3·2]  (convp: conv C→6, K=1, no activation)
p_target = p_head(h1)                                     scalar logit, no activation
```

`coef[k][2i]`, `coef[k][2i+1]` is the real and imaginary part of tap `i`. Tap `i` multiplies `X_{t−i}`, the spectrum `i` hops ago, zero before the stream starts.

**Output spectrum.**

- `Y[k] = Σ_{i=0..2} (coef[k][2i] + j·coef[k][2i+1]) · X_{t−i}[k]` for `k < 96` (deep filter on the *noisy* spectrum).
- `Y[k] = gains[band(k)] · X[k]` for `96 ≤ k ≤ 512` (Nyquist included).

Tensors (C=48, H=192, G=4): 574 328 parameters, 1 622 208 MAC per hop (152 MMAC/s).

```
enc.erb0.weight [48,1,3]    enc.erb0.bias [48]
enc.erb1.weight [48,48,3]   enc.erb1.bias [48]
enc.erb2.weight [48,48,3]   enc.erb2.bias [48]
enc.df0.weight  [48,2,3]    enc.df0.bias  [48]
enc.df1.weight  [48,48,3]   enc.df1.bias  [48]
enc.df2.weight  [48,48,3]   enc.df2.bias  [48]
enc.df3.weight  [48,48,3]   enc.df3.bias  [48]
gru0.weight_ih [4,144,96]   gru0.weight_hh [4,144,48]   gru0.bias_ih [4,144]   gru0.bias_hh [4,144]
gru1.weight_ih [4,144,48]   gru1.weight_hh [4,144,48]   gru1.bias_ih [4,144]   gru1.bias_hh [4,144]
erb_dec.fc.weight  [384,192]  erb_dec.fc.bias  [384]
erb_dec.up2.weight [96,48,3]  erb_dec.up2.bias [96]
erb_dec.up1.weight [96,48,3]  erb_dec.up1.bias [96]
erb_dec.out.weight [1,48,3]   erb_dec.out.bias [1]
df_dec.fc0.weight  [384,192]  df_dec.fc0.bias  [384]
df_dec.fc1.weight  [576,384]  df_dec.fc1.bias  [576]
df_dec.convp.weight [6,48,1]  df_dec.convp.bias [6]
p_head.weight [1,192]         p_head.bias [1]
```

## 5. Candidate B (`arch_id = 2`): FastEnhancer-style U-Net

Config defaults: `{"arch":"oto1-b","channels":32,"gru_hidden":128}`. `channels` (C) and `gru_hidden` (H) are free.

The network sees bins `0..512`. The Nyquist bin is dropped on input and zeroed on output.

**Features:** `feat[k] = comp(X[k])` for `k < 512`, shape `[512][2]`.

**Network.** SiLU after every conv and linear unless noted:

```
e0 = enc0(feat)   conv Co=C Ci=2 K=8 s=4 p=2   [128][C]
e1 = enc1(e0)     conv C→C      K=3 s=2 p=1   [ 64][C]
e2 = enc2(e1)     conv C→C      K=3 s=2 p=1   [ 32][C]
e3 = enc3(e2)     conv C→C      K=3 s=2 p=1   [ 16][C]
h0 = gru0(e3)     GRU I=16C, H   (state; input flattened f·C + c)
h1 = gru1(h0)     GRU I=H,   H   (state)
b  = silu(bott.fc(h1))                         [16][C]
d3 = dec3(cat[b,  e3])  conv 2C→2C K=3 s=1 p=1, sub-pixel r=2 → [ 32][C]
d2 = dec2(cat[d3, e2])  same                    → [ 64][C]
d1 = dec1(cat[d2, e1])  same                    → [128][C]
m  = dec0(cat[d1, e0])  conv 2C→8 K=3 s=1 p=1, no activation, sub-pixel r=4 → [512][2]
p_target = p_head(h1)   scalar logit
```

**Output spectrum.** The mask is a bounded complex ratio mask:

- For `k < 512`: `M = m[k][0] + j·m[k][1]`, then `Y[k] = X[k] · M · tanh(|M|) / max(|M|, 1e-8)`. So `|mask| < 1`, with the phase of `M`.
- `Y[512] = 0`.

Tensors (C=32, H=128): 460 233 parameters, 2 392 192 MAC per hop (224 MMAC/s).

```
enc0.weight [32,2,8]   enc0.bias [32]
enc1.weight [32,32,3]  enc1.bias [32]
enc2.weight [32,32,3]  enc2.bias [32]
enc3.weight [32,32,3]  enc3.bias [32]
gru0.weight_ih [384,512]  gru0.weight_hh [384,128]  gru0.bias_ih [384]  gru0.bias_hh [384]
gru1.weight_ih [384,128]  gru1.weight_hh [384,128]  gru1.bias_ih [384]  gru1.bias_hh [384]
bott.fc.weight [512,128]  bott.fc.bias [512]
dec3.weight [64,64,3]  dec3.bias [64]
dec2.weight [64,64,3]  dec2.bias [64]
dec1.weight [64,64,3]  dec1.bias [64]
dec0.weight [8,64,3]   dec0.bias [8]
p_head.weight [1,128]  p_head.bias [1]
```

## 6. Numerics and parity

| Layer | Gate |
|---|---|
| Kernels vs naive reference (`kernels.rs` tests) | Scalar, SSE2 and wasm `simd128` are bit-identical (same summation order: bias, then `i` ascending, no FMA). NEON and AVX2 use FMA and stay within −90 dB (measured about −130 dB end to end). Activations are bit-identical on every backend |
| SIMD vs scalar, whole model (`tests.rs`) | ≤ −90 dB relative RMS over 300 hops |
| wasm vs native (`web/demo/bench-node.mjs` against `ref-*.f32`) | ≤ −90 dB. The residual comes from `powf`/`log10` differing between libms |
| Rust scalar vs PyTorch (exporter side) | ≤ −90 dB on the reference vectors, max abs ≈ 1e-6 |

- **Silence:** all-zero input gives an all-zero output, because every path multiplies `X`.
- **Subnormals:** no carried state value is subnormal after silence, tiny input or subnormal input, on any backend.
