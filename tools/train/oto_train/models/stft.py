"""Exact causal STFT / iSTFT shared with the pure-Rust kernel (`crates/oto-dsp`).

Contract: ``crates/oto-dsp/src/oto1/SPEC.md`` section 1 (that file decides any disagreement).

* ``SR = 48_000``, ``N_FFT = 1024``, ``HOP = 512``, ``N_BINS = 513``.
* Periodic sqrt-Hann window ``w[n] = sqrt(0.5 * (1 - cos(2*pi*n/1024)))``, used for
  both analysis and synthesis. ``w[n]^2 + w[n+512]^2 == 1`` (COLA), so no extra
  normalisation is applied on synthesis.
* Forward transform: unnormalised ``rfft``. Inverse: ``irfft`` (``1/1024`` scaling).
* Causal framing: 512 zeros are prepended, so frame ``f`` covers input samples
  ``x[(f-1)*512 .. (f+1)*512]``; frame 0 sees ``[zeros(512), x[0..512]]``.
* Streaming: at step ``b`` the caller pushes input block ``x[b*512 .. (b+1)*512]``,
  the kernel analyses frame ``b``, and emits the overlap-add of the second half of
  frame ``b-1`` and the first half of frame ``b``, i.e. the reconstruction of input
  block ``b-1``. Output is therefore delayed by exactly 512 samples; algorithmic
  latency is 1024 samples (21.33 ms) including block buffering.

Batch :func:`istft` returns the *aligned* signal (sample ``n`` of the output is the
reconstruction of input sample ``n``); the last 512 samples are incomplete because
frame ``F`` does not exist yet. The streaming output equals ``[pre_roll(512),
istft(X)[:-512]]`` up to float rounding, where ``pre_roll`` is the first half of the
processed frame 0 (it reconstructs the 512 zeros of left padding; it is exactly zero
for an identity model and is meaningless otherwise, so callers discard or emit it as-is).
Parity tests therefore compare ``stream_out[512:]`` with ``batch_out[:-512]``.

Spectra are carried as real tensors with a trailing ``(re, im)`` axis of size 2 so
that everything works under bf16 autocast and on every backend.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

SR = 48_000
N_FFT = 1024
HOP = 512
N_BINS = N_FFT // 2 + 1
FRAMES_PER_SECOND = SR / HOP  # 93.75
BIN_HZ = SR / N_FFT  # 46.875
#: Streaming inputs, synthesis frames and carried GRU states below this magnitude are flushed to 0.
FLUSH = 1e-15


def sqrt_hann(n_fft: int = N_FFT, dtype: torch.dtype = torch.float32, device=None) -> Tensor:
    """Periodic sqrt-Hann window, computed in float64 then cast (matches Rust f64->f32)."""
    n = torch.arange(n_fft, dtype=torch.float64, device=device)
    w = torch.sqrt(0.5 * (1.0 - torch.cos(2.0 * math.pi * n / n_fft)))
    return w.to(dtype)


def _as_ri(spec: Tensor) -> Tensor:
    return torch.view_as_real(spec)


def _as_complex(spec_ri: Tensor) -> Tensor:
    return torch.view_as_complex(spec_ri.float().contiguous())


def stft(x: Tensor, window: Tensor | None = None) -> Tensor:
    """Causal batch STFT.

    Args:
        x: ``(..., T)`` waveform. ``T`` is right-padded with zeros to a multiple of 512.
    Returns:
        ``(..., F, 513, 2)`` spectrum with ``F = ceil(T / 512)``.
    """
    if window is None:
        window = sqrt_hann(device=x.device)
    x = x.float()
    t = x.shape[-1]
    n_frames = -(-t // HOP)
    pad_right = n_frames * HOP - t
    xp = torch.nn.functional.pad(x, (HOP, pad_right))
    frames = xp.unfold(-1, N_FFT, HOP)  # (..., F, 1024)
    spec = torch.fft.rfft(frames * window, n=N_FFT, dim=-1)
    return _as_ri(spec)


def istft(spec_ri: Tensor, window: Tensor | None = None) -> Tensor:
    """Aligned batch iSTFT (overlap-add with the synthesis window).

    Args:
        spec_ri: ``(..., F, 513, 2)``.
    Returns:
        ``(..., F * 512)`` waveform aligned with the input of :func:`stft`.
    """
    if window is None:
        window = sqrt_hann(device=spec_ri.device)
    frames = torch.fft.irfft(_as_complex(spec_ri), n=N_FFT, dim=-1) * window  # (..., F, 1024)
    n_frames = frames.shape[-2]
    first = frames[..., :HOP]  # contributes to block f-1 (padded index f)
    second = frames[..., HOP:]  # contributes to block f
    # Aligned output block b = second half of frame b + first half of frame b+1.
    nxt = torch.nn.functional.pad(first[..., 1:, :], (0, 0, 0, 1))
    out = second + nxt
    return out.reshape(*out.shape[:-2], n_frames * HOP)


class StreamingSTFT:
    """Frame-by-frame analysis + synthesis with the exact Rust state layout.

    State: ``in_buf`` (previous 512 input samples) and ``ola_buf`` (second half of
    the previous synthesis frame). Works on arbitrary leading batch dims.
    """

    def __init__(self, batch_shape: tuple[int, ...] = (), device=None, dtype=torch.float32):
        self.window = sqrt_hann(device=device, dtype=dtype)
        self.in_buf = torch.zeros(*batch_shape, HOP, device=device, dtype=dtype)
        self.ola_buf = torch.zeros(*batch_shape, HOP, device=device, dtype=dtype)

    def analyze(self, block: Tensor) -> Tensor:
        """Push 512 new samples, return the ``(..., 513, 2)`` spectrum of the new frame."""
        block = block.to(self.in_buf.dtype)
        block = torch.where(block.abs() < FLUSH, torch.zeros_like(block), block)
        frame = torch.cat([self.in_buf, block], dim=-1)
        self.in_buf = block.clone()
        return _as_ri(torch.fft.rfft(frame * self.window, n=N_FFT, dim=-1))

    def synthesize(self, spec_ri: Tensor) -> Tensor:
        """Inverse-transform one frame and return 512 finished output samples."""
        frame = torch.fft.irfft(_as_complex(spec_ri), n=N_FFT, dim=-1) * self.window
        frame = torch.where(frame.abs() < FLUSH, torch.zeros_like(frame), frame)
        out = self.ola_buf + frame[..., :HOP]
        self.ola_buf = frame[..., HOP:].clone()
        return out


# ---------------------------------------------------------------------------
# ERB bands (SPEC.md section 4, pinned by test on both sides)
# ---------------------------------------------------------------------------

ERB_EDGES_32 = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 33, 39, 45, 53, 62, 73, 85, 99, 116, 134, 156,
                182, 211, 245, 284, 329, 381, 442, 513]  # fmt: skip


def freq2erb(f: float) -> float:
    """ERB-rate ``21.4 * log10(1 + 0.00437 f)`` (Glasberg & Moore)."""
    return 21.4 * math.log10(1.0 + 0.00437 * f)


def erb2freq(e: float) -> float:
    return (10.0 ** (e / 21.4) - 1.0) / 0.00437


def erb_band_edges(n_bands: int = 32, min_width: int = 2) -> list[int]:
    """``n_bands + 1`` band edges in bins (first 0, last 513), exactly as ``erb_band_edges`` in Rust.

    ``edge[k] = max(floor(erb^-1(k * erb(24000) / n) / 46.875 + 0.5), edge[k-1] + 2)`` for
    ``0 < k < n``, computed in float64.
    """
    top = freq2erb(SR / 2)
    edges = [0] * (n_bands + 1)
    for k in range(1, n_bands):
        b = math.floor(erb2freq(k * top / n_bands) / BIN_HZ + 0.5)
        edges[k] = max(b, edges[k - 1] + min_width)
    edges[n_bands] = N_BINS
    assert edges[n_bands] >= edges[n_bands - 1] + min_width, edges
    return edges


def erb_matrices(edges: list[int]) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(sel, width, inv)``.

    ``sel``: ``(513, n)`` 0/1 band-membership matrix (``P @ sel`` sums bin power per band),
    ``width``: ``(n,)`` bins per band, ``inv = sel.T`` copies each band gain onto its bins.
    """
    n = len(edges) - 1
    sel = torch.zeros(N_BINS, n)
    for b in range(n):
        sel[edges[b] : edges[b + 1], b] = 1.0
    width = torch.tensor([edges[b + 1] - edges[b] for b in range(n)], dtype=torch.float32)
    return sel, width, sel.T.contiguous()
