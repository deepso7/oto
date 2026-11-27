"""Training losses for Oto-1 (plan 0004, Decision 8).

All terms are NaN-free for all-zero targets and all-zero estimates: every power is
taken on ``|X|^2 + eps``, every log on ``energy + eps``, and no term divides by the
target energy (no SI-SDR).

* :func:`mr_compressed_stft_loss` -- multi-resolution compressed (``c = 0.3``) magnitude +
  complex STFT loss, with per-clip effective-bandwidth masking above ``0.9 * f_max``.
* :func:`over_attenuation_loss` -- asymmetric penalty when the estimate's compressed
  magnitude falls *below* the target's (swallowed speech), on the same masked bins.
* :func:`bandlimited_l1` -- waveform L1 after removing content above ``0.9 * f_max``.
* :func:`silent_target_loss` -- scale-aware level loss on target-silent frames
  (background-only / pre-speech): pushes output energy toward ``-floor_db`` relative to the
  input, then stops (no gradient once the residual is quieter than the floor).
* :func:`presence_bce` + :func:`presence_labels` -- frame-level target-presence head.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from .models import stft as S

DEFAULT_RESOLUTIONS = ((256, 64), (512, 128), (1024, 256), (2048, 512))  # ~5/10/20/40 ms at 48 kHz


def _bin_mask(f_max: Tensor | None, n_fft: int, batch: int, device, ratio: float = 0.9) -> Tensor:
    """``(B, 1, n_bins)`` 0/1 mask keeping bins at or below ``ratio * f_max``."""
    freqs = torch.arange(n_fft // 2 + 1, device=device) * (S.SR / n_fft)
    if f_max is None:
        return torch.ones(batch, 1, freqs.numel(), device=device)
    return (freqs[None, :] <= ratio * f_max[:, None].to(device)).float()[:, None, :]


def _spec(x: Tensor, n_fft: int, hop: int) -> Tensor:
    win = torch.hann_window(n_fft, device=x.device)
    s = torch.stft(x.float(), n_fft, hop, window=win, return_complex=True, center=True)
    return torch.view_as_real(s).transpose(1, 2)  # (B, T, F, 2)


def _compress(s: Tensor, c: float, eps: float) -> tuple[Tensor, Tensor]:
    p = s.square().sum(-1) + eps
    mag_c = p.pow(c / 2)
    cplx_c = s * p.pow((c - 1) / 2).unsqueeze(-1)
    return mag_c, cplx_c


def mr_compressed_stft_loss(
    est: Tensor,
    tgt: Tensor,
    f_max: Tensor | None = None,
    resolutions=DEFAULT_RESOLUTIONS,
    c: float = 0.3,
    complex_weight: float = 1.0,
    oa_weight: float = 0.0,
    eps: float = 1e-10,
) -> tuple[Tensor, Tensor]:
    """Returns ``(spectral_loss, over_attenuation_loss)`` averaged over resolutions.

    ``est``/``tgt``: ``(B, T)`` waveforms; ``f_max``: ``(B,)`` effective bandwidth in Hz.
    """
    total = est.new_zeros(())
    oa_total = est.new_zeros(())
    b = est.shape[0]
    for n_fft, hop in resolutions:
        se, st = _spec(est, n_fft, hop), _spec(tgt, n_fft, hop)
        me, ce = _compress(se, c, eps)
        mt, ct = _compress(st, c, eps)
        mask = _bin_mask(f_max, n_fft, b, est.device)  # (B, 1, F)
        denom = mask.sum().clamp_min(1.0) * se.shape[1]
        mag_l = ((me - mt).square() * mask).sum() / denom
        cplx_l = ((ce - ct).square().sum(-1) * mask).sum() / denom
        total = total + mag_l + complex_weight * cplx_l
        if oa_weight:
            oa_total = oa_total + (torch.relu(mt - me).square() * mask).sum() / denom
    n = len(resolutions)
    return total / n, oa_total / n


def over_attenuation_loss(est: Tensor, tgt: Tensor, f_max: Tensor | None = None, **kw) -> Tensor:
    return mr_compressed_stft_loss(est, tgt, f_max, oa_weight=1.0, **kw)[1]


def bandlimited_l1(est: Tensor, tgt: Tensor, f_max: Tensor | None = None, ratio: float = 0.9) -> Tensor:
    if f_max is None:
        return (est - tgt).abs().mean()
    n = est.shape[-1]
    d = torch.fft.rfft((est - tgt).float(), dim=-1)
    freqs = torch.fft.rfftfreq(n, 1.0 / S.SR).to(est.device)
    keep = (freqs[None, :] <= ratio * f_max[:, None].to(est.device)).to(d.dtype)
    return torch.fft.irfft(d * keep, n=n, dim=-1).abs().mean()


def frame_energy(x: Tensor, frame: int = S.HOP) -> Tensor:
    """``(B, T) -> (B, T // frame)`` mean-square energy per non-overlapping frame."""
    n = x.shape[-1] // frame * frame
    return x[..., :n].float().reshape(*x.shape[:-1], -1, frame).square().mean(-1)


def presence_labels(tgt: Tensor, abs_floor_db: float = -70.0, rel_db: float = -45.0) -> Tensor:
    """Per-hop target-presence labels ``(B, T // 512)``.

    A frame is "present" if its energy is above ``abs_floor_db`` dBFS (full-scale sine = -3 dB)
    and within ``rel_db`` of the clip's loudest frame.
    """
    e = frame_energy(tgt)
    e_db = 10 * torch.log10(e + 1e-12)
    peak = e_db.amax(-1, keepdim=True)
    return ((e_db > abs_floor_db) & (e_db > peak + rel_db)).float()


def silent_target_loss(
    est: Tensor, mix: Tensor, tgt: Tensor, floor_db: float = 50.0, silent: Tensor | None = None
) -> Tensor:
    """Scale-aware loss on frames where the target is silent.

    ``L = mean_over_silent_frames( relu(10 log10((E_out + d) / (E_in + d)) + floor_db) ) / floor_db``

    i.e. the output should be at least ``floor_db`` quieter than the input on those frames.
    It is invariant to the overall input level, finite for all-zero inputs/targets, and gives
    no gradient once enough attenuation is reached. ``d`` is -100 dBFS so digital-silence
    inputs give zero loss instead of an exploding ratio.
    """
    if silent is None:
        silent = 1.0 - presence_labels(tgt)
    d = 1e-10
    eo, ei = frame_energy(est), frame_energy(mix)
    n = min(eo.shape[-1], silent.shape[-1])
    eo, ei, silent = eo[..., :n], ei[..., :n], silent[..., :n]
    ratio_db = 10 * torch.log10(eo + d) - 10 * torch.log10(ei + d)
    per = torch.relu(ratio_db + floor_db) / floor_db
    return (per * silent).sum() / silent.sum().clamp_min(1.0)


def presence_bce(p_logit: Tensor, labels: Tensor) -> Tensor:
    n = min(p_logit.shape[-1], labels.shape[-1])
    return nn.functional.binary_cross_entropy_with_logits(p_logit[..., :n].float(), labels[..., :n])


@dataclass
class LossWeights:
    spectral: float = 1.0
    complex: float = 1.0
    l1: float = 0.2
    over_atten: float = 1.0
    silent: float = 0.1
    presence: float = 0.05
    resolutions: tuple = field(default_factory=lambda: DEFAULT_RESOLUTIONS)


class OtoLoss(nn.Module):
    def __init__(self, w: LossWeights | None = None):
        super().__init__()
        self.w = w or LossWeights()

    def forward(self, est: Tensor, tgt: Tensor, mix: Tensor, p_logit: Tensor, f_max: Tensor | None = None):
        w = self.w
        est, tgt, mix = est.float(), tgt.float(), mix.float()
        spec, oa = mr_compressed_stft_loss(
            est, tgt, f_max, w.resolutions, complex_weight=w.complex, oa_weight=w.over_atten
        )
        l1 = bandlimited_l1(est, tgt, f_max)
        labels = presence_labels(tgt)
        silent = silent_target_loss(est, mix, tgt, silent=1.0 - labels)
        pres = presence_bce(p_logit, labels)
        total = w.spectral * spec + w.over_atten * oa + w.l1 * l1 + w.silent * silent + w.presence * pres
        parts = {"spec": spec.detach(), "oa": oa.detach(), "l1": l1.detach(), "silent": silent.detach(),
                 "presence": pres.detach()}  # fmt: skip
        return total, parts
