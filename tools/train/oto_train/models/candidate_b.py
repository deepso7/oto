"""Candidate B (``arch_id = 2``, ``"oto1-b"``): FastEnhancer-style U-Net. SPEC.md section 5.

Per frame (``X``: ``(513, 2)``; the Nyquist bin 512 is dropped on input and zeroed on output):

* ``feat[k] = comp(X[k])`` for ``k < 512``, channels ``(re, im)``.
* Encoder: ``enc0`` 2 -> C (K=8, s=4, p=2, 512 -> 128 rows), ``enc1..3`` (K=3, s=2: 64, 32, 16 rows).
* ``gru0`` (plain GRU, I = 16C flattened ``f*C + c``), ``gru1``; ``bott.fc`` + SiLU -> ``[16][C]``.
* Decoder: ``dec{3,2,1}(cat[decoder, skip])`` 2C -> 2C + SiLU, sub-pixel r=2; ``dec0`` 2C -> 8, no
  activation, sub-pixel r=4 -> ``m [512][2]``.
* Bounded complex mask ``M * tanh(|M|) / max(|M|, 1e-8)``; ``Y[512] = 0``. ``p_head``: logit.

Every conv except ``dec0`` carries a foldable BatchNorm.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from . import stft as S
from .base import OtoModel
from .layers import FreqConv, GRULayer, comp, complex_mul, flat_fc, linear_macs, subpixel, unflat_fc

F_IN = 512
F_BOTT = 16


@dataclass
class CandidateBConfig:
    """Exactly the ``.otow`` config keys (``arch`` aside)."""

    channels: int = 24
    gru_hidden: int = 128

    def __post_init__(self):
        if self.channels <= 0 or self.gru_hidden <= 0:
            raise ValueError(f"Oto-1 B: inconsistent config {self}")


class Bott(nn.Module):
    def __init__(self, c: int, h: int):
        super().__init__()
        self.fc = nn.Linear(h, F_BOTT * c)


class CandidateB(OtoModel):
    arch_id = 2
    arch_name = "oto1-b"

    def __init__(self, cfg: CandidateBConfig | None = None):
        cfg = cfg or CandidateBConfig()
        super().__init__(cfg)
        c, h = cfg.channels, cfg.gru_hidden
        self.enc0 = FreqConv(2, c, 8, 4, 2)
        self.enc1 = FreqConv(c, c, 3, 2, 1)
        self.enc2 = FreqConv(c, c, 3, 2, 1)
        self.enc3 = FreqConv(c, c, 3, 2, 1)
        self.gru0 = GRULayer(F_BOTT * c, h)
        self.gru1 = GRULayer(h, h)
        self.bott = Bott(c, h)
        self.dec3 = FreqConv(2 * c, 2 * c, 3, 1, 1)
        self.dec2 = FreqConv(2 * c, 2 * c, 3, 1, 1)
        self.dec1 = FreqConv(2 * c, 2 * c, 3, 1, 1)
        self.dec0 = FreqConv(2 * c, 8, 3, 1, 1, bn=False, act=False)
        self.p_head = nn.Linear(h, 1)

    def _encode(self, x: Tensor):
        feat = comp(x[:, :F_IN].float()).transpose(1, 2)  # (N, 2, 512)
        e0 = self.enc0(feat)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        return flat_fc(e3), (e0, e1, e2, e3)

    def _decode(self, h1: Tensor, skips, x: Tensor):
        e0, e1, e2, e3 = skips
        b = unflat_fc(nn.functional.silu(self.bott.fc(h1)), self.cfg.channels)  # (N, C, 16)
        d = subpixel(self.dec3(torch.cat([b, e3], 1)), 2)
        d = subpixel(self.dec2(torch.cat([d, e2], 1)), 2)
        d = subpixel(self.dec1(torch.cat([d, e1], 1)), 2)
        m = subpixel(self.dec0(torch.cat([d, e0], 1)), 4).transpose(1, 2).float()  # (N, 512, 2)
        mag = m.square().sum(-1, keepdim=True).clamp_min(1e-30).sqrt()  # clamp only keeps grads finite
        y = complex_mul(m * (torch.tanh(mag) / mag.clamp_min(1e-8)), x[:, :F_IN].float())
        y = torch.cat([y, torch.zeros_like(y[:, :1])], dim=1)
        return y, self.p_head(h1).squeeze(-1)

    def forward_spec(self, x: Tensor) -> tuple[Tensor, Tensor]:
        b, t = x.shape[:2]
        xf = x.reshape(b * t, S.N_BINS, 2)
        emb, skips = self._encode(xf)
        h0, _ = self.gru0(emb.view(b, t, -1))
        h1, _ = self.gru1(h0)
        y, p = self._decode(h1.reshape(b * t, -1), skips, xf)
        return y.view(b, t, S.N_BINS, 2), p.view(b, t)

    def init_state(self, batch: int, device=None):
        h = self.cfg.gru_hidden
        return {"h0": torch.zeros(batch, h, device=device), "h1": torch.zeros(batch, h, device=device)}

    def step_spec(self, x: Tensor, state):
        emb, skips = self._encode(x)
        _, h0 = self.gru0(emb[:, None], state["h0"], flush=True)
        _, h1 = self.gru1(h0[:, None], state["h1"], flush=True)
        y, p = self._decode(h1, skips, x)
        return y, p, {"h0": h0, "h1": h1}

    def _named(self):
        for n in ("enc0", "enc1", "enc2", "enc3", "gru0", "gru1"):
            yield n, getattr(self, n)
        yield "bott.fc", self.bott.fc
        for n in ("dec3", "dec2", "dec1", "dec0"):
            yield n, getattr(self, n)
        yield "p_head", self.p_head

    def macs_per_frame(self) -> int:
        f_in = {"enc0": 512, "enc1": 128, "enc2": 64, "enc3": 32, "dec3": 16, "dec2": 32, "dec1": 64, "dec0": 128}
        total = 0
        for n, m in self._named():
            if isinstance(m, FreqConv):
                total += m.macs(f_in[n])
            elif isinstance(m, GRULayer):
                total += m.macs()
            else:
                total += linear_macs(m)
        return total
