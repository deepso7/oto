"""Candidate A (``arch_id = 1``, ``"oto1-a"``): DeepFilterNet-style hybrid. SPEC.md section 4.

Per frame (``X``: ``(513, 2)`` noisy spectrum, ``P = |X|^2``):

* Features: ``erb_feat[b] = log10(mean_{k in band b} P[k] + 1e-10)`` over the 32 pinned ERB bands,
  ``df_feat[k] = comp(X[k])`` for ``k < 96``.
* Encoders: ``enc.erb{0,1,2}`` (32 -> 16 -> 8 rows), ``enc.df{0,1,2,3}`` (96 -> 48 -> 24 -> 8 rows,
  the last with stride 3, pad 0); ``emb = e2 + d3`` flattened ``f*C + c``.
* ``gru0`` (grouped, I = 8C), ``group_shuffle``, ``gru1`` (grouped, I = H).
* ERB decoder: ``erb_dec.fc`` + SiLU ``+ e2``, ``erb_dec.up2`` (C -> 2C, sub-pixel r=2) ``+ e1``,
  ``erb_dec.up1`` ``+ e0``, ``erb_dec.out`` -> sigmoid gains ``[32]``.
* DF decoder: ``df_dec.fc0`` + SiLU, ``coef = tanh(df_dec.fc1(q) + df_dec.convp(d0))`` ``[96][6]``.
* ``p_head``: target-presence logit.
* Output: ``Y[k] = sum_i coef_i[k] * X_{t-i}[k]`` for ``k < 96``, ``gains[band(k)] * X[k]`` above.

SiLU follows every conv and linear unless noted; every conv has a foldable BatchNorm except
``erb_dec.out`` and ``df_dec.convp``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from . import stft as S
from .base import OtoModel
from .layers import FreqConv, GRULayer, comp, complex_mul, flat_fc, group_shuffle, linear_macs, subpixel, unflat_fc

N_ERB = 32
N_DF = 96
DF_ORDER = 3
F_EMB = 8


@dataclass
class CandidateAConfig:
    """Exactly the ``.otow`` config keys (``arch`` aside). ``n_erb``, ``n_df_bins``, ``df_order`` are fixed."""

    channels: int = 48
    gru_hidden: int = 192
    gru_groups: int = 4
    n_erb: int = N_ERB
    n_df_bins: int = N_DF
    df_order: int = DF_ORDER

    def __post_init__(self):
        if (self.n_erb, self.n_df_bins, self.df_order) != (N_ERB, N_DF, DF_ORDER):
            raise ValueError("Oto-1 A: n_erb, n_df_bins, df_order are fixed at 32, 96, 3")
        g = self.gru_groups
        if not (self.channels > 0 and g > 0 and self.gru_hidden % g == 0 and (F_EMB * self.channels) % g == 0):
            raise ValueError(f"Oto-1 A: inconsistent config {self}")


class EncA(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.erb0 = FreqConv(1, c, 3, 1, 1)
        self.erb1 = FreqConv(c, c, 3, 2, 1)
        self.erb2 = FreqConv(c, c, 3, 2, 1)
        self.df0 = FreqConv(2, c, 3, 1, 1)
        self.df1 = FreqConv(c, c, 3, 2, 1)
        self.df2 = FreqConv(c, c, 3, 2, 1)
        self.df3 = FreqConv(c, c, 3, 3, 0)


class ErbDec(nn.Module):
    def __init__(self, c: int, h: int):
        super().__init__()
        self.fc = nn.Linear(h, F_EMB * c)
        self.up2 = FreqConv(c, 2 * c, 3, 1, 1)
        self.up1 = FreqConv(c, 2 * c, 3, 1, 1)
        self.out = FreqConv(c, 1, 3, 1, 1, bn=False, act=False)


class DfDec(nn.Module):
    def __init__(self, c: int, h: int):
        super().__init__()
        self.fc0 = nn.Linear(h, F_EMB * c)
        self.fc1 = nn.Linear(F_EMB * c, N_DF * DF_ORDER * 2)
        self.convp = FreqConv(c, DF_ORDER * 2, 1, 1, 0, bn=False, act=False)


class CandidateA(OtoModel):
    arch_id = 1
    arch_name = "oto1-a"

    def __init__(self, cfg: CandidateAConfig | None = None):
        cfg = cfg or CandidateAConfig()
        super().__init__(cfg)
        c, h, g = cfg.channels, cfg.gru_hidden, cfg.gru_groups
        self.edges = S.erb_band_edges(N_ERB)
        sel, width, inv = S.erb_matrices(self.edges)
        self.register_buffer("erb_sel", sel, persistent=False)
        self.register_buffer("erb_width", width, persistent=False)
        self.register_buffer("erb_inv", inv, persistent=False)
        self.enc = EncA(c)
        self.gru0 = GRULayer(F_EMB * c, h, g)
        self.gru1 = GRULayer(h, h, g)
        self.erb_dec = ErbDec(c, h)
        self.df_dec = DfDec(c, h)
        self.p_head = nn.Linear(h, 1)

    # -- per-frame (time-flattened) pieces ------------------------------------

    def _encode(self, x: Tensor):
        """``x``: ``(N, 513, 2)`` -> ``emb (N, 8C)`` and the skips ``(e0, e1, e2, d0)``."""
        enc = self.enc
        x = x.float()
        p = x.square().sum(-1)  # (N, 513)
        erb_feat = torch.log10((p @ self.erb_sel) / self.erb_width + 1e-10)  # (N, 32)
        df_feat = comp(x[:, :N_DF]).transpose(1, 2)  # (N, 2, 96)
        e0 = enc.erb0(erb_feat.unsqueeze(1))
        e1 = enc.erb1(e0)
        e2 = enc.erb2(e1)
        d0 = enc.df0(df_feat)
        d3 = enc.df3(enc.df2(enc.df1(d0)))
        return flat_fc(e2 + d3), (e0, e1, e2, d0)

    def _decode(self, h1: Tensor, skips) -> tuple[Tensor, Tensor, Tensor]:
        e0, e1, e2, d0 = skips
        c = self.cfg.channels
        n = h1.shape[0]
        ed = self.erb_dec
        t = unflat_fc(nn.functional.silu(ed.fc(h1)), c) + e2  # (N, C, 8)
        u = subpixel(ed.up2(t), 2) + e1  # (N, C, 16)
        v = subpixel(ed.up1(u), 2) + e0  # (N, C, 32)
        gains = torch.sigmoid(ed.out(v)).squeeze(1)  # (N, 32)

        dd = self.df_dec
        q = nn.functional.silu(dd.fc0(h1))
        coef = dd.fc1(q).view(n, N_DF, 2 * DF_ORDER) + dd.convp(d0).transpose(1, 2)
        coef = torch.tanh(coef).view(n, N_DF, DF_ORDER, 2)  # [k][tap][re, im]
        return gains, coef, self.p_head(h1).squeeze(-1)

    def _filter(self, x: Tensor, gains: Tensor, coef: Tensor, hist: Tensor) -> Tensor:
        """``x (N, 513, 2)``, ``hist (N, 2, 96, 2)`` = ``[X_{t-1}, X_{t-2}]`` low bins."""
        x = x.float()
        y_hi = x[:, N_DF:] * (gains.float() @ self.erb_inv)[:, N_DF:, None]
        taps = torch.cat([x[:, None, :N_DF], hist.float()], dim=1).transpose(1, 2)  # (N, 96, 3, 2)
        y_lo = complex_mul(coef.float(), taps).sum(2)
        return torch.cat([y_lo, y_hi], dim=1)

    # -- batch -----------------------------------------------------------------

    def forward_spec(self, x: Tensor) -> tuple[Tensor, Tensor]:
        b, t = x.shape[:2]
        xf = x.reshape(b * t, S.N_BINS, 2)
        emb, skips = self._encode(xf)
        h0, _ = self.gru0(emb.view(b, t, -1))
        h1, _ = self.gru1(group_shuffle(h0, self.cfg.gru_groups))
        gains, coef, p_logit = self._decode(h1.reshape(b * t, -1), skips)
        low = x[:, :, :N_DF]
        hist = torch.stack([nn.functional.pad(low, (0, 0, 0, 0, i, 0))[:, :t] for i in range(1, DF_ORDER)], dim=2)
        y = self._filter(xf, gains, coef, hist.reshape(b * t, DF_ORDER - 1, N_DF, 2))
        return y.view(b, t, S.N_BINS, 2), p_logit.view(b, t)

    # -- streaming -------------------------------------------------------------

    def init_state(self, batch: int, device=None):
        h = self.cfg.gru_hidden
        z = lambda *s: torch.zeros(*s, device=device)  # noqa: E731
        return {"h0": z(batch, h), "h1": z(batch, h), "hist": z(batch, DF_ORDER - 1, N_DF, 2)}

    def step_spec(self, x: Tensor, state):
        emb, skips = self._encode(x)
        _, h0 = self.gru0(emb[:, None], state["h0"], flush=True)
        _, h1 = self.gru1(group_shuffle(h0, self.cfg.gru_groups)[:, None], state["h1"], flush=True)
        gains, coef, p_logit = self._decode(h1, skips)
        y = self._filter(x, gains, coef, state["hist"])
        hist = torch.cat([x[:, None, :N_DF].float(), state["hist"][:, :-1]], dim=1)
        return y, p_logit, {"h0": h0, "h1": h1, "hist": hist}

    # -- export ----------------------------------------------------------------

    def _named(self):
        """``(name, module)`` for every exported layer, in SPEC.md order."""
        for n in ("erb0", "erb1", "erb2", "df0", "df1", "df2", "df3"):
            yield f"enc.{n}", getattr(self.enc, n)
        yield "gru0", self.gru0
        yield "gru1", self.gru1
        for n in ("fc", "up2", "up1", "out"):
            yield f"erb_dec.{n}", getattr(self.erb_dec, n)
        for n in ("fc0", "fc1", "convp"):
            yield f"df_dec.{n}", getattr(self.df_dec, n)
        yield "p_head", self.p_head

    def macs_per_frame(self) -> int:
        f_in = {"enc.erb0": 32, "enc.erb1": 32, "enc.erb2": 16, "enc.df0": 96, "enc.df1": 96, "enc.df2": 48,
                "enc.df3": 24, "erb_dec.up2": 8, "erb_dec.up1": 16, "erb_dec.out": 32, "df_dec.convp": 96}  # fmt: skip
        total = 0
        for n, m in self._named():
            if isinstance(m, FreqConv):
                total += m.macs(f_in[n])
            elif isinstance(m, GRULayer):
                total += m.macs()
            else:
                total += linear_macs(m)
        return total
