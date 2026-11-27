"""The op set of ``crates/oto-dsp/src/oto1/SPEC.md`` section 3, in batch-friendly PyTorch.

Every layer here has a 1:1 counterpart in the Rust kernel (``layers.rs``):

* :class:`FreqConv` -- frequency-only conv (time kernel 1) on ``(N, C, F)`` with ``N = B*T``, i.e.
  ``nn.Conv2d(Ci, Co, (1, K), stride=(1, s), padding=(0, p))`` on ``[B, C, T, F]``; optional
  ``BatchNorm1d`` (folded into weight and bias before export) and SiLU.
* :class:`GRULayer` -- one GRU layer split into ``G`` independent groups (block-diagonal weights,
  PyTorch gate order ``r, z, n``). ``G = 1`` is a plain ``nn.GRU``.
* :func:`group_shuffle`, :func:`subpixel`, :func:`comp`, ``nn.Linear``, sigmoid / tanh / SiLU.

Activations are ``(N, C, F)`` here; the kernel stores ``[F][C]``, so every flatten into a linear
or GRU input goes through :func:`flat_fc` (index ``f * C + c``) and back through :func:`unflat_fc`.
Because convolutions never look across time, the only temporal state is the GRU hidden state
(plus the deep-filter history in Candidate A), so batch and streaming agree up to float rounding.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .stft import FLUSH


class FreqConv(nn.Module):
    """``conv(Co, Ci, K, stride, pad)`` with zero padding ``pad`` on both sides of frequency."""

    def __init__(self, cin: int, cout: int, kernel: int, stride: int = 1, pad: int | None = None,
                 bn: bool = True, act: bool = True):  # fmt: skip
        super().__init__()
        self.kernel, self.stride = kernel, stride
        self.pad = kernel // 2 if pad is None else pad
        self.conv = nn.Conv1d(cin, cout, kernel, stride=stride, padding=self.pad, bias=True)
        self.bn: nn.Module = nn.BatchNorm1d(cout) if bn else nn.Identity()
        self.act: nn.Module = nn.SiLU() if act else nn.Identity()
        self.folded = not bn

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))

    def out_len(self, f_in: int) -> int:
        return (f_in + 2 * self.pad - self.kernel) // self.stride + 1

    def macs(self, f_in: int) -> int:
        return self.conv.in_channels * self.conv.out_channels * self.kernel * self.out_len(f_in)

    @torch.no_grad()
    def fold_batchnorm(self) -> None:
        """``W' = W*g/sqrt(var+eps)``, ``b' = (b-mean)*g/sqrt(var+eps) + beta`` (eval statistics)."""
        if self.folded:
            return
        bn = self.bn
        assert isinstance(bn, nn.BatchNorm1d)
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        self.conv.weight.mul_(scale[:, None, None])
        self.conv.bias.copy_((self.conv.bias - bn.running_mean) * scale + bn.bias)
        self.bn = nn.Identity()
        self.folded = True

    def tensors(self) -> dict[str, Tensor]:
        assert self.folded, "fold_batchnorm() before export"
        return {"weight": self.conv.weight, "bias": self.conv.bias}

    @torch.no_grad()
    def load(self, t: dict[str, Tensor]) -> None:
        assert self.folded, "fold_batchnorm() before loading exported tensors"
        w = t["weight"]
        if w.dim() == 4:  # [Co, Ci, 1, K] (Conv2d with a (1, K) kernel)
            assert w.shape[2] == 1, w.shape
            w = w[:, :, 0]
        self.conv.weight.copy_(w)
        self.conv.bias.copy_(t["bias"])


class GRULayer(nn.Module):
    """``gru(I, H, G)``: ``G`` independent ``nn.GRU(I/G, H/G)`` cells.

    Group ``g`` reads ``x[..., g*I/G:(g+1)*I/G]`` and owns ``h[..., g*H/G:(g+1)*H/G]``. Exported as
    ``weight_ih [G, 3H/G, I/G]``, ``weight_hh [G, 3H/G, H/G]``, ``bias_ih / bias_hh [G, 3H/G]``, or
    without the leading ``G`` when ``G = 1`` (plain ``nn.GRU``, ``_l0`` suffix dropped).
    """

    def __init__(self, input_size: int, hidden: int, groups: int = 1):
        super().__init__()
        assert input_size % groups == 0 and hidden % groups == 0
        self.input_size, self.hidden, self.groups = input_size, hidden, groups
        self.cells = nn.ModuleList(nn.GRU(input_size // groups, hidden // groups, batch_first=True)
                                   for _ in range(groups))  # fmt: skip

    def forward(self, x: Tensor, h0: Tensor | None = None, flush: bool = False) -> tuple[Tensor, Tensor]:
        """``x``: ``(B, T, I)``, ``h0``: ``(B, H)``. Returns ``(y (B, T, H), h_T (B, H))``.

        ``flush`` sets ``|h| < 1e-15`` to 0 (the kernel does this every step; streaming uses it).
        """
        hg = self.hidden // self.groups
        xs = x.chunk(self.groups, dim=-1)
        ys, hs = [], []
        for g, cell in enumerate(self.cells):
            h = None if h0 is None else h0[:, g * hg : (g + 1) * hg].unsqueeze(0).contiguous().to(xs[g].dtype)
            y, hn = cell(xs[g].contiguous(), h)
            ys.append(y)
            hs.append(hn[0])
        y, h = torch.cat(ys, -1), torch.cat(hs, -1)
        if flush:
            y = torch.where(y.abs() < FLUSH, torch.zeros_like(y), y)
            h = torch.where(h.abs() < FLUSH, torch.zeros_like(h), h)
        return y, h

    def macs(self) -> int:
        return 3 * (self.hidden // self.groups) * (self.input_size + self.hidden)

    _PARTS = ("weight_ih", "weight_hh", "bias_ih", "bias_hh")

    def tensors(self) -> dict[str, Tensor]:
        out = {p: torch.stack([getattr(c, f"{p}_l0") for c in self.cells], 0) for p in self._PARTS}
        if self.groups == 1:
            out = {k: v[0] for k, v in out.items()}
        return out

    @torch.no_grad()
    def load(self, t: dict[str, Tensor]) -> None:
        for p in self._PARTS:
            v = t[p] if self.groups > 1 else t[p][None]
            for g, c in enumerate(self.cells):
                getattr(c, f"{p}_l0").copy_(v[g])


def group_shuffle(h: Tensor, groups: int) -> Tensor:
    """DFN's grouped-GRU shuffle on the last axis: ``h.view(H/G, G).transpose(0, 1).reshape(H)``."""
    hd = h.shape[-1]
    return h.unflatten(-1, (hd // groups, groups)).transpose(-1, -2).flatten(-2)


def subpixel(y: Tensor, r: int) -> Tensor:
    """``(N, r*C', F) -> (N, C', F*r)``: channel ``j*C' + c`` at ``f`` becomes channel ``c`` at ``f*r + j``."""
    n, ch, f = y.shape
    return y.view(n, r, ch // r, f).permute(0, 2, 3, 1).reshape(n, ch // r, f * r)


def flat_fc(x: Tensor) -> Tensor:
    """``(N, C, F) -> (N, F*C)`` in the kernel's ``[F][C]`` order (index ``f*C + c``)."""
    return x.transpose(1, 2).flatten(1)


def unflat_fc(x: Tensor, c: int) -> Tensor:
    """Inverse of :func:`flat_fc`: ``(N, F*C) -> (N, C, F)``."""
    return x.view(x.shape[0], -1, c).transpose(1, 2)


def comp(spec_ri: Tensor) -> Tensor:
    """``X * (|X|^2 + 1e-12)^(-0.35)`` on ``(..., 2)`` (magnitude ``|X|^0.3``, phase kept)."""
    p = spec_ri.square().sum(-1, keepdim=True) + 1e-12
    return spec_ri * p.pow(-0.35)


def complex_mul(a: Tensor, b: Tensor) -> Tensor:
    ar, ai = a.unbind(-1)
    br, bi = b.unbind(-1)
    return torch.stack([ar * br - ai * bi, ar * bi + ai * br], dim=-1)


def linear_macs(lin: nn.Linear) -> int:
    return lin.in_features * lin.out_features
