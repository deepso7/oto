"""Common batch/streaming plumbing for both Oto-1 candidates."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
from torch import Tensor, nn

from . import stft as S
from .layers import FreqConv, GRULayer  # noqa: F401


class OtoModel(nn.Module):
    """Base class.

    Subclasses implement:

    * ``forward_spec(X) -> (Y, p_logit)`` on ``(B, T, 513, 2)`` spectra (training path).
    * ``init_state(batch, device)`` and ``step_spec(X, state) -> (Y, p_logit, state)`` on one
      ``(B, 513, 2)`` frame (streaming path, mirrors the Rust kernel).
    * ``export_tensors()`` / ``load_tensors()`` / ``macs_per_frame()``.

    The config dataclass holds exactly the SPEC.md config keys, so ``asdict(cfg)`` plus ``arch``
    is the ``.otow`` config JSON.
    """

    arch_id: int = 0
    arch_name: str = ""

    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("window", S.sqrt_hann(), persistent=False)

    # -- batch ---------------------------------------------------------------

    def forward(self, wav: Tensor) -> dict[str, Tensor]:
        """``wav``: ``(B, T)`` at 48 kHz. Returns aligned enhanced waveform, spectra, presence logits."""
        t = wav.shape[-1]
        x = S.stft(wav, self.window)
        y, p_logit = self.forward_spec(x)
        out = S.istft(y, self.window)[..., :t]
        return {"wav": out, "spec": y, "noisy_spec": x, "p_logit": p_logit}

    # -- streaming -----------------------------------------------------------

    def stream(self, batch: int = 1, device=None) -> "Streamer":
        return Streamer(self, batch, device)

    # -- export --------------------------------------------------------------

    @torch.no_grad()
    def fold_batchnorm(self) -> "OtoModel":
        """Fold every BatchNorm into its conv (requires ``eval()`` semantics: running stats)."""
        self.eval()
        for m in self.modules():
            if isinstance(m, FreqConv):
                m.fold_batchnorm()
        return self

    def _named(self):
        """Yield ``(tensor-name prefix, FreqConv | GRULayer | nn.Linear)`` in SPEC.md order."""
        raise NotImplementedError

    def export_tensors(self) -> dict[str, Tensor]:
        """The ``.otow`` tensors (SPEC.md names and shapes, f32). Requires :meth:`fold_batchnorm`."""
        out: dict[str, Tensor] = {}
        for name, m in self._named():
            parts = {"weight": m.weight, "bias": m.bias} if isinstance(m, nn.Linear) else m.tensors()
            for k, v in parts.items():
                out[f"{name}.{k}"] = v.detach().float().contiguous()
        return out

    @torch.no_grad()
    def load_tensors(self, tensors: dict[str, Tensor]) -> "OtoModel":
        """Inverse of :meth:`export_tensors` with the kernel's strictness: every tensor present with
        its exact shape (conv weights may be ``[Co, Ci, K]`` or ``[Co, Ci, 1, K]``), nothing extra."""
        self.fold_batchnorm()
        want = self.export_tensors()
        if set(tensors) != set(want):
            raise KeyError(f"missing {sorted(set(want) - set(tensors))}, unexpected {sorted(set(tensors) - set(want))}")
        convs = {n for n, m in self._named() if isinstance(m, FreqConv)}
        for k, v in tensors.items():
            shape = tuple(v.shape)
            ok = shape == tuple(want[k].shape)
            if not ok and k.endswith(".weight") and k[: -len(".weight")] in convs and len(shape) == 4:
                co, ci, kk = want[k].shape
                ok = shape == (co, ci, 1, kk)
            if not ok:
                raise ValueError(f"{k}: shape {shape}, expected {tuple(want[k].shape)}")
        for name, m in self._named():
            parts = {k.split(".")[-1]: torch.as_tensor(v).float() for k, v in tensors.items()
                     if k.rsplit(".", 1)[0] == name}  # fmt: skip
            if isinstance(m, nn.Linear):
                m.weight.copy_(parts["weight"])
                m.bias.copy_(parts["bias"])
            else:
                m.load(parts)
        return self

    def config_dict(self) -> dict:
        """The flat ``.otow`` config of SPEC.md (``arch`` plus the candidate's integer keys)."""
        return {"arch": self.arch_name, **asdict(self.cfg)}

    def exported_param_count(self) -> int:
        """Parameters in the ``.otow`` (BatchNorm folded); counted without folding ``self``."""
        n = sum(p.numel() for p in self.parameters())
        return n - sum(p.numel() for m in self.modules() if isinstance(m, nn.BatchNorm1d) for p in m.parameters())

    def macs_per_second(self) -> float:
        return self.macs_per_frame() * S.FRAMES_PER_SECOND

    def summary(self) -> str:
        n_train = sum(p.numel() for p in self.parameters())
        lines = [
            f"{self.arch_name} (arch_id={self.arch_id})",
            f"  exported params (BN folded): {self.exported_param_count():,}",
            f"  trainable params (with BatchNorm): {n_train:,}",
            f"  MACs/frame: {self.macs_per_frame():,}",
            f"  MAC/s: {self.macs_per_second() / 1e6:.2f} M ({S.FRAMES_PER_SECOND} frames/s)",
        ]
        return "\n".join(lines)


class Streamer:
    """Waveform-in / waveform-out 512-sample block processor (exact Rust state machine)."""

    def __init__(self, model: OtoModel, batch: int = 1, device=None):
        self.model = model
        self.stft = S.StreamingSTFT((batch,), device=device)
        self.state = model.init_state(batch, device)

    @torch.no_grad()
    def process(self, block: Tensor) -> tuple[Tensor, Tensor]:
        """``block``: ``(B, 512)``. Returns ``(out_block (B, 512), p_logit (B,))``.

        ``out_block`` is the reconstruction of the *previous* input block (512-sample delay).
        """
        x = self.stft.analyze(block)
        y, p, self.state = self.model.step_spec(x, self.state)
        return self.stft.synthesize(y), p

    @torch.no_grad()
    def run(self, wav: Tensor) -> tuple[Tensor, Tensor]:
        """Stream a whole ``(B, T)`` signal (``T`` multiple of 512). Returns delayed output + logits."""
        outs, ps = [], []
        for blk in wav.split(S.HOP, dim=-1):
            o, p = self.process(blk)
            outs.append(o)
            ps.append(p)
        return torch.cat(outs, -1), torch.stack(ps, -1)
