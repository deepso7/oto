"""GPU-side dynamic acoustic mixer (all ops batched torch, run on the training device).

Inputs per batch (all 48 kHz, ``(B, T)`` unless noted):

* ``speech``, ``speech_fmax (B,)``: clean target speech and its effective bandwidth.
* ``noise``: noise / music (already concatenated or looped to ``T``).
* ``rir_near (B, L)``, ``rir_far (B, L)``, ``direct_near (B,)``, ``direct_far (B,)``: paired RIRs
  from the same room with geometric direct-path sample indices.
* Optional ``interferer (B, T)``: second-talker / TV speech for Isolate (Phase 3).

Recipe (Clean, plan 0004 Decision 8):

1. Reverberate the user with the full RIR; the **target** is the same speech convolved with the
   direct path + early reflections (``early_ms`` after the direct peak, half-Hann fade) and
   time-aligned (both shifted so the direct peak lands on sample 0). Dereverb is learned via
   the target.
2. Add noise at a random SNR; random gain.
3. Device simulation on the *input only*: random smooth EQ, mic self-noise, soft/hard clipping,
   and HFP/codec bandwidth limiting (8 / 16 / 24 kHz). If the input is band-limited, the loss
   ``f_max`` is lowered too so the model is never asked to hallucinate missing bands.

Phase 3 (Isolate) hooks, all off by default (``p_interferer = 0``):

* interferers rendered through the *far* RIR, low-passed and loudspeaker-coloured;
* user distance decoupled from interferer distance (``p_user_far``: user also through the far RIR);
* background-first clips (user starts after ``0.5-3 s`` of interferer only) and background-only
  clips (target is all zeros -> exercises the silent-target loss).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

SR = 48_000


@dataclass
class MixConfig:
    snr_db: tuple[float, float] = (-5.0, 30.0)
    p_noise: float = 0.95
    gain_db: tuple[float, float] = (-35.0, -5.0)  # target RMS of the mixture in dBFS
    p_reverb: float = 0.8
    early_ms: float = 50.0
    # device simulation
    p_eq: float = 0.5
    eq_db: float = 8.0
    p_clip: float = 0.1
    p_bandlimit: float = 0.15
    bandlimits: tuple[float, ...] = (4_000.0, 8_000.0, 12_000.0)  # f_max after 8/16/24 kHz codecs
    p_self_noise: float = 0.5
    self_noise_dbfs: tuple[float, float] = (-90.0, -65.0)
    # Phase 3 (Isolate) hooks
    p_interferer: float = 0.0
    sir_db: tuple[float, float] = (0.0, 15.0)
    interferer_lowpass: tuple[float, float] = (3_000.0, 12_000.0)
    p_user_far: float = 0.0
    p_background_first: float = 0.0
    p_background_only: float = 0.0


def fft_conv(x: Tensor, h: Tensor) -> Tensor:
    """Batched linear convolution truncated to ``x``'s length: ``(B, T) * (B, L) -> (B, T)``."""
    t = x.shape[-1]
    n = 1 << math.ceil(math.log2(t + h.shape[-1] - 1))
    y = torch.fft.irfft(torch.fft.rfft(x.float(), n) * torch.fft.rfft(h.float(), n), n)
    return y[..., :t]


def shift_left(h: Tensor, idx: Tensor) -> Tensor:
    """Shift each RIR left by ``idx`` samples (direct peak -> sample 0), zero-filling the tail."""
    L = h.shape[-1]
    ar = torch.arange(L, device=h.device)[None, :] + idx[:, None].clamp(0, L - 1)
    valid = ar < L
    return torch.gather(h, 1, ar.clamp(max=L - 1)) * valid


def early_rir(h_aligned: Tensor, early_ms: float, fade_ms: float = 5.0) -> Tensor:
    """Keep direct + early reflections of an aligned RIR with a half-Hann fade-out."""
    L = h_aligned.shape[-1]
    n_e, n_f = int(early_ms * SR / 1000), int(fade_ms * SR / 1000)
    w = torch.zeros(L, device=h_aligned.device)
    w[: min(n_e, L)] = 1.0
    end = min(n_e + n_f, L)
    if end > n_e:
        w[n_e:end] = 0.5 * (1 + torch.cos(torch.linspace(0, math.pi, end - n_e, device=h_aligned.device)))
    return h_aligned * w


def rms(x: Tensor, eps: float = 1e-12) -> Tensor:
    return x.square().mean(-1).add(eps).sqrt()


def lowpass_fft(x: Tensor, cutoff: Tensor) -> Tensor:
    """Brick-wall-ish low-pass (cosine roll-off over 500 Hz) per batch item; ``cutoff (B,)`` Hz."""
    n = x.shape[-1]
    X = torch.fft.rfft(x.float(), dim=-1)
    f = torch.fft.rfftfreq(n, 1.0 / SR).to(x.device)[None, :]
    c = cutoff[:, None]
    g = torch.clamp((c + 250 - f) / 500, 0, 1)
    g = 0.5 - 0.5 * torch.cos(math.pi * g)
    return torch.fft.irfft(X * g, n=n, dim=-1)


def random_eq(x: Tensor, max_db: float, gen: torch.Generator | None = None) -> Tensor:
    """Smooth random EQ: 8 log-spaced control points, +-max_db, interpolated on a log-f grid."""
    b, n = x.shape
    X = torch.fft.rfft(x.float(), dim=-1)
    f = torch.fft.rfftfreq(n, 1.0 / SR).to(x.device).clamp_min(20.0)
    logf = torch.log10(f)
    pts = torch.linspace(math.log10(50), math.log10(20_000), 8, device=x.device)
    db = (torch.rand(b, 8, generator=gen, device=x.device) * 2 - 1) * max_db
    idx = torch.bucketize(logf, pts).clamp(1, 7)
    t = ((logf - pts[idx - 1]) / (pts[idx] - pts[idx - 1])).clamp(0, 1)
    curve = db[:, idx - 1] * (1 - t) + db[:, idx] * t
    return torch.fft.irfft(X * (10 ** (curve / 20)), n=n, dim=-1)


class Mixer:
    def __init__(self, cfg: MixConfig | None = None, seed: int = 0, device="cpu"):
        self.cfg = cfg or MixConfig()
        self.device = torch.device(device)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)

    def _u(self, b: int, lo: float, hi: float) -> Tensor:
        return lo + (hi - lo) * torch.rand(b, generator=self.gen, device=self.device)

    def _p(self, b: int, p: float) -> Tensor:
        return torch.rand(b, generator=self.gen, device=self.device) < p

    @torch.no_grad()
    def __call__(
        self,
        speech: Tensor,
        speech_fmax: Tensor,
        noise: Tensor | None = None,
        rir_near: Tensor | None = None,
        direct_near: Tensor | None = None,
        rir_far: Tensor | None = None,
        direct_far: Tensor | None = None,
        interferer: Tensor | None = None,
    ) -> dict[str, Tensor]:
        cfg = self.cfg
        dev = self.device
        speech = speech.to(dev).float()
        b, t = speech.shape
        f_max = speech_fmax.to(dev).float().clone()

        # -- user speech: reverb + dereverb target ------------------------------------
        user_rir, user_direct = rir_near, direct_near
        if rir_far is not None and cfg.p_user_far > 0:
            far_user = self._p(b, cfg.p_user_far)[:, None]
            user_rir = torch.where(far_user, rir_far, rir_near)
            user_direct = torch.where(far_user[:, 0], direct_far, direct_near)
        if user_rir is not None:
            h = shift_left(user_rir.to(dev).float(), user_direct.to(dev).long())
            use = self._p(b, cfg.p_reverb)[:, None]
            dirac = torch.zeros_like(h)
            dirac[:, 0] = 1.0
            h_full = torch.where(use, h, dirac)
            h_early = torch.where(use, early_rir(h, cfg.early_ms), dirac)
            # Normalise so the target's direct path keeps the dry speech level.
            norm = h_early[:, :1].abs().clamp_min(1e-4)
            reverberant = fft_conv(speech, h_full / norm)
            target = fft_conv(speech, h_early / norm)
        else:
            reverberant, target = speech, speech.clone()

        # -- background-first / background-only (Phase 3) ------------------------------
        if cfg.p_background_first > 0:
            bf = self._p(b, cfg.p_background_first)
            start = (self._u(b, 0.5, 3.0) * SR).long() * bf
            gate = (torch.arange(t, device=dev)[None, :] >= start[:, None]).float()
            reverberant, target = reverberant * gate, target * gate
        if cfg.p_background_only > 0:
            keep = (~self._p(b, cfg.p_background_only)).float()[:, None]
            reverberant, target = reverberant * keep, target * keep

        mix = reverberant.clone()
        ref = rms(reverberant).clamp_min(1e-5)

        # -- interferers (Phase 3): far RIR + low-pass + loudspeaker colour -------------------
        if interferer is not None and cfg.p_interferer > 0:
            itf = interferer.to(dev).float()
            if rir_far is not None:
                itf = fft_conv(itf, shift_left(rir_far.to(dev).float(), direct_far.to(dev).long()))
            itf = lowpass_fft(itf, self._u(b, *cfg.interferer_lowpass))
            itf = random_eq(itf, 6.0, self.gen)
            r = rms(itf)[:, None].clamp_min(1e-6)
            k = self._u(b, 0.2, 1.0)[:, None]  # loudspeaker / TV dynamic-range compression
            itf = torch.tanh(k * itf / r) / k * r
            sir = self._u(b, *cfg.sir_db)
            gain = ref / rms(itf).clamp_min(1e-6) * 10 ** (-sir / 20)
            # Background-only clips have no user: set interferer level against a nominal -30 dBFS.
            gain = torch.where(rms(reverberant) < 1e-5, 0.0316 / rms(itf).clamp_min(1e-6), gain)
            mix = mix + itf * (gain * self._p(b, cfg.p_interferer))[:, None]

        # -- noise -----------------------------------------------------------------
        if noise is not None:
            nz = noise.to(dev).float()
            snr = self._u(b, *cfg.snr_db)
            gain = ref / rms(nz).clamp_min(1e-6) * 10 ** (-snr / 20)
            gain = torch.where(rms(reverberant) < 1e-5, 0.0316 / rms(nz).clamp_min(1e-6), gain)
            mix = mix + nz * (gain * self._p(b, cfg.p_noise))[:, None]

        # -- device simulation (input only) ---------------------------------------------
        if cfg.p_eq > 0:
            eq = self._p(b, cfg.p_eq)[:, None]
            mix = torch.where(eq, random_eq(mix, cfg.eq_db, self.gen), mix)
        if cfg.p_bandlimit > 0:
            bl = self._p(b, cfg.p_bandlimit)
            choices = torch.tensor(cfg.bandlimits, device=dev)
            cut = choices[torch.randint(len(choices), (b,), generator=self.gen, device=dev)]
            mix = torch.where(bl[:, None], lowpass_fft(mix, cut), mix)
            f_max = torch.where(bl, torch.minimum(f_max, cut), f_max)

        # -- level, self-noise, clipping ---------------------------------------------------
        level = 10 ** (self._u(b, *cfg.gain_db) / 20)
        g = level / rms(mix).clamp_min(1e-6)
        mix, target = mix * g[:, None], target * g[:, None]
        if cfg.p_self_noise > 0:
            sn = 10 ** (self._u(b, *cfg.self_noise_dbfs) / 20) * self._p(b, cfg.p_self_noise)
            mix = mix + torch.randn(b, t, generator=self.gen, device=dev) * sn[:, None]
        if cfg.p_clip > 0:
            clip = self._p(b, cfg.p_clip)[:, None]
            thr = self._u(b, 0.1, 0.6)[:, None]
            mix = torch.where(clip, torch.clamp(mix, -thr, thr), mix)
        mix = mix.clamp(-1.0, 1.0)
        return {"mix": mix, "target": target, "f_max": f_max}
