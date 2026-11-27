"""Effective-bandwidth estimation (Hi-Fi TTS / URGENT method).

The effective bandwidth ``f_max`` is the highest frequency whose *mean* power spectrum
is within ``threshold_db`` (default ``-50 dB``) of the spectrum's peak. Container sample
rate is not trusted. The packer stores ``f_max`` per clip; the losses mask bins above
``0.9 * f_max`` so band-limited clips never teach the model to muffle fullband audio.

Silence and DC/rumble are excluded: only frames within ``active_db`` of the loudest frame
contribute, and bins below ``min_hz`` are ignored when locating the peak.
"""

from __future__ import annotations

import numpy as np

URGENT_RATES = (8000, 16000, 22050, 24000, 32000, 44100, 48000)


def mean_power_spectrum(
    x: np.ndarray, sr: int, n_fft: int = 2048, hop: int = 512, active_db: float = 40.0
) -> tuple[np.ndarray, np.ndarray]:
    """Mean power spectrum over active frames. Returns ``(freqs_hz, power)``."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=-1) if x.shape[-1] <= 8 else x.mean(axis=0)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx] * np.hanning(n_fft)[None, :]
    energy = (frames**2).mean(axis=1)
    if energy.max() <= 0:
        return np.fft.rfftfreq(n_fft, 1.0 / sr), np.zeros(n_fft // 2 + 1)
    active = energy >= energy.max() * 10 ** (-active_db / 10)
    spec = np.abs(np.fft.rfft(frames[active], axis=1)) ** 2
    return np.fft.rfftfreq(n_fft, 1.0 / sr), spec.mean(axis=0)


def estimate_bandwidth(
    x: np.ndarray, sr: int, threshold_db: float = -50.0, n_fft: int = 2048, min_hz: float = 50.0
) -> float:
    """Effective bandwidth in Hz (0.0 for digital silence)."""
    freqs, p = mean_power_spectrum(x, sr, n_fft)
    if p.max() <= 0:
        return 0.0
    valid = freqs >= min_hz
    peak = p[valid].max()
    above = np.nonzero(valid & (p >= peak * 10 ** (threshold_db / 10)))[0]
    return float(freqs[above[-1]]) if len(above) else 0.0


def urgent_target_rate(f_max: float) -> int:
    """Lowest URGENT sample rate whose Nyquist covers ``f_max``."""
    for r in URGENT_RATES:
        if r / 2 >= f_max:
            return r
    return URGENT_RATES[-1]


def band_energy_ratio_db(x: np.ndarray, sr: int, lo_hz: float, hi_hz: float, ref: tuple = (300.0, 4000.0)) -> float:
    """Mean power in ``[lo, hi)`` relative to the speech band ``ref`` (dB). Used by MP3 spot checks."""
    freqs, p = mean_power_spectrum(x, sr)
    band = p[(freqs >= lo_hz) & (freqs < hi_hz)]
    refp = p[(freqs >= ref[0]) & (freqs < ref[1])]
    if band.size == 0 or refp.mean() <= 0:
        return float("-inf")
    return float(10 * np.log10(band.mean() / refp.mean() + 1e-30))
