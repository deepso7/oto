"""Signal metrics in pure Python (numpy is used when installed, for long files only).

Conventions: signals are sequences of floats at a known rate. Windows are (start_s, end_s)
in the *input's* timeline; a processed signal delayed by `delay` samples is read at
`start + delay`. Levels are dBFS RMS (a full-scale sine is −3 dBFS).
"""

from __future__ import annotations

import math

try:  # optional, only used to speed up long manifest files
    import numpy as _np
except ImportError:  # pragma: no cover - depends on the environment
    _np = None

FLOOR_DB = -200.0
_NUMPY_MIN = 1 << 18


def _dot(a, b) -> float:
    n = min(len(a), len(b))
    if _np is not None and n >= _NUMPY_MIN:
        return float(_np.dot(_np.asarray(a[:n], dtype=_np.float64), _np.asarray(b[:n], dtype=_np.float64)))
    return math.fsum(a[i] * b[i] for i in range(n))


def energy(x) -> float:
    return _dot(x, x)


def to_db(power_ratio: float) -> float:
    return 10.0 * math.log10(power_ratio) if power_ratio > 0 else FLOOR_DB


def rms(x) -> float:
    return math.sqrt(energy(x) / len(x)) if len(x) else 0.0


def rms_dbfs(x) -> float:
    r = rms(x)
    return 20.0 * math.log10(r) if r > 0 else FLOOR_DB


def peak_dbfs(x) -> float:
    finite = [abs(s) for s in x if math.isfinite(s)]
    p = max(finite, default=0.0)
    return 20.0 * math.log10(p) if p > 0 else FLOOR_DB


def nan_count(x) -> int:
    return sum(1 for s in x if not math.isfinite(s))


def clip_count(x, threshold: float = 0.999) -> int:
    """Samples at or beyond full scale (NaN/inf excluded; those are counted separately)."""
    return sum(1 for s in x if math.isfinite(s) and abs(s) >= threshold)


def snr_db(ref, est) -> float:
    """Plain SNR of `est` against an already aligned `ref`."""
    n = min(len(ref), len(est))
    err = [est[i] - ref[i] for i in range(n)]
    e = energy(err)
    return to_db(energy(ref[:n]) / e) if e > 0 else math.inf


def si_sdr(ref, est) -> float:
    """Scale-invariant SDR (Le Roux et al. 2019), means removed, on aligned signals."""
    n = min(len(ref), len(est))
    if n == 0:
        return FLOOR_DB
    mr = math.fsum(ref[:n]) / n
    me = math.fsum(est[:n]) / n
    r = [ref[i] - mr for i in range(n)]
    e = [est[i] - me for i in range(n)]
    rr = energy(r)
    if rr == 0:
        return FLOOR_DB
    alpha = _dot(e, r) / rr
    target = [alpha * v for v in r]
    tt = energy(target)
    if tt == 0:
        return FLOOR_DB
    noise = [e[i] - target[i] for i in range(n)]
    nn = energy(noise)
    return to_db(tt / nn) if nn > 0 else math.inf


def shift(x, delay: int, length: int | None = None) -> list[float]:
    """`x` advanced by `delay` samples (undoing a processing delay), zero-padded to `length`."""
    length = len(x) - delay if length is None else length
    out = list(x[delay : delay + length]) if delay >= 0 else [0.0] * (-delay) + list(x[: length + delay])
    return out + [0.0] * (length - len(out))


def window_samples(x, rate: int, windows, delay: int = 0) -> list[float]:
    """Concatenated samples of `x` inside `windows` (seconds), read `delay` samples later.
    Windows are clamped to the signal's length."""
    out: list[float] = []
    for a, b in windows:
        i0 = max(0, int(a * rate) + delay)
        i1 = min(len(x), int(b * rate) + delay)
        if i1 > i0:
            out.extend(x[i0:i1])
    return out


def window_level_db(x, rate: int, windows, delay: int = 0) -> float:
    return rms_dbfs(window_samples(x, rate, windows, delay))


def attenuation_db(inp, in_rate: int, out, out_rate: int, windows, delay: int) -> float:
    """How much quieter `out` is than `inp` inside `windows` (positive = quieter)."""
    return window_level_db(inp, in_rate, windows) - window_level_db(out, out_rate, windows, delay)


def frame_energies(x, rate: int, frame_ms: float = 10.0) -> list[float]:
    hop = max(1, int(rate * frame_ms / 1000))
    return [energy(x[i : i + hop]) / hop for i in range(0, len(x) - hop + 1, hop)]


def envelope_delay(ref, ref_rate: int, out, out_rate: int, max_delay_s: float = 0.25, frame_ms: float = 2.0) -> float:
    """Coarse delay of `out` behind `ref` in seconds, from the cross-correlation of their
    log-energy envelopes. Works across sample rates and survives denoising."""
    def env(x, rate):
        return [math.log10(e + 1e-10) for e in frame_energies(x, rate, frame_ms)]

    a, b = env(ref, ref_rate), env(out, out_rate)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    ma, mb = math.fsum(a) / n, math.fsum(b) / n
    a = [v - ma for v in a]
    b = [v - mb for v in b]
    best, best_lag = -math.inf, 0
    for lag in range(0, int(max_delay_s * 1000 / frame_ms) + 1):
        m = n - lag
        if m <= 0:
            break
        c = math.fsum(a[i] * b[i + lag] for i in range(m)) / m
        if c > best:
            best, best_lag = c, lag
    return best_lag * frame_ms / 1000


def impulse_delays(out, impulses, max_delay: int) -> list[int]:
    """For each input impulse position, where the largest output sample falls after it."""
    delays = []
    for p in impulses:
        seg = out[p : p + max_delay + 1]
        if not seg:
            continue
        mags = [abs(s) if math.isfinite(s) else 0.0 for s in seg]
        delays.append(max(range(len(mags)), key=mags.__getitem__))
    return delays


def step_delay(out, step: int, max_delay: int) -> int | None:
    """Where the largest first difference falls after an input step."""
    seg = out[max(0, step - 1) : step + max_delay + 1]
    if len(seg) < 2:
        return None
    d = [abs(seg[i + 1] - seg[i]) if math.isfinite(seg[i + 1] - seg[i]) else 0.0 for i in range(len(seg) - 1)]
    return max(range(len(d)), key=d.__getitem__)


def median(values):
    v = sorted(values)
    if not v:
        return None
    m = len(v) // 2
    return v[m] if len(v) % 2 else (v[m - 1] + v[m]) / 2


def resample_linear(x, src_rate: int, dst_rate: int) -> list[float]:
    """Linear-interpolation resampler. For level and envelope metrics only, not for quality
    scores (use scipy's resample_poly for PESQ)."""
    if src_rate == dst_rate:
        return list(x)
    n = int(len(x) * dst_rate / src_rate)
    step = src_rate / dst_rate
    out = []
    for i in range(n):
        t = i * step
        j = int(t)
        f = t - j
        a = x[j] if j < len(x) else 0.0
        b = x[j + 1] if j + 1 < len(x) else a
        out.append(a + (b - a) * f)
    return out
