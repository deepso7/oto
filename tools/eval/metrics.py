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
    if n == 0:
        raise ValueError("envelope_delay: signal shorter than one frame")
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


def impulse_delays(out, impulses, max_delay: int, min_amp: float = 0.05) -> list[int]:
    """For each input impulse position, where the largest output sample falls after it.
    Windows whose peak is below `min_amp` are skipped so all-silent output never looks aligned."""
    delays = []
    for p in impulses:
        seg = out[p : p + max_delay + 1]
        if not seg:
            continue
        mags = [abs(s) if math.isfinite(s) else 0.0 for s in seg]
        best_i = max(range(len(mags)), key=mags.__getitem__)
        if mags[best_i] >= min_amp:
            delays.append(best_i)
    return delays


def step_delay(out, step: int, max_delay: int, min_amp: float = 0.05) -> int | None:
    """Where the largest first difference falls after an input step (`None` on silence)."""
    seg = out[max(0, step - 1) : step + max_delay + 1]
    if len(seg) < 2:
        return None
    d = [abs(seg[i + 1] - seg[i]) if math.isfinite(seg[i + 1] - seg[i]) else 0.0 for i in range(len(seg) - 1)]
    best_i = max(range(len(d)), key=d.__getitem__)
    return best_i if d[best_i] >= min_amp else None


def _fft(x: list[complex]) -> list[complex]:
    """Iterative radix-2 Cooley-Tukey FFT (len(x) must be a power of two)."""
    n = len(x)
    a = list(x)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]
    size = 2
    while size <= n:
        half = size // 2
        ang = -2.0 * math.pi / size
        wlen = complex(math.cos(ang), math.sin(ang))
        for start in range(0, n, size):
            w = 1.0 + 0.0j
            for k in range(half):
                u = a[start + k]
                v = a[start + k + half] * w
                a[start + k] = u + v
                a[start + k + half] = u - v
                w *= wlen
        size <<= 1
    return a


def overlap_leakage_db(inp, out, interferer, rate: int, clean=None) -> float | None:
    """Phase-invariant interferer leakage in dB using short-time cross-spectral density.

    For each STFT bin f where the interferer carries meaningful energy (`S_II(f) > 1e-4 * max`),
    `|S_YI(f)|^2 / S_II(f)` estimates interferer power preserved in `out` regardless of
    per-frequency phase shifts (such as an 80 Hz high-pass filter). When `clean` is supplied, its
    per-bin complex projection is subtracted from `out` and `inp` first so target energy at
    overlapping frequencies does not inflate leakage. Returns 10*log10(leaked_out / leaked_in)."""
    sigs = [inp, out, interferer] + ([clean] if clean is not None else [])
    n = min(len(s) for s in sigs)
    nfft = 1024 if rate >= 32000 else 512
    hop = nfft // 2
    if n < nfft:
        return None
    win = [0.5 - 0.5 * math.cos(2.0 * math.pi * k / nfft) for k in range(nfft)]
    bins = nfft // 2 + 1
    s_ii = [0.0] * bins
    s_xi = [0j] * bins
    s_yi = [0j] * bins
    s_cc = [0.0] * bins
    s_xc = [0j] * bins
    s_yc = [0j] * bins
    s_ci = [0j] * bins
    for pos in range(0, n - nfft + 1, hop):
        I = _fft([complex(interferer[pos + k] * win[k]) for k in range(nfft)])[:bins]
        X = _fft([complex(inp[pos + k] * win[k]) for k in range(nfft)])[:bins]
        Y = _fft([complex(out[pos + k] * win[k]) for k in range(nfft)])[:bins]
        C = _fft([complex(clean[pos + k] * win[k]) for k in range(nfft)])[:bins] if clean is not None else None
        for f in range(bins):
            ic = I[f].conjugate()
            s_ii[f] += (I[f] * ic).real
            s_xi[f] += X[f] * ic
            s_yi[f] += Y[f] * ic
            if C is not None:
                cc = C[f].conjugate()
                s_cc[f] += (C[f] * cc).real
                s_xc[f] += X[f] * cc
                s_yc[f] += Y[f] * cc
                s_ci[f] += C[f] * ic
    if clean is not None:
        max_cc = max(s_cc)
        c_floor = max_cc * 1e-6
        for f in range(bins):
            if s_cc[f] > c_floor:
                s_xi[f] -= (s_xc[f] / s_cc[f]) * s_ci[f]
                s_yi[f] -= (s_yc[f] / s_cc[f]) * s_ci[f]
    max_ii = max(s_ii)
    if max_ii <= 1e-20:
        return None
    # Restrict to bins where the interferer carries energy (excludes window sidelobes at target bins).
    floor = max_ii * 1e-4
    p_in = math.fsum((abs(s_xi[f]) ** 2) / s_ii[f] for f in range(bins) if s_ii[f] > floor)
    p_out = math.fsum((abs(s_yi[f]) ** 2) / s_ii[f] for f in range(bins) if s_ii[f] > floor)
    if p_in <= 1e-20:
        return None
    return to_db(p_out / p_in)


HIGH_BAND_HZ = 16000.0


def band_power_spectrum(x, rate: int, nfft: int = 1024, max_frames: int = 256) -> list[float]:
    """Mean Hann-windowed power spectrum (nfft/2+1 bins) over at most `max_frames` frames
    spread evenly across the signal. numpy when installed; the stdlib FFT otherwise."""
    n = len(x)
    if n < nfft:
        return []
    hop = nfft // 2
    starts = list(range(0, n - nfft + 1, hop))
    if len(starts) > max_frames:
        step = len(starts) / max_frames
        starts = [starts[int(i * step)] for i in range(max_frames)]
    win = [0.5 - 0.5 * math.cos(2.0 * math.pi * k / nfft) for k in range(nfft)]
    bins = nfft // 2 + 1
    if _np is not None:
        arr = _np.asarray(x, dtype=_np.float64)
        arr = _np.where(_np.isfinite(arr), arr, 0.0)
        frames = _np.stack([arr[s : s + nfft] for s in starts]) * _np.asarray(win)
        return (_np.abs(_np.fft.rfft(frames, axis=1)) ** 2).mean(axis=0).tolist()
    acc = [0.0] * bins
    for s in starts:
        spec = _fft([complex((x[s + k] if math.isfinite(x[s + k]) else 0.0) * win[k]) for k in range(nfft)])
        for f in range(bins):
            acc[f] += spec[f].real ** 2 + spec[f].imag ** 2
    return [v / len(starts) for v in acc]


def band_energy_ratio_db(x, rate: int, f_lo: float = HIGH_BAND_HZ, f_hi: float | None = None, nfft: int = 1024, max_frames: int = 256) -> float | None:
    """Energy in [f_lo, f_hi) relative to the whole spectrum, in dB (FLOOR_DB when the band is
    empty, None when the signal is silent, too short, or its Nyquist is at or below `f_lo`).
    With the defaults this is the >16 kHz content check for Oto-1's 48 kHz fullband gate: a
    16 kHz-band model resampled to 48 kHz scores near the floor, real fullband speech about
    −25 to −50 dB."""
    if rate / 2 <= f_lo:
        return None
    spec = band_power_spectrum(x, rate, nfft, max_frames)
    if not spec:
        return None
    total = math.fsum(spec)
    if total <= 1e-20:
        return None
    hz = rate / nfft
    band = math.fsum(p for f, p in enumerate(spec) if f * hz >= f_lo and (f_hi is None or f * hz < f_hi))
    return to_db(band / total)


def high_band_metrics(inp, in_rate: int, out, out_rate: int, min_input_db: float = -70.0) -> dict:
    """`high_band_16k_db` (output's >16 kHz share) and `high_band_16k_delta_db` (output minus
    input share) when the input is fullband: rate ≥ 44.1 kHz and >16 kHz content above
    `min_input_db`. Band-limited inputs (the 32 kHz Krisp set, phone audio) return {}, so the
    gate reports them as missing rather than failing a model for its input. When the input is
    fullband and the output rate is below 44.1 kHz, the output cannot carry >16 kHz content and
    scores `FLOOR_DB` so the fullband gate fails rather than being skipped."""
    if in_rate < 44100:
        return {}
    a = band_energy_ratio_db(inp, in_rate)
    if a is None or a < min_input_db:
        return {}
    if out_rate < 44100:
        b = FLOOR_DB
    else:
        b = band_energy_ratio_db(out, out_rate)
        if b is None:
            return {}
    return {"input_high_band_16k_db": a, "high_band_16k_db": b, "high_band_16k_delta_db": b - a}


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
