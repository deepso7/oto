"""Deterministic synthetic CI fixtures. Stdlib only; no recorded audio, so they are
redistributable (docs/training-data-audit.md: CI fixtures are synthetic or CC0).

    python3 tools/eval/fixtures.py OUT_DIR

writes the WAVs plus `fixtures.json`, which describes every file: its rate, the clean
reference, the speech-active and noise-only windows (seconds), the SNR, and the impulse/step
positions (samples) used to check delay alignment.

The "speech" is a harmonic source with a moving pitch, vowel-like formant weighting and
syllable envelopes. It is not speech, so models trained on real speech may treat parts of it
as noise; the CI bars in gates.toml are sanity bars, not quality scores. Quality is measured
on real data with `eval.py --suite manifest`.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

from wavio import write_wav

SEED = 0x070
RATE = 48_000
HFP_RATE = 16_000
DURATION = 4.0
# Speech spans (seconds). Everything outside them is noise only. The first WARMUP seconds are
# excluded from noise-only measurements: models adapt their noise estimate at startup.
SPEECH_SPANS = [(1.0, 2.0), (2.5, 3.5)]
WARMUP = 0.5
# Measurement windows keep this far from speech edges, so onsets and model decay don't count.
GUARD = 0.05
SPEECH_RMS_DBFS = -23.0
NOISE_ONLY_RMS_DBFS = -30.0
IMPULSES_S = [0.25, 0.75, 1.25, 1.75]
IMPULSE_AMP = 0.5
STEP_S = 2.75
STEP_AMP = 0.25
VOICE_TOP_HZ = 5400.0
HUM_DB = -10.0  # mains hum relative to the pink noise

# Vowel-ish formants (Hz, bandwidth Hz).
VOWELS = [
    [(700, 130), (1220, 70), (2600, 160)],  # a
    [(300, 60), (2300, 100), (3000, 120)],  # i
    [(350, 65), (850, 80), (2400, 140)],  # u
    [(500, 90), (1800, 100), (2500, 140)],  # e
]


def db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)


def rms(x, start: int = 0, end: int | None = None) -> float:
    seg = x[start:end]
    if not seg:
        return 0.0
    return math.sqrt(math.fsum(s * s for s in seg) / len(seg))


def _span_mask(n: int, rate: int, spans) -> list[bool]:
    m = [False] * n
    for a, b in spans:
        for i in range(int(a * rate), min(n, int(b * rate))):
            m[i] = True
    return m


def speech_like(rate: int, duration: float = DURATION, spans=SPEECH_SPANS, band_hz: float | None = None, seed: int = SEED) -> list[float]:
    """Voice-like signal inside `spans`, zero elsewhere, normalised to SPEECH_RMS_DBFS over the
    active part. Syllables are a jittered harmonic source (1/k tilt, vowel formants, up to
    VOICE_TOP_HZ or `band_hz`) plus a breath component, with short fricative bursts between
    them. Without the aperiodic parts DPDFNet-2 removes the "voice" as noise (−23 dB at
    Balanced); with them it keeps it within about 2 dB, like DeepFilterNet3."""
    rng = random.Random(seed)
    n = int(rate * duration)
    top = min(band_hz or VOICE_TOP_HZ, rate / 2 * 0.9)
    out = [0.0] * n
    syllable = 0.22  # seconds per syllable, including its gap
    for a, b in spans:
        t0 = a
        while t0 + 0.12 < b:
            length = min(syllable * (0.8 + 0.4 * rng.random()), b - t0)
            vowel = VOWELS[rng.randrange(len(VOWELS))]
            f0_start = 110 + 50 * rng.random()
            f0_end = f0_start * (0.85 + 0.3 * rng.random())
            i0, i1 = int(t0 * rate), min(n, int((t0 + length) * rate))
            count = max(1, i1 - i0)
            # Harmonic weights from the syllable's mean pitch: formant peaks over a 1/k tilt.
            f0_mean = (f0_start + f0_end) / 2
            weights = []
            k = 1
            while k * max(f0_start, f0_end) * 1.03 < top:
                fk = k * f0_mean
                g = sum(1.0 / (1.0 + ((fk - fc) / bw) ** 2) for fc, bw in vowel)
                weights.append((k, (g + 0.02) / k))
                k += 1
            phase = 0.0
            for j in range(count):
                frac = j / count
                f0 = f0_start + (f0_end - f0_start) * frac
                f0 *= (1.0 + 0.02 * (rng.random() - 0.5)) * (1.0 + 0.01 * math.sin(2 * math.pi * 5.5 * j / rate))
                phase += 2 * math.pi * f0 / rate
                # Raised-cosine syllable envelope with a short tail gap.
                if frac >= 0.85:
                    continue
                env = math.sin(math.pi * frac / 0.85) ** 2
                voiced = sum(w * math.sin(k * phase) for k, w in weights)
                out[i0 + j] = env * (voiced + 0.05 * (rng.random() * 2.0 - 1.0))
            # Fricative: 30 ms of first-differenced (high-tilted) noise after the syllable.
            burst = int(0.03 * rate)
            prev = 0.0
            for j in range(min(burst, n - i1)):
                w = rng.random() * 2.0 - 1.0
                out[i1 + j] += 0.15 * (w - prev) * math.sin(math.pi * j / burst)
                prev = w
            t0 += length + 0.03
    mask = _span_mask(n, rate, spans)
    active = [s for s, m in zip(out, mask) if m]
    gain = db_to_lin(SPEECH_RMS_DBFS) / max(rms(active), 1e-12)
    return [s * gain for s in out]


def pink_hum_noise(rate: int, duration: float = DURATION, seed: int = SEED + 1, hum_hz: float = 50.0) -> list[float]:
    """Pink noise (Kellet's filter on uniform white noise) plus mains hum with two harmonics
    HUM_DB below it, normalised to unit RMS. The hum stays well below the pink part because
    the chain's 80 Hz high-pass removes most of it whatever the model does."""
    rng = random.Random(seed)
    n = int(rate * duration)
    b0 = b1 = b2 = b3 = b4 = b5 = b6 = 0.0
    pink = [0.0] * n
    hum = [0.0] * n
    for i in range(n):
        w = rng.random() * 2.0 - 1.0
        b0 = 0.99886 * b0 + w * 0.0555179
        b1 = 0.99332 * b1 + w * 0.0750759
        b2 = 0.96900 * b2 + w * 0.1538520
        b3 = 0.86650 * b3 + w * 0.3104856
        b4 = 0.55000 * b4 + w * 0.5329522
        b5 = -0.7616 * b5 - w * 0.0168980
        pink[i] = b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362
        b6 = w * 0.115926
        t = i / rate
        hum[i] = 0.6 * math.sin(2 * math.pi * hum_hz * t) + 0.3 * math.sin(2 * math.pi * 2 * hum_hz * t) + 0.15 * math.sin(2 * math.pi * 3 * hum_hz * t)
    gp = 1.0 / max(rms(pink), 1e-12)
    gh = db_to_lin(HUM_DB) / max(rms(hum), 1e-12)
    out = [p * gp + h * gh for p, h in zip(pink, hum)]
    g = 1.0 / max(rms(out), 1e-12)
    return [s * g for s in out]


def mix_at_snr(clean: list[float], noise: list[float], snr_db: float, rate: int, spans=SPEECH_SPANS) -> list[float]:
    mask = _span_mask(len(clean), rate, spans)
    speech_rms = rms([s for s, m in zip(clean, mask) if m])
    g = speech_rms / db_to_lin(snr_db) / max(rms(noise), 1e-12)
    return [c + g * w for c, w in zip(clean, noise)]


def windows(spans, duration: float = DURATION, warmup: float = WARMUP, guard: float = GUARD):
    """(speech windows, noise-only windows) in seconds, shrunk by `guard` at every edge."""
    spans = sorted(spans)
    speech = [(a + guard, b - guard) for a, b in spans]
    starts = [warmup] + [b + guard for _, b in spans]
    ends = [a - guard for a, _ in spans] + [duration - guard]
    gaps = [(max(a, warmup), b) for a, b in zip(starts, ends) if b > max(a, warmup)]
    return speech, gaps


def generate(out_dir: str | Path, fmt: str = "pcm16") -> dict:
    """Writes every fixture into `out_dir` and returns the manifest (also `fixtures.json`)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = int(RATE * DURATION)
    speech_w, noise_w = windows(SPEECH_SPANS)
    clean = speech_like(RATE)
    noise = pink_hum_noise(RATE)
    fixtures: list[dict] = []

    def add(name: str, samples, kind: str, rate: int = RATE, **meta):
        path = out / f"{name}.wav"
        write_wav(path, rate, samples, fmt)
        entry = {"name": name, "kind": kind, "path": path.name, "rate": rate, "frames": len(samples)}
        entry.update(meta)
        fixtures.append(entry)

    write_wav(out / "clean_speech_ref.wav", RATE, clean, fmt)
    add("clean_speech", clean, "clean", reference="clean_speech_ref.wav", speech_windows=speech_w, noise_windows=noise_w)
    for snr in (15, 5):
        add(f"noisy_speech_{snr}db", mix_at_snr(clean, noise, snr, RATE), "noisy", reference="clean_speech_ref.wav", snr_db=snr, speech_windows=speech_w, noise_windows=noise_w)
    g = db_to_lin(NOISE_ONLY_RMS_DBFS)
    add("noise_only", [s * g for s in noise], "noise_only", noise_windows=[(WARMUP, DURATION - GUARD)])

    align = [0.0] * n
    impulses = [int(t * RATE) for t in IMPULSES_S]
    for p in impulses:
        align[p] = IMPULSE_AMP
    step = int(STEP_S * RATE)
    for i in range(step, n):
        align[i] = STEP_AMP
    add("impulse_step", align, "alignment", impulses=impulses, step=step)

    add("silence", [0.0] * n, "silence")

    hfp_clean = speech_like(HFP_RATE, band_hz=3400.0, seed=SEED + 2)
    hfp_noise = pink_hum_noise(HFP_RATE, seed=SEED + 3)
    write_wav(out / "hfp_16k_ref.wav", HFP_RATE, hfp_clean, fmt)
    add("hfp_16k_noisy_15db", mix_at_snr(hfp_clean, hfp_noise, 15, HFP_RATE), "noisy", rate=HFP_RATE, reference="hfp_16k_ref.wav", snr_db=15, speech_windows=speech_w, noise_windows=noise_w)

    manifest = {"seed": SEED, "format": fmt, "duration_s": DURATION, "fixtures": fixtures}
    (out / "fixtures.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: fixtures.py OUT_DIR")
    m = generate(sys.argv[1])
    print(f"wrote {len(m['fixtures'])} fixtures to {sys.argv[1]}")
