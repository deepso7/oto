"""Training sample streams: WebDataset shards (real data) and a synthetic source (``--smoke``).

Each yielded sample is a dict of CPU tensors with a fixed segment length ``T``:
``speech (T,)``, ``speech_fmax ()``, ``noise (T,)``, ``rir_near (L,)``, ``direct_near ()``,
``rir_far (L,)``, ``direct_far ()``, ``interferer (T,)``. Mixing happens on the GPU
(:mod:`oto_train.data.mixer`).
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import random
import tarfile
import zipfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

SR = 48_000
RIR_LEN = int(1.0 * SR)


def iter_tar(path: str) -> Iterator[tuple[str, dict[str, bytes]]]:
    """Group ``{key}.{ext}`` tar members into samples (WebDataset convention)."""
    cur_key, cur = None, {}
    with tarfile.open(path) as tf:
        for m in tf:
            if not m.isfile():
                continue
            key, _, ext = m.name.partition(".")
            if key != cur_key and cur:
                yield cur_key, cur
                cur = {}
            cur_key = key
            cur[ext] = tf.extractfile(m).read()
    if cur:
        yield cur_key, cur


def decode_flac(data: bytes) -> np.ndarray:
    import soundfile as sf

    x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    assert sr == SR, f"shards must be 48 kHz, got {sr}"
    return x.mean(axis=1)


class ShardPool:
    """Infinite shuffled stream of ``(audio, meta)`` from a list of tar shards with a shuffle buffer."""

    def __init__(
        self,
        shards: list[str],
        seed: int,
        buffer: int = 256,
        min_len: int = SR,
        max_len: int | None = None,
        max_bytes: int = 32 << 20,
    ):
        self.shards = shards
        self.seed = seed
        self.buffer = buffer
        self.min_len = min_len
        self.max_len = max_len
        self.max_bytes = max_bytes

    def __iter__(self):
        rng = random.Random(self.seed)
        buf: list[tuple[np.ndarray, dict]] = []
        buf_bytes = 0
        while True:
            order = list(self.shards)
            rng.shuffle(order)
            for sh in order:
                for _, files in iter_tar(sh):
                    if "flac" not in files:
                        continue
                    x = decode_flac(files["flac"])
                    if len(x) < self.min_len:
                        continue
                    if self.max_len is not None and len(x) > self.max_len:
                        start = rng.randrange(len(x) - self.max_len + 1)
                        x = x[start : start + self.max_len].copy()
                    meta = json.loads(files.get("json", b"{}"))
                    buf.append((x, meta))
                    buf_bytes += x.nbytes
                    while buf and (len(buf) >= self.buffer or buf_bytes >= self.max_bytes):
                        item = buf.pop(rng.randrange(len(buf)))
                        buf_bytes -= item[0].nbytes
                        yield item


def _crop_or_loop(x: np.ndarray, t: int, rng: random.Random) -> np.ndarray:
    if len(x) >= t:
        s = rng.randrange(len(x) - t + 1)
        return x[s : s + t]
    reps = math.ceil(t / len(x))
    return np.tile(x, reps)[:t]


def _fill_speech(pool_iter, t: int, rng: random.Random) -> tuple[np.ndarray, float]:
    """Concatenate utterances (with short gaps) until ``t`` samples; ``f_max`` = minimum over parts."""
    out, f_max, n = [], 24_000.0, 0
    while n < t:
        x, meta = next(pool_iter)
        f_max = min(f_max, float(meta.get("f_max", 24_000.0)) or 24_000.0)
        x = _crop_or_loop(x, min(len(x), t - n), rng) if n else x[rng.randrange(max(1, len(x) // 2)) :]
        out.append(x)
        n += len(x)
        gap = int(rng.uniform(0.0, 0.4) * SR)
        out.append(np.zeros(gap, np.float32))
        n += gap
    return np.concatenate(out)[:t], f_max


class RirBank:
    """Random paired near/far RIRs from ``rir_gen`` ``.npz`` files (loaded lazily per sample)."""

    def __init__(self, files: list[str]):
        self.files = files

    def sample(self, rng: random.Random):
        if not self.files:
            return None
        d = np.load(rng.choice(self.files))
        meta = json.loads(str(d["meta"]))
        k = rng.randrange(d["far"].shape[0])
        near = _fit(d["near"], RIR_LEN)
        far = _fit(d["far"][k], RIR_LEN)
        return near, meta["direct_idx"][0], far, meta["direct_idx"][1 + k]


def _fit(h: np.ndarray, n: int) -> np.ndarray:
    return h[:n] if len(h) >= n else np.pad(h, (0, n - len(h)))


def _read_rir_rt60(path: str) -> tuple[bool, float]:
    """Read ``meta["rt60"]`` from an RIR ``.npz`` without decompressing the waveforms.

    Returns ``(has_rt60, value)``; synthetic fixtures without an ``rt60`` key return ``(False, 0.0)``.
    Present-but-null/non-numeric values return ``(True, nan)`` so they are rejected by the filter.
    """
    try:
        with zipfile.ZipFile(path) as zf:
            missing = {"meta.npy", "near.npy", "far.npy"} - set(zf.namelist())
            if missing:
                raise KeyError(f"missing {', '.join(sorted(missing))}")
            with zf.open("meta.npy") as fh:
                raw = np.lib.format.read_array(io.BytesIO(fh.read()), allow_pickle=False)
        meta = json.loads(str(raw))
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"invalid RIR archive {path}: {e}") from e
    if not isinstance(meta, dict):
        raise ValueError(f"invalid RIR archive {path}: meta must be a JSON object")
    if "rt60" not in meta:
        return False, 0.0
    val = meta["rt60"]
    if val is None or isinstance(val, bool):
        return True, float("nan")
    try:
        return True, float(val)
    except (TypeError, ValueError):
        return True, float("nan")


class ShardStream(IterableDataset):
    def __init__(
        self,
        root: str,
        split: str,
        seconds: float,
        rir_dir: str | None = None,
        seed: int = 0,
        max_rt60: float | None = 1.5,
    ):
        super().__init__()
        if max_rt60 is not None and (not math.isfinite(max_rt60) or max_rt60 <= 0):
            raise ValueError(f"max_rt60 must be positive and finite (or None), got {max_rt60!r}")
        rootp = Path(root) / split
        self.speech = sorted(str(p) for p in rootp.glob("speech-*.tar"))
        self.noise = sorted(str(p) for p in rootp.glob("noise-*.tar")) + sorted(str(p) for p in rootp.glob("music-*.tar"))
        self.interf = sorted(str(p) for p in rootp.glob("interferer-*.tar")) or self.speech
        if not self.speech:
            raise FileNotFoundError(f"no speech shards under {rootp}")
        if rir_dir:
            all_rirs = sorted(str(p) for p in Path(rir_dir).glob("*.npz"))
            if not all_rirs:
                raise FileNotFoundError(f"no RIR .npz files under {rir_dir}")
            from .rir_gen import room_split

            rirs = [r for r in all_rirs if room_split(int(Path(r).stem)) == split]
            if not rirs:
                raise FileNotFoundError(f"no RIR .npz files for split {split!r} under {rir_dir}")
            if max_rt60 is not None:
                kept = []
                for r in rirs:
                    has_rt60, rt60 = _read_rir_rt60(r)
                    if not has_rt60 or (math.isfinite(rt60) and 0.0 < rt60 <= max_rt60):
                        kept.append(r)
                rirs = kept
                if not rirs:
                    raise FileNotFoundError(
                        f"no RIR .npz files for split {split!r} with 0 < rt60 <= {max_rt60} s under {rir_dir}"
                    )
        else:
            rirs = []
        self.rirs = RirBank(rirs)
        self.t = int(seconds * SR) // 512 * 512
        self.seed, self.split = seed, split

    def __iter__(self):
        wi = get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        seed = int.from_bytes(hashlib.sha256(f"{self.seed}:{wid}:{self.split}".encode()).digest()[:4], "big")
        rng = random.Random(seed)

        def part(lst):
            return lst[wid::nw] or lst

        sp = iter(ShardPool(part(self.speech), seed, max_len=self.t * 2))
        nz = iter(ShardPool(part(self.noise), seed + 1, max_len=self.t)) if self.noise else None
        it = iter(ShardPool(part(self.interf), seed + 2, min_len=SR // 2, max_len=self.t))
        while True:
            speech, f_max = _fill_speech(sp, self.t, rng)
            sample = {"speech": torch.from_numpy(speech.copy()), "speech_fmax": torch.tensor(f_max)}
            sample["noise"] = torch.from_numpy(_crop_or_loop(next(nz)[0], self.t, rng).copy()) if nz else torch.zeros(self.t)
            sample["interferer"] = torch.from_numpy(_crop_or_loop(next(it)[0], self.t, rng).copy())
            r = self.rirs.sample(rng)
            if r is not None:
                near, dn, far, df = r
                sample.update(rir_near=torch.from_numpy(near.astype(np.float32)), direct_near=torch.tensor(dn),
                              rir_far=torch.from_numpy(far.astype(np.float32)), direct_far=torch.tensor(df))  # fmt: skip
            yield sample


# ---------------------------------------------------------------------------
# Synthetic source for --smoke (no datasets needed)
# ---------------------------------------------------------------------------


def synth_speech(t: int, rng: np.random.Generator) -> np.ndarray:
    """Speech-like signal: harmonic stack with vibrato + syllabic AM + fricative bursts up to 20 kHz."""
    n = np.arange(t) / SR
    f0 = rng.uniform(90, 250) * (1 + 0.05 * np.sin(2 * np.pi * rng.uniform(3, 6) * n))
    phase = 2 * np.pi * np.cumsum(f0) / SR
    x = sum((0.6**k) * np.sin(k * phase) for k in range(1, 30) if k * f0.max() < 20_000)
    syll = np.clip(np.sin(2 * np.pi * rng.uniform(2, 5) * n + rng.uniform(0, 6)), 0, None) ** 2
    fric = rng.standard_normal(t) * (np.sin(2 * np.pi * rng.uniform(1, 3) * n) > 0.9)
    y = 0.1 * x * syll + 0.02 * fric
    return (y / (np.abs(y).max() + 1e-9) * 0.5).astype(np.float32)


def synth_rir(rng: np.random.Generator, rt60: float, length: int = RIR_LEN, direct: int = 100) -> np.ndarray:
    h = np.zeros(length, np.float32)
    decay = np.exp(-6.9 * np.arange(length - direct) / (rt60 * SR))
    h[direct:] = (rng.standard_normal(length - direct) * decay * 0.3).astype(np.float32)
    h[direct] = 1.0
    return h


class SyntheticStream(IterableDataset):
    def __init__(self, seconds: float = 2.0, seed: int = 0):
        super().__init__()
        self.t = int(seconds * SR) // 512 * 512
        self.seed = seed

    def __iter__(self):
        wi = get_worker_info()
        rng = np.random.default_rng([self.seed, wi.id if wi else 0])
        while True:
            yield {
                "speech": torch.from_numpy(synth_speech(self.t, rng)),
                "speech_fmax": torch.tensor(float(rng.choice([8_000.0, 20_000.0, 24_000.0]))),
                "noise": torch.from_numpy((rng.standard_normal(self.t) * 0.1).astype(np.float32)),
                "interferer": torch.from_numpy(synth_speech(self.t, rng)),
                "rir_near": torch.from_numpy(synth_rir(rng, rng.uniform(0.2, 0.6))),
                "direct_near": torch.tensor(100),
                "rir_far": torch.from_numpy(synth_rir(rng, rng.uniform(0.4, 1.2))),
                "direct_far": torch.tensor(100),
            }
