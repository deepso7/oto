"""Licence-enforcing WebDataset shard packer for Oto-1 training data (run on ``mito``).

Only sources listed in ``ALLOWLIST`` (mirroring ``docs/training-data-audit.md``) can be
packed; anything else is an error. Per-clip licence filters are enforced for FSD50K
(``*_clips_info_FSD50K.json``) and FMA (``tracks.csv``); HiFiTTS-2 is filtered by
bandwidth, single-speaker flag and per-speaker cap.

Subcommands::

    # 1. Inspect 50 HiFiTTS-2 chapters for MP3 top-band artefacts BEFORE bulk download
    python -m oto_train.data.pack_shards spot-check --hifitts2-manifest m.jsonl \\
        --hifitts2-audio /data/hifitts2 --spot-check 50 --out /data/spot

    # 2. Collect + filter + split + pack + ATTRIBUTION.md
    python -m oto_train.data.pack_shards pack --out /data/shards \\
        --hifitts2-manifest m.jsonl --hifitts2-audio /data/hifitts2 \\
        --vctk /data/VCTK-Corpus-0.92 --fsd50k /data/FSD50K --fma-tracks /data/fma_metadata/tracks.csv \\
        --fma-audio /data/fma_large --workers 16

    # 3. Upload (Cloudflare R2 via S3 API, or a private Hugging Face dataset repo)
    python -m oto_train.data.pack_shards upload --src /data/shards --r2-bucket oto-train --prefix v1/
    python -m oto_train.data.pack_shards upload --src /data/shards --hf-repo me/oto-train-shards

Shard layout (``{out}/{split}/{kind}-{split}-{000000}.tar``): each sample is
``{key}.flac`` (48 kHz mono PCM_16) + ``{key}.json`` (source, licence, speaker, session,
f_max, orig_sr, duration, attribution). ``kind`` is ``speech``, ``noise``, ``music`` or
``interferer``.

Splits are deterministic and **speaker- and session-disjoint**: the split is a hash of the
speaker (speech) or uploader/artist (noise, music), so a speaker, chapter, recordist or
artist never lands on two sides. Split lists (ids only) are written to ``{out}/splits/``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
import sys
import tarfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .bandwidth import band_energy_ratio_db, estimate_bandwidth

TARGET_SR = 48_000

# ---------------------------------------------------------------------------
# Licence allow-list (docs/training-data-audit.md, "Permissive" table)
# ---------------------------------------------------------------------------

ALLOWLIST: dict[str, dict] = {
    "hifitts2": {
        "licence": "CC-BY-4.0 (metadata); audio public domain (LibriVox)",
        "citation": "NVIDIA HiFiTTS-2, https://huggingface.co/datasets/nvidia/hifitts-2 (CC BY 4.0); "
        "audio from LibriVox (public domain), https://librivox.org",
    },
    "vctk": {
        "licence": "CC-BY-4.0",
        "citation": "Yamagishi, Veaux, MacDonald. CSTR VCTK Corpus 0.92, University of Edinburgh, "
        "https://doi.org/10.7488/ds/2645 (CC BY 4.0)",
    },
    "fsd50k": {
        "licence": "per-clip CC0-1.0 / CC-BY (3.0, 4.0)",
        "citation": "Fonseca et al. FSD50K, https://zenodo.org/records/4060432 ; per-clip Freesound licences listed below",
    },
    "fma": {
        "licence": "per-track CC0 / CC-BY",
        "citation": "Defferrard et al. FMA: A Dataset for Music Analysis, https://github.com/mdeff/fma ; "
        "per-track licences listed below",
    },
    # Directory sources packed with their fixed licence (verify the licence file on download).
    "libritts": {"licence": "CC-BY-4.0", "citation": "LibriTTS, https://www.openslr.org/60/ (CC BY 4.0)"},
    "librispeech": {"licence": "CC-BY-4.0", "citation": "LibriSpeech, https://www.openslr.org/12/ (CC BY 4.0)"},
    "mls": {"licence": "CC-BY-4.0", "citation": "Multilingual LibriSpeech, https://www.openslr.org/94/ (CC BY 4.0)"},
    "commonvoice": {"licence": "CC0-1.0", "citation": "Mozilla Common Voice (CC0 1.0)"},
    "globe": {"licence": "CC0-1.0", "citation": "GLOBE, https://huggingface.co/datasets/MushanW/GLOBE_V2 (CC0 1.0)"},
    "ami": {"licence": "CC-BY-4.0", "citation": "AMI Meeting Corpus, https://groups.inf.ed.ac.uk/ami/corpus/ (CC BY 4.0)"},
    "dipco": {"licence": "CDLA-Permissive-1.0", "citation": "DiPCo, Amazon (CDLA-Permissive-1.0)"},
    "dns5_freesound": {"licence": "CC0-1.0", "citation": "DNS Challenge 5 freesound_* noise (CC0); DEMAND removed"},
    "ace": {"licence": "CC-BY-4.0", "citation": "ACE Challenge corpus (CC BY 4.0)"},
    "oto_rir": {"licence": "MIT OR Apache-2.0", "citation": "Oto pyroomacoustics-simulated 48 kHz RIR bank"},
}

# Hard deny-list: refused even if passed as a directory source.
DENYLIST = {"daps", "dds", "ears", "wham", "whamr", "esc50", "demand", "qut", "micirp", "audioset", "voxceleb",
            "yodas", "crema", "ptdb", "slr28_real", "gwa", "mit_ir", "tau", "krisp", "aihub"}  # fmt: skip

CC0_RE = re.compile(r"publicdomain/zero|\bcc0\b|creative commons zero", re.I)
CC_BY_URL_RE = re.compile(r"creativecommons\.org/licenses/by/\d")
RESTRICTIVE_RE = re.compile(
    r"non.?commercial|\bnc\b|share.?alike|\bsa\b|no.?deriv|\bnd\b|sampling|\bby-(nc|sa|nd)", re.I
)

# FSD50K vocabulary labels that contain human voice: routed to interferers, never to noise.
FSD50K_VOICE = {
    "Human_voice", "Speech", "Male_speech_and_man_speaking", "Female_speech_and_woman_speaking",
    "Child_speech_and_kid_speaking", "Conversation", "Speech_synthesizer", "Singing", "Male_singing",
    "Female_singing", "Whispering", "Screaming", "Shout", "Yell", "Laughter", "Giggle", "Chuckle_and_chortle",
    "Crying_and_sobbing", "Chatter", "Crowd", "Cheering", "Applause_and_cheering",
}  # fmt: skip


def fsd50k_licence_ok(url: str) -> bool:
    """Allow only CC0 and plain CC-BY (any version). Rejects BY-NC, BY-SA, Sampling+ etc."""
    u = url.strip().lower()
    if "publicdomain/zero" in u:
        return True
    return bool(CC_BY_URL_RE.search(u)) and "sampling" not in u


def fma_licence_ok(text: str) -> bool:
    """Allow CC0 and plain CC-BY (``Attribution`` without NC / SA / ND). Unknown or empty -> reject."""
    t = (text or "").strip()
    if not t:
        return False
    if CC0_RE.search(t):
        return True
    if "creativecommons.org" in t.lower():
        return fsd50k_licence_ok(t)
    return bool(re.search(r"\battribution\b", t, re.I)) and not RESTRICTIVE_RE.search(t)


def check_source(source: str) -> None:
    if source.lower() in DENYLIST or source not in ALLOWLIST:
        raise SystemExit(f"source {source!r} is not on the training allow-list (docs/training-data-audit.md)")


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------


@dataclass
class Item:
    key: str
    source: str
    path: str
    kind: str  # speech | noise | music | interferer
    speaker: str
    session: str
    licence: str
    licence_url: str = ""
    title: str = ""
    author: str = ""
    url: str = ""
    duration: float = 0.0
    f_max: float = 0.0  # from metadata if known, re-estimated at pack time
    extra: dict = field(default_factory=dict)


def _safe_key(*parts: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", "_".join(parts)).strip("-")[:120]


def _first(d: dict, names: list[str], default=None):
    for n in names:
        if n in d and d[n] not in (None, ""):
            return d[n]
    return default


# -- HiFiTTS-2 ---------------------------------------------------------------


def _is_multispeaker(row: dict, field_name: str | None) -> bool | None:
    """True/False if the row says so, ``None`` if it carries no multi-speaker information."""
    names = [field_name] if field_name else ["is_multispeaker", "multi_speaker", "multispeaker",
                                             "num_speakers", "speaker_count", "n_speakers"]  # fmt: skip
    for k in names:
        if k not in row or row[k] is None:
            continue
        v = row[k]
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return v > 1 if ("count" in k or "num" in k or k.startswith("n_")) else bool(v)
        return str(v).strip().lower() in ("1", "true", "yes", "multi")
    return None


def collect_hifitts2(
    manifests: list[str],
    audio_root: str,
    min_bw: float = 16_000.0,
    max_speaker_hours: float = 0.5,
    path_field: str = "audio_filepath",
    speaker_field: str = "speaker",
    bw_field: str = "bandwidth",
    duration_field: str = "duration",
    multispk_field: str | None = None,
    allow_unknown_multispk: bool = False,
    seed: int = 0,
) -> tuple[list[Item], dict]:
    """Filter HiFiTTS-2 manifest rows. Fail closed: rows without bandwidth are dropped.

    Rows with no multi-speaker information are dropped unless ``allow_unknown_multispk``.
    The per-speaker cap keeps a seeded random subset of whole chapters.
    """
    stats: dict[str, float] = defaultdict(int)
    by_spk: dict[str, list[Item]] = defaultdict(list)
    for mpath in manifests:
        with open(mpath) as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                stats["rows"] += 1
                bw = _first(row, [bw_field, "bandwidth", "estimated_bandwidth"])
                if bw is None:
                    stats["drop_no_bandwidth"] += 1
                    continue
                if float(bw) < min_bw:
                    stats["drop_bandwidth"] += 1
                    continue
                multi = _is_multispeaker(row, multispk_field)
                if multi or (multi is None and not allow_unknown_multispk):
                    stats["drop_multispeaker" if multi else "drop_unknown_multispeaker"] += 1
                    continue
                rel = row[path_field]
                spk = str(_first(row, [speaker_field, "speaker_id", "speaker"]))
                chapter = str(_first(row, ["chapter_id", "chapter", "book_id", "book"], Path(rel).parent.as_posix()))
                by_spk[spk].append(
                    Item(
                        key=_safe_key("hifitts2", spk, Path(rel).stem),
                        source="hifitts2",
                        path=str(Path(audio_root) / rel),
                        kind="speech",
                        speaker=f"hifitts2:{spk}",
                        session=f"hifitts2:{spk}:{chapter}",
                        licence=ALLOWLIST["hifitts2"]["licence"],
                        url="https://librivox.org",
                        duration=float(_first(row, [duration_field, "duration"], 0.0)),
                        f_max=float(bw),
                    )
                )
    rng = random.Random(seed)
    items: list[Item] = []
    cap = max_speaker_hours * 3600
    for spk in sorted(by_spk):
        sessions: dict[str, list[Item]] = defaultdict(list)
        for it in by_spk[spk]:
            sessions[it.session].append(it)
        order = sorted(sessions)
        rng.shuffle(order)
        total = 0.0
        for s in order:
            if total >= cap:
                stats["drop_speaker_cap"] += len(sessions[s])
                continue
            items.extend(sessions[s])
            total += sum(i.duration for i in sessions[s])
    stats["kept"] = len(items)
    stats["speakers"] = len({i.speaker for i in items})
    stats["hours"] = round(sum(i.duration for i in items) / 3600, 2)
    return items, dict(stats)


# -- VCTK 0.92 ---------------------------------------------------------------


def collect_vctk(root: str, mic: str = "mic1", exclude=("p315",)) -> list[Item]:
    """VCTK 0.92 FLACs (``wav48_silence_trimmed/<spk>/<spk>_<utt>_mic1.flac``). Checks the licence file."""
    rootp = Path(root)
    lic = next(iter(sorted(rootp.glob("license*.txt"))), None)
    if lic is None:
        raise SystemExit(f"{root}: license_text.txt not found; refusing to pack VCTK without its licence file")
    if "Attribution 4.0" not in lic.read_text(errors="ignore"):
        raise SystemExit(f"{lic}: expected CC BY 4.0 licence text for VCTK 0.92")
    items = []
    for f in sorted((rootp / "wav48_silence_trimmed").glob(f"*/*_{mic}.flac")):
        spk = f.parent.name
        if spk in exclude:
            continue
        items.append(
            Item(
                key=_safe_key("vctk", f.stem),
                source="vctk",
                path=str(f),
                kind="speech",
                speaker=f"vctk:{spk}",
                session=f"vctk:{spk}",
                licence="CC-BY-4.0",
                licence_url="https://creativecommons.org/licenses/by/4.0/",
            )
        )
    return items


# -- FSD50K ------------------------------------------------------------------


def collect_fsd50k(root: str) -> tuple[list[Item], dict]:
    """Per-clip licence filter over ``{dev,eval}_clips_info_FSD50K.json``.

    The split group is the Freesound uploader, so one recordist never spans train/dev/test.
    Requires the matching ``FSD50K.ground_truth/{dev,eval}.csv`` (or ``ground_truth/{dev,eval}.csv``)
    for every supplied partition so voice clips are never silently routed into the noise pool.
    """
    rootp = Path(root)
    stats: dict[str, int] = defaultdict(int)
    items = []
    for part in ("dev", "eval"):
        info_path = next(iter(rootp.glob(f"**/{part}_clips_info_FSD50K.json")), None)
        if info_path is None:
            continue
        gt_path = next(iter(rootp.glob(f"**/FSD50K.ground_truth/{part}.csv")), None) or next(
            iter(rootp.glob(f"**/ground_truth/{part}.csv")), None
        )
        if gt_path is None:
            raise FileNotFoundError(
                f"FSD50K {part} ground-truth CSV not found under {rootp} "
                f"(expected FSD50K.ground_truth/{part}.csv); refusing to pack without voice-label filtering"
            )
        labels: dict[str, set[str]] = {}
        with open(gt_path) as fh:
            for row in csv.DictReader(fh):
                labels[str(row["fname"])] = set(row["labels"].split(","))
        info = json.loads(info_path.read_text())
        audio_dir = next(iter(rootp.glob(f"**/FSD50K.{part}_audio")), rootp / f"FSD50K.{part}_audio")
        for fname, meta in info.items():
            stats["clips"] += 1
            url = meta.get("license", "")
            if not fsd50k_licence_ok(url):
                stats["drop_licence"] += 1
                continue
            clip_labels = labels.get(str(fname))
            if clip_labels is None:
                stats["drop_unlabeled"] += 1
                continue
            kind = "interferer" if clip_labels & FSD50K_VOICE else "noise"
            stats[kind] += 1
            uploader = meta.get("uploader", "unknown")
            items.append(
                Item(
                    key=_safe_key("fsd50k", str(fname)),
                    source="fsd50k",
                    path=str(audio_dir / f"{fname}.wav"),
                    kind=kind,
                    speaker=f"fsd50k:{uploader}",
                    session=f"fsd50k:{uploader}",
                    licence="CC0-1.0" if "publicdomain/zero" in url else "CC-BY",
                    licence_url=url,
                    title=meta.get("title", ""),
                    author=uploader,
                    url=f"https://freesound.org/s/{fname}/",
                )
            )
    return items, dict(stats)


# -- FMA ---------------------------------------------------------------------


def collect_fma(tracks_csv: str, audio_root: str, require_audio: bool = True) -> tuple[list[Item], dict]:
    """Parse FMA's 3-row-header ``tracks.csv`` and keep CC0 / CC-BY tracks (split group = artist)."""
    stats: dict[str, int] = defaultdict(int)
    items = []
    with open(tracks_csv, newline="") as fh:
        rd = csv.reader(fh)
        top, sub = next(rd), next(rd)
        next(rd)  # "track_id" row
        cols = {(a.strip(), b.strip()): i for i, (a, b) in enumerate(zip(top, sub))}
        i_lic = cols[("track", "license")]
        i_title = cols.get(("track", "title"))
        i_artist = cols.get(("artist", "name"))
        i_artist_id = cols.get(("artist", "id"))
        for row in rd:
            if not row:
                continue
            stats["tracks"] += 1
            lic = row[i_lic]
            if not fma_licence_ok(lic):
                stats["drop_licence"] += 1
                continue
            tid = int(row[0])
            path = Path(audio_root) / f"{tid:06d}"[:3] / f"{tid:06d}.mp3"
            if require_audio and not path.exists():
                stats["missing_audio"] += 1
                continue
            artist = row[i_artist] if i_artist is not None else ""
            artist_id = row[i_artist_id] if i_artist_id is not None else artist
            stats["kept"] += 1
            items.append(
                Item(
                    key=_safe_key("fma", str(tid)),
                    source="fma",
                    path=str(path),
                    kind="music",
                    speaker=f"fma:{artist_id}",
                    session=f"fma:{artist_id}",
                    licence="CC0-1.0" if CC0_RE.search(lic) else "CC-BY",
                    licence_url=lic,
                    title=row[i_title] if i_title is not None else "",
                    author=artist,
                    url=f"https://freemusicarchive.org/track/{tid}",
                )
            )
    return items, dict(stats)


# -- Generic directory source --------------------------------------------------


# Sources that require dedicated per-item licence / quality collectors and must never bypass them via `--dir`.
DEDICATED_ONLY = {"fsd50k", "fma", "hifitts2"}


def collect_dir(source: str, root: str, kind: str, speaker_depth: int = 1) -> list[Item]:
    """Every audio file under ``root`` for an allow-listed directory source.

    Speaker/session = the first ``speaker_depth`` path components below ``root``.
    """
    check_source(source)
    if source in DEDICATED_ONLY:
        raise ValueError(
            f"source {source!r} requires per-item licence/quality filtering; use its dedicated CLI flag instead of --dir"
        )
    rootp = Path(root)
    items = []
    for f in sorted(p for p in rootp.rglob("*") if p.suffix.lower() in (".wav", ".flac", ".mp3", ".ogg")):
        rel = f.relative_to(rootp)
        group = "/".join(rel.parts[:speaker_depth]) if len(rel.parts) > speaker_depth else rel.stem
        items.append(
            Item(
                key=_safe_key(source, *rel.with_suffix("").parts),
                source=source,
                path=str(f),
                kind=kind,
                speaker=f"{source}:{group}",
                session=f"{source}:{group}",
                licence=ALLOWLIST[source]["licence"],
            )
        )
    return items


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def assign_split(group: str, ratios: dict[str, float], salt: str = "oto-v1") -> str:
    h = int.from_bytes(hashlib.sha256(f"{salt}:{group}".encode()).digest()[:8], "big") / 2**64
    acc = 0.0
    for name, r in ratios.items():
        acc += r
        if h < acc:
            return name
    return list(ratios)[-1]


def split_items(items: list[Item], ratios: dict[str, float], salt: str = "oto-v1") -> dict[str, list[Item]]:
    """Split by speaker hash (sessions are nested in speakers, so both are disjoint)."""
    out: dict[str, list[Item]] = {k: [] for k in ratios}
    for it in items:
        out[assign_split(it.speaker, ratios, salt)].append(it)
    check_disjoint(out)
    return out


def check_disjoint(splits: dict[str, list[Item]]) -> None:
    for attr in ("speaker", "session", "path"):
        owner: dict[str, str] = {}
        for name, its in splits.items():
            for it in its:
                v = getattr(it, attr)
                if owner.setdefault(v, name) != name:
                    raise AssertionError(f"{attr} {v!r} appears in both {owner[v]} and {name}")


# ---------------------------------------------------------------------------
# Audio + shard writing
# ---------------------------------------------------------------------------


def load_audio(path: str) -> tuple[np.ndarray, int]:
    import soundfile as sf

    x, sr = sf.read(path, dtype="float32", always_2d=True)
    return x.mean(axis=1), sr


def to_48k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return x
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(sr, TARGET_SR)
    return resample_poly(x, TARGET_SR // g, sr // g).astype(np.float32)


def encode_sample(item_dict: dict) -> tuple[str, bytes, bytes] | None:
    """Worker: decode, measure f_max at the source rate, resample to 48 kHz, FLAC-encode."""
    import soundfile as sf

    try:
        x, sr = load_audio(item_dict["path"])
    except Exception as e:  # noqa: BLE001 - corrupt/missing files are skipped and logged
        print(f"skip {item_dict['path']}: {e}", file=sys.stderr)
        return None
    if len(x) < sr * 0.5 or not np.isfinite(x).all():
        return None
    f_max = estimate_bandwidth(x, sr)
    y = to_48k(x, sr)
    peak = float(np.abs(y).max())
    if peak > 1.0:
        y = y / peak
    buf = io.BytesIO()
    sf.write(buf, y, TARGET_SR, format="FLAC", subtype="PCM_16")
    meta = {k: v for k, v in item_dict.items() if k != "path"}
    meta.update(f_max=round(f_max, 1), meta_f_max=item_dict.get("f_max", 0.0), orig_sr=sr,
                duration=round(len(y) / TARGET_SR, 3), sr=TARGET_SR)  # fmt: skip
    return item_dict["key"], buf.getvalue(), json.dumps(meta, sort_keys=True).encode()


class ShardWriter:
    """Minimal WebDataset-compatible tar writer (``{key}.{ext}`` members, deterministic mtime)."""

    def __init__(self, pattern: str, max_count: int = 2000, max_bytes: int = 1 << 30):
        self.pattern, self.max_count, self.max_bytes = pattern, max_count, max_bytes
        self.idx, self.count, self.bytes = 0, 0, 0
        self.tar: tarfile.TarFile | None = None
        self.paths: list[str] = []

    def _open(self) -> None:
        path = self.pattern % self.idx
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.tar = tarfile.open(path, "w")
        self.paths.append(path)
        self.idx += 1
        self.count = self.bytes = 0

    def write(self, key: str, files: dict[str, bytes]) -> None:
        if self.tar is None or self.count >= self.max_count or self.bytes >= self.max_bytes:
            self.close()
            self._open()
        assert self.tar is not None
        for ext, data in files.items():
            ti = tarfile.TarInfo(f"{key}.{ext}")
            ti.size, ti.mtime, ti.mode = len(data), 0, 0o644
            self.tar.addfile(ti, io.BytesIO(data))
            self.bytes += len(data)
        self.count += 1

    def close(self) -> None:
        if self.tar is not None:
            self.tar.close()
            self.tar = None


def pack_split(items: list[Item], out_dir: Path, split: str, kind: str, workers: int, max_count: int) -> int:
    writer = ShardWriter(str(out_dir / split / f"{kind}-{split}-%06d.tar"), max_count=max_count)
    items = list(items)
    random.Random(f"{split}:{kind}").shuffle(items)  # mix speakers inside shards
    n = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(encode_sample, [asdict(i) for i in items], chunksize=16):
            if res is None:
                continue
            key, flac, meta = res
            writer.write(key, {"flac": flac, "json": meta})
            n += 1
    writer.close()
    return n


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------


def write_attribution(items: list[Item], path: Path) -> None:
    by_src: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        by_src[it.source].append(it)
    lines = [
        "# Training data attribution for Oto-1 weights",
        "",
        "Generated by `tools/train/oto_train/data/pack_shards.py`. Every source below is on the permissive",
        "list of `docs/training-data-audit.md`.",
        "",
    ]
    for src in sorted(by_src):
        its = by_src[src]
        lines += [f"## {src}", "", f"- Licence: {ALLOWLIST[src]['licence']}",
                  f"- Source: {ALLOWLIST[src]['citation']}", f"- Clips used: {len(its):,}", ""]  # fmt: skip
        if src in ("fsd50k", "fma"):
            lines += ["| Title | Author | Licence | URL |", "|---|---|---|---|"]
            for i in sorted(its, key=lambda i: i.key):
                t, a = i.title.replace("|", "/"), i.author.replace("|", "/")
                lines.append(f"| {t} | {a} | {i.licence_url or i.licence} | {i.url} |")
            lines.append("")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Spot check (HiFiTTS-2 MP3 top band)
# ---------------------------------------------------------------------------


def spot_check(items: list[Item], n: int, out: Path, seed: int = 0, utts_per_chapter: int = 3) -> dict:
    """Inspect ``n`` random chapters: f_max plus 16-19 kHz and 19-21 kHz energy vs the speech band.

    A hard MP3 low-pass shows as ``f_max`` pinned near 16 / 19 / 20 kHz across a chapter and a
    cliff between the two top bands. Writes ``spot_check.csv`` + summary JSON and, if matplotlib
    is installed, one spectrogram PNG per utterance for listening/visual review.
    """
    chapters: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        chapters[it.session].append(it)
    picked = random.Random(seed).sample(sorted(chapters), min(n, len(chapters)))
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for ch in picked:
        for it in sorted(chapters[ch], key=lambda i: i.key)[:utts_per_chapter]:
            try:
                x, sr = load_audio(it.path)
            except Exception as e:  # noqa: BLE001
                rows.append({"chapter": ch, "key": it.key, "error": str(e)})
                continue
            rows.append({
                "chapter": ch, "key": it.key, "sr": sr, "meta_f_max": it.f_max,
                "f_max": round(estimate_bandwidth(x, sr), 1),
                "db_16_19k": round(band_energy_ratio_db(x, sr, 16_000, 19_000), 1),
                "db_19_21k": round(band_energy_ratio_db(x, sr, 19_000, 21_000), 1),
            })  # fmt: skip
            _maybe_png(x, sr, out / f"{it.key}.png")
    with open(out / "spot_check.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    fm = [r["f_max"] for r in rows if "f_max" in r]
    summary = {
        "chapters": len(picked),
        "utterances": len(fm),
        "errors": sum("error" in r for r in rows),
        "f_max_median": float(np.median(fm)) if fm else 0.0,
        "frac_f_max_ge_16k": float(np.mean([f >= 16_000 for f in fm])) if fm else 0.0,
        "frac_f_max_ge_19k": float(np.mean([f >= 19_000 for f in fm])) if fm else 0.0,
    }
    (out / "spot_check_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _maybe_png(x: np.ndarray, sr: int, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.specgram(x[: sr * 10], NFFT=2048, Fs=sr, noverlap=1536, cmap="magma")
    ax.set_ylim(0, sr / 2)
    fig.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Upload / download helpers (shards and checkpoints)
# ---------------------------------------------------------------------------


def _r2_client():
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def upload_r2(src: str, bucket: str, prefix: str = "") -> None:
    """Upload a file or directory to Cloudflare R2. Env: ``R2_ACCOUNT_ID``, ``R2_ACCESS_KEY_ID``, ``R2_SECRET_ACCESS_KEY``."""
    s3 = _r2_client()
    srcp = Path(src)
    files = [srcp] if srcp.is_file() else sorted(f for f in srcp.rglob("*") if f.is_file())
    for f in files:
        key = prefix + (f.name if srcp.is_file() else f.relative_to(srcp).as_posix())
        s3.upload_file(str(f), bucket, key)
        print(f"r2://{bucket}/{key}")


def download_r2(bucket: str, prefix: str, dst: str) -> None:
    """Mirror ``r2://bucket/prefix`` into ``dst`` (skips files already present with the same size)."""
    s3 = _r2_client()
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            out = Path(dst) / obj["Key"][len(prefix) :].lstrip("/")
            if out.exists() and out.stat().st_size == obj["Size"]:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, obj["Key"], str(out))


def upload_hf(src: str, repo_id: str, path_in_repo: str = "") -> None:
    """Upload to a **private** Hugging Face dataset repo (``HF_TOKEN`` in env or ``hf auth login``)."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    if Path(src).is_file():
        api.upload_file(path_or_fileobj=src, path_in_repo=path_in_repo + Path(src).name, repo_id=repo_id,
                        repo_type="dataset")  # fmt: skip
    elif path_in_repo:
        api.upload_folder(repo_id=repo_id, folder_path=src, path_in_repo=path_in_repo, repo_type="dataset")
    else:
        api.upload_large_folder(repo_id=repo_id, folder_path=src, repo_type="dataset")


def download_hf(repo_id: str, dst: str, pattern: str | None = None) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id, repo_type="dataset", local_dir=dst, allow_patterns=pattern)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_sources(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("HiFiTTS-2")
    g.add_argument("--hifitts2-manifest", nargs="*", default=[])
    g.add_argument("--hifitts2-audio", default="")
    g.add_argument("--hifitts2-min-bw", type=float, default=16_000.0)
    g.add_argument("--hifitts2-speaker-cap-hours", type=float, default=0.5)
    g.add_argument("--hifitts2-path-field", default="audio_filepath")
    g.add_argument("--hifitts2-speaker-field", default="speaker")
    g.add_argument("--hifitts2-bw-field", default="bandwidth")
    g.add_argument("--hifitts2-duration-field", default="duration")
    g.add_argument("--hifitts2-multispk-field", default=None)
    g.add_argument("--hifitts2-allow-unknown-multispk", action="store_true")
    ap.add_argument("--vctk", help="VCTK-Corpus-0.92 root (with license_text.txt)")
    ap.add_argument("--fsd50k", help="FSD50K root (needs *_clips_info_FSD50K.json + ground_truth/)")
    ap.add_argument("--fma-tracks", help="fma_metadata/tracks.csv")
    ap.add_argument("--fma-audio", help="fma_large (or fma_full) audio root")
    ap.add_argument("--dir", action="append", default=[], metavar="SOURCE:KIND:PATH",
                    help="allow-listed directory source, e.g. ace:noise:/data/ACE or oto_rir:rir:/data/rirs")  # fmt: skip
    ap.add_argument("--seed", type=int, default=0)


def _collect(args) -> tuple[list[Item], dict]:
    items: list[Item] = []
    stats: dict = {}
    if args.hifitts2_manifest:
        its, st = collect_hifitts2(
            args.hifitts2_manifest, args.hifitts2_audio, args.hifitts2_min_bw, args.hifitts2_speaker_cap_hours,
            args.hifitts2_path_field, args.hifitts2_speaker_field, args.hifitts2_bw_field,
            args.hifitts2_duration_field, args.hifitts2_multispk_field, args.hifitts2_allow_unknown_multispk,
            args.seed,
        )  # fmt: skip
        items += its
        stats["hifitts2"] = st
    if args.vctk:
        its = collect_vctk(args.vctk)
        items += its
        stats["vctk"] = {"kept": len(its)}
    if args.fsd50k:
        its, st = collect_fsd50k(args.fsd50k)
        items += its
        stats["fsd50k"] = st
    if args.fma_tracks:
        its, st = collect_fma(args.fma_tracks, args.fma_audio)
        items += its
        stats["fma"] = st
    for spec in args.dir:
        src, kind, path = spec.split(":", 2)
        its = collect_dir(src, path, kind)
        items += its
        stats[src] = {"kept": len(its)}
    for it in items:
        check_source(it.source)
    return items, stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("spot-check", help="inspect HiFiTTS-2 chapters for MP3 top-band artefacts")
    _add_sources(sc)
    sc.add_argument("--spot-check", type=int, default=50, help="number of chapters")
    sc.add_argument("--out", required=True)

    pk = sub.add_parser("pack", help="collect, filter, split, pack shards and write ATTRIBUTION.md")
    _add_sources(pk)
    pk.add_argument("--out", required=True)
    pk.add_argument("--split", default="train=0.96,dev=0.02,test=0.02")
    pk.add_argument("--salt", default="oto-v1", help="split hash salt (change only for a new data version)")
    pk.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    pk.add_argument("--shard-size", type=int, default=2000, help="samples per shard")
    pk.add_argument("--dry-run", action="store_true", help="collect + split + attribution only")

    up = sub.add_parser("upload", help="upload shards to R2 or a private HF dataset repo")
    up.add_argument("--src", required=True)
    up.add_argument("--r2-bucket")
    up.add_argument("--prefix", default="")
    up.add_argument("--hf-repo")

    args = ap.parse_args(argv)
    if args.cmd == "upload":
        if args.r2_bucket:
            upload_r2(args.src, args.r2_bucket, args.prefix)
        if args.hf_repo:
            upload_hf(args.src, args.hf_repo, args.prefix)
        return

    items, stats = _collect(args)
    print(json.dumps(stats, indent=2))
    out = Path(args.out)
    if args.cmd == "spot-check":
        speech = [i for i in items if i.source == "hifitts2"]
        print(json.dumps(spot_check(speech, args.spot_check, out, args.seed), indent=2))
        return

    ratios = {k: float(v) for k, v in (kv.split("=") for kv in args.split.split(","))}
    splits = split_items(items, ratios, args.salt)
    out.mkdir(parents=True, exist_ok=True)
    (out / "splits").mkdir(exist_ok=True)
    for name, its in splits.items():
        doc = {"speakers": sorted({i.speaker for i in its}), "sessions": sorted({i.session for i in its}),
               "keys": sorted(i.key for i in its)}  # fmt: skip
        (out / "splits" / f"{name}.json").write_text(json.dumps(doc, indent=0))
    with open(out / "manifest.jsonl", "w") as fh:
        for name, its in splits.items():
            for it in its:
                fh.write(json.dumps({**asdict(it), "split": name}) + "\n")
    write_attribution(items, out / "ATTRIBUTION.md")
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    if args.dry_run:
        return
    t0 = time.time()
    for name, its in splits.items():
        by_kind: dict[str, list[Item]] = defaultdict(list)
        for it in its:
            by_kind[it.kind].append(it)
        for kind, kits in sorted(by_kind.items()):
            n = pack_split(kits, out, name, kind, args.workers, args.shard_size)
            print(f"{name}/{kind}: {n} samples ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
