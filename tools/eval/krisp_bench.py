#!/usr/bin/env python3
"""Krisp Voice Isolation Benchmark: download, manifest, ASR, scoring (docs/plans/0004, Phase 0).

The dataset (HF `Krisp-AI/VoiceIsolation-Benchmark-Dataset`, 265 files, 32 kHz) and Krisp's
published VI 2.5 outputs (`Krisp-AI/VoiceIsolation-Benchmark-Dataset-Processed`) are CC BY-NC 4.0:
local evaluation only, never in the repo, never for training. Everything lives under
`--data-dir` (default `~/.cache/oto-eval/krisp`); a data dir inside the repo is refused.

    python3 tools/eval/krisp_bench.py download --dry-run       # sizes and free disk, nothing written
    python3 tools/eval/krisp_bench.py download                  # raw audio + Krisp VI 2.5 Default (~2 GB)
    python3 tools/eval/krisp_bench.py manifest                  # <data-dir>/manifest.jsonl for eval.py
    python3 tools/eval/krisp_bench.py asr --cond raw --cond vi_2_5_default   # Parakeet (extra: parakeet)
    python3 tools/eval/krisp_bench.py score --cond raw --cond vi_2_5_default --check
    python3 tools/eval/krisp_bench.py reproduce                 # all of the above; exit 1 if off by > 0.2 WER pts
    python3 tools/eval/krisp_bench.py eval --models dfn3,dpdfnet,nearfield   # Oto models through eval.py
    python3 tools/eval/krisp_bench.py score --clips voxlock/bench/results/krisp.clips.jsonl  # re-pool published counts

Download and manifest are stdlib only. `asr` / `reproduce` / `eval` need the `parakeet` extra
(`uv run --project tools/eval --extra parakeet …`). Scoring is in krisp.py.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import krisp  # noqa: E402

REPO = HERE.parent.parent
HF = "https://huggingface.co"
RAW_REPO = "Krisp-AI/VoiceIsolation-Benchmark-Dataset"
PROCESSED_REPO = "Krisp-AI/VoiceIsolation-Benchmark-Dataset-Processed"
VARIANTS = ("vi_2_5_default", "vi_2_5_balanced", "vi_2_5_lite", "vi_2_5_hd")
DEFAULT_DATA_DIR = Path("~/.cache/oto-eval/krisp")
DEFAULT_RESERVE_GB = 20.0
LICENSE = "CC-BY-NC-4.0"
DATASET = "krisp-voice-isolation-benchmark"
USER_AGENT = "oto-eval/0.1 (krisp_bench.py)"


class BenchError(Exception):
    """Usage or environment problem: exit 2."""


def baseline_name(variant: str) -> str:
    """`vi_2_5_default` -> `krisp_default` (the name gates.toml's `[oto1.isolate]` uses)."""
    return "krisp_" + variant.removeprefix("vi_2_5_")


# ------------------------------------------------------------------------------------------
# Data directory


def data_dir(path: str | Path | None) -> Path:
    """The resolved data dir; refuses any path inside the repository."""
    d = Path(path or DEFAULT_DATA_DIR).expanduser().resolve()
    repo = REPO.resolve()
    if d == repo or repo in d.parents:
        raise BenchError(f"{d} is inside the repository; Krisp data is CC BY-NC and must stay outside it (default {DEFAULT_DATA_DIR})")
    return d


def raw_dir(d: Path, scenario: str) -> Path:
    return d / "raw" / scenario


def variant_dir(d: Path, scenario: str, variant: str) -> Path:
    return d / "processed" / scenario / variant


# ------------------------------------------------------------------------------------------
# Download (stdlib urllib, public dataset; $HF_TOKEN is sent if set)


def _request(url: str) -> urllib.request.Request:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    token = os.environ.get("HF_TOKEN")
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    if token and host == "huggingface.co":
        # Unredirected: urllib strips it on redirects so the token is never sent to CDN/other hosts.
        req.add_unredirected_header("Authorization", f"Bearer {token}")
    return req


def hf_tree(repo: str, subdir: str) -> list[dict]:  # pragma: no cover - network
    """Files under `subdir` of a dataset repo: [{"path", "size"}], following pagination."""
    url = f"{HF}/api/datasets/{repo}/tree/main/{urllib.parse.quote(subdir)}?recursive=true"
    out: list[dict] = []
    while url:
        with urllib.request.urlopen(_request(url), timeout=60) as r:
            out += [{"path": x["path"], "size": int(x.get("size", 0))} for x in json.load(r) if x.get("type") == "file"]
            nxt = None
            for part in (r.headers.get("Link") or "").split(","):
                if 'rel="next"' in part:
                    nxt = part[part.find("<") + 1 : part.find(">")]
            url = nxt
    return out


def plan_download(d: Path, scenarios, variants, lister=hf_tree) -> list[dict]:
    """Files to fetch: [{"url", "dest", "size"}], skipping files already present at the right size.
    `.DS_Store` and other non-WAV/JSONL files are ignored."""
    jobs = []
    sources = [(RAW_REPO, s, s, raw_dir(d, s)) for s in scenarios] + [(PROCESSED_REPO, s, f"{s}/{v}", variant_dir(d, s, v)) for s in scenarios for v in variants]
    for repo, scen, sub, dest_dir in sources:
        for f in lister(repo, sub):
            rel = f["path"][len(sub) + 1 :] if f["path"].startswith(sub + "/") else f["path"]
            if "/" in rel or not rel.endswith((".wav", ".jsonl")):
                continue
            dest = dest_dir / rel
            if dest.is_file() and (f["size"] <= 0 or dest.stat().st_size == f["size"]):
                continue
            jobs.append({"url": f"{HF}/datasets/{repo}/resolve/main/{urllib.parse.quote(f['path'])}", "dest": str(dest), "size": f["size"]})
    # metadata of the processed repo is not needed: raw metadata + directory layout identify files
    return [j for j in jobs if not (PROCESSED_REPO in j["url"] and j["dest"].endswith(".jsonl"))]


def check_disk(d: Path, need_bytes: int, reserve_gb: float = DEFAULT_RESERVE_GB, usage=shutil.disk_usage) -> dict:
    """Refuse a download that would leave less than `reserve_gb` free on the data dir's volume."""
    probe = d
    while not probe.exists():
        probe = probe.parent
    u = usage(probe)
    after = u.free - need_bytes
    info = {"volume_free_gb": round(u.free / 1e9, 2), "need_gb": round(need_bytes / 1e9, 2), "free_after_gb": round(after / 1e9, 2), "reserve_gb": reserve_gb}
    if after < reserve_gb * 1e9:
        raise BenchError(f"not enough disk: {info['volume_free_gb']} GB free, download needs {info['need_gb']} GB and {reserve_gb} GB must stay free (--reserve-gb)")
    return info


def fetch(job: dict, opener=urllib.request.urlopen) -> None:  # pragma: no cover - network
    dest = Path(job["dest"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with opener(_request(job["url"]), timeout=120) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    if job["size"] > 0 and part.stat().st_size != job["size"]:
        part.unlink()
        raise BenchError(f"{job['url']}: got {part.stat().st_size if part.exists() else 0} bytes, expected {job['size']}")
    part.replace(dest)


def cmd_download(args, log=print, lister=hf_tree, fetcher=fetch, usage=shutil.disk_usage) -> int:
    d = data_dir(args.data_dir)
    jobs = plan_download(d, args.scenarios, args.variants, lister)
    need = sum(j["size"] for j in jobs)
    info = check_disk(d, need, args.reserve_gb, usage) if not args.dry_run else None
    if args.dry_run:
        probe = d
        while not probe.exists():
            probe = probe.parent
        free = usage(probe).free
        log(f"{len(jobs)} files, {need / 1e9:.2f} GB to {d}; {free / 1e9:.1f} GB free on the volume (reserve {args.reserve_gb} GB)")
        return 0
    log(f"downloading {len(jobs)} files ({need / 1e9:.2f} GB) to {d}; {info['volume_free_gb']} GB free")
    for k, j in enumerate(jobs, 1):
        fetcher(j)
        if k % 50 == 0 or k == len(jobs):
            log(f"  {k}/{len(jobs)}")
    return 0


# ------------------------------------------------------------------------------------------
# Manifest


def load_metadata(d: Path, scenario: str) -> list[dict]:
    p = raw_dir(d, scenario) / "metadata.jsonl"
    if not p.is_file():
        raise BenchError(f"missing {p}; run `krisp_bench.py download` first")
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def build_manifest(d: Path, scenarios=krisp.SCENARIOS, variants=VARIANTS, limit: int | None = None) -> list[dict]:
    """eval.py manifest entries (absolute paths) for every downloaded file. Baselines are the
    Krisp outputs present on disk, named `krisp_<variant>` (krisp_default, …)."""
    entries = []
    for scen in scenarios:
        if not (raw_dir(d, scen) / "metadata.jsonl").is_file():
            continue  # scenario not downloaded
        for m in load_metadata(d, scen):
            wav = raw_dir(d, scen) / m["file_name"]
            if not wav.is_file():
                continue
            stem = Path(m["file_name"]).stem
            e = {
                "id": f"{scen}/{stem}", "noisy": str(wav), "kind": "speech", "transcript": m["full_transcript"],
                "segments": krisp.validate_segments(m.get("segments")), "speaker_id": m["speaker_id"], "scenario": scen,
                "dataset": DATASET, "license": LICENSE,
                "device": m.get("recording_device"), "environment": m.get("environment"),
            }
            base = {baseline_name(v): str(variant_dir(d, scen, v) / Path(m["file_name"]).name) for v in variants if (variant_dir(d, scen, v) / Path(m["file_name"]).name).is_file()}
            if base:
                e["baselines"] = base
            entries.append(e)
    entries.sort(key=lambda e: e["id"])
    if limit:
        entries = entries[:limit]
    if not entries:
        raise BenchError(f"no Krisp audio under {d}; run `krisp_bench.py download` first")
    return entries


def write_manifest(entries: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"# Krisp Voice Isolation Benchmark ({LICENSE}): local evaluation only, never commit or train on it.\n"
    path.write_text(header + "".join(json.dumps(e) + "\n" for e in entries))
    return path


def manifest_path(d: Path, explicit: str | None) -> Path:
    p = Path(explicit).expanduser().resolve() if explicit else d / "manifest.jsonl"
    repo = REPO.resolve()
    if repo in p.parents:
        raise BenchError(f"{p} is inside the repository; keep Krisp manifests outside it")
    return p


def cmd_manifest(args, log=print) -> int:
    d = data_dir(args.data_dir)
    entries = build_manifest(d, args.scenarios, args.variants, args.limit)
    p = write_manifest(entries, manifest_path(d, args.out))
    spk = len({e["speaker_id"] for e in entries})
    log(f"wrote {p}: {len(entries)} files, {spk} speakers, baselines {sorted({b for e in entries for b in e.get('baselines', {})})}")
    return 0


# ------------------------------------------------------------------------------------------
# ASR cache and scoring


def cond_path(e: dict, cond: str) -> str | None:
    """Audio for a condition: `raw` or a Krisp variant (`vi_2_5_default`, or `krisp_default`)."""
    if cond == "raw":
        return e["noisy"]
    name = cond if cond.startswith("krisp_") else baseline_name(cond)
    return (e.get("baselines") or {}).get(name)


def asr_tag(config: dict) -> str:
    return "-".join(str(config.get(k, "")) for k in ("engine", "model", "quantization")).strip("-").replace("/", "_")


def cache_file(d: Path, tag: str, cond: str) -> Path:
    return d / "asr" / tag / f"{cond}.json"


def run_asr(entries: list[dict], conds, transcriber, d: Path, log=print) -> str:
    """Transcribe every condition with timed words into `<data-dir>/asr/<tag>/<cond>.json`."""
    import asr

    tag = asr_tag(transcriber.config)
    for cond in conds:
        cf = cache_file(d, tag, cond)
        cache = json.loads(cf.read_text()) if cf.is_file() else {}
        todo = [e for e in entries if e["id"] not in cache and cond_path(e, cond)]
        for k, e in enumerate(todo, 1):
            text, words = asr.transcribe(transcriber, cond_path(e, cond), True)
            cache[e["id"]] = {"text": text, "words": words or []}
            if k % 20 == 0 or k == len(todo):
                cf.parent.mkdir(parents=True, exist_ok=True)
                cf.write_text(json.dumps(cache))
                log(f"  asr {cond}: {k}/{len(todo)}")
        if not todo:
            log(f"  asr {cond}: cached ({len(cache)} files)")
    return tag


def rows_for(entries: list[dict], cache: dict, normalizer) -> list[dict]:
    return [krisp.clip_row(e["id"], e["scenario"], e["speaker_id"], e["transcript"], cache[e["id"]]["text"], cache[e["id"]]["words"], e["segments"], normalizer) for e in entries if e["id"] in cache]


_COND_ALIASES = {
    **{baseline_name(v): v for v in VARIANTS},
    **{v.removeprefix("vi_2_5_"): v for v in VARIANTS},
    **{f"krisp_{v}": v for v in VARIANTS},
}


def canonical_cond(cond: str) -> str:
    """Normalize aliases (`krisp_default`, `default`, `krisp_vi_2_5_default`) to `vi_2_5_default`."""
    return _COND_ALIASES.get(cond, cond)


def read_clips_jsonl(path: Path, set_name: str = "krisp") -> dict[str, list[dict]]:
    """VoxLock-format per-clip rows: one JSON object per line, {"set", "cond", "clips": [...]}."""
    out: dict[str, list[dict]] = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("set", set_name) == set_name:
                cond = canonical_cond(r["cond"])
                if cond in out:
                    raise BenchError(f"duplicate condition {cond!r} (from {r['cond']!r}) in {path}")
                out[cond] = r["clips"]
    return out


def score_conditions(rows_by_cond: dict[str, list[dict]], reference: str | None = "vi_2_5_default", n_boot: int = 2000) -> dict:
    """Pooled metrics, speaker-bootstrap CIs, per-scenario WER and (vs `reference`) paired deltas."""
    out = {}
    for cond, rows in rows_by_cond.items():
        krisp._check_unique_ids(rows)
        p = krisp.pooled(rows)
        ci = krisp.speaker_bootstrap(rows, n_boot)
        r = {"files": len(rows), "speakers": ci["speakers"], **{k: p[k] for k in krisp.STAT_KEYS}, "ci": {k: ci[k] for k in krisp.STAT_KEYS}}
        r["scenarios"] = {s: krisp.pooled([x for x in rows if x.get("scen") == s]) for s in krisp.SCENARIOS if any(x.get("scen") == s for x in rows)}
        if reference and cond != reference and reference in rows_by_cond:
            r["vs_reference"] = {"reference": reference, **{k: v for k, v in krisp.paired_bootstrap(rows, rows_by_cond[reference], n_boot).items() if k in krisp.STAT_KEYS}}
        out[cond] = r
    return out


def check_published(scores: dict, tolerance: float = krisp.REPRODUCE_TOLERANCE_WER, required=None) -> list[dict]:
    """Compare reproduced pooled WER with the published rows (plan 0004: within 0.2 points).
    Every present (or `required`) published condition must have all 265 files; incomplete or
    missing published conditions fail instead of being silently dropped."""
    norm_scores: dict[str, dict] = {}
    for k, v in scores.items():
        ck = canonical_cond(k)
        if ck in norm_scores:
            raise ValueError(f"duplicate condition {ck!r} (from {k!r}) in scores")
        norm_scores[ck] = v
    conds = list(norm_scores)
    for c in required or ():
        cc = canonical_cond(c)
        if cc not in conds:
            conds.append(cc)
    checks = []
    for cond in conds:
        pub = krisp.PUBLISHED.get(cond)
        if not pub:
            continue
        s = norm_scores.get(cond)
        if not s or s["files"] != 265:
            files = s["files"] if s else 0
            wer = s["wer"] if s else float("nan")
            leak = s["leak"] if s else float("nan")
            checks.append({"cond": cond, "files": files, "wer": wer, "published_wer": pub["wer"], "diff": wer - pub["wer"], "leak": leak, "published_leak": pub["leak"], "ok": False, "reason": f"incomplete ({files}/265 files)"})
            continue
        diff = s["wer"] - pub["wer"]
        checks.append({"cond": cond, "files": s["files"], "wer": s["wer"], "published_wer": pub["wer"], "diff": diff, "leak": s["leak"], "published_leak": pub["leak"], "ok": abs(diff) <= tolerance + 1e-9})
    return checks


def _f(v, d=2):
    return "–" if v is None or v != v else f"{v:.{d}f}"


def format_scores(scores: dict, checks: list[dict] | None = None) -> str:
    L = ["| Condition | Files / spk | WER % [95% CI] | Missed % | Leak w/min [95% CI] | Phone-call WER % | Δ WER vs ref [CI] | Δ leak vs ref [CI] |", "|---|---|---|---|---|---|---|---|"]
    for cond, s in sorted(scores.items(), key=lambda kv: kv[1]["wer"]):
        ci = s["ci"]
        vs = s.get("vs_reference")
        dw = f"{vs['wer'][0]:+.2f} [{vs['wer'][1]:+.2f}, {vs['wer'][2]:+.2f}]" if vs else "–"
        dl = f"{vs['leak'][0]:+.2f} [{vs['leak'][1]:+.2f}, {vs['leak'][2]:+.2f}]" if vs else "–"
        phone = s["scenarios"].get("phone_calls", {}).get("wer")
        L.append(f"| {cond} | {s['files']} / {s['speakers']} | {_f(s['wer'])} [{_f(ci['wer'][0])}, {_f(ci['wer'][1])}] | {_f(s['missed'])} | {_f(s['leak'])} [{_f(ci['leak'][0])}, {_f(ci['leak'][1])}] | {_f(phone)} | {dw} | {dl} |")
    if checks:
        L.append("")
    for c in checks or []:
        if c.get("reason"):
            L.append(f"MISMATCH {c['cond']}: {c['reason']} (WER {_f(c['wer'])} vs published {c['published_wer']:.2f})")
        else:
            L.append(f"{'OK' if c['ok'] else 'MISMATCH'} {c['cond']}: WER {c['wer']:.2f} vs published {c['published_wer']:.2f} ({c['diff']:+.2f} pts); leak {c['leak']:.2f} vs {c['published_leak']:.2f}")
    return "\n".join(L)


def write_report(d: Path, name: str, scores: dict, rows_by_cond: dict, checks, meta: dict) -> Path:
    rep = d / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / f"{name}.json").write_text(json.dumps({"meta": meta, "scores": scores, "checks": checks}, indent=2, default=str) + "\n")
    with open(rep / f"{name}.clips.jsonl", "w") as f:
        for cond, rows in rows_by_cond.items():
            f.write(json.dumps({"set": "krisp", "cond": cond, "clips": rows}) + "\n")
    (rep / f"{name}.md").write_text(format_scores(scores, checks) + "\n")
    return rep / f"{name}.md"


def _make_transcriber(args):
    import asr

    try:
        return asr.make_transcriber(args.asr_backend, args.asr_model, args.asr_quantization)
    except RuntimeError as e:
        raise BenchError(str(e)) from e


def cmd_asr(args, log=print, transcriber=None) -> int:
    d = data_dir(args.data_dir)
    entries = build_manifest(d, args.scenarios, VARIANTS, args.limit)
    run_asr(entries, args.cond, transcriber or _make_transcriber(args), d, log)
    return 0


def cmd_score(args, log=print, transcriber_config: dict | None = None) -> int:
    meta: dict = {}
    if args.clips:
        rows_by_cond = read_clips_jsonl(Path(args.clips))
        if args.cond:
            wanted = [canonical_cond(c) for c in args.cond]
            rows_by_cond = {c: rows_by_cond[c] for c in wanted if c in rows_by_cond}
        d = data_dir(args.data_dir)
        meta["source"] = str(args.clips)
    else:
        d = data_dir(args.data_dir)
        entries = build_manifest(d, args.scenarios, VARIANTS, args.limit)
        config = transcriber_config or {"engine": "onnx-asr" if args.asr_backend == "parakeet" else args.asr_backend, "model": args.asr_model or ("nemo-parakeet-tdt-0.6b-v2" if args.asr_backend == "parakeet" else "small.en"), "quantization": args.asr_quantization or ("fp32" if args.asr_backend == "parakeet" else "")}
        tag = asr_tag(config)
        normalizer = krisp.KrispNormalizer(args.normalizer)
        rows_by_cond = {}
        for cond in args.cond or ["raw", "vi_2_5_default"]:
            canon = canonical_cond(cond)
            if canon in rows_by_cond:
                raise BenchError(f"duplicate condition {canon!r} (from {cond!r})")
            cf = cache_file(d, tag, cond)
            if not cf.is_file() and canon != cond:
                alt = cache_file(d, tag, canon)
                if alt.is_file():
                    cf = alt
            if not cf.is_file():
                raise BenchError(f"no transcripts for {cond} at {cf}; run `krisp_bench.py asr --cond {canon}`")
            rows_by_cond[canon] = rows_for(entries, json.loads(cf.read_text()), normalizer)
        meta.update(asr=config, normalizer=normalizer.backend)
    if not rows_by_cond:
        raise BenchError("nothing to score")
    scores = score_conditions(rows_by_cond, canonical_cond(args.reference) if args.reference else None, args.bootstrap)
    required = [canonical_cond(c) for c in args.cond] if args.cond else (["raw", "vi_2_5_default"] if not args.clips else None)
    checks = check_published(scores, required=required) if args.check else []
    log(format_scores(scores, checks))
    if not args.clips or args.write:
        log(f"wrote {write_report(d, args.name, scores, rows_by_cond, checks, meta)}")
    if checks and not all(c["ok"] for c in checks):
        return 1
    if args.check and not checks:
        log("nothing to check: needs all 265 files of `raw` or a published Krisp variant")
        return 1
    return 0


def cmd_reproduce(args, log=print) -> int:
    rc = cmd_download(args, log)
    if rc:
        return rc
    cmd_manifest(argparse.Namespace(**{**vars(args), "out": None, "limit": None}), log)
    t = _make_transcriber(args)
    args.cond = ["raw", *args.variants]
    cmd_asr(argparse.Namespace(**{**vars(args), "limit": None}), log, t)
    args.check, args.clips, args.limit = True, None, None
    return cmd_score(args, log, t.config)


def eval_command(args, manifest: Path, d: Path) -> list[str]:
    cmd = [sys.executable, str(HERE / "eval.py"), "--suite", "manifest", "--manifest", str(manifest), "--models", args.models, "--strengths", args.strengths, "--asr", "--asr-backend", args.asr_backend, "--gates-profile", args.gates_profile, "--out", str(Path(args.eval_out).expanduser() if args.eval_out else d / "eval-out"), "--bootstrap", str(args.bootstrap), "--krisp-normalizer", args.normalizer]
    if args.asr_model:
        cmd += ["--asr-model", args.asr_model]
    if args.asr_quantization:
        cmd += ["--asr-quantization", args.asr_quantization]
    return cmd + list(args.eval_args or [])


def cmd_eval(args, log=print, runner=subprocess.call) -> int:
    d = data_dir(args.data_dir)
    entries = build_manifest(d, args.scenarios, args.variants, args.limit)
    m = write_manifest(entries, manifest_path(d, args.out))
    cmd = eval_command(args, m, d)
    log("running: " + " ".join(cmd))
    return runner(cmd)


# ------------------------------------------------------------------------------------------
# CLI


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, variants_default=("vi_2_5_default",)):
        sp.add_argument("--data-dir", default=os.environ.get("OTO_KRISP_DATA"), help=f"outside the repo (default $OTO_KRISP_DATA or {DEFAULT_DATA_DIR})")
        sp.add_argument("--scenarios", type=lambda s: [x for x in s.split(",") if x], default=list(krisp.SCENARIOS))
        sp.add_argument("--variants", type=lambda s: [x for x in s.split(",") if x], default=list(variants_default), help=f"Krisp outputs: {','.join(VARIANTS)}")

    def asr_opts(sp):
        sp.add_argument("--asr-backend", choices=["parakeet", "faster-whisper"], default="parakeet")
        sp.add_argument("--asr-model")
        sp.add_argument("--asr-quantization", help="parakeet: int8 for the smaller export (default fp32, as published)")
        sp.add_argument("--normalizer", choices=["auto", "nemo", "stdlib"], default="auto")
        sp.add_argument("--bootstrap", type=int, default=2000)

    def dl_opts(sp):
        sp.add_argument("--reserve-gb", type=float, default=DEFAULT_RESERVE_GB, help="free space that must remain after the download")
        sp.add_argument("--dry-run", action="store_true")

    sp = sub.add_parser("download", help="fetch audio + metadata from Hugging Face")
    common(sp)
    dl_opts(sp)

    sp = sub.add_parser("manifest", help="write an eval.py JSONL manifest")
    common(sp, VARIANTS)
    sp.add_argument("--out", help="manifest path (default <data-dir>/manifest.jsonl)")
    sp.add_argument("--limit", type=int)

    sp = sub.add_parser("asr", help="transcribe conditions with timed words (cached)")
    common(sp)
    asr_opts(sp)
    sp.add_argument("--cond", action="append", required=True, help="raw or a Krisp variant; repeatable")
    sp.add_argument("--limit", type=int)

    sp = sub.add_parser("score", help="pooled WER / missed / leak with speaker-bootstrap CIs")
    common(sp)
    asr_opts(sp)
    sp.add_argument("--cond", action="append", help="default: raw and vi_2_5_default")
    sp.add_argument("--reference", default="vi_2_5_default", help="paired deltas against this condition")
    sp.add_argument("--clips", help="score VoxLock-format per-clip rows instead of the ASR cache")
    sp.add_argument("--check", action="store_true", help="exit 1 unless WER is within 0.2 points of the published rows")
    sp.add_argument("--name", default="krisp", help="report name under <data-dir>/reports")
    sp.add_argument("--write", action="store_true", help="with --clips: also write the report")
    sp.add_argument("--limit", type=int)

    sp = sub.add_parser("reproduce", help="download + manifest + ASR + score --check for raw and Krisp VI 2.5 Default")
    common(sp)
    asr_opts(sp)
    dl_opts(sp)
    sp.add_argument("--reference", default="vi_2_5_default")
    sp.add_argument("--name", default="krisp-reproduce")

    sp = sub.add_parser("eval", help="score Oto models with eval.py on the Krisp manifest (gates: oto1.isolate)")
    common(sp)
    asr_opts(sp)
    sp.add_argument("--models", default="dfn3,dpdfnet,nearfield")
    sp.add_argument("--strengths", default="balanced")
    sp.add_argument("--gates-profile", default="oto1.isolate")
    sp.add_argument("--out", help="manifest path (default <data-dir>/manifest.jsonl)")
    sp.add_argument("--eval-out", help="eval.py --out (default <data-dir>/eval-out)")
    sp.add_argument("--limit", type=int)
    sp.add_argument("eval_args", nargs="*", help="extra eval.py arguments after `--`")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        return {"download": cmd_download, "manifest": cmd_manifest, "asr": cmd_asr, "score": cmd_score, "reproduce": cmd_reproduce, "eval": cmd_eval}[args.cmd](args)
    except BenchError as e:
        print(f"krisp_bench: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
