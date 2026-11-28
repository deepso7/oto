#!/usr/bin/env python3
"""Oto evaluation harness (docs/plans/0002-e2e-roadmap.md, Phase 1 Track A).

    python3 tools/eval/eval.py --suite ci                       # synthetic fixtures, stdlib only
    python3 tools/eval/eval.py --suite manifest --manifest holdout.jsonl [--asr]
    python3 tools/eval/eval.py --bench [--live]                 # oto bench --json into the scorecard

Every run checks MODELS.toml's SHA-256 hashes, writes `report.json` and `scorecard.md` to
--out, and exits 1 if any gate in gates.toml fails (2 on a usage or environment error).
See tools/eval/README.md.
"""

from __future__ import annotations

import argparse
import datetime as _dt
try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None
import hashlib
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fixtures as fx  # noqa: E402
import krisp  # noqa: E402
import metrics as mx  # noqa: E402
from wavio import Audio, read_wav, write_wav  # noqa: E402

REPO = HERE.parent.parent
DEFAULT_MODELS = ["rnnoise", "dfn3", "dpdfnet2_48khz_hr"]
STRENGTHS = ["light", "balanced", "strong"]
# Engine attenuation caps per strength (oto-engine `Strength`); Strong is unlimited, 60 dB here.
# Used only by the oracle processor.
STRENGTH_CAP_DB = {"light": 12.0, "balanced": 24.0, "strong": 60.0}
ENGINE_RATE = 48_000
# `oto process IN OUT`: --no-delay-comp keeps the chain delay so CI can check it against MODELS.toml.
DEFAULT_TEMPLATE = "{oto} process {input} {output} --model {model} --strength {strength} {bypass} --no-delay-comp"


class HarnessError(Exception):
    """A usage or environment problem: exit code 2."""


# ------------------------------------------------------------------------------------------
# Manifest and gates


@dataclass
class ModelSpec:
    key: str
    cli_model: str
    cli_tier: str | None
    frame_samples: int
    latency_samples: int
    sample_rate: int = ENGINE_RATE
    ship_in_release: bool = False
    status: str = ""
    engine_name: str | None = None
    path: str | None = None
    sha256: str | None = None
    extra_delay_tolerance: int = 0
    raw: dict = field(default_factory=dict)

    @property
    def chain_latency_samples(self) -> int:
        return self.frame_samples + self.latency_samples

    @classmethod
    def from_toml(cls, key: str, t: dict) -> "ModelSpec":
        rs = t.get("resampler_latency_ms")
        tol = int(math.ceil(max(rs) * ENGINE_RATE / 1000)) if rs else 0
        return cls(key=key, cli_model=t.get("cli_model", key), cli_tier=t.get("cli_tier"), frame_samples=int(t["frame_samples"]), latency_samples=int(t["latency_samples"]), sample_rate=int(t.get("sample_rate", ENGINE_RATE)), ship_in_release=bool(t.get("ship_in_release", False)), status=t.get("status", ""), engine_name=t.get("engine_name"), path=t.get("path"), sha256=t.get("sha256"), extra_delay_tolerance=tol, raw=t)


def load_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_models(path: Path) -> dict[str, ModelSpec]:
    return {k: ModelSpec.from_toml(k, v) for k, v in load_toml(path).get("models", {}).items()}


def resolve_models(names: list[str], models: dict[str, ModelSpec]) -> list[ModelSpec]:
    out = []
    for n in names:
        m = models.get(n) or next((s for s in models.values() if s.cli_model == n), None)
        if m is None:
            raise HarnessError(f"unknown model {n!r}; MODELS.toml has {', '.join(models)}")
        out.append(m)
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_hashes(models: dict[str, ModelSpec], repo: Path = REPO) -> list[dict]:
    """One row per model. `ok` is False only for a missing file or a mismatch; models
    without a `path` (embedded weights, or not vendored yet) are `skipped`."""
    rows = []
    for m in models.values():
        row = {"model": m.key, "path": m.path, "expected": m.sha256}
        if not m.path or not m.sha256:
            row.update(status="skipped", ok=True, actual=None)
        elif not (repo / m.path).is_file():
            row.update(status="missing-file", ok=False, actual=None)
        else:
            actual = sha256_file(repo / m.path)
            good = actual == m.sha256.lower()
            row.update(actual=actual, ok=good, status="ok" if good else "mismatch")
        rows.append(row)
    return rows


def ci_gates(gates: dict, model: str, strength: str) -> dict:
    """Shared [ci] bars, then [ci.<strength>], then the model's [models.<m>.ci] and
    [models.<m>.ci.<strength>]. Later tables override earlier ones."""
    out: dict = {}
    for table in (gates.get("ci", {}), gates.get("models", {}).get(model, {}).get("ci", {})):
        out.update({k: v for k, v in table.items() if not isinstance(v, dict)})
        out.update(table.get(strength, {}))
    return out


def model_gates(gates: dict, model: str, prefixes: tuple[str, ...]) -> dict:
    t = gates.get("models", {}).get(model, {})
    return {k: v for k, v in t.items() if not isinstance(v, dict) and k.startswith(prefixes)}


def evaluate_gates(metrics: dict, gates: dict) -> list[dict]:
    """`min_X` / `max_X` gates against metric `X_min` / `X_max` if present, else `X`.
    Keys without those prefixes are ignored. Unmeasured metrics are `missing`."""
    checks = []
    for key, threshold in gates.items():
        if key.startswith("min_"):
            op, name = "min", key[4:]
        elif key.startswith("max_"):
            op, name = "max", key[4:]
        else:
            continue
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            continue
        metric = f"{name}_{op}" if f"{name}_{op}" in metrics else name
        value = metrics.get(metric)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            status = "missing"
        elif op == "min":
            status = "pass" if value >= threshold else "fail"
        else:
            status = "pass" if value <= threshold else "fail"
        checks.append({"gate": key, "metric": metric, "op": op, "threshold": threshold, "value": value, "status": status})
    return checks


# ------------------------------------------------------------------------------------------
# Processors


class ProcessError(Exception):
    pass


class Processor:
    name = "base"

    def process(self, model: ModelSpec, strength: str, inp: Path, out: Path, bypass: bool = False, hint: dict | None = None) -> None:
        raise NotImplementedError


class OtoProcessor(Processor):
    """Runs `oto process`. The command line comes from a template so the harness does not
    have to change when the CLI's flags do (--process-template, $OTO_PROCESS_TEMPLATE)."""

    name = "oto"

    def __init__(self, binary: str, template: str = DEFAULT_TEMPLATE, timeout: float = 300.0):
        self.binary = binary
        self.template = template
        self.timeout = timeout

    def command(self, model: ModelSpec, strength: str, inp: Path, out: Path, bypass: bool) -> list[str]:
        fields = {"oto": shlex.quote(self.binary), "model": model.cli_model, "strength": strength, "bypass": "--no-denoise" if bypass else "", "input": shlex.quote(str(inp)), "output": shlex.quote(str(out))}
        return shlex.split(self.template.format(**fields))

    def process(self, model, strength, inp, out, bypass=False, hint=None):
        cmd = self.command(model, strength, inp, out, bypass)
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise ProcessError(f"{shlex.join(cmd)}: {e}") from e
        if r.returncode != 0 or not Path(out).is_file():
            tail = (r.stderr or r.stdout).strip().splitlines()[-5:]
            raise ProcessError(f"{shlex.join(cmd)} exited {r.returncode}: {' | '.join(tail)}")


def _delayed(a: Audio, model: ModelSpec, x) -> list[float]:
    d = round(model.chain_latency_samples * a.rate / ENGINE_RATE)
    return ([0.0] * d + list(x))[: a.frames]


class PassthroughProcessor(Processor):
    """Delays the input by the model's chain latency and changes nothing else, like a
    bypassed chain. Fails every attenuation gate by design."""

    name = "passthrough"

    def process(self, model, strength, inp, out, bypass=False, hint=None):
        a = read_wav(inp)
        write_wav(out, a.rate, _delayed(a, model, a.mono), "float32")


class OracleProcessor(Processor):
    """A perfect denoiser for inputs with a known clean reference: keeps the reference,
    attenuates the rest by the strength's cap and adds the chain latency. Tests the harness."""

    name = "oracle"

    def process(self, model, strength, inp, out, bypass=False, hint=None):
        a = read_wav(inp)
        x = a.mono
        hint = hint or {}
        if not bypass:
            g = 10 ** (-STRENGTH_CAP_DB[strength] / 20)
            ref_path = hint.get("clean") or (Path(inp).parent / hint["reference"] if hint.get("reference") else None)
            if ref_path:
                ref = read_wav(ref_path).mono
                x = [r + (s - r) * g for s, r in zip(x, ref)]
            elif hint.get("kind") in ("noise_only", "silence", "background_only"):
                x = [s * g for s in x]
        write_wav(out, a.rate, _delayed(a, model, x), "float32")


def find_oto(explicit: str | None) -> str | None:
    for c in [explicit, os.environ.get("OTO_BIN"), str(REPO / "target/release/oto"), str(REPO / "target/debug/oto"), shutil.which("oto")]:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return c
    return None


def oto_has(binary: str, subcommand: str) -> bool:
    try:
        r = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return f"oto {subcommand}" in (r.stdout + r.stderr)


def make_processor(args) -> Processor:
    if args.processor == "passthrough":
        return PassthroughProcessor()
    if args.processor == "oracle":
        return OracleProcessor()
    binary = find_oto(args.oto)
    if binary is None:
        raise HarnessError("no oto binary: build it (cargo build --release -p oto-cli), pass --oto or set OTO_BIN")
    if not oto_has(binary, "process"):
        raise HarnessError(f"{binary} has no `oto process` subcommand; rebuild it, or use --processor passthrough|oracle to test the harness")
    return OtoProcessor(binary, args.process_template)


# ------------------------------------------------------------------------------------------
# Delay calibration and the CI suite


def calibrate_delay(proc: Processor, model: ModelSpec, work: Path, align_fixture: dict, fixtures_dir: Path, max_error: int) -> dict:
    """Runs the impulse/step fixture with processing bypassed (--no-denoise keeps the model's
    delay) and compares the measured delay with MODELS.toml. An output already compensated
    for the delay (measured ≈ 0) is accepted and recorded as such."""
    inp = fixtures_dir / align_fixture["path"]
    out = work / f"{model.key}__bypass__{align_fixture['name']}.wav"
    proc.process(model, "balanced", inp, out, bypass=True, hint=align_fixture)
    y = read_wav(out).mono
    expected = model.chain_latency_samples
    tol = max_error + model.extra_delay_tolerance
    search = expected + tol + ENGINE_RATE // 100
    delays = mx.impulse_delays(y, align_fixture["impulses"], search)
    measured = mx.median(delays)
    step = mx.step_delay(y, align_fixture["step"], search)
    if measured is None or step is None or abs(step - measured) > tol:
        return {"expected_samples": expected, "measured_samples": measured, "impulse_delays": delays, "step_delay_samples": step, "delay_error_samples": math.inf, "compensated": False}
    compensated = abs(measured) <= tol < expected
    err = abs(measured) if compensated else abs(measured - expected)
    if max_error < err <= tol:
        err = max_error  # inside the model's declared resampler tolerance
    return {"expected_samples": expected, "measured_samples": measured, "impulse_delays": delays, "step_delay_samples": step, "compensated": compensated, "delay_error_samples": err, "delay_samples_used": round(measured)}


def fixture_metrics(fixture: dict, fixtures_dir: Path, inp: Audio, out: Audio, delay48: int, bypass_out: Audio | None = None, bypass_ref: Audio | None = None) -> dict:
    """Metrics for one processed fixture. `bypass_out` / `bypass_ref` are the same input and
    its clean reference run through the chain with processing bypassed (--no-denoise): the
    80 Hz high-pass and the limiter stay, so `model_attenuation_db` and the SI-SDR figures
    measure the model alone, while `attenuation_db` is what the listener hears."""
    m: dict = {}
    m["nan_count"] = mx.nan_count(out.mono)
    y = [s if math.isfinite(s) else 0.0 for s in out.mono]
    m["clip_count"] = mx.clip_count(y)
    m["peak_dbfs"] = mx.peak_dbfs(y)
    m["length_ratio"] = (len(y) / out.rate) / (inp.frames / inp.rate) if inp.frames else 0.0
    m["output_dbfs"] = mx.rms_dbfs(y)
    delay = round(delay48 * out.rate / ENGINE_RATE)
    kind = fixture["kind"]
    nw = fixture.get("noise_windows")
    if nw and kind in ("noisy", "noise_only"):
        m["attenuation_db"] = mx.attenuation_db(inp.mono, inp.rate, y, out.rate, nw, delay)
        if bypass_out is not None and bypass_out.rate == out.rate:
            m["model_attenuation_db"] = mx.window_level_db(bypass_out.mono, out.rate, nw, delay) - mx.window_level_db(y, out.rate, nw, delay)
    if fixture.get("reference"):
        ref = read_wav(fixtures_dir / fixture["reference"])
        sw = fixture["speech_windows"]
        m["speech_level_change_db"] = mx.window_level_db(y, out.rate, sw, delay) - mx.window_level_db(ref.mono, ref.rate, sw)
        if bypass_out is not None and bypass_ref is not None and bypass_out.rate == bypass_ref.rate == out.rate:
            n = ref.frames * out.rate // ref.rate
            bref = mx.shift(bypass_ref.mono, delay, n)
            m["si_sdr_in_db"] = mx.si_sdr(bref, mx.shift(bypass_out.mono, delay, n))
            m["si_sdr_out_db"] = mx.si_sdr(bref, mx.shift(y, delay, n))
        elif out.rate == ref.rate == inp.rate:
            m["si_sdr_in_db"] = mx.si_sdr(ref.mono, inp.mono)
            m["si_sdr_out_db"] = mx.si_sdr(ref.mono, mx.shift(y, delay, ref.frames))
        if "si_sdr_out_db" in m:
            m["si_sdr_delta_db"] = m["si_sdr_out_db"] - m["si_sdr_in_db"]
        if kind == "noisy" and inp.rate == ENGINE_RATE:
            env = mx.envelope_delay(ref.mono, ref.rate, y, out.rate)
            m["envelope_delay_ms"] = env * 1000
            m["envelope_delay_error_ms"] = abs(env * 1000 - delay48 * 1000 / ENGINE_RATE)
    return m


def aggregate_ci(per: dict[str, dict], calib: dict) -> dict:
    """Per-fixture metrics → the flat names gates.toml uses."""
    a: dict = {}
    vals = list(per.values())
    a["nan_count"] = sum(v.get("nan_count", 0) for v in vals)
    a["clip_count"] = sum(v.get("clip_count", 0) for v in vals)
    a["peak_dbfs"] = max((v.get("peak_dbfs", mx.FLOOR_DB) for v in vals), default=mx.FLOOR_DB)
    lr = [v["length_ratio"] for v in vals if "length_ratio" in v]
    if lr:
        a["length_ratio_min"], a["length_ratio_max"] = min(lr), max(lr)
    a["delay_error_samples"] = calib.get("delay_error_samples")

    def g(fixture, metric):
        return per.get(fixture, {}).get(metric)

    a["silence_output_dbfs"] = g("silence", "output_dbfs")
    a["clean_speech_level_change_db"] = g("clean_speech", "speech_level_change_db")
    noisy = [v for v in (g("noisy_speech_15db", "speech_level_change_db"), g("noisy_speech_5db", "speech_level_change_db")) if v is not None]
    a["noisy_speech_level_change_db"] = min(noisy) if noisy else None
    a["noise_only_attenuation_db"] = g("noise_only", "attenuation_db")
    a["noisy_15db_attenuation_db"] = g("noisy_speech_15db", "attenuation_db")
    a["noisy_5db_attenuation_db"] = g("noisy_speech_5db", "attenuation_db")
    a["hfp_attenuation_db"] = g("hfp_16k_noisy_15db", "attenuation_db")
    a["noise_only_model_attenuation_db"] = g("noise_only", "model_attenuation_db")
    a["noisy_15db_model_attenuation_db"] = g("noisy_speech_15db", "model_attenuation_db")
    a["hfp_si_sdr_delta_db"] = g("hfp_16k_noisy_15db", "si_sdr_delta_db")
    a["envelope_delay_error_ms"] = g("noisy_speech_15db", "envelope_delay_error_ms")
    for snr in (15, 5):
        for k in ("si_sdr_in_db", "si_sdr_out_db", "si_sdr_delta_db"):
            a[f"noisy_{snr}db_{k}"] = g(f"noisy_speech_{snr}db", k)
    return {k: v for k, v in a.items() if v is not None}


def _ensure_fixtures(fixtures_dir: Path) -> list[dict]:
    path = fixtures_dir / "fixtures.json"
    if not path.is_file():
        fx.generate(fixtures_dir)
    return json.loads(path.read_text())["fixtures"]


def _calibrate(proc, model, work, align_fixture, fixtures_dir, max_err) -> dict:
    try:
        return calibrate_delay(proc, model, work, align_fixture, fixtures_dir, max_err)
    except (ProcessError, ValueError, OSError) as e:
        return {"error": f"delay calibration: {e}"}


def run_bypass(proc: Processor, model: ModelSpec, work: Path, fixtures: list[dict], fixtures_dir: Path) -> tuple[dict[str, Audio], list[str]]:
    """Every fixture that has noise or a reference, and each reference, through the bypassed
    chain once per model (strength does not apply). Keyed by fixture-relative path."""
    paths: list[str] = []
    for f in fixtures:
        if f["kind"] in ("noisy", "noise_only", "clean"):
            for p in (f["path"], f.get("reference")):
                if p and p not in paths:
                    paths.append(p)
    done: dict[str, Audio] = {}
    errors = []
    for p in paths:
        out = work / f"{model.key}__bypass__{Path(p).stem}.wav"
        try:
            proc.process(model, "balanced", fixtures_dir / p, out, bypass=True, hint={"name": Path(p).stem, "kind": "bypass"})
            done[p] = read_wav(out)
        except (ProcessError, ValueError, OSError) as e:
            errors.append(f"bypass {p}: {e}")
    return done, errors


def run_ci(models: list[ModelSpec], strengths: list[str], proc: Processor, gates: dict, out_dir: Path, fixtures_dir: Path | None = None, log=print) -> list[dict]:
    fixtures_dir = fixtures_dir or out_dir / "fixtures"
    fixtures = _ensure_fixtures(fixtures_dir)
    align_fixture = next(f for f in fixtures if f["kind"] == "alignment")
    work = out_dir / "processed"
    work.mkdir(parents=True, exist_ok=True)
    max_err = int(gates.get("ci", {}).get("max_delay_error_samples", 2))
    inputs = {f["name"]: read_wav(fixtures_dir / f["path"]) for f in fixtures}
    results = []
    for model in models:
        calib = _calibrate(proc, model, work, align_fixture, fixtures_dir, max_err)
        delay48 = calib.get("delay_samples_used") or 0
        bypass, bypass_errors = run_bypass(proc, model, work, fixtures, fixtures_dir)
        for strength in strengths:
            row = {"suite": "ci", "model": model.key, "cli_model": model.cli_model, "strength": strength, "calibration": calib, "fixtures": {}, "errors": ([calib["error"]] if "error" in calib else []) + bypass_errors}
            for f in fixtures:
                if f["kind"] == "alignment":
                    continue
                out = work / f"{model.key}__{strength}__{f['name']}.wav"
                try:
                    proc.process(model, strength, fixtures_dir / f["path"], out, hint=f)
                    row["fixtures"][f["name"]] = fixture_metrics(f, fixtures_dir, inputs[f["name"]], read_wav(out), delay48, bypass.get(f["path"]), bypass.get(f.get("reference")))
                except (ProcessError, ValueError, OSError) as e:
                    row["errors"].append(f"{f['name']}: {e}")
            row["metrics"] = aggregate_ci(row["fixtures"], calib)
            row["checks"] = evaluate_gates(row["metrics"], ci_gates(gates, model.key, strength))
            if row["errors"]:
                row["checks"].append({"gate": "process", "metric": "errors", "op": "max", "threshold": 0, "value": len(row["errors"]), "status": "fail"})
            log(f"  {model.key:22} {strength:9} {_verdict(row['checks'])}")
            results.append(row)
    return results


# ------------------------------------------------------------------------------------------
# Manifest suite (external local datasets)


def read_manifest(path: Path) -> list[dict]:
    """JSONL. Required: `noisy`. Optional: `id`, `clean`, `interferer`, `transcript`,
    `interferer_transcript`, `kind` (speech | noise_only | background_only), `noise_windows`,
    `speech_windows` (seconds), `dataset`, `license`. Krisp-benchmark style entries add
    `segments` (labelled primary / mix / secondary / noise), `speaker_id`, `scenario` and
    `baselines` ({name: wav}, e.g. Krisp's published outputs, scored with the same ASR).
    Relative paths are relative to the manifest file."""
    entries = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError as err:
            raise HarnessError(f"{path}:{n}: {err}") from err
        if "noisy" not in e:
            raise HarnessError(f"{path}:{n}: missing `noisy`")
        e.setdefault("id", f"line{n}")
        e.setdefault("kind", "speech")
        if e["kind"] not in ("speech", "noise_only", "background_only"):
            raise HarnessError(f"{path}:{n}: unknown kind {e['kind']!r}")
        def resolve(v):
            p = Path(v).expanduser()
            return str(p if p.is_absolute() else (Path(path).parent / p).resolve())

        for k in ("noisy", "clean", "interferer"):
            if e.get(k):
                e[k] = resolve(e[k])
        if e.get("baselines"):
            if not isinstance(e["baselines"], dict):
                raise HarnessError(f"{path}:{n}: `baselines` must be an object of name -> wav")
            e["baselines"] = {str(k): resolve(v) for k, v in e["baselines"].items() if v}
        if e.get("segments") is not None:
            try:
                e["segments"] = krisp.validate_segments(e["segments"])
            except (ValueError, KeyError, TypeError) as err:
                raise HarnessError(f"{path}:{n}: segments: {err}") from err
        entries.append(e)
    return entries


def projection_gain(x, basis) -> float:
    bb = mx.energy(basis)
    return mx._dot(x, basis) / bb if bb > 0 else 0.0


def pesq_score(ref, ref_rate: int, deg, deg_rate: int):
    """Wideband PESQ at 16 kHz, or None without the `perceptual` extra."""
    try:
        import numpy as np
        from pesq import pesq
    except ImportError:
        return None
    try:
        from math import gcd

        from scipy.signal import resample_poly

        def to16(x, r):
            g = gcd(r, 16000)
            return resample_poly(np.asarray(x, dtype=np.float64), 16000 // g, r // g)
    except ImportError:
        def to16(x, r):
            return np.asarray(mx.resample_linear(x, r, 16000))

    a, b = to16(ref, ref_rate), to16(deg, deg_rate)
    n = min(len(a), len(b))
    try:
        return float(pesq(16000, a[:n], b[:n], "wb"))
    except Exception:  # noqa: BLE001 - pesq raises on silent or too-short input
        return None


def _resample_16k_antialiased(x, rate: int):
    """Anti-aliased resampling to 16 kHz for perceptual models (PESQ, DNSMOS)."""
    import numpy as np

    arr = np.asarray(x, dtype=np.float32)
    if rate == 16000 or not len(arr):
        return arr
    try:
        from scipy.signal import resample_poly

        g = math.gcd(rate, 16000)
        return resample_poly(arr, 16000 // g, rate // g).astype(np.float32)
    except ImportError:
        # Frequency-domain anti-aliased resample (brick-wall at min Nyquist) using numpy rfft.
        n_out = max(1, round(len(arr) * 16000 / rate))
        spec = np.fft.rfft(arr)
        out_spec = np.zeros(n_out // 2 + 1, dtype=spec.dtype)
        keep = min(len(spec), len(out_spec))
        out_spec[:keep] = spec[:keep]
        return (np.fft.irfft(out_spec, n=n_out) * (n_out / len(arr))).astype(np.float32)


class Dnsmos:
    """DNSMOS P.835 with the `sig_bak_ovr.onnx` model from microsoft/DNS-Challenge, supplied
    locally (it is not redistributed here). Non-personalised polynomial mapping."""

    POLY = {"sig": (-0.08397278, 1.22083953, 0.0052439), "bak": (-0.13166888, 1.60915514, -0.39604546), "ovr": (-0.06766283, 1.11546468, 0.04602535)}
    LEN = int(9.01 * 16000)

    def __init__(self, model_path: str):
        import numpy as np
        import onnxruntime as ort

        self.np = np
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def __call__(self, x, rate: int) -> dict:  # pragma: no cover - needs the model file
        np = self.np
        a = _resample_16k_antialiased(x, rate)
        if not len(a):
            a = np.zeros(self.LEN, np.float32)
        while len(a) < self.LEN:
            a = np.concatenate([a, a])
        raw = np.mean([self.session.run(None, {self.input: a[s : s + self.LEN][None, :]})[0][0] for s in range(0, len(a) - self.LEN + 1, 16000)], axis=0)
        return {f"dnsmos_{k}": float(np.polyval(self.POLY[k], raw[i])) for i, k in enumerate(("sig", "bak", "ovr"))}


def _krisp_scored(e: dict) -> bool:
    return e["kind"] == "speech" and bool(e.get("transcript")) and e.get("segments") is not None


def _krisp_row(e: dict, text: str, words, normalizer, offset: float = 0.0) -> dict:
    return krisp.clip_row(e["id"], e.get("scenario") or e.get("dataset") or "", e.get("speaker_id") or e["id"], e["transcript"], text, words, e["segments"], normalizer, offset)


def _write_json_atomic(path: Path, updates: dict) -> None:
    """Lock `<path>.lock`, re-read current on-disk JSON, merge `updates`, and atomically replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with open(lock_path, "a+") as lf:
        if fcntl is not None:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            merged: dict = {}
            if path.is_file():
                try:
                    cur = json.loads(path.read_text())
                    if isinstance(cur, dict):
                        merged.update(cur)
                except (OSError, json.JSONDecodeError):
                    pass
            merged.update(updates)
            fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(merged, f)
                os.replace(tmp_name, path)
            except BaseException:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
        finally:
            if fcntl is not None:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


_KRISP_DATASET = "krisp-voice-isolation-benchmark"
_KRISP_BASELINES = {
    "krisp_default": "vi_2_5_default",
    "krisp_balanced": "vi_2_5_balanced",
    "krisp_lite": "vi_2_5_lite",
    "krisp_hd": "vi_2_5_hd",
}


def _valid_words(words) -> list[tuple[str, float, float]] | None:
    if not isinstance(words, list):
        return None
    out: list[tuple[str, float, float]] = []
    for item in words:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            return None
        w, a, b = item
        if not isinstance(w, str) or isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            return None
        out.append((w, float(a), float(b)))
    return out


def _cached_asr_entry(
    entry: dict | None,
    need_words: bool,
    sha256: str | None = None,
    allow_legacy_no_sha256: bool = False,
    reconstruct_words: bool = False,
) -> tuple[str, list | None] | None:
    if not isinstance(entry, dict) or not isinstance(entry.get("text"), str):
        return None
    if sha256 is not None:
        stored_sha = entry.get("sha256")
        if stored_sha is not None:
            if stored_sha != sha256:
                return None
        elif not allow_legacy_no_sha256:
            return None
    raw_words = entry.get("words")
    if raw_words is None:
        if need_words:
            return None
        word_list = None
    else:
        word_list = _valid_words(raw_words)
        if word_list is None:
            return None
    text = " ".join(w[0] for w in word_list) if (reconstruct_words and word_list) else entry["text"]
    return text, word_list


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _asr_tag(config: dict) -> str:
    return "-".join(str(config.get(k, "")) for k in ("engine", "model", "quantization")).strip("-").replace("/", "_")


def _load_asr_cache(cache_root: Path | None, tag: str, *names: str) -> tuple[Path | None, dict, bool]:
    """Load and merge cache files, with earlier names taking precedence; always checkpoint to `names[0]`.
    Returns `(dest_path, merged_cache, merged_dirty)` where `merged_dirty` is True if merging pulled
    entries not yet persisted in `dest_path`."""
    if cache_root is None or not tag:
        return None, {}, False
    base = cache_root / tag
    dest = base / f"{names[0]}.json"
    dest_data = json.loads(dest.read_text()) if dest.is_file() else {}
    merged: dict = {}
    for n in reversed(names):
        p = base / f"{n}.json"
        if p.is_file():
            merged.update(json.loads(p.read_text()))
    dirty = bool(merged) and (not dest.is_file() or any(k not in dest_data for k in merged))
    return dest, merged, dirty


def _baseline_cache(cache_root: Path | None, tag: str, name: str) -> tuple[Path | None, dict, bool, bool]:
    """Return `(dest_path, merged_cache, is_known_krisp, dirty)`. Known Krisp baselines use canonical
    `vi_2_5_<variant>.json` (merging legacy `krisp_<variant>.json` on read); arbitrary names
    go in a dedicated `baselines/` directory with a safe hashed filename."""
    if cache_root is None or not tag:
        return None, {}, False, False
    canon = _KRISP_BASELINES.get(name)
    if canon is not None:
        p, data, dirty = _load_asr_cache(cache_root, tag, canon, name)
        return p, data, True, dirty
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    safe = f"{_safe(name)[:64] or 'baseline'}-{digest}"
    p = cache_root / tag / "baselines" / f"{safe}.json"
    return p, (json.loads(p.read_text()) if p.is_file() else {}), False, False


def _model_cache(cache_root: Path | None, tag: str, model_key: str, strength: str) -> tuple[Path | None, dict]:
    if cache_root is None or not tag:
        return None, {}
    p = cache_root / tag / "models" / f"{_safe(model_key)}__{_safe(strength)}.json"
    return p, (json.loads(p.read_text()) if p.is_file() else {})


def run_manifest(entries: list[dict], models: list[ModelSpec], strengths: list[str], proc: Processor, gates: dict, out_dir: Path, profile: str | None = None, transcriber=None, dnsmos=None, log=print, latency_gates_by_key: dict[tuple[str, str], set[str]] | None = None, normalizer=None, n_boot: int = 2000, asr_cache_dir: Path | None = None) -> list[dict]:
    work = out_dir / "processed"
    work.mkdir(parents=True, exist_ok=True)
    fixtures_dir = out_dir / "fixtures"
    align_fixture = next(f for f in _ensure_fixtures(fixtures_dir) if f["kind"] == "alignment")
    max_err = int(gates.get("ci", {}).get("max_delay_error_samples", 2))

    # Same-ASR baseline: the unprocessed input (downmixed identically to `oto process`),
    # transcribed once with the same configuration.
    # Krisp-style entries (with `segments`) also get timed words for leak scoring, and their
    # `baselines` (e.g. Krisp's published outputs) are transcribed once with the same ASR.
    import asr as _asr

    raw_text: dict[str, str] = {}
    krisp_raw: dict[str, dict] = {}
    krisp_base: dict[str, dict[str, dict]] = {}
    asr_cfg = getattr(transcriber, "config", {}) if transcriber else {}
    asr_tag = _asr_tag(asr_cfg) if transcriber else ""
    reconstruct_words = asr_cfg.get("engine") == "onnx-asr"
    if transcriber:
        normalizer = normalizer or krisp.KrispNormalizer()
        raw_cf, raw_cache, raw_dirty = _load_asr_cache(asr_cache_dir, asr_tag, "raw")
        raw_updates: dict = dict(raw_cache) if raw_dirty else {}
        base_caches: dict[str, tuple[Path | None, dict, bool, dict]] = {}
        for idx, e in enumerate(entries, 1):
            scored = _krisp_scored(e)
            cid = e["id"]
            is_krisp_ds = e.get("dataset") == _KRISP_DATASET
            raw_sha = _file_sha256(Path(e["noisy"])) if raw_cf is not None else None
            hit = _cached_asr_entry(
                raw_cache.get(cid),
                need_words=scored,
                sha256=raw_sha,
                allow_legacy_no_sha256=is_krisp_ds,
                reconstruct_words=reconstruct_words,
            ) if raw_cf is not None else None
            if hit is not None:
                raw_text[cid], words = hit
            else:
                inp_audio = read_wav(e["noisy"])
                src = e["noisy"]
                if len(inp_audio.channels) != 1:
                    src = work / f"raw__{_safe(cid)}.wav"
                    write_wav(src, inp_audio.rate, inp_audio.mono, "float32")
                raw_text[cid], words = _asr.transcribe(transcriber, src, scored)
                if raw_cf is not None:
                    entry_out = {"text": raw_text[cid], "words": list(words) if words is not None else None, "sha256": raw_sha}
                    raw_cache[cid] = entry_out
                    raw_updates[cid] = entry_out
            if scored:
                krisp_raw[cid] = _krisp_row(e, raw_text[cid], words, normalizer)
                for name, path in (e.get("baselines") or {}).items():
                    if name not in base_caches:
                        bcf, bc, is_known_krisp, bdirty = _baseline_cache(asr_cache_dir, asr_tag, name)
                        base_caches[name] = (bcf, bc, is_known_krisp, dict(bc) if bdirty else {})
                    bcf, bc, is_known_krisp, bupdates = base_caches[name]
                    try:
                        b_sha = _file_sha256(Path(path)) if bcf is not None else None
                    except OSError as err:
                        log(f"  baseline {name} {cid}: {err}")
                        continue
                    bhit = _cached_asr_entry(
                        bc.get(cid),
                        need_words=True,
                        sha256=b_sha,
                        allow_legacy_no_sha256=(is_krisp_ds and is_known_krisp),
                        reconstruct_words=reconstruct_words,
                    ) if bcf is not None else None
                    if bhit is not None:
                        text, bwords = bhit
                    else:
                        try:
                            text, bwords = _asr.transcribe(transcriber, path, True)
                        except (ValueError, OSError) as err:
                            log(f"  baseline {name} {cid}: {err}")
                            continue
                        if bcf is not None:
                            b_entry = {"text": text, "words": list(bwords) if bwords is not None else None, "sha256": b_sha}
                            bc[cid] = b_entry
                            bupdates[cid] = b_entry
                    krisp_base.setdefault(cid, {})[name] = _krisp_row(e, text, bwords, normalizer)
            if idx % 20 == 0 or idx == len(entries):
                if raw_updates and raw_cf is not None:
                    _write_json_atomic(raw_cf, raw_updates)
                    raw_updates.clear()
                for bcf, _bc, _is_k, bupdates in base_caches.values():
                    if bupdates and bcf is not None:
                        _write_json_atomic(bcf, bupdates)
                        bupdates.clear()

    # DNSMOS of each baseline (e.g. Krisp NC captures), once per clip, for paired deltas.
    base_dnsmos: dict[str, dict[str, dict]] = {}
    if dnsmos is not None:
        for e in entries:
            for name, path in (e.get("baselines") or {}).items():
                b = read_wav(path)
                base_dnsmos.setdefault(e["id"], {})[name] = dnsmos([s if math.isfinite(s) else 0.0 for s in b.mono], b.rate)

    profile_gates = _lookup(gates, profile) if profile else {}
    if profile and not profile_gates:
        raise HarnessError(f"no gates table {profile!r} in gates.toml")
    results = []
    for model in models:
        calib = _calibrate(proc, model, work, align_fixture, fixtures_dir, max_err)
        delay48 = calib.get("delay_samples_used") or 0
        for strength in strengths:
            clips, errors = [], [calib["error"]] if "error" in calib else []
            m_cf, m_cache = _model_cache(asr_cache_dir, asr_tag, model.key, strength) if transcriber else (None, {})
            m_updates: dict = {}
            for idx, e in enumerate(entries, 1):
                out = work / f"{model.key}__{strength}__{_safe(e['id'])}.wav"
                try:
                    proc.process(model, strength, Path(e["noisy"]), out, hint=e)
                    digest = _file_sha256(out) if m_cf is not None else None
                    mhit = _cached_asr_entry(m_cache.get(e["id"]), need_words=_krisp_scored(e), sha256=digest, reconstruct_words=reconstruct_words) if m_cf is not None else None
                    c = clip_metrics(e, read_wav(e["noisy"]), read_wav(out), delay48, out, transcriber, raw_text.get(e["id"]), dnsmos, normalizer, cached_asr=mhit)
                    if m_cf is not None and mhit is None and "hyp" in c:
                        w_out = c.pop("_asr_words", None)
                        m_entry = {"text": c["hyp"], "words": list(w_out) if w_out is not None else None, "sha256": digest}
                        m_cache[e["id"]] = m_entry
                        m_updates[e["id"]] = m_entry
                    else:
                        c.pop("_asr_words", None)
                    if e["id"] in krisp_raw:
                        c["krisp_raw"] = krisp_raw[e["id"]]
                    if e["id"] in krisp_base:
                        c["krisp_baselines"] = krisp_base[e["id"]]
                    if e["id"] in base_dnsmos:
                        c["baseline_dnsmos"] = base_dnsmos[e["id"]]
                    c["speaker_id"] = e.get("speaker_id") or e["id"]
                    clips.append(c)
                except (ProcessError, ValueError, OSError) as err:
                    errors.append(f"{e['id']}: {err}")
                if idx % 20 == 0 or idx == len(entries):
                    if m_updates and m_cf is not None:
                        _write_json_atomic(m_cf, m_updates)
                        m_updates.clear()
                    if len(entries) >= 20:
                        log(f"    {model.key}/{strength}: {idx}/{len(entries)}")
            agg = aggregate_manifest(clips, n_boot)
            agg["delay_error_samples"] = calib.get("delay_error_samples")
            # Always enforce baseline output & calibration safety checks alongside any quality profile.
            ci_safety = {k: v for k, v in gates.get("ci", {}).items() if k in ("max_nan_count", "max_clip_count", "max_delay_error_samples")}
            row_gates = dict(ci_safety)
            row_gates.update(profile_gates)
            # Latency/onset gates are evaluated once by matching (model, strength) --latency-report rows.
            if latency_gates_by_key and (model.key, strength) in latency_gates_by_key:
                for k in latency_gates_by_key[(model.key, strength)]:
                    row_gates.pop(k, None)
            checks = evaluate_gates(agg, row_gates)
            if errors:
                checks.append({"gate": "process", "metric": "errors", "op": "max", "threshold": 0, "value": len(errors), "status": "fail"})
            row = {"suite": "manifest", "model": model.key, "cli_model": model.cli_model, "strength": strength, "calibration": calib, "metrics": agg, "clips": clips, "errors": errors, "checks": checks, "gate_profile": profile}
            log(f"  {model.key:22} {strength:9} {_verdict(checks)}")
            results.append(row)
    return results


def clip_metrics(e: dict, inp: Audio, out: Audio, delay48: int, out_path: Path, transcriber=None, raw_text=None, dnsmos=None, normalizer=None, cached_asr: tuple[str, list | None] | None = None) -> dict:
    y = [s if math.isfinite(s) else 0.0 for s in out.mono]
    delay = round(delay48 * out.rate / ENGINE_RATE)
    duration = inp.frames / inp.rate
    c: dict = {"id": e["id"], "kind": e["kind"], "dataset": e.get("dataset"), "duration_s": duration, "nan_count": mx.nan_count(out.mono), "clip_count": mx.clip_count(y), "output_dbfs": mx.rms_dbfs(y)}
    nw = e.get("noise_windows") or ([(min(fx.WARMUP, duration / 2), duration)] if e["kind"] in ("noise_only", "background_only") else None)
    if nw:
        c["attenuation_db"] = mx.attenuation_db(inp.mono, inp.rate, y, out.rate, nw, delay)
    clean_ref = None
    if e.get("clean"):
        ref = read_wav(e["clean"])
        clean_ref = ref
        if ref.rate == out.rate == inp.rate:
            aligned = mx.shift(y, delay, ref.frames)
            c["si_sdr_in_db"] = mx.si_sdr(ref.mono, inp.mono)
            c["si_sdr_out_db"] = mx.si_sdr(ref.mono, aligned)
            c["si_sdr_delta_db"] = c["si_sdr_out_db"] - c["si_sdr_in_db"]
            c["snr_out_db"] = mx.snr_db(ref.mono, aligned)
        p = pesq_score(ref.mono, ref.rate, mx.shift(y, delay), out.rate)
        if p is not None:
            c["pesq_wb"] = p
    if e.get("interferer"):
        itf = read_wav(e["interferer"])
        if itf.rate == out.rate == inp.rate:
            clean_mono = clean_ref.mono if (clean_ref is not None and clean_ref.rate == itf.rate) else None
            leak = mx.overlap_leakage_db(inp.mono, mx.shift(y, delay, itf.frames), itf.mono, itf.rate, clean=clean_mono)
            if leak is not None:
                c["overlap_leakage_db"] = leak
    if e["kind"] == "speech":
        c.update(mx.high_band_metrics(inp.mono, inp.rate, y, out.rate))
    if dnsmos is not None:
        c.update(dnsmos(y, out.rate))
    if transcriber is not None:
        import asr

        if cached_asr is not None:
            hyp, words = cached_asr
        else:
            hyp, words = asr.transcribe(transcriber, out_path, _krisp_scored(e))
            c["_asr_words"] = words
        c["hyp"], c["raw_hyp"] = hyp, raw_text
        if _krisp_scored(e):
            # Word times are in the output's timeline: undo the chain delay before matching segments.
            c["krisp"] = _krisp_row(e, hyp, words, normalizer or krisp.KrispNormalizer(), delay48 / ENGINE_RATE)
        if e.get("transcript") and e["kind"] == "speech":
            ref_words = asr.normalize(e["transcript"])
            c["ref_words"] = len(ref_words)
            c["errors"] = asr.align(ref_words, asr.normalize(hyp)).errors
            c["raw_errors"] = asr.align(ref_words, asr.normalize(raw_text or "")).errors
            c["wer"] = asr.wer(e["transcript"], hyp)
            c["first_word_deleted"] = asr.first_word_deleted(e["transcript"], hyp)
            if e.get("interferer_transcript"):
                c["wrong_speaker"] = asr.wrong_speaker(hyp, e["transcript"], e["interferer_transcript"])
        if e["kind"] == "background_only":
            c["leaked_words_per_min"] = asr.words_per_minute(hyp, duration)
        if e["kind"] == "noise_only":
            c["hallucinated_words_per_min"] = asr.words_per_minute(hyp, duration)
            c["raw_hallucinated_words_per_min"] = asr.words_per_minute(raw_text or "", duration)
    return c


def aggregate_manifest(clips: list[dict], n_boot: int = 2000) -> dict:
    a: dict = {"clips": len(clips)}

    def mean(key):
        v = [c[key] for c in clips if c.get(key) is not None and math.isfinite(c[key])]
        return sum(v) / len(v) if v else None

    for k in ("si_sdr_in_db", "si_sdr_out_db", "si_sdr_delta_db", "snr_out_db", "pesq_wb", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovr", "overlap_leakage_db", "input_high_band_16k_db", "high_band_16k_db", "high_band_16k_delta_db"):
        a[k] = mean(k)
    a["quiet_attenuation_db"] = mean("attenuation_db")
    a["nan_count"] = sum(c["nan_count"] for c in clips)
    a["clip_count"] = sum(c["clip_count"] for c in clips)
    scored = [c for c in clips if "ref_words" in c]
    if scored:
        words = sum(c["ref_words"] for c in scored) or 1
        a["target_wer"] = sum(c["errors"] for c in scored) / words
        a["raw_wer"] = sum(c["raw_errors"] for c in scored) / words
        a["target_wer_delta"] = a["target_wer"] - a["raw_wer"]
        a["first_word_deletion_rate"] = sum(c["first_word_deleted"] for c in scored) / len(scored)
    ws = [c for c in clips if "wrong_speaker" in c]
    if ws:
        a["wrong_speaker_rate"] = sum(c["wrong_speaker"] for c in ws) / len(ws)

    def per_min(key, kind):
        sel = [c for c in clips if c["kind"] == kind and key in c]
        minutes = sum(c["duration_s"] for c in sel) / 60
        return sum(c[key] * c["duration_s"] / 60 for c in sel) / minutes if minutes else None

    a["background_leaked_words_per_min"] = per_min("leaked_words_per_min", "background_only")
    a["noise_only_hallucinated_words_per_min"] = per_min("hallucinated_words_per_min", "noise_only")
    a["raw_noise_only_hallucinated_words_per_min"] = per_min("raw_hallucinated_words_per_min", "noise_only")
    a.update(aggregate_krisp(clips, n_boot))
    a.update(aggregate_baseline_dnsmos(clips, n_boot))
    return {k: v for k, v in a.items() if v is not None}


def aggregate_baseline_dnsmos(clips: list[dict], n_boot: int = 2000) -> dict:
    """`dnsmos_<k>_delta_vs_<baseline>` (mean of per-clip Oto − baseline) with `_ci_lo` /
    `_ci_hi` from a speaker-clustered bootstrap, for every baseline scored with DNSMOS."""
    a: dict = {}
    names = sorted({n for c in clips for n in (c.get("baseline_dnsmos") or {})})
    for name in names:
        for k in ("sig", "bak", "ovr"):
            key = f"dnsmos_{k}"
            sel = [c for c in clips if key in c and key in (c.get("baseline_dnsmos") or {}).get(name, {})]
            if not sel:
                continue
            point, lo, hi = krisp.cluster_mean_ci([c[key] - c["baseline_dnsmos"][name][key] for c in sel], [c.get("speaker_id", c["id"]) for c in sel], n_boot)
            base = f"{key}_delta_vs_{name}"
            a[base], a[base + "_ci_lo"], a[base + "_ci_hi"] = point, lo, hi
    return a


def aggregate_krisp(clips: list[dict], n_boot: int = 2000) -> dict:
    """Krisp benchmark metrics (krisp.py): pooled WER, missed caller words, secondary-segment
    leak and noise-segment hallucinations with speaker-bootstrap CIs, for the model (no
    prefix), the raw input (`raw_`) and each baseline (`<name>_`), plus paired
    `<metric>_delta_vs_raw` / `_delta_vs_<baseline>` intervals (model minus reference)."""
    rows = [c["krisp"] for c in clips if "krisp" in c]
    if not rows:
        return {}
    a = krisp.summary_metrics(rows, n_boot=n_boot)
    raw = [c["krisp_raw"] for c in clips if "krisp" in c and "krisp_raw" in c]
    refs: dict[str, list[dict]] = {"raw": raw} if raw else {}
    for c in clips:
        if "krisp" in c:
            for name, row in (c.get("krisp_baselines") or {}).items():
                refs.setdefault(name, []).append(row)
    for name, ref_rows in refs.items():
        a.update(krisp.summary_metrics(ref_rows, prefix=f"{name}_", n_boot=n_boot))
        a.update(krisp.delta_metrics(rows, ref_rows, name, n_boot=n_boot))
    a["krisp_references"] = sorted(refs)
    return a


# ------------------------------------------------------------------------------------------
# Bench and fault-injection reports


BENCH_KEYS = {
    "bench_p50_us": ("p50_us",),
    "bench_p99_us": ("p99_us",),
    "bench_p999_us": ("p999_us", "p99_9_us"),
    "bench_max_us": ("max_us",),
    "bench_realtime_factor": ("realtime_factor", "rtf"),
    "bench_cpu_percent": ("cpu_percent",),
    "bench_latency_samples": ("latency_samples",),
}
STREAMS = ("capture", "output", "worker")


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _last_json_object(text: str) -> dict:
    for line in reversed(text.strip().splitlines()):
        if line.strip().startswith("{"):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"no JSON object in bench output: {text[:200]!r}") from e
    if not isinstance(obj, dict):
        raise ValueError("bench JSON is not an object")
    return obj


def parse_bench_json(text: str) -> dict:
    """Metrics from the last JSON object printed by `oto bench --json` (offline: one chain call
    per 10 ms block) or `oto bench --live --json` (the real engine: per-stream callback timing
    under `stats.capture|output|worker`, deadline misses, starvations, overflows, xruns).
    The parsed object is kept under `raw`."""
    obj = _last_json_object(text)
    out: dict = {"raw": obj}
    stats = obj.get("stats")
    if isinstance(stats, dict):
        out["live_running"] = bool(stats.get("running"))
        if _num(obj.get("cpu_percent")):
            out["bench_cpu_percent"] = obj["cpu_percent"]
        misses = 0
        active_streams = 0
        misses_complete = True
        for stream in STREAMS:
            c = stats.get(stream)
            if not isinstance(c, dict) or not (_num(c.get("callbacks")) and c["callbacks"] > 0):
                continue
            active_streams += 1
            for k in ("p50_us", "p99_us", "p999_us", "max_us", "period_us", "callbacks"):
                if _num(c.get(k)):
                    out[f"live_{stream}_{k}"] = c[k]
            if _num(c.get("deadline_misses")):
                misses += int(c["deadline_misses"])
            else:
                misses_complete = False
        if active_streams > 0 and misses_complete:
            out["live_deadline_misses"] = misses
        for k in ("starvations", "worker_overflows", "underruns", "resyncs", "overflows"):
            if _num(stats.get(k)):
                out[f"live_{k}"] = stats[k]
        if _num(stats.get("stream_starts")) and stats["stream_starts"] > 0:
            out["live_restarts"] = max(0, int(stats["stream_starts"]) - 1)
        if _num(stats.get("estimated_latency_ms")):
            out["live_estimated_latency_ms"] = stats["estimated_latency_ms"]
        out["live_model"] = stats.get("model")
        if isinstance(obj.get("errors"), list):
            out["live_errors"] = len(obj["errors"])
        if isinstance(obj.get("notices"), list):
            out["live_fallbacks"] = len(obj["notices"])
        return out
    for name, keys in BENCH_KEYS.items():
        for k in keys:
            if _num(obj.get(k)):
                out[name] = obj[k]
                break
    return out


# Hard bars for every live run: a run with an engine error, capture-ring overflow, stream
# restart, or silent model fallback measures the wrong thing (plan 0001).
LIVE_HARD_GATES = {
    "max_live_errors": 0,
    "max_live_fallbacks": 0,
    "max_live_underruns": 0,
    "max_live_resyncs": 0,
    "max_live_overflows": 0,
    "max_live_worker_overflows": 0,
    "max_live_restarts": 0,
}

LIVE_REQUIRED_COUNTERS = (
    "live_deadline_misses",
    "live_starvations",
    "live_overflows",
    "live_underruns",
    "live_resyncs",
    "live_worker_overflows",
    "live_restarts",
    "live_errors",
    "live_fallbacks",
)


def bench_command(binary: str, m: ModelSpec, seconds: float, live: bool) -> list[str]:
    if live:
        if not m.cli_tier:
            raise ValueError(f"{m.key} has no cli_tier in MODELS.toml; `oto bench --live` runs a --tier")
        return [binary, "bench", "--live", "--tier", m.cli_tier, "--seconds", str(seconds), "--json"]
    return [binary, "bench", "--model", m.cli_model, "--seconds", str(seconds), "--json"]


def bench_checks(metrics: dict, m: ModelSpec, gates: dict, live: bool) -> list[dict]:
    if live:
        g = model_gates(gates, m.key, ("max_live_", "max_bench_cpu_percent"))
        g.update(LIVE_HARD_GATES)
        checks = evaluate_gates(metrics, g)
        # Missing required counters or inactive/wrong-model streams must fail, not pass as "missing".
        for c in checks:
            if c["status"] == "missing" and c["metric"] in LIVE_REQUIRED_COUNTERS:
                c["status"] = "fail"
        if not metrics.get("live_running"):
            checks.append({"gate": "live_running", "metric": "live_running", "op": "min", "threshold": 1, "value": 0, "status": "fail"})
        expected_model = m.engine_name
        got_model = metrics.get("live_model")
        if expected_model and got_model != expected_model:
            checks.append({"gate": "live_model", "metric": "live_model", "op": "eq", "threshold": expected_model, "value": got_model, "status": "fail"})
        for stream in ("capture", "output"):
            if (metrics.get(f"live_{stream}_callbacks") or 0) <= 0:
                checks.append({"gate": f"live_{stream}_active", "metric": f"live_{stream}_callbacks", "op": "min", "threshold": 1, "value": metrics.get(f"live_{stream}_callbacks", 0), "status": "fail"})
        if m.raw.get("placement") == "worker" and (metrics.get("live_worker_callbacks") or 0) <= 0:
            checks.append({"gate": "live_worker_active", "metric": "live_worker_callbacks", "op": "min", "threshold": 1, "value": metrics.get("live_worker_callbacks", 0), "status": "fail"})
        return checks
    g = model_gates(gates, m.key, ("min_bench_", "max_bench_"))
    if "bench_latency_samples" in metrics:
        metrics["bench_chain_latency_error_samples"] = abs(metrics["bench_latency_samples"] - m.chain_latency_samples)
        g["max_bench_chain_latency_error_samples"] = 0
    return evaluate_gates(metrics, g)


def _resolve_engine_model(name: str | None, all_models: dict[str, ModelSpec]) -> ModelSpec | None:
    if not name:
        return None
    for spec in all_models.values():
        if name in (spec.engine_name, spec.key, spec.cli_model):
            return spec
    return None


def _latency_metric_for_platform(name: str | None) -> str | None:
    if not isinstance(name, str):
        return None
    p = name.strip().lower()
    if p in ("macos", "darwin"):
        return "macos_e2e_latency_ms"
    if p == "linux":
        return "linux_e2e_latency_ms"
    return None


def evaluate_latency_report(obj: dict, all_models: dict[str, ModelSpec], gates: dict, profile: str | None = None) -> dict:
    """Validates a saved `oto latency --stimulus speech --json` report and checks its E2E
    latency ceiling, speech-onset miss rate, and live stream counters against gates.toml."""
    if not isinstance(obj, dict):
        raise ValueError("latency report JSON must be an object")
    reported_platform = obj.get("platform")
    lat_metric = _latency_metric_for_platform(reported_platform)
    reported_strength = obj.get("strength")

    stats = obj.get("stats") if isinstance(obj.get("stats"), dict) else {}
    model_name = stats.get("model")
    spec = _resolve_engine_model(model_name, all_models)
    model_key = spec.key if spec else (str(model_name) if model_name else "unknown")

    metrics: dict = {
        "platform": reported_platform,
        "stimulus": obj.get("stimulus"),
        "strength": reported_strength,
        "denoise": obj.get("denoise"),
        "expected": obj.get("expected"),
        "detected": obj.get("detected"),
        "missed": obj.get("missed"),
        "ok": bool(obj.get("ok")),
        "live_running": bool(stats.get("running")),
        "live_model": model_name,
    }
    if isinstance(obj.get("errors"), list):
        metrics["live_errors"] = len(obj["errors"])
    if isinstance(obj.get("notices"), list):
        metrics["live_fallbacks"] = len(obj["notices"])
    if _num(obj.get("median_ms")):
        metrics["median_ms"] = obj["median_ms"]
        if lat_metric:
            metrics[lat_metric] = obj["median_ms"]
    if _num(obj.get("p95_ms")):
        metrics["p95_ms"] = obj["p95_ms"]
    if _num(obj.get("miss_rate")):
        metrics["speech_onset_miss_rate"] = obj["miss_rate"]
    misses = 0
    active_streams = 0
    misses_complete = True
    for stream in STREAMS:
        c = stats.get(stream)
        if isinstance(c, dict) and _num(c.get("callbacks")) and c["callbacks"] > 0:
            active_streams += 1
            metrics[f"live_{stream}_callbacks"] = c["callbacks"]
            if _num(c.get("deadline_misses")):
                misses += int(c["deadline_misses"])
            else:
                misses_complete = False
    if stats:
        if active_streams > 0 and misses_complete:
            metrics["live_deadline_misses"] = misses
        if _num(stats.get("stream_starts")) and stats["stream_starts"] > 0:
            metrics["live_restarts"] = max(0, int(stats["stream_starts"]) - 1)
        for k in ("starvations", "worker_overflows", "underruns", "resyncs", "overflows"):
            if _num(stats.get(k)):
                metrics[f"live_{k}"] = stats[k]

    checks: list[dict] = []
    if lat_metric is None:
        checks.append({"gate": "latency_platform", "metric": "platform", "op": "eq", "threshold": "macos|linux", "value": reported_platform, "status": "fail"})
    if obj.get("stimulus") != "speech":
        checks.append({"gate": "latency_stimulus", "metric": "stimulus", "op": "eq", "threshold": "speech", "value": obj.get("stimulus"), "status": "fail"})
    if reported_strength not in STRENGTHS:
        checks.append({"gate": "latency_strength", "metric": "strength", "op": "eq", "threshold": "|".join(STRENGTHS), "value": reported_strength, "status": "fail"})
    if obj.get("denoise") is not True:
        checks.append({"gate": "latency_denoise", "metric": "denoise", "op": "eq", "threshold": True, "value": obj.get("denoise"), "status": "fail"})
    if spec is None:
        checks.append({"gate": "latency_model", "metric": "live_model", "op": "eq", "threshold": "known model", "value": model_name, "status": "fail"})
    if not metrics["live_running"]:
        checks.append({"gate": "live_running", "metric": "live_running", "op": "min", "threshold": 1, "value": 0, "status": "fail"})
    if not metrics["ok"]:
        checks.append({"gate": "latency_ok", "metric": "ok", "op": "min", "threshold": 1, "value": 0, "status": "fail"})
    if not (_num(obj.get("expected")) and obj["expected"] > 0):
        checks.append({"gate": "latency_expected", "metric": "expected", "op": "min", "threshold": 1, "value": obj.get("expected"), "status": "fail"})
    for stream in ("capture", "output"):
        if (metrics.get(f"live_{stream}_callbacks") or 0) <= 0:
            checks.append({"gate": f"live_{stream}_active", "metric": f"live_{stream}_callbacks", "op": "min", "threshold": 1, "value": metrics.get(f"live_{stream}_callbacks", 0), "status": "fail"})
    if spec is not None and spec.raw.get("placement") == "worker" and (metrics.get("live_worker_callbacks") or 0) <= 0:
        checks.append({"gate": "live_worker_active", "metric": "live_worker_callbacks", "op": "min", "threshold": 1, "value": metrics.get("live_worker_callbacks", 0), "status": "fail"})

    g: dict = dict(LIVE_HARD_GATES)
    if spec is not None:
        g.update(model_gates(gates, spec.key, ("max_live_",)))
        if lat_metric:
            lat_gate = f"max_{lat_metric}"
            limit = gates.get("models", {}).get(spec.key, {}).get(lat_gate)
            if limit is not None:
                g[lat_gate] = limit
    if profile:
        pg = _lookup(gates, profile)
        if not pg:
            raise HarnessError(f"no gates table {profile!r} in gates.toml")
        if "max_speech_onset_miss_rate" in pg:
            g["max_speech_onset_miss_rate"] = pg["max_speech_onset_miss_rate"]
        if lat_metric:
            lat_gate = f"max_{lat_metric}"
            if lat_gate in pg:
                g[lat_gate] = min(g[lat_gate], pg[lat_gate]) if lat_gate in g else pg[lat_gate]

    gate_checks = evaluate_gates(metrics, g)
    required = set(LIVE_REQUIRED_COUNTERS) | {"speech_onset_miss_rate"} | ({lat_metric} if lat_metric else set())
    for c in gate_checks:
        if c["status"] == "missing" and c["metric"] in required:
            c["status"] = "fail"
    checks.extend(gate_checks)
    return {"suite": "latency", "model": model_key, "strength": reported_strength or "", "metrics": metrics, "checks": checks}


def run_bench(binary: str, models: list[ModelSpec], gates: dict, seconds: float, live: bool, log=print) -> list[dict]:
    rows = []
    for m in models:
        row: dict = {"suite": "bench", "model": m.key, "strength": "live" if live else "", "live": live}
        try:
            cmd = bench_command(binary, m, seconds, live)
            row["command"] = shlex.join(cmd)
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=seconds * 4 + 120)
            if r.returncode != 0:
                raise ValueError(f"exited {r.returncode}: {r.stderr.strip()[-300:]}")
            metrics = parse_bench_json(r.stdout)
            row["raw"] = metrics.pop("raw")
            row["metrics"] = metrics
            row["checks"] = bench_checks(metrics, m, gates, live)
        except (OSError, subprocess.TimeoutExpired, ValueError) as e:
            row["metrics"] = {}
            row["checks"] = [{"gate": "bench", "metric": "error", "op": "max", "threshold": 0, "value": str(e), "status": "fail"}]
        log(f"  bench {m.key:22} {_verdict(row['checks'])}")
        rows.append(row)
    return rows


def evaluate_fault_report(report: dict, gates: dict) -> dict:
    """Checks an engine fault-injection report against [voice_isolation.fault_injection].
    Format: {"model": ..., "cases": [{"case", "strength", "leaked_samples", "leaked_frames",
    "fallback_to_other_model", "isolator"}]}. Every case × strength must be present: the bar is
    hard. `real_model_cases` must have run the real model, not the "silencer" stand-in."""
    g = gates.get("voice_isolation", {}).get("fault_injection", {})
    seen = {(c.get("case"), c.get("strength")): c for c in report.get("cases", [])}
    checks = []
    for case in g.get("cases", []):
        for strength in g.get("strengths", STRENGTHS):
            c = seen.get((case, strength))
            if c is None:
                checks.append({"gate": f"{case}/{strength}", "metric": "present", "op": "min", "threshold": 1, "value": 0, "status": "fail"})
                continue
            if case in g.get("real_model_cases", []):
                iso = c.get("isolator")
                want = g.get("real_model_isolator", "nearfield")
                checks.append({"gate": f"{case}/{strength}/real_model", "metric": "isolator", "op": "eq", "threshold": want, "value": iso, "status": "pass" if iso == want else "fail"})
            checked = c.get("checked_samples")
            ok_checked = isinstance(checked, int) and not isinstance(checked, bool) and checked > 0
            checks.append({"gate": f"{case}/{strength}/checked_samples", "metric": "checked_samples", "op": "min", "threshold": 1, "value": checked, "status": "pass" if ok_checked else "fail"})
            for metric in ("leaked_samples", "leaked_frames", "fallback_to_other_model"):
                limit = g.get(f"max_{metric}")
                if limit is None:
                    continue
                v = c.get(metric)
                checks.append({"gate": f"{case}/{strength}/max_{metric}", "metric": metric, "op": "max", "threshold": limit, "value": v, "status": "fail" if v is None or v > limit else "pass"})
    return {"suite": "fault_injection", "model": report.get("model", "voice_isolation"), "strength": "all", "metrics": {}, "checks": checks}


# ------------------------------------------------------------------------------------------
# Reports


def _lookup(d: dict, dotted: str) -> dict:
    for part in dotted.split("."):
        d = d.get(part, {}) if isinstance(d, dict) else {}
    return d if isinstance(d, dict) else {}


def _safe(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(s))[:120]


def _verdict(checks: list[dict]) -> str:
    fails = [c for c in checks if c["status"] == "fail"]
    missing = sum(c["status"] == "missing" for c in checks)
    if fails:
        return "FAIL (" + ", ".join(c["gate"] for c in fails[:6]) + (" …" if len(fails) > 6 else "") + ")"
    return "pass" + (f" ({missing} missing)" if missing else "")


def _fmt(v, digits=1) -> str:
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if math.isinf(v):
            return "∞" if v > 0 else "−∞"
        return f"{v:.{digits}f}"
    return str(v)


def summarize(rows: list[dict], hashes: list[dict], strict: bool) -> dict:
    checks = [c for r in rows for c in r.get("checks", [])]
    failed = sum(c["status"] == "fail" for c in checks) + sum(not h["ok"] for h in hashes)
    missing = sum(c["status"] == "missing" for c in checks)
    passed = sum(c["status"] == "pass" for c in checks) + sum(h["status"] == "ok" for h in hashes)
    return {"passed": passed, "failed": failed, "missing": missing, "strict": strict, "ok": failed == 0 and (missing == 0 or not strict)}


def markdown(report: dict) -> str:
    meta, s = report["meta"], report["summary"]
    L = ["# Oto evaluation scorecard", "", f"{meta['date']} · processor `{meta['processor']}` · {meta['platform']} · gates `{Path(meta['gates']).name}`"]
    if meta.get("provisional"):
        L += ["", "> Gates are **provisional** (gates.toml `[meta]`): calibrate with measured values before freezing."]
    L += ["", f"**{'PASS' if s['ok'] else 'FAIL'}**: {s['passed']} passed, {s['failed']} failed, {s['missing']} missing{' (strict)' if s['strict'] else ''}.", ""]
    L += ["## Model provenance (MODELS.toml)", "", "| Model | File | SHA-256 | Result |", "|---|---|---|---|"]
    for h in report["hashes"]:
        L.append(f"| {h['model']} | {h['path'] or 'embedded / not vendored'} | `{(h['expected'] or '–')[:16]}` | {h['status']} |")
    ci = [r for r in report["results"] if r["suite"] == "ci"]
    if ci:
        L += ["", "## CI suite (synthetic fixtures)", "", "| Model | Strength | Noise-only atten. dB (model) | Noisy 15 / 5 dB atten. dB | SI-SDR Δ @5 dB | Clean speech Δ dB | Silence out dBFS | Delay err (smp) | NaN / clip | Result |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in ci:
            m = r["metrics"]
            L.append(f"| {r['model']} | {r['strength']} | {_fmt(m.get('noise_only_attenuation_db'))} ({_fmt(m.get('noise_only_model_attenuation_db'))}) | {_fmt(m.get('noisy_15db_attenuation_db'))} / {_fmt(m.get('noisy_5db_attenuation_db'))} | {_fmt(m.get('noisy_5db_si_sdr_delta_db'))} | {_fmt(m.get('clean_speech_level_change_db'))} | {_fmt(m.get('silence_output_dbfs'), 0)} | {_fmt(m.get('delay_error_samples'), 0)} | {m.get('nan_count', '–')} / {m.get('clip_count', '–')} | {_verdict(r['checks'])} |")
    man = [r for r in report["results"] if r["suite"] == "manifest"]
    if man:
        L += ["", "## Manifest suite", "", "| Model | Strength | Clips | SI-SDR Δ | Quiet atten. dB | PESQ-WB | DNSMOS OVR | WER Δ | 1st-word del. | Wrong spk. | BG leaked w/min | Noise halluc. w/min | Overlap leak dB | Result |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in man:
            m = r["metrics"]
            L.append(f"| {r['model']} | {r['strength']} | {m.get('clips', 0)} | {_fmt(m.get('si_sdr_delta_db'))} | {_fmt(m.get('quiet_attenuation_db'))} | {_fmt(m.get('pesq_wb'), 2)} | {_fmt(m.get('dnsmos_ovr'), 2)} | {_fmt(m.get('target_wer_delta'), 3)} | {_fmt(m.get('first_word_deletion_rate'), 3)} | {_fmt(m.get('wrong_speaker_rate'), 3)} | {_fmt(m.get('background_leaked_words_per_min'))} | {_fmt(m.get('noise_only_hallucinated_words_per_min'))} | {_fmt(m.get('overlap_leakage_db'))} | {_verdict(r['checks'])} |")
        kr = [r for r in man if "krisp_wer_pct" in r["metrics"]]
        if kr:
            L += ["", "### Krisp Voice Isolation Benchmark scoring", "", "Pooled WER, caller words missed and leaked words per minute of secondary-only speech; [95% speaker-bootstrap interval]. Deltas are model minus reference (paired).", "", "| Row | Files / speakers | WER % | Missed % | Leak w/min | Phone-call WER % |", "|---|---|---|---|---|---|"]

            def ci(m, k, d=2):
                lo, hi = m.get(k + "_ci_lo"), m.get(k + "_ci_hi")
                return f"{_fmt(m.get(k), d)}" + (f" [{_fmt(lo, d)}, {_fmt(hi, d)}]" if lo is not None else "")

            shown: set[str] = set()
            for r in kr:
                m = r["metrics"]
                for ref in m.get("krisp_references", []):
                    if ref in shown:
                        continue
                    shown.add(ref)
                    p = f"{ref}_"
                    L.append(f"| {ref} | {m.get(p + 'krisp_files', '–')} / {m.get(p + 'krisp_speakers', '–')} | {ci(m, p + 'krisp_wer_pct')} | {ci(m, p + 'missed_caller_pct')} | {ci(m, p + 'secondary_leaked_words_per_min')} | {_fmt(m.get(p + 'phone_calls_krisp_wer_pct'), 2)} |")
                L.append(f"| **{r['model']} / {r['strength']}** | {m.get('krisp_files')} / {m.get('krisp_speakers')} | {ci(m, 'krisp_wer_pct')} | {ci(m, 'missed_caller_pct')} | {ci(m, 'secondary_leaked_words_per_min')} | {_fmt(m.get('phone_calls_krisp_wer_pct'), 2)} |")
                for ref in m.get("krisp_references", []):
                    if ref != "raw":
                        L.append(f"| ↳ Δ vs {ref} | | {ci(m, f'krisp_wer_pct_delta_vs_{ref}')} | {ci(m, f'missed_caller_pct_delta_vs_{ref}')} | {ci(m, f'secondary_leaked_words_per_min_delta_vs_{ref}')} | |")
        if meta.get("asr"):
            L += ["", f"Same-ASR head-to-head, one configuration for every row and the raw input: `{json.dumps(meta['asr'])}`."]
    bench = [r for r in report["results"] if r["suite"] == "bench"]
    if bench:
        L += ["", "## Bench (`oto bench --json`)", "", "| Model | p50 µs | p99 µs | p99.9 µs | max µs | RTF | CPU % | Deadline misses | Starvations | Result |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in bench:
            m = dict(r["metrics"])
            if r.get("live"):  # the stream that does the model's work: worker if it ran, else capture
                src = "worker" if "live_worker_p50_us" in m else "capture"
                for k in ("p50", "p99", "p999", "max"):
                    m[f"bench_{k}_us"] = m.get(f"live_{src}_{k}_us")
            L.append(f"| {r['model']}{' (live)' if r.get('live') else ''} | {_fmt(m.get('bench_p50_us'))} | {_fmt(m.get('bench_p99_us'))} | {_fmt(m.get('bench_p999_us'))} | {_fmt(m.get('bench_max_us'))} | {_fmt(m.get('bench_realtime_factor'), 0)} | {_fmt(m.get('bench_cpu_percent'), 2)} | {_fmt(m.get('live_deadline_misses'), 0)} | {_fmt(m.get('live_starvations'), 0)} | {_verdict(r['checks'])} |")
    lat = [r for r in report["results"] if r["suite"] == "latency"]
    if lat:
        L += ["", "## Latency (`oto latency --stimulus speech --json`)", "", "| Model | Median ms | p95 ms | Detected / Expected | Miss rate | Deadline misses | Starvations | Result |", "|---|---|---|---|---|---|---|---|"]
        for r in lat:
            m = r["metrics"]
            L.append(f"| {r['model']} | {_fmt(m.get('median_ms'))} | {_fmt(m.get('p95_ms'))} | {m.get('detected', '–')} / {m.get('expected', '–')} | {_fmt(m.get('speech_onset_miss_rate'), 3)} | {_fmt(m.get('live_deadline_misses'), 0)} | {_fmt(m.get('live_starvations'), 0)} | {_verdict(r['checks'])} |")
    fails = [(r, c) for r in report["results"] for c in r.get("checks", []) if c["status"] == "fail"]
    if fails:
        L += ["", "## Failed gates", "", "| Suite | Model | Strength | Gate | Value | Threshold |", "|---|---|---|---|---|---|"]
        for r, c in fails:
            L.append(f"| {r['suite']} | {r['model']} | {r.get('strength', '')} | `{c['gate']}` | {_fmt(c['value'], 2)} | {c['threshold']} |")
    errors = [(r, e) for r in report["results"] for e in r.get("errors", [])]
    if errors:
        L += ["", "## Errors", ""] + [f"- {r['model']} / {r.get('strength', '')}: {e}" for r, e in errors[:50]]
    return "\n".join(L) + "\n"


def _clean_floats(o):
    if isinstance(o, float) and not math.isfinite(o):
        return None if math.isnan(o) else ("inf" if o > 0 else "-inf")
    if isinstance(o, dict):
        return {k: _clean_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean_floats(v) for v in o]
    return o


def write_report(report: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    j, md = out_dir / "report.json", out_dir / "scorecard.md"
    j.write_text(json.dumps(_clean_floats(report), indent=2, default=str) + "\n")
    md.write_text(markdown(report))
    return j, md


# ------------------------------------------------------------------------------------------
# CLI


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", choices=["ci", "manifest"], help="ci: synthetic fixtures; manifest: an external JSONL dataset")
    p.add_argument("--manifest", type=Path, help="JSONL file for --suite manifest (see README)")
    p.add_argument("--models", default=",".join(DEFAULT_MODELS), help="comma-separated MODELS.toml keys or CLI names (default: %(default)s)")
    p.add_argument("--strengths", default=",".join(STRENGTHS))
    p.add_argument("--processor", choices=["oto", "passthrough", "oracle"], default="oto", help="passthrough / oracle test the harness without the binary")
    p.add_argument("--oto", help="oto binary (default: $OTO_BIN, target/release/oto, target/debug/oto, PATH)")
    p.add_argument("--process-template", default=os.environ.get("OTO_PROCESS_TEMPLATE", DEFAULT_TEMPLATE), help="command template; fields {oto} {model} {strength} {bypass} {input} {output}")
    p.add_argument("--bench", "--ota-bench", dest="bench", action="store_true", help="also run `oto bench --json` per model")
    p.add_argument("--live", action="store_true", help="pass --live to oto bench (realtime-paced: deadline misses, starvations)")
    p.add_argument("--bench-seconds", type=float, default=10.0)
    p.add_argument("--fault-report", type=Path, help="engine fault-injection JSON, checked against [voice_isolation.fault_injection]")
    p.add_argument("--latency-report", type=Path, action="append", default=[], help="saved `oto latency --stimulus speech --json` file (repeatable)")
    p.add_argument("--gates-profile", help="gates.toml table for the manifest/latency suite, e.g. voice_isolation.opt_in")
    p.add_argument("--asr", action="store_true", help="same-ASR head-to-head (extra: asr for faster-whisper, parakeet for Parakeet TDT)")
    p.add_argument("--asr-backend", choices=["faster-whisper", "parakeet"], default="faster-whisper", help="parakeet = NVIDIA Parakeet TDT 0.6B v2 via onnx-asr, the Krisp benchmark's recogniser")
    p.add_argument("--asr-model", help="backend model (default: small.en for faster-whisper, nemo-parakeet-tdt-0.6b-v2 for parakeet)")
    p.add_argument("--asr-quantization", help="parakeet only: e.g. int8 (default fp32)")
    p.add_argument("--asr-cache-dir", type=Path, help="reuse/save per-clip ASR transcripts under <dir>/<asr-tag>/<cond>.json")
    p.add_argument("--krisp-normalizer", choices=["auto", "nemo", "stdlib"], default="auto", help="text normalisation for Krisp-style entries (auto: NeMo TN if installed)")
    p.add_argument("--bootstrap", type=int, default=2000, help="speaker-bootstrap resamples for Krisp-style CIs")
    p.add_argument("--dnsmos-model", help="local DNSMOS sig_bak_ovr.onnx (extra: perceptual)")
    p.add_argument("--gates", type=Path, default=HERE / "gates.toml")
    p.add_argument("--models-toml", type=Path, default=REPO / "MODELS.toml")
    p.add_argument("--fixtures-dir", type=Path, help="reuse fixtures written by fixtures.py")
    p.add_argument("--out", type=Path, default=HERE / "out", help="report and audio directory (default: tools/eval/out, git-ignored)")
    p.add_argument("--strict", action="store_true", help="missing metrics fail too")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except HarnessError as e:
        print(f"eval: {e}", file=sys.stderr)
        return 2


def run(args, log=print) -> int:
    if not (args.suite or args.bench or args.fault_report or args.latency_report):
        raise HarnessError("nothing to do: pass --suite ci|manifest, --bench, --latency-report or --fault-report")
    if args.suite == "manifest" and not args.manifest:
        raise HarnessError("--suite manifest needs --manifest FILE.jsonl")
    gates = load_toml(args.gates)
    all_models = load_models(args.models_toml)
    models = resolve_models([m for m in args.models.split(",") if m], all_models)
    strengths = [s for s in args.strengths.split(",") if s]
    bad = [s for s in strengths if s not in STRENGTHS]
    if bad:
        raise HarnessError(f"unknown strength {bad[0]!r}; use {', '.join(STRENGTHS)}")

    log("MODELS.toml hashes:")
    hashes = verify_hashes(all_models, args.models_toml.parent)
    for h in hashes:
        log(f"  {h['model']:22} {h['status']}")

    gm = gates.get("meta", {})
    meta = {"date": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), "gates": str(args.gates), "provisional": bool(gm.get("provisional_latency") or gm.get("provisional_ci_bars")), "python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}", "processor": args.processor, "models": [m.key for m in models], "strengths": strengths}
    results: list[dict] = []
    args.out.mkdir(parents=True, exist_ok=True)

    latency_rows: list[dict] = []
    latency_gates_by_key: dict[tuple[str, str], set[str]] = {}
    if args.latency_report:
        for path in args.latency_report:
            row = evaluate_latency_report(json.loads(path.read_text()), all_models, gates, args.gates_profile)
            latency_rows.append(row)
            if row["strength"] in STRENGTHS:
                latency_gates_by_key.setdefault((row["model"], row["strength"]), set()).update(c["gate"] for c in row["checks"])

    if args.suite:
        proc = make_processor(args)
        if isinstance(proc, OtoProcessor):
            meta.update(oto=proc.binary, process_template=proc.template)
        log(f"suite {args.suite}:")
        if args.suite == "ci":
            results += run_ci(models, strengths, proc, gates, args.out, args.fixtures_dir, log)
        else:
            entries = read_manifest(args.manifest)
            meta["manifest"] = str(args.manifest)
            transcriber = None
            normalizer = None
            if args.asr:
                from asr import make_transcriber

                try:
                    transcriber = make_transcriber(args.asr_backend, args.asr_model, args.asr_quantization)
                    normalizer = krisp.KrispNormalizer(args.krisp_normalizer)
                except RuntimeError as e:
                    raise HarnessError(str(e)) from e
                meta["asr"] = transcriber.config
                if any(e.get("segments") is not None for e in entries):
                    meta["krisp_normalizer"] = normalizer.backend
            dnsmos = None
            if args.dnsmos_model:
                try:
                    dnsmos = Dnsmos(args.dnsmos_model)
                except ImportError as e:
                    raise HarnessError("DNSMOS needs the `perceptual` extra (onnxruntime, numpy)") from e
            results += run_manifest(entries, models, strengths, proc, gates, args.out, args.gates_profile, transcriber, dnsmos, log, latency_gates_by_key=latency_gates_by_key, normalizer=normalizer, n_boot=args.bootstrap, asr_cache_dir=args.asr_cache_dir)

    if args.bench:
        binary = find_oto(args.oto)
        if binary is None:
            raise HarnessError("--bench needs the oto binary (--oto or $OTO_BIN)")
        log("bench:")
        results += run_bench(binary, models, gates, args.bench_seconds, args.live, log)

    if args.fault_report:
        results.append(evaluate_fault_report(json.loads(args.fault_report.read_text()), gates))

    if latency_rows:
        log("latency:")
        for row in latency_rows:
            log(f"  latency {row['model']:20} {_verdict(row['checks'])}")
            results.append(row)

    report = {"meta": meta, "hashes": hashes, "results": results}
    report["summary"] = summarize(results, hashes, args.strict)
    j, md = write_report(report, args.out)
    s = report["summary"]
    log(f"{'PASS' if s['ok'] else 'FAIL'}: {s['passed']} passed, {s['failed']} failed, {s['missing']} missing")
    log(f"wrote {md} and {j}")
    return 0 if s["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
