"""Tests for the evaluation harness. Stdlib only:

    python3 -m unittest discover -s tools/eval -v

The live `oto process` test runs only when a built oto binary has the subcommand
(target/release/oto, target/debug/oto, $OTO_BIN or PATH); everything else uses the
passthrough/oracle processors or a fake `oto` script.
"""

from __future__ import annotations

import io
import json
import math
import os
import stat
import struct
import sys
import tempfile
import tomllib
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import asr  # noqa: E402
import eval as harness  # noqa: E402
import fixtures as fx  # noqa: E402
import metrics as mx  # noqa: E402
from wavio import read_wav, write_wav  # noqa: E402

REPO = HERE.parent.parent
_FIXTURES: tempfile.TemporaryDirectory | None = None


def shared_fixtures() -> Path:
    """Fixtures are generated once per test run (about a second)."""
    global _FIXTURES
    if _FIXTURES is None:
        _FIXTURES = tempfile.TemporaryDirectory(prefix="oto-eval-fixtures-")
        fx.generate(_FIXTURES.name)
    return Path(_FIXTURES.name)


def tearDownModule():
    if _FIXTURES is not None:
        _FIXTURES.cleanup()


def sine(freq, rate, seconds, amp=1.0):
    return [amp * math.sin(2 * math.pi * freq * i / rate) for i in range(int(rate * seconds))]


def quiet_log(*_a, **_k):
    pass


class WavTests(unittest.TestCase):
    def test_pcm16_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            x = [0.0, 0.5, -0.5, 0.25, -1.0, 0.999]
            write_wav(Path(d) / "a.wav", 16000, x, "pcm16")
            a = read_wav(Path(d) / "a.wav")
            self.assertEqual(a.rate, 16000)
            self.assertEqual(len(a.channels), 1)
            for got, want in zip(a.mono, x):
                self.assertAlmostEqual(got, want, delta=1 / 32768 + 1e-9)

    def test_float32_round_trip_keeps_overs(self):
        with tempfile.TemporaryDirectory() as d:
            x = [0.0, 1.5, -2.0, 0.125]
            write_wav(Path(d) / "f.wav", 48000, x, "float32")
            a = read_wav(Path(d) / "f.wav")
            self.assertEqual(a.mono, x)

    def test_pcm16_clips_on_write(self):
        with tempfile.TemporaryDirectory() as d:
            write_wav(Path(d) / "c.wav", 8000, [2.0, -2.0], "pcm16")
            a = read_wav(Path(d) / "c.wav")
            self.assertAlmostEqual(a.mono[0], 32767 / 32768)
            self.assertEqual(a.mono[1], -1.0)

    def test_mono_matches_live_downmix_rules(self):
        # Stereo averages both channels (opposite polarity cancels); >2ch takes channel 0.
        stereo = harness.Audio(48000, [[0.5, -0.25], [-0.5, 0.75]])
        self.assertEqual(stereo.mono, [0.0, 0.25])
        quad = harness.Audio(48000, [[0.4, -0.2], [0.9, 0.9], [0.9, 0.9], [0.9, 0.9]])
        self.assertEqual(quad.mono, [0.4, -0.2])

    def test_si_sdr_muted_estimate_scores_at_floor_not_inf(self):
        ref = sine(440, 16000, 0.1, 0.5)
        muted = [0.0] * len(ref)
        self.assertEqual(mx.si_sdr(ref, muted), mx.FLOOR_DB)
        self.assertEqual(mx.si_sdr(ref, [0.25 * s for s in ref]), math.inf)

    def _raw_wav(self, tag, bits, nch, payload, extensible=False):
        width = bits // 8
        if extensible:
            fmt = struct.pack("<HHIIHH", 0xFFFE, nch, 48000, 48000 * width * nch, width * nch, bits)
            fmt += struct.pack("<HHI", 22, bits, 0) + struct.pack("<H", tag) + b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
        else:
            fmt = struct.pack("<HHIIHH", tag, nch, 48000, 48000 * width * nch, width * nch, bits)
        body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
        body += b"LIST" + struct.pack("<I", 4) + b"INFO"  # an unrelated chunk to skip
        body += b"data" + struct.pack("<I", len(payload)) + payload
        return b"RIFF" + struct.pack("<I", len(body)) + body

    def test_reads_24bit_stereo_and_extensible_float(self):
        with tempfile.TemporaryDirectory() as d:
            # 24-bit stereo: L = +0.5, R = -0.25
            l, r = int(0.5 * 8388608), int(-0.25 * 8388608) & 0xFFFFFF
            frame = l.to_bytes(3, "little") + r.to_bytes(3, "little")
            p = Path(d) / "s24.wav"
            p.write_bytes(self._raw_wav(1, 24, 2, frame * 3))
            a = read_wav(p)
            self.assertEqual(len(a.channels), 2)
            self.assertEqual(a.channels[0], [0.5] * 3)
            self.assertEqual(a.channels[1], [-0.25] * 3)
            q = Path(d) / "ext.wav"
            q.write_bytes(self._raw_wav(3, 32, 1, struct.pack("<3f", 0.1, -0.2, 0.3), extensible=True))
            self.assertEqual([round(v, 6) for v in read_wav(q).mono], [0.1, -0.2, 0.3])

    def test_rejects_non_wav(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.wav"
            p.write_bytes(b"not a wav file at all")
            with self.assertRaises(ValueError):
                read_wav(p)


class FixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = shared_fixtures()
        cls.manifest = json.loads((cls.dir / "fixtures.json").read_text())
        cls.by_name = {f["name"]: f for f in cls.manifest["fixtures"]}

    def test_suite_contents(self):
        self.assertEqual(set(self.by_name), {"clean_speech", "noisy_speech_15db", "noisy_speech_5db", "noise_only", "impulse_step", "silence", "hfp_16k_noisy_15db"})
        for f in self.manifest["fixtures"]:
            a = read_wav(self.dir / f["path"])
            self.assertEqual(a.rate, f["rate"])
            self.assertEqual(a.frames, f["frames"])
            self.assertLess(mx.peak_dbfs(a.mono), -0.5, f["name"])
        self.assertEqual(self.by_name["hfp_16k_noisy_15db"]["rate"], 16000)

    def test_deterministic(self):
        with tempfile.TemporaryDirectory() as d:
            fx.generate(d)
            for f in self.manifest["fixtures"]:
                self.assertEqual((Path(d) / f["path"]).read_bytes(), (self.dir / f["path"]).read_bytes(), f["name"])
            self.assertEqual(json.loads((Path(d) / "fixtures.json").read_text()), self.manifest)

    def test_noisy_mixes_hit_their_snr(self):
        ref = read_wav(self.dir / "clean_speech_ref.wav").mono
        spans = fx.SPEECH_SPANS
        for name, snr in (("noisy_speech_15db", 15), ("noisy_speech_5db", 5)):
            noisy = read_wav(self.dir / self.by_name[name]["path"]).mono
            noise = [a - b for a, b in zip(noisy, ref)]
            speech = mx.window_samples(ref, fx.RATE, spans)
            got = 20 * math.log10(mx.rms(speech) / mx.rms(noise))
            self.assertAlmostEqual(got, snr, delta=0.1, msg=name)

    def test_speech_level_and_gaps(self):
        f = self.by_name["clean_speech"]
        ref = read_wav(self.dir / f["reference"]).mono
        self.assertAlmostEqual(mx.rms_dbfs(mx.window_samples(ref, fx.RATE, fx.SPEECH_SPANS)), fx.SPEECH_RMS_DBFS, delta=0.1)
        self.assertEqual(mx.rms(mx.window_samples(ref, fx.RATE, f["noise_windows"])), 0.0)

    def test_noise_only_level(self):
        x = read_wav(self.dir / "noise_only.wav").mono
        self.assertAlmostEqual(mx.rms_dbfs(x), fx.NOISE_ONLY_RMS_DBFS, delta=0.2)

    def test_alignment_fixture(self):
        f = self.by_name["impulse_step"]
        x = read_wav(self.dir / f["path"]).mono
        for p in f["impulses"]:
            self.assertAlmostEqual(x[p], fx.IMPULSE_AMP, delta=1e-4)
            self.assertEqual(x[p - 1], 0.0)
        self.assertEqual(x[f["step"] - 1], 0.0)
        self.assertAlmostEqual(x[f["step"]], fx.STEP_AMP, delta=1e-4)

    def test_silence_is_digital_zero(self):
        self.assertTrue(all(s == 0.0 for s in read_wav(self.dir / "silence.wav").mono))

    def test_windows_exclude_warmup_and_guard_edges(self):
        speech, noise = fx.windows([(1.0, 2.0), (2.5, 3.5)], duration=4.0, warmup=0.5, guard=0.05)
        self.assertEqual(speech, [(1.05, 1.95), (2.55, 3.45)])
        self.assertEqual(noise, [(0.5, 0.95), (2.05, 2.45), (3.55, 3.95)])


class MetricTests(unittest.TestCase):
    def test_rms_and_peak(self):
        x = sine(1000, 48000, 1.0)
        self.assertAlmostEqual(mx.rms_dbfs(x), -3.0103, delta=0.01)
        self.assertAlmostEqual(mx.peak_dbfs(x), 0.0, delta=0.01)
        self.assertEqual(mx.rms_dbfs([0.0] * 10), mx.FLOOR_DB)
        self.assertEqual(mx.rms([]), 0.0)

    def test_nan_and_clip_counts(self):
        x = [0.0, float("nan"), float("inf"), 1.0, -1.2, 0.5]
        self.assertEqual(mx.nan_count(x), 2)
        self.assertEqual(mx.clip_count(x), 2)

    def test_si_sdr_is_scale_invariant(self):
        ref = sine(440, 16000, 0.5)
        self.assertGreater(mx.si_sdr(ref, [0.3 * s for s in ref]), 100)
        self.assertEqual(mx.si_sdr([0.0] * 10, ref[:10]), mx.FLOOR_DB)

    def test_si_sdr_and_snr_with_known_noise(self):
        ref = sine(440, 16000, 1.0)
        noise = sine(3000, 16000, 1.0, amp=0.1)  # orthogonal over whole periods: 20 dB below
        est = [a + b for a, b in zip(ref, noise)]
        self.assertAlmostEqual(mx.snr_db(ref, est), 20.0, delta=0.05)
        self.assertAlmostEqual(mx.si_sdr(ref, est), 20.0, delta=0.05)
        self.assertEqual(mx.snr_db(ref, ref), math.inf)

    def test_shift_and_windows(self):
        x = list(range(10))
        self.assertEqual(mx.shift(x, 3), [3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(mx.shift(x, 3, 9), [3, 4, 5, 6, 7, 8, 9, 0.0, 0.0])
        self.assertEqual(mx.shift(x, -2, 4), [0.0, 0.0, 0, 1])
        self.assertEqual(mx.window_samples(x, 10, [(0.2, 0.4), (0.8, 2.0)]), [2, 3, 8, 9])
        self.assertEqual(mx.window_samples(x, 10, [(0.2, 0.4)], delay=1), [3, 4])

    def test_attenuation_with_delay_and_rate_change(self):
        inp = sine(200, 48000, 1.0, 0.5)
        delay = 480
        out = [0.0] * delay + [s * 0.1 for s in inp[:-delay]]  # 20 dB down, 10 ms late
        self.assertAlmostEqual(mx.attenuation_db(inp, 48000, out, 48000, [(0.1, 0.9)], delay), 20.0, delta=0.01)
        out16 = mx.resample_linear(out, 48000, 16000)
        self.assertAlmostEqual(mx.attenuation_db(inp, 48000, out16, 16000, [(0.1, 0.9)], delay // 3), 20.0, delta=0.1)

    def test_envelope_delay(self):
        ref = [0.0] * 4800 + sine(300, 48000, 0.2, 0.5) + [0.0] * 4800 + sine(500, 48000, 0.1, 0.5) + [0.0] * 9600
        d = 2880  # 60 ms
        out = [0.0] * d + ref[:-d]
        self.assertAlmostEqual(mx.envelope_delay(ref, 48000, out, 48000), 0.060, delta=0.0025)

    def test_impulse_and_step_delays(self):
        y = [0.0] * 2000
        for p in (100, 600):
            y[p + 37] = 0.4
            y[p + 38] = -0.1
        self.assertEqual(mx.impulse_delays(y, [100, 600], 200), [37, 37])
        z = [0.0] * 300 + [0.25] * 300
        self.assertEqual(mx.step_delay(z, 250, 100), 50)
        self.assertEqual(mx.median([3, 1, 2]), 2)
        self.assertEqual(mx.median([1, 2, 3, 4]), 2.5)
        self.assertIsNone(mx.median([]))

    def test_resample_linear(self):
        x = [float(i) for i in range(48)]
        y = mx.resample_linear(x, 48000, 16000)
        self.assertEqual(len(y), 16)
        self.assertEqual(y[:3], [0.0, 3.0, 6.0])


class AsrTextTests(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(asr.normalize("Hello, World! It's   'fine'."), ["hello", "world", "it's", "fine"])

    def test_wer(self):
        self.assertEqual(asr.wer("the cat sat", "the cat sat"), 0.0)
        self.assertAlmostEqual(asr.wer("the cat sat on the mat", "the cat sat on mat"), 1 / 6)
        self.assertAlmostEqual(asr.wer("a b c", "a x c d"), 2 / 3)
        self.assertEqual(asr.wer("", ""), 0.0)
        a = asr.align(["a", "b", "c"], ["b", "c"])
        self.assertEqual((a.substitutions, a.deletions, a.insertions), (0, 1, 0))
        self.assertEqual(a.deleted, [True, False, False])

    def test_first_word_deletion(self):
        self.assertTrue(asr.first_word_deleted("hello there my friend", "there my friend"))
        self.assertTrue(asr.first_word_deleted("hello there", "yellow there"))
        self.assertFalse(asr.first_word_deleted("hello there", "hello there"))
        self.assertFalse(asr.first_word_deleted("hello there", "um hello there"))
        self.assertFalse(asr.first_word_deleted("", "anything"))

    def test_rates_and_wrong_speaker(self):
        self.assertEqual(asr.words_per_minute("one two three", 30.0), 6.0)
        self.assertEqual(asr.words_per_minute("", 30.0), 0.0)
        self.assertTrue(asr.wrong_speaker("turn on the tv", "please call me back", "turn on the tv"))
        self.assertFalse(asr.wrong_speaker("please call me back", "please call me back", "turn on the tv"))


class GateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gates = harness.load_toml(HERE / "gates.toml")

    def test_evaluate_min_max_missing(self):
        checks = harness.evaluate_gates({"a": 5.0, "b": 1.0, "len_min": 0.9, "len_max": 1.2, "nan": float("nan")}, {"min_a": 4, "max_b": 0.5, "min_len": 0.95, "max_len": 1.3, "max_c": 1, "max_nan": 0, "note": "ignored", "max_flag": True})
        by = {c["gate"]: c for c in checks}
        self.assertEqual(by["min_a"]["status"], "pass")
        self.assertEqual(by["max_b"]["status"], "fail")
        self.assertEqual((by["min_len"]["metric"], by["min_len"]["status"]), ("len_min", "fail"))
        self.assertEqual((by["max_len"]["metric"], by["max_len"]["status"]), ("len_max", "pass"))
        self.assertEqual(by["max_c"]["status"], "missing")
        self.assertEqual(by["max_nan"]["status"], "missing")
        self.assertNotIn("note", by)
        self.assertNotIn("max_flag", by)

    def test_ci_gate_layering(self):
        g = harness.ci_gates(self.gates, "rnnoise", "balanced")
        self.assertEqual(g["max_nan_count"], 0)  # shared
        self.assertEqual(g["max_noise_only_model_attenuation_db"], 24.5)  # [ci.balanced]
        self.assertEqual(g["min_noise_only_attenuation_db"], 18.0)
        self.assertEqual(harness.ci_gates(self.gates, "rnnoise", "strong")["min_noise_only_attenuation_db"], 25.0)  # model override wins
        self.assertEqual(harness.ci_gates(self.gates, "dfn3", "strong")["min_noise_only_attenuation_db"], 40.0)
        self.assertEqual(harness.ci_gates(self.gates, "dfn3", "light")["max_noise_only_model_attenuation_db"], 12.5)
        self.assertNotIn("max_noise_only_model_attenuation_db", harness.ci_gates(self.gates, "dfn3", "strong"))

    def test_gates_file_shape(self):
        g = self.gates
        self.assertTrue(g["meta"]["provisional_latency"])
        self.assertIn("before freezing", g["meta"]["note"])
        models = harness.load_toml(REPO / "MODELS.toml")["models"]
        for key in ("rnnoise", "dfn3", "dpdfnet2_48khz_hr", "fastenhancer_s_48khz", "nearfield_v2_3_fast"):
            self.assertIn(key, models)
            m = g["models"][key]
            for k in ("chain_latency_ms", "max_macos_e2e_latency_ms", "max_linux_e2e_latency_ms", "max_bench_cpu_percent"):
                self.assertIn(k, m, f"{key}.{k}")
            self.assertLessEqual(m["chain_latency_ms"], m["max_macos_e2e_latency_ms"])
        self.assertEqual(g["models"]["dfn3"]["max_macos_e2e_latency_ms"], 65.0)
        self.assertEqual(g["models"]["dpdfnet2_48khz_hr"]["max_macos_e2e_latency_ms"], 85.0)

    def test_voice_isolation_gates(self):
        vi = self.gates["voice_isolation"]
        fi = vi["fault_injection"]
        self.assertEqual(set(fi["cases"]), {"startup", "starvation", "overflow_flush", "inference_error", "load_failure", "mode_switch_preload"})
        self.assertEqual(fi["strengths"], harness.STRENGTHS)
        self.assertEqual((fi["max_leaked_samples"], fi["max_leaked_frames"], fi["max_fallback_to_other_model"]), (0, 0, 0))
        needed = {"max_background_leaked_words_per_min", "max_overlap_leakage_db", "max_first_word_deletion_rate", "max_wrong_speaker_rate", "max_target_wer_delta", "max_speech_onset_miss_rate"}
        for profile in ("opt_in", "default_promotion"):
            self.assertLessEqual(needed, set(vi[profile]), profile)
        for k in needed:  # promotion is stricter on every shared gate
            self.assertLess(vi["default_promotion"][k], vi["opt_in"][k], k)

    def test_fault_report(self):
        fi = self.gates["voice_isolation"]["fault_injection"]
        cases = [{"case": c, "strength": s, "leaked_samples": 0, "leaked_frames": 0, "fallback_to_other_model": 0} for c in fi["cases"] for s in fi["strengths"]]
        ok = harness.evaluate_fault_report({"cases": cases}, self.gates)
        self.assertTrue(all(c["status"] == "pass" for c in ok["checks"]))
        cases[0]["leaked_samples"] = 3
        bad = harness.evaluate_fault_report({"cases": cases[:-1]}, self.gates)
        failed = {c["gate"] for c in bad["checks"] if c["status"] == "fail"}
        self.assertIn(f"{cases[0]['case']}/{cases[0]['strength']}/max_leaked_samples", failed)
        self.assertIn(f"{cases[-1]['case']}/{cases[-1]['strength']}", failed)  # missing case fails


class ModelsManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(REPO / "MODELS.toml", "rb") as f:
            cls.toml = tomllib.load(f)
        cls.models = harness.load_models(REPO / "MODELS.toml")

    def test_sha256_matches_vendored_models(self):
        onnx = sorted((REPO / "crates/oto-dsp/models").glob("*.onnx"))
        self.assertTrue(onnx, "no ONNX models in crates/oto-dsp/models")
        listed = {m.path for m in self.models.values() if m.path}
        for p in onnx:
            self.assertIn(str(p.relative_to(REPO)), listed, f"{p.name} is not in MODELS.toml")
        rows = harness.verify_hashes(self.models, REPO)
        for r in rows:
            self.assertTrue(r["ok"], r)
        self.assertEqual({r["model"] for r in rows if r["status"] == "ok"}, {"dpdfnet2_48khz_hr", "fastenhancer_s_48khz"})

    def test_hash_mismatch_and_missing_file_fail(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "m.onnx").write_bytes(b"weights")
            spec = harness.ModelSpec("x", "x", None, 480, 480, path="m.onnx", sha256="0" * 64)
            gone = harness.ModelSpec("y", "y", None, 480, 480, path="gone.onnx", sha256="0" * 64)
            rows = {r["model"]: r for r in harness.verify_hashes({"x": spec, "y": gone}, Path(d))}
            self.assertEqual((rows["x"]["status"], rows["x"]["ok"]), ("mismatch", False))
            self.assertEqual((rows["y"]["status"], rows["y"]["ok"]), ("missing-file", False))

    def test_entries(self):
        m = self.toml["models"]
        self.assertEqual((m["rnnoise"]["version"], m["rnnoise"]["license_code"], m["rnnoise"]["frame_samples"], m["rnnoise"]["latency_samples"]), ("0.5.2", "BSD-3-Clause", 480, 480))
        self.assertTrue(m["rnnoise"]["ship_in_release"])
        self.assertEqual(m["dfn3"]["commit"], "7b14b895fe7fc42cab72b4a4c1e49c982bd6404b")
        self.assertEqual((m["dfn3"]["license_code"], m["dfn3"]["frame_samples"], m["dfn3"]["latency_samples"]), ("MIT OR Apache-2.0", 480, 1440))
        d = m["dpdfnet2_48khz_hr"]
        self.assertEqual((d["license_weights"], d["frame_samples"], d["latency_samples"]), ("Apache-2.0", 480, 2400))
        self.assertEqual(d["architectures"], ["aarch64-apple-darwin", "x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"])
        fe = m["fastenhancer_s_48khz"]
        self.assertEqual((fe["ship_in_release"], fe["status"], fe["license_weights"]), (False, "evaluation-only", "unstated"))
        nf = m["nearfield_v2_3_fast"]
        self.assertEqual((nf["ship_in_release"], nf["status"], nf["tag"], nf["frame_samples"], nf["model_sample_rate"]), (False, "candidate-pending-phase2-gates", "v2.3.0", 384, 16000))
        self.assertEqual(nf["latency_samples"], 48 * 48)
        self.assertTrue(nf["notice_required"])
        for key, spec in m.items():
            if spec["ship_in_release"]:
                self.assertNotIn("unstated", spec["license_weights"], key)

    def test_chain_latency_matches_gates(self):
        gates = harness.load_toml(HERE / "gates.toml")
        for key, spec in self.models.items():
            want = gates["models"][key]["chain_latency_ms"]
            got = spec.chain_latency_samples * 1000 / harness.ENGINE_RATE + (max(spec.raw.get("resampler_latency_ms", [0])))
            self.assertAlmostEqual(got, want, delta=0.5, msg=key)

    def test_resolve_by_key_or_cli_name(self):
        self.assertEqual([m.key for m in harness.resolve_models(["dpdfnet", "dfn3"], self.models)], ["dpdfnet2_48khz_hr", "dfn3"])
        with self.assertRaises(harness.HarnessError):
            harness.resolve_models(["nope"], self.models)


class BenchParseTests(unittest.TestCase):
    def test_offline_bench_json(self):
        text = 'loading\n{"model":"DeepFilterNet3","frames":1000,"block":480,"p50_us":700.5,"p99_us":850,"p999_us":990,"max_us":1200,"mean_us":720,"realtime_factor":14.2,"cpu_percent":7.04,"latency_ms":40,"latency_samples":1920}\n'
        m = harness.parse_bench_json(text)
        self.assertEqual((m["bench_p50_us"], m["bench_p999_us"], m["bench_realtime_factor"], m["bench_latency_samples"]), (700.5, 990, 14.2, 1920))
        spec = harness.load_models(REPO / "MODELS.toml")["dfn3"]
        gates = harness.load_toml(HERE / "gates.toml")
        m.pop("raw")
        checks = harness.bench_checks(m, spec, gates, live=False)
        self.assertTrue(checks and all(c["status"] == "pass" for c in checks), checks)
        m["bench_latency_samples"] = 1919
        self.assertIn("fail", {c["status"] for c in harness.bench_checks(m, spec, gates, live=False)})

    def test_live_bench_json(self):
        cb = lambda p99, misses: {"callbacks": 100, "period_us": 5333, "p50_us": 100, "p99_us": p99, "p999_us": p99 + 50, "max_us": p99 + 100, "first_us": 900, "deadline_misses": misses}
        obj = {"tier": "standard", "denoise": True, "seconds": 5, "cpu_percent": 3.1, "errors": [], "notices": [], "stats": {"running": True, "model": "DeepFilterNet3", "estimated_latency_ms": 61.0, "underruns": 0, "resyncs": 0, "overflows": 0, "starvations": 0, "worker_overflows": 0, "capture": cb(200, 0), "output": cb(150, 1), "worker": cb(1300, 0)}}
        m = harness.parse_bench_json(json.dumps(obj))
        self.assertEqual((m["live_deadline_misses"], m["live_worker_p999_us"], m["bench_cpu_percent"], m["live_starvations"]), (1, 1350, 3.1, 0))
        m.pop("raw")
        spec = harness.load_models(REPO / "MODELS.toml")["dfn3"]
        gates = harness.load_toml(HERE / "gates.toml")
        failed = {c["gate"] for c in harness.bench_checks(m, spec, gates, live=True) if c["status"] == "fail"}
        self.assertEqual(failed, {"max_live_deadline_misses"})

        # Empty object, wrong model, or capture-ring overflows must fail even without --strict.
        empty = harness.parse_bench_json("{}")
        empty.pop("raw")
        self.assertTrue(any(c["status"] == "fail" for c in harness.bench_checks(empty, spec, gates, live=True)))

        obj["stats"]["output"] = cb(150, 0)
        obj["stats"]["overflows"] = 9
        ovf = harness.parse_bench_json(json.dumps(obj))
        ovf.pop("raw")
        self.assertIn("max_live_overflows", {c["gate"] for c in harness.bench_checks(ovf, spec, gates, live=True) if c["status"] == "fail"})

        obj["stats"]["overflows"] = 0
        obj["stats"]["model"] = "RNNoise"
        wrong = harness.parse_bench_json(json.dumps(obj))
        wrong.pop("raw")
        self.assertIn("live_model", {c["gate"] for c in harness.bench_checks(wrong, spec, gates, live=True) if c["status"] == "fail"})

        # Omitted per-stream deadline_misses, errors, or notices must fail, not default to 0.
        stripped = json.loads(json.dumps(obj))
        stripped["stats"]["model"] = "DeepFilterNet3"
        del stripped["errors"]
        del stripped["notices"]
        del stripped["stats"]["worker"]["deadline_misses"]
        sm = harness.parse_bench_json(json.dumps(stripped))
        sm.pop("raw")
        failed = {c["gate"] for c in harness.bench_checks(sm, spec, gates, live=True) if c["status"] == "fail"}
        self.assertTrue({"max_live_errors", "max_live_fallbacks", "max_live_deadline_misses"} <= failed, failed)

    def test_latency_report_gates(self):
        cb = lambda: {"callbacks": 200, "period_us": 5333, "p50_us": 100, "p99_us": 500, "p999_us": 600, "max_us": 700, "first_us": 900, "deadline_misses": 0}
        obj = {
            "platform": "macos",
            "stimulus": "speech",
            "tier": "standard",
            "denoise": True,
            "seconds": 6,
            "injections": 8,
            "expected": 8,
            "detected": 8,
            "missed": 0,
            "miss_rate": 0.0,
            "min_ms": 62.0,
            "median_ms": 62.5,
            "p95_ms": 63.5,
            "max_ms": 63.5,
            "ok": True,
            "errors": [],
            "notices": [],
            "stats": {"running": True, "model": "DeepFilterNet3", "underruns": 0, "resyncs": 0, "overflows": 0, "starvations": 0, "worker_overflows": 0, "capture": cb(), "output": cb(), "worker": cb()},
        }
        models = harness.load_models(REPO / "MODELS.toml")
        gates = harness.load_toml(HERE / "gates.toml")
        row = harness.evaluate_latency_report(obj, models, gates, profile="voice_isolation.opt_in")
        self.assertEqual(row["model"], "dfn3")
        self.assertIn("macos_e2e_latency_ms", row["metrics"])
        self.assertTrue(all(c["status"] == "pass" for c in row["checks"]), row["checks"])

        # S2: removing stats.worker from a worker-placed model (dfn3) must fail.
        no_worker = json.loads(json.dumps(obj))
        del no_worker["stats"]["worker"]
        nw_row = harness.evaluate_latency_report(no_worker, models, gates, profile="voice_isolation.opt_in")
        self.assertIn("live_worker_active", {c["gate"] for c in nw_row["checks"] if c["status"] == "fail"})

        # S12: profile ceiling must not loosen a stricter model ceiling (RNNoise 40 ms vs profile 65 ms).
        rnn_slow = json.loads(json.dumps(obj))
        rnn_slow["stats"]["model"] = "RNNoise"
        del rnn_slow["stats"]["worker"]
        rnn_slow["median_ms"] = 50.0
        rnn_row = harness.evaluate_latency_report(rnn_slow, models, gates, profile="voice_isolation.default_promotion")
        self.assertIn("max_macos_e2e_latency_ms", {c["gate"] for c in rnn_row["checks"] if c["status"] == "fail"})

        bad = json.loads(json.dumps(obj))
        bad["stimulus"] = "tone"
        bad["denoise"] = False
        bad["median_ms"] = 70.0
        bad["miss_rate"] = 0.1
        bad_row = harness.evaluate_latency_report(bad, models, gates, profile="voice_isolation.opt_in")
        failed = {c["gate"] for c in bad_row["checks"] if c["status"] == "fail"}
        self.assertTrue({"latency_stimulus", "latency_denoise", "max_macos_e2e_latency_ms", "max_speech_onset_miss_rate"} <= failed, failed)

        # S11: missing platform always fails (no fallback to host OS).
        no_plat = json.loads(json.dumps(obj))
        del no_plat["platform"]
        np_row = harness.evaluate_latency_report(no_plat, models, gates, profile="voice_isolation.opt_in")
        self.assertIn("latency_platform", {c["gate"] for c in np_row["checks"] if c["status"] == "fail"})

        # S2: non-numeric callbacks (bool/str) fail activity instead of passing or raising TypeError.
        for bad_cb in (True, "200"):
            bc = json.loads(json.dumps(obj))
            bc["stats"]["capture"]["callbacks"] = bad_cb
            bc_row = harness.evaluate_latency_report(bc, models, gates, profile="voice_isolation.opt_in")
            self.assertIn("live_capture_active", {c["gate"] for c in bc_row["checks"] if c["status"] == "fail"})

        # S13: misspelled --gates-profile raises HarnessError in latency-only runs too.
        with self.assertRaises(harness.HarnessError):
            harness.evaluate_latency_report(obj, models, gates, profile="voice_isolation.typo")

    def test_manifest_deduplicates_latency_gates_only_when_report_matches(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            w = sine(440, 16000, 0.25, 0.2)
            write_wav(p / "n.wav", 16000, w, "float32")
            write_wav(p / "c.wav", 16000, w, "float32")
            entries = [{"id": "a", "noisy": str(p / "n.wav"), "clean": str(p / "c.wav"), "kind": "speech"}]
            models = harness.resolve_models(["dfn3"], harness.load_models(REPO / "MODELS.toml"))
            gates = harness.load_toml(HERE / "gates.toml")
            # Without a latency report, max_speech_onset_miss_rate is flagged as missing.
            r_missing = harness.run_manifest(entries, models, ["balanced"], harness.OracleProcessor(), gates, p / "o1", profile="voice_isolation.opt_in", log=quiet_log)
            gates_missing = {c["gate"]: c["status"] for c in r_missing[0]["checks"]}
            self.assertEqual(gates_missing.get("max_speech_onset_miss_rate"), "missing")
            # A Linux-only latency report removes max_speech_onset_miss_rate, NOT max_macos_e2e_latency_ms.
            r_linux_only = harness.run_manifest(entries, models, ["balanced"], harness.OracleProcessor(), gates, p / "o2", profile="voice_isolation.default_promotion", log=quiet_log, latency_gates_by_model={"dfn3": {"max_speech_onset_miss_rate", "max_linux_e2e_latency_ms"}})
            gates_linux = {c["gate"]: c["status"] for c in r_linux_only[0]["checks"]}
            self.assertNotIn("max_speech_onset_miss_rate", gates_linux)
            self.assertEqual(gates_linux.get("max_macos_e2e_latency_ms"), "missing")

    def test_bench_command_uses_tier_when_live(self):
        spec = harness.load_models(REPO / "MODELS.toml")["dpdfnet2_48khz_hr"]
        self.assertEqual(harness.bench_command("oto", spec, 5, False)[1:4], ["bench", "--model", "dpdfnet"])
        self.assertEqual(harness.bench_command("oto", spec, 5, True)[1:5], ["bench", "--live", "--tier", "max"])
        with self.assertRaises(ValueError):
            harness.bench_command("oto", harness.load_models(REPO / "MODELS.toml")["nearfield_v2_3_fast"], 5, True)

    def test_rejects_garbage(self):
        with self.assertRaises(ValueError):
            harness.parse_bench_json("no json here")


FAKE_OTO = '''#!{python}
"""Stands in for `oto process IN OUT --model M --strength S [--no-denoise] [--no-delay-comp]`:
a bypassed chain (input delayed by the model's chain latency)."""
import sys
sys.path.insert(0, {here!r})
from wavio import read_wav, write_wav
LAT = {{"rnnoise": 960, "dfn3": 1920, "dpdfnet": 2880, "fastenhancer": 1024}}
a = sys.argv[1:]
if a == ["--help"]:
    print("USAGE:\\n    oto process INPUT.wav OUTPUT.wav [--model M]"); sys.exit(0)
assert a[0] == "process", a
inp, out = a[1], a[2]
model = a[a.index("--model") + 1]
w = read_wav(inp)
d = 0 if "--no-delay-comp" not in a else round(LAT[model] * w.rate / 48000)
write_wav(out, w.rate, [0.0] * d + w.mono, "float32")
'''


class HarnessEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="oto-eval-test-")
        self.out = Path(self.tmp.name)
        self.gates = harness.load_toml(HERE / "gates.toml")
        self.models = harness.load_models(REPO / "MODELS.toml")

    def tearDown(self):
        self.tmp.cleanup()

    def ci(self, proc, models=("dfn3",), strengths=("balanced",)):
        return harness.run_ci(harness.resolve_models(list(models), self.models), list(strengths), proc, self.gates, self.out, shared_fixtures(), quiet_log)

    def test_oracle_passes_every_ci_gate(self):
        rows = self.ci(harness.OracleProcessor(), ("rnnoise", "dpdfnet2_48khz_hr"), ("light", "strong"))
        self.assertEqual(len(rows), 4)
        for r in rows:
            statuses = {c["status"] for c in r["checks"]}
            self.assertEqual(statuses, {"pass"}, [c for c in r["checks"] if c["status"] != "pass"])
            self.assertEqual(r["calibration"]["delay_error_samples"], 0)
            self.assertFalse(r["calibration"]["compensated"])
        light = rows[0]["metrics"]
        self.assertAlmostEqual(light["noise_only_attenuation_db"], 12.0, delta=0.1)
        self.assertAlmostEqual(light["noise_only_model_attenuation_db"], 12.0, delta=0.1)
        self.assertAlmostEqual(light["noisy_5db_si_sdr_delta_db"], 12.0, delta=0.5)
        self.assertAlmostEqual(light["hfp_attenuation_db"], 12.0, delta=0.5)
        self.assertLess(light["envelope_delay_error_ms"], 3.0)

    def test_passthrough_fails_attenuation_only(self):
        (r,) = self.ci(harness.PassthroughProcessor())
        failed = {c["gate"] for c in r["checks"] if c["status"] == "fail"}
        self.assertEqual(failed, {"min_noise_only_attenuation_db", "min_noisy_15db_attenuation_db", "min_noisy_5db_attenuation_db", "min_hfp_attenuation_db", "min_noisy_5db_si_sdr_delta_db"})
        self.assertAlmostEqual(r["metrics"]["noise_only_model_attenuation_db"], 0.0, delta=1e-9)
        self.assertEqual(r["metrics"]["delay_error_samples"], 0)

    def test_wrong_delay_is_caught(self):
        class Late(harness.PassthroughProcessor):
            def process(self, model, strength, inp, out, bypass=False, hint=None):
                a = read_wav(inp)
                write_wav(out, a.rate, ([0.0] * (model.chain_latency_samples + 48) + a.mono)[: a.frames], "float32")

        (r,) = self.ci(Late())
        self.assertEqual(r["calibration"]["measured_samples"], 1968)
        self.assertIn("max_delay_error_samples", {c["gate"] for c in r["checks"] if c["status"] == "fail"})

    def test_strength_cap_is_checked_against_bypass(self):
        class TooStrong(harness.OracleProcessor):
            def process(self, model, strength, inp, out, bypass=False, hint=None):
                super().process(model, "strong" if not bypass else strength, inp, out, bypass, hint)

        (r,) = self.ci(TooStrong(), strengths=("light",))
        self.assertIn("max_noise_only_model_attenuation_db", {c["gate"] for c in r["checks"] if c["status"] == "fail"})

    def test_nan_and_clipping_are_caught(self):
        class Broken(harness.OracleProcessor):
            def process(self, model, strength, inp, out, bypass=False, hint=None):
                super().process(model, strength, inp, out, bypass, hint)
                if not bypass and hint and hint["name"] == "noise_only":
                    a = read_wav(out)
                    y = a.mono
                    y[1000], y[2000] = float("nan"), 1.5
                    write_wav(out, a.rate, y, "float32")

        (r,) = self.ci(Broken())
        failed = {c["gate"] for c in r["checks"] if c["status"] == "fail"}
        self.assertTrue({"max_nan_count", "max_clip_count", "max_peak_dbfs"} <= failed, failed)

    def test_process_errors_fail_the_row(self):
        class Crashes(harness.Processor):
            def process(self, *a, **k):
                raise harness.ProcessError("boom")

        (r,) = self.ci(Crashes())
        self.assertTrue(r["errors"])
        self.assertIn("process", {c["gate"] for c in r["checks"] if c["status"] == "fail"})

    def test_fake_oto_binary_through_the_cli(self):
        fake = self.out / "oto"
        fake.write_text(FAKE_OTO.format(python=sys.executable, here=str(HERE)))
        fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
        self.assertTrue(harness.oto_has(str(fake), "process"))
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = harness.main(["--suite", "ci", "--oto", str(fake), "--models", "rnnoise", "--strengths", "light", "--fixtures-dir", str(shared_fixtures()), "--out", str(self.out / "run")])
        self.assertEqual(code, 1)  # a bypassed chain removes no noise
        report = json.loads((self.out / "run/report.json").read_text())
        (row,) = report["results"]
        self.assertEqual(row["calibration"]["measured_samples"], 960)
        self.assertEqual(row["metrics"]["delay_error_samples"], 0)
        self.assertEqual(report["meta"]["oto"], str(fake))
        md = (self.out / "run/scorecard.md").read_text()
        self.assertIn("| rnnoise | light |", md)
        self.assertIn("min_noise_only_attenuation_db", md)
        self.assertIn("dpdfnet2_48khz_hr | crates/oto-dsp/models/dpdfnet2_48khz_hr.onnx", md)
        # Without --no-delay-comp the output is compensated; the harness accepts and records it.
        with redirect_stdout(io.StringIO()):
            harness.main(["--suite", "ci", "--oto", str(fake), "--process-template", "{oto} process {input} {output} --model {model} --strength {strength} {bypass}", "--models", "rnnoise", "--strengths", "light", "--fixtures-dir", str(shared_fixtures()), "--out", str(self.out / "comp")])
        row = json.loads((self.out / "comp/report.json").read_text())["results"][0]
        self.assertTrue(row["calibration"]["compensated"])
        self.assertEqual(row["metrics"]["delay_error_samples"], 0)

    def test_process_template(self):
        p = harness.OtoProcessor("/path with space/oto")
        spec = self.models["dpdfnet2_48khz_hr"]
        self.assertEqual(p.command(spec, "strong", Path("/a b/in.wav"), Path("/o.wav"), True), ["/path with space/oto", "process", "/a b/in.wav", "/o.wav", "--model", "dpdfnet", "--strength", "strong", "--no-denoise", "--no-delay-comp"])
        self.assertNotIn("--no-denoise", p.command(spec, "light", Path("i"), Path("o"), False))

    def test_cli_usage_errors(self):
        with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            self.assertEqual(harness.main([]), 2)
            self.assertEqual(harness.main(["--suite", "manifest"]), 2)
            self.assertEqual(harness.main(["--suite", "ci", "--processor", "oracle", "--models", "nope", "--out", str(self.out)]), 2)
            self.assertEqual(harness.main(["--suite", "ci", "--processor", "oracle", "--strengths", "max", "--out", str(self.out)]), 2)
            fake = self.out / "old-oto"
            fake.write_text("#!/bin/sh\necho 'USAGE: oto run | oto bench'\n")
            fake.chmod(0o755)
            self.assertFalse(harness.oto_has(str(fake), "process"))
            self.assertEqual(harness.main(["--suite", "ci", "--oto", str(fake), "--out", str(self.out)]), 2)

    def test_cli_oracle_passes_and_writes_reports(self):
        with redirect_stdout(io.StringIO()):
            code = harness.main(["--suite", "ci", "--processor", "oracle", "--models", "fastenhancer", "--strengths", "balanced", "--fixtures-dir", str(shared_fixtures()), "--out", str(self.out)])
        self.assertEqual(code, 0)
        report = json.loads((self.out / "report.json").read_text())
        self.assertTrue(report["summary"]["ok"])
        self.assertTrue(report["meta"]["provisional"])
        self.assertTrue(all(h["ok"] for h in report["hashes"]))
        self.assertIn("**PASS**", (self.out / "scorecard.md").read_text())

    def test_manifest_suite_with_fake_asr(self):
        fixtures = shared_fixtures()
        lines = [
            {"id": "speech-5db", "noisy": str(fixtures / "noisy_speech_5db.wav"), "clean": str(fixtures / "clean_speech_ref.wav"), "transcript": "please call me back today", "interferer_transcript": "turn on the tv", "noise_windows": [[0.5, 0.95]], "dataset": "synthetic", "license": "CC0"},
            {"id": "noise", "kind": "noise_only", "noisy": "noise_only.wav"},
            {"id": "bg", "kind": "background_only", "noisy": "noise_only.wav"},
        ]
        manifest = self.out / "m.jsonl"
        manifest.write_text("# local holdout\n" + "\n".join(json.dumps(l) for l in lines) + "\n")
        entries = harness.read_manifest(manifest)
        self.assertEqual(entries[1]["noisy"], str((self.out / "noise_only.wav").resolve()))
        for e in entries[1:]:
            e["noisy"] = str(fixtures / "noise_only.wav")

        def fake_asr(path):  # raw input: noisy words; processed: model-dependent text
            name = Path(path).name
            if "speech" in name:
                return "please call me back today" if "__" in name else "please fall me back today"
            if name.endswith("__noise.wav"):
                return ""
            if name.endswith("__bg.wav"):
                return "turn on"
            return "uh"

        rows = harness.run_manifest(entries, [self.models["dfn3"]], ["balanced"], harness.OracleProcessor(), self.gates, self.out / "run", "voice_isolation.opt_in", fake_asr, None, quiet_log)
        (row,) = rows
        m = row["metrics"]
        self.assertEqual(m["clips"], 3)
        self.assertAlmostEqual(m["si_sdr_delta_db"], 24.0, delta=0.5)
        self.assertAlmostEqual(m["quiet_attenuation_db"], 24.0, delta=0.5)
        self.assertEqual((m["target_wer"], m["raw_wer"]), (0.0, 0.2))
        self.assertAlmostEqual(m["target_wer_delta"], -0.2)
        self.assertEqual((m["first_word_deletion_rate"], m["wrong_speaker_rate"]), (0.0, 0.0))
        self.assertAlmostEqual(m["background_leaked_words_per_min"], 2 / (4 / 60))
        self.assertEqual(m["noise_only_hallucinated_words_per_min"], 0.0)
        self.assertAlmostEqual(m["raw_noise_only_hallucinated_words_per_min"], 1 / (4 / 60))
        by = {c["gate"]: c["status"] for c in row["checks"]}
        self.assertEqual(by["max_background_leaked_words_per_min"], "fail")
        self.assertEqual(by["max_target_wer_delta"], "pass")
        self.assertEqual(by["max_speech_onset_miss_rate"], "missing")  # comes from `oto latency`
        report = {"meta": {"date": "d", "processor": "oracle", "platform": "p", "gates": "gates.toml", "asr": {"engine": "fake"}}, "hashes": [], "results": rows}
        report["summary"] = harness.summarize(rows, [], strict=True)
        self.assertFalse(report["summary"]["ok"])
        self.assertIn("Same-ASR head-to-head", harness.markdown(report))

    def test_manifest_overlap_leakage(self):
        rate = 48000
        target = sine(220, rate, 1.0, 0.3)
        itf = sine(1250, rate, 1.0, 0.3)
        d = self.out
        write_wav(d / "t.wav", rate, target, "float32")
        write_wav(d / "i.wav", rate, itf, "float32")
        write_wav(d / "mix.wav", rate, [a + b for a, b in zip(target, itf)], "float32")
        spec = self.models["rnnoise"]
        lag = spec.chain_latency_samples
        write_wav(d / "out.wav", rate, ([0.0] * lag + [a + 0.1 * b for a, b in zip(target, itf)])[:rate], "float32")
        e = {"id": "x", "kind": "speech", "noisy": str(d / "mix.wav"), "clean": str(d / "t.wav"), "interferer": str(d / "i.wav")}
        c = harness.clip_metrics(e, read_wav(d / "mix.wav"), read_wav(d / "out.wav"), lag, d / "out.wav")
        self.assertAlmostEqual(c["overlap_leakage_db"], -20.0, delta=0.5)

    def test_manifest_validation(self):
        bad = self.out / "bad.jsonl"
        bad.write_text('{"id": "x"}\n')
        with self.assertRaises(harness.HarnessError):
            harness.read_manifest(bad)
        bad.write_text('{"noisy": "a.wav", "kind": "music"}\n')
        with self.assertRaises(harness.HarnessError):
            harness.read_manifest(bad)


def _live_oto():
    b = harness.find_oto(None)
    return b if b and harness.oto_has(b, "process") else None


@unittest.skipUnless(_live_oto(), "no built oto binary with `oto process` (cargo build -p oto-cli)")
class LiveOtoTests(unittest.TestCase):
    def test_rnnoise_bypass_delay_matches_models_toml(self):
        with tempfile.TemporaryDirectory() as d:
            proc = harness.OtoProcessor(_live_oto())
            fixtures = shared_fixtures()
            align = next(f for f in json.loads((fixtures / "fixtures.json").read_text())["fixtures"] if f["kind"] == "alignment")
            spec = harness.load_models(REPO / "MODELS.toml")["rnnoise"]
            cal = harness.calibrate_delay(proc, spec, Path(d), align, fixtures, 2)
            self.assertLessEqual(cal["delay_error_samples"], 2, cal)


if __name__ == "__main__":
    unittest.main()
