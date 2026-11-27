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
import krisp  # noqa: E402
import krisp_bench  # noqa: E402
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
        cases = [{"case": c, "strength": s, "leaked_samples": 0, "leaked_frames": 0, "fallback_to_other_model": 0, "checked_samples": 4800, "isolator": "nearfield" if c == "inference_error" else "silencer"} for c in fi["cases"] for s in fi["strengths"]]
        ok = harness.evaluate_fault_report({"cases": cases}, self.gates)
        self.assertTrue(all(c["status"] == "pass" for c in ok["checks"]), ok["checks"])
        # inference_error must run the real "nearfield" model (rejecting silencer, mock, empty, bool) and check >0 samples.
        for bad_iso in ("silencer", "mock", "", False, None):
            mock = [dict(c, isolator=bad_iso) if c["case"] == "inference_error" else c for c in cases]
            failed = {c["gate"] for c in harness.evaluate_fault_report({"cases": mock}, self.gates)["checks"] if c["status"] == "fail"}
            self.assertEqual(failed, {f"inference_error/{s}/real_model" for s in fi["strengths"]}, bad_iso)
        for bad_checked in (0, 0.5, float("inf"), True, None):
            zero_checked = [dict(c, checked_samples=bad_checked) if c["case"] == "startup" else c for c in cases]
            failed_zero = {c["gate"] for c in harness.evaluate_fault_report({"cases": zero_checked}, self.gates)["checks"] if c["status"] == "fail"}
            self.assertEqual(failed_zero, {f"startup/{s}/checked_samples" for s in fi["strengths"]}, bad_checked)
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
        self.assertEqual({r["model"] for r in rows if r["status"] == "ok"}, {"dpdfnet2_48khz_hr", "fastenhancer_s_48khz", "nearfield_v2_3_fast"})

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
        self.assertEqual((nf["latency_samples"], nf["cli_model"], nf["cli_tier"], nf["cargo_feature"]), (2014, "nearfield", "isolate", "nearfield"))
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
        obj = {"tier": "standard", "strength": "balanced", "denoise": True, "seconds": 5, "cpu_percent": 3.1, "errors": [], "notices": [], "stats": {"running": True, "stream_starts": 1, "model": "DeepFilterNet3", "estimated_latency_ms": 61.0, "underruns": 0, "resyncs": 0, "overflows": 0, "starvations": 0, "worker_overflows": 0, "capture": cb(200, 0), "output": cb(150, 1), "worker": cb(1300, 0)}}
        m = harness.parse_bench_json(json.dumps(obj))
        self.assertEqual((m["live_deadline_misses"], m["live_worker_p999_us"], m["bench_cpu_percent"], m["live_starvations"], m["live_restarts"]), (1, 1350, 3.1, 0, 0))
        m.pop("raw")
        spec = harness.load_models(REPO / "MODELS.toml")["dfn3"]
        gates = harness.load_toml(HERE / "gates.toml")
        failed = {c["gate"] for c in harness.bench_checks(m, spec, gates, live=True) if c["status"] == "fail"}
        self.assertEqual(failed, {"max_live_deadline_misses"})

        # Stream restart during live bench fails max_live_restarts even if final counters are 0.
        restarted = json.loads(json.dumps(obj))
        restarted["stats"]["output"] = cb(150, 0)
        restarted["stats"]["stream_starts"] = 2
        rm = harness.parse_bench_json(json.dumps(restarted))
        rm.pop("raw")
        self.assertIn("max_live_restarts", {c["gate"] for c in harness.bench_checks(rm, spec, gates, live=True) if c["status"] == "fail"})

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
            "strength": "balanced",
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
            "stats": {"running": True, "stream_starts": 1, "model": "DeepFilterNet3", "underruns": 0, "resyncs": 0, "overflows": 0, "starvations": 0, "worker_overflows": 0, "capture": cb(), "output": cb(), "worker": cb()},
        }
        models = harness.load_models(REPO / "MODELS.toml")
        gates = harness.load_toml(HERE / "gates.toml")
        row = harness.evaluate_latency_report(obj, models, gates, profile="voice_isolation.opt_in")
        self.assertEqual((row["model"], row["strength"]), ("dfn3", "balanced"))
        self.assertIn("macos_e2e_latency_ms", row["metrics"])
        self.assertTrue(all(c["status"] == "pass" for c in row["checks"]), row["checks"])

        # Missing/invalid strength fails latency_strength.
        no_str = json.loads(json.dumps(obj))
        del no_str["strength"]
        self.assertIn("latency_strength", {c["gate"] for c in harness.evaluate_latency_report(no_str, models, gates)["checks"] if c["status"] == "fail"})

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
            # A passing Light latency report must NOT clear Strong's latency gates.
            r_wrong_strength = harness.run_manifest(entries, models, ["strong"], harness.OracleProcessor(), gates, p / "o2", profile="voice_isolation.opt_in", log=quiet_log, latency_gates_by_key={("dfn3", "light"): {"max_speech_onset_miss_rate"}})
            gates_strong = {c["gate"]: c["status"] for c in r_wrong_strength[0]["checks"]}
            self.assertEqual(gates_strong.get("max_speech_onset_miss_rate"), "missing")
            # A Linux-only balanced report removes max_speech_onset_miss_rate for balanced, NOT max_macos_e2e_latency_ms.
            r_linux_only = harness.run_manifest(entries, models, ["balanced"], harness.OracleProcessor(), gates, p / "o3", profile="voice_isolation.default_promotion", log=quiet_log, latency_gates_by_key={("dfn3", "balanced"): {"max_speech_onset_miss_rate", "max_linux_e2e_latency_ms"}})
            gates_linux = {c["gate"]: c["status"] for c in r_linux_only[0]["checks"]}
            self.assertNotIn("max_speech_onset_miss_rate", gates_linux)
            self.assertEqual(gates_linux.get("max_macos_e2e_latency_ms"), "missing")
            # Baseline safety gates (max_nan_count, max_clip_count, max_delay_error_samples) always run in manifest mode.
            self.assertTrue({"max_nan_count", "max_clip_count", "max_delay_error_samples"} <= set(gates_linux))

    def test_greptile_regressions(self):
        # 1. Phase-invariant overlap leakage: a 90-degree phase-shifted 140 Hz interferer + 440 Hz target stays near 0 dB, not -40 dB.
        rate = 16000
        n = rate // 2
        target = [0.4 * math.sin(2 * math.pi * 440 * i / rate) for i in range(n)]
        itf = [0.4 * math.sin(2 * math.pi * 140 * i / rate) for i in range(n)]
        itf_shifted = [0.4 * math.cos(2 * math.pi * 140 * i / rate) for i in range(n)]
        inp = [a + b for a, b in zip(target, itf)]
        out_hpf_like = [a + b for a, b in zip(target, itf_shifted)]
        leak_same = mx.overlap_leakage_db(inp, out_hpf_like, itf, rate)
        self.assertIsNotNone(leak_same)
        self.assertGreater(leak_same, -1.0)
        out_attenuated = [a + 0.1 * b for a, b in zip(target, itf_shifted)]
        leak_20 = mx.overlap_leakage_db(inp, out_attenuated, itf, rate)
        self.assertAlmostEqual(leak_20, -20.0, delta=0.5)

        # 5. envelope_delay raises ValueError (not ZeroDivisionError) on empty/too-short signals.
        with self.assertRaises(ValueError):
            mx.envelope_delay([0.0] * 1000, 16000, [], 16000)

        # 6. first_word_deleted does not flag leading fillers before a preserved first word.
        self.assertFalse(asr.first_word_deleted("hello there", "well um hello there"))
        self.assertTrue(asr.first_word_deleted("hello there", "there"))
        self.assertTrue(asr.first_word_deleted("hello there", "yellow there"))

        # 7. All-silent calibration output fails impulse_delays / step_delay instead of returning delay 0.
        silence = [0.0] * 48000
        self.assertEqual(mx.impulse_delays(silence, [12000, 24000], 4000), [])
        self.assertIsNone(mx.step_delay(silence, 30000, 4000))

        # 9. Bad WAV headers (4-bit ADPCM, truncated fmt, rate=0) raise ValueError, not ZeroDivisionError/struct.error.
        with tempfile.TemporaryDirectory() as d:
            adpcm = Path(d) / "adpcm.wav"
            fmt = struct.pack("<HHIIHH", 2, 1, 16000, 8000, 1, 4)
            body = b"WAVEfmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", 4) + b"\x00\x00\x00\x00"
            adpcm.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
            with self.assertRaises(ValueError):
                read_wav(adpcm)
            short_fmt = Path(d) / "short.wav"
            body2 = b"WAVEfmt " + struct.pack("<I", 4) + b"\x01\x00\x01\x00" + b"data" + struct.pack("<I", 0)
            short_fmt.write_bytes(b"RIFF" + struct.pack("<I", len(body2)) + body2)
            with self.assertRaises(ValueError):
                read_wav(short_fmt)

    def test_bench_command_uses_tier_when_live(self):
        spec = harness.load_models(REPO / "MODELS.toml")["dpdfnet2_48khz_hr"]
        self.assertEqual(harness.bench_command("oto", spec, 5, False)[1:4], ["bench", "--model", "dpdfnet"])
        self.assertEqual(harness.bench_command("oto", spec, 5, True)[1:5], ["bench", "--live", "--tier", "max"])
        self.assertEqual(harness.bench_command("oto", harness.load_models(REPO / "MODELS.toml")["nearfield_v2_3_fast"], 5, True)[1:5], ["bench", "--live", "--tier", "isolate"])
        with self.assertRaises(ValueError):  # a model with no live tier cannot be benched live
            harness.bench_command("oto", harness.ModelSpec("x", "x", None, 480, 480), 5, True)

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

# ------------------------------------------------------------------------------------------
# Krisp Voice Isolation Benchmark (krisp.py, krisp_bench.py), Parakeet helpers, high band


def kseg(start, end, label, text=None):
    return {"start": start, "end": end, "label": label, "text": text}


def krow(i, caller, H, S=0, D=0, I=0, leak=0, bgmin=0.0, scen="work", noise_words=0, noise_min=0.0):
    return {"i": i, "scen": scen, "caller": caller, "H": H, "S": S, "D": D, "I": I, "leak": leak, "bgmin": bgmin, "noise_words": noise_words, "noise_min": noise_min}


class KrispNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.norm = krisp.KrispNormalizer("stdlib")

    def test_regex_rules(self):
        self.assertEqual(krisp.split_general_id("ticket AB-1234 now"), "ticket A B 1 2 3 4 now")
        self.assertEqual(krisp.split_digit_first("ref 12345B ok"), "ref 1 2 3 4 5 B ok")
        self.assertEqual(krisp.split_digit_first("the 1990s"), "the 1990s")
        self.assertEqual(krisp.split_standalone_4digits("PIN 4821 and 1969"), "PIN 4 8 2 1 and 1969")
        self.assertEqual(krisp.split_standalone_4digits("$5000 or 5000 dollars"), "$5000 or 5000 dollars")
        self.assertEqual(krisp.split_adjacent_digit_groups("call 555 123 4567"), "call 5 5 5 1 2 3 4 5 6 7")

    def test_number_words(self):
        self.assertEqual(krisp.number_words(0), "zero")
        self.assertEqual(krisp.number_words(123), "one hundred twenty three")
        self.assertEqual(krisp.number_words(35000), "thirty five thousand")
        self.assertEqual(krisp.number_words(1_000_001), "one million one")
        self.assertEqual(krisp.ordinal_words(14), "fourteenth")
        self.assertEqual(krisp.ordinal_words(21), "twenty first")
        self.assertEqual(krisp.ordinal_words(40), "fortieth")
        self.assertEqual(krisp.year_words(1969), "nineteen sixty nine")
        self.assertEqual(krisp.year_words(2024), "twenty twenty four")
        self.assertEqual(krisp.year_words(2005), "two thousand five")
        self.assertEqual(krisp.year_words(1905), "nineteen oh five")
        self.assertEqual(krisp.year_words(1900), "nineteen hundred")

    def test_stdlib_verbalize(self):
        v = krisp.stdlib_verbalize
        self.assertEqual(v("$5.50"), "five dollars fifty cents")
        self.assertEqual(v("$1 and £3"), "one dollar and three pounds")
        self.assertEqual(v("$2 million"), "two million dollars")
        self.assertEqual(v("50%"), "fifty percent")
        self.assertEqual(v("at 8:45 a.m."), "at eight forty five a m")
        self.assertEqual(v("at 8:05"), "at eight oh five")
        self.assertEqual(v("by 9:00"), "by nine o'clock")
        self.assertEqual(v("on August 14th"), "on August fourteenth")
        self.assertEqual(v("In 1969, 35,000 feet"), "In nineteen sixty nine, thirty five thousand feet")
        self.assertEqual(v("3.5 stars"), "three point five stars")
        self.assertEqual(v("the 1990s"), "the nineteen nineties")
        self.assertEqual(v("R&D"), "R and D")

    def test_jiwer_transform(self):
        self.assertEqual(krisp.jiwer_words("I won't, can't; let's go. They're here, it's John's, I'd, we'll, I've, I'm"), "i will not can not let us go they are here it is john is i would we will i have i am".split())
        self.assertEqual(krisp.jiwer_words("Um, uh, so... mhm OK-ish"), ["so", "ok", "ish"])
        self.assertEqual(krisp.jiwer_words(""), [])

    def test_full_normalizer_matches_spoken_form(self):
        written = "Uh, we leave on August 14th at 8:45 from Platform 6. Don't forget $20!"
        spoken = "we leave on august fourteenth at eight forty five from platform six do not forget twenty dollars"
        self.assertEqual(self.norm(written), spoken.split())
        self.assertEqual(self.norm(written), self.norm(spoken))
        self.assertEqual(self.norm(None), [])
        self.assertEqual(self.norm.backend, "stdlib")

    def test_leak_tokens(self):
        self.assertEqual(krisp.leak_tokens("Don't,"), ["don", "t"])  # as the published scorer counts
        self.assertEqual(krisp.leak_tokens("Uh,"), [])
        self.assertEqual(krisp.leak_tokens("Hello."), ["hello"])

    def test_nemo_backend_without_package(self):
        try:
            import nemo_text_processing  # noqa: F401
        except ImportError:
            with self.assertRaises(RuntimeError):
                krisp.KrispNormalizer("nemo")
            self.assertEqual(krisp.KrispNormalizer("auto").backend, "stdlib")
        else:  # pragma: no cover
            self.skipTest("nemo_text_processing installed")


class KrispMetricTests(unittest.TestCase):
    def setUp(self):
        self.norm = krisp.KrispNormalizer("stdlib")

    def test_segments_validation_and_regions(self):
        segs = krisp.validate_segments([kseg(0, 5, "primary", "hi"), kseg(5, 9, "mix", "x"), kseg(9, 12, "secondary"), kseg(12, 12.4, "secondary"), kseg(12.4, 14, "noise")])
        self.assertEqual(krisp.regions(segs, "secondary"), [(9.25, 11.75)])  # 0.4 s segment skipped
        self.assertEqual(krisp.regions(segs, "noise"), [(12.65, 13.75)])
        self.assertAlmostEqual(krisp.segment_minutes(segs, "secondary"), 3.4 / 60)
        with self.assertRaises(ValueError):
            krisp.validate_segments([kseg(0, 1, "music")])
        with self.assertRaises(ValueError):
            krisp.validate_segments([kseg(2, 1, "primary")])

    def test_words_in_regions_uses_midpoint_and_offset(self):
        regs = [(10.0, 12.0)]
        words = [("hello", 9.5, 10.6), ("there,", 11.0, 11.4), ("uh", 11.5, 11.6), ("don't", 11.7, 12.2), ("late", 12.1, 12.5)]
        self.assertEqual(krisp.words_in_regions(words, regs), 1 + 1 + 0 + 2)  # hello mid 10.05; don't mid 11.95
        self.assertEqual(krisp.words_in_regions(words, regs, offset=0.3), 1 + 0 + 2 + 1)  # late mid 12.3-0.3; hello mid 9.75 out

    def test_clip_row_counts(self):
        segs = [kseg(0, 4, "primary", "please call me back"), kseg(4, 8, "secondary"), kseg(8, 10, "noise")]
        words = [("Please", 0.2, 0.5), ("call", 0.6, 0.9), ("me", 1.0, 1.1), ("turn", 5.0, 5.3), ("on", 5.4, 5.5), ("the", 5.6, 5.7), ("TV.", 5.8, 6.0), ("hmm", 9.0, 9.2), ("yes", 9.3, 9.5)]
        r = krisp.clip_row("work/1", "work", "spk1", "Please call me back.", "Please call me. Turn on the TV yes", words, segs, self.norm)
        self.assertEqual((r["H"], r["S"], r["D"], r["I"]), (3, 1, 0, 4))  # back->turn, +on the tv yes
        self.assertEqual(r["leak"], 4)
        self.assertAlmostEqual(r["bgmin"], 3.5 / 60, places=5)
        self.assertEqual(r["noise_words"], 1)  # "hmm" is a filler
        self.assertAlmostEqual(r["noise_min"], 1.5 / 60, places=5)
        self.assertEqual((r["i"], r["scen"], r["caller"]), ("work/1", "work", "spk1"))

    def test_pooled_is_corpus_level(self):
        rows = [krow("a", "s1", H=9, S=1, leak=3, bgmin=0.5), krow("b", "s2", H=90, D=5, I=5, leak=0, bgmin=1.0)]
        p = krisp.pooled(rows)
        self.assertAlmostEqual(p["wer"], 100 * 11 / 105)  # not the mean of 10% and 10.5%
        self.assertAlmostEqual(p["missed"], 100 * 6 / 105)
        self.assertAlmostEqual(p["del"], 100 * 5 / 105)
        self.assertAlmostEqual(p["ins"], 100 * 5 / 105)
        self.assertAlmostEqual(p["leak"], 3 / 1.5)
        self.assertTrue(math.isnan(p["noise_wpm"]))
        self.assertTrue(math.isnan(krisp.pooled([])["wer"]))

    def test_percentile_matches_numpy_linear(self):
        self.assertEqual(krisp.percentile([4, 1, 3, 2], 25), 1.75)
        self.assertEqual(krisp.percentile([1, float("nan"), 3], 50), 2.0)
        self.assertTrue(math.isnan(krisp.percentile([], 50)))

    def test_speaker_bootstrap(self):
        rows = [krow(f"c{k}", f"s{k % 5}", H=90 + k, S=k % 4, D=1, I=k % 3, leak=k % 3, bgmin=0.3) for k in range(20)]
        ci = krisp.speaker_bootstrap(rows, n_boot=500, seed=1)
        p = krisp.pooled(rows)
        self.assertEqual(ci["speakers"], 5)
        for k in ("wer", "missed", "leak"):
            lo, hi = ci[k]
            self.assertLessEqual(lo, p[k])
            self.assertGreaterEqual(hi, p[k])
            self.assertLess(lo, hi)
        again = krisp.speaker_bootstrap(rows, n_boot=500, seed=1)  # deterministic
        self.assertEqual({k: ci[k] for k in ("wer", "missed", "leak")}, {k: again[k] for k in ("wer", "missed", "leak")})
        same = [krow(f"c{k}", f"s{k}", H=95, S=5, leak=2, bgmin=1.0) for k in range(6)]  # identical speakers
        ci = krisp.speaker_bootstrap(same, n_boot=200)
        self.assertAlmostEqual(ci["wer"][0], 5.0)
        self.assertAlmostEqual(ci["wer"][1], 5.0)
        self.assertAlmostEqual(ci["leak"][0], 2.0)

    def test_paired_bootstrap(self):
        ref = [krow(f"c{k}", f"s{k % 8}", H=90, S=6, D=1, I=3, leak=4 + k % 3, bgmin=0.5) for k in range(24)]
        better = [dict(r, S=r["S"] - 2, H=r["H"] + 2, leak=0) for r in ref]
        d = krisp.paired_bootstrap(better, ref, n_boot=300)
        self.assertAlmostEqual(d["wer"][0], -200 / 97)
        self.assertLess(d["wer"][2], 0)  # significantly better: whole interval below zero
        self.assertLess(d["leak"][2], 0)
        self.assertEqual((d["clips"], d["speakers"]), (24, 8))
        z = krisp.paired_bootstrap(ref, ref, n_boot=50)
        self.assertEqual(z["wer"], (0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            krisp.paired_bootstrap(ref, [krow("other", "s", H=1)])

    def test_summary_and_delta_metric_names(self):
        rows = [krow("work/1", "a", H=95, S=5, leak=2, bgmin=1.0), krow("phone_calls/1", "b", H=96, S=4, scen="phone_calls", noise_words=1, noise_min=0.5)]
        m = krisp.summary_metrics(rows, n_boot=100)
        self.assertAlmostEqual(m["krisp_wer_pct"], 4.5)
        self.assertAlmostEqual(m["phone_calls_krisp_wer_pct"], 4.0)
        self.assertAlmostEqual(m["work_secondary_leaked_words_per_min"], 2.0)
        self.assertAlmostEqual(m["noise_hallucinated_words_per_min"], 2.0)
        self.assertEqual((m["krisp_files"], m["krisp_speakers"]), (2, 2))
        for k in ("krisp_wer_pct_ci_lo", "krisp_wer_pct_ci_hi", "missed_caller_pct", "secondary_leaked_words_per_min_ci_hi"):
            self.assertIn(k, m)
        self.assertIn("raw_krisp_wer_pct", krisp.summary_metrics(rows, prefix="raw_", n_boot=10))
        d = krisp.delta_metrics(rows, rows, "krisp_default", n_boot=10)
        self.assertEqual(d["krisp_wer_pct_delta_vs_krisp_default"], 0.0)
        self.assertIn("secondary_leaked_words_per_min_delta_vs_krisp_default_ci_hi", d)
        self.assertEqual(krisp.summary_metrics([]), {})

    def test_cluster_mean_ci(self):
        point, lo, hi = krisp.cluster_mean_ci([0.1, 0.3, 0.2, 0.2], ["a", "a", "b", "b"], n_boot=200)
        self.assertAlmostEqual(point, 0.2)
        self.assertAlmostEqual(lo, 0.2)
        self.assertAlmostEqual(hi, 0.2)
        self.assertTrue(math.isnan(krisp.cluster_mean_ci([None], ["a"])[0]))

    def test_published_reference_rows(self):
        self.assertEqual(krisp.PUBLISHED["raw"], {"wer": 23.21, "missed": 4.28, "leak": 66.48})
        self.assertEqual((krisp.PUBLISHED["vi_2_5_default"]["wer"], krisp.PUBLISHED["vi_2_5_default"]["leak"]), (5.30, 1.52))


class HighBandTests(unittest.TestCase):
    @staticmethod
    def noise(rate, seconds, seed=3):
        import random as _r

        g = _r.Random(seed)
        return [g.uniform(-0.5, 0.5) for _ in range(int(rate * seconds))]

    def test_white_noise_share_and_band_limited_signal(self):
        x = self.noise(48000, 0.5)
        r = mx.band_energy_ratio_db(x, 48000)
        self.assertAlmostEqual(r, 10 * math.log10(8 / 24), delta=0.6)  # 16–24 kHz of a flat 0–24 kHz spectrum
        self.assertLess(mx.band_energy_ratio_db(sine(1000, 48000, 0.5, 0.5), 48000), -80.0)
        self.assertIsNone(mx.band_energy_ratio_db(x, 32000))  # Nyquist 16 kHz: no band to measure
        self.assertIsNone(mx.band_energy_ratio_db([0.0] * 4800, 48000))
        self.assertIsNone(mx.band_energy_ratio_db([0.1] * 100, 48000))
        self.assertAlmostEqual(mx.band_energy_ratio_db(x, 48000, 4000, 8000), 10 * math.log10(4 / 24), delta=0.6)

    def test_stdlib_fft_path_matches_numpy_path(self):
        x = [a + b for a, b in zip(self.noise(48000, 0.2), sine(18000, 48000, 0.2, 0.2))]
        saved = mx._np
        try:
            mx._np = None
            pure = mx.band_energy_ratio_db(x, 48000, max_frames=8)
        finally:
            mx._np = saved
        self.assertAlmostEqual(pure, mx.band_energy_ratio_db(x, 48000, max_frames=8), places=6)

    def test_high_band_metrics(self):
        rate = 48000
        fullband = [a + b for a, b in zip(sine(300, rate, 0.5, 0.3), self.noise(rate, 0.5))]
        m = mx.high_band_metrics(fullband, rate, fullband, rate)
        self.assertAlmostEqual(m["high_band_16k_delta_db"], 0.0, places=6)
        self.assertGreater(m["high_band_16k_db"], -60.0)
        lowpassed = [a + b for a, b in zip(sine(300, rate, 0.5, 0.3), sine(3000, rate, 0.5, 0.05))]
        m = mx.high_band_metrics(fullband, rate, lowpassed, rate)
        self.assertLess(m["high_band_16k_db"], -60.0)  # what a 16 kHz-band model looks like
        self.assertLess(m["high_band_16k_delta_db"], -20.0)
        self.assertEqual(mx.high_band_metrics(fullband[:16000], 32000, fullband, rate), {})  # 32 kHz input: not fullband
        self.assertEqual(mx.high_band_metrics(sine(300, rate, 0.5), rate, fullband, rate), {})  # no >16 kHz content in
        narrow = mx.high_band_metrics(fullband, rate, sine(300, 16000, 0.5, 0.3), 16000)
        self.assertEqual(narrow["high_band_16k_db"], mx.FLOOR_DB)
        self.assertLess(narrow["high_band_16k_delta_db"], -100.0)

    def test_gates_cover_fullband(self):
        g = harness.load_toml(HERE / "gates.toml")["oto1"]["clean"]
        checks = {c["gate"]: c["status"] for c in harness.evaluate_gates({"high_band_16k_db": -110.0, "high_band_16k_delta_db": -70.0}, g)}
        self.assertEqual((checks["min_high_band_16k_db"], checks["min_high_band_16k_delta_db"]), ("fail", "fail"))


class AsrBackendTests(unittest.TestCase):
    def test_words_from_tokens(self):
        self.assertEqual(asr.words_from_tokens([" hel", "lo", " world", ","], [0.0, 0.08, 0.4, 0.48]), [("hello", 0.0, 0.16), ("world,", 0.4, 0.56)])
        self.assertEqual(asr.words_from_tokens(["▁it", "'s", "▁ok"], [1.0, 1.08, 2.0]), [("it's", 1.0, 1.16), ("ok", 2.0, 2.08)])
        self.assertEqual(asr.words_from_tokens(["no", "space"], [0.0, 0.08]), [("nospace", 0.0, 0.16)])
        self.assertEqual(asr.words_from_tokens([], []), [])
        self.assertEqual(asr.words_from_tokens(None, None), [])

    def test_make_transcriber(self):
        with self.assertRaises(ValueError):
            asr.make_transcriber("kaldi")
        try:
            import onnx_asr  # noqa: F401
        except ImportError:
            with self.assertRaisesRegex(RuntimeError, "parakeet"):
                asr.make_transcriber("parakeet")
        self.assertEqual(asr.ParakeetTranscriber.DEFAULT_MODEL, "nemo-parakeet-tdt-0.6b-v2")

    def test_transcribe_helper(self):
        class Timed:
            def __call__(self, p):
                return "text"

            def transcribe_words(self, p):
                return "text", [("text", 0.0, 0.1)]

        self.assertEqual(asr.transcribe(Timed(), "a.wav", True), ("text", [("text", 0.0, 0.1)]))
        self.assertEqual(asr.transcribe(Timed(), "a.wav", False), ("text", None))
        self.assertEqual(asr.transcribe(lambda p: "plain", "a.wav", True), ("plain", None))

    def test_pyproject_extras_keep_base_stdlib(self):
        pp = tomllib.loads((HERE / "pyproject.toml").read_text())["project"]
        self.assertEqual(pp["dependencies"], [])
        self.assertTrue(any(d.startswith("onnx-asr") for d in pp["optional-dependencies"]["parakeet"]))


def make_krisp_data(root: Path, rate: int = 16000, seconds: float = 3.0) -> Path:
    """A tiny fake Krisp data dir: 2 work files (2 speakers), 1 phone call, VI Default for work."""
    meta = {
        "work": [
            {"file_name": "001.wav", "id": "001", "speaker_id": "speaker1", "gender": "male", "full_transcript": "Please call me back at 8:45.", "segments": [kseg(0.0, 1.0, "primary", "Please call"), kseg(1.0, 1.5, "mix", "me back at 8:45."), kseg(1.5, 2.6, "secondary"), kseg(2.6, 3.0, "noise")], "recording_device": "jabra", "environment": "small_room"},
            {"file_name": "002.wav", "id": "002", "speaker_id": "speaker2", "gender": "female", "full_transcript": "The order ships on the 14th.", "segments": [kseg(0.0, 1.2, "primary", "The order ships on the 14th."), kseg(1.2, 2.8, "secondary"), kseg(2.8, 3.0, "noise")], "recording_device": "airpods", "environment": "open_space"},
        ],
        "phone_calls": [
            {"file_name": "107.wav", "id": "107", "speaker_id": "speaker1", "gender": "male", "full_transcript": "Hi Olivia, I booked the tickets.", "segments": [kseg(0.2, 2.9, "primary", "Hi Olivia, I booked the tickets.")], "recording_device": "airpods", "environment": "car", "phone": "iphone_16", "car_model": None},
        ],
    }
    for scen, rows in meta.items():
        d = krisp_bench.raw_dir(root, scen)
        d.mkdir(parents=True)
        (d / "metadata.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        for r in rows:
            write_wav(d / r["file_name"], rate, sine(220, rate, seconds, 0.2), "pcm16")
    v = krisp_bench.variant_dir(root, "work", "vi_2_5_default")
    v.mkdir(parents=True)
    for r in meta["work"]:
        write_wav(v / r["file_name"], rate, sine(220, rate, seconds, 0.1), "pcm16")
    return root


class FakeTimedAsr:
    """Raw input leaks the background talker; Krisp's output leaks one word; anything else
    (the model's output) is clean. Word times follow the fake data's segments."""

    config = {"engine": "fake", "model": "timed", "quantization": "none"}
    CLEAN = {"001": "please call me back at 8:45", "002": "the order ships on the 14th", "107": "hi olivia i booked the tickets"}

    def __init__(self):
        self.calls = []

    def transcribe_words(self, path):
        self.calls.append(str(path))
        p = Path(path)
        key = next(k for k in self.CLEAN if k in p.name)
        words = [(w, 0.1 + 0.1 * i, 0.15 + 0.1 * i) for i, w in enumerate(self.CLEAN[key].split())]
        if key == "107":
            return self.CLEAN[key], words
        if p.name.startswith("raw__") or "/raw/" in str(p):
            leak = [("turn", 1.9, 2.0), ("on", 2.0, 2.1), ("the", 2.1, 2.2), ("tv", 2.2, 2.3)]
        elif "vi_2_5_default" in str(p):
            leak = [("tv", 2.0, 2.1)]
        else:
            leak = []
        return " ".join(w for w, _, _ in words + leak), words + leak

    def __call__(self, path):
        return self.transcribe_words(path)[0]


class KrispBenchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="oto-krisp-test-")
        self.root = make_krisp_data(Path(self.tmp.name) / "krisp")

    def tearDown(self):
        self.tmp.cleanup()

    def test_data_dir_must_be_outside_repo(self):
        for bad in (REPO, REPO / "tools/eval/out/krisp", HERE):
            with self.assertRaises(krisp_bench.BenchError):
                krisp_bench.data_dir(bad)
        self.assertEqual(krisp_bench.data_dir(self.root), self.root.resolve())
        self.assertEqual(krisp_bench.data_dir(None), Path("~/.cache/oto-eval/krisp").expanduser().resolve())
        with self.assertRaises(krisp_bench.BenchError):
            krisp_bench.manifest_path(self.root, str(REPO / "krisp.jsonl"))
        with redirect_stderr(io.StringIO()):
            self.assertEqual(krisp_bench.main(["manifest", "--data-dir", str(REPO / "data")]), 2)

    def test_build_manifest(self):
        entries = krisp_bench.build_manifest(self.root)
        self.assertEqual([e["id"] for e in entries], ["phone_calls/107", "work/001", "work/002"])
        w = entries[1]
        self.assertEqual((w["kind"], w["speaker_id"], w["scenario"], w["license"]), ("speech", "speaker1", "work", "CC-BY-NC-4.0"))
        self.assertEqual(w["transcript"], "Please call me back at 8:45.")
        self.assertEqual([s["label"] for s in w["segments"]], ["primary", "mix", "secondary", "noise"])
        self.assertEqual(set(w["baselines"]), {"krisp_default"})
        self.assertTrue(Path(w["baselines"]["krisp_default"]).is_file())
        self.assertNotIn("baselines", entries[0])  # no VI output downloaded for phone calls
        self.assertEqual(len(krisp_bench.build_manifest(self.root, limit=2)), 2)
        self.assertEqual([e["id"] for e in krisp_bench.build_manifest(self.root, scenarios=["work"])], ["work/001", "work/002"])
        with self.assertRaises(krisp_bench.BenchError):
            krisp_bench.build_manifest(Path(self.tmp.name) / "empty")

    def test_manifest_round_trips_through_eval(self):
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(krisp_bench.main(["manifest", "--data-dir", str(self.root)]), 0)
        self.assertIn("3 files, 2 speakers", out.getvalue())
        m = self.root / "manifest.jsonl"
        self.assertTrue(m.read_text().startswith("# Krisp Voice Isolation Benchmark (CC-BY-NC-4.0)"))
        entries = harness.read_manifest(m)
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[1]["segments"][2]["label"], "secondary")
        self.assertTrue(Path(entries[1]["baselines"]["krisp_default"]).is_absolute())
        bad = self.root / "bad.jsonl"
        bad.write_text(json.dumps({"noisy": "a.wav", "segments": [kseg(0, 1, "music")]}) + "\n")
        with self.assertRaises(harness.HarnessError):
            harness.read_manifest(bad)
        bad.write_text(json.dumps({"noisy": "a.wav", "baselines": ["x.wav"]}) + "\n")
        with self.assertRaises(harness.HarnessError):
            harness.read_manifest(bad)

    def test_plan_download_and_disk_guard(self):
        def lister(repo, sub):
            if repo == krisp_bench.RAW_REPO:
                return [{"path": f"{sub}/metadata.jsonl", "size": 10}, {"path": f"{sub}/001.wav", "size": (self.root / "raw" / sub / "001.wav").stat().st_size if sub == "work" else 5}, {"path": f"{sub}/999.wav", "size": 1000}, {"path": f"{sub}/nested/x.wav", "size": 1}]
            return [{"path": f"{sub}/001.wav", "size": 7}, {"path": f"{sub}/.DS_Store", "size": 6148}]

        jobs = krisp_bench.plan_download(self.root, ["work"], ["vi_2_5_default"], lister)
        dests = sorted(Path(j["dest"]).relative_to(self.root).as_posix() for j in jobs)
        # metadata.jsonl and work/001.wav already present at the same size are skipped (meta size differs: refetched)
        self.assertEqual(dests, ["processed/work/vi_2_5_default/001.wav", "raw/work/999.wav", "raw/work/metadata.jsonl"])
        self.assertTrue(all(j["url"].startswith("https://huggingface.co/datasets/Krisp-AI/") and "/resolve/main/" in j["url"] for j in jobs))

        class U:
            def __init__(self, free):
                self.free = free

        info = krisp_bench.check_disk(self.root / "new" / "dir", 2 * 10**9, 20.0, usage=lambda p: U(60 * 10**9))
        self.assertEqual((info["need_gb"], info["free_after_gb"]), (2.0, 58.0))
        with self.assertRaisesRegex(krisp_bench.BenchError, "not enough disk"):
            krisp_bench.check_disk(self.root, 2 * 10**9, 20.0, usage=lambda p: U(21 * 10**9))

        args = krisp_bench.parse_args(["download", "--data-dir", str(self.root), "--scenarios", "work", "--dry-run"])
        fetched, lines = [], []
        self.assertEqual(krisp_bench.cmd_download(args, lines.append, lister, fetched.append, lambda p: U(60 * 10**9)), 0)
        self.assertEqual(fetched, [])  # dry run writes nothing
        self.assertIn("3 files", lines[0])
        args = krisp_bench.parse_args(["download", "--data-dir", str(self.root), "--scenarios", "work"])
        self.assertEqual(krisp_bench.cmd_download(args, lines.append, lister, fetched.append, lambda p: U(60 * 10**9)), 0)
        self.assertEqual(len(fetched), 3)
        with self.assertRaises(krisp_bench.BenchError):
            krisp_bench.cmd_download(args, lines.append, lister, fetched.append, lambda p: U(10**9))

    def test_asr_cache_score_and_report(self):
        fake = FakeTimedAsr()
        entries = krisp_bench.build_manifest(self.root)
        tag = krisp_bench.run_asr(entries, ["raw", "vi_2_5_default"], fake, self.root, quiet_log)
        self.assertEqual(tag, "fake-timed-none")
        n = len(fake.calls)
        self.assertEqual(n, 3 + 2)  # VI Default exists for the 2 work files only
        krisp_bench.run_asr(entries, ["raw"], fake, self.root, quiet_log)
        self.assertEqual(len(fake.calls), n)  # cached
        args = krisp_bench.parse_args(["score", "--data-dir", str(self.root), "--cond", "raw", "--cond", "vi_2_5_default", "--normalizer", "stdlib", "--bootstrap", "50"])
        lines = []
        self.assertEqual(krisp_bench.cmd_score(args, lines.append, fake.config), 0)
        rep = json.loads((self.root / "reports" / "krisp.json").read_text())
        raw, vi = rep["scores"]["raw"], rep["scores"]["vi_2_5_default"]
        self.assertEqual((raw["files"], raw["speakers"], vi["files"]), (3, 2, 2))
        self.assertAlmostEqual(raw["leak"], 8 / ((1.1 - 0.5 + 1.6 - 0.5) / 60), places=1)  # 4 leaked words in each work file
        self.assertAlmostEqual(vi["leak"], 2 / ((1.1 - 0.5 + 1.6 - 0.5) / 60), places=1)
        self.assertAlmostEqual(vi["wer"], 2 / 14 * 100)  # one inserted "tv" per work file; "8:45" is 3 of the 14 reference words
        self.assertIn("vs_reference", raw)
        self.assertEqual(rep["checks"], [])  # not the full 265 files
        self.assertTrue((self.root / "reports" / "krisp.clips.jsonl").is_file())
        self.assertIn("| raw | 3 / 2 |", "\n".join(lines))
        args = krisp_bench.parse_args(["score", "--data-dir", str(self.root), "--cond", "vi_2_5_hd"])
        with self.assertRaisesRegex(krisp_bench.BenchError, "no transcripts"):
            krisp_bench.cmd_score(args, quiet_log, fake.config)

    def test_score_published_clips_and_check(self):
        def rows(H, S, D, I, leak, bgmin):
            out = [krow(f"work/{k}", f"speaker{k % 47}", 0) for k in range(264)]
            return [krow("work/x", "speaker0", H, S, D, I, leak, bgmin)] + out

        raw = rows(9572, 300, 128, 1893, 6648, 100.0)  # 23.21% WER, 66.48 w/min
        vi = rows(9579, 320, 101, 109, 152, 100.0)  # 5.30% WER, 1.52 w/min
        clips = Path(self.tmp.name) / "clips.jsonl"
        clips.write_text(json.dumps({"set": "krisp", "cond": "raw", "clips": raw}) + "\n" + json.dumps({"set": "krisp", "cond": "krisp_vi_2_5_default", "clips": vi}) + "\n" + json.dumps({"set": "ami", "cond": "raw", "clips": []}) + "\n")
        got = krisp_bench.read_clips_jsonl(clips)
        self.assertEqual(sorted(got), ["raw", "vi_2_5_default"])
        scores = krisp_bench.score_conditions(got, "vi_2_5_default", n_boot=20)
        checks = {c["cond"]: c for c in krisp_bench.check_published(scores)}
        self.assertTrue(checks["raw"]["ok"] and checks["vi_2_5_default"]["ok"], checks)
        self.assertAlmostEqual(checks["raw"]["leak"], 66.48)
        lines = []
        args = krisp_bench.parse_args(["score", "--data-dir", str(self.root), "--clips", str(clips), "--check", "--bootstrap", "20"])
        self.assertEqual(krisp_bench.cmd_score(args, lines.append), 0)
        self.assertIn("OK raw: WER 23.21 vs published 23.21", "\n".join(lines))
        self.assertFalse((self.root / "reports").exists())  # --clips without --write
        off = rows(9572, 300, 128, 1950, 6648, 100.0)  # 23.78%: outside 0.2 points
        clips.write_text(json.dumps({"set": "krisp", "cond": "raw", "clips": off}) + "\n")
        self.assertEqual(krisp_bench.cmd_score(args, quiet_log), 1)

    def test_eval_command(self):
        args = krisp_bench.parse_args(["eval", "--data-dir", str(self.root), "--models", "dfn3", "--asr-quantization", "int8", "--", "--strict"])
        calls = []
        self.assertEqual(krisp_bench.cmd_eval(args, quiet_log, lambda cmd: calls.append(cmd) or 0), 0)
        (cmd,) = calls
        self.assertEqual(cmd[1], str(HERE / "eval.py"))
        joined = " ".join(cmd)
        for part in ("--suite manifest", f"--manifest {self.root.resolve() / 'manifest.jsonl'}", "--models dfn3", "--asr --asr-backend parakeet", "--gates-profile oto1.isolate", f"--out {self.root.resolve() / 'eval-out'}", "--asr-quantization int8"):
            self.assertIn(part, joined)
        self.assertEqual(cmd[-1], "--strict")
        self.assertEqual(krisp_bench.parse_args(["eval"]).models, "dfn3,dpdfnet,nearfield")


class KrispEvalIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="oto-krisp-eval-")
        self.root = make_krisp_data(Path(self.tmp.name) / "krisp")
        self.gates = harness.load_toml(HERE / "gates.toml")
        self.models = harness.load_models(REPO / "MODELS.toml")

    def tearDown(self):
        self.tmp.cleanup()

    def test_manifest_suite_scores_krisp_rows(self):
        m = krisp_bench.write_manifest(krisp_bench.build_manifest(self.root), self.root / "manifest.jsonl")
        entries = harness.read_manifest(m)
        fake = FakeTimedAsr()
        rows = harness.run_manifest(entries, [self.models["dfn3"]], ["balanced"], harness.OracleProcessor(), self.gates, Path(self.tmp.name) / "run", "oto1.isolate", fake, None, quiet_log, normalizer=krisp.KrispNormalizer("stdlib"), n_boot=100)
        (row,) = rows
        mt = row["metrics"]
        self.assertEqual(mt["krisp_files"], 3)
        self.assertEqual(mt["krisp_speakers"], 2)
        self.assertEqual(mt["krisp_wer_pct"], 0.0)  # model output: clean transcript
        self.assertEqual(mt["secondary_leaked_words_per_min"], 0.0)
        self.assertGreater(mt["raw_secondary_leaked_words_per_min"], 100)
        self.assertEqual(mt["krisp_default_krisp_files"], 2)
        self.assertEqual(mt["krisp_references"], ["krisp_default", "raw"])
        self.assertLess(mt["secondary_leaked_words_per_min_delta_vs_krisp_default"], 0)
        self.assertLess(mt["krisp_wer_pct_delta_vs_raw"], 0)
        self.assertEqual(mt["phone_calls_krisp_wer_pct"], 0.0)
        by = {c["gate"]: c for c in row["checks"]}
        self.assertEqual(by["max_phone_calls_krisp_wer_pct"]["status"], "pass")
        self.assertEqual(by["max_krisp_wer_pct_delta_vs_krisp_default_ci_lo"]["status"], "pass")
        self.assertEqual(by["max_secondary_leaked_words_per_min_delta_vs_krisp_default_ci_hi"]["status"], "pass")
        self.assertEqual(by["min_krisp_files"]["status"], "fail")  # 3 of 265 files
        self.assertEqual(by["min_krisp_speakers"]["status"], "fail")
        self.assertEqual(mt["krisp_matched_files_vs_krisp_default"], 2)
        self.assertEqual(mt["krisp_matched_speakers_vs_krisp_default"], 2)
        self.assertEqual(by["min_krisp_matched_files_vs_krisp_default"]["status"], "fail")
        self.assertEqual(by["min_krisp_matched_speakers_vs_krisp_default"]["status"], "fail")
        self.assertEqual(by["max_heap_allocations_per_frame"]["status"], "missing")
        # raw, baseline and model each transcribed once per clip (baselines are not re-run per row)
        self.assertEqual(len(fake.calls), 3 + 2 + 3)
        report = {"meta": {"date": "d", "processor": "oracle", "platform": "p", "gates": "gates.toml", "asr": fake.config}, "hashes": [], "results": rows}
        report["summary"] = harness.summarize(rows, [], strict=False)
        md = harness.markdown(report)
        self.assertIn("### Krisp Voice Isolation Benchmark scoring", md)
        self.assertIn("| **dfn3 / balanced** | 3 / 2 |", md)
        self.assertIn("↳ Δ vs krisp_default", md)

    def test_word_times_are_shifted_by_the_chain_delay(self):
        e = harness.read_manifest(krisp_bench.write_manifest(krisp_bench.build_manifest(self.root, scenarios=["work"]), self.root / "m.jsonl"))[0]

        class Late:
            def transcribe_words(self, p):  # one background word just after the secondary region ends, in output time
                return "please call me back at 8:45 tv", [("please", 0.1, 0.2), ("tv", 2.38, 2.42)]

        inp = read_wav(e["noisy"])
        c0 = harness.clip_metrics(e, inp, inp, 0, Path(e["noisy"]), Late(), "", None, krisp.KrispNormalizer("stdlib"))
        c1 = harness.clip_metrics(e, inp, inp, 2400, Path(e["noisy"]), Late(), "", None, krisp.KrispNormalizer("stdlib"))  # 50 ms at 48 kHz
        self.assertEqual((c0["krisp"]["leak"], c1["krisp"]["leak"]), (0, 1))

    def test_baseline_dnsmos_deltas(self):
        clips = [{"id": f"c{k}", "kind": "speech", "duration_s": 1, "nan_count": 0, "clip_count": 0, "speaker_id": f"s{k % 2}", "dnsmos_sig": 3.5, "dnsmos_ovr": 3.0, "baseline_dnsmos": {"krisp_nc": {"dnsmos_sig": 3.4, "dnsmos_ovr": 3.1, "dnsmos_bak": 4.0}}} for k in range(4)]
        a = harness.aggregate_manifest(clips, n_boot=50)
        self.assertAlmostEqual(a["dnsmos_sig_delta_vs_krisp_nc"], 0.1)
        self.assertAlmostEqual(a["dnsmos_ovr_delta_vs_krisp_nc_ci_hi"], -0.1)
        self.assertNotIn("dnsmos_bak_delta_vs_krisp_nc", a)  # the model row has no BAK
        checks = {c["gate"]: c["status"] for c in harness.evaluate_gates(a, self.gates["oto1"]["clean"])}
        self.assertEqual((checks["min_dnsmos_sig_delta_vs_krisp_nc_ci_hi"], checks["min_dnsmos_ovr_delta_vs_krisp_nc_ci_hi"]), ("pass", "fail"))


class Oto1GateTests(unittest.TestCase):
    def test_plan_0004_gates(self):
        g = harness.load_toml(HERE / "gates.toml")
        self.assertIn("ci", g)  # older sections untouched
        self.assertIn("opt_in", g["voice_isolation"])
        o = g["oto1"]
        self.assertEqual((o["sample_rate_hz"], o["n_fft"], o["hop"]), (48000, 1024, 512))
        clean, iso = o["clean"], o["isolate"]
        self.assertEqual(clean["max_chain_latency_ms"], 21.33)
        self.assertEqual(clean["max_macos_e2e_latency_ms"], 40.0)
        self.assertEqual(clean["max_bench_cpu_percent"], 5.0)
        self.assertEqual(clean["max_heap_allocations_per_frame"], 0)
        self.assertEqual(clean["max_wasm_brotli_bytes"], 1_500_000)
        self.assertEqual((clean["max_browser_paced_p99_ms"], clean["max_browser_racing_mean_ms"]), (2.0, 0.5))
        self.assertEqual(clean["min_dnsmos_bak_delta_vs_krisp_nc"], -0.1)
        self.assertEqual(clean["max_target_wer_delta"], 0.002)
        self.assertLessEqual(clean["max_chain_latency_ms"] * 512 / 1000 * 48000 / 512, 1024 + 1)  # 21.33 ms ≈ 1024 samples
        self.assertEqual(iso["max_chain_latency_ms"], 21.33)
        self.assertEqual(iso["max_phone_calls_krisp_wer_pct"], 4.15)
        self.assertEqual(iso["max_krisp_wer_pct_delta_vs_krisp_default_ci_lo"], 0.0)
        self.assertLess(iso["max_secondary_leaked_words_per_min_delta_vs_krisp_default_ci_hi"], 0.0)
        self.assertEqual((iso["min_krisp_files"], iso["min_krisp_speakers"]), (265, 47))
        self.assertEqual((iso["min_krisp_matched_files_vs_krisp_default"], iso["min_krisp_matched_speakers_vs_krisp_default"]), (265, 47))
        # Gate names must match metric names krisp.py / eval.py produce.
        produced = set(krisp.summary_metrics([krow("phone_calls/1", "a", 9, 1, leak=1, bgmin=1.0, scen="phone_calls")], n_boot=5))
        produced |= set(krisp.delta_metrics([krow("x", "a", 9, 1, leak=1, bgmin=1.0)], [krow("x", "a", 9, 1, leak=1, bgmin=1.0)], "krisp_default", n_boot=5))
        for k in ("max_phone_calls_krisp_wer_pct", "max_krisp_wer_pct_delta_vs_krisp_default_ci_lo", "max_secondary_leaked_words_per_min_delta_vs_krisp_default_ci_hi", "min_krisp_files", "min_krisp_speakers", "min_krisp_matched_files_vs_krisp_default", "min_krisp_matched_speakers_vs_krisp_default"):
            self.assertIn(k[4:], produced, k)
        self.assertEqual(krisp_bench.baseline_name("vi_2_5_default"), "krisp_default")

    def test_check_published_fails_on_incomplete_condition_and_token_unredirected(self):
        # 265 raw matching published WER, but only 264 krisp default clips -> must fail!
        scores = {
            "raw": {"files": 265, "wer": 23.21, "leak": 66.48},
            "vi_2_5_default": {"files": 264, "wer": 5.30, "leak": 1.52},
        }
        checks = krisp_bench.check_published(scores, required=["raw", "vi_2_5_default"])
        by = {c["cond"]: c for c in checks}
        self.assertTrue(by["raw"]["ok"])
        self.assertFalse(by["vi_2_5_default"]["ok"])
        self.assertIn("264/265", by["vi_2_5_default"]["reason"])

        # Alias `krisp_default`, `default`, and `krisp_vi_2_5_default` all resolve to `vi_2_5_default`, and collisions fail.
        alias_scores = {"raw": scores["raw"], "krisp_default": scores["vi_2_5_default"]}
        alias_checks = {c["cond"]: c for c in krisp_bench.check_published(alias_scores, required=["raw", "krisp_default"])}
        self.assertFalse(alias_checks["vi_2_5_default"]["ok"])
        self.assertEqual(krisp_bench.canonical_cond("default"), "vi_2_5_default")
        self.assertEqual(krisp_bench.canonical_cond("krisp_vi_2_5_default"), "vi_2_5_default")
        with self.assertRaises(ValueError):
            krisp_bench.check_published({"krisp_default": scores["vi_2_5_default"], "vi_2_5_default": scores["vi_2_5_default"]})

        # Duplicate or missing/null clip IDs must be rejected before scoring/pairing so coverage cannot be spoofed.
        dup = [krow("x", "a", 9, 1), krow("x", "a", 9, 1)]
        with self.assertRaises(ValueError):
            krisp.summary_metrics(dup, n_boot=5)
        with self.assertRaises(ValueError):
            krisp.paired_bootstrap(dup, [krow("x", "a", 9, 1)], n_boot=5)
        null_id = [krow("x", "a", 9, 1), {**krow("y", "a", 9, 1), "i": None}]
        with self.assertRaises(ValueError):
            krisp.summary_metrics(null_id, n_boot=5)
        with self.assertRaises(ValueError):
            krisp.paired_bootstrap(null_id, null_id, n_boot=5)

        old = os.environ.get("HF_TOKEN")
        try:
            os.environ["HF_TOKEN"] = "secret-token"
            hf_req = krisp_bench._request("https://huggingface.co/datasets/Krisp-AI/x")
            self.assertNotIn("Authorization", hf_req.headers)
            self.assertEqual(hf_req.unredirected_hdrs.get("Authorization"), "Bearer secret-token")
            other_req = krisp_bench._request("https://example.com/file.wav")
            self.assertNotIn("Authorization", other_req.headers)
            self.assertNotIn("Authorization", other_req.unredirected_hdrs)
        finally:
            if old is None:
                os.environ.pop("HF_TOKEN", None)
            else:
                os.environ["HF_TOKEN"] = old

    def test_cached_score_with_alias_preserves_paired_deltas(self):
        with tempfile.TemporaryDirectory(prefix="oto-krisp-alias-") as td:
            root = make_krisp_data(Path(td) / "krisp")
            fake = FakeTimedAsr()
            entries = krisp_bench.build_manifest(root)
            tag = krisp_bench.run_asr(entries, ["raw", "krisp_default"], fake, root, quiet_log)
            alias_cf = krisp_bench.cache_file(root, tag, "krisp_default")
            self.assertTrue(alias_cf.is_file())
            # A partial canonical cache must not shadow the explicitly requested alias cache.
            canon_cf = krisp_bench.cache_file(root, tag, "vi_2_5_default")
            alias_data = json.loads(alias_cf.read_text())
            first_id = next(iter(alias_data))
            canon_cf.write_text(json.dumps({first_id: alias_data[first_id]}))
            args = krisp_bench.parse_args(["score", "--data-dir", str(root), "--cond", "raw", "--cond", "krisp_default", "--reference", "krisp_default", "--normalizer", "stdlib", "--bootstrap", "20", "--name", "alias"])
            rc = krisp_bench.cmd_score(args, quiet_log, fake.config)
            self.assertEqual(rc, 0)
            rep = json.loads((root / "reports" / "alias.json").read_text())
            self.assertEqual(rep["scores"]["vi_2_5_default"]["files"], len(alias_data))
            vs = rep["scores"]["raw"].get("vs_reference")
            self.assertIsNotNone(vs)
            self.assertEqual(vs["reference"], "vi_2_5_default")
            self.assertIn("wer", vs)
            self.assertIn("leak", vs)


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
