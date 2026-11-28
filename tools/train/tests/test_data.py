import csv
import json

import numpy as np
import pytest
import torch

from oto_train.data import pack_shards as P
from oto_train.data.bandwidth import estimate_bandwidth, urgent_target_rate
from oto_train.data.mixer import Mixer, MixConfig
from oto_train.data.stream import SyntheticStream, iter_tar


# -- licence filters -----------------------------------------------------------


@pytest.mark.parametrize("url,ok", [
    ("http://creativecommons.org/publicdomain/zero/1.0/", True),
    ("http://creativecommons.org/licenses/by/3.0/", True),
    ("https://creativecommons.org/licenses/by/4.0/", True),
    ("http://creativecommons.org/licenses/by-nc/3.0/", False),
    ("http://creativecommons.org/licenses/by-sa/4.0/", False),
    ("http://creativecommons.org/licenses/sampling+/1.0/", False),
    ("", False),
])
def test_fsd50k_licence(url, ok):
    assert P.fsd50k_licence_ok(url) is ok


@pytest.mark.parametrize("text,ok", [
    ("Attribution", True),
    ("Attribution 4.0 International", True),
    ("Creative Commons Attribution 3.0 United States", True),
    ("CC0 1.0 Universal", True),
    ("Attribution-NonCommercial-ShareAlike 3.0 International", False),
    ("Attribution-NonCommercial", False),
    ("Attribution-ShareAlike", False),
    ("Attribution-NoDerivatives 4.0 International", False),
    ("Attribution-NonCommercial-NoDerivatives (aka Music Sharing) 3.0 International", False),
    ("FMA-Limited: Download Only", False),
    ("Free Art License", False),
    ("", False),
])
def test_fma_licence(text, ok):
    assert P.fma_licence_ok(text) is ok


def test_denylist_and_unknown_sources():
    for s in ("daps", "ears", "demand", "wham", "voxceleb", "something_new"):
        with pytest.raises(SystemExit):
            P.check_source(s)
    P.check_source("vctk")


def test_collect_fsd50k(tmp_path):
    root = tmp_path / "FSD50K"
    (root / "FSD50K.metadata").mkdir(parents=True)
    gt = root / "FSD50K.ground_truth"
    gt.mkdir()
    info = {
        "1": {"license": "http://creativecommons.org/publicdomain/zero/1.0/", "uploader": "u1", "title": "rain"},
        "2": {"license": "http://creativecommons.org/licenses/by/3.0/", "uploader": "u2", "title": "talk"},
        "3": {"license": "http://creativecommons.org/licenses/by-nc/3.0/", "uploader": "u3", "title": "nc"},
        "4": {"license": "http://creativecommons.org/licenses/sampling+/1.0/", "uploader": "u4", "title": "s+"},
        "5": {"license": "http://creativecommons.org/publicdomain/zero/1.0/", "uploader": "u5", "title": "missing_label"},
    }
    (root / "FSD50K.metadata" / "dev_clips_info_FSD50K.json").write_text(json.dumps(info))
    (gt / "dev.csv").write_text("fname,labels,mids,split\n1,Rain,/m/x,train\n2,\"Speech,Human_voice\",/m/y,train\n")
    items, stats = P.collect_fsd50k(str(root))
    assert {i.key for i in items} == {"fsd50k_1", "fsd50k_2"}
    assert {i.key: i.kind for i in items} == {"fsd50k_1": "noise", "fsd50k_2": "interferer"}
    assert stats["drop_licence"] == 2
    assert stats["drop_unlabeled"] == 1

    # Missing ground-truth labels for a supplied partition must fail closed (never route voice into noise).
    (gt / "dev.csv").unlink()
    with pytest.raises(FileNotFoundError):
        P.collect_fsd50k(str(root))


def test_collect_dir_refuses_dedicated_filter_sources(tmp_path):
    for src in ("fsd50k", "fma", "hifitts2"):
        with pytest.raises(ValueError):
            P.collect_dir(src, str(tmp_path), "noise")


def test_collect_fma(tmp_path):
    p = tmp_path / "tracks.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["", "artist", "artist", "track", "track"])
        w.writerow(["", "id", "name", "license", "title"])
        w.writerow(["track_id", "", "", "", ""])
        w.writerow(["2", "10", "A", "Attribution 4.0 International", "ok"])
        w.writerow(["3", "11", "B", "Attribution-NonCommercial 3.0 International", "nc"])
        w.writerow(["5", "12", "C", "CC0 1.0 Universal", "cc0"])
        w.writerow(["7", "13", "D", "", "unknown"])
    items, stats = P.collect_fma(str(p), str(tmp_path), require_audio=False)
    assert [i.key for i in items] == ["fma_2", "fma_5"]
    assert items[0].path.endswith("000/000002.mp3") and items[1].licence == "CC0-1.0"
    assert stats["drop_licence"] == 2


def _hifitts_rows():
    rows = []
    for spk, bw, multi in [("s1", 20000, False), ("s2", 14000, False), ("s3", 20000, True), ("s4", 18000, None)]:
        for ch in range(4):
            for u in range(3):
                r = {"audio_filepath": f"{spk}/{ch}/{u}.flac", "speaker": spk, "bandwidth": bw, "duration": 300.0,
                     "chapter_id": ch, "snr": 32.0}
                if multi is not None:
                    r["is_multispeaker"] = multi
                rows.append(r)
    rows.append({"audio_filepath": "s5/0/0.flac", "speaker": "s5", "duration": 3.0, "is_multispeaker": False, "snr": 32.0})
    return rows


def test_hifitts2_filters(tmp_path):
    m = tmp_path / "m.jsonl"
    m.write_text("\n".join(json.dumps(r) for r in _hifitts_rows()))
    items, st = P.collect_hifitts2([str(m)], "/audio", min_bw=16_000, max_speaker_hours=0.5)
    assert {i.speaker for i in items} == {"hifitts2:s1"}  # s2 bandwidth, s3 multi, s4 unknown, s5 no bw
    assert st["drop_no_bandwidth"] == 1 and st["drop_multispeaker"] == 12 and st["drop_unknown_multispeaker"] == 12
    # cap: 0.5 h = 1800 s; chapters are 900 s each -> exactly 2 whole chapters kept, all keys unique across chapters
    assert sum(i.duration for i in items) == 1800 and len({i.session for i in items}) == 2
    assert len({i.key for i in items}) == len(items)
    items2, _ = P.collect_hifitts2([str(m)], "/audio", allow_unknown_multispk=True, max_speaker_hours=10)
    assert {i.speaker for i in items2} == {"hifitts2:s1", "hifitts2:s4"}
    assert len({i.key for i in items2}) == len(items2)


def test_hifitts2_string_speaker_counts_and_quality_gates(tmp_path):
    rows = [
        {"audio_filepath": "a/ch1/0.flac", "speaker": "a", "bandwidth": 20000, "duration": 5.0, "num_speakers": "1", "snr": 30.0},
        {"audio_filepath": "a/ch2/0.flac", "speaker": "a", "bandwidth": 20000, "duration": 5.0, "num_speakers": "1", "dnsmos_ovr": 3.8},
        {"audio_filepath": "b/ch1/0.flac", "speaker": "b", "bandwidth": 20000, "duration": 5.0, "num_speakers": "2", "snr": 35.0},
        {"audio_filepath": "c/ch1/0.flac", "speaker": "c", "bandwidth": 20000, "duration": 5.0, "num_speakers": "bad", "snr": 35.0},
        {"audio_filepath": "d/ch1/0.flac", "speaker": "d", "bandwidth": 20000, "duration": 5.0, "num_speakers": "1", "snr": 10.0},
        {"audio_filepath": "e/ch1/0.flac", "speaker": "e", "bandwidth": 20000, "duration": 5.0, "num_speakers": "1", "dnsmos_ovr": 2.4},
        {"audio_filepath": "f/ch1/0.flac", "speaker": "f", "bandwidth": 20000, "duration": 5.0, "num_speakers": "1"},
    ]
    m = tmp_path / "q.jsonl"
    m.write_text("\n".join(json.dumps(r) for r in rows))
    items, st = P.collect_hifitts2([str(m)], "/audio", max_speaker_hours=10)
    assert {i.speaker for i in items} == {"hifitts2:a"}
    assert len(items) == 2 and len({i.key for i in items}) == 2
    assert st["drop_multispeaker"] == 1 and st["drop_unknown_multispeaker"] == 1
    assert st["drop_snr"] == 1 and st["drop_dnsmos"] == 1 and st["drop_no_quality"] == 1

    items_unscored, _ = P.collect_hifitts2([str(m)], "/audio", max_speaker_hours=10, allow_unscored_quality=True)
    assert {i.speaker for i in items_unscored} == {"hifitts2:a", "hifitts2:f"}


def test_split_rejects_duplicate_keys():
    a = P.Item(key="dup", source="vctk", path="/a", kind="speech", speaker="s1", session="s1", licence="CC-BY-4.0")
    b = P.Item(key="dup", source="vctk", path="/b", kind="speech", speaker="s1", session="s1", licence="CC-BY-4.0")
    with pytest.raises(ValueError, match="duplicate item key"):
        P.split_items([a, b], {"train": 0.8, "dev": 0.2})


def test_shard_stream_rir_split_isolation_and_pool_bounds(tmp_path):
    import soundfile as sf
    from oto_train.data.rir_gen import room_split
    from oto_train.data.stream import SR, ShardPool, ShardStream

    # Create a tiny valid speech shard with a 6 s clip.
    split_dir = tmp_path / "shards" / "dev"
    split_dir.mkdir(parents=True)
    wav = tmp_path / "long.wav"
    sf.write(wav, (0.1 * np.random.default_rng(0).standard_normal(SR * 6)).astype(np.float32), SR)
    from dataclasses import asdict

    it = P.Item(key="vctk_long", source="vctk", path=str(wav), kind="speech", speaker="s", session="s", licence="CC-BY-4.0")
    key, flac, meta = P.encode_sample(asdict(it))
    w = P.ShardWriter(str(split_dir / "speech-dev-%06d.tar"), max_count=10)
    w.write(key, {"flac": flac, "json": meta})
    w.close()

    # Pool crops clips above max_len and flushes when max_bytes is reached.
    pool = ShardPool(w.paths, seed=0, buffer=256, min_len=SR, max_len=SR * 2, max_bytes=SR * 4)
    x, _ = next(iter(pool))
    assert len(x) == SR * 2

    # RIR dir containing only train rooms must fail for dev instead of leaking train rooms.
    rir_dir = tmp_path / "rirs"
    rir_dir.mkdir()
    train_ids = [i for i in range(50) if room_split(i) == "train"][:2]
    for rid in train_ids:
        np.savez(rir_dir / f"{rid}.npz", near=np.zeros(1024, np.float32), far=np.zeros((1, 1024), np.float32),
                 meta=json.dumps({"direct_idx": [0, 0]}))
    with pytest.raises(FileNotFoundError, match="split 'dev'"):
        ShardStream(str(tmp_path / "shards"), "dev", seconds=1.0, rir_dir=str(rir_dir))

    # RT60 filtering keeps rooms in (0, max_rt60] (and fixtures omitting rt60) and drops long/NaN/null/<=0 rooms.
    dev_ids = [i for i in range(400) if room_split(i) == "dev"][:7]
    for rid, rt60 in zip(dev_ids[:6], [0.45, 2.8, float("nan"), None, 0.0, -0.2]):
        meta_doc = {"direct_idx": [0, 0]} if rid == dev_ids[5] else {"direct_idx": [0, 0], "rt60": rt60}
        if rid == dev_ids[5]:
            # Fixture without rt60 key (exempt) vs explicit rt60=null / <=0 (rejected)
            pass
        np.savez(rir_dir / f"{rid}.npz", near=np.zeros(1024, np.float32), far=np.zeros((1, 1024), np.float32),
                 meta=json.dumps(meta_doc))
    # Add an explicit negative-rt60 room too.
    np.savez(rir_dir / f"{dev_ids[6]}.npz", near=np.zeros(1024, np.float32), far=np.zeros((1, 1024), np.float32),
             meta=json.dumps({"direct_idx": [0, 0], "rt60": -0.2}))
    ds = ShardStream(str(tmp_path / "shards"), "dev", seconds=1.0, rir_dir=str(rir_dir), max_rt60=1.5)
    assert {int(P.Path(p).stem) for p in ds.rirs.files} == {dev_ids[0], dev_ids[5]}
    ds_all = ShardStream(str(tmp_path / "shards"), "dev", seconds=1.0, rir_dir=str(rir_dir), max_rt60=None)
    assert len(ds_all.rirs.files) == 7
    with pytest.raises(FileNotFoundError, match="0 < rt60 <= 0.1"):
        (rir_dir / f"{dev_ids[5]}.npz").unlink()
        ShardStream(str(tmp_path / "shards"), "dev", seconds=1.0, rir_dir=str(rir_dir), max_rt60=0.1)
    with pytest.raises(ValueError, match="max_rt60"):
        ShardStream(str(tmp_path / "shards"), "dev", seconds=1.0, rir_dir=str(rir_dir), max_rt60=0.0)

    # Malformed RIR archives (missing meta.npy or missing far.npy) fail fast and name the offending file.
    bad_rir = rir_dir / f"{dev_ids[5]}.npz"
    np.savez(bad_rir, near=np.zeros(16, np.float32))
    with pytest.raises(ValueError, match=bad_rir.name):
        ShardStream(str(tmp_path / "shards"), "dev", seconds=1.0, rir_dir=str(rir_dir), max_rt60=1.5)
    np.savez(bad_rir, near=np.zeros(16, np.float32), meta=json.dumps({"direct_idx": [0, 0], "rt60": 0.4}))
    with pytest.raises(ValueError, match=f"{bad_rir.name}.*far\\.npy"):
        ShardStream(str(tmp_path / "shards"), "dev", seconds=1.0, rir_dir=str(rir_dir), max_rt60=1.5)


def test_parse_max_rt60_rejects_non_finite():
    import argparse
    from oto_train.data.rir_gen import parse_max_rt60

    assert parse_max_rt60("1.5") == 1.5
    assert parse_max_rt60("0") is None
    assert parse_max_rt60("-1") is None
    for bad in ("nan", "inf", "-inf", "abc"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_max_rt60(bad)


def test_pack_split_drop_counts_and_repack_cleanup(tmp_path):
    import soundfile as sf

    good = tmp_path / "good.wav"
    short = tmp_path / "short.wav"
    bad = tmp_path / "nan.wav"
    sf.write(good, (0.1 * np.random.default_rng(0).standard_normal(48_000)).astype(np.float32), 48_000)
    sf.write(short, np.zeros(1000, np.float32), 48_000)
    sf.write(bad, np.full(48_000, np.nan, np.float32), 48_000, subtype="FLOAT")

    items = [
        P.Item(key="k_good", source="vctk", path=str(good), kind="speech", speaker="s1", session="s1", licence="CC-BY-4.0"),
        P.Item(key="k_short", source="vctk", path=str(short), kind="speech", speaker="s2", session="s2", licence="CC-BY-4.0"),
        P.Item(key="k_nan", source="vctk", path=str(bad), kind="speech", speaker="s3", session="s3", licence="CC-BY-4.0"),
        P.Item(key="k_miss", source="vctk", path=str(tmp_path / "missing.wav"), kind="speech", speaker="s4", session="s4", licence="CC-BY-4.0"),
    ]
    # Pre-seed a stale higher-index shard: a failed repack must keep it, a successful repack must remove it.
    stale_dir = tmp_path / "out" / "train"
    stale_dir.mkdir(parents=True)
    stale = stale_dir / "speech-train-000005.tar"
    stale.write_bytes(b"stale")
    with pytest.raises(ValueError, match="workers"):
        P.pack_split(items, tmp_path / "out", "train", "speech", workers=0, max_count=10)
    assert stale.exists()

    drops: dict[str, int] = {}
    n = P.pack_split(items, tmp_path / "out", "train", "speech", workers=1, max_count=10, drop_counts=drops)
    assert n == 1
    assert drops == {"drop_short": 1, "drop_nonfinite": 1, "drop_error": 1}
    assert not stale.exists()

    # CLI repack into the same dir with omitted splits cleans stale dev/test shards, preserves hyphenated kinds,
    # and rolls back cleanly if a second group fails mid-pack.
    src_dir = tmp_path / "src_ace"
    src_dir.mkdir()
    for i in range(6):
        sf.write(src_dir / f"c{i}.wav", (0.1 * np.random.default_rng(i).standard_normal(48_000)).astype(np.float32), 48_000)
    cli_out = tmp_path / "cli_out"
    P.main(["pack", "--dir", f"ace:noise-extra:{src_dir}", "--out", str(cli_out), "--split", "train=0.5,dev=0.5", "--workers", "1"])
    assert list((cli_out / "train").glob("noise-extra-train-*.tar"))
    assert list((cli_out / "dev").glob("noise-extra-dev-*.tar"))
    P.main(["pack", "--dir", f"ace:noise-extra:{src_dir}", "--out", str(cli_out), "--split", "train=1", "--workers", "1"])
    assert list((cli_out / "train").glob("noise-extra-train-*.tar"))
    assert not list((cli_out / "dev").glob("*.tar"))
    assert not (cli_out / "splits" / "dev.json").exists()
    st = json.loads((cli_out / "stats.json").read_text())
    assert list(st["pack"].keys()) == ["train/noise-extra"] and st["pack"]["train/noise-extra"]["packed"] == 6

    # Mid-pack failure leaves the previous published dataset intact and removes staging dirs.
    orig_pack = P.pack_split
    calls = {"n": 0}

    def boom(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("injected pack failure")
        return orig_pack(*a, **kw)

    orig_stats = (cli_out / "stats.json").read_text()
    try:
        P.pack_split = boom
        with pytest.raises(RuntimeError, match="injected pack failure"):
            P.main(["pack", "--dir", f"ace:noise:{src_dir}", "--dir", f"librispeech:speech:{src_dir}",
                    "--out", str(cli_out), "--split", "train=1", "--workers", "1"])
    finally:
        P.pack_split = orig_pack
    assert (cli_out / "stats.json").read_text() == orig_stats
    assert not list((cli_out / "train").glob("noise-train-*.tar"))
    assert not list(cli_out.glob(".tmp-*"))

    # Failure while removing a stale file rolls back every published replacement.
    pub_dir = tmp_path / "pub_rb"
    pub_dir.mkdir()
    src_f = pub_dir / "new.tar"
    dst_f = pub_dir / "old.tar"
    stale_ok = pub_dir / "stale_ok.tar"
    stale_fail = pub_dir / "stale_fail.tar"
    bak_dir = pub_dir / "bak"
    bak_dir.mkdir()
    src_f.write_bytes(b"new")
    dst_f.write_bytes(b"old")
    stale_ok.write_bytes(b"s1")
    stale_fail.write_bytes(b"s2")
    real_replace = P.os.replace

    def fail_stale(src, dst):
        if P.Path(src) == stale_fail:
            raise PermissionError("read-only dir")
        return real_replace(src, dst)

    try:
        P.os.replace = fail_stale
        with pytest.raises(PermissionError, match="read-only"):
            P._publish_files([(src_f, dst_f)], [stale_ok, stale_fail], bak_dir)
    finally:
        P.os.replace = real_replace
    assert dst_f.read_bytes() == b"old"
    assert stale_ok.read_bytes() == b"s1"
    assert stale_fail.read_bytes() == b"s2"


def test_split_speaker_and_session_disjoint():
    items = [P.Item(key=f"k{s}_{u}", source="vctk", path=f"/{s}/{u}", kind="speech", speaker=f"spk{s}",
                    session=f"spk{s}:c{u % 3}", licence="CC-BY-4.0") for s in range(300) for u in range(5)]
    splits = P.split_items(items, {"train": 0.8, "dev": 0.1, "test": 0.1})
    assert all(splits[k] for k in splits)
    P.check_disjoint(splits)
    again = P.split_items(items, {"train": 0.8, "dev": 0.1, "test": 0.1})
    assert [i.key for i in again["dev"]] == [i.key for i in splits["dev"]]
    bad = {"train": [items[0]], "dev": [items[1]]}
    with pytest.raises(AssertionError):
        P.check_disjoint(bad)


def test_attribution(tmp_path):
    items = [P.Item(key="fsd50k_2", source="fsd50k", path="", kind="noise", speaker="u", session="u", licence="CC-BY",
                    licence_url="http://creativecommons.org/licenses/by/3.0/", title="t|x", author="bob",
                    url="https://freesound.org/s/2/")]
    P.write_attribution(items, tmp_path / "ATTRIBUTION.md")
    txt = (tmp_path / "ATTRIBUTION.md").read_text()
    assert "## fsd50k" in txt and "bob" in txt and "t/x" in txt


def test_shard_roundtrip(tmp_path):
    import soundfile as sf

    wav = tmp_path / "a.wav"
    sf.write(wav, (0.1 * np.random.default_rng(0).standard_normal(44_100)).astype(np.float32), 44_100)
    item = P.Item(key="vctk_a", source="vctk", path=str(wav), kind="speech", speaker="s", session="s", licence="CC-BY-4.0")
    from dataclasses import asdict

    key, flac, meta = P.encode_sample(asdict(item))
    w = P.ShardWriter(str(tmp_path / "speech-train-%06d.tar"), max_count=1)
    w.write(key, {"flac": flac, "json": meta})
    w.write(key + "b", {"flac": flac, "json": meta})
    w.close()
    assert len(w.paths) == 2
    samples = list(iter_tar(w.paths[0]))
    assert samples[0][0] == "vctk_a"
    m = json.loads(samples[0][1]["json"])
    assert m["sr"] == 48_000 and m["orig_sr"] == 44_100 and 20_000 < m["f_max"] <= 22_050


# -- bandwidth -------------------------------------------------------------------


@pytest.mark.parametrize("cutoff", [4_000, 8_000, 16_000])
def test_bandwidth_estimator(cutoff):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(48_000 * 2)
    X = np.fft.rfft(x)
    X[np.fft.rfftfreq(len(x), 1 / 48_000) > cutoff] = 0  # brick-wall band limit
    y = np.fft.irfft(X, len(x))
    f = estimate_bandwidth(y, 48_000)
    assert cutoff * 0.97 < f < cutoff * 1.05, f
    assert estimate_bandwidth(x, 48_000) > 23_000
    assert estimate_bandwidth(np.zeros(48_000), 48_000) == 0.0
    assert urgent_target_rate(7_900) == 16_000 and urgent_target_rate(23_000) == 48_000


# -- mixer ---------------------------------------------------------------------------


@pytest.mark.parametrize("isolate", [False, True])
def test_mixer_finite(isolate):
    from torch.utils.data import DataLoader

    batch = next(iter(DataLoader(SyntheticStream(1.0), batch_size=4)))
    cfg = MixConfig(p_interferer=1, p_user_far=0.5, p_background_first=0.5, p_background_only=0.5) if isolate else MixConfig()
    out = Mixer(cfg, seed=1)(**batch)
    assert out["mix"].shape == batch["speech"].shape
    assert all(torch.isfinite(v).all() for v in out.values())
    assert out["mix"].abs().max() <= 1.0
    assert (out["f_max"] <= batch["speech_fmax"]).all()


def test_mixer_dereverb_target_is_early_only():
    from oto_train.data.mixer import early_rir, shift_left

    h = torch.zeros(1, 48_000)
    h[0, 100] = 1.0
    h[0, 100 + 4_800] = 0.5  # 100 ms reflection -> removed from the target
    a = shift_left(h, torch.tensor([100]))
    assert a[0, 0] == 1.0
    e = early_rir(a, 50.0)
    assert e[0, 4_800] == 0 and e[0, 0] == 1.0


def test_rir_gen_ism():
    pytest.importorskip("pyroomacoustics")
    from oto_train.data.rir_gen import simulate_room

    res = simulate_room(7, ray_tracing=False)
    assert res is not None
    near, far, meta = res
    assert np.isfinite(near).all() and far.shape[0] == 2
    assert 0.15 <= meta["near_dist"] <= 0.6 and all(1.2 <= d <= 6.0 for d in meta["far_dist"])
    assert meta["direct_idx"][0] < min(meta["direct_idx"][1:])
