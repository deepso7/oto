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
                     "chapter_id": ch}
                if multi is not None:
                    r["is_multispeaker"] = multi
                rows.append(r)
    rows.append({"audio_filepath": "s5/0/0.flac", "speaker": "s5", "duration": 3.0, "is_multispeaker": False})
    return rows


def test_hifitts2_filters(tmp_path):
    m = tmp_path / "m.jsonl"
    m.write_text("\n".join(json.dumps(r) for r in _hifitts_rows()))
    items, st = P.collect_hifitts2([str(m)], "/audio", min_bw=16_000, max_speaker_hours=0.5)
    assert {i.speaker for i in items} == {"hifitts2:s1"}  # s2 bandwidth, s3 multi, s4 unknown, s5 no bw
    assert st["drop_no_bandwidth"] == 1 and st["drop_multispeaker"] == 12 and st["drop_unknown_multispeaker"] == 12
    # cap: 0.5 h = 1800 s; chapters are 900 s each -> exactly 2 whole chapters kept
    assert sum(i.duration for i in items) == 1800 and len({i.session for i in items}) == 2
    items2, _ = P.collect_hifitts2([str(m)], "/audio", allow_unknown_multispk=True, max_speaker_hours=10)
    assert {i.speaker for i in items2} == {"hifitts2:s1", "hifitts2:s4"}


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
