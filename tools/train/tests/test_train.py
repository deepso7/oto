import pytest
import torch

from oto_train.data.mixer import MixConfig, Mixer
from oto_train.export_otow import read_otow
from oto_train.losses import LossWeights, OtoLoss
from oto_train.models import stft as S
from oto_train.train import cli, main, run_batch, stage


@pytest.mark.parametrize("arch", ["a", "b"])
def test_smoke_train_and_resume(arch, tmp_path):
    res = main(["--smoke", "--arch", arch, "--ckpt-dir", str(tmp_path)])
    assert res["step"] == 4 and res["loss"] == res["loss"]  # not NaN
    o = read_otow(tmp_path / "best.otow")
    assert o.arch_id == {"a": 1, "b": 2}[arch] and o.config["meta_step"] == 4 and o.config["arch"] == f"oto1-{arch}"
    res2 = main(["--smoke", "--arch", arch, "--ckpt-dir", str(tmp_path), "--steps", "6"])
    assert res2["step"] == 6


def test_init_from_conflicts_with_existing_latest_and_resumes_own_finetune(tmp_path):
    base = tmp_path / "base"
    main(["--smoke", "--arch", "a", "--steps", "2", "--ckpt-dir", str(base)])
    # Reusing the same ckpt_dir with --init-from must fail instead of silently ignoring --init-from.
    with pytest.raises(ValueError, match="conflicts with existing"):
        main(["--smoke", "--arch", "a", "--steps", "3", "--ckpt-dir", str(base), "--init-from", str(base / "best.pt")])
    # In a fresh ckpt_dir, --init-from initializes cleanly and a subsequent restart resumes that fine-tune.
    ft = tmp_path / "ft"
    r1 = main(["--smoke", "--arch", "a", "--steps", "2", "--ckpt-dir", str(ft), "--init-from", str(base / "best.pt")])
    assert r1["step"] == 2
    r2 = main(["--smoke", "--arch", "a", "--steps", "3", "--ckpt-dir", str(ft), "--init-from", str(base / "best.pt")])
    assert r2["step"] == 3


def test_cli_returns_none(tmp_path):
    assert cli(["--smoke", "--arch", "a", "--steps", "1", "--ckpt-dir", str(tmp_path)]) is None


def test_stage_hf_nested_path_cached_matches_first_call(tmp_path, monkeypatch):
    from oto_train.data import pack_shards as P

    calls = []

    def fake_dl(repo, dst, pattern=None):
        calls.append((repo, dst, pattern))
        (tmp_path / "stage" / "shards" / "v1").mkdir(parents=True, exist_ok=True)
        (tmp_path / "stage" / "shards" / "v1" / "old.tar").write_bytes(b"v1")

    monkeypatch.setattr(P, "download_hf", fake_dl)
    url = "hf://user/repo/shards/v1"
    dst = str(tmp_path / "stage")
    first = stage(url, dst)
    second = stage(url, dst)
    assert first == second == str(tmp_path / "stage" / "shards" / "v1")
    assert len(calls) == 1

    # Switching source into the same stage dir must clear stale shards first.
    def fake_dl_v2(repo, dst, pattern=None):
        (tmp_path / "stage" / "shards" / "v2").mkdir(parents=True, exist_ok=True)
        (tmp_path / "stage" / "shards" / "v2" / "new.tar").write_bytes(b"v2")

    monkeypatch.setattr(P, "download_hf", fake_dl_v2)
    third = stage("hf://user/repo/shards/v2", dst)
    assert third == str(tmp_path / "stage" / "shards" / "v2")
    assert not (tmp_path / "stage" / "shards" / "v1" / "old.tar").exists()
    assert (tmp_path / "stage" / "shards" / "v2" / "new.tar").exists()


def test_stage_hf_tar_extracts_and_caches(tmp_path, monkeypatch):
    import tarfile
    from oto_train.data import pack_shards as P

    calls = []

    def fake_dl(repo, dst, pattern=None):
        calls.append((repo, dst, pattern))
        tar_p = tmp_path / "stage_rirs" / "rirs.tar"
        tar_p.parent.mkdir(parents=True, exist_ok=True)
        npz = tmp_path / "000000.npz"
        npz.write_bytes(b"npz")
        with tarfile.open(tar_p, "w") as tf:
            tf.add(npz, arcname="rirs/000000.npz")

    monkeypatch.setattr(P, "download_hf", fake_dl)
    url = "hf://user/repo/rirs.tar"
    dst = str(tmp_path / "stage_rirs")
    first = stage(url, dst)
    second = stage(url, dst)
    assert first == second == str(tmp_path / "stage_rirs" / "rirs")
    assert (tmp_path / "stage_rirs" / "rirs" / "000000.npz").exists()
    assert not (tmp_path / "stage_rirs" / "rirs.tar").exists()
    assert len(calls) == 1 and calls[0][2] == "rirs.tar"


def test_main_stages_local_rir_tar(tmp_path):
    import json
    import tarfile
    from dataclasses import asdict
    import numpy as np
    import soundfile as sf
    from oto_train.data import pack_shards as P
    from oto_train.data.rir_gen import room_split
    from oto_train.data.stream import SR

    # Minimal valid train + dev speech shards and a local rirs.tar containing train + dev rooms.
    wav = tmp_path / "s.wav"
    sf.write(wav, (0.1 * np.random.default_rng(0).standard_normal(SR * 2)).astype(np.float32), SR)
    it = P.Item(key="vctk_s", source="vctk", path=str(wav), kind="speech", speaker="s", session="s", licence="CC-BY-4.0")
    key, flac, meta = P.encode_sample(asdict(it))
    shards_root = tmp_path / "shards"
    for split in ("train", "dev"):
        w = P.ShardWriter(str(shards_root / split / f"speech-{split}-%06d.tar"), max_count=10)
        w.write(key, {"flac": flac, "json": meta})
        w.close()

    rir_tar = tmp_path / "rirs.tar"
    train_id = next(i for i in range(100) if room_split(i) == "train")
    dev_id = next(i for i in range(100) if room_split(i) == "dev")
    with tarfile.open(rir_tar, "w") as tf:
        for rid in (train_id, dev_id):
            p = tmp_path / f"{rid:06d}.npz"
            near = np.zeros(1024, np.float32)
            near[16] = 1.0
            far = np.zeros((1, 1024), np.float32)
            far[0, 32] = 1.0
            np.savez(p, near=near, far=far, meta=json.dumps({"direct_idx": [16, 32], "rt60": 0.4}))
            tf.add(p, arcname=f"rirs/{rid:06d}.npz")

    res = main([
        "--arch", "b", "--data", str(shards_root), "--rir-data", str(rir_tar),
        "--stage-dir", str(tmp_path / "stage"), "--ckpt-dir", str(tmp_path / "ckpt"),
        "--steps", "1", "--batch-size", "1", "--seconds", "1", "--val-every", "1",
        "--val-batches", "1", "--workers", "0", "--warmup", "1", "--log-every", "1",
    ])
    assert res["step"] == 1
    assert (tmp_path / "stage" / "rirs" / "rirs" / f"{train_id:06d}.npz").exists()


def test_run_batch_drops_incomplete_trailing_istft_hop():
    class IdentityStftModel(torch.nn.Module):
        def forward(self, x):
            spec = S.stft(x)
            return {"wav": S.istft(spec), "p_logit": torch.full((x.shape[0], spec.shape[1]), 10.0)}

    t = S.HOP * 10
    tone = 0.2 * torch.sin(2 * torch.pi * 440 * torch.arange(t) / S.SR)[None]
    batch = {
        "speech": tone,
        "speech_fmax": torch.tensor([24_000.0]),
        "noise": torch.zeros_like(tone),
        "interferer": torch.zeros_like(tone),
    }
    mixer = Mixer(MixConfig(p_reverb=0.0, p_noise=0.0, p_interferer=0.0, p_background_only=0.0, p_background_first=0.0,
                            p_bandlimit=0.0, p_eq=0.0, p_self_noise=0.0, p_clip=0.0), seed=0)
    loss, parts = run_batch(IdentityStftModel(), mixer, OtoLoss(LossWeights()), batch, torch.device("cpu"), False)
    assert float(parts["l1"]) < 1e-5 and float(parts["spec"]) < 1e-5 and float(loss) < 1e-3


def test_resume_and_init_from_use_checkpoint_model_cfg(tmp_path):
    run32 = tmp_path / "b32"
    main(["--smoke", "--arch", "b", "--set", "channels=32", "--steps", "2", "--ckpt-dir", str(run32)])
    # Auto-resume without --set must inherit channels=32 from latest.pt instead of default 24.
    r2 = main(["--smoke", "--arch", "b", "--steps", "3", "--ckpt-dir", str(run32)])
    assert r2["step"] == 3
    assert read_otow(run32 / "best.otow").config["channels"] == 32
    # Conflicting explicit override must fail clearly.
    with pytest.raises(ValueError, match="conflicts with checkpoint"):
        main(["--smoke", "--arch", "b", "--set", "channels=24", "--steps", "4", "--ckpt-dir", str(run32)])
    # Fine-tuning via --init-from without --set also inherits channels=32.
    ft = tmp_path / "ft32"
    r3 = main(["--smoke", "--arch", "b", "--steps", "2", "--ckpt-dir", str(ft), "--init-from", str(run32 / "best.pt")])
    assert r3["step"] == 2
    assert read_otow(ft / "best.otow").config["channels"] == 32
    # Weights-only legacy Candidate B checkpoint (no model_cfg) also infers channels=32.
    raw_ck = torch.load(run32 / "best.pt", map_location="cpu", weights_only=True)
    legacy = tmp_path / "legacy_b32.pt"
    torch.save({"model": raw_ck["model"]}, legacy)
    ft_leg = tmp_path / "ft_leg"
    r4 = main(["--smoke", "--arch", "b", "--steps", "2", "--ckpt-dir", str(ft_leg), "--init-from", str(legacy)])
    assert r4["step"] == 2
    assert read_otow(ft_leg / "best.otow").config["channels"] == 32
