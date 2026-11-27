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
