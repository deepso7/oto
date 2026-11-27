import pytest

from oto_train.export_otow import read_otow
from oto_train.train import cli, main, stage


@pytest.mark.parametrize("arch", ["a", "b"])
def test_smoke_train_and_resume(arch, tmp_path):
    res = main(["--smoke", "--arch", arch, "--ckpt-dir", str(tmp_path)])
    assert res["step"] == 4 and res["loss"] == res["loss"]  # not NaN
    o = read_otow(tmp_path / "best.otow")
    assert o.arch_id == {"a": 1, "b": 2}[arch] and o.config["meta_step"] == 4 and o.config["arch"] == f"oto1-{arch}"
    res2 = main(["--smoke", "--arch", arch, "--ckpt-dir", str(tmp_path), "--steps", "6"])
    assert res2["step"] == 6


def test_cli_returns_none(tmp_path):
    assert cli(["--smoke", "--arch", "a", "--steps", "1", "--ckpt-dir", str(tmp_path)]) is None


def test_stage_hf_nested_path_cached_matches_first_call(tmp_path, monkeypatch):
    from oto_train.data import pack_shards as P

    calls = []

    def fake_dl(repo, dst, pattern=None):
        calls.append((repo, dst, pattern))
        (tmp_path / "stage" / "shards" / "v1").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(P, "download_hf", fake_dl)
    url = "hf://user/repo/shards/v1"
    dst = str(tmp_path / "stage")
    first = stage(url, dst)
    second = stage(url, dst)
    assert first == second == str(tmp_path / "stage" / "shards" / "v1")
    assert len(calls) == 1
