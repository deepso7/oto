import copy

import pytest
import torch

from conftest import rel_db
from oto_train.export_otow import _randomise_bn
from oto_train.models import build_model
from oto_train.models import stft as S

ARCHS = ["a", "b"]


def _model(arch):
    m = build_model(arch)
    _randomise_bn(m, 0)
    return m.eval()


def _signal(seconds=2.0, batch=2):
    n = int(seconds * S.SR) // 512 * 512
    t = torch.arange(n) / S.SR
    x = 0.2 * torch.sin(2 * torch.pi * 180 * t) * (1 + torch.sin(2 * torch.pi * 3 * t))
    return (x + 0.05 * torch.randn(batch, n)).float()


@pytest.mark.parametrize("arch", ARCHS)
def test_batch_vs_streaming(arch):
    m = _model(arch)
    x = _signal()
    with torch.no_grad():
        out = m(x)
    ys, ps = m.stream(x.shape[0]).run(x)
    db = rel_db(ys[:, 512:], out["wav"][:, :-512])
    print(f"{arch}: stream vs batch waveform {db:.1f} dB")
    assert db <= -90
    assert rel_db(ps, out["p_logit"]) <= -90


@pytest.mark.parametrize("arch", ARCHS)
def test_step_spec_matches_forward_spec(arch):
    m = _model(arch)
    X = S.stft(_signal(1.0))
    with torch.no_grad():
        Y, P = m.forward_spec(X)
        st = m.init_state(X.shape[0])
        ys = []
        for f in range(X.shape[1]):
            y, _, st = m.step_spec(X[:, f], st)
            ys.append(y)
    assert rel_db(torch.stack(ys, 1), Y) <= -90


@pytest.mark.parametrize("arch", ARCHS)
def test_fold_batchnorm(arch):
    m = _model(arch)
    folded = copy.deepcopy(m).fold_batchnorm()
    assert not any(isinstance(x, torch.nn.BatchNorm1d) for x in folded.modules())
    x = _signal(1.0)
    with torch.no_grad():
        a, b = m(x), folded(x)
    db = rel_db(b["wav"], a["wav"])
    print(f"{arch}: fold BN {db:.1f} dB")
    assert db <= -90
    assert rel_db(b["p_logit"], a["p_logit"]) <= -90


def test_param_and_mac_counts():
    """SPEC.md sections 4 and 5: exact exported tensors, parameter and MAC counts."""
    want = {"a": (574_328, 1_622_208), "b": (460_233, 2_392_192)}
    for arch in ARCHS:
        m = build_model(arch)
        n_exp = m.exported_param_count()
        assert (n_exp, m.macs_per_frame()) == want[arch]
        t = m.fold_batchnorm().export_tensors()
        assert sum(v.numel() for v in t.values()) == n_exp


def test_tensor_names_and_shapes():
    a = {k: tuple(v.shape) for k, v in build_model("a").fold_batchnorm().export_tensors().items()}
    assert a["enc.erb0.weight"] == (48, 1, 3) and a["enc.df3.weight"] == (48, 48, 3)
    assert a["gru0.weight_ih"] == (4, 144, 96) and a["gru1.bias_hh"] == (4, 144)
    assert a["erb_dec.up2.weight"] == (96, 48, 3) and a["erb_dec.out.bias"] == (1,)
    assert a["df_dec.fc1.weight"] == (576, 384) and a["df_dec.convp.weight"] == (6, 48, 1)
    assert a["p_head.weight"] == (1, 192) and len(a) == 38
    b = {k: tuple(v.shape) for k, v in build_model("b").fold_batchnorm().export_tensors().items()}
    assert b["enc0.weight"] == (32, 2, 8) and b["gru0.weight_ih"] == (384, 512) and b["gru1.bias_ih"] == (384,)
    assert b["bott.fc.weight"] == (512, 128) and b["dec3.weight"] == (64, 64, 3) and b["dec0.weight"] == (8, 64, 3)
    assert b["p_head.weight"] == (1, 128) and len(b) == 28


def test_config_json():
    assert build_model("a").config_dict() == {"arch": "oto1-a", "channels": 48, "df_order": 3, "gru_groups": 4,
                                              "gru_hidden": 192, "n_df_bins": 96, "n_erb": 32}  # fmt: skip
    assert build_model("b").config_dict() == {"arch": "oto1-b", "channels": 32, "gru_hidden": 128}
    with pytest.raises(ValueError):
        build_model("a", n_df_bins=64)
    with pytest.raises(ValueError):
        build_model("a", gru_hidden=190)


def test_primitives():
    from oto_train.models.layers import comp, group_shuffle, subpixel

    h = torch.arange(12.0)
    out = group_shuffle(h, 4)
    for a in range(3):
        for b in range(4):
            assert out[b * 3 + a] == h[a * 4 + b]
    y = torch.randn(1, 2 * 5, 7)  # r = 2, C' = 5
    z = subpixel(y, 2)
    assert z.shape == (1, 5, 14)
    for j in range(2):
        for c in range(5):
            assert torch.equal(z[0, c, j::2], y[0, j * 5 + c])
    x = torch.tensor([[3.0, 4.0]])
    assert torch.allclose(comp(x).norm(dim=-1), torch.tensor([5.0**0.3]), rtol=1e-6)


@pytest.mark.parametrize("arch", ARCHS)
def test_training_mode_backward(arch):
    m = build_model(arch).train()
    x = _signal(0.5)
    out = m(x)
    (out["wav"].square().mean() + out["p_logit"].mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
