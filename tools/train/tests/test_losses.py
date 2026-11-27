import pytest
import torch

from oto_train.losses import OtoLoss, mr_compressed_stft_loss, presence_labels, silent_target_loss
from oto_train.data.stream import synth_speech

import numpy as np


def _speech(b=2, n=48_000):
    rng = np.random.default_rng(0)
    return torch.stack([torch.from_numpy(synth_speech(n, rng)) for _ in range(b)])


CASES = {
    "speech": lambda s: (s + 0.05 * torch.randn_like(s), s, s + 0.1 * torch.randn_like(s)),
    "silent_target": lambda s: (0.01 * torch.randn_like(s), torch.zeros_like(s), 0.1 * torch.randn_like(s)),
    "all_zero": lambda s: (torch.zeros_like(s), torch.zeros_like(s), torch.zeros_like(s)),
    "zero_estimate": lambda s: (torch.zeros_like(s), s, s),
}


@pytest.mark.parametrize("case", list(CASES))
def test_loss_finite_with_finite_grads(case):
    s = _speech()
    est, tgt, mix = CASES[case](s)
    est = est.clone().requires_grad_(True)
    p = torch.zeros(2, s.shape[1] // 512, requires_grad=True)
    loss, parts = OtoLoss()(est, tgt, mix, p, torch.tensor([24_000.0, 8_000.0]))
    assert torch.isfinite(loss), parts
    assert all(torch.isfinite(v) for v in parts.values())
    loss.backward()
    assert torch.isfinite(est.grad).all() and torch.isfinite(p.grad).all()


def test_bandwidth_mask_ignores_high_band():
    s = _speech(1)
    s = s + 1e-3 * torch.randn_like(s)  # realistic noise floor (compression magnifies FFT leakage into exact zeros)
    hf = torch.sin(2 * torch.pi * 18_000 * torch.arange(s.shape[1]) / 48_000)[None] * 0.1
    l_full, _ = mr_compressed_stft_loss(s + hf, s, torch.tensor([24_000.0]))
    l_masked, _ = mr_compressed_stft_loss(s + hf, s, torch.tensor([8_000.0]))
    assert l_masked < 0.1 * l_full


def test_over_attenuation_is_asymmetric():
    s = _speech(1)
    _, oa_quiet = mr_compressed_stft_loss(0.5 * s, s, oa_weight=1.0)
    _, oa_loud = mr_compressed_stft_loss(1.5 * s, s, oa_weight=1.0)
    assert oa_quiet > 0 and oa_loud < 1e-3 * oa_quiet


def test_silent_target_loss_scale_aware():
    mix = 0.1 * torch.randn(1, 48_000)
    tgt = torch.zeros_like(mix)
    l_pass = silent_target_loss(mix, mix, tgt)
    l_scaled = silent_target_loss(100 * mix, 100 * mix, tgt)
    l_supp = silent_target_loss(mix * 1e-3, mix, tgt)
    assert abs(float(l_pass) - 1.0) < 1e-3 and abs(float(l_scaled) - 1.0) < 1e-3
    assert float(l_supp) < 0.5
    assert float(silent_target_loss(mix * 1e-4, mix, tgt)) == 0.0


def test_presence_labels():
    s = _speech(1)
    s[:, :24_064] = 0
    lab = presence_labels(s)
    assert lab[0, :40].sum() == 0 and lab[0, 50:].mean() > 0.3
