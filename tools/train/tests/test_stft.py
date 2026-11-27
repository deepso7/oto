import math

import torch

from conftest import rel_db
from oto_train.models import stft as S


def test_window_cola_exact():
    w = S.sqrt_hann(dtype=torch.float64)
    assert w.shape == (1024,)
    assert torch.allclose(w[:512] ** 2 + w[512:] ** 2, torch.ones(512, dtype=torch.float64), atol=1e-15)
    n = 37
    assert abs(float(w[n]) - math.sqrt(0.5 * (1 - math.cos(2 * math.pi * n / 1024)))) < 1e-15
    assert float(w[0]) == 0.0


def test_batch_perfect_reconstruction():
    x = torch.randn(3, 512 * 50)
    y = S.istft(S.stft(x))
    assert y.shape == x.shape
    assert rel_db(y[:, :-512], x[:, :-512]) < -120


def test_frame0_sees_left_padding():
    x = torch.randn(1, 2048)
    X = S.stft(x)
    frame0 = torch.cat([torch.zeros(512), x[0, :512]]) * S.sqrt_hann()
    ref = torch.view_as_real(torch.fft.rfft(frame0))
    assert torch.allclose(X[0, 0], ref, atol=1e-5)


def test_streaming_matches_batch():
    x = torch.randn(2, 512 * 40)
    st = S.StreamingSTFT((2,))
    specs, outs = [], []
    for blk in x.split(512, -1):
        s = st.analyze(blk)
        specs.append(s)
        outs.append(st.synthesize(s))
    spec = torch.stack(specs, 1)
    assert rel_db(spec, S.stft(x)) < -130
    out = torch.cat(outs, -1)
    # identity model: pre-roll block is exactly zero, then a 512-sample delayed copy
    assert out[:, :512].abs().max() < 1e-6
    assert rel_db(out[:, 512:], x[:, :-512]) < -120


def test_erb_bands():
    e = S.erb_band_edges(32)
    assert e == S.ERB_EDGES_32 == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 33, 39, 45, 53, 62, 73, 85, 99,
                                   116, 134, 156, 182, 211, 245, 284, 329, 381, 442, 513]  # fmt: skip
    sel, width, inv = S.erb_matrices(e)
    assert torch.equal(sel.sum(0), width) and int(width.sum()) == 513
    assert torch.equal(inv.sum(0), torch.ones(513))


def test_streaming_flushes_tiny_input():
    st = S.StreamingSTFT((1,))
    X = st.analyze(torch.full((1, 512), 1e-16))
    assert torch.count_nonzero(X) == 0
