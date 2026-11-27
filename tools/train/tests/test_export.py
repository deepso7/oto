import hashlib
import struct

import numpy as np
import pytest
import torch

from conftest import rel_db
from oto_train.export_otow import (
    ALIGN, DTYPE_F16, DTYPE_F32, FLAG_ENROLLMENT, FLAG_ISOLATE, OtowError, _randomise_bn, export, model_from_otow,
    read_otow, write_otow,
)
from oto_train.models import build_model


def _exported(arch, dtype, tmp_path):
    m = build_model(arch)
    _randomise_bn(m, 1)
    m.eval()
    blob = export(m, tmp_path / f"{arch}.otow", dtype, FLAG_ISOLATE, tmp_path / f"{arch}.safetensors")
    return m, blob


@pytest.mark.parametrize("arch", ["a", "b"])
def test_roundtrip_f32_exact(arch, tmp_path):
    m, blob = _exported(arch, DTYPE_F32, tmp_path)
    o = read_otow(blob)
    assert o.arch_id == m.arch_id and o.dtype == DTYPE_F32 and o.flags == FLAG_ISOLATE
    ref = m.export_tensors()
    assert list(o.tensors) == list(ref)
    for k, v in ref.items():
        assert np.array_equal(o.tensors[k], v.numpy()), k
    from safetensors.torch import load_file

    st = load_file(tmp_path / f"{arch}.safetensors")
    assert all(torch.equal(st[k], v) for k, v in ref.items())
    # model rebuilt from the file is bit-identical in behaviour
    x = torch.randn(1, 512 * 30)
    with torch.no_grad():
        assert rel_db(model_from_otow(blob)(x)["wav"], m(x)["wav"]) <= -90


@pytest.mark.parametrize("arch", ["a", "b"])
def test_roundtrip_f16(arch, tmp_path):
    m, blob = _exported(arch, DTYPE_F16, tmp_path)
    o = read_otow(blob)
    for k, v in m.export_tensors().items():
        assert np.array_equal(o.tensors[k], v.numpy().astype(np.float16).astype(np.float32)), k
    x = 0.1 * torch.randn(1, 512 * 30)
    with torch.no_grad():
        db = rel_db(model_from_otow(blob)(x)["wav"], m(x)["wav"])
    print(f"{arch}: f16 weights vs f32 {db:.1f} dB")
    assert db < -30


def test_layout_alignment_and_header(tmp_path):
    t = {"w": torch.arange(5, dtype=torch.float32), "m": torch.ones(2, 3, 4, 5)}
    cfg = {"b": 1, "a": [1.5, "x"]}
    blob = write_otow(tmp_path / "t.otow", t, cfg, arch_id=2, dtype=DTYPE_F32, flags=FLAG_ENROLLMENT)
    assert blob[:4] == b"OTOW" and len(blob) % ALIGN == 0
    version, arch, dtype, flags, clen, n, table = struct.unpack_from("<7I", blob, 4)
    assert (version, arch, dtype, flags, n) == (1, 2, DTYPE_F32, FLAG_ENROLLMENT, 2)
    assert blob[64 : 64 + clen] == b'{"a":[1.5,"x"],"b":1}'
    assert table % 64 == 0 and table >= 64 + clen
    assert blob[32:64] == hashlib.sha256(blob[64:]).digest()
    for i in range(n):
        e = table + 96 * i
        ndim, *shape, dt, off = struct.unpack_from("<I4II Q", blob, e + 64)
        assert off % 64 == 0 and dt == DTYPE_F32
    o = read_otow(blob)
    assert o.tensors["m"].shape == (2, 3, 4, 5) and o.config == cfg


def test_tamper_detected(tmp_path):
    _, blob = _exported("b", DTYPE_F16, tmp_path)
    bad = bytearray(blob)
    bad[-100] ^= 0x01
    with pytest.raises(OtowError, match="SHA-256"):
        read_otow(bytes(bad))
    bad = bytearray(blob)
    bad[0:4] = b"OTOX"
    with pytest.raises(OtowError, match="magic"):
        read_otow(bytes(bad))


def test_rejects_bad_tensors(tmp_path):
    with pytest.raises(ValueError):
        write_otow(tmp_path / "x", {"x" * 64: torch.ones(1)}, {}, 1)
    with pytest.raises(ValueError):
        write_otow(tmp_path / "x", {"x": torch.ones(1, 1, 1, 1, 1)}, {}, 1)
    with pytest.raises(ValueError):
        write_otow(tmp_path / "x", {"x": torch.full((2,), 1e6)}, {}, 1, DTYPE_F16)
