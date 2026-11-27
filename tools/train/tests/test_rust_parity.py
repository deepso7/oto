"""Two-way PyTorch <-> Rust (<-> wasm) parity against ``crates/oto-dsp`` (SPEC.md section 6: <= -90 dB).

* Rust -> PyTorch: the kernel's random models (``oto1_bench --write``) load into PyTorch with
  :func:`load_otow_into_model` and reproduce ``ref-{a,b}-out.f32`` in streaming and batch mode.
* PyTorch -> Rust: PyTorch exports (non-trivial BatchNorm folded, f32 and f16, and a ``--smoke``
  training checkpoint) run through the kernel on every native backend (``oto1_bench --eval``) and
  through the wasm builds in Node, matching PyTorch run on the exact stored weights.

Needs ``cargo`` (skipped otherwise); the wasm check also needs ``node`` and ``web/demo/wasm/*.wasm``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import rel_db
from oto_train.export_otow import DTYPE_F16, DTYPE_F32, _randomise_bn, export, load_otow_into_model
from oto_train.models import build_model
from oto_train.models import stft as S

ROOT = Path(__file__).resolve().parents[3]
REF = ROOT / "target" / "oto1"
WASM = ROOT / "web" / "demo" / "wasm"
GATE = -90.0
HOPS = 200

pytestmark = pytest.mark.skipif(shutil.which("cargo") is None, reason="needs cargo")

_CARGO = ["cargo", "run", "-q", "--release", "-p", "oto-dsp", "--no-default-features", "--features", "oto1",
          "--example", "oto1_bench", "--"]  # fmt: skip


def _f32(p: Path) -> torch.Tensor:
    return torch.from_numpy(np.fromfile(p, dtype="<f4").copy())


def _write_f32(p: Path, x: torch.Tensor) -> None:
    x.detach().cpu().numpy().astype("<f4").tofile(p)


@pytest.fixture(scope="module")
def rust_refs():
    """Fresh ``target/oto1/{random-*.otow, ref-*.f32}`` from the kernel (short benchmark run)."""
    subprocess.run([*_CARGO, "--write", "10", "0.2"], cwd=ROOT, check=True, capture_output=True)
    return REF


def _rust_eval(otow: Path, x: torch.Tensor, tmp: Path) -> dict[str, torch.Tensor]:
    """Run ``x`` through the kernel on every native backend; ``{backend: output}``."""
    inp = tmp / "in.f32"
    _write_f32(inp, x)
    subprocess.run([*_CARGO, "--eval", str(otow), str(inp), str(tmp / "out")], cwd=ROOT, check=True,
                   capture_output=True)  # fmt: skip
    return {p.name.split(".")[1]: _f32(p) for p in sorted(tmp.glob("out.*.f32"))}


_NODE_EVAL = r"""
import { readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
const [core, wasm, otow, input, out] = process.argv.slice(1);
const { otoInstance, HOP } = createRequire(import.meta.url)(core);
const b = readFileSync(input);
const x = new Float32Array(b.buffer, b.byteOffset, b.length / 4);
const inst = otoInstance(new WebAssembly.Module(readFileSync(wasm)), { weights: readFileSync(otow) });
const y = new Float32Array(x.length);
for (let h = 0; h < x.length / HOP; h++) {
  inst.input.set(x.subarray(h * HOP, (h + 1) * HOP));
  inst.w.oto_process_hop();
  y.set(inst.output, h * HOP);
}
writeFileSync(out, Buffer.from(y.buffer));
"""


def _wasm_eval(otow: Path, x: torch.Tensor, tmp: Path) -> dict[str, torch.Tensor]:
    inp = tmp / "in.f32"
    _write_f32(inp, x)
    outs = {}
    for build in ("simd", "scalar"):
        out = tmp / f"wasm-{build}.f32"
        subprocess.run(["node", "--input-type=module", "-e", _NODE_EVAL, str(ROOT / "web/demo/bench-core.js"),
                        str(WASM / f"oto-{build}.wasm"), str(otow), str(inp), str(out)],
                       check=True, capture_output=True)  # fmt: skip
        outs[f"wasm-{build}"] = _f32(out)
    return outs


def _signal(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    n = HOPS * S.HOP
    t = torch.arange(n, dtype=torch.float64) / S.SR
    x = 0.2 * torch.sin(2 * torch.pi * 220 * t) * (1 + 0.5 * torch.sin(2 * torch.pi * 3 * t))
    return (x + 0.05 * torch.randn(n, generator=g, dtype=torch.float64)).float()


def _torch_out(model, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(streaming, batch)`` outputs of ``x`` (1-D), both on the kernel's 512-sample-delayed grid."""
    ys, _ = model.stream(1).run(x[None])
    with torch.no_grad():
        yb = model(x[None])["wav"][0]
    return ys[0], torch.cat([ys[0, : S.HOP], yb[: -S.HOP]])  # batch has no pre-roll: borrow streaming's


# -- Rust -> PyTorch -------------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ["a", "b"])
def test_rust_random_model_in_pytorch(arch, rust_refs):
    model = load_otow_into_model(rust_refs / f"random-{arch}.otow")
    assert model.arch_id == {"a": 1, "b": 2}[arch]
    x, ref = _f32(rust_refs / "ref-input.f32"), _f32(rust_refs / f"ref-{arch}-out.f32")
    ys, yb = _torch_out(model, x)
    db_s, db_b = rel_db(ys, ref), rel_db(yb[S.HOP :], ref[S.HOP :])
    print(f"Rust->PyTorch {arch.upper()}: streaming {db_s:.1f} dB, batch {db_b:.1f} dB, "
          f"max abs {float((ys - ref).abs().max()):.2e}")  # fmt: skip
    assert db_s <= GATE and db_b <= GATE


# -- PyTorch -> Rust / wasm ------------------------------------------------------------------------


def _random_export(arch: str, dtype: int, tmp: Path, seed: int = 3) -> Path:
    torch.manual_seed(seed)
    m = build_model(arch)
    _randomise_bn(m, seed)  # non-trivial running stats, gamma and beta, folded at export
    m.eval()
    path = tmp / f"{arch}-{dtype}.otow"
    export(m, path, dtype)
    return path


def _check_pytorch_to_rust(otow: Path, tmp: Path, label: str, wasm: bool = False) -> None:
    model = load_otow_into_model(otow)  # PyTorch on the exact stored (f32 or f16-rounded) weights
    x = _signal(7)
    ys, yb = _torch_out(model, x)
    outs = _rust_eval(otow, x, tmp)
    assert "scalar" in outs
    if wasm:
        outs |= _wasm_eval(otow, x, tmp)
    for be, y in outs.items():
        db_s, db_b = rel_db(y, ys), rel_db(y[S.HOP :], yb[S.HOP :])
        print(f"PyTorch->{be} {label}: vs streaming {db_s:.1f} dB, vs batch {db_b:.1f} dB")
        assert db_s <= GATE and db_b <= GATE, (be, db_s, db_b)


@pytest.mark.parametrize("dtype", [DTYPE_F32, DTYPE_F16], ids=["f32", "f16"])
@pytest.mark.parametrize("arch", ["a", "b"])
def test_pytorch_export_in_rust(arch, dtype, tmp_path):
    label = f"{arch.upper()} {'f32' if dtype == DTYPE_F32 else 'f16'}"
    _check_pytorch_to_rust(_random_export(arch, dtype, tmp_path), tmp_path, label)


@pytest.mark.skipif(shutil.which("node") is None or not (WASM / "oto-simd.wasm").exists(),
                    reason="needs node and web/demo/wasm builds")  # fmt: skip
@pytest.mark.parametrize("arch", ["a", "b"])
def test_pytorch_export_in_wasm(arch, tmp_path):
    _check_pytorch_to_rust(_random_export(arch, DTYPE_F16, tmp_path), tmp_path, f"{arch.upper()} f16", wasm=True)


@pytest.mark.parametrize("arch", ["a", "b"])
def test_smoke_checkpoint_in_rust(arch, tmp_path):
    from oto_train.train import main

    main(["--smoke", "--arch", arch, "--ckpt-dir", str(tmp_path / "ckpt")])
    _check_pytorch_to_rust(tmp_path / "ckpt" / "best.otow", tmp_path, f"{arch.upper()} smoke ckpt")
