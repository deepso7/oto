"""Export a trained (or random) Oto-1 candidate to ``.safetensors`` and ``.otow`` v1.

``.otow`` v1 layout (little-endian, every section 64-byte aligned)::

    [0..4)    magic b"OTOW"
    [4..8)    u32 version = 1
    [8..12)   u32 arch_id (1 = Candidate A, 2 = Candidate B)
    [12..16)  u32 dtype (1 = f16, 2 = f32)
    [16..20)  u32 flags (bit 0 isolate_mode, bit 1 enrollment)
    [20..24)  u32 config_len
    [24..28)  u32 num_tensors
    [28..32)  u32 table_offset
    [32..64)  SHA-256 of bytes [64..EOF)
    [64..64+config_len)  flat canonical JSON config (sorted keys, no whitespace), zero-padded,
                         e.g. {"arch":"oto1-b","channels":32,"gru_hidden":128}
    [table_offset..)     num_tensors x 96-byte entries:
                           [0..64) UTF-8 name, zero-padded (<= 63 bytes)
                           [64..68) u32 ndim (1..=4)
                           [68..84) u32[4] shape (unused dims 0)
                           [84..88) u32 dtype
                           [88..96) u64 absolute byte offset (64-aligned)
    payloads             row-major little-endian, each zero-padded to 64 bytes

The contract is ``crates/oto-dsp/src/oto1/SPEC.md``. BatchNorm is folded before export.
Tensor order is the model's ``export_tensors()`` order (deterministic; the kernel accepts any).
:func:`load_otow_into_model` loads any v1 file (also the Rust-written random models) into PyTorch.
Usage::

    python -m oto_train.export_otow --arch a --random --seed 0 --out out/a_random.otow \\
        --safetensors out/a_random.safetensors --parity-fixture out/a_parity.safetensors
    python -m oto_train.export_otow --ckpt runs/a/best.pt --out out/oto1_clean.otow --dtype f16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .models import OtoModel, build_model

MAGIC = b"OTOW"
VERSION = 1
HEADER_LEN = 64
ENTRY_LEN = 96
NAME_LEN = 64
ALIGN = 64
DTYPE_F16 = 1
DTYPE_F32 = 2
FLAG_ISOLATE = 1 << 0
FLAG_ENROLLMENT = 1 << 1
_NP_DTYPE = {DTYPE_F16: np.dtype("<f2"), DTYPE_F32: np.dtype("<f4")}


def align(n: int, a: int = ALIGN) -> int:
    return (n + a - 1) // a * a


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def write_otow(
    path: str | Path,
    tensors: dict[str, torch.Tensor | np.ndarray],
    config: dict,
    arch_id: int,
    dtype: int = DTYPE_F16,
    flags: int = 0,
) -> bytes:
    """Write ``.otow`` v1 and return the file bytes."""
    if dtype not in _NP_DTYPE:
        raise ValueError(f"unknown dtype {dtype}")
    np_dtype = _NP_DTYPE[dtype]
    cfg = canonical_json(config)
    table_offset = align(HEADER_LEN + len(cfg))
    n = len(tensors)
    data_start = align(table_offset + n * ENTRY_LEN)

    table = bytearray()
    payload = bytearray()
    offset = data_start
    for name, t in tensors.items():
        arr = t.detach().cpu().float().numpy() if isinstance(t, torch.Tensor) else np.asarray(t, dtype=np.float32)
        name_b = name.encode("utf-8")
        if not 0 < len(name_b) < NAME_LEN:
            raise ValueError(f"tensor name must be 1..63 bytes: {name!r}")
        if not 1 <= arr.ndim <= 4:
            raise ValueError(f"{name}: ndim {arr.ndim} not in 1..=4")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name}: non-finite weights")
        if dtype == DTYPE_F16 and np.abs(arr).max(initial=0.0) > 65504.0:
            raise ValueError(f"{name}: value overflows f16")
        shape = list(arr.shape) + [0] * (4 - arr.ndim)
        table += name_b.ljust(NAME_LEN, b"\0")
        table += struct.pack("<I4II Q", arr.ndim, *shape, dtype, offset)
        raw = np.ascontiguousarray(arr).astype(np_dtype).tobytes()
        payload += raw + b"\0" * (align(len(raw)) - len(raw))
        offset += align(len(raw))

    body = bytearray()
    body += cfg + b"\0" * (table_offset - HEADER_LEN - len(cfg))
    body += table + b"\0" * (data_start - table_offset - len(table))
    body += payload
    digest = hashlib.sha256(body).digest()
    header = MAGIC + struct.pack("<7I", VERSION, arch_id, dtype, flags, len(cfg), n, table_offset) + digest
    assert len(header) == HEADER_LEN
    blob = bytes(header + body)
    assert len(blob) % ALIGN == 0
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(blob)
    return blob


@dataclass
class Otow:
    arch_id: int
    dtype: int
    flags: int
    config: dict
    tensors: dict[str, np.ndarray]  # always float32 views of the stored data
    sha256: bytes


class OtowError(ValueError):
    pass


def read_otow(path_or_bytes: str | Path | bytes) -> Otow:
    """Parse and fully verify an ``.otow`` v1 file (magic, version, SHA-256, alignment, bounds)."""
    blob = path_or_bytes if isinstance(path_or_bytes, bytes) else Path(path_or_bytes).read_bytes()
    if len(blob) < HEADER_LEN or blob[:4] != MAGIC:
        raise OtowError("bad magic")
    version, arch_id, dtype, flags, config_len, n, table_offset = struct.unpack_from("<7I", blob, 4)
    if version != VERSION:
        raise OtowError(f"unsupported version {version}")
    digest = blob[32:64]
    if hashlib.sha256(blob[64:]).digest() != digest:
        raise OtowError("SHA-256 mismatch")
    if dtype not in _NP_DTYPE:
        raise OtowError(f"bad dtype {dtype}")
    if table_offset % ALIGN or table_offset < HEADER_LEN + config_len or len(blob) % ALIGN:
        raise OtowError("misaligned table or file")
    if table_offset + n * ENTRY_LEN > len(blob):
        raise OtowError("tensor table out of bounds")
    config = json.loads(blob[HEADER_LEN : HEADER_LEN + config_len].decode())
    if canonical_json(config) != blob[HEADER_LEN : HEADER_LEN + config_len]:
        raise OtowError("config JSON is not canonical")
    tensors: dict[str, np.ndarray] = {}
    for i in range(n):
        e = table_offset + i * ENTRY_LEN
        name = blob[e : e + NAME_LEN].rstrip(b"\0").decode("utf-8")
        ndim, s0, s1, s2, s3, tdtype, off = struct.unpack_from("<I4II Q", blob, e + NAME_LEN)
        shape = (s0, s1, s2, s3)
        if not 1 <= ndim <= 4 or any(shape[ndim:]) or tdtype not in _NP_DTYPE:
            raise OtowError(f"bad entry {name!r}")
        if off % ALIGN:
            raise OtowError(f"{name}: misaligned payload")
        count = int(np.prod(shape[:ndim]))
        nbytes = count * _NP_DTYPE[tdtype].itemsize
        if off + nbytes > len(blob):
            raise OtowError(f"{name}: payload out of bounds")
        if name in tensors:
            raise OtowError(f"duplicate tensor {name}")
        arr = np.frombuffer(blob, dtype=_NP_DTYPE[tdtype], count=count, offset=off).reshape(shape[:ndim])
        tensors[name] = arr.astype(np.float32)
    return Otow(arch_id, dtype, flags, config, tensors, digest)


# ---------------------------------------------------------------------------
# Model <-> tensors
# ---------------------------------------------------------------------------


def export_model(model: OtoModel) -> tuple[dict[str, torch.Tensor], dict]:
    """Fold BatchNorm (in place) and return ``(tensors, config)``."""
    model.fold_batchnorm()
    return model.export_tensors(), model.config_dict()


def import_tensors(model: OtoModel, tensors: dict[str, np.ndarray | torch.Tensor]) -> OtoModel:
    """Load exported tensors into ``model`` (BN folded first); inverse of ``export_tensors``."""
    return model.load_tensors({k: torch.as_tensor(np.asarray(v, dtype=np.float32)) for k, v in tensors.items()})


#: SPEC.md config keys the kernel reads, per ``arch_id`` (everything else in the config is ignored).
_ARCH_KEYS = {1: ("channels", "gru_hidden", "gru_groups", "n_erb", "n_df_bins", "df_order"), 2: ("channels", "gru_hidden")}


def load_otow_into_model(path_or_bytes, model: OtoModel | None = None) -> OtoModel:
    """Build (or fill) a model from any ``.otow`` v1 file, including Rust-written ones.

    The architecture comes from the header ``arch_id``; the width from the config keys the kernel
    reads (missing keys take the SPEC.md defaults). Tensors are loaded with the kernel's strictness.
    Weights are the stored values widened to f32, so an f16 file runs on its f16-rounded weights.
    """
    o = read_otow(path_or_bytes)
    if o.flags & FLAG_ENROLLMENT:
        raise OtowError("enrolled models are not supported by kernel v1")
    if o.arch_id not in _ARCH_KEYS:
        raise OtowError(f"unknown arch_id {o.arch_id}")
    if model is None:
        model = build_model(o.arch_id, **{k: o.config[k] for k in _ARCH_KEYS[o.arch_id] if k in o.config})
    elif model.arch_id != o.arch_id:
        raise OtowError(f"file arch_id {o.arch_id} != model arch_id {model.arch_id}")
    return import_tensors(model, o.tensors).eval()


model_from_otow = load_otow_into_model


def otow_flags(isolate: bool = False, enrollment: bool = False) -> int:
    return (FLAG_ISOLATE if isolate else 0) | (FLAG_ENROLLMENT if enrollment else 0)


def export(
    model: OtoModel,
    out: str | Path,
    dtype: int = DTYPE_F16,
    flags: int = 0,
    safetensors_path: str | Path | None = None,
    extra_config: dict | None = None,
) -> bytes:
    from safetensors.torch import save_file

    tensors, config = export_model(model)
    if extra_config:
        # The config is flat (SPEC.md section 2); the kernel ignores keys it does not read.
        config = {**config, **{f"meta_{k}": v for k, v in extra_config.items() if v is not None}}
    if safetensors_path:
        Path(safetensors_path).parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(safetensors_path), metadata={"config": canonical_json(config).decode()})
    return write_otow(out, tensors, config, model.arch_id, dtype, flags)


@torch.no_grad()
def parity_fixture(model: OtoModel, path: str | Path, seconds: float = 2.0, seed: int = 0) -> None:
    """Save a Rust parity fixture: input, streaming output, logits, and first-frame spectra.

    ``input`` is a deterministic mix of a harmonic tone and white noise (multiple of 512 samples).
    ``stream_out`` is the exact 512-sample-delayed streaming output (sample 0..512 = pre-roll).
    """
    from safetensors.torch import save_file

    from .models import stft as S

    g = torch.Generator().manual_seed(seed)
    n = int(seconds * S.SR) // S.HOP * S.HOP
    t = torch.arange(n, dtype=torch.float64) / S.SR
    x = 0.1 * torch.sin(2 * torch.pi * 220 * t) + 0.05 * torch.sin(2 * torch.pi * 3300 * t)
    x = (x + 0.02 * torch.randn(n, generator=g, dtype=torch.float64)).float()[None]
    model.eval()
    out, p = model.stream(1).run(x)
    spec_in = S.stft(x)[0, :8]
    spec_out, _ = model.forward_spec(S.stft(x)[:, :8])
    save_file(
        {"input": x[0], "stream_out": out[0], "p_logit": p[0], "spec_in": spec_in, "spec_out": spec_out[0].contiguous()},
        str(path),
        metadata={"note": "stream_out[512:] == batch_out[:-512]; spec_* are frames 0..8 (re, im)"},
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", help="training checkpoint (.pt) written by oto_train.train")
    ap.add_argument("--arch", choices=["a", "b"], help="architecture (required with --random)")
    ap.add_argument("--random", action="store_true", help="export random weights (parity / browser benchmarks)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", action="append", default=[], metavar="K=V", help="config override, e.g. channels=64")
    ap.add_argument("--out", required=True, help=".otow output path")
    ap.add_argument("--dtype", choices=["f16", "f32"], default="f16")
    ap.add_argument("--safetensors", help="also write folded tensors as .safetensors")
    ap.add_argument("--parity-fixture", help="also write a streaming parity fixture (.safetensors, f32)")
    ap.add_argument("--isolate", action="store_true")
    ap.add_argument("--enrollment", action="store_true")
    args = ap.parse_args(argv)

    overrides = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        overrides[k] = json.loads(v)
    meta: dict = {}
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        model = build_model(ck["arch"], **ck["model_cfg"])
        model.load_state_dict(ck["ema"] if ck.get("ema") else ck["model"])
        meta = {"step": ck.get("step"), "val_loss": ck.get("best_val")}
    elif args.random:
        if not args.arch:
            ap.error("--random needs --arch")
        torch.manual_seed(args.seed)
        model = build_model(args.arch, **overrides)
        _randomise_bn(model, args.seed)
        meta = {"random_seed": args.seed}
    else:
        ap.error("pass --ckpt or --random")
    model.eval()
    dtype = DTYPE_F16 if args.dtype == "f16" else DTYPE_F32
    blob = export(model, args.out, dtype, otow_flags(args.isolate, args.enrollment), args.safetensors, meta)
    o = read_otow(blob)
    print(f"wrote {args.out}: {len(blob):,} bytes, {len(o.tensors)} tensors, sha256 {o.sha256.hex()}")
    if args.parity_fixture:
        # Fixture runs the model with the *stored* (possibly f16-rounded) weights.
        parity_fixture(load_otow_into_model(blob), args.parity_fixture, seed=args.seed)
        print(f"wrote {args.parity_fixture}")


@torch.no_grad()
def _randomise_bn(model: OtoModel, seed: int) -> None:
    """Give BatchNorm non-trivial running stats so folding is exercised by random exports."""
    g = torch.Generator().manual_seed(seed + 1)
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            m.running_mean.copy_(0.1 * torch.randn(m.num_features, generator=g))
            m.running_var.copy_(0.5 + torch.rand(m.num_features, generator=g))
            m.weight.copy_(1.0 + 0.1 * torch.randn(m.num_features, generator=g))
            m.bias.copy_(0.1 * torch.randn(m.num_features, generator=g))


if __name__ == "__main__":
    main()
