"""Oto-1 trainer: GPU-side mixing, bf16, auto-resuming checkpoints, ``.otow`` export of the best model.

Examples::

    # Dry run on synthetic data, no datasets (CPU is fine)
    python -m oto_train.train --smoke --arch a --ckpt-dir /tmp/oto-smoke

    # Colab / rented GPU: stage shards from R2 to local NVMe, checkpoint every 20 min to R2
    python -m oto_train.train --arch a --data r2://oto-train/v1/ --stage-dir /content/data \\
        --rir-data r2://oto-train/rirs/ --ckpt-dir /content/ckpt --sync r2://oto-train/runs/a1/ \\
        --batch-size 32 --seconds 4 --steps 300000 --compile

    # Long-sequence fine-tune of the best Clean checkpoint (30-60 s segments; use a fresh --ckpt-dir)
    python -m oto_train.train --arch a --init-from /content/ckpt/best.pt --ckpt-dir /content/ckpt-long --long-seq ...

    # Phase 3 Isolate fine-tune (interferers, background-first/only, decoupled distance)
    python -m oto_train.train --arch a --init-from clean_best.pt --isolate ...

Checkpoints: ``{ckpt_dir}/latest.pt`` (every ``--ckpt-minutes``, on SIGTERM/SIGINT, and at the
end) and ``{ckpt_dir}/best.pt`` + ``best.otow`` + ``best.safetensors`` on validation improvement.
A restarted job with the same ``--ckpt-dir`` / ``--sync`` resumes automatically.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import signal
import tarfile
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data.mixer import Mixer, MixConfig
from .data.rir_gen import parse_max_rt60
from .data.stream import ShardStream, SyntheticStream
from .losses import LossWeights, OtoLoss
from .models import build_model
from .models.stft import HOP


def _clear_dir(root: Path) -> None:
    for child in root.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _norm_path(p: str | None) -> str | None:
    return str(Path(p).resolve()) if p else None


# ---------------------------------------------------------------------------
# Remote storage (r2://bucket/prefix, hf://user/repo[/path], or a local path)
# ---------------------------------------------------------------------------


def _split_remote(url: str) -> tuple[str, str, str]:
    scheme, rest = url.split("://", 1)
    if scheme == "hf":
        parts = rest.split("/")
        return scheme, "/".join(parts[:2]), "/".join(parts[2:])
    bucket, _, prefix = rest.partition("/")
    return scheme, bucket, prefix


def _extract_tar(tar_path: Path, dst: Path) -> str:
    """Safely extract a ``.tar`` archive into ``dst`` and return the extracted data directory."""
    with tarfile.open(tar_path) as tf:
        tf.extractall(dst, filter="data")
    tar_path.unlink(missing_ok=True)
    stem_dir = dst / tar_path.stem
    return str(stem_dir) if stem_dir.is_dir() else str(dst)


def stage(src: str, dst: str) -> str:
    """Copy ``src`` (R2 / HF / local, e.g. Google Drive mount) to local NVMe ``dst`` once."""
    from .data import pack_shards as P

    root = Path(dst)
    root.mkdir(parents=True, exist_ok=True)
    resolved = str(root)
    hf_repo = hf_path = ""
    if src.startswith("hf://"):
        _, hf_repo, hf_path = _split_remote(src)
        hf_path = hf_path.strip("/")
        if hf_path and not hf_path.endswith(".tar"):
            resolved = str(root / hf_path)
    if src.rstrip("/").endswith(".tar"):
        stem_dir = root / Path(src.rstrip("/")).stem
        resolved = str(stem_dir) if stem_dir.is_dir() else str(root)
    marker = root / ".staged"
    staging = root / ".staging"
    if marker.exists() and marker.read_text() == src:
        return resolved
    if (marker.exists() and marker.read_text() != src) or (staging.exists() and staging.read_text() != src):
        _clear_dir(root)
    staging.write_text(src)
    t0 = time.time()
    if src.startswith("r2://"):
        _, bucket, prefix = _split_remote(src)
        P.download_r2(bucket, prefix, str(root))
    elif src.startswith("hf://"):
        pattern = None if not hf_path else (hf_path if hf_path.endswith(".tar") else f"{hf_path}/**")
        P.download_hf(hf_repo, str(root), pattern)
        if hf_path.endswith(".tar"):
            resolved = _extract_tar(root / hf_path, root)
    else:
        srcp = Path(src)
        if srcp.is_file() and srcp.suffix == ".tar":
            with tarfile.open(srcp) as tf:
                tf.extractall(root, filter="data")
            stem_dir = root / srcp.stem
            resolved = str(stem_dir) if stem_dir.is_dir() else str(root)
        else:
            shutil.copytree(src, str(root), dirs_exist_ok=True)
    os.replace(staging, marker)
    print(f"staged {src} -> {resolved} in {time.time() - t0:.0f}s", flush=True)
    return resolved


def sync_up(path: Path, remote: str | None) -> None:
    if not remote or not path.exists():
        return
    from .data import pack_shards as P

    try:
        if remote.startswith("r2://"):
            _, bucket, prefix = _split_remote(remote)
            P.upload_r2(str(path), bucket, prefix)
        elif remote.startswith("hf://"):
            _, repo, sub = _split_remote(remote)
            P.upload_hf(str(path), repo, sub)
        else:
            Path(remote).mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, Path(remote) / path.name)
    except Exception as e:  # noqa: BLE001 - never kill a run because an upload failed
        print(f"warning: sync of {path} to {remote} failed: {e}", flush=True)


def sync_down(remote: str | None, name: str, dst: Path) -> None:
    if not remote or dst.exists():
        return
    try:
        if remote.startswith("r2://"):
            from .data.pack_shards import _r2_client

            _, bucket, prefix = _split_remote(remote)
            _r2_client().download_file(bucket, prefix + name, str(dst))
        elif remote.startswith("hf://"):
            from huggingface_hub import hf_hub_download

            _, repo, sub = _split_remote(remote)
            sub_clean = sub.strip("/")
            remote_file = f"{sub_clean}/{name}" if sub_clean else name
            shutil.copy2(hf_hub_download(repo, remote_file, repo_type="dataset"), dst)
        elif (Path(remote) / name).exists():
            shutil.copy2(Path(remote) / name, dst)
    except Exception as e:  # noqa: BLE001
        print(f"no remote checkpoint {name} ({e})", flush=True)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_ckpt(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].lerp_(v.float(), 1 - self.decay)
            else:
                self.shadow[k].copy_(v)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=["a", "b"], default="a")
    ap.add_argument("--set", action="append", default=[], metavar="K=V", help="model config override")
    ap.add_argument("--smoke", action="store_true", help="synthetic data, a few steps, export at the end")
    ap.add_argument("--data", help="shard root: local dir, r2://bucket/prefix/ or hf://user/repo/path")
    ap.add_argument("--stage-dir", default="/content/data", help="local NVMe staging dir for --data")
    ap.add_argument("--rir-data", help="RIR bank (.npz from rir_gen): local dir, r2:// or hf://")
    ap.add_argument(
        "--max-rt60", type=parse_max_rt60, default=1.5, help="ignore RIRs with RT60 above this in seconds (finite <=0 disables)"
    )
    ap.add_argument("--ckpt-dir", default="/content/ckpt")
    ap.add_argument("--sync", help="mirror checkpoints to r2://, hf:// or a local/Drive path")
    ap.add_argument("--ckpt-minutes", type=float, default=20.0)
    ap.add_argument("--init-from", help="initialise weights from a checkpoint (fine-tunes); resets optimizer")
    ap.add_argument("--seconds", type=float, default=4.0, help="segment length")
    ap.add_argument("--long-seq", action="store_true", help="30-60 s segment fine-tune preset")
    ap.add_argument("--isolate", action="store_true", help="Phase 3 mixing recipe (interferers etc.)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-batches", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--compile", action="store_true", help="torch.compile the model")
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--export-dtype", choices=["f16", "f32"], default="f16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args(argv)
    if args.smoke:
        args.steps = min(args.steps, 4) if args.steps == 300_000 else args.steps
        args.batch_size = min(args.batch_size, 2)
        args.seconds = min(args.seconds, 1.0)
        args.val_every = args.steps
        args.val_batches = 1
        args.workers = 0
        args.warmup = 1
        args.log_every = 1
    if args.long_seq:
        args.seconds = max(args.seconds, 30.0)
        args.batch_size = min(args.batch_size, 8)
        args.lr = min(args.lr, 2e-4)
        args.warmup = min(args.warmup, 500)
    return args


def mix_config(args) -> MixConfig:
    if args.isolate:
        return MixConfig(p_interferer=0.7, p_user_far=0.2, p_background_first=0.25, p_background_only=0.1)
    return MixConfig()


def make_loader(args, split: str, data_root: str | None, rir_root: str | None, seed: int) -> DataLoader:
    if args.smoke or not data_root:
        ds = SyntheticStream(args.seconds, seed)
    else:
        ds = ShardStream(data_root, split, args.seconds, rir_root, seed, max_rt60=args.max_rt60)
    nw = args.workers if split == "train" else min(args.workers, 2)
    return DataLoader(ds, batch_size=args.batch_size, num_workers=nw, pin_memory=torch.cuda.is_available(),
                      persistent_workers=nw > 0, prefetch_factor=4 if nw > 0 else None)  # fmt: skip


def run_batch(model, mixer: Mixer, loss_fn: OtoLoss, batch: dict, device, bf16: bool):
    mixed = mixer(**{k: v.to(device, non_blocking=True) for k, v in batch.items()})
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
        out = model(mixed["mix"])
    # Batch iSTFT leaves the last HOP (512) samples incomplete because frame F does not exist yet.
    valid = out["wav"].shape[-1] - HOP
    if valid > 0:
        wav = out["wav"][..., :valid]
        tgt = mixed["target"][..., :valid]
        mix = mixed["mix"][..., :valid]
        p_logit = out["p_logit"][..., : valid // HOP]
    else:
        wav, tgt, mix, p_logit = out["wav"], mixed["target"], mixed["mix"], out["p_logit"]
    return loss_fn(wav, tgt, mix, p_logit, mixed["f_max"])


def main(argv=None) -> dict:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16 = device.type == "cuda" and not args.no_bf16 and torch.cuda.is_bf16_supported()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    overrides = {k: json.loads(v) for k, v in (kv.split("=", 1) for kv in args.set)}

    latest = ckpt_dir / "latest.pt"
    sync_down(args.sync, "latest.pt", latest)
    sync_down(args.sync, "best.pt", ckpt_dir / "best.pt")

    def _ckpt_cfg(ck: dict, label: str) -> dict:
        ck_arch = ck.get("arch")
        if ck_arch is not None and ck_arch != args.arch:
            raise ValueError(f"--arch {args.arch!r} conflicts with {label} (arch={ck_arch!r})")
        ck_cfg = dict(ck.get("model_cfg") or {})
        if args.arch == "b" and "channels" not in ck_cfg:
            sd = ck.get("ema") or ck.get("model") or {}
            if "enc0.conv.weight" in sd:
                ck_cfg["channels"] = int(sd["enc0.conv.weight"].shape[0])
        for k, v in overrides.items():
            if k in ck_cfg and ck_cfg[k] != v:
                raise ValueError(f"--set {k}={v!r} conflicts with {label} ({k}={ck_cfg[k]!r})")
        return {**ck_cfg, **overrides}

    resumed_ck = None
    init_ck = None
    model_cfg = dict(overrides)
    if latest.exists():
        resumed_ck = torch.load(latest, map_location=device, weights_only=True)
        if args.init_from:
            ck_init = _norm_path((resumed_ck.get("args") or {}).get("init_from"))
            if ck_init != _norm_path(args.init_from):
                raise ValueError(
                    f"--init-from {args.init_from!r} conflicts with existing {latest} "
                    f"(initialized from {ck_init!r}); pass a fresh --ckpt-dir for fine-tuning"
                )
        model_cfg = _ckpt_cfg(resumed_ck, f"checkpoint {latest}")
    elif args.init_from:
        init_ck = torch.load(args.init_from, map_location=device, weights_only=True)
        model_cfg = _ckpt_cfg(init_ck, f"--init-from {args.init_from}")

    model = build_model(args.arch, **model_cfg).to(device)
    print(model.summary(), flush=True)

    data_root = rir_root = None
    if not args.smoke and args.data:
        data_root = stage(args.data, str(Path(args.stage_dir) / "data")) if "://" in args.data else args.data
    if not args.smoke and args.rir_data:
        rir_root = stage(args.rir_data, str(Path(args.stage_dir) / "rirs")) if "://" in args.rir_data else args.rir_data

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))

    ema = EMA(model, args.ema)
    step, best_val = 0, float("inf")

    if resumed_ck is not None:
        model.load_state_dict(resumed_ck["model"])
        opt.load_state_dict(resumed_ck["opt"])
        ema.shadow = {k: v.to(device) for k, v in resumed_ck["ema"].items()}
        step, best_val = resumed_ck["step"], resumed_ck.get("best_val", float("inf"))
        torch.set_rng_state(resumed_ck["rng_cpu"].to(device="cpu", dtype=torch.uint8))
        print(f"resumed from {latest} at step {step} (best val {best_val:.4f})", flush=True)
    elif init_ck is not None:
        model.load_state_dict(init_ck["ema"] if init_ck.get("ema") else init_ck["model"])
        ema = EMA(model, args.ema)
        print(f"initialised from {args.init_from}", flush=True)

    fwd_model = torch.compile(model) if args.compile else model
    loss_fn = OtoLoss(LossWeights())
    mixer = Mixer(mix_config(args), seed=args.seed + step, device=device)

    def state() -> dict:
        return {"arch": args.arch, "model_cfg": asdict(model.cfg), "model": model.state_dict(), "opt": opt.state_dict(),
                "ema": ema.shadow, "step": step, "best_val": best_val, "args": vars(args),
                "rng_cpu": torch.get_rng_state()}  # fmt: skip

    stop = {"flag": False}

    def _on_signal(signum, _frame):
        print(f"signal {signum}: checkpointing and exiting", flush=True)
        stop["flag"] = True

    for s in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(s, _on_signal)
        except ValueError:  # not in main thread (e.g. some notebook runners)
            pass

    @torch.no_grad()
    def validate() -> float:
        ema_model = copy.deepcopy(model)
        ema_model.load_state_dict({k: v.to(ema_model.state_dict()[k].dtype) for k, v in ema.shadow.items()})
        ema_model.eval()
        vmixer = Mixer(mix_config(args), seed=12345, device=device)
        vl = make_loader(args, "dev", data_root, rir_root, seed=999)
        total, n = 0.0, 0
        for i, batch in enumerate(vl):
            if i >= args.val_batches:
                break
            loss, _ = run_batch(ema_model, vmixer, loss_fn, batch, device, bf16)
            total += float(loss)
            n += 1
        return total / max(n, 1), ema_model

    loader = make_loader(args, "train", data_root, rir_root, seed=args.seed + step)
    last_ckpt = time.time()
    t0, seen = time.time(), 0
    result: dict = {}
    model.train()
    for batch in loader:
        if step >= args.steps or stop["flag"]:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        loss, parts = run_batch(fwd_model, mixer, loss_fn, batch, device, bf16)
        if not torch.isfinite(loss):
            print(f"step {step}: non-finite loss {parts}, skipping batch", flush=True)
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        ema.update(model)
        step += 1
        seen += batch["speech"].shape[0] * batch["speech"].shape[1]
        result = {"step": step, "loss": float(loss.detach()), **{k: float(v) for k, v in parts.items()}, "grad_norm": float(gn)}
        if step % args.log_every == 0:
            rt = seen / 48_000 / max(time.time() - t0, 1e-6)
            print(json.dumps({**result, "lr": lr_at(step), "audio_s_per_s": round(rt, 1)}), flush=True)

        if step % args.val_every == 0 or step == args.steps:
            val, ema_model = validate()
            result["val"] = val
            print(json.dumps({"step": step, "val": val, "best": min(best_val, val)}), flush=True)
            if val < best_val:
                best_val = val
                save_ckpt(ckpt_dir / "best.pt", state())
                _export_best(ema_model, ckpt_dir, args, step, val)
                for name in ("best.pt", "best.otow", "best.safetensors"):
                    sync_up(ckpt_dir / name, args.sync)
            model.train()

        if time.time() - last_ckpt > args.ckpt_minutes * 60:
            save_ckpt(latest, state())
            sync_up(latest, args.sync)
            last_ckpt = time.time()

    save_ckpt(latest, state())
    sync_up(latest, args.sync)
    result["best_val"] = best_val
    return result


def _export_best(ema_model, ckpt_dir: Path, args, step: int, val: float) -> None:
    from .export_otow import DTYPE_F16, DTYPE_F32, export, otow_flags

    m = copy.deepcopy(ema_model).cpu().float().eval()
    export(m, ckpt_dir / "best.otow", DTYPE_F16 if args.export_dtype == "f16" else DTYPE_F32,
           otow_flags(isolate=args.isolate), ckpt_dir / "best.safetensors",
           {"step": step, "val_loss": val})  # fmt: skip


def cli(argv=None) -> None:
    """Console entry point (returns None so ``sys.exit`` exits 0 on success)."""
    main(argv)


if __name__ == "__main__":
    cli()
