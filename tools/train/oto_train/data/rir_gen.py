"""Paired near/far 48 kHz RIR bank generator (pyroomacoustics, MIT).

For each simulated shoebox room one microphone (cardioid or omni, random orientation) gets:

* ``near``: one user position ``0.15-0.6 m`` from the mic (headset / laptop user),
* ``far_k``: ``n_far`` interferer positions ``1.2-6 m`` from the mic (TV, second talker),

all rendered in the *same* room so near/far differ only in distance, angle and directivity.
Rooms use the hybrid simulator (image sources up to ``max_order`` + ray-traced late tail),
frequency-dependent material absorption, air absorption and source/mic directivity (the
image-source part honours directivity; pyroomacoustics' ray tracer treats sources as omni).

Output: ``{out}/{room:06d}.npz`` with ``near (L,)``, ``far (n_far, L)``, and JSON-able
metadata (dims, materials, rt60, distances, direct-path sample index). Use ``--shard`` to
generate disjoint room ranges on several machines. Train/dev/test room splits are by room
index (``room % 50 == 0`` -> dev, ``room % 50 == 1`` -> test).

    python -m oto_train.data.rir_gen --out /data/rirs --rooms 20000 --workers 16
"""

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SR = 48_000
C_SOUND = 343.0

# (wall material names from pyroomacoustics' database, weight). Mixed absorbers per room.
WALLS = ["hard_surface", "brickwork", "rough_concrete", "plasterboard", "wooden_lining", "glass_window",
         "gypsum_board", "curtains_cotton_0.5", "panel_fabric_covered_6pcf", "concrete_floor"]  # fmt: skip
FLOORS = ["carpet_cotton", "carpet_thin", "linoleum_on_concrete", "audience_floor", "concrete_floor", "felt_5mm"]
CEILINGS = ["ceiling_plasterboard", "ceiling_fissured_tile", "ceiling_perforated_gypsum_board", "hard_surface"]


def room_split(room: int) -> str:
    return "dev" if room % 50 == 0 else "test" if room % 50 == 1 else "train"


def _random_point_at(rng, center, dist, dims, margin=0.3, tries=200):
    for _ in range(tries):
        v = rng.normal(size=3)
        v[2] *= 0.3  # mostly horizontal
        v /= np.linalg.norm(v)
        p = center + dist * v
        if np.all(p > margin) and np.all(p < np.asarray(dims) - margin):
            return p
    return None


def _directivity(pra, kind: str, azimuth: float, colatitude: float):
    """Version-tolerant cardioid-family constructor (pyroomacoustics 0.7 / 0.8 APIs)."""
    from pyroomacoustics import directivities as D

    orient = D.DirectionVector(azimuth=azimuth, colatitude=colatitude, degrees=True)
    if kind == "omni":
        return None
    if hasattr(D, "CardioidFamily") and hasattr(D, "DirectivityPattern"):
        pat = {"cardioid": D.DirectivityPattern.CARDIOID, "hypercardioid": D.DirectivityPattern.HYPERCARDIOID,
               "subcardioid": D.DirectivityPattern.SUBCARDIOID}[kind]  # fmt: skip
        return D.CardioidFamily(orientation=orient, pattern_enum=pat)
    cls = {"cardioid": D.Cardioid, "hypercardioid": D.HyperCardioid, "subcardioid": D.SubCardioid}[kind]
    return cls(orient)


def simulate_room(room: int, seed: int = 0, n_far: int = 2, max_len_s: float = 1.5, ray_tracing: bool = True):
    """Simulate one room; returns ``(near, far, meta)`` or ``None`` if placement failed."""
    import pyroomacoustics as pra

    rng = np.random.default_rng([seed, room])
    # Volume-aware sizes: living rooms / offices / meeting rooms / halls.
    dims = np.array([rng.uniform(2.5, 12.0), rng.uniform(2.5, 10.0), rng.uniform(2.3, 4.5)])
    mats = {
        "east": rng.choice(WALLS), "west": rng.choice(WALLS), "north": rng.choice(WALLS),
        "south": rng.choice(WALLS), "floor": rng.choice(FLOORS), "ceiling": rng.choice(CEILINGS),
    }  # fmt: skip
    materials = {k: pra.Material(v, "rpg_skyline" if rng.random() < 0.3 else None) for k, v in mats.items()}
    mic = np.array([rng.uniform(0.5, dims[0] - 0.5), rng.uniform(0.5, dims[1] - 0.5), rng.uniform(0.7, 1.6)])
    near_d = rng.uniform(0.15, 0.6)
    user = _random_point_at(rng, mic, near_d, dims, margin=0.1)
    fars, far_d = [], []
    for _ in range(n_far):
        d = rng.uniform(1.2, 6.0)
        p = _random_point_at(rng, mic, d, dims)
        if p is None:
            d = 1.2
            p = _random_point_at(rng, mic, d, dims)
        if p is None:
            return None
        fars.append(p)
        far_d.append(d)
    if user is None:
        return None

    mic_kind = rng.choice(["omni", "cardioid", "subcardioid"], p=[0.5, 0.3, 0.2])
    to_user = user - mic
    az = math.degrees(math.atan2(to_user[1], to_user[0])) + rng.uniform(-60, 60)
    mic_dir = _directivity(pra, mic_kind, az, 90.0 + rng.uniform(-30, 30))

    room_obj = pra.ShoeBox(
        dims, fs=SR, materials=materials, max_order=int(rng.integers(8, 18)) if ray_tracing else 30,
        ray_tracing=ray_tracing, air_absorption=True, use_rand_ism=True, max_rand_disp=0.05,
    )  # fmt: skip
    if ray_tracing:
        room_obj.set_ray_tracing(receiver_radius=0.5, n_rays=5000, energy_thres=1e-7)
    # Talker faces roughly toward the mic; TV/loudspeaker interferers face anywhere.
    room_obj.add_source(user, directivity=_directivity(pra, "subcardioid", az + 180 + rng.uniform(-45, 45), 90.0))
    for p in fars:
        src_dir = _directivity(pra, rng.choice(["omni", "cardioid", "subcardioid"]), rng.uniform(0, 360), 90.0)
        room_obj.add_source(p, directivity=src_dir)
    room_obj.add_microphone(mic, directivity=mic_dir)
    room_obj.compute_rir()

    n = int(max_len_s * SR)
    rirs = [np.asarray(room_obj.rir[0][s], dtype=np.float32)[:n] for s in range(1 + n_far)]
    L = max(len(r) for r in rirs)
    rirs = [np.pad(r, (0, L - len(r))) for r in rirs]
    near, far = rirs[0], np.stack(rirs[1:])
    try:
        rt60 = float(np.median(room_obj.measure_rt60()))
    except Exception:  # noqa: BLE001
        rt60 = float("nan")
    # Geometric direct-path index (argmax is unreliable when directivity attenuates the direct ray).
    offset = pra.constants.get("frac_delay_length") // 2
    directs = [int(round(np.linalg.norm(p - mic) / C_SOUND * SR)) + offset for p in [user, *fars]]
    meta = {
        "room": room, "split": room_split(room), "dims": dims.round(3).tolist(), "materials": mats,
        "mic": mic.round(3).tolist(), "mic_directivity": str(mic_kind), "near_dist": round(float(near_d), 3),
        "far_dist": [round(float(d), 3) for d in far_d], "rt60": round(rt60, 3), "direct_idx": directs, "sr": SR,
    }  # fmt: skip
    return near, far, meta


def _work(args):
    room, out, seed, n_far, ray_tracing = args
    path = Path(out) / f"{room:06d}.npz"
    if path.exists():
        return room, "exists"
    res = simulate_room(room, seed, n_far, ray_tracing=ray_tracing)
    if res is None:
        return room, "placement_failed"
    near, far, meta = res
    np.savez_compressed(path, near=near, far=far, meta=json.dumps(meta))
    return room, meta["rt60"]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rooms", type=int, default=20_000)
    ap.add_argument("--start", type=int, default=0, help="first room index (for multi-machine generation)")
    ap.add_argument("--n-far", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--no-ray-tracing", action="store_true", help="pure ISM (fast, for tests)")
    args = ap.parse_args(argv)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    jobs = [(r, args.out, args.seed, args.n_far, not args.no_ray_tracing)
            for r in range(args.start, args.start + args.rooms)]  # fmt: skip
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for room, status in ex.map(_work, jobs, chunksize=4):
            if room % 100 == 0:
                print(f"room {room}: {status}", flush=True)


if __name__ == "__main__":
    main()
