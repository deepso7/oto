"""Oto-1 candidate models (batch + exact streaming) and the shared STFT."""

from .base import OtoModel, Streamer
from .candidate_a import CandidateA, CandidateAConfig
from .candidate_b import CandidateB, CandidateBConfig

ARCHS = {"a": (CandidateA, CandidateAConfig), "b": (CandidateB, CandidateBConfig)}
ARCH_IDS = {1: "a", 2: "b"}
ARCH_NAMES = {"oto1-a": "a", "oto1-b": "b"}


def build_model(arch: str, **overrides) -> OtoModel:
    """``arch``: ``"a"`` / ``"b"``, arch_id ``1`` / ``2`` or ``"oto1-a"`` / ``"oto1-b"``; overrides go to the config."""
    arch = ARCH_IDS.get(int(arch), arch) if str(arch).isdigit() else ARCH_NAMES.get(arch, arch).lower()
    cls, cfg_cls = ARCHS[arch]
    return cls(cfg_cls(**overrides))


def main() -> None:
    """Print parameter and MAC counts for both candidates: ``python -m oto_train.models``."""
    for arch in ("a", "b"):
        m = build_model(arch)
        print(m.summary())


__all__ = ["OtoModel", "Streamer", "CandidateA", "CandidateAConfig", "CandidateB", "CandidateBConfig", "build_model"]
