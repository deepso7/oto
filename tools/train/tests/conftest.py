import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def rel_db(a: torch.Tensor, ref: torch.Tensor) -> float:
    """Relative RMS error in dB (``-inf`` if identical)."""
    err = (a.double() - ref.double()).square().mean().sqrt()
    den = ref.double().square().mean().sqrt().clamp_min(1e-30)
    return float(20 * torch.log10(err / den + 1e-300))


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
