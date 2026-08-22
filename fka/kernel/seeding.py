"""One place to seed everything, called before anything random happens.

CLAUDE.md says "seed everything", and Phase 1 did not: `train_d3` calls `torch.manual_seed` at the
*start of training*, which is after the run script has already constructed the model. Model
initialisation therefore came from an unseeded global RNG, so **no D3 run before 2026-08-02 is
exactly reproducible** — same seed, same corpus fingerprint, different weights.

That is how the 50M stability check's NaN (step ≤251) escaped: three 400-step reproduction attempts
under the same nominal seed initialised three different models, and the defect is init-dependent.
A seed that does not cover initialisation is not a seed; it is a label.

Call `seed_everything(seed)` at the top of a run script, before building anything.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, *, deterministic_algorithms: bool = False) -> int:
    """Seed python, numpy and torch (CPU + all devices). Returns the seed, for logging."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is optional for the data-only paths
        return seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_algorithms:
        # Off by default: it makes some ROCm kernels fall back to slow paths, and bitwise
        # determinism is not what we need — reproducible *initialisation* is.
        torch.use_deterministic_algorithms(True, warn_only=True)
    return seed
