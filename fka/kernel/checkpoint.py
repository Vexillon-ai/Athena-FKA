"""Run checkpoints: weights, config, and RNG state.

Policy (CLAUDE.md): every non-smoke run writes one before exiting. The diagnostics you want are
almost never the ones you planned, and without weights each one costs a full retrain.

RNG state is saved alongside because a follow-up run that reproduces the *model* but not the
*stream* diverges from the run being explained — and then you are debugging two models.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import torch


def _as_dict(obj) -> dict:
    if obj is None:
        return {}
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return dict(obj)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    model_config=None,
    train_config=None,
    extra: dict | None = None,
) -> Path:
    """Write weights + config + RNG state. Returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "model_config": _as_dict(model_config),
        "train_config": _as_dict(train_config),
        "rng": {
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
        "extra": extra or {},
    }
    torch.save(payload, path)
    # A sidecar so a checkpoint's provenance is greppable without loading torch.
    path.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "model_config": payload["model_config"],
                "train_config": payload["train_config"],
                "extra": payload["extra"],
                "n_params": sum(p.numel() for p in model.parameters()),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return path


def load_checkpoint(path: str | Path, model: torch.nn.Module | None = None, *, restore_rng=False):
    """Load a checkpoint. If ``model`` is given its weights are loaded in place."""
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    if model is not None:
        model.load_state_dict(blob["model_state"])
    if restore_rng:
        torch.set_rng_state(blob["rng"]["torch"])
        if blob["rng"]["torch_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(blob["rng"]["torch_cuda"])
        np.random.set_state(blob["rng"]["numpy"])
        random.setstate(blob["rng"]["python"])
    return blob
