"""Measurement harness: capacity, crosstalk, noise-robustness, leakage.

Lazy re-exports, for the same reason as :mod:`fka.data` — keeps ``python -m fka.eval.capacity``
from double-importing the module.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from fka.eval.capacity import (
        CapacityReport,
        DictMemory,
        Query,
        RandomGuessMemory,
        RelationResult,
        capacity_curve,
        measure_capacity,
    )

_EXPORTS = {
    "CapacityReport": "fka.eval.capacity",
    "DictMemory": "fka.eval.capacity",
    "Query": "fka.eval.capacity",
    "RandomGuessMemory": "fka.eval.capacity",
    "RelationResult": "fka.eval.capacity",
    "capacity_curve": "fka.eval.capacity",
    "measure_capacity": "fka.eval.capacity",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(_EXPORTS[name]), name)


def __dir__() -> list[str]:
    return __all__
