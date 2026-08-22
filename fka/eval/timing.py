"""Microbenchmark timing — warmup plus best-of-k, with device-aware synchronisation.

**Standing policy for this project: never time a single cold call.** The rule is not stylistic.
On 2026-08-01 a 4096² fp32 matmul on gfx1151 measured 0.44 TFLOP/s timed cold and 2.11 TFLOP/s
warmed up — a 4.8x difference from timing methodology alone, large enough to look like broken
hardware. See ``experiments/2026-07-31_env-validation/notes.md``.

Why the specific choices here:

* **Warmup** absorbs one-time costs that get misattributed to the operation: rocBLAS
  initialisation, kernel load and autotune, allocator growth, and lazy module init.
* **Best-of-k**, not mean, is the headline. Timing noise on a shared desktop is one-sided —
  interference only ever makes a run *slower* — so the minimum is the best estimator of the
  operation's intrinsic cost. The mean measures the machine's background load as much as the code.
* **Median and stdev are reported alongside** so a run whose spread is large announces itself
  instead of quietly producing a confident-looking number.
* **Device synchronisation** wraps every sample. GPU work is asynchronous; without a sync you are
  timing kernel *launch*, which for small kernels is close to free and reads as an impossible
  speedup.

This matters most where a benchmark's conclusion is a *shape* rather than a number — the Phase 2
router study claims an O(√N) log-log slope (research plan §3.5), and a fixed cold-start overhead
that does not scale with N will bend that line toward a flatter, better-looking exponent.

``torch`` is imported softly so this stays usable for pure-NumPy benchmarks (the router scaling
sweep does not need a GPU to be meaningful).
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

try:  # soft dependency: timing is useful without torch installed
    import torch
except ImportError:  # pragma: no cover - exercised only in torch-free environments
    torch = None  # type: ignore[assignment]

DEFAULT_WARMUP = 3
DEFAULT_REPEATS = 10

#: Relative spread (stdev/median) above which a measurement is flagged unstable rather than
#: silently reported. Chosen loosely: this is a "look again" signal, not a hard threshold.
INSTABILITY_THRESHOLD = 0.25


def synchronize(device: Any = None) -> None:
    """Block until queued device work has finished. No-op on CPU or without torch."""
    if torch is None or device is None:
        return
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass(frozen=True)
class TimingResult:
    """Timings for one benchmarked operation. ``best`` is the number to quote."""

    name: str
    samples: tuple[float, ...]
    warmup: int
    device: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def repeats(self) -> int:
        return len(self.samples)

    @property
    def best(self) -> float:
        return min(self.samples)

    @property
    def median(self) -> float:
        return statistics.median(self.samples)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.samples)

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.samples) if len(self.samples) > 1 else 0.0

    @property
    def relative_spread(self) -> float:
        """``stdev / median`` — how much the machine interfered during measurement."""
        return self.stdev / self.median if self.median > 0 else 0.0

    @property
    def stable(self) -> bool:
        return self.relative_spread <= INSTABILITY_THRESHOLD

    def throughput(self, units: float) -> float:
        """``units`` per second at the best sample, e.g. tokens/sec or FLOP/s."""
        return units / self.best if self.best > 0 else float("inf")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "device": self.device,
            "warmup": self.warmup,
            "repeats": self.repeats,
            "seconds_best": self.best,
            "seconds_median": self.median,
            "seconds_mean": self.mean,
            "seconds_stdev": self.stdev,
            "relative_spread": self.relative_spread,
            "stable": self.stable,
            **self.metadata,
        }

    def __str__(self) -> str:
        flag = "" if self.stable else f"  UNSTABLE (spread {self.relative_spread:.1%})"
        return (
            f"{self.name}: best {self.best * 1e3:.3f} ms, median {self.median * 1e3:.3f} ms "
            f"(warmup {self.warmup}, repeats {self.repeats}){flag}"
        )


def benchmark(
    fn: Callable[[], Any],
    *,
    name: str = "benchmark",
    warmup: int = DEFAULT_WARMUP,
    repeats: int = DEFAULT_REPEATS,
    device: Any = None,
    metadata: dict | None = None,
) -> TimingResult:
    """Time ``fn`` with ``warmup`` untimed calls followed by ``repeats`` timed ones.

    ``fn`` takes no arguments; bind them with a lambda or ``functools.partial``. Its return value
    is discarded, but it must be *kept alive long enough to be computed* — with torch that means
    the callable should not return a lazily-evaluated handle without touching it, which is why
    every sample is followed by a device synchronise.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")

    for _ in range(warmup):
        fn()
    synchronize(device)

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        synchronize(device)
        samples.append(time.perf_counter() - start)

    return TimingResult(
        name=name,
        samples=tuple(samples),
        warmup=warmup,
        device=str(device) if device is not None else None,
        metadata=dict(metadata or {}),
    )


def benchmark_throughput(
    fn: Callable[[], Any],
    units_per_call: float,
    *,
    name: str = "benchmark",
    unit: str = "units",
    **kwargs,
) -> tuple[TimingResult, float]:
    """:func:`benchmark` plus the derived rate, e.g. tokens/sec.

    Returns ``(result, rate)``; the rate is also stored in the result's metadata so it survives
    into ``to_dict()`` and any JSON written from it.
    """
    result = benchmark(fn, name=name, **kwargs)
    rate = result.throughput(units_per_call)
    return (
        TimingResult(
            name=result.name,
            samples=result.samples,
            warmup=result.warmup,
            device=result.device,
            metadata={
                **result.metadata,
                f"{unit}_per_sec": rate,
                f"{unit}_per_call": units_per_call,
            },
        ),
        rate,
    )
