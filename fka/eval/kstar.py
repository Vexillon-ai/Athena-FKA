"""K* — the discriminable-key cliff, under the criterion frozen in M5 §5.50.

    K*(P, surface) = the key count at which chance-corrected accuracy crosses 20%,
    interpolated LOG-LINEARLY in K between the two bracketing arms.

Option (b), a fitted knee, was tested against the 1M ladder and rejected on the data: corrected
accuracy flattens toward its floor and neither a power law nor an exponential has an asymptote
there, so both mis-predict the tail (M5 §5.50.1). The threshold crossing uses only the bracketing
pair and is insensitive to exactly the region the fits get wrong.

**The threshold level is not load-bearing** — 10% to 30% moves K* by at most ~15% and never out of
its bracket — which is the property that made it safe to freeze blind to the 10M measurement.
`threshold_sensitivity` reports that table so the claim travels with the number.

Every quote is **surface-indexed** (M5 §5.47): `K*_syllable` and `K*_verbose` are different
quantities and are not interchangeable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Frozen 2026-08-03, before any 10M arm ran. Changing it changes what K* means.
THRESHOLD = 0.20

# MEASURED by replication 2026-08-03 (M5 §5.102.2), replacing a guessed 0.015 that was wrong by
# 3x to 26x depending on regime. The single-band constant is RETIRED: run-to-run spread is not one
# number.
#
#   non-cliff  sigma ~4.6 pts  -- 10M@24k 4.42, 1M@16k 4.97, 1M@24k 4.43 (three seeds each,
#                                two model sizes, both sides of the transition)
#   cliff      sigma 12.8-39.4 -- 10M@32k 12.80, 10M@28k 39.35
#
# BAND CORRECTED 2026-08-05 (M5 §5.149.2), CONSTANTS DELIBERATELY UNCHANGED. Both figures above
# were taken at n=3. Over all four cliff-region arms at n=5 the band is sigma 12.7-44.5:
#   10M@24k 44.47 | 1M@24k 43.57 | 10M@28k 33.10 | 10M@32k 12.73
# The upper edge rose because the widest arm was never in the band, and 10M@28k FELL 39.35 -> 33.10.
# NOISE_SIGMA_CLIFF is the band's LOWER edge, so the flag is more eager than the measured spread
# warrants.
#
# RULED 2026-08-05: STAYS LIVE, as an ACCEPTED CONSERVATIVE BIAS. The bias runs in the safe
# direction — an over-eager flag summons scrutiny and can never suppress it, which is the opposite
# of the failure mode family member 5 exists to catch. Every firing this constant has ever produced
# was resolved by REPLICATION rather than by the threshold, so no verdict in the track rests on
# where the line sits. Re-deriving it therefore fails the price-consequence test: it would
# re-adjudicate every within-noise call in the track to change no conclusion. Recorded at the point
# of use so the bias is visible; any future change ships with the re-adjudication list.
#
# These are PER-ARM SIGMAS, not flag thresholds. The distinction cost a bug: using sigma itself as
# the threshold fires at 1 sigma, i.e. on roughly a third of clean sweeps. The difference of two
# independent arms has sd sigma*sqrt(2), so the threshold is k*sigma*sqrt(2) with k matching the
# outlier screen's k = 3 (M5 §5.84.1).
NOISE_SIGMA_NON_CLIFF = 0.046
NOISE_SIGMA_CLIFF = 0.128
FLAG_K = 3

#: Flag sensitivity that follows: **54.3 points at the cliff**, 19.5 off it. That is very
#: insensitive, and it is the honest consequence of measured cliff variance rather than a choice —
#: **a single-seed sweep can barely detect non-monotonicity at a cliff at all**, which is precisely
#: why §5.89 makes replication mandatory at cliff arms.


@dataclass(frozen=True)
class Arm:
    """One key-count arm: how many distinct keys, and what chance-corrected accuracy it reached."""

    keys: int
    corrected: float  # chance-corrected, in [0, 1]

    def __post_init__(self) -> None:
        if self.keys <= 0:
            raise ValueError(f"keys must be positive, got {self.keys}")


@dataclass(frozen=True)
class KStar:
    """A located cliff, or an explicit statement that the arms did not bracket one."""

    value: float | None
    bracket: tuple[int, int] | None
    bound: str | None  # ">" / "<" when every arm is on one side of the threshold
    threshold: float
    surface: str
    n_params: int | None
    non_monotone: tuple[tuple[int, int], ...] = ()

    @property
    def located(self) -> bool:
        return self.value is not None

    @property
    def params_per_key(self) -> float | None:
        if self.value is None or self.n_params is None:
            return None
        return self.n_params / self.value

    def __str__(self) -> str:
        if self.value is None:
            side = "unlocated"
            if self.bound and self.bracket:
                side = f"K* {self.bound} {self.bracket[0]:,}"
            return f"K*_{self.surface} {side} (threshold {self.threshold:.0%})"
        ratio = f", {self.params_per_key:.1f} params/key" if self.params_per_key else ""
        lo, hi = self.bracket  # type: ignore[misc]
        return (
            f"K*_{self.surface} = {self.value:,.0f} keys"
            f" (bracket {lo:,}-{hi:,}, threshold {self.threshold:.0%}{ratio})"
        )


def _non_monotone_pairs(
    arms: list[Arm], sigma: float = NOISE_SIGMA_CLIFF, k: float = FLAG_K
) -> tuple[tuple[int, int], ...]:
    """Adjacent pairs where MORE keys read materially BETTER — §5.55's mis-specification clause.

    ``sigma`` is a MEASURED per-arm spread (§5.102.2), not a guess, and the threshold is
    ``k * sigma * sqrt(2)`` — the sd of a DIFFERENCE of two independent arms. Pass
    ``NOISE_SIGMA_NON_CLIFF`` only when both arms are demonstrably away from the transition; the
    default is the cliff sigma, because a sweep that straddles its own cliff is when this matters.
    """
    threshold = k * sigma * math.sqrt(2)
    bad = []
    for lo, hi in zip(arms, arms[1:]):
        if hi.corrected > lo.corrected + threshold:
            bad.append((lo.keys, hi.keys))
    return tuple(bad)


def kstar(
    arms,
    *,
    threshold: float = THRESHOLD,
    surface: str = "syllable",
    n_params: int | None = None,
) -> KStar:
    """Locate the cliff by log-linear interpolation between the bracketing arms.

    Returns an *unlocated* result rather than a fabricated one when every arm sits on the same
    side of the threshold: an extrapolated cliff is exactly the reading-the-instrument failure the
    criterion exists to avoid.
    """
    arms = sorted(arms, key=lambda a: a.keys)
    if len(arms) < 2:
        raise ValueError("K* needs at least two arms to interpolate between")
    non_monotone = _non_monotone_pairs(arms)

    for lo, hi in zip(arms, arms[1:]):
        if lo.corrected >= threshold > hi.corrected:
            span = lo.corrected - hi.corrected
            frac = (lo.corrected - threshold) / span
            log_k = math.log(lo.keys) + frac * (math.log(hi.keys) - math.log(lo.keys))
            return KStar(
                value=math.exp(log_k),
                bracket=(lo.keys, hi.keys),
                bound=None,
                threshold=threshold,
                surface=surface,
                n_params=n_params,
                non_monotone=non_monotone,
            )

    every_alive = all(a.corrected >= threshold for a in arms)
    bound = ">" if every_alive else "<"
    edge = arms[-1].keys if every_alive else arms[0].keys
    return KStar(
        value=None,
        bracket=(edge, edge),
        bound=bound,
        threshold=threshold,
        surface=surface,
        n_params=n_params,
        non_monotone=non_monotone,
    )


def threshold_sensitivity(arms, levels=(0.10, 0.15, 0.20, 0.25, 0.30), **kw) -> dict[float, float]:
    """K* at several thresholds — the evidence that the level is a reporting convention."""
    out = {}
    for level in levels:
        located = kstar(arms, threshold=level, **kw)
        if located.value is not None:
            out[level] = located.value
    return out


@dataclass(frozen=True)
class ScalingFit:
    """A candidate law for K*(P), fitted to the measured (params, K*) points."""

    form: str
    predict: object  # Callable[[float], float]; kept loose so the dataclass stays frozen-friendly
    rms: float
    n_points: int

    @property
    def determined(self) -> bool:
        """False when the fit has as many parameters as points — no residual is available."""
        return self.n_points > 2

    def __call__(self, n_params: float) -> float:
        return self.predict(n_params)  # type: ignore[operator]


def fit_power(points: list[tuple[int, float]]) -> ScalingFit:
    """K* = a * P^alpha, fitted in log-log by least squares."""
    xs = [math.log(p) for p, _ in points]
    ys = [math.log(k) for _, k in points]
    alpha, intercept = _least_squares(xs, ys)
    predict = lambda p: math.exp(intercept + alpha * math.log(p))  # noqa: E731
    rms = _rms(points, predict)
    return ScalingFit(f"K* = {math.exp(intercept):.3g} * P^{alpha:.3f}", predict, rms, len(points))


def fit_log(points: list[tuple[int, float]]) -> ScalingFit:
    """K* = a * log2(P) + b, fitted by least squares."""
    xs = [math.log2(p) for p, _ in points]
    ys = [k for _, k in points]
    slope, intercept = _least_squares(xs, ys)
    predict = lambda p: slope * math.log2(p) + intercept  # noqa: E731
    rms = _rms(points, predict)
    return ScalingFit(f"K* = {slope:,.0f} * log2(P) + {intercept:,.0f}", predict, rms, len(points))


def params_for_keys(fit: ScalingFit, keys: float, *, hi: float = 1e18) -> float | None:
    """Invert a fit: how many parameters to reach ``keys`` discriminable keys. None if unreachable."""
    lo = 1.0
    if fit(hi) < keys:
        return None
    for _ in range(400):
        mid = math.sqrt(lo * hi)
        if fit(mid) < keys:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def _least_squares(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        raise ValueError("cannot fit: all points share one x")
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    return slope, my - slope * mx


def _rms(points: list[tuple[int, float]], predict) -> float:
    return math.sqrt(sum((predict(p) - k) ** 2 for p, k in points) / len(points))
