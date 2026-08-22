"""Validation for the margin instrument, designed BEFORE it reports anything (M3 ruling 1).

The discriminator pair is the point: **partial degradation** (every margin shrunken, none lost)
and **subset-total degradation** (most intact, a fraction inverted), at *matched means*.
An instrument that cannot separate those cannot answer the shape question, and the previous two
channels could not — which is why this one is validated before use rather than after.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from fka.eval.margin import MarginSet, MarginTrajectories, retrieval_margins

N, DIM = 200, 32
RNG = np.random.default_rng(0)


def _reference(n: int = N) -> np.ndarray:
    return np.full(n, 1.0)


def _partial(scale: float, n: int = N) -> MarginSet:
    """Every fact shrunken by the same factor: nothing lost, everything weaker."""
    return MarginSet(fact_ids=np.arange(n), margins=np.full(n, scale), load=1.0)


def _subset_total(frac_lost: float, n: int = N, lost_value: float = -1.0) -> MarginSet:
    """A subset falls off entirely; the rest are untouched."""
    k = int(frac_lost * n)
    m = np.concatenate([np.full(k, lost_value), np.ones(n - k)])
    return MarginSet(fact_ids=np.arange(n), margins=m, load=1.0)


# -- the discriminator pair ----------------------------------------------------------------


def test_matched_means_partial_vs_subset_total_are_separated():
    """THE load-bearing test. Identical means, opposite mechanisms; the instrument sees both."""
    # subset-total with 25% lost at -1.0: mean = 0.25*(-1) + 0.75*(1) = 0.5
    subset = _subset_total(0.25)
    partial = _partial(0.5)  # every margin at 0.5
    ref = _reference()

    assert partial.mean == pytest.approx(subset.mean), "the premise: means must match"

    # Inversion separates them outright.
    assert partial.inverted_fraction == 0.0
    assert subset.inverted_fraction == pytest.approx(0.25)

    # And so does the shape of the degradation, in the opposite direction.
    assert partial.degraded_fraction(ref) == 1.0, "partial: everything in the intermediate band"
    assert subset.degraded_fraction(ref) == 0.0, "subset-total: nothing intermediate"
    assert subset.intact_fraction(ref) == pytest.approx(0.75)


def test_verified_red_the_mean_alone_would_report_them_identical():
    """Negative control on the readout: a mean-only instrument fails this pair by construction."""
    subset, partial = _subset_total(0.25), _partial(0.5)
    assert partial.mean == subset.mean
    assert partial.retrieved_fraction != subset.retrieved_fraction


def test_gold_stub_a_perfect_key_set_has_large_positive_margins():
    keys = F.normalize(torch.randn(N, DIM, generator=torch.Generator().manual_seed(0)), dim=-1)
    m = retrieval_margins(keys, keys, torch.arange(N))
    assert float(m.min()) > 0, "every key must beat every other on its own query"


def test_verified_red_a_shuffled_key_set_inverts_the_margins():
    """The instrument must be able to report catastrophe, on a key set built to be wrong."""
    g = torch.Generator().manual_seed(0)
    keys = F.normalize(torch.randn(N, DIM, generator=g), dim=-1)
    queries = F.normalize(torch.randn(N, DIM, generator=g), dim=-1)  # unrelated
    m = retrieval_margins(keys, queries, torch.arange(N))
    assert float((m < 0).float().mean()) > 0.9


# -- trajectories, the primary readout ------------------------------------------------------


def _traj(margins: np.ndarray) -> MarginTrajectories:
    return MarginTrajectories(
        fact_ids=np.arange(margins.shape[0]),
        loads=np.arange(margins.shape[1], dtype=float),
        margins=margins,
    )


def test_collapse_sharpness_separates_a_cliff_from_a_slide_PER_FACT():
    """No aggregation between measurement and conclusion: each fact is classified on its own."""
    steps = 5
    slide = _traj(np.tile(np.linspace(1.0, 0.0, steps), (10, 1)))
    cliff = _traj(np.tile(np.array([1.0, 1.0, 1.0, 0.0, 0.0]), (10, 1)))

    assert np.nanmedian(cliff.collapse_sharpness()) == pytest.approx(1.0)
    assert np.nanmedian(slide.collapse_sharpness()) == pytest.approx(1.0 / (steps - 1))
    assert cliff.summary()["fraction_cliff_like"] == 1.0
    assert slide.summary()["fraction_slide_like"] == 1.0


def test_a_mixed_population_reports_as_mixed_not_as_one_branch():
    """M3 §3.1 requires mixed signatures to be reported as mixed."""
    steps = 5
    m = np.concatenate([
        np.tile(np.array([1.0, 1.0, 1.0, 0.0, 0.0]), (5, 1)),
        np.tile(np.linspace(1.0, 0.0, steps), (5, 1)),
    ])
    s = _traj(m).summary()
    assert 0.0 < s["fraction_cliff_like"] < 1.0
    assert 0.0 < s["fraction_slide_like"] < 1.0


def test_trajectories_join_by_fact_id_not_position():
    sets = []
    for load in (1.0, 2.0):
        order = RNG.permutation(N)
        sets.append(MarginSet(fact_ids=order, margins=order.astype(float) / N - load * 0.1,
                              load=load))
    tr = MarginTrajectories.from_sets(sets)
    assert list(tr.fact_ids) == sorted(tr.fact_ids)
    # fact f has margin f/N - 0.1*load at each load, by construction
    for row, f in enumerate(tr.fact_ids[:10]):
        assert tr.margins[row, 0] == pytest.approx(f / N - 0.1)
        assert tr.margins[row, 1] == pytest.approx(f / N - 0.2)


def test_duplicate_queries_for_one_fact_are_averaged_not_dropped():
    s = MarginSet(fact_ids=np.array([7, 7, 9]), margins=np.array([1.0, 0.0, 0.5]), load=1.0)
    tr = MarginTrajectories.from_sets([s])
    assert list(tr.fact_ids) == [7, 9]
    assert tr.margins[0, 0] == pytest.approx(0.5)


def test_per_class_splits_come_from_the_instrument():
    s = MarginSet(fact_ids=np.arange(N), margins=RNG.normal(size=N), load=1.0,
                  classes=np.arange(N) % 2, class_names={0: "works_with", 1: "attribute"})
    assert set(s.per_class()) == {"works_with", "attribute"}
    tr = MarginTrajectories.from_sets([s, s])
    assert set(tr.summary()["per_class"]) == {"works_with", "attribute"}
