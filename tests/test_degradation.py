"""Gold stub + VERIFIED RED for the M3 §3.1 shape instrument, before it reports anything.

The load-bearing test is `test_the_MEAN_cannot_distinguish_what_the_instrument_can`: two synthetic
populations with **identical means** and opposite shapes. If the instrument could not separate them
it would be decoration, and the compression sweep would be asking a question its readout cannot
answer — which is precisely why M3 §3.1 required this instrument before the sweep.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from fka.eval.degradation import (
    QualityDistribution,
    classify_shape,
    per_fact_quality,
    transition_width,
)
from fka.store.base import IdentityStore
from fka.store.s1_factorized import S1Config, S1FactorizedStore

DIM, N = 64, 128


def _content(n: int = N, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, DIM, generator=g), dim=-1)


def _bimodal(frac_clean: float, n: int = 1000) -> QualityDistribution:
    """All-or-nothing: the attractor-basin signature."""
    k = int(frac_clean * n)
    q = np.concatenate([np.ones(k), np.zeros(n - k)])
    return QualityDistribution(fact_ids=np.arange(n), quality=q)


def _spread(mean: float, n: int = 1000) -> QualityDistribution:
    """Everything partially degraded: the superposition signature."""
    return QualityDistribution(fact_ids=np.arange(n), quality=np.full(n, mean))


# -- the discriminating test ---------------------------------------------------------------


def test_the_MEAN_cannot_distinguish_what_the_instrument_can():
    """Identical means, opposite shapes. The mean is blind; the instrument is not."""
    cliff = _bimodal(0.5)
    smooth = _spread(0.5)

    assert cliff.mean == pytest.approx(smooth.mean), "the premise: means must match exactly"
    assert cliff.bimodal_fraction == 0.0
    assert smooth.bimodal_fraction == 1.0
    assert cliff.clean_fraction == 0.5 and cliff.lost_fraction == 0.5
    assert smooth.clean_fraction == 0.0 and smooth.lost_fraction == 0.0


def test_gold_stub_a_perfect_store_is_all_clean_and_not_intermediate():
    """The instrument gate: a lossless store must read 1.0 with an empty intermediate band."""
    store = IdentityStore()
    ids = store.write(_content())
    for kind in ("reconstruction", "addressing"):
        d = per_fact_quality(store, ids, kind=kind)
        assert d.mean > 0.999, kind
        assert d.bimodal_fraction == 0.0, kind
        assert d.clean_fraction == 1.0, kind


def test_verified_red_a_broken_store_is_seen_as_broken():
    """The instrument must be able to report failure, on a store built to fail.

    Quantising the residual to 2 bits with no codebook capacity destroys the codes; if the
    instrument still read "clean" it would be reporting its own construction.
    """
    store = S1FactorizedStore(
        S1Config(latent_dim=DIM, n_stages=1, codebook_size=2, residual_dim=2, residual_bits=2)
    )
    ids = store.write(_content())
    d = per_fact_quality(store, ids, kind="reconstruction")
    assert d.mean < 0.9, f"a deliberately broken store read as healthy: {d}"
    assert d.clean_fraction < 1.0


# -- the statistics ------------------------------------------------------------------------


def test_addressing_quality_is_a_normalised_rank_and_is_1_when_nothing_outranks():
    store = IdentityStore()
    ids = store.write(_content())
    d = per_fact_quality(store, ids, kind="addressing")
    assert d.quality.max() <= 1.0 and d.quality.min() >= 0.0
    assert d.mean == pytest.approx(1.0, abs=1e-6)


def test_reconstruction_and_addressing_can_disagree():
    """Separate numbers: M3's first point had 46% recon error at 100% addressability."""
    store = S1FactorizedStore(
        S1Config(latent_dim=DIM, n_stages=4, codebook_size=64, residual_dim=0)
    )
    ids = store.write(_content())
    recon = per_fact_quality(store, ids, kind="reconstruction")
    addr = per_fact_quality(store, ids, kind="addressing")
    assert addr.mean > recon.mean, (
        f"addressing {addr.mean:.3f} should survive better than reconstruction {recon.mean:.3f}"
    )


def test_per_class_splits_are_produced_by_the_instrument_not_the_caller():
    """M3 §4.1 makes them mandatory, so they belong in the instrument (CLAUDE.md convention)."""
    store = IdentityStore()
    ids = store.write(_content())
    classes = np.arange(N) % 4
    d = per_fact_quality(store, ids, classes=classes,
                         class_names={0: "birth_year", 3: "works_with"})
    assert set(d.per_class) == {"birth_year", "1", "2", "works_with"}
    assert all(v["n"] == N // 4 for v in d.per_class.values())


def test_join_is_by_fact_id_not_row_position():
    store = IdentityStore()
    ids = store.write(_content())
    ordered = per_fact_quality(store, ids, fact_ids=np.arange(N))
    perm = np.random.default_rng(0).permutation(N)
    shuffled = per_fact_quality(store, ids[perm], fact_ids=np.arange(N)[perm])
    a = dict(zip(ordered.fact_ids.tolist(), ordered.quality.tolist(), strict=True))
    b = dict(zip(shuffled.fact_ids.tolist(), shuffled.quality.tolist(), strict=True))
    assert all(a[k] == pytest.approx(b[k], abs=1e-5) for k in a)


# -- transition width ----------------------------------------------------------------------


def test_transition_width_is_narrow_for_a_cliff_and_wide_for_a_slide():
    cliff = transition_width([1, 2, 3, 4, 5], [1.0, 1.0, 0.95, 0.05, 0.0])
    slide = transition_width([1, 2, 3, 4, 5], [1.0, 0.75, 0.5, 0.25, 0.0])
    assert cliff["width"] < slide["width"]
    assert cliff["normalised_width"] < slide["normalised_width"]


def test_transition_width_reports_none_rather_than_fabricating_a_number():
    """A curve that never crosses 10% has no width; inventing one is worse than reporting none."""
    tw = transition_width([1, 2, 3], [1.0, 0.99, 0.98])
    assert tw["width"] is None and tw["normalised_width"] is None


def test_classify_shape_returns_evidence_and_never_a_bare_verdict():
    dists = [_bimodal(1.0), _bimodal(0.5), _bimodal(0.0)]
    out = classify_shape(dists, [1, 2, 3])
    assert out["peak_intermediate_fraction"] == 0.0
    assert "means" in out and "transition" in out
    assert "verdict" not in out and "shape" not in out
