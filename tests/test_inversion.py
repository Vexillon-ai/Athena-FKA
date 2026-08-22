"""Gates for the §10.2 query-inversion search path, before it reports any number."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from fka.router.inversion import (
    EntityInverter,
    GoldStubInverter,
    InverterConfig,
    entity_recovery_by_inversion,
    fit_inverter,
)

N_E, N_R, DIM = 200, 4, 32


def _codebook(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(N_E, DIM, generator=g), dim=-1)


def test_gold_stub_recovers_every_entity_at_m_1():
    """The instrument gate. If this is not exactly 1.0, no `h` number means anything."""
    codes = _codebook()
    ents = np.arange(N_E)
    stub = GoldStubInverter(codes, torch.from_numpy(ents))
    res = entity_recovery_by_inversion(stub, torch.randn(N_E, DIM), ents, codes, N_R)
    assert res.at_1 == 1.0
    assert res.m_for(0.99) == 1


def test_recovery_falls_to_chance_for_an_uninformative_map():
    """A map carrying nothing about the query must not produce a usable shortlist."""

    class Constant(torch.nn.Module):
        def forward(self, q):
            return F.normalize(torch.ones(len(q), DIM), dim=-1)

    codes = _codebook()
    res = entity_recovery_by_inversion(
        Constant(), torch.randn(N_E, DIM), np.arange(N_E), codes, N_R
    )
    assert res.at_1 < 0.05
    assert res.m_for(0.99) is None or res.m_for(0.99) > 64


def test_curve_is_monotone_and_candidate_cost_is_reported():
    codes = _codebook()
    stub = GoldStubInverter(codes, torch.from_numpy(np.arange(N_E)))
    res = entity_recovery_by_inversion(stub, torch.randn(N_E, DIM), np.arange(N_E), codes, N_R)
    values = [res.curve[m] for m in sorted(res.curve)]
    assert values == sorted(values)
    assert res.to_dict()["candidate_slots"]["8"] == 8 * N_R


def test_join_is_by_value_not_row_position():
    codes = _codebook()
    ents = np.arange(N_E)
    perm = np.random.default_rng(0).permutation(N_E)
    q = torch.randn(N_E, DIM)

    class ByContent(torch.nn.Module):
        """Answers from the query's content, so a permuted batch must score identically."""

        def forward(self, x):
            return F.normalize(codes[x[:, 0].round().long() % N_E], dim=-1)

    probe = torch.zeros(N_E, DIM)
    probe[:, 0] = torch.from_numpy(ents).float()
    a = entity_recovery_by_inversion(ByContent(), probe, ents, codes, N_R)
    b = entity_recovery_by_inversion(ByContent(), probe[perm], ents[perm], codes, N_R)
    assert a.at_1 == b.at_1 == 1.0
    del q


def test_a_learned_inverter_can_actually_invert_hadamard_binding():
    """Sanity that the fit works at all, on the exact structure the real probe faces.

    Queries are `normalize(e_code ⊙ r_code)` — the oracle's own addresses — and `h` must recover the
    entity code **without being told the relation**. Fitted on 80% of entities and scored on the
    held-out 20%: a generalisation check in miniature, not a memorisation one.

    **What this asserts is the CONTRAST, not a threshold**, and the reason is a real finding.
    Inversion generalises only with enough training entities to pin the map: at 1,600 training
    entities the probe measures **99.8%** held-out recovery
    (`experiments/2026-08-02_m2-searchability/`), while at the 160 this cheap test can afford, the
    MLP drives training cosine loss to ~0.008 and still recovers **7.5%** held out — it memorised.

    So the assertion is that the evaluator **can tell those apart**. If it could not, a later
    memorising inverter would report a healthy number and nobody would know. Asserting a weak
    version of the generalisation claim here instead would be a probe that cannot reach the regime
    it is claiming to test, which is its own documented failure mode.
    """
    g = torch.Generator().manual_seed(1)
    dim = 64
    codes = F.normalize(torch.randn(N_E, dim, generator=g), dim=-1)
    rel = F.normalize(torch.randn(N_R, dim, generator=g), dim=-1)
    ents = np.repeat(np.arange(N_E), N_R)
    rels = np.tile(np.arange(N_R), N_E)
    queries = F.normalize(codes[ents] * rel[rels], dim=-1)

    train = ents < int(0.8 * N_E)
    inverter = EntityInverter(InverterConfig(latent_dim=dim, hidden=256, n_layers=2))
    losses = fit_inverter(
        inverter, queries[train], codes[ents[train]], steps=400, lr=3e-3, log_every=10**9
    )

    fitted = entity_recovery_by_inversion(inverter, queries[train], ents[train], codes, N_R)
    held = entity_recovery_by_inversion(inverter, queries[~train], ents[~train], codes, N_R)

    assert losses[-1] < 0.1, "the fit itself must work — otherwise this measures nothing"
    assert fitted.at_1 > 0.9, f"training-set recovery should be near perfect, got {fitted.at_1:.1%}"
    assert held.at_1 < fitted.at_1 - 0.3, (
        "at this entity count the inverter memorises; the evaluator must show the gap "
        f"(train {fitted.at_1:.1%}, held out {held.at_1:.1%})"
    )


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="against"):
        entity_recovery_by_inversion(
            GoldStubInverter(_codebook(), torch.arange(N_E)),
            torch.randn(5, DIM), np.arange(N_E), _codebook(), N_R,
        )
