"""The separability index must be shown to discriminate before it is ever reported (M2 §9.8).

Per the validated-statistics pattern (CLAUDE.md): a reported statistic ships with a test proving it
separates the cases it claims to separate, or it is decoration.

The two constructions here are not invented for the test — they are **the same pair that decided
Stage A**, where a product-key fit reached 100.0% on concatenative keys and 16.6% on `e ⊙ r` keys
under an identical fit. So the statistic is validated against a case whose answer is already known
by an independent route.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from fka.eval.separability import additive_surrogate, entity_recovery, separability

#: The frozen M1 interface width. Testing at some other dim would measure a different instrument:
#: the surrogate's quality depends on it, and 64 is the one that will be reported against.
N_E, N_R, DIM = 400, 4, 64


def _fact_ids() -> np.ndarray:
    return np.arange(N_E * N_R, dtype=np.int64)


def _concatenative_keys(seed: int = 0) -> torch.Tensor:
    """`[a_e ; b_r]` with unit-norm halves — separable by construction.

    Constant total norm (√2 for every key), so normalisation is a global scale and the key set is
    *exactly* additive rather than approximately so. Anything below 1.0 here is the instrument.
    """
    g = torch.Generator().manual_seed(seed)
    a = F.normalize(torch.randn(N_E, DIM // 2, generator=g), dim=-1)
    b = F.normalize(torch.randn(N_R, DIM // 2, generator=g), dim=-1)
    keys = torch.empty(N_R * N_E, DIM)
    for r in range(N_R):
        for e in range(N_E):
            keys[r * N_E + e] = torch.cat([a[e], b[r]])
    return F.normalize(keys, dim=-1)


def _hadamard_keys(seed: int = 0, n_e: int = N_E) -> torch.Tensor:
    """`normalize(entity_code ⊙ relation_code)` — the oracle binding Stage A retired."""
    g = torch.Generator().manual_seed(seed)
    e_code = F.normalize(torch.randn(n_e, DIM, generator=g), dim=-1)
    r_code = F.normalize(torch.randn(N_R, DIM, generator=g), dim=-1)
    return F.normalize(
        (e_code.unsqueeze(0) * r_code.unsqueeze(1)).reshape(N_R * n_e, DIM), dim=-1
    )


def test_surrogate_is_exact_on_an_additive_key_set():
    """The closed form must reproduce a set that is already in the family, to numerical noise."""
    g = torch.Generator().manual_seed(1)
    a = torch.randn(N_E, DIM, generator=g)
    b = torch.randn(N_R, DIM, generator=g)
    keys = torch.stack([a[e] + b[r] for r in range(N_R) for e in range(N_E)])
    assert torch.allclose(additive_surrogate(keys, N_E, N_R), keys, atol=1e-5)


def test_surrogate_rejects_an_incomplete_grid():
    with pytest.raises(ValueError, match="complete grid"):
        additive_surrogate(torch.randn(N_E * N_R - 1, DIM), N_E, N_R)


def test_separable_keys_score_high_and_bound_keys_score_low():
    """THE load-bearing assertion: the statistic must tell the two geometries apart.

    Queries are the true keys themselves, so retrieval under the learned keys is perfect in both
    cases and the *only* thing that can move the index is the surrogate.
    """
    ids = _fact_ids()

    concat = _concatenative_keys()
    sep = separability(concat, concat, ids, N_E, N_R)
    assert sep.reference_recall_at_1 == 1.0
    assert sep.separability_index > 0.99, "an exactly-additive key set must survive its surrogate"
    assert sep.residual_fraction < 0.01

    hadamard = _hadamard_keys()
    bound = separability(hadamard, hadamard, ids, N_E, N_R)
    assert bound.reference_recall_at_1 == 1.0
    assert bound.separability_index < 0.55, "multiplicative binding must NOT survive its surrogate"
    assert bound.residual_fraction > 0.5

    # The gap is the discrimination, and it has to be large enough to read through run-to-run
    # noise rather than merely being in the right direction.
    assert sep.separability_index - bound.separability_index > 0.4


def test_the_multiplicative_floor_is_not_zero_and_depends_on_corpus_size():
    """Calibration, without which the reported number cannot be read.

    A fully multiplicative key set does **not** score ≈0: its additive surrogate still recovers the
    *entity*, so the index sits well above chance while carrying no binding at all. The floor is
    therefore a function of the corpus, and reading a joint-fit result against an assumed floor of
    zero would overstate its separability by exactly this amount.

    Pinned here so the floor is a measured reference rather than an intuition, and asserted only as
    an ordering — the corpus-size dependence is the claim, not the specific values.
    """
    floors = {}
    for n_e in (40, 400, 2000):
        keys = _hadamard_keys(n_e=n_e)
        ids = np.arange(n_e * N_R, dtype=np.int64)
        floors[n_e] = separability(keys, keys, ids, n_e, N_R).separability_index

    assert all(f > 0.15 for f in floors.values()), f"floor is not near zero: {floors}"
    assert floors[40] > floors[400] > floors[2000], (
        f"the floor must fall as the address space grows: {floors}"
    )


def test_bound_keys_fail_the_surrogate_by_going_relation_blind():
    """*How* it fails is part of the claim: the fit keeps the entity and loses the relation.

    An additive surrogate of `e ⊙ r` averages a key over the relations, which retains the entity
    and destroys the binding — so its errors should land overwhelmingly on the same entity. If the
    index dropped for some other reason, this assertion is what notices.
    """
    ids = _fact_ids()
    keys = _hadamard_keys()
    surrogate = additive_surrogate(keys, N_E, N_R)
    got = (F.normalize(keys, dim=-1) @ F.normalize(surrogate, dim=-1).T).argmax(dim=-1).numpy()

    wrong = got != ids
    assert wrong.any()
    same_entity = (got[wrong] % N_E) == (ids[wrong] % N_E)
    chance = (N_R - 1) / (N_E * N_R - 1)
    assert same_entity.mean() > 0.7
    # Stated as enrichment because the absolute fraction moves with dim and corpus size, while
    # "the errors are relation-blind" is the claim and it survives both.
    assert same_entity.mean() / chance > 20


def test_relative_index_is_conditioned_on_the_key_sets_own_recall():
    """A surrogate cannot beat the keys it approximates, so the raw index is ceilinged.

    Same failure shape as the follow-rate inversion (M1 §1): reporting only the raw number would
    understate separability whenever the probe set is hard.
    """
    ids = _fact_ids()
    keys = _concatenative_keys()
    noisy = keys + 0.9 * torch.randn(keys.shape, generator=torch.Generator().manual_seed(3))
    res = separability(keys, F.normalize(noisy, dim=-1), ids, N_E, N_R)
    assert res.reference_recall_at_1 < 1.0
    assert res.relative_index >= res.separability_index


def test_join_is_by_fact_id_not_row_position():
    ids = _fact_ids()
    keys = _concatenative_keys()
    perm = np.random.default_rng(0).permutation(len(ids))
    shuffled = separability(keys, keys[perm], ids[perm], N_E, N_R)
    assert shuffled.separability_index == pytest.approx(
        separability(keys, keys, ids, N_E, N_R).separability_index
    )


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="against"):
        separability(_concatenative_keys(), torch.randn(5, DIM), _fact_ids(), N_E, N_R)


# ---------------------------------------------------------------------------------------
# Entity recovery — the two-stage entity-first hypothesis (M2 §10.1)
# ---------------------------------------------------------------------------------------


def test_entity_recovery_is_high_where_slot_recall_collapses():
    """The hypothesis in one assertion, on the key set that motivated it.

    A Hadamard key set scores far below 1.0 on the separability index because the surrogate goes
    relation-blind. Entity recovery is the complement of that failure: the entity survives even
    though the relation does not. If both numbers were low the two-stage scheme would be dead, and
    this test is what would say so.
    """
    ids = _fact_ids()
    keys = _hadamard_keys()
    slot_level = separability(keys, keys, ids, N_E, N_R).separability_index
    ent = entity_recovery(keys, keys, ids, N_E, N_R)

    assert ent.entity_recovery_at_1 > slot_level + 0.3, (
        f"entity recovery {ent.entity_recovery_at_1:.3f} must clearly exceed slot-level "
        f"separability {slot_level:.3f} — otherwise there is nothing to shortlist with"
    )
    assert ent.curve[8] >= ent.curve[1], "the curve must be monotone in m"


def test_entity_recovery_curve_is_monotone_and_reaches_one():
    ids = _fact_ids()
    ent = entity_recovery(_hadamard_keys(), _hadamard_keys(), ids, N_E, N_R,
                          ms=(1, 2, 4, 8, 16, 32, 64, 128, 256, N_E))
    values = [ent.curve[m] for m in sorted(ent.curve)]
    assert values == sorted(values)
    assert ent.curve[N_E] == 1.0, "every entity is in the top-n_entities by construction"


def test_candidate_recall_equals_entity_recovery_and_states_its_cost():
    ids = _fact_ids()
    ent = entity_recovery(_hadamard_keys(), _hadamard_keys(), ids, N_E, N_R)
    assert ent.candidate_recall(4) == ent.curve[4]
    assert ent.to_dict()["candidate_set_sizes"]["4"] == 4 * N_R


def test_entity_recovery_falls_to_chance_when_the_QUERIES_carry_nothing():
    """The negative control, and the first version of it was wrong in an instructive way.

    Randomising the *keys* does NOT drive entity recovery down: with queries drawn from the key set,
    the surrogate's entity mean `ā_e = (1/n_r) Σ_r k(e, r)` literally contains the query, so the
    true entity wins on any key set whatsoever (measured: 92.3% on uniform random keys). The
    quantity is a property of the query-key alignment, not of the keys alone.

    So the control that can actually fail is randomising the **queries**. This also states the
    caveat that travels with every reported figure: entity recovery is high only where `q·k(e, r)`
    is already large, so it must be read next to slot-level recall, never instead of it.
    """
    g = torch.Generator().manual_seed(7)
    keys = _hadamard_keys()
    queries = F.normalize(torch.randn(N_E * N_R, DIM, generator=g), dim=-1)
    ent = entity_recovery(keys, queries, _fact_ids(), N_E, N_R)
    assert ent.entity_recovery_at_1 < 0.02, "uninformative queries must give chance-level recovery"
    assert ent.curve[64] < 0.25, "and the whole curve must stay near chance, not just m=1"
