"""Instrument gate for the router eval — written BEFORE any router trains.

M2 §2 requires this: a known-answer stub through the deployed eval path, so the first router
number ever produced is measured by an instrument already shown to catch the failure it looks for.

The decisive test here is not "the oracle stub scores 1.0". It is
`test_pooled_mean_cannot_see_what_the_binding_split_sees`: a stub that confuses `(e, r)` with
`(e, r')` — a router that found the entity and ignored the relation — must be caught by the
same-entity margin **and missed by the pooled mean**. If the pooled mean caught it too, the
per-class split would be decoration, and the gate design would not be earning its complexity.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.instrument

from fka.data.corpus_gen import CorpusConfig, generate_corpus  # noqa: E402
from fka.eval.router_eval import (  # noqa: E402
    aligned_slot_map,
    evaluate_router,
    shuffled_slot_map,
)

RELATIONS = ["birth_year", "birth_city", "employer", "works_with"]


class OracleStubRouter:
    """Returns the correct slot first, by construction, and scores it highest.

    Mimics ProductKeyRouter's surface (`__call__`, `slot_scores`) so it travels the real eval path.
    Identity comes from the query row's *content*, never its position: each probe's query is a
    one-hot-ish vector encoding its target slot, so a permuted batch still resolves correctly.
    """

    def __init__(self, n_slots: int, k: int = 8, confuse_to: np.ndarray | None = None):
        self.n_slots = n_slots
        self.k = k
        # If set, the stub answers with confuse_to[target] instead of target — a deliberate,
        # structured mistake used to prove the gate can see a specific failure mode.
        self.confuse_to = confuse_to

    def _targets(self, q: torch.Tensor) -> torch.Tensor:
        # The query's first component carries its target slot; see `_queries_for` below.
        return q[:, 0].round().long()

    def __call__(self, q: torch.Tensor):
        t = self._targets(q)
        answer = t if self.confuse_to is None else torch.from_numpy(self.confuse_to)[t]
        slots = answer.unsqueeze(1).repeat(1, self.k)
        # Pad the rest with other slots so recall@k is not trivially 1 for every slot.
        filler = (answer.unsqueeze(1) + torch.arange(1, self.k, dtype=torch.long)) % self.n_slots
        slots = torch.cat([answer.unsqueeze(1), filler], dim=1)
        scores = torch.linspace(1.0, 0.0, self.k).unsqueeze(0).repeat(len(q), 1)
        return slots, scores

    def slot_scores(self, q: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        t = self._targets(q)
        answer = t if self.confuse_to is None else torch.from_numpy(self.confuse_to)[t]
        return (slots == answer.unsqueeze(1)).float()


@pytest.fixture(scope="module")
def world():
    corpus = generate_corpus(CorpusConfig(n_entities=120, seed=0, n_coworkers=1))
    smap = shuffled_slot_map(corpus, RELATIONS, seed=0)
    rng = np.random.default_rng(0)
    fact_ids = rng.choice(corpus.n_entities * len(RELATIONS), size=64, replace=False)
    return corpus, smap, fact_ids


def _queries_for(smap, fact_ids) -> torch.Tensor:
    """A query whose first component is its target slot — content-addressed, not positional."""
    q = torch.zeros(len(fact_ids), 64)
    q[:, 0] = torch.from_numpy(smap.fact_to_slot[fact_ids]).float()
    return q


def test_oracle_stub_scores_exactly_one(world):
    """THE gate. Below 1.0 means the eval is broken, not the router."""
    corpus, smap, fact_ids = world
    res = evaluate_router(
        OracleStubRouter(smap.n_slots), _queries_for(smap, fact_ids), fact_ids, smap,
        corpus, RELATIONS,
    )
    assert res.recall_at_1 == 1.0, f"eval scored a by-construction-correct router at {res!s}"
    assert res.worst_binding_margin > 0


def test_score_is_invariant_under_probe_permutation(world):
    """Identity must not be positional — the class of defect that bit Phase 1 three times."""
    corpus, smap, fact_ids = world
    order = np.random.default_rng(5).permutation(len(fact_ids))
    shuffled = fact_ids[order]
    res = evaluate_router(
        OracleStubRouter(smap.n_slots), _queries_for(smap, shuffled), shuffled, smap,
        corpus, RELATIONS,
    )
    assert res.recall_at_1 == 1.0


def test_pooled_mean_cannot_see_what_the_binding_split_sees(world):
    """The load-bearing test: why the eval reports per-class margins instead of one number.

    A relation-blind router answers `(e, r')` for `(e, r)`. The same-entity margin must catch it.
    A pooled mean over all confusers must NOT — if it did, the per-class split would be decoration
    and M2 §2's design would be unjustified complexity.
    """
    corpus, smap, fact_ids = world
    n_e, n_r = corpus.n_entities, len(RELATIONS)

    # Build the confusion: every fact answers with a DIFFERENT relation on the same entity.
    n_facts = n_e * n_r
    confuse = np.arange(smap.n_slots, dtype=np.int64)
    for f in range(n_facts):
        e, r = f % n_e, f // n_e
        wrong = ((r + 1) % n_r) * n_e + e
        confuse[smap.fact_to_slot[f]] = smap.fact_to_slot[wrong]

    res = evaluate_router(
        OracleStubRouter(smap.n_slots, confuse_to=confuse),
        _queries_for(smap, fact_ids), fact_ids, smap, corpus, RELATIONS,
    )

    assert res.recall_at_1 == 0.0, "the corrupted stub should never return the right slot"
    assert res.margins["same-entity"].minimum < 0, (
        "the same-entity margin failed to notice a relation-blind router — the gate is blind to "
        "the exact failure it exists to detect"
    )
    # And the easy class is fooled, which is the justification for not gating on it.
    assert res.margins["unrelated"].minimum >= 0, (
        "the unrelated-confuser margin caught it too, so the per-class split is not "
        "load-bearing — revisit M2 section 2 before trusting this design"
    )


def test_aligned_and_shuffled_maps_are_both_injective(world):
    """Both experimental configurations must be valid slot assignments before either is run."""
    corpus, _, _ = world
    for smap in (aligned_slot_map(corpus, RELATIONS), shuffled_slot_map(corpus, RELATIONS)):
        assert len(set(smap.fact_to_slot.tolist())) == corpus.n_entities * len(RELATIONS)
        assert smap.fact_to_slot.max() < smap.n_slots


# ---------------------------------------------------------------------------------------
# The supervised/unsupervised address split (2026-08-02)
# ---------------------------------------------------------------------------------------


def test_unseen_split_catches_a_router_that_only_knows_supervised_addresses(world):
    """The load-bearing assertion for the split: pooled recall must NOT be able to hide it.

    This is the defect the split was added for. Fork (a)'s probe set was entity-held-out in M1's
    sense, but **85.8% of its target FACTS had been supervised as training targets**, so a router
    that had merely memorised supervised addresses scored high overall while saying nothing about
    composition. A memorising router is simulated exactly: right on supervised targets, wrong on
    the rest, in the same 85.8/14.2 proportion the real probe set had.
    """
    corpus, smap, fact_ids = world
    n_slots = smap.n_slots
    supervised = fact_ids[: int(round(0.858 * len(fact_ids)))]

    # Answer the true slot for supervised facts, a neighbouring slot for everything else.
    confuse_to = np.arange(n_slots)
    unseen = [f for f in fact_ids if f not in set(supervised.tolist())]
    for f in unseen:
        confuse_to[smap.fact_to_slot[f]] = (smap.fact_to_slot[f] + 1) % n_slots

    res = evaluate_router(
        OracleStubRouter(n_slots, confuse_to=confuse_to),
        _queries_for(smap, fact_ids), fact_ids, smap, corpus, RELATIONS,
        supervised_fact_ids=supervised,
    )

    assert res.recall_at_1 > 0.8, "pooled recall looks healthy — which is exactly the trap"
    assert res.n_unseen == len(unseen)
    assert res.recall_at_1_unseen == 0.0, "the split must expose what the pooled number hides"
    assert "UNSUPERVISED ADDRESSES" in str(res)


def test_unseen_split_is_absent_unless_asked_for(world):
    """It has to be opt-in, but its absence must be visible rather than silently zero."""
    corpus, smap, fact_ids = world
    res = evaluate_router(
        OracleStubRouter(smap.n_slots), _queries_for(smap, fact_ids), fact_ids, smap,
        corpus, RELATIONS,
    )
    assert res.recall_at_1_unseen is None and res.n_unseen == 0
    assert "UNSUPERVISED" not in str(res)


def test_unseen_split_agrees_with_pooled_recall_when_nothing_was_supervised(world):
    """Sanity: with an empty supervised set the split is the pooled number, not a new quantity."""
    corpus, smap, fact_ids = world
    res = evaluate_router(
        OracleStubRouter(smap.n_slots), _queries_for(smap, fact_ids), fact_ids, smap,
        corpus, RELATIONS, supervised_fact_ids=np.array([], dtype=np.int64),
    )
    assert res.n_unseen == len(fact_ids)
    assert res.recall_at_1_unseen == res.recall_at_1 == 1.0
