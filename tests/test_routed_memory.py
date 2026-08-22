"""The learned stack must speak the oracle's interface exactly, or the deployed evaluator lies.

`RoutedLatentMemory` exists so `evaluate_d3` can score a learned router through the *same* path
that produced M1's numbers. That only holds if slot ids really are fact ids and the value substrate
really is untouched — both asserted here, because a transposition between them would show up as a
plausible-looking accuracy rather than as an error.
"""

from __future__ import annotations

import torch

from fka.data.corpus_gen import CorpusConfig, generate_corpus
from fka.kernel.latent_memory import LatentCodebook, OracleLatentMemory
from fka.router.composed_keys import ComposedKeyConfig, ComposedKeyTable
from fka.router.routed_memory import RoutedLatentMemory, oracle_key_stub

DIM = 16


def _fixture(n_entities: int = 24):
    corpus = generate_corpus(CorpusConfig(n_entities=n_entities, seed=0, n_coworkers=1))
    oracle = OracleLatentMemory(corpus, LatentCodebook.build(corpus, dim=DIM, seed=0))
    table = ComposedKeyTable(
        ComposedKeyConfig(
            n_entities=corpus.n_entities,
            n_relations=len(corpus.relations),
            key_dim=DIM,
            comp_dim=8,
            hidden=16,
            mode="bilinear",
        )
    )
    routed = RoutedLatentMemory(oracle, table, list(corpus.relations))
    return corpus, oracle, routed


def test_slot_ids_are_fact_ids():
    """The whole no-translation-table claim in one assertion."""
    corpus, oracle, routed = _fixture()
    n_e = corpus.n_entities
    for fact_id in range(len(routed)):
        entity, relation = fact_id % n_e, list(corpus.relations)[fact_id // n_e]
        assert oracle.fact_index[(entity, relation)] == fact_id


def test_the_oracle_key_stub_reproduces_the_oracle_exactly():
    """The gate for this eval path: oracle keys in the wrapper must retrieve like the oracle.

    Driven with the oracle's own keys as queries, so retrieval is unambiguous and any disagreement
    is the wrapper. This is what would catch a fact-id/slot transposition, which no amount of
    inspecting the router could.
    """
    corpus, oracle, _ = _fixture()
    stub = oracle_key_stub(oracle, list(corpus.relations))
    queries = oracle.keys

    assert torch.equal(stub.retrieved_index(queries), oracle.retrieved_index(queries))
    want = torch.arange(len(oracle))
    assert torch.equal(stub.retrieved_index(queries), want), "stub must be exactly right, not close"
    assert torch.allclose(stub.read(queries, hard=True), oracle.read(queries, hard=True))


def test_values_come_from_the_frozen_substrate_untouched():
    """Phase 2 replaces addressing, not values. A learned value table would be a different claim."""
    _, oracle, routed = _fixture()
    assert routed.values_matrix.data_ptr() == oracle.values_matrix.data_ptr()
    assert not routed.values_matrix.requires_grad


def test_keys_are_differentiable_and_reach_the_composition():
    _, _, routed = _fixture()
    routed.read(torch.randn(4, DIM)).sum().backward()
    assert routed.table.entity.weight.grad is not None
    assert routed.table.entity.weight.grad.abs().sum() > 0


def test_freezing_snapshots_keys_and_thawing_restores_the_live_path():
    """A stale cache during training would silently score a router that no longer exists."""
    _, _, routed = _fixture()
    routed.freeze_keys()
    frozen = routed.keys.clone()
    with torch.no_grad():
        routed.table.entity.weight.add_(1.0)
    assert torch.equal(routed.keys, frozen), "frozen keys must not follow the table"
    routed.thaw_keys()
    assert not torch.equal(routed.keys, frozen), "thawed keys must follow the table again"


def test_hard_read_matches_a_manual_argmax_over_the_learned_keys():
    _, _, routed = _fixture()
    q = torch.randn(8, DIM)
    idx = torch.nn.functional.normalize(q, dim=-1) @ routed.keys.T
    assert torch.allclose(routed.read(q, hard=True), routed.values_matrix[idx.argmax(-1)])


def test_cached_keys_computes_once_and_still_accumulates_gradient():
    """The per-step cache must be an optimisation, never a change of answer."""
    _, _, routed = _fixture()
    q = torch.randn(4, DIM)

    routed.table.zero_grad()
    with routed.cached_keys():
        calls = routed.keys
        (routed.read(q).sum() + routed.read(q).sum()).backward()
        assert routed.keys.data_ptr() == calls.data_ptr(), "cache must return the same tensor"
    cached_grad = routed.table.entity.weight.grad.clone()

    routed.table.zero_grad()
    (routed.read(q).sum() + routed.read(q).sum()).backward()
    assert torch.allclose(cached_grad, routed.table.entity.weight.grad, atol=1e-6)
    assert routed._step_keys is None, "the cache must not outlive its context"
