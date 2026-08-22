"""Fork (a)'s architecture constraint, asserted rather than trusted (M2 section 9.1)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from fka.router.composed_keys import (  # noqa: E402
    ComposedKeyConfig,
    ComposedKeyTable,
    key_spread,
)


def _table(mode="mlp", n_e=2000, n_r=4):
    torch.manual_seed(0)
    return ComposedKeyTable(ComposedKeyConfig(n_entities=n_e, n_relations=n_r, mode=mode))


@pytest.mark.parametrize("mode", ["mlp", "bilinear"])
def test_parameters_scale_with_factors_not_with_pairs(mode):
    """THE constraint. Doubling the entity count must not scale with entities x relations.

    A free per-(e, r) key table would grow by n_relations x the embedding for each new entity;
    a composed table grows by exactly one embedding row.
    """
    small, big = _table(mode, n_e=1000), _table(mode, n_e=2000)
    growth = big.n_params - small.n_params
    assert growth == 1000 * small.cfg.comp_dim, (
        "key parameters do not grow as O(n_entities); a per-pair table has crept in, which makes "
        "the entity holdout unpassable in principle (M2 section 9.1)"
    )


def test_no_parameter_tensor_is_indexed_by_fact():
    """Guards the guard: no parameter may have n_entities * n_relations rows."""
    t = _table()
    n_facts = t.cfg.n_entities * t.cfg.n_relations
    for name, p in t.named_parameters():
        assert n_facts not in tuple(p.shape), f"{name} has a per-fact dimension"


@pytest.mark.parametrize("mode", ["mlp", "bilinear"])
def test_keys_for_unseen_entities_are_computable(mode):
    """The property the holdout tests: an entity never optimised still yields a finite key.

    This is what a free per-pair table cannot do, and it is why the constraint exists.
    """
    t = _table(mode)
    k = t(torch.tensor([1999, 1998]), torch.tensor([0, 3]))
    assert k.shape == (2, t.cfg.key_dim)
    assert torch.isfinite(k).all()


def test_key_depends_on_both_factors():
    """A composition that ignores a factor is failure mode 9.3(a) in its starkest form."""
    t = _table()
    base = t(torch.tensor([5]), torch.tensor([0]))
    other_entity = t(torch.tensor([6]), torch.tensor([0]))
    other_relation = t(torch.tensor([5]), torch.tensor([1]))
    assert not torch.allclose(base, other_entity, atol=1e-6), "key ignores the entity"
    assert not torch.allclose(base, other_relation, atol=1e-6), "key ignores the relation"


def test_all_keys_follows_corpus_fact_id_order():
    """fact_id = relation_index * n_entities + entity_index -- the corpus convention."""
    t = _table(n_e=7, n_r=3)
    allk = t.all_keys()
    assert allk.shape == (21, t.cfg.key_dim)
    for r in (0, 2):
        for e in (0, 6):
            assert torch.allclose(
                allk[r * 7 + e], t(torch.tensor([e]), torch.tensor([r]))[0], atol=1e-6
            )


def test_gradients_reach_both_embedding_tables():
    t = _table()
    t(torch.tensor([1, 2]), torch.tensor([0, 1])).sum().backward()
    assert t.entity.weight.grad.abs().sum() > 0
    assert t.relation.weight.grad.abs().sum() > 0


def test_key_spread_detects_collapse():
    """The statistic must actually discriminate, or reporting it is decoration."""
    healthy = torch.randn(512, 64)
    collapsed = torch.randn(1, 64).repeat(512, 1) + 1e-4 * torch.randn(512, 64)

    h, c = key_spread(healthy), key_spread(collapsed)
    assert h["effective_rank"] > 10 * c["effective_rank"]
    assert c["mean_cosine"] > 0.99
    assert h["mean_cosine"] < 0.5
