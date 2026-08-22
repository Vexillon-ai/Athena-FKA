"""Reachability on the VALUE path, verified RED against the old independent-value build (M3 §16.2).

The invariant, restated for values: **the marginal cost of one more fact must be bounded by
`log2(|value space|)`, not by the latent dimension** — and reconstruction of a value that is in a
table must be *exact*.

Both tests below run against **both** designs. A test that only passed for the redesign would prove
nothing; the old build is the red case and it must fail for the stated reason.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from fka.store.pointer_values import PointerValueStore
from fka.store.s1_factorized import S1Config, S1FactorizedStore

DIM = 64
SPACES = {"birth_year": 100, "birth_city": 512, "employer": 1024, "works_with": 2000}


def _tables(seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {n: F.normalize(torch.randn(k, DIM, generator=g), dim=-1) for n, k in SPACES.items()}


def _corpus_values(tables, per_space: int = 500, seed: int = 1) -> torch.Tensor:
    """Values drawn FROM the tables, as the corpus's own values are."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for t in tables.values():
        idx = torch.randint(0, t.shape[0], (per_space,), generator=g)
        out.append(t[idx])
    return torch.cat(out)


def _pointer_store(tables) -> PointerValueStore:
    return PointerValueStore(tables, latent_dim=DIM,
                             shared_tables_are_free=frozenset({"works_with"}))


def _old_store() -> S1FactorizedStore:
    """The independent-value build measured at §14's knee: 4 stages, K=256, no residual."""
    return S1FactorizedStore(S1Config(latent_dim=DIM, n_stages=4, codebook_size=256,
                                      residual_dim=0))


# -- the verified-red pair -----------------------------------------------------------------


def test_marginal_cost_is_bounded_by_the_value_space_and_the_OLD_BUILD_FAILS_THAT():
    """The invariant, and the design it was written against.

    The old build spends its full per-slot budget regardless of how small the vocabulary is; the
    redesign spends the pointer width. That difference is the whole of §14.3's named fix.
    """
    tables = _tables()
    values = _corpus_values(tables)
    bound = math.log2(max(SPACES.values())) + math.log2(len(SPACES))

    new = _pointer_store(tables)
    new.write(values)
    assert new.marginal_storage_bits <= bound, "redesign must respect the vocabulary bound"

    old = _old_store()
    old.write(values)
    old_marginal = old.cost_model()["per_fact_storage_bits"]
    assert old_marginal > bound, (
        f"the RED case must fail the bound it was written for: old {old_marginal} vs {bound}"
    )
    assert new.marginal_storage_bits < old_marginal


def test_reconstruction_is_EXACT_for_pointers_and_LOSSY_for_the_old_build():
    tables = _tables()
    values = _corpus_values(tables)
    ids = torch.arange(len(values))

    new = _pointer_store(tables)
    new.write(values)
    assert float(new.recon_error(ids).max()) < 1e-5, "a pointer into a table must be exact"

    old = _old_store()
    old.write(values)
    assert float(old.recon_error(ids).mean()) > 0.05, "the RED case must actually be lossy"


# -- structural sharing --------------------------------------------------------------------


def test_entity_valued_relations_cost_a_pointer_and_no_new_table():
    """`works_with` returns codes the key path already stores; its table is not charged twice."""
    tables = _tables()
    store = _pointer_store(tables)
    store.write(_corpus_values(tables))
    charged = store.cost_model()["shared_parameters"]
    uncharged = PointerValueStore(tables, latent_dim=DIM).cost_model()["shared_parameters"]
    assert charged < uncharged
    assert uncharged - charged == SPACES["works_with"] * DIM
    assert "works_with" in store.cost_model()["per_fact_detail"]["tables_not_charged"]


def test_the_table_selector_is_charged_not_derived():
    """It could be derived from the slot id; charging it is the conservative reading (M3 §16)."""
    tables = _tables()
    store = _pointer_store(tables)
    store.write(_corpus_values(tables))
    assert store.selector_bits == pytest.approx(math.log2(len(SPACES)))
    assert store.marginal_storage_bits > 0
    detail = store.cost_model()["per_fact_detail"]
    assert detail["selector_bits"] == pytest.approx(2.0)


def test_marginal_cost_reflects_the_mix_actually_written_not_an_unweighted_average():
    tables = _tables()
    only_small = tables["birth_year"][:50]
    store = _pointer_store(tables)
    store.write(only_small)
    expected = math.log2(SPACES["birth_year"]) + math.log2(len(SPACES))
    assert store.marginal_storage_bits == pytest.approx(expected, abs=1e-6)


def test_a_value_outside_every_table_degrades_rather_than_lying():
    tables = _tables()
    store = _pointer_store(tables)
    novel = F.normalize(torch.randn(8, DIM, generator=torch.Generator().manual_seed(9)), dim=-1)
    ids = store.write(novel)
    assert float(store.recon_error(ids).mean()) > 0.1, "an out-of-vocabulary value must show error"


def test_shared_tables_are_never_modified_by_a_write():
    tables = _tables()
    store = _pointer_store(tables)
    before = {k: v.clone() for k, v in store.tables.items()}
    store.write(_corpus_values(tables))
    assert all(torch.equal(before[k], store.tables[k]) for k in before)
