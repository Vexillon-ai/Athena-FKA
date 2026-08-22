"""The REACHABILITY invariant (M2 §9.1-rev), tested by gradient support rather than by counting.

The load-bearing test is `test_reachability_invariant_*`: it must **pass** for content keys and
**fail** for the composed-key table that fork (a) used. A test that only passed for the new design
would prove nothing — the old design also passed §9.1's original invariant, and that is exactly why
fork (a) reached a real run with a defect it was specified to be free of (§11.3).

The invariant, stated as the property that actually matters:

    computing a held-out entity's key must touch **zero parameter rows unique to it** —
    every row it reaches must also be reached by entities the router was trained on.

Relations are exempt by ruling: a closed vocabulary fixed by the schema, so there is no held-out
relation and no unreachable row to create.
"""

from __future__ import annotations

import pytest
import torch

from fka.router.composed_keys import ComposedKeyConfig, ComposedKeyTable
from fka.router.content_keys import ContentKeyConfig, ContentKeyTable, parameter_row_support

N_E, N_R, LATENT, KEY = 24, 4, 16, 16
TRAIN, HELD = list(range(20)), 20  # entity 20 is never "trained"


def _codes(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(N_E, LATENT, generator=g), dim=-1)


def _content(mode: str = "bilinear") -> ContentKeyTable:
    return ContentKeyTable(
        ContentKeyConfig(n_relations=N_R, latent_dim=LATENT, key_dim=KEY, comp_dim=8,
                         hidden=16, encoder_hidden=16, mode=mode),
        _codes(),
    )


def _composed(mode: str = "bilinear") -> ComposedKeyTable:
    return ComposedKeyTable(ComposedKeyConfig(
        n_entities=N_E, n_relations=N_R, key_dim=KEY, comp_dim=8, hidden=16, mode=mode
    ))


def _unique_to_heldout(table) -> set:
    """Parameter rows the held-out entity reaches that no training entity reaches."""
    held = parameter_row_support(table, HELD, 0)
    trained: set = set()
    for e in TRAIN:
        trained |= parameter_row_support(table, e, 0)
    return held - trained


@pytest.mark.parametrize("mode", ["bilinear", "mlp"])
def test_reachability_invariant_holds_for_content_keys(mode):
    """Every row a held-out entity touches is a row trained entities touch too."""
    assert _unique_to_heldout(_content(mode)) == set()


@pytest.mark.parametrize("mode", ["bilinear", "mlp"])
def test_reachability_invariant_FAILS_for_the_composed_key_table(mode):
    """The test must be able to fail, and must fail on the design that actually failed.

    `ComposedKeyTable` satisfies §9.1's original count invariant. It reaches one parameter row no
    training entity ever reaches — its own embedding — and that single row is the whole 0.0%.
    """
    unique = _unique_to_heldout(_composed(mode))
    assert unique == {("entity.weight", HELD)}, (
        f"expected exactly the held-out entity's own embedding row, got {unique}"
    )


def test_parameter_count_is_independent_of_world_size():
    """`n_entities` may not appear in any parameter shape — a cheap corollary of reachability."""
    small = ContentKeyTable(
        ContentKeyConfig(n_relations=N_R, latent_dim=LATENT, key_dim=KEY, comp_dim=8,
                         hidden=16, encoder_hidden=16),
        _codes()[:8],
    )
    big = ContentKeyTable(
        ContentKeyConfig(n_relations=N_R, latent_dim=LATENT, key_dim=KEY, comp_dim=8,
                         hidden=16, encoder_hidden=16),
        _codes(),
    )
    assert small.n_params == big.n_params
    # The composed table is the contrast: three times the entities, more parameters.
    assert (
        ComposedKeyTable(ComposedKeyConfig(n_entities=8, n_relations=N_R, key_dim=KEY,
                                           comp_dim=8, hidden=16)).n_params
        < ComposedKeyTable(ComposedKeyConfig(n_entities=N_E, n_relations=N_R, key_dim=KEY,
                                             comp_dim=8, hidden=16)).n_params
    )


def test_codes_are_a_frozen_buffer_not_a_parameter():
    """A learnable codebook would let a per-entity address back in through the back door."""
    table = _content()
    assert "codes" not in dict(table.named_parameters())
    assert not table.codes.requires_grad
    names = dict(table.named_buffers())
    assert "codes" in names


def test_relation_rows_are_reachable_only_from_their_own_relation_and_that_is_allowed():
    """The relation table IS id-indexed: intended, not overlooked (closed vocabulary)."""
    table = _content()
    r0 = parameter_row_support(table, TRAIN[0], 0)
    r1 = parameter_row_support(table, TRAIN[0], 1)
    assert ("relation.weight", 0) in r0 and ("relation.weight", 0) not in r1


def test_keys_from_codes_need_no_entity_id_at_all():
    """The deployment signature: an entity the table has never indexed still gets a key."""
    table = _content()
    novel = torch.nn.functional.normalize(torch.randn(3, LATENT), dim=-1)
    keys = table.keys_for_codes(novel, torch.tensor([0, 1, 2]))
    assert keys.shape == (3, KEY)
    assert torch.allclose(keys.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_forward_by_id_matches_forward_by_code():
    table = _content()
    ids, rels = torch.tensor([1, 5, 9]), torch.tensor([0, 2, 3])
    assert torch.allclose(table(ids, rels), table.keys_for_codes(table.codes[ids], rels))


def test_all_keys_is_in_corpus_fact_id_order():
    table = _content()
    keys = table.all_keys()
    assert keys.shape == (N_E * N_R, KEY)
    for fact_id in (0, 7, N_E + 3, 3 * N_E + 11):
        e, r = fact_id % N_E, fact_id // N_E
        assert torch.allclose(
            keys[fact_id], table(torch.tensor([e]), torch.tensor([r]))[0], atol=1e-6
        )
