"""The frozen `KnowledgeStore` contract (M3 §10), run against EVERY implementation.

Parameterised over designs deliberately: the phase's product is a comparison between S1, S2 and S3,
and a comparison is only meaningful if every design is held to the same guarantees. A new store that
cannot pass this suite has not implemented the interface, whatever its numbers say.

The two directions of the invalidation check are the load-bearing part. A store may declare anything
it likes about what a write perturbs, but:

* slots it declares **untouched** must genuinely be untouched (honesty), and
* declaring *everything* touched passes edit-locality trivially and fails usefulness.

Testing only the first would let a store pass by declaring maximal damage.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from fka.store.base import M1_LATENT_DIM, IdentityStore, KnowledgeStore
from fka.store.s1_factorized import S1Config, S1FactorizedStore

N, DIM = 256, M1_LATENT_DIM


def _content(n: int = N, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, DIM, generator=g), dim=-1)


def _identity() -> KnowledgeStore:
    return IdentityStore()


def _s1_lightest() -> KnowledgeStore:
    """residual_dim == latent_dim: exact reconstruction, the lightest point on the surface."""
    return S1FactorizedStore(S1Config(n_stages=2, codebook_size=32, residual_dim=DIM))


def _s1_compressed() -> KnowledgeStore:
    return S1FactorizedStore(S1Config(n_stages=4, codebook_size=64, residual_dim=0))


STORES = {"identity": _identity, "s1_lightest": _s1_lightest, "s1_compressed": _s1_compressed}


@pytest.fixture(params=sorted(STORES))
def store(request):
    s = STORES[request.param]()
    s.write(_content())
    return s


# -- geometry -----------------------------------------------------------------------------


def test_public_latents_are_all_in_the_frozen_m1_space(store):
    """One geometry, no per-client latent spaces (M3 §10.2). Internals may differ; this may not."""
    assert store.latent_dim == M1_LATENT_DIM
    ids = torch.arange(8)
    assert store.reconstruct(ids).shape == (8, M1_LATENT_DIM)
    assert store.target(ids).shape == (8, M1_LATENT_DIM)


def test_write_returns_ids_in_the_order_given(store):
    fresh = _content(4, seed=7)
    ids = store.write(fresh)
    assert ids.shape == (4,)
    assert torch.allclose(store.target(ids), fresh, atol=1e-6)


def test_content_of_the_wrong_width_is_refused(store):
    with pytest.raises(ValueError, match="dim"):
        store.write(torch.randn(3, DIM + 1))


# -- the two latents ------------------------------------------------------------------------


def test_target_is_the_clean_code_and_reconstruct_may_differ(store):
    """`target` is Phase 4's training signal, so it must be exact even when `reconstruct` is not."""
    ids = torch.arange(N)
    assert torch.allclose(store.target(ids), _content(), atol=1e-6)
    assert store.reconstruct(ids).shape == store.target(ids).shape


def test_recon_error_agrees_with_the_two_latents_it_summarises(store):
    ids = torch.arange(32)
    manual = (store.reconstruct(ids) - store.target(ids)).norm(dim=-1) / store.target(ids).norm(
        dim=-1
    )
    assert torch.allclose(store.recon_error(ids), manual, atol=1e-5)
    assert store.recon_error(ids).shape == (32,)


def test_a_lossless_store_has_zero_reconstruction_error():
    """The gate: with no compression, `reconstruct` IS `target`, so any later shortfall is
    compression rather than plumbing."""
    for name in ("identity", "s1_lightest"):
        s = STORES[name]()
        s.write(_content())
        assert float(s.recon_error(torch.arange(N)).max()) < 1e-4, name


def test_a_compressed_store_actually_loses_something():
    """Otherwise the degradation surface has no axis to move along and the suite proves nothing."""
    s = _s1_compressed()
    s.write(_content())
    assert float(s.recon_error(torch.arange(N)).mean()) > 0.01


# -- scoring ------------------------------------------------------------------------------


def test_score_is_an_energy_lower_is_better(store):
    """Sign is part of the contract: three clients compare these values (M3 §10.1)."""
    ids = torch.arange(N)
    latent = store.reconstruct(ids[:16])
    energies = store.score(latent, ids)
    own = energies[torch.arange(16), torch.arange(16)]
    best = energies.min(dim=1).values
    assert bool((own <= best + 1e-5).all()), "a slot must score best on itself"
    assert float(own.mean()) < 0


def test_score_is_batched_and_broadcasts_a_shared_slot_list(store):
    latent = torch.randn(5, DIM)
    assert store.score(latent, torch.arange(9)).shape == (5, 9)
    per_row = torch.arange(9).unsqueeze(0).expand(5, -1)
    assert torch.allclose(store.score(latent, torch.arange(9)), store.score(latent, per_row))


# -- deletion -----------------------------------------------------------------------------


def test_reading_a_deleted_slot_is_an_error_not_a_zero_vector(store):
    """A zero vector would flow silently into a key and read as a bad address."""
    store.delete(torch.tensor([3]))
    with pytest.raises(KeyError):
        store.reconstruct(torch.tensor([3]))


# -- the declared invalidation model, both directions --------------------------------------


def test_slots_declared_untouched_are_genuinely_untouched(store):
    """Honesty of the declaration. This is the edit-locality instrument in test form."""
    ids = torch.arange(N)
    before = store.reconstruct(ids).clone()

    new_ids = store.write(_content(8, seed=3))
    declared = set(store.declared_invalidation(new_ids).tolist())
    untouched = torch.tensor([i for i in range(N) if i not in declared])

    after = store.reconstruct(untouched)
    base = before[untouched]
    drift = (after - base).norm(dim=-1) / base.norm(dim=-1).clamp(min=1e-9)
    assert float(drift.max()) < 0.01, "a slot declared untouched moved by more than 1%"


def test_declaring_everything_would_fail_the_usefulness_side(store):
    """The other direction: a store cannot buy locality by declaring maximal damage."""
    new_ids = store.write(_content(4, seed=5))
    declared = store.declared_invalidation(new_ids)
    assert len(declared) <= len(new_ids) * 4, (
        "a write declaring wide invalidation passes locality trivially; that is not a pass"
    )


def test_cost_model_declares_what_a_write_touches(store):
    cm = store.cost_model()
    assert {"design", "n_slots", "shared_parameters", "write_touches"} <= set(cm)
    assert isinstance(cm["write_touches"], str) and cm["write_touches"]


# -- the honesty gate, made RED-ABLE (CLAUDE.md pattern 2a) ---------------------------------


class _DishonestStore(IdentityStore):
    """Perturbs every existing slot on write, and declares only the new ones.

    Exactly the store the honesty direction exists to reject: locality by declaration rather than
    by construction. 2% drift is under no test's eye except this one's, and is the realistic shape
    of the defect — a refit that nudges neighbours, not one that destroys them.
    """

    def write(self, content: torch.Tensor) -> torch.Tensor:
        if self._codes is not None and self._codes.shape[0]:
            g = torch.Generator().manual_seed(11)
            noise = torch.randn(self._codes.shape, generator=g)
            self._codes = self._codes + 0.02 * noise * self._codes.norm(dim=-1, keepdim=True)
        return super().write(content)


def test_honesty_gate_is_red_against_a_store_that_damages_undeclared_slots():
    """Audit finding (2026-08-02): this gate was decorative until this case existed.

    None of the three registered designs perturbs an existing slot on write, so
    ``test_slots_declared_untouched_are_genuinely_untouched`` was green **by construction** and had
    never been in a position to fail. Per CLAUDE.md pattern 2a a gate is red-able only when shown
    red against the defect it names, under conditions that let the defect express itself — here,
    a store whose write actually moves undeclared slots.
    """
    store = _DishonestStore()
    store.write(_content())
    ids = torch.arange(N)
    before = store.reconstruct(ids).clone()

    new_ids = store.write(_content(8, seed=3))
    declared = set(store.declared_invalidation(new_ids).tolist())
    untouched = torch.tensor([i for i in range(N) if i not in declared])

    after = store.reconstruct(untouched)
    base = before[untouched]
    drift = (after - base).norm(dim=-1) / base.norm(dim=-1).clamp(min=1e-9)
    assert float(drift.max()) >= 0.01, (
        "the dishonest store went undetected — the honesty gate cannot see the defect it names"
    )
