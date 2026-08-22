"""R1 product-key router: shape, exactness, and the properties the science depends on."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from fka.router.product_key import ProductKeyConfig, ProductKeyRouter  # noqa: E402


def _router(n_slots=8100, topk=8, k_half=None, axes=None, seed=0):
    torch.manual_seed(seed)
    return ProductKeyRouter(
        ProductKeyConfig(n_slots=n_slots, topk=topk, topk_half=k_half, axes=axes)
    )


def test_shapes_and_slot_range():
    r = _router()
    slots, scores = r(torch.randn(16, 64))
    assert slots.shape == scores.shape == (16, 8)
    assert int(slots.min()) >= 0 and int(slots.max()) < r.cfg.n_slots


def test_scores_are_sorted_descending():
    _, scores = _router()(torch.randn(8, 64))
    assert torch.all(scores[:, :-1] >= scores[:, 1:])


def test_shortlist_search_is_exact_when_k_half_ge_k():
    """Not a benchmark — a theorem about outer sums, asserted so a refactor cannot break it.

    The top-k of `a_i + b_j` lies inside `topk_k(a) x topk_k(b)`. This is why "recall@k vs exact
    NN over the same key set" is 1.0 by construction and must never be reported as a result
    (see the module docstring and M2 section 4).
    """
    r = _router(topk=8, k_half=8)
    q = torch.randn(32, 64)
    got, _ = r(q)
    want, _ = r.exact_topk(q, k=8)
    assert torch.equal(got.sort(dim=-1).values, want.sort(dim=-1).values)


def test_shortlist_can_miss_when_k_half_is_too_small():
    """Guards the guard: exactness is a consequence of k_half >= k, not an accident of the code."""
    r = _router(topk=8, k_half=2)
    q = torch.randn(64, 64)
    got, _ = r(q)
    want, _ = r.exact_topk(q, k=8)
    overlap = (got.unsqueeze(2) == want.unsqueeze(1)).any(dim=2).float().mean()
    assert overlap < 1.0, "with k_half < k the shortlist must be able to miss; it never did"


def test_padding_slots_are_never_returned():
    """n_sub^2 exceeds n_slots whenever N is not a perfect square; those ids address nothing."""
    r = _router(n_slots=8000)  # 90x90 = 8100 grid
    slots, _ = r(torch.randn(256, 64))
    assert int(slots.max()) < 8000


def test_slot_scores_agrees_with_forward():
    """The hard-negative eval scores *named* slots; it must agree with the search path."""
    r = _router()
    q = torch.randn(16, 64)
    slots, scores = r(q)
    assert torch.allclose(r.slot_scores(q, slots), scores, atol=1e-5)


def test_asymmetric_axes_address_every_fact():
    """The 2000x4 positive control: an axis layout matching the corpus's own factorisation."""
    r = _router(n_slots=8000, axes=(2000, 4), topk=4, k_half=4)
    assert r.cfg.n_sub1 == 2000 and r.cfg.n_sub2 == 4
    slots, _ = r(torch.randn(8, 64))
    assert int(slots.max()) < 8000


def test_parameter_count_is_two_sqrt_n_vectors():
    """The reason this exists at all: 2*sqrt(N) vectors describe N slots."""
    r = _router(n_slots=1_000_000)
    assert r.cfg.n_sub1 == 1000
    assert r.n_params == 2 * 1000 * 32


def test_gradients_reach_both_codebooks():
    r = _router()
    _, scores = r(torch.randn(8, 64))
    scores.sum().backward()
    assert r.keys1.grad is not None and r.keys1.grad.abs().sum() > 0
    assert r.keys2.grad is not None and r.keys2.grad.abs().sum() > 0


def test_rejects_k_half_larger_than_the_smaller_axis():
    with pytest.raises(ValueError, match="half-keys"):
        _router(n_slots=8000, axes=(2000, 4), topk=8, k_half=8)
