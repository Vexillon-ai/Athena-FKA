"""D3 latent interface: compositional addressing, frozen codebook, iterative retrieval.

The load-bearing property is that addresses are *composed* rather than enumerated. If
``key(entity, relation)`` were an independent random vector per fact, the kernel would need one
learned mapping per fact — storing the corpus in its own weights — and an entity-level holdout
could not be passed even in principle.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from fka.data.corpus_gen import generate_corpus  # noqa: E402
from fka.kernel.latent_kernel import LatentKernelConfig, LatentReasoningKernel  # noqa: E402
from fka.kernel.latent_memory import LatentCodebook, OracleLatentMemory  # noqa: E402


@pytest.fixture(scope="module")
def corpus():
    return generate_corpus(n_entities=40, seed=0, n_coworkers=1)


@pytest.fixture(scope="module")
def memory(corpus):
    return OracleLatentMemory(corpus, LatentCodebook.build(corpus, dim=64, seed=0))


def test_codebook_is_deterministic_and_normalised(corpus):
    a = LatentCodebook.build(corpus, dim=64, seed=0)
    b = LatentCodebook.build(corpus, dim=64, seed=0)
    assert torch.allclose(a.entity, b.entity)
    assert torch.allclose(a.entity.norm(dim=-1), torch.ones(corpus.n_entities), atol=1e-5)
    assert not torch.allclose(a.entity, LatentCodebook.build(corpus, dim=64, seed=1).entity)


def test_addresses_are_composed_not_enumerated(memory, corpus):
    """key(e, r) must be a function of the entity and relation codes, so binding generalises."""
    cb = memory.codebook
    for entity_id in (0, 7, 39):
        for relation in ("birth_year", "birth_city"):
            expected = cb.bind(cb.entity[entity_id], relation)
            stored = memory.keys[memory.fact_index[(entity_id, relation)]]
            assert torch.allclose(expected, stored, atol=1e-5)


def test_binding_the_same_entity_with_different_relations_gives_distinct_addresses(memory):
    cb = memory.codebook
    a = cb.bind(cb.entity[3], "birth_year")
    b = cb.bind(cb.entity[3], "birth_city")
    assert abs(float(a @ b)) < 0.5, "relation must actually change the address"


def test_exact_address_retrieves_the_right_value(memory, corpus):
    """Hard read with the true key must return exactly the stored value latent."""
    cb = memory.codebook
    for entity_id in (0, 5, 11):
        key = cb.bind(cb.entity[entity_id], "birth_city")
        got = memory.read(key.unsqueeze(0), hard=True)[0]
        value_idx = int(corpus.values["birth_city"][entity_id])
        assert torch.allclose(got, cb.value["birth_city"][value_idx], atol=1e-5)


def test_works_with_returns_the_partner_entity_code(memory, corpus):
    """The composition path: hop 1 must return something re-bindable, not an opaque value."""
    cb = memory.codebook
    for entity_id in (0, 9, 21):
        key = cb.bind(cb.entity[entity_id], "works_with")
        got = memory.read(key.unsqueeze(0), hard=True)[0]
        partner = int(corpus.values["works_with"][entity_id][0])
        assert torch.allclose(got, cb.entity[partner], atol=1e-5)
        # ...and re-binding it forms a valid second address.
        second = cb.bind(got, "birth_year")
        assert torch.allclose(second, memory.keys[memory.fact_index[(partner, "birth_year")]],
                              atol=1e-5)


def test_two_hop_chain_resolves_purely_in_latent_space(memory, corpus):
    """The full capability, with no learning involved: bind, read, re-bind, read."""
    cb = memory.codebook
    entity_id = 4
    partner_code = memory.read(cb.bind(cb.entity[entity_id], "works_with").unsqueeze(0), hard=True)
    answer = memory.read(cb.bind(partner_code[0], "birth_city").unsqueeze(0), hard=True)[0]
    partner = int(corpus.values["works_with"][entity_id][0])
    expected_idx = int(corpus.values["birth_city"][partner])
    assert torch.allclose(answer, cb.value["birth_city"][expected_idx], atol=1e-5)


def test_soft_read_is_differentiable_and_hard_read_is_not_required_to_be(memory):
    q = torch.randn(2, memory.codebook.dim, requires_grad=True)
    memory.read(q).sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()


def test_disabled_memory_returns_zeros(memory):
    off = memory.disabled_copy()
    z = off.read(torch.randn(3, memory.codebook.dim))
    assert torch.count_nonzero(z) == 0
    assert memory.enabled and not off.enabled


def test_codebook_is_frozen(memory):
    """Nothing the kernel achieves may come from the memory reshaping itself."""
    for tensor in (memory.keys, memory.values_matrix, memory.codebook.entity):
        assert not tensor.requires_grad


# --- the kernel -------------------------------------------------------------------------


def _kernel(n_hops=2, latent_dim=64):
    cfg = LatentKernelConfig(
        vocab_size=40, block_size=32, n_layer=4, n_head=2, n_embd=32,
        latent_dim=latent_dim, n_read_heads=1, cross_attn_every=2, n_hops=n_hops,
    )
    return LatentReasoningKernel(cfg), cfg


def test_forward_shapes_and_retrieval_count(memory):
    model, cfg = _kernel()
    B, T = 3, 20
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    subj = torch.randn(B, cfg.latent_dim)
    logits, loss, info = model(
        idx, subj, torch.zeros(B, dtype=torch.long),
        torch.tensor([[5, 10]] * B), memory,
        targets=idx, loss_mask=torch.ones(B, T),
    )
    assert logits.shape == (B, T, cfg.vocab_size)
    assert torch.isfinite(loss)
    assert len(info["queries"]) == 2, "one query per hop"
    assert info["latents"].shape == (B, 2, cfg.latent_dim)


def test_gradient_reaches_the_first_hop_query_through_the_second_retrieval(memory):
    """The composition path must be differentiable end to end, or hop 1 never learns."""
    model, cfg = _kernel()
    B, T = 2, 20
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    _, loss, _ = model(
        idx, torch.randn(B, cfg.latent_dim), torch.zeros(B, dtype=torch.long),
        torch.tensor([[4, 9]] * B), memory, targets=idx, loss_mask=torch.ones(B, T),
    )
    loss.backward()
    assert model.query_out.weight.grad is not None
    assert torch.isfinite(model.query_out.weight.grad).all()
    assert model.query_out.weight.grad.abs().sum() > 0
    assert model.subject_in.weight.grad.abs().sum() > 0, "subject latent must receive gradient"


def test_latents_are_not_visible_before_they_are_retrieved(memory):
    """Causality over the read channel: a position may only see earlier hops' latents."""
    model, cfg = _kernel()
    B, T = 1, 20
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    subj = torch.randn(B, cfg.latent_dim)
    qpos = torch.tensor([[6, 12]])
    with torch.no_grad():
        base, _, _ = model(idx, subj, torch.zeros(B, dtype=torch.long), qpos, memory)
    # Positions at or before the first query cannot depend on any retrieved latent, so a
    # different memory must not change them.
    other = OracleLatentMemory(
        memory.corpus, LatentCodebook.build(memory.corpus, dim=cfg.latent_dim, seed=99)
    )
    with torch.no_grad():
        alt, _, _ = model(idx, subj, torch.zeros(B, dtype=torch.long), qpos, other)
    assert torch.allclose(base[:, :7], alt[:, :7], atol=1e-4)
    assert not torch.allclose(base[:, 13:], alt[:, 13:], atol=1e-4)


def test_parameter_count_stays_near_the_named_size():
    cfg = LatentKernelConfig(
        vocab_size=81, block_size=192, n_layer=8, n_head=8, n_embd=320, latent_dim=128
    )
    model = LatentReasoningKernel(cfg)
    assert 9e6 < model.n_params() < 16e6, model.n_params()
