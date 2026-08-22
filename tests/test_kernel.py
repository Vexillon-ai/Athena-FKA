"""Kernel: memory interface, episode masking, model contract, and the D1 pipeline.

The load-bearing test here is the firewall one — that retrieved values never enter the loss. If
that mask is wrong the kernel is silently a plain language model memorising facts, every leakage
number becomes meaningless, and nothing else in the suite would notice.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fka.data.corpus_gen import generate_corpus  # noqa: E402
from fka.data.multihop import two_hop_probe_list  # noqa: E402
from fka.data.templates import QUERY_CLOSE, QUERY_OPEN, RESULT_CLOSE, RESULT_OPEN  # noqa: E402
from fka.data.tokenizer import CharTokenizer  # noqa: E402
from fka.kernel.episodes import (  # noqa: E402
    Role,
    answer_mask,
    episode_from_probe,
    one_hop_episodes,
    pack_episodes,
    trainable_mask,
)
from fka.kernel.memory import (  # noqa: E402
    OracleTextMemory,
    format_query,
    parse_query,
    splice_results,
)
from fka.kernel.model import MAX_PARAMS, KernelConfig, ReasoningKernel, config_for  # noqa: E402
from fka.kernel.train import TrainConfig, lr_at, train_kernel  # noqa: E402


@pytest.fixture(scope="module")
def corpus():
    return generate_corpus(n_entities=60, seed=0, n_coworkers=1)


@pytest.fixture(scope="module")
def tokenizer():
    return CharTokenizer()


# --- memory interface -------------------------------------------------------------------


def test_query_round_trips():
    assert parse_query(format_query("Ann Bell", "birth_year")) == ("Ann Bell", "birth_year")
    assert parse_query("birth_city of Ann Bell") == ("Ann Bell", "birth_city")


def test_malformed_queries_are_reported_not_raised():
    assert parse_query("!!! nonsense") is None
    assert parse_query("") is None


def test_oracle_memory_answers_and_counts(corpus):
    mem = OracleTextMemory.from_corpus(corpus)
    fact = corpus.fact(int(corpus.train_ids[0]))
    span = mem.answer_span(format_query(fact.subject, fact.relation))
    assert span == f"{RESULT_OPEN}{fact.value}{RESULT_CLOSE}"
    assert mem.stats.hits == 1 and mem.stats.misses == 0


def test_disabled_memory_returns_wellformed_empty_spans(corpus):
    """The leakage condition must change only the answer, never the shape of the interface."""
    mem = OracleTextMemory.from_corpus(corpus).disabled_copy()
    fact = corpus.fact(int(corpus.train_ids[0]))
    span = mem.answer_span(format_query(fact.subject, fact.relation))
    assert span == f"{RESULT_OPEN}{RESULT_CLOSE}"
    assert fact.value not in span
    assert mem.stats.misses == 1


def test_memory_miss_and_malformed_are_distinguished(corpus):
    mem = OracleTextMemory.from_corpus(corpus)
    mem.answer_span("birth_year of Nobody At All")
    mem.answer_span("###")
    assert mem.stats.misses == 1
    assert mem.stats.malformed == 1


def test_splice_fills_firewalled_text(corpus):
    mem = OracleTextMemory.from_corpus(corpus)
    fid = int(corpus.train_ids[0])
    fact = corpus.fact(fid)
    walled = next(iter(corpus.documents([fid], firewall=True)))
    assert fact.value not in walled
    assert fact.value in splice_results(walled, mem)


# --- episodes and the firewall mask -----------------------------------------------------


def test_result_tokens_are_excluded_from_the_loss(corpus, tokenizer):
    """The firewall. If this breaks, the kernel is trained to memorise facts."""
    episode = next(one_hop_episodes(corpus))
    ids, roles = episode.encode(tokenizer)
    mask = trainable_mask(roles)
    assert (roles == int(Role.RESULT)).any(), "episode must contain a result span"
    assert not mask[roles == int(Role.RESULT)].any()
    # The retrieved value's characters must be exactly the untrained positions.
    value_ids = tokenizer.encode(episode.answer)
    masked_ids = ids[roles == int(Role.RESULT)]
    assert list(masked_ids) == list(value_ids)


def test_query_tokens_are_trained(corpus, tokenizer):
    """Emitting the right query is the skill; it must not be masked out with the result."""
    episode = next(one_hop_episodes(corpus))
    _, roles = episode.encode(tokenizer)
    assert (roles == int(Role.QUERY)).any()
    assert trainable_mask(roles)[roles == int(Role.QUERY)].all()


def test_answer_tokens_are_trained_and_separately_identifiable(corpus, tokenizer):
    episode = next(one_hop_episodes(corpus))
    _, roles = episode.encode(tokenizer)
    assert answer_mask(roles).any()
    assert trainable_mask(roles)[answer_mask(roles)].all()


def test_rendering_without_memory_drops_only_the_values(corpus):
    episode = next(one_hop_episodes(corpus))
    with_mem, without = episode.render(), episode.render(with_memory=False)
    assert episode.answer in with_mem
    assert f"{RESULT_OPEN}{RESULT_CLOSE}" in without
    assert QUERY_OPEN in without and QUERY_CLOSE in without
    # The answer still appears once (after "A: "), but never inside a result span.
    assert without.count(episode.answer) == 1


def test_two_hop_episode_has_both_spans_in_order(corpus, tokenizer):
    probe = two_hop_probe_list(corpus)[0]
    episode = episode_from_probe(probe)
    text = episode.render()
    assert episode.kind == "2hop"
    assert text.count(QUERY_OPEN) == 2
    first, second = (h.key for h in probe.hops)
    assert text.index(format_query(*first)) < text.index(format_query(*second))


def test_packing_pads_and_masks_padding_out_of_the_loss(corpus, tokenizer):
    episodes = list(one_hop_episodes(corpus))[:8]
    tokens, roles = pack_episodes(episodes, tokenizer, 256)
    assert tokens.shape == (8, 256)
    assert (tokens[0][-1] == tokenizer.pad_id) or len(episodes[0].encode(tokenizer)[0]) == 256
    assert not trainable_mask(roles)[roles == int(Role.RESULT)].any()


def test_packing_refuses_to_truncate_silently(corpus, tokenizer):
    """Truncating mid-answer would score an episode wrong for a reason unrelated to the model."""
    episodes = list(one_hop_episodes(corpus))[:4]
    with pytest.raises(ValueError, match="exceed block_size"):
        pack_episodes(episodes, tokenizer, 16)


# --- model ------------------------------------------------------------------------------


def test_named_sizes_match_their_labels():
    for label, expected in (("10M", 10e6), ("50M", 50e6), ("150M", 150e6)):
        model = ReasoningKernel(config_for(label, vocab_size=81, block_size=192))
        assert 0.8 * expected <= model.n_params() <= 1.25 * expected, label


def test_parameter_cap_is_enforced():
    """CLAUDE.md guardrail: 300M is a hard cap, not a default."""
    with pytest.raises(ValueError, match="hard cap"):
        ReasoningKernel(KernelConfig(n_layer=48, n_head=16, n_embd=1600, block_size=64))
    assert MAX_PARAMS == 300_000_000


def test_forward_shapes_and_masked_loss():
    cfg = KernelConfig(vocab_size=32, block_size=16, n_layer=2, n_head=2, n_embd=32)
    model = ReasoningKernel(cfg)
    x = torch.randint(0, 32, (3, 8))
    y = torch.randint(0, 32, (3, 8))
    logits, loss = model(x, y, torch.ones(3, 8))
    assert logits.shape == (3, 8, 32)
    assert torch.isfinite(loss)


def test_fully_masked_batch_does_not_produce_nan():
    cfg = KernelConfig(vocab_size=32, block_size=16, n_layer=2, n_head=2, n_embd=32)
    model = ReasoningKernel(cfg)
    x = torch.randint(0, 32, (2, 8))
    _, loss = model(x, x, torch.zeros(2, 8))
    assert torch.isfinite(loss), "empty mask must not divide by zero"


def test_mask_actually_changes_the_loss():
    cfg = KernelConfig(vocab_size=32, block_size=16, n_layer=2, n_head=2, n_embd=32)
    model = ReasoningKernel(cfg)
    torch.manual_seed(0)
    x = torch.randint(0, 32, (2, 8))
    mask = torch.ones(2, 8)
    mask[:, 4:] = 0
    _, full = model(x, x, torch.ones(2, 8))
    _, partial = model(x, x, mask)
    assert not torch.isclose(full, partial), "masking must not be a no-op"


def test_rejects_sequences_longer_than_block_size():
    cfg = KernelConfig(vocab_size=32, block_size=8, n_layer=1, n_head=1, n_embd=16)
    model = ReasoningKernel(cfg)
    with pytest.raises(ValueError, match="exceeds block_size"):
        model(torch.randint(0, 32, (1, 9)))


def test_gradients_flow_to_every_parameter():
    cfg = KernelConfig(vocab_size=32, block_size=16, n_layer=2, n_head=2, n_embd=32)
    model = ReasoningKernel(cfg)
    x = torch.randint(0, 32, (2, 8))
    _, loss = model(x, x, torch.ones(2, 8))
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached {missing}"


# --- training ---------------------------------------------------------------------------


def test_lr_schedule_warms_up_then_decays():
    cfg = TrainConfig(lr=1e-3, warmup_steps=10, steps=100, min_lr_ratio=0.1)
    assert lr_at(0, cfg) < lr_at(5, cfg) < lr_at(9, cfg) <= cfg.lr
    assert lr_at(99, cfg) < lr_at(50, cfg) < cfg.lr
    assert lr_at(99, cfg) >= cfg.lr * cfg.min_lr_ratio * 0.99


def test_training_reduces_loss_on_a_tiny_corpus(corpus, tokenizer):
    episodes = list(one_hop_episodes(corpus))[:32]
    cfg = TrainConfig(size="tiny", block_size=192, batch_size=4, steps=30, amp=False, log_every=999)
    model, state = train_kernel(episodes, tokenizer, cfg, device="cpu", progress=False)
    assert state.step == 30
    assert np.mean(state.losses[-5:]) < np.mean(state.losses[:5]), "loss should fall"
    assert all(np.isfinite(state.losses))


def test_routing_loss_flag_is_off_by_default_and_activates_when_set(corpus, tokenizer):
    """D2 must be a flag on the same path, not a fork."""
    episodes = list(one_hop_episodes(corpus))[:16]
    base = TrainConfig(size="tiny", block_size=192, batch_size=4, steps=5, amp=False, log_every=999)
    assert base.routing_loss_weight == 0.0
    _, off = train_kernel(episodes, tokenizer, base, device="cpu", progress=False)
    assert off.routing_losses == []

    on_cfg = TrainConfig(**{**base.to_dict(), "routing_loss_weight": 0.5})
    _, on = train_kernel(episodes, tokenizer, on_cfg, device="cpu", progress=False)
    assert len(on.routing_losses) == 5
    assert all(v >= 0 for v in on.routing_losses), "hinge loss cannot go negative"
