"""Instrument gate: the evaluator must score a known-perfect model at exactly 100%.

An evaluator is a measuring device, and this project has now lost three investigation cycles to
one that was silently wrong. The self-test drives a **gold-emitting stub** through the *deployed*
eval path: if the scorer cannot give 100% to a model that emits the correct answer by construction,
every number it has ever produced is suspect.

The stub is deliberately trivial and the assertion is deliberately exact. `>= 0.99` would have
passed the batching bug that motivated this file at 200 episodes only by luck of composition.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# Standing instrument gate, in make smoke as of 2026-08-02. The instrument marker is kept so
# the gate can be run alone (pytest -m instrument), not to exclude it.
#
# History worth keeping: this file was committed RED and caught two distinct alignment defects -
# a per-batch answer-start slicer in the deployed evaluator, and a prefix-keyed stub that
# collapsed every same-relation episode onto one row (in D3 the subject is a latent, so
# same-relation prompts are character-identical). Both looked like model behaviour.
pytestmark = pytest.mark.instrument

from fka.data.corpus_gen import CorpusConfig, generate_corpus  # noqa: E402
from fka.data.multihop import chain_probe_list  # noqa: E402
from fka.eval.latent_leakage import _collect, decode_accuracy  # noqa: E402
from fka.kernel.latent_episodes import (  # noqa: E402
    d3_tokenizer,
    episode_from_probe,
    pack,
)
from fka.kernel.latent_memory import LatentCodebook, OracleLatentMemory  # noqa: E402
from fka.kernel.latent_train import evaluate_d3  # noqa: E402


class GoldStub(torch.nn.Module):
    """Emits the gold continuation by construction: argmax at position t is gold[t+1].

    Mimics LatentReasoningKernel's signature so it travels the real eval path untouched.
    """

    def __init__(
        self,
        gold: torch.Tensor,
        vocab: int,
        subject_codes: torch.Tensor,
        corrupt_row: int | None = None,
    ):
        super().__init__()
        self.gold = gold
        self.vocab = vocab
        self.subject_codes = subject_codes  # (N, dim), one per episode
        self.corrupt_row = corrupt_row
        # Shortest answer-start across the set, so the match window never reaches generated text.
        self.prompt_match_len = 60
        self.cfg = type("C", (), {"block_size": gold.shape[1]})()

    def forward(self, idx, subject_code, subj_pos, qvec_pos, memory, **kw):
        """Identify each row by (subject latent, token prefix), then emit its gold continuation.

        The subject latent is essential and the token prefix alone is not enough: in D3 the
        subject never appears as text, so every same-relation episode has a *character-identical*
        prefix. A prefix-keyed stub silently collapses them all onto the first row — which is
        exactly the bug this stub had, and it looked like an evaluator defect.

        Keying on batch position would be worse still: it breaks the moment the evaluator regroups.
        """
        B, T = idx.shape
        logits = torch.zeros(B, T, self.vocab)
        for b in range(B):
            # Match on the prompt region only. Beyond it the corrupted stub emits wrong tokens
            # that feed back into the input, so a full-prefix match would fail to identify its
            # own episode — the stub must stay identifiable even while deliberately misbehaving.
            k = min(T, self.prompt_match_len)
            same_text = (self.gold[:, :k] == idx[b, :k]).all(dim=1)
            same_subject = (self.subject_codes == subject_code[b]).all(dim=1)
            candidates = (same_text & same_subject).nonzero()
            assert len(candidates) == 1, (
                f"episode not uniquely identifiable from (subject, prefix): "
                f"{len(candidates)} candidates"
            )
            row = int(candidates[0, 0])
            for t in range(T):
                nxt = int(self.gold[row, t + 1]) if t + 1 < self.gold.shape[1] else 0
                if self.corrupt_row is not None and row == self.corrupt_row:
                    nxt = (nxt + 1) % self.vocab
                logits[b, t, nxt] = 10.0
        return logits, None, {"queries": [], "latents": None, "retrieved": []}

    def eval(self):
        return self


@pytest.fixture(scope="module")
def fixture():
    corpus = generate_corpus(CorpusConfig(n_entities=300, seed=0, n_coworkers=1))
    tok = d3_tokenizer()
    mem = OracleLatentMemory(corpus, LatentCodebook.build(corpus, dim=32, seed=0))
    n2i = {n: i for i, n in enumerate(corpus.entity_names)}
    probes = chain_probe_list(corpus, 3)
    np.random.default_rng(0).shuffle(probes)
    probes = probes[:48]
    eps = [episode_from_probe(p, corpus, n2i) for p in probes]
    packed = pack(eps, tok, 128, mem.fact_index)
    codes = mem.codebook.entity[torch.from_numpy(packed.subject_ids)]
    return mem, tok, packed, codes, torch.from_numpy(packed.tokens)


def test_eval_batch_mixes_answer_start_positions(fixture):
    """Guards the guard: if every episode had the same length this test proves nothing."""
    _, _, packed, _, _ = fixture
    starts = {int(np.flatnonzero(packed.answer_mask[i])[0]) for i in range(len(packed.tokens))}
    assert len(starts) > 1, (
        "fixture no longer mixes episode lengths, so it cannot detect a per-batch slicer; "
        "pick probes spanning several tail relations"
    )


def test_evaluator_scores_a_gold_emitting_model_at_exactly_one(fixture):
    """THE instrument gate. Anything below 1.0 means the scorer is broken, not the model."""
    mem, tok, packed, codes, gold = fixture
    acc = decode_accuracy(GoldStub(gold, tok.vocab_size, codes), mem, packed, tok, device="cpu")
    assert acc == 1.0, (
        f"evaluator scored a by-construction-correct model at {acc:.1%}. "
        f"See experiments/2026-08-02_d3-codec-remeasure/QUARANTINE.md"
    )


def test_selftest_is_sensitive_to_a_corrupted_model(fixture):
    """And it must not pass everything: one deliberately wrong row must show up."""
    mem, tok, packed, codes, gold = fixture
    acc = decode_accuracy(
        GoldStub(gold, tok.vocab_size, codes, corrupt_row=0), mem, packed, tok, device="cpu",
    )
    expected = 1.0 - 1.0 / len(packed.tokens)
    assert acc == pytest.approx(expected, abs=1e-9), (
        "a single corrupted episode should cost exactly one episode's worth of accuracy"
    )


def test_score_is_invariant_under_batch_permutation(fixture):
    """Permutation invariance: the behavioural guarantee that identity is not positional.

    Both defects this file has caught were identity-by-position — the evaluator slicing a batch at
    element 0's answer-start, and the stub keying rows by prefix. Neither survives a shuffle.
    With the episode_id join in place this is now structural rather than incidental, but the
    behavioural assertion stays: it is what would notice if a future change reintroduced an
    ordering assumption somewhere the join does not reach.
    """
    mem, tok, packed, codes, gold = fixture
    order = np.random.default_rng(7).permutation(len(packed))
    acc = decode_accuracy(
        GoldStub(gold, tok.vocab_size, codes), mem, packed.select(order), tok, device="cpu"
    )
    assert acc == 1.0, f"score changed under permutation: {acc:.1%} — identity is positional"


def test_permuted_corrupted_stub_still_costs_exactly_one_episode(fixture):
    """The negative side must survive shuffling too, or the sensitivity check is luck."""
    mem, tok, packed, codes, gold = fixture
    n = len(packed)
    order = np.random.default_rng(11).permutation(n)
    acc = decode_accuracy(
        GoldStub(gold, tok.vocab_size, codes, corrupt_row=0),
        mem, packed.select(order), tok, device="cpu",
    )
    assert acc == pytest.approx(1.0 - 1.0 / n, abs=1e-9)


# ---------------------------------------------------------------------------------------
# the episode_id join itself
# ---------------------------------------------------------------------------------------


def test_episode_ids_are_unique_across_independently_packed_sets(fixture):
    """Seen and held-out sets are built by separate `pack()` calls and joined against separately.

    If their ids overlapped, a mixed-up set would score *plausibly* instead of raising — which is
    the whole failure mode the join exists to make impossible.
    """
    _, tok, packed, _, _ = fixture
    corpus = generate_corpus(CorpusConfig(n_entities=300, seed=0, n_coworkers=1))
    n2i = {n: i for i, n in enumerate(corpus.entity_names)}
    mem2 = OracleLatentMemory(corpus, LatentCodebook.build(corpus, dim=32, seed=0))
    other = pack(
        [episode_from_probe(p, corpus, n2i) for p in chain_probe_list(corpus, 3)[:16]],
        tok, 128, mem2.fact_index,
    )
    assert not (set(packed.episode_id.tolist()) & set(other.episode_id.tolist()))


def test_scoring_boundary_rejects_a_missing_episode(fixture):
    """The assertion must be live: silently scoring 47 of 48 episodes is how this class hides."""
    _, _, packed, _, _ = fixture
    predictions = {int(e): "whatever" for e in packed.episode_id[1:]}
    with pytest.raises(AssertionError, match="never scored"):
        _collect(packed, predictions)


def test_evaluate_d3_also_scores_the_gold_stub_at_exactly_one(fixture):
    """The *other* evaluator. It had no self-test, and that is precisely how it kept the bug.

    ``decode_accuracy`` was fixed and gated; ``evaluate_d3`` — the function that produced the
    sweep's headline accuracies — kept the identical per-batch slicer for another two commits
    because nothing was watching it. One instrument gate per deployed eval path, not per module.
    """
    mem, tok, packed, codes, gold = fixture
    res, _ = evaluate_d3(
        GoldStub(gold, tok.vocab_size, codes), mem, packed, tok, device="cpu", batch_size=16
    )
    assert res.accuracy == 1.0, (
        f"evaluate_d3 scored a by-construction-correct model at {res.accuracy:.1%}"
    )
    assert res.n_correct == len(packed)


def test_evaluate_d3_is_permutation_invariant_and_sensitive(fixture):
    """Both ends, shuffled: one corrupted episode costs exactly one, whatever the order."""
    mem, tok, packed, codes, gold = fixture
    n = len(packed)
    order = np.random.default_rng(3).permutation(n)
    res, _ = evaluate_d3(
        GoldStub(gold, tok.vocab_size, codes, corrupt_row=0),
        mem, packed.select(order), tok, device="cpu", batch_size=16,
    )
    assert res.n_correct == n - 1
    assert sum(res.per_episode.values()) == n - 1


def test_scoring_boundary_rejects_a_foreign_episode(fixture):
    """And an id from another set must not be quietly absorbed."""
    _, _, packed, _, _ = fixture
    predictions = {int(e): "whatever" for e in packed.episode_id}
    predictions[max(predictions) + 10_000] = "from another set"
    with pytest.raises(AssertionError, match="belong to no episode"):
        _collect(packed, predictions)
