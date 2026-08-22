"""Probing the dense baseline: batched greedy QA, scored by the frozen capacity harness.

The prompt is exactly the surface the QA channel taught (``Q: <canonical question> A: ``) and the
model continues it. No memory is in the loop — there is nothing to put there.

**Truncate at ``<eos>``; never filter it out.** CLAUDE.md records the free-running phantom: an
ad-hoc audit script *filtered* ``<eos>`` from the generated ids instead of truncating there, so
every prediction carried ~20 tokens of post-``<eos>`` garbage and a healthy model measured 1.6%.
That defect lived in a throwaway script while the deployed evaluator was correct, which is why
this path ships with a gold stub whose script deliberately emits garbage after ``<eos>``.

**The gold stub drives the real path.** :class:`ScriptedModel` is a language model in the only
sense that matters here — it returns logits — so the stub exercises batched generation, the
per-row cursor, eos truncation, decoding and scoring. A stub that merely returned strings would
gate the scorer and leave everything upstream of it unprotected, which is the "one gate per
deployed eval path" rule (CLAUDE.md).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from fka.data.corpus_gen import KnowledgeCorpus
from fka.data.tokenizer import CharTokenizer
from fka.dense.surface import Surface, VerboseSurface
from fka.eval.capacity import CapacityReport, Query, measure_capacity

#: Longest answer the corpus can produce is a two-name ``works_with`` set; 64 characters covers it
#: with headroom. Truncation here would score a correct answer wrong, so it is generous on purpose.
DEFAULT_MAX_NEW_TOKENS = 64


class PromptBuilder:
    """Probe prompts for a corpus under a surface — the same string the QA channel trained.

    Subjects are materialised per relation once: under the terse surface the subject is a record
    id derived from the entity index, which the frozen :class:`Query` does not carry.
    """

    def __init__(self, corpus: KnowledgeCorpus, surface: Surface | None = None) -> None:
        self.corpus = corpus
        self.surface = surface or VerboseSurface()
        self._subjects: dict[str, list[str]] = {}

    def subject(self, relation: str, entity_id: int) -> str:
        cached = self._subjects.get(relation)
        if cached is None:
            cached = self._subjects[relation] = self.surface.subjects(self.corpus, relation)
        return cached[entity_id]

    def for_pair(self, pair) -> str:
        relation = pair.key[1]
        entity = int(pair.fact_id) % self.corpus.n_entities
        return self.surface.probe_prompt(relation, self.subject(relation, entity))


def truncate_at_eos(ids: Sequence[int], eos_id: int) -> list[int]:
    """Everything before the first ``<eos>``. Not a filter — see the module docstring."""
    out: list[int] = []
    for i in ids:
        if int(i) == eos_id:
            break
        out.append(int(i))
    return out


@torch.no_grad()
def generate_answers(
    model,
    tokenizer: CharTokenizer,
    prompts_text: Sequence[str],
    *,
    device="cpu",
    amp_dtype=None,
    batch_size: int = 128,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    stop_at_eos: bool = True,
) -> list[str]:
    """Greedy continuations of ``Q: … A: ``, batched over ragged prompt lengths.

    Prompts are **right**-aligned into a padded buffer with a per-row write cursor rather than
    left-padded: this model uses absolute positions ``arange(T)``, so left-padding would shift
    every prompt to a position range it was never trained at and the probe would be measuring
    positional generalisation instead of recall.

    ``stop_at_eos=False`` is a **gate instrument, not a run mode.** With early stopping on,
    truncating at ``<eos>`` and filtering ``<eos>`` out are indistinguishable — nothing is ever
    generated past it — so a gold stub cannot see the defect CLAUDE.md records. Running the full
    token budget forces post-``<eos>`` content into the span and makes truncation load-bearing,
    which is what ``test_gold_stub_needs_truncation_not_filtering`` measures.
    """
    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()
    pad_id, eos_id = tokenizer.pad_id, tokenizer.eos_id
    block_size = getattr(getattr(model, "cfg", None), "block_size", 1 << 30)
    answers: list[str] = []

    for start in range(0, len(prompts_text), batch_size):
        chunk = prompts_text[start : start + batch_size]
        prompts = [tokenizer.encode(text) for text in chunk]
        lengths = np.array([len(p) for p in prompts], dtype=np.int64)
        width = int(lengths.max()) + max_new_tokens
        if width > block_size:
            raise ValueError(
                f"probe needs {width} positions but block_size is {block_size}; a truncated "
                f"prompt would score a correct model wrong"
            )
        idx = torch.full((len(chunk), width), pad_id, dtype=torch.long, device=device)
        for i, p in enumerate(prompts):
            idx[i, : len(p)] = torch.tensor(p, dtype=torch.long, device=device)
        cursor = torch.tensor(lengths, dtype=torch.long, device=device)
        done = torch.zeros(len(chunk), dtype=torch.bool, device=device)
        rows = torch.arange(len(chunk), device=device)

        for _ in range(max_new_tokens):
            active = ~done
            if not bool(active.any()):
                break
            end = int(cursor.max())
            window = idx[:, :end]
            if amp_dtype is not None:
                with torch.autocast(device_type=torch.device(device).type, dtype=amp_dtype):
                    logits, _ = model(window)
            else:
                logits, _ = model(window)
            nxt = logits[rows, cursor - 1, :].argmax(dim=-1)
            # Only unfinished rows are written to. Writing pad over a finished row's <eos> would
            # erase the very token the truncation gate is there to test.
            write_rows, write_cols = rows[active], cursor[active]
            idx[write_rows, write_cols] = nxt[active]
            cursor[active] = write_cols + 1
            if stop_at_eos:
                done[write_rows] = nxt[active] == eos_id

        for i, p in enumerate(prompts):
            span = idx[i, len(p) : int(cursor[i])].tolist()
            # skip_special stays OFF: a stray <pad> or marker must be visible as a wrong answer,
            # not silently swept out of the string before scoring.
            answers.append(tokenizer.decode(truncate_at_eos(span, eos_id)))
    if was_training and hasattr(model, "train"):
        model.train()
    return answers


class DenseRecall:
    """``recall(query) -> str`` over a trained dense LM, with answers computed in batches."""

    def __init__(
        self,
        model,
        tokenizer: CharTokenizer,
        *,
        device="cpu",
        amp_dtype=None,
        batch_size: int = 128,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        stop_at_eos: bool = True,
        surface: Surface | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.amp_dtype = amp_dtype
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.stop_at_eos = stop_at_eos
        self.surface = surface or VerboseSurface()
        self._prompts: PromptBuilder | None = None
        self._cache: dict[int, str] = {}

    @property
    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()))

    def prefill(self, corpus: KnowledgeCorpus, fact_ids) -> None:
        if self._prompts is None or self._prompts.corpus is not corpus:
            self._prompts = PromptBuilder(corpus, self.surface)
        pairs = list(corpus.memory_pairs(fact_ids))
        answers = generate_answers(
            self.model,
            self.tokenizer,
            [self._prompts.for_pair(p) for p in pairs],
            device=self.device,
            amp_dtype=self.amp_dtype,
            batch_size=self.batch_size,
            max_new_tokens=self.max_new_tokens,
            stop_at_eos=self.stop_at_eos,
        )
        self._cache.update({p.fact_id: a for p, a in zip(pairs, answers, strict=True)})

    def recall(self, query: Query) -> str | None:
        cached = self._cache.get(query.fact_id)
        if cached is not None:
            return cached
        if self._prompts is None:
            raise RuntimeError(
                "call prefill() first: the probe prompt is a function of the surface and the "
                "corpus, and Query does not carry the entity index a terse subject needs"
            )
        answer = generate_answers(
            self.model,
            self.tokenizer,
            [self._prompts.for_pair(query)],
            device=self.device,
            amp_dtype=self.amp_dtype,
            batch_size=1,
            max_new_tokens=self.max_new_tokens,
            stop_at_eos=self.stop_at_eos,
        )[0]
        self._cache[query.fact_id] = answer
        return answer


# =======================================================================================
# gold stub and its verified-red counterpart (M5 §4.2)
# =======================================================================================


class ScriptedModel(nn.Module):
    """A 'language model' that emits a fixed script per row — the gold stub's engine.

    It has no parameters and no memory; it exists so the probe path can be driven with a known
    answer. Anything the real path does wrong — cursor arithmetic, eos handling, decoding,
    scoring — shows up here as a number below 100%.
    """

    @dataclass(frozen=True)
    class _Cfg:
        block_size: int

    def __init__(
        self,
        scripts: dict[str, list[int]],
        tokenizer: CharTokenizer,
        block_size=4096,
        answer_prefix: str = " A: ",
    ):
        super().__init__()
        self.scripts = scripts
        self.tokenizer = tokenizer
        self.answer_prefix = answer_prefix
        self.cfg = self._Cfg(block_size=block_size)

    def forward(self, idx: torch.Tensor, targets=None, loss_mask=None, *, reduction="mean"):
        B, T = idx.shape
        V = self.tokenizer.vocab_size
        logits = torch.zeros(B, T, V, device=idx.device)
        for i in range(B):
            text = self.tokenizer.decode(
                [int(t) for t in idx[i].tolist() if int(t) != self.tokenizer.pad_id]
            )
            # The row is [prompt][generated so far]; the prompt ends at the last answer prefix,
            # which no answer can contain. Re-deriving it each step keeps the script index correct
            # as the row grows.
            cut = text.rfind(self.answer_prefix)
            prompt_text = text if cut < 0 else text[: cut + len(self.answer_prefix)]
            script = self.scripts.get(prompt_text.strip(), [self.tokenizer.eos_id])
            prompt_len = len(self.tokenizer.encode(prompt_text))
            for p in range(T):
                k = p - prompt_len + 1
                token = script[k] if 0 <= k < len(script) else self.tokenizer.eos_id
                logits[i, p, token] = 1.0
        return logits, None


def _scripted_recall(
    corpus: KnowledgeCorpus,
    tokenizer: CharTokenizer,
    fact_ids,
    answer_of,
    *,
    trailing_garbage: str = "ZZZ nonsense",
    stop_at_eos: bool = True,
    surface: Surface | None = None,
) -> DenseRecall:
    """Build a stub whose script is ``answer_of(pair)`` then ``<eos>`` then garbage.

    The garbage is only reachable with ``stop_at_eos=False``; that combination is what makes
    truncation load-bearing and the gate red-able. See :func:`generate_answers`.
    """
    surface = surface or VerboseSurface()
    prompts = PromptBuilder(corpus, surface)
    scripts: dict[str, list[int]] = {}
    for pair in corpus.memory_pairs(fact_ids):
        tail = tokenizer.encode(answer_of(pair)) + [tokenizer.eos_id]
        tail += tokenizer.encode(trailing_garbage)
        scripts[prompts.for_pair(pair).strip()] = tail
    model = ScriptedModel(scripts, tokenizer, answer_prefix=surface.answer_prefix)
    return DenseRecall(
        model, tokenizer, batch_size=16, stop_at_eos=stop_at_eos, surface=surface
    )


def GoldStubRecall(
    corpus: KnowledgeCorpus,
    tokenizer: CharTokenizer,
    fact_ids,
    *,
    stop_at_eos: bool = True,
    surface: Surface | None = None,
) -> DenseRecall:
    """Scores 100% by construction. If it does not, the probe path is broken, not the model."""
    return _scripted_recall(
        corpus, tokenizer, fact_ids, lambda pair: pair.answer,
        stop_at_eos=stop_at_eos, surface=surface,
    )


def WrongAnswerRecall(
    corpus: KnowledgeCorpus,
    tokenizer: CharTokenizer,
    fact_ids,
    seed: int = 0,
    *,
    surface: Surface | None = None,
) -> DenseRecall:
    """The verified-red half: format-perfect, factually wrong. Must NOT score above chance.

    A probe path that scored this highly would be matching on shape rather than content, and the
    gold stub alone cannot see that — it passes for a scorer that returns True unconditionally.
    """
    rng = np.random.default_rng(seed)
    others = {
        r: [corpus.value_of(r, int(e)) for e in rng.integers(0, corpus.n_entities, size=64)]
        for r in corpus.relations
    }

    def wrong(pair):
        pool = [v for v in others[pair.key[1]] if v != pair.answer]
        return pool[int(rng.integers(0, len(pool)))] if pool else "unknown"

    return _scripted_recall(corpus, tokenizer, fact_ids, wrong, surface=surface)


# =======================================================================================
# the measurement
# =======================================================================================


def dense_capacity(
    model,
    tokenizer: CharTokenizer,
    corpus: KnowledgeCorpus,
    fact_ids,
    *,
    device="cpu",
    amp_dtype=None,
    batch_size: int = 128,
    n_params: int | None = None,
    surface: Surface | None = None,
) -> CapacityReport:
    """Probe and account, through the frozen capacity harness (nothing bespoke to this baseline).

    ``n_params`` defaults to every parameter in the model: for a dense LM the whole network *is*
    the store, and exempting the embeddings or the head would be M3 §10.5's gaming counterexample
    2 wearing a different hat.
    """
    recall = (
        model
        if isinstance(model, DenseRecall)
        else DenseRecall(
            model, tokenizer, device=device, amp_dtype=amp_dtype, batch_size=batch_size,
            surface=surface,
        )
    )
    recall.prefill(corpus, fact_ids)
    return measure_capacity(
        recall, corpus, fact_ids=fact_ids, n_params=n_params or recall.n_params
    )


__all__ = [
    "DEFAULT_MAX_NEW_TOKENS",
    "DenseRecall",
    "GoldStubRecall",
    "PromptBuilder",
    "ScriptedModel",
    "WrongAnswerRecall",
    "dense_capacity",
    "generate_answers",
    "truncate_at_eos",
]
