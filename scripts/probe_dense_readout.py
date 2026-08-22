"""Is the fact absent, or present and unreadable? The registered alternative readout (M5 §1.7).

The exposure ladder scores the dense baseline by **generation**. If that reads ~0 at every rung,
two completely different worlds are consistent with it:

* **not stored** — the weights do not contain the fact; the ladder is measuring under-exposure and
  the registered branches apply;
* **stored but unreadable** — the fact is in the weights and the greedy QA channel cannot get it
  out. Then the ladder is measuring *readout*, the exposure axis is not the axis being varied, and
  the mis-specification response applies.

The pre-registration named the instrument that separates them: **the log-probability of the gold
value under teacher forcing**, which needs no generation and therefore cannot fail for any reason
generation fails for. It is scored two ways:

1. **2AFC** — total NLL of the gold value span against a distractor drawn from the *same value
   space*, in the *same sentence*. Chance is exactly 50% by construction, and it is a
   discrimination test, so it is insensitive to the model's overall fluency.
2. **Teacher-forced exact** — argmax at every value position equals the gold value.

Both are read on the **statement** surface, which is the channel the fact was actually taught
through; a template whose ``{value}`` precedes ``{subject}`` is excluded, because there the value
is unpredictable by construction and the number would measure template choice.

A control on **held-out-entity distractors** is not needed here: the distractor is a different
value for the *same* entity in the *same* sentence, so anything the model knows about the sentence
other than the fact cancels.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from fka.data import templates as T  # noqa: E402
from fka.data.corpus_gen import generate_corpus  # noqa: E402
from fka.data.tokenizer import CharTokenizer  # noqa: E402
from fka.dense.stream import DenseCorpusStream, DenseDataConfig  # noqa: E402
from fka.kernel.model import ReasoningKernel, config_for  # noqa: E402


def value_last_variant(relation: str) -> int:
    """A template whose ``{value}`` follows ``{subject}``.

    Variant 4 of several relations puts the value first ("Born in {value}, {subject} ..."), where
    the value is genuinely unpredictable and a low score would say nothing about storage.
    """
    for i, template in enumerate(T.relation_templates(relation).statements):
        if template.index("{value}") > template.index("{subject}"):
            return i
    raise ValueError(f"{relation}: no template places the value after the subject")


@torch.no_grad()
def span_nll(
    model, tokenizer: CharTokenizer, texts: list[str], spans: list[tuple[int, int]], device
) -> np.ndarray:
    """Mean NLL over ``[start, end)`` character positions of each text (char tokenizer: 1:1)."""
    out = np.zeros(len(texts))
    for i in range(0, len(texts), 32):
        chunk, chunk_spans = texts[i : i + 32], spans[i : i + 32]
        ids = [tokenizer.encode(t) for t in chunk]
        width = max(len(x) for x in ids)
        batch = torch.full((len(ids), width), tokenizer.pad_id, dtype=torch.long, device=device)
        for j, x in enumerate(ids):
            batch[j, : len(x)] = torch.tensor(x, dtype=torch.long, device=device)
        logits, _ = model(batch[:, :-1])
        logp = F.log_softmax(logits.float(), dim=-1)
        targets = batch[:, 1:]
        token_lp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        for j, (start, end) in enumerate(chunk_spans):
            # target position p predicts token p+1, so value token at index k is at target k-1
            out[i + j] = -token_lp[j, start - 1 : end - 1].mean().item()
    return out


@torch.no_grad()
def span_exact(
    model, tokenizer: CharTokenizer, texts: list[str], spans: list[tuple[int, int]], device
) -> np.ndarray:
    out = np.zeros(len(texts), dtype=bool)
    for i in range(0, len(texts), 32):
        chunk, chunk_spans = texts[i : i + 32], spans[i : i + 32]
        ids = [tokenizer.encode(t) for t in chunk]
        width = max(len(x) for x in ids)
        batch = torch.full((len(ids), width), tokenizer.pad_id, dtype=torch.long, device=device)
        for j, x in enumerate(ids):
            batch[j, : len(x)] = torch.tensor(x, dtype=torch.long, device=device)
        logits, _ = model(batch[:, :-1])
        pred = logits.argmax(dim=-1)
        for j, (start, end) in enumerate(chunk_spans):
            out[i + j] = bool(
                (pred[j, start - 1 : end - 1] == batch[j, start:end]).all().item()
            )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--size", default="1M")
    p.add_argument("--n-entities", type=int, default=31_686)
    p.add_argument("--corpus-seed", type=int, default=0)
    p.add_argument("--probe-fraction", type=float, default=0.06)
    p.add_argument("--n-given-names", type=int, default=4096)
    p.add_argument("--n-surnames", type=int, default=4096)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    tok = CharTokenizer()
    corpus = generate_corpus(
        n_entities=args.n_entities, seed=args.corpus_seed, probe_fraction=args.probe_fraction,
        n_given_names=args.n_given_names, n_surnames=args.n_surnames,
    )
    stream = DenseCorpusStream(corpus, tok, DenseDataConfig(exposures=1, seed=args.corpus_seed))
    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ReasoningKernel(
        config_for(args.size, vocab_size=tok.vocab_size, block_size=args.block_size)
    )
    model.load_state_dict(blob["model_state"])
    model.to(args.device).eval()
    print(f"== {args.checkpoint}  step {blob['step']:,}  epoch {blob['epoch']}  "
          f"loss {blob['losses'][-1]:.4f}")
    print("   readout: teacher-forced, no generation (M5 §1.7 registered alternative)")

    rng = np.random.default_rng(args.seed)
    for label, ids in (
        ("qa_heldout", stream.probe_fact_ids()),
        ("qa_trained", stream.probe_fact_ids(qa_trained=True)),
    ):
        by_relation: dict[str, list[int]] = {}
        for fid in ids:
            by_relation.setdefault(corpus.relations[int(fid) // corpus.n_entities], []).append(
                int(fid)
            )
        print(f"\n== {label}")
        for relation, fids in by_relation.items():
            variant = value_last_variant(relation)
            pick = rng.permutation(len(fids))[: args.n]
            gold_texts, gold_spans, bad_texts, bad_spans = [], [], [], []
            for k in pick:
                entity = int(fids[k]) % corpus.n_entities
                subject = corpus.subject_key(relation, entity)
                gold = corpus.value_of(relation, entity)
                other_entity = int(rng.integers(0, corpus.n_entities))
                distractor = corpus.value_of(relation, other_entity)
                if distractor == gold:
                    continue
                for value, texts, spans in (
                    (gold, gold_texts, gold_spans),
                    (distractor, bad_texts, bad_spans),
                ):
                    text = T.render_statement(relation, subject, value, variant)
                    start = text.index(value)
                    texts.append(text)
                    spans.append((start, start + len(value)))

            nll_gold = span_nll(model, tok, gold_texts, gold_spans, args.device)
            nll_bad = span_nll(model, tok, bad_texts, bad_spans, args.device)
            exact = span_exact(model, tok, gold_texts, gold_spans, args.device)
            afc = float((nll_gold < nll_bad).mean())
            print(
                f"   {relation:<12} n={len(gold_texts):>4}  2AFC {afc:7.2%} (chance 50.00%)  "
                f"teacher-forced exact {exact.mean():7.2%}  "
                f"NLL gold {nll_gold.mean():.4f} vs distractor {nll_bad.mean():.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
