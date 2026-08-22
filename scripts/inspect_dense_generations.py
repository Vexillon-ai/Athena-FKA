"""What did the dense baseline actually say? Storage failure vs format failure, per relation.

A near-zero accuracy has two completely different causes and the score cannot tell them apart:

* **format failure** — the model does not produce a well-formed answer at all, so a fact it holds
  is scored wrong. The registered mis-specification response applies.
* **storage failure** — the model answers fluently with a plausible value of the right relation
  and simply does not know this one. That is the honest sub-knee regime, and the ladder is
  measuring what it claims to measure.

This runs the **deployed** decode path (:func:`fka.dense.probe.generate_answers`), not a
reimplementation of it. CLAUDE.md's free-running phantom was a throwaway audit script that got
`<eos>` handling wrong while the deployed evaluator had it right, and two diagnosis cycles were
spent on the difference.

CPU by default so it can be run against a checkpoint while a GPU job holds the lock.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from fka.data.corpus_gen import generate_corpus  # noqa: E402
from fka.data.tokenizer import CharTokenizer  # noqa: E402
from fka.dense.probe import generate_answers  # noqa: E402
from fka.dense.stream import DenseCorpusStream, DenseDataConfig  # noqa: E402
from fka.eval.capacity import answers_match  # noqa: E402
from fka.kernel.model import ReasoningKernel, config_for  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--size", default="1M")
    p.add_argument("--n-entities", type=int, default=31_686)
    p.add_argument("--corpus-seed", type=int, default=0)
    p.add_argument("--probe-fraction", type=float, default=0.06)
    p.add_argument("--n-given-names", type=int, default=4096)
    p.add_argument("--n-surnames", type=int, default=4096)
    p.add_argument("--qa-entity-fraction", type=float, default=0.5)
    p.add_argument("--per-relation", type=int, default=6, help="examples printed per relation")
    p.add_argument("--n", type=int, default=400, help="probes scored per half")
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    tok = CharTokenizer()
    corpus = generate_corpus(
        n_entities=args.n_entities, seed=args.corpus_seed, probe_fraction=args.probe_fraction,
        n_given_names=args.n_given_names, n_surnames=args.n_surnames,
    )
    stream = DenseCorpusStream(
        corpus, tok,
        DenseDataConfig(exposures=1, qa_entity_fraction=args.qa_entity_fraction,
                        seed=args.corpus_seed),
    )
    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ReasoningKernel(
        config_for(args.size, vocab_size=tok.vocab_size, block_size=args.block_size)
    )
    model.load_state_dict(blob["model_state"])
    model.to(args.device).eval()
    print(f"== {args.checkpoint}  step {blob['step']:,}  epoch {blob['epoch']}  "
          f"loss {blob['losses'][-1]:.4f}")

    for label, ids in (
        ("qa_heldout", stream.probe_fact_ids()),
        ("qa_trained", stream.probe_fact_ids(qa_trained=True)),
    ):
        rng = np.random.default_rng(0)
        pick = ids[rng.permutation(ids.size)[: args.n]]
        pairs = list(corpus.memory_pairs(np.sort(pick)))
        got = generate_answers(
            model, tok, [q.query for q in pairs], device=args.device, batch_size=64
        )
        by_relation: dict[str, list] = {}
        for pair, answer in zip(pairs, got, strict=True):
            by_relation.setdefault(pair.key[1], []).append((pair, answer))

        print(f"\n== {label}")
        for relation, rows in by_relation.items():
            correct = sum(answers_match(a, p.answer, relation) for p, a in rows)
            # "Well-formed" = the string is a member of this relation's value space. That is the
            # split between not knowing the fact and not knowing how to answer at all.
            legal = corpus.spaces[relation].values if relation != "works_with" else ()
            legal_set = {str(v).casefold() for v in legal}
            wellformed = (
                sum(a.strip().casefold() in legal_set for _, a in rows)
                if legal_set
                else sum(" and " in a for _, a in rows)
            )
            print(f"   {relation:<12} n={len(rows):>4}  correct {correct / len(rows):7.2%}  "
                  f"well-formed {wellformed / len(rows):7.2%}  "
                  f"distinct answers {len({a for _, a in rows}):>4}  "
                  f"modal {Counter(a for _, a in rows).most_common(1)[0]}")
            for pair, answer in rows[: args.per_relation]:
                flag = "OK " if answers_match(answer, pair.answer, relation) else "   "
                print(f"     {flag} got {answer!r:<44} gold {pair.answer!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
