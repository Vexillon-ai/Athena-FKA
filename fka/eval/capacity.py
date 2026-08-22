"""Bits-per-parameter capacity measurement.

This is the instrument the whole program is judged by. Research plan §4.5 sets the bar at
**>4 bits/param** for at least one substrate design (versus the ~2 bits/param dense-transformer
ceiling), and §8's M3 milestone makes ~3 bits/param a go/no-go gate. So the arithmetic here has
to be defensible.

The measurement
---------------
Every probe is a fact whose exact information content is known by construction (see
:mod:`fka.data.corpus_gen`). We ask the model, score the answer, and add up bits::

    stored_bits = Σ_probes  bits(fact) * correct(fact)

Two refinements that matter:

**Chance correction.** A model that answers "1950" to every birth-year question gets 1% accuracy
on a 100-way question, which naively credits it with 1% of the corpus entropy. It stored nothing.
So we also report a chance-corrected figure, per relation::

    corrected_accuracy = max(0, (accuracy - chance) / (1 - chance))

where ``chance`` is the hit rate of a guesser sampling from the value space's own distribution
(``Σ p²``; ``1/K`` for a uniform space). Raw and corrected are both reported — raw is what the
literature usually quotes, corrected is what we should believe.

**Extrapolation.** Probes are a sample of the stored facts, so ``estimated_corpus_bits`` scales
the measured fraction back up to the whole corpus. With a 10% probe fraction that is a 10x
extrapolation, and it is only meaningful because probes are sampled uniformly at random.

The model interface
-------------------
Anything with ``recall(query: Query) -> str``. :class:`Query` carries both the structured address
``(subject, relation)`` and the canonical natural-language question, so a dict-backed oracle, a
learned KnowledgeStore and a language model can all be measured by the same harness without
adapters. Returning ``None`` or ``""`` counts as an abstention (scored wrong, but tracked
separately — a store that knows when it does not know is worth more than one that confabulates).

CLI::

    python -m fka.eval.capacity --smoke
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from fka.data.corpus_gen import COWORKER_JOINER, KnowledgeCorpus, MemoryPair, generate_corpus


@dataclass(frozen=True)
class Query:
    """What a memory is asked. Carries both addressing forms; use whichever fits the model."""

    fact_id: int
    key: tuple[str, str]
    text: str
    subject: str
    relation: str

    @classmethod
    def from_pair(cls, pair: MemoryPair) -> Query:
        return cls(
            fact_id=pair.fact_id,
            key=pair.key,
            text=pair.query,
            subject=pair.key[0],
            relation=pair.key[1],
        )


@runtime_checkable
class RecallModel(Protocol):
    """Any memory that can be asked for a fact."""

    def recall(self, query: Query) -> str | None: ...


# =======================================================================================
# scoring
# =======================================================================================


def normalize_answer(answer: str | None, *, case_sensitive: bool = False) -> str:
    if answer is None:
        return ""
    text = " ".join(str(answer).split())
    return text if case_sensitive else text.casefold()


def answers_match(
    predicted: str | None, gold: str, relation: str, *, case_sensitive: bool = False
) -> bool:
    """Exact match, except that multi-valued answers are compared as unordered sets.

    ``works_with`` answers name a set of colleagues; "A and B" and "B and A" are the same fact,
    and penalising order would measure formatting rather than storage.
    """
    pred = normalize_answer(predicted, case_sensitive=case_sensitive)
    truth = normalize_answer(gold, case_sensitive=case_sensitive)
    if not pred:
        return False
    if relation == "works_with":
        sep = normalize_answer(COWORKER_JOINER, case_sensitive=case_sensitive)
        return {p.strip() for p in pred.split(sep)} == {t.strip() for t in truth.split(sep)}
    return pred == truth


def chance_accuracy(corpus: KnowledgeCorpus, relation: str) -> float:
    """Hit rate of a guesser drawing from the relation's own value distribution (``Σ p²``)."""
    space = corpus.spaces[relation]
    if relation == "works_with":
        return 1.0 / space.size
    if relation == "full_name":
        return 1.0 / space.size
    p = space.probabilities()
    return float((p**2).sum())


# =======================================================================================
# results
# =======================================================================================


@dataclass
class RelationResult:
    relation: str
    n_probes: int
    n_correct: int
    n_abstained: int
    probe_bits: float
    stored_bits: float
    chance: float

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_probes if self.n_probes else 0.0

    @property
    def corrected_accuracy(self) -> float:
        if self.n_probes == 0 or self.chance >= 1.0:
            return 0.0
        return max(0.0, (self.accuracy - self.chance) / (1.0 - self.chance))

    @property
    def corrected_bits(self) -> float:
        return self.corrected_accuracy * self.probe_bits

    def to_dict(self) -> dict:
        d = asdict(self)
        d |= {
            "accuracy": self.accuracy,
            "corrected_accuracy": self.corrected_accuracy,
            "corrected_bits": self.corrected_bits,
        }
        return d


@dataclass
class CapacityReport:
    """Result of one capacity measurement."""

    n_probes: int
    n_correct: int
    n_abstained: int
    probe_bits: float
    stored_bits: float
    corrected_stored_bits: float
    corpus_total_bits: float
    corpus_stored_bits: float
    n_params: int | None
    per_relation: dict[str, RelationResult] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_probes if self.n_probes else 0.0

    @property
    def fraction_of_entropy(self) -> float:
        """Share of the probed facts' information content that the model actually holds."""
        return self.stored_bits / self.probe_bits if self.probe_bits else 0.0

    @property
    def corrected_fraction_of_entropy(self) -> float:
        return self.corrected_stored_bits / self.probe_bits if self.probe_bits else 0.0

    @property
    def estimated_corpus_bits(self) -> float:
        """Measured fraction extrapolated over every fact the model was given."""
        return self.fraction_of_entropy * self.corpus_stored_bits

    @property
    def estimated_corpus_bits_corrected(self) -> float:
        return self.corrected_fraction_of_entropy * self.corpus_stored_bits

    @property
    def bits_per_param(self) -> float | None:
        if not self.n_params:
            return None
        return self.estimated_corpus_bits / self.n_params

    @property
    def corrected_bits_per_param(self) -> float | None:
        if not self.n_params:
            return None
        return self.estimated_corpus_bits_corrected / self.n_params

    def to_dict(self) -> dict:
        return {
            "n_probes": self.n_probes,
            "n_correct": self.n_correct,
            "n_abstained": self.n_abstained,
            "accuracy": self.accuracy,
            "probe_bits": self.probe_bits,
            "stored_bits": self.stored_bits,
            "corrected_stored_bits": self.corrected_stored_bits,
            "fraction_of_entropy": self.fraction_of_entropy,
            "corrected_fraction_of_entropy": self.corrected_fraction_of_entropy,
            "corpus_total_bits": self.corpus_total_bits,
            "corpus_stored_bits": self.corpus_stored_bits,
            "estimated_corpus_bits": self.estimated_corpus_bits,
            "estimated_corpus_bits_corrected": self.estimated_corpus_bits_corrected,
            "n_params": self.n_params,
            "bits_per_param": self.bits_per_param,
            "corrected_bits_per_param": self.corrected_bits_per_param,
            "per_relation": {k: v.to_dict() for k, v in self.per_relation.items()},
        }

    def __str__(self) -> str:
        bpp = self.bits_per_param
        lines = [
            f"probes            {self.n_probes:,}  "
            f"(correct {self.n_correct:,}, abstained {self.n_abstained:,})",
            f"accuracy          {self.accuracy:7.2%}",
            f"probe bits        {self.probe_bits:14,.1f}",
            f"stored bits       {self.stored_bits:14,.1f}   "
            f"({self.fraction_of_entropy:6.2%} of probed entropy)",
            f"  chance-corrected{self.corrected_stored_bits:14,.1f}   "
            f"({self.corrected_fraction_of_entropy:6.2%})",
            f"corpus bits       {self.corpus_stored_bits:14,.1f} given, "
            f"{self.estimated_corpus_bits:,.1f} estimated stored",
            f"parameters        {self.n_params if self.n_params else 'n/a'}",
            f"bits/param        {f'{bpp:.4f}' if bpp is not None else 'n/a'}"
            + (
                f"   (corrected {self.corrected_bits_per_param:.4f})"
                if self.corrected_bits_per_param is not None
                else ""
            ),
        ]
        lines.append("per relation:")
        header = f"    {'relation':14s} {'n':>6s} {'acc':>8s} {'corr.acc':>9s} {'bits':>12s}"
        lines.append(header)
        for name, r in self.per_relation.items():
            lines.append(
                f"    {name:14s} {r.n_probes:6,d} {r.accuracy:8.2%} "
                f"{r.corrected_accuracy:9.2%} {r.stored_bits:12,.1f}"
            )
        return "\n".join(lines)


# =======================================================================================
# measurement
# =======================================================================================


def count_parameters(model: object) -> int | None:
    """Parameter count for a torch module, an object exposing ``n_params``, or None."""
    n = getattr(model, "n_params", None)
    if callable(n):
        return int(n())
    if isinstance(n, int):
        return n
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            return int(sum(p.numel() for p in parameters()))
        except (TypeError, AttributeError):
            return None
    return None


def measure_capacity(
    model: RecallModel,
    corpus: KnowledgeCorpus,
    *,
    fact_ids: Sequence[int] | np.ndarray | None = None,
    n_params: int | None = None,
    case_sensitive: bool = False,
) -> CapacityReport:
    """Probe ``model`` over ``fact_ids`` (default: the corpus probe split) and account the bits.

    ``n_params`` defaults to whatever :func:`count_parameters` can infer; pass it explicitly for
    models where the meaningful denominator is not "every tensor element" (for example a store
    whose codebooks are shared across facts).
    """
    if n_params is None:
        n_params = count_parameters(model)

    per_relation: dict[str, RelationResult] = {}
    for pair in corpus.memory_pairs(fact_ids):
        relation = pair.key[1]
        result = per_relation.get(relation)
        if result is None:
            result = per_relation[relation] = RelationResult(
                relation=relation,
                n_probes=0,
                n_correct=0,
                n_abstained=0,
                probe_bits=0.0,
                stored_bits=0.0,
                chance=chance_accuracy(corpus, relation),
            )
        predicted = model.recall(Query.from_pair(pair))
        correct = answers_match(predicted, pair.answer, relation, case_sensitive=case_sensitive)
        result.n_probes += 1
        result.probe_bits += pair.bits
        if predicted is None or not str(predicted).strip():
            result.n_abstained += 1
        if correct:
            result.n_correct += 1
            result.stored_bits += pair.bits

    ordered = {r: per_relation[r] for r in corpus.relations if r in per_relation}
    return CapacityReport(
        n_probes=sum(r.n_probes for r in ordered.values()),
        n_correct=sum(r.n_correct for r in ordered.values()),
        n_abstained=sum(r.n_abstained for r in ordered.values()),
        probe_bits=sum(r.probe_bits for r in ordered.values()),
        stored_bits=sum(r.stored_bits for r in ordered.values()),
        corrected_stored_bits=sum(r.corrected_bits for r in ordered.values()),
        corpus_total_bits=corpus.total_bits,
        corpus_stored_bits=corpus.stored_bits,
        n_params=n_params,
        per_relation=ordered,
    )


def capacity_curve(
    build_model: Callable[[KnowledgeCorpus, np.ndarray], RecallModel],
    corpus: KnowledgeCorpus,
    fact_counts: Iterable[int],
    *,
    n_params: int | None = None,
) -> list[tuple[int, CapacityReport]]:
    """Sweep the number of inserted facts and measure each time — the capacity-knee instrument.

    Research plan §4.4 step 2: insert K facts, measure exact-decode accuracy versus K, and locate
    the knee where accuracy collapses. Bits/param *at the knee* is the headline number, not
    bits/param at any convenient K.

    ``build_model`` receives the corpus and the fact ids to insert, and returns a fresh model
    holding exactly those facts. Probes are drawn from the inserted set, so accuracy measures
    retention rather than coverage.
    """
    train = corpus.train_ids
    out: list[tuple[int, CapacityReport]] = []
    for k in fact_counts:
        k = int(min(k, len(train)))
        subset = train[:k]
        model = build_model(corpus, subset)
        report = measure_capacity(model, corpus, fact_ids=subset, n_params=n_params)
        out.append((k, report))
    return out


# =======================================================================================
# reference memories — the controls every capacity number should be read against
# =======================================================================================


class DictMemory:
    """A Python dict behind the recall interface: the oracle memory and the positive control.

    Research plan §2.4 uses exactly this as the Phase-1 stand-in memory so kernel training can
    start before Phases 2-4 exist. It must recover ~100% of corpus entropy; if it does not, the
    harness is broken, not the model.
    """

    def __init__(self, mapping: dict[tuple[str, str], str]) -> None:
        self._mapping = dict(mapping)

    @classmethod
    def from_corpus(
        cls, corpus: KnowledgeCorpus, fact_ids: Sequence[int] | np.ndarray | None = None
    ) -> DictMemory:
        return cls(corpus.as_oracle_dict(fact_ids))

    def recall(self, query: Query) -> str | None:
        return self._mapping.get(query.key)

    @property
    def n_params(self) -> int:
        """Stored scalars, counted generously as one per character of every stored answer.

        A dict is not a parametric model, so bits/param for it is a diagnostic, not a result.
        """
        return sum(len(v) for v in self._mapping.values())

    def __len__(self) -> int:
        return len(self._mapping)


class RandomGuessMemory:
    """Guesses from each relation's value distribution: the negative control.

    Its measured ``stored_bits`` shows the floor that any real result must clear, and its
    *chance-corrected* bits should sit at ~0 by construction.
    """

    def __init__(self, corpus: KnowledgeCorpus, seed: int = 0) -> None:
        self.corpus = corpus
        self.rng = np.random.default_rng(seed)

    def recall(self, query: Query) -> str:
        relation = query.relation
        corpus = self.corpus
        if relation == "works_with":
            k = corpus.config.n_coworkers
            picks = self.rng.choice(corpus.n_entities, size=k, replace=False)
            return COWORKER_JOINER.join(sorted(corpus.entity_name(int(i)) for i in picks))
        if relation == "full_name":
            return corpus.entity_name(int(self.rng.integers(0, corpus.n_entities)))
        space = corpus.spaces[relation]
        return space.sample(self.rng)

    @property
    def n_params(self) -> int:
        return 0


# =======================================================================================
# CLI
# =======================================================================================


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Measure fact-recall capacity of a memory against a synthetic corpus."
    )
    p.add_argument("--n-entities", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--probe-fraction", type=float, default=0.1)
    p.add_argument("--json", type=str, default=None)
    p.add_argument("--smoke", action="store_true", help="tiny corpus; runs in about a second")
    args = p.parse_args(argv)

    if args.smoke:
        args.n_entities = 500

    corpus = generate_corpus(
        n_entities=args.n_entities, seed=args.seed, probe_fraction=args.probe_fraction
    )
    print(f"== {corpus!r}\n")

    reports = {
        "dict_oracle": measure_capacity(DictMemory.from_corpus(corpus), corpus),
        "random_guess": measure_capacity(RandomGuessMemory(corpus, seed=args.seed), corpus),
    }
    for name, report in reports.items():
        print(f"-- {name}")
        print(report)
        print()

    if args.json:
        Path(args.json).write_text(
            json.dumps({k: v.to_dict() for k, v in reports.items()}, indent=2), encoding="utf-8"
        )
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
