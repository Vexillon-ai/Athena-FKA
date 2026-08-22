"""The two M1 gate measurements: knowledge leakage and multi-hop composition.

Both come from research plan §2.5, and both are pass/fail gates rather than numbers to admire.

**Leakage.** Ask the kernel fact questions with memory *disabled* — every query it emits comes
back as an empty result span. Recall should collapse to near chance. If it does not, the kernel
memorised facts parametrically, and every later bits-per-parameter comparison is invalid because
we would be crediting the substrate for knowledge the kernel already holds. Success criterion:
within 5 points of the chance baseline, against >95% with memory.

Chance here is *measured*, not assumed. The theoretical floor (1/100 for birth years) ignores the
fact that a kernel which has learned the output format will guess a plausible year rather than
random characters. So the baseline is an empirical most-frequent-answer guesser over the same
probes, which is the honest bar a leaking model has to clear.

**Composition.** 2-hop questions where the second query's address is only knowable after reading
the first result. Success criterion: >85% at some kernel size. Because the memory fills every
result span and the kernel is barred from writing them, a correct answer really does require
having asked both questions in the right order.

Failures are attributed rather than merely counted: a wrong 2-hop answer is classified as a
routing failure (asked the wrong thing), a copy failure (retrieved right, answered wrong), or a
format failure (never produced a well-formed query at all). That distinction is what tells us
whether to change the kernel or the interface.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from fka.data.corpus_gen import MemoryPair
from fka.data.multihop import MultiHopProbe
from fka.eval.capacity import answers_match
from fka.kernel.generate import answer_question
from fka.kernel.memory import OracleTextMemory, TextMemory

#: Gate thresholds from research plan §2.5.
LEAKAGE_MARGIN = 0.05
COMPOSITION_TARGET = 0.85
RECALL_TARGET = 0.95


@dataclass
class RecallResult:
    """Accuracy over 1-hop probes under one memory condition."""

    condition: str
    n: int
    n_correct: int
    n_malformed: int
    memory_hit_rate: float
    per_relation: dict[str, tuple[int, int]] = field(default_factory=dict)
    examples: list[dict] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n if self.n else 0.0

    def to_dict(self) -> dict:
        return {
            "condition": self.condition,
            "n": self.n,
            "n_correct": self.n_correct,
            "accuracy": self.accuracy,
            "n_malformed": self.n_malformed,
            "memory_hit_rate": self.memory_hit_rate,
            "per_relation": {
                k: {"n": n, "correct": c, "accuracy": c / n if n else 0.0}
                for k, (n, c) in self.per_relation.items()
            },
            "examples": self.examples,
        }


@dataclass
class CompositionResult:
    n: int
    n_correct: int
    n_routing_failures: int
    n_copy_failures: int
    n_format_failures: int
    mean_hops: float
    examples: list[dict] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n if self.n else 0.0

    @property
    def passes(self) -> bool:
        return self.accuracy >= COMPOSITION_TARGET

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "n_correct": self.n_correct,
            "accuracy": self.accuracy,
            "n_routing_failures": self.n_routing_failures,
            "n_copy_failures": self.n_copy_failures,
            "n_format_failures": self.n_format_failures,
            "mean_hops": self.mean_hops,
            "passes": self.passes,
            "target": COMPOSITION_TARGET,
            "examples": self.examples,
        }


@dataclass
class LeakageResult:
    with_memory: RecallResult
    without_memory: RecallResult
    chance_baseline: float

    @property
    def excess_over_chance(self) -> float:
        """How much better than guessing the kernel does with no memory. This is the leak."""
        return self.without_memory.accuracy - self.chance_baseline

    @property
    def passes(self) -> bool:
        return (
            self.excess_over_chance < LEAKAGE_MARGIN
            and self.with_memory.accuracy >= RECALL_TARGET
        )

    def to_dict(self) -> dict:
        return {
            "with_memory": self.with_memory.to_dict(),
            "without_memory": self.without_memory.to_dict(),
            "chance_baseline": self.chance_baseline,
            "excess_over_chance": self.excess_over_chance,
            "leakage_margin": LEAKAGE_MARGIN,
            "recall_target": RECALL_TARGET,
            "passes": self.passes,
        }

    def __str__(self) -> str:
        verdict = "PASS" if self.passes else "FAIL"
        return (
            f"leakage [{verdict}]  with memory {self.with_memory.accuracy:.1%} "
            f"(target >={RECALL_TARGET:.0%})  without memory "
            f"{self.without_memory.accuracy:.1%} vs chance {self.chance_baseline:.1%} "
            f"-> excess {self.excess_over_chance:+.1%} (allowed <{LEAKAGE_MARGIN:.0%})"
        )


def most_frequent_answer_baseline(pairs: Sequence[MemoryPair]) -> float:
    """Accuracy of always answering each relation's most common value.

    The honest floor for a model that knows the answer *format* but not the facts — strictly
    stronger than 1/K random guessing, so a leak has to beat a real opponent.
    """
    by_relation: dict[str, Counter] = {}
    for pair in pairs:
        by_relation.setdefault(pair.key[1], Counter())[pair.answer] += 1
    if not by_relation:
        return 0.0
    hits = sum(counter.most_common(1)[0][1] for counter in by_relation.values())
    return hits / len(pairs)


def evaluate_recall(
    model,
    tokenizer,
    memory: TextMemory,
    pairs: Sequence[MemoryPair],
    *,
    condition: str,
    device: Any = "cpu",
    amp_dtype: Any = None,
    n_examples: int = 5,
) -> RecallResult:
    memory.reset_stats()
    n_correct = n_malformed = 0
    per_relation: dict[str, list[int]] = {}
    examples: list[dict] = []

    for pair in pairs:
        relation = pair.key[1]
        trace = answer_question(
            model, tokenizer, memory, pair.query, device=device, amp_dtype=amp_dtype
        )
        correct = answers_match(trace.answer, pair.answer, relation)
        if not trace.queries:
            n_malformed += 1
        stats = per_relation.setdefault(relation, [0, 0])
        stats[0] += 1
        stats[1] += int(correct)
        n_correct += int(correct)
        if len(examples) < n_examples:
            examples.append({**trace.to_dict(), "expected": pair.answer, "correct": correct})

    return RecallResult(
        condition=condition,
        n=len(pairs),
        n_correct=n_correct,
        n_malformed=n_malformed,
        memory_hit_rate=memory.stats.hit_rate,
        per_relation={k: (v[0], v[1]) for k, v in per_relation.items()},
        examples=examples,
    )


def leakage_test(
    model,
    tokenizer,
    memory: OracleTextMemory,
    pairs: Sequence[MemoryPair],
    *,
    device: Any = "cpu",
    amp_dtype: Any = None,
) -> LeakageResult:
    """Run the same probes with memory enabled and disabled (research plan §2.5)."""
    with_mem = evaluate_recall(
        model, tokenizer, memory, pairs,
        condition="memory_enabled", device=device, amp_dtype=amp_dtype,
    )
    without_mem = evaluate_recall(
        model, tokenizer, memory.disabled_copy(), pairs,
        condition="memory_disabled", device=device, amp_dtype=amp_dtype,
    )
    return LeakageResult(
        with_memory=with_mem,
        without_memory=without_mem,
        chance_baseline=most_frequent_answer_baseline(pairs),
    )


def composition_test(
    model,
    tokenizer,
    memory: TextMemory,
    probes: Sequence[MultiHopProbe],
    *,
    device: Any = "cpu",
    amp_dtype: Any = None,
    n_examples: int = 5,
) -> CompositionResult:
    """2-hop accuracy with failures attributed to routing, copying, or format."""
    memory.reset_stats()
    n_correct = routing_failures = copy_failures = format_failures = 0
    total_hops = 0
    examples: list[dict] = []

    for probe in probes:
        trace = answer_question(
            model, tokenizer, memory, probe.query,
            device=device, amp_dtype=amp_dtype, max_new_tokens=128,
        )
        total_hops += trace.n_hops
        correct = answers_match(trace.answer, probe.answer, probe.tail_relation)
        n_correct += int(correct)

        if not correct:
            expected_keys = [f"{r} of {s}" for s, r in (h.key for h in probe.hops)]
            asked_all = all(
                any(exp.strip() == got.strip() for got in trace.queries) for exp in expected_keys
            )
            if not trace.queries:
                format_failures += 1
            elif asked_all:
                # Retrieved everything it needed and still answered wrong.
                copy_failures += 1
            else:
                routing_failures += 1

        if len(examples) < n_examples:
            examples.append(
                {
                    **trace.to_dict(),
                    "expected": probe.answer,
                    "expected_queries": [f"{r} of {s}" for s, r in (h.key for h in probe.hops)],
                    "correct": correct,
                }
            )

    return CompositionResult(
        n=len(probes),
        n_correct=n_correct,
        n_routing_failures=routing_failures,
        n_copy_failures=copy_failures,
        n_format_failures=format_failures,
        mean_hops=total_hops / len(probes) if probes else 0.0,
        examples=examples,
    )
