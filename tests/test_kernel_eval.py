"""Gate metrics: the chance baseline, the leakage verdict, and failure attribution.

These are tested with stub models rather than a trained kernel. The point is that the *verdict
logic* is right — a harness that reports PASS when a model has memorised everything would be
worse than no harness at all.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from fka.data.corpus_gen import generate_corpus  # noqa: E402
from fka.data.multihop import two_hop_probe_list  # noqa: E402
from fka.eval.kernel_eval import (  # noqa: E402
    COMPOSITION_TARGET,
    LEAKAGE_MARGIN,
    RECALL_TARGET,
    CompositionResult,
    LeakageResult,
    RecallResult,
    most_frequent_answer_baseline,
)


@pytest.fixture(scope="module")
def corpus():
    return generate_corpus(n_entities=80, seed=0, n_coworkers=1)


def _recall(condition: str, n: int, n_correct: int) -> RecallResult:
    return RecallResult(
        condition=condition, n=n, n_correct=n_correct, n_malformed=0, memory_hit_rate=1.0
    )


def test_baseline_is_stronger_than_uniform_guessing(corpus):
    """A model that knows the answer format guesses plausibly; the floor must reflect that."""
    pairs = list(corpus.memory_pairs(corpus.train_ids))
    baseline = most_frequent_answer_baseline(pairs)
    assert 0.0 < baseline < 0.5
    # birth_year is 1-of-100 uniform, so a most-frequent-value guesser beats 1/100 on it.
    years = [p for p in pairs if p.key[1] == "birth_year"]
    assert most_frequent_answer_baseline(years) > 1 / 100


def test_baseline_of_a_constant_relation_is_one():
    class P:
        def __init__(self, answer):
            self.key = ("x", "r")
            self.answer = answer

    assert most_frequent_answer_baseline([P("a"), P("a"), P("a")]) == pytest.approx(1.0)


def test_leakage_passes_only_when_memory_works_and_nothing_leaks():
    good = LeakageResult(
        with_memory=_recall("on", 100, 100),
        without_memory=_recall("off", 100, 2),
        chance_baseline=0.02,
    )
    assert good.excess_over_chance == pytest.approx(0.0)
    assert good.passes


def test_leakage_fails_when_the_kernel_answers_without_memory():
    leaked = LeakageResult(
        with_memory=_recall("on", 100, 100),
        without_memory=_recall("off", 100, 60),
        chance_baseline=0.02,
    )
    assert leaked.excess_over_chance > LEAKAGE_MARGIN
    assert not leaked.passes, "a kernel that answers without memory has memorised the facts"


def test_leakage_fails_when_memory_itself_does_not_work():
    """Near-chance without memory is meaningless if recall with memory is also near chance."""
    broken = LeakageResult(
        with_memory=_recall("on", 100, 30),
        without_memory=_recall("off", 100, 2),
        chance_baseline=0.02,
    )
    assert broken.excess_over_chance < LEAKAGE_MARGIN
    assert not broken.passes
    assert RECALL_TARGET == 0.95


def test_composition_verdict_tracks_the_target():
    assert CompositionResult(100, 90, 5, 5, 0, 2.0).passes
    assert not CompositionResult(100, 80, 15, 5, 0, 2.0).passes
    assert COMPOSITION_TARGET == 0.85


def test_result_dicts_are_json_ready():
    import json

    result = LeakageResult(_recall("on", 10, 10), _recall("off", 10, 0), 0.1)
    json.dumps(result.to_dict())
    json.dumps(CompositionResult(10, 9, 1, 0, 0, 2.0).to_dict())


def test_composition_probes_exist_for_the_fixture(corpus):
    """Guards the fixture itself: an empty probe list would make every gate vacuously pass."""
    assert len(two_hop_probe_list(corpus)) > 0
    assert CompositionResult(0, 0, 0, 0, 0, 0.0).accuracy == 0.0
