"""Capacity harness: the two controls, partial-knowledge linearity, and chance correction.

The controls are the point. A dict must recover ~100% of corpus entropy and a random guesser
~0%; if either drifts, every bits-per-parameter number the project reports is suspect.
"""

from __future__ import annotations

import math

import pytest

from fka.data.corpus_gen import generate_corpus
from fka.eval.capacity import (
    DictMemory,
    Query,
    RandomGuessMemory,
    answers_match,
    capacity_curve,
    chance_accuracy,
    measure_capacity,
    normalize_answer,
)


@pytest.fixture(scope="module")
def corpus():
    return generate_corpus(n_entities=600, seed=0, probe_fraction=0.25)


# --- the two controls -------------------------------------------------------------------


def test_dict_oracle_recovers_essentially_all_of_the_entropy(corpus):
    report = measure_capacity(DictMemory.from_corpus(corpus), corpus)
    assert report.accuracy == 1.0
    assert report.fraction_of_entropy == pytest.approx(1.0)
    assert report.corrected_fraction_of_entropy == pytest.approx(1.0)
    assert report.stored_bits == pytest.approx(report.probe_bits)
    assert report.estimated_corpus_bits == pytest.approx(corpus.stored_bits)
    assert report.n_abstained == 0


def test_random_guessing_recovers_essentially_nothing(corpus):
    report = measure_capacity(RandomGuessMemory(corpus, seed=0), corpus)
    assert report.fraction_of_entropy < 0.02
    assert report.corrected_fraction_of_entropy < 0.02
    assert report.estimated_corpus_bits < 0.02 * corpus.stored_bits


def test_chance_correction_zeroes_out_a_constant_answerer(corpus):
    """A model that always says "1950" is right 1% of the time and has stored nothing."""

    class ConstantMemory:
        def recall(self, query: Query) -> str:
            return "1950"

    ids = corpus.fact_ids_for("birth_year")
    report = measure_capacity(ConstantMemory(), corpus, fact_ids=ids)
    assert report.accuracy == pytest.approx(0.01, abs=0.02)
    assert report.stored_bits > 0, "raw accounting credits chance hits"
    assert report.corrected_stored_bits == pytest.approx(0.0, abs=0.05 * report.probe_bits)


def test_abstention_is_tracked_and_scored_wrong(corpus):
    class SilentMemory:
        def recall(self, query: Query) -> None:
            return None

    report = measure_capacity(SilentMemory(), corpus)
    assert report.n_abstained == report.n_probes
    assert report.stored_bits == 0.0


# --- partial knowledge ------------------------------------------------------------------


def test_a_memory_holding_half_the_facts_scores_about_half_the_bits(corpus):
    """Bits accounting must be linear in what is actually stored, not just correlated with it."""
    kept = corpus.train_ids[::2]
    report = measure_capacity(DictMemory.from_corpus(corpus, kept), corpus)
    assert report.accuracy == pytest.approx(0.5, abs=0.06)
    assert report.fraction_of_entropy == pytest.approx(0.5, abs=0.08)


def test_relations_are_accounted_separately(corpus):
    """Knowing only birth years must credit only birth-year bits."""
    ids = corpus.fact_ids_for("birth_year")
    report = measure_capacity(DictMemory.from_corpus(corpus, ids), corpus)
    assert report.per_relation["birth_year"].accuracy == 1.0
    for relation in corpus.relations:
        if relation != "birth_year":
            assert report.per_relation[relation].accuracy == 0.0
    expected = report.per_relation["birth_year"].probe_bits
    assert report.stored_bits == pytest.approx(expected)


def test_bits_per_param_uses_the_supplied_denominator(corpus):
    report = measure_capacity(DictMemory.from_corpus(corpus), corpus, n_params=1000)
    assert report.n_params == 1000
    assert report.bits_per_param == pytest.approx(corpus.stored_bits / 1000)


def test_bits_per_param_is_none_without_a_denominator(corpus):
    class Anonymous:
        def recall(self, query: Query) -> str:
            return ""

    assert measure_capacity(Anonymous(), corpus).bits_per_param is None


# --- scoring rules ----------------------------------------------------------------------


def test_multi_valued_answers_compare_as_sets():
    assert answers_match("Ann Bell and Cy Dole", "Cy Dole and Ann Bell", "works_with")
    assert not answers_match("Ann Bell and Zed Fox", "Cy Dole and Ann Bell", "works_with")


def test_scalar_answers_compare_exactly_but_forgive_case_and_spacing():
    assert answers_match("  northvale ", "Northvale", "birth_city")
    assert not answers_match("Northvale", "Southvale", "birth_city")
    assert not answers_match("  northvale ", "Northvale", "birth_city", case_sensitive=True)


def test_empty_and_none_answers_never_match():
    assert not answers_match(None, "1931", "birth_year")
    assert not answers_match("", "1931", "birth_year")
    assert normalize_answer(None) == ""


def test_chance_accuracy_matches_the_value_space(corpus):
    assert chance_accuracy(corpus, "birth_year") == pytest.approx(1 / 100)
    assert chance_accuracy(corpus, "birth_city") == pytest.approx(1 / 512)
    assert chance_accuracy(corpus, "employer") == pytest.approx(1 / 1024)
    assert chance_accuracy(corpus, "works_with") == pytest.approx(1 / math.comb(599, 2))


def test_chance_accuracy_is_higher_for_a_skewed_space():
    zipf = generate_corpus(n_entities=300, seed=0, birth_city_distribution="zipf")
    uniform = generate_corpus(n_entities=300, seed=0, birth_city_distribution="uniform")
    assert chance_accuracy(zipf, "birth_city") > chance_accuracy(uniform, "birth_city")


# --- the capacity-knee instrument -------------------------------------------------------


def test_capacity_curve_measures_each_insertion_size(corpus):
    curve = capacity_curve(
        lambda c, ids: DictMemory.from_corpus(c, ids), corpus, [50, 200, 800]
    )
    assert [k for k, _ in curve] == [50, 200, 800]
    for k, report in curve:
        assert report.n_probes == k, "probes are drawn from the inserted facts"
        assert report.accuracy == 1.0, "a dict has no capacity knee"
    assert curve[-1][1].stored_bits > curve[0][1].stored_bits


def test_capacity_curve_exposes_a_knee_when_the_store_is_bounded(corpus):
    """A memory that can only hold 100 facts should hold its bits flat past that point."""

    class BoundedMemory:
        capacity = 100

        def __init__(self, c, ids):
            self._mapping = c.as_oracle_dict(ids[: self.capacity])

        def recall(self, query: Query) -> str | None:
            return self._mapping.get(query.key)

    curve = capacity_curve(BoundedMemory, corpus, [50, 100, 400, 1200])
    accuracies = [r.accuracy for _, r in curve]
    assert accuracies[0] == 1.0 and accuracies[1] == 1.0
    assert accuracies[2] < 0.5 and accuracies[3] < accuracies[2]
    stored = [r.stored_bits for _, r in curve]
    assert stored[2] == pytest.approx(stored[3], rel=0.15), "bits plateau past the knee"
