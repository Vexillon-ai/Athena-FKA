"""Corpus generator: determinism, entropy accounting, rendering, splits, and the fact firewall."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fka.data import templates as T
from fka.data.corpus_gen import CorpusConfig, generate_corpus

SMALL = dict(n_entities=300, seed=0)


@pytest.fixture(scope="module")
def corpus():
    return generate_corpus(**SMALL)


# --- determinism ------------------------------------------------------------------------


def test_same_seed_gives_an_identical_corpus(corpus):
    other = generate_corpus(**SMALL)
    assert other.fingerprint() == corpus.fingerprint()
    assert other.entity_names == corpus.entity_names
    assert list(other.documents()) == list(corpus.documents())
    assert [p.answer for p in other.memory_pairs()] == [p.answer for p in corpus.memory_pairs()]


def test_different_seed_gives_a_different_corpus(corpus):
    other = generate_corpus(**{**SMALL, "seed": 1})
    assert other.fingerprint() != corpus.fingerprint()


def test_fingerprint_reacts_to_config_changes(corpus):
    assert generate_corpus(**{**SMALL, "n_cities": 256}).fingerprint() != corpus.fingerprint()


def test_document_order_is_stable_and_shuffle_is_reproducible(corpus):
    assert list(corpus.documents(shuffle=True)) == list(corpus.documents(shuffle=True))
    assert list(corpus.documents(shuffle=True)) != list(corpus.documents())


# --- entropy accounting -----------------------------------------------------------------


def test_total_bits_is_the_sum_of_per_fact_bits(corpus):
    by_hand = sum(corpus.fact(i).bits for i in range(corpus.n_facts))
    assert corpus.total_bits == pytest.approx(by_hand)
    assert corpus.total_bits == pytest.approx(sum(corpus.bits_by_relation().values()))


def test_per_relation_bits_match_closed_forms(corpus):
    n = corpus.n_entities
    expected = {
        "birth_year": math.log2(100),
        "birth_city": math.log2(512),
        "employer": math.log2(1024),
        "works_with": math.log2(math.comb(n - 1, 2)),
    }
    for relation, bits_per_fact in expected.items():
        total = corpus.bits_by_relation()[relation]
        assert total == pytest.approx(bits_per_fact * n), relation


def test_bits_of_ids_agrees_with_individual_facts(corpus):
    ids = corpus.probe_ids
    assert corpus.bits_of_ids(ids) == pytest.approx(sum(corpus.fact(int(i)).bits for i in ids))


def test_zipf_relation_lowers_entropy_and_varies_per_fact():
    uniform = generate_corpus(**SMALL, birth_city_distribution="uniform")
    zipf = generate_corpus(**SMALL, birth_city_distribution="zipf", zipf_s=1.0)
    assert zipf.bits_by_relation()["birth_city"] < uniform.bits_by_relation()["birth_city"]
    assert len(set(np.round(uniform.bits["birth_city"], 6))) == 1
    assert len(set(np.round(zipf.bits["birth_city"], 6))) > 1


def test_stored_bits_excludes_heldout():
    c = generate_corpus(**SMALL, heldout_fraction=0.2)
    assert len(c.heldout_ids) > 0
    assert c.stored_bits < c.total_bits
    assert c.stored_bits == pytest.approx(c.bits_of_ids(c.train_ids))


# --- fact addressing --------------------------------------------------------------------


def test_fact_id_round_trips_to_relation_and_subject(corpus):
    for fid in (0, 1, corpus.n_entities, corpus.n_facts - 1):
        fact = corpus.fact(fid)
        assert fact.fact_id == fid
        assert corpus.relation_of(fid) == fact.relation
        assert int(corpus.subject_of(fid)) == fact.subject_id
        assert fid in corpus.fact_ids_for(fact.relation)


def test_fact_count_matches_relations_times_entities(corpus):
    assert corpus.n_facts == len(corpus.relations) * corpus.n_entities


def test_entity_names_are_unique(corpus):
    assert len(set(corpus.entity_names)) == corpus.n_entities


# --- splits -----------------------------------------------------------------------------


def test_probe_facts_are_a_subset_of_train_and_disjoint_from_heldout():
    c = generate_corpus(n_entities=800, seed=2, probe_fraction=0.2, heldout_fraction=0.1)
    train, probe, heldout = set(c.train_ids.tolist()), set(c.probe_ids.tolist()), set(
        c.heldout_ids.tolist()
    )
    assert probe <= train, "we measure recall of facts the model was actually given"
    assert not (probe & heldout)
    assert train | heldout == set(range(c.n_facts))
    assert not (train & heldout)


def test_split_fractions_are_approximately_as_configured():
    c = generate_corpus(n_entities=3000, seed=4, probe_fraction=0.25, heldout_fraction=0.1)
    assert len(c.heldout_ids) / c.n_facts == pytest.approx(0.1, abs=0.02)
    assert len(c.probe_ids) / len(c.train_ids) == pytest.approx(0.25, abs=0.02)


def test_zero_heldout_by_default(corpus):
    assert len(corpus.heldout_ids) == 0
    assert corpus.stored_bits == pytest.approx(corpus.total_bits)


# --- rendering --------------------------------------------------------------------------


def test_documents_contain_subject_and_value(corpus):
    for fid in list(corpus.train_ids)[:200]:
        fact = corpus.fact(int(fid))
        doc = next(iter(corpus.documents([fid])))
        assert fact.subject in doc
        assert fact.value in doc


def test_statements_per_fact_emits_that_many_distinct_renderings():
    c = generate_corpus(n_entities=60, seed=0, statements_per_fact=3)
    for fid in list(c.train_ids)[:40]:
        docs = list(c.documents([fid]))
        assert len(docs) == 3
        assert len(set(docs)) == 3, "paraphrases of one fact must differ"


def test_every_template_gets_used_across_a_corpus(corpus):
    """Variant selection must spread over the templates, not collapse onto one."""
    for relation in corpus.relations:
        ids = corpus.fact_ids_for(relation)
        docs = list(corpus.documents(ids))
        n_expected = T.n_variants(relation)
        shapes = {
            tuple(sorted(w for w in d.split() if w.isalpha() and w.islower())) for d in docs
        }
        assert len(shapes) >= n_expected - 1, f"{relation}: templates look under-used"


def test_probes_use_a_single_canonical_question_form(corpus):
    """A probe measures storage; its phrasing must not be a second experimental variable."""
    for relation in corpus.relations:
        ids = corpus.fact_ids_for(relation)
        pairs = list(corpus.memory_pairs(ids[:50]))
        skeletons = {p.query.replace(p.key[0], "<S>") for p in pairs}
        assert len(skeletons) == 1, f"{relation}: probe phrasing varies"


def test_memory_pairs_answer_matches_the_fact(corpus):
    for pair in list(corpus.memory_pairs(corpus.train_ids[:100])):
        fact = corpus.fact(pair.fact_id)
        assert pair.answer == fact.value
        assert pair.key == (fact.subject, fact.relation)
        assert pair.bits == pytest.approx(fact.bits)


def test_works_with_answer_is_a_sorted_join_of_real_entities(corpus):
    fid = int(corpus.fact_ids_for("works_with")[0])
    fact = corpus.fact(fid)
    names = fact.value.split(" and ")
    assert len(names) == corpus.config.n_coworkers
    assert names == sorted(names)
    assert set(names) <= set(corpus.entity_names)
    assert fact.subject not in names


# --- fact firewall ----------------------------------------------------------------------


def test_firewalled_documents_never_contain_a_fact_value(corpus):
    """The core guarantee of research plan §2.4: values reach the kernel only via the interface."""
    ids = list(corpus.train_ids)[:400]
    for fid in ids:
        fact = corpus.fact(int(fid))
        doc = next(iter(corpus.documents([fid], firewall=True)))
        assert fact.value not in doc, f"{fact.relation} value leaked into firewalled text"
        assert T.QUERY_OPEN in doc and T.RESULT_OPEN in doc
        assert fact.subject in doc, "the query span must still address the subject"


def test_firewalled_documents_keep_sentence_shape(corpus):
    fid = int(corpus.train_ids[0])
    plain = next(iter(corpus.documents([fid])))
    walled = next(iter(corpus.documents([fid], firewall=True)))
    assert plain != walled
    assert plain.endswith(".") and walled.endswith(".")


def test_oracle_dict_covers_the_training_facts(corpus):
    oracle = corpus.as_oracle_dict()
    assert len(oracle) == len(corpus.train_ids)
    fact = corpus.fact(int(corpus.train_ids[0]))
    assert oracle[(fact.subject, fact.relation)] == fact.value


def test_oracle_dict_omits_heldout_facts():
    c = generate_corpus(n_entities=400, seed=6, heldout_fraction=0.25)
    oracle = c.as_oracle_dict()
    for fid in list(c.heldout_ids)[:50]:
        fact = c.fact(int(fid))
        assert oracle.get((fact.subject, fact.relation)) != fact.value


# --- optional relations and configuration -----------------------------------------------


def test_name_facts_are_off_by_default_and_keyed_by_record_id(corpus):
    assert "full_name" not in corpus.relations
    c = generate_corpus(n_entities=100, seed=0, include_name_facts=True)
    assert "full_name" in c.relations
    fact = c.fact(int(c.fact_ids_for("full_name")[0]))
    assert fact.subject.startswith("#"), "reverse lookup must not be keyed by its own answer"
    assert fact.value == c.entity_name(fact.subject_id)
    assert fact.value not in fact.subject


def test_name_space_headroom_is_enforced():
    with pytest.raises(ValueError, match="name space is too small"):
        CorpusConfig(n_entities=10_000, n_given_names=32, n_surnames=32)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(n_entities=1),
        dict(probe_fraction=1.5),
        dict(heldout_fraction=1.0),
        dict(statements_per_fact=0),
        dict(relations=("birth_year", "birth_year")),
        dict(relations=("no_such_relation",)),
        dict(birth_city_distribution="lognormal"),
    ],
)
def test_invalid_configs_are_rejected(kwargs):
    with pytest.raises((ValueError, KeyError)):
        CorpusConfig(**kwargs)


def test_scales_to_a_larger_world_without_changing_semantics():
    """Cheap guard on the 10^4 -> 10^6 scaling claim; the arithmetic must not depend on size."""
    c = generate_corpus(n_entities=20_000, seed=0)
    assert c.n_facts == 80_000
    assert len(set(c.entity_names)) == 20_000
    assert c.bits_by_relation()["birth_year"] == pytest.approx(math.log2(100) * 20_000)
