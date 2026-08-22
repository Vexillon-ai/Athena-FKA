"""2-hop probes: chain validity, entropy accounting, and the well-posedness precondition."""

from __future__ import annotations

import pytest

from fka.data.corpus_gen import generate_corpus
from fka.data.multihop import DEFAULT_TAIL_RELATIONS, two_hop_probe_list, two_hop_probes


@pytest.fixture(scope="module")
def corpus():
    return generate_corpus(n_entities=120, seed=0, n_coworkers=1)


def test_requires_a_single_valued_chain_relation():
    """"The colleague of X" has no unique referent with 2 coworkers, so the probe is ill-posed."""
    multi = generate_corpus(n_entities=50, seed=0, n_coworkers=2)
    with pytest.raises(ValueError, match="unique referent"):
        two_hop_probe_list(multi)


def test_chain_resolves_to_the_real_second_hop_value(corpus):
    for probe in two_hop_probe_list(corpus)[:40]:
        hop1, hop2 = probe.hops
        partner = hop1.answer
        assert hop1.key[1] == "works_with"
        assert hop2.key[0] == partner, "second hop must be addressed to hop 1's answer"
        assert probe.answer == hop2.answer
        # The answer really is the partner's attribute, not the subject's.
        partner_id = corpus.entity_names.index(partner)
        assert corpus.value_of(hop2.key[1], partner_id) == probe.answer


def test_second_query_is_unknowable_before_the_first_retrieval(corpus):
    """The whole point of the composition test: hop 2's address is not in the question."""
    for probe in two_hop_probe_list(corpus)[:40]:
        assert probe.hops[1].key[0] not in probe.query


def test_supporting_fact_ids_point_at_the_right_facts(corpus):
    for probe in two_hop_probe_list(corpus)[:30]:
        for hop in probe.hops:
            fact = corpus.fact(hop.fact_id)
            assert (fact.subject, fact.relation) == hop.key
            assert fact.value == hop.answer


def test_bits_are_the_answers_entropy_not_the_chains(corpus):
    """Hop 1 is addressing cost; counting it would overstate what a correct answer shows."""
    for probe in two_hop_probe_list(corpus)[:20]:
        tail_fact = corpus.fact(probe.hops[1].fact_id)
        assert probe.bits == pytest.approx(tail_fact.bits)


def test_covers_every_tail_relation(corpus):
    relations = {p.tail_relation for p in two_hop_probe_list(corpus)}
    assert relations == set(DEFAULT_TAIL_RELATIONS)
    assert "works_with" not in relations, "chaining works_with to itself is a different probe"


def test_probe_count_is_entities_times_tail_relations(corpus):
    probes = two_hop_probe_list(corpus)
    assert len(probes) == corpus.n_entities * len(DEFAULT_TAIL_RELATIONS)


def test_determinism(corpus):
    a = [(p.query, p.answer) for p in two_hop_probe_list(corpus)]
    b = [(p.query, p.answer) for p in two_hop_probe_list(generate_corpus(
        n_entities=120, seed=0, n_coworkers=1))]
    assert a == b


def test_heldout_chains_are_skipped():
    c = generate_corpus(n_entities=200, seed=1, n_coworkers=1, heldout_fraction=0.3)
    heldout = set(c.heldout_ids.tolist())
    probes = two_hop_probe_list(c, exclude_heldout=True)
    assert probes, "some chains should survive"
    for probe in probes:
        assert not (set(probe.supporting_fact_ids) & heldout)
    assert len(probes) < len(two_hop_probe_list(c, exclude_heldout=False))


def test_unknown_tail_relation_is_rejected(corpus):
    with pytest.raises(ValueError, match="not in corpus"):
        list(two_hop_probes(corpus, tail_relations=("favourite_colour",)))


def test_entity_subset_limits_generation(corpus):
    probes = two_hop_probe_list(corpus, entity_ids=[0, 1, 2])
    assert len(probes) == 3 * len(DEFAULT_TAIL_RELATIONS)
    assert {p.subject for p in probes} == {corpus.entity_name(i) for i in (0, 1, 2)}
