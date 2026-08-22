"""Multi-hop probes: questions whose answer requires chaining two facts.

A 2-hop question is ``works_with`` followed by an attribute:

    X --works_with--> Y --birth_city--> "Northvale"
    "In what city was the colleague of X born?"

**Why this is the composition test and not a copying test.** At evaluation the model is not handed
the chain. It is given only the question, and must emit its own ``<query>`` spans; the memory
fills the ``<result>`` spans. To answer, it has to issue ``works_with of X``, *read Y out of the
returned result*, and then compose a second query ``birth_city of Y`` — a query whose text it
could not have known before the first retrieval. Getting the final answer right is then a copy
from the second result, which is fine: the skill under test is forming the second query, and a
model that cannot will retrieve the wrong fact and answer wrongly.

**Why ``n_coworkers`` must be 1.** With more than one coworker "the colleague of X" has no unique
referent, and a wrong-but-reasonable answer would be scored as a failure. The chain relation has
to be single-valued for the probe to be well posed, so :func:`two_hop_probes` requires it rather
than silently picking one.

Information content: the answer's entropy is the *second* hop's bits. The first hop's bits are the
cost of locating the right entity, not of the answer, so counting them would overstate what a
correct answer demonstrates.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from fka.data.corpus_gen import KnowledgeCorpus

#: Relations reachable as the second hop. ``works_with`` is excluded: chaining it to itself gives
#: "the colleague of the colleague of X", which is a different (and much noisier) probe.
DEFAULT_TAIL_RELATIONS: tuple[str, ...] = ("birth_year", "birth_city", "employer")

#: One canonical phrasing per tail relation, matching the single-probe-form policy in
#: :mod:`fka.data.templates` — probe wording must not become a second experimental variable.
TWO_HOP_QUESTIONS: dict[str, str] = {
    "birth_year": "In what year was the colleague of {subject} born?",
    "birth_city": "In what city was the colleague of {subject} born?",
    "employer": "Who does the colleague of {subject} work for?",
}

#: Chain phrasing by hop count. Index 2 is the 2-hop wording above; index 3 adds another link.
#: One canonical phrasing per (hops, relation) — probe wording is never an experimental variable.
CHAIN_QUESTIONS: dict[int, dict[str, str]] = {
    2: TWO_HOP_QUESTIONS,
    3: {
        "birth_year": "In what year was the colleague of the colleague of {subject} born?",
        "birth_city": "In what city was the colleague of the colleague of {subject} born?",
        "employer": "Who does the colleague of the colleague of {subject} work for?",
    },
}


@dataclass(frozen=True)
class Hop:
    """One retrieval step: what the model must ask, and what memory returns."""

    key: tuple[str, str]
    answer: str
    fact_id: int


@dataclass(frozen=True)
class MultiHopProbe:
    """A question requiring ``len(hops)`` retrievals, with the chain that solves it."""

    query: str
    answer: str
    hops: tuple[Hop, ...]
    bits: float
    subject: str
    tail_relation: str

    @property
    def n_hops(self) -> int:
        return len(self.hops)

    @property
    def supporting_fact_ids(self) -> tuple[int, ...]:
        return tuple(h.fact_id for h in self.hops)


def two_hop_probes(
    corpus: KnowledgeCorpus,
    *,
    entity_ids: Sequence[int] | np.ndarray | None = None,
    tail_relations: Sequence[str] = DEFAULT_TAIL_RELATIONS,
    exclude_heldout: bool = True,
) -> Iterator[MultiHopProbe]:
    """Yield 2-hop probes over ``corpus``.

    Every supporting fact is checked against the corpus splits: a probe whose chain touches a
    held-out fact is skipped when ``exclude_heldout`` is set, because the model was never given
    that fact and failing it would measure coverage rather than composition.
    """
    if corpus.config.n_coworkers != 1:
        raise ValueError(
            f"two-hop probes need a single-valued chain relation, but n_coworkers="
            f"{corpus.config.n_coworkers}. 'the colleague of X' has no unique referent otherwise. "
            f"Generate the corpus with n_coworkers=1."
        )
    if "works_with" not in corpus.relations:
        raise ValueError("two-hop probes need the works_with relation in the corpus")

    unknown = set(tail_relations) - set(corpus.relations)
    if unknown:
        raise ValueError(f"tail relations not in corpus: {sorted(unknown)}")

    ids = range(corpus.n_entities) if entity_ids is None else entity_ids
    heldout = set(corpus.heldout_ids.tolist()) if exclude_heldout else set()
    chain_rel_index = corpus.relations.index("works_with")

    for entity_id in ids:
        entity_id = int(entity_id)
        chain_fact_id = chain_rel_index * corpus.n_entities + entity_id
        if chain_fact_id in heldout:
            continue
        partner = int(corpus.values["works_with"][entity_id][0])
        subject = corpus.entity_name(entity_id)
        partner_name = corpus.entity_name(partner)

        hop1 = Hop(
            key=(subject, "works_with"),
            answer=partner_name,
            fact_id=chain_fact_id,
        )
        for relation in tail_relations:
            tail_fact_id = corpus.relations.index(relation) * corpus.n_entities + partner
            if tail_fact_id in heldout:
                continue
            value = corpus.value_of(relation, partner)
            yield MultiHopProbe(
                query=TWO_HOP_QUESTIONS[relation].format(subject=subject),
                answer=value,
                hops=(
                    hop1,
                    Hop(key=(partner_name, relation), answer=value, fact_id=tail_fact_id),
                ),
                # The answer's own entropy; hop 1 is addressing cost, not answer content.
                bits=float(corpus.bits[relation][partner]),
                subject=subject,
                tail_relation=relation,
            )


def two_hop_probe_list(corpus: KnowledgeCorpus, **kwargs) -> list[MultiHopProbe]:
    return list(two_hop_probes(corpus, **kwargs))


def chain_probes(
    corpus: KnowledgeCorpus,
    n_hops: int,
    *,
    entity_ids: Sequence[int] | np.ndarray | None = None,
    tail_relations: Sequence[str] = DEFAULT_TAIL_RELATIONS,
    exclude_heldout: bool = True,
) -> Iterator[MultiHopProbe]:
    """Generalise :func:`two_hop_probes` to ``n_hops`` (``n_hops - 1`` ``works_with`` links).

    **Cycles are excluded, and that is not cosmetic.** With ``n_coworkers=1`` the colleague map is
    a random functional graph, so 2-cycles are common: if X's colleague is Y and Y's colleague is
    X, then "the colleague of the colleague of X" *is* X, and a 3-hop probe collapses into a 1-hop
    one that a non-composing model can answer by ignoring the chain entirely. Any chain revisiting
    an entity is dropped.
    """
    if n_hops not in CHAIN_QUESTIONS:
        raise ValueError(f"no canonical phrasing for {n_hops}-hop chains; known: "
                         f"{sorted(CHAIN_QUESTIONS)}")
    if corpus.config.n_coworkers != 1:
        raise ValueError(
            f"chain probes need a single-valued chain relation, but n_coworkers="
            f"{corpus.config.n_coworkers}. Generate the corpus with n_coworkers=1."
        )

    unknown = set(tail_relations) - set(corpus.relations)
    if unknown:
        raise ValueError(f"tail relations not in corpus: {sorted(unknown)}")

    ids = range(corpus.n_entities) if entity_ids is None else entity_ids
    heldout = set(corpus.heldout_ids.tolist()) if exclude_heldout else set()
    chain_rel_index = corpus.relations.index("works_with")
    n_links = n_hops - 1

    for entity_id in ids:
        entity_id = int(entity_id)
        walk = [entity_id]
        hops: list[Hop] = []
        broken = False
        for _ in range(n_links):
            current = walk[-1]
            fact_id = chain_rel_index * corpus.n_entities + current
            if fact_id in heldout:
                broken = True
                break
            nxt = int(corpus.values["works_with"][current][0])
            if nxt in walk:  # cycle: the chain would collapse to a shorter one
                broken = True
                break
            hops.append(
                Hop(key=(corpus.entity_name(current), "works_with"),
                    answer=corpus.entity_name(nxt), fact_id=fact_id)
            )
            walk.append(nxt)
        if broken:
            continue

        final = walk[-1]
        for relation in tail_relations:
            tail_fact_id = corpus.relations.index(relation) * corpus.n_entities + final
            if tail_fact_id in heldout:
                continue
            value = corpus.value_of(relation, final)
            yield MultiHopProbe(
                query=CHAIN_QUESTIONS[n_hops][relation].format(
                    subject=corpus.entity_name(entity_id)
                ),
                answer=value,
                hops=(
                    *hops,
                    Hop(key=(corpus.entity_name(final), relation),
                        answer=value, fact_id=tail_fact_id),
                ),
                bits=float(corpus.bits[relation][final]),
                subject=corpus.entity_name(entity_id),
                tail_relation=relation,
            )


def chain_probe_list(corpus: KnowledgeCorpus, n_hops: int, **kwargs) -> list[MultiHopProbe]:
    return list(chain_probes(corpus, n_hops, **kwargs))
