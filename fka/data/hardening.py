"""Gate hardening: entity-level holdout and surface-form distractors.

The M1 gates passed at 100% with chain-level holdout, which means they no longer discriminate.
Two independent tightenings, each closing a different escape route:

**Entity-level holdout.** A disjoint set of entities that is never the *subject* of any training
episode — 1-hop or chain. At evaluation the kernel is asked about entities it has never been
asked about. Chain-level holdout only withheld the *pairing*; the kernel had still formed queries
about every entity by name, so "generalises to a new chain" and "has a learned per-entity query
habit" were indistinguishable. Entity holdout separates them.

Held-out entities may still appear as retrieved *values* (a training chain can point at one), and
that is deliberate: severing that too would change the graph rather than the split, and the
capability under test is forming a query about an unfamiliar subject, not never having seen the
name.

**Surface-form distractors.** Extra entities whose names are one edit away from real ones, with
independently sampled attributes, present in the memory but never in training. They change the
failure mode of a sloppy address from a visible miss into a *confident wrong answer*: without
them, a near-miss query returns an empty result and the kernel gets a free signal that it erred.
Retrieval that works only because wrong keys hit nothing is not retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fka.data.corpus_gen import KnowledgeCorpus
from fka.data.templates import DEFAULT_RELATIONS


@dataclass(frozen=True)
class EntitySplit:
    """A partition of entity ids into training subjects and held-out probe subjects."""

    train: np.ndarray
    heldout: np.ndarray

    def __post_init__(self) -> None:
        if set(self.train.tolist()) & set(self.heldout.tolist()):
            raise ValueError("entity split is not disjoint")

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_heldout(self) -> int:
        return len(self.heldout)

    def to_dict(self) -> dict:
        return {"n_train": self.n_train, "n_heldout": self.n_heldout}


def entity_split(corpus: KnowledgeCorpus, fraction: float = 0.2, seed: int = 0) -> EntitySplit:
    """Partition entities into training subjects and held-out probe subjects."""
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(corpus.n_entities)
    n_heldout = max(1, int(corpus.n_entities * fraction))
    return EntitySplit(train=np.sort(order[n_heldout:]), heldout=np.sort(order[:n_heldout]))


def _edit_variants(name: str, rng: np.random.Generator, n: int) -> list[str]:
    """Near-miss spellings of ``name``: one duplicated, dropped, or swapped character."""
    given, _, surname = name.partition(" ")
    out: set[str] = set()
    attempts = 0
    while len(out) < n and attempts < 40:
        attempts += 1
        target = surname if rng.random() < 0.5 else given
        if len(target) < 3:
            continue
        i = int(rng.integers(1, len(target) - 1))
        mode = int(rng.integers(0, 3))
        if mode == 0:  # duplicate a character
            mutated = target[:i] + target[i] + target[i:]
        elif mode == 1:  # drop a character
            mutated = target[:i] + target[i + 1 :]
        else:  # swap adjacent characters
            mutated = target[:i] + target[i + 1] + target[i] + target[i + 2 :]
        candidate = (
            f"{given} {mutated}" if target is surname else f"{mutated} {surname}"
        )
        if candidate != name:
            out.add(candidate)
    return sorted(out)[:n]


def distractor_facts(
    corpus: KnowledgeCorpus,
    *,
    entity_ids: np.ndarray | None = None,
    per_entity: int = 1,
    relations: tuple[str, ...] = DEFAULT_RELATIONS,
    seed: int = 0,
) -> dict[tuple[str, str], str]:
    """Oracle entries for near-name entities that do not exist in the corpus.

    Returned as a ``(subject, relation) -> value`` mapping to merge into a memory. Values are
    drawn from the same value spaces as real facts, so a distractor answer is indistinguishable
    from a real one by surface form alone — which is the point.
    """
    rng = np.random.default_rng(seed)
    ids = range(corpus.n_entities) if entity_ids is None else entity_ids
    real_names = set(corpus.entity_names)
    out: dict[tuple[str, str], str] = {}

    for entity_id in ids:
        name = corpus.entity_name(int(entity_id))
        for variant in _edit_variants(name, rng, per_entity):
            if variant in real_names:
                continue  # would shadow a genuine fact
            for relation in relations:
                if relation == "works_with":
                    partner = int(rng.integers(0, corpus.n_entities))
                    out[(variant, relation)] = corpus.entity_name(partner)
                else:
                    space = corpus.spaces[relation]
                    out[(variant, relation)] = space.sample(rng)
    return out


def harden_memory(memory, distractors: dict[tuple[str, str], str]):
    """Merge distractors into an ``OracleTextMemory`` without overwriting real facts."""
    added = 0
    for key, value in distractors.items():
        if key not in memory.mapping:
            memory.mapping[key] = value
            added += 1
    return added
