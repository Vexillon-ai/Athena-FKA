"""Seeded synthetic knowledge-world generator — the project's primary scientific instrument.

We generate a world of synthetic people with attributes (birth year, birth city, employer) and a
relation (works with), render it into natural language, and — critically — know *exactly* how many
bits of information the world contains. Ground truth is known by construction and duplication is
controlled, so "how many bits did this architecture actually store?" becomes a measurement rather
than an estimate. Web text cannot support that question: you never know what the model already saw.

Design notes worth knowing before changing anything here:

* **Facts are addressed, not stored as objects.** With ``R`` relations over ``N`` entities there
  are exactly ``R * N`` facts and ``fact_id = relation_index * N + entity_index``. Nothing is
  materialised per fact, which is what lets the default 10^4 world scale to 10^6 entities without
  a redesign. :class:`Fact` objects are built on demand at the boundary.

* **Names are keys, not facts.** Every probe is phrased in terms of an entity's name, so counting
  the name as recallable knowledge would double-count it. Names are sampled without replacement
  (probes must be unambiguous) from a space asserted to be >= 100x the entity count. Set
  ``include_name_facts=True`` to add reverse lookup ("who holds record #123?") as a real relation.

* **Entropy is summed self-information.** Each fact carries ``bits = -log2 P(value)`` under the
  distribution it was actually drawn from (see :mod:`fka.data.vocab`), so ``corpus.total_bits`` is
  the corpus's information content, and it stays correct when relations switch from uniform to
  Zipf sampling.

* **Determinism has a caveat.** Same seed and config give the same corpus, but NumPy does not
  guarantee ``Generator`` streams across NumPy versions. :meth:`KnowledgeCorpus.fingerprint`
  exists so a run can record what it actually generated and detect drift later.

CLI::

    python -m fka.data.corpus_gen --n-entities 10000 --seed 0
    python -m fka.data.corpus_gen --smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from fka.data import templates as T
from fka.data.vocab import (
    CoworkerSetSpace,
    UniformSpace,
    UniqueNameSpace,
    ValueSpace,
    ZipfSpace,
    cities,
    employers,
    given_names,
    surnames,
    years,
)

#: Multi-valued answers are rendered by joining with this, and scored after sorting.
COWORKER_JOINER = " and "

#: Minimum ratio of name-space size to entity count. Guarantees name collisions are rare enough
#: that per-name self-information (log2 capacity) is within ~0.015 bits of the exact
#: without-replacement value. See UniqueNameSpace's docstring.
NAME_SPACE_HEADROOM = 100


@dataclass(frozen=True)
class CorpusConfig:
    """Everything needed to regenerate a corpus byte-for-byte (given a fixed NumPy version)."""

    n_entities: int = 10_000
    seed: int = 0
    relations: tuple[str, ...] = T.DEFAULT_RELATIONS
    include_name_facts: bool = False

    # value-space sizes
    n_given_names: int = 4096
    n_surnames: int = 4096
    n_cities: int = 512
    n_employers: int = 1024
    birth_year_range: tuple[int, int] = (1900, 2000)
    n_coworkers: int = 2

    # sampling distributions: "uniform" or "zipf"
    birth_city_distribution: str = "uniform"
    employer_distribution: str = "uniform"
    zipf_s: float = 1.0

    # rendering and splits
    statements_per_fact: int = 1
    probe_fraction: float = 0.1
    heldout_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.n_entities < 2:
            raise ValueError("need at least 2 entities (works_with requires a partner)")
        for r in self.relations:
            T.relation_templates(r)  # raises on unknown relation
        if len(set(self.relations)) != len(self.relations):
            raise ValueError(f"duplicate relation in {self.relations}")
        if not 0.0 <= self.probe_fraction <= 1.0:
            raise ValueError("probe_fraction must be in [0, 1]")
        if not 0.0 <= self.heldout_fraction < 1.0:
            raise ValueError("heldout_fraction must be in [0, 1)")
        if self.statements_per_fact < 1:
            raise ValueError("statements_per_fact must be >= 1")
        for label, mode in (
            ("birth_city_distribution", self.birth_city_distribution),
            ("employer_distribution", self.employer_distribution),
        ):
            if mode not in ("uniform", "zipf"):
                raise ValueError(f"{label} must be 'uniform' or 'zipf', got {mode!r}")

        capacity = self.n_given_names * self.n_surnames
        required = NAME_SPACE_HEADROOM * self.n_entities
        if capacity < required:
            raise ValueError(
                f"name space is too small: {self.n_given_names} given x {self.n_surnames} "
                f"surnames = {capacity:,} combinations, but {self.n_entities:,} entities need "
                f">= {required:,} ({NAME_SPACE_HEADROOM}x headroom) so that sampling without "
                f"replacement stays close to i.i.d. and per-name bits remain exact. "
                f"Raise n_given_names / n_surnames."
            )

    @property
    def active_relations(self) -> tuple[str, ...]:
        if self.include_name_facts and "full_name" not in self.relations:
            return (*self.relations, "full_name")
        return self.relations

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Fact:
    """One atomic fact, materialised at the API boundary."""

    fact_id: int
    subject_id: int
    subject: str
    relation: str
    value: str
    bits: float


@dataclass(frozen=True)
class MemoryPair:
    """A fact as the memory interface sees it: an address, a question, and an answer.

    ``key`` is the structured address ``(subject, relation)`` a router should learn to hash;
    ``query`` is the canonical natural-language form of the same thing.
    """

    fact_id: int
    key: tuple[str, str]
    query: str
    answer: str
    bits: float


class KnowledgeCorpus:
    """A generated world. Construct via :func:`generate_corpus`, not directly."""

    def __init__(
        self,
        config: CorpusConfig,
        name_space: UniqueNameSpace,
        entity_name_idx: np.ndarray,
        spaces: dict[str, ValueSpace],
        values: dict[str, np.ndarray],
        bits: dict[str, np.ndarray],
        variant_offset: np.ndarray,
        probe_mask: np.ndarray,
        heldout_mask: np.ndarray,
    ) -> None:
        self.config = config
        self.relations: tuple[str, ...] = config.active_relations
        self.name_space = name_space
        self.entity_name_idx = entity_name_idx
        self.spaces = spaces
        self.values = values
        self.bits = bits
        self._variant_offset = variant_offset
        self.probe_mask = probe_mask
        self.heldout_mask = heldout_mask
        self._names_cache: tuple[str, ...] | None = None

    # -- basic geometry ------------------------------------------------------------------

    @property
    def n_entities(self) -> int:
        return self.config.n_entities

    @property
    def n_facts(self) -> int:
        return len(self.relations) * self.n_entities

    def relation_of(self, fact_id: int | np.ndarray):
        idx = np.asarray(fact_id) // self.n_entities
        if np.ndim(fact_id) == 0:
            return self.relations[int(idx)]
        return idx

    def subject_of(self, fact_id: int | np.ndarray):
        return np.asarray(fact_id) % self.n_entities

    def fact_ids_for(self, relation: str) -> np.ndarray:
        r = self.relations.index(relation)
        return np.arange(r * self.n_entities, (r + 1) * self.n_entities, dtype=np.int64)

    # -- entity names --------------------------------------------------------------------

    @property
    def entity_names(self) -> tuple[str, ...]:
        if self._names_cache is None:
            self._names_cache = tuple(
                self.name_space.render(int(i)) for i in self.entity_name_idx
            )
        return self._names_cache

    def entity_name(self, entity_id: int) -> str:
        return self.name_space.render(int(self.entity_name_idx[entity_id]))

    def subject_key(self, relation: str, entity_id: int) -> str:
        """What probes address this fact by.

        Every relation is keyed by the entity's name, except ``full_name`` — a reverse lookup,
        which must be keyed by the record id or it would contain its own answer.
        """
        if relation == "full_name":
            return f"#{entity_id:07d}"
        return self.entity_name(entity_id)

    # -- values --------------------------------------------------------------------------

    def value_of(self, relation: str, entity_id: int) -> str:
        """The canonical answer string for one fact."""
        if relation == "works_with":
            partners = sorted(self.entity_name(int(j)) for j in self.values[relation][entity_id])
            return COWORKER_JOINER.join(partners)
        if relation == "full_name":
            return self.entity_name(entity_id)
        space = self.spaces[relation]
        return space.values[int(self.values[relation][entity_id])]

    def bits_of(self, fact_id: int) -> float:
        relation = self.relations[fact_id // self.n_entities]
        return float(self.bits[relation][fact_id % self.n_entities])

    def fact(self, fact_id: int) -> Fact:
        r, e = divmod(int(fact_id), self.n_entities)
        relation = self.relations[r]
        return Fact(
            fact_id=int(fact_id),
            subject_id=e,
            subject=self.subject_key(relation, e),
            relation=relation,
            value=self.value_of(relation, e),
            bits=float(self.bits[relation][e]),
        )

    def facts(self, fact_ids: Sequence[int] | np.ndarray | None = None) -> Iterator[Fact]:
        ids = self.train_ids if fact_ids is None else np.asarray(fact_ids)
        for fid in ids:
            yield self.fact(int(fid))

    # -- entropy accounting --------------------------------------------------------------

    @property
    def total_bits(self) -> float:
        """Information content of every fact in the world, held out or not."""
        return float(sum(self.bits[r].sum() for r in self.relations))

    @property
    def stored_bits(self) -> float:
        """Information content of the facts a model is actually given (excludes held-out)."""
        return self.bits_of_ids(self.train_ids)

    def bits_of_ids(self, fact_ids: Sequence[int] | np.ndarray) -> float:
        ids = np.asarray(fact_ids, dtype=np.int64)
        r, e = np.divmod(ids, self.n_entities)
        table = np.stack([self.bits[rel] for rel in self.relations])
        return float(table[r, e].sum())

    def bits_by_relation(self) -> dict[str, float]:
        return {r: float(self.bits[r].sum()) for r in self.relations}

    # -- splits --------------------------------------------------------------------------

    @property
    def train_ids(self) -> np.ndarray:
        """Facts the model is allowed to learn/store. Probe facts are a subset of these."""
        return np.flatnonzero(~self.heldout_mask)

    @property
    def probe_ids(self) -> np.ndarray:
        """Stored facts we query at evaluation time — the capacity measurement sample."""
        return np.flatnonzero(self.probe_mask)

    @property
    def heldout_ids(self) -> np.ndarray:
        """Facts never shown to the model. Query these for leakage / guessing baselines."""
        return np.flatnonzero(self.heldout_mask)

    # -- surfaces ------------------------------------------------------------------------

    def documents(
        self,
        fact_ids: Sequence[int] | np.ndarray | None = None,
        *,
        firewall: bool = False,
        shuffle: bool = False,
    ) -> Iterator[str]:
        """Render facts as natural-language statements.

        ``firewall=True`` is the *fact firewall* of research plan §2.4: values are replaced by
        empty ``<query>…</query><result></result>`` spans, so a kernel trained on this stream
        meets every atomic fact only through the memory interface and can never memorise one
        from text. Pair it with :meth:`memory_pairs` to supply the answers.
        """
        ids = self.train_ids if fact_ids is None else np.asarray(fact_ids, dtype=np.int64)
        k = self.config.statements_per_fact
        pairs = [(int(fid), j) for fid in ids for j in range(k)]
        if shuffle:
            rng = np.random.default_rng(self.config.seed + 1)
            order = rng.permutation(len(pairs))
            pairs = [pairs[i] for i in order]
        for fid, j in pairs:
            r, e = divmod(fid, self.n_entities)
            relation = self.relations[r]
            variant = int(self._variant_offset[r, e]) + j
            subject = self.subject_key(relation, e)
            if firewall:
                yield T.render_firewalled_statement(relation, subject, variant)
            else:
                yield T.render_statement(relation, subject, self.value_of(relation, e), variant)

    def memory_pairs(
        self, fact_ids: Sequence[int] | np.ndarray | None = None
    ) -> Iterator[MemoryPair]:
        """Facts as ``(query, answer)`` memory-interface pairs, in canonical probe phrasing."""
        ids = self.probe_ids if fact_ids is None else np.asarray(fact_ids, dtype=np.int64)
        for fid in ids:
            r, e = divmod(int(fid), self.n_entities)
            relation = self.relations[r]
            subject = self.subject_key(relation, e)
            yield MemoryPair(
                fact_id=int(fid),
                key=(subject, relation),
                query=T.render_question(relation, subject),
                answer=self.value_of(relation, e),
                bits=float(self.bits[relation][e]),
            )

    def as_oracle_dict(
        self, fact_ids: Sequence[int] | np.ndarray | None = None
    ) -> dict[tuple[str, str], str]:
        """An oracle memory: ``(subject, relation) -> answer`` over the stored facts.

        This is the Phase-1 stand-in memory (research plan §2.4 step 3) and the positive control
        for the capacity harness — a dict must recover ~100% of corpus entropy.
        """
        ids = self.train_ids if fact_ids is None else np.asarray(fact_ids, dtype=np.int64)
        return {p.key: p.answer for p in self.memory_pairs(ids)}

    # -- provenance ----------------------------------------------------------------------

    def fingerprint(self) -> str:
        """Content hash of the generated world.

        Recorded alongside experiment configs so a corpus that silently changed — a NumPy upgrade
        shifting the ``Generator`` stream, an edited value space — is caught rather than quietly
        invalidating a comparison.
        """
        h = hashlib.sha256()
        h.update(json.dumps(self.config.to_dict(), sort_keys=True).encode())
        h.update(self.entity_name_idx.tobytes())
        for r in self.relations:
            h.update(r.encode())
            h.update(np.ascontiguousarray(self.values[r]).tobytes())
            h.update(np.round(self.bits[r], 9).tobytes())
        h.update(self.probe_mask.tobytes())
        h.update(self.heldout_mask.tobytes())
        return h.hexdigest()

    def summary(self) -> dict:
        return {
            "n_entities": self.n_entities,
            "n_facts": self.n_facts,
            "relations": list(self.relations),
            "total_bits": self.total_bits,
            "bits_per_fact": self.total_bits / self.n_facts,
            "bits_by_relation": self.bits_by_relation(),
            "n_train": int(len(self.train_ids)),
            "n_probe": int(len(self.probe_mask.nonzero()[0])),
            "n_heldout": int(len(self.heldout_mask.nonzero()[0])),
            "fingerprint": self.fingerprint(),
        }

    def __repr__(self) -> str:
        return (
            f"KnowledgeCorpus(n_entities={self.n_entities:,}, n_facts={self.n_facts:,}, "
            f"total_bits={self.total_bits:,.0f})"
        )


# =======================================================================================
# generation
# =======================================================================================


def _build_space(relation: str, config: CorpusConfig) -> ValueSpace:
    if relation == "birth_year":
        return UniformSpace(years(*config.birth_year_range), "birth_year")
    if relation == "birth_city":
        values = cities(config.n_cities)
        if config.birth_city_distribution == "zipf":
            return ZipfSpace(values, "birth_city", s=config.zipf_s)
        return UniformSpace(values, "birth_city")
    if relation == "employer":
        values = employers(config.n_employers)
        if config.employer_distribution == "zipf":
            return ZipfSpace(values, "employer", s=config.zipf_s)
        return UniformSpace(values, "employer")
    if relation == "works_with":
        return CoworkerSetSpace(config.n_entities, config.n_coworkers)
    if relation == "full_name":
        return UniqueNameSpace(
            given_names(config.n_given_names), surnames(config.n_surnames)
        )
    raise KeyError(f"no value space defined for relation {relation!r}")


def generate_corpus(config: CorpusConfig | None = None, **overrides) -> KnowledgeCorpus:
    """Generate a knowledge world. Same ``(config, seed)`` gives the same world.

    ``overrides`` are applied on top of ``config`` (or the defaults), so
    ``generate_corpus(n_entities=1000, seed=3)`` works without building a config first.
    """
    config = config or CorpusConfig()
    if overrides:
        config = replace(config, **overrides)

    rng = np.random.default_rng(config.seed)
    n = config.n_entities
    relations = config.active_relations

    name_space = UniqueNameSpace(given_names(config.n_given_names), surnames(config.n_surnames))
    entity_name_idx = name_space.sample_indices(rng, n)

    spaces: dict[str, ValueSpace] = {}
    values: dict[str, np.ndarray] = {}
    bits: dict[str, np.ndarray] = {}
    for relation in relations:
        space = _build_space(relation, config)
        spaces[relation] = space
        if relation == "works_with":
            drawn = space.sample_indices(rng, n)
            values[relation] = drawn
            bits[relation] = np.full(n, space.bits(), dtype=np.float64)
        elif relation == "full_name":
            # The name was already drawn as the entity key; as a fact it just points back at it.
            values[relation] = entity_name_idx
            bits[relation] = np.full(n, space.bits(""), dtype=np.float64)
        else:
            drawn = space.sample_indices(rng, n)
            values[relation] = drawn
            bits[relation] = space.bits_of_indices(drawn).astype(np.float64)

    # Which template each fact leads with; +j gives the j-th paraphrase.
    variant_offset = rng.integers(0, 256, size=(len(relations), n), dtype=np.uint8)

    # Splits. One uniform draw per fact keeps the assignment independent of relation order.
    u = rng.random(len(relations) * n)
    heldout_mask = u < config.heldout_fraction
    retained = ~heldout_mask
    probe_cut = config.heldout_fraction + config.probe_fraction * (1.0 - config.heldout_fraction)
    probe_mask = retained & (u < probe_cut)

    return KnowledgeCorpus(
        config=config,
        name_space=name_space,
        entity_name_idx=entity_name_idx,
        spaces=spaces,
        values=values,
        bits=bits,
        variant_offset=variant_offset,
        probe_mask=probe_mask,
        heldout_mask=heldout_mask,
    )


# =======================================================================================
# CLI
# =======================================================================================


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate and inspect a synthetic knowledge world.")
    p.add_argument("--n-entities", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-coworkers", type=int, default=2)
    p.add_argument("--statements-per-fact", type=int, default=1)
    p.add_argument("--probe-fraction", type=float, default=0.1)
    p.add_argument("--heldout-fraction", type=float, default=0.0)
    p.add_argument("--birth-city-distribution", choices=("uniform", "zipf"), default="uniform")
    p.add_argument("--employer-distribution", choices=("uniform", "zipf"), default="uniform")
    p.add_argument("--zipf-s", type=float, default=1.0)
    p.add_argument("--include-name-facts", action="store_true")
    p.add_argument("--show", type=int, default=5, help="sample documents to print")
    p.add_argument("--summary-json", type=str, default=None)
    p.add_argument("--smoke", action="store_true", help="tiny world; runs in well under a second")
    args = p.parse_args(argv)

    if args.smoke:
        args.n_entities, args.show = 200, 3

    config = CorpusConfig(
        n_entities=args.n_entities,
        seed=args.seed,
        n_coworkers=args.n_coworkers,
        statements_per_fact=args.statements_per_fact,
        probe_fraction=args.probe_fraction,
        heldout_fraction=args.heldout_fraction,
        birth_city_distribution=args.birth_city_distribution,
        employer_distribution=args.employer_distribution,
        zipf_s=args.zipf_s,
        include_name_facts=args.include_name_facts,
    )
    corpus = generate_corpus(config)
    summary = corpus.summary()

    print(f"== {corpus!r}")
    for key, value in summary.items():
        if key == "bits_by_relation":
            print("   bits_by_relation:")
            for r, b in value.items():
                print(f"       {r:14s} {b:14,.1f}  ({b / corpus.n_entities:6.3f} bits/entity)")
        else:
            print(f"   {key:18s} {value if not isinstance(value, float) else f'{value:,.2f}'}")

    if args.show:
        print("\n-- sample documents")
        for doc in list(corpus.documents(corpus.train_ids[: args.show * 4]))[: args.show]:
            print(f"   {doc}")
        print("\n-- same facts, firewalled")
        for doc in list(
            corpus.documents(corpus.train_ids[: args.show * 4], firewall=True)
        )[: args.show]:
            print(f"   {doc}")
        print("\n-- probe pairs")
        for pair in list(corpus.memory_pairs())[: args.show]:
            print(f"   Q: {pair.query}\n   A: {pair.answer}   ({pair.bits:.2f} bits)")

    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nwrote {args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
