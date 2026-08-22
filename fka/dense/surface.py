"""Rendering surfaces for the dense baseline — the token-density axis (M5 §5.6).

A surface decides how a fact becomes *text*. It decides nothing else: the facts, the value spaces,
and every per-fact bit count are properties of :mod:`fka.data.corpus_gen` and are untouched here.
That separation is what makes the axis safe to move, and it is worth stating precisely because a
change to the corpus's information content would invalidate every capacity number in the program.

**(a) Entropy accounting is untouched.** `bits(value) = -log2 P(value)` under the distribution
`sample()` draws from. A surface changes the *characters* a fact is written in, never the value
space it is drawn from, never which value was drawn, and never how many of them there are. The
corpus fingerprint is identical across surfaces; `corpus.total_bits` is identical across surfaces.

**(c) FKA-side capacity numbers are rendering-independent, structurally.** Our architecture never
reads a rendered document. The substrate is populated programmatically from
`(entity_id, relation) -> value` and keys are computed from `store.reconstruct()`; M3's entire
measurement chain — 52 bits/entity, 0.4387 marginal, the bits-vs-N curve — runs off the corpus
object, not off text. Rendering feeds exactly two consumers: the **dense baseline's training
stream** and the **probe surface**. So moving this axis cannot move an FKA number, and no result
already recorded needs re-deriving.

**(b) Before/after bits/token is measured**, not asserted — `bits_per_token()`, reported for every
surface at every N used.

Why the axis exists at all: at ~56 tokens/fact and ~13 bits/fact this corpus carries **0.16
bits/token**, and most of those tokens are *key material* — entity names, which M0's own rule
excludes from the corpus's information content ("names are keys, not facts"). The dense baseline
was therefore spending the majority of its compute on characters that carry no measured bits, and
that is what put a saturated 150M run 43.8 days out of reach (M5 §5.4).

**The terse surface is a candidate, not an assumption.** Shortening the key surface could in
principle make the task *harder* for SGD rather than cheaper — a 7-digit record id and a two-word
name carry about the same ~23 bits, but nothing guarantees they are equally learnable. So the
switch is gated on a measured **surface-equivalence control** (§5.6.2), not on this argument.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

import numpy as np

from fka.data import templates as T
from fka.data.corpus_gen import KnowledgeCorpus
from fka.data.tokenizer import DEFAULT_CHARS, DEFAULT_SPECIALS, CharTokenizer
from fka.dense.syllables import SYLLABLE_CHARS, name_for, readable, units_needed

QUESTION_PREFIX = "Q: "
ANSWER_PREFIX = " A: "

#: One character per relation for the terse surface. Every code is inside the corpus charset, so
#: the tokenizer is unchanged — a surface may not alter the vocabulary either.
RELATION_CODE: dict[str, str] = {
    "birth_year": "y",
    "birth_city": "c",
    "employer": "e",
    "works_with": "w",
    "full_name": "n",
}


@runtime_checkable
class Surface(Protocol):
    """How facts are written. Values are never rewritten — the frozen scorer compares them."""

    name: str
    #: The delimiter after which a probe's answer begins. No value may contain it, because the
    #: gold stub locates the prompt boundary by its last occurrence.
    answer_prefix: str

    def n_variants(self, relation: str) -> int: ...
    def subjects(self, corpus: KnowledgeCorpus, relation: str) -> list[str]: ...
    def statement(self, relation: str, subject: str, value: str, variant: int) -> str: ...
    def probe_prompt(self, relation: str, subject: str) -> str: ...

    def qa_line(self, relation: str, subject: str, value: str) -> str:
        """The QA training line. Must be exactly ``probe_prompt(...) + value``."""


@dataclass(frozen=True)
class VerboseSurface:
    """Natural language — the M1/M0 surface, and the default. Five paraphrases per relation."""

    name: str = "verbose"
    answer_prefix: str = ANSWER_PREFIX

    def n_variants(self, relation: str) -> int:
        return len(T.relation_templates(relation).statements)

    def subjects(self, corpus: KnowledgeCorpus, relation: str) -> list[str]:
        return [corpus.subject_key(relation, e) for e in range(corpus.n_entities)]

    def statement(self, relation: str, subject: str, value: str, variant: int) -> str:
        return T.render_statement(relation, subject, value, variant)

    def probe_prompt(self, relation: str, subject: str) -> str:
        return f"{QUESTION_PREFIX}{T.render_question(relation, subject)}{ANSWER_PREFIX}"

    def qa_line(self, relation: str, subject: str, value: str) -> str:
        return self.probe_prompt(relation, subject) + value


#: Structural paraphrases for the terse surface. Variety is part of the registered exposure regime
#: (epoch j uses template j mod n_variants), so a surface with a single form would confound the
#: density change with a collapse of the variety knob. Field ORDER varies, mirroring the verbose
#: templates' "Born in {value}, {subject}..." inversions.
_TERSE_STATEMENTS: tuple[str, ...] = (
    "{subject}|{code}|{value}",
    "{subject} {code} {value}",
    "{code}|{subject}|{value}",
    "{subject}.{code}.{value}",
    "{value}|{code}|{subject}",
)


@dataclass(frozen=True)
class TerseSurface:
    """Field-delimited records. Same facts, same values, far fewer characters of scaffolding.

    ``subject_style`` is the part that actually moves the needle and the part that carries risk:

    * ``"name"`` keeps the entity name and strips only English scaffolding (~1.4x denser);
    * ``"record_id"`` addresses the entity by ``#0001234`` — the corpus's *own* canonical
      record-id form, already used by ``subject_key`` for ``full_name`` — which is ~1.9x denser
      because it replaces ~18 characters of key material with 8 carrying the same ~23 bits.

    Values are never touched, so ``fka.eval.capacity``'s scorer applies unchanged and there is no
    second scoring path to keep in sync (CLAUDE.md: one gate per deployed eval path).
    """

    subject_style: str = "record_id"
    name: str = "terse"
    answer_prefix: str = ":"

    def __post_init__(self) -> None:
        if self.subject_style not in ("name", "record_id"):
            raise ValueError(f"unknown subject_style {self.subject_style!r}")

    def n_variants(self, relation: str) -> int:
        return len(_TERSE_STATEMENTS)

    def subjects(self, corpus: KnowledgeCorpus, relation: str) -> list[str]:
        if self.subject_style == "name":
            return [corpus.subject_key(relation, e) for e in range(corpus.n_entities)]
        width = max(7, len(str(corpus.n_entities - 1)))
        return [f"#{e:0{width}d}" for e in range(corpus.n_entities)]

    def statement(self, relation: str, subject: str, value: str, variant: int) -> str:
        return _TERSE_STATEMENTS[variant % len(_TERSE_STATEMENTS)].format(
            subject=subject, code=RELATION_CODE[relation], value=value
        )

    def probe_prompt(self, relation: str, subject: str) -> str:
        return f"?{subject}|{RELATION_CODE[relation]}{self.answer_prefix}"

    def qa_line(self, relation: str, subject: str, value: str) -> str:
        return self.probe_prompt(relation, subject) + value


#: Injective scramble from entity index to name index. Without it, adjacent entities would get
#: adjacent (and near-identical) names, manufacturing structure the world does not have. The
#: multiplier is odd and every capacity is a power of 608 = 2^5 * 19, so the map is a bijection.
_SCRAMBLE = 2654435761


@dataclass(frozen=True)
class SyllableSurface:
    """The FAIR surface (M5 §5.32): entities spelled in a few units from a FIXED inventory.

    A name is ``units_needed(N)`` syllables — 2 at N <= 369,664, 3 to 225M — each rendered as a
    single private-use codepoint, so **one syllable is one character and one token**. Token count
    per name therefore grows only *logarithmically* in N while the vocabulary never moves, which is
    what BPE does and what name-part atomic could not (§5.32).

    Field-delimited templates, inherited from the terse surface: §5.27.3 rules that the density
    revision and the tokenizer revision ship as **one** change, not two.

    **Entity-valued relations are refused.** ``works_with``'s value is another entity, and rendering
    it would give the same entity two names — its corpus spelling as a value, its syllable spelling
    as a subject. A value renderer is the next build item; refusing is the honest interim.
    """

    name: str = "syllable"
    answer_prefix: str = ":"

    #: Override the name length in syllables. ``None`` uses the minimum that addresses the corpus
    #: distinctly, which is what every measurement to date used. The override exists for M5 §5.57's
    #: discriminator: at N = 288,000 the 2-unit space is 77.9% full, so key crowding and *name-space
    #: saturation* co-vary exactly, and only forcing a wider name separates them. It is a genuine
    #: surface change — tokens/fact rises — so it moves ``surface_fingerprint`` and never the world.
    name_units: int | None = None

    def n_variants(self, relation: str) -> int:
        return len(_TERSE_STATEMENTS)

    def units_for(self, n_entities: int) -> int:
        units = units_needed(n_entities) if self.name_units is None else self.name_units
        if len(SYLLABLE_CHARS) ** units < n_entities:
            raise ValueError(
                f"name_units={units} addresses only {len(SYLLABLE_CHARS) ** units:,} names, "
                f"which cannot spell {n_entities:,} entities distinctly"
            )
        return units

    def subjects(self, corpus: KnowledgeCorpus, relation: str) -> list[str]:
        if relation == "works_with":
            raise ValueError(
                "SyllableSurface cannot render works_with: its value is an entity, which would be "
                "spelled as a corpus name while the same entity is spelled as syllables when it is "
                "a subject. Supply a value renderer first (M5 §5.32 scope limit)."
            )
        n = corpus.n_entities
        units = self.units_for(n)
        capacity = len(SYLLABLE_CHARS) ** units
        return [name_for((e * _SCRAMBLE) % capacity, units) for e in range(n)]

    def statement(self, relation: str, subject: str, value: str, variant: int) -> str:
        return _TERSE_STATEMENTS[variant % len(_TERSE_STATEMENTS)].format(
            subject=subject, code=RELATION_CODE[relation], value=value
        )

    def probe_prompt(self, relation: str, subject: str) -> str:
        return f"?{subject}|{RELATION_CODE[relation]}{self.answer_prefix}"

    def qa_line(self, relation: str, subject: str, value: str) -> str:
        return self.probe_prompt(relation, subject) + value


def syllable_tokenizer() -> CharTokenizer:
    """The charset plus one atomic character per syllable. Vocabulary is FIXED at 688."""
    return CharTokenizer(
        chars=tuple(DEFAULT_CHARS) + SYLLABLE_CHARS, specials=DEFAULT_SPECIALS
    )


@dataclass(frozen=True)
class BPESurface:
    """The syllable world spelled in LETTERS, for a BPE to fragment as it will (M5 §5.132).

    Identical to :class:`SyllableSurface` in every respect except the alphabet: the same entity
    index, the same scramble, the same terse templates, the same values — but names are rendered
    through :func:`readable`, so ```` becomes ``"baba"``.

    **This is required, not cosmetic.** A BPE trained on English has no private-use codepoints in
    its base vocabulary, so the syllable rendering would encode entirely to ``<unk>`` — an
    unlearnable stream that would have read 0% and looked like a capacity result. Caught by the
    stream's roundtrip test before any arm ran.

    Surface coordinates (M5 §5.112) at 2,000 entities: **4.263 tokens/name at vocab 735**,
    **3.877 at 3,193**, against syllable's 2.000 and a character surface's 6.780.
    """

    name: str = "bpe"
    answer_prefix: str = ":"
    name_units: int | None = None

    def n_variants(self, relation: str) -> int:
        return len(_TERSE_STATEMENTS)

    def units_for(self, n_entities: int) -> int:
        units = units_needed(n_entities) if self.name_units is None else self.name_units
        if len(SYLLABLE_CHARS) ** units < n_entities:
            raise ValueError(
                f"name_units={units} addresses only {len(SYLLABLE_CHARS) ** units:,} names, "
                f"which cannot spell {n_entities:,} entities distinctly"
            )
        return units

    def subjects(self, corpus: KnowledgeCorpus, relation: str) -> list[str]:
        if relation == "works_with":
            raise ValueError(
                "BPESurface cannot render works_with, for the same reason SyllableSurface cannot: "
                "its value is an entity, which would be spelled two different ways (M5 §5.32)."
            )
        n = corpus.n_entities
        units = self.units_for(n)
        capacity = len(SYLLABLE_CHARS) ** units
        return [readable(name_for((e * _SCRAMBLE) % capacity, units)) for e in range(n)]

    def statement(self, relation: str, subject: str, value: str, variant: int) -> str:
        return _TERSE_STATEMENTS[variant % len(_TERSE_STATEMENTS)].format(
            subject=subject, code=RELATION_CODE[relation], value=value
        )

    def probe_prompt(self, relation: str, subject: str) -> str:
        return f"?{subject}|{RELATION_CODE[relation]}{self.answer_prefix}"

    def qa_line(self, relation: str, subject: str, value: str) -> str:
        return self.probe_prompt(relation, subject) + value


SURFACES: dict[str, Surface] = {
    "bpe": BPESurface(),
    "syllable": SyllableSurface(),
    "verbose": VerboseSurface(),
    "terse": TerseSurface(subject_style="record_id"),
    "terse_named": TerseSurface(subject_style="name"),
}


def surface_for(name: str, *, name_units: int | None = None) -> Surface:
    try:
        surface = SURFACES[name]
    except KeyError:
        raise KeyError(f"unknown surface {name!r}; known: {sorted(SURFACES)}") from None
    if name_units is None:
        return surface
    if not isinstance(surface, SyllableSurface):
        raise ValueError(f"name_units applies to the syllable surface only, not {name!r}")
    return replace(surface, name_units=name_units)


def assert_answer_prefix_is_unambiguous(surface: Surface, corpus: KnowledgeCorpus) -> None:
    """No value may contain the answer delimiter, or prompt/answer cannot be split back apart.

    The gold stub finds the prompt boundary with ``rfind(answer_prefix)``; a value containing the
    delimiter would silently shift that boundary and the stub would score a correct path wrong.
    """
    for relation in corpus.relations:
        for entity in range(min(corpus.n_entities, 512)):
            value = corpus.value_of(relation, entity)
            if surface.answer_prefix in value:
                raise ValueError(
                    f"{surface.name}: value {value!r} for {relation} contains the answer "
                    f"delimiter {surface.answer_prefix!r}"
                )


def surface_fingerprint(
    surface: Surface,
    tokenizer: CharTokenizer,
    corpus: KnowledgeCorpus,
    *,
    n_sample: int = 64,
    seed: int = 0,
) -> str:
    """Content hash of the RENDERING — the companion to `corpus.fingerprint()` (M5 §5.31).

    `corpus.fingerprint()` pins the **world**: config, entity name *indices*, values, bits, split
    masks. By design it never touches rendered strings, so it is **structurally blind** to a change
    of tokenizer, template, or name construction. A corpus generator edit that produced entirely
    different name strings for the same entities would leave it byte-identical.

    This pins the **surface**: the tokenizer's vocabulary, every statement template and probe prompt
    form, and a deterministic sample of actually-rendered facts. Between the two, a change is
    visible in exactly one — which is what makes "the world is unchanged, only the characters moved"
    a checkable claim rather than an assurance.

    The rendered sample is the load-bearing part. Templates alone would miss a change in how
    *subjects* are constructed, which is precisely the corpus-generator change under consideration.
    """
    h = hashlib.sha256()
    h.update(surface.name.encode())
    h.update(surface.answer_prefix.encode())
    h.update(repr(tokenizer.itos).encode())
    for relation in sorted(corpus.relations):
        h.update(relation.encode())
        for variant in range(surface.n_variants(relation)):
            h.update(surface.statement(relation, "<S>", "<V>", variant).encode())
        h.update(surface.probe_prompt(relation, "<S>").encode())

    rng = np.random.default_rng(seed)
    picks = np.sort(rng.choice(corpus.n_facts, size=min(n_sample, corpus.n_facts), replace=False))
    subjects: dict[str, list[str]] = {}
    for fact_id in picks:
        r, e = divmod(int(fact_id), corpus.n_entities)
        relation = corpus.relations[r]
        if relation not in subjects:
            subjects[relation] = surface.subjects(corpus, relation)
        value = corpus.value_of(relation, e)
        h.update(surface.statement(relation, subjects[relation][e], value, 0).encode())
        h.update(surface.qa_line(relation, subjects[relation][e], value).encode())
    return h.hexdigest()


__all__ = [
    "ANSWER_PREFIX",
    "QUESTION_PREFIX",
    "RELATION_CODE",
    "SURFACES",
    "Surface",
    "SyllableSurface",
    "TerseSurface",
    "VerboseSurface",
    "assert_answer_prefix_is_unambiguous",
    "surface_fingerprint",
    "surface_for",
    "syllable_tokenizer",
]
