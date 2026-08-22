"""The dense baseline's token stream: facts in the text, firewall OFF (M5 §4.1).

Three deliberate design decisions, each of which would otherwise silently handicap the baseline.

**1. The fact firewall is OFF and that is the whole point.** M1's loss mask exists to stop *our*
kernel memorising facts. Applying it here would rig the comparison: storing facts in weights is
precisely the dense baseline's job. So every statement carries its value as literal text, every
token is trained, and no ``<query>``/``<result>`` marker appears anywhere in the stream. The
absence of markers is asserted, not assumed — a stray one would mean the two systems are not
reading the same world.

**2. Exposures are epochs, and template variety is a first-class knob.** The capacity literature
(Allen-Zhu & Li, *Physics of Language Models* 3.3) reports that a dense transformer approaches its
~2 bits/parameter ceiling only after **~1000 exposures per fact**, sits materially below it at
~100, and that how many *distinct* renderings a fact appears in changes how much of it is stored.
A baseline trained for a handful of passes is under-exposed, and an under-exposed baseline loses
for a reason that has nothing to do with the architecture. So epoch ``j`` renders each fact with
template ``(offset + j) mod n_variants``: exposures and variety move together the way they do in
the literature this number is compared against.

**3. The probe format is trained on a DISJOINT entity split, at EQUAL exposure.** A model that has
only ever seen declarative statements will fail a question it has never been asked, and that
failure is a format failure being scored as a storage failure. The literature's own protocol is to
teach the QA surface on some entities and probe it on others, so only ``qa_entity_fraction`` of
entities ever appear as ``Q: … A: …`` pairs and probes are drawn **only from the complement**.

The QA rendering **replaces** one statement rather than adding to it (one exposure in
``qa_period``). Appending it would give QA-train entities more total exposures than probe
entities, and then the two halves would differ in *exposure* as well as *surface* — which matters
because the headline extrapolates the probe half's storage rate over the whole corpus. With the
substitution, both halves receive exactly ``exposures`` renderings per fact and that
extrapolation is a statement about surface only. The QA-trained half is still scored, separately
and labelled, as the registered check on it.

Rendering is cached per template variant. Text for epoch ``j`` depends on ``j`` only through
``j mod variant_period``, so at most ``variant_period`` buffers are ever built; epochs differ after
that by window ordering and stream offset, which the trainer owns.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import reduce

import numpy as np

from fka.data import templates as T
from fka.data.corpus_gen import KnowledgeCorpus
from fka.data.tokenizer import CharTokenizer
from fka.dense.surface import (
    ANSWER_PREFIX,
    QUESTION_PREFIX,
    Surface,
    assert_answer_prefix_is_unambiguous,
    surface_fingerprint,
    surface_for,
)

#: Placeholder byte for the document separator, chosen outside the corpus charset so it cannot
#: collide with real text. Mapped to ``<eos>`` by the encoding LUT.
_SEP_BYTE = 0x01
_SEP = chr(_SEP_BYTE)

#: Cap on how many distinct rendered epochs are cached. The natural period is the lcm of the
#: per-relation template counts (5 for the default relations, 15 once ``full_name`` joins them).
MAX_VARIANT_PERIOD = 30

__all_prefixes__ = (QUESTION_PREFIX, ANSWER_PREFIX)  # re-exported for callers that import here


@dataclass(frozen=True)
class DenseDataConfig:
    """The exposure regime. Pre-registered before generation (M5 §4.2)."""

    #: Passes over the fact set. One pass = one exposure per fact.
    exposures: int = 100
    #: Distinct statement paraphrases used per relation. ``None`` = every template available.
    n_variants: int | None = None
    #: Share of entities whose facts appear in QA form. Probes come from the complement.
    qa_entity_fraction: float = 0.5
    #: For a QA-train fact, one exposure in ``qa_period`` is rendered as QA *instead of* a
    #: statement. Default 5 matches the template count, so the rendering period stays 5 and the
    #: number of cached epoch buffers does not multiply. 0 disables the QA channel entirely.
    qa_period: int = 5
    #: Rendering surface (:mod:`fka.dense.surface`). Changes characters, never facts or bits.
    #: **Never changed mid-run** — it is a between-ladder axis (M5 §5.6).
    surface: str = "verbose"
    #: Syllables per entity name; ``None`` = the minimum that addresses the corpus. See M5 §5.57 —
    #: forcing a wider name separates key crowding from name-space saturation, which co-vary at
    #: the default width. Syllable surface only.
    name_units: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.exposures < 1:
            raise ValueError("exposures must be >= 1")
        if self.n_variants is not None and self.n_variants < 1:
            raise ValueError("n_variants must be >= 1 or None")
        if not 0.0 <= self.qa_entity_fraction < 1.0:
            raise ValueError(
                "qa_entity_fraction must be in [0, 1): with no held-out entities every probe "
                "would ask a question the model was trained to answer"
            )
        if self.qa_period < 0:
            raise ValueError("qa_period must be >= 0")

    def to_dict(self) -> dict:
        return asdict(self)


def _encoding_lut(tokenizer: CharTokenizer) -> tuple[np.ndarray, str, np.dtype]:
    """Codepoint -> token id, so a 100M-character epoch encodes as one vectorised gather.

    ``CharTokenizer.encode`` is a per-character Python loop with a regex pass for specials — correct
    and far too slow at exposure-ladder volumes. This is the same mapping over the single-character
    part of the vocabulary, with one reserved codepoint for the document separator.

    Two widths, because the syllable surface (M5 §5.32) needs 689 symbols and the character surface
    needs 80:

    * **narrow** (<= 256 symbols): latin-1 bytes -> ``uint8`` ids, a 256-entry table;
    * **wide**: UTF-16LE code units -> ``uint16`` ids, a 65,536-entry table. Every symbol this
      program uses is ASCII or private-use, both inside the BMP, so one code unit is one character.
    """
    narrow = tokenizer.vocab_size <= 256
    size, encoding = (256, "latin-1") if narrow else (1 << 16, "utf-16-le")
    dtype = np.uint8 if narrow else np.uint16
    lut = np.full(size, tokenizer.unk_id, dtype=dtype)
    for symbol, index in tokenizer.stoi.items():
        if len(symbol) == 1 and ord(symbol) < size:
            lut[ord(symbol)] = index
    lut[_SEP_BYTE] = tokenizer.eos_id
    return lut, encoding, dtype


class DenseCorpusStream:
    """Renders a :class:`KnowledgeCorpus` as a plain-LM token stream.

    Construct once; call :meth:`epoch_tokens` per epoch. Values are stringified once at
    construction because ``works_with`` renders two entity names per fact and doing that inside
    the epoch loop dominates the cost at scale.
    """

    def __init__(
        self,
        corpus: KnowledgeCorpus,
        tokenizer: CharTokenizer,
        cfg: DenseDataConfig | None = None,
    ) -> None:
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.cfg = cfg or DenseDataConfig()
        # A BPE tokenizer is NOT one-char-one-token, so the LUT fast path cannot represent it
        # (M5 §5.132.5). Detected by asking the tokenizer, never by checking its class name.
        self._char_level = all(len(sym) == 1 for sym in tokenizer.stoi if not sym.startswith("<"))
        self._lut, self._encoding, self._token_dtype = (
            _encoding_lut(tokenizer) if self._char_level else (None, None, np.int32)
        )
        self._buffers: dict[int, np.ndarray] = {}

        n = corpus.n_entities
        rng = np.random.default_rng(self.cfg.seed ^ 0x5EED)
        n_qa = int(round(self.cfg.qa_entity_fraction * n))
        order = rng.permutation(n)
        self.qa_train_entities: np.ndarray = np.zeros(n, dtype=bool)
        self.qa_train_entities[order[:n_qa]] = True

        self.surface: Surface = surface_for(self.cfg.surface, name_units=self.cfg.name_units)
        assert_answer_prefix_is_unambiguous(self.surface, corpus)

        # Per-relation surface material, computed once.
        self._subjects: dict[str, list[str]] = {}
        self._values: dict[str, list[str]] = {}
        self._n_variants: dict[str, int] = {}
        for relation in corpus.relations:
            available = self.surface.n_variants(relation)
            self._n_variants[relation] = (
                available if self.cfg.n_variants is None else min(self.cfg.n_variants, available)
            )
            self._subjects[relation] = self.surface.subjects(corpus, relation)
            self._values[relation] = [corpus.value_of(relation, e) for e in range(n)]

        period = reduce(math.lcm, self._n_variants.values(), 1)
        if self.cfg.qa_period:
            period = math.lcm(period, self.cfg.qa_period)
        if period > MAX_VARIANT_PERIOD:
            raise ValueError(
                f"rendering period {period} exceeds {MAX_VARIANT_PERIOD}: qa_period="
                f"{self.cfg.qa_period} against template counts {sorted(set(self._n_variants.values()))} "
                f"multiplies the cached epoch buffers. Pick a qa_period that divides the template "
                f"count (5 by default)."
            )
        self.variant_period = period

        # Only facts the model is given. Held-out facts stay out of the stream entirely.
        self._train_mask = ~corpus.heldout_mask

    # -- geometry ------------------------------------------------------------------------

    @property
    def n_relations(self) -> int:
        return len(self.corpus.relations)

    def probe_fact_ids(self, *, qa_trained: bool = False) -> np.ndarray:
        """Probe facts, restricted to one side of the QA entity split.

        ``qa_trained=False`` is the headline set: entities whose questions were never answered in
        training, so a correct answer can only have come from the statement channel.
        """
        ids = self.corpus.probe_ids
        entity = self.corpus.subject_of(ids)
        keep = self.qa_train_entities[entity] if qa_trained else ~self.qa_train_entities[entity]
        return ids[keep]

    # -- rendering -----------------------------------------------------------------------

    def _documents(self, variant: int) -> list[str]:
        """One rendering of every fact this variant. Exactly one line per fact, QA or statement."""
        corpus, cfg = self.corpus, self.cfg
        n = corpus.n_entities
        qa_period = cfg.qa_period
        lines: list[str] = []
        surface = self.surface
        for r_index, relation in enumerate(corpus.relations):
            period = self._n_variants[relation]
            offsets = corpus._variant_offset[r_index]
            subjects = self._subjects[relation]
            values = self._values[relation]
            base = r_index * n
            for e in range(n):
                if not self._train_mask[base + e]:
                    continue
                phase = int(offsets[e]) + variant
                if qa_period and self.qa_train_entities[e] and phase % qa_period == 0:
                    lines.append(surface.qa_line(relation, subjects[e], values[e]))
                else:
                    lines.append(
                        surface.statement(relation, subjects[e], values[e], phase % period)
                    )
        return lines

    def _variant_tokens(self, variant: int, epoch: int | None = None) -> np.ndarray:
        """Tokens for one exposure. ``epoch`` selects the SHUFFLE; ``variant`` selects the wording.

        **The shuffle must not repeat (M5 §5.63).** Seeding it on ``variant`` alone gives exactly
        ``variant_period`` distinct orderings — five — each presented ~107 times byte-identical at
        E = 535. A 10M-parameter model memorises that stream instead of the facts: its subject-token
        loss sits 4.15 nats below uniform on the trained ordering and 10.66 nats *above* uniform on
        a fresh one, while its *value* loss is worse than a 1M model's. It reported a 2.2x lower
        training loss and 7.4x lower recall, which reads as a scaling result and is a data-loader
        artifact.

        So the order is seeded on the **epoch**, and the cache keeps only the wording-dependent work.
        Re-tokenising costs ~0.1 s per epoch against ~11 s of compute at 10M.
        """
        lines = self._documents(variant)
        # Shuffle per EPOCH, so no ordering is ever presented twice.
        key = variant if epoch is None else epoch
        rng = np.random.default_rng((self.cfg.seed, key, 0xD0C))
        order = rng.permutation(len(lines))
        text = _SEP.join(lines[i] for i in order) + _SEP
        if self._char_level:
            raw = np.frombuffer(text.encode(self._encoding), dtype=self._lut.dtype.type
                                if self._lut.dtype == np.uint8 else np.uint16)
            ids = self._lut[raw]
        else:
            # Multi-character tokens: the vectorised gather cannot apply, so encode per line and
            # splice the separator in as its own id. Slower by ~30x, which is affordable because
            # the BPE arms are 1M/10M and the epoch is re-rendered once per exposure anyway.
            pieces = []
            sep = np.array([self.tokenizer.eos_id], dtype=np.int32)
            for line in text.split(_SEP):
                if line:
                    pieces.append(np.asarray(self.tokenizer.encode(line), dtype=np.int32))
                pieces.append(sep)
            ids = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int32)
        if bool((ids == self.tokenizer.unk_id).any()):
            if self._char_level:
                bad = sorted({chr(c) for c in raw[ids == self.tokenizer.unk_id]})
                raise ValueError(f"characters outside the tokenizer charset in the stream: {bad}")
            # Multi-character tokenizer: report the offending SUBSTRINGS, since there is no
            # character-to-id correspondence to point at. A BPE whose base vocabulary misses a
            # surface character produces <unk> here rather than an unlearnable silent failure.
            missing = sorted({c for c in text if c not in self.tokenizer.stoi and c != _SEP})
            raise ValueError(
                f"{int((ids == self.tokenizer.unk_id).sum())} <unk> tokens in the stream; "
                f"characters absent from the tokenizer's base vocabulary: {missing}"
            )
        self._assert_no_markers(ids)
        return ids

    def _assert_no_markers(self, ids: np.ndarray) -> None:
        """The firewall is OFF, so no interface marker may appear. Structural, not stylistic.

        A marker in this stream would mean the dense baseline is reading a firewalled surface —
        i.e. being handicapped in exactly the way M5 §4.1 forbids — and nothing in the loss curve
        or the accuracy would show it.
        """
        marker_ids = np.array(
            [self.tokenizer.stoi[m] for m in T.MARKERS if m in self.tokenizer.stoi],
            dtype=ids.dtype,
        )
        if marker_ids.size and bool(np.isin(ids, marker_ids).any()):
            raise ValueError(
                "memory-interface markers appear in the dense stream — the fact firewall is on "
                "and the baseline is being handicapped (M5 §4.1)"
            )

    def epoch_tokens(self, epoch: int) -> np.ndarray:
        """The token stream for one exposure of every fact. Flat, ``<eos>``-separated.

        The **wording** cycles with ``variant_period``; the **order** never repeats (M5 §5.63).
        """
        return self._variant_tokens(epoch % self.variant_period, epoch=epoch)

    # -- accounting ----------------------------------------------------------------------

    @property
    def tokens_per_epoch(self) -> int:
        return int(self.epoch_tokens(0).size)

    def total_tokens(self) -> int:
        return self.tokens_per_epoch * self.cfg.exposures

    def bits_per_token(self) -> float:
        """Corpus information content per training token — the density axis's own metric (M5 §5.6).

        Measured, never asserted. The *numerator* is a property of the corpus and is identical
        across surfaces by construction; only the denominator moves. That is precisely what makes
        this axis safe: a surface change that altered the numerator would have altered the world.
        """
        return self.corpus.stored_bits / self.tokens_per_epoch

    def tokens_per_name(self) -> float:
        """Mean tokens an entity name costs on this surface — a SURFACE COORDINATE (M5 §5.110).

        Measured from the rendered subjects rather than asserted: the syllable surface is 2 by
        construction, but a BPE surface fragments names however its merges happen to fall, and the
        whole point of the coordinate is that the number is not knowable from the config.
        """
        totals, count = 0, 0
        for relation in self.corpus.relations:
            subjects = self._subjects.get(relation)
            if not subjects:
                continue
            sample = subjects[:: max(1, len(subjects) // 256)]
            totals += sum(len(self.tokenizer.encode(name)) for name in sample)
            count += len(sample)
        return totals / max(1, count)

    def exposure_report(self) -> dict:
        """What the regime actually delivers — quoted in the run plan, not inferred from config."""
        n_train_facts = int(self._train_mask.sum())
        n_qa = int(self.qa_train_entities.sum())
        return {
            "exposures_per_fact": self.cfg.exposures,
            "distinct_renderings_per_fact": {
                r: min(self.cfg.exposures, v) for r, v in self._n_variants.items()
            },
            "qa_period": self.cfg.qa_period,
            "qa_share_of_exposures": (1.0 / self.cfg.qa_period) if self.cfg.qa_period else 0.0,
            "variant_period": self.variant_period,
            "n_train_facts": n_train_facts,
            "n_qa_train_entities": n_qa,
            "n_probe_entities": int(self.corpus.n_entities - n_qa),
            "n_probe_facts_heldout_qa": int(self.probe_fact_ids().size),
            "n_probe_facts_qa_trained": int(self.probe_fact_ids(qa_trained=True).size),
            "surface": self.surface.name,
            # SURFACE COORDINATES (M5 §5.110). Every surface-dependent number ships with all three,
            # because no achievable surface comparison holds them all fixed and any two of them
            # varying makes a difference un-attributable. Embedding SHARE needs the model width and
            # is therefore reported by the runner, not here.
            "vocabulary_size": int(self.tokenizer.vocab_size),
            "tokens_per_name": self.tokens_per_name(),
            # Two fingerprints, because one cannot see what the other pins (M5 §5.31).
            "surface_fingerprint": surface_fingerprint(
                self.surface, self.tokenizer, self.corpus
            ),
            "tokens_per_epoch": self.tokens_per_epoch,
            "tokens_per_fact": self.tokens_per_epoch / max(1, n_train_facts),
            "bits_per_token": self.bits_per_token(),
            "total_tokens": self.total_tokens(),
            "corpus_bits_per_fact": self.corpus.total_bits / self.corpus.n_facts,
            "corpus_stored_bits": self.corpus.stored_bits,
            "world_fingerprint": self.corpus.fingerprint(),
            "fingerprint": self.corpus.fingerprint(),  # legacy key, kept for older artifacts
        }


__all__ = ["DenseCorpusStream", "DenseDataConfig"]
