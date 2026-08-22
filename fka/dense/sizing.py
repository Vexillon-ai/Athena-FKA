"""Sizing a dense run. **Load is measured against VALUE ENTROPY ONLY** (M5 §5.25).

### The rule, and the mistake it replaces

`corpus.total_bits` is the corpus's scored information: `bits(value) = -log2 P(value)` summed over
facts. M0's rule — **names are keys, not facts** — excludes entity names, because every probe is
phrased in terms of a name and counting it would double-count. **That is the correct denominator
for sizing, and it is the one the reference capacity study uses**, which is what makes our loads
comparable to its 2 bits/param.

**§5.8 briefly added a key-material term to this denominator and it was wrong.** The intuition was
that the weights must hold the name to tell it from `N-1` others. But a key is **presented at query
time**: an associative memory owes enough bits to reproduce the *value* given the key, not to store
the key. Adding `n_entities * log2(name-space)` inflated the load axis by up to **44%** at
N = 31,686 and put every sizing row in the wrong place.

### Why it was worse than merely wrong

The added term prices **key discrimination as storage** — which was precisely the hypothesis under
test when the sizing was used to design its discriminator. A denominator that already assumes the
answer cannot return the other one:

> **A denominator that prices the hypothesis is an instrument that cannot return the other answer.**

Under the retired accounting the key-discrimination test read "inconclusive, load 0.56x, inside the
collapse zone". Under this one it reads **load 0.12x with 4x the keys and recall at chance** —
clean, and decisive in the opposite direction. Same data, and the accounting chose the verdict.

`key_bits()` is **kept as a reported diagnostic** — it is a real and interesting quantity, 44% of
capacity at N = 31,686 — but it is no longer in any denominator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from fka.data.corpus_gen import KnowledgeCorpus

#: The dense-transformer figure the M3 gates are stated against (research plan §4.5). Used only to
#: SIZE runs; the measured value is what a run reports, and the two must never be conflated.
NOMINAL_BITS_PER_PARAM = 2.0


def key_bits(corpus: KnowledgeCorpus) -> float:
    """Bits of entity-key material the weights must hold and the scorer never credits."""
    return corpus.n_entities * math.log2(corpus.name_space.size)


def required_storage_bits(corpus: KnowledgeCorpus) -> float:
    """**RETIRED as a sizing denominator** (M5 §5.25); kept only so old artifacts stay readable.

    Adds key material to the value entropy. Keys are presented at query time and are not stored, so
    this over-counts; worse, it prices the key-discrimination hypothesis into the instrument used
    to test it. Use :func:`storage_bits`.
    """
    return corpus.total_bits + key_bits(corpus)


def storage_bits(corpus: KnowledgeCorpus) -> float:
    """What the weights owe: the corpus's **value entropy**. The reference study's denominator."""
    return corpus.total_bits


def capacity_bits(n_params: int, bits_per_param: float = NOMINAL_BITS_PER_PARAM) -> float:
    return n_params * bits_per_param


@dataclass(frozen=True)
class SizingReport:
    n_entities: int
    n_params: int
    scored_bits: float
    key_bits: float
    required_bits: float
    capacity_bits: float

    @property
    def load(self) -> float:
        """Value entropy over capacity — the reference accounting. **>1 is over-capacity.**"""
        return self.scored_bits / self.capacity_bits if self.capacity_bits else float("inf")

    @property
    def load_with_keys_counted(self) -> float:
        """The retired accounting, reported so old rows can be re-read (M5 §5.25)."""
        return self.required_bits / self.capacity_bits if self.capacity_bits else float("inf")

    @property
    def key_share_of_capacity(self) -> float:
        """Key material as a fraction of capacity — a diagnostic, never a denominator."""
        return self.key_bits / self.capacity_bits if self.capacity_bits else float("inf")

    def to_dict(self) -> dict:
        return {
            "n_entities": self.n_entities,
            "n_params": self.n_params,
            "scored_bits": self.scored_bits,
            "key_bits": self.key_bits,
            "required_bits": self.required_bits,
            "capacity_bits": self.capacity_bits,
            "load": self.load,
            "load_with_keys_counted": self.load_with_keys_counted,
            "key_share_of_capacity": self.key_share_of_capacity,
        }

    def __str__(self) -> str:
        return (
            f"N={self.n_entities:,} P={self.n_params:,}  value entropy "
            f"{self.scored_bits / 1e6:.3f}M vs capacity {self.capacity_bits / 1e6:.3f}M  ->  "
            f"load {self.load:.2f}x   [keys {self.key_bits / 1e6:.3f}M = "
            f"{self.key_share_of_capacity:.0%} of capacity, diagnostic only]"
        )


def size_run(
    corpus: KnowledgeCorpus, n_params: int, bits_per_param: float = NOMINAL_BITS_PER_PARAM
) -> SizingReport:
    return SizingReport(
        n_entities=corpus.n_entities,
        n_params=n_params,
        scored_bits=corpus.total_bits,
        key_bits=key_bits(corpus),
        required_bits=required_storage_bits(corpus),
        capacity_bits=capacity_bits(n_params, bits_per_param),
    )


__all__ = [
    "NOMINAL_BITS_PER_PARAM",
    "SizingReport",
    "capacity_bits",
    "key_bits",
    "required_storage_bits",
    "size_run",
    "storage_bits",
]
