"""Storage accounting for M3 — **knowledge bits stored / storage bits required** (M3 §10.5).

Pre-registered before any capacity measurement, because this is exactly the accounting choice that a
favourable number would make it tempting to settle afterwards.

    bits_per_bit = (knowledge bits recovered) / (per-fact bits x n_facts + shared bits)

**"Storage bits" = everything ``reconstruct()`` and the key path need at inference.** Pointers and
residuals per fact, at their **stored width**; codebooks, the relation table and the key encoder
``f`` amortised over the fact set they serve. **Nothing is exempt**, and the same accounting
applies to S1, S2, S3 and to ``IdentityStore``.

The three gaming counterexamples, recorded so the rule reads as load-bearing
----------------------------------------------------------------------------
Each is excluded by "everything the inference path needs", and each is otherwise reachable:

1. **Free pointers.** Declare per-slot pointers "data, not parameters" and drop them from the
   denominator. A store with one enormous codebook and one pointer per fact then looks arbitrarily
   efficient, while the pointer is carrying all the information.
2. **Hidden dictionary.** Leave the codebooks out because they are "shared infrastructure". Taken to
   its limit this is one codebook entry per fact — a lookup table reported as compression.
3. **Memorising encoder.** Push the information into ``f``'s weights and count only the substrate.
   The store shrinks to nothing and the key encoder becomes the store. M2 §10.3.4 already showed a
   learned map on this path *can* memorise, so this is not hypothetical.

Two widths, both reported
-------------------------
Shared parameters are counted at **8 bits** (int8-equivalent) for the headline, because the dense
baseline this is measured against — ~2 bits/param, i.e. **0.25 bits/bit** — comes from models
established to survive int8 quantisation. The **fp32** figure is reported beside it as the
conservative reading. Quoting only the flattering one would be the accounting equivalent of picking
the branch after seeing the number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

#: M3 gates, restated in bits/bit (M3 §2.1). Exact restatement of the bits/param gates under the
#: int8 convention: 3 bits/param / 8 bits/param = 0.375; 4 / 8 = 0.5; dense ~2 / 8 = 0.25.
GATE_PASS = 0.375
GATE_TARGET = 0.5
DENSE_BASELINE = 0.25


def pointer_bits(codebook_size: int) -> float:
    """Bits to index one codebook. Fractional by design — an entropy bound, not a byte count."""
    return math.log2(codebook_size)


@dataclass
class StorageAccount:
    """One design at one operating point, with every bit its inference path needs."""

    n_facts: int
    #: Bits stored per fact: pointers + residual, at STORED width.
    per_fact_bits: float
    #: Parameters shared across facts, counted ONCE: codebooks, relation table, key encoder `f`.
    shared_params: int
    #: Knowledge bits actually recovered (corpus entropy of correctly-decoded facts).
    knowledge_bits: float
    breakdown: dict = field(default_factory=dict)

    def storage_bits(self, param_bits: int = 8) -> float:
        return self.per_fact_bits * self.n_facts + self.shared_params * param_bits

    def bits_per_bit(self, param_bits: int = 8) -> float:
        denom = self.storage_bits(param_bits)
        return self.knowledge_bits / denom if denom else 0.0

    @property
    def headline(self) -> float:
        """int8-equivalent — the figure the gates are stated against."""
        return self.bits_per_bit(8)

    @property
    def conservative(self) -> float:
        """fp32 shared parameters — reported beside the headline, never instead of it."""
        return self.bits_per_bit(32)

    @property
    def verdict(self) -> str:
        b = self.headline
        if b >= GATE_TARGET:
            return "PASS (meets target)"
        if b >= GATE_PASS:
            return "PASS BUT UNDER-DELIVERS"
        return "FAIL"

    def to_dict(self) -> dict:
        return {
            "n_facts": self.n_facts,
            "per_fact_bits": self.per_fact_bits,
            "shared_params": self.shared_params,
            "knowledge_bits": self.knowledge_bits,
            "storage_bits_int8": self.storage_bits(8),
            "storage_bits_fp32": self.storage_bits(32),
            "bits_per_bit_int8": self.headline,
            "bits_per_bit_fp32": self.conservative,
            "gate_pass": GATE_PASS,
            "gate_target": GATE_TARGET,
            "dense_baseline": DENSE_BASELINE,
            "verdict": self.verdict,
            "breakdown": self.breakdown,
        }

    def __str__(self) -> str:
        return (
            f"{self.headline:.4f} bits/bit int8 ({self.conservative:.4f} fp32)   "
            f"[pass {GATE_PASS}, target {GATE_TARGET}, dense {DENSE_BASELINE}]   {self.verdict}"
        )


def account_for(
    store,
    *,
    n_facts: int,
    knowledge_bits: float,
    key_path_params: int = 0,
    key_path_label: str = "key encoder f + relation table",
) -> StorageAccount:
    """Build the account from a store's declared `cost_model()` plus the key path's parameters.

    `key_path_params` is not optional in spirit: omitting it is gaming counterexample 3. It is a
    separate argument only because `f` is not owned by the store.
    """
    cm = store.cost_model()
    missing = {"per_fact_storage_bits", "shared_parameters"} - set(cm)
    if missing:
        raise ValueError(
            f"{type(store).__name__}.cost_model() is missing {sorted(missing)} — a store must "
            "declare everything its inference path needs (M3 §10.5)"
        )
    return StorageAccount(
        n_facts=n_facts,
        per_fact_bits=float(cm["per_fact_storage_bits"]),
        shared_params=int(cm["shared_parameters"]) + int(key_path_params),
        knowledge_bits=knowledge_bits,
        breakdown={
            "design": cm.get("design"),
            "store_shared_parameters": int(cm["shared_parameters"]),
            "key_path_parameters": int(key_path_params),
            "key_path_label": key_path_label,
            "per_fact_detail": cm.get("per_fact_detail"),
        },
    )
