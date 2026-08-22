"""Query-space confusion: how close is an emitted query to the right key, versus the wrong ones?

The gate number tells you *whether* retrieval succeeded. This tells you *by how much*, which is
what distinguishes "the addressing is robust" from "the addressing is correct by a hair and the
distractors simply were not near enough to matter".

For each emitted query ``q`` we record, in the kernel's own query space:

* ``correct_sim`` — cosine similarity to the true key
* ``best_wrong_sim`` — the highest similarity to any *other* key in the memory
* ``margin = correct_sim - best_wrong_sim`` — positive iff hard argmax retrieves correctly

A large positive margin means wrong keys are nowhere near being selected, so adding more
near-miss *surface forms* cannot harden this interface: distance in address space is what matters,
and for a composed address ``entity_code ⊙ relation_code`` it has no relationship to string
similarity. A margin near zero means retrieval is genuinely contested and distractors bite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QueryConfusion:
    """Margin statistics over a set of emitted queries."""

    n: int
    margin_mean: float
    margin_median: float
    margin_p05: float
    margin_min: float
    fraction_positive: float
    correct_sim_mean: float
    best_wrong_sim_mean: float

    @classmethod
    def from_samples(
        cls, margins: list[float], correct_sim: list[float], best_wrong_sim: list[float]
    ) -> QueryConfusion:
        m = np.asarray(margins, dtype=np.float64)
        if m.size == 0:
            return cls(0, 0, 0, 0, 0, 0, 0, 0)
        return cls(
            n=int(m.size),
            margin_mean=float(m.mean()),
            margin_median=float(np.median(m)),
            margin_p05=float(np.percentile(m, 5)),
            margin_min=float(m.min()),
            fraction_positive=float((m > 0).mean()),
            correct_sim_mean=float(np.mean(correct_sim)),
            best_wrong_sim_mean=float(np.mean(best_wrong_sim)),
        )

    @property
    def contested(self) -> bool:
        """True if a meaningful share of queries are close to selecting a wrong key.

        The threshold is a reading aid, not a gate: a 5th-percentile margin under 0.05 means one
        query in twenty is within a whisker of mis-addressing.
        """
        return self.margin_p05 < 0.05

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "margin_mean": self.margin_mean,
            "margin_median": self.margin_median,
            "margin_p05": self.margin_p05,
            "margin_min": self.margin_min,
            "fraction_positive": self.fraction_positive,
            "correct_sim_mean": self.correct_sim_mean,
            "best_wrong_sim_mean": self.best_wrong_sim_mean,
            "contested": self.contested,
        }

    def __str__(self) -> str:
        verdict = "CONTESTED" if self.contested else "uncontested"
        return (
            f"margin mean {self.margin_mean:+.3f}, median {self.margin_median:+.3f}, "
            f"p05 {self.margin_p05:+.3f}, min {self.margin_min:+.3f}; "
            f"{self.fraction_positive:.1%} positive; correct {self.correct_sim_mean:.3f} vs "
            f"best-wrong {self.best_wrong_sim_mean:.3f} [{verdict}]"
        )
