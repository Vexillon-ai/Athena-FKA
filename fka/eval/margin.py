"""Per-query retrieval MARGIN, and per-fact margin trajectories — the primary shape readout.

Registered in M3 §11.4 after the first shape attempt could not conclude. The earlier channels
failed for two different reasons, both worth keeping in view:

* store-internal addressing **saturated at 1.000** — a code still outranks its neighbours inside the
  store long after the downstream path has stopped working;
* per-fact end-to-end correctness was **binary at 1-2 queries per fact**, so its "bimodality" was
  reading sampling granularity rather than mechanism.

The margin fixes both. It is continuous, it is measured on the deployed path, and it keeps degrading
after correctness has already hit zero:

    margin(query) = cos(query, key[true]) - max_{j != true} cos(query, key[j])

Positive means retrieved; the *size* says by how much; negative says how badly it was beaten.

Trajectories are primary, distributions secondary
--------------------------------------------------
The shape question is about **what happens to a fact as load increases**, so the readout is the
per-fact margin *trajectory* across loads — no aggregation step between the measurement and the
conclusion. Fact identity is threaded by id throughout, never by position.

Aggregating first is precisely how the previous attempt lost the mechanism: a mean over facts, or
even a distribution at one load, cannot distinguish "every fact slid a little" from "a fifth of the
facts fell off a cliff" once the two have the same mean. The trajectory can, per fact, without
anybody choosing a threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def retrieval_margins(
    keys: torch.Tensor, queries: torch.Tensor, target_ids: torch.Tensor
) -> torch.Tensor:
    """`(Q,)` margins for each query against its own target, over the full key set."""
    sims = F.normalize(queries.float(), dim=-1) @ F.normalize(keys.float(), dim=-1).T
    correct = sims.gather(1, target_ids.view(-1, 1)).squeeze(1)
    best_wrong = sims.scatter(1, target_ids.view(-1, 1), float("-inf")).max(dim=1).values
    return correct - best_wrong


@dataclass
class MarginSet:
    """Margins at ONE load, joined by fact id."""

    fact_ids: np.ndarray
    margins: np.ndarray
    load: float
    label: str = ""
    classes: np.ndarray | None = None
    class_names: dict = field(default_factory=dict)

    @property
    def mean(self) -> float:
        return float(self.margins.mean()) if self.margins.size else 0.0

    @property
    def retrieved_fraction(self) -> float:
        return float((self.margins > 0).mean()) if self.margins.size else 0.0

    @property
    def inverted_fraction(self) -> float:
        """Margin below zero: the fact lost outright. **Subset-total degradation's signature.**"""
        return float((self.margins < 0).mean()) if self.margins.size else 0.0

    def intact_fraction(self, reference: np.ndarray, keep: float = 0.5) -> float:
        """Facts still holding at least `keep` of their reference margin."""
        if not self.margins.size:
            return 0.0
        ref = np.where(np.abs(reference) < 1e-9, 1e-9, reference)
        return float((self.margins / ref >= keep).mean())

    def degraded_fraction(self, reference: np.ndarray, band=(0.05, 0.95)) -> float:
        """Facts in the intermediate band of their own reference. **Partial degradation's
        signature**: everything shrunken, nothing lost."""
        if not self.margins.size:
            return 0.0
        ref = np.where(np.abs(reference) < 1e-9, 1e-9, reference)
        r = self.margins / ref
        return float(((r > band[0]) & (r < band[1])).mean())

    def per_class(self) -> dict:
        if self.classes is None:
            return {}
        out = {}
        for c in np.unique(self.classes):
            m = self.classes == c
            out[str(self.class_names.get(int(c), int(c)))] = {
                "n": int(m.sum()),
                "mean_margin": float(self.margins[m].mean()),
                "retrieved_fraction": float((self.margins[m] > 0).mean()),
                "inverted_fraction": float((self.margins[m] < 0).mean()),
            }
        return out

    def to_dict(self) -> dict:
        return {
            "label": self.label, "load": self.load, "n": int(self.margins.size),
            "mean_margin": self.mean, "retrieved_fraction": self.retrieved_fraction,
            "inverted_fraction": self.inverted_fraction, "per_class": self.per_class(),
        }


@dataclass
class MarginTrajectories:
    """Per-fact margin across loads. **Primary readout — no aggregation before the conclusion.**"""

    fact_ids: np.ndarray  # (F,)
    loads: np.ndarray  # (L,) ascending in DIFFICULTY (descending in bits)
    margins: np.ndarray  # (F, L)
    classes: np.ndarray | None = None
    class_names: dict = field(default_factory=dict)

    @classmethod
    def from_sets(cls, sets: list[MarginSet]) -> MarginTrajectories:
        """Join per-fact across loads BY FACT ID; a fact missing at any load is dropped, loudly."""
        common = set(sets[0].fact_ids.tolist())
        for s in sets[1:]:
            common &= set(s.fact_ids.tolist())
        ids = np.array(sorted(common))
        cols = []
        for s in sets:
            # Average duplicate queries for the same fact — a fact may be probed more than once.
            lookup: dict[int, list[float]] = {}
            for f, m in zip(s.fact_ids.tolist(), s.margins.tolist(), strict=True):
                lookup.setdefault(int(f), []).append(float(m))
            cols.append(np.array([np.mean(lookup[int(i)]) for i in ids]))
        classes = None
        if sets[0].classes is not None:
            first = {int(f): int(c) for f, c in
                     zip(sets[0].fact_ids.tolist(), sets[0].classes.tolist(), strict=True)}
            classes = np.array([first[int(i)] for i in ids])
        return cls(fact_ids=ids, loads=np.array([s.load for s in sets]),
                   margins=np.stack(cols, axis=1), classes=classes,
                   class_names=sets[0].class_names)

    def collapse_sharpness(self) -> np.ndarray:
        """Per fact: the largest single-step drop as a share of its total drop.

        ~1.0 means one step did all the damage (a **cliff**); ~1/(L-1) means the loss was spread
        evenly across steps (a **slide**). Computed per fact, so no threshold is chosen for anyone.
        """
        d = -np.diff(self.margins, axis=1)  # positive where the margin fell
        total = d.sum(axis=1)
        biggest = d.max(axis=1)
        safe = np.where(np.abs(total) < 1e-9, np.nan, total)
        return biggest / safe

    def summary(self) -> dict:
        sharp = self.collapse_sharpness()
        finite = sharp[np.isfinite(sharp)]
        n_steps = max(1, self.margins.shape[1] - 1)
        out = {
            "n_facts": int(self.fact_ids.size),
            "n_loads": int(self.loads.size),
            "uniform_slide_reference": 1.0 / n_steps,
            "sharpness_median": float(np.median(finite)) if finite.size else None,
            "sharpness_p90": float(np.percentile(finite, 90)) if finite.size else None,
            "fraction_cliff_like": float((finite > 0.75).mean()) if finite.size else None,
            "fraction_slide_like": float((finite < 2.0 / n_steps).mean()) if finite.size else None,
            "ever_inverted_fraction": float((self.margins < 0).any(axis=1).mean()),
            "monotone_fraction": float((np.diff(self.margins, axis=1) <= 1e-9).all(axis=1).mean()),
        }
        if self.classes is not None:
            per = {}
            for c in np.unique(self.classes):
                m = self.classes == c
                s = sharp[m][np.isfinite(sharp[m])]
                per[str(self.class_names.get(int(c), int(c)))] = {
                    "n": int(m.sum()),
                    "sharpness_median": float(np.median(s)) if s.size else None,
                    "ever_inverted_fraction": float((self.margins[m] < 0).any(axis=1).mean()),
                }
            out["per_class"] = per
        return out
