"""M3 §3.1 — the per-fact decode-quality distribution, the SHAPE instrument.

Built **before** the compression sweep, because the sweep exists to ask a question the mean cannot
answer. Two mechanisms produce a falling accuracy curve and are distinguished only by the shape of
the per-fact distribution underneath it:

======================  ===========================  ==========================================
shape                   mechanism                    signature
======================  ===========================  ==========================================
plateau-then-cliff      attractor basins             per-fact quality **bimodal**; narrow
                                                     transition; failures all-or-nothing
smooth decay            superposition interference   per-fact quality **unimodal, spreading**;
                                                     wide transition; failures partial
======================  ===========================  ==========================================

Three statistics, all registered in advance (M3 §3.1):

1. **Bimodality** — fraction of facts in the intermediate band (quality in ``(0.1, 0.9)``). Basins
   predict this stays near zero *through* the transition; interference predicts it grows and then
   shrinks as everything degrades.
2. **Transition width** — the load interval over which mean quality falls 90% -> 10%, normalised by
   the knee. A cliff is narrow; interference is wide.
3. **Per-fact trajectory** — does an individual fact hold and then collapse, or slide? Requires
   tracking the *same* facts across loads, so identity is carried by **fact id, never by position**.

Two qualities, because they answer different questions
------------------------------------------------------
``reconstruction`` — cosine between ``reconstruct`` and ``target``: the store's own fidelity.
``addressing`` — the normalised rank of the true slot when its own clean code is used as the query:
1.0 means nothing outranks it, 0.0 means everything does. **Addressing is the one the phase's claim
turns on**; reconstruction is reported beside it because a design can lose one and keep the other,
and M3's first point already showed exactly that (46% reconstruction error, 100% addressability).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

#: Facts in this band are neither cleanly decoded nor cleanly lost — the interference signature.
INTERMEDIATE_BAND = (0.1, 0.9)


@dataclass
class QualityDistribution:
    """Per-fact quality at ONE load, joined by fact id."""

    fact_ids: np.ndarray
    quality: np.ndarray
    label: str = ""
    per_class: dict[str, dict] = field(default_factory=dict)

    @property
    def mean(self) -> float:
        return float(self.quality.mean()) if self.quality.size else 0.0

    @property
    def bimodal_fraction(self) -> float:
        """Fraction in the intermediate band. **Near zero = all-or-nothing = basins.**"""
        lo, hi = INTERMEDIATE_BAND
        if not self.quality.size:
            return 0.0
        return float(((self.quality > lo) & (self.quality < hi)).mean())

    @property
    def clean_fraction(self) -> float:
        return float((self.quality >= INTERMEDIATE_BAND[1]).mean()) if self.quality.size else 0.0

    @property
    def lost_fraction(self) -> float:
        return float((self.quality <= INTERMEDIATE_BAND[0]).mean()) if self.quality.size else 0.0

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "n": int(self.quality.size),
            "mean": self.mean,
            "bimodal_fraction": self.bimodal_fraction,
            "clean_fraction": self.clean_fraction,
            "lost_fraction": self.lost_fraction,
            "per_class": self.per_class,
        }

    def __str__(self) -> str:
        return (
            f"mean {self.mean:.3f}  clean {self.clean_fraction:.1%}  "
            f"intermediate {self.bimodal_fraction:.1%}  lost {self.lost_fraction:.1%}"
        )


@torch.no_grad()
def per_fact_quality(
    store,
    slot_ids: torch.Tensor,
    *,
    kind: str = "addressing",
    fact_ids: np.ndarray | None = None,
    classes: np.ndarray | None = None,
    class_names: dict | None = None,
    label: str = "",
) -> QualityDistribution:
    """Per-fact quality over `slot_ids`.

    `kind="addressing"` — normalised rank of the true slot when queried with its own clean code.
    `kind="reconstruction"` — cosine of reconstruction against the clean code, clamped to [0, 1].

    `classes` gives a per-fact class id so the per-class split is produced here rather than being
    left to each caller (M3 §4.1 makes it mandatory, and mandatory things belong in the instrument).
    """
    if kind == "reconstruction":
        q = F.cosine_similarity(store.reconstruct(slot_ids), store.target(slot_ids), dim=-1)
        quality = q.clamp(min=0.0, max=1.0).cpu().numpy()
    elif kind == "addressing":
        # Energy of every slot against each fact's own clean code; lower is better, so the true
        # slot's rank is how many slots beat it.
        clean = store.target(slot_ids)
        energies = store.score(clean, slot_ids)  # (N, N)
        own = energies.diagonal()
        beaten = (energies < own.unsqueeze(1)).sum(dim=1).float()
        n = max(1, energies.shape[1] - 1)
        quality = (1.0 - beaten / n).cpu().numpy()
    else:
        raise ValueError(f"unknown quality kind {kind!r}")

    ids = np.asarray(fact_ids) if fact_ids is not None else slot_ids.cpu().numpy()
    dist = QualityDistribution(fact_ids=ids, quality=quality, label=label)

    if classes is not None:
        names = class_names or {}
        for c in np.unique(classes):
            m = classes == c
            sub = QualityDistribution(fact_ids=ids[m], quality=quality[m])
            dist.per_class[str(names.get(int(c), int(c)))] = sub.to_dict()
    return dist


def transition_width(loads: list[float], means: list[float]) -> dict:
    """Where mean quality falls 90% -> 10%, and how wide that is relative to the knee.

    Returned as a dict with `None`s rather than a bare number: a curve that never crosses one of the
    thresholds has no width, and reporting a fabricated one would be worse than reporting nothing.
    """
    loads, means = list(loads), list(means)
    order = np.argsort(loads)
    x = np.array(loads, dtype=float)[order]
    y = np.array(means, dtype=float)[order]

    def crossing(level: float) -> float | None:
        for i in range(len(y) - 1):
            if (y[i] >= level >= y[i + 1]) and y[i] != y[i + 1]:
                t = (y[i] - level) / (y[i] - y[i + 1])
                return float(x[i] + t * (x[i + 1] - x[i]))
        return None

    hi, lo = crossing(0.9), crossing(0.1)
    width = None if (hi is None or lo is None) else lo - hi
    return {
        "load_at_90pct": hi,
        "load_at_10pct": lo,
        "width": width,
        "normalised_width": None if (width is None or not hi) else width / hi,
    }


def classify_shape(dists: list[QualityDistribution], loads: list[float]) -> dict:
    """Report the evidence for each shape. **Never returns a bare verdict.**

    A single label would invite quoting it without the numbers that produced it, and M3 §3.1 already
    says a mixed signature must be reported as mixed rather than forced into a branch.
    """
    means = [d.mean for d in dists]
    peak_intermediate = max((d.bimodal_fraction for d in dists), default=0.0)
    tw = transition_width(loads, means)
    return {
        "peak_intermediate_fraction": peak_intermediate,
        "transition": tw,
        "means": means,
        "loads": list(loads),
        # Evidence, not a verdict. Thresholds are deliberately absent: the phase has no calibrated
        # value for "narrow" yet, and inventing one here would freeze a guess as a finding.
        "evidence_for_basins": "low peak intermediate fraction + narrow normalised width",
        "evidence_for_superposition": "high peak intermediate fraction + wide normalised width",
    }
