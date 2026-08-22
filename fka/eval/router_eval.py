"""Router evaluation against **binding confusers**, plus the slot maps that make it honest.

Pre-registered in `docs/decision_records/M2_router.md` §2, before any router trains.

Why not recall@k against random negatives
-----------------------------------------
With 8,000 slots, the negatives that discriminate a *bound* address from an *entity-shaped* one
are the 3 slots sharing the target's entity and the 1,999 sharing its relation. A mean margin over
all 7,999 negatives is dominated by cases that were never in doubt — the same way Phase 1's 3-hop
gate saturated because its discriminating cases were a vanishing fraction of the data.

So this module reports margins **per confuser class** and gates on the worst:

    same-entity `(e, r')`  -> a hit here means the router found the ENTITY and ignored the relation
    same-relation `(e', r)` -> a hit here means it found the RELATION and ignored the entity
    unrelated `(e', r')`    -> easy; reported only to show how uninformative it is

A router that scores well on the pooled mean while failing `(e, r')` has learned to address
entities, not facts, and the pooled mean cannot see it. That asymmetry is the whole point.

Scoring is by fact id join throughout — never by row position (see M1 §5, defects 3-5).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from fka.data.corpus_gen import KnowledgeCorpus

#: Gate: the worst binding-confuser margin must be positive by this much, in score units.
#: Provisional — the freeze value belongs in the M2 record once a router has actually trained.
MIN_BINDING_MARGIN = 0.0


@dataclass(frozen=True)
class SlotMap:
    """Which product-key slot each fact lives in.

    The assignment is a **pre-registered experimental variable**, not an implementation detail:
    aligning the grid axes with (entity, relation) would hand the router the factorisation the
    experiment exists to test. See M2 §3.
    """

    fact_to_slot: np.ndarray  # (n_facts,) -> slot id
    axes: tuple[int, int]
    mode: str

    @property
    def n_slots(self) -> int:
        return self.axes[0] * self.axes[1]

    def __post_init__(self) -> None:
        if len(set(self.fact_to_slot.tolist())) != len(self.fact_to_slot):
            raise ValueError("slot assignment is not injective — two facts share a slot")
        if self.fact_to_slot.max(initial=0) >= self.n_slots:
            raise ValueError("slot assignment points outside the grid")


def aligned_slot_map(corpus: KnowledgeCorpus, relations: list[str]) -> SlotMap:
    """POSITIVE CONTROL ONLY. Axis 1 = entity, axis 2 = relation.

    Not `O(√N)` — the axes are 2,000 × 4 — and not a result. It bounds what the machinery can do
    when the factorisation is *given*, which is the ceiling the shuffled map is measured against.
    """
    n_e, n_r = corpus.n_entities, len(relations)
    fact_to_slot = np.empty(n_e * n_r, dtype=np.int64)
    for r_idx in range(n_r):
        for e in range(n_e):
            fact_to_slot[r_idx * n_e + e] = e * n_r + r_idx
    return SlotMap(fact_to_slot, (n_e, n_r), "aligned")


def shuffled_slot_map(
    corpus: KnowledgeCorpus, relations: list[str], *, seed: int = 0
) -> SlotMap:
    """THE EXPERIMENT. A square `√N × √N` grid with facts scattered by a fixed permutation.

    Entity and relation are spread across both axes, so separable scoring cannot inherit the
    corpus's own factorisation; any binding the router shows has to have been learned.
    """
    n_facts = corpus.n_entities * len(relations)
    side = int(np.ceil(np.sqrt(n_facts)))
    perm = np.random.default_rng(seed).permutation(side * side)[:n_facts]
    return SlotMap(perm.astype(np.int64), (side, side), f"shuffled(seed={seed})")


def identity_slot_map(corpus: KnowledgeCorpus, relations: list[str]) -> SlotMap:
    """Square grid, facts in natural `fact_id` order. Between the other two: not aligned with
    (entity, relation), but not scrambled either — consecutive entities share a row."""
    n_facts = corpus.n_entities * len(relations)
    side = int(np.ceil(np.sqrt(n_facts)))
    return SlotMap(np.arange(n_facts, dtype=np.int64), (side, side), "identity")


@dataclass
class ConfuserMargins:
    """Margin statistics for one confuser class."""

    name: str
    n: int
    mean: float
    p05: float
    minimum: float
    fraction_positive: float

    @property
    def contested(self) -> bool:
        return self.minimum <= MIN_BINDING_MARGIN

    def to_dict(self) -> dict:
        return {
            "name": self.name, "n": self.n, "mean": self.mean, "p05": self.p05,
            "min": self.minimum, "fraction_positive": self.fraction_positive,
            "contested": self.contested,
        }

    def __str__(self) -> str:
        flag = "  [CONTESTED]" if self.contested else ""
        return (
            f"{self.name:<14} mean {self.mean:+.4f}  p05 {self.p05:+.4f}  "
            f"min {self.minimum:+.4f}  {self.fraction_positive:.1%} positive{flag}"
        )


@dataclass
class RouterEvalResult:
    n_probes: int
    recall_at_1: float
    recall_at_k: float
    k: int
    margins: dict[str, ConfuserMargins] = field(default_factory=dict)
    #: recall@1 restricted to targets whose ADDRESS was never supervised. See `evaluate_router`.
    recall_at_1_unseen: float | None = None
    n_unseen: int = 0

    @property
    def worst_binding_margin(self) -> float:
        """The gate quantity: the minimum over the two BINDING confusers, ignoring the easy one."""
        binding = [m for n, m in self.margins.items() if n != "unrelated"]
        return min((m.minimum for m in binding), default=float("-inf"))

    @property
    def passes(self) -> bool:
        return self.recall_at_1 >= 1.0 and self.worst_binding_margin > MIN_BINDING_MARGIN

    def to_dict(self) -> dict:
        return {
            "n_probes": self.n_probes,
            "recall_at_1": self.recall_at_1,
            "recall_at_k": self.recall_at_k,
            "k": self.k,
            "margins": {n: m.to_dict() for n, m in self.margins.items()},
            "worst_binding_margin": self.worst_binding_margin,
            "recall_at_1_unseen": self.recall_at_1_unseen,
            "n_unseen": self.n_unseen,
            "passes": self.passes,
        }

    def __str__(self) -> str:
        lines = [
            f"recall@1 {self.recall_at_1:.1%}  recall@{self.k} {self.recall_at_k:.1%}  "
            f"(n={self.n_probes})"
        ]
        if self.recall_at_1_unseen is not None:
            lines[0] += (
                f"   [UNSUPERVISED ADDRESSES: {self.recall_at_1_unseen:.1%}, n={self.n_unseen}]"
            )
        lines += [f"   {m}" for m in self.margins.values()]
        return "\n".join(lines)


def _margins(name: str, values: np.ndarray) -> ConfuserMargins:
    return ConfuserMargins(
        name=name,
        n=int(values.size),
        mean=float(values.mean()) if values.size else 0.0,
        p05=float(np.percentile(values, 5)) if values.size else 0.0,
        minimum=float(values.min()) if values.size else 0.0,
        fraction_positive=float((values > 0).mean()) if values.size else 0.0,
    )


@torch.no_grad()
def evaluate_router(
    router,
    queries: torch.Tensor,
    fact_ids: np.ndarray,
    slot_map: SlotMap,
    corpus: KnowledgeCorpus,
    relations: list[str],
    *,
    n_confusers: int = 8,
    seed: int = 0,
    supervised_fact_ids: np.ndarray | set | None = None,
) -> RouterEvalResult:
    """Recall and per-class binding margins for `queries[i]` targeting `fact_ids[i]`.

    `fact_id = relation_index * n_entities + entity_index` (corpus convention). Everything is
    joined through `fact_ids`; row order is never assumed to mean anything.

    `supervised_fact_ids` — every fact that appeared as a **retrieval target during training** —
    splits out `recall_at_1_unseen`, and it is not optional in spirit.

    **Why this split exists (2026-08-02).** Fork (a)'s probes are entity-held-out in M1's sense:
    the subject was never a training subject. That is the correct holdout for the *kernel*, because
    it stops query formation being a per-name habit. It is the **wrong** holdout for the *router*,
    whose addresses are supervised whenever a fact is a retrieval *target* — including as hop 2 or
    3 of a chain whose subject was a training entity. Measured on the joint fit's own probe set,
    **85.8% of eval target facts had already been supervised as training targets**, so the headline
    number was mostly address recall and only incidentally address composition.

    P3 in §1 — "a query composed from a retrieved latent lands correctly on entities never
    addressed" — is the claim, and only the unseen subset can speak to it. §9.5's non-composing
    failure argument said a held-out entity's address "must be computed"; that argument was wrong,
    and this is the instrument that makes the error visible instead of invisible.
    """
    if len(queries) != len(fact_ids):
        raise ValueError(f"{len(queries)} queries against {len(fact_ids)} targets")
    if len(set(fact_ids.tolist())) != len(fact_ids):
        raise ValueError("duplicate fact ids in the probe set — the join would be ambiguous")

    n_e, n_r = corpus.n_entities, len(relations)
    device = queries.device
    target_slots = torch.from_numpy(slot_map.fact_to_slot[fact_ids]).to(device)

    slots, _ = router(queries)
    hit1_per = (slots[:, 0] == target_slots).float()
    hit1 = hit1_per.mean().item()
    hitk = (slots == target_slots.unsqueeze(1)).any(dim=1).float().mean().item()

    unseen_recall, n_unseen = None, 0
    if supervised_fact_ids is not None:
        seen = set(int(f) for f in supervised_fact_ids)
        mask = torch.tensor(
            [int(f) not in seen for f in fact_ids], dtype=torch.bool, device=device
        )
        n_unseen = int(mask.sum())
        unseen_recall = float(hit1_per[mask].mean()) if n_unseen else None

    target_score = router.slot_scores(queries, target_slots.unsqueeze(1)).squeeze(1)

    rng = np.random.default_rng(seed)
    ent = fact_ids % n_e
    rel = fact_ids // n_e

    def confuser_margin(name: str, build) -> ConfuserMargins:
        cols = []
        for i in range(len(fact_ids)):
            cand = build(int(ent[i]), int(rel[i]))
            if not cand:
                continue
            take = cand if len(cand) <= n_confusers else [
                cand[j] for j in rng.choice(len(cand), size=n_confusers, replace=False)
            ]
            cols.append((i, take))
        if not cols:
            return _margins(name, np.array([]))
        width = max(len(c) for _, c in cols)
        idx = np.zeros((len(cols), width), dtype=np.int64)
        valid = np.zeros((len(cols), width), dtype=bool)
        for row, (_, take) in enumerate(cols):
            idx[row, : len(take)] = slot_map.fact_to_slot[np.array(take)]
            valid[row, : len(take)] = True
        rows = torch.tensor([i for i, _ in cols], device=device)
        scores = router.slot_scores(
            queries[rows], torch.from_numpy(idx).to(device)
        )
        scores = scores.masked_fill(~torch.from_numpy(valid).to(device), float("-inf"))
        best_confuser = scores.max(dim=1).values
        return _margins(name, (target_score[rows] - best_confuser).cpu().numpy())

    margins = {
        # same entity, every other relation
        "same-entity": confuser_margin(
            "same-entity", lambda e, r: [r2 * n_e + e for r2 in range(n_r) if r2 != r]
        ),
        # same relation, other entities
        "same-relation": confuser_margin(
            "same-relation",
            lambda e, r: [r * n_e + int(e2) for e2 in rng.choice(n_e, size=min(n_e, 64),
                                                                 replace=False) if int(e2) != e],
        ),
        # neither — the easy class, reported to show it cannot discriminate
        "unrelated": confuser_margin(
            "unrelated",
            lambda e, r: [
                int(f) for f in rng.choice(n_e * n_r, size=64, replace=False)
                if int(f) % n_e != e and int(f) // n_e != r
            ],
        ),
    }

    return RouterEvalResult(
        n_probes=len(fact_ids),
        recall_at_1=hit1,
        recall_at_k=hitk,
        k=int(slots.shape[1]),
        margins=margins,
        recall_at_1_unseen=unseen_recall,
        n_unseen=n_unseen,
    )
