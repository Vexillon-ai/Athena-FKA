"""How separable did the learned key geometry turn out to be? (M2 §9.8)

This is the bridge between fork (a) and the Phase 2 exit criterion. Fork (a) asks whether an
address space can be *learned*; §10 asks whether it can be *searched* sub-linearly. Those are
different properties, and the number that connects them is how close the converged keys sit to the
set a product-key shortlist can represent exactly.

The surrogate family, and why it is the right one
-------------------------------------------------
A product-key score decomposes as ``q·k(e, r) = q·a_e + q·b_r`` — additively separable across the
two factors. So the keys a product-key memory can represent are exactly those of the form::

    k(e, r) = a_e + b_r

and the question "how separable are these keys?" is the question "how well are they approximated by
that family?". Concatenative keys ``[a_e ; b_r]`` are the special case with disjoint supports;
nothing here needs the split to be axis-aligned, which is one arbitrary choice fewer.

The L2-optimal member has a closed form — the two-way additive decomposition, i.e. entity means plus
relation means minus the grand mean. **No optimiser, no learning rate, no seed:** the statistic is a
deterministic function of the key matrix. That matters because a fitted statistic can under-report
purely by being under-fitted, and a low separability reading would then be an artifact of the fit
rather than a property of the geometry.

Reported as a pair
------------------
``separability_index`` — recall@1 under the surrogate, using real kernel queries. This is the
operational number: it is what a product-key shortlist would have to work with.

``residual_fraction`` — ``‖K − K̂‖² / ‖K − μ‖²``, the share of key variance no additive surrogate can
explain. Query-free, so it cannot be flattered by a weak probe set, and it is the honest number to
quote if the probe set is small.

Calibration — the floor is NOT zero
-----------------------------------
A fully multiplicative key set still scores well above chance, because its additive surrogate
recovers the **entity** and loses only the relation. At our deployed configuration (2,000 entities,
4 relations, dim 64), measured with perfect queries:

===================================  =====  ========
key set                              index  residual
===================================  =====  ========
``normalize(e ⊙ r)``   (FLOOR)       0.260     0.763
``[a_e ; b_r]``        (CEILING)     1.000     0.000
===================================  =====  ========

So an index of ~0.26 means *fully multiplicative*, not "somewhat separable", and reading a result
against an assumed floor of zero would overstate its separability by that whole margin. The floor
falls as the address space grows (``tests/test_separability.py`` pins the ordering), so it is
recomputed for the configuration in hand rather than quoted from here.

The generous grid is deliberate (M2 §9.8): the surrogate is fitted on the true *(entity, relation)*
factorisation, so a **low** index is strong evidence of non-separability, while a **high** index is
an upper bound that still owes §10 a balanced-axis story — our grid is 2,000 × 4, not √N × √N.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SeparabilityResult:
    """The separability of a key set, measured against the product-key-representable family."""

    n_probes: int
    #: recall@1 under the additive surrogate, with real queries. The operational number.
    separability_index: float
    #: recall@1 under the *learned* keys on the same probes — the ceiling this is measured against.
    reference_recall_at_1: float
    #: share of key variance no additive surrogate can explain. Query-free.
    residual_fraction: float
    #: mean cosine between each learned key and its surrogate. Geometry, not ranking.
    mean_key_cosine: float
    n_entities: int
    n_relations: int

    @property
    def relative_index(self) -> float:
        """Separability as a fraction of what the learned keys themselves achieve.

        The raw index is capped by the query set's quality: keys that only reach 60% recall cannot
        have a surrogate that reaches 100%. Reporting the ratio alongside is the conditional-gate
        rule (CLAUDE.md) applied to a statistic rather than to a gate.
        """
        return (
            self.separability_index / self.reference_recall_at_1
            if self.reference_recall_at_1 > 0
            else 0.0
        )

    def to_dict(self) -> dict:
        return {
            "n_probes": self.n_probes,
            "separability_index": self.separability_index,
            "reference_recall_at_1": self.reference_recall_at_1,
            "relative_index": self.relative_index,
            "residual_fraction": self.residual_fraction,
            "mean_key_cosine": self.mean_key_cosine,
            "n_entities": self.n_entities,
            "n_relations": self.n_relations,
        }

    def __str__(self) -> str:
        return (
            f"separability index {self.separability_index:.1%} "
            f"(learned keys {self.reference_recall_at_1:.1%}, "
            f"relative {self.relative_index:.1%})  "
            f"residual {self.residual_fraction:.1%}  "
            f"mean key cos {self.mean_key_cosine:+.4f}"
        )


def additive_surrogate(keys: torch.Tensor, n_entities: int, n_relations: int) -> torch.Tensor:
    """The L2-optimal additively separable approximation of ``keys``.

    ``keys`` is ``(n_entities * n_relations, dim)`` in corpus fact-id order
    (``relation_index * n_entities + entity_index``); the return has the same shape and order.

    Closed form: ``k̂(e, r) = mean_r K[e, r] + mean_e K[e, r] − mean K``. Every member of the family
    ``a_e + b_r`` is reachable and this one minimises the squared error, so a low agreement with it
    is a statement about the keys and never about the fit.
    """
    expected = n_entities * n_relations
    if keys.shape[0] != expected:
        raise ValueError(
            f"{keys.shape[0]} keys for a {n_entities}x{n_relations} grid (expected {expected}) — "
            "the surrogate is only defined on the complete grid"
        )
    k = keys.float().view(n_relations, n_entities, -1)
    grand = k.mean(dim=(0, 1), keepdim=True)
    per_entity = k.mean(dim=0, keepdim=True)  # a_e, averaged over relations
    per_relation = k.mean(dim=1, keepdim=True)  # b_r, averaged over entities
    return (per_entity + per_relation - grand).reshape(expected, -1)


@torch.no_grad()
def separability(
    keys: torch.Tensor,
    queries: torch.Tensor,
    fact_ids: np.ndarray,
    n_entities: int,
    n_relations: int,
) -> SeparabilityResult:
    """Measure how much of a key set's addressing survives an additively separable approximation.

    ``queries[i]`` targets ``fact_ids[i]``; the join is by fact id, never by row position.
    """
    if len(queries) != len(fact_ids):
        raise ValueError(f"{len(queries)} queries against {len(fact_ids)} targets")

    keys = keys.float()
    surrogate = additive_surrogate(keys, n_entities, n_relations)

    centred = keys - keys.mean(dim=0, keepdim=True)
    residual = float((keys - surrogate).pow(2).sum() / centred.pow(2).sum().clamp(min=1e-12))
    key_cos = float(F.cosine_similarity(keys, surrogate, dim=-1).mean())

    q = F.normalize(queries.float(), dim=-1)
    targets = torch.from_numpy(np.asarray(fact_ids)).to(q.device)

    def recall_at_1(key_set: torch.Tensor) -> float:
        got = (q @ F.normalize(key_set, dim=-1).T).argmax(dim=-1)
        return float((got == targets).float().mean())

    return SeparabilityResult(
        n_probes=len(fact_ids),
        separability_index=recall_at_1(surrogate),
        reference_recall_at_1=recall_at_1(keys),
        residual_fraction=residual,
        mean_key_cosine=key_cos,
        n_entities=n_entities,
        n_relations=n_relations,
    )


@dataclass
class EntityRecoveryResult:
    """The falsifiable number behind the two-stage entity-first hypothesis (M2 §10.1)."""

    n_probes: int
    #: fraction whose true ENTITY is the surrogate's top-1 entity, ignoring the relation
    entity_recovery_at_1: float
    #: m -> fraction whose true entity is in the surrogate's top-m entities
    curve: dict[int, float]
    n_relations: int

    def candidate_recall(self, m: int) -> float:
        """Recall of a candidate set built as *the top-m entities' slots*.

        Equal to entity recovery at `m` by construction: if the true entity is in the top-m, its
        `n_relations` slots are all in the candidate set, so the true slot is too. Dense re-rank
        over that set is exact (`tests/test_router.py`), so this IS the searchability number and
        the candidate set costs `m * n_relations` slots.
        """
        return self.curve[m]

    def to_dict(self) -> dict:
        return {
            "n_probes": self.n_probes,
            "entity_recovery_at_1": self.entity_recovery_at_1,
            "curve": {str(m): v for m, v in self.curve.items()},
            "n_relations": self.n_relations,
            "candidate_set_sizes": {str(m): m * self.n_relations for m in self.curve},
        }

    def __str__(self) -> str:
        pts = "  ".join(f"m={m}:{v:.1%}" for m, v in sorted(self.curve.items()))
        return f"entity recovery @1 {self.entity_recovery_at_1:.1%}   {pts}"


@torch.no_grad()
def entity_recovery(
    keys: torch.Tensor,
    queries: torch.Tensor,
    fact_ids: np.ndarray,
    n_entities: int,
    n_relations: int,
    ms: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128),
) -> EntityRecoveryResult:
    """Does the separable surrogate recover the ENTITY, even when it cannot resolve the relation?

    §11.3 found the additive surrogate of a bound key set fails *relation-blind*: it keeps the
    entity and loses the relation. That failure mode is a shortlist strategy if it is reliable —
    generate candidates by entity with the cheap separable score, then let an exact dense re-rank
    over `m * n_relations` slots pick the relation. This measures the only part that can fail.
    """
    surrogate = F.normalize(
        additive_surrogate(keys.float(), n_entities, n_relations), dim=-1
    )
    q = F.normalize(queries.float(), dim=-1)
    scores = q @ surrogate.T  # (P, n_entities * n_relations)

    # Best slot per entity: the shortlist is over entities, so a relation-blind score is fine.
    per_entity = scores.view(-1, n_relations, n_entities).max(dim=1).values  # (P, n_entities)
    truth = torch.from_numpy(np.asarray(fact_ids) % n_entities).to(q.device)

    order = per_entity.argsort(dim=-1, descending=True)
    rank = (order == truth.unsqueeze(1)).float().argmax(dim=-1)
    curve = {
        m: float((rank < m).float().mean()) for m in ms if m <= n_entities
    }
    return EntityRecoveryResult(
        n_probes=len(fact_ids),
        entity_recovery_at_1=curve.get(1, 0.0),
        curve=curve,
        n_relations=n_relations,
    )


#: Recorded because the first negative control for `entity_recovery` was wrong.
#:
#: Entity recovery does **not** fall when the key set is randomised: the surrogate's entity mean
#: contains the query whenever queries are drawn from the key set, so uniform random keys still
#: score 92.3%. It falls only when the *queries* carry no information about the keys. The quantity
#: therefore measures query-key alignment aggregated to entity level, and is high only where
#: `q · k(e, r)` is already large.
#:
#: **Consequence for reporting: entity recovery is read NEXT TO slot-level recall, never instead
#: of it.** A high figure on a key set whose slot recall is 0% would say the shortlist preserves
#: nothing useful, because there was nothing to preserve.
ENTITY_RECOVERY_CAVEAT = __doc__
