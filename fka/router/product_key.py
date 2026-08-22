"""R1 — product-key routing: O(√N) top-k over a factorised key space.

Research plan §3.3. A query is split in half; each half searches its own codebook of √N
half-keys; the Cartesian combination of the two shortlists is re-scored and the best `k` survive.
Cost is `O(√N·d)` instead of `O(N·d)`, which is the whole point at 10⁶–10⁹ slots.

Slot `(i, j)` has score::

    score(i, j) = q₁ · c₁ᵢ  +  q₂ · c₂ⱼ

and slot id `i * n_sub + j`.

Two properties that must be understood before any number from this module is interpreted
--------------------------------------------------------------------------------------

**1. The shortlist search is EXACT, not approximate.** The top-k of an outer *sum* is always
contained in `topk_half(a) × topk_half(b)` when `k_half ≥ k`: any pair outside both shortlists is
dominated by the pair of individual maxima. So "recall@k versus exact nearest neighbour over this
key set" is **1.0 by construction** — it is a theorem about outer sums, not a measurement of a
design. `test_router.py` asserts it as a *correctness* property, and it must never be reported as
an experimental result. (Research plan §3.5 lists "recall@k ≥ 0.95 vs exact NN" as a success
criterion; read literally against a product-key set it is unfalsifiable. The falsifiable version is
property 2.)

**2. The real question is expressivity, not search.** A product-key set is a rank-structured
subset of all possible key sets: `N` slots described by `2√N` vectors. It cannot represent an
arbitrary set of `N` keys. So the question Phase 2 actually has to answer is whether a *learned*
product-key set can reproduce the geometry the oracle gets from `normalize(e ⊙ r)` — including
which facts are near-addressable from which queries. That is measured by
`fka.eval.router_eval`, against binding confusers, and it is the number that can fail.

See `docs/decision_records/M2_router.md` §3 for the alignment hazard: making axis 1 the entity and
axis 2 the relation would supply the factorisation the experiment is supposed to test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ProductKeyConfig:
    """`n_slots` slots on an `n_sub1 × n_sub2` grid; slot id is `i * n_sub2 + j`.

    Axes default to `ceil(sqrt(n_slots))` each, which is the `O(√N)` design. They are separately
    sizable because the *asymmetric* case is a needed control, not a curiosity: our fact space
    factors as 2,000 entities × 4 relations, so a square grid **cannot** be aligned with the
    oracle's own factorisation. See `docs/decision_records/M2_router.md` §3.
    """

    n_slots: int
    query_dim: int = 64  # frozen M1 interface width
    topk: int = 8  # slots returned
    topk_half: int | None = None  # shortlist per half; defaults to `topk` (exact, see module doc)
    normalize_keys: bool = True
    axes: tuple[int, int] | None = None  # explicit (n_sub1, n_sub2); None -> square

    def __post_init__(self) -> None:
        if self.axes is not None and self.axes[0] * self.axes[1] < self.n_slots:
            raise ValueError(
                f"axes {self.axes} give {self.axes[0] * self.axes[1]} slots, "
                f"fewer than the {self.n_slots} required"
            )

    @property
    def n_sub1(self) -> int:
        return self.axes[0] if self.axes else self._square

    @property
    def n_sub2(self) -> int:
        return self.axes[1] if self.axes else self._square

    @property
    def _square(self) -> int:
        return math.isqrt(self.n_slots - 1) + 1 if self.n_slots > 1 else 1

    @property
    def half_dim(self) -> int:
        if self.query_dim % 2:
            raise ValueError(f"query_dim must be even to split in half, got {self.query_dim}")
        return self.query_dim // 2

    @property
    def k_half(self) -> int:
        return self.topk_half or self.topk


class ProductKeyRouter(nn.Module):
    """Two learned half-key codebooks with a shortlist-and-recombine search.

    ``forward(q)`` returns ``(slot_ids, scores)``, both ``(B, topk)``, sorted descending.
    """

    def __init__(self, cfg: ProductKeyConfig) -> None:
        super().__init__()
        if cfg.n_slots < 1:
            raise ValueError("n_slots must be positive")
        if cfg.k_half > min(cfg.n_sub1, cfg.n_sub2):
            raise ValueError(
                f"k_half={cfg.k_half} exceeds the {min(cfg.n_sub1, cfg.n_sub2)} half-keys on the "
                f"smaller axis; lower topk_half or use more slots"
            )
        self.cfg = cfg
        half = cfg.half_dim
        # Unit-norm init: with normalize_keys the scale is irrelevant, and without it this keeps
        # initial scores O(1) rather than O(sqrt(half_dim)).
        self.keys1 = nn.Parameter(F.normalize(torch.randn(cfg.n_sub1, half), dim=-1))
        self.keys2 = nn.Parameter(F.normalize(torch.randn(cfg.n_sub2, half), dim=-1))
        # Slots beyond n_slots exist only because n_sub^2 >= n_slots; they must never be returned.
        self.register_buffer(
            "_n_valid", torch.tensor(cfg.n_slots, dtype=torch.long), persistent=False
        )

    def extra_repr(self) -> str:
        c = self.cfg
        return (
            f"n_slots={c.n_slots}, axes={c.n_sub1}x{c.n_sub2}, query_dim={c.query_dim}, "
            f"topk={c.topk}, k_half={c.k_half}"
        )

    @property
    def n_params(self) -> int:
        return self.keys1.numel() + self.keys2.numel()

    def _halves(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if q.shape[-1] != self.cfg.query_dim:
            raise ValueError(f"query dim {q.shape[-1]} != configured {self.cfg.query_dim}")
        q1, q2 = q.split(self.cfg.half_dim, dim=-1)
        if self.cfg.normalize_keys:
            q1, q2 = F.normalize(q1, dim=-1), F.normalize(q2, dim=-1)
        return q1, q2

    def _codebooks(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.normalize_keys:
            return F.normalize(self.keys1, dim=-1), F.normalize(self.keys2, dim=-1)
        return self.keys1, self.keys2

    def forward(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Shortlist each half, recombine, return the best ``topk`` slots and their scores."""
        cfg = self.cfg
        q1, q2 = self._halves(q)
        c1, c2 = self._codebooks()

        # (B, k_half) each — the only place N enters is through the axis sizes, ~sqrt(N).
        v1, i1 = (q1 @ c1.T).topk(min(cfg.k_half, cfg.n_sub1), dim=-1)
        v2, i2 = (q2 @ c2.T).topk(min(cfg.k_half, cfg.n_sub2), dim=-1)

        # Cartesian recombination: (B, k1, k2).
        scores = v1.unsqueeze(-1) + v2.unsqueeze(-2)
        slots = i1.unsqueeze(-1) * cfg.n_sub2 + i2.unsqueeze(-2)
        scores, slots = scores.flatten(1), slots.flatten(1)

        # n_sub^2 can exceed n_slots; those ids address nothing.
        scores = scores.masked_fill(slots >= self._n_valid, float("-inf"))

        k = min(cfg.topk, scores.shape[-1])
        best, order = scores.topk(k, dim=-1)
        return slots.gather(-1, order), best

    def all_scores(self, q: torch.Tensor) -> torch.Tensor:
        """Differentiable scores for every slot — ``(B, n_slots)``.

        `O(N)` and therefore not a forward path; it exists so training can put a cross-entropy
        over all slots rather than over a shortlist, which would make the gradient depend on
        which slots the shortlist happened to return.
        """
        cfg = self.cfg
        q1, q2 = self._halves(q)
        c1, c2 = self._codebooks()
        full = ((q1 @ c1.T).unsqueeze(-1) + (q2 @ c2.T).unsqueeze(-2)).flatten(1)
        return full[:, : cfg.n_slots]

    @torch.no_grad()
    def exact_topk(
        self, q: torch.Tensor, k: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Brute-force top-k over all ``n_slots``. The reference the shortlist is checked against.

        Materialises the full ``n_sub × n_sub`` score matrix per query, so it is `O(N)` in both
        time and memory — usable as a test oracle and for small-N study points, never in a
        forward pass. Batch it in small chunks at large N.
        """
        cfg = self.cfg
        k = k or cfg.topk
        q1, q2 = self._halves(q)
        c1, c2 = self._codebooks()
        full = (q1 @ c1.T).unsqueeze(-1) + (q2 @ c2.T).unsqueeze(-2)  # (B, n_sub1, n_sub2)
        full = full.flatten(1)
        if cfg.n_sub1 * cfg.n_sub2 > cfg.n_slots:
            ids = torch.arange(full.shape[-1], device=full.device)
            full = full.masked_fill(ids >= self._n_valid, float("-inf"))
        best, slots = full.topk(min(k, cfg.n_slots), dim=-1)
        return slots, best

    @torch.no_grad()
    def slot_scores(self, q: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        """Score specific slots for specific queries — ``q`` (B, d), ``slots`` (B, m) -> (B, m).

        Used by the hard-negative evaluation, which needs the score of a *named* confuser slot
        rather than whatever the shortlist happened to return.
        """
        q1, q2 = self._halves(q)
        c1, c2 = self._codebooks()
        i, j = slots // self.cfg.n_sub2, slots % self.cfg.n_sub2
        s1 = (q1.unsqueeze(1) * c1[i]).sum(-1)
        s2 = (q2.unsqueeze(1) * c2[j]).sum(-1)
        return s1 + s2


def slot_id(i: int, j: int, n_sub2: int) -> int:
    return i * n_sub2 + j
