"""Dense scoring over a key set, plus its gold stub — the shared fork (a) eval instrument.

Moved here **verbatim** from `scripts/run_fork_a0.py` when the joint fit needed the same path.
Verbatim matters: Stage a0's numbers were produced by this code, and a re-implementation that
scored "the same way" would make the joint fit's numbers incomparable to a0's by exactly the amount
nobody would notice. It is the sibling-evaluator failure (M1 §5) avoided in advance rather than
diagnosed afterwards.

Dense, so slot id == fact id: there is no grid here, and `identity_slot_map` makes that explicit.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class DenseKeyRouter:
    """Adapter so a key set travels the same eval path as the product-key router."""

    def __init__(self, keys: torch.Tensor, k: int = 8):
        self.keys, self.k = keys, k

    def __call__(self, q):
        scores = F.normalize(q, dim=-1) @ F.normalize(self.keys, dim=-1).T
        v, i = scores.topk(min(self.k, self.keys.shape[0]), dim=-1)
        return i, v

    def slot_scores(self, q, slots):
        kn = F.normalize(self.keys, dim=-1)[slots]
        return (F.normalize(q, dim=-1).unsqueeze(1) * kn).sum(-1)


class GoldStubRouter(DenseKeyRouter):
    """Instrument gate: answers each query's true slot by construction.

    Per the one-gate-per-eval-path rule, no fork (a) number is admissible until this scores exactly
    1.0 recall@1 with a positive worst-case binding margin through the very same evaluator.
    """

    def __init__(self, targets: torch.Tensor, n_slots: int, k: int = 8):
        self.t, self.n_slots, self.k = targets, n_slots, k

    def __call__(self, q):
        a = self.t[: len(q)]
        filler = (a.unsqueeze(1) + torch.arange(1, self.k, device=q.device)) % self.n_slots
        return torch.cat([a.unsqueeze(1), filler], 1), torch.linspace(
            1, 0, self.k, device=q.device
        ).unsqueeze(0).repeat(len(q), 1)

    def slot_scores(self, q, slots):
        return (slots == self.t[: len(q)].unsqueeze(1)).float()
