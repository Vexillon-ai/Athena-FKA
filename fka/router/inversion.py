"""§10.2: recover the ENTITY from the query, then shortlist by nearest neighbour over the codebook.

The two-stage search hypothesis (§10.1) first tried to shortlist through the additive **surrogate of
the keys**, and got 66.1% entity recovery where the dense keys themselves scored 100%. That gap is
the clue: dense scoring at 100% proves the entity information is already **in the query** — the
correct slot cannot be top-1 of 8,000 otherwise. The surrogate was approximating the wrong object.

So approximate nothing. Learn `h : query -> entity_code` directly, and generate candidates by
nearest neighbour over the **entity codebook** — 2,000 unit vectors that exist at inference anyway.
The candidate set is the recovered entity's `n_relations` slots, and a dense re-rank over a
candidate set is exact by construction (`tests/test_router.py`), so entity recovery at `m` **is**
candidate recall at `m * n_relations`.

Reachability (§9.1-rev) holds: no parameter of `h` is indexed by entity id, so a never-supervised
entity's recovery travels the same weights as every other entity's.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class InverterConfig:
    latent_dim: int = 64
    hidden: int = 512
    n_layers: int = 2


class EntityInverter(nn.Module):
    """`h(query) -> entity_code`. A plain MLP; the point is the target, not the architecture."""

    def __init__(self, cfg: InverterConfig) -> None:
        super().__init__()
        self.cfg = cfg
        dims = [cfg.latent_dim] + [cfg.hidden] * cfg.n_layers
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:], strict=True):
            layers += [nn.Linear(a, b), nn.GELU()]
        layers.append(nn.Linear(dims[-1], cfg.latent_dim))
        self.net = nn.Sequential(*layers)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(F.normalize(query, dim=-1)), dim=-1)


class GoldStubInverter(nn.Module):
    """Instrument gate: returns the true entity code by construction.

    Per the one-gate-per-eval-path rule, no `h` number is admissible until this scores exactly 100%
    entity recovery at m=1 through the same evaluator.
    """

    def __init__(self, codes: torch.Tensor, entity_ids: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("answer", F.normalize(codes[entity_ids], dim=-1), persistent=False)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        return self.answer[: len(query)]


@dataclass
class RecoveryResult:
    """Entity recovery through a candidate shortlist over the codebook."""

    n_probes: int
    n_codebook: int
    n_relations: int
    curve: dict[int, float]

    @property
    def at_1(self) -> float:
        return self.curve.get(1, 0.0)

    def m_for(self, target: float) -> int | None:
        """Smallest `m` reaching `target` recovery, or None if the curve never does."""
        for m in sorted(self.curve):
            if self.curve[m] >= target:
                return m
        return None

    def to_dict(self) -> dict:
        return {
            "n_probes": self.n_probes,
            "n_codebook": self.n_codebook,
            "n_relations": self.n_relations,
            "entity_recovery_at_1": self.at_1,
            "curve": {str(m): v for m, v in self.curve.items()},
            "candidate_slots": {str(m): m * self.n_relations for m in self.curve},
            "m_for_99": self.m_for(0.99),
            "m_for_95": self.m_for(0.95),
        }

    def __str__(self) -> str:
        pts = "  ".join(f"m={m}:{v:.1%}" for m, v in sorted(self.curve.items()))
        m99 = self.m_for(0.99)
        cost = (
            f" ({m99 * self.n_relations} candidate slots of "
            f"{self.n_codebook * self.n_relations:,})" if m99 else ""
        )
        return (
            f"entity recovery @1 {self.at_1:.1%}   {pts}\n"
            f"      m for 99% = {m99 if m99 is not None else 'NOT REACHED'}{cost}"
        )


@torch.no_grad()
def entity_recovery_by_inversion(
    inverter: nn.Module,
    queries: torch.Tensor,
    entity_ids: np.ndarray,
    codebook: torch.Tensor,
    n_relations: int,
    ms: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256),
) -> RecoveryResult:
    """Rank the codebook by similarity to `h(query)`; where does the true entity land?

    `entity_ids[i]` is the true entity for `queries[i]`. Joined by value, never by row position.
    """
    if len(queries) != len(entity_ids):
        raise ValueError(f"{len(queries)} queries against {len(entity_ids)} entities")
    predicted = inverter(queries)
    scores = predicted @ F.normalize(codebook.to(predicted.dtype), dim=-1).T
    truth = torch.from_numpy(np.asarray(entity_ids)).to(scores.device)

    order = scores.argsort(dim=-1, descending=True)
    rank = (order == truth.unsqueeze(1)).float().argmax(dim=-1)
    n_e = codebook.shape[0]
    return RecoveryResult(
        n_probes=len(entity_ids),
        n_codebook=n_e,
        n_relations=n_relations,
        curve={m: float((rank < m).float().mean()) for m in ms if m <= n_e},
    )


def fit_inverter(
    inverter: nn.Module,
    queries: torch.Tensor,
    targets: torch.Tensor,
    *,
    steps: int,
    lr: float,
    batch: int = 1024,
    log_every: int = 500,
) -> list[float]:
    """Fit `h` by cosine loss toward the true entity code.

    Cosine rather than MSE because only the *direction* matters — the codebook is unit vectors and
    the shortlist scores by cosine, so fitting magnitude would optimise something never used.
    """
    opt = torch.optim.AdamW(inverter.parameters(), lr=lr)
    n = len(queries)
    losses = []
    for step in range(steps):
        idx = torch.randint(0, n, (min(batch, n),), device=queries.device)
        loss = (1 - F.cosine_similarity(inverter(queries[idx]), targets[idx], dim=-1)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if step % log_every == 0 or step == steps - 1:
            print(f"      step {step + 1:>5}/{steps}  cosine loss {losses[-1]:.5f}")
    return losses
