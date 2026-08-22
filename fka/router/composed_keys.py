"""Fork (a): router keys COMPUTED from entity and relation components, never stored per pair.

`docs/decision_records/M2_router.md` §9.1 makes this constraint non-negotiable, and the module is
written so that violating it requires deleting code rather than adding it: there is no parameter
tensor indexed by fact id anywhere here, and `n_facts` never appears in a parameter shape.

**Why the constraint is load-bearing.** A free per-`(e, r)` key makes the entity-level holdout
unpassable *in principle*: a held-out entity's key would be an independently-initialised parameter
with no computable path from anything the kernel can emit, so a 0% result would be a statement
about initialisation rather than about binding. It would also store one learned parameter per fact
— the corpus in the router's weights, which is the failure the whole architecture exists to avoid.

What Phase 2 must preserve is the **compositional property**, not `e ⊙ r` specifically. M1 §2.1
recorded Hadamard binding as an oracle-side assumption so that Phases 2–4 could reproduce its
*effect* by their own means; Stage A then showed that forcing that exact algebra through a
separable product key is a representational dead end (§8). So the composition here is *learned*,
and the only thing held fixed is that it is a function of the two factors.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ComposedKeyConfig:
    n_entities: int
    n_relations: int
    key_dim: int = 64  # frozen M1 interface width
    comp_dim: int = 128  # per-factor embedding width
    hidden: int = 256
    mode: str = "mlp"  # "mlp" | "bilinear"
    normalize: bool = True


class ComposedKeyTable(nn.Module):
    """`key(e, r) = compose(entity_emb[e], relation_emb[r])`, differentiable end to end.

    Parameter count is `O(n_entities + n_relations)`, **not** `O(n_entities x n_relations)`; that
    ratio is asserted in the tests, because it is the property the whole fork rests on.
    """

    def __init__(self, cfg: ComposedKeyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.entity = nn.Embedding(cfg.n_entities, cfg.comp_dim)
        self.relation = nn.Embedding(cfg.n_relations, cfg.comp_dim)
        nn.init.normal_(self.entity.weight, std=0.02)
        nn.init.normal_(self.relation.weight, std=0.02)

        if cfg.mode == "mlp":
            self.compose = nn.Sequential(
                nn.Linear(2 * cfg.comp_dim, cfg.hidden),
                nn.GELU(),
                nn.Linear(cfg.hidden, cfg.key_dim),
            )
        elif cfg.mode == "bilinear":
            # A low-rank bilinear map: keeps a multiplicative interaction available without
            # hard-coding Hadamard binding, which is exactly the algebra Stage A retired.
            self.compose = nn.Bilinear(cfg.comp_dim, cfg.comp_dim, cfg.key_dim)
        else:
            raise ValueError(f"unknown composition mode {cfg.mode!r}")

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, entity_ids: torch.Tensor, relation_ids: torch.Tensor) -> torch.Tensor:
        """`(N,)` entity ids and `(N,)` relation ids -> `(N, key_dim)` keys."""
        e = self.entity(entity_ids)
        r = self.relation(relation_ids)
        k = (
            self.compose(torch.cat([e, r], dim=-1))
            if self.cfg.mode == "mlp"
            else self.compose(e, r)
        )
        return F.normalize(k, dim=-1) if self.cfg.normalize else k

    def all_keys(self, n_entities: int | None = None) -> torch.Tensor:
        """Every key, in corpus fact-id order (`relation_index * n_entities + entity_index`)."""
        n_e = n_entities or self.cfg.n_entities
        dev = self.entity.weight.device
        ents = torch.arange(n_e, device=dev).repeat(self.cfg.n_relations)
        rels = torch.arange(self.cfg.n_relations, device=dev).repeat_interleave(n_e)
        return self(ents, rels)


@torch.no_grad()
def key_spread(keys: torch.Tensor, sample: int = 2048, seed: int = 0) -> dict:
    """Collapse statistics for failure mode 9.3(a) — reported pass or fail.

    `mean_cosine` near 1.0, or `effective_rank` near 1, means the address space has collapsed and
    any retrieval score is measuring agreement between two co-adapted halves rather than an
    address space (§9.6).
    """
    n = keys.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(n, generator=g)[: min(sample, n)].to(keys.device)
    k = F.normalize(keys[idx].float(), dim=-1)

    sims = k @ k.T
    off = ~torch.eye(len(k), dtype=torch.bool, device=k.device)
    # Participation ratio of the covariance spectrum: a smooth, threshold-free effective rank.
    ev = torch.linalg.eigvalsh(torch.cov(k.T)).clamp(min=0)
    eff_rank = float((ev.sum() ** 2) / (ev.pow(2).sum() + 1e-12))

    return {
        "n_sampled": int(len(k)),
        "mean_cosine": float(sims[off].mean()),
        "max_offdiag_cosine": float(sims[off].max()),
        "effective_rank": eff_rank,
        "key_dim": int(keys.shape[1]),
        "effective_rank_fraction": eff_rank / keys.shape[1],
    }
