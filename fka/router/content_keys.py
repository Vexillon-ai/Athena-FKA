"""Fork (c): keys COMPUTED FROM CONTENT. No parameter on the key path is indexed by entity id.

M2 §9.1-rev. This module exists because §9.1's original invariant was satisfied by a design that
failed for exactly the reason §9.1 named.

What went wrong with the count invariant
----------------------------------------
§9.1 forbade free per-`(e, r)` keys and enforced it as a **parameter-shape invariant**: parameter
count must be `O(n_e + n_r)`, not `O(n_e x n_r)`. `ComposedKeyTable` satisfies that and still keeps
a free **per-entity** embedding — a parameter row that a held-out entity alone would touch, and
which therefore receives no positive supervision. Measured (§11.3): never-supervised entities
scored **0.0% (0/154)** while new `(e, r)` *pairings* of trained entities scored **100.0% (37/37)**.
A switch, not a gradient.

The count was never the property that mattered. **Reachability** was:

    every input to the key computation must be derivable at inference time for an entity the
    router has never seen, and no parameter row may be reachable only from that entity.

So the entity side becomes a **learned encoder over the frozen codebook latent**, and the codebook
is data — the very vector the kernel already receives as `subject_code`. Adding an entity to the
world adds **zero parameters**; `n_entities` appears in no parameter shape at all, which is the
invariant `tests/test_content_keys.py` asserts by gradient support rather than by counting.

The relation table stays a free embedding, and that is legitimate: relations are a **closed
vocabulary and interface machinery**, fixed by the schema. There is no held-out relation to
generalise to, so there is no unreachable row to create.

What a failure would mean here, and why it differs from Stage A
---------------------------------------------------------------
**The oracle's own solution is inside this hypothesis class.** With `f` linear (the identity is
available), `g` returning the relation code, and a bilinear composition — `Bilinear(x, y)_k =
x^T W_k y` reproduces the Hadamard product at `W_k = e_k e_k^T` — the composition can express
`normalize(e_code (x) r_code)` exactly. Stage A's finding was *representational*: no parameters
existed that could express the target. Here they do exist. **A fork (c) failure would therefore be
an optimisation or generalisation finding, not a representational one**, and must be reported as
such rather than as another "cannot express" result.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ContentKeyConfig:
    n_relations: int
    #: Width of the frozen codebook latent — the encoder's INPUT, not a parameter shape.
    latent_dim: int = 64
    key_dim: int = 64  # frozen M1 interface width
    comp_dim: int = 128
    hidden: int = 256
    encoder_hidden: int = 256
    mode: str = "bilinear"  # "mlp" | "bilinear", downstream of the encoder
    normalize: bool = True


class ContentKeyTable(nn.Module):
    """`key(e, r) = compose(f(entity_code[e]), g(r))`, with `f` a learned encoder over content.

    `entity_code` is the FROZEN codebook, registered as a buffer: data the kernel already holds,
    never a parameter. So the parameter count is independent of `n_entities`, and a held-out
    entity's key travels the same weights every trained entity's key travels.
    """

    def __init__(self, cfg: ContentKeyConfig, entity_codes: torch.Tensor) -> None:
        super().__init__()
        self.cfg = cfg
        if entity_codes.shape[1] != cfg.latent_dim:
            raise ValueError(
                f"codes are {entity_codes.shape[1]}-dim, config says {cfg.latent_dim}"
            )
        # A buffer, not a parameter: it is the same vector the kernel receives as `subject_code`,
        # and it is frozen, so nothing here can learn a per-entity address.
        self.register_buffer("codes", entity_codes.detach().clone(), persistent=False)

        self.encode = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.encoder_hidden),
            nn.GELU(),
            nn.Linear(cfg.encoder_hidden, cfg.comp_dim),
        )
        self.relation = nn.Embedding(cfg.n_relations, cfg.comp_dim)
        nn.init.normal_(self.relation.weight, std=0.02)

        if cfg.mode == "mlp":
            self.compose = nn.Sequential(
                nn.Linear(2 * cfg.comp_dim, cfg.hidden),
                nn.GELU(),
                nn.Linear(cfg.hidden, cfg.key_dim),
            )
        elif cfg.mode == "bilinear":
            self.compose = nn.Bilinear(cfg.comp_dim, cfg.comp_dim, cfg.key_dim)
        else:
            raise ValueError(f"unknown composition mode {cfg.mode!r}")

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def keys_for_codes(
        self, entity_codes: torch.Tensor, relation_ids: torch.Tensor
    ) -> torch.Tensor:
        """The real signature: keys from *codes*, so no entity id is needed at all."""
        e = self.encode(entity_codes)
        r = self.relation(relation_ids)
        k = (
            self.compose(torch.cat([e, r], dim=-1))
            if self.cfg.mode == "mlp"
            else self.compose(e, r)
        )
        return F.normalize(k, dim=-1) if self.cfg.normalize else k

    def forward(self, entity_ids: torch.Tensor, relation_ids: torch.Tensor) -> torch.Tensor:
        """Id-taking form, for interface compatibility with the composed-key table.

        The ids index a frozen buffer, never a parameter, so this is a lookup of *data*.
        """
        return self.keys_for_codes(self.codes[entity_ids], relation_ids)

    def all_keys(self, n_entities: int | None = None) -> torch.Tensor:
        """Every key, in corpus fact-id order (`relation_index * n_entities + entity_index`)."""
        n_e = n_entities or self.codes.shape[0]
        dev = self.codes.device
        ents = torch.arange(n_e, device=dev).repeat(self.cfg.n_relations)
        rels = torch.arange(self.cfg.n_relations, device=dev).repeat_interleave(n_e)
        return self(ents, rels)


@torch.enable_grad()
def parameter_row_support(
    table: nn.Module, entity_id: int, relation_id: int, *, device=None
) -> set[tuple[str, int]]:
    """Which `(parameter name, row)` pairs the key for one `(e, r)` actually touches.

    Measured by gradient rather than by reading shapes, because the property under test is about
    what the computation *reaches*, and a shape can satisfy a count while hiding an unreachable row
    — which is precisely how `ComposedKeyTable` passed §9.1's original invariant and failed its
    rationale (§11.3).
    """
    dev = device or next(table.parameters()).device
    table.zero_grad(set_to_none=True)
    key = table(
        torch.tensor([entity_id], device=dev), torch.tensor([relation_id], device=dev)
    )
    key.sum().backward()

    support: set[tuple[str, int]] = set()
    for name, prm in table.named_parameters():
        if prm.grad is None:
            continue
        g = prm.grad
        if g.dim() == 0:
            if float(g.abs()) > 0:
                support.add((name, 0))
            continue
        rows = g.reshape(g.shape[0], -1).abs().sum(dim=1)
        support.update((name, int(i)) for i in torch.nonzero(rows).flatten().tolist())
    table.zero_grad(set_to_none=True)
    return support
