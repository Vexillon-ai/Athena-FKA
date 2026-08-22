"""The `KnowledgeStore` API — FROZEN 2026-08-02 (`docs/decision_records/M3_substrate.md` §10).

Frozen **before the first implementation**, with **Phase 4 as a co-client rather than a downstream
consumer**, so S1/S2/S3 are built against it and the diffusion retriever does not force a re-freeze.

Two latents, deliberately
-------------------------
``reconstruct`` returns the **lossy** post-compression latent — the integration path — while
``target`` returns the **clean** stored code — the denoiser's training signal. A store exposing only
one forces the other client to reach inside it, and reaching inside is how an interface stops being
one.

``recon_error`` is queryable rather than left to the caller, so the degradation surface stays
comparable across designs: the store owns the metric. The residual vector is still available to
anyone who wants it as ``reconstruct - target``.

``score`` is an ENERGY: **lower is better**
-------------------------------------------
Fixed here because three clients compare its outputs — retriever guidance, hallucination abstention
(a threshold), and dense re-rank. A sign flip between two designs would silently invert an
abstention rule, which is the kind of defect that reads as a model result. A cosine-similarity store
returns *negative* cosine.

Non-constraints, recorded so they are not accidentally imported
---------------------------------------------------------------
**Settling dynamics, noise schedules and step policies are RETRIEVER-owned.** The substrate is
store / reconstruct / score only: no iteration count, no temperature, no annealing schedule, no
opinion about how often ``score`` is called. An attractor-flavoured store invites a ``settle()``
method, and the moment it exists the Phase 3 / Phase 4 boundary is gone and neither can be measured
alone.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

#: The frozen M1 interface width. Every latent crossing this API is in this space, or passes through
#: ONE declared fixed adapter owned by the store. Internals may use any dimension; the public
#: geometry is single and fixed, because per-client geometries are how an interface stops being one.
M1_LATENT_DIM = 64


class KnowledgeStore(ABC):
    """Store, reconstruct, score. Nothing else."""

    #: Public latent width. Must equal :data:`M1_LATENT_DIM`; asserted by the contract suite.
    latent_dim: int = M1_LATENT_DIM

    # -- writing -----------------------------------------------------------------------------

    @abstractmethod
    def write(self, content: torch.Tensor) -> torch.Tensor:
        """``(N, latent_dim)`` latents -> ``(N,)`` slot ids, in the order given."""

    @abstractmethod
    def delete(self, slot_ids: torch.Tensor) -> None:
        """Remove slots. Reading a deleted slot is an error, not a zero vector."""

    # -- reading -----------------------------------------------------------------------------

    @abstractmethod
    def reconstruct(self, slot_ids: torch.Tensor) -> torch.Tensor:
        """The **lossy**, post-compression latent. The integration path consumes this."""

    @abstractmethod
    def target(self, slot_ids: torch.Tensor) -> torch.Tensor:
        """The **clean** stored code. Phase 4's denoiser trains toward this."""

    def recon_error(self, slot_ids: torch.Tensor) -> torch.Tensor:
        """Per-slot reconstruction error, ``(N,)``. Relative L2 unless a design overrides it.

        Queryable rather than caller-computed so the degradation surface is comparable across
        designs — the store owns the metric. Override only with a documented reason.
        """
        clean = self.target(slot_ids)
        err = (self.reconstruct(slot_ids) - clean).norm(dim=-1)
        return err / clean.norm(dim=-1).clamp(min=1e-9)

    # -- scoring -----------------------------------------------------------------------------

    @abstractmethod
    def score(self, latent: torch.Tensor, slot_ids: torch.Tensor) -> torch.Tensor:
        """ENERGY of ``latent`` against each slot — **lower is better**.

        ``latent`` is ``(B, latent_dim)``; ``slot_ids`` is ``(B, K)`` or ``(K,)`` broadcast over the
        batch. Returns ``(B, K)``. Batched because the retriever calls it per denoising step.
        """

    # -- the declared cost / invalidation model ----------------------------------------------

    @abstractmethod
    def declared_invalidation(self, slot_ids: torch.Tensor) -> torch.Tensor:
        """Slots whose reconstruction **may** have changed as a result of touching ``slot_ids``.

        The edit-locality instrument reads this **declaration**, never the store's internals, which
        makes the declaration itself testable in both directions:

        * a store declaring *everything* passes locality trivially and fails usefulness;
        * a store declaring *too little* is caught by measurement disagreeing with it.

        Both checks live in the contract suite.
        """

    @abstractmethod
    def cost_model(self) -> dict:
        """Per-design declared costs. **Required keys** (M3 §10.5, specified after the API froze):

        * ``shared_parameters`` — parameters the inference path needs, counted once;
        * ``per_fact_storage_bits`` — pointers + residual per fact, at their STORED width;
        * ``write_touches`` — prose, for the edit-locality declaration.

        Specifying the dict's contents is a refinement *within* the frozen surface: the method and
        its signature are unchanged, and every design is held to the same keys so the bits/bit
        comparison is apples to apples.
        """

    # -- helpers, not part of the contract ----------------------------------------------------

    def _broadcast_slots(self, latent: torch.Tensor, slot_ids: torch.Tensor) -> torch.Tensor:
        if slot_ids.dim() == 1:
            return slot_ids.unsqueeze(0).expand(latent.shape[0], -1)
        return slot_ids


class IdentityStore(KnowledgeStore):
    """No compression at all: ``reconstruct == target``. The reference implementation and the GATE.

    Its job is to be substituted wherever a real store goes, so that any shortfall observed with a
    real store can be attributed to compression rather than to the plumbing. Per the
    one-gate-per-eval-path rule, no substrate number is admissible until this reproduces the
    pre-substrate result exactly.
    """

    def __init__(self, latent_dim: int = M1_LATENT_DIM) -> None:
        self.latent_dim = latent_dim
        self._codes: torch.Tensor | None = None
        self._alive: torch.Tensor | None = None

    def write(self, content: torch.Tensor) -> torch.Tensor:
        if content.shape[-1] != self.latent_dim:
            raise ValueError(f"content is {content.shape[-1]}-dim, store is {self.latent_dim}")
        start = 0 if self._codes is None else self._codes.shape[0]
        self._codes = content.detach().clone() if self._codes is None else torch.cat(
            [self._codes, content.detach().clone()]
        )
        alive = torch.ones(content.shape[0], dtype=torch.bool, device=content.device)
        self._alive = alive if self._alive is None else torch.cat([self._alive, alive])
        return torch.arange(start, start + content.shape[0], device=content.device)

    def delete(self, slot_ids: torch.Tensor) -> None:
        self._alive[slot_ids] = False

    def _check(self, slot_ids: torch.Tensor) -> None:
        if self._codes is None:
            raise RuntimeError("nothing written")
        if not bool(self._alive[slot_ids].all()):
            raise KeyError("read of a deleted slot — deleted slots are an error, not a zero vector")

    def reconstruct(self, slot_ids: torch.Tensor) -> torch.Tensor:
        self._check(slot_ids)
        return self._codes[slot_ids]

    def target(self, slot_ids: torch.Tensor) -> torch.Tensor:
        self._check(slot_ids)
        return self._codes[slot_ids]

    def score(self, latent: torch.Tensor, slot_ids: torch.Tensor) -> torch.Tensor:
        slots = self._broadcast_slots(latent, slot_ids)
        keys = torch.nn.functional.normalize(self.reconstruct(slots), dim=-1)
        q = torch.nn.functional.normalize(latent, dim=-1).unsqueeze(1)
        return -(q * keys).sum(-1)  # negative cosine: an ENERGY, lower is better

    def declared_invalidation(self, slot_ids: torch.Tensor) -> torch.Tensor:
        return slot_ids  # nothing is shared, so a write touches only what it wrote

    def cost_model(self) -> dict:
        n = 0 if self._codes is None else self._codes.shape[0]
        return {
            "design": "identity",
            "n_slots": n,
            "shared_parameters": 0,
            # Identical accounting to every other design (M3 §10.5): a lossless store pays the
            # full fp32 latent per fact, which is exactly why it is the wrong end of the surface.
            "per_fact_storage_bits": self.latent_dim * 32,
            "per_fact_detail": {"pointer_bits": 0.0, "residual_bits": self.latent_dim * 32,
                                "residual_dim": self.latent_dim, "residual_width": 32},
            "write_touches": "the written slot only",
        }

    def __len__(self) -> int:
        return 0 if self._codes is None else int(self._alive.sum())

    def __repr__(self) -> str:
        return f"IdentityStore({len(self)} live slots, dim={self.latent_dim})"
