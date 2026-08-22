"""S1 — factorized triple codes (research plan §4.3), the first substrate build.

A fact costs a **pointer into shared codebooks** plus a **small residual**. Compression comes from
structure: related facts reuse the same shared components, so adding a fact that resembles existing
ones costs little. That is RQ3.2's claim, and this is the design that tests it most directly.

Why S1 first (M3 §6)
--------------------
It keeps ``content(e)`` an explicit, addressable object, so fork (c)'s ``f(content(e))`` attaches
with no adapter. And its failure would be the most informative: if *explicit* factorisation cannot
hold addressability, S2's implicit factorisation is unlikely to.

Construction
------------
``recon(slot) = normalize( sum_c codebook_c[ptr_c(slot)] + residual_basis @ coeff(slot) )``

* ``n_stages`` shared codebooks of ``codebook_size`` entries, fitted **once** and then FROZEN;
* per-slot pointers, one per stage, assigned greedily stage by stage against the running residual —
  the residual-quantization schedule, which is what makes the stages share work rather than compete;
* an optional ``residual_dim`` coefficient vector in a fixed random basis, carrying whatever the
  codebooks could not.

**The compression knob is (`n_stages`, `codebook_size`, `residual_dim`).** ``residual_dim ==
latent_dim`` reconstructs exactly and is the *lightest* point on the degradation surface: it makes
the plumbing measurable before compression is allowed to bite.

Frozen codebooks are a design decision with a declared consequence
------------------------------------------------------------------
Codebooks are fitted on the first ``write`` and never refitted, so a later write perturbs **only the
slot it wrote** — that is what ``declared_invalidation`` returns, and it is what makes edit-locality
attainable. A variant that refits on every write would have to declare *every* slot invalidated, and
the contract suite would hold it to that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from fka.store.base import M1_LATENT_DIM, KnowledgeStore


@dataclass(frozen=True)
class S1Config:
    latent_dim: int = M1_LATENT_DIM
    #: Shared codebooks. More stages = finer approximation, more bits per slot.
    n_stages: int = 4
    codebook_size: int = 256
    #: Residual coefficients per slot. `latent_dim` reconstructs exactly (the lightest point).
    residual_dim: int = 0
    #: STORED width of each residual coefficient. A first-class axis of the compression sweep
    #: (M3 §10.5 ruling 3), not a post-hoc trick: coefficients are genuinely quantised to this
    #: width at write time, so reconstruction degrades and the metric sees the saving.
    residual_bits: int = 32
    #: Rows per chunk for fitting and encoding. Bounds every intermediate to `chunk x K` rather
    #: than `N x K`, which is what made 2M abort natively (M3 §19.4).
    chunk: int = 65_536
    seed: int = 0

    @property
    def bits_per_slot(self) -> float:
        """Everything stored PER FACT: pointers at their entropy width plus quantised residual."""
        return (
            self.n_stages * math.log2(self.codebook_size)
            + self.residual_dim * self.residual_bits
        )


class S1FactorizedStore(KnowledgeStore):
    """Shared codebooks + per-slot pointers + optional residual."""

    def __init__(self, cfg: S1Config) -> None:
        self.cfg = cfg
        self.latent_dim = cfg.latent_dim
        if cfg.residual_dim > cfg.latent_dim:
            raise ValueError(
                f"residual_dim {cfg.residual_dim} exceeds latent_dim {cfg.latent_dim}; "
                "there is nothing beyond the space to carry"
            )
        g = torch.Generator().manual_seed(cfg.seed)
        # ORTHONORMAL rows, via QR. A random normalised-row matrix is not orthogonal, so
        # `basis.T @ basis` would not be a projector and even residual_dim == latent_dim would
        # fail to reconstruct — which is exactly what the contract suite caught. With an
        # orthonormal basis the residual path is an honest orthogonal projection at every r, and
        # exact at r == latent_dim.
        self._basis = (
            torch.linalg.qr(
                torch.randn(cfg.latent_dim, cfg.residual_dim, generator=g)
            ).Q.T.contiguous()
            if cfg.residual_dim
            else None
        )
        self._codebooks: torch.Tensor | None = None  # (n_stages, codebook_size, latent_dim)
        self._ptr: torch.Tensor | None = None  # (n_slots, n_stages)
        self._coeff: torch.Tensor | None = None  # (n_slots, residual_dim)
        self._clean: torch.Tensor | None = None  # (n_slots, latent_dim)
        self._alive: torch.Tensor | None = None

    # -- fitting ---------------------------------------------------------------------------

    def _assign(self, x: torch.Tensor, book: torch.Tensor) -> torch.Tensor:
        """Nearest codeword per row, in chunks. Intermediates are `chunk x K`, never `N x K`."""
        out = torch.empty(x.shape[0], dtype=torch.long, device=x.device)
        for start in range(0, x.shape[0], self.cfg.chunk):
            stop = min(start + self.cfg.chunk, x.shape[0])
            out[start:stop] = torch.cdist(x[start:stop], book).argmin(dim=1)
        return out

    def _fit_codebooks(self, content: torch.Tensor) -> None:
        """Fit once, from the first batch written, then freeze.

        Entries are drawn from the data itself (with a k-means refinement per stage) rather than
        randomly, because a random codebook makes the first stage useless and silently shifts all
        the work onto the residual — which would look like "S1 compresses badly" and would actually
        be initialisation.

        **Chunked, and the Lloyd update is a scatter rather than a loop over K.** The original wrote
        `for k in range(codebook_size): book[k] = residual[assign == k].mean(0)`, which allocates a
        masked copy per centroid; at 500k entities that drove peak allocation to 25 GB and at 2M it
        aborted natively out of the HIP allocator. `index_add_` does the same arithmetic in one pass
        with no per-centroid temporaries (M3 §19.4).
        """
        cfg = self.cfg
        g = torch.Generator(device="cpu").manual_seed(cfg.seed)
        books, residual = [], content.detach().clone()
        for _ in range(cfg.n_stages):
            n = residual.shape[0]
            idx = torch.randperm(n, generator=g)[: cfg.codebook_size].to(residual.device)
            book = residual[idx].clone()
            if book.shape[0] < cfg.codebook_size:  # pad a small corpus
                pad = cfg.codebook_size - book.shape[0]
                book = torch.cat([book, residual[torch.randint(0, n, (pad,), generator=g)]])
            for _ in range(5):  # Lloyd iterations; cheap and worth it
                assign = self._assign(residual, book)
                sums = torch.zeros_like(book).index_add_(0, assign, residual)
                counts = torch.zeros(cfg.codebook_size, device=book.device).index_add_(
                    0, assign, torch.ones(n, device=book.device)
                )
                nonempty = counts > 0
                book = torch.where(
                    nonempty.unsqueeze(1), sums / counts.clamp(min=1).unsqueeze(1), book
                )
            books.append(book)
            residual = residual - book[self._assign(residual, book)]
        self._codebooks = torch.stack(books)

    def _encode(self, content: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Greedy residual quantization against the frozen codebooks, then the residual basis."""
        residual = content.detach().clone()
        ptrs = []
        for stage in range(self.cfg.n_stages):
            book = self._codebooks[stage]
            idx = self._assign(residual, book)
            ptrs.append(idx)
            residual = residual - book[idx]
        ptr = torch.stack(ptrs, dim=1)
        if self._basis is None:
            return ptr, torch.zeros(content.shape[0], 0, device=content.device)
        coeff = residual @ self._basis.to(residual.device).T
        return ptr, self._quantise(coeff)

    def _quantise(self, coeff: torch.Tensor) -> torch.Tensor:
        """Uniform quantisation of the residual coefficients to `residual_bits`.

        Applied at WRITE time so the stored object really is this coarse. If quantisation were
        applied only in the accounting, the sweep would report savings the reconstruction never
        paid for — which is the "post-hoc trick" the ruling forbids.
        """
        bits = self.cfg.residual_bits
        if bits >= 32 or coeff.numel() == 0:
            return coeff
        scale = coeff.abs().max().clamp(min=1e-12)
        levels = 2 ** (bits - 1) - 1  # signed, symmetric
        return torch.round(coeff / scale * levels) / levels * scale

    # -- the API ---------------------------------------------------------------------------

    def write(self, content: torch.Tensor) -> torch.Tensor:
        if content.shape[-1] != self.latent_dim:
            raise ValueError(f"content is {content.shape[-1]}-dim, store is {self.latent_dim}")
        if self._codebooks is None:
            self._fit_codebooks(content)
        else:
            self._codebooks = self._codebooks.to(content.device)

        ptr, coeff = self._encode(content)
        start = 0 if self._ptr is None else self._ptr.shape[0]
        alive = torch.ones(content.shape[0], dtype=torch.bool, device=content.device)

        def cat(old, new):
            return new if old is None else torch.cat([old.to(new.device), new])

        self._ptr = cat(self._ptr, ptr)
        self._coeff = cat(self._coeff, coeff)
        self._clean = cat(self._clean, content.detach().clone())
        self._alive = cat(self._alive, alive)
        return torch.arange(start, start + content.shape[0], device=content.device)

    def delete(self, slot_ids: torch.Tensor) -> None:
        self._alive[slot_ids] = False

    def _check(self, slot_ids: torch.Tensor) -> None:
        if self._ptr is None:
            raise RuntimeError("nothing written")
        if not bool(self._alive[slot_ids].all()):
            raise KeyError("read of a deleted slot — deleted slots are an error, not a zero vector")

    def reconstruct(self, slot_ids: torch.Tensor) -> torch.Tensor:
        self._check(slot_ids)
        flat = slot_ids.reshape(-1)
        ptr = self._ptr[flat]
        out = torch.zeros(flat.shape[0], self.latent_dim, device=flat.device)
        for stage in range(self.cfg.n_stages):
            out = out + self._codebooks[stage][ptr[:, stage]]
        if self._basis is not None and self._basis.shape[0]:
            out = out + self._coeff[flat] @ self._basis.to(out.device)
        return out.reshape(*slot_ids.shape, self.latent_dim)

    def target(self, slot_ids: torch.Tensor) -> torch.Tensor:
        self._check(slot_ids)
        return self._clean[slot_ids]

    def score(self, latent: torch.Tensor, slot_ids: torch.Tensor) -> torch.Tensor:
        slots = self._broadcast_slots(latent, slot_ids)
        keys = F.normalize(self.reconstruct(slots), dim=-1)
        q = F.normalize(latent, dim=-1).unsqueeze(1)
        return -(q * keys).sum(-1)  # ENERGY: lower is better

    def declared_invalidation(self, slot_ids: torch.Tensor) -> torch.Tensor:
        """Only the touched slots: codebooks freeze after the first write, so nothing else moves.

        A variant refitting codebooks on every write would have to return every live slot here, and
        the contract suite would hold it to the declaration.
        """
        return slot_ids

    def cost_model(self) -> dict:
        cfg = self.cfg
        n = 0 if self._ptr is None else self._ptr.shape[0]
        shared = cfg.n_stages * cfg.codebook_size * cfg.latent_dim
        return {
            "design": "S1-factorized",
            "n_slots": n,
            # Everything the inference path needs (M3 §10.5). Pointers are NOT exempt: treating
            # them as "data rather than parameters" is gaming counterexample 1.
            "shared_parameters": shared,
            "per_fact_storage_bits": cfg.bits_per_slot,
            "per_fact_detail": {
                "pointer_bits": cfg.n_stages * math.log2(cfg.codebook_size),
                "residual_bits": cfg.residual_dim * cfg.residual_bits,
                "residual_dim": cfg.residual_dim,
                "residual_width": cfg.residual_bits,
            },
            "bits_per_slot": cfg.bits_per_slot,
            "pointers_per_slot": cfg.n_stages,
            "write_touches": "the written slot only (codebooks frozen after first write)",
        }

    def __len__(self) -> int:
        return 0 if self._alive is None else int(self._alive.sum())

    def __repr__(self) -> str:
        c = self.cfg
        return (
            f"S1FactorizedStore({len(self)} live slots, {c.n_stages}x{c.codebook_size} codebooks, "
            f"residual_dim={c.residual_dim}, {c.bits_per_slot:.0f} bits/slot)"
        )
