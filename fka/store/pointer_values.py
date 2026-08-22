"""Values as POINTERS into shared tables — the value-side redesign (M3 §16).

§14.3 named the defect in the first build: it stored 8,000 value slots as if independent, when the
corpus says otherwise. `works_with` returns a partner's **entity code**, and those are already
stored for the key path; each attribute relation draws from a value space of 100–1,024. So a value
should cost a *pointer*, which is research plan §4.3's own spec for S1 and was simply not built.

The marginal cost is optimal by construction, and that is worth stating plainly
--------------------------------------------------------------------------------
With uniform value spaces, ``log2(|space|)`` **is** the value's entropy, so a pointer is the
entropy-optimal encoding. Measured on our corpus: 73,219 pointer bits against 73,218 knowledge bits.
**That is not a result about this design being clever** — it is what a pointer costs. The real
question is whether the *shared* machinery amortises, which is what M3 §15's load-knee regime exists
to measure, and why both the amortised and the marginal figure are always reported.

The tables are discovered, not declared
----------------------------------------
``write`` searches every shared table for the nearest entry, so the store *finds* the structure in
what it is given rather than being told where to look. A value that is genuinely in a table is
reconstructed **exactly**; one that is not degrades to its nearest neighbour, and ``recon_error``
says so. That keeps the design honest on inputs it was not built for.

The table selector is COUNTED, not derived
-------------------------------------------
A slot id determines its relation, so the table index could be derived for free rather than stored.
It is charged anyway (``log2(n_tables)`` bits per fact). Deriving it would be defensible and would
improve the headline; charging it is the conservative reading, and M3 §10.5 says nothing is exempt.
"""

from __future__ import annotations

import math

import torch

from fka.store.base import M1_LATENT_DIM, KnowledgeStore


class PointerValueStore(KnowledgeStore):
    """Shared value tables plus one pointer per fact. Exact whenever the value is in a table."""

    def __init__(
        self,
        tables: dict[str, torch.Tensor],
        *,
        latent_dim: int = M1_LATENT_DIM,
        shared_tables_are_free: frozenset[str] = frozenset(),
    ) -> None:
        """`shared_tables_are_free` names tables already paid for elsewhere (e.g. the entity codes,
        which the key path stores anyway). Their *contents* are not charged twice; their pointers
        always are."""
        if not tables:
            raise ValueError("a pointer store needs at least one table")
        self.latent_dim = latent_dim
        self.names = sorted(tables)
        self.tables = {k: tables[k].detach().clone() for k in self.names}
        self.free = frozenset(shared_tables_are_free)
        for name, t in self.tables.items():
            if t.shape[1] != latent_dim:
                raise ValueError(f"table {name!r} is {t.shape[1]}-dim, store is {latent_dim}")
        self._table_id: torch.Tensor | None = None
        self._index: torch.Tensor | None = None
        self._clean: torch.Tensor | None = None
        self._alive: torch.Tensor | None = None

    # -- storage arithmetic ------------------------------------------------------------------

    @property
    def selector_bits(self) -> float:
        return math.log2(len(self.names))

    def pointer_bits(self, name: str) -> float:
        return math.log2(self.tables[name].shape[0])

    @property
    def marginal_storage_bits(self) -> float:
        """Bits to store ONE more fact — the scaling limit (M3 §15.1).

        The mean over what has actually been written, so it reflects the corpus's own mix rather
        than an unweighted average over tables.
        """
        if self._table_id is None or not len(self._table_id):
            return self.selector_bits + max(self.pointer_bits(n) for n in self.names)
        per = torch.tensor([self.pointer_bits(n) for n in self.names])
        return float(per[self._table_id.cpu()].mean()) + self.selector_bits

    @property
    def shared_parameters(self) -> int:
        return sum(
            t.numel() for n, t in self.tables.items() if n not in self.free
        )

    # -- the API -----------------------------------------------------------------------------

    def write(self, content: torch.Tensor) -> torch.Tensor:
        if content.shape[-1] != self.latent_dim:
            raise ValueError(f"content is {content.shape[-1]}-dim, store is {self.latent_dim}")
        best_d = None
        best_t = best_i = None
        for tid, name in enumerate(self.names):
            table = self.tables[name].to(content.device)
            d, i = torch.cdist(content, table).min(dim=1)
            if best_d is None:
                best_d, best_t, best_i = d, torch.full_like(i, tid), i
            else:
                take = d < best_d
                best_d = torch.where(take, d, best_d)
                best_t = torch.where(take, torch.full_like(i, tid), best_t)
                best_i = torch.where(take, i, best_i)

        start = 0 if self._table_id is None else self._table_id.shape[0]
        alive = torch.ones(content.shape[0], dtype=torch.bool, device=content.device)

        def cat(old, new):
            return new if old is None else torch.cat([old.to(new.device), new])

        self._table_id = cat(self._table_id, best_t)
        self._index = cat(self._index, best_i)
        self._clean = cat(self._clean, content.detach().clone())
        self._alive = cat(self._alive, alive)
        return torch.arange(start, start + content.shape[0], device=content.device)

    def delete(self, slot_ids: torch.Tensor) -> None:
        self._alive[slot_ids] = False

    def _check(self, slot_ids: torch.Tensor) -> None:
        if self._table_id is None:
            raise RuntimeError("nothing written")
        if not bool(self._alive[slot_ids].all()):
            raise KeyError("read of a deleted slot — deleted slots are an error, not a zero vector")

    def reconstruct(self, slot_ids: torch.Tensor) -> torch.Tensor:
        self._check(slot_ids)
        flat = slot_ids.reshape(-1)
        tid, idx = self._table_id[flat], self._index[flat]
        out = torch.empty(flat.shape[0], self.latent_dim, device=flat.device)
        for t, name in enumerate(self.names):
            m = tid == t
            if bool(m.any()):
                out[m] = self.tables[name].to(out.device)[idx[m]]
        return out.reshape(*slot_ids.shape, self.latent_dim)

    def target(self, slot_ids: torch.Tensor) -> torch.Tensor:
        self._check(slot_ids)
        return self._clean[slot_ids]

    def score(self, latent: torch.Tensor, slot_ids: torch.Tensor) -> torch.Tensor:
        slots = self._broadcast_slots(latent, slot_ids)
        keys = torch.nn.functional.normalize(self.reconstruct(slots), dim=-1)
        q = torch.nn.functional.normalize(latent, dim=-1).unsqueeze(1)
        return -(q * keys).sum(-1)  # ENERGY: lower is better

    def declared_invalidation(self, slot_ids: torch.Tensor) -> torch.Tensor:
        """Only the written slots: the tables are shared but never modified by a write."""
        return slot_ids

    def cost_model(self) -> dict:
        n = 0 if self._table_id is None else self._table_id.shape[0]
        return {
            "design": "S1-pointer-values",
            "n_slots": n,
            "shared_parameters": self.shared_parameters,
            "per_fact_storage_bits": self.marginal_storage_bits,
            "per_fact_detail": {
                "selector_bits": self.selector_bits,
                "pointer_bits_by_table": {n_: self.pointer_bits(n_) for n_ in self.names},
                "tables_not_charged": sorted(self.free),
            },
            "write_touches": "the written slot only (tables are read-only)",
        }

    def __len__(self) -> int:
        return 0 if self._alive is None else int(self._alive.sum())

    def __repr__(self) -> str:
        return (
            f"PointerValueStore({len(self)} slots, {len(self.names)} tables, "
            f"{self.marginal_storage_bits:.2f} bits/fact marginal)"
        )
