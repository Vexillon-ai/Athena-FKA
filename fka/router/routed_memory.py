"""The Phase 2 stack: LEARNED addresses over the frozen value substrate.

Phase 2 replaces the oracle's *addressing*, not its values. The codebook stays frozen — nothing the
router achieves may come from the values reshaping themselves to be easy — so this wraps a
:class:`~fka.router.composed_keys.ComposedKeyTable` around
:class:`~fka.kernel.latent_memory.OracleLatentMemory`'s value matrix and presents the identical
interface the kernel already speaks (``read`` / ``retrieved_index`` / ``keys``).

Presenting the same interface is the point, not convenience: it means the *deployed* evaluator
(``evaluate_d3``) scores the learned stack through the same code path that produced M1's numbers,
so a Phase 2 result and a Phase 1 result are comparable without a second evaluator to keep in sync.
That is the one-gate-per-eval-path rule read forwards instead of backwards.

Slot ids are fact ids
---------------------
Dense scoring, no grid, keys built in corpus fact-id order (``relation_index * n_entities +
entity_index``) — so ``retrieved_index`` returns a fact id directly and the comparison against
``hop_fact_index`` needs no translation table. A translation table is exactly where an off-by-one
between "slot" and "fact" would hide, and ``gold_keys_reproduce_the_oracle`` in the joint driver
exists to catch it if this ever stops being true.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F

from fka.kernel.latent_memory import OracleLatentMemory
from fka.router.composed_keys import ComposedKeyTable


class RoutedLatentMemory:
    """Learned composed keys, frozen oracle values, oracle-compatible interface."""

    def __init__(
        self,
        oracle: OracleLatentMemory,
        table: ComposedKeyTable,
        relations: list[str],
        *,
        temperature: float | None = None,
    ) -> None:
        self.oracle = oracle
        self.table = table
        self.relations = relations
        self.corpus = oracle.corpus
        self.codebook = oracle.codebook
        self.values_matrix = oracle.values_matrix
        self.fact_index = oracle.fact_index
        self.temperature = oracle.temperature if temperature is None else temperature
        self.enabled = True

        n_e, n_r = oracle.corpus.n_entities, len(relations)
        dev = oracle.values_matrix.device
        # Fact-id order, built once: fact_id = relation_index * n_entities + entity_index.
        self._entity_ids = torch.arange(n_e * n_r, device=dev) % n_e
        self._relation_ids = torch.arange(n_e * n_r, device=dev) // n_e
        self._frozen_keys: torch.Tensor | None = None
        self._step_keys: torch.Tensor | None = None

    # -- keys -------------------------------------------------------------------------------

    def compute_keys(self) -> torch.Tensor:
        """Every key, differentiable, in fact-id order."""
        return F.normalize(self.table(self._entity_ids, self._relation_ids), dim=-1)

    @property
    def keys(self) -> torch.Tensor:
        """The key matrix — frozen snapshot, per-step cache, or freshly computed, in that order."""
        if self._frozen_keys is not None:
            return self._frozen_keys
        if self._step_keys is not None:
            return self._step_keys
        return self.compute_keys()

    @contextmanager
    def cached_keys(self):
        """Compute the key table **once** for the enclosing training step, keeping it in the graph.

        The kernel calls ``read`` once per hop and again on the final pass, so a naive property
        would rebuild all `N` keys three or four times per step. Caching a single tensor and reusing
        it is also the autograd-correct choice: one node with several consumers accumulates
        gradient, whereas repeated construction would build several independent subgraphs of the
        same parameters — same answer, more memory, more time.
        """
        previous = self._step_keys
        self._step_keys = self.compute_keys()
        try:
            yield self
        finally:
            self._step_keys = previous

    def freeze_keys(self) -> RoutedLatentMemory:
        """Snapshot the current keys for evaluation. Undone by :meth:`thaw_keys`."""
        with torch.no_grad():
            self._frozen_keys = self.compute_keys().detach()
        return self

    def thaw_keys(self) -> RoutedLatentMemory:
        self._frozen_keys = None
        return self

    # -- the OracleLatentMemory interface ----------------------------------------------------

    def read(self, query: torch.Tensor, *, hard: bool = False) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros_like(query)
        # .to(query.dtype): the kernel hands us fp32 while an autocast region may have produced
        # bf16 keys, and a mixed matmul refuses. Exactly the dtype mismatch that reached a real
        # sweep run through the subject-projection path, so it is handled rather than assumed.
        scores = F.normalize(query, dim=-1) @ self.keys.to(query.dtype).T
        if hard:
            return self.values_matrix[scores.argmax(dim=-1)]
        return torch.softmax(scores / self.temperature, dim=-1) @ self.values_matrix.to(
            scores.dtype
        )

    def retrieved_index(self, query: torch.Tensor) -> torch.Tensor:
        return (F.normalize(query, dim=-1) @ self.keys.to(query.dtype).T).argmax(dim=-1)

    def to(self, device) -> RoutedLatentMemory:
        self.oracle = self.oracle.to(device)
        self.table = self.table.to(device)
        self.values_matrix = self.oracle.values_matrix
        self.codebook = self.oracle.codebook
        self._entity_ids = self._entity_ids.to(device)
        self._relation_ids = self._relation_ids.to(device)
        if self._frozen_keys is not None:
            self._frozen_keys = self._frozen_keys.to(device)
        return self

    def disabled_copy(self) -> RoutedLatentMemory:
        clone = object.__new__(RoutedLatentMemory)
        clone.__dict__ = dict(self.__dict__)
        clone.enabled = False
        return clone

    def __len__(self) -> int:
        return int(self._entity_ids.shape[0])

    def __repr__(self) -> str:
        frozen = "frozen keys" if self._frozen_keys is not None else "live keys"
        return f"RoutedLatentMemory({len(self):,} facts, {self.table.cfg.mode}, {frozen})"


class _OracleKeyStub(RoutedLatentMemory):
    """Instrument gate: the learned stack wearing the ORACLE's keys.

    Per the one-gate-per-eval-path rule, the learned-stack eval path gets its own stub before it
    reports a number. Substituting the oracle's own keys must reproduce the oracle's own retrieval
    **exactly**; any shortfall is the wrapper — a fact-id/slot transposition, a stale key cache, a
    device mismatch — and not the router, because no router is involved.
    """

    def __init__(self, oracle: OracleLatentMemory, relations: list[str]) -> None:
        self.oracle = oracle
        self.table = None
        self.relations = relations
        self.corpus = oracle.corpus
        self.codebook = oracle.codebook
        self.values_matrix = oracle.values_matrix
        self.fact_index = oracle.fact_index
        self.temperature = oracle.temperature
        self.enabled = True
        self._step_keys = None
        self._frozen_keys = oracle.keys
        n_e, n_r = oracle.corpus.n_entities, len(relations)
        dev = oracle.values_matrix.device
        self._entity_ids = torch.arange(n_e * n_r, device=dev) % n_e
        self._relation_ids = torch.arange(n_e * n_r, device=dev) // n_e

    def compute_keys(self) -> torch.Tensor:
        return self._frozen_keys

    def __repr__(self) -> str:
        return f"_OracleKeyStub({len(self):,} facts)"


def oracle_key_stub(oracle: OracleLatentMemory, relations: list[str]) -> RoutedLatentMemory:
    """The gate for the learned-stack eval path. See :class:`_OracleKeyStub`."""
    return _OracleKeyStub(oracle, relations)
