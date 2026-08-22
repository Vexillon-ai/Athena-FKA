"""D3: a memory addressed by continuous vectors, over a FIXED random codebook.

Design D3 in research plan §2.3 — the kernel emits query *vectors* rather than text and receives
*latents* rather than strings. The oracle here is deliberately frozen: no part of the codebook is
learnable, so nothing the kernel achieves can come from the memory reshaping itself to be easy.

The central design decision, and the reason a naive version of this is worthless
--------------------------------------------------------------------------------
If each fact's address were an independent random vector, the kernel would have to learn one
mapping per fact — storing the corpus in its own weights, which is exactly what this architecture
exists to avoid, and which would make an entity-level holdout impossible to pass by construction.

So addresses are **composed, not enumerated**::

    key(entity, relation) = normalize( entity_code[entity] ⊙ relation_code[relation] )

with element-wise product as the binding operation over fixed random codes (the cheap
Hadamard/HRR-style binding). The kernel therefore has one thing to learn — *how to bind* — and it
generalises to entities it has never addressed, because binding is entity-agnostic. That is what
makes "compose the hop-2 query from the hop-1 latent" a real, falsifiable capability rather than a
lookup table.

Values follow the same logic:

* ``works_with`` returns the partner's **entity code** — so a retrieved latent can be re-bound
  with the next relation code to form the following query. This is the composition path.
* attribute relations return a fixed code per **value**, not per fact. The kernel's readout has to
  decode ~1,600 value codes regardless of how many facts exist, so decoding capacity does not grow
  with the corpus. Names are never decoded to text at all — only re-bound.

Retrieval is soft (softmax over key similarity) during training so gradients reach the query
vector, and hard (argmax) at evaluation so the reported number reflects real addressing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from fka.data.corpus_gen import KnowledgeCorpus


def _random_codes(n: int, dim: int, generator: torch.Generator) -> torch.Tensor:
    """``n`` fixed unit vectors in ``dim`` dimensions."""
    codes = torch.randn(n, dim, generator=generator)
    return F.normalize(codes, dim=-1)


@dataclass
class LatentCodebook:
    """Frozen codes for entities, relations, and attribute values."""

    entity: torch.Tensor  # (n_entities, dim)
    relation: dict[str, torch.Tensor]  # relation -> (dim,)
    value: dict[str, torch.Tensor]  # relation -> (n_values, dim)
    dim: int

    @classmethod
    def build(cls, corpus: KnowledgeCorpus, dim: int, seed: int = 0) -> LatentCodebook:
        g = torch.Generator().manual_seed(seed)
        entity = _random_codes(corpus.n_entities, dim, g)
        relation = {r: _random_codes(1, dim, g)[0] for r in corpus.relations}
        value = {
            r: _random_codes(corpus.spaces[r].size, dim, g)
            for r in corpus.relations
            if r != "works_with"
        }
        return cls(entity=entity, relation=relation, value=value, dim=dim)

    def to(self, device) -> LatentCodebook:
        return LatentCodebook(
            entity=self.entity.to(device),
            relation={k: v.to(device) for k, v in self.relation.items()},
            value={k: v.to(device) for k, v in self.value.items()},
            dim=self.dim,
        )

    def bind(self, entity_code: torch.Tensor, relation: str) -> torch.Tensor:
        """The address of ``(entity, relation)``. Entity-agnostic, hence generalisable."""
        return F.normalize(entity_code * self.relation[relation], dim=-1)


class OracleLatentMemory:
    """Frozen key/value tables over every fact, addressed by cosine similarity.

    Shares the spirit of :class:`fka.kernel.memory.OracleTextMemory` — ground truth, no learning,
    hit/miss accounting — but the interface is vectors on both sides.
    """

    def __init__(
        self,
        corpus: KnowledgeCorpus,
        codebook: LatentCodebook,
        *,
        temperature: float = 0.05,
        enabled: bool = True,
    ) -> None:
        self.corpus = corpus
        self.codebook = codebook
        self.temperature = temperature
        self.enabled = enabled

        keys, values, self.fact_index = [], [], {}
        for relation in corpus.relations:
            rel_code = codebook.relation[relation]
            for entity_id in range(corpus.n_entities):
                key = F.normalize(codebook.entity[entity_id] * rel_code, dim=-1)
                if relation == "works_with":
                    partner = int(corpus.values["works_with"][entity_id][0])
                    value = codebook.entity[partner]
                else:
                    value_idx = int(corpus.values[relation][entity_id])
                    value = codebook.value[relation][value_idx]
                self.fact_index[(entity_id, relation)] = len(keys)
                keys.append(key)
                values.append(value)

        self.keys = torch.stack(keys)  # (n_facts, dim)
        self.values_matrix = torch.stack(values)  # (n_facts, dim)

    def to(self, device) -> OracleLatentMemory:
        self.keys = self.keys.to(device)
        self.values_matrix = self.values_matrix.to(device)
        self.codebook = self.codebook.to(device)
        return self

    def read(self, query: torch.Tensor, *, hard: bool = False) -> torch.Tensor:
        """Retrieve latents for a batch of query vectors, shape ``(B, dim) -> (B, dim)``.

        Soft (default) keeps the operation differentiable so the query vector receives gradient.
        Hard argmax is what evaluation uses: a soft blend can look correct while the top-1 address
        is wrong, which would flatter the retrieval numbers.
        """
        if not self.enabled:
            return torch.zeros_like(query)
        query = F.normalize(query, dim=-1)
        scores = query @ self.keys.T
        if hard:
            return self.values_matrix[scores.argmax(dim=-1)]
        return torch.softmax(scores / self.temperature, dim=-1) @ self.values_matrix

    def retrieved_index(self, query: torch.Tensor) -> torch.Tensor:
        """Which fact a query actually addresses — for attributing retrieval failures."""
        return (F.normalize(query, dim=-1) @ self.keys.T).argmax(dim=-1)

    def disabled_copy(self) -> OracleLatentMemory:
        clone = object.__new__(OracleLatentMemory)
        clone.__dict__ = dict(self.__dict__)
        clone.enabled = False
        return clone

    def __len__(self) -> int:
        return self.keys.shape[0]

    def __repr__(self) -> str:
        state = "enabled" if self.enabled else "DISABLED"
        return f"OracleLatentMemory({len(self):,} facts, dim={self.codebook.dim}, {state})"
