"""The kernel's memory interface.

This is the seam the whole architecture turns on, so it is abstract from day one even though the
only implementation today is a Python dict. Phase 2-4 replace what sits behind it — a product-key
router into a compressed substrate, retrieved by iterative denoising — and nothing on the kernel
side should have to change when they do.

Two levels are anticipated:

* :class:`TextMemory` (this file, now) — the interface is *text*. The kernel emits
  ``<query>relation of subject</query>`` and the memory writes back
  ``<result>value</result>``. This is design D1 from research plan §2.3: teacher-force the
  retrieval interface with an oracle so kernel training decouples from Phases 2-4.
* ``LatentMemory`` (later) — the interface is a *vector*. The kernel emits a continuous query
  vector and receives a retrieved latent through cross-attention (design D3). The abstract base
  deliberately does not assume string-shaped queries beyond what :class:`TextMemory` needs, so
  adding it is an addition rather than a rewrite.

Every memory records its own hit/miss statistics. That is not bookkeeping for its own sake: when
the end-to-end system starts failing, research plan §6.4 wants each error attributed to a stage —
routing miss, substrate crosstalk, or denoiser hallucination — and a memory that silently returns
nothing is indistinguishable from a kernel that asked the wrong question unless it counts.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field

from fka.data.templates import QUERY_CLOSE, QUERY_OPEN, RESULT_CLOSE, RESULT_OPEN

#: How a query span's text encodes an address. Kept as one place so the kernel, the corpus
#: firewall renderer and the memory cannot drift apart on the format.
QUERY_FORMAT = "{relation} of {subject}"
_QUERY_RE = re.compile(r"^\s*(?P<relation>[A-Za-z_][A-Za-z0-9_]*)\s+of\s+(?P<subject>.+?)\s*$")


def format_query(subject: str, relation: str) -> str:
    return QUERY_FORMAT.format(relation=relation, subject=subject)


def parse_query(text: str) -> tuple[str, str] | None:
    """Parse a query span's contents into ``(subject, relation)``, or None if malformed.

    A malformed query is a real outcome, not an error: at evaluation the model generates these
    itself and an untrained kernel will emit nonsense. The caller counts it as a miss.
    """
    match = _QUERY_RE.match(text)
    if match is None:
        return None
    return match.group("subject"), match.group("relation")


@dataclass
class MemoryStats:
    """Per-run retrieval accounting, so failures can be attributed to a stage."""

    lookups: int = 0
    hits: int = 0
    misses: int = 0
    malformed: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def reset(self) -> None:
        self.lookups = self.hits = self.misses = self.malformed = 0

    def to_dict(self) -> dict:
        return {
            "lookups": self.lookups,
            "hits": self.hits,
            "misses": self.misses,
            "malformed": self.malformed,
            "hit_rate": self.hit_rate,
        }


class MemoryInterface(ABC):
    """Base class for anything the kernel can query for a fact."""

    def __init__(self) -> None:
        self.stats = MemoryStats()

    @property
    @abstractmethod
    def enabled(self) -> bool:
        """Whether the memory answers at all.

        Disabling is the leakage test (research plan §2.5): with memory off, fact recall must fall
        to near chance, or the kernel memorised parametrically and the capacity comparison is void.
        """

    @abstractmethod
    def lookup(self, subject: str, relation: str) -> str | None:
        """Return the stored value, or None for a miss."""

    def reset_stats(self) -> None:
        self.stats.reset()


class TextMemory(MemoryInterface):
    """A memory addressed by, and answering with, text spans."""

    def answer_span(self, query_text: str) -> str:
        """Turn a query span's contents into the full ``<result>…</result>`` span to splice in.

        Always returns a well-formed span, empty on a miss. An empty span is what the kernel must
        learn to treat as "memory could not help" — and it is exactly what the leakage test feeds
        it for every query.
        """
        self.stats.lookups += 1
        parsed = parse_query(query_text)
        if parsed is None:
            self.stats.malformed += 1
            return f"{RESULT_OPEN}{RESULT_CLOSE}"
        if not self.enabled:
            self.stats.misses += 1
            return f"{RESULT_OPEN}{RESULT_CLOSE}"
        value = self.lookup(*parsed)
        if value is None:
            self.stats.misses += 1
            return f"{RESULT_OPEN}{RESULT_CLOSE}"
        self.stats.hits += 1
        return f"{RESULT_OPEN}{value}{RESULT_CLOSE}"


@dataclass(eq=False)
class OracleTextMemory(TextMemory):
    """Ground-truth memory backed by the corpus oracle dict.

    Research plan §2.4 step 3: use a Python dict first so kernel training is decoupled from
    Phases 2-4. This is also the positive control — with it enabled, fact recall should exceed
    95%; anything less is a kernel or plumbing problem, not a memory problem.

    ``enabled=False`` is the leakage condition. It still consumes queries and returns
    well-formed empty spans, so the *only* thing that changes between the two conditions is
    whether the answer is present.
    """

    mapping: Mapping[tuple[str, str], str] = field(default_factory=dict)
    is_enabled: bool = True

    def __post_init__(self) -> None:
        super().__init__()
        self.mapping = dict(self.mapping)

    @classmethod
    def from_corpus(cls, corpus, fact_ids=None, *, enabled: bool = True) -> OracleTextMemory:
        return cls(mapping=corpus.as_oracle_dict(fact_ids), is_enabled=enabled)

    @property
    def enabled(self) -> bool:
        return self.is_enabled

    def lookup(self, subject: str, relation: str) -> str | None:
        return self.mapping.get((subject, relation))

    def disabled_copy(self) -> OracleTextMemory:
        """The same memory with answers withheld — the leakage-test condition."""
        return OracleTextMemory(mapping=self.mapping, is_enabled=False)

    def __len__(self) -> int:
        return len(self.mapping)

    def __repr__(self) -> str:
        state = "enabled" if self.is_enabled else "DISABLED"
        return f"OracleTextMemory({len(self.mapping):,} facts, {state})"


def splice_results(text: str, memory: TextMemory) -> str:
    """Fill every empty ``<result></result>`` span in ``text`` using ``memory``.

    Used to build teacher-forced training episodes from firewalled text. Generation-time filling
    is different — see ``fka.kernel.generate`` — because there the query spans do not exist yet.
    """
    out: list[str] = []
    pos = 0
    pattern = re.compile(
        re.escape(QUERY_OPEN) + r"(.*?)" + re.escape(QUERY_CLOSE)
        + re.escape(RESULT_OPEN) + r"(.*?)" + re.escape(RESULT_CLOSE),
        re.DOTALL,
    )
    for match in pattern.finditer(text):
        out.append(text[pos : match.start()])
        out.append(f"{QUERY_OPEN}{match.group(1)}{QUERY_CLOSE}")
        out.append(memory.answer_span(match.group(1)))
        pos = match.end()
    out.append(text[pos:])
    return "".join(out)
