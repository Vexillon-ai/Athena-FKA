"""Training episodes: questions, retrieval spans, and the loss mask that enforces the firewall.

An episode is one question answered through the memory interface::

    Q: In what city was Calo Reburns born? <query>birth_city of Calo Reburns</query>
    <result>Northvale</result> A: Northvale<eos>

**The loss mask is the firewall, and it is the subtle part.** The kernel is trained on every token
*except the contents of ``<result>`` spans*. Those tokens are input, not target: they are what the
memory supplied, and training the kernel to predict them would teach it to store facts in its own
weights — precisely what this architecture exists to avoid. Query spans, by contrast, *are*
trained: emitting the right query is the skill being learned.

**The final answer is trained too, and that is deliberate.** It duplicates the retrieved value, so
in principle the kernel could learn the question→answer mapping directly and bypass memory
entirely. We do not prevent that by construction — we *measure* it. That is exactly what the
leakage test is for (research plan §2.5), and the D2 routing loss is the mitigation if it fails.
Designing the leak out of the data would hide the phenomenon the milestone is meant to detect.

Each episode can be rendered two ways — with memory (result spans filled) and without (result
spans empty). The pair is what the leakage test and the D2 routing loss both need.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from fka.data.corpus_gen import KnowledgeCorpus, MemoryPair
from fka.data.multihop import MultiHopProbe
from fka.data.templates import QUERY_CLOSE, QUERY_OPEN, RESULT_CLOSE, RESULT_OPEN
from fka.data.tokenizer import CharTokenizer
from fka.kernel.memory import format_query

ANSWER_PREFIX = " A: "
QUESTION_PREFIX = "Q: "


class Role(IntEnum):
    """What part of an episode a token belongs to. Drives masking and metrics."""

    PROMPT = 0  # the question and scaffolding
    QUERY = 1  # inside a <query> span — trained: the kernel must learn to ask
    RESULT = 2  # inside a <result> span — NOT trained: this is memory's output
    ANSWER = 3  # the final answer — trained, and the target of the D2 routing loss


@dataclass(frozen=True)
class Episode:
    """One question-through-memory example, as role-tagged text segments."""

    segments: tuple[tuple[str, Role], ...]
    answer: str
    kind: str  # "1hop" | "2hop"
    bits: float
    subject: str
    relation: str

    def render(self, *, with_memory: bool = True) -> str:
        """The episode as text. ``with_memory=False`` empties every result span."""
        return "".join(
            "" if (role is Role.RESULT and not with_memory) else text
            for text, role in self.segments
        )

    def encode(
        self, tokenizer: CharTokenizer, *, with_memory: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(token_ids, roles)`` as parallel int arrays."""
        ids: list[int] = []
        roles: list[int] = []
        for text, role in self.segments:
            if role is Role.RESULT and not with_memory:
                continue
            chunk = tokenizer.encode(text)
            ids.extend(chunk)
            roles.extend([int(role)] * len(chunk))
        return np.array(ids, dtype=np.int64), np.array(roles, dtype=np.int8)


def _spans_for_hops(hops: Iterable[tuple[str, str, str]]) -> list[tuple[str, Role]]:
    """Build query/result segments from ``(subject, relation, value)`` triples."""
    segments: list[tuple[str, Role]] = []
    for subject, relation, value in hops:
        segments.append((f"{QUERY_OPEN}{format_query(subject, relation)}{QUERY_CLOSE}", Role.QUERY))
        segments.append((RESULT_OPEN, Role.PROMPT))
        segments.append((value, Role.RESULT))
        segments.append((RESULT_CLOSE, Role.PROMPT))
    return segments


def episode_from_pair(pair: MemoryPair, eos: str = "<eos>") -> Episode:
    """A 1-hop episode from a corpus probe."""
    subject, relation = pair.key
    segments: list[tuple[str, Role]] = [(f"{QUESTION_PREFIX}{pair.query} ", Role.PROMPT)]
    segments += _spans_for_hops([(subject, relation, pair.answer)])
    segments.append((ANSWER_PREFIX, Role.PROMPT))
    segments.append((pair.answer, Role.ANSWER))
    segments.append((eos, Role.PROMPT))
    return Episode(
        segments=tuple(segments),
        answer=pair.answer,
        kind="1hop",
        bits=pair.bits,
        subject=subject,
        relation=relation,
    )


def episode_from_probe(probe: MultiHopProbe, eos: str = "<eos>") -> Episode:
    """A 2-hop episode. The second query's text depends on the first result — the whole point."""
    segments: list[tuple[str, Role]] = [(f"{QUESTION_PREFIX}{probe.query} ", Role.PROMPT)]
    segments += _spans_for_hops([(h.key[0], h.key[1], h.answer) for h in probe.hops])
    segments.append((ANSWER_PREFIX, Role.PROMPT))
    segments.append((probe.answer, Role.ANSWER))
    segments.append((eos, Role.PROMPT))
    return Episode(
        segments=tuple(segments),
        answer=probe.answer,
        kind="2hop",
        bits=probe.bits,
        subject=probe.subject,
        relation=probe.tail_relation,
    )


def one_hop_episodes(
    corpus: KnowledgeCorpus, fact_ids: Sequence[int] | np.ndarray | None = None
) -> Iterator[Episode]:
    for pair in corpus.memory_pairs(fact_ids if fact_ids is not None else corpus.train_ids):
        yield episode_from_pair(pair)


def pack_episodes(
    episodes: Sequence[Episode],
    tokenizer: CharTokenizer,
    block_size: int,
    *,
    with_memory: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode episodes into fixed-width ``(tokens, roles)`` arrays, right-padded.

    Episodes are kept whole rather than concatenated into a continuous stream. Packing several
    into one context would let the model attend across unrelated questions, which at best wastes
    context and at worst leaks one episode's answer into another's prediction.

    Padding is marked with ``Role.RESULT`` so it is excluded from the loss by the same rule that
    excludes retrieved values — one masking rule, not two.
    """
    pad_id = tokenizer.pad_id
    tokens = np.full((len(episodes), block_size), pad_id, dtype=np.int64)
    roles = np.full((len(episodes), block_size), int(Role.RESULT), dtype=np.int8)
    overflow = 0
    for i, episode in enumerate(episodes):
        ids, role_ids = episode.encode(tokenizer, with_memory=with_memory)
        if len(ids) > block_size:
            overflow += 1
            ids, role_ids = ids[:block_size], role_ids[:block_size]
        tokens[i, : len(ids)] = ids
        roles[i, : len(role_ids)] = role_ids
    if overflow:
        raise ValueError(
            f"{overflow} of {len(episodes)} episodes exceed block_size={block_size} and would be "
            f"truncated mid-answer, silently scoring them wrong. Raise block_size "
            f"(longest episode needs {max_episode_length(episodes, tokenizer)} tokens)."
        )
    return tokens, roles


def max_episode_length(episodes: Sequence[Episode], tokenizer: CharTokenizer) -> int:
    return max(len(e.encode(tokenizer)[0]) for e in episodes) if episodes else 0


def trainable_mask(roles: np.ndarray) -> np.ndarray:
    """True where the token should contribute to the loss: everything but retrieved values."""
    return roles != int(Role.RESULT)


def answer_mask(roles: np.ndarray) -> np.ndarray:
    """True on final-answer tokens — the D2 routing loss and answer accuracy both key off this."""
    return roles == int(Role.ANSWER)
