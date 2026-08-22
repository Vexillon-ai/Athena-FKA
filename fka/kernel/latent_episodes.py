"""D3 episodes: a subject latent, query slots, and an answer the kernel must decode.

Layout::

    <subj> Q: In what city was the colleague of the colleague of them born?
    <qvec><qvec><qvec> A: Stonewood<eos>

(shown wrapped; it is one line)

Three things about this are load-bearing:

* **The subject's name never appears as text.** ``<subj>``'s input embedding is a learned
  projection of its frozen entity code. Consequently two questions about different entities are
  *character-identical* — all discriminating information is in the latent. That is what makes the
  leakage probe airtight: there is no string in the input correlated with the answer.
* **One ``<qvec>`` per hop.** The hidden state at each becomes a query vector. Their positions are
  fixed by the template, but nothing about the *content* of the address is supplied.
* **Only the answer is trained** for the language loss. Everything before it is input.

The retrieval targets are carried alongside so training can supervise the query vectors, which is
the fair analogue of D1 having its query span text inside the trained region.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from fka.data.corpus_gen import KnowledgeCorpus
from fka.data.multihop import MultiHopProbe
from fka.data.tokenizer import DEFAULT_SPECIALS, CharTokenizer

SUBJ = "<subj>"
QVEC = "<qvec>"
D3_SPECIALS: tuple[str, ...] = (*DEFAULT_SPECIALS, SUBJ, QVEC)

ANSWER_PREFIX = " A: "

#: Templates carry no subject string — the subject arrives as a latent. "them" is a placeholder
#: that is identical across every entity, on purpose.
CHAIN_TEMPLATES: dict[int, dict[str, str]] = {
    1: {
        "birth_year": "Q: In what year were they born? ",
        "birth_city": "Q: In what city were they born? ",
        "employer": "Q: Who do they work for? ",
    },
    2: {
        "birth_year": "Q: In what year was the colleague of them born? ",
        "birth_city": "Q: In what city was the colleague of them born? ",
        "employer": "Q: Who does the colleague of them work for? ",
    },
    3: {
        "birth_year": "Q: In what year was the colleague of the colleague of them born? ",
        "birth_city": "Q: In what city was the colleague of the colleague of them born? ",
        "employer": "Q: Who does the colleague of the colleague of them work for? ",
    },
}


def d3_tokenizer() -> CharTokenizer:
    return CharTokenizer(specials=D3_SPECIALS)


#: Process-global episode-id source. Ids are join keys and nothing else — never indices, never
#: ordered, never persisted — so process-order dependence is harmless, and a single counter buys
#: the property that actually matters: two independently built sets (seen vs held-out) can never
#: collide, so a mix-up surfaces as a missing key rather than as a silently wrong score.
_EPISODE_IDS = itertools.count(1)


def _next_episode_id() -> int:
    return next(_EPISODE_IDS)


@dataclass(frozen=True)
class LatentEpisode:
    """One D3 example plus everything needed to supervise and diagnose its retrievals."""

    text: str
    subject_id: int
    answer: str
    tail_relation: str
    n_hops: int
    #: ``(entity_id, relation)`` addressed at each hop — the retrieval-supervision target.
    hop_addresses: tuple[tuple[int, str], ...]
    #: Identity, assigned here at generation and carried unchanged to the scoring boundary.
    episode_id: int = field(default_factory=_next_episode_id)

    def encode(self, tokenizer: CharTokenizer) -> tuple[np.ndarray, int, np.ndarray, int]:
        """Return ``(ids, subj_pos, qvec_pos, answer_start)``."""
        ids = np.array(tokenizer.encode(self.text), dtype=np.int64)
        subj_pos = int(np.flatnonzero(ids == tokenizer.stoi[SUBJ])[0])
        qvec_pos = np.flatnonzero(ids == tokenizer.stoi[QVEC]).astype(np.int64)
        prefix = self.text.index(ANSWER_PREFIX) + len(ANSWER_PREFIX)
        answer_start = len(tokenizer.encode(self.text[:prefix]))
        return ids, subj_pos, qvec_pos, answer_start


def episode_from_probe(
    probe: MultiHopProbe, corpus: KnowledgeCorpus, name_to_id: dict[str, int]
) -> LatentEpisode:
    n_hops = probe.n_hops
    template = CHAIN_TEMPLATES[n_hops][probe.tail_relation]
    text = f"{SUBJ}{template}{QVEC * n_hops}{ANSWER_PREFIX}{probe.answer}<eos>"
    return LatentEpisode(
        text=text,
        subject_id=name_to_id[probe.subject],
        answer=probe.answer,
        tail_relation=probe.tail_relation,
        n_hops=n_hops,
        hop_addresses=tuple((name_to_id[h.key[0]], h.key[1]) for h in probe.hops),
    )


def one_hop_episode(
    corpus: KnowledgeCorpus, entity_id: int, relation: str
) -> LatentEpisode:
    value = corpus.value_of(relation, entity_id)
    text = f"{SUBJ}{CHAIN_TEMPLATES[1][relation]}{QVEC}{ANSWER_PREFIX}{value}<eos>"
    return LatentEpisode(
        text=text,
        subject_id=entity_id,
        answer=value,
        tail_relation=relation,
        n_hops=1,
        hop_addresses=((entity_id, relation),),
    )


@dataclass(frozen=True)
class Gold:
    """What an episode should answer, retrieved by id rather than by row."""

    answer: str
    relation: str


@dataclass
class PackedEpisodes:
    """Fixed-width batchable arrays. Episodes are grouped by hop count before packing.

    Grouping matters: the number of retrieval rounds is a property of the batch, not of an
    individual example, because the forward loop runs once per hop for the whole batch.

    **The gold answers travel inside this object, keyed by ``episode_id``.** They used to be
    passed alongside as parallel ``answers``/``relations`` sequences aligned to row order, which
    is the arrangement that produced two separate scoring defects: any regrouping between packing
    and scoring silently re-paired predictions with the wrong gold. There is now no row-order
    correspondence left to get wrong — reorder, subset, or batch these arrays however you like and
    the join still lands on the right answer.
    """

    tokens: np.ndarray  # (N, T)
    subj_pos: np.ndarray  # (N,)
    qvec_pos: np.ndarray  # (N, n_hops)
    answer_mask: np.ndarray  # (N, T) bool, aligned to *targets* (shifted)
    subject_ids: np.ndarray  # (N,)
    hop_fact_index: np.ndarray  # (N, n_hops) index into the memory's key table
    n_hops: int
    episode_id: np.ndarray  # (N,) join key, assigned at episode generation
    gold: dict[int, Gold]  # episode_id -> expected answer; not row-aligned by construction

    def __post_init__(self) -> None:
        n = len(self.tokens)
        if len(self.episode_id) != n:
            raise ValueError(f"episode_id has {len(self.episode_id)} entries for {n} episodes")
        if len(set(self.episode_id.tolist())) != n:
            raise ValueError("episode ids are not unique — a join would silently drop rows")

    def __len__(self) -> int:
        return len(self.tokens)

    def gold_for(self, episode_id: int) -> Gold:
        """The scoring boundary. A missing id is an error, never a zero score."""
        g = self.gold.get(int(episode_id))
        if g is None:
            raise KeyError(
                f"episode {int(episode_id)} has no gold answer — the id was invented or lost "
                f"somewhere between packing and scoring"
            )
        return g

    def batches_by_answer_start(self, batch_size: int):
        """Yield row-index groups that share an answer-start position.

        Episodes of different tail relations have different template lengths, so a naive
        contiguous batch mixes answer-start positions. Slicing such a batch at one position cuts
        most of its episodes at the wrong token — the defect that invalidated three rounds of
        measurement (see experiments/2026-08-02_d3-codec-remeasure/QUARANTINE.md). Grouping first
        keeps the batched fast path while making the slice correct for every member.

        This lives here, on the data, because *two* evaluators needed it and only one got it.
        """
        starts = np.array(
            [int(np.flatnonzero(self.answer_mask[i])[0]) for i in range(len(self.tokens))]
        )
        for start in np.unique(starts):
            rows = np.flatnonzero(starts == start)
            for k in range(0, len(rows), batch_size):
                yield rows[k : k + batch_size]

    def select(self, idx: np.ndarray) -> PackedEpisodes:
        """A reordered or subset view. Use this instead of rebuilding the dataclass by hand:
        a hand-built copy is one forgotten field away from re-introducing a misalignment."""
        idx = np.asarray(idx)
        return PackedEpisodes(
            tokens=self.tokens[idx],
            subj_pos=self.subj_pos[idx],
            qvec_pos=self.qvec_pos[idx],
            answer_mask=self.answer_mask[idx],
            subject_ids=self.subject_ids[idx],
            hop_fact_index=self.hop_fact_index[idx],
            n_hops=self.n_hops,
            episode_id=self.episode_id[idx],
            gold=self.gold,
        )


def pack(
    episodes: Sequence[LatentEpisode],
    tokenizer: CharTokenizer,
    block_size: int,
    fact_index: dict[tuple[int, str], int],
) -> PackedEpisodes:
    if not episodes:
        raise ValueError("no episodes to pack")
    n_hops = episodes[0].n_hops
    if any(e.n_hops != n_hops for e in episodes):
        raise ValueError("pack() requires a uniform hop count; group episodes first")

    n = len(episodes)
    tokens = np.full((n, block_size), tokenizer.pad_id, dtype=np.int64)
    answer_mask = np.zeros((n, block_size - 1), dtype=bool)
    subj_pos = np.zeros(n, dtype=np.int64)
    qvec_pos = np.zeros((n, n_hops), dtype=np.int64)
    subject_ids = np.zeros(n, dtype=np.int64)
    hop_fact_index = np.zeros((n, n_hops), dtype=np.int64)
    episode_id = np.array([ep.episode_id for ep in episodes], dtype=np.int64)
    gold = {ep.episode_id: Gold(ep.answer, ep.tail_relation) for ep in episodes}

    for i, ep in enumerate(episodes):
        ids, sp, qp, answer_start = ep.encode(tokenizer)
        if len(ids) > block_size:
            raise ValueError(f"episode needs {len(ids)} tokens, block_size is {block_size}")
        tokens[i, : len(ids)] = ids
        subj_pos[i] = sp
        qvec_pos[i] = qp
        subject_ids[i] = ep.subject_id
        hop_fact_index[i] = [fact_index[a] for a in ep.hop_addresses]
        # Targets are inputs shifted left by one, so target position t predicts token t+1.
        answer_mask[i, answer_start - 1 : len(ids) - 1] = True

    return PackedEpisodes(
        tokens=tokens,
        subj_pos=subj_pos,
        qvec_pos=qvec_pos,
        answer_mask=answer_mask,
        subject_ids=subject_ids,
        hop_fact_index=hop_fact_index,
        n_hops=n_hops,
        episode_id=episode_id,
        gold=gold,
    )
