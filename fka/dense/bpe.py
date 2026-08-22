"""A minimal byte-pair-encoding tokenizer, for the surface arm the reconciliation owes (M5 §5.82).

**The merge corpus is a design variable, not an implementation detail.** Merges learned from our own
rendered corpus would let recurring entity names earn their own merges, whose limit is one token per
name — entity-atomic, which §5.32 disqualified for *deleting* the binding task rather than measuring
it. So merges come from **generic English** (the repository's own prose), and our synthetic names
fragment however they fragment. That is the reference's actual situation: their person names were
not in their tokenizer's training objective either.

Vocabulary size is set by the **embedding-share constraint**, not by taste: <= 10% of parameters in
the embedding table gives **735 tokens at 1M** and **3,193 at 10M** (M5 §5.79).

The tokenizer mirrors `CharTokenizer`'s surface (`stoi`/`itos`/`encode`/`decode`/`vocab_size`/
`eos_id`/`unk_id`) so downstream code can hold either — but note it is **not** a one-char-one-token
tokenizer, so the character-LUT fast path in `DenseCorpusStream` does not apply to it.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

#: Split points that stop merges spanning word boundaries — the standard BPE pre-tokenisation.
_WORD = re.compile(r"\S+|\s+")


def _word_freqs(text: str) -> Counter:
    return Counter(_WORD.findall(text))


def train_bpe(
    text: str,
    vocab_size: int,
    specials: tuple[str, ...] = (),
    required_chars: str = "",
) -> list[tuple[str, str]]:
    """Learn merges from ``text`` until the vocabulary reaches ``vocab_size``.

    Returns the ordered merge list. Base vocabulary is the set of characters in ``text`` **plus
    ``required_chars``**; merges are learned greedily on the most frequent adjacent pair.

    **``required_chars`` is not optional in practice.** The merge corpus is generic English and the
    rendered surface is not: a delimiter or digit absent from the prose would tokenise to ``<unk>``
    and the fact would be unlearnable. Pass every character the surface can emit.
    """
    freqs = _word_freqs(text)
    base = sorted({c for w in freqs for c in w} | set(required_chars))
    budget = vocab_size - len(base) - len(specials)
    if budget <= 0:
        raise ValueError(
            f"vocab_size {vocab_size} leaves no room for merges: {len(base)} base characters "
            f"plus {len(specials)} specials"
        )
    words = {tuple(w): n for w, n in freqs.items()}
    merges: list[tuple[str, str]] = []
    for _ in range(budget):
        pairs: Counter = Counter()
        for sym, n in words.items():
            for a, b in zip(sym, sym[1:]):
                pairs[(a, b)] += n
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merges.append(best)
        joined = best[0] + best[1]
        rebuilt = {}
        for sym, n in words.items():
            out, i = [], 0
            while i < len(sym):
                if i + 1 < len(sym) and (sym[i], sym[i + 1]) == best:
                    out.append(joined)
                    i += 2
                else:
                    out.append(sym[i])
                    i += 1
            rebuilt[tuple(out)] = rebuilt.get(tuple(out), 0) + n
        words = rebuilt
    return merges


@dataclass
class BPETokenizer:
    """A trained BPE, exposing `CharTokenizer`'s interface.

    **Not one-char-one-token**, which is the whole point: `tokens_per_name` becomes a measured
    surface coordinate rather than a constant (M5 §5.112).
    """

    merges: list[tuple[str, str]]
    base_chars: tuple[str, ...]  # MUST cover every character the surface can render
    #: Must cover every id the deployed paths ask for. `probe.py` needs pad/eos, and a
    #: missing one surfaces as an AttributeError mid-run rather than at construction.
    specials: tuple[str, ...] = ("<unk>", "<eos>", "<pad>")
    stoi: dict = field(init=False)
    itos: list = field(init=False)
    _rank: dict = field(init=False, repr=False)

    def __post_init__(self) -> None:
        vocab = list(self.specials) + list(self.base_chars)
        vocab += [a + b for a, b in self.merges]
        # A merge can reproduce an existing symbol only if the corpus was degenerate; guard anyway.
        seen, ordered = set(), []
        for s in vocab:
            if s not in seen:
                seen.add(s)
                ordered.append(s)
        self.itos = ordered
        self.stoi = {s: i for i, s in enumerate(ordered)}
        self._rank = {pair: i for i, pair in enumerate(self.merges)}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def unk_id(self) -> int:
        return self.stoi["<unk>"]

    @property
    def eos_id(self) -> int:
        return self.stoi["<eos>"]

    @property
    def pad_id(self) -> int:
        return self.stoi["<pad>"]

    def _merge_word(self, word: str) -> list[str]:
        sym = list(word)
        while len(sym) > 1:
            best, at = None, None
            for i, pair in enumerate(zip(sym, sym[1:])):
                r = self._rank.get(pair)
                if r is not None and (best is None or r < best):
                    best, at = r, i
            if at is None:
                break
            sym[at : at + 2] = [sym[at] + sym[at + 1]]
        return sym

    def tokens(self, text: str) -> list[str]:
        return [t for w in _WORD.findall(text) for t in self._merge_word(w)]

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(t, self.unk_id) for t in self.tokens(text)]

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids if 0 <= i < len(self.itos))


def fragment_statistics(tokenizer: BPETokenizer, names: list[str]) -> dict:
    """The CERTIFICATION the BPE arm is gated on (M5 §5.82.2) — three statistics, not one.

    ``tokens_per_name`` alone cannot distinguish the two failure modes. A name split into 4 tokens
    drawn from a reused inventory of ~200 fragments **is** the reference property (novel compositions
    over a learned sub-inventory); a name split into 4 tokens that appear nowhere else is
    character-level tokenisation wearing a BPE label.
    """
    per_name = [tokenizer.tokens(n) for n in names]
    counts = [len(t) for t in per_name]
    flat = [t for toks in per_name for t in toks]
    inventory = Counter(flat)
    singletons = sum(1 for _, c in inventory.items() if c == 1)
    chars = sum(len(n) for n in names)
    return {
        "n_names": len(names),
        # CORRECTED 2026-08-04: `fragment_reuse_mean` does NOT separate a learned sub-inventory from
        # character-level tokenisation — names built from few distinct letters score HIGH reuse
        # precisely because their fragments are single characters. The statistic that separates them
        # is fragment LENGTH: at 1.0 chars/token the surface IS character-level whatever else it
        # scores. Caught by its own discrimination test.
        "mean_fragment_chars": chars / max(1, sum(counts)),
        "mean_name_chars": chars / max(1, len(names)),
        "tokens_per_name_mean": sum(counts) / max(1, len(counts)),
        "tokens_per_name_min": min(counts) if counts else 0,
        "tokens_per_name_max": max(counts) if counts else 0,
        "sub_inventory_size": len(inventory),
        "fragment_reuse_mean": len(flat) / max(1, len(inventory)),
        "singleton_fragment_fraction": singletons / max(1, len(inventory)),
        "vocabulary_size": tokenizer.vocab_size,
    }
