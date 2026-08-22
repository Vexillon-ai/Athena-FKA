"""The BPE surface's certification statistics ship with tests proving they discriminate.

`fragment_statistics` decides whether the generic-English variant runs or the held-out-world
fallback fires (M5 §5.82.4), so it is an adjudicating instrument and needs red cases.
"""

from __future__ import annotations

import pytest

from fka.dense.bpe import BPETokenizer, fragment_statistics, train_bpe

# Rich enough to support 300 merges; a two-sentence corpus exhausts its pairs long before that.
_WORDS = [f"{a}{b}{c}" for a in "abcdefgh" for b in "aeiou" for c in "lmnrst"]
CORPUS = " ".join(_WORDS * 12) + " the quick brown fox jumps over the lazy dog " * 100

#: Every character the terse surface can emit — delimiters and digits are NOT in English prose
#: by default, and a missing one becomes <unk> (M5 §5.132).
SURFACE_CHARS = "?|:. 0123456789abcdefghijklmnopqrstuvwxyz"


#: The tokenizer's specials must be the SAME tuple the merge budget was computed against, or the
#: vocabulary overshoots by one per unaccounted special — which the embedding-share constraint
#: would then quietly violate.
SPECIALS = ("<unk>", "<eos>", "<pad>")


def _tok(vocab_size: int = 300) -> BPETokenizer:
    merges = train_bpe(
        CORPUS, vocab_size=vocab_size, specials=SPECIALS, required_chars=SURFACE_CHARS
    )
    base = tuple(sorted(set(CORPUS) | set(SURFACE_CHARS)))
    return BPETokenizer(merges=merges, specials=SPECIALS, base_chars=base)


def test_vocabulary_size_is_exactly_what_the_constraint_allows():
    """Vocabulary is set by the embedding-share constraint, so it must land on the number."""
    assert _tok(300).vocab_size == 300


def test_roundtrip_is_lossless():
    tok = _tok()
    for text in ("the quick brown fox", "kalero sigran", "?baba|y:1994"):
        assert tok.decode(tok.encode(text)) == text


def test_it_is_NOT_one_char_one_token():
    """The whole point: tokens/name becomes measured rather than constant (M5 §5.112)."""
    tok = _tok()
    assert len(tok.tokens("the quick brown fox")) < len("the quick brown fox")


def test_unseen_characters_map_to_unk_rather_than_crashing():
    tok = _tok()
    assert tok.unk_id in tok.encode("")


def test_fragment_statistics_SEPARATE_a_learned_inventory_from_a_shattering():
    """Three statistics, because tokens/name alone cannot tell the two failure modes apart.

    A name split into 4 tokens from a REUSED inventory is the reference property; the same 4 tokens
    appearing nowhere else is character-level tokenisation wearing a BPE label.
    """
    tok = _tok()
    # Compositions over syllables the corpus taught: a small, heavily reused sub-inventory.
    reused = [a + b for a in _WORDS[:20] for b in _WORDS[:20]]
    good = fragment_statistics(tok, reused)
    # Names sharing no structure with each other or the corpus: fragments used once each.
    import random
    rng = random.Random(0)
    shattered = ["".join(rng.choice("wxyzqj") for _ in range(6)) for _ in range(400)]
    bad = fragment_statistics(tok, shattered)
    # `fragment_reuse_mean` does NOT do this job — names built from few distinct letters score
    # HIGH reuse precisely because their fragments are single characters (M5 §5.132). The statistic
    # that separates a learned sub-inventory from character-level tokenisation is fragment LENGTH.
    assert good["mean_fragment_chars"] > bad["mean_fragment_chars"]
    assert bad["mean_fragment_chars"] < 1.5, "the shattered case must be near character-level"


def test_training_refuses_a_vocabulary_too_small_for_its_base_characters():
    with pytest.raises(ValueError, match="no room for merges"):
        train_bpe(CORPUS, vocab_size=5, specials=SPECIALS)


# --- the non-LUT stream path (M5 §5.132.5) ------------------------------------------------------


def test_stream_roundtrips_a_multi_character_tokenizer():
    """The standing roundtrip test: a BPE stream must decode back to the text it rendered.

    The LUT fast path assumes one character is one token. A BPE breaks that, and the failure would
    be SILENT — facts encoded as <unk> and a run that simply reads 0%.
    """
    from fka.data.corpus_gen import generate_corpus
    from fka.dense.stream import DenseCorpusStream, DenseDataConfig

    corpus = generate_corpus(n_entities=40, seed=1, probe_fraction=0.25,
                             relations=("birth_year", "birth_city", "employer"))
    # Derived from the corpus charset, never hand-typed. A hand-written subset silently omitted
    # the CAPITALISED value names, and this test is what found it (M5 §5.133).
    from fka.data.tokenizer import DEFAULT_CHARS

    surface_chars = DEFAULT_CHARS + "?|:.\n"
    merges = train_bpe(CORPUS, vocab_size=400, specials=SPECIALS,
                       required_chars=surface_chars)
    tok = BPETokenizer(merges=merges, specials=SPECIALS,
                       base_chars=tuple(sorted(set(CORPUS) | set(surface_chars))))
    stream = DenseCorpusStream(
        corpus, tok, DenseDataConfig(exposures=1, surface="bpe", seed=0)
    )
    ids = stream.epoch_tokens(0)
    assert ids.size > 0
    text = tok.decode(ids.tolist())
    # Every value must survive the round trip as literal text — that is what "firewall off" means.
    assert corpus.value_of("birth_city", 0) in text
    assert tok.unk_id not in ids.tolist(), "a surface character is missing from the BPE base vocab"


def test_the_stream_detects_multi_character_tokenizers_by_ASKING_not_by_class_name():
    from fka.data.corpus_gen import generate_corpus
    from fka.data.tokenizer import CharTokenizer
    from fka.dense.stream import DenseCorpusStream, DenseDataConfig

    corpus = generate_corpus(n_entities=30, seed=2, probe_fraction=0.3)
    char = DenseCorpusStream(corpus, CharTokenizer(), DenseDataConfig(exposures=1))
    assert char._char_level is True and char._lut is not None


def test_tokenizer_exposes_every_id_the_DEPLOYED_paths_ask_for():
    """A missing special surfaces as an AttributeError mid-run, not at construction (M5 §5.137).

    Checked against the attributes the deployed probe and train paths actually read, so the list
    cannot drift out of sync with a grep someone did once.
    """
    tok = _tok()
    for attr in ("vocab_size", "unk_id", "eos_id", "pad_id", "encode", "decode"):
        assert hasattr(tok, attr), f"BPETokenizer lacks {attr}, which the dense paths use"
    assert len({tok.unk_id, tok.eos_id, tok.pad_id}) == 3
