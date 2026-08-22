"""Character tokenizer: round-tripping, atomic special tokens, and corpus coverage."""

from __future__ import annotations

import pytest

from fka.data.corpus_gen import generate_corpus
from fka.data.templates import QUERY_CLOSE, QUERY_OPEN, RESULT_CLOSE, RESULT_OPEN
from fka.data.tokenizer import DEFAULT_SPECIALS, CharTokenizer


@pytest.fixture(scope="module")
def tok():
    return CharTokenizer()


def test_round_trip(tok):
    text = "Rovina Liwell was born in the year 1931."
    assert tok.decode(tok.encode(text)) == text


def test_ids_are_stable_and_specials_come_first(tok):
    assert tok.pad_id == 0
    assert tok.itos[: len(DEFAULT_SPECIALS)] == DEFAULT_SPECIALS
    assert tok.vocab_size == len(DEFAULT_SPECIALS) + len(tok.chars)


def test_bos_and_eos_are_optional(tok):
    ids = tok.encode("hi", bos=True, eos=True)
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
    assert tok.decode(ids, skip_special=True) == "hi"


def test_markers_encode_as_single_tokens(tok):
    """The kernel should see one interface boundary symbol, not seven characters of markup."""
    text = f"{QUERY_OPEN}birth_year of X{QUERY_CLOSE}{RESULT_OPEN}1931{RESULT_CLOSE}"
    ids = tok.encode(text)
    assert ids[0] == tok.stoi[QUERY_OPEN]
    assert ids[-1] == tok.stoi[RESULT_CLOSE]
    assert tok.decode(ids) == text
    assert len(ids) == len("birth_year of X") + len("1931") + 4


def test_closing_marker_is_not_split_into_the_opening_one(tok):
    """`</query>` must match whole, not as `<` + `query>` or shadowed by `<query>`."""
    ids = tok.encode(f"{QUERY_OPEN}a{QUERY_CLOSE}")
    assert ids == [tok.stoi[QUERY_OPEN], tok.stoi["a"], tok.stoi[QUERY_CLOSE]]


def test_unknown_characters_raise_by_default(tok):
    with pytest.raises(KeyError, match="not in the tokenizer charset"):
        tok.encode("café")


def test_unknown_characters_map_to_unk_when_not_strict(tok):
    ids = tok.encode("café", strict=False)
    assert ids[-1] == tok.unk_id
    assert tok.decode(ids) == "caf<unk>"


def test_from_texts_builds_exactly_the_observed_charset():
    built = CharTokenizer.from_texts(["abc", f"{QUERY_OPEN}ab{QUERY_CLOSE}"])
    assert set(built.chars) == set("abc")
    assert QUERY_OPEN in built.specials
    assert "<" not in built.chars, "special-token brackets must not leak in as characters"


def test_default_charset_covers_everything_the_generator_emits():
    """A generated corpus must be encodable without <unk>, or training silently loses facts."""
    corpus = generate_corpus(n_entities=300, seed=0, include_name_facts=True)
    tok = CharTokenizer()
    for doc in corpus.documents():
        tok.encode(doc)
    for doc in corpus.documents(firewall=True):
        tok.encode(doc)
    for pair in corpus.memory_pairs():
        tok.encode(pair.query)
        tok.encode(pair.answer)


def test_save_and_load_round_trip(tmp_path, tok):
    path = tmp_path / "tokenizer.json"
    tok.save(path)
    assert CharTokenizer.load(path) == tok


def test_rejects_malformed_construction():
    with pytest.raises(ValueError, match="single characters"):
        CharTokenizer(chars=["ab"])
    with pytest.raises(ValueError, match="duplicate character"):
        CharTokenizer(chars="aa")
    with pytest.raises(ValueError, match="overlap"):
        CharTokenizer(chars="a", specials=("a", "<pad>", "<bos>", "<eos>", "<unk>"))
