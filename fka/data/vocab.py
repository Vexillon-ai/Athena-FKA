"""Value spaces for the synthetic knowledge world.

Two layers live here.

**The syllable forge.** Vocabularies (given names, surnames, cities, employers) are built by
positionally decoding an integer index across a stack of syllable tables. That makes the i-th
word of a vocabulary stable forever, makes vocabulary capacity exactly the product of the table
sizes, and lets us scale to 10^6 entities without shipping a word list.

Crucially, vocabularies are a pure function of the **schema** (the requested size) and never of
the corpus seed. Two corpora generated with different seeds must share identical value spaces,
because the information content of a fact is a property of the space it was drawn from — if the
space moved with the seed, entropies would not be comparable across runs.

**Value spaces.** A :class:`ValueSpace` couples a set of values to the distribution facts are
actually drawn from, and answers the only question the capacity instrument cares about:

    bits(value) = -log2 P(value)

the self-information of that value under the *actual* sampling distribution. Uniform spaces give
the familiar ``log2(K)``; Zipf spaces give small numbers for head values and large ones for the
tail, which is exactly what makes the long-tail track interesting. Summing ``bits`` over a corpus
gives its total information content in bits — the denominator of every bits-per-parameter number
in this project.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from math import comb, log2, prod

import numpy as np

# =======================================================================================
# the syllable forge
# =======================================================================================

_GIVEN_ONSET = ("b", "br", "c", "d", "dr", "f", "g", "h", "j", "k", "l", "m", "n", "r", "s", "t")
_GIVEN_VOWEL = ("a", "e", "i", "o", "u", "ia", "ea", "ou")
_GIVEN_CODA = (
    "n", "l", "r", "s", "na", "lo", "ra", "ka", "mi", "no", "vin", "dor", "lyn", "ric", "san", "ta",
)
# Tails must not re-create a coda boundary: with "a" in this table, coda "n" + tail "a" and
# coda "na" + tail "" both render "na", silently collapsing 384 of the 16384 given names into
# duplicates. _vocabulary asserts distinctness so a future edit cannot reintroduce that.
_GIVEN_TAIL = ("", "us", "ine", "el", "or", "iel", "ana", "eth")

_SUR_ONSET = ("B", "Ch", "D", "F", "G", "H", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V")
_SUR_VOWEL = ("a", "e", "i", "o", "u", "ai", "ee", "oo")
_SUR_CODA = (
    "ford", "well", "ton", "sky", "man", "berg", "wick", "stad", "ley", "gard", "holm", "burn",
    "field", "mont", "quist", "ram",
)
_SUR_TAIL = (
    "", "s", "son", "ova", "sen", "ski", "ini", "escu", "ridge", "worth", "by", "dt", "ez", "off",
    "yn", "ard",
)

_CITY_PREFIX = (
    "North", "South", "East", "West", "New", "Old", "Red", "Green", "Grey", "Silver", "Iron",
    "Stone", "Amber", "Frost", "Ash", "Bright", "Clear", "Deep", "Fair", "Glen", "High", "Long",
    "Marsh", "Oak", "Pine", "Rock", "Salt", "Sand", "Star", "Still", "Thorn", "Wind",
)
_CITY_SUFFIX = (
    "vale", "brook", "ford", "port", "haven", "ridge", "field", "wood", "mere", "gate", "cliff",
    "moor", "burgh", "stead", "hollow", "reach", "crest", "shore", "bury", "dale", "fell", "garth",
    "hurst", "keep", "landing", "march", "pass", "quay", "run", "spire", "tarn", "watch",
)

_FIRM_ONSET = (
    "Ax", "Bel", "Cor", "Dyn", "El", "Fer", "Gal", "Hy", "Ir", "Jun", "Kel", "Lum", "Mer", "Nov",
    "Or", "Pri", "Qua", "Rho", "Sol", "Ter", "Umb", "Ver", "Wex", "Xen", "Yar", "Zen", "Ath",
    "Bra", "Cyg", "Del", "Eos", "Fyn",
)
_FIRM_STEM = (
    "tek", "dyne", "corp", "lith", "wave", "core", "flux", "gen", "line", "mark", "path", "sphere",
    "trex", "volt", "works", "zon",
)
_FIRM_SUFFIX = (
    " Systems", " Labs", " Industries", " Holdings", " Dynamics", " Partners", " Analytics",
    " Technologies",
)

_GIVEN_TABLES = (_GIVEN_ONSET, _GIVEN_VOWEL, _GIVEN_CODA, _GIVEN_TAIL)   # capacity 16384
_SUR_TABLES = (_SUR_ONSET, _SUR_VOWEL, _SUR_CODA, _SUR_TAIL)             # capacity 32768
_CITY_TABLES = (_CITY_PREFIX, _CITY_SUFFIX)                              # capacity 1024
_FIRM_TABLES = (_FIRM_ONSET, _FIRM_STEM, _FIRM_SUFFIX)                   # capacity 4096

Tables = tuple[tuple[str, ...], ...]


def _capacity(tables: Tables) -> int:
    return prod(len(t) for t in tables)


def _forge(index: int, tables: Tables) -> str:
    """Decode ``index`` positionally across ``tables`` and concatenate the chosen syllables.

    The first table varies fastest, so consecutive indices differ in their first syllable and
    a slice of the vocabulary looks varied rather than clustered.
    """
    parts = []
    for table in tables:
        parts.append(table[index % len(table)])
        index //= len(table)
    return "".join(parts)


def _vocabulary(n: int, tables: Tables, label: str) -> tuple[str, ...]:
    cap = _capacity(tables)
    if n < 1:
        raise ValueError(f"{label} vocabulary size must be >= 1, got {n}")
    if n > cap:
        raise ValueError(
            f"{label} vocabulary supports at most {cap} distinct values, got {n}. "
            f"Widen the syllable tables in fka/data/vocab.py to go higher."
        )
    words = tuple(_forge(i, tables) for i in range(n))
    # Concatenating variable-length syllables is not injective for arbitrary tables: if one
    # table can end where the next begins, two indices render the same word. That would break
    # name uniqueness (probes become ambiguous) and quietly shrink the value space below the
    # size its bits() claims, so it is checked rather than trusted.
    if len(set(words)) != len(words):
        seen: set[str] = set()
        collision = next(w for w in words if w in seen or seen.add(w))  # type: ignore[func-returns-value]
        raise ValueError(
            f"{label} syllable tables are ambiguous: distinct indices both render {collision!r}. "
            f"Adjust the tables in fka/data/vocab.py so no syllable is a prefix of a "
            f"concatenation of the following ones."
        )
    return words


GIVEN_NAME_CAPACITY = _capacity(_GIVEN_TABLES)
SURNAME_CAPACITY = _capacity(_SUR_TABLES)
CITY_CAPACITY = _capacity(_CITY_TABLES)
EMPLOYER_CAPACITY = _capacity(_FIRM_TABLES)


def given_names(n: int) -> tuple[str, ...]:
    """``n`` distinct given names, e.g. ``Ban``, ``Bral``."""
    return tuple(w.capitalize() for w in _vocabulary(n, _GIVEN_TABLES, "given name"))


def surnames(n: int) -> tuple[str, ...]:
    """``n`` distinct surnames, e.g. ``Baford``, ``Chewell``."""
    return _vocabulary(n, _SUR_TABLES, "surname")


def cities(n: int) -> tuple[str, ...]:
    """``n`` distinct city names, e.g. ``Northvale``, ``Southbrook``."""
    return _vocabulary(n, _CITY_TABLES, "city")


def employers(n: int) -> tuple[str, ...]:
    """``n`` distinct company names, e.g. ``Axtek Systems``."""
    return _vocabulary(n, _FIRM_TABLES, "employer")


def years(start: int, end: int) -> tuple[str, ...]:
    """Years in ``[start, end)`` rendered as strings, so every value space is str-valued."""
    if end <= start:
        raise ValueError(f"empty year range [{start}, {end})")
    return tuple(str(y) for y in range(start, end))


# =======================================================================================
# value spaces
# =======================================================================================


class ValueSpace(ABC):
    """A finite set of values plus the distribution from which facts draw them.

    Subclasses must make ``sample`` and ``bits`` consistent: ``bits(v)`` is ``-log2 P(v)``
    under the distribution ``sample`` actually draws from. The capacity instrument relies on
    that being true — if the two drift apart, every bits-per-parameter number is wrong.
    """

    def __init__(self, values: Sequence[str], name: str) -> None:
        if len(values) == 0:
            raise ValueError(f"{name}: value space cannot be empty")
        self.name = name
        self.values: tuple[str, ...] = tuple(values)
        self._index: dict[str, int] = {v: i for i, v in enumerate(self.values)}
        if len(self._index) != len(self.values):
            raise ValueError(f"{name}: value space contains duplicates")

    def __len__(self) -> int:
        return len(self.values)

    @property
    def size(self) -> int:
        return len(self.values)

    def index_of(self, value: str) -> int:
        return self._index[value]

    # -- sampling ------------------------------------------------------------------------

    def sample(self, rng: np.random.Generator) -> str:
        """Draw a single value."""
        return self.values[int(self.sample_indices(rng, 1)[0])]

    @abstractmethod
    def sample_indices(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw ``n`` value indices. Vectorised path used by the corpus generator."""

    # -- information content -------------------------------------------------------------

    @abstractmethod
    def probabilities(self) -> np.ndarray:
        """Per-value probabilities under the sampling distribution; sums to 1."""

    def bits(self, value: str) -> float:
        """Self-information ``-log2 P(value)`` in bits."""
        return float(self.bits_of_indices(np.array([self.index_of(value)]))[0])

    def bits_of_indices(self, indices: np.ndarray) -> np.ndarray:
        return -np.log2(self.probabilities()[indices])

    @property
    def entropy(self) -> float:
        """Shannon entropy of the space in bits — the expected cost of one fact."""
        p = self.probabilities()
        return float(-(p * np.log2(p)).sum())

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, size={self.size})"


class UniformSpace(ValueSpace):
    """Every value equally likely: ``bits = log2(K)`` for all values."""

    def sample_indices(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.integers(0, self.size, size=n, dtype=np.int64)

    def probabilities(self) -> np.ndarray:
        return np.full(self.size, 1.0 / self.size)

    def bits_of_indices(self, indices: np.ndarray) -> np.ndarray:
        # Closed form, and avoids materialising a length-K probability vector per call.
        return np.full(len(indices), log2(self.size))

    def bits(self, value: str) -> float:
        self.index_of(value)  # validates membership
        return log2(self.size)


class ZipfSpace(ValueSpace):
    """Rank-ordered power law: ``P(rank r) ∝ (r + 1) ** -s``, normalised over the space.

    Values keep their vocabulary order, so ``values[0]`` is the head (cheapest to store, most
    frequently seen) and the tail is expensive. This is the distribution the
    compression-through-structure experiment needs (research plan §4.5): a store that exploits
    shared structure should win more on correlated, head-heavy data than on uniform data.
    """

    def __init__(self, values: Sequence[str], name: str, s: float = 1.0) -> None:
        super().__init__(values, name)
        if s <= 0:
            raise ValueError(f"{name}: Zipf exponent must be > 0, got {s}")
        self.s = float(s)
        weights = (np.arange(1, self.size + 1, dtype=np.float64)) ** (-self.s)
        self._probs = weights / weights.sum()

    def sample_indices(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.choice(self.size, size=n, p=self._probs).astype(np.int64)

    def probabilities(self) -> np.ndarray:
        return self._probs


class UniqueNameSpace(ValueSpace):
    """Full names drawn **without replacement** from the given-name x surname product.

    Entity names are the query key for every probe ("In what year was <name> born?"), so they
    must be unique or the probes are ill-posed.

    On information content: by exchangeability the marginal distribution of the i-th draw is
    uniform over the whole product, so ``bits(value) = log2(capacity)`` is exact as a per-fact
    self-information. Summing it over ``n`` names does slightly overstate their *joint* entropy
    (the true joint is ``sum_i log2(capacity - i)``). The corpus generator asserts
    ``capacity >= 100 * n_entities``, which bounds that overstatement at
    ``log2(100/99) ~ 0.0145`` bits per name — negligible next to the ~6-26 bits carried by the
    attribute facts, and the reason the 100x assertion exists.
    """

    def __init__(self, given: Sequence[str], surname: Sequence[str], name: str = "full_name"):
        self._given = tuple(given)
        self._surnames = tuple(surname)
        if len(set(self._given)) != len(self._given):
            raise ValueError("given-name pool contains duplicates; names would not be unique")
        if len(set(self._surnames)) != len(self._surnames):
            raise ValueError("surname pool contains duplicates; names would not be unique")
        self._capacity = len(self._given) * len(self._surnames)
        # Values are enumerated lazily; materialising 5x10^8 strings is not an option.
        self.name = name
        self.values = ()
        self._index = {}

    def __len__(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        return self._capacity

    def render(self, index: int) -> str:
        given = self._given[index % len(self._given)]
        surname = self._surnames[index // len(self._given)]
        return f"{given} {surname}"

    def index_of(self, value: str) -> int:
        given, surname = value.split(" ", 1)
        return self._given.index(given) + len(self._given) * self._surnames.index(surname)

    def sample_indices(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """``n`` distinct indices, sampled without replacement.

        Rejection-resampling rather than ``rng.choice(capacity, replace=False)``, which would
        allocate an array the size of the whole product (gigabytes at 10^6 entities).
        """
        if n > self._capacity:
            raise ValueError(f"cannot draw {n} distinct names from a space of {self._capacity}")
        out = rng.integers(0, self._capacity, size=n, dtype=np.int64)
        while True:
            unique, first_pos = np.unique(out, return_index=True)
            if len(unique) == n:
                return out
            duplicate_mask = np.ones(n, dtype=bool)
            duplicate_mask[first_pos] = False
            out[duplicate_mask] = rng.integers(
                0, self._capacity, size=int(duplicate_mask.sum()), dtype=np.int64
            )

    def probabilities(self) -> np.ndarray:
        return np.full(self._capacity, 1.0 / self._capacity)

    def bits_of_indices(self, indices: np.ndarray) -> np.ndarray:
        return np.full(len(indices), log2(self._capacity))

    def bits(self, value: str) -> float:
        return log2(self._capacity)

    @property
    def entropy(self) -> float:
        return log2(self._capacity)

    def __repr__(self) -> str:
        return f"UniqueNameSpace(size={self._capacity})"


class CoworkerSetSpace(ValueSpace):
    """The value of ``works_with``: an unordered set of ``k`` distinct *other* entities.

    The space is subject-relative — entity i chooses from the other ``n_entities - 1`` — but
    its cardinality, and hence the information content, is the same for every subject:

        bits = log2( C(n_entities - 1, k) )

    Values are entity indices rather than strings, so this class overrides the string-oriented
    parts of the base interface.
    """

    def __init__(self, n_entities: int, k: int, name: str = "works_with") -> None:
        if n_entities < 2:
            raise ValueError("works_with needs at least 2 entities")
        if not 1 <= k <= n_entities - 1:
            raise ValueError(f"k must be in [1, {n_entities - 1}], got {k}")
        self.name = name
        self.n_entities = n_entities
        self.k = k
        self._bits = log2(comb(n_entities - 1, k))
        self.values = ()
        self._index = {}

    def __len__(self) -> int:
        return self.size

    @property
    def size(self) -> int:
        return comb(self.n_entities - 1, self.k)

    def sample_indices(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Coworker sets for subjects ``0..n-1``; shape ``(n, k)``, never containing the subject.

        Draw from ``[0, n_entities - 1)`` and shift indices at or above the subject up by one,
        which maps the reduced range onto "everyone but me" without rejection.
        """
        subjects = np.arange(n, dtype=np.int64)[:, None]
        out = rng.integers(0, self.n_entities - 1, size=(n, self.k), dtype=np.int64)
        out += out >= subjects
        while True:
            # Reject rows that drew the same coworker twice.
            out.sort(axis=1)
            bad = (np.diff(out, axis=1) == 0).any(axis=1)
            if not bad.any():
                return out
            redraw = rng.integers(
                0, self.n_entities - 1, size=(int(bad.sum()), self.k), dtype=np.int64
            )
            redraw += redraw >= subjects[bad]
            out[bad] = redraw

    def probabilities(self) -> np.ndarray:
        raise NotImplementedError(
            "CoworkerSetSpace is subject-relative and its value set is combinatorial; "
            "use .bits()/.entropy, which are constant across subjects."
        )

    def bits(self, value: object = None) -> float:
        return self._bits

    def bits_of_indices(self, indices: np.ndarray) -> np.ndarray:
        return np.full(len(indices), self._bits)

    @property
    def entropy(self) -> float:
        return self._bits

    def __repr__(self) -> str:
        return f"CoworkerSetSpace(n_entities={self.n_entities}, k={self.k}, bits={self._bits:.2f})"
