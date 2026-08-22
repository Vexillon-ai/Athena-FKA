"""A fixed syllable inventory for the fair surface (M5 §5.32).

**The key-discrimination verdict says the dense baseline binds on discriminable keys, not on bits.**
A character-level tokenizer spreads an entity's identity across ~11 tokens of shared alphabet; the
reference regime spells a name in a handful of subword units from a large fixed vocabulary. This
module supplies the latter.

Three properties, in the order they matter:

1. **Fixed inventory, logarithmic name length.** ``N_SYLLABLES`` units, names of
   ``ceil(log(100N)/log(N_SYLLABLES))`` ~ **3** units at every N this program can reach. The
   vocabulary does **not** grow with N — which is what disqualified name-part atomic tokenisation
   (§5.32): its vocabulary grew as ``sqrt(100N)`` and therefore diluted toward one-token-per-entity,
   the option already rejected for *advantaging* the baseline past the reference regime.

2. **One codepoint per syllable.** Each syllable is assigned a single character in the Unicode
   private-use area, so a syllable is one character *and* one token. That keeps the whole rendering
   path textual — templates, probes, the fast vectorised encoder — with no token-space special
   casing. The human-readable spelling is retained in :data:`SYLLABLE_SPELLING` for logs and
   diagnostics.

3. **It is a SURFACE, not a corpus change.** §5.29.1 expected this to require widening the
   generator's syllable tables. It does not: :meth:`Surface.subjects` already owns how an entity is
   spelled, so the inventory lives here and ``fka/data/vocab.py`` is untouched. The world is pinned
   by construction rather than by luck — ``corpus.fingerprint()`` and ``corpus.total_bits`` cannot
   move, because nothing they hash is involved.

**Scope limit, enforced rather than documented.** Relations whose *values* are entities
(``works_with``) would render the same entity under two different names — its corpus name as a
value, its syllable name as a subject. The surface refuses those relations until a value renderer
exists (registered as the next build item); the intervention (§5.35) is ``birth_year``-only and is
unaffected.
"""

from __future__ import annotations

#: Start of the Unicode private-use area. Syllables occupy one codepoint each from here.
PRIVATE_USE_BASE = 0xE000

_ONSETS = (
    "b", "br", "c", "ch", "cl", "d", "dr", "f", "fl", "g", "gl", "gr", "h", "j", "k", "kr",
    "l", "m", "n", "p", "pl", "pr", "qu", "r", "s", "sh", "sk", "sl", "sn", "sp", "st", "t",
    "th", "tr", "v", "w", "y", "z",
)
_NUCLEI = ("a", "e", "i", "o", "u", "ae", "ea", "ou")
#: Kept deliberately small. 38 x 8 x 2 = 608 syllables, which is the ~600 target: large enough that
#: a name is 2-3 units across this program's whole N range, small enough that the embedding table
#: stays a few percent of any model worth measuring (2.2% at 10M).
_CODAS = ("", "n")


def _build() -> tuple[str, ...]:
    """Deterministic, injective, and sorted so ids are stable across runs."""
    out = {f"{o}{n}{c}" for o in _ONSETS for n in _NUCLEI for c in _CODAS}
    return tuple(sorted(out))


#: Human-readable spelling of each syllable, by id.
SYLLABLE_SPELLING: tuple[str, ...] = _build()

#: The atomic rendering character for each syllable, by id.
SYLLABLE_CHARS: tuple[str, ...] = tuple(
    chr(PRIVATE_USE_BASE + i) for i in range(len(SYLLABLE_SPELLING))
)

N_SYLLABLES: int = len(SYLLABLE_SPELLING)

_CHAR_TO_SPELLING = dict(zip(SYLLABLE_CHARS, SYLLABLE_SPELLING, strict=True))


def units_needed(n_names: int) -> int:
    """Syllables per name to address ``n_names`` distinctly. Logarithmic in N, by construction."""
    units, capacity = 1, N_SYLLABLES
    while capacity < max(1, n_names):
        units += 1
        capacity *= N_SYLLABLES
    return units


def name_for(index: int, units: int) -> str:
    """The ``index``-th name as ``units`` syllable characters. Injective for ``index < N**units``."""
    out = []
    for _ in range(units):
        index, rem = divmod(index, N_SYLLABLES)
        out.append(SYLLABLE_CHARS[rem])
    return "".join(reversed(out))


def readable(text: str) -> str:
    """Render syllable characters back to letters, for logs and error messages only."""
    return "".join(_CHAR_TO_SPELLING.get(ch, ch) for ch in text)


__all__ = [
    "N_SYLLABLES",
    "PRIVATE_USE_BASE",
    "SYLLABLE_CHARS",
    "SYLLABLE_SPELLING",
    "name_for",
    "readable",
    "units_needed",
]
