"""Character-level tokenizer.

Character level is the right default for the synthetic track and is explicitly a placeholder:
it keeps the vocabulary tiny (~70 symbols), removes tokenizer artefacts as a confound in capacity
measurements, and means no fact can be memorised as a single convenient token. A BPE tokenizer
becomes worth revisiting when the real-data (Wikipedia) track starts — see research plan §1.

The one non-character concession is *special tokens*: the memory-interface markers
(``<query>``, ``<result>``, …) are atomic symbols rather than seven separate characters, so the
kernel sees an unambiguous interface boundary instead of having to learn to parse angle brackets.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

from fka.data.templates import MARKERS

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"

#: Reserved symbols, in a fixed order so ids are stable across runs.
DEFAULT_SPECIALS: tuple[str, ...] = (PAD, BOS, EOS, UNK, *MARKERS)

#: Characters the synthetic corpus generator can emit. Fixing this makes the tokenizer a pure
#: function of the schema, exactly like the value spaces in :mod:`fka.data.vocab`.
DEFAULT_CHARS: str = (
    " !',-.0123456789:?"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "#_|"  # '_' appears inside relation names in firewall query spans (e.g. birth_year)
)


class CharTokenizer:
    """Maps text to character ids, treating ``specials`` as atomic tokens.

    Ids are assigned specials-first, then characters in the order given, so
    ``tokenizer.pad_id == 0`` and the id space is stable as long as the inputs are.
    """

    def __init__(
        self,
        chars: Sequence[str] | str = DEFAULT_CHARS,
        specials: Sequence[str] = DEFAULT_SPECIALS,
    ) -> None:
        specials = tuple(specials)
        chars = tuple(chars)
        if len(set(specials)) != len(specials):
            raise ValueError("duplicate special token")
        if len(set(chars)) != len(chars):
            raise ValueError("duplicate character in charset")
        if any(len(c) != 1 for c in chars):
            raise ValueError("charset entries must be single characters")
        overlap = set(specials) & set(chars)
        if overlap:
            raise ValueError(f"specials overlap the charset: {sorted(overlap)}")

        self.specials = specials
        self.chars = chars
        self.itos: tuple[str, ...] = specials + chars
        self.stoi: dict[str, int] = {s: i for i, s in enumerate(self.itos)}
        # Longest-first alternation so </query> never matches as <query> would-be prefix.
        self._special_re = re.compile(
            "|".join(re.escape(s) for s in sorted(specials, key=len, reverse=True))
        )

    # -- construction --------------------------------------------------------------------

    @classmethod
    def from_texts(
        cls, texts: Iterable[str], specials: Sequence[str] = DEFAULT_SPECIALS
    ) -> CharTokenizer:
        """Build a tokenizer over exactly the characters present in ``texts``.

        Special tokens are stripped before the charset is collected, so their angle brackets do
        not leak in as ordinary characters.
        """
        specials = tuple(specials)
        strip = re.compile("|".join(re.escape(s) for s in sorted(specials, key=len, reverse=True)))
        seen: set[str] = set()
        for text in texts:
            seen.update(strip.sub("", text))
        return cls(chars="".join(sorted(seen)), specials=specials)

    # -- properties ----------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK]

    # -- coding --------------------------------------------------------------------------

    def encode(
        self, text: str, *, bos: bool = False, eos: bool = False, strict: bool = True
    ) -> list[int]:
        """Encode ``text``. Unknown characters raise unless ``strict=False`` (then ``<unk>``)."""
        ids: list[int] = [self.bos_id] if bos else []
        pos = 0
        for match in self._special_re.finditer(text):
            ids.extend(self._encode_chars(text[pos : match.start()], strict))
            ids.append(self.stoi[match.group()])
            pos = match.end()
        ids.extend(self._encode_chars(text[pos:], strict))
        if eos:
            ids.append(self.eos_id)
        return ids

    def _encode_chars(self, chunk: str, strict: bool) -> list[int]:
        out = []
        for ch in chunk:
            token = self.stoi.get(ch)
            if token is None:
                if strict:
                    raise KeyError(
                        f"character {ch!r} (U+{ord(ch):04X}) is not in the tokenizer charset; "
                        f"pass strict=False to substitute {UNK}"
                    )
                token = self.unk_id
            out.append(token)
        return out

    def decode(self, ids: Iterable[int], *, skip_special: bool = False) -> str:
        pieces = []
        for i in ids:
            symbol = self.itos[i]
            if skip_special and symbol in self.specials:
                continue
            pieces.append(symbol)
        return "".join(pieces)

    # -- persistence ---------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"specials": list(self.specials), "chars": "".join(self.chars)}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> CharTokenizer:
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(chars=blob["chars"], specials=blob["specials"])

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size}, n_specials={len(self.specials)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CharTokenizer):
            return NotImplemented
        return self.itos == other.itos
