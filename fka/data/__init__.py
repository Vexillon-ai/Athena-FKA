"""Synthetic knowledge worlds, tokenizer, and real-data loaders.

Re-exports are lazy (PEP 562). Importing the submodules eagerly here would make
``python -m fka.data.corpus_gen`` load ``corpus_gen`` twice and emit a ``runpy`` warning, which
matters because the CLIs are how experiments are launched.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-time convenience for type checkers only
    from fka.data.corpus_gen import (
        CorpusConfig,
        Fact,
        KnowledgeCorpus,
        MemoryPair,
        generate_corpus,
    )
    from fka.data.tokenizer import CharTokenizer
    from fka.data.vocab import (
        CoworkerSetSpace,
        UniformSpace,
        UniqueNameSpace,
        ValueSpace,
        ZipfSpace,
    )

_EXPORTS = {
    "CorpusConfig": "fka.data.corpus_gen",
    "Fact": "fka.data.corpus_gen",
    "KnowledgeCorpus": "fka.data.corpus_gen",
    "MemoryPair": "fka.data.corpus_gen",
    "generate_corpus": "fka.data.corpus_gen",
    "CharTokenizer": "fka.data.tokenizer",
    "CoworkerSetSpace": "fka.data.vocab",
    "UniformSpace": "fka.data.vocab",
    "UniqueNameSpace": "fka.data.vocab",
    "ValueSpace": "fka.data.vocab",
    "ZipfSpace": "fka.data.vocab",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(_EXPORTS[name]), name)


def __dir__() -> list[str]:
    return __all__
