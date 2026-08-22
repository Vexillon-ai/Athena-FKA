"""Phase 5's dense baseline — a plain language model over the same knowledge world.

This package exists to make the comparison **fair**, which is a stronger requirement than making
it *possible*. M5 §4.1: the dense baseline is not our kernel with memory switched off, it is a
proper plain LM that is given every chance — facts in the token stream, no loss mask, no memory
machinery anywhere in its graph, its own tuned recipe.

Nothing in this package may import :mod:`fka.kernel.memory`, :mod:`fka.kernel.episodes`,
:mod:`fka.kernel.latent_memory` or :mod:`fka.retriever`. That is enforced by
``tests/test_dense_baseline.py`` as a source-level invariant, not by convention.
"""

from fka.dense.probe import DenseRecall, GoldStubRecall, WrongAnswerRecall, dense_capacity
from fka.dense.stream import DenseDataConfig, DenseCorpusStream
from fka.dense.train import DenseTrainConfig, DenseTrainState, train_dense

__all__ = [
    "DenseCorpusStream",
    "DenseDataConfig",
    "DenseRecall",
    "DenseTrainConfig",
    "DenseTrainState",
    "GoldStubRecall",
    "WrongAnswerRecall",
    "dense_capacity",
    "train_dense",
]
