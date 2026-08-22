"""A seed that does not cover model initialisation is a label, not a seed.

Regression tier for a bug that reached every D3 run: `train_d3` calls `torch.manual_seed` at the
start of *training*, which is after the run script has already built the model. Initialisation
therefore came from an unseeded global RNG.

Nothing caught it because every individual test seeded its own model, and the experiment scripts
were never run twice and compared. This drives the **deployed entry point** twice in one process
and demands the same number out — the only tier that could have seen it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fka.kernel.latent_kernel import LatentKernelConfig, LatentReasoningKernel  # noqa: E402
from fka.kernel.seeding import seed_everything  # noqa: E402


def test_seed_everything_makes_model_init_deterministic():
    """The unit-level property: same seed in, same weights out."""
    cfg = LatentKernelConfig(vocab_size=83, block_size=32, n_layer=2, n_head=2, n_embd=64,
                             latent_dim=32, n_read_heads=1, cross_attn_every=2)
    seed_everything(1234)
    a = LatentReasoningKernel(cfg)
    seed_everything(1234)
    b = LatentReasoningKernel(cfg)
    for (na, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        assert torch.equal(pa, pb), f"{na} differs under an identical seed"


def test_different_seeds_actually_differ():
    """Guards the guard: if seeding were a no-op the test above would pass trivially."""
    cfg = LatentKernelConfig(vocab_size=83, block_size=32, n_layer=2, n_head=2, n_embd=64,
                             latent_dim=32, n_read_heads=1, cross_attn_every=2)
    seed_everything(1)
    a = LatentReasoningKernel(cfg)
    seed_everything(2)
    b = LatentReasoningKernel(cfg)
    assert not all(
        torch.equal(pa, pb)
        for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters(), strict=True)
    )


@pytest.mark.instrument
def test_the_deployed_d3_entry_point_is_reproducible(capsys):
    """The tier that could have caught it: run the real script twice, compare its output.

    A unit test on `seed_everything` cannot see a script that forgets to call it. This can.
    """
    from scripts.run_d3_kernel import main

    def run() -> str:
        main(["--smoke", "--seed", "7"])
        return capsys.readouterr().out

    first, second = run(), run()
    # Loss lines carry the trajectory; comparing them catches init, batch order and dropout alike.
    losses = [
        [ln for ln in out.splitlines() if "loss" in ln and "step" in ln] for out in (first, second)
    ]
    assert losses[0], "no loss lines captured — the smoke path changed shape"
    assert losses[0] == losses[1], (
        "two identically-seeded runs of the deployed entry point diverged:\n"
        f"  {losses[0]}\n  {losses[1]}"
    )
