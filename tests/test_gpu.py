"""GPU tier: tests that need the real device. Excluded from `make smoke` (CI has no GPU).

    pytest -m gpu

Everything here exists because CPU-only coverage structurally cannot see it. Per CLAUDE.md, any
bug that reaches a real run earns a regression test at the tier that could have caught it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from fka.data.corpus_gen import generate_corpus  # noqa: E402
from fka.kernel.latent_kernel import LatentKernelConfig, LatentReasoningKernel  # noqa: E402
from fka.kernel.latent_memory import LatentCodebook, OracleLatentMemory  # noqa: E402
from fka.kernel.model import KernelConfig, ReasoningKernel  # noqa: E402

pytestmark = pytest.mark.gpu


def _device():
    if not torch.cuda.is_available():
        pytest.skip("no GPU visible to torch")
    try:  # availability is not liveness on this box - see CLAUDE.md
        torch.zeros(1, device="cuda")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"GPU visible but unusable ({type(exc).__name__}); running over RDP?")
    return torch.device("cuda")


@pytest.fixture(scope="module")
def corpus():
    return generate_corpus(n_entities=40, seed=0, n_coworkers=1)


def test_latent_kernel_forward_under_bf16_autocast(corpus):
    """REGRESSION (2026-08-01): bf16/fp32 mismatch in the subject-projection index-put.

    Under autocast ``subject_in`` returns bf16 while the embedding stream is fp32, and the
    index-put refuses to mix them. This reached a real sweep run: `make smoke` passed throughout
    and always would have, because it runs on CPU where autocast is never exercised.
    """
    device = _device()
    cfg = LatentKernelConfig(
        vocab_size=40, block_size=32, n_layer=2, n_head=2, n_embd=32, latent_dim=32,
    )
    model = LatentReasoningKernel(cfg).to(device)
    memory = OracleLatentMemory(corpus, LatentCodebook.build(corpus, dim=32, seed=0)).to(device)

    B, T = 2, 16
    idx = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    subj = torch.randn(B, cfg.latent_dim, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, loss, info = model(
            idx, subj, torch.zeros(B, dtype=torch.long, device=device),
            torch.tensor([[4, 9]] * B, device=device), memory,
            targets=idx, loss_mask=torch.ones(B, T, device=device),
        )
    assert logits.shape == (B, T, cfg.vocab_size)
    assert torch.isfinite(loss)
    assert len(info["queries"]) == 2


def test_latent_kernel_backward_under_autocast(corpus):
    """The forward fix must not leave gradients broken across the dtype boundary."""
    device = _device()
    cfg = LatentKernelConfig(
        vocab_size=40, block_size=32, n_layer=2, n_head=2, n_embd=32, latent_dim=32,
    )
    model = LatentReasoningKernel(cfg).to(device)
    memory = OracleLatentMemory(corpus, LatentCodebook.build(corpus, dim=32, seed=0)).to(device)
    idx = torch.randint(0, cfg.vocab_size, (2, 16), device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, loss, _ = model(
            idx, torch.randn(2, cfg.latent_dim, device=device),
            torch.zeros(2, dtype=torch.long, device=device),
            torch.tensor([[4, 9]] * 2, device=device), memory,
            targets=idx, loss_mask=torch.ones(2, 16, device=device),
        )
    loss.backward()
    assert model.subject_in.weight.grad is not None
    assert torch.isfinite(model.subject_in.weight.grad).all()
    assert model.query_out.weight.grad.abs().sum() > 0


def test_d1_kernel_under_autocast():
    """The D1 path shares the init and masking code; check it survives autocast too."""
    device = _device()
    model = ReasoningKernel(
        KernelConfig(vocab_size=40, block_size=32, n_layer=2, n_head=2, n_embd=32)
    ).to(device)
    idx = torch.randint(0, 40, (2, 16), device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, loss = model(idx, idx, torch.ones(2, 16, device=device))
    loss.backward()
    assert torch.isfinite(loss)


def test_sdpa_matches_the_math_backend_on_this_gpu():
    """AOTriton flash attention is experimental on gfx1151; verify it stays numerically sound.

    CLAUDE.md recommends TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 for a 30% speedup. The failure
    mode of a bad attention kernel is a model that trains to a worse loss, not one that crashes,
    so this is checked rather than assumed after every ROCm bump.
    """
    device = _device()
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 64, 16, device=device) for _ in range(3))
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(out, ref, atol=1e-4), (out - ref).abs().max().item()


def test_memory_read_is_stable_in_bf16(corpus):
    """Retrieval runs in fp32 by design; confirm a bf16 query does not change the address."""
    device = _device()
    memory = OracleLatentMemory(corpus, LatentCodebook.build(corpus, dim=64, seed=0)).to(device)
    cb = memory.codebook
    key = cb.bind(cb.entity[3], "birth_city").unsqueeze(0)
    assert torch.equal(
        memory.retrieved_index(key), memory.retrieved_index(key.to(torch.bfloat16).float())
    )
