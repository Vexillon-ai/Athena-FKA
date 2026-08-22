"""The reasoning kernel: a small decoder-only transformer (research plan §2).

nanoGPT-class on purpose — full control and a small readable file matter more here than features,
because this model is a scientific instrument and every behaviour has to be attributable.

Two project-specific details:

* **Loss is masked per token.** ``forward`` takes a mask so retrieved values can be excluded from
  the objective (see :mod:`fka.kernel.episodes`). Without that this is just a language model that
  memorises facts, which is the baseline we are trying to beat, not the thing we are building.
* **Size is capped at 300M parameters** (CLAUDE.md guardrail, research plan §9 scope-creep risk).
  The named configs are the sizing policy: iterate at 10M, confirm at 50M, use 150M only for
  decision-record runs.

Attention goes through ``F.scaled_dot_product_attention`` — no flash-attn dependency. On this box
set ``TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`` or it silently falls back to the math backend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Hard guardrail from CLAUDE.md. Exceeding it is scope creep, not ambition.
MAX_PARAMS = 300_000_000


@dataclass(frozen=True)
class KernelConfig:
    vocab_size: int = 80
    block_size: int = 384
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 320
    dropout: float = 0.0
    bias: bool = False

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd={self.n_embd} must be divisible by n_head={self.n_head}")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


#: Sizing policy (CLAUDE.md): iterate at 10M, confirm at 50M, 150M only for decision records.
#: Parameter counts are the transformer body; embeddings add little at char-level vocab.
KERNEL_SIZES: dict[str, dict] = {
    # Not part of the sizing policy: a CI/--smoke config that exercises every code path in
    # seconds on CPU. Never report a result from it.
    "tiny": dict(n_layer=2, n_head=2, n_embd=64),
    # Also not part of the sizing policy: dense-baseline exposure-ladder sizes (M5 §4.2). The
    # exposure knee has to be located on a corpus small enough to sweep 100+ passes over, which
    # means a model small enough for that corpus to saturate it. No kernel result is reported
    # from these.
    "1M": dict(n_layer=4, n_head=4, n_embd=128),
    "3M": dict(n_layer=6, n_head=6, n_embd=192),
    "10M": dict(n_layer=8, n_head=8, n_embd=320),
    "50M": dict(n_layer=12, n_head=12, n_embd=576),
    "150M": dict(n_layer=16, n_head=16, n_embd=896),
}


def config_for(size: str, **overrides) -> KernelConfig:
    if size not in KERNEL_SIZES:
        raise KeyError(f"unknown kernel size {size!r}; known: {sorted(KERNEL_SIZES)}")
    return KernelConfig(**{**KERNEL_SIZES[size], **overrides})


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: KernelConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: KernelConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: KernelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class ReasoningKernel(nn.Module):
    """Decoder-only transformer with per-token loss masking."""

    def __init__(self, cfg: KernelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.wte.weight  # weight tying

        self.apply(self._init_weights)
        # Scale residual-path projections by 1/sqrt(2*n_layer) so the residual stream variance
        # does not grow with depth. Getting this wrong looks like "training is just slow" —
        # it cost us an hour on the smoke test (2026-07-31 notes).
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

        n = self.n_params()
        if n > MAX_PARAMS:
            raise ValueError(
                f"kernel has {n:,} parameters, over the {MAX_PARAMS:,} hard cap in CLAUDE.md "
                f"(research plan §9, scope creep). Shrink the config or change the guardrail "
                f"deliberately."
            )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def n_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wpe.weight.numel()
        return n

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        *,
        reduction: str = "mean",
    ):
        """Run the kernel.

        ``loss_mask`` is aligned with ``targets`` and selects which positions contribute. With
        ``reduction="none"`` the per-token loss is returned unreduced, which the routing loss and
        per-episode metrics both need.
        """
        B, T = idx.shape
        if self.cfg.block_size < T:
            raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if targets is None:
            return logits, None

        per_token = F.cross_entropy(
            logits.view(-1, logits.size(-1)).float(),
            targets.reshape(-1),
            reduction="none",
        ).view(B, T)
        if loss_mask is not None:
            per_token = per_token * loss_mask
        if reduction == "none":
            return logits, per_token
        denom = loss_mask.sum() if loss_mask is not None else per_token.numel()
        # A batch with nothing trainable would otherwise produce a silent NaN.
        loss = per_token.sum() / denom.clamp(min=1) if loss_mask is not None else per_token.mean()
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 0.0,
        stop_ids: set[int] | None = None,
    ) -> torch.Tensor:
        """Greedy by default: these are factual probes, so sampling noise is only a confound."""
        for _ in range(max_new_tokens):
            window = idx[:, -self.cfg.block_size :]
            logits, _ = self(window)
            logits = logits[:, -1, :]
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            else:
                nxt = logits.argmax(dim=-1, keepdim=True)
            idx = torch.cat((idx, nxt), dim=1)
            if stop_ids and idx.size(0) == 1 and int(nxt.item()) in stop_ids:
                break
        return idx
