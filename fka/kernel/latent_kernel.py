"""D3: a kernel that emits continuous query vectors and reads latents by cross-attention.

Differences from D1 that matter:

* The subject enters as a **latent**, not as text: the ``<subj>`` token's input embedding is a
  learned projection of the subject's frozen entity code. So query formation cannot be a per-name
  habit, and an entity-level holdout is a real generalisation test.
* At each ``<qvec>`` position the kernel projects its hidden state to a **query vector**, which
  addresses the frozen memory. Nothing about the address is supplied.
* Retrieved latents enter through **cross-attention** (one read head to start), masked so a
  position can only see latents retrieved strictly before it.
* The answer is produced by the ordinary LM head — a **learned readout** from the retrieved value
  latent to tokens. The answer never appears in the input as text.

Retrieval is iterative, and it has to be
----------------------------------------
Hop *k*'s query depends on hop *k-1*'s latent, so one parallel forward cannot produce both. The
forward runs ``n_hops + 1`` times: each pass fills in one more latent, and the last pass sees them
all. The whole chain stays in the autograd graph, so gradient reaches the hop-1 query through the
hop-2 retrieval — which is precisely the composition path under test.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from fka.kernel.model import MAX_PARAMS, Block, KernelConfig


@dataclass(frozen=True)
class LatentKernelConfig(KernelConfig):
    latent_dim: int = 128
    n_read_heads: int = 1
    cross_attn_every: int = 2  # insert a read layer every N blocks
    n_hops: int = 2


class MemoryCrossAttention(nn.Module):
    """Read retrieved latents into the residual stream. Queries from the stream, KV from memory."""

    def __init__(self, cfg: LatentKernelConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_read_heads
        self.ln = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.q = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.kv = nn.Linear(cfg.latent_dim, 2 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)

    def forward(self, x: torch.Tensor, latents: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``x`` (B,T,C); ``latents`` (B,H,latent_dim); ``mask`` (B,T,H) True where readable."""
        B, T, C = x.shape
        H = latents.shape[1]
        q = self.q(self.ln(x)).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k, v = self.kv(latents).split(C, dim=-1)
        k = k.view(B, H, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, H, self.n_head, C // self.n_head).transpose(1, 2)
        attn_mask = mask.unsqueeze(1)  # (B,1,T,H) broadcast over heads
        # A row with nothing readable would make softmax produce NaN; allow it to attend to
        # nothing by keeping one slot open and zeroing the contribution afterwards.
        empty = ~attn_mask.any(dim=-1, keepdim=True)
        attn_mask = attn_mask | empty
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = y * (~empty.squeeze(1)).to(y.dtype)
        return x + self.proj(y)


class LatentReasoningKernel(nn.Module):
    """Decoder-only kernel with a latent memory interface (design D3)."""

    def __init__(self, cfg: LatentKernelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.reads = nn.ModuleList(
            MemoryCrossAttention(cfg)
            for i in range(cfg.n_layer)
            if (i + 1) % cfg.cross_attn_every == 0
        )
        self._read_at = {
            i: j
            for j, i in enumerate(
                i for i in range(cfg.n_layer) if (i + 1) % cfg.cross_attn_every == 0
            )
        }
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.wte.weight

        # The two ends of the latent interface, both learned.
        self.subject_in = nn.Linear(cfg.latent_dim, cfg.n_embd, bias=False)
        self.query_out = nn.Linear(cfg.n_embd, cfg.latent_dim, bias=False)

        self.apply(ReasoningKernelInit.init)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * cfg.n_layer) ** 0.5)

        if self.n_params() > MAX_PARAMS:
            raise ValueError(f"kernel has {self.n_params():,} params, over the cap {MAX_PARAMS:,}")

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _backbone(
        self,
        idx: torch.Tensor,
        subject_code: torch.Tensor,
        subj_pos: torch.Tensor,
        latents: torch.Tensor,
        read_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        # Replace the <subj> token's embedding with the projected entity code.
        x = x.clone()
        # .to(x.dtype): under autocast the projection returns bf16 while the embedding stream is
        # fp32, and an index-put refuses to mix them. Only reachable on GPU, so a CPU-only smoke
        # test cannot catch it.
        x[torch.arange(B, device=idx.device), subj_pos] = self.subject_in(subject_code).to(x.dtype)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self._read_at:
                x = self.reads[self._read_at[i]](x, latents, read_mask)
        return self.ln_f(x)

    def forward(
        self,
        idx: torch.Tensor,
        subject_code: torch.Tensor,
        subj_pos: torch.Tensor,
        qvec_pos: torch.Tensor,
        memory,
        *,
        targets: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        hard_read: bool = False,
        override_last_latent: torch.Tensor | None = None,
        zero_subject: bool = False,
    ):
        """Run the kernel with ``n_hops`` retrievals interleaved.

        ``qvec_pos`` is (B, n_hops): the positions whose hidden states become query vectors.
        Returns ``(logits, loss, info)`` where ``info`` carries the queries and retrieved indices
        so retrieval failures can be attributed.

        ``override_last_latent`` replaces the final hop's retrieved latent *after* retrieval, so
        the query and the addressing are untouched and only the value the readout sees changes.
        It is the mechanism behind the latent-substitution test: if the answer follows the
        substituted latent, the read channel is genuinely carrying the answer; if it sticks with
        the value the memory would have returned, the kernel is answering from somewhere else.

        ``zero_subject`` blanks the injected subject code — note the confound that the subject
        code legitimately feeds hop-1 query formation, so a drop under this ablation is not by
        itself evidence of a shortcut.
        """
        B, T = idx.shape
        n_hops = qvec_pos.shape[1]
        dim = self.cfg.latent_dim
        device = idx.device

        if zero_subject:
            subject_code = torch.zeros_like(subject_code)
        latents = torch.zeros(B, n_hops, dim, device=device, dtype=self.wte.weight.dtype)
        positions = torch.arange(T, device=device).view(1, T, 1)
        # Position t may read latent k only if k's query position is strictly earlier.
        visible = positions > qvec_pos.view(B, 1, n_hops)

        queries, retrieved = [], []
        for hop in range(n_hops):
            read_mask = visible & (
                torch.arange(n_hops, device=device).view(1, 1, n_hops) < hop
            )
            h = self._backbone(idx, subject_code, subj_pos, latents, read_mask)
            q = self.query_out(h[torch.arange(B, device=device), qvec_pos[:, hop]])
            z = memory.read(q.float(), hard=hard_read).to(latents.dtype)
            latents = latents.clone()
            latents[:, hop] = z
            queries.append(q)
            retrieved.append(z)

        if override_last_latent is not None:
            latents = latents.clone()
            latents[:, -1] = override_last_latent.to(latents.dtype)

        h = self._backbone(idx, subject_code, subj_pos, latents, visible)
        logits = self.head(h)
        info = {"queries": queries, "latents": latents, "retrieved": retrieved}

        if targets is None:
            return logits, None, info

        per_token = F.cross_entropy(
            logits.view(-1, logits.size(-1)).float(), targets.reshape(-1), reduction="none"
        ).view(B, T)
        if loss_mask is not None:
            per_token = per_token * loss_mask
            loss = per_token.sum() / loss_mask.sum().clamp(min=1)
        else:
            loss = per_token.mean()
        return logits, loss, info


class ReasoningKernelInit:
    """Shared init so D1 and D3 cannot drift apart on initialisation."""

    @staticmethod
    def init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
