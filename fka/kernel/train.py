"""Training the reasoning kernel on memory-interface episodes (design D1, research plan §2.3).

The kernel learns to ask the memory for facts and to compose the retrieved values into an answer.
Retrieved values are excluded from the loss (see :mod:`fka.kernel.episodes`), so nothing in the
objective rewards storing a fact in the weights — but nothing structurally prevents it either,
which is what the leakage test measures.

**D2 (the routing loss) is a flag, not a fork.** ``TrainConfig.routing_loss_weight > 0`` adds an
auxiliary term that explicitly punishes answering correctly when memory is withheld. It is off by
default so that D1's leakage number is measured honestly first; turning it on is the mitigation if
D1 leaks, and having both paths in one file keeps them comparable.

The routing loss is a *hinge*, not a negated cross-entropy::

    routing = relu(target_ce - ce_without_memory)

Negating the no-memory cross-entropy would reward driving it to infinity, which destabilises
training and teaches the model to emit garbage. The hinge only pushes the no-memory answer up to
chance level (``ln(vocab_size)``) and then stops caring — "don't know it without memory" rather
than "be maximally wrong".
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from fka.data.tokenizer import CharTokenizer
from fka.kernel.episodes import Episode, Role, answer_mask, pack_episodes, trainable_mask
from fka.kernel.model import KernelConfig, ReasoningKernel


@dataclass
class TrainConfig:
    size: str = "10M"
    block_size: int = 384
    batch_size: int = 32
    steps: int = 2000
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 0
    amp: bool = True  # bf16 autocast: 3.3x throughput, +0.009 loss gap (2026-08-01)
    eval_every: int = 250
    log_every: int = 50

    #: D2. Off by default — measure D1's leakage honestly before mitigating it.
    routing_loss_weight: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainState:
    step: int = 0
    losses: list[float] = field(default_factory=list)
    routing_losses: list[float] = field(default_factory=list)
    tokens_per_sec: float = 0.0
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "steps": self.step,
            "final_loss": self.losses[-1] if self.losses else None,
            "loss_curve": self.losses[:: max(1, len(self.losses) // 40)],
            "final_routing_loss": self.routing_losses[-1] if self.routing_losses else None,
            "tokens_per_sec": self.tokens_per_sec,
            "seconds": self.seconds,
        }


class EpisodeDataset:
    """Fixed-width token/role batches, with and without memory.

    Both renderings are materialised up front. The no-memory copy is only touched when the routing
    loss is on, but building it here keeps the two views aligned by construction rather than by a
    second code path that could drift.
    """

    def __init__(
        self,
        episodes: Sequence[Episode],
        tokenizer: CharTokenizer,
        block_size: int,
        *,
        build_no_memory: bool = False,
    ) -> None:
        self.tokens, self.roles = pack_episodes(episodes, tokenizer, block_size)
        self.no_memory = (
            pack_episodes(episodes, tokenizer, block_size, with_memory=False)
            if build_no_memory
            else None
        )
        self.n = len(episodes)

    def batch(self, rng: np.random.Generator, batch_size: int, device) -> dict[str, torch.Tensor]:
        idx = rng.integers(0, self.n, size=batch_size)
        out = {
            "tokens": torch.from_numpy(self.tokens[idx]).to(device, non_blocking=True),
            "roles": torch.from_numpy(self.roles[idx].astype(np.int64)).to(device),
        }
        if self.no_memory is not None:
            nm_tokens, nm_roles = self.no_memory
            out["nm_tokens"] = torch.from_numpy(nm_tokens[idx]).to(device, non_blocking=True)
            out["nm_roles"] = torch.from_numpy(nm_roles[idx].astype(np.int64)).to(device)
        return out


def _shift(tokens: torch.Tensor, roles: torch.Tensor):
    """Next-token targets, with roles aligned to the *target* position."""
    return tokens[:, :-1], tokens[:, 1:], roles[:, 1:]


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup then cosine decay to ``min_lr_ratio * lr``."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    progress = min(1.0, progress)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * cosine)


def train_kernel(
    episodes: Sequence[Episode],
    tokenizer: CharTokenizer,
    cfg: TrainConfig,
    *,
    device: torch.device | str = "cpu",
    progress: bool = True,
) -> tuple[ReasoningKernel, TrainState]:
    device = torch.device(device)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    from fka.kernel.model import config_for

    model_cfg: KernelConfig = config_for(
        cfg.size, vocab_size=tokenizer.vocab_size, block_size=cfg.block_size
    )
    model = ReasoningKernel(model_cfg).to(device)

    use_routing = cfg.routing_loss_weight > 0
    data = EpisodeDataset(episodes, tokenizer, cfg.block_size, build_no_memory=use_routing)
    chance_ce = math.log(tokenizer.vocab_size)

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=(0.9, 0.95),
    )
    amp_dtype = torch.bfloat16 if (cfg.amp and device.type == "cuda") else None
    state = TrainState()

    def forward_loss(tokens, roles, mask_fn):
        x, y, target_roles = _shift(tokens, roles)
        mask = torch.from_numpy(mask_fn(target_roles.cpu().numpy())).to(device).float()
        if amp_dtype is None:
            return model(x, y, mask)[1], mask
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            return model(x, y, mask)[1], mask

    t0 = time.perf_counter()
    for step in range(cfg.steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, cfg)

        batch = data.batch(rng, cfg.batch_size, device)
        loss, _ = forward_loss(batch["tokens"], batch["roles"], trainable_mask)

        routing = torch.zeros((), device=device)
        if use_routing:
            # Same episodes with the result spans emptied: how well could the kernel answer
            # with no memory at all? Push that toward chance, no further.
            nm_loss, _ = forward_loss(batch["nm_tokens"], batch["nm_roles"], answer_mask)
            routing = torch.relu(torch.tensor(chance_ce, device=device) - nm_loss)
            state.routing_losses.append(float(routing.detach()))

        total = loss + cfg.routing_loss_weight * routing
        opt.zero_grad(set_to_none=True)
        total.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        state.step = step + 1
        state.losses.append(float(loss.detach()))
        if progress and (step % cfg.log_every == 0 or step == cfg.steps - 1):
            extra = f"  routing {float(routing.detach()):.4f}" if use_routing else ""
            print(f"   step {step + 1:>5}/{cfg.steps}  loss {state.losses[-1]:.4f}{extra}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    state.seconds = time.perf_counter() - t0
    state.tokens_per_sec = cfg.steps * cfg.batch_size * cfg.block_size / state.seconds
    return model, state


def save_run(path: str | Path, cfg: TrainConfig, state: TrainState, extra: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"train_config": cfg.to_dict(), "train": state.to_dict(), **(extra or {})},
                   indent=2),
        encoding="utf-8",
    )


__all__ = [
    "EpisodeDataset",
    "Role",
    "TrainConfig",
    "TrainState",
    "lr_at",
    "save_run",
    "train_kernel",
]
