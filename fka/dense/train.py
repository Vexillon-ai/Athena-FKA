"""Plain-LM training for the dense baseline — its own recipe, no memory machinery (M5 §4.1).

What makes this a *fair* object rather than our kernel with a switch flipped:

* **No loss mask.** Every token is trained, including fact values. ``ReasoningKernel.forward`` is
  called with ``loss_mask=None``, so the firewall does not exist on this path at all.
* **Nothing from the memory interface is imported here.** Not episodes, not the oracle, not the
  router, not the store. ``tests/test_dense_baseline.py`` asserts that at source level.
* **Its own recipe.** Warmup, per-size LR and the plain-LM protocol sizing row (123k/42k/16k
  tok/s), not the D3 row — a D3 step costs ``n_hops+1`` forwards and quoting it here would
  understate the baseline's affordable exposure count by ~3x.

Two conventions are load-bearing in the signature.

**The schedule horizon is separate from the number of steps executed** (``total_steps`` vs
``run_steps``). CLAUDE.md's reproduction-probe rule: a probe that recomputes the cosine schedule
from its own length is at a different learning rate than the run it claims to reproduce, and two
such probes once "survived" a divergence they were never exposed to. A probe here inherits
``total_steps`` from the real run and only shortens ``run_steps``.

**Rolling checkpoints carry the optimizer and the epoch cursor**, because this run is expected to
span sessions. A checkpoint without optimizer state resumes into a different trajectory, which is
the training-time version of "debugging two different models".
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from fka.dense.stream import DenseCorpusStream
from fka.eval.timing import benchmark
from fka.kernel.model import KernelConfig, ReasoningKernel, config_for


@dataclass
class DenseTrainConfig:
    """The dense baseline's own recipe. Defaults are the 10M ladder's; larger sizes retune LR."""

    size: str = "10M"
    block_size: int = 512
    batch_size: int = 32
    lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 0
    amp: bool = True  # bf16 autocast (CLAUDE.md: 3.3x, +0.009 loss gap)
    log_every: int = 100
    checkpoint_every: int = 2000
    health_every: int = 200

    def to_dict(self) -> dict:
        return asdict(self)


#: Per-size learning rates. 10M trains clean at 6e-4 on a plain-LM objective; the larger sizes
#: are stepped down because CLAUDE.md records that *the recipe does not transfer across sizes* —
#: 50M diverged to NaN at 3e-4 and at 1e-4 on the D3 objective. These are starting points to be
#: confirmed by a completed run, never by a spot check.
SIZE_LR: dict[str, float] = {
    "tiny": 1e-3, "1M": 1e-3, "3M": 8e-4, "10M": 6e-4, "50M": 3e-4, "150M": 2e-4,
}

#: Weight decay **as a function of corpus size**, transcribed from the capacity-scaling study whose
#: 2 bits/param figure this baseline exists to reproduce (arXiv 2404.05405, appendix; `bioS(N)`).
#: Keys are the upper bound of each of their bands.
#:
#: **This is a schedule, not a constant, and that is the point.** Decoupled AdamW decay pulls every
#: weight toward zero by `lr * wd` per step regardless of the data, while the gradient pressure
#: defending any *single* fact scales with that fact's share of the token stream — which falls as
#: `1/n_facts`. At N = 500 a fact holds 1/2,000 of the stream; at N = 31,686, 1/126,744, a **63x**
#: smaller share against the same pull. So the decay-vs-gradient equilibrium crosses as load rises,
#: and a decay that is harmless on a small corpus erases facts on a large one.
#:
#: We ran a constant **0.1** — 5x their value at our N — inherited from generic LM defaults, and the
#: resulting stall was very nearly reported as a property of dense models (M5 §5.16).
LITERATURE_WEIGHT_DECAY: tuple[tuple[int, float], ...] = (
    (200_000, 0.02),
    (1_000_000, 0.01),
    (2_000_000, 0.005),
    (10**18, 0.002),
)


def weight_decay_for(n_entities: int) -> float:
    """The reference study's weight decay at this corpus size."""
    for bound, wd in LITERATURE_WEIGHT_DECAY:
        if n_entities <= bound:
            return wd
    return LITERATURE_WEIGHT_DECAY[-1][1]


def recipe_for(size: str, **overrides) -> DenseTrainConfig:
    """Per-size defaults, with any explicit override winning (including ``lr``)."""
    overrides.setdefault("lr", SIZE_LR.get(size, 3e-4))
    return DenseTrainConfig(size=size, **overrides)


@dataclass
class DenseTrainState:
    step: int = 0
    epoch: int = 0
    losses: list[float] = field(default_factory=list)
    seconds: float = 0.0
    tokens_per_sec: float = 0.0
    peak_bytes: int = 0
    health: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "steps": self.step,
            "epochs": self.epoch,
            "final_loss": self.losses[-1] if self.losses else None,
            "loss_curve": self.losses[:: max(1, len(self.losses) // 60)],
            "seconds": self.seconds,
            "tokens_per_sec": self.tokens_per_sec,
            "peak_bytes": self.peak_bytes,
            "health": self.health,
        }


def lr_at(step: int, cfg: DenseTrainConfig, total_steps: int) -> float:
    """Linear warmup then cosine decay over ``total_steps`` — the horizon, not the run length."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * cosine)


class EpochWindows:
    """Random windows over one epoch's flat token stream.

    The stream is continuous rather than one-document-per-row: a plain LM is most token-efficient
    that way, and padding rows to the longest statement would spend a large share of the baseline's
    compute on ``<pad>``. Each epoch draws a fresh byte offset before windowing, so document
    boundaries do not land in the same place twice even when the rendered text repeats.
    """

    def __init__(self, tokens: np.ndarray, block_size: int, batch_size: int, seed: tuple) -> None:
        rng = np.random.default_rng(seed)
        offset = int(rng.integers(0, block_size)) if tokens.size > block_size else 0
        usable = tokens[offset:]
        self.tokens = usable
        self.block_size = block_size
        self.batch_size = batch_size
        self.n_windows = max(0, (usable.size - 1) // block_size)
        self.order = rng.permutation(self.n_windows)
        self.cursor = 0
        self._arange = np.arange(block_size + 1, dtype=np.int64)

    @property
    def n_batches(self) -> int:
        return self.n_windows // self.batch_size

    def has_batch(self) -> bool:
        return self.cursor + self.batch_size <= self.n_windows

    def next_batch(self, device) -> tuple[torch.Tensor, torch.Tensor]:
        starts = self.order[self.cursor : self.cursor + self.batch_size] * self.block_size
        self.cursor += self.batch_size
        chunk = self.tokens[starts[:, None] + self._arange]
        block = torch.from_numpy(chunk.astype(np.int64)).to(device, non_blocking=True)
        return block[:, :-1], block[:, 1:]


def _build(stream: DenseCorpusStream, cfg: DenseTrainConfig, device) -> ReasoningKernel:
    model_cfg: KernelConfig = config_for(
        cfg.size, vocab_size=stream.tokenizer.vocab_size, block_size=cfg.block_size
    )
    return ReasoningKernel(model_cfg).to(device)


def steps_per_epoch(stream: DenseCorpusStream, cfg: DenseTrainConfig) -> int:
    windows = max(0, (stream.tokens_per_epoch - 1) // cfg.block_size)
    return windows // cfg.batch_size


def plan_run(stream: DenseCorpusStream, cfg: DenseTrainConfig) -> dict:
    """Derived run geometry. ``exposures`` is the knob; step count follows from it."""
    per_epoch = steps_per_epoch(stream, cfg)
    total = per_epoch * stream.cfg.exposures
    return {
        "steps_per_epoch": per_epoch,
        "total_steps": total,
        "tokens_per_step": cfg.batch_size * cfg.block_size,
        "tokens_total": total * cfg.batch_size * cfg.block_size,
        "exposures": stream.cfg.exposures,
    }


def save_dense_checkpoint(path: str | Path, model, opt, cfg, state, *, extra=None) -> Path:
    """Weights + optimizer + cursor + RNG. Everything a cross-session resume needs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(model.cfg),
            "optimizer_state": opt.state_dict(),
            "train_config": cfg.to_dict(),
            "step": state.step,
            "epoch": state.epoch,
            "losses": state.losses,
            "rng": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
            "extra": extra or {},
        },
        path,
    )
    path.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "step": state.step,
                "epoch": state.epoch,
                "final_loss": state.losses[-1] if state.losses else None,
                "train_config": cfg.to_dict(),
                "n_params": sum(p.numel() for p in model.parameters()),
                "extra": extra or {},
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return path


class NonFiniteLoss(RuntimeError):
    """Training produced a non-finite loss. Raised loudly — 50M's NaN cost ~1,500 GPU steps."""


def train_dense(
    stream: DenseCorpusStream,
    cfg: DenseTrainConfig,
    *,
    device: torch.device | str = "cpu",
    total_steps: int | None = None,
    run_steps: int | None = None,
    out_dir: str | Path | None = None,
    resume: str | Path | None = None,
    progress: bool = True,
) -> tuple[ReasoningKernel, DenseTrainState]:
    """Train a plain LM on ``stream``.

    ``total_steps`` is the LR schedule's horizon (defaults to the full exposure regime);
    ``run_steps`` is how many steps this invocation executes. They differ for probes and for
    cross-session resumes, and conflating them is the schedule-horizon defect CLAUDE.md records.
    """
    device = torch.device(device)
    plan = plan_run(stream, cfg)
    total_steps = int(total_steps or plan["total_steps"])
    if total_steps < 1:
        raise ValueError(
            f"the regime yields {total_steps} steps: {stream.tokens_per_epoch:,} tokens/epoch is "
            f"under one batch of {cfg.batch_size}x{cfg.block_size}"
        )

    model = _build(stream, cfg, device)
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
    state = DenseTrainState()

    if resume is not None:
        blob = torch.load(Path(resume), map_location=device, weights_only=False)
        model.load_state_dict(blob["model_state"])
        opt.load_state_dict(blob["optimizer_state"])
        state.step, state.epoch = int(blob["step"]), int(blob["epoch"])
        state.losses = list(blob.get("losses", []))
        if progress:
            print(f"   resumed from {resume} at step {state.step:,} (epoch {state.epoch})")

    stop_at = total_steps if run_steps is None else min(total_steps, state.step + int(run_steps))
    amp_dtype = torch.bfloat16 if (cfg.amp and device.type == "cuda") else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    windows: EpochWindows | None = None
    start_step = state.step
    t0 = time.perf_counter()
    while state.step < stop_at:
        if windows is None or not windows.has_batch():
            tokens = stream.epoch_tokens(state.epoch)
            windows = EpochWindows(
                tokens, cfg.block_size, cfg.batch_size, (cfg.seed, state.epoch, 0xE9)
            )
            state.epoch += 1
            if windows.n_batches < 1:
                raise ValueError("epoch stream is shorter than one batch")

        for group in opt.param_groups:
            group["lr"] = lr_at(state.step, cfg, total_steps)

        x, y = windows.next_batch(device)
        if amp_dtype is None:
            _, loss = model(x, y)  # loss_mask=None: plain LM, every token trained
        else:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                _, loss = model(x, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        state.step += 1
        value = float(loss.detach())
        state.losses.append(value)

        if not math.isfinite(value):
            if out_dir:
                save_dense_checkpoint(
                    Path(out_dir) / "nonfinite.pt", model, opt, cfg, state,
                    extra={"reason": "non-finite loss"},
                )
            raise NonFiniteLoss(
                f"loss is {value} at step {state.step:,} (lr {lr_at(state.step, cfg, total_steps):.2e}). "
                f"The recipe does not transfer across sizes; retune before re-launching."
            )

        if state.step % cfg.health_every == 0:
            state.health.append(
                {
                    "step": state.step,
                    "loss": value,
                    "lr": lr_at(state.step, cfg, total_steps),
                    "grad_finite": all(
                        bool(torch.isfinite(p.grad).all())
                        for p in model.parameters()
                        if p.grad is not None
                    ),
                    "elapsed": time.perf_counter() - t0,
                }
            )
        if progress and (state.step % cfg.log_every == 0 or state.step == stop_at):
            done = state.step / total_steps
            print(
                f"   step {state.step:>7,}/{total_steps:,} ({done:5.1%})  "
                f"epoch {state.epoch:>4}  loss {value:.4f}  "
                f"lr {lr_at(state.step, cfg, total_steps):.2e}"
            )
        if out_dir and cfg.checkpoint_every and state.step % cfg.checkpoint_every == 0:
            save_dense_checkpoint(Path(out_dir) / "rolling.pt", model, opt, cfg, state)

    if device.type == "cuda":
        torch.cuda.synchronize()
        state.peak_bytes = int(torch.cuda.max_memory_allocated())
    state.seconds = time.perf_counter() - t0
    executed = state.step - start_step
    state.tokens_per_sec = (
        executed * cfg.batch_size * cfg.block_size / state.seconds if state.seconds > 0 else 0.0
    )
    if out_dir:
        save_dense_checkpoint(Path(out_dir) / "final.pt", model, opt, cfg, state)
    return model, state


def probe_cost(
    stream: DenseCorpusStream,
    cfg: DenseTrainConfig,
    *,
    device: torch.device | str = "cpu",
    repeats: int = 10,
) -> dict:
    """Measured step cost AND peak allocation — both mandatory before a sized run.

    CLAUDE.md, twice: never time a cold call, and *a timing probe is not a memory probe*. The
    2M scale run died in the HIP allocator with a perfectly accurate timing projection in hand,
    because peak allocation grew superlinearly in a term the probe never exercised. So this
    measures the full train step — forward, backward, optimizer — and reports
    ``max_memory_allocated`` across it.
    """
    device = torch.device(device)
    model = _build(stream, cfg, device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    windows = EpochWindows(
        stream.epoch_tokens(0), cfg.block_size, cfg.batch_size, (cfg.seed, 0, 0xE9)
    )
    amp_dtype = torch.bfloat16 if (cfg.amp and device.type == "cuda") else None
    batches = [windows.next_batch(device) for _ in range(min(4, windows.n_batches))]
    if not batches:
        raise ValueError("stream too small to form a batch for the probe")

    counter = {"i": 0}

    def step() -> None:
        x, y = batches[counter["i"] % len(batches)]
        counter["i"] += 1
        if amp_dtype is None:
            _, loss = model(x, y)
        else:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    result = benchmark(step, name=f"dense-step-{cfg.size}", device=device, repeats=repeats)
    peak = int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0

    tokens_per_step = cfg.batch_size * cfg.block_size
    plan = plan_run(stream, cfg)
    return {
        "size": cfg.size,
        "n_params": model.n_params(),
        "seconds_per_step": result.best,
        "seconds_per_step_median": result.median,
        "stable": result.stable,
        "tokens_per_sec": tokens_per_step / result.best,
        "peak_bytes": peak,
        "peak_gb": peak / 1e9,
        "batch_size": cfg.batch_size,
        "block_size": cfg.block_size,
        **plan,
        "projected_seconds": plan["total_steps"] * result.best,
        "projected_hours": plan["total_steps"] * result.best / 3600,
    }


__all__ = [
    "DenseTrainConfig",
    "DenseTrainState",
    "EpochWindows",
    "NonFiniteLoss",
    "SIZE_LR",
    "lr_at",
    "plan_run",
    "probe_cost",
    "recipe_for",
    "save_dense_checkpoint",
    "steps_per_epoch",
    "train_dense",
]
