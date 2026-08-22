"""Environment validation for the FKA training box.

Four checks, in increasing order of how much they exercise the stack:

1. Device visibility (``torch.cuda.is_available()`` — ROCm exposes the iGPU as ``cuda``).
2. A 4096x4096 fp32 matmul on device, verified against the CPU result.
3. Forward + backward through a 4-layer transformer block using
   ``F.scaled_dot_product_attention`` (the only attention path this project uses — see
   CLAUDE.md: no flash-attn).
4. A 200-step training loop of a ~2M-param GPT on random tokens, reporting tokens/sec.

The tiny GPT here is deliberately *local to this script* rather than imported from
``fka.kernel``: Phase 1 has not started and this file must not pre-empt its design.

Usage
-----
    python scripts/smoke_gpu.py                 # full run on the best available device
    python scripts/smoke_gpu.py --smoke         # small + CPU, finishes in seconds (CI)
    python scripts/smoke_gpu.py --device cpu    # force CPU
    python scripts/smoke_gpu.py --json out.json # also dump machine-readable results
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Repo-root on sys.path so this runs as `python scripts/smoke_gpu.py` without an install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fka.eval.timing import benchmark, synchronize  # noqa: E402

# ---------------------------------------------------------------------------------------
# memory reporting
# ---------------------------------------------------------------------------------------


def gtt_used_bytes() -> int | None:
    """Bytes of GTT (graphics translation table) memory in use, or None if unavailable.

    On Linux/ROCm this is the meaningful number for an APU: ``rocm-smi`` reports 0% VRAM on
    Strix Halo because the iGPU has no dedicated pool, so the real allocation shows up as
    GTT. On Windows the sysfs tree does not exist and we fall back to torch's own counters.
    """
    for path in sorted(glob.glob("/sys/class/drm/card*/device/mem_info_gtt_used")):
        try:
            with open(path) as fh:
                return int(fh.read().strip())
        except (OSError, ValueError):
            continue
    return None


def memory_report(device: torch.device) -> dict[str, float | None]:
    report: dict[str, float | None] = {
        "gtt_used_mib": None,
        "torch_peak_allocated_mib": None,
        "torch_peak_reserved_mib": None,
        "device_free_mib": None,
        "device_total_mib": None,
    }
    gtt = gtt_used_bytes()
    if gtt is not None:
        report["gtt_used_mib"] = gtt / 2**20
    if device.type == "cuda":
        report["torch_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        report["torch_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
        try:
            free, total = torch.cuda.mem_get_info()
            report["device_free_mib"] = free / 2**20
            report["device_total_mib"] = total / 2**20
        except (RuntimeError, AttributeError):
            pass
    return report


#: Device-aware sync lives in fka.eval.timing so benchmarks and checks cannot disagree about it.
sync = synchronize


# ---------------------------------------------------------------------------------------
# a minimal GPT, local to this script
# ---------------------------------------------------------------------------------------


@dataclass
class GPTConfig:
    vocab_size: int = 512
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 6
    n_embd: int = 192


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, n_head, T, head_dim)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.wte.weight  # weight tying
        self.apply(self._init_weights)
        # nanoGPT convention: scale residual-path projections by 1/sqrt(2 * n_layer) so the
        # residual stream variance does not grow with depth.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def n_params(self) -> int:
        # wpe is counted; the tied head shares wte's storage so it is counted once.
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------------------


@dataclass
class Check:
    name: str
    ok: bool
    detail: dict = field(default_factory=dict)
    error: str | None = None


def check_device(device: torch.device) -> Check:
    detail = {
        "torch_version": torch.__version__,
        "hip_version": getattr(torch.version, "hip", None),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "selected_device": str(device),
        "platform": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        detail |= {
            "device_name": props.name,
            "gcn_arch": getattr(props, "gcnArchName", None),
            "total_memory_mib": props.total_memory / 2**20,
            "multi_processor_count": props.multi_processor_count,
        }
    return Check("device", ok=True, detail=detail)


def check_matmul(device: torch.device, n: int, repeats: int = 10) -> Check:
    """Large fp32 matmul on device, verified against CPU.

    Timing is best-of-``repeats`` after a warmup. The first matmul on a fresh ROCm process pays
    rocBLAS initialisation and kernel load, which is easily 10-20x the steady-state cost — timing
    a single cold call understates throughput badly enough to look like a hardware fault.
    """
    # Reduced-precision fp32 paths would make this a precision test rather than a
    # correctness test, so pin the highest precision for the comparison.
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        g = torch.Generator().manual_seed(0)
        a_cpu = torch.randn(n, n, generator=g, dtype=torch.float32)
        b_cpu = torch.randn(n, n, generator=g, dtype=torch.float32)
        ref = a_cpu @ b_cpu

        a, b = a_cpu.to(device), b_cpu.to(device)

        timing = benchmark(
            lambda: a @ b, name=f"matmul_{n}", repeats=repeats, device=device
        )
        elapsed = timing.best
        out_cpu = (a @ b).float().cpu()
        abs_err = (out_cpu - ref).abs()
        rel_fro = (abs_err.norm() / ref.norm()).item()
        ok = math.isfinite(rel_fro) and rel_fro < 1e-5
        return Check(
            "matmul",
            ok=ok,
            detail={
                "n": n,
                "seconds": elapsed,
                "seconds_median": timing.median,
                "repeats": timing.repeats,
                "stable": timing.stable,
                "relative_spread": timing.relative_spread,
                "tflops": (2 * n**3) / elapsed / 1e12,
                "max_abs_err": abs_err.max().item(),
                "rel_frobenius_err": rel_fro,
                "tolerance": 1e-5,
                "note": (
                    "degenerate: device is CPU, so this compares CPU against itself"
                    if device.type == "cpu"
                    else "device result compared against CPU reference"
                ),
            },
        )
    finally:
        torch.set_float32_matmul_precision(prev)


def check_transformer_backward(device: torch.device, cfg: GPTConfig, batch: int) -> Check:
    """Forward + backward through 4 transformer blocks built on SDPA."""
    torch.manual_seed(0)
    blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)]).to(device)
    x = torch.randn(batch, cfg.block_size, cfg.n_embd, device=device, requires_grad=True)

    sync(device)
    t0 = time.perf_counter()
    loss = blocks(x).square().mean()
    loss.backward()
    sync(device)
    elapsed = time.perf_counter() - t0

    grads = [p.grad for p in blocks.parameters() if p.grad is not None]
    all_finite = all(torch.isfinite(g).all().item() for g in grads)
    any_nonzero = any((g != 0).any().item() for g in grads)
    n_without_grad = sum(1 for p in blocks.parameters() if p.grad is None)
    return Check(
        "transformer_fwd_bwd",
        ok=all_finite and any_nonzero and n_without_grad == 0 and math.isfinite(loss.item()),
        detail={
            "n_layer": cfg.n_layer,
            "batch": batch,
            "seq_len": cfg.block_size,
            "loss": loss.item(),
            "seconds": elapsed,
            "grads_finite": all_finite,
            "grads_nonzero": any_nonzero,
            "params_without_grad": n_without_grad,
            "sdpa_backend": "F.scaled_dot_product_attention (default backend selection)",
        },
    )


def check_training_loop(
    device: torch.device,
    cfg: GPTConfig,
    batch: int,
    steps: int,
    amp_dtype: torch.dtype | None = None,
) -> Check:
    """Time a training loop on random tokens and report tokens/sec.

    The data is a small fixed pool of random token sequences cycled repeatedly, so the loss
    *should* fall well below ln(vocab_size) — the model is memorising the pool. That makes the
    loss curve a usable numerical-health signal rather than a flat line: a precision problem shows
    up as a curve that stalls or diverges relative to the fp32 run at the same seed.

    ``amp_dtype`` enables autocast (bf16 here). No GradScaler: bf16 has the same exponent range as
    fp32, so gradient underflow is not the concern it is with fp16.
    """
    torch.manual_seed(0)
    model = TinyGPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    gen = torch.Generator(device="cpu").manual_seed(1234)
    data = torch.randint(
        0, cfg.vocab_size, (batch * 8, cfg.block_size + 1), generator=gen, dtype=torch.long
    ).to(device)

    def get_batch(step: int) -> tuple[torch.Tensor, torch.Tensor]:
        lo = (step * batch) % (data.size(0) - batch + 1)
        chunk = data[lo : lo + batch]
        return chunk[:, :-1], chunk[:, 1:]

    def train_step(step: int) -> torch.Tensor:
        x, y = get_batch(step)
        if amp_dtype is None:
            _, loss = model(x, y)
        else:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss.detach()

    warmup = min(5, max(1, steps // 20))
    for step in range(warmup):  # excluded from timing: kernel autotune + allocator warmup
        train_step(step)
    sync(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    losses = []
    t0 = time.perf_counter()
    for step in range(steps):
        losses.append(train_step(warmup + step))
    sync(device)
    elapsed = time.perf_counter() - t0

    losses = [v.item() for v in losses]
    tokens = steps * batch * cfg.block_size
    tail = losses[-max(1, steps // 10) :]
    # Short label deliberately: it is the key the bf16-vs-fp32 comparison looks up, and
    # str(torch.bfloat16) would give "bfloat16" and silently miss.
    label = {None: "fp32", torch.bfloat16: "bf16", torch.float16: "fp16"}.get(
        amp_dtype, str(amp_dtype).replace("torch.", "")
    )
    return Check(
        f"training_loop_{label}",
        ok=all(math.isfinite(v) for v in losses) and losses[-1] < losses[0],
        detail={
            "dtype": label,
            "n_params": model.n_params(),
            "steps": steps,
            "batch": batch,
            "seq_len": cfg.block_size,
            "seconds": elapsed,
            "ms_per_step": 1000 * elapsed / steps,
            "tokens_per_sec": tokens / elapsed,
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "min_loss": min(losses),
            "mean_loss_last_decile": sum(tail) / len(tail),
            "expected_random_loss": math.log(cfg.vocab_size),
            # Downsampled so the JSON stays readable but the shape is recoverable.
            "loss_curve": losses[:: max(1, steps // 20)],
        },
    )


# ---------------------------------------------------------------------------------------


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--matmul-n", type=int, default=4096)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument(
        "--dtype",
        choices=("fp32", "bf16", "both"),
        default="both",
        help="precision for the training loop; 'both' also reports the bf16-vs-fp32 comparison",
    )
    p.add_argument("--json", type=str, default=None, help="write results to this JSON file")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="tiny CPU-sized configuration for CI; runs in seconds",
    )
    args = p.parse_args(argv)

    if args.smoke:
        args.device = "cpu" if args.device == "auto" else args.device
        args.matmul_n, args.steps, args.batch = 256, 5, 2

    device = resolve_device(args.device)
    cfg = GPTConfig(block_size=64 if args.smoke else 256)

    checks = [check_device(device)]
    print(f"== FKA environment smoke test ==  device={device}")
    for k, v in checks[0].detail.items():
        print(f"   {k:24s} {v}")

    if args.device == "auto" and device.type == "cpu":
        print("\n!! No GPU visible to torch — running on CPU.")

    # bf16 autocast is only meaningful on the GPU here; CPU bf16 on this stack is a slow
    # emulation path and would produce a misleading comparison.
    dtypes: list[torch.dtype | None] = [None]
    if args.dtype in ("bf16", "both") and device.type == "cuda":
        dtypes.append(torch.bfloat16)
    if args.dtype == "bf16" and device.type == "cuda":
        dtypes = [torch.bfloat16]

    stages = [
        ("matmul", lambda: check_matmul(device, args.matmul_n)),
        ("transformer_fwd_bwd", lambda: check_transformer_backward(device, cfg, args.batch)),
    ]
    stages += [
        (
            f"training_loop_{'fp32' if dt is None else 'bf16'}",
            lambda dt=dt: check_training_loop(device, cfg, args.batch, args.steps, dt),
        )
        for dt in dtypes
    ]
    for name, fn in stages:
        print(f"\n-- {name}")
        try:
            check = fn()
        except Exception as exc:  # noqa: BLE001 - we want the failure recorded, not raised
            check = Check(name, ok=False, error=f"{type(exc).__name__}: {exc}")
            print(f"   FAILED: {check.error}")
        else:
            for k, v in check.detail.items():
                print(f"   {k:24s} {v:.6g}" if isinstance(v, float) else f"   {k:24s} {v}")
            print(f"   -> {'PASS' if check.ok else 'FAIL'}")
        checks.append(check)

    runs = {c.detail.get("dtype"): c.detail for c in checks if c.name.startswith("training_loop")}
    comparison: dict[str, float] = {}
    if "fp32" in runs and "bf16" in runs:
        fp32, bf16 = runs["fp32"], runs["bf16"]
        comparison = {
            "bf16_speedup": bf16["tokens_per_sec"] / fp32["tokens_per_sec"],
            "fp32_tokens_per_sec": fp32["tokens_per_sec"],
            "bf16_tokens_per_sec": bf16["tokens_per_sec"],
            "fp32_last_loss": fp32["last_loss"],
            "bf16_last_loss": bf16["last_loss"],
            # Same seed and same data, so a healthy bf16 run should track fp32 closely. A large
            # gap means precision is costing optimisation quality, not just throughput.
            "last_loss_gap": bf16["last_loss"] - fp32["last_loss"],
        }
        print("\n-- bf16 vs fp32")
        print(f"   {'throughput':24s} {comparison['bf16_speedup']:.3f}x")
        print(f"   {'fp32 tokens/sec':24s} {fp32['tokens_per_sec']:,.0f}")
        print(f"   {'bf16 tokens/sec':24s} {bf16['tokens_per_sec']:,.0f}")
        print(
            f"   {'final loss':24s} fp32 {fp32['last_loss']:.4f} vs bf16 "
            f"{bf16['last_loss']:.4f}  (gap {comparison['last_loss_gap']:+.4f})"
        )

    mem = memory_report(device)
    print("\n-- memory")
    for k, v in mem.items():
        print(f"   {k:24s} {'n/a' if v is None else f'{v:.1f}'}")
    if mem["gtt_used_mib"] is None and device.type == "cuda":
        print("   (no /sys/class/drm GTT counters on this platform; torch counters used)")

    passed = all(c.ok for c in checks)
    print(f"\n== {'ALL CHECKS PASSED' if passed else 'FAILURES PRESENT'} ==")

    if args.json:
        payload = {
            "passed": passed,
            "device": str(device),
            "memory": mem,
            "bf16_vs_fp32": comparison,
            "checks": [
                {"name": c.name, "ok": c.ok, "detail": c.detail, "error": c.error} for c in checks
            ],
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {args.json}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
