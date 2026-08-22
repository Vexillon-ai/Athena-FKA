"""R1 scaling study: is product-key lookup sub-linear, and with what slope?

    python scripts/run_router_scaling.py --out experiments/<run>/scaling.json

Research plan §3.5 asks for a log-log latency slope ≈ 0.5. Every point goes through
`fka.eval.timing.benchmark` (warmup + best-of-k + device sync) — standing policy, and load-bearing
here because the headline *is* a slope: a cold-timed matmul on this box read 0.44 TFLOP/s against a
warm 2.11, and that 4.8× is larger than the effect being measured.

Reported alongside the slope: **the fit residual**. A slope near 0.5 through points that do not lie
on a line is not evidence of O(√N), and a coefficient quoted without its residual cannot be
distinguished from one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fka.eval.timing import benchmark  # noqa: E402
from fka.kernel.seeding import seed_everything  # noqa: E402
from fka.router.product_key import ProductKeyConfig, ProductKeyRouter  # noqa: E402


def fit_loglog(xs: list[float], ys: list[float]) -> dict:
    """Least-squares slope in log-log space, with the residual that makes it interpretable."""
    lx, ly = np.log10(np.asarray(xs)), np.log10(np.asarray(ys))
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    resid = ly - pred
    ss_res = float((resid**2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "max_abs_residual_dex": float(np.abs(resid).max()),
        "rms_residual_dex": float(np.sqrt((resid**2).mean())),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exponents", type=float, nargs="+", default=[4, 4.5, 5, 5.5, 6, 6.5, 7])
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--query-dim", type=int, default=64)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--device", default="auto")
    p.add_argument("--baseline-max", type=float, default=6.0,
                   help="highest exponent at which to also time the O(N) brute-force reference")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    seed_everything(args.seed)

    print(f"== R1 product-key scaling on {device}")
    print(f"   batch {args.batch}, topk {args.topk}, query_dim {args.query_dim}, "
          f"best-of-{args.repeats} after warmup")
    print(f"\n   {'slots':>12}  {'axes':>13}  {'params':>11}  {'lookup ms':>10}  "
          f"{'brute ms':>9}  {'speedup':>8}")

    points = []
    for e in args.exponents:
        n = int(round(10**e))
        cfg = ProductKeyConfig(n_slots=n, query_dim=args.query_dim, topk=args.topk)
        router = ProductKeyRouter(cfg).to(device)
        q = torch.randn(args.batch, args.query_dim, device=device)

        with torch.no_grad():
            timing = benchmark(
                lambda r=router, qq=q: r(qq), repeats=args.repeats, device=device
            )

        brute_ms = None
        if e <= args.baseline_max:
            # O(N) reference: materialises every slot score. Memory-bound and the reason the
            # study stops at 10^6 for the baseline even though the router itself goes further.
            with torch.no_grad():
                brute = benchmark(
                    lambda r=router, qq=q: r.exact_topk(qq, k=args.topk),
                    repeats=args.repeats, device=device,
                )
            brute_ms = brute.best * 1e3

        ms = timing.best * 1e3
        speed = f"{brute_ms / ms:7.1f}x" if brute_ms else "        -"
        print(f"   {n:>12,}  {cfg.n_sub1:>6,}x{cfg.n_sub2:<6,}  {router.n_params:>11,}  "
              f"{ms:>10.4f}  {brute_ms if brute_ms else float('nan'):>9.4f}  {speed}")

        points.append({
            "n_slots": n, "axes": [cfg.n_sub1, cfg.n_sub2], "n_params": router.n_params,
            "lookup_ms": ms, "brute_ms": brute_ms, "stable": timing.stable,
            "lookup_ms_median": timing.median * 1e3,
        })
        del router, q
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fit = fit_loglog([p["n_slots"] for p in points], [p["lookup_ms"] for p in points])
    brute_pts = [p for p in points if p["brute_ms"]]
    brute_fit = (
        fit_loglog([p["n_slots"] for p in brute_pts], [p["brute_ms"] for p in brute_pts])
        if len(brute_pts) > 2 else None
    )

    print(f"\n   product-key log-log slope : {fit['slope']:+.3f}  "
          f"(r2 {fit['r2']:.3f}, max residual {fit['max_abs_residual_dex']:.3f} dex)")
    if brute_fit:
        print(f"   brute-force  log-log slope : {brute_fit['slope']:+.3f}  "
              f"(r2 {brute_fit['r2']:.3f})   [expected ~1.0]")
    unstable = [p["n_slots"] for p in points if not p["stable"]]
    if unstable:
        print(f"   !! timing unstable at N = {unstable} — treat those points as indicative only")

    payload = {
        "device": str(device), "batch": args.batch, "topk": args.topk,
        "query_dim": args.query_dim, "repeats": args.repeats,
        "points": points, "fit": fit, "brute_fit": brute_fit,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
