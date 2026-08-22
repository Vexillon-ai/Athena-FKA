"""Is `h : query -> entity_code` learnable at all? (§10.2, sizing probe before the real run)

Cheapest possible form of the §10.2 probe: **no kernel, no router, no checkpoint.** Queries are the
oracle's own addresses `normalize(e_code (x) r_code)`, which is what the kernel emits when it is
working (M1 confirms it retrieves at 100% per hop). If `h` cannot invert *those*, it cannot invert
the kernel's, and the expensive run is not worth launching.

Also reports the analytic reference — re-multiplying by the relation code, the VSA unbinding
operation — which needs to know `r` and so is **not deployable**, but bounds what information the
query carries.

Fitted on 80% of entities, scored on the held-out 20%: a generalisation check, not a fit check.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fka.kernel.gpu_lock import gpu_lock  # noqa: E402
from fka.kernel.seeding import seed_everything  # noqa: E402
from fka.router.inversion import (  # noqa: E402
    EntityInverter,
    InverterConfig,
    entity_recovery_by_inversion,
    fit_inverter,
)


class _Fixed(torch.nn.Module):
    """Wraps precomputed predictions so they travel the same evaluator as a learned `h`."""

    def __init__(self, v):
        super().__init__()
        self.v = v

    def forward(self, x):
        return self.v[: len(x)]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-entities", type=int, default=2000)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--n-relations", type=int, default=4)
    p.add_argument("--holdout", type=float, default=0.2)
    p.add_argument("--steps", type=int, nargs="+", default=[1000, 4000])
    p.add_argument("--hidden", type=int, nargs="+", default=[512])
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    lock = gpu_lock() if device.type == "cuda" else None
    if lock is not None:
        lock.__enter__()
    try:
        return _run(args, device)
    finally:
        if lock is not None:
            lock.__exit__(None, None, None)


def _run(args, device) -> int:
    seed_everything(args.seed)
    n_e, n_r, dim = args.n_entities, args.n_relations, args.dim
    g = torch.Generator().manual_seed(args.seed)
    codes = F.normalize(torch.randn(n_e, dim, generator=g), dim=-1).to(device)
    rel = F.normalize(torch.randn(n_r, dim, generator=g), dim=-1).to(device)

    ents = np.repeat(np.arange(n_e), n_r)
    rels = np.tile(np.arange(n_r), n_e)
    queries = F.normalize(codes[ents] * rel[rels], dim=-1)
    n_train = int((1 - args.holdout) * n_e)
    train = ents < n_train

    print(f"== §10.2 inversion feasibility on {device}")
    print(f"   {n_e:,} entities x {n_r} relations, dim {dim}; "
          f"fit on {n_train:,} entities, score on {n_e - n_train:,} held out")

    results = {}

    # Analytic reference: re-multiply by the relation code (VSA unbinding). Needs r, so it is a
    # BOUND on the information available, never a deployable mechanism.
    analytic = F.normalize(queries[~train] * rel[rels[~train]], dim=-1)
    ref = entity_recovery_by_inversion(
        _Fixed(analytic), queries[~train], ents[~train], codes, n_r
    )
    print(f"\n   analytic re-multiply (KNOWS r, not deployable)  @1 {ref.at_1:.1%}  "
          f"@8 {ref.curve[8]:.1%}  @64 {ref.curve[64]:.1%}  m99={ref.m_for(0.99)}")
    results["analytic_reference"] = ref.to_dict()

    for hidden in args.hidden:
        for steps in args.steps:
            t0 = time.perf_counter()
            seed_everything(args.seed)
            inv = EntityInverter(
                InverterConfig(latent_dim=dim, hidden=hidden, n_layers=2)
            ).to(device)
            fit_inverter(inv, queries[train], codes[ents[train]], steps=steps, lr=args.lr,
                         log_every=max(1, steps // 2))
            tr = entity_recovery_by_inversion(inv, queries[train], ents[train], codes, n_r)
            he = entity_recovery_by_inversion(inv, queries[~train], ents[~train], codes, n_r)
            print(f"   learned h  hidden={hidden} steps={steps} ({time.perf_counter() - t0:.0f}s)  "
                  f"train@1 {tr.at_1:.1%}   HELD-OUT@1 {he.at_1:.1%}  @8 {he.curve[8]:.1%}  "
                  f"@64 {he.curve[64]:.1%}  m99={he.m_for(0.99)}")
            results[f"h_hidden{hidden}_steps{steps}"] = {
                "train": tr.to_dict(), "heldout": he.to_dict(),
                "n_params": inv.n_params, "seconds": time.perf_counter() - t0,
            }

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps({"config": vars(args), "device": str(device), "results": results}, indent=2),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
