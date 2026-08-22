"""The M3 compression sweep — ONE KNOB AT A TIME, with the §3.1 shape instrument.

    python scripts/run_m3_sweep.py --checkpoint <kernel> --router <router.pt> --query-head <h>

Run only after the shape instrument exists (M3 §3.1 ruling): the mean cannot answer the question the
sweep exists to ask, so a sweep reported as means would be a sweep that cannot conclude.

**One knob at a time.** Every configuration differs from the baseline in exactly one of
``n_stages`` / ``codebook_size`` / ``residual_dim`` / ``residual_bits``. The first slice taken at
the first point moved two knobs between its last two rows, which left its transition unresolved by
construction; this does not repeat that.

**Stored width is a real axis** (ruling 3): residual coefficients are genuinely quantised at write
time, so a narrower width degrades reconstruction and the accounting sees the saving.

Reported at every configuration, per class, always:
  * addressability through the learned stack — never-supervised, per relation;
  * the §3.1 per-fact distributions (reconstruction and addressing) with bimodality;
  * stored bits per fact, and the compression ratio against a lossless fp32 latent.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fka.data.corpus_gen import CorpusConfig, generate_corpus  # noqa: E402
from fka.data.hardening import entity_split  # noqa: E402
from fka.eval.degradation import (  # noqa: E402
    QualityDistribution,
    classify_shape,
    per_fact_quality,
)
from fka.kernel.checkpoint import load_checkpoint  # noqa: E402
from fka.kernel.gpu_lock import gpu_lock  # noqa: E402
from fka.kernel.latent_episodes import d3_tokenizer, pack  # noqa: E402
from fka.kernel.latent_kernel import LatentKernelConfig, LatentReasoningKernel  # noqa: E402
from fka.kernel.latent_memory import LatentCodebook, OracleLatentMemory  # noqa: E402
from fka.kernel.seeding import seed_everything  # noqa: E402
from fka.store.base import IdentityStore  # noqa: E402
from fka.store.s1_factorized import S1Config, S1FactorizedStore  # noqa: E402
from scripts.run_d3_kernel import build_data  # noqa: E402
from scripts.run_fork_c import address_holdout, supervised_addresses  # noqa: E402
from scripts.run_s1_first_point import build_table, measure  # noqa: E402

#: Everything varies from here, one field at a time.
BASELINE = S1Config(n_stages=4, codebook_size=256, residual_dim=8, residual_bits=8)

SWEEP = {
    "n_stages": [1, 2, 4, 8],
    "codebook_size": [16, 64, 256, 1024],
    "residual_dim": [0, 2, 4, 8, 16, 32, 64],
    "residual_bits": [2, 4, 8, 16, 32],
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--router", required=True)
    p.add_argument("--query-head", default=None)
    p.add_argument("--n-entities", type=int, default=2000)
    p.add_argument("--n-eval", type=int, default=400)
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
    t0 = time.perf_counter()
    blob = load_checkpoint(args.checkpoint)
    meta, mcfg = blob["extra"], blob["model_config"]
    corpus = generate_corpus(
        CorpusConfig(n_entities=args.n_entities, seed=meta["codebook_seed"], n_coworkers=1)
    )
    tok = d3_tokenizer()
    split = entity_split(corpus, fraction=0.2, seed=meta["codebook_seed"])
    oracle = OracleLatentMemory(
        corpus, LatentCodebook.build(corpus, dim=meta["latent_dim"], seed=meta["codebook_seed"])
    ).to(device)
    model = LatentReasoningKernel(LatentKernelConfig(**mcfg)).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    if args.query_head:
        head = torch.load(args.query_head, map_location=device, weights_only=False)
        model.query_out.weight.data.copy_(head["query_out.weight"].to(device))
    amp = torch.bfloat16 if device.type == "cuda" else None

    raw, heldout_eps, _ = build_data(
        corpus, tok, split, [2, 3], oracle.fact_index, meta["block_size"], meta["codebook_seed"]
    )
    groups, _ = address_holdout(raw, split.heldout, corpus.n_entities)
    supervised = supervised_addresses(groups)
    packed = pack(heldout_eps[: args.n_eval], tok, meta["block_size"], oracle.fact_index)
    codes = F.normalize(oracle.codebook.entity, dim=-1).to(device)
    n_e, dim = corpus.n_entities, meta["latent_dim"]
    slots = torch.arange(n_e, device=device)
    lossless_bits = dim * 32

    print(f"== M3 compression sweep, one knob at a time, on {device}")
    print(f"   baseline: stages={BASELINE.n_stages} K={BASELINE.codebook_size} "
          f"r={BASELINE.residual_dim} width={BASELINE.residual_bits} "
          f"({BASELINE.bits_per_slot:.0f} bits/fact vs {lossless_bits} lossless)")

    # -- gate first, on every eval path this script drives ---------------------------------
    ident = IdentityStore()
    ident.write(codes)
    gate = measure(model, oracle, build_table(meta, ident, n_e, device, args.router),
                   packed, supervised, corpus, device, amp)
    gq = per_fact_quality(ident, slots, kind="addressing")
    print(f"\n-- gate: IdentityStore  never-sup {gate['never_supervised']:.1%}, "
          f"shape instrument mean {gq.mean:.4f}, intermediate {gq.bimodal_fraction:.1%}")
    if gate["never_supervised"] < 0.999 or gq.mean < 0.999:
        raise SystemExit("GATE FAILED — no sweep number is admissible")
    print("   -> OK")

    results = {}
    for knob, values in SWEEP.items():
        print(f"\n-- knob: {knob}")
        rows, dists = [], []
        for v in values:
            cfg = replace(BASELINE, latent_dim=dim, seed=args.seed, **{knob: v})
            if cfg.residual_dim > dim:
                continue
            store = S1FactorizedStore(cfg)
            store.write(codes)
            tbl = build_table(meta, store, n_e, device, args.router)
            addr = measure(model, oracle, tbl, packed, supervised, corpus, device, amp)
            recon_q = per_fact_quality(store, slots, kind="reconstruction", label=f"{knob}={v}")
            addr_q = per_fact_quality(store, slots, kind="addressing", label=f"{knob}={v}")
            # The channel the shape question actually lives on: per-fact END-TO-END retrieval.
            pf = addr["per_fact_never_supervised"]
            keys = sorted(pf)
            e2e_q = QualityDistribution(
                fact_ids=np.array(keys), label=f"{knob}={v}",
                quality=np.array([pf[k] for k in keys], dtype=float),
                per_class={
                    name: {"n": x["n"], "mean": x["recall"]}
                    for name, x in addr["per_relation_never_supervised"].items()
                },
            )
            dists.append(e2e_q)

            per_rel = addr["per_relation_never_supervised"]
            rows.append({
                "value": v,
                "bits_per_fact": cfg.bits_per_slot,
                "compression_ratio": lossless_bits / max(cfg.bits_per_slot, 1e-9),
                "never_supervised": addr["never_supervised"],
                "per_relation_never_supervised": per_rel,
                "reconstruction": recon_q.to_dict(),
                "addressing_store_internal": addr_q.to_dict(),
                "addressing_end_to_end": e2e_q.to_dict(),
            })
            lows = "  ".join(
                f"{k}:{'-' if x['recall'] is None else format(x['recall'], '.0%')}"
                for k, x in per_rel.items()
            )
            print(f"   {knob}={v:<5} {cfg.bits_per_slot:>7.0f} b/fact "
                  f"({lossless_bits / max(cfg.bits_per_slot, 1e-9):>6.1f}x)  "
                  f"never-sup {addr['never_supervised']:>6.1%}  "
                  f"recon {recon_q.mean:.3f}  store-sep {addr_q.mean:.3f}  "
                  f"E2E clean {e2e_q.clean_fraction:>5.1%} interm {e2e_q.bimodal_fraction:>5.1%} "
                  f"lost {e2e_q.lost_fraction:>5.1%}   {lows}")

        results[knob] = {
            "values": values,
            "rows": rows,
            # Evidence only — `classify_shape` deliberately returns no verdict (M3 §3.1).
            "shape_evidence": classify_shape(dists, [r["bits_per_fact"] for r in rows]),
        }

    payload = {"checkpoint": str(args.checkpoint), "router": str(args.router),
               "device": str(device), "seconds": time.perf_counter() - t0,
               "corpus_fingerprint": corpus.fingerprint(), "config": vars(args),
               "baseline": BASELINE.__dict__, "lossless_bits_per_fact": lossless_bits,
               "gate": gate, "sweep": results}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {args.out}")
    print(f"\n== done in {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
