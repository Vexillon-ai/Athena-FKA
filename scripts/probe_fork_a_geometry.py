"""Three cheap measurements the joint fit left open. (M2 §9.8, §9.9)

    python scripts/probe_fork_a_geometry.py --checkpoint experiments/<run>/checkpoint_dim64.pt

**1. The capacity-scaled bilinear control (§9.9), which is the registered FIRST diagnostic now that
the joint fit has struggled.** Stage a0's caveat — that its compositions were not capacity-matched
— is currently a judgement, and interpreting any shortfall as *representational* over an unmeasured
capacity confound is the same error class as reading a probe's survival as the system's survival.
Re-runs a0's fit with `comp_dim` raised to hold ~10x the key-parameter budget, everything else
identical.

**2. The harvest confound the joint run surfaced.** Stage a0 harvested every query through the
ORACLE memory, so each hop's query was composed from a *correct* latent. The joint driver harvests
through the LEARNED stack, where a wrong retrieval poisons the next hop's query. Those two numbers
are not comparable, and the joint run's warm start read 38.0% where a0 read 57.8% — which is either
a weaker fit or a stricter measurement, and the difference matters for what the joint fit is
compared against. Both are measured here, on the same keys.

**3. The matched separability floor and ceiling.** The calibration in `fka.eval.separability` was
measured with *perfect* queries. The learned keys are scored with real ones, which ceilings the
index, so the floor must be recomputed under the same queries to be an honest reference:

    floor    = normalize(e ⊙ r)  — the oracle's own multiplicative binding
    ceiling  = [a_e ; b_r]       — concatenative, separable by construction

Without those two lines the reported index is a number with no scale.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fka.data.corpus_gen import CorpusConfig, generate_corpus  # noqa: E402
from fka.data.hardening import entity_split  # noqa: E402
from fka.eval.router_eval import evaluate_router, identity_slot_map  # noqa: E402
from fka.eval.separability import separability  # noqa: E402
from fka.kernel.checkpoint import load_checkpoint  # noqa: E402
from fka.kernel.gpu_lock import gpu_lock  # noqa: E402
from fka.kernel.latent_episodes import d3_tokenizer, pack  # noqa: E402
from fka.kernel.latent_kernel import LatentKernelConfig, LatentReasoningKernel  # noqa: E402
from fka.kernel.latent_memory import LatentCodebook, OracleLatentMemory  # noqa: E402
from fka.kernel.seeding import seed_everything  # noqa: E402
from fka.router.composed_keys import ComposedKeyConfig, ComposedKeyTable, key_spread  # noqa: E402
from fka.router.dense import DenseKeyRouter  # noqa: E402
from fka.router.routed_memory import RoutedLatentMemory  # noqa: E402
from scripts.run_d3_kernel import build_data  # noqa: E402
from scripts.run_fork_a_joint import RELATIONS, fit_router_only, harvest  # noqa: E402


def unique_probe_set(queries, targets, device):
    """One query per distinct target — `evaluate_router` joins on fact id and needs uniqueness."""
    uniq, first = np.unique(targets.cpu().numpy(), return_index=True)
    return queries[torch.from_numpy(first).to(device)], uniq


def score(keys, q_eval, uniq, smap, corpus, tag):
    res = evaluate_router(DenseKeyRouter(keys), q_eval, uniq, smap, corpus, RELATIONS)
    sep = separability(keys, q_eval, uniq, corpus.n_entities, len(RELATIONS))
    spread = key_spread(keys)
    print(f"   {tag:<34} recall@1 {res.recall_at_1:>6.1%}   worst margin "
          f"{res.worst_binding_margin:+.4f}   sep {sep.separability_index:>6.1%} "
          f"(rel {sep.relative_index:>6.1%})   rank {spread['effective_rank']:.1f}")
    return {"recall_at_1": res.recall_at_1, "worst_binding_margin": res.worst_binding_margin,
            "separability": sep.to_dict(), "key_spread": spread}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--comp-dims", type=int, nargs="+", default=[128, 448])
    p.add_argument("--steps", type=int, default=1500, help="a0 used exactly this many")
    p.add_argument("--n-entities", type=int, default=2000)
    p.add_argument("--n-train-harvest", type=int, default=1200)
    p.add_argument("--n-eval", type=int, default=400)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--temp", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--force-gpu-lock", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    lock = gpu_lock(force=args.force_gpu_lock) if device.type == "cuda" else None
    if lock is not None:
        lock.__enter__()
    try:
        return _run(args, device)
    finally:
        if lock is not None:
            lock.__exit__(None, None, None)


def _run(args, device) -> int:
    seed_everything(args.seed)
    blob = load_checkpoint(args.checkpoint)
    meta, mcfg = blob["extra"], blob["model_config"]
    corpus = generate_corpus(
        CorpusConfig(n_entities=args.n_entities, seed=meta["codebook_seed"], n_coworkers=1)
    )
    tok = d3_tokenizer()
    block_size = meta["block_size"]
    split = entity_split(corpus, fraction=0.2, seed=meta["codebook_seed"])
    oracle = OracleLatentMemory(
        corpus, LatentCodebook.build(corpus, dim=meta["latent_dim"], seed=meta["codebook_seed"])
    ).to(device)
    model = LatentReasoningKernel(LatentKernelConfig(**mcfg)).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    amp = torch.bfloat16 if device.type == "cuda" else None

    groups, heldout_eps, _ = build_data(
        corpus, tok, split, [2, 3], oracle.fact_index, block_size, meta["codebook_seed"]
    )
    packed_held = pack(heldout_eps[: args.n_eval], tok, block_size, oracle.fact_index)
    train_3hop = groups[-1]
    smap = identity_slot_map(corpus, RELATIONS)

    print(f"== Fork (a) geometry probe on {device}  [kernel FROZEN]")
    sel = np.arange(min(args.n_train_harvest, len(train_3hop.tokens)))
    q_tr, t_tr, _ = harvest(model, oracle, train_3hop.select(sel), device, amp)
    q_or, t_or, _ = harvest(model, oracle, packed_held, device, amp)
    q_eval_or, uniq_or = unique_probe_set(q_or, t_or, device)
    print(f"   {len(q_tr):,} training queries, {len(uniq_or):,} distinct held-out targets")

    n_facts = corpus.n_entities * len(RELATIONS)
    ents = torch.arange(n_facts, device=device) % corpus.n_entities
    rels = torch.arange(n_facts, device=device) // corpus.n_entities

    # -- 3. matched floor and ceiling, under the SAME queries the learned keys are scored with --
    print("\n-- separability reference points, scored with the same real queries")
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    half = meta["latent_dim"] // 2
    a = F.normalize(torch.randn(corpus.n_entities, half, generator=g), dim=-1).to(device)
    b = F.normalize(torch.randn(len(RELATIONS), half, generator=g), dim=-1).to(device)
    concat = F.normalize(torch.cat([a[ents], b[rels]], dim=-1), dim=-1)
    refs = {
        "floor_hadamard": score(oracle.keys, q_eval_or, uniq_or, smap, corpus,
                                "FLOOR  normalize(e (x) r)"),
        "ceiling_concatenative": score(concat, q_eval_or, uniq_or, smap, corpus,
                                       "CEILING  [a_e ; b_r]"),
    }

    # -- 1 & 2. capacity control, each scored under BOTH harvests -----------------------------
    results = {}
    for comp_dim in args.comp_dims:
        seed_everything(args.seed)
        table = ComposedKeyTable(ComposedKeyConfig(
            n_entities=corpus.n_entities, n_relations=len(RELATIONS),
            key_dim=meta["latent_dim"], comp_dim=comp_dim, mode="bilinear",
        )).to(device)
        scale = comp_dim == args.comp_dims[0]
        print(f"\n-- bilinear comp_dim {comp_dim}: {table.n_params:,} key params"
              f"{'  (a0 replica)' if scale else '  (CAPACITY CONTROL)'}")
        losses = fit_router_only(
            table, ents, rels, q_tr, t_tr, steps=args.steps, lr=args.lr, temp=args.temp,
            log_every=500,
        )
        keys = F.normalize(table(ents, rels), dim=-1).detach()

        # (2) the same keys under the two harvests. Only the QUERIES differ.
        routed = RoutedLatentMemory(oracle, table, RELATIONS).to(device).freeze_keys()
        q_rt, t_rt, h_rt = harvest(model, routed, packed_held, device, amp)
        q_eval_rt, uniq_rt = unique_probe_set(q_rt, t_rt, device)
        n_hops = packed_held.n_hops
        hit = routed.retrieved_index(q_rt) == t_rt
        per_hop = [float(hit[h_rt == h].float().mean()) for h in range(n_hops)]

        entry = {
            "comp_dim": comp_dim,
            "n_params": table.n_params,
            "final_loss": losses[-1],
            "oracle_harvest": score(keys, q_eval_or, uniq_or, smap, corpus,
                                    "ORACLE harvest (a0-comparable)"),
            "routed_harvest": score(keys, q_eval_rt, uniq_rt, smap, corpus,
                                    "LEARNED-stack harvest (deployed)"),
            "per_hop_retrieval_routed": per_hop,
        }
        print(f"   {'per-hop through learned stack':<34} "
              f"{['%.1f%%' % (100 * h) for h in per_hop]}")
        results[str(comp_dim)] = entry

    # -- verdicts, against thresholds fixed in §9.9 -------------------------------------------
    base, scaled = results[str(args.comp_dims[0])], results[str(args.comp_dims[-1])]
    d_recall = scaled["oracle_harvest"]["recall_at_1"] - base["oracle_harvest"]["recall_at_1"]
    d_margin = (scaled["oracle_harvest"]["worst_binding_margin"]
                - base["oracle_harvest"]["worst_binding_margin"])
    print(f"\n== §9.9 capacity control: {base['n_params']:,} -> {scaled['n_params']:,} key params "
          f"({scaled['n_params'] / base['n_params']:.1f}x)")
    print(f"   held-out recall@1  {base['oracle_harvest']['recall_at_1']:.1%} -> "
          f"{scaled['oracle_harvest']['recall_at_1']:.1%}   ({d_recall:+.1%})")
    print(f"   worst margin       {base['oracle_harvest']['worst_binding_margin']:+.4f} -> "
          f"{scaled['oracle_harvest']['worst_binding_margin']:+.4f}   ({d_margin:+.4f})")
    capacity_bound = d_recall > 0.10 and d_margin > 0
    print("   -> " + ("a0's shortfall was PARTLY CAPACITY; the notes must be amended"
                      if capacity_bound else
                      "capacity is NOT the limit; a0's representational reading stands on a "
                      "measurement"))

    payload = {"checkpoint": str(args.checkpoint), "device": str(device),
               "corpus_fingerprint": corpus.fingerprint(), "config": vars(args),
               "references": refs, "fits": results,
               "capacity_bound": bool(capacity_bound),
               "delta_recall": d_recall, "delta_margin": d_margin}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
