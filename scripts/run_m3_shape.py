"""M3 shape verdict + the relation/hop CONFOUND BREAK, on existing checkpoints. No retraining.

    python scripts/run_m3_shape.py --checkpoint <kernel> --router <router.pt> --query-head <h>

Two things the previous sweep could not do, both blocking:

**1. Break the confound.** The sweep scored by relation, but `works_with` facts were hops 0-1 and
attribute facts were hop 2, so relation type and hop depth co-varied perfectly. All four cells of
the 2x2 already exist in the corpus and needed only to be *scored separately* — no new probe
generation and no retraining:

    works_with @ DIRECT    hop 0 of a 3-hop chain (subject code injected)
    works_with @ COMPOSED  hop 1 of a 3-hop chain (query built from a retrieved latent)
    attribute  @ DIRECT    hop 0 of a 1-hop episode, held-out subject
    attribute  @ COMPOSED  hop 2 of a 3-hop chain

Registered interpretations, fixed before the run: a **hop-depth** effect is a *retriever-side*
finding (composed queries carry noise; Phase 4's cleaning regime owns it). A **relation-type**
effect is a *substrate* finding (what compresses first).

**2. The shape verdict**, from per-fact margin trajectories across a single-knob compression ladder.
Trajectories are the primary readout; distributions are secondary; per class throughout.
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
from fka.eval.margin import MarginSet, MarginTrajectories, retrieval_margins  # noqa: E402
from fka.kernel.checkpoint import load_checkpoint  # noqa: E402
from fka.kernel.gpu_lock import gpu_lock  # noqa: E402
from fka.kernel.latent_episodes import d3_tokenizer, one_hop_episode, pack  # noqa: E402
from fka.kernel.latent_kernel import LatentKernelConfig, LatentReasoningKernel  # noqa: E402
from fka.kernel.latent_memory import LatentCodebook, OracleLatentMemory  # noqa: E402
from fka.kernel.seeding import seed_everything  # noqa: E402
from fka.router.routed_memory import RoutedLatentMemory  # noqa: E402
from fka.store.base import IdentityStore  # noqa: E402
from fka.store.s1_factorized import S1Config, S1FactorizedStore  # noqa: E402
from scripts.run_d3_kernel import build_data  # noqa: E402
from scripts.run_fork_a_joint import RELATIONS, harvest  # noqa: E402
from scripts.run_fork_c import address_holdout, supervised_addresses  # noqa: E402
from scripts.run_s1_first_point import build_table  # noqa: E402

#: One knob (codebook size) at fixed stages and no residual — a clean ladder in bits/fact.
LADDER_BASE = S1Config(n_stages=2, codebook_size=256, residual_dim=0, residual_bits=8)
LADDER = [2, 4, 8, 16, 32, 64, 256, 1024]

CELLS = {(0, "direct"): "works_with@direct", (0, "composed"): "works_with@composed",
         (1, "direct"): "attribute@direct", (1, "composed"): "attribute@composed"}
CELL_ID = {name: i for i, name in enumerate(sorted(CELLS.values()))}


@torch.no_grad()
def collect(model, routed, packed, device, amp, n_e, is_one_hop: bool):
    """Queries, target fact ids, and the (relation-type, hop-kind) cell for each query."""
    queries, targets, hops = harvest(model, routed, packed, device, amp)
    t = targets.cpu().numpy()
    h = hops.cpu().numpy()
    rel = t // n_e
    works_with = RELATIONS.index("works_with")
    # hop 0 is DIRECT: the subject's code is injected. Every later hop is COMPOSED from a latent.
    kind = np.where(h == 0, "direct", "composed")
    rel_type = np.where(rel == works_with, 0, 1)
    cells = np.array(
        [CELL_ID[CELLS[(int(r), str(k))]] for r, k in zip(rel_type, kind, strict=True)]
    )
    if is_one_hop:
        assert (h == 0).all(), "a 1-hop probe set must be entirely direct"
    return queries, t, cells


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--router", required=True)
    p.add_argument("--query-head", default=None)
    p.add_argument("--n-entities", type=int, default=2000)
    p.add_argument("--n-eval", type=int, default=400)
    p.add_argument("--n-one-hop", type=int, default=600)
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


def _run(args, device) -> int:  # noqa: C901 - a driver
    seed_everything(args.seed)
    t0 = time.perf_counter()
    blob = load_checkpoint(args.checkpoint)
    meta, mcfg = blob["extra"], blob["model_config"]
    corpus = generate_corpus(
        CorpusConfig(n_entities=args.n_entities, seed=meta["codebook_seed"], n_coworkers=1)
    )
    tok = d3_tokenizer()
    bs = meta["block_size"]
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
        corpus, tok, split, [2, 3], oracle.fact_index, bs, meta["codebook_seed"]
    )
    groups, _ = address_holdout(raw, split.heldout, corpus.n_entities)
    supervised = supervised_addresses(groups)
    packed3 = pack(heldout_eps[: args.n_eval], tok, bs, oracle.fact_index)

    # attribute @ DIRECT — the missing cell. Held-out subjects, 1-hop episodes, no retraining.
    rng = np.random.default_rng(args.seed)
    attrs = [r for r in RELATIONS if r != "works_with"]
    one_hop = [
        one_hop_episode(corpus, int(e), attrs[int(rng.integers(0, len(attrs)))])
        for e in rng.permutation(split.heldout)[: args.n_one_hop]
    ]
    packed1 = pack(one_hop, tok, bs, oracle.fact_index)

    # works_with @ COMPOSED already exists as hop 1 of the 3-hop chains — no new probes needed.
    n_e, dim = corpus.n_entities, meta["latent_dim"]
    codes = F.normalize(oracle.codebook.entity, dim=-1).to(device)
    names = {v: k for k, v in CELL_ID.items()}

    print(f"== M3 shape + confound break on {device} (no retraining)")

    def run_config(store, label, load):
        tbl = build_table(meta, store, n_e, device, args.router)
        routed = RoutedLatentMemory(oracle, tbl, RELATIONS).to(device).freeze_keys()
        keys = routed.keys
        q3, t3, c3 = collect(model, routed, packed3, device, amp, n_e, False)
        q1, t1, c1 = collect(model, routed, packed1, device, amp, n_e, True)
        q = torch.cat([q3, q1])
        t = np.concatenate([t3, t1])
        c = np.concatenate([c3, c1])
        keep = ~np.isin(t, supervised)  # never-supervised addresses only
        tt = torch.from_numpy(t[keep]).to(device)
        m = retrieval_margins(keys, q[torch.from_numpy(keep).to(device)], tt).cpu().numpy()
        return MarginSet(fact_ids=t[keep], margins=m, load=load, label=label,
                         classes=c[keep], class_names=names)

    # -- gate ------------------------------------------------------------------------------
    ident = IdentityStore()
    ident.write(codes)
    g = run_config(ident, "IdentityStore", float("inf"))
    print(f"\n-- gate: IdentityStore  min margin {g.margins.min():+.4f}, "
          f"retrieved {g.retrieved_fraction:.1%}, inverted {g.inverted_fraction:.1%}")
    for name, v in g.per_class().items():
        print(f"      {name:<22} n={v['n']:<5} mean margin {v['mean_margin']:+.4f}  "
              f"retrieved {v['retrieved_fraction']:.1%}")
    if g.retrieved_fraction < 0.999:
        raise SystemExit("GATE FAILED — lossless store does not retrieve everything")
    print("   -> OK")
    reference = None

    # -- the ladder ------------------------------------------------------------------------
    print(f"\n-- compression ladder: {LADDER_BASE.n_stages} stages, no residual, K varying")
    sets = []
    for k in sorted(LADDER, reverse=True):  # easiest first, so `loads` ascends in difficulty
        cfg = replace(LADDER_BASE, latent_dim=dim, codebook_size=k, seed=args.seed)
        store = S1FactorizedStore(cfg)
        store.write(codes)
        ms = run_config(store, f"K={k}", cfg.bits_per_slot)
        sets.append(ms)
        if reference is None:
            reference = ms.margins.copy()
        pc = "  ".join(
            f"{n.split('@')[0][:4]}@{n.split('@')[1][:4]}:{v['retrieved_fraction']:.0%}"
            for n, v in sorted(ms.per_class().items())
        )
        print(f"   K={k:<5} {cfg.bits_per_slot:>5.0f} b/fact  margin {ms.mean:+.4f}  "
              f"retr {ms.retrieved_fraction:>6.1%}  inv {ms.inverted_fraction:>6.1%}   {pc}")

    # -- the 2x2, at the most degraded rung that still has signal --------------------------
    print("\n== CONFOUND BREAK — retrieved fraction by relation type x hop depth")
    print(f"   {'bits/fact':<10} " + "  ".join(f"{n:<22}" for n in sorted(names.values())))
    for ms in sets:
        row = ms.per_class()
        cells = "  ".join(
            f"{row.get(n, {}).get('retrieved_fraction', float('nan')):<22.1%}"
            for n in sorted(names.values())
        )
        print(f"   {ms.load:<10.0f} {cells}")

    # -- shape verdict from trajectories ----------------------------------------------------
    traj = MarginTrajectories.from_sets(sets)
    summ = traj.summary()
    print("\n== SHAPE — per-fact margin trajectories (primary readout)")
    print(f"   {summ['n_facts']} facts across {summ['n_loads']} loads; "
          f"uniform-slide reference = {summ['uniform_slide_reference']:.3f}")
    print(f"   collapse sharpness: median {summ['sharpness_median']:.3f}, "
          f"p90 {summ['sharpness_p90']:.3f}")
    print(f"   cliff-like (>0.75) {summ['fraction_cliff_like']:.1%}   "
          f"slide-like (<2/steps) {summ['fraction_slide_like']:.1%}   "
          f"monotone {summ['monotone_fraction']:.1%}   "
          f"ever inverted {summ['ever_inverted_fraction']:.1%}")
    for name, v in sorted(summ.get("per_class", {}).items()):
        print(f"      {name:<22} n={v['n']:<5} sharpness median "
              f"{'n/a' if v['sharpness_median'] is None else format(v['sharpness_median'], '.3f')}"
              f"   ever inverted {v['ever_inverted_fraction']:.1%}")

    payload = {
        "checkpoint": str(args.checkpoint), "router": str(args.router), "device": str(device),
        "seconds": time.perf_counter() - t0, "corpus_fingerprint": corpus.fingerprint(),
        "config": vars(args), "ladder": [s.to_dict() for s in sets],
        "gate": g.to_dict(), "trajectories": summ,
        "trajectory_matrix": {"fact_ids": traj.fact_ids.tolist(),
                              "loads": traj.loads.tolist(),
                              "margins": traj.margins.tolist()},
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
