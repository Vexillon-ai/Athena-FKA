"""M3 CAPACITY KNEE — fine ladder, per depth-class, with the frozen bits/bit gate.

    python scripts/run_m3_knee.py --checkpoint <kernel> --router <router.pt> --query-head <h>

One experiment, two readouts (M3 ruling 1): the load axis is sampled **densely through the 16 -> 12
bits/fact interval**, where every cliff observed so far lives, so the same run locates the knee and
sharpens the shape verdict.

**Per depth-class (ruling 2).** §12 established that degradation is a hop-depth effect, not a
relation-type one, so the substrate's own number is the **direct-query knee**. The composed-query
knee is reported as the **no-denoiser integrated floor**, and the direct-composed gap at each load
is recorded explicitly as **Phase 4's measured target**. Charging composed-query noise to the
substrate would bill Phase 3 for Phase 4's unbuilt component.

**Values are stored too, and that is not a detail.** Until now the substrate held only the entity
codes the key path reads; the 8,000 fact *values* still came from the frozen oracle matrix. A
bits/bit figure computed that way omits value storage from the denominator and flatters the design
by a large factor. Here a second store holds `values_matrix` under the same config, so the
accounting covers everything the inference path needs — which is what M3 §10.5 requires.
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
from fka.eval.accounting import GATE_PASS, GATE_TARGET, StorageAccount  # noqa: E402
from fka.eval.margin import MarginSet, MarginTrajectories, retrieval_margins  # noqa: E402
from fka.kernel.checkpoint import load_checkpoint  # noqa: E402
from fka.kernel.gpu_lock import gpu_lock  # noqa: E402
from fka.kernel.latent_episodes import d3_tokenizer, one_hop_episode, pack  # noqa: E402
from fka.kernel.latent_kernel import LatentKernelConfig, LatentReasoningKernel  # noqa: E402
from fka.kernel.latent_memory import LatentCodebook, OracleLatentMemory  # noqa: E402
from fka.kernel.latent_train import evaluate_d3  # noqa: E402
from fka.kernel.seeding import seed_everything  # noqa: E402
from fka.router.routed_memory import RoutedLatentMemory  # noqa: E402
from fka.store.base import IdentityStore  # noqa: E402
from fka.store.s1_factorized import S1Config, S1FactorizedStore  # noqa: E402
from scripts.run_d3_kernel import build_data  # noqa: E402
from scripts.run_fork_a_joint import RELATIONS, harvest  # noqa: E402
from scripts.run_fork_c import address_holdout, supervised_addresses  # noqa: E402
from scripts.run_s1_first_point import build_table  # noqa: E402

LADDER_BASE = S1Config(n_stages=2, codebook_size=256, residual_dim=0, residual_bits=8)
#: TWO one-knob ladders, each varying only `codebook_size` at a fixed stage count.
#: bits = n_stages * log2(K).
#:
#: The 2-stage family is dense through 12-16 bits because every cliff observed so far lives there —
#: it carries the SHAPE readout. It also never reaches the knee criterion, which is itself a result:
#: two stages cannot hold this corpus at any codebook size tried.
#: The 4-stage family spans the region where the standing results already put full addressability
#: (32 bits -> 99.8%), so it carries the KNEE readout.
FAMILIES = {
    2: [10.0, 11.0, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 17.0, 18.0, 20.0],
    4: [16.0, 20.0, 24.0, 28.0, 30.0, 32.0, 34.0, 36.0, 40.0, 44.0],
}
SHAPE_FAMILY = 2
KNEE_CRITERION = 0.99

DEPTH = {0: "direct", 1: "composed"}
HOPS = {0: "hop0(direct)", 1: "hop1", 2: "hop2"}


def cfg_for(bits: float, stages: int, dim: int, seed: int) -> S1Config:
    k = max(2, int(round(2 ** (bits / stages))))
    return replace(LADDER_BASE, latent_dim=dim, n_stages=stages, codebook_size=k, seed=seed)


@torch.no_grad()
def gather(model, routed, packs, device, amp, supervised):
    """Queries, targets, depth class and absolute hop, over every probe set."""
    qs, ts, ds, hs = [], [], [], []
    for packed in packs:
        q, t, h = harvest(model, routed, packed, device, amp)
        qs.append(q)
        ts.append(t.cpu().numpy())
        hh = h.cpu().numpy()
        ds.append(np.where(hh == 0, 0, 1))
        hs.append(hh)
    q = torch.cat(qs)
    t, d, h = np.concatenate(ts), np.concatenate(ds), np.concatenate(hs)
    keep = ~np.isin(t, supervised)
    return q[torch.from_numpy(keep).to(device)], t[keep], d[keep], h[keep]


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
    tok, bs = d3_tokenizer(), meta["block_size"]
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
    rng = np.random.default_rng(args.seed)
    attrs = [r for r in RELATIONS if r != "works_with"]
    packed1 = pack(
        [one_hop_episode(corpus, int(e), attrs[int(rng.integers(0, len(attrs)))])
         for e in rng.permutation(split.heldout)[: args.n_one_hop]],
        tok, bs, oracle.fact_index,
    )
    packs = [packed3, packed1]
    n_e, dim = corpus.n_entities, meta["latent_dim"]
    codes = F.normalize(oracle.codebook.entity, dim=-1).to(device)
    values = oracle.values_matrix.to(device)

    print(f"== M3 capacity knee on {device}")
    print(f"   corpus {corpus.n_facts:,} facts, {corpus.total_bits:,.0f} knowledge bits "
          f"({corpus.total_bits / corpus.n_facts:.2f} bits/fact)")

    def routed_for(store):
        return RoutedLatentMemory(
            oracle, build_table(meta, store, n_e, device, args.router), RELATIONS
        ).to(device).freeze_keys()

    # -- gate ------------------------------------------------------------------------------
    ident = IdentityStore()
    ident.write(codes)
    r_ident = routed_for(ident)
    q, t, d, h = gather(model, r_ident, packs, device, amp, supervised)
    m = retrieval_margins(r_ident.keys, q, torch.from_numpy(t).to(device)).cpu().numpy()
    print(f"\n-- gate: IdentityStore retrieved {float((m > 0).mean()):.1%}, "
          f"min margin {m.min():+.4f}")
    if float((m > 0).mean()) < 0.999:
        raise SystemExit("GATE FAILED")
    print("   -> OK")

    # -- Stage A: the two one-knob ladders ------------------------------------------------
    all_rows: dict[int, list] = {}
    shape_sets = []
    for stages, bit_list in FAMILIES.items():
        print(f"\n-- ladder: {stages} stages, no residual, K varying ({len(bit_list)} rungs)")
        print(f"   {'bits':>6} {'K':>6}  {'direct':>8} {'composed':>9} {'GAP':>7}   "
              f"{'hop0':>7} {'hop1':>7} {'hop2':>7}")
        rows = []
        for bits in sorted(bit_list, reverse=True):
            cfg = cfg_for(bits, stages, dim, args.seed)
            store = S1FactorizedStore(cfg)
            store.write(codes)
            routed = routed_for(store)
            q, t, d, h = gather(model, routed, packs, device, amp, supervised)
            m = retrieval_margins(routed.keys, q, torch.from_numpy(t).to(device)).cpu().numpy()

            direct = float((m[d == 0] > 0).mean())
            comp = float((m[d == 1] > 0).mean())
            by_hop = {HOPS[k]: float((m[h == k] > 0).mean()) for k in sorted(set(h.tolist()))}
            if stages == SHAPE_FAMILY:
                shape_sets.append(MarginSet(
                    fact_ids=t, margins=m, load=cfg.bits_per_slot,
                    label=f"K={cfg.codebook_size}", classes=d, class_names=DEPTH))
            rows.append({"stages": stages, "bits": cfg.bits_per_slot,
                         "codebook_size": cfg.codebook_size, "direct": direct, "composed": comp,
                         "gap": direct - comp, "by_abs_hop": by_hop,
                         "mean_margin": float(m.mean())})
            print(f"   {cfg.bits_per_slot:>6.1f} {cfg.codebook_size:>6}  {direct:>8.1%} "
                  f"{comp:>9.1%} {direct - comp:>7.1%}   " +
                  " ".join(f"{by_hop.get(HOPS[k], float('nan')):>7.1%}" for k in (0, 1, 2)))
        all_rows[stages] = rows

    rows = [r for rs in all_rows.values() for r in rs]

    def knee(key: str) -> dict:
        ok = [r for r in rows if r[key] >= KNEE_CRITERION]
        return min(ok, key=lambda r: r["bits"]) if ok else {}

    knee_direct, knee_comp = knee("direct"), knee("composed")
    print(f"\n== KNEE at retrieval >= {KNEE_CRITERION:.0%}")
    print(f"   DIRECT (substrate's number)      {knee_direct.get('bits', float('nan')):.1f} "
          f"bits/fact (K={knee_direct.get('codebook_size')})")
    print(f"   COMPOSED (no-denoiser floor)     {knee_comp.get('bits', float('nan')):.1f} "
          f"bits/fact (K={knee_comp.get('codebook_size')})")

    # -- Stage A2: sharpened shape + the 44.2% diagnostic ----------------------------------
    traj = MarginTrajectories.from_sets(shape_sets)
    summ = traj.summary()
    print("\n== SHAPE, sharpened on the fine ladder")
    print(f"   {summ['n_facts']} facts x {summ['n_loads']} loads; uniform-slide reference "
          f"{summ['uniform_slide_reference']:.3f}")
    print(f"   sharpness median {summ['sharpness_median']:.3f}   "
          f"cliff-like {summ['fraction_cliff_like']:.1%}   "
          f"slide-like {summ['fraction_slide_like']:.1%}   "
          f"monotone {summ['monotone_fraction']:.1%}")
    print("   44.2% MINORITY DIAGNOSTIC: does non-cliff resolve into cliff at finer granularity?")
    print(f"      coarse ladder (8 loads) cliff-like was 55.8%; fine ladder "
          f"({summ['n_loads']} loads) is {summ['fraction_cliff_like']:.1%}")

    # -- Stage B: the gate at the knee, WITH values stored ---------------------------------
    print("\n== GATE AT THE KNEE — values stored in the substrate too")
    gate_rows = {}
    for name, kn in (("direct-knee", knee_direct), ("composed-knee", knee_comp)):
        if not kn:
            continue
        cfg = cfg_for(kn["bits"], int(kn["stages"]), dim, args.seed)
        key_store = S1FactorizedStore(cfg)
        key_store.write(codes)
        val_store = S1FactorizedStore(cfg)
        val_store.write(values)

        routed = routed_for(key_store)
        routed.values_matrix = val_store.reconstruct(torch.arange(len(values), device=device))
        res, _ = evaluate_d3(model, routed, packed3, tok, device=device, amp_dtype=amp)

        tail = packed3.hop_fact_index[:, -1]
        correct_ids = np.array([
            tail[i] for i, eid in enumerate(packed3.episode_id) if res.per_episode.get(int(eid))
        ])
        known = corpus.bits_of_ids(correct_ids) if correct_ids.size else 0.0

        key_cm, val_cm = key_store.cost_model(), val_store.cost_model()
        table_params = sum(p.numel() for p in build_table(
            meta, key_store, n_e, device, args.router).parameters())
        storage = (
            key_cm["per_fact_storage_bits"] * n_e
            + val_cm["per_fact_storage_bits"] * len(values)
        )
        shared = key_cm["shared_parameters"] + val_cm["shared_parameters"] + table_params
        # Scale the sampled knowledge up to the corpus the store actually holds: the probe set is
        # a sample of the facts, exactly as `fka.eval.capacity` extrapolates.
        scale = corpus.n_facts / max(1, len(packed3.tokens))
        acc = StorageAccount(n_facts=1, per_fact_bits=storage, shared_params=shared,
                             knowledge_bits=known * scale,
                             breakdown={"key_slots": n_e, "value_slots": len(values),
                                        "key_path_params": table_params,
                                        "probe_accuracy": res.accuracy,
                                        "extrapolation_factor": scale})
        gate_rows[name] = {"bits_per_fact": cfg.bits_per_slot, "codebook_size": cfg.codebook_size,
                           "end_to_end_accuracy": res.accuracy, **acc.to_dict()}
        print(f"   {name:<14} {cfg.bits_per_slot:.1f} b/fact  end-to-end {res.accuracy:.1%}  "
              f"{acc}")

    print(f"\n   gates: PASS >= {GATE_PASS}, TARGET > {GATE_TARGET}")

    payload = {"checkpoint": str(args.checkpoint), "device": str(device),
               "seconds": time.perf_counter() - t0,
               "corpus": {"n_facts": corpus.n_facts, "total_bits": corpus.total_bits},
               "config": vars(args), "ladders": all_rows,
               "knee": {"direct": knee_direct, "composed": knee_comp,
                        "criterion": KNEE_CRITERION},
               "shape": summ, "gate_at_knee": gate_rows,
               "trajectory_matrix": {"fact_ids": traj.fact_ids.tolist(),
                                     "loads": traj.loads.tolist(),
                                     "margins": traj.margins.tolist()}}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    print(f"\n== done in {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
