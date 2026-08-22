"""Fork (c): content-computed keys, on an ADDRESS-LEVEL holdout.

    python scripts/run_fork_c.py --checkpoint experiments/<run>/checkpoint_dim64.pt --timing-probe
    python scripts/run_fork_c.py --checkpoint ... --warm-steps 3000 --joint-steps 3000

Pre-registered in `docs/decision_records/M2_router.md` §9.1-rev and §12, before this ran.

Two things changed from fork (a), and both were forced by §11:

**1. The invariant is reachability, not parameter count.** `ContentKeyTable` has no parameter
indexed by entity id; the entity side is a learned encoder over the frozen codebook latent — the
same vector the kernel already receives as `subject_code`. `tests/test_content_keys.py` asserts
this by gradient support, and asserts that the *old* table fails the same check.

**2. The holdout is at the level of ADDRESSES.** Fork (a) inherited M1's subject-level entity
holdout, under which 85.8% of eval addresses had already been supervised. Here every training
episode that touches a held-out entity *at any hop* is dropped, so a held-out entity's address is
never supervised, by construction and not by luck. The headline gate is the never-supervised class.

Cost: 80.3% of training episodes survive the filter, and 0 of 400 held-out entities appear as a
supervised address — both asserted at run time, because a silent regression here would restore
exactly the defect this run exists to remove.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fka.data.corpus_gen import CorpusConfig, generate_corpus  # noqa: E402
from fka.data.hardening import entity_split  # noqa: E402
from fka.eval.latent_leakage import latent_substitution_test  # noqa: E402
from fka.eval.router_eval import evaluate_router, identity_slot_map  # noqa: E402
from fka.eval.separability import entity_recovery, separability  # noqa: E402
from fka.eval.timing import benchmark  # noqa: E402
from fka.kernel.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from fka.kernel.gpu_lock import gpu_lock  # noqa: E402
from fka.kernel.latent_episodes import d3_tokenizer, pack  # noqa: E402
from fka.kernel.latent_kernel import LatentKernelConfig, LatentReasoningKernel  # noqa: E402
from fka.kernel.latent_memory import LatentCodebook, OracleLatentMemory  # noqa: E402
from fka.kernel.latent_train import _to_device, evaluate_d3  # noqa: E402
from fka.kernel.seeding import seed_everything  # noqa: E402
from fka.router.composed_keys import key_spread  # noqa: E402
from fka.router.content_keys import ContentKeyConfig, ContentKeyTable  # noqa: E402
from fka.router.dense import DenseKeyRouter, GoldStubRouter  # noqa: E402
from fka.router.routed_memory import RoutedLatentMemory, oracle_key_stub  # noqa: E402
from scripts.run_d3_kernel import build_data  # noqa: E402
from scripts.run_fork_a_joint import (  # noqa: E402
    RELATIONS,
    fit_router_only,
    harvest,
    joint_step,
    learned_stack_path_gate,
    oscillation_triggered,
)

GATE_RETRIEVAL = 0.99  # parity: the oracle scores 100% on this class too
GATE_STICK = 0.05
TRIGGER_ORACLE_E2E = 0.95
#: Fork (a) never-supervised recall, the number fork (c) has to beat to mean anything.
FORK_A_NEVER_SUPERVISED = 0.0


def address_holdout(groups, heldout_entities: np.ndarray, n_entities: int):
    """Drop every training episode touching a held-out entity AT ANY HOP.

    Subject-level holdout is not enough and that is the whole lesson of §11.2: a router's address is
    supervised whenever a fact is a retrieval *target*, so an episode whose subject is a training
    entity still teaches the addresses of the entities its chain passes through.
    """
    kept, dropped = [], 0
    for g in groups:
        ent = g.hop_fact_index % n_entities
        keep = ~np.isin(ent, heldout_entities).any(axis=1)
        dropped += int((~keep).sum())
        kept.append(g.select(np.flatnonzero(keep)))
    return kept, dropped


def supervised_addresses(groups) -> np.ndarray:
    return np.unique(np.concatenate([g.hop_fact_index.ravel() for g in groups]))


@torch.no_grad()
def assess(model, routed, packed, corpus, device, amp, smap, *, n_hops, supervised):
    """Every table carries the address split. The never-supervised class is the headline."""
    routed.freeze_keys()
    queries, targets, hops = harvest(model, routed, packed, device, amp)
    correct = routed.retrieved_index(queries) == targets

    t_np = targets.cpu().numpy()
    unseen = torch.from_numpy(~np.isin(t_np, supervised)).to(device)
    h_np = hops.cpu().numpy()

    def frac(mask):
        return float(correct[mask].float().mean()) if int(mask.sum()) else None

    per_hop, per_hop_unseen = [], []
    for h in range(n_hops):
        hop_mask = hops == h
        per_hop.append(frac(hop_mask))
        per_hop_unseen.append(frac(hop_mask & unseen))

    uniq, first = np.unique(t_np, return_index=True)
    q_eval = queries[torch.from_numpy(first).to(device)]
    keys = routed.keys

    router_res = evaluate_router(
        DenseKeyRouter(keys), q_eval, uniq, smap, corpus, RELATIONS,
        supervised_fact_ids=supervised,
    )
    sep = separability(keys, q_eval, uniq, corpus.n_entities, len(RELATIONS))

    # §10.1: measured on the never-supervised class, which is the only one that can falsify it.
    unseen_u = ~np.isin(uniq, supervised)
    ent_rec = (
        entity_recovery(keys, q_eval[torch.from_numpy(unseen_u).to(device)], uniq[unseen_u],
                        corpus.n_entities, len(RELATIONS))
        if unseen_u.any() else None
    )
    spread = key_spread(keys)
    routed.thaw_keys()

    return {
        "retrieval_all": float(correct.float().mean()),
        "retrieval_never_supervised": frac(unseen),
        "retrieval_supervised": frac(~unseen),
        "n_never_supervised": int(unseen.sum()),
        "per_hop_retrieval": per_hop,
        "per_hop_retrieval_never_supervised": per_hop_unseen,
        "router": router_res.to_dict(),
        "worst_binding_margin": router_res.worst_binding_margin,
        "key_spread": spread,
        "separability": sep.to_dict(),
        "entity_recovery": ent_rec.to_dict() if ent_rec else None,
        "n_hops_by_class": {
            "never_supervised": int(unseen.sum()), "supervised": int((~unseen).sum()),
            "by_hop": [int(((hops == h) & unseen).sum()) for h in range(n_hops)],
        },
        "_router_str": str(router_res),
        "_sep_str": str(sep),
        "_ent_str": str(ent_rec) if ent_rec else "n/a",
        "_h": h_np,
    }


def _show(tag, a):
    ns = a["retrieval_never_supervised"]
    print(f"   [{tag}]")
    print(f"   NEVER-SUPERVISED retrieval {ns:.1%} (n={a['n_never_supervised']})   "
          f"supervised {a['retrieval_supervised']:.1%}   all {a['retrieval_all']:.1%}")
    hops = ["%.1f%%" % (100 * h) if h is not None else "-"
            for h in a["per_hop_retrieval_never_supervised"]]
    print(f"   per-hop never-supervised {hops}")
    print(f"   {a['_router_str']}")
    print(f"   {a['_sep_str']}")
    print(f"   {a['_ent_str']}")
    print(f"   effective rank {a['key_spread']['effective_rank']:.1f}/"
          f"{a['key_spread']['key_dim']}   worst margin {a['worst_binding_margin']:+.4f}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--n-entities", type=int, default=2000)
    p.add_argument("--warm-steps", type=int, default=3000)
    p.add_argument("--joint-steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--query-lr-scale", type=float, default=0.1)
    p.add_argument("--temp", type=float, default=0.05)
    p.add_argument("--comp-dim", type=int, default=128)
    p.add_argument("--encoder-hidden", type=int, default=256)
    p.add_argument("--mode", default="bilinear", choices=("bilinear", "mlp"))
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--n-eval", type=int, default=400)
    p.add_argument("--n-train-harvest", type=int, default=1200)
    p.add_argument("--alt-block", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--force-gpu-lock", action="store_true")
    p.add_argument("--label", default="forkc")
    p.add_argument("--out", default=None)
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--timing-probe", action="store_true",
                   help="measure step cost and project wall clock, then exit without training")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)

    if args.smoke:
        args.warm_steps, args.joint_steps = 20, 10
        args.eval_every, args.n_eval, args.n_train_harvest = 10, 16, 32
        args.batch_size, args.device = 4, "cpu"
        args.n_entities, args.comp_dim, args.encoder_hidden = 40, 16, 16

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


def _run(args, device) -> int:  # noqa: C901 - a driver; the sequence is the point
    seed_everything(args.seed)
    t_start = time.perf_counter()

    if args.smoke:
        meta = {"codebook_seed": 0, "latent_dim": 16, "block_size": 128}
        mcfg = {"vocab_size": len(d3_tokenizer().stoi), "block_size": 128, "n_layer": 2,
                "n_head": 2, "n_embd": 32, "latent_dim": 16, "n_read_heads": 1,
                "cross_attn_every": 2, "n_hops": 2}
    else:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required outside --smoke")
        blob = load_checkpoint(args.checkpoint)
        meta, mcfg = blob["extra"], blob["model_config"]

    corpus = generate_corpus(
        CorpusConfig(n_entities=args.n_entities, seed=meta["codebook_seed"], n_coworkers=1)
    )
    if not args.smoke and corpus.fingerprint() != meta["corpus_fingerprint"]:
        raise SystemExit("corpus fingerprint mismatch with the checkpoint")
    if list(corpus.relations) != RELATIONS:
        raise SystemExit(f"relation order {list(corpus.relations)} != {RELATIONS}")

    tok = d3_tokenizer()
    block_size = meta["block_size"]
    split = entity_split(corpus, fraction=0.2, seed=meta["codebook_seed"])
    oracle = OracleLatentMemory(
        corpus, LatentCodebook.build(corpus, dim=meta["latent_dim"], seed=meta["codebook_seed"])
    ).to(device)
    model = LatentReasoningKernel(LatentKernelConfig(**mcfg)).to(device)
    if not args.smoke:
        load_checkpoint(args.checkpoint, model)
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    amp = torch.bfloat16 if device.type == "cuda" else None

    raw_groups, heldout_eps, _ = build_data(
        corpus, tok, split, [2, 3], oracle.fact_index, block_size, meta["codebook_seed"]
    )
    groups, dropped = address_holdout(raw_groups, split.heldout, corpus.n_entities)
    supervised = supervised_addresses(groups)
    packed_held = pack(heldout_eps[: args.n_eval], tok, block_size, oracle.fact_index)
    n_hops = packed_held.n_hops
    smap = identity_slot_map(corpus, RELATIONS)

    kept = sum(len(g.tokens) for g in groups)
    total = sum(len(g.tokens) for g in raw_groups)
    leaked = len({int(f) % corpus.n_entities for f in supervised} & set(split.heldout.tolist()))
    print(f"== Fork (c) — CONTENT keys, ADDRESS-LEVEL holdout, on {device}  [{args.label}]")
    print(f"   training episodes {kept:,}/{total:,} kept ({kept / total:.1%}), {dropped:,} dropped")
    print(f"   supervised addresses {len(supervised):,}/{corpus.n_entities * len(RELATIONS):,} "
          f"({len(supervised) / (corpus.n_entities * len(RELATIONS)):.1%})")
    print(f"   held-out entities appearing as a supervised address: {leaked}  (must be 0)")
    if leaked:
        raise SystemExit("ADDRESS HOLDOUT LEAKED — the run would measure the fork (a) defect again")

    # -- instrument gates ------------------------------------------------------------------
    print("\n-- instrument gate 1/2: ORACLE keys through the learned-stack eval path")
    ok_stub, stub = learned_stack_path_gate(
        model, oracle, oracle_key_stub(oracle, RELATIONS), packed_held, device, amp, n_hops
    )
    print(f"   queries identical {stub['queries_identical']}, addresses identical "
          f"{stub['addresses_identical']}  -> {'OK' if ok_stub else 'FAILED'}")
    if not ok_stub:
        raise SystemExit("learned-stack eval path gate FAILED; no fork (c) number is admissible")

    print("-- instrument gate 2/2: gold stub through the REBUILT router eval path")
    ids = np.arange(len(oracle))
    gt = torch.from_numpy(smap.fact_to_slot[ids]).to(device)
    g = evaluate_router(
        GoldStubRouter(gt, smap.n_slots), oracle.keys, ids, smap, corpus, RELATIONS,
        supervised_fact_ids=supervised,
    )
    stub_ok = (
        g.recall_at_1 == 1.0 and g.worst_binding_margin > 0 and g.recall_at_1_unseen == 1.0
    )
    print(f"   recall@1 {g.recall_at_1:.1%}, NEVER-SUPERVISED {g.recall_at_1_unseen:.1%} "
          f"(n={g.n_unseen}), worst margin {g.worst_binding_margin:+.4f}  "
          f"-> {'OK' if stub_ok else 'FAILED'}")
    if not stub_ok:
        raise SystemExit("router eval path gate FAILED; no fork (c) number is admissible")

    # -- the table -------------------------------------------------------------------------
    table = ContentKeyTable(
        ContentKeyConfig(
            n_relations=len(RELATIONS), latent_dim=meta["latent_dim"],
            key_dim=meta["latent_dim"], comp_dim=args.comp_dim,
            encoder_hidden=args.encoder_hidden, mode=args.mode,
        ),
        oracle.codebook.entity,
    ).to(device)
    routed = RoutedLatentMemory(oracle, table, RELATIONS).to(device)
    n_facts = corpus.n_entities * len(RELATIONS)
    ents = torch.arange(n_facts, device=device) % corpus.n_entities
    rels = torch.arange(n_facts, device=device) // corpus.n_entities

    train_3hop = groups[-1]
    sel = np.arange(min(args.n_train_harvest, len(train_3hop.tokens)))
    q_tr, t_tr, _ = harvest(model, oracle, train_3hop.select(sel), device, amp)
    codes = oracle.codebook.entity

    # -- timing probe: size the run from a MEASURED step cost (standing rule) ---------------
    if args.timing_probe:
        print(f"\n-- timing probe ({table.n_params:,} key params, batch {args.batch_size})")
        model.query_out.weight.requires_grad_(True)
        packed0 = groups[-1]
        idx = np.arange(min(args.batch_size, len(packed0.tokens)))
        batch = _to_device(packed0, idx, device)

        warm = benchmark(
            lambda: fit_router_only(table, ents, rels, q_tr, t_tr, steps=1, lr=args.lr,
                                    temp=args.temp, log_every=10**9),
            name="router-only step", warmup=3, repeats=10, device=device,
        )

        def one_joint():
            table.zero_grad(set_to_none=True)
            model.query_out.grad = None
            joint_step(model, routed, batch, codes[batch["subject_ids"]], amp, args.temp)

        joint = benchmark(one_joint, name="joint step", warmup=3, repeats=10, device=device)
        proj = args.warm_steps * warm.best + args.joint_steps * joint.best
        print(f"   router-only  {warm.best * 1e3:8.1f} ms/step  "
              f"(spread {warm.relative_spread:.1%})")
        print(f"   joint        {joint.best * 1e3:8.1f} ms/step  "
              f"(spread {joint.relative_spread:.1%})")
        print(f"   projected training: {args.warm_steps:,} warm + {args.joint_steps:,} joint "
              f"= {proj / 60:.1f} min (evaluation excluded)")
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps({
                "router_only_ms": warm.best * 1e3, "joint_ms": joint.best * 1e3,
                "projected_seconds": proj, "key_params": table.n_params,
                "config": vars(args)}, indent=2), encoding="utf-8")
            print(f"\nwrote {args.out}")
        return 0

    # -- Stage 1: router-only against the frozen kernel -------------------------------------
    print(f"\n-- Stage 1: router-only, kernel frozen ({args.mode}, {table.n_params:,} key params, "
          f"independent of n_entities)")
    print(f"   harvested {len(q_tr):,} training queries through the ORACLE memory")
    warm_losses = fit_router_only(
        table, ents, rels, q_tr, t_tr, steps=args.warm_steps, lr=args.lr, temp=args.temp
    )
    warm = assess(model, routed, packed_held, corpus, device, amp, smap,
                  n_hops=n_hops, supervised=supervised)
    _show(f"Stage 1 — {args.warm_steps} router-only steps", warm)

    # -- Stage 2: staged unfreeze -----------------------------------------------------------
    model.query_out.weight.requires_grad_(True)
    opt = torch.optim.AdamW([
        {"params": list(table.parameters()), "lr": args.lr, "name": "router"},
        {"params": [model.query_out.weight], "lr": args.lr * args.query_lr_scale,
         "name": "query_head"},
    ])
    base_lrs = [grp["lr"] for grp in opt.param_groups]
    print(f"\n-- Stage 2: staged unfreeze — query head at {args.query_lr_scale:g}x")

    rng = np.random.default_rng(args.seed)
    history = []
    margins = [warm["worst_binding_margin"]]
    recalls = [warm["retrieval_never_supervised"] or 0.0]
    alternating_from = None
    model.train()

    for step in range(args.joint_steps):
        cos = 0.5 * (1 + math.cos(math.pi * min(1.0, step / max(1, args.joint_steps))))
        for gi, grp in enumerate(opt.param_groups):
            lr = base_lrs[gi] * cos
            if alternating_from is not None:
                router_block = ((step - alternating_from) // args.alt_block) % 2 == 0
                if (grp["name"] == "router") != router_block:
                    lr = 0.0
            grp["lr"] = lr

        packed = groups[int(rng.integers(0, len(groups)))]
        idx = rng.integers(0, len(packed.tokens), size=args.batch_size)
        batch = _to_device(packed, idx, device)
        opt.zero_grad(set_to_none=True)
        a_loss, r_loss = joint_step(
            model, routed, batch, codes[batch["subject_ids"]], amp, args.temp
        )
        torch.nn.utils.clip_grad_norm_(
            [prm for grp in opt.param_groups for prm in grp["params"]], 1.0
        )
        opt.step()

        if (step + 1) % args.eval_every == 0 or step == args.joint_steps - 1:
            model.eval()
            snap = assess(model, routed, packed_held, corpus, device, amp, smap,
                          n_hops=n_hops, supervised=supervised)
            model.train()
            snap["step"], snap["answer_loss"], snap["retrieval_loss"] = step + 1, a_loss, r_loss
            history.append(snap)
            margins.append(snap["worst_binding_margin"])
            recalls.append(snap["retrieval_never_supervised"] or 0.0)
            print(f"   step {step + 1:>5}/{args.joint_steps}  answer {a_loss:.4f}  "
                  f"retrieval {r_loss:.4f}  NEVER-SUP {recalls[-1]:.1%}  "
                  f"sup {snap['retrieval_supervised']:.1%}  "
                  f"margin {margins[-1]:+.4f}  rank {snap['key_spread']['effective_rank']:.1f}")
            if alternating_from is None and oscillation_triggered(margins, recalls):
                alternating_from = step + 1
                print(f"   ** §9.7 alternation trigger fired at step {step + 1}")

    model.eval()
    seconds = time.perf_counter() - t_start

    # -- gates -------------------------------------------------------------------------------
    final = assess(model, routed, packed_held, corpus, device, amp, smap,
                   n_hops=n_hops, supervised=supervised)
    print("\n== Gates (§12.4) — headline is the NEVER-SUPERVISED class")
    _show("final", final)
    routed.freeze_keys()

    stick_pack = train_3hop.select(np.arange(min(200, len(train_3hop.tokens))))
    sub = latent_substitution_test(
        model, routed, stick_pack, tok, device=device, amp_dtype=amp, seed=args.seed
    )
    e2e_routed, _ = evaluate_d3(model, routed, packed_held, tok, device=device, amp_dtype=amp)
    e2e_oracle, _ = evaluate_d3(model, oracle, packed_held, tok, device=device, amp_dtype=amp)

    ns = final["retrieval_never_supervised"]
    gates = {
        "never_supervised_retrieval": (ns, GATE_RETRIEVAL, ns >= GATE_RETRIEVAL),
        "worst_binding_margin": (
            final["worst_binding_margin"], 0.0, final["worst_binding_margin"] > 0
        ),
        "substitution_stick_rate": (sub.stick_rate, GATE_STICK, sub.stick_rate <= GATE_STICK),
    }
    print()
    for name, (value, threshold, ok) in gates.items():
        print(f"   {name:<28} {value:+.4f}  (gate {threshold:g})  {'PASS' if ok else 'FAIL'}")
    print(f"   {'kernel damage via ORACLE':<28} {e2e_oracle.accuracy:.1%}  "
          f"{'ok' if e2e_oracle.accuracy >= TRIGGER_ORACLE_E2E else 'TRIGGERED'}")
    print(f"   {'end-to-end via learned stack':<28} {e2e_routed.accuracy:.1%}  (reported)")
    print(f"   {'fork (a) on this class':<28} {FORK_A_NEVER_SUPERVISED:.1%}  "
          f"-> fork (c) moves it to {ns:.1%}")

    passed = all(ok for _, _, ok in gates.values())
    print(f"\n== Fork (c): {'PASS' if passed else 'FAIL'}")

    payload = {
        "label": args.label, "checkpoint": str(args.checkpoint), "device": str(device),
        "seconds": seconds, "corpus_fingerprint": corpus.fingerprint(), "config": vars(args),
        "key_params": table.n_params, "kernel_params": model.n_params(),
        "address_holdout": {
            "episodes_kept": kept, "episodes_total": total, "dropped": dropped,
            "supervised_addresses": int(len(supervised)),
            "heldout_entities_leaked": leaked,
        },
        "stage1": {k: v for k, v in warm.items() if not k.startswith("_")},
        "history": [{k: v for k, v in h.items() if not k.startswith("_")} for h in history],
        "final": {k: v for k, v in final.items() if not k.startswith("_")},
        "substitution": sub.to_dict(),
        "e2e_learned_stack": e2e_routed.to_dict(), "e2e_oracle": e2e_oracle.to_dict(),
        "gates": {k: {"value": v, "threshold": t, "pass": ok} for k, (v, t, ok) in gates.items()},
        "stage1_final_loss": warm_losses[-1] if warm_losses else None,
        "alternation_from_step": alternating_from,
        "verdict": "pass" if passed else "fail",
    }
    for blk in [payload["stage1"], payload["final"], *payload["history"]]:
        blk.pop("_h", None)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {args.out}")

    if args.checkpoint_dir:
        d = Path(args.checkpoint_dir)
        d.mkdir(parents=True, exist_ok=True)
        head_path = d / f"query_head_{args.label}.pt"
        torch.save({"query_out.weight": model.query_out.weight.detach().cpu()}, head_path)
        save_checkpoint(
            d / f"router_{args.label}.pt", table, model_config=table.cfg,
            train_config=vars(args),
            extra={"query_head_file": head_path.name, "verdict": payload["verdict"],
                   "corpus_fingerprint": corpus.fingerprint(),
                   "kernel_checkpoint": str(args.checkpoint)},
        )
        print(f"wrote {d / f'router_{args.label}.pt'} and {head_path.name}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
