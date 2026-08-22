"""Fork (a) joint fit: learn the address space and the query space TOGETHER.

    python scripts/run_fork_a_joint.py --checkpoint experiments/<run>/checkpoint_dim64.pt

Pre-registered in `docs/decision_records/M2_router.md` §9, amended in §9.7-9.9 before this ran.

Stage a0 answered the cheap question first and said no: composed keys reach 57.8% against the
**frozen** kernel's queries, with negative worst-case binding margins and an effective rank of
16/64. Serving a fixed client is not sufficient, so the query side has to move — which is what
makes this fit necessary rather than merely permitted.

Two stages (§9.7):

    Stage 1  router-only warm start   kernel frozen; keys fitted to harvested queries
    Stage 2  staged unfreeze          `query_out` trainable at 0.1x the router's LR

with alternation as the registered fallback if margins oscillate, on a trigger fixed in advance.

What is measured, and what is only reported
-------------------------------------------
Gates (§9.4): per-hop retrieval on entity-held-out probes through the learned stack, worst-case
binding margin, substitution stick rate, and a kernel-damage check through the ORACLE memory —
that last one measured through a path the router does not touch, so a drop can only mean kernel
damage (§9.3(c)).

Reported, never gated: end-to-end 3-hop (ceilinged by the frozen kernel), key spread, and the
separability index (§9.8), which is the number that tells §10 whether searchability is close to
solved or is the hard open problem.
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fka.data.corpus_gen import CorpusConfig, generate_corpus  # noqa: E402
from fka.data.hardening import entity_split  # noqa: E402
from fka.eval.latent_leakage import latent_substitution_test  # noqa: E402
from fka.eval.router_eval import evaluate_router, identity_slot_map  # noqa: E402
from fka.eval.separability import separability  # noqa: E402
from fka.kernel.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from fka.kernel.gpu_lock import gpu_lock  # noqa: E402
from fka.kernel.latent_episodes import d3_tokenizer, pack  # noqa: E402
from fka.kernel.latent_kernel import LatentKernelConfig, LatentReasoningKernel  # noqa: E402
from fka.kernel.latent_memory import LatentCodebook, OracleLatentMemory  # noqa: E402
from fka.kernel.latent_train import _to_device, evaluate_d3  # noqa: E402
from fka.kernel.seeding import seed_everything  # noqa: E402
from fka.router.composed_keys import ComposedKeyConfig, ComposedKeyTable, key_spread  # noqa: E402
from fka.router.dense import DenseKeyRouter, GoldStubRouter  # noqa: E402
from fka.router.routed_memory import RoutedLatentMemory, oracle_key_stub  # noqa: E402
from scripts.run_d3_kernel import build_data  # noqa: E402

RELATIONS = ["birth_year", "birth_city", "employer", "works_with"]

#: §9.4 gates.
GATE_RETRIEVAL = 0.99
GATE_STICK = 0.05
#: §9.3(c) regression trigger, measured through the ORACLE memory.
TRIGGER_ORACLE_E2E = 0.95


# =======================================================================================
# harvest / assess
# =======================================================================================


@torch.no_grad()
def harvest(model, memory, packed, device, amp, batch=64):
    """Emitted queries, their target fact ids, and their hop index — through `memory`.

    The memory argument is load-bearing rather than incidental: harvesting through the **routed**
    memory means hop *k*'s query is composed from the value the *learned* router returned, which is
    what "per-hop retrieval through the learned stack" has to mean. Harvesting through the oracle
    and then scoring the router would measure a stack that is never deployed.
    """
    qs, ts, hs = [], [], []
    for start in range(0, len(packed.tokens), batch):
        sl = np.arange(start, min(start + batch, len(packed.tokens)))
        b = _to_device(packed, sl, device)
        code = memory.codebook.entity[b["subject_ids"]]

        def run(b=b, code=code):
            return model(b["tokens"][:, :-1], code, b["subj_pos"], b["qvec_pos"],
                         memory, hard_read=True)

        if amp is None:
            _, _, info = run()
        else:
            with torch.autocast(device_type=device.type, dtype=amp):
                _, _, info = run()
        for hop, q in enumerate(info["queries"]):
            qs.append(q.float().detach())
            ts.append(b["hop_fact_index"][:, hop])
            hs.append(torch.full((len(sl),), hop, device=device))
    return torch.cat(qs), torch.cat(ts), torch.cat(hs)


@torch.no_grad()
def learned_stack_path_gate(model, oracle, stub, packed, device, amp, n_hops):
    """Gate the learned-stack eval path by EQUALITY against the oracle path, not by a threshold.

    An earlier version of this gate demanded per-hop retrieval >= 99.9% from the stub. That is the
    conditional-gate violation CLAUDE.md exists to prevent: the stub's attainable maximum is capped
    by the *kernel's* query quality, so the gate would report failure on a perfectly correct wrapper
    driven by a weak kernel — and it was unrunnable in `--smoke` for exactly that reason, which is
    how the defect surfaced.

    The unceilinged complement is an equality. Substituting the oracle's own keys into the wrapper
    must reproduce the oracle's own queries and its own addresses **exactly**, at any kernel
    quality. Any difference is the wrapper — a fact-id/slot transposition, a stale cache, a dtype
    mismatch — because no router is involved on either side.
    """
    q_o, t_o, h_o = harvest(model, oracle, packed, device, amp)
    q_s, _, _ = harvest(model, stub, packed, device, amp)

    queries_identical = bool(torch.allclose(q_o, q_s, atol=1e-5))
    addresses_identical = bool(torch.equal(oracle.retrieved_index(q_o), stub.retrieved_index(q_o)))
    oracle_hit = oracle.retrieved_index(q_o) == t_o
    wrapper_hit = stub.retrieved_index(q_s) == t_o
    oracle_per_hop = [float(oracle_hit[h_o == h].float().mean()) for h in range(n_hops)]
    wrapper_per_hop = [float(wrapper_hit[h_o == h].float().mean()) for h in range(n_hops)]

    ok = queries_identical and addresses_identical and wrapper_per_hop == oracle_per_hop
    return ok, {
        "oracle_per_hop": oracle_per_hop,
        "wrapper_per_hop": wrapper_per_hop,
        "queries_identical": queries_identical,
        "addresses_identical": addresses_identical,
    }


@torch.no_grad()
def assess(model, routed, packed, corpus, device, amp, smap, *, n_hops, supervised=None):
    """Per-hop retrieval, binding margins, key spread and separability — one harvest, one pass.

    `supervised` is every fact that was a **retrieval target during training**. It splits out
    recall on addresses the router was never taught, which is the only subset that speaks to P3.
    """
    routed.freeze_keys()
    queries, targets, hops = harvest(model, routed, packed, device, amp)
    got = routed.retrieved_index(queries)
    correct = (got == targets)

    per_hop = [float(correct[hops == h].float().mean()) for h in range(n_hops)]
    unseen_mask = None
    if supervised is not None:
        t_np = targets.cpu().numpy()
        unseen_mask = torch.from_numpy(~np.isin(t_np, supervised)).to(device)

    # One query per unique target: `evaluate_router` joins on fact id and needs them unique.
    uniq, first = np.unique(targets.cpu().numpy(), return_index=True)
    q_eval = queries[torch.from_numpy(first).to(device)]

    keys = routed.keys
    router_res = evaluate_router(
        DenseKeyRouter(keys), q_eval, uniq, smap, corpus, RELATIONS,
        supervised_fact_ids=supervised,
    )
    sep = separability(keys, q_eval, uniq, corpus.n_entities, len(RELATIONS))
    spread = key_spread(keys)
    routed.thaw_keys()

    return {
        "per_hop_retrieval": per_hop,
        "retrieval_accuracy": float(correct.float().mean()),
        "retrieval_accuracy_unseen": (
            float(correct[unseen_mask].float().mean()) if unseen_mask is not None
            and int(unseen_mask.sum()) else None
        ),
        "n_unseen_queries": int(unseen_mask.sum()) if unseen_mask is not None else 0,
        "n_distinct_targets": int(len(uniq)),
        "router": router_res.to_dict(),
        "worst_binding_margin": router_res.worst_binding_margin,
        "key_spread": spread,
        "separability": sep.to_dict(),
        "_router_str": str(router_res),
        "_sep_str": str(sep),
    }


# =======================================================================================
# stages
# =======================================================================================


def fit_router_only(table, ents, rels, queries, targets, *, steps, lr, temp, log_every=200):
    """Stage 1: the warm start. Kernel frozen, only `key(e, r)` moves — Stage a0's fit, exactly."""
    opt = torch.optim.AdamW(table.parameters(), lr=lr)
    n = len(queries)
    losses = []
    for step in range(steps):
        idx = torch.randint(0, n, (min(1024, n),), device=queries.device)
        keys = F.normalize(table(ents, rels), dim=-1)
        logits = F.normalize(queries[idx], dim=-1) @ keys.T / temp
        loss = F.cross_entropy(logits, targets[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if step % log_every == 0 or step == steps - 1:
            print(f"      step {step + 1:>5}/{steps}  loss {losses[-1]:.4f}")
    return losses


def joint_step(model, routed, batch, subject_code, amp, temp):
    """One joint forward/backward. Returns (answer_loss, retrieval_loss).

    The read goes through the **routed** memory, so the router is in the loop the kernel actually
    uses; the retrieval term supervises which slot (§9.2) and never where that slot sits.
    """
    x, y = batch["tokens"][:, :-1], batch["tokens"][:, 1:]
    mask = batch["answer_mask"].float()

    with routed.cached_keys():
        def run():
            return model(x, subject_code, batch["subj_pos"], batch["qvec_pos"], routed,
                         targets=y, loss_mask=mask)

        if amp is None:
            _, answer_loss, info = run()
        else:
            with torch.autocast(device_type=x.device.type, dtype=amp):
                _, answer_loss, info = run()

        keys = routed.keys.float()
        retrieval_loss = torch.zeros((), device=x.device)
        for hop, q in enumerate(info["queries"]):
            scores = F.normalize(q.float(), dim=-1) @ keys.T / temp
            retrieval_loss = retrieval_loss + F.cross_entropy(
                scores, batch["hop_fact_index"][:, hop]
            )
        retrieval_loss = retrieval_loss / max(1, len(info["queries"]))
        (answer_loss + retrieval_loss).backward()

    return float(answer_loss.detach()), float(retrieval_loss.detach())


def oscillation_triggered(margins: list[float], recalls: list[float]) -> bool:
    """§9.7's alternation trigger, fixed in advance so 'oscillating' is not decided after the fact.

    Two consecutive checkpoint-over-checkpoint drops in the worst binding margin, while held-out
    retrieval has not improved by more than a point across the same span.
    """
    if len(margins) < 3:
        return False
    a, b, c = margins[-3:]
    return c < b < a and (recalls[-1] - recalls[-3]) <= 0.01


# =======================================================================================
# driver
# =======================================================================================


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--n-entities", type=int, default=2000)
    p.add_argument("--warm-steps", type=int, default=3000)
    p.add_argument("--a0-steps", type=int, default=1500,
                   help="assess here first: Stage a0 used exactly this many router-only steps")
    p.add_argument("--joint-steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3, help="router LR")
    p.add_argument("--query-lr-scale", type=float, default=0.1, help="§9.7: query head at 0.1x")
    p.add_argument("--temp", type=float, default=0.05)
    p.add_argument("--comp-dim", type=int, default=128)
    p.add_argument("--mode", default="bilinear", choices=("bilinear", "mlp"))
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--n-eval", type=int, default=400)
    p.add_argument("--n-train-harvest", type=int, default=1200)
    p.add_argument("--alt-block", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--force-gpu-lock", action="store_true")
    p.add_argument("--label", default="joint")
    p.add_argument("--out", default=None)
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--load-router", default=None,
                   help="re-score a saved router instead of training one")
    p.add_argument("--load-query-head", default=None)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)

    if args.smoke:
        args.warm_steps, args.joint_steps, args.a0_steps = 20, 10, 10
        args.eval_every, args.n_eval, args.n_train_harvest = 10, 16, 32
        args.batch_size, args.device = 4, "cpu"
        args.n_entities, args.comp_dim = 40, 16
    elif not args.checkpoint:
        raise SystemExit("--checkpoint is required outside --smoke")

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
        # A tiny untrained kernel: every path runs, no number means anything, under two minutes.
        meta = {"codebook_seed": 0, "latent_dim": 16, "block_size": 128,
                "corpus_fingerprint": None}
        mcfg = {"vocab_size": len(d3_tokenizer().stoi), "block_size": 128, "n_layer": 2,
                "n_head": 2, "n_embd": 32, "latent_dim": 16, "n_read_heads": 1,
                "cross_attn_every": 2, "n_hops": 2}
    else:
        blob = load_checkpoint(args.checkpoint)
        meta, mcfg = blob["extra"], blob["model_config"]
    corpus = generate_corpus(
        CorpusConfig(n_entities=args.n_entities, seed=meta["codebook_seed"], n_coworkers=1)
    )
    if not args.smoke and corpus.fingerprint() != meta["corpus_fingerprint"]:
        raise SystemExit("corpus fingerprint mismatch with the checkpoint")
    if list(corpus.relations) != RELATIONS:
        raise SystemExit(
            f"relation order {list(corpus.relations)} != {RELATIONS}; fact ids would not line up"
        )

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

    groups, heldout_eps, _ = build_data(
        corpus, tok, split, [2, 3], oracle.fact_index, block_size, meta["codebook_seed"]
    )
    packed_held = pack(heldout_eps[: args.n_eval], tok, block_size, oracle.fact_index)
    n_hops = packed_held.n_hops
    train_3hop = groups[-1]
    smap = identity_slot_map(corpus, RELATIONS)

    # Every fact that is a retrieval TARGET anywhere in training. Its address is supervised, so it
    # cannot speak to compositional generalisation however held-out its subject was (§9.5 erratum).
    supervised = np.unique(np.concatenate([g.hop_fact_index.ravel() for g in groups]))

    print(f"== Fork (a) JOINT FIT on {device}   [{args.label}]")
    print(f"   kernel {model.n_params():,} params from {args.checkpoint}")
    print(f"   {len(packed_held.tokens)} entity-held-out {n_hops}-hop probes")

    # -- instrument gates, both before any number -----------------------------------------
    print("\n-- instrument gate 1/2: ORACLE keys through the learned-stack eval path")
    ok_stub, stub = learned_stack_path_gate(
        model, oracle, oracle_key_stub(oracle, RELATIONS), packed_held, device, amp, n_hops
    )
    print(f"   oracle path per-hop  {['%.1f%%' % (100 * h) for h in stub['oracle_per_hop']]}")
    print(f"   wrapper per-hop      {['%.1f%%' % (100 * h) for h in stub['wrapper_per_hop']]}")
    print(f"   queries identical {stub['queries_identical']}, "
          f"addresses identical {stub['addresses_identical']}  "
          f"-> {'OK' if ok_stub else 'FAILED'}")
    if not ok_stub:
        raise SystemExit("learned-stack eval path gate FAILED; no joint-fit number is admissible")

    print("-- instrument gate 2/2: gold stub through the router eval path")
    gt = torch.from_numpy(smap.fact_to_slot[np.arange(len(oracle))]).to(device)
    g = evaluate_router(
        GoldStubRouter(gt, smap.n_slots),
        oracle.keys[: len(gt)], np.arange(len(oracle)), smap, corpus, RELATIONS,
    )
    print(f"   recall@1 {g.recall_at_1:.1%}  worst margin {g.worst_binding_margin:+.4f}  "
          f"{'OK' if g.recall_at_1 == 1.0 and g.worst_binding_margin > 0 else 'FAILED'}")
    if not (g.recall_at_1 == 1.0 and g.worst_binding_margin > 0):
        raise SystemExit("router eval path gate FAILED; no joint-fit number is admissible")

    # -- Stage 1: router-only warm start ---------------------------------------------------
    table = ComposedKeyTable(ComposedKeyConfig(
        n_entities=corpus.n_entities, n_relations=len(RELATIONS),
        key_dim=meta["latent_dim"], comp_dim=args.comp_dim, mode=args.mode,
    )).to(device)
    routed = RoutedLatentMemory(oracle, table, RELATIONS).to(device)

    n_facts = corpus.n_entities * len(RELATIONS)
    ents = torch.arange(n_facts, device=device) % corpus.n_entities
    rels = torch.arange(n_facts, device=device) // corpus.n_entities

    if args.load_router:
        # Re-score a saved state instead of training one, so a new instrument can be applied to a
        # finished run without paying for the run again.
        load_checkpoint(args.load_router, table)
        if args.load_query_head:
            head = torch.load(args.load_query_head, map_location=device, weights_only=False)
            model.query_out.weight.data.copy_(head["query_out.weight"].to(device))
        args.warm_steps = args.joint_steps = 0
        print(f"\n-- RE-SCORING {args.load_router}"
              f"{' + ' + args.load_query_head if args.load_query_head else ''}")

    print(f"\n-- Stage 1: router-only warm start ({args.mode}, {table.n_params:,} key params)")
    harvest_src = train_3hop
    sel = np.arange(min(args.n_train_harvest, len(harvest_src.tokens)))
    q_tr, t_tr, _ = harvest(model, oracle, harvest_src.select(sel), device, amp)
    print(f"   harvested {len(q_tr):,} training queries through the ORACLE memory")
    # Assessed in chunks so the a0-equivalent point (`--a0-steps`) is reported before the
    # continuation runs past it. Without that, "the warm start scored X" would be a number from a
    # different fit length than a0's and the comparison would quietly stop being one.
    warm_losses, warm_history, done = [], [], 0
    while True:
        if done < args.warm_steps:
            chunk = min(
                args.a0_steps if done == 0 else args.warm_steps - done, args.warm_steps - done
            )
            warm_losses += fit_router_only(
                table, ents, rels, q_tr, t_tr, steps=chunk, lr=args.lr, temp=args.temp
            )
            done += chunk
        warm = assess(model, routed, packed_held, corpus, device, amp, smap, n_hops=n_hops,
                      supervised=supervised)
        warm["step"] = done
        warm_history.append(warm)
        tag = "  <- a0-equivalent" if done == args.a0_steps else ""
        print(f"   [{done}/{args.warm_steps} steps]{tag}")
        print(f"   {warm['_router_str']}")
        print(f"   per-hop {['%.1f%%' % (100 * h) for h in warm['per_hop_retrieval']]}  "
              f"worst margin {warm['worst_binding_margin']:+.4f}")
        print(f"   {warm['_sep_str']}")
        print(f"   effective rank {warm['key_spread']['effective_rank']:.1f}/"
              f"{warm['key_spread']['key_dim']}")
        if done >= args.warm_steps:
            break

    # -- Stage 2: staged unfreeze ----------------------------------------------------------
    model.query_out.weight.requires_grad_(True)
    opt = torch.optim.AdamW([
        {"params": list(table.parameters()), "lr": args.lr, "name": "router"},
        {"params": [model.query_out.weight], "lr": args.lr * args.query_lr_scale,
         "name": "query_head"},
    ])
    base_lrs = [g["lr"] for g in opt.param_groups]

    print(f"\n-- Stage 2: staged unfreeze — query head at {args.query_lr_scale:g}x "
          f"(lr {base_lrs[1]:.2e} vs router {base_lrs[0]:.2e})")

    rng = np.random.default_rng(args.seed)
    history, margins, recalls = [], [warm["worst_binding_margin"]], [
        float(np.mean(warm["per_hop_retrieval"]))
    ]
    alternating_from = None
    codes = oracle.codebook.entity
    model.train()

    for step in range(args.joint_steps):
        cos = 0.5 * (1 + math.cos(math.pi * min(1.0, step / max(1, args.joint_steps))))
        for gi, grp in enumerate(opt.param_groups):
            lr = base_lrs[gi] * cos
            if alternating_from is not None:
                # Alternate whole blocks: router blocks, then query-head blocks.
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
            [p for grp in opt.param_groups for p in grp["params"]], 1.0
        )
        opt.step()

        if (step + 1) % args.eval_every == 0 or step == args.joint_steps - 1:
            model.eval()
            snap = assess(model, routed, packed_held, corpus, device, amp, smap,
                          n_hops=n_hops, supervised=supervised)
            model.train()
            snap["step"] = step + 1
            snap["answer_loss"], snap["retrieval_loss"] = a_loss, r_loss
            history.append(snap)
            margins.append(snap["worst_binding_margin"])
            recalls.append(float(np.mean(snap["per_hop_retrieval"])))
            print(f"   step {step + 1:>5}/{args.joint_steps}  answer {a_loss:.4f}  "
                  f"retrieval {r_loss:.4f}  held-out retrieval {recalls[-1]:.1%}  "
                  f"worst margin {margins[-1]:+.4f}  "
                  f"sep {snap['separability']['separability_index']:.1%}  "
                  f"rank {snap['key_spread']['effective_rank']:.1f}")
            if alternating_from is None and oscillation_triggered(margins, recalls):
                alternating_from = step + 1
                print(f"   ** §9.7 alternation trigger fired at step {step + 1} — "
                      f"switching to router/query-head blocks of {args.alt_block}")

    model.eval()
    seconds = time.perf_counter() - t_start

    # -- final gates -----------------------------------------------------------------------
    print("\n== Gates (§9.4)")
    final = assess(model, routed, packed_held, corpus, device, amp, smap, n_hops=n_hops,
                      supervised=supervised)
    routed.freeze_keys()

    print(f"   {final['_router_str']}")
    print(f"   {final['_sep_str']}")

    stick_pack = train_3hop.select(np.arange(min(200, len(train_3hop.tokens))))
    sub = latent_substitution_test(
        model, routed, stick_pack, tok, device=device, amp_dtype=amp, seed=args.seed
    )
    e2e_routed, _ = evaluate_d3(
        model, routed, packed_held, tok, device=device, amp_dtype=amp
    )
    e2e_oracle, _ = evaluate_d3(
        model, oracle, packed_held, tok, device=device, amp_dtype=amp
    )

    per_hop = final["per_hop_retrieval"]
    gates = {
        "per_hop_retrieval": (min(per_hop), GATE_RETRIEVAL, min(per_hop) >= GATE_RETRIEVAL),
        "worst_binding_margin": (
            final["worst_binding_margin"], 0.0, final["worst_binding_margin"] > 0
        ),
        "substitution_stick_rate": (sub.stick_rate, GATE_STICK, sub.stick_rate <= GATE_STICK),
    }
    for name, (value, threshold, ok) in gates.items():
        print(f"   {name:<26} {value:+.4f}  (gate {threshold:g})  {'PASS' if ok else 'FAIL'}")
    print(f"   {'kernel damage via ORACLE':<26} {e2e_oracle.accuracy:.1%}  "
          f"(trigger < {TRIGGER_ORACLE_E2E:.0%})  "
          f"{'ok' if e2e_oracle.accuracy >= TRIGGER_ORACLE_E2E else 'TRIGGERED'}")
    print(f"   {'end-to-end via learned stack':<26} {e2e_routed.accuracy:.1%}  "
          f"(reported, NOT gated — ceilinged by the frozen kernel)")

    passed = all(ok for _, _, ok in gates.values())
    drift = e2e_oracle.accuracy < TRIGGER_ORACLE_E2E
    print(f"\n== Fork (a) joint fit: {'PASS' if passed else 'FAIL'}"
          f"{'  [QUERY-HEAD DRIFT TRIGGERED — §9.3(c) fallback applies]' if drift else ''}")

    # §9.6: retrieval high while the geometry says collapse is not a pass.
    if passed and (
        final["key_spread"]["effective_rank"] < 4 or final["worst_binding_margin"] < 1e-3
    ):
        print("   ** §9.6 MIS-SPECIFICATION: retrieval passed with a collapsed key space. "
              "Freeze the router and re-fit the query head from scratch before reporting this.")

    payload = {
        "label": args.label,
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "seconds": seconds,
        "corpus_fingerprint": corpus.fingerprint(),
        "config": vars(args),
        "kernel_params": model.n_params(),
        "key_params": table.n_params,
        "instrument_gates": {"learned_stack": ok_stub, "router_eval": True},
        "stage1_warm": [{k: v for k, v in w.items() if not k.startswith("_")}
                        for w in warm_history],
        "stage1_final_loss": warm_losses[-1] if warm_losses else None,
        "history": [{k: v for k, v in h.items() if not k.startswith("_")} for h in history],
        "final": {k: v for k, v in final.items() if not k.startswith("_")},
        "substitution": sub.to_dict(),
        "e2e_learned_stack": e2e_routed.to_dict(),
        "e2e_oracle": e2e_oracle.to_dict(),
        "gates": {k: {"value": v, "threshold": t, "pass": ok} for k, (v, t, ok) in gates.items()},
        "alternation_from_step": alternating_from,
        "query_head_drift_triggered": drift,
        "verdict": "pass" if passed else "fail",
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")

    # Every real run saves a checkpoint: router table, fine-tuned query head, RNG state.
    if args.checkpoint_dir:
        d = Path(args.checkpoint_dir)
        d.mkdir(parents=True, exist_ok=True)
        # The fine-tuned query head goes in its own file: the sidecar meta.json stringifies
        # whatever lands in `extra`, and a 64x576 tensor rendered as text is not provenance.
        head_path = d / f"query_head_{args.label}.pt"
        torch.save({"query_out.weight": model.query_out.weight.detach().cpu()}, head_path)
        save_checkpoint(
            d / f"router_{args.label}.pt", table,
            model_config=table.cfg, train_config=vars(args),
            extra={"query_head_file": head_path.name,
                   "corpus_fingerprint": corpus.fingerprint(),
                   "kernel_checkpoint": str(args.checkpoint), "verdict": payload["verdict"]},
        )
        print(f"wrote {d / f'router_{args.label}.pt'} and {head_path.name}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
