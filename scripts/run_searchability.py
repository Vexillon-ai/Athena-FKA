"""§10.2: two-stage entity-first search over fork (c)'s learned keys, with REAL kernel queries.

    python scripts/run_searchability.py --checkpoint <kernel> --router <router.pt> --query-head <h>

Pre-registered in `docs/decision_records/M2_router.md` §10.2 before running. The feasibility probe
(`probe_inversion_feasibility.py`) established that `h : query -> entity_code` is learnable from the
oracle's own addresses; this measures it where it has to work — on the queries the kernel actually
emits through the learned router, scored on the **never-supervised** class.

Four things are measured, in cost order:

1. **Gate**: a gold-stub `h` must score exactly 100% recovery at m=1 (one-gate-per-eval-path).
2. **Headline**: entity recovery on the never-supervised class, and the candidate recall@m curve.
3. **Control (§10.2.4)**: recovery on entity codes drawn fresh from the codebook distribution and
   never in the corpus at all — if it holds there, `h` is a function of code geometry rather than of
   the training set. A **sample-complexity sweep** accompanies it, because the unit test showed the
   inverter memorises when given too few training entities (train 100% / held-out 7.5% at 160).
4. **Scaling**: the codebook padded with synthetic distractors to 10^6. Queries and targets stay
   real; only the haystack grows, so this measures shortlist degradation with `N` directly and needs
   no synthetic queries.
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

from fka.data.corpus_gen import CorpusConfig, generate_corpus  # noqa: E402
from fka.data.hardening import entity_split  # noqa: E402
from fka.kernel.checkpoint import load_checkpoint  # noqa: E402
from fka.kernel.gpu_lock import gpu_lock  # noqa: E402
from fka.kernel.latent_episodes import d3_tokenizer, pack  # noqa: E402
from fka.kernel.latent_kernel import LatentKernelConfig, LatentReasoningKernel  # noqa: E402
from fka.kernel.latent_memory import LatentCodebook, OracleLatentMemory  # noqa: E402
from fka.kernel.seeding import seed_everything  # noqa: E402
from fka.router.content_keys import ContentKeyConfig, ContentKeyTable  # noqa: E402
from fka.router.inversion import (  # noqa: E402
    EntityInverter,
    GoldStubInverter,
    InverterConfig,
    entity_recovery_by_inversion,
    fit_inverter,
)
from fka.router.routed_memory import RoutedLatentMemory  # noqa: E402
from scripts.run_d3_kernel import build_data  # noqa: E402
from scripts.run_fork_a_joint import RELATIONS, harvest  # noqa: E402
from scripts.run_fork_c import address_holdout, supervised_addresses  # noqa: E402

GATE_RECOVERY = 0.99


class _Fixed(torch.nn.Module):
    def __init__(self, v):
        super().__init__()
        self.v = v

    def forward(self, x):
        return self.v[: len(x)]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--router", required=True)
    p.add_argument("--query-head", default=None)
    p.add_argument("--n-entities", type=int, default=2000)
    p.add_argument("--n-eval", type=int, default=400)
    p.add_argument("--n-train-harvest", type=int, default=1600)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--scale-to", type=int, nargs="+", default=[2000, 10000, 100000, 1000000])
    p.add_argument("--sample-complexity", type=int, nargs="+", default=[100, 400, 800, 1600])
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
    if args.query_head:
        head = torch.load(args.query_head, map_location=device, weights_only=False)
        model.query_out.weight.data.copy_(head["query_out.weight"].to(device))
    amp = torch.bfloat16 if device.type == "cuda" else None

    table = ContentKeyTable(
        ContentKeyConfig(n_relations=len(RELATIONS), latent_dim=meta["latent_dim"],
                         key_dim=meta["latent_dim"], comp_dim=128, mode="bilinear"),
        oracle.codebook.entity,
    ).to(device)
    load_checkpoint(args.router, table)
    routed = RoutedLatentMemory(oracle, table, RELATIONS).to(device).freeze_keys()

    raw_groups, heldout_eps, _ = build_data(
        corpus, tok, split, [2, 3], oracle.fact_index, block_size, meta["codebook_seed"]
    )
    groups, _ = address_holdout(raw_groups, split.heldout, corpus.n_entities)
    supervised = supervised_addresses(groups)
    packed_held = pack(heldout_eps[: args.n_eval], tok, block_size, oracle.fact_index)
    n_e, n_r = corpus.n_entities, len(RELATIONS)
    codes = F.normalize(oracle.codebook.entity, dim=-1)

    print(f"== §10.2 two-stage entity-first search on {device}")

    # -- harvest REAL kernel queries through the LEARNED stack -----------------------------
    # From EVERY training group, not just the 3-hop one. Harvesting 3-hop alone gave `h` 3,200
    # `works_with` examples against ~533 per attribute relation, and the first run's whole residual
    # was that imbalance: 100% recovery on `works_with` (549/549) against 6-22% on the attributes,
    # with the sample-complexity sweep showing ~800 entities per relation are needed. Inverting
    # `e (x) r` is a separate map per relation code, so relation coverage is the thing to balance —
    # a defect in the probe, fixed here, not a property of the key geometry.
    per_group = max(1, args.n_train_harvest // len(groups))
    q_parts, t_parts = [], []
    for grp in groups:
        sel = np.arange(min(per_group, len(grp.tokens)))
        if not len(sel):
            continue
        qg, tg, _ = harvest(model, routed, grp.select(sel), device, amp)
        q_parts.append(qg)
        t_parts.append(tg)
    q_tr, t_tr = torch.cat(q_parts), torch.cat(t_parts)
    q_te, t_te, _ = harvest(model, routed, packed_held, device, amp)
    e_tr, e_te = t_tr.cpu().numpy() % n_e, t_te.cpu().numpy() % n_e

    never = ~np.isin(t_te.cpu().numpy(), supervised)
    q_ns, e_ns = q_te[torch.from_numpy(never).to(device)], e_te[never]
    print(f"   {len(q_tr):,} training queries ({len(set(e_tr.tolist())):,} distinct entities), "
          f"{len(q_ns):,} never-supervised eval queries "
          f"({len(set(e_ns.tolist())):,} distinct entities)")

    # -- gate ------------------------------------------------------------------------------
    stub = GoldStubInverter(codes, torch.from_numpy(e_ns).to(device)).to(device)
    g = entity_recovery_by_inversion(stub, q_ns, e_ns, codes, n_r)
    print(f"\n-- instrument gate: gold-stub h   recovery@1 {g.at_1:.1%}  "
          f"{'OK' if g.at_1 == 1.0 else 'FAILED'}")
    if g.at_1 != 1.0:
        raise SystemExit("inversion eval path gate FAILED; no h number is admissible")

    results = {}

    # -- headline --------------------------------------------------------------------------
    print(f"\n-- fitting h on {len(q_tr):,} SUPERVISED-entity queries")
    seed_everything(args.seed)
    inv = EntityInverter(
        InverterConfig(latent_dim=meta["latent_dim"], hidden=args.hidden, n_layers=2)
    ).to(device)
    fit_inverter(inv, q_tr, codes[torch.from_numpy(e_tr).to(device)], steps=args.steps,
                 lr=args.lr, log_every=max(1, args.steps // 2))

    on_train = entity_recovery_by_inversion(inv, q_tr, e_tr, codes, n_r)
    headline = entity_recovery_by_inversion(inv, q_ns, e_ns, codes, n_r)
    print(f"\n   train-set recovery      {on_train.at_1:.1%}")
    print(f"   NEVER-SUPERVISED        {headline}")
    results["headline"] = headline.to_dict()
    results["on_train"] = on_train.to_dict()

    # -- control: fresh codes never in the corpus ------------------------------------------
    gg = torch.Generator().manual_seed(args.seed + 1)
    n_fresh = 400
    fresh = F.normalize(
        torch.randn(n_fresh, meta["latent_dim"], generator=gg), dim=-1
    ).to(device)
    rel_codes = torch.stack([oracle.codebook.relation[r] for r in RELATIONS]).to(device)
    # Draw the control's relations from the EVAL mix, not uniformly. The first version drew them
    # uniformly and read 36.2%, which looked like memorisation and was arithmetic: recovery is
    # relation-dependent, so a control with a different relation mix measures the mix.
    fresh_r = np.random.default_rng(args.seed).choice(
        (t_te.cpu().numpy()[never] // n_e), size=n_fresh, replace=True
    )
    fresh_q = F.normalize(fresh * rel_codes[torch.from_numpy(fresh_r).to(device)], dim=-1)
    aug = torch.cat([codes, fresh], dim=0)
    ctrl = entity_recovery_by_inversion(
        inv, fresh_q, np.arange(n_e, n_e + n_fresh), aug, n_r
    )
    print(f"\n   CONTROL, codes never in the corpus (haystack {len(aug):,}): "
          f"recovery@1 {ctrl.at_1:.1%}")
    results["fresh_code_control"] = ctrl.to_dict()

    # -- sample complexity -----------------------------------------------------------------
    print("\n-- sample complexity: how many training entities does h need?")
    sc = {}
    uniq_tr = np.array(sorted(set(e_tr.tolist())))
    for k in args.sample_complexity:
        if k > len(uniq_tr):
            continue
        keep = set(uniq_tr[:k].tolist())
        mask = np.array([e in keep for e in e_tr])
        seed_everything(args.seed)
        h_k = EntityInverter(
            InverterConfig(latent_dim=meta["latent_dim"], hidden=args.hidden, n_layers=2)
        ).to(device)
        fit_inverter(h_k, q_tr[torch.from_numpy(mask).to(device)],
                     codes[torch.from_numpy(e_tr[mask]).to(device)],
                     steps=args.steps, lr=args.lr, log_every=10**9)
        r = entity_recovery_by_inversion(h_k, q_ns, e_ns, codes, n_r)
        sc[k] = r.to_dict()
        print(f"   {k:>5} training entities ({int(mask.sum()):>5} queries)  "
              f"never-supervised recovery@1 {r.at_1:.1%}   m99={r.m_for(0.99)}")
    results["sample_complexity"] = sc

    # -- scaling: pad the haystack with synthetic distractors ------------------------------
    print("\n-- scaling: real queries and targets, synthetic distractors padding the codebook")
    scaling = {}
    gg2 = torch.Generator().manual_seed(args.seed + 2)
    for n in args.scale_to:
        if n < n_e:
            continue
        pad = n - n_e
        book = codes if pad == 0 else torch.cat([
            codes,
            F.normalize(torch.randn(pad, meta["latent_dim"], generator=gg2), dim=-1).to(device),
        ])
        r = entity_recovery_by_inversion(
            inv, q_ns, e_ns, book, n_r, ms=(1, 2, 4, 8, 16, 32, 64, 128, 256, 1024)
        )
        scaling[n] = r.to_dict()
        m99 = r.m_for(0.99)
        cost = (
            f"   ({m99 * n_r} of {n * n_r:,} slots = {m99 * n_r / (n * n_r):.4%})" if m99 else ""
        )
        print(f"   N={n:>9,}  recovery@1 {r.at_1:.1%}   m for 99% = "
              f"{m99 if m99 is not None else 'NOT REACHED'}{cost}")
        del book
    results["scaling"] = scaling

    passed = headline.at_1 >= GATE_RECOVERY
    print(f"\n== §10.2 gate: never-supervised entity recovery {headline.at_1:.1%} "
          f"(gate {GATE_RECOVERY:.0%}) -> {'PASS' if passed else 'FAIL'}")
    print("   " + ("two-stage search SOLVED without touching the learned geometry"
                   if passed else "the separability regulariser branch applies (§10.2.2)"))

    payload = {
        "checkpoint": str(args.checkpoint), "router": str(args.router),
        "device": str(device), "seconds": time.perf_counter() - t0,
        "corpus_fingerprint": corpus.fingerprint(), "config": vars(args),
        "inverter_params": inv.n_params, "gate_passed": bool(passed), "results": results,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
