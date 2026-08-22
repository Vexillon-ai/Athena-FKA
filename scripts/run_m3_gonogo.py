"""M3 GO/NO-GO under the pre-registered regime (M3 §15) with the redesigned value path (§16).

    python scripts/run_m3_gonogo.py --checkpoint <kernel> --router <router.pt> --query-head <h>

Three §2 conditions at ONE operating point: capacity (bits/bit), addressability (never-supervised
retrieval ≥ 99%), edit locality (< 1%).

**Both figures always reported** (§15.1): the **amortised** bits/bit at the load knee — the gate
number — and the **marginal** per-fact bits — the scaling limit. Neither reading stands alone.

The load curve is grounded rather than projected: marginal cost is *measured* at increasing `K` by
writing synthetic facts, and only the amortisation arithmetic is then applied to the measured
quantities.
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
from fka.eval.accounting import DENSE_BASELINE, GATE_PASS, GATE_TARGET, StorageAccount  # noqa: E402
from fka.kernel.checkpoint import load_checkpoint  # noqa: E402
from fka.kernel.gpu_lock import gpu_lock  # noqa: E402
from fka.kernel.latent_episodes import d3_tokenizer, one_hop_episode, pack  # noqa: E402
from fka.kernel.latent_kernel import LatentKernelConfig, LatentReasoningKernel  # noqa: E402
from fka.kernel.latent_memory import LatentCodebook, OracleLatentMemory  # noqa: E402
from fka.kernel.seeding import seed_everything  # noqa: E402
from fka.router.routed_memory import RoutedLatentMemory  # noqa: E402
from fka.store.base import IdentityStore  # noqa: E402
from fka.store.pointer_values import PointerValueStore  # noqa: E402
from fka.store.s1_factorized import S1Config, S1FactorizedStore  # noqa: E402
from scripts.run_d3_kernel import build_data  # noqa: E402
from scripts.run_fork_a_joint import RELATIONS, harvest  # noqa: E402
from scripts.run_fork_c import address_holdout, supervised_addresses  # noqa: E402
from scripts.run_s1_first_point import build_table  # noqa: E402

#: The addressing knee located in §14.1.
KEY_CFG = S1Config(n_stages=4, codebook_size=256, residual_dim=0, residual_bits=8)
LOAD_SCALES = [1, 10, 100, 1000]


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
    codebook = LatentCodebook.build(corpus, dim=meta["latent_dim"], seed=meta["codebook_seed"])
    oracle = OracleLatentMemory(corpus, codebook).to(device)
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
    n_e, dim = corpus.n_entities, meta["latent_dim"]
    codes = F.normalize(codebook.entity, dim=-1).to(device)
    values = oracle.values_matrix.to(device)

    print(f"== M3 GO/NO-GO — redesigned value path, regime per §15, on {device}")
    print(f"   corpus {corpus.n_facts:,} facts, {corpus.total_bits:,.0f} knowledge bits")

    # -- the two stores --------------------------------------------------------------------
    key_store = S1FactorizedStore(replace(KEY_CFG, latent_dim=dim, seed=args.seed))
    key_store.write(codes)
    tables = {"works_with": codes.cpu(), **{r: codebook.value[r].cpu() for r in attrs}}
    val_store = PointerValueStore(tables, latent_dim=dim,
                                  shared_tables_are_free=frozenset({"works_with"}))
    val_store.write(values)
    print(f"   value store: {val_store}")
    max_err = float(val_store.recon_error(torch.arange(len(values), device=device)).max())
    print(f"   value reconstruction error (max) {max_err:.2e}")

    # -- condition 2: addressability, per class and per depth -------------------------------
    def addressability(store):
        routed = RoutedLatentMemory(
            oracle, build_table(meta, store, n_e, device, args.router), RELATIONS
        ).to(device).freeze_keys()
        out = {}
        qs, ts, ds, rs = [], [], [], []
        for packed in (packed3, packed1):
            q, t, h = harvest(model, routed, packed, device, amp)
            qs.append(q)
            ts.append(t.cpu().numpy())
            hh = h.cpu().numpy()
            ds.append(np.where(hh == 0, "direct", "composed"))
            rs.append(t.cpu().numpy() // n_e)
        q = torch.cat(qs)
        t, d, r = np.concatenate(ts), np.concatenate(ds), np.concatenate(rs)
        keep = ~np.isin(t, supervised)
        ok = (routed.retrieved_index(q).cpu().numpy() == t)[keep]
        d, r = d[keep], r[keep]
        out["never_supervised"] = float(ok.mean())
        out["by_depth"] = {k: float(ok[d == k].mean()) for k in ("direct", "composed")}
        out["by_relation"] = {
            RELATIONS[i]: float(ok[r == i].mean()) for i in sorted(set(r.tolist()))
        }
        out["n"] = int(ok.size)
        return out

    print("\n-- gate: IdentityStore keys (no compression) must reproduce fork (c)")
    ident = IdentityStore()
    ident.write(codes)
    gate = addressability(ident)
    print(f"   never-supervised {gate['never_supervised']:.1%}  {gate['by_depth']}")
    if gate["never_supervised"] < 0.999:
        raise SystemExit("GATE FAILED — no go/no-go number is admissible")
    print("   -> OK")

    addr = addressability(key_store)
    print(f"\n-- condition 2 ADDRESSABILITY at the key knee ({KEY_CFG.bits_per_slot:.0f} b/entity)")
    print(f"   never-supervised {addr['never_supervised']:.1%} (n={addr['n']})   "
          f"gate >= 99%   {'PASS' if addr['never_supervised'] >= 0.99 else 'FAIL'}")
    print(f"   by depth   {({k: f'{v:.1%}' for k, v in addr['by_depth'].items()})}")
    print(f"   by relation {({k: f'{v:.1%}' for k, v in addr['by_relation'].items()})}")

    # -- condition 3: edit locality ---------------------------------------------------------
    ids = torch.arange(len(values), device=device)
    before = val_store.reconstruct(ids).clone()
    new_ids = val_store.write(values[:64])
    declared = set(val_store.declared_invalidation(new_ids).cpu().tolist())
    untouched = torch.tensor([i for i in range(len(values)) if i not in declared], device=device)
    drift = (val_store.reconstruct(untouched) - before[untouched]).norm(dim=-1)
    drift = drift / before[untouched].norm(dim=-1).clamp(min=1e-9)
    locality = float(drift.max())
    print(f"\n-- condition 3 EDIT LOCALITY  max drift on undeclared slots {locality:.2e}   "
          f"gate < 1%   {'PASS' if locality < 0.01 else 'FAIL'}")

    # -- condition 1: capacity, both figures, over the load curve ---------------------------
    key_cm, val_cm = key_store.cost_model(), val_store.cost_model()
    table_params = sum(p.numel() for p in build_table(
        meta, key_store, n_e, device, args.router).parameters())
    #: Per-fact costs that scale with the world.
    per_fact_value = val_cm["per_fact_storage_bits"]
    per_fact_key = key_cm["per_fact_storage_bits"] / (corpus.n_facts / n_e)  # entity cost / facts
    per_fact = per_fact_value + per_fact_key
    #: Fixed machinery that does NOT scale: value tables, key codebooks, key encoder f.
    fixed_params = val_cm["shared_parameters"] + key_cm["shared_parameters"] + table_params
    bits_per_fact_knowledge = corpus.total_bits / corpus.n_facts

    print("\n-- condition 1 CAPACITY, both figures (§15.1)")
    print(f"   MARGINAL per-fact bits (scaling limit): {per_fact:.2f} "
          f"(value {per_fact_value:.2f} + key {per_fact_key:.2f})")
    print(f"   marginal bits/bit = {bits_per_fact_knowledge / per_fact:.4f}  "
          f"<- what one more fact costs; cannot be improved by scaling")
    print(f"   fixed machinery: {fixed_params:,} params = {fixed_params * 8:,} bits (int8)")

    # Measured, not assumed: marginal cost must stay flat as K grows.
    print("\n   marginal cost measured at increasing K (synthetic facts drawn from the tables):")
    g = torch.Generator().manual_seed(args.seed)
    load_rows = []
    for scale in LOAD_SCALES:
        n = corpus.n_facts * scale
        probe = PointerValueStore(tables, latent_dim=dim,
                                  shared_tables_are_free=frozenset({"works_with"}))
        idx = torch.randint(0, len(values), (min(n, 200_000),), generator=g)
        probe.write(values[idx.to(device)])
        marg = probe.cost_model()["per_fact_storage_bits"] + per_fact_key
        know = bits_per_fact_knowledge * n
        acc = StorageAccount(n_facts=1, per_fact_bits=marg * n, shared_params=fixed_params,
                             knowledge_bits=know)
        load_rows.append({"n_facts": n, "marginal_bits": marg,
                          "amortised_bits_per_bit": acc.headline, "verdict": acc.verdict})
        print(f"      K={n:>10,}  marginal {marg:>6.2f} b/fact   amortised "
              f"{acc.headline:.4f} bits/bit   {acc.verdict}")

    # Where the amortised figure crosses the gates — arithmetic on measured quantities.
    def crossing(target: float) -> float | None:
        denom = bits_per_fact_knowledge - target * per_fact
        return None if denom <= 0 else target * fixed_params * 8 / denom

    n_pass, n_target = crossing(GATE_PASS), crossing(GATE_TARGET)
    print("\n   crossings (arithmetic on the measured marginal and fixed costs):")
    def fmt(v):
        return "never (marginal cost is below the gate)" if v is None else f"{v:,.0f} facts"

    print(f"      PASS  ({GATE_PASS}) at K = {fmt(n_pass)}")
    print(f"      TARGET({GATE_TARGET}) at K = {fmt(n_target)}")

    knee_row = next((r for r in load_rows if r["amortised_bits_per_bit"] >= GATE_PASS), None)
    print("\n== §2 GO/NO-GO at the load knee")
    c1 = knee_row is not None
    c2 = addr["never_supervised"] >= 0.99
    c3 = locality < 0.01
    where = "" if not c1 else (
        f" at K={knee_row['n_facts']:,} ({knee_row['amortised_bits_per_bit']:.4f})"
    )
    print(f"   1 capacity      "
          f"{'PASS' if c1 else 'not reached on the measured ladder'}{where}")
    print(f"   2 addressability {'PASS' if c2 else 'FAIL'}  ({addr['never_supervised']:.1%})")
    print(f"   3 edit locality  {'PASS' if c3 else 'FAIL'}  ({locality:.2e})")
    print(f"\n   dense baseline {DENSE_BASELINE}; marginal limit "
          f"{bits_per_fact_knowledge / per_fact:.4f}")

    payload = {"checkpoint": str(args.checkpoint), "device": str(device),
               "seconds": time.perf_counter() - t0, "config": vars(args),
               "corpus": {"n_facts": corpus.n_facts, "total_bits": corpus.total_bits},
               "gate_identity": gate, "addressability": addr, "edit_locality": locality,
               "marginal_bits_per_fact": per_fact,
               "marginal_bits_per_bit": bits_per_fact_knowledge / per_fact,
               "fixed_params": fixed_params, "load_curve": load_rows,
               "crossing_pass_facts": n_pass, "crossing_target_facts": n_target,
               "conditions": {"capacity": c1, "addressability": c2, "edit_locality": c3}}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    print(f"\n== done in {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
