"""DIM-LIFTING experiment — both bills in one run (M2 §17.1 candidate 2, M3 §21).

    python scripts/run_m3_lift.py --checkpoint <k> --router <r> --query-head <h>

`f` lifts dim-64 content into an internal key space of dimension `d` before quantisation. The lift
is a fixed **isometry** (orthonormal columns, `L^T L = I_64`), so it is information-preserving and
exactly invertible on its image: quantise in `d`-space, project back with `L^T`, and the component
of the quantisation error orthogonal to the 64-dim image **vanishes**. That is the mechanism the
address-cost bill is hoping for.

Per `d`, both bills, measured together so a split cannot be missed:

  (a) **address cost** — min bits/entity for >= 99% never-supervised addressability at 2M;
  (b) **crowding** — clean-code shortlist recovery@1 and `m` for 99%.

Coherence is logged against `sqrt(2 ln N / d)` at each `d`, because that prediction is the whole
argument for expecting double duty.

**Accounting unchanged**: per-entity cost stays `n_stages * log2(K)` bits at every `d`. Shared
codebooks grow as `n_stages * K * d` and are reported separately — they amortise, but they are not
free (M3 §10.5).

**Registered uncertainty: lifting harvests geometry, not information.** An isometry preserves
every angle, so the *intrinsic* dimension of the content is still 64 whatever `d` is. If bill (b)
is set by the intrinsic geometry rather than the ambient one, lifting cannot touch it — and the
mis-specification clause fires: **if the two bills respond differently, double duty is falsified in
the direction of the split, and they are billed separately from then on.**
"""

from __future__ import annotations

import argparse
import json
import math
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
from fka.kernel.checkpoint import load_checkpoint  # noqa: E402
from fka.kernel.gpu_lock import gpu_lock  # noqa: E402
from fka.kernel.latent_episodes import d3_tokenizer, one_hop_episode, pack  # noqa: E402
from fka.kernel.latent_kernel import LatentKernelConfig, LatentReasoningKernel  # noqa: E402
from fka.kernel.latent_memory import LatentCodebook, OracleLatentMemory  # noqa: E402
from fka.kernel.seeding import seed_everything  # noqa: E402
from fka.router.content_keys import ContentKeyConfig, ContentKeyTable  # noqa: E402
from fka.router.inversion import (  # noqa: E402
    EntityInverter,
    InverterConfig,
    entity_recovery_by_inversion,
    fit_inverter,
)
from fka.store.s1_factorized import S1FactorizedStore  # noqa: E402
from scripts.run_d3_kernel import build_data  # noqa: E402
from scripts.run_fork_a_joint import RELATIONS, harvest  # noqa: E402
from scripts.run_fork_c import address_holdout, supervised_addresses  # noqa: E402
from scripts.run_m3_curve import BASE, chunked_argmax  # noqa: E402

DIMS = [64, 128, 256]
BIT_LEVELS = [40, 44, 48, 52, 56]


def isometry(src: int, dst: int, seed: int, device) -> torch.Tensor:
    """`(src, dst)` with orthonormal columns-of-the-transpose: `L @ L.T == I_src`."""
    if dst == src:
        return torch.eye(src, device=device)
    g = torch.Generator().manual_seed(seed)
    q = torch.linalg.qr(torch.randn(dst, src, generator=g)).Q  # (dst, src), orthonormal columns
    return q.T.contiguous().to(device)  # (src, dst); L @ L.T = I_src


def coherence(codes: torch.Tensor, sample: int = 4096, seed: int = 0) -> float:
    """Max off-diagonal cosine over a sample — the quantity `sqrt(2 ln N / d)` predicts."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(codes.shape[0], generator=g)[:sample].to(codes.device)
    c = F.normalize(codes[idx].float(), dim=-1)
    s = c @ c.T
    s.fill_diagonal_(-2.0)
    return float(s.max())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--router", required=True)
    p.add_argument("--query-head", default=None)
    p.add_argument("--n", type=int, default=2_000_000)
    p.add_argument("--dims", type=int, nargs="+", default=DIMS)
    p.add_argument("--n-eval", type=int, default=400)
    p.add_argument("--n-one-hop", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--chunk", type=int, default=65_536,
                   help="store-fit chunk; must shrink as d and K grow (memory, not time)")
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
    dim = meta["latent_dim"]
    corpus = generate_corpus(CorpusConfig(n_entities=2000, seed=0, n_coworkers=1))
    tok, bs = d3_tokenizer(), meta["block_size"]
    split = entity_split(corpus, fraction=0.2, seed=0)
    cb = LatentCodebook.build(corpus, dim=dim, seed=0)
    oracle = OracleLatentMemory(corpus, cb).to(device)
    model = LatentReasoningKernel(LatentKernelConfig(**mcfg)).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    if args.query_head:
        head = torch.load(args.query_head, map_location=device, weights_only=False)
        model.query_out.weight.data.copy_(head["query_out.weight"].to(device))
    amp = torch.bfloat16 if device.type == "cuda" else None

    raw, held, _ = build_data(corpus, tok, split, [2, 3], oracle.fact_index, bs, 0)
    groups, _ = address_holdout(raw, split.heldout, corpus.n_entities)
    sup = supervised_addresses(groups)
    ph = pack(held[: args.n_eval], tok, bs, oracle.fact_index)
    rng = np.random.default_rng(args.seed)
    attrs = [r for r in RELATIONS if r != "works_with"]
    p1 = pack([one_hop_episode(corpus, int(e), attrs[int(rng.integers(0, len(attrs)))])
               for e in rng.permutation(split.heldout)[: args.n_one_hop]], tok, bs,
              oracle.fact_index)
    n0, n_rel, N = corpus.n_entities, len(RELATIONS), args.n
    real = F.normalize(cb.entity, dim=-1).to(device)

    qs, ts = [], []
    for pk in (ph, p1):
        q, t, _ = harvest(model, oracle, pk, device, amp)
        qs.append(q)
        ts.append(t.cpu().numpy())
    q_all, t_all = torch.cat(qs), np.concatenate(ts)
    keep = ~np.isin(t_all, sup)
    q_probe = q_all[torch.from_numpy(keep).to(device)]
    rel_p, ent_p = t_all[keep] // n0, t_all[keep] % n0

    g = torch.Generator(device="cpu").manual_seed(N)
    codes = torch.cat([real,
                       F.normalize(torch.randn(N - n0, dim, generator=g), dim=-1).to(device)])
    print(f"== dim-lifting at N={N:,}   ({len(ent_p)} never-supervised probes)")
    print(f"   coherence of the 64-dim content: measured {coherence(codes):.4f}, "
          f"sqrt(2 ln N / 64) = {math.sqrt(2 * math.log(N) / 64):.4f}")

    seed_everything(args.seed)
    inv = EntityInverter(InverterConfig(latent_dim=dim, hidden=512, n_layers=2)).to(device)
    q_tr, t_tr, _ = harvest(model, oracle, groups[-1], device, amp)
    fit_inverter(inv, q_tr, real[t_tr.cpu().numpy() % n0], steps=4000, lr=3e-3, log_every=10**9)

    results = []
    for d in args.dims:
        L = isometry(dim, d, args.seed, device)  # (64, d)
        lifted = codes @ L  # exact isometry: angles and norms preserved
        coh = coherence(lifted)
        pred = math.sqrt(2 * math.log(N) / d)
        print(f"\n-- d = {d}   coherence measured {coh:.4f}   "
              f"sqrt(2 ln N / d) predicts {pred:.4f}")

        # (b) CROWDING: clean-code shortlist, over the lifted-then-projected clean codes.
        clean_back = (lifted @ L.T) if d != dim else codes
        rec = entity_recovery_by_inversion(inv, q_probe, ent_p, clean_back, n_rel,
                                           ms=(1, 8, 64, 256, 1024))
        print(f"   (b) crowding  clean-code recovery@1 {rec.at_1:>6.1%}   "
              f"m for 99% = {rec.m_for(0.99)}")

        # (a) ADDRESS COST: quantise in d-space, project back with L^T, feed the trained f.
        best = None
        for bits in BIT_LEVELS:
            cfg = replace(BASE, latent_dim=d, seed=args.seed, chunk=args.chunk,
                          codebook_size=int(round(2 ** (bits / BASE.n_stages))))
            store = S1FactorizedStore(cfg)
            store.write(lifted)
            recon_d = store.reconstruct(torch.arange(N, device=device))
            recon = recon_d @ L.T  # the out-of-image error component vanishes here
            table = ContentKeyTable(
                ContentKeyConfig(n_relations=n_rel, latent_dim=dim, key_dim=dim, comp_dim=128,
                                 mode="bilinear"), recon).to(device)
            load_checkpoint(args.router, table)
            table.codes = recon
            got = chunked_argmax(q_probe, table, N, n_rel, device)
            acc = float((got == torch.from_numpy(rel_p * N + ent_p).to(device)).float().mean())
            err = float(((recon - codes).norm(dim=-1) / codes.norm(dim=-1)).mean())
            print(f"       {bits:>3} bits/entity  addressability {acc:>6.1%}  "
                  f"recon err {err:.3f}  shared {cfg.n_stages * cfg.codebook_size * d:,} params")
            del store, recon_d, recon, table, got
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if acc >= 0.99:
                best = bits
                break
        shown = best if best else "> " + str(BIT_LEVELS[-1])
        print(f"   (a) address   min bits/entity for >= 99%: {shown}")
        results.append({"d": d, "coherence": coh, "coherence_pred": pred,
                        "crowding_recovery_at_1": rec.at_1, "crowding_m99": rec.m_for(0.99),
                        "min_bits": best})
        del lifted, clean_back
        if device.type == "cuda":
            torch.cuda.empty_cache()

    base = results[0]
    print("\n== BOTH BILLS")
    print(f"   {'d':>5} {'coherence':>10} {'min bits':>9} {'crowding@1':>11}")
    for r in results:
        print(f"   {r['d']:>5} {r['coherence']:>10.4f} "
              f"{str(r['min_bits']):>9} {r['crowding_recovery_at_1']:>11.1%}")
    addr_moved = any(r["min_bits"] is not None and base["min_bits"] is not None
                     and r["min_bits"] < base["min_bits"] for r in results[1:])
    crowd_moved = any(r["crowding_recovery_at_1"] > base["crowding_recovery_at_1"] + 0.02
                      for r in results[1:])
    print(f"\n   address bill responds to lifting: {addr_moved}")
    print(f"   crowding bill responds to lifting: {crowd_moved}")
    if addr_moved != crowd_moved:
        print("   ** MIS-SPECIFICATION CLAUSE FIRED: the two bills respond differently.")
        print("      DOUBLE DUTY IS FALSIFIED in the direction of the split; bill separately.")

    payload = {"device": str(device), "seconds": time.perf_counter() - t0, "config": vars(args),
               "results": results, "address_bill_responds": addr_moved,
               "crowding_bill_responds": crowd_moved,
               "double_duty_falsified": addr_moved != crowd_moved}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    print(f"\n== done in {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
