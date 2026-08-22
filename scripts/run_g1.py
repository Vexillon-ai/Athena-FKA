"""G1 evaluation against M4 §7's registered metric. Gates first, then the rung.

    python scripts/run_g1.py --checkpoint <k> --router <r> --query-head <h>

Capability is tested at 500K, which is affordable; the break-even **verdict is priced at 2M**, where
the budget lives. Those are different `N`, and M4 §7.2's scaling asymmetry makes saying which one
mandatory: `P_max` grows linearly with `n_facts`, so "too big" is meaningless without its `N`.
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
from fka.kernel.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from fka.kernel.gpu_lock import gpu_lock  # noqa: E402
from fka.kernel.latent_episodes import d3_tokenizer, one_hop_episode, pack  # noqa: E402
from fka.kernel.latent_kernel import LatentKernelConfig, LatentReasoningKernel  # noqa: E402
from fka.kernel.latent_memory import LatentCodebook, OracleLatentMemory  # noqa: E402
from fka.kernel.seeding import seed_everything  # noqa: E402
from fka.retriever.g1 import G1Config, G1Denoiser, IdentityDenoiser, fit_g1  # noqa: E402
from fka.router.content_keys import ContentKeyConfig, ContentKeyTable  # noqa: E402
from fka.store.s1_factorized import S1FactorizedStore  # noqa: E402
from scripts.run_d3_kernel import build_data  # noqa: E402
from scripts.run_fork_a_joint import RELATIONS, harvest  # noqa: E402
from scripts.run_fork_c import address_holdout, supervised_addresses  # noqa: E402
from scripts.run_m3_curve import BASE, chunked_argmax  # noqa: E402

RUNGS = [40, 44, 48]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--router", required=True)
    p.add_argument("--query-head", default=None)
    p.add_argument("--n", type=int, default=500_000)
    p.add_argument("--train-bits", type=int, default=44)
    p.add_argument("--hidden", type=int, default=768)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--n-eval", type=int, default=400)
    p.add_argument("--n-one-hop", type=int, default=600)
    p.add_argument("--chunk", type=int, default=16384)
    p.add_argument("--probe", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None)
    p.add_argument("--checkpoint-dir", default=None)
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

    qs, ts, ds = [], [], []
    for pk in (ph, p1):
        q, t, h = harvest(model, oracle, pk, device, amp)
        qs.append(q)
        ts.append(t.cpu().numpy())
        ds.append(np.where(h.cpu().numpy() == 0, "direct", "composed"))
    q_all, t_all, d_all = torch.cat(qs), np.concatenate(ts), np.concatenate(ds)
    keep = ~np.isin(t_all, sup)
    q_probe, d_probe = q_all[torch.from_numpy(keep).to(device)], d_all[keep]
    rel_p, ent_p = t_all[keep] // n0, t_all[keep] % n0

    g = torch.Generator(device="cpu").manual_seed(N)
    codes = torch.cat([real, F.normalize(torch.randn(N - n0, dim, generator=g), dim=-1).to(device)])
    print(f"== G1 at N={N:,}   ({len(ent_p)} never-supervised probes)")

    # Reconstructions are built ONCE per rung and reused by every arm.
    #
    # They must be: the chunked Lloyd update uses `index_add_`, which is atomic and therefore
    # run-to-run non-deterministic in float. Re-fitting the store per arm made the no-denoiser and
    # identity ladders differ by ~0.1 points from that alone, so an exact-equality gate could never
    # pass and — worse — the difference would have been attributed to the denoiser. Fitting once
    # removes the noise from the comparison; the non-determinism itself is recorded as a defect.
    def store_at(bits):
        cfg = replace(BASE, latent_dim=dim, seed=args.seed, chunk=args.chunk,
                      codebook_size=int(round(2 ** (bits / BASE.n_stages))))
        s = S1FactorizedStore(cfg)
        s.write(codes)
        return s

    print("\n-- building reconstructions once per rung (shared by every arm)")
    recons = {}
    for bits in RUNGS:
        st = store_at(bits)
        recons[bits] = st.reconstruct(torch.arange(N, device=device)).clone()
        del st
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def ladder(denoiser, steps, label):
        out = {}
        for bits in RUNGS:
            recon = recons[bits]
            with torch.no_grad():
                # Normalisation belongs to the SHARED path, not to the denoiser. The verified-red
                # null caught this: an identity denoiser that normalised while the no-denoiser
                # baseline did not shifted the ladder by up to 1.3 points, which would have been
                # scored as G1's effect. Both arms normalise; G1's gain is refinement alone.
                cleaned = (
                    denoiser(recon, steps=steps) if denoiser is not None
                    else F.normalize(recon, dim=-1)
                )
            tb = ContentKeyTable(
                ContentKeyConfig(n_relations=n_rel, latent_dim=dim, key_dim=dim, comp_dim=128,
                                 mode="bilinear"), cleaned).to(device)
            load_checkpoint(args.router, tb)
            tb.codes = cleaned
            got = chunked_argmax(q_probe, tb, N, n_rel, device)
            want = torch.from_numpy(rel_p * N + ent_p).to(device)
            ok = (got == want).cpu().numpy()
            del cleaned, tb, got
            if device.type == "cuda":
                torch.cuda.empty_cache()
            out[bits] = {
                "acc": float(ok.mean()),
                "by_depth": {k: float(ok[d_probe == k].mean()) for k in ("direct", "composed")},
                "by_relation": {RELATIONS[i]: float(ok[rel_p == i].mean())
                                for i in sorted(set(rel_p.tolist()))},
            }
            print(f"   {label:<18} {bits:>3} bits  acc {out[bits]['acc']:>6.1%}  "
                  f"direct {out[bits]['by_depth']['direct']:>6.1%}  "
                  f"composed {out[bits]['by_depth']['composed']:>6.1%}")
        return out

    print("\n-- GATE 1: no-denoiser ladder (the reference)")
    base = ladder(None, 0, "no denoiser")
    print("\n-- GATE 2, VERIFIED-RED NULL: identity denoiser must reproduce it EXACTLY")
    ident = ladder(IdentityDenoiser(), 1, "identity")
    same = all(abs(base[b]["acc"] - ident[b]["acc"]) < 1e-12 for b in RUNGS)
    print(f"   identical at every rung: {same}")
    if not same:
        raise SystemExit("VERIFIED-RED NULL FAILED — the harness changes the measurement")

    if args.probe:
        peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
        print(f"\n-- probe: {time.perf_counter() - t0:.0f}s, peak {peak:.2f} GB")
        return 0

    print(f"\n-- training G1 on (reconstruct @ {args.train_bits} bits, target) pairs")
    st = store_at(args.train_bits)
    idx = torch.randperm(N, device=device)[: min(N, 400_000)]
    noisy, clean = st.reconstruct(idx), codes[idx]
    del st
    seed_everything(args.seed)
    g1 = G1Denoiser(G1Config(latent_dim=dim, hidden=args.hidden)).to(device)
    print(f"   G1: {g1.n_params:,} params")
    fit_g1(g1, noisy, clean, steps=args.steps, log_every=max(1, args.steps // 4))
    del noisy, clean
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\n-- step-count vs recovery (the adaptive-compute curve)")
    curves = {k: ladder(g1, k, f"G1 x{k}") for k in (1, 2, 4, 8)}

    print("\n== RESULT")
    hdr = " ".join(f"{'G1 x' + str(k):>9}" for k in curves)
    print(f"   {'rung':>5} {'no-denoiser':>12} " + hdr)
    for bits in RUNGS:
        print(f"   {bits:>5} {base[bits]['acc']:>12.1%} "
              + " ".join(f"{curves[k][bits]['acc']:>9.1%}" for k in curves))

    def min_rung(tbl):
        ok = [b for b in RUNGS if tbl[b]["acc"] >= 0.99]
        return min(ok) if ok else None

    base_rung = min_rung(base)
    best = {k: min_rung(v) for k, v in curves.items()}
    n_facts = n_rel * N
    print(f"\n   min rung >= 99%: no denoiser {base_rung}, G1 {best}")
    print(f"   budget at N={N:,} ({n_facts:,} facts): "
          + ", ".join(f"{r}b -> {(47.9 - r) / n_rel * n_facts / 8:,.0f}" for r in (44, 40))
          + " int8 params")
    print(f"   G1 costs {g1.n_params:,} int8 params")
    print("   VERDICT priced at 2M (M4 §7.2): budgets 2,010,644 (44 rung) / 3,059,220 (40 rung).")

    payload = {"n": N, "seconds": time.perf_counter() - t0, "config": vars(args),
               "g1_params": g1.n_params, "no_denoiser": base, "identity": ident,
               "verified_red_null_passed": same,
               "g1": {str(k): v for k, v in curves.items()},
               "min_rung": {"no_denoiser": base_rung, "g1": {str(k): v for k, v in best.items()}}}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    if args.checkpoint_dir:
        d = Path(args.checkpoint_dir)
        d.mkdir(parents=True, exist_ok=True)
        save_checkpoint(d / "g1.pt", g1, model_config=g1.cfg, train_config=vars(args),
                        extra={"n": N, "train_bits": args.train_bits})
        print(f"wrote {d / 'g1.pt'}")
    print(f"\n== done in {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
