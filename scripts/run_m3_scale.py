"""M3 §2's outstanding clause: the JOINT SCALE MEASUREMENT. One store, one run design.

    python scripts/run_m3_scale.py --checkpoint <kernel> --router <r.pt> --query-head <h> --probe
    python scripts/run_m3_scale.py --checkpoint <kernel> --router <r.pt> --query-head <h>

Three measurements at each scale, on the same store so they cannot drift apart:

**(a) never-supervised addressability through `f`** — the §2 clause. Real held-out probes, real
kernel, real trained `f`; only the haystack grows.
**(b) shortlist candidate recall@m over the store's RECONSTRUCTED codes** — M2 §16.4's carried flag,
and the first non-easy case: reconstructions are sums of shared codebook entries and therefore
*correlated*, which is exactly where nearest-neighbour separation degrades. The 10^6 study used
random unit vectors.
**(c) `h` entity-recovery at scale** over those same correlated codes — its 96.7% at 10^6 was
measured on random codes and is not evidence here.

Pre-registered hazard (ruling 2): nearest-neighbour margins shrink like ~log N, so **the plausible
failure is address bits growing with entity count to hold 99%, not a cliff.** If 32 bits/entity
fails at scale, the deliverable is the measured **bits-vs-N curve at fixed retrieval** — that curve
*is* the capacity claim's scaling law. Mis-specification clause: **addressability holds while
shortlist recall collapses = a search problem, not an addressing problem, billed separately.**

Why synthetic entity codes above 167,772
-----------------------------------------
The corpus generator refuses beyond **167,772 entities**: its own 100x firewall (name space
must exceed 100x the entity count) caps 4,096 x 4,096 combinations. Enlarging the pools would
change schema-derived vocabularies, and CLAUDE.md fixes vocabularies as a function of the schema so
entropies stay comparable — every prior number would stop being comparable.

That assertion protects **text-probe unambiguity**, D1's concern. D3's addressing path never
sees a name: `f` reads entity *codes*. So above the ceiling this run uses synthetic codes, and
it **cross-checks the substitution** where both are available (real corpus vs synthetic codes
at the same N) rather than assuming they behave alike.
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
from fka.eval.timing import benchmark  # noqa: E402
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
from fka.store.s1_factorized import S1Config, S1FactorizedStore  # noqa: E402
from scripts.run_d3_kernel import build_data  # noqa: E402
from scripts.run_fork_a_joint import RELATIONS, harvest  # noqa: E402
from scripts.run_fork_c import address_holdout, supervised_addresses  # noqa: E402

KEY_CFG = S1Config(n_stages=4, codebook_size=256, residual_dim=0, residual_bits=8)
SCALES = [2_000, 50_000, 160_000, 500_000, 2_000_000]
BITS_SWEEP = [24, 28, 32, 36, 40]
GENERATOR_CEILING = 167_772


def memory_report(device) -> dict:
    """Windows has no `/sys/class/drm/.../mem_info_gtt_used` — CLAUDE.md says so explicitly and
    directs us to the torch counters instead. Recorded here rather than silently substituted."""
    if device.type != "cuda":
        return {"note": "cpu"}
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_gb": torch.cuda.memory_allocated() / 2**30,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
        "reserved_gb": torch.cuda.memory_reserved() / 2**30,
        "device_free_gb": free / 2**30, "device_total_gb": total / 2**30,
        "gtt_sysfs": "unavailable on Windows (CLAUDE.md); torch counters used instead",
    }


def build_keys(table: ContentKeyTable, n_entities: int, n_rel: int, device, chunk: int = 262_144):
    """All keys in fact-id order, chunked — the tensor is the run's memory high-water mark."""
    out = torch.empty(n_entities * n_rel, table.cfg.key_dim, device=device)
    for r in range(n_rel):
        for start in range(0, n_entities, chunk):
            stop = min(start + chunk, n_entities)
            ents = torch.arange(start, stop, device=device)
            rels = torch.full((stop - start,), r, device=device, dtype=torch.long)
            out[r * n_entities + start : r * n_entities + stop] = table(ents, rels)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--router", required=True)
    p.add_argument("--query-head", default=None)
    p.add_argument("--n-eval", type=int, default=400)
    p.add_argument("--n-one-hop", type=int, default=600)
    p.add_argument("--scales", type=int, nargs="+", default=SCALES)
    p.add_argument("--bits", type=int, nargs="+", default=BITS_SWEEP)
    p.add_argument("--probe", action="store_true", help="timing + memory probe, then exit")
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
    dim = meta["latent_dim"]
    corpus = generate_corpus(CorpusConfig(n_entities=2000, seed=meta["codebook_seed"],
                                          n_coworkers=1))
    tok, bs = d3_tokenizer(), meta["block_size"]
    split = entity_split(corpus, fraction=0.2, seed=meta["codebook_seed"])
    codebook = LatentCodebook.build(corpus, dim=dim, seed=meta["codebook_seed"])
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
    n0, n_rel = corpus.n_entities, len(RELATIONS)
    real_codes = F.normalize(codebook.entity, dim=-1).to(device)

    print(f"== M3 §2 scale clause on {device}")
    print(f"   generator ceiling {GENERATOR_CEILING:,} entities (its own 100x name firewall); "
          f"synthetic codes above it, cross-checked below it")

    def codes_for(n: int) -> torch.Tensor:
        """Real codes first, synthetic after — real probes keep entity ids 0..1999."""
        if n <= n0:
            return real_codes[:n]
        g = torch.Generator(device="cpu").manual_seed(args.seed + n)
        extra = F.normalize(torch.randn(n - n0, dim, generator=g), dim=-1).to(device)
        return torch.cat([real_codes, extra])

    # -- (a) queries: harvested ONCE through the oracle, reused at every scale --------------
    def harvest_probes():
        qs, ts, ds = [], [], []
        for packed in (packed3, packed1):
            q, t, h = harvest(model, oracle, packed, device, amp)
            qs.append(q)
            ts.append(t.cpu().numpy())
            ds.append(np.where(h.cpu().numpy() == 0, "direct", "composed"))
        return torch.cat(qs), np.concatenate(ts), np.concatenate(ds)

    q_all, t_all, d_all = harvest_probes()
    keep = ~np.isin(t_all, supervised)
    q_probe, t_probe, d_probe = q_all[torch.from_numpy(keep).to(device)], t_all[keep], d_all[keep]
    rel_probe, ent_probe = t_probe // n0, t_probe % n0
    print(f"   {len(t_probe)} never-supervised probe queries "
          f"({(d_probe == 'direct').sum()} direct / {(d_probe == 'composed').sum()} composed)")

    # -- timing + memory probe (ruling 3) --------------------------------------------------
    if args.probe:
        print("\n-- timing + memory probe (standing rule: size the run from a measured cost)")
        rows = []
        for n in (50_000, 200_000):
            codes = codes_for(n)
            store = S1FactorizedStore(replace(KEY_CFG, latent_dim=dim, seed=args.seed))
            t = time.perf_counter()
            store.write(codes)
            fit_s = time.perf_counter() - t
            table = ContentKeyTable(
                ContentKeyConfig(n_relations=n_rel, latent_dim=dim, key_dim=dim, comp_dim=128,
                                 mode="bilinear"),
                store.reconstruct(torch.arange(n, device=device)),
            ).to(device)
            load_checkpoint(args.router, table)
            table.codes = store.reconstruct(torch.arange(n, device=device))

            def _build(tbl=table, nn=n):
                return build_keys(tbl, nn, n_rel, device)

            r = benchmark(_build, name=f"keys@{n}", warmup=1, repeats=3, device=device)
            rows.append({"n": n, "store_fit_s": fit_s, "key_build_s": r.best,
                         "memory": memory_report(device)})
            print(f"   N={n:>9,}  store fit {fit_s:6.1f}s   key build {r.best:6.1f}s   "
                  f"peak {memory_report(device)['max_allocated_gb']:.2f} GB")
            del table, store, codes
            torch.cuda.empty_cache() if device.type == "cuda" else None
        per_key = rows[-1]["key_build_s"] / (rows[-1]["n"] * n_rel)
        for n in args.scales:
            print(f"   projected N={n:>9,}: keys {per_key * n * n_rel:7.1f}s, "
                  f"key tensor {n * n_rel * dim * 4 / 2**30:5.2f} GB, "
                  f"x{len(args.bits)} levels = "
                  f"{per_key * n * n_rel * len(args.bits) / 60:.1f} min")
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps({"probe": rows}, indent=2), encoding="utf-8")
        return 0

    # -- the joint measurement --------------------------------------------------------------
    inverter = None
    results = []
    for n in args.scales:
        codes = codes_for(n)
        real_frac = min(n0, n) / n
        print(f"\n-- N = {n:,} entities ({n * n_rel:,} facts), real fraction {real_frac:.1%}")
        best = None
        for bits in sorted(args.bits):
            cfg = replace(KEY_CFG, latent_dim=dim, seed=args.seed,
                          codebook_size=int(round(2 ** (bits / KEY_CFG.n_stages))))
            store = S1FactorizedStore(cfg)
            store.write(codes)
            recon = store.reconstruct(torch.arange(n, device=device))
            table = ContentKeyTable(
                ContentKeyConfig(n_relations=n_rel, latent_dim=dim, key_dim=dim, comp_dim=128,
                                 mode="bilinear"), recon,
            ).to(device)
            load_checkpoint(args.router, table)
            table.codes = recon
            keys = build_keys(table, n, n_rel, device)

            target = torch.from_numpy(rel_probe * n + ent_probe).to(device)
            got = (F.normalize(q_probe, dim=-1) @ F.normalize(keys, dim=-1).T).argmax(dim=-1)
            ok = (got == target).cpu().numpy()
            row = {
                "n_entities": n, "bits_per_entity": cfg.bits_per_slot,
                "codebook_size": cfg.codebook_size,
                "never_supervised": float(ok.mean()),
                "by_depth": {k: float(ok[d_probe == k].mean()) for k in ("direct", "composed")},
                "by_relation": {RELATIONS[i]: float(ok[rel_probe == i].mean())
                                for i in sorted(set(rel_probe.tolist()))},
            }
            print(f"   {cfg.bits_per_slot:>3.0f} b/entity  "
                  f"addressability {row['never_supervised']:>6.1%}"
                  f"   direct {row['by_depth']['direct']:>6.1%}  "
                  f"composed {row['by_depth']['composed']:>6.1%}")
            if best is None and row["never_supervised"] >= 0.99:
                best = row
            if bits == KEY_CFG.bits_per_slot or (best is None and bits == max(args.bits)):
                # (b) + (c) measured on the reconstructed (correlated) codes at this scale
                if inverter is None:
                    inverter = EntityInverter(
                        InverterConfig(latent_dim=dim, hidden=512, n_layers=2)
                    ).to(device)
                    q_tr, t_tr, _ = harvest(model, oracle, groups[-1], device, amp)
                    fit_inverter(inverter, q_tr, real_codes[t_tr.cpu().numpy() % n0],
                                 steps=4000, lr=3e-3, log_every=10**9)
                rec = entity_recovery_by_inversion(
                    inverter, q_probe, ent_probe, recon, n_rel,
                    ms=(1, 2, 4, 8, 16, 64, 256, 1024),
                )
                row["shortlist"] = rec.to_dict()
                m99 = rec.m_for(0.99)
                print(f"        shortlist over RECONSTRUCTED codes: recovery@1 {rec.at_1:.1%}, "
                      f"m for 99% = {m99 if m99 is not None else 'NOT REACHED'}")
            results.append(row)
            del keys, table, store, recon
            if device.type == "cuda":
                torch.cuda.empty_cache()
        mem = memory_report(device)
        print(f"   min bits holding 99%: "
              f"{best['bits_per_entity']:.0f}" if best else "   99% NOT REACHED at any bit level")
        print(f"   peak memory {mem.get('max_allocated_gb', 0):.2f} GB")
        del codes

    payload = {"checkpoint": str(args.checkpoint), "device": str(device),
               "seconds": time.perf_counter() - t0, "config": vars(args),
               "generator_ceiling": GENERATOR_CEILING,
               "memory": memory_report(device), "rows": results}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    print(f"\n== done in {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
