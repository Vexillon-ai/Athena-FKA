"""Complete the bits-vs-N curve to 2M, and FIT ITS FUNCTIONAL FORM. (M3 §20)

    python scripts/run_m3_curve.py --checkpoint <k> --router <r> --query-head <h> --probe
    python scripts/run_m3_curve.py --checkpoint <k> --router <r> --query-head <h>

**The curve's shape IS the §2 verdict** (ruling 2), so the discrimination is pre-registered:

    log form    bits = c * log2(N) + b     -> ~48-52 bits/entity at 2M, ~0.387 bits/bit, gate holds
    power form  bits = a * N^p             -> the gate fails at some N; report which

Both fits are reported with residuals; neither is chosen by eye.

**Memory probe is mandatory alongside the timing probe** (convention ratified 2026-08-02: a timing
probe is not a memory probe). The store fit is now chunked and its Lloyd update is a scatter, so
intermediates are bounded by `chunk x K`; the probe verifies the 500k footprint has actually fallen
before 2M is attempted.

Everything above 167,772 entities uses synthetic codes (the generator's own 100x name firewall), and
the real-vs-synthetic cross-check remains owed — recorded, not resolved.
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
from fka.kernel.checkpoint import load_checkpoint  # noqa: E402
from fka.kernel.gpu_lock import gpu_lock  # noqa: E402
from fka.kernel.latent_episodes import d3_tokenizer, one_hop_episode, pack  # noqa: E402
from fka.kernel.latent_kernel import LatentKernelConfig, LatentReasoningKernel  # noqa: E402
from fka.kernel.latent_memory import LatentCodebook, OracleLatentMemory  # noqa: E402
from fka.kernel.seeding import seed_everything  # noqa: E402
from fka.router.content_keys import ContentKeyConfig, ContentKeyTable  # noqa: E402
from fka.store.s1_factorized import S1Config, S1FactorizedStore  # noqa: E402
from scripts.run_d3_kernel import build_data  # noqa: E402
from scripts.run_fork_a_joint import RELATIONS, harvest  # noqa: E402
from scripts.run_fork_c import address_holdout, supervised_addresses  # noqa: E402

BASE = S1Config(n_stages=4, codebook_size=256, residual_dim=0, residual_bits=8)
N_POINTS = [2_000, 5_000, 12_000, 30_000, 75_000, 160_000, 400_000, 1_000_000, 2_000_000]
BIT_LEVELS = [24, 28, 32, 36, 40, 44, 48, 52, 56, 60]
CRITERION = 0.99


def mem() -> dict:
    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    return {"peak_gb": torch.cuda.max_memory_allocated() / 2**30,
            "free_gb": free / 2**30, "total_gb": total / 2**30}


@torch.no_grad()
def chunked_argmax(queries: torch.Tensor, table: ContentKeyTable, n_ent: int, n_rel: int,
                   device, chunk: int = 131_072) -> torch.Tensor:
    """Best slot per query WITHOUT materialising the full key matrix.

    Streams key blocks and keeps a running max, so peak allocation is `chunk x dim` plus
    `n_queries x chunk` rather than `n_ent * n_rel x dim`. At 2M entities the full matrix is 2 GB,
    which fits — but the fit's intermediates did not, and bounding both is what makes 2M reachable.
    """
    q = F.normalize(queries.float(), dim=-1)
    best = torch.full((q.shape[0],), -2.0, device=device)
    arg = torch.zeros(q.shape[0], dtype=torch.long, device=device)
    for r in range(n_rel):
        for start in range(0, n_ent, chunk):
            stop = min(start + chunk, n_ent)
            ents = torch.arange(start, stop, device=device)
            rels = torch.full((stop - start,), r, device=device, dtype=torch.long)
            keys = F.normalize(table(ents, rels), dim=-1)
            s = q @ keys.T
            v, i = s.max(dim=1)
            take = v > best
            best = torch.where(take, v, best)
            arg = torch.where(take, i + start + r * n_ent, arg)
            del keys, s
    return arg


def fit_forms(ns: list[int], bits: list[float]) -> dict:
    """Both pre-registered forms, with residuals. Neither is chosen by eye."""
    x, y = np.array(ns, dtype=float), np.array(bits, dtype=float)
    log_c, log_b = np.polyfit(np.log2(x), y, 1)
    log_pred = log_c * np.log2(x) + log_b
    pw = np.polyfit(np.log(x), np.log(y), 1)
    p, log_a = pw[0], pw[1]
    pow_pred = np.exp(log_a) * x**p

    def rms(pred):
        return float(np.sqrt(np.mean((y - pred) ** 2)))

    return {
        "log": {"c": float(log_c), "b": float(log_b), "rms": rms(log_pred),
                "predict_2M": float(log_c * np.log2(2_000_000) + log_b)},
        "power": {"a": float(np.exp(log_a)), "p": float(p), "rms": rms(pow_pred),
                  "predict_2M": float(np.exp(log_a) * 2_000_000**p)},
        "n": ns, "bits": bits,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--router", required=True)
    p.add_argument("--query-head", default=None)
    p.add_argument("--n-eval", type=int, default=400)
    p.add_argument("--n-one-hop", type=int, default=600)
    p.add_argument("--points", type=int, nargs="+", default=N_POINTS)
    p.add_argument("--probe", action="store_true")
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
    n0, n_rel = corpus.n_entities, len(RELATIONS)
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
    print(f"== M3 bits-vs-N curve on {device}   ({len(ent_p)} never-supervised probes)")

    def codes_for(n):
        if n <= n0:
            return real[:n]
        g = torch.Generator(device="cpu").manual_seed(args.seed + n)
        return torch.cat([real,
                          F.normalize(torch.randn(n - n0, dim, generator=g), dim=-1).to(device)])

    def accuracy(n, bits):
        cfg = replace(BASE, latent_dim=dim, seed=args.seed,
                      codebook_size=int(round(2 ** (bits / BASE.n_stages))))
        store = S1FactorizedStore(cfg)
        store.write(codes_for(n))
        recon = store.reconstruct(torch.arange(n, device=device))
        table = ContentKeyTable(
            ContentKeyConfig(n_relations=n_rel, latent_dim=dim, key_dim=dim, comp_dim=128,
                             mode="bilinear"), recon).to(device)
        load_checkpoint(args.router, table)
        table.codes = recon
        got = chunked_argmax(q_probe, table, n, n_rel, device)
        want = torch.from_numpy(rel_p * n + ent_p).to(device)
        acc = float((got == want).float().mean())
        del store, recon, table, got
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return acc

    if args.probe:
        print("\n-- MANDATORY memory + timing probe (a timing probe is not a memory probe)")
        for n in (160_000, 500_000):
            torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
            t = time.perf_counter()
            a = accuracy(n, 40)
            m = mem()
            print(f"   N={n:>9,}  acc {a:.1%}   {time.perf_counter() - t:6.1f}s   "
                  f"peak {m.get('peak_gb', 0):6.2f} GB  (was 8.21 / 25.07 GB unchunked)")
        print(f"   free {mem().get('free_gb', 0):.1f} of {mem().get('total_gb', 0):.1f} GB")
        return 0

    rows, ns, bits_at = [], [], []
    for n in args.points:
        torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
        lo, hi, best, seen = 0, len(BIT_LEVELS) - 1, None, {}
        while lo <= hi:  # bisect the bit ladder: ~log2(10) evaluations per N, not 10
            mid = (lo + hi) // 2
            b = BIT_LEVELS[mid]
            seen[b] = accuracy(n, b)
            if seen[b] >= CRITERION:
                best = b
                hi = mid - 1
            else:
                lo = mid + 1
        m = mem()
        rows.append({"n": n, "min_bits": best, "measured": seen, "peak_gb": m.get("peak_gb")})
        print(f"   N={n:>9,}  min bits >= 99%: "
              f"{best if best is not None else '> ' + str(BIT_LEVELS[-1])}   "
              f"({', '.join(f'{k}:{v:.1%}' for k, v in sorted(seen.items()))})   "
              f"peak {m.get('peak_gb', 0):.2f} GB")
        if best is not None:
            ns.append(n)
            bits_at.append(float(best))

    fits = fit_forms(ns, bits_at) if len(ns) >= 3 else {}
    if fits:
        print("\n== FUNCTIONAL FORM (pre-registered discrimination; both reported)")
        print(f"   log form   bits = {fits['log']['c']:.2f}*log2(N) + {fits['log']['b']:.2f}   "
              f"RMS {fits['log']['rms']:.2f}   -> {fits['log']['predict_2M']:.1f} bits at 2M")
        print(f"   power form bits = {fits['power']['a']:.2f}*N^{fits['power']['p']:.3f}      "
              f"RMS {fits['power']['rms']:.2f}   -> {fits['power']['predict_2M']:.1f} bits at 2M")
        better = "log" if fits["log"]["rms"] <= fits["power"]["rms"] else "power"
        print(f"   lower residual: {better.upper()} form")
        measured_2m = next((r["min_bits"] for r in rows if r["n"] == 2_000_000), None)
        if measured_2m:
            key_bits = measured_2m / n_rel
            marginal = 11.13 + key_bits
            print("\n== §2 AT THE MEASURED 2M POINT")
            print(f"   min bits/entity {measured_2m}  -> key {key_bits:.2f} b/fact, "
                  f"marginal {marginal:.2f} b/fact -> {9.15 / marginal:.4f} bits/bit "
                  f"(gate 0.375)  {'PASS' if 9.15 / marginal >= 0.375 else 'FAIL'}")

    payload = {"device": str(device), "seconds": time.perf_counter() - t0, "config": vars(args),
               "rows": rows, "fits": fits, "memory": mem()}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    print(f"\n== done in {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
