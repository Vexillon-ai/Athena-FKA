"""The registered FIRST POINT on the degradation surface (M3 §2.3).

    python scripts/run_s1_first_point.py --checkpoint <kernel> --router <router.pt> --query-head <h>

Fork (c) addressability with `f` reading **`store.reconstruct()`** instead of the clean codebook —
at the **lightest** compression, where reconstruction is exact. Nothing else changes: same kernel,
same trained router, same address-level holdout, same evaluator.

**The point of the lightest point is that it should change nothing.** It is the gate that separates
plumbing from compression: if addressability drops here, the substrate integration is wrong, and
every later point on the surface would be measuring that bug instead of compression.

Per-class splits from the start (M3 §4.1): `works_with` returns an entity code and the other three
return value codes, so a pooled figure could hide a substrate that handles one family and not the
other.
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
from fka.router.routed_memory import RoutedLatentMemory  # noqa: E402
from fka.store.base import IdentityStore  # noqa: E402
from fka.store.s1_factorized import S1Config, S1FactorizedStore  # noqa: E402
from scripts.run_d3_kernel import build_data  # noqa: E402
from scripts.run_fork_a_joint import RELATIONS, harvest  # noqa: E402
from scripts.run_fork_c import address_holdout, supervised_addresses  # noqa: E402


def build_table(meta, store, n_entities, device, router_path):
    """Fork (c)'s trained key encoder, reading the substrate's reconstruction as content."""
    slots = torch.arange(n_entities, device=device)
    content = store.reconstruct(slots)
    table = ContentKeyTable(
        ContentKeyConfig(n_relations=len(RELATIONS), latent_dim=meta["latent_dim"],
                         key_dim=meta["latent_dim"], comp_dim=128, mode="bilinear"),
        content,
    ).to(device)
    load_checkpoint(router_path, table)
    # load_checkpoint restores the buffer too, so re-install the reconstruction afterwards.
    table.codes = content.detach().clone()
    return table


@torch.no_grad()
def measure(model, oracle, table, packed, supervised, corpus, device, amp):
    routed = RoutedLatentMemory(oracle, table, RELATIONS).to(device).freeze_keys()
    queries, targets, _ = harvest(model, routed, packed, device, amp)
    ok = (routed.retrieved_index(queries) == targets).cpu().numpy()
    t = targets.cpu().numpy()
    never = ~np.isin(t, supervised)
    rel = t // corpus.n_entities

    # Per-FACT end-to-end correctness, keyed by fact id. The §3.1 instrument's store-internal
    # `addressing` channel saturates at 1.000 on the compression axis — it measures whether a code
    # still outranks its neighbours inside the store, which stays true long after the downstream
    # path has stopped working. The quantity that actually degrades is this one, so the shape
    # question needs it per fact or the sweep cannot conclude.
    per_fact: dict[int, list] = {}
    for fid, hit in zip(t[never].tolist(), ok[never].tolist(), strict=True):
        per_fact.setdefault(int(fid), []).append(float(hit))

    out = {
        "per_fact_never_supervised": {k: float(np.mean(v)) for k, v in per_fact.items()},
        "never_supervised": float(ok[never].mean()) if never.any() else None,
        "supervised": float(ok[~never].mean()) if (~never).any() else None,
        "all": float(ok.mean()),
        "n_never_supervised": int(never.sum()),
        "per_relation_never_supervised": {},
    }
    for r, name in enumerate(RELATIONS):
        m = never & (rel == r)
        out["per_relation_never_supervised"][name] = {
            "recall": float(ok[m].mean()) if m.any() else None, "n": int(m.sum())
        }
    return out


def _show(tag, res, err):
    per = "  ".join(
        f"{k}:{'-' if v['recall'] is None else format(v['recall'], '.1%')}(n={v['n']})"
        for k, v in res["per_relation_never_supervised"].items()
    )
    ns = res["never_supervised"]
    print(f"   {tag:<26} NEVER-SUP {ns:.1%}  sup {res['supervised']:.1%}  "
          f"all {res['all']:.1%}   recon_err {err:.2e}")
    print(f"   {'':<26} per relation: {per}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--router", required=True)
    p.add_argument("--query-head", default=None)
    p.add_argument("--n-entities", type=int, default=2000)
    p.add_argument("--n-eval", type=int, default=400)
    p.add_argument("--sweep", action="store_true",
                   help="also walk the compression axis — the first SLICE of the surface")
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


def _run(args, device) -> int:
    seed_everything(args.seed)
    t0 = time.perf_counter()
    blob = load_checkpoint(args.checkpoint)
    meta, mcfg = blob["extra"], blob["model_config"]
    corpus = generate_corpus(
        CorpusConfig(n_entities=args.n_entities, seed=meta["codebook_seed"], n_coworkers=1)
    )
    tok = d3_tokenizer()
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
        corpus, tok, split, [2, 3], oracle.fact_index, meta["block_size"], meta["codebook_seed"]
    )
    groups, _ = address_holdout(raw, split.heldout, corpus.n_entities)
    supervised = supervised_addresses(groups)
    packed = pack(heldout_eps[: args.n_eval], tok, meta["block_size"], oracle.fact_index)
    codes = F.normalize(oracle.codebook.entity, dim=-1).to(device)
    n_e, dim = corpus.n_entities, meta["latent_dim"]

    print(f"== M3 first point — fork (c) addressability through store.reconstruct(), {device}")
    print(f"   {len(packed.tokens)} held-out 3-hop probes; address-level holdout")

    results = {}
    slots = torch.arange(n_e, device=device)

    # -- GATE: the lossless store must reproduce fork (c) exactly --------------------------
    print("\n-- instrument gate: IdentityStore (no compression) must change nothing")
    ident = IdentityStore()
    ident.write(codes)
    tbl = build_table(meta, ident, n_e, device, args.router)
    gate = measure(model, oracle, tbl, packed, supervised, corpus, device, amp)
    _show("IdentityStore", gate, float(ident.recon_error(slots).max()))
    results["gate_identity"] = gate
    if gate["never_supervised"] < 0.999:
        raise SystemExit(
            f"GATE FAILED: lossless substrate gives {gate['never_supervised']:.1%}, not fork (c)'s "
            "100.0% — the integration is wrong and no compressed number would be admissible"
        )
    print("   -> OK: plumbing is transparent, so later shortfalls are compression")

    # -- the registered FIRST POINT: S1 at its lightest ------------------------------------
    print("\n-- FIRST POINT: S1 at lightest compression (residual_dim = latent_dim, exact)")
    s1 = S1FactorizedStore(S1Config(latent_dim=dim, n_stages=2, codebook_size=256,
                                    residual_dim=dim, seed=args.seed))
    s1.write(codes)
    tbl = build_table(meta, s1, n_e, device, args.router)
    first = measure(model, oracle, tbl, packed, supervised, corpus, device, amp)
    _show("S1 lightest", first, float(s1.recon_error(slots).max()))
    results["first_point"] = {**first, "cost_model": s1.cost_model(),
                              "recon_error_mean": float(s1.recon_error(slots).mean())}

    # -- optional first SLICE of the surface -----------------------------------------------
    if args.sweep:
        print("\n-- first slice of the degradation surface (compression axis only)")
        slice_ = {}
        for n_stages, k, r in [(4, 256, 32), (4, 256, 16), (4, 256, 8), (4, 256, 0), (2, 64, 0)]:
            st = S1FactorizedStore(S1Config(latent_dim=dim, n_stages=n_stages, codebook_size=k,
                                            residual_dim=r, seed=args.seed))
            st.write(codes)
            tbl = build_table(meta, st, n_e, device, args.router)
            res = measure(model, oracle, tbl, packed, supervised, corpus, device, amp)
            err = float(st.recon_error(slots).mean())
            _show(f"stages={n_stages} K={k} r={r}", res, err)
            slice_[f"s{n_stages}_k{k}_r{r}"] = {
                **res, "recon_error_mean": err, "bits_per_slot": st.cfg.bits_per_slot
            }
        results["surface_slice"] = slice_

    payload = {"checkpoint": str(args.checkpoint), "router": str(args.router),
               "device": str(device), "seconds": time.perf_counter() - t0,
               "corpus_fingerprint": corpus.fingerprint(), "config": vars(args),
               "results": results}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
