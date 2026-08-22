"""M2 Stage A: can a product-key set express the oracle's `e ⊙ r` key geometry at all?

    python scripts/run_router_stage_a.py --out experiments/<run>/stage_a.json

Pre-registered in `docs/decision_records/M2_router.md` §7. No kernel, no episodes: the query for
fact `(e, r)` is the oracle's own key, `normalize(entity_code[e] ⊙ relation_code[r])`, and the
target is that fact's slot. This is Stage B's precondition — if `2√N` vectors cannot separate `N`
oracle keys when handed perfect queries, nothing downstream can work, and finding that out costs
minutes instead of hours.

Headline is **SHUFFLED**. ALIGNED (axes = entity × relation) is a positive control that hands the
router our own factorisation and is never the result (§3, the alignment hazard).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fka.data.corpus_gen import CorpusConfig, generate_corpus  # noqa: E402
from fka.eval.router_eval import (  # noqa: E402
    aligned_slot_map,
    evaluate_router,
    identity_slot_map,
    shuffled_slot_map,
)
from fka.kernel.latent_memory import LatentCodebook, OracleLatentMemory  # noqa: E402
from fka.kernel.seeding import seed_everything  # noqa: E402
from fka.router.product_key import ProductKeyConfig, ProductKeyRouter  # noqa: E402

RELATIONS = ["birth_year", "birth_city", "employer", "works_with"]


class _OracleStub:
    """Returns each query's correct slot by construction. The instrument gate, at THIS scale.

    The standing router self-test runs on a 120-entity fixture; M2 section 3's registered response
    to a failed positive control is to gate on *the same slot set* before any number is admissible.
    So the gate runs inline here and the script refuses to report if it does not score 1.0.

    Identity comes from the query's content (nearest oracle key), never its row -- the alignment
    class that bit Phase 1 three times.
    """

    def __init__(self, keys, fact_to_slot, k=8, n_slots=0):
        self.keys, self.n_slots, self.k = keys, n_slots, k
        self.slots = torch.from_numpy(fact_to_slot).to(keys.device)

    def _answer(self, q):
        return self.slots[(q @ self.keys.T).argmax(dim=-1)]

    def __call__(self, q):
        a = self._answer(q)
        filler = (a.unsqueeze(1) + torch.arange(1, self.k, device=q.device)) % max(1, self.n_slots)
        return torch.cat([a.unsqueeze(1), filler], dim=1), torch.linspace(
            1.0, 0.0, self.k, device=q.device
        ).unsqueeze(0).repeat(len(q), 1)

    def slot_scores(self, q, slots):
        return (slots == self._answer(q).unsqueeze(1)).float()


def concatenative_keys(memory, relations, dim_half):
    """k = normalize([e_half ; r_half]) -- representable by a product key BY CONSTRUCTION.

    The constructive control for M2 section 7. Product-key scoring is additively separable across
    the two query halves; this key set is built so that half 1 depends only on the entity and
    half 2 only on the relation, which is exactly what an axis-aligned grid can express. The
    optimum is therefore known to exist and ~100% is attainable.

    Contrast with the oracle's `normalize(e * r)`, where EVERY coordinate depends on both factors.
    """
    cb = memory.codebook
    n_e = cb.entity.shape[0]
    keys = []
    for r in relations:
        e_half = cb.entity[:, :dim_half]
        r_half = cb.relation[r][:dim_half].unsqueeze(0).expand(n_e, -1)
        keys.append(F.normalize(torch.cat([e_half, r_half], dim=-1), dim=-1))
    return torch.cat(keys, dim=0)


def fit(router, queries, targets, *, steps, lr, temp, device, log_every=200):
    opt = torch.optim.AdamW(router.parameters(), lr=lr)
    n = len(queries)
    for step in range(steps):
        idx = torch.randint(0, n, (min(512, n),), device=device)
        loss = F.cross_entropy(router.all_scores(queries[idx]) / temp, targets[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            print(f"      step {step + 1:>5}/{steps}  loss {float(loss):.4f}")
    return float(loss)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-entities", type=int, default=2000)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--temp", type=float, default=0.05)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--keys", choices=["oracle", "concat"], default="oracle",
                   help="'concat' is the constructive control: product-key-representable keys")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    seed_everything(args.seed)
    corpus = generate_corpus(
        CorpusConfig(n_entities=args.n_entities, seed=args.seed, n_coworkers=1)
    )
    memory = OracleLatentMemory(
        corpus, LatentCodebook.build(corpus, dim=64, seed=args.seed)
    ).to(device)

    # The oracle's own keys, in fact-id order. These ARE the queries: Stage A asks about
    # expressivity, so the query side is made perfect on purpose.
    if args.keys == "concat":
        keys = concatenative_keys(memory, RELATIONS, memory.codebook.dim // 2).detach().to(device)
    else:
        keys = memory.keys.detach().to(device)
    n_facts = keys.shape[0]
    fact_ids = np.arange(n_facts)
    print(f"== M2 Stage A on {device}: {n_facts:,} {args.keys} keys, dim {keys.shape[1]}")

    configs = {
        "shuffled": (shuffled_slot_map(corpus, RELATIONS, seed=args.seed), None),
        "identity": (identity_slot_map(corpus, RELATIONS), None),
        "aligned": (aligned_slot_map(corpus, RELATIONS), (corpus.n_entities, len(RELATIONS))),
    }

    # ---- instrument gate, BEFORE any router is fitted ---------------------------------
    print("")
    print("-- instrument gate: oracle stub through the Stage A eval path, at this scale")
    for name, (smap, _axes) in configs.items():
        stub = _OracleStub(keys, smap.fact_to_slot, n_slots=smap.n_slots)
        g = evaluate_router(stub, keys, fact_ids, smap, corpus, RELATIONS, seed=args.seed)
        ok = g.recall_at_1 == 1.0 and g.worst_binding_margin > 0
        print(f"   {name:<9} recall@1 {g.recall_at_1:.1%}  worst binding margin "
              f"{g.worst_binding_margin:+.4f}  {'OK' if ok else 'FAILED'}")
        if not ok:
            raise SystemExit(
                f"instrument gate FAILED on '{name}': the eval path cannot score a "
                f"by-construction-correct router at 1.0, so no Stage A number from this "
                f"configuration is admissible (M2 section 3)."
            )

    results = {}
    for name, (smap, axes) in configs.items():
        role = "HEADLINE" if name == "shuffled" else (
            "positive control" if name == "aligned" else "intermediate"
        )
        print(f"\n-- {name}  [{role}]  axes {axes or 'square'}")
        seed_everything(args.seed)
        # n_slots is the GRID size, not the fact count: a square grid over 8,000 facts has 8,100
        # positions and the shuffled map legitimately uses the whole of it. Sizing the router to
        # the fact count instead makes valid slot ids unaddressable.
        cfg = ProductKeyConfig(
            n_slots=smap.n_slots, query_dim=keys.shape[1], topk=args.topk,
            topk_half=min(args.topk, min(axes) if axes else args.topk), axes=axes,
        )
        router = ProductKeyRouter(cfg).to(device)
        targets = torch.from_numpy(smap.fact_to_slot).to(device)
        final = fit(router, keys, targets, steps=args.steps, lr=args.lr,
                    temp=args.temp, device=device)

        res = evaluate_router(router, keys, fact_ids, smap, corpus, RELATIONS, seed=args.seed)
        print(f"   {res}")
        print(f"   worst binding margin {res.worst_binding_margin:+.4f}  "
              f"key params {router.n_params:,}  -> {'PASS' if res.passes else 'FAIL'}")
        results[name] = {
            "role": role, "axes": list(axes) if axes else [cfg.n_sub1, cfg.n_sub2],
            "final_loss": final, "n_params": router.n_params, **res.to_dict(),
        }

    head = results["shuffled"]
    print(f"\n== Stage A headline (SHUFFLED): recall@1 {head['recall_at_1']:.1%}, "
          f"worst binding margin {head['worst_binding_margin']:+.4f} -> "
          f"{'PASS' if head['passes'] else 'FAIL'}")
    print(f"   positive control (ALIGNED): recall@1 {results['aligned']['recall_at_1']:.1%}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps({"corpus_fingerprint": corpus.fingerprint(), "device": str(device),
                        "steps": args.steps, "key_set": args.keys,
                        "instrument_gate": "passed", "results": results}, indent=2),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")
    return 0 if head["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
