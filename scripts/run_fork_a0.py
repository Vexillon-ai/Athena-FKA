"""Stage a0: can composed keys serve the FROZEN kernel's existing query geometry?

    python scripts/run_fork_a0.py --checkpoint experiments/<run>/checkpoint_dim64.pt

Pre-registered as an amendment to `docs/decision_records/M2_router.md` §9. Nothing is fine-tuned:
the kernel is frozen, its emitted query vectors are harvested, and only `key(e, r)` is fitted to
them.

**Legitimacy.** Kernel-emitted queries exist at inference time — serving a fixed client's geometry
is the deployment question, not a shortcut. If composed keys can be fitted to them, fork (a) closes
early: compositional addressing is learnable against a fixed client, with no joint fit and
therefore no co-adaptation risk (§9.3(b)) and no query-head drift risk (§9.3(c)) at all.

**Decision rule, fixed before the run:**

    a0 passes its gates  -> fork (a) closes early; proceed to searchability.
    a0 fails             -> joint fit proceeds as registered, and a0's failure mode says
                            which side has to move.
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
from fka.data.hardening import entity_split  # noqa: E402
from fka.data.multihop import chain_probe_list  # noqa: E402
from fka.eval.router_eval import evaluate_router, identity_slot_map  # noqa: E402
from fka.kernel.checkpoint import load_checkpoint  # noqa: E402
from fka.kernel.gpu_lock import gpu_lock  # noqa: E402
from fka.kernel.latent_episodes import d3_tokenizer, episode_from_probe, pack  # noqa: E402
from fka.kernel.latent_kernel import LatentKernelConfig, LatentReasoningKernel  # noqa: E402
from fka.kernel.latent_memory import LatentCodebook, OracleLatentMemory  # noqa: E402
from fka.kernel.latent_train import _to_device  # noqa: E402
from fka.kernel.seeding import seed_everything  # noqa: E402
from fka.router.composed_keys import (  # noqa: E402
    ComposedKeyConfig,
    ComposedKeyTable,
    key_spread,
)
from fka.router.dense import DenseKeyRouter, GoldStubRouter  # noqa: E402

RELATIONS = ["birth_year", "birth_city", "employer", "works_with"]


@torch.no_grad()
def harvest(model, memory, packed, device, amp, batch=64):
    """Emitted query vectors and their target fact ids, from the frozen kernel."""
    qs, ts = [], []
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
    return torch.cat(qs), torch.cat(ts)


def fit_keys(table, ents, rels, queries, targets, *, steps, lr, temp, device, n_facts):
    opt = torch.optim.AdamW(table.parameters(), lr=lr)
    n = len(queries)
    for step in range(steps):
        idx = torch.randint(0, n, (min(1024, n),), device=device)
        keys = F.normalize(table(ents, rels), dim=-1)
        logits = F.normalize(queries[idx], dim=-1) @ keys.T / temp
        loss = F.cross_entropy(logits, targets[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 200 == 0 or step == steps - 1:
            print(f"      step {step + 1:>5}/{steps}  loss {float(loss.detach()):.4f}")
    return float(loss.detach())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n-entities", type=int, default=2000)
    p.add_argument("--n-train-probes", type=int, default=1200)
    p.add_argument("--n-eval-probes", type=int, default=400)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--temp", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--force-gpu-lock", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    if device.type == "cuda":
        gpu_lock(force=args.force_gpu_lock).__enter__()
    seed_everything(args.seed)

    blob = load_checkpoint(args.checkpoint)
    meta, mcfg = blob["extra"], blob["model_config"]
    corpus = generate_corpus(
        CorpusConfig(n_entities=args.n_entities, seed=meta["codebook_seed"], n_coworkers=1)
    )
    if corpus.fingerprint() != meta["corpus_fingerprint"]:
        raise SystemExit("corpus fingerprint mismatch with the checkpoint")

    tok = d3_tokenizer()
    split = entity_split(corpus, fraction=0.2, seed=meta["codebook_seed"])
    memory = OracleLatentMemory(
        corpus, LatentCodebook.build(corpus, dim=meta["latent_dim"], seed=meta["codebook_seed"])
    ).to(device)
    model = LatentReasoningKernel(LatentKernelConfig(**mcfg)).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)  # FROZEN. a0's whole legitimacy rests on this.
    amp = torch.bfloat16 if device.type == "cuda" else None

    n2i = {n: i for i, n in enumerate(corpus.entity_names)}
    train_e = set(split.train.tolist())
    probes = chain_probe_list(corpus, 3)
    np.random.default_rng(meta["codebook_seed"]).shuffle(probes)
    seen = [q for q in probes if n2i[q.subject] in train_e][: args.n_train_probes]
    held = [q for q in probes if n2i[q.subject] not in train_e][: args.n_eval_probes]

    def prep(ps):
        return pack([episode_from_probe(q, corpus, n2i) for q in ps], tok,
                    meta["block_size"], memory.fact_index)

    print(f"== Fork (a) Stage a0 on {device}  [kernel FROZEN]")
    print(f"   {len(seen)} train / {len(held)} entity-held-out 3-hop probes")

    q_tr, t_tr = harvest(model, memory, prep(seen), device, amp)
    q_te, t_te = harvest(model, memory, prep(held), device, amp)
    print(f"   harvested queries: {len(q_tr):,} train / {len(q_te):,} held-out")

    n_facts = corpus.n_entities * len(RELATIONS)
    ents = torch.arange(n_facts, device=device) % corpus.n_entities
    rels = torch.arange(n_facts, device=device) // corpus.n_entities

    # One query per unique held-out target: evaluate_router joins on fact id and needs them unique.
    uniq, first = np.unique(t_te.cpu().numpy(), return_index=True)
    q_eval, fact_ids = q_te[torch.from_numpy(first).to(device)], uniq
    smap = identity_slot_map(corpus, RELATIONS)
    print(f"   eval on {len(fact_ids):,} distinct held-out target facts")

    print("\n-- instrument gate: gold stub through the a0 eval path")
    gt = torch.from_numpy(smap.fact_to_slot[fact_ids]).to(device)
    g = evaluate_router(GoldStubRouter(gt, smap.n_slots), q_eval, fact_ids, smap, corpus, RELATIONS)
    if not (g.recall_at_1 == 1.0 and g.worst_binding_margin > 0):
        raise SystemExit(f"instrument gate FAILED ({g!s}); no a0 number is admissible")
    print(f"   recall@1 {g.recall_at_1:.1%}, worst binding margin "
          f"{g.worst_binding_margin:+.4f}  OK")

    results = {}
    for mode in ("mlp", "bilinear"):
        print(f"\n-- composition: {mode}")
        seed_everything(args.seed)
        table = ComposedKeyTable(ComposedKeyConfig(
            n_entities=corpus.n_entities, n_relations=len(RELATIONS),
            key_dim=meta["latent_dim"], mode=mode,
        )).to(device)
        loss = fit_keys(table, ents, rels, q_tr, t_tr, steps=args.steps, lr=args.lr,
                        temp=args.temp, device=device, n_facts=n_facts)
        keys = F.normalize(table(ents, rels), dim=-1).detach()
        res = evaluate_router(DenseKeyRouter(keys), q_eval, fact_ids, smap, corpus, RELATIONS)
        spread = key_spread(keys)
        print(f"   {res}")
        print(f"   worst binding margin {res.worst_binding_margin:+.4f}   "
              f"key params {table.n_params:,}")
        print(f"   key spread: mean cos {spread['mean_cosine']:+.4f}, effective rank "
              f"{spread['effective_rank']:.1f}/{spread['key_dim']} "
              f"({spread['effective_rank_fraction']:.1%})")
        print(f"   -> {'PASS' if res.passes else 'FAIL'}")
        results[mode] = {"final_loss": loss, "n_params": table.n_params,
                         "key_spread": spread, **res.to_dict()}

    best = max(results, key=lambda m: results[m]["recall_at_1"])
    passed = results[best]["passes"]
    print(f"\n== Stage a0: best={best}, held-out recall@1 "
          f"{results[best]['recall_at_1']:.1%}, worst binding margin "
          f"{results[best]['worst_binding_margin']:+.4f} -> {'PASS' if passed else 'FAIL'}")
    print("   " + ("fork (a) CLOSES EARLY: no joint fit, no drift risk." if passed
                   else "joint fit proceeds as registered (M2 §9)."))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"checkpoint": str(args.checkpoint), "device": str(device),
             "corpus_fingerprint": corpus.fingerprint(), "instrument_gate": "passed",
             "steps": args.steps, "results": results, "verdict": "pass" if passed else "fail"},
            indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
