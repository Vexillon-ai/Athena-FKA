"""Run the full M1 gate battery against a saved D3 checkpoint. No training.

    python scripts/run_gate_battery.py --checkpoint experiments/<run>/checkpoint_dim64.pt \
        --out experiments/2026-08-02_d3-clean-rescore/battery_codec.json

Four gates, pre-registered in experiments/2026-08-02_d3-clean-rescore/notes.md:

    A  3-hop entity-held-out end-to-end     >= 95%   (with failure attribution)
    B  latent-side leakage: decode split    reported, localising only
    C  latent substitution, STICK rate      <= 5%    (not follow — it is ceilinged)
    D  codec context-free decode            == 100%

Everything is scored through the id-joined evaluator. The battery also runs the
**answer-start composition check** on gate A: the old measurement's 62.5% was the exact
fraction of episodes its slicer cut correctly, so a re-score that reproduces 62.5% means a
third alignment defect, not a model result. That is the pre-registered mis-specification
observation and it is checked automatically rather than left to be noticed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fka.data.corpus_gen import CorpusConfig, generate_corpus  # noqa: E402
from fka.data.hardening import entity_split  # noqa: E402
from fka.data.multihop import chain_probe_list  # noqa: E402
from fka.eval.latent_leakage import (  # noqa: E402
    SUBSTITUTION_MAX_STICK,
    codec_decode_accuracy,
    decode_accuracy,
    latent_substitution_test,
    subject_ablation_accuracy,
)
from fka.kernel.checkpoint import load_checkpoint  # noqa: E402
from fka.kernel.latent_episodes import d3_tokenizer, episode_from_probe, pack  # noqa: E402
from fka.kernel.latent_kernel import LatentKernelConfig, LatentReasoningKernel  # noqa: E402
from fka.kernel.latent_memory import LatentCodebook, OracleLatentMemory  # noqa: E402
from fka.kernel.latent_train import evaluate_d3  # noqa: E402

#: Pre-registered thresholds. None of these is capped by another component's accuracy.
GATE_END_TO_END = 0.95
GATE_CODEC_DECODE = 1.0
GATE_RETRIEVAL = 0.99

#: Measured, not assumed — most_frequent_answer_baseline on hardened D1 3-hop.
CHANCE_BASELINE = 0.025


def verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n-probes", type=int, default=200)
    p.add_argument("--hops", type=int, default=3)
    p.add_argument("--n-entities", type=int, default=2000)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None)
    p.add_argument("--label", default=None, help="name for this checkpoint in the report")
    args = p.parse_args(argv)

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    blob = load_checkpoint(args.checkpoint)
    meta, mcfg = blob["extra"], blob["model_config"]
    dim, seed = meta["latent_dim"], meta["codebook_seed"]
    label = args.label or Path(args.checkpoint).parent.name

    corpus = generate_corpus(CorpusConfig(n_entities=args.n_entities, seed=seed, n_coworkers=1))
    if corpus.fingerprint() != meta["corpus_fingerprint"]:
        raise SystemExit(
            "corpus fingerprint does not match the checkpoint — the battery would be scoring "
            "this model against a different world than it was trained on"
        )
    tokenizer = d3_tokenizer()
    split = entity_split(corpus, fraction=0.2, seed=seed)
    memory = OracleLatentMemory(corpus, LatentCodebook.build(corpus, dim=dim, seed=seed)).to(device)

    model = LatentReasoningKernel(LatentKernelConfig(**mcfg)).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()
    amp = torch.bfloat16 if device.type == "cuda" else None

    name_to_id = {n: i for i, n in enumerate(corpus.entity_names)}
    train_entities = set(split.train.tolist())

    # Reproduce the sweep's held-out selection exactly: one rng, hop counts in order. The eval set
    # must be the same 200 probes the quarantined numbers were produced on, or the comparison is
    # between two different measurements rather than two versions of one.
    rng = np.random.default_rng(seed)
    heldout = seen = None
    for n_hops in (2, args.hops):
        probes = chain_probe_list(corpus, n_hops)
        rng.shuffle(probes)
        if n_hops == args.hops:
            heldout = [p for p in probes if name_to_id[p.subject] not in train_entities]
            seen = [p for p in probes if name_to_id[p.subject] in train_entities]

    def prep(ps):
        return pack(
            [episode_from_probe(p, corpus, name_to_id) for p in ps],
            tokenizer, meta["block_size"], memory.fact_index,
        )

    held_packed = prep(heldout[: args.n_probes])
    seen_packed = prep(seen[: args.n_probes])

    print(f"== M1 gate battery  [{label}]  dim {dim}, {device}, {args.hops}-hop")
    print(f"   corpus {corpus.fingerprint()[:16]}  {split.n_train:,} train / "
          f"{split.n_heldout:,} held-out entities")
    print(f"   git at train time: {meta.get('git_hash', '?')}  "
          f"steps {blob['train_config'].get('steps')}  "
          f"codec loss {'yes' if 'codec_loss_weight' in blob['train_config'] else 'no'}")

    # ---- A: 3-hop entity-held-out end to end -------------------------------------------
    res, _ = evaluate_d3(model, memory, held_packed, tokenizer, device=device, amp_dtype=amp)
    a_ok = res.accuracy >= GATE_END_TO_END
    print(f"\n-- A  {args.hops}-hop entity-held-out end-to-end  [{verdict(a_ok)}]")
    print(f"      {res.accuracy:.1%}  ({res.n_correct}/{res.n})   "
          f"threshold {GATE_END_TO_END:.0%}, chance {CHANCE_BASELINE:.1%}")
    print(f"      failures: routing {res.n_routing_failures}, copy {res.n_copy_failures}, "
          f"format {res.n_format_failures}")
    print(f"      retrieval per hop {[f'{v:.1%}' for v in res.per_hop_retrieval]}")

    # The composition check, run automatically. The old evaluator could only score episodes whose
    # answer-start matched their batch leader's; that fraction was 62.5% and so was the result.
    starts = np.array(
        [int(np.flatnonzero(held_packed.answer_mask[i])[0]) for i in range(len(held_packed))]
    )
    by_start = {}
    for s in np.unique(starts):
        rows = np.flatnonzero(starts == s)
        ids = [int(held_packed.episode_id[r]) for r in rows]
        by_start[int(s)] = {
            "n": len(ids),
            "accuracy": sum(res.per_episode[i] for i in ids) / len(ids),
        }
    print("      by answer-start (the old slicer's fault line):")
    for s, d in sorted(by_start.items()):
        print(f"        start {s}: {d['accuracy']:.1%} of {d['n']}")

    majority = max(by_start.items(), key=lambda kv: kv[1]["n"])
    ceiling = majority[1]["n"] / len(held_packed)
    mis_specified = abs(res.accuracy - ceiling) < 1e-9 and majority[1]["accuracy"] == 1.0
    if mis_specified:
        print(f"\n      !! MIS-SPECIFICATION: accuracy equals the old slicing ceiling "
              f"({ceiling:.1%}) exactly, with the majority group perfect. Per the "
              f"pre-registration this is a third alignment defect, not a model result. "
              f"Stop and run the gold stub against THIS eval set.")

    # ---- B: latent-side leakage, decode split ------------------------------------------
    acc_seen = decode_accuracy(model, memory, seen_packed, tokenizer, device=device, amp_dtype=amp)
    print("\n-- B  decode split  [localising, not a gate]")
    print(f"      training-entity subjects {acc_seen:.1%} / held-out {res.accuracy:.1%}   "
          f"gap {res.accuracy - acc_seen:+.1%}")

    # ---- C: latent substitution --------------------------------------------------------
    sub = latent_substitution_test(
        model, memory, seen_packed, tokenizer, device=device, amp_dtype=amp, seed=seed
    )
    print(f"\n-- C  latent substitution  [{verdict(sub.passes)}]  (permanent latent-leakage gate)")
    print(f"      {sub}")
    cf = sub.conditional_follow_rate(acc_seen)
    print(f"      conditional follow (reported, never gated): "
          f"{cf:.1%}" if cf is not None else "      conditional follow: n/a")

    # ---- D: codec context-free decode --------------------------------------------------
    codec = codec_decode_accuracy(
        model, memory, tokenizer, device=device, amp_dtype=amp,
        block_size=meta["block_size"], seed=seed,
    )
    d_ok = codec.accuracy >= GATE_CODEC_DECODE
    print(f"\n-- D  codec context-free decode  [{verdict(d_ok)}]")
    print(f"      {codec}")
    for r, v in sorted(codec.per_relation.items()):
        print(f"        {r:<12} {v:.1%}")

    # ---- confounded, read last ---------------------------------------------------------
    abl = subject_ablation_accuracy(
        model, memory, seen_packed, tokenizer, device=device, amp_dtype=amp
    )
    print(f"\n-- (confounded) subject-path ablation: {abl:.1%} (was {acc_seen:.1%}) — the subject "
          f"code legitimately feeds hop-1 queries, so only an ABSENCE of a drop is informative")

    r_ok = min(res.per_hop_retrieval) >= GATE_RETRIEVAL
    all_pass = a_ok and sub.passes and d_ok and r_ok and not mis_specified
    print(f"\n== {label}: {'ALL GATES PASS' if all_pass else 'NOT CLEAR'}"
          f"   (A {verdict(a_ok)}, C {verdict(sub.passes)}, D {verdict(d_ok)}, "
          f"retrieval {verdict(r_ok)})")

    payload = {
        "label": label,
        "checkpoint": str(args.checkpoint),
        "latent_dim": dim,
        "train_steps": blob["train_config"].get("steps"),
        "train_git_hash": meta.get("git_hash"),
        "corpus_fingerprint": corpus.fingerprint(),
        "device": str(device),
        "chance_baseline": CHANCE_BASELINE,
        "gate_a_end_to_end": {**res.to_dict(), "threshold": GATE_END_TO_END, "passes": a_ok},
        "answer_start_composition": by_start,
        "mis_specification_fired": bool(mis_specified),
        "gate_b_decode_split": {
            "seen": acc_seen, "heldout": res.accuracy, "gap": res.accuracy - acc_seen
        },
        "gate_c_substitution": sub.to_dict(decode_accuracy=acc_seen),
        "gate_d_codec_decode": {**codec.to_dict(), "threshold": GATE_CODEC_DECODE, "passes": d_ok},
        "subject_ablation": {"seen_accuracy": abl, "baseline": acc_seen},
        "retrieval_passes": r_ok,
        "all_gates_pass": all_pass,
        "thresholds": {
            "end_to_end": GATE_END_TO_END,
            "substitution_max_stick": SUBSTITUTION_MAX_STICK,
            "codec_decode": GATE_CODEC_DECODE,
            "retrieval": GATE_RETRIEVAL,
        },
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
