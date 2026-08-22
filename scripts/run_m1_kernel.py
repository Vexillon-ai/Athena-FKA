"""M1: train the D1 reasoning kernel and run the two gate measurements.

    python scripts/run_m1_kernel.py --size 10M --out experiments/<date>_m1-kernel-10M
    python scripts/run_m1_kernel.py --smoke          # tiny, CPU, seconds

Trains on 1-hop and 2-hop episodes routed through an oracle text memory, then reports:

  * **leakage** — fact recall with memory disabled, against a measured chance baseline
  * **composition** — 2-hop accuracy, with failures attributed to routing / copying / format

Both are gates from research plan §2.5, not metrics to admire. Everything needed to reproduce a
run — config, corpus fingerprint, git hash, throughput — is written to the output directory.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fka.data.corpus_gen import CorpusConfig, generate_corpus  # noqa: E402
from fka.data.hardening import distractor_facts, entity_split, harden_memory  # noqa: E402
from fka.data.multihop import chain_probe_list  # noqa: E402
from fka.data.tokenizer import CharTokenizer  # noqa: E402
from fka.eval.kernel_eval import composition_test, leakage_test  # noqa: E402
from fka.kernel.episodes import (  # noqa: E402
    episode_from_probe,
    max_episode_length,
    one_hop_episodes,
)
from fka.kernel.memory import OracleTextMemory  # noqa: E402
from fka.kernel.train import TrainConfig, save_run, train_kernel  # noqa: E402


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort, not a hard dependency
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", default="10M", choices=("tiny", "10M", "50M", "150M"))
    p.add_argument("--n-entities", type=int, default=2000)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--routing-loss-weight", type=float, default=0.0, help="D2; 0 disables")
    p.add_argument("--n-probes", type=int, default=200, help="1-hop probes for the leakage test")
    p.add_argument("--n-composition", type=int, default=200)
    p.add_argument(
        "--entity-holdout",
        type=float,
        default=0.2,
        help="fraction of entities never used as a training subject (hardened gate)",
    )
    p.add_argument("--hops", type=int, default=2, choices=(2, 3))
    p.add_argument(
        "--distractors",
        type=int,
        default=0,
        help="near-name distractor entities per real entity, added to memory only",
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None, help="experiment directory")
    p.add_argument(
        "--throughput-only",
        action="store_true",
        help="train briefly and report tokens/sec, skipping the gate tests (sizing policy)",
    )
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)

    if args.smoke:
        # 'tiny' matters as much as the small step count: generation does a full forward per
        # token, so a 10M model on CPU turns a 3-probe eval into minutes.
        args.size = "tiny"
        args.n_entities, args.steps, args.batch_size = 60, 3, 4
        args.n_probes = args.n_composition = 3
        args.device = "cpu" if args.device == "auto" else args.device

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )

    # n_coworkers=1 is required: "the colleague of X" must have a unique referent or the 2-hop
    # probe is ill-posed. See fka/data/multihop.py.
    corpus = generate_corpus(
        CorpusConfig(n_entities=args.n_entities, seed=args.seed, n_coworkers=1, probe_fraction=0.15)
    )
    tokenizer = CharTokenizer()

    rng = np.random.default_rng(args.seed)

    # HARDENED GATE (a): entity-level holdout. A disjoint entity set that is never the subject of
    # any training episode. Chain-level holdout only withheld the pairing, leaving "generalises"
    # and "has a per-entity query habit" indistinguishable.
    split = entity_split(corpus, fraction=args.entity_holdout, seed=args.seed)
    train_entities = set(split.train.tolist())

    one_hop = [
        ep
        for ep in one_hop_episodes(corpus)
        if corpus.entity_names.index(ep.subject) in train_entities
    ] if args.entity_holdout > 0 else list(one_hop_episodes(corpus))

    # HARDENED GATE (b): chains of the requested depth, cycles excluded.
    all_probes = chain_probe_list(corpus, args.hops)
    rng.shuffle(all_probes)
    name_to_id = {n: i for i, n in enumerate(corpus.entity_names)}
    train_probes = [p for p in all_probes if name_to_id[p.subject] in train_entities]
    holdout_probes = [p for p in all_probes if name_to_id[p.subject] not in train_entities]
    two_hop = [episode_from_probe(pr) for pr in train_probes]

    episodes = one_hop + two_hop
    needed = max_episode_length(episodes, tokenizer)
    block_size = int(min(1024, 64 * ((needed + 63) // 64)))

    print(f"== M1 kernel {args.size} on {device}")
    print(f"   corpus         {corpus!r}")
    print(f"   fingerprint    {corpus.fingerprint()}")
    print(
        f"   entities       {split.n_train:,} train / {split.n_heldout:,} HELD OUT "
        f"(never a training subject)"
    )
    print(
        f"   episodes       {len(one_hop):,} 1-hop + {len(two_hop):,} {args.hops}-hop trained, "
        f"{len(holdout_probes):,} chains held out"
    )
    print(f"   block_size     {block_size} (longest episode {needed} tokens)")
    print(f"   vocab          {tokenizer.vocab_size}")

    cfg = TrainConfig(
        size=args.size,
        block_size=block_size,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        seed=args.seed,
        routing_loss_weight=args.routing_loss_weight,
    )
    print(f"\n-- training ({'D2 routing loss ON' if args.routing_loss_weight else 'D1'})")
    model, state = train_kernel(episodes, tokenizer, cfg, device=device)
    print(f"   {state.tokens_per_sec:,.0f} tokens/sec over {state.seconds:.1f}s")
    print(f"   params {model.n_params():,}")

    if args.throughput_only:
        print(
            f"\n== {args.size}: {state.tokens_per_sec:,.0f} tokens/sec, "
            f"{model.n_params():,} params"
        )
        if args.out:
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            save_run(
                out / f"throughput_{args.size}.json",
                cfg,
                state,
                extra={"git_hash": git_hash(), "device": str(device), "n_params": model.n_params()},
            )
        return 0

    amp_dtype = torch.bfloat16 if device.type == "cuda" else None
    memory = OracleTextMemory.from_corpus(corpus)

    # HARDENED GATE (c): surface-form distractors, in memory only. Without them a near-miss
    # address returns an empty span, which hands the kernel a free signal that it erred.
    n_distractors = 0
    if args.distractors:
        n_distractors = harden_memory(
            memory,
            distractor_facts(corpus, per_entity=args.distractors, seed=args.seed),
        )
        print(f"\n   memory hardened with {n_distractors:,} distractor facts")

    probe_pairs = list(corpus.memory_pairs(corpus.probe_ids))[: args.n_probes]
    print(f"\n-- leakage test ({len(probe_pairs)} probes)")
    leak = leakage_test(
        model, tokenizer, memory, probe_pairs, device=device, amp_dtype=amp_dtype
    )
    print(f"   {leak}")

    comp_probes = holdout_probes[: args.n_composition]
    print(
        f"\n-- composition test ({len(comp_probes)} {args.hops}-hop chains, "
        f"ENTITY-HELD-OUT subjects)"
    )
    comp = composition_test(
        model, tokenizer, memory, comp_probes, device=device, amp_dtype=amp_dtype
    )
    verdict = "PASS" if comp.passes else "FAIL"
    print(
        f"   {args.hops}-hop accuracy [{verdict}] {comp.accuracy:.1%} (target >={0.85:.0%})  "
        f"mean hops {comp.mean_hops:.2f}"
    )
    print(
        f"   failures: routing {comp.n_routing_failures}, copy {comp.n_copy_failures}, "
        f"format {comp.n_format_failures}"
    )

    # Same measurement on chains the kernel *was* trained on. The gap between the two is the
    # only way to tell composition apart from chain memorisation.
    seen_probes = train_probes[: args.n_composition]
    comp_seen = composition_test(
        model, tokenizer, memory, seen_probes, device=device, amp_dtype=amp_dtype
    )
    print(
        f"   (for reference, chains seen in training: {comp_seen.accuracy:.1%} — "
        f"gap {comp.accuracy - comp_seen.accuracy:+.1%})"
    )

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        save_run(
            out / "metrics.json",
            cfg,
            state,
            extra={
                "git_hash": git_hash(),
                "device": str(device),
                "n_params": model.n_params(),
                "corpus": corpus.summary(),
                "hardening": {
                    "hops": args.hops,
                    "entity_holdout_fraction": args.entity_holdout,
                    "entity_split": split.to_dict(),
                    "n_distractor_facts": n_distractors,
                    "holdout_level": "entity",
                },
                "n_episodes": {
                    "one_hop": len(one_hop),
                    "chain_trained": len(two_hop),
                    "chain_heldout": len(holdout_probes),
                },
                "leakage": leak.to_dict(),
                "composition": comp.to_dict(),
                "composition_seen_in_training": comp_seen.to_dict(),
            },
        )
        (out / "config.yaml").write_text(
            json.dumps({"train": cfg.to_dict(), "corpus": corpus.config.to_dict()}, indent=2),
            encoding="utf-8",
        )
        print(f"\nwrote {out}/metrics.json")

    return 0 if (leak.passes and comp.passes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
