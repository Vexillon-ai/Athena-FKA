"""D3: train the latent-interface kernel and sweep query/latent dimension.

    python scripts/run_d3_kernel.py --dims 64 128 256 --out experiments/2026-08-01_d3-sweep
    python scripts/run_d3_kernel.py --smoke

Holdout levels are identical to the hardened D1 baseline so the comparison is like-for-like.
Pre-registration for this experiment is in the output directory's notes.md, committed before the
first run.
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
from fka.data.hardening import entity_split  # noqa: E402
from fka.data.multihop import chain_probe_list  # noqa: E402
from fka.eval.query_diagnostics import QueryConfusion  # noqa: E402
from fka.kernel.checkpoint import save_checkpoint  # noqa: E402
from fka.kernel.gpu_lock import gpu_lock  # noqa: E402
from fka.kernel.latent_episodes import (  # noqa: E402
    d3_tokenizer,
    episode_from_probe,
    one_hop_episode,
    pack,
)
from fka.kernel.latent_kernel import LatentKernelConfig, LatentReasoningKernel  # noqa: E402
from fka.kernel.latent_memory import LatentCodebook, OracleLatentMemory  # noqa: E402
from fka.kernel.latent_train import CodecBatcher, D3TrainConfig, evaluate_d3, train_d3  # noqa: E402
from fka.kernel.model import KERNEL_SIZES  # noqa: E402
from fka.kernel.seeding import seed_everything  # noqa: E402

#: Hardened D1, same corpus and seed — the line D3 is reported against.
D1_BASELINE = {"composition_3hop": 0.975, "routing_failures": 5, "leakage_excess": -0.025}


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def build_data(corpus, tokenizer, split, hops, fact_index, block_size, seed):
    name_to_id = {n: i for i, n in enumerate(corpus.entity_names)}
    train_entities = set(split.train.tolist())
    rng = np.random.default_rng(seed)

    groups_train, heldout_eps, heldout_meta = [], [], []
    # 1-hop episodes teach the readout and single-step addressing.
    one_hop = [
        one_hop_episode(corpus, e, r)
        for e in split.train
        for r in ("birth_year", "birth_city", "employer")
    ]
    groups_train.append(pack(one_hop, tokenizer, block_size, fact_index))

    for n_hops in hops:
        probes = chain_probe_list(corpus, n_hops)
        rng.shuffle(probes)
        train = [p for p in probes if name_to_id[p.subject] in train_entities]
        held = [p for p in probes if name_to_id[p.subject] not in train_entities]
        groups_train.append(
            pack([episode_from_probe(p, corpus, name_to_id) for p in train],
                 tokenizer, block_size, fact_index)
        )
        if n_hops == max(hops):
            heldout_eps = [episode_from_probe(p, corpus, name_to_id) for p in held]
            heldout_meta = held
    return groups_train, heldout_eps, heldout_meta


def main(argv: list[str] | None = None) -> int:
    """Entry point. Device work is wrapped in the box's single-GPU-job lock."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256])
    p.add_argument("--size", default="10M")
    p.add_argument("--n-entities", type=int, default=2000)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hops", type=int, nargs="+", default=[2, 3])
    p.add_argument("--entity-holdout", type=float, default=0.2)
    p.add_argument("--n-eval", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None)
    p.add_argument("--codec", action="store_true", help="fallback (a): context-free codec loss")
    p.add_argument("--no-amp", action="store_true", help="fp32 training; isolates bf16 effects")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--checkpoint-every", type=int, default=500,
                   help="rolling mid-run checkpoints; 0 disables")
    p.add_argument("--checkpoint-keep", type=int, default=3)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force-gpu-lock", action="store_true",
                   help="take the GPU lock even if another pid holds it (see fka.kernel.gpu_lock)")
    args = p.parse_args(argv)

    if args.smoke:
        args.dims, args.size, args.n_entities = [32], "tiny", 60
        args.steps, args.batch_size, args.n_eval = 3, 4, 4
        args.hops = [2]
        args.device = "cpu" if args.device == "auto" else args.device

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    if device.type == "cuda":
        stack = gpu_lock(force=args.force_gpu_lock)
        stack.__enter__()

    # Before ANY module is constructed: model init draws from the global torch RNG, and
    # train_d3's own manual_seed comes too late to cover it.
    seed_everything(args.seed)

    corpus = generate_corpus(
        CorpusConfig(n_entities=args.n_entities, seed=args.seed, n_coworkers=1)
    )
    tokenizer = d3_tokenizer()
    split = entity_split(corpus, fraction=args.entity_holdout, seed=args.seed)

    print(f"== D3 sweep on {device}")
    print(f"   corpus       {corpus!r}  fingerprint {corpus.fingerprint()[:16]}")
    print(f"   entities     {split.n_train:,} train / {split.n_heldout:,} HELD OUT")
    print(f"   vocab        {tokenizer.vocab_size}   dims {args.dims}")

    results = {}
    for dim in args.dims:
        print(f"\n########## latent_dim = {dim} ##########")
        codebook = LatentCodebook.build(corpus, dim=dim, seed=args.seed)
        memory = OracleLatentMemory(corpus, codebook).to(device)
        block_size = 128
        groups, heldout_eps, heldout_meta = build_data(
            corpus, tokenizer, split, args.hops, memory.fact_index, block_size, args.seed
        )

        cfg = LatentKernelConfig(
            vocab_size=tokenizer.vocab_size, block_size=block_size,
            latent_dim=dim, n_read_heads=1, cross_attn_every=2,
            **KERNEL_SIZES[args.size],
        )
        model = LatentReasoningKernel(cfg).to(device)
        print(f"   params {model.n_params():,}")

        tcfg = D3TrainConfig(
            steps=args.steps, batch_size=args.batch_size, seed=args.seed,
            amp=not args.no_amp, lr=args.lr, grad_clip=args.grad_clip,
            checkpoint_every=args.checkpoint_every, checkpoint_keep=args.checkpoint_keep,
        )
        codec = CodecBatcher(memory, tokenizer, block_size) if args.codec else None
        state = train_d3(
            model, memory, groups, tcfg, device=device, codec=codec,
            checkpoint_dir=(Path(args.out) if args.out and not args.smoke else None),
            checkpoint_extra={
                "latent_dim": dim, "codebook_seed": args.seed,
                "corpus_fingerprint": corpus.fingerprint(), "block_size": block_size,
            },
        )
        print(f"   {state.tokens_per_sec:,.0f} tokens/sec over {state.seconds:.1f}s")

        # Policy (CLAUDE.md): every real run persists weights + config + RNG state. The
        # diagnostics you want are almost never the ones you planned.
        if args.out and not args.smoke:
            ckpt = save_checkpoint(
                Path(args.out) / f"checkpoint_dim{dim}.pt",
                model,
                model_config=cfg,
                train_config=tcfg,
                extra={
                    "latent_dim": dim,
                    "codebook_seed": args.seed,
                    "corpus_fingerprint": corpus.fingerprint(),
                    "entity_split": split.to_dict(),
                    "git_hash": git_hash(),
                    "block_size": block_size,
                },
            )
            print(f"   checkpoint -> {ckpt.name}")

        eval_eps = heldout_eps[: args.n_eval]
        packed = pack(eval_eps, tokenizer, block_size, memory.fact_index)
        amp = torch.bfloat16 if device.type == "cuda" else None
        res, diag = evaluate_d3(model, memory, packed, tokenizer, device=device, amp_dtype=amp)
        conf = QueryConfusion.from_samples(
            diag["margins"], diag["correct_sim"], diag["best_wrong_sim"]
        )
        gap = res.accuracy - D1_BASELINE["composition_3hop"]
        print(
            f"   {max(args.hops)}-hop entity-held-out: {res.accuracy:.1%} "
            f"(D1 {D1_BASELINE['composition_3hop']:.1%}, gap {gap:+.1%})"
        )
        print(
            f"   failures: routing {res.n_routing_failures}, copy {res.n_copy_failures}, "
            f"format {res.n_format_failures}   retrieval {res.retrieval_accuracy:.1%} "
            f"per-hop {[f'{v:.1%}' for v in res.per_hop_retrieval]}"
        )
        print(f"   query space: {conf}")

        results[str(dim)] = {
            "latent_dim": dim,
            "n_params": model.n_params(),
            "train": state.to_dict(),
            "eval": res.to_dict(),
            "query_confusion": conf.to_dict(),
            "gap_vs_d1": gap,
        }

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "sweep.json").write_text(
            json.dumps(
                {
                    "git_hash": git_hash(),
                    "device": str(device),
                    "corpus": corpus.summary(),
                    "entity_split": split.to_dict(),
                    "d1_baseline": D1_BASELINE,
                    "results": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {out}/sweep.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


