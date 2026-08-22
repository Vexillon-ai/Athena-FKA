"""Replay the codec batch stream and ask when each degenerate shape FIRST appears.

    python scripts/probe_codec_composition.py --steps 4000 --failure-step 3110

The 50M divergence is a non-finite *gradient* under a finite forward, localised to the codec
loss's backward (experiments/2026-08-02_d3-50M-stability/notes.md). The mechanism is unknown, and
this is the cheapest way to name it: **bookkeeping, not training.**

Codec batch composition is a pure function of the RNG stream — `train_d3` draws in a fixed order
(dataset pick, episode indices, the codec coin flip, then the codec batch itself), so the exact
sequence of batches can be replayed with **no model, no GPU, and no gradient**, in seconds.

If some degenerate shape first occurs at exactly the failing step, that coincidence names the
mechanism by bookkeeping and the driver hypothesis closes — a batch-composition-determined failure
is numerics, not a kernel defect. If the degenerate shape occurs from step ~1, it cannot explain a
failure at step 3110 and the cost-ordered probes take over.

Nothing here needs to be right about *why* a shape is dangerous. It only needs to report first
occurrences honestly.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fka.data.corpus_gen import CorpusConfig, generate_corpus  # noqa: E402
from fka.data.hardening import entity_split  # noqa: E402
from fka.kernel.latent_episodes import d3_tokenizer  # noqa: E402
from fka.kernel.latent_memory import LatentCodebook, OracleLatentMemory  # noqa: E402
from fka.kernel.latent_train import CodecBatcher, D3TrainConfig  # noqa: E402
from fka.kernel.seeding import seed_everything  # noqa: E402
from scripts.run_d3_kernel import build_data  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--failure-step", type=int, default=3110)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    seed_everything(args.seed)
    device = torch.device("cpu")
    corpus = generate_corpus(CorpusConfig(n_entities=2000, seed=args.seed, n_coworkers=1))
    tokenizer = d3_tokenizer()
    split = entity_split(corpus, fraction=0.2, seed=args.seed)
    memory = OracleLatentMemory(corpus, LatentCodebook.build(corpus, dim=64, seed=args.seed))
    block_size = 128
    groups, _, _ = build_data(
        corpus, tokenizer, split, [2, 3], memory.fact_index, block_size, args.seed
    )
    codec = CodecBatcher(memory, tokenizer, block_size)
    cfg = D3TrainConfig(seed=args.seed, batch_size=args.batch_size)

    # Exactly train_d3's draw order. Any deviation makes this a different stream and the whole
    # probe worthless -- the same class of error as the schedule-horizon confound.
    rng = np.random.default_rng(args.seed)
    first_seen: dict[str, int] = {}
    counts: dict[str, int] = defaultdict(int)
    codec_steps = 0
    per_step = []
    mask_sigs: set = set()

    def note(prop: str, step: int) -> None:
        counts[prop] += 1
        first_seen.setdefault(prop, step)

    for step in range(1, args.steps + 1):
        packed = groups[int(rng.integers(0, len(groups)))]
        rng.integers(0, len(packed.tokens), size=cfg.batch_size)  # episode indices

        if not (rng.random() < cfg.codec_batch_ratio):
            continue
        codec_steps += 1
        cb = codec.batch(rng, cfg.batch_size, device)
        amask = cb["answer_mask"]
        per_row = amask.sum(dim=1)
        lens = per_row.tolist()

        # Degenerate shapes, each named so a coincidence is interpretable rather than suggestive.
        if int((per_row == 0).sum()):
            note("row_with_zero_target_positions", step)
        if min(lens) == 1:
            note("row_with_single_target_position", step)
        if len(set(lens)) == 1:
            note("all_rows_same_length", step)
        if max(lens) >= block_size - codec.prompt_len - 1:
            note("row_at_block_limit", step)

        # The hypothesis under test is the degenerate single-<qvec> EMPTY-ROW shape in the
        # cross-attention mask. That mask is a function of (qvec position, sequence length),
        # both of which come from the constant codec prompt -- so it is measured here per batch
        # rather than assumed constant, because "obviously constant" is how the last three
        # confounds got in.
        T = cb["tokens"].shape[1] - 1
        qpos = int(cb["qvec_pos"][0, 0])
        # A position can read latent 0 only strictly after the qvec position; everything at or
        # before it has nothing readable -> an "empty" row that must be handled without NaN.
        n_empty_rows = qpos + 1
        mask_sigs.add((T, qpos, n_empty_rows, int(cb["subj_pos"][0])))

        per_step.append({"step": step, "min_len": min(lens), "max_len": max(lens),
                         "n_zero": int((per_row == 0).sum())})

    print(f"== codec batch composition over {args.steps} steps")
    print(f"   codec batches drawn : {codec_steps} "
          f"({codec_steps / args.steps:.1%}, ratio {cfg.codec_batch_ratio})")
    print(f"   first codec step    : {per_step[0]['step'] if per_step else 'none'}")
    print(f"   failing step        : {args.failure_step}")
    print("\n   degenerate shape                 first seen    occurrences")
    if not first_seen:
        print("   (none of the tracked shapes ever occurred)")
    for prop, step in sorted(first_seen.items(), key=lambda kv: kv[1]):
        print(f"   {prop:<32} {step:>10}    {counts[prop]:>11}")

    print("")
    print(f"   cross-attention mask signatures across all {codec_steps} codec batches:")
    for sig in sorted(mask_sigs):
        print(f"      T={sig[0]}, qvec_pos={sig[1]}, empty_rows={sig[2]}, subj_pos={sig[3]}")
    if len(mask_sigs) == 1:
        print("      -> INVARIANT. The degenerate empty-row shape is present in EVERY codec")
        print(f"         batch from the first one (step {per_step[0]['step']}), not first at "
              f"step {args.failure_step}.")
        print("         It therefore cannot explain the failure TIMING on its own.")

    coincident = [p for p, s in first_seen.items() if s == args.failure_step]
    print()
    if coincident:
        print(f"   ** COINCIDENT with the failing step: {coincident}")
        print("      Mechanism named by bookkeeping; driver hypothesis closes.")
    else:
        print(f"   No tracked shape first occurs at step {args.failure_step}.")
        print("      Batch composition does not explain the failure timing on its own —")
        print("      fall back to the cost-ordered probes (backward hooks, fp32 codec, anomaly).")

    payload = {
        "steps": args.steps, "failure_step": args.failure_step,
        "codec_steps": codec_steps, "first_seen": first_seen, "counts": dict(counts),
        "coincident": coincident,
        "mask_signatures": sorted(mask_sigs),
        "mask_shape_invariant": len(mask_sigs) == 1,
        "first_codec_step": per_step[0]["step"] if per_step else None,
        "min_len_overall": min((d["min_len"] for d in per_step), default=None),
        "max_len_overall": max((d["max_len"] for d in per_step), default=None),
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
