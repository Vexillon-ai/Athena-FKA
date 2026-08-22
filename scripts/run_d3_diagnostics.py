"""Run the three latent-side shortcut diagnostics against a saved D3 checkpoint.

    python scripts/run_d3_diagnostics.py --checkpoint experiments/<run>/checkpoint_dim64.pt

Order and interpretation are pre-registered in docs/decision_records/M1_kernel_interface.md §3b:
(a) decode split localises the gap, (b) latent substitution is decisive and is the permanent
latent-side leakage gate, (c) subject ablation is confounded and read last.
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
    decode_accuracy,
    latent_substitution_test,
    subject_ablation_accuracy,
)
from fka.kernel.checkpoint import load_checkpoint  # noqa: E402
from fka.kernel.latent_episodes import d3_tokenizer, episode_from_probe, pack  # noqa: E402
from fka.kernel.latent_kernel import LatentKernelConfig, LatentReasoningKernel  # noqa: E402
from fka.kernel.latent_memory import LatentCodebook, OracleLatentMemory  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n-probes", type=int, default=200)
    p.add_argument("--hops", type=int, default=3)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    blob = load_checkpoint(args.checkpoint)
    meta, mcfg = blob["extra"], blob["model_config"]
    dim, seed = meta["latent_dim"], meta["codebook_seed"]

    corpus = generate_corpus(CorpusConfig(n_entities=2000, seed=seed, n_coworkers=1))
    if corpus.fingerprint() != meta["corpus_fingerprint"]:
        raise SystemExit(
            "corpus fingerprint does not match the checkpoint — the diagnostics would be run "
            "against a different world than the model was trained on"
        )
    tokenizer = d3_tokenizer()
    split = entity_split(corpus, fraction=0.2, seed=seed)
    codebook = LatentCodebook.build(corpus, dim=dim, seed=seed)
    memory = OracleLatentMemory(corpus, codebook).to(device)

    cfg = LatentKernelConfig(**{k: v for k, v in mcfg.items()})
    model = LatentReasoningKernel(cfg).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()
    amp = torch.bfloat16 if device.type == "cuda" else None

    name_to_id = {n: i for i, n in enumerate(corpus.entity_names)}
    train_entities = set(split.train.tolist())
    probes = chain_probe_list(corpus, args.hops)
    np.random.default_rng(seed).shuffle(probes)
    seen = [p for p in probes if name_to_id[p.subject] in train_entities][: args.n_probes]
    held = [p for p in probes if name_to_id[p.subject] not in train_entities][: args.n_probes]

    def prep(ps):
        eps = [episode_from_probe(p, corpus, name_to_id) for p in ps]
        return pack(eps, tokenizer, meta["block_size"], memory.fact_index)

    seen_packed = prep(seen)
    held_packed = prep(held)

    print(f"== D3 shortcut diagnostics  (dim {dim}, {len(seen)} seen / {len(held)} held-out)")

    print("\n-- (a) decode split")
    acc_seen = decode_accuracy(model, memory, seen_packed, tokenizer, device=device, amp_dtype=amp)
    acc_held = decode_accuracy(model, memory, held_packed, tokenizer, device=device, amp_dtype=amp)
    print(f"   training-entity subjects : {acc_seen:.1%}")
    print(f"   entity-held-out subjects : {acc_held:.1%}")
    print(f"   generalisation gap       : {acc_held - acc_seen:+.1%}")

    print("\n-- (b) latent substitution  [DECISIVE / permanent gate]")
    sub = latent_substitution_test(model, memory, seen_packed, tokenizer,
                                   device=device, amp_dtype=amp, seed=seed)
    print(f"   {sub}")

    print("\n-- (c) subject-path ablation  [confounded — subject code feeds hop-1 queries]")
    abl_seen = subject_ablation_accuracy(model, memory, seen_packed, tokenizer,
                                         device=device, amp_dtype=amp)
    print(f"   training-entity accuracy with subject code blanked: {abl_seen:.1%} "
          f"(was {acc_seen:.1%})")

    payload = {
        "checkpoint": str(args.checkpoint),
        "latent_dim": dim,
        "decode_split": {"seen": acc_seen, "heldout": acc_held, "gap": acc_held - acc_seen},
        "latent_substitution": sub.to_dict(decode_accuracy=acc_seen),
        "subject_ablation": {"seen_accuracy": abl_seen, "baseline": acc_seen},
        "gate_passes": sub.passes,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0 if sub.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())

