"""Compare the recipe-audit arms against the incumbent and the modal-value baseline (M5 §5.14-19).

Prints one row per arm. The columns that decide the audit are `heldout` and its distance from the
**modal-value baseline** — the score a model that has stored nothing gets by emitting each
relation's most likely value. Loss alone cannot settle it: the incumbent's loss and arm 2's differ
by less than 0.02 while both sit exactly at that baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fka.data.corpus_gen import generate_corpus  # noqa: E402
from fka.eval.capacity import chance_accuracy  # noqa: E402

#: The incumbent, measured before the audit opened (E=64 rung of the exposure ladder).
INCUMBENT = {"arm": "incumbent", "lr": 1e-3, "wd": 0.1, "warmup": 200,
             "loss": 0.4579, "heldout": 0.0034, "trained": 0.0034}


def modal_baseline(n_entities: int, probe_fraction: float, seed: int = 0) -> float:
    """Accuracy of emitting each relation's value-space mode, weighted by the probe mix."""
    corpus = generate_corpus(
        n_entities=n_entities, seed=seed, probe_fraction=probe_fraction
    )
    pairs = list(corpus.memory_pairs(corpus.probe_ids))
    if not pairs:
        return 0.0
    return sum(chance_accuracy(corpus, p.key[1]) for p in pairs) / len(pairs)


def load_arm(directory: Path) -> dict | None:
    blob = directory / "ladder.json"
    if not blob.exists():
        return None
    rows = json.loads(blob.read_text(encoding="utf-8")).get("ladder", [])
    if not rows or "qa_heldout" not in rows[-1]:
        return None
    row = rows[-1]
    return {
        "arm": directory.name,
        "lr": row.get("args", {}).get("lr"),
        "loss": row.get("final_loss"),
        "heldout": row["qa_heldout"]["accuracy"],
        "trained": row["qa_trained"]["accuracy"],
        "bits_per_param": row["qa_heldout"]["bits_per_param"],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", default="experiments/2026-08-02_dense-recipe-audit")
    p.add_argument("--n-entities", type=int, default=31_686)
    p.add_argument("--probe-fraction", type=float, default=0.06)
    args = p.parse_args(argv)

    base = modal_baseline(args.n_entities, args.probe_fraction)
    print(f"modal-value baseline at N={args.n_entities:,}: {base:.4%}")
    print(f"\n{'arm':>22} {'loss':>8} {'heldout':>9} {'trained':>9} {'x baseline':>11}")
    rows = [INCUMBENT]
    for d in sorted(Path(args.dir).iterdir()):
        if d.is_dir():
            row = load_arm(d)
            if row:
                rows.append(row)
    for row in rows:
        held = row["heldout"]
        print(
            f"{row['arm']:>22} {row.get('loss', 0):>8.4f} {held:>8.2%} "
            f"{row.get('trained', 0):>8.2%} {held / base if base else 0:>10.2f}x"
        )
    print("\n'x baseline' near 1.0 means the arm stored NOTHING and is emitting the mode.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
