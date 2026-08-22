"""Turn a K* sweep's log into the verdict, under the criterion frozen in M5 §5.50.

Reports, in this order and deliberately so:

1. **per-relation splits, before any pooled figure** — a single learned component spanning
   heterogeneous classes has a per-class competence profile, and the pooled figure is a statement
   about the sampling mix as much as about the model (CLAUDE.md);
2. the **recovered chance floor per class**, cross-checked against the value space it must equal —
   a free instrument check that would catch a corrected-accuracy sign or denominator error;
3. the pooled corrected accuracy and the located `K*`;
4. the **threshold-sensitivity table**, so the "level is not load-bearing" claim travels with the
   number rather than being asserted once in M5 5.50.2;
5. the **envelope verdict** against the laws registered in §5.51, before the measurement existed.

Usage:  python scripts/summarise_kstar.py experiments/<run>/run.log --params 10220160
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fka.eval.kstar import Arm, kstar, threshold_sensitivity  # noqa: E402

CORPUS = re.compile(r"KnowledgeCorpus\(n_entities=([\d,]+), n_facts=([\d,]+), total_bits=([\d,]+)")
ROW = re.compile(
    r"(qa_heldout|qa_trained)\s+(\w+)\s+n=\s*([\d,]+)\s+acc\s+([\d.]+)%\s+corrected\s+([\d.]+)%"
)
POOLED = re.compile(
    r"(qa_heldout|qa_trained)\s+POOLED\s+n=\s*([\d,]+)\s+acc\s+([\d.]+)%\s+"
    r"corrected\s+([\d.]+)%\s+bits/param\s+([\d.]+)"
)

# The value spaces the sweep fixes (M5 §5.58), and therefore the chance floor each relation owes.
EXPECTED_CHANCE = {"birth_year": 1 / 10, "birth_city": 1 / 23, "employer": 1 / 32}

# Registered before any 10M arm ran (M5 §5.51), at a 10.86x parameter ratio over the 1M anchor.
ENVELOPES = {"linear": 214_500, "sqrt(P)": 65_100, "log(P)": 23_200, "flat": 19_752}
ANCHOR_KEYS, ANCHOR_PARAMS = 19_752, 941_312


def _int(text: str) -> int:
    return int(text.replace(",", ""))


def parse(log: str) -> list[dict]:
    """One record per key-count arm, in log order."""
    arms: list[dict] = []
    for line in log.splitlines():
        corpus = CORPUS.search(line)
        if corpus:
            arms.append(
                {
                    "keys": _int(corpus.group(1)),
                    "facts": _int(corpus.group(2)),
                    "bits": _int(corpus.group(3)),
                    "relations": {},
                }
            )
            continue
        if not arms:
            continue
        pooled = POOLED.search(line)
        if pooled:
            arms[-1][pooled.group(1)] = {
                "n": _int(pooled.group(2)),
                "accuracy": float(pooled.group(3)) / 100,
                "corrected": float(pooled.group(4)) / 100,
                "bits_per_param": float(pooled.group(5)),
            }
            continue
        row = ROW.search(line)
        if row and row.group(2) != "POOLED":
            arms[-1]["relations"].setdefault(row.group(1), {})[row.group(2)] = {
                "n": _int(row.group(3)),
                "accuracy": float(row.group(4)) / 100,
                "corrected": float(row.group(5)) / 100,
            }
    return arms


def recovered_chance(accuracy: float, corrected: float) -> float | None:
    """Invert corrected = (acc - chance) / (1 - chance). A free check on the whole probe path.

    **Uninformative when corrected is CLAMPED at zero.** A model scoring at or below chance reports
    corrected = 0, and the inversion then returns the raw accuracy rather than the floor — so a
    below-chance arm would read as a chance-floor "mismatch" that is really just the clamp. Returns
    None there, because a check that fires on its own clamp is a check that cries wolf on every dead
    arm, which is most of this sweep.
    """
    if corrected >= 1.0 or corrected <= 0.0:
        return None
    return (accuracy - corrected) / (1 - corrected)


def report(arms: list[dict], n_params: int, surface: str, split: str) -> None:
    print(f"\n=== per-relation, BEFORE any pooled figure ({split}, surface {surface}) ===")
    for arm in arms:
        per = arm["relations"].get(split, {})
        if not per:
            continue
        print(f"\n  keys {arm['keys']:>7,}   load {arm['bits'] / (2 * n_params):.3f}x")
        for relation, row in sorted(per.items()):
            chance = recovered_chance(row["accuracy"], row["corrected"])
            owed = EXPECTED_CHANCE.get(relation)
            flag = ""
            if chance is not None and owed is not None:
                flag = " OK" if abs(chance - owed) < 0.01 else f" MISMATCH (owes {owed:.3f})"
            shown = f"{chance:.4f}" if chance is not None else "   n/a"
            print(
                f"    {relation:<12} n={row['n']:>5,}  acc {row['accuracy']:>7.2%}"
                f"  corrected {row['corrected']:>7.2%}  chance {shown}{flag}"
            )

    print(f"\n=== pooled ({split}) ===")
    for arm in arms:
        row = arm.get(split)
        if row:
            print(
                f"  keys {arm['keys']:>7,}  n={row['n']:>5,}  acc {row['accuracy']:>7.2%}"
                f"  corrected {row['corrected']:>7.2%}  bits/param {row['bits_per_param']:.4f}"
            )

    points = [Arm(a["keys"], a[split]["corrected"]) for a in arms if split in a]
    if len(points) < 2:
        print("\n  (fewer than two completed arms — K* cannot be located)")
        return
    located = kstar(points, surface=surface, n_params=n_params)
    print(f"\n=== K* ===\n  {located}")
    if located.non_monotone:
        print(
            f"  MIS-SPECIFICATION FLAG (M5 5.55): more keys read materially better at "
            f"{located.non_monotone} — route to the M5 5.56/5.57 confounds, do not fit a law."
        )

    table = threshold_sensitivity(points, surface=surface, n_params=n_params)
    if table:
        print("\n=== threshold sensitivity (the level is a reporting convention, M5 5.50.2) ===")
        print("  " + "  ".join(f"{level:.0%}: {value:>9,.0f}" for level, value in sorted(table.items())))

    if located.value is None:
        print("\n  K* unlocated -> no envelope verdict; the arms did not bracket the cliff.")
        return

    print(f"\n  params per discriminable key: {n_params / located.value:.1f}"
          f"   (anchor at 1M: {ANCHOR_PARAMS / ANCHOR_KEYS:.1f})")

    if n_params == ANCHOR_PARAMS:
        print("  (this IS the anchor — no envelope verdict against itself)")
        return

    print("\n=== envelope verdict, against M5 5.51 registered before the measurement ===")
    for name, predicted in sorted(ENVELOPES.items(), key=lambda kv: -kv[1]):
        ratio = located.value / predicted
        print(f"  {name:<8} predicted {predicted:>9,}   measured/predicted {ratio:>6.2f}x")
    import math

    exponent = math.log(located.value / ANCHOR_KEYS) / math.log(n_params / ANCHOR_PARAMS)
    print(f"\n  implied two-point exponent: K* ~ P^{exponent:.3f}")
    print(f"  3M prediction from that law (M5 5.59): {ANCHOR_KEYS * (2_887_296 / ANCHOR_PARAMS) ** exponent:,.0f}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("log", type=Path)
    p.add_argument("--params", type=int, required=True)
    p.add_argument("--surface", default="syllable")
    p.add_argument("--split", default="qa_heldout", choices=("qa_heldout", "qa_trained"))
    args = p.parse_args()

    arms = parse(args.log.read_text(encoding="utf-8", errors="replace"))
    if not arms:
        print("no arms found in log")
        return 1
    report(arms, args.params, args.surface, args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
