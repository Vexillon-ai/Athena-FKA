"""Recover a ladder's per-rung results from its log.

The ladder writes its JSON only when every rung completes, so a run that is paused — as the
E = 1024 rung was, deliberately, to let a validity check run ahead of it — leaves its finished
rungs recorded only in the log. This turns those back into a machine-readable artifact, so the
record does not depend on a run reaching its end.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

RUNG = re.compile(r"== exposures=(\d+)\s+([\d,]+) steps")
STEP = re.compile(r"step\s+([\d,]+)/([\d,]+).*loss ([\d.]+)")
POOLED = re.compile(
    r"(qa_heldout|qa_trained)\s+POOLED\s+n=\s*([\d,]+)\s+acc\s+([\d.]+)%\s+"
    r"corrected\s+([\d.]+)%\s+bits/param\s+([\d.]+)"
)
LEGACY = re.compile(
    r"(qa_heldout|qa_trained)\s+n=([\d,]+)\s+acc ([\d.]+)%\s+corrected ([\d.]+)%\s+"
    r"bits/param ([\d.]+)"
)


def _int(text: str) -> int:
    return int(text.replace(",", ""))


def parse(log: str) -> list[dict]:
    rows: list[dict] = []
    current: dict | None = None
    for line in log.splitlines():
        rung = RUNG.search(line)
        if rung:
            current = {"exposures": _int(rung.group(1)), "planned_steps": _int(rung.group(2))}
            rows.append(current)
            continue
        if current is None:
            continue
        step = STEP.search(line)
        if step:
            current["steps_done"] = _int(step.group(1))
            current["loss"] = float(step.group(3))
        for pattern in (POOLED, LEGACY):
            hit = pattern.search(line)
            if hit:
                current[hit.group(1)] = {
                    "n": _int(hit.group(2)),
                    "accuracy": float(hit.group(3)) / 100,
                    "corrected": float(hit.group(4)) / 100,
                    "bits_per_param": float(hit.group(5)),
                }
                break
    for row in rows:
        row["complete"] = "qa_heldout" in row and row.get("steps_done") == row["planned_steps"]
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log")
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)

    rows = parse(Path(args.log).read_text(encoding="utf-8", errors="replace"))
    header = f"{'E':>6} {'steps':>10} {'loss':>8} {'heldout':>9} {'trained':>9} {'bits/param':>11}"
    print(header)
    for row in rows:
        held, trained = row.get("qa_heldout"), row.get("qa_trained")
        if not held:
            print(
                f"{row['exposures']:>6} {row.get('steps_done', 0):>10,} "
                f"{row.get('loss', float('nan')):>8.4f} {'PAUSED':>9} {'-':>9} {'-':>11}"
            )
            continue
        print(
            f"{row['exposures']:>6} {row.get('steps_done', 0):>10,} {row.get('loss', 0):>8.4f} "
            f"{held['accuracy']:>8.2%} {trained['accuracy'] if trained else 0:>8.2%} "
            f"{held['bits_per_param']:>11.4f}"
        )
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
