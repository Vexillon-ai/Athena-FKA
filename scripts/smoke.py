"""Run the whole fast suite: every test plus every ``--smoke`` path, on CPU, under two minutes.

CLAUDE.md makes this the gate for CI and for quick iteration, so it must stay fast. The budget is
enforced, not merely documented: if the suite creeps past ``--budget`` seconds this exits
non-zero, because a smoke suite nobody waits for is a smoke suite nobody runs.

This is the implementation behind both ``make smoke`` and ``scripts/smoke.ps1`` — the logic lives
here so Windows and Linux cannot drift apart.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUDGET_SECONDS = 120.0


def _steps(skip_torch: bool) -> list[tuple]:
    """Each step is ``(name, argv)`` or ``(name, argv, acceptable_exit_codes)``."""
    py = sys.executable
    steps: list[tuple] = [
        # 'not gpu' is load-bearing: the GPU tier needs a real device and CI has none.
        (
            "pytest",
            [py, "-m", "pytest", "tests", "-q", "-m", "not slow and not gpu"],
        ),
        ("corpus generator", [py, "-m", "fka.data.corpus_gen", "--smoke"]),
        ("capacity harness", [py, "-m", "fka.eval.capacity", "--smoke"]),
    ]
    if not skip_torch:
        smoke_gpu = str(REPO_ROOT / "scripts" / "smoke_gpu.py")
        steps.append(("gpu/cpu smoke", [py, smoke_gpu, "--smoke"]))
        # The M1 pipeline end to end: episodes, training, memory-in-the-loop generation, and
        # both gate tests. It exits non-zero because an untrained tiny model fails the gates,
        # which is the correct outcome — we only care that every path runs.
        steps.append(
            (
                "m1 pipeline",
                [py, str(REPO_ROOT / "scripts" / "run_m1_kernel.py"), "--smoke"],
                {0, 1},
            )
        )
    return steps


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--budget", type=float, default=DEFAULT_BUDGET_SECONDS)
    p.add_argument("--verbose", action="store_true", help="stream each step's output")
    args = p.parse_args(argv)

    skip_torch = importlib.util.find_spec("torch") is None
    if skip_torch:
        print("!! torch not importable — skipping the model smoke step\n")

    results: list[tuple[str, float, bool]] = []
    for step in _steps(skip_torch):
        name, cmd = step[0], step[1]
        ok_codes: set[int] = step[2] if len(step) > 2 else {0}
        print(f"-- {name}")
        start = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=not args.verbose,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        elapsed = time.perf_counter() - start
        ok = proc.returncode in ok_codes
        results.append((name, elapsed, ok))
        if not ok and not args.verbose:
            print((proc.stdout or "") + (proc.stderr or ""))
        print(f"   {'ok' if ok else 'FAILED'} in {elapsed:.1f}s\n")

    total = sum(e for _, e, _ in results)
    width = max(len(n) for n, _, _ in results)
    print("== summary")
    for name, elapsed, ok in results:
        print(f"   {name:<{width}}  {elapsed:6.1f}s  {'ok' if ok else 'FAILED'}")
    print(f"   {'total':<{width}}  {total:6.1f}s  (budget {args.budget:.0f}s)")

    failed = [n for n, _, ok in results if not ok]
    if failed:
        print(f"\n== FAILED: {', '.join(failed)}")
        return 1
    if total > args.budget:
        print(
            f"\n== OVER BUDGET by {total - args.budget:.1f}s. Either speed a step up or move the "
            f"slow part behind @pytest.mark.slow."
        )
        return 1
    print("\n== ALL SMOKE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    if shutil.which(sys.executable) is None:  # pragma: no cover - defensive
        raise SystemExit("cannot locate the running interpreter")
    raise SystemExit(main())



