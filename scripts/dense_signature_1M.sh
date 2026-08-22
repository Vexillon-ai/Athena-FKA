#!/usr/bin/env bash
# ARM-INDEXED SIGNATURE TEST at 1M — the crossing pair to n = 10 (M5 §5.119).
#
# Ruled at n = 5. A power check run BEFORE wording the branches says n = 5 CANNOT DECIDE:
#
#     n = 5   |r| must exceed 0.878 for p < 0.05
#     n = 10  |r| must exceed 0.632
#
# The 10M correlations that prompted this are r = -0.356 (p = 0.556) and r = -0.788 (p = 0.113).
# NEITHER is significant at n = 5. The statistic is underpowered in BOTH directions and everything
# read from it so far is uninterpretable — including the -0.788 that was reported as supporting
# ill-conditioning.
#
# So the pair goes to n = 10, which is affordable only because these are 1M arms (0.19 h each):
#   18,000 seeds 3-9   |   20,000 seeds 3-9   = 14 runs, ~2.7 h
# Seeds 3 and 4 land first, so the ruled n = 5 point is reached inside the first ~0.8 h and the run
# can be stopped there if the extension is not wanted.
#
# BRANCHES, disjoint and worded before any number (|r| >= 0.632 = "strong" at n = 10):
#   SPLIT   one arm strong, one weak -> arm-indexed account CONFIRMED at a second size;
#                                       provisional findings re-derived under it
#   BOTH STRONG                       -> ill-conditioning takes the field; the asymmetry entry
#                                       (§5.90) likely dies
#   BOTH WEAK                         -> equivalent-optima holds at 1M; the 10M @ 32,000 reading is
#                                       then the anomaly and is re-examined, not generalised
#
# Every P still ships with sigma (§5.113.2). Completion marker only through success (CLAUDE.md 2e).
set -u
cd "$(dirname "$0")/.."
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
OUT=experiments/2026-08-03_dense-kstar-replicate
R=birth_year,birth_city,employer
COMMON="--mode ladder --sizes 1M --relations $R --surface syllable --n-cities 23 \
  --n-employers 32 --birth-year-range 1990 2000 --n-given-names 8192 --n-surnames 8192 \
  --max-probes 1500 --log-every 8000 --lr 0.001 --weight-decay 0.02 --warmup-steps 1000"

arm () {  # keys exposures probe_fraction seed
  .venv/Scripts/python.exe -u scripts/run_dense_baseline.py $COMMON \
    --n-entities "$1" --ladder "$2" --probe-fraction "$3" --seed "$4" \
    --out "$OUT/1M_K$1_s$4" \
    || { echo "ARM $1 seed $4 FAILED — aborted, no completion marker"; return 1; }
}

for s in 3 4; do arm 18000 476 0.12 $s || exit 1; arm 20000 428 0.12 $s || exit 1; done
echo "== n=5 REACHED at both arms (the ruled point) — continuing to n=10"
for s in 5 6 7 8 9; do arm 18000 476 0.12 $s || exit 1; arm 20000 428 0.12 $s || exit 1; done
echo "1M SIGNATURE PAIR n=10 COMPLETE"
