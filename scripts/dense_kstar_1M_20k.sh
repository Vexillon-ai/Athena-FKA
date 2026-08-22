#!/usr/bin/env bash
# Bracket the 1M P-curve anchor: 20,000 keys x 3 seeds, fixed stream (M5 §5.102.5).
#
# K*_P(1M) = 19,596 is currently the MIDPOINT OF AN INTERVAL, not a measurement: P jumps
# 1.000 (16,000: 44.49 / 38.67 / 34.61, all alive) -> 0.000 (24,000: 10.62 / 6.23 / 1.76, all dead)
# with nothing in between. A point at 20,000 brackets the 0.5 crossing.
#
# It runs BEFORE the 5-seed 10M cliff arms because it narrows alpha's DENOMINATOR at a tenth the
# numerator's price (0.19 h/arm against 0.83), and because §5.102.1 certified 1M as the variance
# workhorse — sigma_logK agrees at 0.86x, so 1M spread transfers to 10M error bars.
#
# alpha = ln(K10/K1)/2.3848, so tightening K1 tightens the interval directly:
#   currently  K1 in [16,000, 24,000] -> alpha in [0.065, 0.291]
#   if 20,000 is ALIVE (P >= 2/3): crossing moves into [20,000, 24,000], alpha_max falls to ~0.197
#   if 20,000 is DEAD  (P <= 1/3): crossing moves into [16,000, 20,000], alpha_min rises to ~0.098
#   if 20,000 is SPLIT (P = 1/3 or 2/3 with seeds straddling): the crossing is AT 20,000 and the
#                       stochastic zone reaches 1M as well — which would be its own finding, since
#                       every 1M arm replicated so far has been unanimous (3/3 or 0/3).
#
# Everything else inherited verbatim: syllable surface, value spaces 23/32/10, literature recipe,
# 8192x8192 name pool, fixed stream, step-matched. Only --seed varies.
# The completion marker is reachable only through success (CLAUDE.md 2e).
set -u
cd "$(dirname "$0")/.."
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
OUT=experiments/2026-08-03_dense-kstar-replicate
R=birth_year,birth_city,employer
COMMON="--mode ladder --sizes 1M --relations $R --surface syllable --n-cities 23 \
  --n-employers 32 --birth-year-range 1990 2000 --n-given-names 8192 --n-surnames 8192 \
  --max-probes 1500 --log-every 8000 --lr 0.001 --weight-decay 0.02 --warmup-steps 1000"

arm () {  # seed
  .venv/Scripts/python.exe -u scripts/run_dense_baseline.py $COMMON \
    --n-entities 20000 --ladder 428 --probe-fraction 0.12 --seed "$1" \
    --out "$OUT/1M_K20000_s$1" \
    || { echo "ARM 20000 seed $1 FAILED — aborted, no completion marker"; return 1; }
}

arm 0 || exit 1
arm 1 || exit 1
arm 2 || exit 1
echo "1M K=20,000 THREE-SEED ARM COMPLETE"
