#!/usr/bin/env bash
# Pin the 1M P-curve crossing: 18,000 keys x 3 seeds, fixed stream (M5 §5.104).
#
# The anchor has moved DOWN twice as its bracket tightened — 21,451 (point-estimate definition) ->
# 19,596 (16k-24k bracket) -> 17,889 (16k-20k bracket) — each time landing near the LOW edge of the
# bracket containing it. This arm tests whether that is real compression or an artifact of dead
# draws, and it is the cheapest remaining way to tighten alpha's denominator (0.19 h/arm).
#
# ALL THREE OUTCOMES PRE-WORDED (M5 §5.104.1), so none can be narrated after the fact:
#
#   DEAD  (P <= 1/3)  crossing pinned to [16,000, 18,000]; the downward drift was REAL COMPRESSION,
#                     K*_P(1M) ~ 16,971, alpha_min rises again
#   ALIVE (P >= 2/3)  crossing in [18,000, 20,000]; the midpoint RISES to ~18,974 and the drift was
#                     a DEAD-DRAW ARTIFACT — three coarse brackets each happening to sit high
#   SPLIT (seeds straddle 20%)  the first INTERIOR point on the 1M P-curve. The crossing then
#                     INTERPOLATES rather than being bracketed, at +-0.17 in P at n = 3, and the
#                     anchor becomes a curve rather than collapsing to a point. This would also be
#                     the first 1M arm to show run-level stochasticity, which would put the
#                     variance-workhorse certification (§5.102.1) back under review.
#
# Until this lands the anchor is quoted WITH ITS BRACKET ATTACHED.
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
    --n-entities 18000 --ladder 476 --probe-fraction 0.12 --seed "$1" \
    --out "$OUT/1M_K18000_s$1" \
    || { echo "ARM 18000 seed $1 FAILED — aborted, no completion marker"; return 1; }
}

arm 0 || exit 1
arm 1 || exit 1
arm 2 || exit 1
echo "1M K=18,000 THREE-SEED ARM COMPLETE"
