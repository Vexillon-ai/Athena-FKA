#!/usr/bin/env bash
# FINE BRACKET for K*_syllable(10M) (M5 §5.76).
#
# The coarse sweep brackets it at 16,000 (93.27%) -> 32,000 (13.78%), giving K* = 30,311 under the
# frozen criterion. That bracket spans **2x** with a **79-point** drop across it, which is far too
# coarse to quote: log-linear interpolation over a 2x span assumes a shape the two endpoints cannot
# constrain, and the criterion's +-15% threshold-insensitivity property **could not be checked at
# the 10% level on this pair** (both arms sit above 10%, so it does not bracket).
#
# Two arms, placed to tighten the bracket from either side of the estimate:
#
#   24,000   geometric-ish midpoint of 16,000-32,000; also the key count where the 1M ladder has a
#            measured point (10.62%), so it doubles as a same-key cross-size comparison
#   28,000   just below the estimate, so that if it is alive the final pair is 28,000-32,000 — a
#            **1.14x** span rather than 2x
#
# Expected from K* = 30,311: both arms ALIVE, final bracket 28,000-32,000. If 24,000 comes back
# DEAD the estimate was high and the final bracket is 16,000-24,000 — either way the span tightens
# by at least 33% and the shape between the endpoints becomes measured rather than assumed.
#
# Same frozen criterion (§5.50), same everything else as the coarse sweep (§5.58): syllable surface
# at 2 units, value spaces 23/32/10 => 4.2818 bits/fact, literature recipe, 8192x8192 name pool,
# fixed stream, step-matched at ~23,000. ~0.83 h per arm.
#
# The completion marker is reachable only through success (CLAUDE.md 2e).
set -u
cd "$(dirname "$0")/.."
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
OUT=experiments/2026-08-03_dense-kstar-10M-rerun
R=birth_year,birth_city,employer
COMMON="--mode ladder --sizes 10M --relations $R --surface syllable --n-cities 23 \
  --n-employers 32 --birth-year-range 1990 2000 --n-given-names 8192 --n-surnames 8192 \
  --max-probes 1500 --log-every 4000 --lr 0.001 --weight-decay 0.02 --warmup-steps 1000"

arm () {  # keys exposures probe_fraction
  .venv/Scripts/python.exe -u scripts/run_dense_baseline.py $COMMON \
    --n-entities "$1" --ladder "$2" --probe-fraction "$3" --out "$OUT/10M_K$1" \
    || { echo "ARM $1 FAILED — fine bracket aborted, no completion marker"; return 1; }
}

arm 24000 357 0.10 || exit 1
arm 28000 306 0.09 || exit 1
echo "K-STAR 10M FINE BRACKET COMPLETE"
