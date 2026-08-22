#!/usr/bin/env bash
# K*_syllable(10M) COARSE SWEEP, second attempt — fixed stream, gate already passed (M5 §5.71).
#
# Prerequisites, both discharged before this runs:
#   * trainability gate: 10M @ 16,000 keys, fixed stream = 93.27% corrected (>= 30%). PASSED.
#   * re-anchor:         K*_syllable(1M) = 21,451 keys = 43.9 params/key, fixed stream.
#
# Envelopes RE-DERIVED from the new anchor at a 10.86x parameter ratio (forms unchanged, registered
# blind in §5.51; only the numeric values move because the anchor was re-measured — §5.67.3):
#
#     linear 232,901  |  sqrt(P) 70,682  |  log(P) 25,170  |  flat 21,451
#
# The three arms discriminate all four, and — noted rather than relied on — they land at the SAME
# key counts the first attempt used, because the anchor moved only 8.6%:
#
#     32,000   above flat (21,451) and log (25,170), below sqrt (70,682)  -> alive iff >= sqrt
#     96,000   above sqrt, below linear (232,901)                          -> alive iff >= linear
#    288,000   above linear                                                -> alive iff super-linear
#
# Everything else inherited verbatim from §5.58: syllable surface at 2 units, fixed value spaces
# 23/32/10 => 4.2818 bits/fact, literature recipe, 8192x8192 name pool, step-matched at ~23,000,
# per-relation splits before any pooled figure, chance floors recovered per arm.
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
    || { echo "ARM $1 FAILED — sweep aborted, no completion marker"; return 1; }
}

arm  32000 268 0.08 || exit 1
arm  96000  89 0.03 || exit 1
arm 288000  29 0.01 || exit 1
echo "K-STAR 10M COARSE SWEEP 2 COMPLETE"
