#!/usr/bin/env bash
# K*_syllable(10M) BRACKET + the positive control the coarse sweep did not have (M5 §5.61).
#
# Arm A of the coarse sweep read 1.12% corrected at 32,000 keys, against the 1M anchor's 2.15% at
# the same point — both dead. That excludes sqrt(P) (65,100) and linear (214,500) outright and puts
# K*_syl(10M) BELOW 32,000, where the coarse sweep placed no arms. The surviving envelopes are
# flat (19,752) and log(P) (23,200), and 16,000 / 24,000 bracket both.
#
# The 16,000 arm does DOUBLE DUTY and runs first:
#
#   POSITIVE CONTROL — the 1M anchor read 34.77% corrected at 16,000 keys under the identical
#   surface, value spaces and recipe. A 10.86x larger model cannot legitimately do worse.
#
#       GATE (fixed before the run, M5 §5.61.2):  corrected >= 30%
#         PASS -> the recipe trains at 10M; arm A is a key-capacity reading; bracket is valid
#         FAIL -> every 10M arm measures the OPTIMISER, not key capacity. The sweep is VOID
#                 pending an LR re-certification at 10M (SIZE_LR says 6e-4; the literature
#                 recipe forces 1e-3, which is what every arm here uses).
#
#   ** If the gate fails, DO NOT run the 24,000 arm. ** There is nothing to bracket.
#
# Everything else is inherited from the coarse sweep unchanged (M5 §5.58): syllable surface at 2
# units, fixed value spaces 23/32/10 => 4.2818 bits/fact, literature recipe, 8192x8192 name pool,
# step-matched at ~23,000. E mirrors the 1M anchor's own counts at these key counts (535 / 357).
set -u
cd "$(dirname "$0")/.."
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
OUT=experiments/2026-08-03_dense-kstar-10M
R=birth_year,birth_city,employer
COMMON="--mode ladder --sizes 10M --relations $R --surface syllable --n-cities 23 \
  --n-employers 32 --birth-year-range 1990 2000 --n-given-names 8192 --n-surnames 8192 \
  --max-probes 1500 --log-every 4000 --lr 0.001 --weight-decay 0.02 --warmup-steps 1000"

arm () {  # keys exposures probe_fraction
  .venv/Scripts/python.exe -u scripts/run_dense_baseline.py $COMMON \
    --n-entities "$1" --ladder "$2" --probe-fraction "$3" --out "$OUT/K$1"
}

arm 16000 535 0.15   # positive control AND lower bracket point — gate >= 30% corrected
arm 24000 357 0.10   # upper bracket point — separates flat from log(P)
echo "K-STAR 10M BRACKET COMPLETE"
