#!/usr/bin/env bash
# K*_syllable(10M), SECOND ATTEMPT — under the fixed stream, gate first (M5 §5.66).
#
# The first attempt was VOID: DenseCorpusStream seeded its shuffle per VARIANT, giving five
# orderings each shown ~107 times byte-identical, and the 10M model memorised the stream instead of
# the facts (§5.63 — subject-token loss +7.41 nats on a fresh permutation against +0.12 at 1M, and
# a WORSE value loss than the 1M model). The shuffle is now seeded per EPOCH.
#
# Arm order is the ruling: the TRAINABILITY GATE runs FIRST, as its own arm, and nothing at the new
# setting may be read until it passes (CLAUDE.md, "trainability is a confound at every scale or
# regime change").
#
#   arm 0   1M  @ 16,000   re-derive the reference under the FIXED stream, like-for-like (11 min).
#                          Also converts §5.64's inheritance claim from inference to measurement:
#                          if this reads ~34.77%, the ordering leak provably never touched the 1M
#                          corpus of results. If arm 0 itself lands < 30%, the fixed stream changed
#                          the task and the sequence STOPS for re-registration.
#   arm 1   10M @ 16,000   THE GATE: >= 30% chance-corrected against arm 0's number.
#                          FAIL -> every 10M arm is void again; do not run arms 2+.
#   arms 2  10M @ 32,000 / 96,000 / 288,000   coarse sweep, design unchanged from §5.58.
#
# Envelopes (§5.51) and the outcome->reconciliation map (§5.55) are UNCHANGED and un-consumed; the
# first attempt's void is noted inline at §5.51 so there is no second-look ambiguity.
#
# Everything else is inherited from §5.58 verbatim: syllable surface at 2 units, fixed value spaces
# 23/32/10 => 4.2818 bits/fact, literature recipe, 8192x8192 name pool at EVERY arm including 1M so
# the comparison is same-world this time (§5.65a), step-matched at ~23,000.
set -u
cd "$(dirname "$0")/.."
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
OUT=experiments/2026-08-03_dense-kstar-10M-rerun
R=birth_year,birth_city,employer
COMMON="--mode ladder --relations $R --surface syllable --n-cities 23 \
  --n-employers 32 --birth-year-range 1990 2000 --n-given-names 8192 --n-surnames 8192 \
  --max-probes 1500 --log-every 4000 --lr 0.001 --weight-decay 0.02 --warmup-steps 1000"

# The completion marker is reachable only through success (CLAUDE.md 2e).
arm () {  # size keys exposures probe_fraction
  .venv/Scripts/python.exe -u scripts/run_dense_baseline.py $COMMON --sizes "$1" \
    --n-entities "$2" --ladder "$3" --probe-fraction "$4" --out "$OUT/$1_K$2" \
    || { echo "ARM $1/$2 FAILED — sequence aborted, no completion marker"; return 1; }
}

arm 1M  16000 535 0.15 || exit 1
echo "== ARM 0 COMPLETE (reference). Gate arm next."
arm 10M 16000 535 0.15 || exit 1
echo "== GATE ARM COMPLETE — read it before the coarse sweep runs."
echo "K-STAR 10M RERUN GATE STAGE COMPLETE"
