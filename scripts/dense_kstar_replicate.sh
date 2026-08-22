#!/usr/bin/env bash
# REPLICATION at the cliff — the mis-specification clause fired (M5 §5.83).
#
# The fine bracket came back NON-MONOTONE far outside the registered band:
#
#     24,000 -> 84.14%   28,000 -> 0.85%   32,000 -> 13.78%
#
# 28,000 reads 12.93 points WORSE than a larger key count, against a +-1.5 point band. Per §5.55 no
# law may be fitted to this sweep. Both registered routes are ELIMINATED: §5.56's exposure
# starvation does not apply (E = 306 sits between the neighbours' 357 and 268) and §5.57's
# name-space saturation does not apply (28,000 of 369,664 = 7.6% full).
#
# What the loss says: training was NORMAL and perfectly ordered in key count —
#
#     16k 0.8565 | 24k 0.9092 | 28k 0.9303 | 32k 0.9719 | 96k 1.1707 | 288k 1.2554
#
# 28,000's loss sits exactly where it belongs between its neighbours. The anomaly is in ACCURACY
# only, which rules out a training failure and points at the third, unregistered cause: the cliff is
# very sharp (84 -> ~1 point over a 1.17x key increase) and arms sitting ON it have large run-to-run
# variance, so a single run per key count cannot locate it.
#
# THE BAND ITSELF WAS NEVER MEASURED. The +-1.5 points came from the 1M ladder's scatter ACROSS key
# counts, not from repeated runs AT one key count — an assumption dressed as a measurement, and the
# instrument this run needs. Every "within noise" judgement in the dense record inherits it.
#
# So: replicate the two arms that disagree, varying ONLY the training seed (model init). The world,
# the corpus, the data ordering and the recipe are all held fixed, so the spread measured here is
# training variance and nothing else.
#
#   READOUT
#     spread small at both -> one of the two runs is an outlier; identify it and the cliff is real
#     spread LARGE at both -> the cliff cannot be located by single runs at all, and the frozen
#                             criterion needs replication built into it before any K* is quotable
#
# 4 arms x 0.83 h = ~3.3 h. The completion marker is reachable only through success (CLAUDE.md 2e).
set -u
cd "$(dirname "$0")/.."
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
OUT=experiments/2026-08-03_dense-kstar-replicate
R=birth_year,birth_city,employer
COMMON="--mode ladder --sizes 10M --relations $R --surface syllable --n-cities 23 \
  --n-employers 32 --birth-year-range 1990 2000 --n-given-names 8192 --n-surnames 8192 \
  --max-probes 1500 --log-every 8000 --lr 0.001 --weight-decay 0.02 --warmup-steps 1000"

arm () {  # keys exposures probe_fraction seed
  .venv/Scripts/python.exe -u scripts/run_dense_baseline.py $COMMON \
    --n-entities "$1" --ladder "$2" --probe-fraction "$3" --seed "$4" \
    --out "$OUT/K$1_s$4" \
    || { echo "ARM $1 seed $4 FAILED — replication aborted, no completion marker"; return 1; }
}

arm 28000 306 0.09 1 || exit 1
arm 28000 306 0.09 2 || exit 1
arm 32000 268 0.08 1 || exit 1
arm 32000 268 0.08 2 || exit 1
echo "K-STAR REPLICATION COMPLETE"
