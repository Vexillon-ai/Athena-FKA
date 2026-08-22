#!/usr/bin/env bash
# STAGE 2 of the variance measurement (M5 §5.84.2, §5.86) — queued behind the cliff replication.
#
#   MID-LADDER regime  24,000 @ 10M, seeds 1,2   — prices the NON-critical arms, so NOISE_POINTS can
#                                                  be regime-indexed if the two regimes differ
#   1M ANCHOR          16,000 and 24,000 @ 1M, seeds 1,2 — prices K*_syl(1M) = 21,451 at a fifth the
#                                                  cost per arm (§5.86)
#
# ~1.7 h (10M mid-ladder) + ~0.76 h (1M anchor) = ~2.5 h.
set -u
cd "$(dirname "$0")/.."
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
OUT=experiments/2026-08-03_dense-kstar-replicate
R=birth_year,birth_city,employer
COMMON="--relations $R --surface syllable --n-cities 23 --n-employers 32 \
  --birth-year-range 1990 2000 --n-given-names 8192 --n-surnames 8192 \
  --max-probes 1500 --log-every 8000 --lr 0.001 --weight-decay 0.02 --warmup-steps 1000 --mode ladder"

arm () {  # size keys exposures probe_fraction seed
  .venv/Scripts/python.exe -u scripts/run_dense_baseline.py $COMMON --sizes "$1" \
    --n-entities "$2" --ladder "$3" --probe-fraction "$4" --seed "$5" \
    --out "$OUT/$1_K$2_s$5" \
    || { echo "ARM $1/$2 seed $5 FAILED — aborted, no completion marker"; return 1; }
}

arm 10M 24000 357 0.10 1 || exit 1
arm 10M 24000 357 0.10 2 || exit 1
arm 1M  16000 535 0.15 1 || exit 1
arm 1M  16000 535 0.15 2 || exit 1
arm 1M  24000 357 0.10 1 || exit 1
arm 1M  24000 357 0.10 2 || exit 1
echo "VARIANCE STAGE 2 COMPLETE"
