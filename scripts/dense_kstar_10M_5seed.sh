#!/usr/bin/env bash
# 5-SEED CLIFF ARMS at 10M — narrow K*_P(10M) (M5 §5.89, §5.114).
#
# Seeds 0,1,2 are already run at both key counts:
#   28,000  0.85 / 64.34 / 72.86   P = 2/3   sigma 39.35   <- genuine bimodality
#   32,000  13.78 / 11.45 / 34.69  P = 1/3   sigma 12.80
#
# Adding seeds 3 and 4 takes P to 1/5 resolution at both, which is what narrows alpha's NUMERATOR.
#
# BRANCH SCHEME, repaired and audited at n=5 before this runs (§5.114) — the earlier
# DEAD/ALIVE/SPLIT boundaries were ambiguous at 1/5 and 4/5:
#
#     unanimous dead   P = 0/5
#     unanimous alive  P = 5/5
#     interior         P in {1/5, 2/5, 3/5, 4/5}  -> the crossing INTERPOLATES between key counts
#
# P = 0.5 is NOT attainable at odd n, so the crossing is never read off a single arm.
# Every interior P is reported WITH its arm's sigma (§5.113.2): P alone cannot tell genuine
# bimodality (28,000, sigma 39.4) from threshold proximity (1M @ 18,000, sigma 5.04).
#
# ~0.83 h/arm x 4 arms = ~3.3 h. Completion marker reachable only through success (CLAUDE.md 2e).
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
    || { echo "ARM $1 seed $4 FAILED — aborted, no completion marker"; return 1; }
}

arm 28000 306 0.09 3 || exit 1
arm 28000 306 0.09 4 || exit 1
arm 32000 268 0.08 3 || exit 1
arm 32000 268 0.08 4 || exit 1
echo "10M CLIFF 5-SEED COMPLETE"
