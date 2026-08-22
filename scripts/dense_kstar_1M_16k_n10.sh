#!/usr/bin/env bash
# The LAST n=3 arm: 1M @ 16,000 to n=10 (M5 §5.125.5).
#
# 16,000 read P = 3/3 on three seeds and now bounds K*_P(1M) from BELOW. The 18,000 arm went
# 2/3 -> 0.30 when taken from n=3 to n=10, so a 3/3 reading is exactly the resolution that just
# proved unreliable. alpha's lower edge depends on this arm.
#
# 7 runs x 0.19 h = ~1.3 h. Completion marker only through success (CLAUDE.md 2e).
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

for s in 3 4 5 6 7 8 9; do arm 16000 535 0.15 $s || exit 1; done
echo "1M K=16,000 n=10 COMPLETE"
