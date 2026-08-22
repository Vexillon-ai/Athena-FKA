#!/usr/bin/env bash
# Locate K*_syllable(1M) — the discriminable-key cliff on the FAIR surface (M5 §5.47).
#
# The near-cliff discriminator leaves the syllable surface still at 34.77% corrected at 16,000
# keys, so K*_syl(1M) > 16,000 and is unlocated. The 10M ladder needs a 1M anchor to predict its
# own bracket from, and guessing that bracket at 49 min/arm is the expensive way to find it.
#
#   keys      16,000 / 24,000 / 32,000 / 48,000
#   surface   syllable (the fair surface; every K* quote carries its surface — §5.46)
#   values    FIXED per-fact spaces (23 cities, 32 employers, 10 years) across all arms
#   recipe    literature-matched
#   compute   ~23,000 steps every arm; E set per K so steps match
#
# **Value-load is NOT matched across arms here, and that is deliberate.** Holding total load fixed
# over a 3x key range would need per-fact spaces of ~1.4 bits — four-value relations with a 25%
# chance floor, where corrected accuracy is mostly noise. Instead per-fact content is held fixed and
# load is REPORTED per point. This is sound only because the matched-load control (§5.26) already
# established key count as causal at 8k vs 16k; this sweep locates the cliff, it does not re-prove
# causation.
set -u
cd "$(dirname "$0")/.."
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
OUT=experiments/2026-08-03_dense-kstar-1M
R=birth_year,birth_city,employer
COMMON="--mode ladder --sizes 1M --relations $R --surface syllable --n-cities 23 \
  --n-employers 32 --birth-year-range 1990 2000 --max-probes 1500 --log-every 8000 \
  --lr 0.001 --weight-decay 0.02 --warmup-steps 1000"

# The completion marker is a PREDICATE, not an epilogue (CLAUDE.md 2c) — see dense_kstar_10M.sh.
arm () {  # keys exposures probe_fraction
  .venv/Scripts/python.exe -u scripts/run_dense_baseline.py $COMMON \
    --n-entities "$1" --ladder "$2" --probe-fraction "$3" --out "$OUT/K$1" \
    || { echo "ARM $1 FAILED — sweep aborted, no completion marker"; return 1; }
}

arm 16000 535 0.15 || exit 1
arm 24000 357 0.10 || exit 1
arm 32000 268 0.08 || exit 1
arm 48000 177 0.06 || exit 1
echo "K-STAR 1M COMPLETE"
