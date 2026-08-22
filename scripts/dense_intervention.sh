#!/usr/bin/env bash
# THE INTERVENTION (M5 §5.35) — three roles, two arms.
#
# Converse of the key-count control. The control held the surface fixed and varied the key COUNT;
# this holds the key count and the bits fixed and varies the key SURFACE. Together they close the
# causal loop on key discrimination in both directions.
#
#   corpus   birth_year only, 31,686 keys, value load 0.12x   (identical to the char-level run)
#   surface  syllable: 2-unit names from a fixed 608-syllable inventory
#   recipe   literature-matched (lr 1e-3, wd 0.02, warmup 1,000)
#   char-level result to beat: 1.33% raw / 0.34% corrected, against a 1.00% chance floor
#
# TWO arms, because "step-matched" and "exposure-matched" price different things and the ruling
# names both a causal role and a magnitude role:
#
#   A  exposure-matched (E=256, 4,864 steps)   — equal exposures, 4.6x LESS compute than the
#      char-level run. Isolates the SURFACE: any recovery here cannot be bought with training.
#      This is the causal-confirmation arm, and it is conservative.
#   B  step-matched     (E=1184, 22,496 steps) — equal compute to the char-level run, which at
#      4.5x the token density buys 4.6x the exposures. Prices the FULL lift of key-surface
#      fidelity in recall terms, density benefit included.
#
# The gap between them prices the exposure component of the lift separately from the surface.
set -u
cd "$(dirname "$0")/.."
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
OUT=experiments/2026-08-02_dense-intervention
COMMON="--mode ladder --sizes 1M --n-entities 31686 --relations birth_year --surface syllable \
  --probe-fraction 0.25 --max-probes 1500 --lr 0.001 --weight-decay 0.02 --warmup-steps 1000"

.venv/Scripts/python.exe -u scripts/run_dense_baseline.py $COMMON --ladder 256 \
  --log-every 2000 --out "$OUT/A_exposure_matched"

.venv/Scripts/python.exe -u scripts/run_dense_baseline.py $COMMON --ladder 1184 \
  --log-every 8000 --out "$OUT/B_step_matched"

echo "INTERVENTION COMPLETE"
