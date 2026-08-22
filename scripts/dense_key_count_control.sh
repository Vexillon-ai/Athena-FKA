#!/usr/bin/env bash
# CONFOUND CONTROL for the key-discrimination verdict (M5 §5.26).
#
# The §5.15 discriminator compared `birth_year`-only at 31,686 keys against four relations at
# 8,000 keys: keys varied, but so did the relation mix, so "one-relation readout is intrinsically
# harder" was not excluded. This closes that.
#
#   ARM A   8,000 keys   205,151 bits   3 relations, default value spaces
#   ARM B  16,000 keys   205,528 bits   3 relations, value spaces shrunk to halve per-fact bits
#
# Value-load matched to 0.2%, relation mix identical, same tokenizer, same recipe. Only the KEY
# COUNT moves (2x).
#
# `works_with` is excluded from both arms because its value space **is** the entity set, so its
# entropy tracks N and cannot be held fixed while the key count changes — the one relation that
# makes this control impossible to construct.
#
# Exposures are matched at E=256, so arm B receives ~2x the optimisation steps. That is deliberate
# and CONSERVATIVE: the extra compute favours the high-key arm, i.e. works against the hypothesis
# under test.
set -u
cd "$(dirname "$0")/.."
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
OUT=experiments/2026-08-02_dense-key-count-control
R=birth_year,birth_city,employer

.venv/Scripts/python.exe -u scripts/run_dense_baseline.py --mode ladder --sizes 1M \
  --n-entities 8000 --relations "$R" --ladder 256 --probe-fraction 0.2 --max-probes 1500 \
  --log-every 8000 --lr 0.001 --weight-decay 0.02 --warmup-steps 1000 --out "$OUT/A_8000keys"

.venv/Scripts/python.exe -u scripts/run_dense_baseline.py --mode ladder --sizes 1M \
  --n-entities 16000 --relations "$R" --ladder 256 --probe-fraction 0.1 --max-probes 1500 \
  --log-every 8000 --lr 0.001 --weight-decay 0.02 --warmup-steps 1000 \
  --n-cities 23 --n-employers 32 --birth-year-range 1990 2000 --out "$OUT/B_16000keys"

echo "KEY-COUNT CONTROL COMPLETE"
