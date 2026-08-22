#!/usr/bin/env bash
# RE-ANCHOR: K*_syllable(1M) under the FIXED stream (M5 §5.67).
#
# Arm 0 of the re-run measured 1M @ 16,000 keys at 44.49% chance-corrected, against 34.77% under the
# old five-ordering stream — a +9.7 point improvement. The ordering leak was not only a 10M
# catastrophe; it was a real if milder COST at 1M, because five fixed orderings place every fact in
# the same context neighbourhood ~107 times and local context is then a partial crutch. Removing it
# leaves the subject->value association as the only way to reduce loss.
#
# Consequences this ladder exists to settle:
#
#   * K*_syl(1M) = 19,752 was measured under the old stream and is now a LOWER BOUND, not an anchor.
#   * The §5.51 envelopes are FUNCTIONS of that anchor (linear = 10.86 x anchor, and so on). Their
#     FORMS were registered blind and do not change; their NUMERIC VALUES must be re-derived from a
#     fixed-stream anchor. That is the same registration evaluated at a corrected input, not a
#     second look — and it happens BEFORE any valid 10M arm exists, so it cannot be data-fitted.
#   * The 10M coarse sweep's arm placement was chosen from the old anchor and is re-derived with it.
#
# Same ladder as the original anchor (16k / 24k / 32k / 48k), which still brackets the cliff: 16,000
# reads 44.49% and the threshold is 20%. Same frozen criterion (§5.50), same value spaces, same
# recipe, step-matched. 8192x8192 name pool at every arm so this is same-world with the 10M arms
# (§5.65a) — the ORIGINAL anchor used 4096 and was therefore cross-world.
#
# ~0.19 h per arm, ~0.76 h total.
set -u
cd "$(dirname "$0")/.."
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
OUT=experiments/2026-08-03_dense-kstar-1M-rerun
R=birth_year,birth_city,employer
COMMON="--mode ladder --sizes 1M --relations $R --surface syllable --n-cities 23 \
  --n-employers 32 --birth-year-range 1990 2000 --n-given-names 8192 --n-surnames 8192 \
  --max-probes 1500 --log-every 8000 --lr 0.001 --weight-decay 0.02 --warmup-steps 1000"

# The completion marker is reachable only through success (CLAUDE.md 2e).
arm () {  # keys exposures probe_fraction
  .venv/Scripts/python.exe -u scripts/run_dense_baseline.py $COMMON \
    --n-entities "$1" --ladder "$2" --probe-fraction "$3" --out "$OUT/K$1" \
    || { echo "ARM $1 FAILED — ladder aborted, no completion marker"; return 1; }
}

arm 16000 535 0.15 || exit 1
arm 24000 357 0.10 || exit 1
arm 32000 268 0.08 || exit 1
arm 48000 177 0.06 || exit 1
echo "K-STAR 1M RE-ANCHOR COMPLETE"
