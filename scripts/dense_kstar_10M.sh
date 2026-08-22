#!/usr/bin/env bash
# COARSE SWEEP for K*_syllable(10M) — does key capacity scale with parameters? (M5 §5.58)
#
# The 1M anchor is K*_syl(1M) = 19,752 keys = 47.7 params/key, under the criterion frozen blind to
# 10M (§5.50). At a 10.86x parameter ratio the registered envelopes are:
#
#     linear 214,500  |  sqrt(P) 65,100  |  log(P) 23,200  |  flat 19,752
#
# Three arms separate all four:
#
#     32,000   flat / log (dead here) from sqrt / linear (alive)
#     96,000   sqrt (dead) from linear (alive)
#    288,000   linear (dead just past 214K) from super-linear
#
#   surface   syllable — 2 units per name at every arm (units_needed is 2 up to 369,664), so
#             tokens/fact is identical at 14.8 across the sweep and the arms are comparable
#   values    FIXED per-fact spaces (23 cities, 32 employers, 10 years) — 4.2818 bits/fact,
#             identical to 12 significant figures at all three key counts
#   recipe    literature-matched (lr 1e-3, wd 0.02, warmup 1,000)
#   compute   STEP-matched at ~23,000 (23,048 / 23,140 / 22,678), so E falls 9x across the sweep
#   names     8192 x 8192 given/surnames at every arm — 288,000 entities need more than the
#             default 4096^2 to clear the generator's 100x headroom rule. Proved inert: total_bits
#             and bits/fact identical, and SyllableSurface renders from the entity INDEX, so the
#             subject strings are byte-identical (§5.58).
#
# Load is 0.020x / 0.060x / 0.181x of 2 bits/param. **No arm is anywhere near a bits limit** — that
# is the design. Value-load is held fixed per fact and REPORTED per arm rather than matched across
# arms, exactly as in the 1M anchor: matching total load over a 9x key range needs ~0.5-bit value
# spaces, where corrected accuracy is noise. Causation was already established by the matched-load
# control (§5.26); this sweep LOCATES the cliff.
#
# Two confounds are registered with discriminators, to be run only if the 288,000 arm decides the
# verdict: exposure starvation at E=29 (§5.56, 2x-steps re-run) and 2-unit name-space saturation at
# 77.9% full (§5.57, 3-unit re-run). §5.56 runs first — it is cheaper and needs no new code.
set -u
cd "$(dirname "$0")/.."
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
OUT=experiments/2026-08-03_dense-kstar-10M
R=birth_year,birth_city,employer
COMMON="--mode ladder --sizes 10M --relations $R --surface syllable --n-cities 23 \
  --n-employers 32 --birth-year-range 1990 2000 --n-given-names 8192 --n-surnames 8192 \
  --max-probes 1500 --log-every 4000 --lr 0.001 --weight-decay 0.02 --warmup-steps 1000"

# The completion marker is a PREDICATE, not an epilogue (CLAUDE.md 2c). Without the `|| exit`, a
# sweep whose every arm refused the GPU lock still printed "COMPLETE" — which is what a downstream
# chain greps for. Observed 2026-08-03 on the bracket script; fixed in every sweep script.
arm () {  # keys exposures probe_fraction
  .venv/Scripts/python.exe -u scripts/run_dense_baseline.py $COMMON \
    --n-entities "$1" --ladder "$2" --probe-fraction "$3" --out "$OUT/K$1" \
    || { echo "ARM $1 FAILED — sweep aborted, no completion marker"; return 1; }
}

arm  32000 268 0.08 || exit 1
arm  96000  89 0.03 || exit 1
arm 288000  29 0.01 || exit 1
echo "K-STAR 10M COARSE SWEEP COMPLETE"
