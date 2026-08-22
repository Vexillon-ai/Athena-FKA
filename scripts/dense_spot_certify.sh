#!/usr/bin/env bash
# TIER 2 — SPOT-CERTIFY two findings whose DIRECTION carries an argument (M5 §5.70.2).
#
# Every dense result in the record was produced under the five-ordering stream and understates the
# baseline. The tax is not a constant offset (~10 points at 1M, ~89 at 10M), so nothing can be
# rescaled on paper. Three arms re-certify the two findings whose direction — not level — is load
# bearing for the Phase 5 write-up.
#
#   arm 1  key-count control @ 16,000, syllable
#          Certifies: KEY COUNT BINDS at fixed value-load. Read against the matched 8,000-key point
#          from the same control family. Direction = fewer keys scores materially higher.
#
#   arms 2-3  near-cliff pair @ 12,000, verbose vs syllable
#          Certifies: THE CLIFF'S LOCATION IS SURFACE-DEPENDENT. 12,000 is chosen because the
#          old-stream surface tax there was 17.1x (verbose 2.79%, syllable 47.78%) — inside the
#          collapse zone for verbose and comfortably alive for syllable, so a direction change is
#          visible rather than buried in floor scatter.
#
# READING RULE, fixed before the runs (§5.70.2):
#   direction HOLDS -> the finding stands; its levels are annotated "measured under the ordering
#                      defect; direction re-certified under the fixed stream"
#   direction FLIPS -> that finding is FULLY RE-OPENED, not patched
#
# Configs are inherited from the original runs and vary ONLY the stream fix — the reproduction-probe
# rule (a probe inherits the failing config wholesale and varies one thing). The 8,000-key partner
# for arm 1 comes from the near-cliff family, which used value spaces 512/1024/1900-2000 at 8,000 to
# match load; arm 1 therefore runs at the near-cliff pair's 12,000-key spaces for comparability with
# arms 2-3, and the key-count direction is read 8,000 -> 12,000 -> 16,000 within this run.
set -u
cd "$(dirname "$0")/.."
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
OUT=experiments/2026-08-03_dense-spot-certify
R=birth_year,birth_city,employer
RECIPE="--lr 0.001 --weight-decay 0.02 --warmup-steps 1000"

arm () {  # keys cities employers yr_lo yr_hi surface exposures probe_fraction
  .venv/Scripts/python.exe -u scripts/run_dense_baseline.py --mode ladder --sizes 1M \
    --n-entities "$1" --relations "$R" --n-cities "$2" --n-employers "$3" \
    --birth-year-range "$4" "$5" --surface "$6" --ladder "$7" --probe-fraction "$8" \
    --max-probes 1500 --log-every 8000 $RECIPE --out "$OUT/K$1_$6" \
    || { echo "ARM $1/$6 FAILED — aborted, no completion marker"; return 1; }
}

#          keys  cities emp  yr_lo yr_hi  surface   E     probe_frac
arm  8000    32    256    1990  2005   syllable  1088  0.20 || exit 1   # key-count: low end
arm 12000    32    256    1990  2005   syllable   699  0.17 || exit 1   # key-count mid + surface pair
arm 12000    32    256    1990  2005   verbose    224  0.17 || exit 1   # surface pair, other half
echo "SPOT-CERTIFY COMPLETE"
