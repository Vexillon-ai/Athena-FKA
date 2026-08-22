#!/usr/bin/env bash
# NEAR-CLIFF SURFACE DISCRIMINATOR (M5 §5.44).
#
# The intervention tested the surface at 31,686 keys — already far past the 1M model's cliff — and
# found nothing. That rules out surface RESCUING a collapsed operating point; it does not rule out
# surface moving the cliff's LOCATION. This tests exactly that, by bracketing the cliff.
#
#   key counts   8,000 / 12,000 / 16,000   (the control put the 1M cliff in this range)
#   surfaces     verbose (char-level) vs syllable
#   value-load   matched to within 1.1% across all three key counts, by shrinking value spaces
#   mix          3 relations throughout (works_with excluded — its value space IS the entity set,
#                so its load cannot be held fixed while the key count moves; §5.28)
#   recipe       literature-matched (lr 1e-3, wd 0.02, warmup 1,000)
#   compute      ~22,500 steps every arm — E is set PER SURFACE because the syllable surface is
#                4.5x denser, so matched exposures would silently hand it 4.6x less compute
#                (§5.42.1, the confound that voided intervention arm A)
#
# Routes, registered before the run:
#   cliff MOVES with surface  -> surface matters at the margin; scoped rehabilitation of the
#                                tokenizer arc, and the ladder's surface choice becomes load-bearing
#   cliff INVARIANT           -> key count is the whole story, surface is fully closed as a variable
set -u
cd "$(dirname "$0")/.."
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
OUT=experiments/2026-08-02_dense-near-cliff
R=birth_year,birth_city,employer
RECIPE="--lr 0.001 --weight-decay 0.02 --warmup-steps 1000"

arm () {  # keys  cities  employers  year_lo year_hi  surface  exposures  probe_fraction
  .venv/Scripts/python.exe -u scripts/run_dense_baseline.py --mode ladder --sizes 1M \
    --n-entities "$1" --relations "$R" --n-cities "$2" --n-employers "$3" \
    --birth-year-range "$4" "$5" --surface "$6" --ladder "$7" --probe-fraction "$8" \
    --max-probes 1500 --log-every 8000 $RECIPE --out "$OUT/K$1_$6"
}

#            keys  cities emp  yr_lo yr_hi  surface   E     probe_frac
arm  8000    512   1024   1900  2000   verbose   336   0.20
arm  8000    512   1024   1900  2000   syllable  1088  0.20
arm 12000    32    256    1990  2005   verbose   224   0.17
arm 12000    32    256    1990  2005   syllable  699   0.17
arm 16000    23    32     1990  2000   verbose   168   0.15
arm 16000    23    32     1990  2000   syllable  535   0.15
echo "NEAR-CLIFF DISCRIMINATOR COMPLETE"
