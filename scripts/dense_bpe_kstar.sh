#!/usr/bin/env bash
# LOCATE K*_BPE(1M) — the bracket purchase the headline is gated on (M5 §5.138).
#
# The control's arms were both placed to bracket K*_syllable(1M) = 16,971, and BPE came back ALIVE
# 5/5 at BOTH, so K*_BPE(1M) > 18,000 and is unbracketed above. The reconciliation is stated in
# PARAMS PER KEY, so an accuracy ratio cannot fill the caption slot — only K* can.
#
#   22,000  E=390  ->  23,010 steps
#   26,000  E=329  ->  23,030 steps
#   n = 5 per arm, P reported WITH sigma (M5 §5.113.2). A third arm is placed by the surviving side.
#
# BRANCHES, enumerated over the ATTAINABLE range and required to TILE it (M5 §5.138.1 — the
# undercoverage failure that let T < 1 land in a row whose wording was false):
#
#   K*_BPE <= 16,971            REFUTATION, named in advance: the accuracy advantage does NOT
#                               translate into keys; the surface factor is <= 1.0 in the units that
#                               matter, and RECONCILED is not claimable from this arm.
#   16,971 < K*_BPE <= 22,000   surface factor 1.0x-1.30x; amortisation must carry the rest
#   22,000 < K*_BPE <= 26,000   surface factor 1.30x-1.53x
#   26,000 < K*_BPE             surface factor > 1.53x; a THIRD arm above 26,000 is required
#                               before the factor is quoted, since an unbracketed K* is not a K*
#
# Every row states the factor in params/key, which is what the reconciliation consumes. Embedding
# share is quoted alongside (§5.139), and per-symbol occurrence is annotated as a suspected
# unmeasured coordinate that no comparison may smuggle in meanwhile.
#
# ~0.25 h/arm x 10 = ~2.5 h. Completion marker only through success (CLAUDE.md 2e).
set -u
cd "$(dirname "$0")/.."
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
OUT=experiments/2026-08-04_dense-bpe-control
R=birth_year,birth_city,employer
COMMON="--mode ladder --sizes 1M --relations $R --surface bpe --bpe-vocab 735 --n-cities 23 \
  --n-employers 32 --birth-year-range 1990 2000 --n-given-names 8192 --n-surnames 8192 \
  --max-probes 1500 --log-every 8000 --lr 0.001 --weight-decay 0.02 --warmup-steps 1000"

arm () {  # keys exposures probe_fraction seed
  .venv/Scripts/python.exe -u scripts/run_dense_baseline.py $COMMON \
    --n-entities "$1" --ladder "$2" --probe-fraction "$3" --seed "$4" \
    --out "$OUT/K$1_s$4" \
    || { echo "ARM $1 seed $4 FAILED — aborted, no completion marker"; return 1; }
}

# RESUMED at seed 2: seeds 0-1 completed before a transient device abort (M5 §5.142).
for s in 2 3 4; do
  arm 22000 390 0.10 $s || exit 1
  arm 26000 329 0.09 $s || exit 1
done
echo "BPE K-STAR BRACKET COMPLETE"
