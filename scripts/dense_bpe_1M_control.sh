#!/usr/bin/env bash
# THE BPE 1M CONTROL (M5 §5.106, §5.110) — the arm that chooses the frontier's dense-side row.
#
#   BPE-735 vs syllable-689, at 1M, at the two arms bracketing K*_P(1M) = 16,971.
#
# SURFACE COORDINATES, measured (M5 §5.112) — three matched, one varying:
#
#            vocabulary   embedding share   tokens/name   tokens/fact
#   syllable    689           9.5%             2.00          14.83
#   BPE-735     735          ~10.0%            4.15          14.71
#
# tokens/FACT matching at 0.8% is incidental and useful: BPE spends more on names and less on
# values than the syllable tokenizer (which is character-level for values), and the two nearly
# cancel. So the arms are compute-matched at the SAME exposures, and tokens/name is the one
# coordinate that materially moves. That is a cleaner isolation than §5.109 anticipated.
#
# Compute is matched per surface anyway, by steps not exposures (the confound that voided
# intervention arm A): 535 x 43 = 23,005 steps at 16,000; 479 x 48 = 22,992 at 18,000 — identical
# to the syllable arms already run at n=10.
#
# READOUT: T = syllable_corrected / bpe_corrected at each key count.
#   T >= 5x        scheme+length is the mechanism; §5.46 survives ~whole; third surface NOT run
#   1.5x - 5x      INTERMEDIATE -> the padded-verbose third surface fires (§5.111)
#   T <= 1.5x      length inert over 2.0-4.26 tokens/name ONLY; third surface NOT run
#
# and the ASYMMETRIC INFERENCE (§5.135) governs what each licenses: a large T extends toward the
# reference's regime; a small T says nothing about the unreachable region (chars/token ~4).
#
# 5 seeds per arm against the syllable side's 10 — the asymmetry is stated, not hidden.
# ~0.25 h per arm (0.19 training + 0.06 BPE tokenisation, both measured), 10 arms, ~2.5 h.
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

for s in 0 1 2 3 4; do
  arm 16000 535 0.15 $s || exit 1
  arm 18000 479 0.12 $s || exit 1
done
echo "BPE 1M CONTROL COMPLETE"
