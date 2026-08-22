#!/usr/bin/env bash
# Stage 2 — the dense ceiling, measured to CONVERGENCE under the CERTIFIED recipe (M5 §5.19-20).
#
# Usage: dense_stage2.sh <weight_decay> <warmup> [lr]
#
# Everything about this run is gated on certification, so the recipe is an argument rather than a
# default: nothing here may quietly re-introduce the wd=0.1 that stalled the sweep.
#
# Protocol, pre-registered:
#   * loads 0.71x (N=16,000) and 1.44x (N=31,686), plus 1.00x (N=22,000) as the peak candidate;
#   * at each, scale E until the CONVERGENCE CRITERION holds: heldout gain < 2 points per
#     E-doubling. Read peak position AND height from converged points only;
#   * stage-1 values at E=1024 are under-converged lower bounds and are annotated as such.
#
# The E ladder restarts from a low rung at every N because stage 1's convergence readings were
# taken under the stalled recipe and are VOID (§5.20) — "E=1024 is not converged" was a fact about
# wd=0.1, not about this corpus.
#
# N=16,000's 80,000-step checkpoint is NOT resumed: it was taken under the stalled recipe and is a
# record of that run, not a prefix of this one.
set -u
cd "$(dirname "$0")/.."
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

WD="${1:?usage: dense_stage2.sh <weight_decay> <warmup> [lr]}"
WARM="${2:?usage: dense_stage2.sh <weight_decay> <warmup> [lr]}"
LR="${3:-0.001}"
OUT=experiments/2026-08-02_dense-stage2

point () {  # n_entities  E-ladder  probe_fraction
  local n="$1" ladder="$2" pf="$3" tag="N$1"
  for i in $(seq 1 400); do
    .venv/Scripts/python.exe -u scripts/run_dense_baseline.py --mode ladder --sizes 1M \
      --n-entities "$n" --ladder "$ladder" --probe-fraction "$pf" --max-probes 1500 \
      --log-every 40000 --lr "$LR" --weight-decay "$WD" --warmup-steps "$WARM" \
      --out "$OUT/$tag" 2>&1 | tee "$OUT/$tag.out" && return 0
    if ! grep -q "GPU is already in use" "$OUT/$tag.out"; then
      echo "[$tag] FAILED for a reason other than the GPU lock — aborting stage 2"
      tail -20 "$OUT/$tag.out"; return 1
    fi
    echo "[$tag attempt $i] GPU busy, retrying in 120s"; sleep 120
  done
}

mkdir -p "$OUT"
echo "== stage 2 under certified recipe: lr=$LR wd=$WD warmup=$WARM"
point 16000 64,256,1024 0.06
point 31686 64,256,1024 0.06
point 22000 64,256,1024 0.06
echo "STAGE 2 COMPLETE"
