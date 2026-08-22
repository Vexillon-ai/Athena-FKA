#!/usr/bin/env bash
# High-load recipe audit (M5 §5.14) — give the incumbent its best fight before reporting a collapse.
#
# CLAUDE.md: *the training recipe does not transfer across sizes* (50M NaN'd at every LR tried).
# The regime that changed here is LOAD, not size, and the symptom is the same shape: loss stalls
# (0.4579 -> 0.4519 for 4x the compute at 1.44x load; 0.4445 -> 0.4455 at 0.71x) while lower loads
# keep learning. So the recipe is audited at the collapsed point before any collapse is reported.
#
# Three arms against the incumbent (lr 1e-3, wd 0.1, loss 0.4579 at E=64):
#   lower LR   — the stall could be a too-large step in a sharper basin
#   higher LR  — or a too-small one, if it is stuck rather than bouncing
#   wd 0       — weight decay penalises exactly the large weights memorisation needs. Harmless
#                with capacity to spare, potentially binding once the corpus fills the model.
#                This is the arm with a load-DEPENDENT mechanism, so it is the prime suspect.
#
# Complete runs at a fixed horizon (E=64, 25,728 steps), not truncated probes: the
# reproduction-probe rule says a probe must not recompute the schedule from its own length.
set -u
cd "$(dirname "$0")/.."
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
OUT=experiments/2026-08-02_dense-recipe-audit

run () {  # tag lr wd
  for i in $(seq 1 200); do
    .venv/Scripts/python.exe -u scripts/run_dense_baseline.py --mode ladder --sizes 1M \
      --n-entities 31686 --ladder 64 --probe-fraction 0.06 --max-probes 1500 --log-every 8000 \
      --lr "$2" --weight-decay "$3" --out "$OUT/$1" 2>&1 | tee "$OUT/$1.out" && return 0
    # Retry ONLY on the GPU lock. Retrying on any failure would have hidden a real error behind
    # 200 identical "GPU busy" lines — which is exactly what it did on the first attempt.
    if ! grep -q "GPU is already in use" "$OUT/$1.out"; then
      echo "[$1] FAILED for a reason other than the GPU lock — aborting the audit"
      tail -20 "$OUT/$1.out"
      return 1
    fi
    echo "[$1 attempt $i] GPU busy, retrying in 120s"
    sleep 120
  done
}

run lr3e-4_wd0.1 0.0003 0.1
run lr3e-3_wd0.1 0.003  0.1
run lr1e-3_wd0.0 0.001  0.0
echo "AUDIT COMPLETE"
