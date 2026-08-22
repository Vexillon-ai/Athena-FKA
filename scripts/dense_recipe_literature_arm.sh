#!/usr/bin/env bash
# Fourth audit arm: the LITERATURE-MATCHED recipe (M5 §5.16).
#
# The capacity-scaling paper's own appendix, for bioS(N):
#   N = 10K-200K : weight decay 0.02,  lr 0.001
#   N = 500K-1M  : weight decay 0.01,  lr 0.0005
#   N >= 2M      : weight decay 0.005-0.002, lr 0.0003-0.0005
#   1K warmup steps, cosine 1 -> 0.1, context length 512.
#
# Our N = 31,686 sits in their first band, so the matched recipe is wd 0.02 / lr 0.001 / warmup
# 1000. We ran **wd 0.1** — five times their value — inherited from generic LM defaults. Note also
# that their weight decay FALLS as N rises: they met the load-dependence and tuned around it.
#
# warmup is passed explicitly rather than changed as a default, because arms 1-3 are already
# running at warmup 200 and moving a default under them would make the arms incomparable.
set -u
cd "$(dirname "$0")/.."
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
OUT=experiments/2026-08-02_dense-recipe-audit/lit_wd0.02_warm1000

for i in $(seq 1 400); do
  .venv/Scripts/python.exe -u scripts/run_dense_baseline.py --mode ladder --sizes 1M \
    --n-entities 31686 --ladder 64 --probe-fraction 0.06 --max-probes 1500 --log-every 8000 \
    --lr 0.001 --weight-decay 0.02 --warmup-steps 1000 --out "$OUT" 2>&1 | tee "$OUT.out" && break
  if ! grep -q "GPU is already in use" "$OUT.out"; then
    echo "FAILED for a reason other than the GPU lock — aborting"; tail -20 "$OUT.out"; exit 1
  fi
  echo "[lit arm attempt $i] GPU busy, retrying in 120s"
  sleep 120
done
