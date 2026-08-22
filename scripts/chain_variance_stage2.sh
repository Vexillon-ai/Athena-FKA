#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
LOG=experiments/2026-08-03_dense-kstar-replicate/run.log
while ! grep -q "REPLICATION COMPLETE\|ARM .* FAILED" "$LOG"; do
  if ! tasklist //FI "IMAGENAME eq python.exe" | grep -q python.exe; then
    echo "ABORT: replication died with no marker. Stage 2 NOT launched."; exit 1
  fi
  sleep 60
done
grep -q "ARM .* FAILED" "$LOG" && { echo "ABORT: replication had a failed arm."; exit 1; }
echo "replication complete -> launching variance stage 2"
bash scripts/dense_variance_stage2.sh > experiments/2026-08-03_dense-kstar-replicate/stage2.log 2>&1
echo "STAGE 2 FINISHED"
