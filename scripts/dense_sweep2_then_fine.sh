#!/usr/bin/env bash
# Chain the fine bracket behind sweep 2's OWN success marker (CLAUDE.md 2c/2e).
set -u
cd "$(dirname "$0")/.."
LOG=experiments/2026-08-03_dense-kstar-10M-rerun/coarse2.log
while ! grep -q "SWEEP 2 COMPLETE\|ARM .* FAILED" "$LOG"; do
  if ! tasklist //FI "IMAGENAME eq python.exe" | grep -q python.exe; then
    echo "ABORT: sweep 2 died with no completion marker. Fine bracket NOT launched."; exit 1
  fi
  sleep 60
done
if grep -q "ARM .* FAILED" "$LOG"; then
  echo "ABORT: sweep 2 reported a failed arm. Fine bracket NOT launched."; exit 1
fi
echo "sweep 2 complete -> launching the fine bracket (M5 5.76)"
bash scripts/dense_kstar_10M_fine.sh > experiments/2026-08-03_dense-kstar-10M-rerun/fine.log 2>&1
echo "FINE BRACKET STAGE FINISHED"
