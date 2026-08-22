#!/usr/bin/env bash
# Wait for the trainability gate, APPLY ITS REGISTERED THRESHOLD, and proceed only on a pass.
#
# The decision rule is fixed in advance (M5 §5.61.2, §5.67.4): 10M @ 16,000 keys on the fixed stream
# must reach >= 30% chance-corrected. This automates that rule and nothing else — it does not
# choose the threshold, it applies it.
#
# Every branch that is not a clean pass ABORTS LOUDLY. A chain that proceeds on anything it fails
# to parse is the catch-and-continue defect (CLAUDE.md 2c), and a chain that announces success it
# did not verify is 2e. Both were hit in this session; neither is repeated here.
set -u
cd "$(dirname "$0")/.."
GATE_LOG=experiments/2026-08-03_dense-kstar-10M-rerun/run.log
THRESHOLD=30.0

while ! grep -q "GATE ARM COMPLETE\|ARM .* FAILED" "$GATE_LOG"; do
  if ! tasklist //FI "IMAGENAME eq python.exe" | grep -q python.exe; then
    echo "ABORT: no python process and no gate marker — the gate run died. Nothing launched."
    exit 1
  fi
  sleep 60
done

if grep -q "ARM .* FAILED" "$GATE_LOG"; then
  echo "ABORT: the gate stage reported a failed arm. Nothing launched."
  exit 1
fi

# The gate's own number: the LAST qa_heldout POOLED in the log, which is the 10M arm (arm 0 is 1M
# and precedes it). Parsed strictly — an unparseable log aborts rather than defaulting.
CORRECTED=$(grep "qa_heldout   POOLED" "$GATE_LOG" | tail -1 | sed -n 's/.*corrected \+\([0-9.]\+\)%.*/\1/p')
if [ -z "$CORRECTED" ]; then
  echo "ABORT: could not parse the gate's corrected accuracy from $GATE_LOG. Nothing launched."
  exit 1
fi

echo "GATE: 10M @ 16,000 fixed stream = ${CORRECTED}% corrected (threshold ${THRESHOLD}%)"
if awk -v c="$CORRECTED" -v t="$THRESHOLD" 'BEGIN{exit !(c+0 >= t+0)}'; then
  echo "GATE PASSED -> launching the 1M re-anchor ladder (M5 §5.67.4 step 2)"
  bash scripts/dense_kstar_1M_rerun.sh \
    > experiments/2026-08-03_dense-kstar-1M-rerun/run.log 2>&1
  echo "RE-ANCHOR STAGE FINISHED"
else
  echo "GATE FAILED at ${CORRECTED}% < ${THRESHOLD}% -> every 10M arm is VOID again."
  echo "Nothing launched. The training path at 10M is the finding, not key capacity."
  exit 2
fi
