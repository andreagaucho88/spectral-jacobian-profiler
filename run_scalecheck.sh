#!/usr/bin/env bash
# Scale-sanity check: Qwen2.5-1.5B and 3B, MINIMAL config, hidden+output axes.
#
# SAFETY: this script does NOT run until the primary sweep (run_full_n40.sh) has
# fully finished, so it never competes with it for CPU/RAM. Until then it only
# polls (negligible cost). Output goes to a SEPARATE dir (results_spectral_scalecheck)
# so it can never corrupt the primary results. runner.py is checkpointed/resumable.
#
# The Qwen2.5 family shares ONE tokenizer, so the 0.5B length-matched set is
# length-matched for 1.5B/3B too -> we reuse it (identical token lengths).
set -u -o pipefail
cd "$(dirname "$0")"   # repo root (this script lives there)

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# ---- 0. wait for the primary run to finish (poll only) ---------------------
log "scale-check waiting for primary run (run_full_n40.sh) to finish..."
while pgrep -f "run_full_n40.sh" >/dev/null 2>&1; do sleep 300; done
if ! grep -q "==== ALL DONE ====" results_fullrun.log 2>/dev/null; then
  log "WARNING: primary driver gone but no ALL DONE marker; proceeding anyway (resources are free)."
fi
log "primary run finished -> starting scale-check (24 GB now free)."

# ---- 1. minimal scale-check config -----------------------------------------
JSON="lengthmatched_prompts/lengthmatched_prompts__Qwen__Qwen2.5-0.5B-Instruct.json"
OUT="results_spectral_scalecheck"
NP=8; KTOP=3; NPROBES=6; NITER=8
MODELS="Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct"

log "self-test gate"; python3 spectral.py >/dev/null 2>&1 && log "self-test PASS" || { log "SELF-TEST FAILED"; exit 1; }

for MODEL in $MODELS; do
  log "#### scale-check $MODEL  (n=$NP k=$KTOP probes=$NPROBES iter=$NITER, fi=-2, Fisher on) ####"
  # pre-norm first; Fisher kept enabled. Errors are logged as rows, sweep continues.
  python3 runner.py --model "$MODEL" --prompts-json "$JSON" \
      --out-dir "$OUT" --n-prompts "$NP" --k-top "$KTOP" \
      --n-probes "$NPROBES" --n-iter "$NITER" --final-indices -2 \
      --device cpu --dtype float32 \
      && log "  $MODEL pre-norm done" || log "  $MODEL FAILED (see error rows / manifest)"
done

log "==== SCALECHECK DONE ===="
find "$OUT" -name spectral_summary_by_category.csv 2>/dev/null | sort
find "$OUT" -name run_manifest.json 2>/dev/null | sort
