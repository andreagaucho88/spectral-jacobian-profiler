#!/usr/bin/env bash
# Full spectral grid for the arXiv revision.
#   2 models x {pre-norm (-2), post-norm (-1)} x n=40/category x n_iter=15.
# Non-destructive: backs up the verified n=20 results first.
# Run detached; progress in results_fullrun.log.
set -u -o pipefail

cd "$(dirname "$0")"   # repo root (this script lives there)

MODELS="Qwen/Qwen2.5-0.5B-Instruct HuggingFaceTB/SmolLM2-360M-Instruct"
NP=40; KTOP=6; NPROBES=32; NITER=15; EPS=1e-3; NDIR=8; NLEG=200
LM_DIR=lengthmatched_prompts
LEG_DIR=results_legacy
SPEC_DIR=results_spectral
ANA_DIR=results_analysis

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---- 0. numeric gate + backup of verified n=20 artifacts -------------------
log "==== 0. self-test gate ===="
python3 spectral.py >/dev/null 2>&1 && log "self-test PASS" || { log "SELF-TEST FAILED - aborting"; exit 1; }

BK="backup_n20_$(date '+%Y%m%d_%H%M%S')"
mkdir -p "$BK"
for d in "$SPEC_DIR" "$ANA_DIR"; do
  [ -d "$d" ] && cp -R "$d" "$BK/" && log "backed up $d -> $BK/"
done

# ---- per-model pipeline ----------------------------------------------------
for MODEL in $MODELS; do
  SLUG="${MODEL//\//__}"
  JSON="$LM_DIR/lengthmatched_prompts__${SLUG}.json"
  log "################ MODEL $MODEL ################"
  [ -f "$JSON" ] || { log "MISSING prompts JSON $JSON - skipping model"; continue; }

  # 1. legacy Algorithm 1 (cheap; needed for bulk-consistency + joint axes).
  LEG_CSV="$LEG_DIR/${SLUG}/legacy_per_prompt.csv"
  if [ -f "$LEG_CSV" ]; then
    log "1. legacy present ($LEG_CSV) - keeping"
  else
    log "1. legacy sweep (n=$NLEG, K=$NDIR)"
    python3 legacy.py --model "$MODEL" --prompts-json "$JSON" \
        --out-dir "$LEG_DIR" --n-prompts "$NLEG" --n-directions "$NDIR" \
        --epsilon "$EPS" --device cpu --dtype float32 \
        && log "   legacy done" || log "   legacy FAILED"
  fi

  # 2. spectral Algorithm 2, both read-out sites, n=40, n_iter=15.
  log "2. spectral sweep  n=$NP  k=$KTOP  probes=$NPROBES  iter=$NITER  fi=-2,-1"
  python3 runner.py --model "$MODEL" --prompts-json "$JSON" \
      --out-dir "$SPEC_DIR" --n-prompts "$NP" --k-top "$KTOP" \
      --n-probes "$NPROBES" --n-iter "$NITER" --final-indices -2 -1 \
      --device cpu --dtype float32 \
      && log "   spectral done" || log "   spectral FAILED for $MODEL"

  # 3. joint analysis: pre-norm (legacy + spectral -2) and post-norm (spectral -1).
  log "3. analysis pre_norm"
  python3 analysis.py --legacy-csv "$LEG_CSV" \
      --spectral-csv "$SPEC_DIR/${SLUG}/final_index_-2/spectral_per_prompt.csv" \
      --model "$MODEL" --out-dir "$ANA_DIR/${SLUG}/pre_norm" \
      && log "   pre_norm done" || log "   pre_norm FAILED"

  log "3. analysis post_norm"
  python3 analysis.py \
      --spectral-csv "$SPEC_DIR/${SLUG}/final_index_-1/spectral_per_prompt.csv" \
      --model "$MODEL" --out-dir "$ANA_DIR/${SLUG}/post_norm" \
      && log "   post_norm done" || log "   post_norm FAILED"
done

log "==== ALL DONE ===="
find "$ANA_DIR" -name joint_axes.csv 2>/dev/null | sort
find "$SPEC_DIR" -name spectral_summary_by_category.csv 2>/dev/null | sort
