#!/usr/bin/env bash
# Full pipeline for the hidden-state dynamics paper.
#
# Run with:  bash run.sh          (NOT `source run.sh`, so a failure can't kill the shell)
#
# The pipeline is idempotent per stage (each step writes into its own directory
# and only re-runs if the outputs are missing). Total wall time on a Mac M4 Pro
# with defaults: a few hours per model for the spectral sweep; the rest is
# minutes.
#
# Stages:
#   1. spectral core self-test (no torch needed) - a hard gate
#   2. length-matched prompt generation, one JSON per model
#   3. legacy Algorithm 1 sweep (random-direction FTLE + entropy/p_max)
#   4. spectral Algorithm 2 sweep (sigma_max, stable rank, chi_F, ...)
#   5. joint statistical analysis per model
#
# Overrides:
#   MODELS               space-separated HF model names
#   N_PROMPTS_LM         prompts per category in the matched set  (default 300)
#   N_PROMPTS_LEGACY     prompts per category for the legacy sweep (default 200)
#   N_PROMPTS_SPECTRAL   prompts per category for the spectral sweep (default 60)
#   N_DIRECTIONS_LEGACY  random directions per prompt in Algorithm 1 (default 8)
#   K_TOP, N_PROBES, N_ITER   Algorithm 2 knobs (defaults tuned for M4 Pro)

set -e -u -o pipefail

MODELS="${MODELS:-Qwen/Qwen2.5-0.5B-Instruct HuggingFaceTB/SmolLM2-360M-Instruct}"
N_PROMPTS_LM="${N_PROMPTS_LM:-300}"
N_PROMPTS_LEGACY="${N_PROMPTS_LEGACY:-200}"
N_PROMPTS_SPECTRAL="${N_PROMPTS_SPECTRAL:-60}"
N_DIRECTIONS_LEGACY="${N_DIRECTIONS_LEGACY:-8}"
K_TOP="${K_TOP:-6}"
N_PROBES="${N_PROBES:-32}"
N_ITER="${N_ITER:-15}"
EPSILON="${EPSILON:-1e-3}"

LM_DIR="lengthmatched_prompts"
LEG_DIR="results_legacy"
SPEC_DIR="results_spectral"
ANA_DIR="results_analysis"

echo "==== 1. spectral core self-test (numpy) ===="
python3 spectral.py

for MODEL in $MODELS; do
  SLUG="${MODEL//\//__}"
  JSON="$LM_DIR/lengthmatched_prompts__${SLUG}.json"

  echo
  echo "==== 2. length-matched prompts  [$MODEL] ===="
  if [ -f "$JSON" ]; then
    echo "  found $JSON, skipping generation"
  else
    python3 prompts.py \
        --model "$MODEL" \
        --target-per-category "$N_PROMPTS_LM" \
        --max-per-template 5 \
        --out-dir "$LM_DIR"
  fi

  echo
  echo "==== 3. legacy Algorithm 1  [$MODEL] ===="
  LEG_CSV="$LEG_DIR/${SLUG}/legacy_per_prompt.csv"
  if [ -f "$LEG_CSV" ]; then
    echo "  found $LEG_CSV, skipping"
  else
    python3 legacy.py \
        --model "$MODEL" \
        --prompts-json "$JSON" \
        --out-dir "$LEG_DIR" \
        --n-prompts "$N_PROMPTS_LEGACY" \
        --n-directions "$N_DIRECTIONS_LEGACY" \
        --epsilon "$EPSILON"
  fi

  echo
  echo "==== 4. spectral Algorithm 2  [$MODEL] ===="
  SPEC_CSV_PRE="$SPEC_DIR/${SLUG}/final_index_-2/spectral_per_prompt.csv"
  if [ -f "$SPEC_CSV_PRE" ]; then
    echo "  found $SPEC_CSV_PRE, skipping"
  else
    python3 runner.py \
        --model "$MODEL" \
        --prompts-json "$JSON" \
        --out-dir "$SPEC_DIR" \
        --n-prompts "$N_PROMPTS_SPECTRAL" \
        --k-top "$K_TOP" \
        --n-probes "$N_PROBES" \
        --n-iter "$N_ITER" \
        --final-indices -2 -1
  fi

  echo
  echo "==== 5. joint analysis  [$MODEL] ===="
  # analyze the pre-norm spectral run against the legacy run;
  # then, separately, the post-norm spectral run (no legacy comparison, since
  # the legacy protocol tracks pre-norm by definition).
  python3 analysis.py \
      --legacy-csv    "$LEG_DIR/${SLUG}/legacy_per_prompt.csv" \
      --spectral-csv  "$SPEC_DIR/${SLUG}/final_index_-2/spectral_per_prompt.csv" \
      --model "$MODEL" \
      --out-dir "$ANA_DIR/${SLUG}/pre_norm"

  python3 analysis.py \
      --spectral-csv  "$SPEC_DIR/${SLUG}/final_index_-1/spectral_per_prompt.csv" \
      --model "$MODEL" \
      --out-dir "$ANA_DIR/${SLUG}/post_norm"
done

echo
echo "==== done ===="
find "$ANA_DIR" -name joint_axes.csv 2>/dev/null | sort
