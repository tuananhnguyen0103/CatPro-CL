#!/usr/bin/env bash
# ================================================================
# run_baselines_5seeds.sh — Run all baselines x 5 seeds
#
# Usage:
#   cd <repo_root>
#   bash scripts/run_baselines_5seeds.sh
#
# Override defaults via env vars:
#   DATA=~/data/diginetica_unified/cold_20 bash scripts/run_baselines_5seeds.sh
# ================================================================

set -uo pipefail

CATPROCL="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${DATA:-$HOME/data/retailrocket_unified/cold_20}"
OUT="${OUT:-$HOME/results_baselines}"
DATASET="${DATASET:-retailrocket}"
LOGS="$OUT/logs"

mkdir -p "$OUT" "$LOGS"
cd "$CATPROCL"

SEEDS=(42 0 1 2 3)
declare -a BASELINES=("gcegnn" "cl4srec" "nclsbr" "nirgnn" "m2trec" "letitgo")

TOTAL=$(( ${#BASELINES[@]} * ${#SEEDS[@]} ))
DONE=0
START_ALL=$(date +%s)

echo "================================================================"
echo " Baselines x 5 seeds — $(date)"
echo " REPO   : $CATPROCL"
echo " DATA   : $DATA"
echo " OUT    : $OUT"
echo " Total runs: $TOTAL"
echo "================================================================"

for script in "${BASELINES[@]}"; do
    echo "--- $script ---"
    for seed in "${SEEDS[@]}"; do
        DONE=$((DONE + 1))
        LOG="$LOGS/${script}_seed${seed}.log"
        echo -n "  [$DONE/$TOTAL] $script seed=$seed ... "
        START=$(date +%s)
        python3 baselines/${script}.py \
            --data_dir   "$DATA" \
            --output_dir "$OUT" \
            --dataset    "$DATASET" \
            --seed       "$seed" \
            2>&1 | tee "$LOG"
        ELAPSED=$(( $(date +%s) - START ))
        echo "done in ${ELAPSED}s"
    done
done

TOTAL_TIME=$(( $(date +%s) - START_ALL ))
echo "================================================================"
echo " ALL DONE — $TOTAL runs in $(( TOTAL_TIME / 60 ))m$(( TOTAL_TIME % 60 ))s"
echo " Results: $OUT/"
echo "================================================================"
