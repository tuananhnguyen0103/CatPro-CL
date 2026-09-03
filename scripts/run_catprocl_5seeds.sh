#!/usr/bin/env bash
# ================================================================
# run_catprocl_5seeds.sh — Run CatPro-CL (A11) x 5 seeds
#
# Usage:
#   cd <repo_root>
#   bash scripts/run_catprocl_5seeds.sh
#
# Override defaults:
#   ABLATIONS="A1" bash scripts/run_catprocl_5seeds.sh
#   DATA=~/data/diginetica_unified/cold_20 bash scripts/run_catprocl_5seeds.sh
# ================================================================

set -euo pipefail

CATPROCL="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${DATA:-$HOME/data/retailrocket_unified/cold_20}"
OUT="${OUT:-$HOME/results}"
DATASET="${DATASET:-retailrocket}"
CONFIG="${CONFIG:-$CATPROCL/configs/catprocl_retailrocket_a11.yaml}"

mkdir -p "$OUT"
cd "$CATPROCL"

SEEDS=(42 0 1 2 3)
ABLATIONS="${ABLATIONS:-A11}"

TOTAL=$(( ${#SEEDS[@]} ))
DONE=0
START_ALL=$(date +%s)

echo "================================================================"
echo " CatPro-CL $ABLATIONS x 5 seeds — $(date)"
echo " REPO   : $CATPROCL"
echo " CONFIG : $CONFIG"
echo " OUT    : $OUT"
echo "================================================================"

for seed in "${SEEDS[@]}"; do
    DONE=$((DONE + 1))
    echo "[$DONE/$TOTAL] ablation=$ABLATIONS seed=$seed"
    python3 catprocl/train.py \
        --config    "$CONFIG" \
        --ablation  "$ABLATIONS" \
        --seed      "$seed" \
        --output_dir "$OUT"
done

TOTAL_TIME=$(( $(date +%s) - START_ALL ))
echo "================================================================"
echo " DONE — 5 seeds in $(( TOTAL_TIME / 60 ))m$(( TOTAL_TIME % 60 ))s"
echo " Results: $OUT/"
echo "================================================================"
