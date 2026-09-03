#!/usr/bin/env bash
# ================================================================
# run_baselines.sh — Run all neural baselines (single seed)
#
# Usage:
#   bash scripts/run_baselines.sh                    # seed=42
#   bash scripts/run_baselines.sh --seed 0
# ================================================================

set -euo pipefail

# ── Defaults (override via environment variables) ─────────────────
DATA="${DATA:-$HOME/data/retailrocket_unified/cold_20}"
RESULTS="${RESULTS:-$HOME/results}"
LOGS="${LOGS:-$HOME/results/logs}"
SEED=42

# ── Parse args ────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --seed) SEED="$2"; shift 2 ;;
        --data_dir) DATA="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$RESULTS" "$LOGS"
CATPROCL="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CATPROCL"

echo "================================================================"
echo " run_baselines.sh — seed=$SEED"
echo " DATA   : $DATA"
echo " RESULTS: $RESULTS"
echo "================================================================"

BASELINES=(gcegnn cl4srec nclsbr nirgnn m2trec letitgo)

for script in "${BASELINES[@]}"; do
    echo "--- $script ---"
    python3 baselines/${script}.py \
        --data_dir   "$DATA" \
        --output_dir "$RESULTS" \
        --dataset    retailrocket \
        --seed       "$SEED" \
        2>&1 | tee "$LOGS/${script}_seed${SEED}.log"
done

echo "Done. Results in $RESULTS/"
