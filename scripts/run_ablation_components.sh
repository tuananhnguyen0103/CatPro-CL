#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# run_ablation_components.sh
# Chạy ablation tắt/bật λ (InfoNCE) và ρ (PSM) trên 3 datasets × 5 seeds
# Mục đích: trả lời reviewer A4 — chứng minh đóng góp độc lập của từng component
#
# 3 variants × 3 datasets × 5 seeds = 45 runs
# Variant:
#   no_infonce : λ=0.0, ρ=0.05  → chỉ PSM hoạt động
#   no_psm     : λ=0.5, ρ=0.0   → chỉ InfoNCE hoạt động
#   no_both    : λ=0.0, ρ=0.0   → prototype bank + cold inference only
# A11 (λ=0.5, ρ=0.05) đã có ở v6 → không cần chạy lại
#
# Cách chạy:
#   cd <repo_root>
#   bash scripts/run_ablation_components.sh 2>&1 | tee logs/ablation_components.log
#
# Kết quả lưu tại:
#   ~/results_ablation/{dataset}/{variant}/{dataset}_{variant}_seed{seed}.json
# ═══════════════════════════════════════════════════════════════════════════════

set -e

SEEDS=(42 0 1 2 3)
DATASETS=(retailrocket diginetica cellphones)
VARIANTS=(no_infonce no_psm no_both)

# Map dataset → config prefix
declare -A DS_PREFIX
DS_PREFIX[retailrocket]="rr"
DS_PREFIX[diginetica]="diginetica"
DS_PREFIX[cellphones]="cellphones"

TOTAL=$((${#DATASETS[@]} * ${#VARIANTS[@]} * ${#SEEDS[@]}))
COUNT=0

echo "════════════════════════════════════════════════════"
echo " Ablation Component Grid Search"
echo " Total runs: $TOTAL (3 datasets × 3 variants × 5 seeds)"
echo " Start: $(date)"
echo "════════════════════════════════════════════════════"

for DS in "${DATASETS[@]}"; do
  for VARIANT in "${VARIANTS[@]}"; do
    CFG="configs/ablation_components/${DS_PREFIX[$DS]}_${VARIANT}.yaml"

    # Kiểm tra config tồn tại
    if [ ! -f "$CFG" ]; then
      echo "[ERROR] Config not found: $CFG"
      exit 1
    fi

    for SEED in "${SEEDS[@]}"; do
      COUNT=$((COUNT + 1))
      echo ""
      echo "── [$COUNT/$TOTAL] ds=$DS  variant=$VARIANT  seed=$SEED ──"
      echo "   config: $CFG"

      python catprocl/train.py \
        --config "$CFG" \
        seed=$SEED \
        output_dir="~/results_ablation/${DS}/${VARIANT}" \
        checkpoint_dir="~/checkpoints_ablation/${DS}/${VARIANT}" \
        log_dir="~/logs_ablation/${DS}/${VARIANT}"

      echo "   ✓ Done: $DS/$VARIANT/seed$SEED"
    done
  done
done

echo ""
echo "════════════════════════════════════════════════════"
echo " ALL DONE: $COUNT/$TOTAL runs completed"
echo " End: $(date)"
echo "════════════════════════════════════════════════════"
echo ""
echo "Results in: ~/results_ablation/"
echo "To pull results, run:"
echo "  python scripts/collect_ablation_results.py"
