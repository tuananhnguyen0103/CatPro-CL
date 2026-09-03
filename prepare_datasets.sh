#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# prepare_datasets.sh
# Chạy trên ds_server (Linux, conda env ai310):
#   1. Download raw data
#   2. Preprocess → unified format
#   3. Cold-start split (10/20/30)
#   4. Build graph cache
#   5. Run A1-A8 ablations × 5 seeds cho từng dataset
#
# Usage:
#   conda activate ai310
#   cd <repo_root>
#   bash prepare_datasets.sh [diginetica|yoochoose|both] [preprocess_only]
#
# Ví dụ:
#   bash prepare_datasets.sh both              # full pipeline cả 2 dataset
#   bash prepare_datasets.sh diginetica        # chỉ Diginetica
#   bash prepare_datasets.sh yoochoose preprocess_only   # chỉ preprocess Yoochoose
# ═══════════════════════════════════════════════════════════════════════════════

set -e   # dừng nếu có lỗi
DATASET="${1:-both}"
MODE="${2:-full}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$REPO_DIR/../data"
RAW_DIR="$DATA_DIR/raw"
SEEDS=(42 0 1 2 3)
ABLATIONS=(A1 A3 A4 A5 A6 A7 A8)

echo "════════════════════════════════════════"
echo " CatPro-CL Dataset Preparation Script"
echo " Dataset: $DATASET | Mode: $MODE"
echo " Repo:    $REPO_DIR"
echo "════════════════════════════════════════"


# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────
check_conda() {
    if ! conda info --envs | grep -q "ai310"; then
        echo "ERROR: conda env 'ai310' not found. Run: conda create -n ai310 python=3.10"
        exit 1
    fi
    echo "✓ conda env ai310 found"
}

download_diginetica() {
    local RAW="$RAW_DIR/diginetica"
    mkdir -p "$RAW"

    if [ -f "$RAW/train-item-views.csv" ] && [ -f "$RAW/product-categories.csv" ]; then
        echo "✓ Diginetica raw data already exists, skipping download"
        return
    fi

    echo "Downloading Diginetica..."
    echo ""
    echo "  Diginetica không có public direct link ổn định."
    echo "  Hãy download thủ công từ một trong các nguồn sau:"
    echo ""
    echo "  Option 1 (Kaggle):"
    echo "    kaggle datasets download -d chadgostopp/recsys-challenge-2015"
    echo "    → Đây là Yoochoose, không phải Diginetica!"
    echo ""
    echo "  Option 2 (CIKM 2016 - thường dùng nhất trong papers):"
    echo "    Tìm mirror trên GitHub hoặc liên hệ tác giả SR-GNN"
    echo "    GCE-GNN repo: https://github.com/CCIIPLab/GCE-GNN"
    echo "    → Trong README họ có link download dataset"
    echo ""
    echo "  Files cần thiết sau khi download:"
    echo "    $RAW/train-item-views.csv"
    echo "    $RAW/product-categories.csv"
    echo ""
    echo "  Sau khi có file, chạy lại script này."

    # Try common Google Drive mirrors used in SBR papers
    # (Uncomment if you have gdown installed and a valid Drive link)
    # pip install gdown -q
    # gdown "https://drive.google.com/uc?id=DIGINETICA_FILE_ID" -O "$RAW/diginetica.zip"
    # unzip "$RAW/diginetica.zip" -d "$RAW/"

    echo ""
    read -r -p "Bạn đã đặt file vào đúng chỗ chưa? [y/N] " answer
    if [[ ! "$answer" =~ ^[Yy]$ ]]; then
        echo "Skipping Diginetica preprocessing."
        return 1
    fi
}

download_yoochoose() {
    local RAW="$RAW_DIR/yoochoose"
    mkdir -p "$RAW"

    if [ -f "$RAW/yoochoose-clicks.dat" ]; then
        echo "✓ Yoochoose raw data already exists, skipping download"
        return
    fi

    echo "Downloading Yoochoose (RecSys 2015)..."

    # Try official AWS S3 link
    YOOCHOOSE_URL="https://s3-eu-west-1.amazonaws.com/yc-rdata/yoochoose-data.7z"
    ARCHIVE="$RAW/yoochoose-data.7z"

    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "$ARCHIVE" "$YOOCHOOSE_URL" || {
            echo "wget failed, trying curl..."
            curl -L -o "$ARCHIVE" "$YOOCHOOSE_URL"
        }
    elif command -v curl &>/dev/null; then
        curl -L --progress-bar -o "$ARCHIVE" "$YOOCHOOSE_URL"
    else
        echo "ERROR: Neither wget nor curl found. Install one of them."
        exit 1
    fi

    # Extract
    if command -v 7z &>/dev/null; then
        7z x "$ARCHIVE" -o"$RAW/" -y
    elif command -v p7zip &>/dev/null; then
        p7zip -d "$ARCHIVE"
    else
        echo "7zip not found. Install: sudo apt-get install p7zip-full"
        echo "Then extract manually: 7z x $ARCHIVE -o$RAW/"
        exit 1
    fi

    echo "✓ Yoochoose downloaded and extracted"
}


# ─── PREPROCESSING ────────────────────────────────────────────────────────────
preprocess_diginetica() {
    local UNIFIED="$DATA_DIR/diginetica_unified"
    if [ -f "$UNIFIED/meta.json" ]; then
        echo "✓ Diginetica unified already exists, skipping preprocess"
    else
        echo "Preprocessing Diginetica..."
        python preprocessing/preprocess_diginetica.py \
            --input_dir  "$RAW_DIR/diginetica" \
            --output_dir "$UNIFIED"
    fi

    echo "Running cold-start split for Diginetica..."
    python preprocessing/cold_start_split.py \
        --data_dir "$UNIFIED" \
        --ratios 10 20 30

    echo "Running sanity check..."
    python preprocessing/sanity_check.py \
        --data_dir "$UNIFIED/cold_20"

    echo "✓ Diginetica preprocessing complete"
}

preprocess_yoochoose() {
    local UNIFIED="$DATA_DIR/yoochoose_unified"
    if [ -f "$UNIFIED/meta.json" ]; then
        echo "✓ Yoochoose unified already exists, skipping preprocess"
    else
        echo "Preprocessing Yoochoose 1/64..."
        python preprocessing/preprocess_yoochoose.py \
            --input_dir  "$RAW_DIR/yoochoose" \
            --output_dir "$UNIFIED"
    fi

    echo "Running cold-start split for Yoochoose..."
    python preprocessing/cold_start_split.py \
        --data_dir "$UNIFIED" \
        --ratios 10 20 30

    echo "Running sanity check..."
    python preprocessing/sanity_check.py \
        --data_dir "$UNIFIED/cold_20"

    echo "✓ Yoochoose preprocessing complete"
}


# ─── TRAINING A1-A8 × 5 SEEDS ────────────────────────────────────────────────
run_ablations() {
    local DATASET_NAME="$1"   # diginetica or yoochoose_1_64
    local CONFIG="$2"         # path to config YAML

    echo ""
    echo "══════════════════════════════════════════"
    echo " Training $DATASET_NAME — A1-A8 × 5 seeds"
    echo "══════════════════════════════════════════"

    # train.py saves JSON to output_dir in config (= ../results/ relative to repo root)
    local RESULTS_DIR="$REPO_DIR/../results"
    local LOG_DIR="$REPO_DIR/../logs"
    mkdir -p "$RESULTS_DIR" "$LOG_DIR"

    for ABLATION in "${ABLATIONS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            # train.py naming: {dataset}_{ablation}_seed{seed}.json
            local OUT_JSON="$RESULTS_DIR/${DATASET_NAME}_${ABLATION}_seed${SEED}.json"

            if [ -f "$OUT_JSON" ]; then
                echo "  [SKIP] $ABLATION seed=$SEED — already done"
                continue
            fi

            echo ""
            echo "  ▶ $ABLATION seed=$SEED"
            python catprocl/train.py \
                --config   "$CONFIG" \
                --ablation "$ABLATION" \
                --seed     "$SEED" \
                2>&1 | tee "$LOG_DIR/${DATASET_NAME}_${ABLATION}_seed${SEED}.log"

            echo "  ✓ $ABLATION seed=$SEED done"
        done
    done

    echo ""
    echo "✓ All ablations for $DATASET_NAME complete"
}

update_expected_cold() {
    local CONFIG="$1"
    local DATA_DIR_PATH="$2"
    local META="$DATA_DIR_PATH/meta.json"

    if [ -f "$META" ]; then
        local N_COLD
        N_COLD=$(python -c "import json; m=json.load(open('$META')); print(m.get('n_cold_items', -1))")
        echo "  Updating expected_cold=$N_COLD in $CONFIG"
        sed -i "s/expected_cold: -1/expected_cold: $N_COLD/" "$CONFIG"
    fi
}


# ─── MAIN ─────────────────────────────────────────────────────────────────────
cd "$REPO_DIR"
check_conda

DO_DIGINETICA=false
DO_YOOCHOOSE=false

case "$DATASET" in
    diginetica) DO_DIGINETICA=true ;;
    yoochoose)  DO_YOOCHOOSE=true  ;;
    both)       DO_DIGINETICA=true; DO_YOOCHOOSE=true ;;
    *)
        echo "Usage: $0 [diginetica|yoochoose|both] [preprocess_only]"
        exit 1
        ;;
esac

# ── Diginetica pipeline ───────────────────────────────────────────────────────
if $DO_DIGINETICA; then
    echo ""
    echo "────── DIGINETICA ──────"
    download_diginetica || true   # allow skipping if download fails

    if [ -f "$RAW_DIR/diginetica/train-item-views.csv" ]; then
        preprocess_diginetica
        update_expected_cold \
            "$REPO_DIR/configs/catprocl_diginetica.yaml" \
            "$DATA_DIR/diginetica_unified/cold_20"

        if [ "$MODE" != "preprocess_only" ]; then
            run_ablations "diginetica" "$REPO_DIR/configs/catprocl_diginetica.yaml"
        fi
    fi
fi

# ── Yoochoose pipeline ────────────────────────────────────────────────────────
if $DO_YOOCHOOSE; then
    echo ""
    echo "────── YOOCHOOSE 1/64 ──────"
    download_yoochoose

    if [ -f "$RAW_DIR/yoochoose/yoochoose-clicks.dat" ]; then
        preprocess_yoochoose
        update_expected_cold \
            "$REPO_DIR/configs/catprocl_yoochoose.yaml" \
            "$DATA_DIR/yoochoose_unified/cold_20"

        if [ "$MODE" != "preprocess_only" ]; then
            run_ablations "yoochoose_1_64" "$REPO_DIR/configs/catprocl_yoochoose.yaml"
        fi
    fi
fi

echo ""
echo "════════════════════════════════════════"
echo " DONE"
echo "════════════════════════════════════════"
