"""
preprocess_diginetica.py
------------------------
Raw Diginetica (CIKM 2016) → 7 unified files:
  sessions_train.txt, sessions_val.txt, sessions_test.txt
  item2cat.json, cat2items.json, cat_parent.json, meta.json

Usage (2 cách):

  # Cách 1 — dùng rs_datasets (tự động download, KHUYÊN DÙNG):
  pip install rs_datasets
  python preprocessing/preprocess_diginetica.py \
      --output_dir data/diginetica_unified

  # Cách 2 — từ raw CSV (download thủ công):
  python preprocessing/preprocess_diginetica.py \
      --input_dir  data/raw/diginetica \
      --output_dir data/diginetica_unified

Input (Cách 2):
  train-item-views.csv       - columns: sessionId, userId, itemId, timeframe, eventdate
  product-categories.csv     - columns: productId, categoryId

Input (Cách 1 — rs_datasets):
  d.views      - columns: session_id, user_id, item_id, timeframe, date
  d.categories - columns: item_id, category_id

Split strategy (standard SR-GNN / GCE-GNN):
  Test  = sessions trong last 7 ngày
  Val   = 7 ngày trước test
  Train = phần còn lại

Filter:
  MIN_SESSION_LEN = 2
  MIN_ITEM_FREQ   = 5
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta

import pandas as pd


# ─── Constants (defaults — can be overridden via CLI args) ────────────────────
MIN_SESSION_LEN = 2
MAX_SESSION_LEN = 50   # override with --max_session_len
MIN_ITEM_FREQ   = 5
TEST_DAYS       = 7
VAL_DAYS        = 7


# ─── Nguồn 1: rs_datasets ─────────────────────────────────────────────────────
def load_from_rs_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Dùng thư viện rs_datasets để load Diginetica.
    Trả về (views_df, categories_df) đã normalize column names.

    views_df columns      : sessionId, itemId, timeframe, eventdate
    categories_df columns : productId, categoryId
    """
    try:
        from rs_datasets import Diginetica
    except ImportError:
        raise ImportError(
            "rs_datasets chưa được cài.\n"
            "Chạy: pip install rs_datasets\n"
            "Hoặc dùng --input_dir để load từ CSV thủ công."
        )

    print("  Downloading/loading Diginetica via rs_datasets...")
    d = Diginetica()

    # d.views: session_id, user_id, item_id, timeframe, date
    views = d.views.rename(columns={
        "session_id": "sessionId",
        "item_id":    "itemId",
        "timeframe":  "timeframe",
        "date":       "eventdate",
    })[["sessionId", "itemId", "timeframe", "eventdate"]]

    # d.categories: item_id, category_id
    cats = d.categories.rename(columns={
        "item_id":     "productId",
        "category_id": "categoryId",
    })[["productId", "categoryId"]]

    print(f"  views rows:      {len(views):,}")
    print(f"  categories rows: {len(cats):,}")
    return views, cats


# ─── Nguồn 2: raw CSV files ───────────────────────────────────────────────────
def load_from_csv(input_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load từ raw CSV files (semicolon-separated, format CIKM 2016).

    train-item-views.csv  : sessionId;userId;itemId;timeframe;eventdate
    product-categories.csv: itemId;categoryId
    """
    views_path = os.path.join(input_dir, "train-item-views.csv")
    cats_path  = os.path.join(input_dir, "product-categories.csv")

    if not os.path.exists(views_path):
        raise FileNotFoundError(
            f"Không tìm thấy: {views_path}\n"
            "Download từ Kaggle: kaggle datasets download -d profalbusdumbledore/diginetica-dataset"
        )
    if not os.path.exists(cats_path):
        raise FileNotFoundError(f"Không tìm thấy: {cats_path}")

    # Diginetica dùng semicolon (;) làm separator
    views = pd.read_csv(views_path, sep=";",
                        dtype={"sessionId": "Int64", "itemId": "Int64"},
                        na_values=["NA", ""])
    views = views.dropna(subset=["sessionId", "itemId"])
    views["sessionId"] = views["sessionId"].astype(int)
    views["itemId"]    = views["itemId"].astype(int)
    views = views[["sessionId", "itemId", "timeframe", "eventdate"]]

    # product-categories.csv: itemId;categoryId (không có productId)
    cats = pd.read_csv(cats_path, sep=";")
    # Normalize: rename itemId → productId để dùng chung interface
    cats = cats.rename(columns={"itemId": "productId"})
    cats = cats[["productId", "categoryId"]]

    print(f"  views rows:      {len(views):,}")
    print(f"  categories rows: {len(cats):,}")
    return views, cats


# ─── Step 1: item → category mapping ─────────────────────────────────────────
def build_item2cat(cats_df: pd.DataFrame) -> dict[int, int]:
    cats_df = cats_df.drop_duplicates(subset="productId", keep="first")
    item2cat = dict(zip(cats_df["productId"].astype(int),
                        cats_df["categoryId"].astype(int)))
    print(f"  Items with category: {len(item2cat):,}")
    return item2cat


# ─── Step 2: Build sessions ───────────────────────────────────────────────────
def build_sessions(
    views_df: pd.DataFrame,
    item2cat: dict,
) -> tuple[list[list[int]], list[str]]:
    """
    Nhóm views theo sessionId, sort theo timeframe.
    Chỉ giữ items có category.
    Trả về (sessions, dates) — date lấy theo click cuối của session.
    """
    # Chỉ giữ items có category
    views_df = views_df[views_df["itemId"].isin(item2cat)].copy()

    # eventdate có thể là string "2016-05-09" hoặc datetime
    views_df["eventdate"] = views_df["eventdate"].astype(str).str[:10]

    # Sort theo session rồi timeframe
    views_df = views_df.sort_values(["sessionId", "timeframe"]).reset_index(drop=True)

    print(f"  Rows after category filter: {len(views_df):,}")

    sessions = []
    dates = []
    for sid, grp in views_df.groupby("sessionId", sort=False):
        items = grp["itemId"].tolist()
        last_date = grp["eventdate"].iloc[-1]
        sessions.append(items)
        dates.append(str(last_date))

    print(f"  Raw sessions: {len(sessions):,}")
    return sessions, dates


# ─── Step 3: Time-based split ─────────────────────────────────────────────────
def split_by_time(
    sessions: list[list[int]],
    dates: list[str],
) -> tuple[list, list, list]:
    parsed = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
    max_date = max(parsed)

    test_start = max_date - timedelta(days=TEST_DAYS - 1)
    val_start  = test_start - timedelta(days=VAL_DAYS)

    train, val, test = [], [], []
    for sess, dt in zip(sessions, parsed):
        if dt >= test_start:
            test.append(sess)
        elif dt >= val_start:
            val.append(sess)
        else:
            train.append(sess)

    print(f"  Time split: train={len(train):,} | val={len(val):,} | test={len(test):,}")
    return train, val, test


# ─── Step 4: Filter & re-index ───────────────────────────────────────────────
def remap_items(
    train: list, val: list, test: list,
    item2cat_raw: dict,
    max_session_len: int = MAX_SESSION_LEN,
) -> tuple:
    # Đếm freq trong train
    freq: dict[int, int] = defaultdict(int)
    for sess in train:
        for it in sess:
            freq[it] += 1

    valid_items = {it for it, cnt in freq.items() if cnt >= MIN_ITEM_FREQ}
    print(f"  Items freq ≥ {MIN_ITEM_FREQ}: {len(valid_items):,} / {len(freq):,}")

    def clean(sessions):
        out = []
        for sess in sessions:
            filtered = [it for it in sess if it in valid_items]
            if len(filtered) < MIN_SESSION_LEN:
                continue
            if len(filtered) > max_session_len:
                filtered = filtered[-max_session_len:]   # giữ lại CUỐI session
            out.append(filtered)
        return out

    train_f = clean(train)
    val_f   = clean(val)
    test_f  = clean(test)
    print(f"  After filter: train={len(train_f):,} | val={len(val_f):,} | test={len(test_f):,}")

    # Collect all items
    all_items: set[int] = set()
    for split in [train_f, val_f, test_f]:
        for sess in split:
            all_items.update(sess)

    # Re-index 1-based (0 = padding)
    sorted_items = sorted(all_items)
    old2new = {old: new for new, old in enumerate(sorted_items, start=1)}
    n_items = len(sorted_items)

    def remap(sessions):
        return [[old2new[it] for it in sess] for sess in sessions]

    train_r = remap(train_f)
    val_r   = remap(val_f)
    test_r  = remap(test_f)

    # Category mappings
    item2cat: dict[int, int] = {}
    for old_id, new_id in old2new.items():
        cat = item2cat_raw.get(old_id)
        if cat is not None:
            item2cat[new_id] = int(cat)

    # Re-index categories 1-based
    all_cats = sorted(set(item2cat.values()))
    old_cat2new = {c: i for i, c in enumerate(all_cats, start=1)}
    item2cat = {it: old_cat2new[c] for it, c in item2cat.items()}

    cat2items: dict[int, list[int]] = defaultdict(list)
    for it, cat in item2cat.items():
        cat2items[cat].append(it)

    n_cats = len(all_cats)

    # Diginetica không có category hierarchy
    cat_parent = {str(cat): None for cat in range(1, n_cats + 1)}

    print(f"  n_items={n_items:,} | n_cats={n_cats:,}")

    meta = {
        "dataset":         "diginetica",
        "n_items":         n_items,
        "n_cats":          n_cats,
        "n_train":         len(train_r),
        "n_val":           len(val_r),
        "n_test":          len(test_r),
        "min_session_len": MIN_SESSION_LEN,
        "max_session_len": max_session_len,
        "min_item_freq":   MIN_ITEM_FREQ,
        "split_strategy":  f"time_based_last{TEST_DAYS}days_test_prev{VAL_DAYS}days_val",
        "item2cat":        {str(k): v for k, v in item2cat.items()},
    }

    return train_r, val_r, test_r, dict(item2cat), dict(cat2items), cat_parent, meta


# ─── Step 5: Write unified files ─────────────────────────────────────────────
def write_unified(output_dir, train, val, test, item2cat, cat2items, cat_parent, meta):
    os.makedirs(output_dir, exist_ok=True)

    def write_sessions(sessions, fname):
        with open(os.path.join(output_dir, fname), "w") as f:
            for sess in sessions:
                f.write(" ".join(map(str, sess)) + "\n")

    write_sessions(train, "sessions_train.txt")
    write_sessions(val,   "sessions_val.txt")
    write_sessions(test,  "sessions_test.txt")

    with open(os.path.join(output_dir, "item2cat.json"), "w") as f:
        json.dump({str(k): v for k, v in item2cat.items()}, f)

    with open(os.path.join(output_dir, "cat2items.json"), "w") as f:
        json.dump({str(k): v for k, v in cat2items.items()}, f)

    with open(os.path.join(output_dir, "cat_parent.json"), "w") as f:
        json.dump(cat_parent, f)

    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Written to: {output_dir}")
    print(f"  sessions_train.txt : {len(train):,} sessions")
    print(f"  sessions_val.txt   : {len(val):,} sessions")
    print(f"  sessions_test.txt  : {len(test):,} sessions")
    print(f"  n_items={meta['n_items']:,} | n_cats={meta['n_cats']:,}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Preprocess Diginetica dataset")
    parser.add_argument(
        "--input_dir", default=None,
        help="Dir chứa train-item-views.csv + product-categories.csv. "
             "Bỏ qua nếu dùng rs_datasets."
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Output dir cho 7 unified files"
    )
    parser.add_argument(
        "--max_session_len", type=int, default=MAX_SESSION_LEN,
        help=f"Độ dài session tối đa (default={MAX_SESSION_LEN}). "
             "Dùng --max_session_len 5 để tạo dataset len2-5."
    )
    args = parser.parse_args()

    print("=== Preprocess Diginetica ===\n")

    # ── Chọn nguồn data ──────────────────────────────────────────────────────
    if args.input_dir is None:
        print("Nguồn: rs_datasets (tự động download)")
        print("Step 1: Load via rs_datasets")
        views_df, cats_df = load_from_rs_datasets()
    else:
        print(f"Nguồn: CSV files từ {args.input_dir}")
        print("Step 1: Load từ CSV files")
        views_df, cats_df = load_from_csv(args.input_dir)

    print("\nStep 2: Build item→category mapping")
    item2cat_raw = build_item2cat(cats_df)

    print("\nStep 3: Build sessions")
    sessions, dates = build_sessions(views_df, item2cat_raw)

    print("\nStep 4: Time-based split")
    train, val, test = split_by_time(sessions, dates)

    print(f"\nStep 5: Filter + re-index (max_session_len={args.max_session_len})")
    train_r, val_r, test_r, item2cat, cat2items, cat_parent, meta = remap_items(
        train, val, test, item2cat_raw,
        max_session_len=args.max_session_len,
    )

    print("\nStep 6: Write unified files")
    write_unified(args.output_dir, train_r, val_r, test_r,
                  item2cat, cat2items, cat_parent, meta)

    print("\n✓ Done! Bước tiếp theo:")
    print(f"  python preprocessing/cold_start_split.py \\")
    print(f"      --data_dir {args.output_dir} --ratios 10 20 30")
    print(f"\nNếu đây là run với max_session_len={args.max_session_len}:")
    print(f"  Kiểm tra meta.json: max_session_len phải = {args.max_session_len}")


if __name__ == "__main__":
    main()
