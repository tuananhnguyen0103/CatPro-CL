"""
preprocess_yoochoose.py
-----------------------
Raw Yoochoose 1/64 → 7 unified files:
  sessions_train.txt, sessions_val.txt, sessions_test.txt
  item2cat.json, cat2items.json, cat_parent.json, meta.json

Usage:
  python preprocessing/preprocess_yoochoose.py \
      --input_dir data/raw/yoochoose \
      --output_dir data/yoochoose_unified

Input file (from RecSys 2015 Challenge):
  yoochoose-clicks.dat   - SessionID,Timestamp,ItemID,Category

Download:
  https://www.kaggle.com/datasets/chadgostopp/recsys-challenge-2015
  OR: https://recsys.acm.org/recsys15/challenge/

1/64 strategy (standard in SR-GNN / GCE-GNN papers):
  - Sort all session IDs, keep only the LAST 1/64 fraction
  - Mimics the "most recent behavior" subsetting in the original challenge

Split strategy:
  - Test  = sessions in last 1 day
  - Val   = sessions in previous 1 day
  - Train = all earlier sessions

Category:
  - Yoochoose clicks include a Category field per click
  - Assign each item the MOST FREQUENT category observed across all its clicks
  - Items with no valid category (0 or NaN) are filtered out
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import pandas as pd


# ─── Constants ────────────────────────────────────────────────────────────────
MIN_SESSION_LEN  = 2
MAX_SESSION_LEN  = 50
MIN_ITEM_FREQ    = 5
FRACTION         = 64    # keep last 1/64 of sessions
TEST_DAYS        = 1     # last N days → test
VAL_DAYS         = 1     # previous N days → val
INVALID_CATEGORY = {"0", ""}  # treat these as "no category"


# ─── Step 1: Load clicks ──────────────────────────────────────────────────────
def load_clicks(input_dir: str) -> pd.DataFrame:
    fpath = os.path.join(input_dir, "yoochoose-clicks.dat")
    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f"yoochoose-clicks.dat not found in {input_dir}.\n"
            "Download from: https://www.kaggle.com/datasets/chadgostopp/recsys-challenge-2015"
        )

    # Columns: SessionID, Timestamp, ItemID, Category
    df = pd.read_csv(
        fpath,
        header=None,
        names=["session_id", "timestamp", "item_id", "category"],
        dtype={"session_id": int, "item_id": int, "category": str},
    )
    print(f"  Total clicks: {len(df):,}")
    return df


# ─── Step 2: 1/64 subsetting ──────────────────────────────────────────────────
def subset_1over64(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the last 1/FRACTION session IDs (highest session IDs)."""
    all_sids = sorted(df["session_id"].unique())
    n_keep = max(1, len(all_sids) // FRACTION)
    keep_sids = set(all_sids[-n_keep:])
    df_sub = df[df["session_id"].isin(keep_sids)].copy()
    print(f"  1/{FRACTION} subset: {n_keep:,} sessions / {len(all_sids):,} total")
    print(f"  Clicks after subset: {len(df_sub):,}")
    return df_sub


# ─── Step 3: Build item → category mapping ───────────────────────────────────
def build_item2cat(df: pd.DataFrame) -> dict[int, int]:
    """
    For each item, pick the most frequently observed category.
    Filter out items where the dominant category is invalid (0 or empty).
    """
    # Filter invalid category rows
    df_valid = df[~df["category"].isin(INVALID_CATEGORY)].copy()
    df_valid["category"] = df_valid["category"].str.strip()
    df_valid = df_valid[df_valid["category"] != ""]

    # Most frequent category per item
    item2cat_raw: dict[int, int] = {}
    for item_id, grp in df_valid.groupby("item_id"):
        most_common_cat = grp["category"].value_counts().idxmax()
        try:
            item2cat_raw[int(item_id)] = int(most_common_cat)
        except ValueError:
            # Non-numeric category string: hash to int
            item2cat_raw[int(item_id)] = abs(hash(most_common_cat)) % (10**6)

    print(f"  Items with valid category: {len(item2cat_raw):,}")
    return item2cat_raw


# ─── Step 4: Build sessions ───────────────────────────────────────────────────
def build_sessions(
    df: pd.DataFrame,
    item2cat: dict,
) -> tuple[list[list[int]], list[datetime]]:
    """
    Group clicks by session_id, sort by timestamp within each session.
    Keep only items that have a valid category.
    Returns: (sessions, session_last_timestamps)
    """
    # Keep only items with a valid category
    df = df[df["item_id"].isin(item2cat)].copy()

    # Parse timestamps (format: 2014-04-07T10:51:09.277Z)
    df["ts_parsed"] = pd.to_datetime(df["timestamp"], utc=True)

    # Sort within sessions
    df = df.sort_values(["session_id", "ts_parsed"]).reset_index(drop=True)

    sessions = []
    session_dates = []
    for sid, grp in df.groupby("session_id", sort=False):
        items = grp["item_id"].tolist()
        last_ts = grp["ts_parsed"].iloc[-1].to_pydatetime()
        sessions.append(items)
        session_dates.append(last_ts)

    print(f"  Sessions with valid items: {len(sessions):,}")
    return sessions, session_dates


# ─── Step 5: Time-based split ─────────────────────────────────────────────────
def split_by_time(
    sessions: list,
    dates: list[datetime],
) -> tuple[list, list, list]:
    max_date = max(d.replace(tzinfo=None) for d in dates)
    dates_naive = [d.replace(tzinfo=None) for d in dates]

    test_start = max_date - timedelta(days=TEST_DAYS)
    val_start  = test_start - timedelta(days=VAL_DAYS)

    train, val, test = [], [], []
    for sess, dt in zip(sessions, dates_naive):
        if dt >= test_start:
            test.append(sess)
        elif dt >= val_start:
            val.append(sess)
        else:
            train.append(sess)

    print(f"  Time split: train={len(train):,} | val={len(val):,} | test={len(test):,}")
    return train, val, test


# ─── Step 6: Filter & re-index ───────────────────────────────────────────────
def remap_items(
    train: list, val: list, test: list,
    item2cat_raw: dict,
) -> tuple:
    # Count frequency in train
    freq: dict[int, int] = defaultdict(int)
    for sess in train:
        for it in sess:
            freq[it] += 1

    valid_items = {it for it, cnt in freq.items() if cnt >= MIN_ITEM_FREQ}
    print(f"  Items with freq ≥ {MIN_ITEM_FREQ}: {len(valid_items):,}")

    def clean(sessions):
        out = []
        for sess in sessions:
            filtered = [it for it in sess if it in valid_items]
            if len(filtered) < MIN_SESSION_LEN:
                continue
            if len(filtered) > MAX_SESSION_LEN:
                filtered = filtered[-MAX_SESSION_LEN:]
            out.append(filtered)
        return out

    train_f = clean(train)
    val_f   = clean(val)
    test_f  = clean(test)

    print(f"  After filter: train={len(train_f):,} | val={len(val_f):,} | test={len(test_f):,}")

    # Collect all items present after filtering
    all_items: set[int] = set()
    for split in [train_f, val_f, test_f]:
        for sess in split:
            all_items.update(sess)

    # Re-index 1-based
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

    # Re-index categories to 1..C
    all_cats = sorted(set(item2cat.values()))
    old_cat2new = {c: i for i, c in enumerate(all_cats, start=1)}
    item2cat = {it: old_cat2new[c] for it, c in item2cat.items()}

    cat2items: dict[int, list[int]] = defaultdict(list)
    for it, cat in item2cat.items():
        cat2items[cat].append(it)

    n_cats = len(all_cats)

    # Yoochoose has no category hierarchy
    cat_parent = {str(cat): None for cat in range(1, n_cats + 1)}

    print(f"  n_items={n_items:,} | n_cats={n_cats:,}")

    meta = {
        "dataset": "yoochoose_1_64",
        "n_items": n_items,
        "n_cats": n_cats,
        "n_train": len(train_r),
        "n_val": len(val_r),
        "n_test": len(test_r),
        "min_session_len": MIN_SESSION_LEN,
        "min_item_freq": MIN_ITEM_FREQ,
        "split_strategy": f"time_based_1over{FRACTION}_last{TEST_DAYS}day_test_prev{VAL_DAYS}day_val",
        "item2cat": {str(k): v for k, v in item2cat.items()},
    }

    return (
        train_r, val_r, test_r,
        dict(item2cat),
        dict(cat2items),
        cat_parent,
        meta,
    )


# ─── Step 7: Write unified files ─────────────────────────────────────────────
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
    parser = argparse.ArgumentParser(description="Preprocess Yoochoose 1/64 dataset")
    parser.add_argument("--input_dir",    required=True, help="Dir with yoochoose-clicks.dat")
    parser.add_argument("--output_dir",   required=True, help="Output dir for unified 7 files")
    parser.add_argument("--skip_subset",  action="store_true",
                        help="Skip 1/64 subsetting (use when file is already 1/64 subset)")
    args = parser.parse_args()

    print("=== Preprocess Yoochoose 1/64 ===\n")

    print("Step 1: Load clicks")
    df = load_clicks(args.input_dir)

    print("\nStep 2: 1/64 subsetting")
    if args.skip_subset:
        print(f"  [SKIP] --skip_subset flag set — file already contains 1/{FRACTION} data")
        print(f"  Sessions: {df['session_id'].nunique():,} | Clicks: {len(df):,}")
    else:
        df = subset_1over64(df)

    print("\nStep 3: Build item→category mapping")
    item2cat_raw = build_item2cat(df)

    print("\nStep 4: Build sessions")
    sessions, dates = build_sessions(df, item2cat_raw)

    print("\nStep 5: Time-based split")
    train, val, test = split_by_time(sessions, dates)

    print("\nStep 6: Filter + re-index")
    train_r, val_r, test_r, item2cat, cat2items, cat_parent, meta = remap_items(
        train, val, test, item2cat_raw
    )

    print("\nStep 7: Write unified files")
    write_unified(args.output_dir, train_r, val_r, test_r,
                  item2cat, cat2items, cat_parent, meta)

    print("\nDone! Next step:")
    print(f"  python preprocessing/cold_start_split.py --data_dir {args.output_dir} --ratios 10 20 30")


if __name__ == "__main__":
    main()
