"""
preprocess_retailrocket.py
--------------------------
Raw RetailRocket → 7 unified files:
  sessions_train.txt, sessions_val.txt, sessions_test.txt
  item2cat.json, cat2items.json, cat_parent.json, meta.json

Usage:
  python preprocessing/preprocess_retailrocket.py \
      --input_dir data/raw/retailrocket \
      --output_dir data/retailrocket_unified

Input files (from Kaggle RetailRocket dataset):
  events.csv              - user events (view/addtocart/transaction)
  item_properties_part1.csv
  item_properties_part2.csv
"""

import argparse
import json
import os
import random
from collections import defaultdict

import pandas as pd


# ─── Constants ────────────────────────────────────────────────────────────────
SESSION_GAP_MINUTES = 30   # Gap to split user clicks into sessions
MIN_SESSION_LEN = 2        # Drop sessions shorter than this
MAX_SESSION_LEN = 50       # Truncate sessions longer than this
MIN_ITEM_FREQ = 5          # Drop items appearing fewer than this times
CATEGORY_PROPERTY = "categoryid"  # Property name in item_properties files


# ─── Step 1: Load item → category mapping ─────────────────────────────────────
def load_item2cat(input_dir: str) -> dict[int, int]:
    """Read item_properties files and extract categoryid for each item."""
    dfs = []
    for fname in ["item_properties_part1.csv", "item_properties_part2.csv"]:
        fpath = os.path.join(input_dir, fname)
        if os.path.exists(fpath):
            dfs.append(pd.read_csv(fpath))

    if not dfs:
        raise FileNotFoundError(
            f"No item_properties files found in {input_dir}. "
            "Download from Kaggle: retailrocket-recommender-system-dataset"
        )

    props = pd.concat(dfs, ignore_index=True)
    cat_props = props[props["property"] == CATEGORY_PROPERTY][["itemid", "value"]].copy()
    # Keep last known category per item
    cat_props = cat_props.drop_duplicates(subset="itemid", keep="last")
    item2cat_raw = dict(zip(cat_props["itemid"], cat_props["value"]))
    print(f"  Items with category: {len(item2cat_raw):,}")
    return item2cat_raw


# ─── Step 2: Build sessions from events ───────────────────────────────────────
def build_sessions(input_dir: str, item2cat_raw: dict) -> list[list[int]]:
    """
    Read events.csv, filter view events, group clicks by user,
    split into sessions using 30-minute gap.
    Returns list of sessions (each session = list of raw item IDs).
    """
    events_path = os.path.join(input_dir, "events.csv")
    if not os.path.exists(events_path):
        raise FileNotFoundError(f"events.csv not found in {input_dir}")

    df = pd.read_csv(events_path)
    # Keep only 'view' events (most common, what SR-GNN papers use)
    df = df[df["event"] == "view"].copy()
    # Keep only items that have a category
    df = df[df["itemid"].isin(item2cat_raw)].copy()
    # Sort by user then timestamp
    df = df.sort_values(["visitorid", "timestamp"]).reset_index(drop=True)

    print(f"  View events with category: {len(df):,}")

    sessions = []
    gap_ms = SESSION_GAP_MINUTES * 60 * 1000  # RetailRocket timestamps are in ms

    for _, user_df in df.groupby("visitorid"):
        timestamps = user_df["timestamp"].tolist()
        items = user_df["itemid"].tolist()

        current_session = [items[0]]
        for i in range(1, len(items)):
            if timestamps[i] - timestamps[i - 1] > gap_ms:
                sessions.append(current_session)
                current_session = [items[i]]
            else:
                current_session.append(items[i])
        sessions.append(current_session)

    print(f"  Raw sessions: {len(sessions):,}")
    return sessions


# ─── Step 3: Filter + re-index ────────────────────────────────────────────────
def filter_and_reindex(
    sessions: list[list[int]],
    item2cat_raw: dict,
) -> tuple[list[list[int]], dict, dict, dict, dict]:
    """
    Iterative k-core filtering (lặp cho đến khi ổn định):
      1. Xóa items xuất hiện < MIN_ITEM_FREQ lần
      2. Xóa sessions có độ dài < MIN_SESSION_LEN sau khi xóa items
      3. Lặp lại vì xóa sessions → một số items giảm frequency → lại xóa tiếp
    Sau khi ổn định:
      4. Cắt sessions dài hơn MAX_SESSION_LEN (giữ MAX_SESSION_LEN item cuối)
      5. Re-index items và categories về integer liên tục (1-based, 0=padding)

    Returns:
        sessions_reindexed: filtered sessions with new item IDs
        item2cat:   {new_item_id: new_cat_id}
        cat2items:  {new_cat_id: [new_item_ids]}
        cat_parent: {new_cat_id: None}  (RetailRocket has flat categories)
        raw2new:    {raw_item_id: new_item_id}
    """
    # ── Iterative k-core filtering ────────────────────────────────────────────
    current = sessions
    iteration = 0
    while True:
        iteration += 1

        # Count item frequencies trong tập sessions hiện tại
        item_freq: dict[int, int] = defaultdict(int)
        for sess in current:
            for item in sess:
                item_freq[item] += 1

        # Lọc sessions: xóa item hiếm, drop session ngắn
        filtered = []
        for sess in current:
            cleaned = [it for it in sess if item_freq[it] >= MIN_ITEM_FREQ]
            if len(cleaned) >= MIN_SESSION_LEN:
                filtered.append(cleaned)

        print(f"  Iter {iteration}: sessions={len(filtered):,} | "
              f"items={len({it for s in filtered for it in s}):,}")

        # Dừng khi không còn thay đổi
        if len(filtered) == len(current):
            break
        current = filtered

    # ── Cắt session dài sau khi đã ổn định ───────────────────────────────────
    filtered = []
    for sess in current:
        if len(sess) > MAX_SESSION_LEN:
            sess = sess[-MAX_SESSION_LEN:]
        filtered.append(sess)

    print(f"  Sessions after filtering: {len(filtered):,}")

    # Collect all items that appear after filtering
    valid_items = set()
    for sess in filtered:
        valid_items.update(sess)

    # Re-index items (1-based, 0 = padding)
    sorted_items = sorted(valid_items)
    raw2new = {raw: new for new, raw in enumerate(sorted_items, start=1)}

    # Re-index categories
    raw_cats = {item2cat_raw[raw] for raw in sorted_items if raw in item2cat_raw}
    sorted_cats = sorted(raw_cats)
    rawcat2new = {raw: new for new, raw in enumerate(sorted_cats, start=0)}

    # Build mappings with new IDs
    item2cat: dict[int, int] = {}
    cat2items: dict[int, list[int]] = defaultdict(list)
    for raw_item in sorted_items:
        new_item = raw2new[raw_item]
        raw_cat = item2cat_raw.get(raw_item)
        if raw_cat is not None and raw_cat in rawcat2new:
            new_cat = rawcat2new[raw_cat]
            item2cat[new_item] = new_cat
            cat2items[new_cat].append(new_item)

    # Re-index sessions
    sessions_reindexed = [
        [raw2new[it] for it in sess if it in raw2new]
        for sess in filtered
    ]
    sessions_reindexed = [s for s in sessions_reindexed if len(s) >= MIN_SESSION_LEN]

    n_items = len(raw2new)
    n_cats = len(rawcat2new)
    print(f"  Items after re-index: {n_items:,}")
    print(f"  Categories: {n_cats:,}")
    print(f"  Sessions after re-index: {len(sessions_reindexed):,}")

    cat_parent = {cat_id: None for cat_id in range(n_cats)}  # flat taxonomy
    return sessions_reindexed, item2cat, dict(cat2items), cat_parent, raw2new


# ─── Step 4: Time-based split 80/10/10 ────────────────────────────────────────
def time_split(sessions: list[list[int]]) -> tuple[list, list, list]:
    """
    Sessions are already ordered by time (we sorted by timestamp in build_sessions).
    80% train / 10% val / 10% test.
    """
    n = len(sessions)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train = sessions[:n_train]
    val = sessions[n_train: n_train + n_val]
    test = sessions[n_train + n_val:]
    print(f"  Split → train: {len(train):,} | val: {len(val):,} | test: {len(test):,}")
    return train, val, test


# ─── Step 5: Write outputs ────────────────────────────────────────────────────
def write_sessions(sessions: list[list[int]], fpath: str) -> None:
    """Write sessions to text file — one session per line, items space-separated."""
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, "w") as f:
        for sess in sessions:
            f.write(" ".join(map(str, sess)) + "\n")


def save_outputs(
    output_dir: str,
    train: list, val: list, test: list,
    item2cat: dict, cat2items: dict, cat_parent: dict,
    n_items: int, n_cats: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    write_sessions(train, os.path.join(output_dir, "sessions_train.txt"))
    write_sessions(val,   os.path.join(output_dir, "sessions_val.txt"))
    write_sessions(test,  os.path.join(output_dir, "sessions_test.txt"))

    # JSON files — keys must be strings for JSON compatibility
    with open(os.path.join(output_dir, "item2cat.json"), "w") as f:
        json.dump({str(k): v for k, v in item2cat.items()}, f)

    with open(os.path.join(output_dir, "cat2items.json"), "w") as f:
        json.dump({str(k): v for k, v in cat2items.items()}, f)

    with open(os.path.join(output_dir, "cat_parent.json"), "w") as f:
        json.dump({str(k): v for k, v in cat_parent.items()}, f)

    meta = {
        "n_items": n_items,
        "n_cats": n_cats,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "min_session_len": MIN_SESSION_LEN,
        "max_session_len": MAX_SESSION_LEN,
        "min_item_freq": MIN_ITEM_FREQ,
        "dataset": "retailrocket",
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ Saved 7 unified files to: {output_dir}")
    print(f"  n_items={n_items:,} | n_cats={n_cats:,}")
    print(f"  train={len(train):,} | val={len(val):,} | test={len(test):,}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  required=True, help="Folder with events.csv and item_properties files")
    parser.add_argument("--output_dir", required=True, help="Where to save the 7 unified files")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print("=== RetailRocket Preprocessing ===")
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}\n")

    print("[1/4] Loading item→category mapping...")
    item2cat_raw = load_item2cat(args.input_dir)

    print("[2/4] Building sessions from events...")
    sessions = build_sessions(args.input_dir, item2cat_raw)

    print("[3/4] Filtering and re-indexing...")
    sessions_ri, item2cat, cat2items, cat_parent, _ = filter_and_reindex(sessions, item2cat_raw)

    print("[4/4] Time-based split 80/10/10...")
    train, val, test = time_split(sessions_ri)

    save_outputs(
        args.output_dir, train, val, test,
        item2cat, cat2items, cat_parent,
        n_items=max(item2cat.keys()) if item2cat else 0,
        n_cats=len(cat2items),
    )


if __name__ == "__main__":
    main()
