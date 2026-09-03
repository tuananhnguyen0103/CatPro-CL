"""
preprocess_amazon_cellphones.py
--------------------------------
Amazon Cell Phones & Accessories (2018, 5-core) → 7 unified files:
  sessions_train.txt, sessions_val.txt, sessions_test.txt
  item2cat.json, cat2items.json, cat_parent.json, meta.json

Also saves taxonomy info:
  item2tax.json  → {item_id: {"t1": ..., "t2": ..., "t3": ..., "brand": ..., "price": ...}}

Usage:
  python preprocessing/preprocess_amazon_cellphones.py \
      --review_file  data/raw/amazon_cellphones/Cell_Phones_and_Accessories_5.json.gz \
      --meta_file    data/raw/amazon_cellphones/meta_Cell_Phones_and_Accessories.json.gz \
      --output_dir   data/amazon_cellphones_unified

Input files (from http://deepyeti.ucsd.edu/jianmo/amazon/):
  Cell_Phones_and_Accessories_5.json.gz   - 5-core reviews
  meta_Cell_Phones_and_Accessories.json.gz - item metadata (taxonomy, brand, price)

Protocol:
  - Group reviews by user, sort by timestamp → sessions
  - Split into sessions using MAX_DAYS_GAP between consecutive reviews
  - Min item frequency filter
  - Cold split: cold_20 (stratified by category)
  - Output format: same 7-file format as RetailRocket/Diginetica
"""

import argparse
import gzip
import json
import os
import random
from collections import defaultdict

import pandas as pd


# ─── Constants ────────────────────────────────────────────────────────────────
MAX_DAYS_GAP   = 7      # Gap in days to split user reviews into sessions
MIN_SESSION_LEN = 2
MAX_SESSION_LEN = 50
MIN_ITEM_FREQ   = 5
TEST_DAYS       = 7     # Last 7 days → test (same as DC2R paper)
VAL_RATIO       = 0.1   # 10% of remaining → val


def load_jsonl_gz(filepath):
    """Load a .json.gz file where each line is a JSON object."""
    data = []
    opener = gzip.open if filepath.endswith('.gz') else open
    with opener(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


# ─── Step 1: Load metadata (taxonomy, brand, price) ───────────────────────────
def load_metadata(meta_file):
    """
    Returns:
        item2tax: {asin: {"t1": str, "t2": str, "t3": str, "brand": str, "price": float}}
        asin2title: {asin: title}
    """
    print(f"Loading metadata from {meta_file} ...")
    meta_list = load_jsonl_gz(meta_file)
    print(f"  Raw metadata records: {len(meta_list):,}")

    item2tax = {}
    for m in meta_list:
        asin = m.get('asin', '').strip()
        if not asin:
            continue

        # Taxonomy: Amazon uses 'category' list (t1→t2→...→tN, coarse to fine)
        cats = m.get('category', [])
        if not cats:
            cats = m.get('categories', [[]])[0] if m.get('categories') else []
        # Normalize to list of strings
        if isinstance(cats, str):
            cats = [cats]

        # Assign t1, t2, t3 (pad with last value if shorter)
        t1 = cats[0].strip() if len(cats) > 0 else 'Unknown'
        t2 = cats[1].strip() if len(cats) > 1 else t1
        t3 = cats[-1].strip() if len(cats) > 0 else 'Unknown'  # finest level

        brand = str(m.get('brand', '')).strip() or 'Unknown'
        try:
            price = float(str(m.get('price', '0')).replace('$', '').replace(',', ''))
        except (ValueError, TypeError):
            price = 0.0

        item2tax[asin] = {
            't1': t1,
            't2': t2,
            't3': t3,
            'brand': brand,
            'price': price,
            'title': str(m.get('title', '')).strip()[:100],
        }

    print(f"  Items with metadata: {len(item2tax):,}")
    return item2tax


# ─── Step 2: Load reviews and build sessions ──────────────────────────────────
def build_sessions(review_file, item2tax):
    """
    Load reviews, group by user, sort by time, split into sessions.
    Returns list of (session, unix_timestamp_of_last_item) sorted by time.
    """
    print(f"Loading reviews from {review_file} ...")
    reviews = load_jsonl_gz(review_file)
    print(f"  Raw reviews: {len(reviews):,}")

    # Filter to items with metadata
    valid_asins = set(item2tax.keys())

    # Group by user
    user_reviews = defaultdict(list)
    for r in reviews:
        asin = r.get('asin', '').strip()
        reviewer = r.get('reviewerID', '').strip()
        ts = r.get('unixReviewTime', 0)
        if asin in valid_asins and reviewer and ts:
            user_reviews[reviewer].append((ts, asin))

    print(f"  Users with reviews: {len(user_reviews):,}")

    # Build sessions: split by MAX_DAYS_GAP
    gap_seconds = MAX_DAYS_GAP * 86400
    all_sessions = []  # (session_items, last_timestamp)

    for reviewer, events in user_reviews.items():
        events.sort(key=lambda x: x[0])  # sort by time

        current_session = []
        current_ts = []

        for ts, asin in events:
            if current_session and (ts - current_ts[-1]) > gap_seconds:
                # Gap too large → save current session, start new
                if len(current_session) >= MIN_SESSION_LEN:
                    all_sessions.append((current_session[:], current_ts[-1]))
                current_session = [asin]
                current_ts = [ts]
            else:
                current_session.append(asin)
                current_ts.append(ts)

        # Save last session
        if len(current_session) >= MIN_SESSION_LEN:
            all_sessions.append((current_session[:], current_ts[-1]))

    print(f"  Raw sessions (before freq filter): {len(all_sessions):,}")
    return all_sessions


# ─── Step 3: Map ASINs to integer IDs ─────────────────────────────────────────
def remap_items(sessions, item2tax, min_freq=5):
    """Filter by frequency and remap ASIN → integer ID starting from 1."""
    freq = defaultdict(int)
    for sess, _ in sessions:
        for asin in sess:
            freq[asin] += 1

    valid = {asin for asin, f in freq.items() if f >= min_freq}
    print(f"  Items with freq >= {min_freq}: {len(valid):,}")

    # Filter sessions
    filtered = []
    for sess, ts in sessions:
        new_sess = [a for a in sess if a in valid]
        new_sess = new_sess[:MAX_SESSION_LEN]
        if len(new_sess) >= MIN_SESSION_LEN:
            filtered.append((new_sess, ts))

    print(f"  Sessions after freq filter: {len(filtered):,}")

    # Build integer mapping
    all_asins = sorted({a for sess, _ in filtered for a in sess})
    asin2id = {a: i+1 for i, a in enumerate(all_asins)}  # 1-indexed

    # Remap sessions
    remapped = [([asin2id[a] for a in sess], ts) for sess, ts in filtered]

    # Build taxonomy mapping with integer IDs
    # Build category mapping from t3 (finest level)
    t3_vals = sorted({item2tax[a]['t3'] for a in all_asins})
    t3_to_catid = {t: i+1 for i, t in enumerate(t3_vals)}

    # t1 → super-category
    t1_vals = sorted({item2tax[a]['t1'] for a in all_asins})
    t1_to_id = {t: i+1 for i, t in enumerate(t1_vals)}

    # t2 → mid-category
    t2_vals = sorted({item2tax[a]['t2'] for a in all_asins})
    t2_to_id = {t: i+1 for i, t in enumerate(t2_vals)}

    item2cat   = {}  # int_id → t3_catid (main category for prototype bank)
    item2tax_int = {}  # int_id → {t1_id, t2_id, t3_id, brand, price}
    for asin in all_asins:
        iid = asin2id[asin]
        tax = item2tax[asin]
        t3_id = t3_to_catid[tax['t3']]
        item2cat[iid] = t3_id
        item2tax_int[iid] = {
            't1_id': t1_to_id[tax['t1']],
            't2_id': t2_to_id[tax['t2']],
            't3_id': t3_id,
            't1': tax['t1'],
            't2': tax['t2'],
            't3': tax['t3'],
            'brand': tax['brand'],
            'price': tax['price'],
        }

    # cat_parent: t3 → t2 (for hierarchy)
    # Build cat_parent: for each t3 category, find its t2 parent
    t3_to_t2 = {}
    for asin in all_asins:
        tax = item2tax[asin]
        t3_id = t3_to_catid[tax['t3']]
        t2_id = t2_to_id[tax['t2']]
        t3_to_t2[t3_id] = t2_id

    cat_parent = {str(cid): t3_to_t2.get(cid) for cid in range(1, len(t3_vals)+1)}

    # cat2items
    cat2items = defaultdict(list)
    for iid, cid in item2cat.items():
        cat2items[cid].append(iid)

    n_items = len(all_asins)
    n_cats  = len(t3_vals)

    print(f"  Final items: {n_items:,}")
    print(f"  t1 categories (super): {len(t1_vals)}")
    print(f"  t2 categories (mid):   {len(t2_vals)}")
    print(f"  t3 categories (fine):  {n_cats}")

    return remapped, item2cat, cat2items, cat_parent, item2tax_int, n_items, n_cats


# ─── Step 4: Train/val/test split ─────────────────────────────────────────────
def split_sessions(sessions, test_ratio=0.1, val_ratio=0.1):
    """
    Split by ratio (time-ordered): last test_ratio → test,
    of remaining: val_ratio → val, rest → train.
    Amazon review data spans years, so time-window (N days) gives almost no test.
    Ratio-based split is more appropriate.
    """
    sessions.sort(key=lambda x: x[1])  # sort by timestamp

    if not sessions:
        raise ValueError("No sessions to split!")

    n = len(sessions)
    n_test = max(1, int(n * test_ratio))
    n_val  = max(1, int((n - n_test) * val_ratio))

    test_sess  = [s for s, _ in sessions[-n_test:]]
    train_val  = sessions[:-n_test]

    # Shuffle train_val then split
    random.shuffle(train_val)
    val_sess   = [s for s, _ in train_val[:n_val]]
    train_sess = [s for s, _ in train_val[n_val:]]

    print(f"  Train: {len(train_sess):,}  Val: {len(val_sess):,}  Test: {len(test_sess):,}")
    return train_sess, val_sess, test_sess


# ─── Step 5: Save unified format ──────────────────────────────────────────────
def save_sessions(sessions, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        for sess in sessions:
            f.write(' '.join(map(str, sess)) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--review_file', required=True,
                        help='Path to Cell_Phones_and_Accessories_5.json.gz')
    parser.add_argument('--meta_file', required=True,
                        help='Path to meta_Cell_Phones_and_Accessories.json.gz')
    parser.add_argument('--output_dir', default='data/amazon_cellphones_unified',
                        help='Output directory for unified files')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min_freq', type=int, default=MIN_ITEM_FREQ)
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Fraction of sessions (by time) used as test set')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Fraction of non-test sessions used as val set')
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Step 1: Load metadata")
    item2tax_raw = load_metadata(args.meta_file)

    print("\nStep 2: Build sessions from reviews")
    sessions_raw = build_sessions(args.review_file, item2tax_raw)

    print("\nStep 3: Remap items")
    sessions, item2cat, cat2items, cat_parent, item2tax_int, n_items, n_cats = \
        remap_items(sessions_raw, item2tax_raw, min_freq=args.min_freq)

    print("\nStep 4: Split train/val/test")
    train, val, test = split_sessions(sessions, args.test_ratio, args.val_ratio)

    print("\nStep 5: Save unified files")
    save_sessions(train, os.path.join(args.output_dir, 'sessions_train.txt'))
    save_sessions(val,   os.path.join(args.output_dir, 'sessions_val.txt'))
    save_sessions(test,  os.path.join(args.output_dir, 'sessions_test.txt'))

    # item2cat: {str(item_id): cat_id}
    with open(os.path.join(args.output_dir, 'item2cat.json'), 'w') as f:
        json.dump({str(k): v for k, v in item2cat.items()}, f)

    # cat2items: {str(cat_id): [item_ids]}
    with open(os.path.join(args.output_dir, 'cat2items.json'), 'w') as f:
        json.dump({str(k): v for k, v in cat2items.items()}, f)

    with open(os.path.join(args.output_dir, 'cat_parent.json'), 'w') as f:
        json.dump(cat_parent, f)

    # item2tax: full taxonomy info (for DC2R compatibility)
    with open(os.path.join(args.output_dir, 'item2tax.json'), 'w') as f:
        json.dump({str(k): v for k, v in item2tax_int.items()}, f)

    meta = {
        'dataset': 'amazon_cellphones',
        'n_items': n_items,
        'n_cats': n_cats,
        'n_train': len(train),
        'n_val': len(val),
        'n_test': len(test),
        'min_item_freq': args.min_freq,
        'test_ratio': args.test_ratio,
        'val_ratio': args.val_ratio,
        'category_level': 't3 (finest taxonomy level)',
        'taxonomy_levels': 't1 (super) → t2 (mid) → t3 (fine = category)',
    }
    with open(os.path.join(args.output_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Done! Output: {args.output_dir}")
    print(f"   n_items={n_items:,}  n_cats={n_cats}")
    print(f"   train={len(train):,}  val={len(val):,}  test={len(test):,}")
    print(f"\nNext steps:")
    print(f"  1. Run cold_start_split.py on {args.output_dir}")
    print(f"  2. Run build_graphs.py")
    print(f"  3. Train with catprocl_amazon_cellphones.yaml")


if __name__ == '__main__':
    main()
