"""
popularity.py — Popularity Baseline
-------------------------------------
Recommend top-K most popular items (by training frequency) to every session.
Cold items sẽ không bao giờ được recommend (không xuất hiện trong train).

Usage:
    python baselines/popularity.py --data_dir <path_to_cold_20>
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_sessions(path: str) -> list[list[int]]:
    sessions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items = list(map(int, line.split()))
                sessions.append(items)
    return sessions


def evaluate_popularity(
    train_sessions: list[list[int]],
    test_sessions: list[list[int]],
    cold_items: set[int],
    ks: list[int] = [10, 20],
) -> dict:
    # Đếm tần suất item trong train
    counter = Counter()
    for sess in train_sessions:
        for item in sess[:-1]:   # bỏ target (item cuối)
            counter[item] += 1

    # Top-K popular items (sắp xếp giảm dần)
    top_items = [item for item, _ in counter.most_common()]

    hits   = {k: 0 for k in ks}
    rr     = {k: 0.0 for k in ks}
    c_hits = {k: 0 for k in ks}
    c_rr   = {k: 0.0 for k in ks}

    n_overall = 0
    n_cold    = 0

    for sess in test_sessions:
        if len(sess) < 2:
            continue
        target = sess[-1]
        is_cold = target in cold_items
        n_overall += 1
        if is_cold:
            n_cold += 1

        for k in ks:
            recs = top_items[:k]
            if target in recs:
                hits[k] += 1
                rank = recs.index(target) + 1
                rr[k] += 1.0 / rank
                if is_cold:
                    c_hits[k] += 1
                    c_rr[k] += 1.0 / rank

    results = {}
    for k in ks:
        results[f"HR@{k}"]      = hits[k] / n_overall if n_overall > 0 else 0.0
        results[f"MRR@{k}"]     = rr[k]   / n_overall if n_overall > 0 else 0.0
        results[f"Cold_HR@{k}"] = c_hits[k] / n_cold  if n_cold > 0 else 0.0
        results[f"Cold_MRR@{k}"]= c_rr[k]   / n_cold  if n_cold > 0 else 0.0
    results["n_overall"] = n_overall
    results["n_cold"]    = n_cold
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="../results")
    parser.add_argument("--dataset", default="retailrocket")
    parser.add_argument("--ks", nargs="+", type=int, default=[10, 20])
    args = parser.parse_args()

    data_dir = args.data_dir

    print("Loading data ...")
    train = load_sessions(os.path.join(data_dir, "sessions_train.txt"))
    test  = load_sessions(os.path.join(data_dir, "sessions_test.txt"))

    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    cold_items = set(meta.get("cold_items", []))

    print(f"  train={len(train):,} | test={len(test):,} | cold_items={len(cold_items):,}")

    results = evaluate_popularity(train, test, cold_items, ks=args.ks)

    print("\n" + "=" * 60)
    print(f"Popularity Baseline — {args.dataset}")
    for k in args.ks:
        print(f"  Overall | HR@{k}={results[f'HR@{k}']:.4f}  MRR@{k}={results[f'MRR@{k}']:.4f}")
    for k in args.ks:
        print(f"  Cold    | HR@{k}={results[f'Cold_HR@{k}']:.4f}  MRR@{k}={results[f'Cold_MRR@{k}']:.4f}")
    print(f"  n_overall={results['n_overall']:,} | n_cold={results['n_cold']:,}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    out = {
        "dataset":  args.dataset,
        "ablation": "Popularity",
        "seed":     0,
        "test":     results,
    }
    fname = os.path.join(args.output_dir, f"{args.dataset}_Popularity.json")
    with open(fname, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved → {fname}")


if __name__ == "__main__":
    main()
