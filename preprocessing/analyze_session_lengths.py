"""
analyze_session_lengths.py
Thống kê phân phối độ dài session cho bất kỳ unified dataset nào.

Usage:
    python preprocessing/analyze_session_lengths.py \
        --data_dir ~/data/retailrocket_unified
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def analyze(data_dir: str):
    base = Path(data_dir)
    splits = {
        "train": base / "sessions_train.txt",
        "val":   base / "sessions_val.txt",
        "test":  base / "sessions_test.txt",
    }

    meta_path = base / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    print(f"\n{'='*60}")
    print(f"Dataset: {base.name}")
    print(f"  n_items : {meta.get('n_items', '?'):,}")
    print(f"  n_cats  : {meta.get('n_cats', '?'):,}")
    print(f"{'='*60}")

    for split_name, fpath in splits.items():
        if not fpath.exists():
            print(f"\n[{split_name}] FILE NOT FOUND")
            continue

        lengths = []
        with open(fpath) as f:
            for line in f:
                items = line.strip().split()
                if items:
                    lengths.append(len(items))

        if not lengths:
            print(f"\n[{split_name}] EMPTY")
            continue

        total = len(lengths)
        cnt = Counter(lengths)
        lengths_sorted = sorted(lengths)

        avg  = sum(lengths) / total
        med  = lengths_sorted[total // 2]
        mn   = lengths_sorted[0]
        mx   = lengths_sorted[-1]
        p25  = lengths_sorted[int(total * 0.25)]
        p75  = lengths_sorted[int(total * 0.75)]
        p90  = lengths_sorted[int(total * 0.90)]
        p95  = lengths_sorted[int(total * 0.95)]

        print(f"\n[{split_name.upper()}]  n_sessions={total:,}")
        print(f"  min={mn}  max={mx}  mean={avg:.2f}  median={med}")
        print(f"  P25={p25}  P75={p75}  P90={p90}  P95={p95}")
        print(f"\n  Phân phối theo độ dài (top 15):")
        print(f"  {'Len':>5} | {'Count':>10} | {'%':>6} | Cumul%")
        cumul = 0
        for length in sorted(cnt.keys()):
            c = cnt[length]
            pct = 100 * c / total
            cumul += pct
            bar = '█' * min(int(pct / 2), 30)
            print(f"  {length:>5} | {c:>10,} | {pct:>5.1f}% | {cumul:>5.1f}%  {bar}")
            if length >= 15 and cumul > 99:
                remaining = {k: v for k, v in cnt.items() if k > 15}
                if remaining:
                    r_count = sum(remaining.values())
                    r_pct = 100 * r_count / total
                    print(f"  {'>15':>5} | {r_count:>10,} | {r_pct:>5.1f}% | (grouped)")
                break

        # Maxlen coverage
        print(f"\n  Coverage nếu dùng maxlen:")
        for ml in [3, 4, 5, 6, 10]:
            kept = sum(v for k, v in cnt.items() if k >= 2)  # all >= min_len=2
            affected = sum(v for k, v in cnt.items() if k > ml)
            pct_affected = 100 * affected / total
            print(f"    maxlen={ml}: {affected:,} sessions bị truncate ({pct_affected:.1f}%)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    args = p.parse_args()
    analyze(args.data_dir)


if __name__ == "__main__":
    main()
