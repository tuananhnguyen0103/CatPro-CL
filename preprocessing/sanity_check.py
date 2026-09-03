"""
sanity_check.py
---------------
Verify the cold_XX split is correct before training.

Checks:
  1. cold_items count matches expected (9,052 for RetailRocket cold_20)
  2. No cold items leak into train or val sessions
  3. Cold items appear in test sessions (as targets)
  4. All 7 files exist and are non-empty
  5. Item IDs in sessions are within [1, n_items]

Usage:
  python preprocessing/sanity_check.py \
      --data_dir data/retailrocket_unified/cold_20 \
      --expected_cold 9052
"""

import argparse
import json
import os


def load_sessions(fpath: str) -> list[list[int]]:
    sessions = []
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if line:
                sessions.append(list(map(int, line.split())))
    return sessions


def check_files_exist(data_dir: str) -> bool:
    required = [
        "sessions_train.txt", "sessions_val.txt", "sessions_test.txt",
        "item2cat.json", "cat2items.json", "cat_parent.json", "meta.json",
    ]
    ok = True
    for fname in required:
        fpath = os.path.join(data_dir, fname)
        exists = os.path.exists(fpath)
        size   = os.path.getsize(fpath) if exists else 0
        status = "✓" if exists and size > 0 else "✗"
        print(f"  {status} {fname}  ({size:,} bytes)")
        if not exists or size == 0:
            ok = False
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",       required=True,   help="cold_XX folder to verify")
    parser.add_argument("--expected_cold",  type=int, default=None,
                        help="Expected number of cold items (e.g. 9052 for RR cold_20)")
    args = parser.parse_args()

    print(f"=== Sanity Check: {args.data_dir} ===\n")
    passed = 0
    failed = 0

    # ── Check 1: files exist ──────────────────────────────────────────────────
    print("[1] Required files:")
    files_ok = check_files_exist(args.data_dir)
    if files_ok:
        print("  PASS: all 7 files present\n")
        passed += 1
    else:
        print("  FAIL: missing files\n")
        failed += 1
        return  # can't continue without files

    # ── Load everything ───────────────────────────────────────────────────────
    meta = json.load(open(os.path.join(args.data_dir, "meta.json")))
    n_items    = meta["n_items"]
    cold_items = set(meta.get("cold_items", []))

    train_sess = load_sessions(os.path.join(args.data_dir, "sessions_train.txt"))
    val_sess   = load_sessions(os.path.join(args.data_dir, "sessions_val.txt"))
    test_sess  = load_sessions(os.path.join(args.data_dir, "sessions_test.txt"))

    # ── Check 2: cold_items count ─────────────────────────────────────────────
    print(f"[2] Cold items count: {len(cold_items):,}")
    if len(cold_items) == 0:
        print("  FAIL: cold_items is empty! Run cold_start_split.py first.\n")
        failed += 1
    elif args.expected_cold and len(cold_items) != args.expected_cold:
        print(f"  WARN: expected {args.expected_cold:,}, got {len(cold_items):,}\n")
    else:
        expected_str = f" (expected {args.expected_cold:,})" if args.expected_cold else ""
        print(f"  PASS{expected_str}\n")
        passed += 1

    # ── Check 3: non-leaking (cold items NOT in train/val) ────────────────────
    print("[3] Non-leaking check (cold items must not appear in train/val):")
    train_items = {it for sess in train_sess for it in sess}
    val_items   = {it for sess in val_sess   for it in sess}

    leak_train = cold_items & train_items
    leak_val   = cold_items & val_items

    if len(leak_train) == 0 and len(leak_val) == 0:
        print(f"  PASS: cold ∩ train = 0 | cold ∩ val = 0\n")
        passed += 1
    else:
        if leak_train:
            print(f"  FAIL: {len(leak_train):,} cold items found in train! e.g. {list(leak_train)[:5]}")
        if leak_val:
            print(f"  FAIL: {len(leak_val):,} cold items found in val! e.g. {list(leak_val)[:5]}")
        print()
        failed += 1

    # ── Check 4: cold items appear in test (as targets) ───────────────────────
    print("[4] Cold items in test sessions (need > 0 for evaluation):")
    test_targets = {sess[-1] for sess in test_sess if sess}
    cold_in_test = cold_items & test_targets
    if len(cold_in_test) > 0:
        print(f"  PASS: {len(cold_in_test):,} cold items appear as test targets\n")
        passed += 1
    else:
        print("  WARN: 0 cold items appear as test targets — cold metrics will be 0\n")

    # ── Check 5: item IDs in valid range ─────────────────────────────────────
    print("[5] Item ID range check:")
    all_items = (
        {it for sess in train_sess for it in sess}
        | {it for sess in val_sess   for it in sess}
        | {it for sess in test_sess  for it in sess}
    )
    out_of_range = {it for it in all_items if it < 1 or it > n_items}
    if len(out_of_range) == 0:
        print(f"  PASS: all item IDs in [1, {n_items:,}]\n")
        passed += 1
    else:
        print(f"  FAIL: {len(out_of_range):,} items out of range [1, {n_items:,}]: {list(out_of_range)[:5]}\n")
        failed += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 50)
    print(f"RESULT: {passed} passed, {failed} failed")
    print(f"  n_items={n_items:,} | cold_items={len(cold_items):,}")
    print(f"  train={len(train_sess):,} | val={len(val_sess):,} | test={len(test_sess):,}")

    if failed == 0:
        print("\n✓ All checks passed. Ready to build graphs and train.")
    else:
        print("\n✗ Fix the issues above before training.")
        exit(1)


if __name__ == "__main__":
    main()
