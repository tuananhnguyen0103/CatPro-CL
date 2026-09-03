"""
cold_start_split.py
-------------------
Apply cold-start protocol on top of the unified 7-file format.

Protocol (per paper):
  - Stratified: for each category c, randomly select cold_ratio% of its items
  - Nested: cold_10 ⊂ cold_20 ⊂ cold_30 (same items, just subsets)
  - Non-leaking: cold items removed from ALL session prefixes (train/val/test)
  - Cold items can appear ONLY as the target (last item) in test sessions

Cold-start evaluation principle:
  A cold item must NEVER appear in any session prefix (context), across ALL splits.
  This ensures:
    1. Model never sees cold items in context → no self-correlation artifact
    2. Cold HR for ID-embedding baselines (CORE, SR-GNN...) → ≈ 0% as expected
    3. CatPro-CL advantage is measured purely via category prototype, not revisit effect

Usage:
  python preprocessing/cold_start_split.py \
      --data_dir data/retailrocket_unified \
      --ratios 10 20 30
"""

import argparse
import json
import os
import random
from collections import defaultdict


# ─── Load unified files ────────────────────────────────────────────────────────
def load_sessions(fpath: str) -> list[list[int]]:
    sessions = []
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if line:
                sessions.append(list(map(int, line.split())))
    return sessions


def load_json(fpath: str) -> dict:
    with open(fpath) as f:
        return json.load(f)


# ─── Stratified cold-item sampling (nested) ───────────────────────────────────
def sample_cold_items_nested(
    cat2items: dict[str, list[int]],
    ratios: list[int],
    seed: int = 42,
) -> dict[int, set[int]]:
    """
    For each ratio in ascending order, sample cold items.
    Nested guarantee: cold_10 ⊂ cold_20 ⊂ cold_30.

    Returns: {ratio: set_of_cold_item_ids}
    """
    rng = random.Random(seed)
    ratios_sorted = sorted(ratios)  # [10, 20, 30]

    # Per-category, sample the LARGEST ratio first, then take subsets
    max_ratio = ratios_sorted[-1]
    cold_by_ratio: dict[int, set[int]] = {r: set() for r in ratios_sorted}

    for cat_str, items in cat2items.items():
        if not items:
            continue
        items_shuffled = items.copy()
        rng.shuffle(items_shuffled)

        # Sample max_ratio% for the largest ratio
        n_max = max(1, int(len(items_shuffled) * max_ratio / 100))
        cold_max = items_shuffled[:n_max]

        # Nested subsets for smaller ratios
        for ratio in ratios_sorted:
            n = max(1, int(len(items_shuffled) * ratio / 100))
            cold_by_ratio[ratio].update(cold_max[:n])  # always subset of cold_max

    for ratio in ratios_sorted:
        print(f"  cold_{ratio}: {len(cold_by_ratio[ratio]):,} items")

    return cold_by_ratio


# ─── Apply cold split to sessions ─────────────────────────────────────────────
def apply_cold_to_sessions(
    sessions: list[list[int]],
    cold_items: set[int],
    split: str,  # "train", "val", or "test"
) -> list[list[int]]:
    """
    Unified cold-item removal for ALL splits.

    train / val:
      - Remove cold items from the ENTIRE session (prefix + target)
      - Drop session if len < 2 after removal
      - (Cold items must not appear as train targets; they have no learned embedding)

    test:
      - Remove cold items from the PREFIX only (session[:-1])
      - Keep the target (session[-1]) unchanged — can be warm OR cold
      - Drop session if prefix after removal is empty (< 1 item)

    Result: cold items NEVER appear in any session prefix across all splits.
    This eliminates the self-correlation artifact in mean-pooling models (CORE, etc.)
    and ensures Cold HR reflects true cold-start capability, not revisit effect.
    """
    if split == "test":
        cleaned = []
        for sess in sessions:
            prefix = [it for it in sess[:-1] if it not in cold_items]
            target = sess[-1]
            if len(prefix) >= 1:          # need ≥1 context item
                cleaned.append(prefix + [target])
        return cleaned

    # train / val: remove cold items everywhere
    cleaned = []
    for sess in sessions:
        filtered = [it for it in sess if it not in cold_items]
        if len(filtered) >= 2:
            cleaned.append(filtered)
    return cleaned


# ─── Write a cold_XX subfolder ────────────────────────────────────────────────
def write_cold_split(
    base_dir: str,
    ratio: int,
    train: list, val: list, test_orig: list,
    cold_items: set[int],
    item2cat: dict, cat2items: dict, cat_parent: dict,
    meta_root: dict,
) -> None:
    out_dir = os.path.join(base_dir, f"cold_{ratio}")
    os.makedirs(out_dir, exist_ok=True)

    # Apply cold protocol
    train_clean = apply_cold_to_sessions(train, cold_items, "train")
    val_clean   = apply_cold_to_sessions(val,   cold_items, "val")
    test_clean  = apply_cold_to_sessions(test_orig, cold_items, "test")

    # Write session files
    def write_sessions(sessions, fpath):
        with open(fpath, "w") as f:
            for sess in sessions:
                f.write(" ".join(map(str, sess)) + "\n")

    write_sessions(train_clean, os.path.join(out_dir, "sessions_train.txt"))
    write_sessions(val_clean,   os.path.join(out_dir, "sessions_val.txt"))
    write_sessions(test_clean,  os.path.join(out_dir, "sessions_test.txt"))

    # Copy category mappings (unchanged)
    for fname, obj in [
        ("item2cat.json", item2cat),
        ("cat2items.json", cat2items),
        ("cat_parent.json", cat_parent),
    ]:
        with open(os.path.join(out_dir, fname), "w") as f:
            json.dump(obj, f)

    # Meta with cold info
    meta = {
        **meta_root,
        "cold_ratio": ratio,
        "cold_items": sorted(cold_items),     # ← KEY: all loaders read from here
        "n_cold_items": len(cold_items),
        "n_train": len(train_clean),
        "n_val": len(val_clean),
        "n_test": len(test_clean),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Verify non-leaking: cold items must not appear in ANY prefix
    train_items = {it for sess in train_clean for it in sess}
    leak_train = cold_items & train_items
    assert len(leak_train) == 0, f"LEAK in train: {len(leak_train)} cold items!"

    val_items = {it for sess in val_clean for it in sess}
    leak_val = cold_items & val_items
    assert len(leak_val) == 0, f"LEAK in val: {len(leak_val)} cold items!"

    test_prefix_items = {it for sess in test_clean for it in sess[:-1]}
    leak_test_prefix = cold_items & test_prefix_items
    assert len(leak_test_prefix) == 0, \
        f"LEAK in test prefix: {len(leak_test_prefix)} cold items in context!"

    # Count cold-target test sessions
    n_cold_test = sum(1 for sess in test_clean if sess[-1] in cold_items)

    print(f"  cold_{ratio}/ → train={len(train_clean):,} | val={len(val_clean):,} | test={len(test_clean):,} | cold_items={len(cold_items):,}")
    print(f"    Non-leak check: PASS (train={len(leak_train)}, val={len(leak_val)}, test_prefix={len(leak_test_prefix)})")
    print(f"    Cold-target test sessions: {n_cold_test:,} / {len(test_clean):,}")


# ─── Sanity: verify nested property ───────────────────────────────────────────
def verify_nested(cold_by_ratio: dict[int, set[int]], ratios: list[int]) -> None:
    ratios_sorted = sorted(ratios)
    for i in range(len(ratios_sorted) - 1):
        r_small = ratios_sorted[i]
        r_large = ratios_sorted[i + 1]
        assert cold_by_ratio[r_small].issubset(cold_by_ratio[r_large]), \
            f"Nested violation: cold_{r_small} ⊄ cold_{r_large}"
    print("  Nested check: PASS (cold_10 ⊂ cold_20 ⊂ cold_30)")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Unified data dir with 7 files")
    parser.add_argument("--ratios", nargs="+", type=int, default=[10, 20, 30])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=== Cold-Start Split ===")
    print(f"Data dir: {args.data_dir}")
    print(f"Ratios:   {args.ratios}")
    print(f"Seed:     {args.seed}\n")

    # Load unified files
    train = load_sessions(os.path.join(args.data_dir, "sessions_train.txt"))
    val   = load_sessions(os.path.join(args.data_dir, "sessions_val.txt"))
    test  = load_sessions(os.path.join(args.data_dir, "sessions_test.txt"))

    item2cat   = load_json(os.path.join(args.data_dir, "item2cat.json"))
    cat2items  = load_json(os.path.join(args.data_dir, "cat2items.json"))
    cat_parent = load_json(os.path.join(args.data_dir, "cat_parent.json"))
    meta_root  = load_json(os.path.join(args.data_dir, "meta.json"))

    print(f"Loaded: train={len(train):,} | val={len(val):,} | test={len(test):,}")
    print(f"Items: {meta_root['n_items']:,} | Cats: {meta_root['n_cats']:,}\n")

    print("[1/3] Sampling cold items (nested)...")
    cold_by_ratio = sample_cold_items_nested(
        {k: [int(i) for i in v] for k, v in cat2items.items()},
        args.ratios,
        seed=args.seed,
    )

    print("\n[2/3] Verifying nested property...")
    verify_nested(cold_by_ratio, args.ratios)

    print("\n[3/3] Writing cold split folders...")
    for ratio in sorted(args.ratios):
        write_cold_split(
            base_dir=args.data_dir,
            ratio=ratio,
            train=train, val=val, test_orig=test,
            cold_items=cold_by_ratio[ratio],
            item2cat=item2cat,
            cat2items=cat2items,
            cat_parent=cat_parent,
            meta_root=meta_root,
        )

    print(f"\n✓ Done. Created: {['cold_' + str(r) for r in sorted(args.ratios)]}")
    print("  Next: python preprocessing/sanity_check.py --data_dir <data_dir>/cold_20")


if __name__ == "__main__":
    main()
