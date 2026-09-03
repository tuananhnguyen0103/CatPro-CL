"""
data_loader.py
--------------
Unified entry point for loading data.

Flow:
  1. Check if cache exists at data_dir/cache/
  2. If NOT: call build_graphs.py automatically (runs once, ~5-10 min)
  3. If YES:  load from .pkl in ~5 seconds
  4. Return (train_loader, val_loader, test_loader, data_dict)

Usage in train.py:
  from catprocl.data_loader import get_dataloaders

  train_loader, val_loader, test_loader, data = get_dataloaders(
      data_dir="data/retailrocket_unified/cold_20",
      batch_size=100,
  )
  # data["n_items"], data["cold_items"], data["item2cat"], etc.

  # SANITY CHECK — always verify this:
  assert len(data["cold_items"]) == 9052, f"Expected 9052, got {len(data['cold_items'])}"
"""

import json
import os
import pickle
import platform
import sys
from typing import Optional
from multiprocessing import cpu_count

import torch
from torch.utils.data import DataLoader

from catprocl.dataset import SessionGraphCollator, SessionGraphDataset


# ─── Load from cache ──────────────────────────────────────────────────────────
def _load_pkl(fpath: str) -> list:
    with open(fpath, "rb") as f:
        return pickle.load(f)


def _load_cache(cache_dir: str) -> tuple[list, list, list, dict]:
    """Load all .pkl files and cache_meta.json from cache_dir."""
    print(f"Loading graph cache from: {cache_dir}")

    train_graphs = _load_pkl(os.path.join(cache_dir, "train_graphs.pkl"))
    val_graphs   = _load_pkl(os.path.join(cache_dir, "val_graphs.pkl"))
    test_graphs  = _load_pkl(os.path.join(cache_dir, "test_graphs.pkl"))

    with open(os.path.join(cache_dir, "cache_meta.json")) as f:
        cache_meta = json.load(f)

    print(f"  Loaded: train={len(train_graphs):,} | val={len(val_graphs):,} | test={len(test_graphs):,}")
    print(f"  n_items={cache_meta['n_items']:,} | cold_items={cache_meta['n_cold']:,}")

    return train_graphs, val_graphs, test_graphs, cache_meta


# ─── Build cache if missing ───────────────────────────────────────────────────
def _ensure_cache(data_dir: str, cache_dir: str, maxlen: int = 0) -> None:
    """Call build_graphs.py if cache doesn't exist yet."""
    required = ["train_graphs.pkl", "val_graphs.pkl", "test_graphs.pkl", "cache_meta.json"]
    if all(os.path.exists(os.path.join(cache_dir, f)) for f in required):
        return  # cache exists

    print(f"Cache not found at {cache_dir}. Building now (runs once)...")
    if maxlen > 0:
        print(f"  Session maxlen={maxlen} (keep last {maxlen} items per session)")
    print("This may take 5-10 minutes for RetailRocket.\n")

    # Import and call directly (avoids subprocess overhead)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from preprocessing.build_graphs import build_and_cache
    build_and_cache(data_dir, cache_dir, force=False, maxlen=maxlen)


# ─── Main entry point ─────────────────────────────────────────────────────────
def _auto_num_workers() -> int:
    """
    Tự động chọn num_workers phù hợp với OS:
    - Windows: 0 (tránh lỗi spawn multiprocessing với CUDA)
    - Linux (server): min(4, cpu_count // 2) — safe với CUDA + fork
    """
    if platform.system() == "Windows":
        return 0
    return min(6, max(1, cpu_count() // 2))


def get_dataloaders(
    data_dir: str,
    batch_size: int = 100,
    num_workers: int = -1,     # -1 = auto-detect theo OS
    cache_dir: Optional[str] = None,
    pin_memory: bool = True,
    maxlen: int = 0,           # 0 = full session; N = keep last N items
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Returns: (train_loader, val_loader, test_loader, data_dict)

    data_dict keys:
      n_items     : int
      n_cats      : int
      cold_items  : set[int]
      item2cat    : dict[int, int]   {item_id: cat_id}
      cat2items   : dict[int, list]  {cat_id: [item_ids]}  (warm items only in train)
    """
    data_dir = os.path.expanduser(data_dir)
    if cache_dir is None:
        # CATPROCL_CACHE_ROOT: redirect cache ra ngoài data_dir
        # Dùng khi data_dir là read-only (ví dụ: symlink vào Kaggle input)
        _cache_root = os.environ.get("CATPROCL_CACHE_ROOT")
        if _cache_root:
            # Tạo key từ data_dir để phân biệt các variant
            _key = data_dir.rstrip("/").replace("/", "_").strip("_")
            if maxlen > 0:
                cache_dir = os.path.join(_cache_root, _key, f"cache_maxlen{maxlen}")
            else:
                cache_dir = os.path.join(_cache_root, _key, "cache")
        elif maxlen > 0:
            cache_dir = os.path.join(data_dir, f"cache_maxlen{maxlen}")
        else:
            cache_dir = os.path.join(data_dir, "cache")

    # Auto num_workers
    if num_workers == -1:
        num_workers = _auto_num_workers()

    # Build cache if needed
    _ensure_cache(data_dir, cache_dir, maxlen=maxlen)

    # Load from cache
    train_graphs, val_graphs, test_graphs, meta = _load_cache(cache_dir)

    # Build datasets
    train_ds = SessionGraphDataset(train_graphs)
    val_ds   = SessionGraphDataset(val_graphs)
    test_ds  = SessionGraphDataset(test_graphs)

    collate_fn = SessionGraphCollator()

    # persistent_workers=True: workers không bị kill/restart giữa các epoch
    # → tiết kiệm ~1-2s mỗi epoch (quan trọng khi train 30 epochs)
    # Chỉ dùng khi num_workers > 0
    persist = (num_workers > 0)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=pin_memory, drop_last=False,
        persistent_workers=persist,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=pin_memory, drop_last=False,
        persistent_workers=persist,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=pin_memory, drop_last=False,
        persistent_workers=persist,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    print(f"  DataLoader: num_workers={num_workers} | "
          f"persistent_workers={persist} | pin_memory={pin_memory}")

    # Build data_dict for training
    cold_items = set(meta["cold_items"])
    item2cat   = {int(k): int(v) for k, v in meta["item2cat"].items()}

    # cat2items: only warm (non-cold) items — used for prototype bank warm_start
    cat2items_warm: dict[int, list[int]] = {}
    for item_id, cat_id in item2cat.items():
        if item_id not in cold_items:
            cat2items_warm.setdefault(cat_id, []).append(item_id)

    data_dict = {
        "n_items":        meta["n_items"],
        "n_cats":         meta.get("n_cats", 0),
        "cold_items":     cold_items,
        "item2cat":       item2cat,
        "cat2items_warm": cat2items_warm,
    }

    # ── Sanity check ──────────────────────────────────────────────────────────
    n_cold = len(cold_items)
    print(f"\nData loaded | n_items={meta['n_items']:,} | cold_items={n_cold:,} "
          f"| train_batches={len(train_loader):,} | val_batches={len(val_loader):,}")

    if n_cold == 0:
        raise RuntimeError("cold_items is empty! Run cold_start_split.py first.")
    if n_cold < 1000:
        print(f"  ⚠ WARNING: cold_items={n_cold} seems too low. "
              "Check data_dir points to cold_20/ and data_loader.py is synced.")

    return train_loader, val_loader, test_loader, data_dict


# ─── Allow direct test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=100)
    args = parser.parse_args()

    train_loader, val_loader, test_loader, data = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )

    # Peek at one batch
    batch = next(iter(train_loader))
    print("\nSample batch shapes:")
    for k, v in batch.items():
        print(f"  {k}: {tuple(v.shape)}")
    print(f"\ncold_items count: {len(data['cold_items']):,}")
    print(f"item2cat entries: {len(data['item2cat']):,}")
    print(f"cat2items_warm cats: {len(data['cat2items_warm']):,}")
