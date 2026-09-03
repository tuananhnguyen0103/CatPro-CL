"""
build_graphs.py
---------------
Pre-compute session graphs for SR-GNN and save to .pkl cache.
This runs ONCE. DataLoader then loads from cache in ~5 seconds.

Why caching matters:
  Without cache: each training run rebuilds graphs → 30min-1h wasted
  With cache:    load from .pkl → DataLoader ready in <10 seconds

What gets pre-computed per session:
  node_ids  : unique item IDs in this session (local re-indexed, 1-based within session)
  A_in      : normalized in-adjacency matrix  (n_nodes, n_nodes)
  A_out     : normalized out-adjacency matrix (n_nodes, n_nodes)
  seq_idx   : session items as local indices into node_ids
  target    : global item ID of next item (prediction target)
  alias_idx : alias for seq_idx (same thing, kept for SR-GNN compat)
  mask      : 1 for valid positions, 0 for padding (set during collation)

Usage:
  python preprocessing/build_graphs.py \
      --data_dir data/retailrocket_unified/cold_20 \
      --output_dir data/retailrocket_unified/cold_20/cache

  # Or let data_loader.py call this automatically on first run.
"""

import argparse
import json
import os
import pickle
import sys
from multiprocessing import Pool, cpu_count
from typing import TypedDict

import numpy as np
from tqdm import tqdm


# ─── Types ────────────────────────────────────────────────────────────────────
class SessionGraph(TypedDict):
    node_ids: list[int]    # Global item IDs of unique nodes in session
    A_in:     np.ndarray   # (n_nodes, n_nodes) float32 — in-edges normalized
    A_out:    np.ndarray   # (n_nodes, n_nodes) float32 — out-edges normalized
    seq_idx:  list[int]    # Items in session as local indices (0-based into node_ids)
    target:   int          # Global item ID to predict
    n_nodes:  int          # Number of unique nodes


# ─── Session → Graph conversion ───────────────────────────────────────────────
def session_to_graph(session: list[int]) -> SessionGraph:
    """
    Convert one session (list of global item IDs) to an SR-GNN graph.

    SR-GNN treats a session as a directed graph:
      - Nodes: unique items in the session
      - Edges: consecutive item pairs  item[i] → item[i+1]
      - A_out[i][j] = normalized weight of edge node_i → node_j
      - A_in[i][j]  = normalized weight of edge node_j → node_i

    The last item is the TARGET (what we predict).
    The graph is built from session[:-1] (prefix).
    """
    # Split: prefix for graph, last item as target
    inputs  = session[:-1]   # items to build graph from
    target  = session[-1]    # item to predict

    # Unique nodes in order of first appearance
    node_ids = list(dict.fromkeys(inputs))  # preserves order, deduplicates
    n_nodes  = len(node_ids)

    # Local index map: global_id → local 0-based index
    global_to_local = {gid: i for i, gid in enumerate(node_ids)}

    # Build adjacency (allow multi-edges, normalize later)
    A_out = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    A_in  = np.zeros((n_nodes, n_nodes), dtype=np.float32)

    for i in range(len(inputs) - 1):
        src = global_to_local[inputs[i]]
        dst = global_to_local[inputs[i + 1]]
        A_out[src][dst] += 1
        A_in[dst][src]  += 1

    # Row-normalize (divide each row by its sum; rows with sum=0 stay 0)
    # QUAN TRỌNG: phải dùng out=np.zeros_like() để tránh uninitialized memory
    # Không có out= thì np.divide để garbage values ở vị trí where=False → NaN trong GNN
    row_sum_out = A_out.sum(axis=1, keepdims=True)
    row_sum_in  = A_in.sum(axis=1, keepdims=True)
    A_out = np.divide(A_out, row_sum_out, out=np.zeros_like(A_out), where=row_sum_out != 0)
    A_in  = np.divide(A_in,  row_sum_in,  out=np.zeros_like(A_in),  where=row_sum_in  != 0)

    # Session items as local indices (the "alias" sequence SR-GNN uses for attention)
    seq_idx = [global_to_local[it] for it in inputs]

    return SessionGraph(
        node_ids=node_ids,
        A_in=A_in,
        A_out=A_out,
        seq_idx=seq_idx,
        target=target,
        n_nodes=n_nodes,
    )


# ─── Worker function (must be top-level for multiprocessing pickle) ───────────
def _worker(sess: list[int]):
    """Top-level worker: convert 1 session → graph. Used by Pool.imap."""
    if len(sess) < 2:
        return None
    return session_to_graph(sess)


# ─── Process a full split (parallel) ─────────────────────────────────────────
def process_sessions(
    sessions: list[list[int]],
    split_name: str,
    n_workers: int = 0,
) -> list[SessionGraph]:
    """
    Convert all sessions in a split to graphs.

    n_workers=0  → auto-detect (cpu_count - 1, min 1)
    n_workers=1  → single-process (debug mode)
    n_workers>1  → explicit parallel

    NOTE: multiprocessing.Pool requires the script to run under
    `if __name__ == '__main__':` on Windows. On Linux server it works fine.
    """
    valid = [s for s in sessions if len(s) >= 2]
    n = len(valid)

    if n_workers == 0:
        n_workers = max(1, cpu_count() - 1)

    print(f"  [{split_name}] {n:,} sessions | {n_workers} workers")

    # Single-process fallback (debug only)
    if n_workers == 1:
        graphs = []
        for sess in tqdm(valid, desc=f"  {split_name}", unit="sess",
                         dynamic_ncols=True, colour="cyan"):
            graphs.append(session_to_graph(sess))
        return graphs

    # Parallel on Linux (server)
    graphs = []
    chunk  = max(1, n // (n_workers * 4))   # chunksize → giảm IPC overhead
    with Pool(processes=n_workers) as pool:
        for g in tqdm(
            pool.imap(_worker, valid, chunksize=chunk),
            total=n,
            desc=f"  {split_name}",
            unit="sess",
            dynamic_ncols=True,
            colour="cyan",
        ):
            if g is not None:
                graphs.append(g)

    print(f"  [{split_name}] {len(graphs):,} graphs built")
    return graphs


# ─── Load sessions ─────────────────────────────────────────────────────────────
def load_sessions(fpath: str) -> list[list[int]]:
    sessions = []
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if line:
                sessions.append(list(map(int, line.split())))
    return sessions


# ─── Load cold items and item2cat ─────────────────────────────────────────────
def load_meta(data_dir: str) -> dict:
    with open(os.path.join(data_dir, "meta.json")) as f:
        return json.load(f)


def load_item2cat(data_dir: str) -> dict[int, int]:
    with open(os.path.join(data_dir, "item2cat.json")) as f:
        raw = json.load(f)
    return {int(k): int(v) for k, v in raw.items()}


# ─── Main ─────────────────────────────────────────────────────────────────────
def build_and_cache(data_dir: str, output_dir: str, force: bool = False, n_workers: int = 0, maxlen: int = 0) -> None:
    """
    Build graphs for all 3 splits and save to output_dir/*.pkl.
    Skips if cache already exists (unless force=True).

    maxlen: nếu > 0, truncate mỗi session xuống maxlen cuối (keep last N items).
            Dùng cho các biến thể maxlen3/4/5/6.
    """
    os.makedirs(output_dir, exist_ok=True)

    cache_files = {
        "train": os.path.join(output_dir, "train_graphs.pkl"),
        "val":   os.path.join(output_dir, "val_graphs.pkl"),
        "test":  os.path.join(output_dir, "test_graphs.pkl"),
        "meta":  os.path.join(output_dir, "cache_meta.json"),
    }

    # Check if cache already exists
    if not force and all(os.path.exists(v) for v in cache_files.values()):
        print(f"✓ Cache already exists at {output_dir}. Use --force to rebuild.")
        return

    print(f"Building graph cache → {output_dir}")
    if maxlen > 0:
        print(f"  Session truncation: maxlen={maxlen} (keep last {maxlen} items)")
    print("(This runs ONCE. Future training loads from cache in <10s)\n")

    # Load sessions
    print("[1/4] Loading sessions...")
    train_sess = load_sessions(os.path.join(data_dir, "sessions_train.txt"))
    val_sess   = load_sessions(os.path.join(data_dir, "sessions_val.txt"))
    test_sess  = load_sessions(os.path.join(data_dir, "sessions_test.txt"))
    print(f"  train={len(train_sess):,} | val={len(val_sess):,} | test={len(test_sess):,}")

    # Apply maxlen truncation (keep last N items, min length still 2 for prefix+target)
    if maxlen > 0:
        train_sess = [s[-maxlen:] for s in train_sess if len(s[-maxlen:]) >= 2]
        val_sess   = [s[-maxlen:] for s in val_sess   if len(s[-maxlen:]) >= 2]
        test_sess  = [s[-maxlen:] for s in test_sess  if len(s[-maxlen:]) >= 2]
        print(f"  After maxlen={maxlen}: train={len(train_sess):,} | val={len(val_sess):,} | test={len(test_sess):,}")

    # Load mappings
    meta     = load_meta(data_dir)
    item2cat = load_item2cat(data_dir)
    cold_items = set(meta.get("cold_items", []))

    print(f"  n_items={meta['n_items']:,} | cold_items={len(cold_items):,}")
    if len(cold_items) == 0:
        print("  ⚠ WARNING: cold_items is empty. Did you run cold_start_split.py?")

    # Build graphs (parallel)
    print(f"\n[2/4] Building train graphs (parallel, workers={n_workers or 'auto'})...")
    train_graphs = process_sessions(train_sess, "train", n_workers)

    print("\n[3/4] Building val graphs...")
    val_graphs = process_sessions(val_sess, "val", n_workers)

    print("\n[4/4] Building test graphs...")
    test_graphs = process_sessions(test_sess, "test", n_workers)

    # Save to pkl
    print("\nSaving cache...")
    for name, graphs in [("train", train_graphs), ("val", val_graphs), ("test", test_graphs)]:
        fpath = cache_files[name]
        with open(fpath, "wb") as f:
            pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = os.path.getsize(fpath) / 1024 / 1024
        print(f"  {name}_graphs.pkl → {size_mb:.1f} MB")

    # Save cache metadata
    cache_meta = {
        "n_items":     meta["n_items"],
        "n_cats":      meta.get("n_cats", 0),
        "cold_items":  sorted(cold_items),
        "n_cold":      len(cold_items),
        "n_train":     len(train_graphs),
        "n_val":       len(val_graphs),
        "n_test":      len(test_graphs),
        "item2cat":    {str(k): v for k, v in item2cat.items()},
        "data_dir":    data_dir,
        "maxlen":      maxlen,
    }
    with open(cache_files["meta"], "w") as f:
        json.dump(cache_meta, f, indent=2)

    print(f"\n✓ Cache complete!")
    print(f"  Sanity: cold_items = {len(cold_items):,}  ← should be 9,052 for RetailRocket cold_20")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   required=True,  help="cold_XX folder với sessions + meta.json")
    parser.add_argument("--output_dir", default=None,   help="Nơi lưu .pkl (default: data_dir/cache)")
    parser.add_argument("--force",      action="store_true", help="Rebuild dù cache đã tồn tại")
    parser.add_argument("--n_workers",  type=int, default=0,
                        help="Số CPU workers (0=auto, 1=single, N=cụ thể). "
                             "Trên Windows tự động fallback về 1.")
    parser.add_argument("--maxlen",     type=int, default=0,
                        help="Truncate sessions to last N items (0=no truncation). "
                             "Output dir tự động là cache_maxlenN/ khi > 0.")
    args = parser.parse_args()

    if args.maxlen > 0:
        default_out = os.path.join(args.data_dir, f"cache_maxlen{args.maxlen}")
    else:
        default_out = os.path.join(args.data_dir, "cache")
    output_dir = args.output_dir or default_out
    build_and_cache(args.data_dir, output_dir, force=args.force, n_workers=args.n_workers, maxlen=args.maxlen)


if __name__ == "__main__":
    main()
