"""
dc2r_official.py
----------------
DC2R baseline using the ORIGINAL source code (dc2r_src/model.py + utils.py).
Differences from DC2R's main.py:
  - Input data: our cold_20 7-file format (converted to DC2R pickle on-the-fly)
  - Evaluation: our HR@K/MRR@K with Overall vs Cold split
  - Early stopping: on VAL set (not test set like DC2R's main.py)
  - --dataset  : flexible dataset name (no longer hardcoded to 'cellphones')
  - --use_cat_as_attr: for datasets WITHOUT brand/price (RetailRocket, Diginetica)
      → attr1 = leaf_cat_id, attr2 = parent_cat_id (or leaf_cat if no parent)
      → taxo1 = leaf_cat_id, taxo2 = taxo3 = parent_cat_id
    This is the closest approximation to DC2R's taxonomy when only flat
    category + optional hierarchy (cat_parent.json) is available.

Dependencies:
  pip install recbole   (for TransformerEncoder used inside DC2Rtrm)

Usage — CellPhones (with full brand/price/taxo data):
  python baselines/dc2r_official.py \
      --data_dir ~/data/amazon_cellphones_unified/cold_20 \
      --output_dir ~/results_v6/data/cellphones/fullen \
      --dataset cellphones --seed 42

Usage — RetailRocket / Diginetica (category only):
  python baselines/dc2r_official.py \
      --data_dir ~/data/retailrocket_unified/cold_20 \
      --output_dir ~/results_v6/data/retailrocket/fullen \
      --dataset retailrocket --use_cat_as_attr --seed 42
"""

import argparse
import json
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# ── Import DC2R's original source ─────────────────────────────────────────────
_DC2R_SRC = os.path.join(os.path.dirname(__file__), 'dc2r_src')
sys.path.insert(0, _DC2R_SRC)
from model import DC2Rtrm, trans_to_cuda, trans_to_cpu   # DC2R originals
from utils import Data as DC2RData                        # DC2R Data class


# ─── Seed ─────────────────────────────────────────────────────────────────────

def set_seed(seed):
    seed = int(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_sessions(filepath, maxlen=0):
    """Load sessions from txt file. maxlen>0: keep last N items per session."""
    sessions = []
    with open(filepath) as f:
        for line in f:
            items = list(map(int, line.strip().split()))
            if maxlen > 0:
                items = items[-maxlen:]
            if len(items) >= 2:
                sessions.append(items)
    return sessions


def sessions_to_dc2r_pairs(sessions, augment=False):
    """
    Convert full sessions → DC2R (prefix_list, target) pairs.
    augment=True: all-prefix pairs (for training)
    augment=False: last-item-only (for val/test)
    """
    seqs, tgts = [], []
    for sess in sessions:
        if augment:
            for i in range(2, len(sess) + 1):
                prefix = sess[:i - 1]
                target = sess[i - 1]
                seqs.append(prefix)
                tgts.append(target)
        else:
            seqs.append(sess[:-1])
            tgts.append(sess[-1])
    return (seqs, tgts)


def build_attr_taxo_arrays(item2tax, n_items):
    """
    Build all_attr and all_taxo arrays from item2tax.json.

    all_attr[i] = [brand_id, price_bucket]  — indexed by item_id (0 = padding)
    all_taxo[i] = [t1_id, t2_id, t3_id]    — indexed by item_id (0 = padding)

    DC2R uses ONE embedding table for attr1+attr2 (n_attr1 must cover both IDs).
    DC2R uses ONE embedding table for t1+t2+t3 (n_taxo must cover all IDs).
    """
    # Build brand map
    brands = sorted({str(v.get('brand', '')) for v in item2tax.values()})
    brand2id = {b: i + 1 for i, b in enumerate(brands)}  # 1-indexed; 0=unknown

    # Price buckets
    prices = [float(v.get('price', 0)) for v in item2tax.values()
              if float(v.get('price', 0)) > 0]
    if prices:
        p_min, p_max = min(prices), max(prices)
        n_buckets = 20
        bucket_w  = max((p_max - p_min) / n_buckets, 1e-9)
    else:
        p_min, bucket_w, n_buckets = 0.0, 1.0, 20

    def price_bucket(p):
        p = float(p) if p else 0.0
        if p <= 0:
            return 0
        b = int((p - p_min) / bucket_w)
        return min(b + 1, n_buckets)

    # Arrays indexed 0..n_items (0 = padding)
    all_attr = [[0, 0]] * (n_items + 1)
    all_taxo = [[0, 0, 0]] * (n_items + 1)

    max_attr_id = 0
    max_taxo_id = 0

    for iid_str, tax in item2tax.items():
        iid = int(iid_str)
        if not (1 <= iid <= n_items):
            continue
        b_id  = brand2id.get(str(tax.get('brand', '')), 0)
        p_id  = price_bucket(tax.get('price', 0))
        t1_id = int(tax.get('t1_id', 0))
        t2_id = int(tax.get('t2_id', 0))
        t3_id = int(tax.get('t3_id', 0))

        all_attr[iid] = [b_id, p_id]
        all_taxo[iid] = [t1_id, t2_id, t3_id]

        max_attr_id = max(max_attr_id, b_id, p_id)
        max_taxo_id = max(max_taxo_id, t1_id, t2_id, t3_id)

    n_attr1 = max_attr_id + 1   # size of shared attribute embedding table
    n_taxo  = max_taxo_id + 1   # size of shared taxonomy embedding table

    return all_attr, all_taxo, n_attr1, n_taxo


def build_attr_taxo_from_cat(item2cat_raw, cat_parent_raw, n_items):
    """
    Build attr/taxo arrays for datasets without brand/price (RR, Diginetica).

    Mapping:
      attr1[item] = leaf_cat_id  (1-indexed, shared embedding table with attr2)
      attr2[item] = parent_cat_id (or leaf_cat_id if no parent exists)
      taxo1[item] = leaf_cat_id
      taxo2[item] = parent_cat_id (or leaf_cat_id)
      taxo3[item] = parent_cat_id (or leaf_cat_id)  ← same as taxo2 (2-level max)

    item2cat_raw  : {str(item_id): int(cat_id)}
    cat_parent_raw: {str(cat_id): parent_id_or_null}
    """
    # Build cat_id → compact index (1-indexed; 0 = padding)
    all_cats = sorted({int(c) for c in item2cat_raw.values()})
    cat2idx  = {c: i + 1 for i, c in enumerate(all_cats)}

    # Build cat_id → parent_cat_id (int or None)
    cat2parent = {}
    for k, v in cat_parent_raw.items():
        cat = int(k)
        cat2parent[cat] = int(v) if v is not None else None

    max_idx = len(cat2idx)   # largest compact index used

    all_attr = [[0, 0]] * (n_items + 1)
    all_taxo = [[0, 0, 0]] * (n_items + 1)

    for iid_str, cat in item2cat_raw.items():
        iid = int(iid_str)
        if not (1 <= iid <= n_items):
            continue
        cat    = int(cat)
        c_idx  = cat2idx.get(cat, 0)
        parent = cat2parent.get(cat, None)
        p_idx  = cat2idx.get(parent, c_idx) if parent is not None else c_idx

        all_attr[iid] = [c_idx, p_idx]
        all_taxo[iid] = [c_idx, p_idx, p_idx]

    n_attr1 = max_idx + 1
    n_taxo  = max_idx + 1
    return all_attr, all_taxo, n_attr1, n_taxo


def load_cold_items(data_dir):
    # 1) Standalone cold_items.json (legacy)
    cold_file = os.path.join(data_dir, 'cold_items.json')
    if os.path.exists(cold_file):
        return set(json.load(open(cold_file)))
    # 2) Read from meta.json (our format: cold_start_split.py saves "cold_items" list)
    meta = json.load(open(os.path.join(data_dir, 'meta.json')))
    if 'cold_items' in meta:
        return set(meta['cold_items'])
    # 3) Last-resort fallback: assume cold items are the last n_cold IDs (WRONG
    #    for stratified splits — only valid if items are assigned sequentially by cold status)
    n_items = meta['n_items']
    n_cold  = meta.get('n_cold_items', 0)
    return set(range(n_items - n_cold + 1, n_items + 1))


# ─── Pre-cache get_slice (Opt-1) ──────────────────────────────────────────────

def build_batch_cache(data, attr_data, taxo_data, batch_size=100, desc="caching"):
    """
    Pre-compute get_slice() for every batch and store in memory as pinned CPU tensors.
    DC2R's shuffle is a no-op (np.random.shuffle commented out in utils.py),
    so batch content is identical every epoch → safe to cache once.

    Opt-2: Pre-convert list[np.ndarray] → pinned CPU tensor during build (one-time cost).
    This eliminates the slow torch.Tensor(list_of_ndarrays) conversion every forward pass.
    non_blocking=True in forward then uses async DMA transfer to GPU.

    Returns: list of pre-computed slice tuples (pinned CPU tensors), one per batch.
    """
    slices = data.generate_batch(batch_size)
    cache  = []
    for idx in tqdm(slices, desc=f"  {desc}", unit="batch", leave=False):
        alias_inputs, items, mask, targets, \
            attr1, attr2, taxo1, taxo2, taxo3, ca_a1, ca_a2, A = \
            data.get_slice(idx, attr_data, taxo_data)
        cache.append((
            torch.LongTensor(np.array(alias_inputs)).pin_memory(),
            torch.LongTensor(np.array(items)).pin_memory(),
            torch.LongTensor(np.array(mask)).pin_memory(),
            torch.LongTensor(np.array(targets)).pin_memory(),
            torch.LongTensor(np.array(attr1)).pin_memory(),
            torch.LongTensor(np.array(attr2)).pin_memory(),
            torch.LongTensor(np.array(taxo1)).pin_memory(),
            torch.LongTensor(np.array(taxo2)).pin_memory(),
            torch.LongTensor(np.array(taxo3)).pin_memory(),
            torch.LongTensor(np.array(ca_a1)).pin_memory(),
            torch.LongTensor(np.array(ca_a2)).pin_memory(),
            torch.from_numpy(np.array(A, dtype=np.float32)).pin_memory(),
        ))
    return cache


# ─── DC2R forward from cached slice ───────────────────────────────────────────

def dc2r_forward_cached(model, cached_slice, sentinel):
    """
    DC2R forward using pre-built pinned CPU tensors (from build_batch_cache).
    non_blocking=True enables async DMA transfer: GPU compute overlaps with H2D copy.
    """
    alias_inputs, items, mask, targets, \
        attr1, attr2, taxo1, taxo2, taxo3, ca_a1, ca_a2, A = cached_slice

    dev = next(model.parameters()).device
    alias_inputs = alias_inputs.to(dev, non_blocking=True)
    items        = items.to(dev, non_blocking=True)
    mask         = mask.to(dev, non_blocking=True)
    A            = A.to(dev, non_blocking=True)
    attr1        = attr1.to(dev, non_blocking=True)
    attr2        = attr2.to(dev, non_blocking=True)
    taxo1        = taxo1.to(dev, non_blocking=True)
    taxo2        = taxo2.to(dev, non_blocking=True)
    taxo3        = taxo3.to(dev, non_blocking=True)
    ca_a1        = ca_a1.to(dev, non_blocking=True)
    ca_a2        = ca_a2.to(dev, non_blocking=True)

    hidden, E_p, attr_emb, I_p = model(
        alias_inputs, A, mask, items, attr1, attr2, taxo1, taxo2, taxo3)

    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack(
        [get(i) for i in torch.arange(len(alias_inputs)).long()])
    zeroloss = model.zeroshot(seq_hidden, attr_emb, mask, alias_inputs)

    score1 = model.compute_scores(E_p)
    if sentinel:
        score2 = model.compute_cand_scores(I_p, ca_a1[1:], ca_a2[1:])
    else:
        score2 = model.compute_cand_scores(I_p, ca_a1, ca_a2)

    score = score1 + score2
    return targets, score, zeroloss


# ─── Training ─────────────────────────────────────────────────────────────────

def train_one_epoch(model, train_cache, sentinel, scaler, use_amp):
    """
    Training epoch using pre-cached batches + optional AMP.
    Opt-1: train_cache replaces get_slice() call every batch.
    Opt-2: autocast() uses fp16 Tensor Cores on RTX 3090.
    """
    model.scheduler.step()   # DC2R gốc: step trước khi train
    model.train()
    total_loss = 0.0
    pbar = tqdm(train_cache, desc="  train", unit="batch", leave=False)

    for cached_slice in pbar:
        model.optimizer.zero_grad()

        with autocast(enabled=use_amp):
            targets, scores, zeroloss = dc2r_forward_cached(model, cached_slice, sentinel)
            # targets already a pinned LongTensor from cache — move to GPU in-place
            dev = next(model.parameters()).device
            targets_t = targets.to(dev, non_blocking=True)
            loss = model.loss_function(scores, targets_t - 1) + zeroloss

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(model.optimizer)
            scaler.update()
        else:
            loss.backward()
            model.optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(len(train_cache), 1)


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_dc2r(model, eval_cache, sentinel, cold_items, use_amp, ks=(10, 20)):
    """
    Evaluate using pre-cached batches + optional AMP.
    Opt-1: eval_cache replaces get_slice().
    Opt-2: autocast() speeds up Transformer inference.
    Opt-3: argpartition O(n) instead of argsort O(n log n).
    """
    model.eval()
    hits   = {k: [] for k in ks}
    mrr    = {k: [] for k in ks}
    c_hits = {k: [] for k in ks}
    c_mrr  = {k: [] for k in ks}
    n_cold = 0
    max_k  = max(ks)

    with torch.no_grad():
        for cached_slice in tqdm(eval_cache, desc="  eval ", unit="batch", leave=False):
            with autocast(enabled=use_amp):
                targets, scores, _ = dc2r_forward_cached(model, cached_slice, sentinel)
            scores_np = trans_to_cpu(scores).detach().float().numpy()

            for score_row, target in zip(scores_np, targets):
                target_pos = int(target) - 1
                is_cold    = int(target) in cold_items

                # Opt-3: argpartition O(n) >> argsort O(n log n) for large n_items
                part  = np.argpartition(score_row, -max_k)[-max_k:]
                order = np.argsort(score_row[part])[::-1]
                top_max = part[order]   # indices of top-max_k items, sorted desc

                for k in ks:
                    topk = top_max[:k]
                    hit  = int(np.isin(target_pos, topk))
                    rank = np.where(topk == target_pos)[0]
                    rec  = float(1.0 / (rank[0] + 1)) if len(rank) > 0 else 0.0
                    hits[k].append(hit)
                    mrr[k].append(rec)
                    if is_cold:
                        c_hits[k].append(hit)
                        c_mrr[k].append(rec)

                if is_cold:
                    n_cold += 1

    n_total = len(hits[ks[0]])
    results = {'overall': {}, 'cold': {}}
    for k in ks:
        results['overall'][f'HR@{k}']  = float(np.mean(hits[k]))
        results['overall'][f'MRR@{k}'] = float(np.mean(mrr[k]))
        results['cold'][f'HR@{k}']  = float(np.mean(c_hits[k])) if c_hits[k] else 0.0
        results['cold'][f'MRR@{k}'] = float(np.mean(c_mrr[k])) if c_mrr[k] else 0.0
    results['n_total'] = n_total
    results['n_cold']  = n_cold
    return results


def print_results(tag, res):
    ov, co = res['overall'], res['cold']
    print("=" * 60)
    print(f"TEST RESULTS — {tag}")
    print(f"  Overall | HR@10={ov.get('HR@10',0):.4f}  HR@20={ov.get('HR@20',0):.4f}  "
          f"MRR@10={ov.get('MRR@10',0):.4f}  MRR@20={ov.get('MRR@20',0):.4f}  "
          f"(n={res['n_total']})")
    print(f"  Cold    | HR@10={co.get('HR@10',0):.4f}  HR@20={co.get('HR@20',0):.4f}  "
          f"MRR@10={co.get('MRR@10',0):.4f}  MRR@20={co.get('MRR@20',0):.4f}  "
          f"(n={res['n_cold']})")
    print("=" * 60)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='DC2R official — our data + our evaluator')
    parser.add_argument('--data_dir',       required=True,
                        help='Path to cold_20 directory')
    parser.add_argument('--output_dir',     default='results_baseline/cellphones')
    parser.add_argument('--dataset',        default='cellphones',
                        help='Dataset name tag used in output filename')
    parser.add_argument('--use_cat_as_attr', action='store_true', default=False,
                        help='Use item2cat + cat_parent as attr/taxo (for RR/Digi without brand/price)')

    # DC2R hyperparams (from DC2R's main.py, CellPhones defaults)
    parser.add_argument('--hiddenSize',         type=int,   default=100)
    parser.add_argument('--batchSize',          type=int,   default=100)
    parser.add_argument('--epoch',              type=int,   default=30)
    parser.add_argument('--lr',                 type=float, default=0.005)
    parser.add_argument('--lr_dc',              type=float, default=0.1)
    parser.add_argument('--lr_dc_step',         type=int,   default=3)
    parser.add_argument('--l2',                 type=float, default=1e-5)
    parser.add_argument('--gama',               type=float, default=1.0)
    parser.add_argument('--temperature',        type=float, default=0.005)
    parser.add_argument('--sentinel',           type=lambda x: x.lower() == 'true',  default=False)
    parser.add_argument('--maxlen',             type=int,   default=0,
                        help='Truncate sessions to last N items (0=full session)')
    parser.add_argument('--n_layers',           type=int,   default=2)
    parser.add_argument('--n_heads',            type=int,   default=2)
    parser.add_argument('--inner_size',         type=int,   default=256)
    parser.add_argument('--hidden_dropout_prob',type=float, default=0.2)
    parser.add_argument('--attn_dropout_prob',  type=float, default=0.2)
    parser.add_argument('--layer_norm_eps',     type=float, default=1e-12)
    parser.add_argument('--initializer_range',  type=float, default=0.02)
    parser.add_argument('--hidden_act',         type=str,   default='gelu')
    parser.add_argument('--patience',           type=int,   default=5)
    parser.add_argument('--seed',               type=int,   default=42)
    parser.add_argument('--gpu_id',             type=str,   default='0')
    parser.add_argument('--use_amp',            type=lambda x: x.lower() == 'true',
                        default=True,
                        help='AMP fp16 (Opt-2). Default True on CUDA, safe to disable.')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    out_json = os.path.join(args.output_dir, f'{args.dataset}_DC2R_seed{args.seed}.json')
    if os.path.exists(out_json):
        print(f'[SKIP] {out_json} already exists.')
        return

    # ── Load our data ──
    print(f'Loading data from {args.data_dir} ...')
    print(f'  dataset={args.dataset} | use_cat_as_attr={args.use_cat_as_attr}')
    meta    = json.load(open(os.path.join(args.data_dir, 'meta.json')))
    n_items = meta['n_items']
    cold_items = load_cold_items(args.data_dir)
    print(f'  n_items={n_items:,} | cold_items={len(cold_items):,}')

    train_sess = load_sessions(os.path.join(args.data_dir, 'sessions_train.txt'), maxlen=args.maxlen)
    val_sess   = load_sessions(os.path.join(args.data_dir, 'sessions_val.txt'),   maxlen=args.maxlen)
    test_sess  = load_sessions(os.path.join(args.data_dir, 'sessions_test.txt'),  maxlen=args.maxlen)

    # ── Build attr/taxo arrays ─────────────────────────────────────────────────
    parent_dir    = os.path.dirname(args.data_dir)

    if args.use_cat_as_attr:
        # ── Mode: RR / Diginetica — use item2cat + cat_parent ──
        item2cat_path = os.path.join(args.data_dir, 'item2cat.json')
        if not os.path.exists(item2cat_path):
            item2cat_path = os.path.join(parent_dir, 'item2cat.json')
        cat_parent_path = os.path.join(args.data_dir, 'cat_parent.json')
        if not os.path.exists(cat_parent_path):
            cat_parent_path = os.path.join(parent_dir, 'cat_parent.json')

        item2cat_raw   = json.load(open(item2cat_path))
        cat_parent_raw = json.load(open(cat_parent_path)) if os.path.exists(cat_parent_path) else {}
        all_attr, all_taxo, n_attr1, n_taxo = build_attr_taxo_from_cat(
            item2cat_raw, cat_parent_raw, n_items)
        n_with_parent = sum(1 for v in cat_parent_raw.values() if v is not None)
        print(f'  use_cat_as_attr: n_attr1={n_attr1}  n_taxo={n_taxo}  '
              f'cats_with_parent={n_with_parent}/{len(cat_parent_raw)}')
    else:
        # ── Mode: CellPhones — use item2tax.json (brand + price + taxo) ──
        item2tax_path = os.path.join(parent_dir, 'item2tax.json')
        if not os.path.exists(item2tax_path):
            item2tax_path = os.path.join(args.data_dir, 'item2tax.json')

        if os.path.exists(item2tax_path):
            item2tax = json.load(open(item2tax_path))
            all_attr, all_taxo, n_attr1, n_taxo = build_attr_taxo_arrays(item2tax, n_items)
            print(f'  item2tax: n_attr1={n_attr1}  n_taxo={n_taxo}')
        else:
            print('  WARNING: item2tax.json not found — using dummy attrs/taxo')
            all_attr  = [[0, 0]]  * (n_items + 1)
            all_taxo  = [[0, 0, 0]] * (n_items + 1)
            n_attr1, n_taxo = 2, 2

    # Patch args with our computed n_attr1 / n_taxo / n_node
    args.n_attr1 = n_attr1
    args.n_taxo  = n_taxo
    n_node       = n_items + 1   # DC2R embedding size (includes padding at 0)

    # ── Convert sessions → DC2R Data objects ──
    print('Converting sessions to DC2R format ...')
    train_pairs = sessions_to_dc2r_pairs(train_sess, augment=True)
    val_pairs   = sessions_to_dc2r_pairs(val_sess,   augment=False)
    test_pairs  = sessions_to_dc2r_pairs(test_sess,  augment=False)
    print(f'  train pairs (all-prefix): {len(train_pairs[0]):,}')
    print(f'  val pairs:  {len(val_pairs[0]):,}')
    print(f'  test pairs: {len(test_pairs[0]):,}')

    train_data = DC2RData(train_pairs, shuffle=True)
    val_data   = DC2RData(val_pairs,   shuffle=False)
    test_data  = DC2RData(test_pairs,  shuffle=False)

    # ── Build DC2Rtrm model ──
    use_amp = args.use_amp and torch.cuda.is_available()
    print(f'\nBuilding DC2Rtrm model | n_node={n_node} | hidden={args.hiddenSize}')
    model = trans_to_cuda(DC2Rtrm(args, n_node))
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Parameters: {n_params:,}')
    print(f'  sentinel={args.sentinel} | temperature={args.temperature}')
    print(f'  n_attr1={n_attr1} | n_taxo={n_taxo}')
    print(f'  use_amp={use_amp}')

    # ── Opt-1: Pre-cache get_slice() for all batches ──────────────────────────
    # get_slice() builds adjacency matrices with Python loops — expensive per batch.
    # DC2R's shuffle is a no-op → same batches every epoch → cache once, reuse.
    print('\nPre-caching graph batches (one-time CPU cost)...')
    t_cache = time.time()
    train_cache = build_batch_cache(train_data, all_attr, all_taxo, args.batchSize, "cache train")
    val_cache   = build_batch_cache(val_data,   all_attr, all_taxo, args.batchSize, "cache val  ")
    test_cache  = build_batch_cache(test_data,  all_attr, all_taxo, args.batchSize, "cache test ")
    print(f'  Cache built in {time.time() - t_cache:.1f}s '
          f'({len(train_cache)} train / {len(val_cache)} val / {len(test_cache)} test batches)')

    # ── Opt-2: AMP GradScaler ─────────────────────────────────────────────────
    scaler = GradScaler(enabled=use_amp)

    # ── Training loop ──
    best_val_hr20 = 0.0
    best_epoch    = 0
    bad_counter   = 0
    best_state    = None
    epoch_log     = []   # history per epoch

    for epoch in range(args.epoch):
        t0   = time.time()
        loss = train_one_epoch(model, train_cache, args.sentinel, scaler, use_amp)
        val_res  = evaluate_dc2r(model, val_cache, args.sentinel, cold_items, use_amp)
        val_hr20 = val_res['overall'].get('HR@20', 0.0)
        val_mrr20 = val_res['overall'].get('MRR@20', 0.0)
        elapsed  = time.time() - t0

        improved = val_hr20 > best_val_hr20
        if improved:
            best_val_hr20 = val_hr20
            best_epoch    = epoch
            bad_counter   = 0
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            mark = '← best'
        else:
            bad_counter += 1
            mark = f'(no improve {bad_counter}/{args.patience})'

        epoch_log.append({
            'epoch':      epoch + 1,
            'loss':       round(loss, 6),
            'val_hr20':   round(val_hr20, 6),
            'val_mrr20':  round(val_mrr20, 6),
            'elapsed_s':  round(elapsed, 1),
            'best':       improved,
        })

        print(f'  Epoch {epoch+1:3d} | loss={loss:.4f} | Val HR@20={val_hr20:.4f} | '
              f'{elapsed:.1f}s | {mark}')

        if bad_counter >= args.patience:
            print(f'Early stopping at epoch {epoch+1} (best epoch={best_epoch+1})')
            break

    # ── Test evaluation with best model ──
    if best_state is not None:
        model.load_state_dict({k: v.cuda() if torch.cuda.is_available() else v
                               for k, v in best_state.items()})

    test_res = evaluate_dc2r(model, test_cache, args.sentinel, cold_items, use_amp)
    print_results(f'{args.dataset} | DC2R (official) | seed={args.seed}', test_res)

    # ── Save ──
    output = {
        'model':          'DC2R_official',
        'dataset':        args.dataset,
        'use_cat_as_attr': args.use_cat_as_attr,
        'seed':           args.seed,
        'best_epoch':  best_epoch + 1,
        'best_val_hr20': best_val_hr20,
        'n_epochs_run': len(epoch_log),
        'hparams': {
            'hiddenSize':   args.hiddenSize,
            'n_attr1':      n_attr1,
            'n_taxo':       n_taxo,
            'n_node':       n_node,
            'temperature':  args.temperature,
            'sentinel':     args.sentinel,
            'lr':           args.lr,
            'l2':           args.l2,
            'gama':         args.gama,
            'n_layers':     args.n_layers,
            'n_heads':      args.n_heads,
        },
        'training_history': epoch_log,
        'results': test_res,
    }
    with open(out_json, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'Results saved → {out_json}')


if __name__ == '__main__':
    main()
