"""
evaluator.py — Full-ranking Evaluation (dùng chung cho MỌI baseline)
----------------------------------------------------------------------
Full-ranking: rank tất cả n_items, không dùng sampled negatives.
Tách Overall metrics vs Cold metrics.

Metrics: HR@K, MRR@K  (K = 10, 20 theo paper)
"""

import torch
import numpy as np
from typing import Optional


def evaluate(
    model_score_fn,              # callable(batch) → (B, n_items+1) logit scores
    data_loader,                 # DataLoader của val hoặc test
    item_emb: torch.Tensor,      # (n_items+1, d) — có thể là cold-replaced
    cold_items: set[int],        # set of cold item IDs
    ks: list[int] = [10, 20],
    device: str = "cuda",
) -> dict[str, float]:
    """
    Chạy full-ranking evaluation trên toàn bộ data_loader.

    model_score_fn: nhận batch dict, trả về (B, n_items+1) scores.
    Caller truyền hàm này để evaluator không phụ thuộc vào model cụ thể.

    Returns dict:
        HR@K, MRR@K                                         (Overall)
        Cold_HR@K, Cold_MRR@K                               (All cold targets)
        StrictCold_HR@K, StrictCold_MRR@K                  (First-time cold only)
        RevisitCold_HR@K, RevisitCold_MRR@K                (Revisit cold only)
        n_overall, n_cold, n_strict_cold, n_revisit_cold   (sample counts)

    Cold split:
      - "Cold"        : target in cold_items (includes both revisit and first-time)
      - "StrictCold"  : cold target NOT in session prefix (node_ids) → first-time cold
      - "RevisitCold" : cold target IS in session prefix → self-correlation present

    Note: node_ids in batch = unique items in session[:-1] (prefix, excl. target).
    A cold item that appeared earlier in the session shows up in node_ids — this
    creates self-correlation in mean-pooling models (e.g. CORE) that inflates Cold HR.
    StrictCold removes this effect and measures true cold-start capability.
    """
    all_hr   = {k: [] for k in ks}
    all_mrr  = {k: [] for k in ks}
    all_ndcg = {k: [] for k in ks}
    cold_hr   = {k: [] for k in ks}
    cold_mrr  = {k: [] for k in ks}
    cold_ndcg = {k: [] for k in ks}
    warm_hr   = {k: [] for k in ks}          # warm targets (not in cold_items)
    strict_cold_hr  = {k: [] for k in ks}   # first-time cold (not in prefix)
    strict_cold_mrr = {k: [] for k in ks}
    revisit_cold_hr  = {k: [] for k in ks}  # cold target appeared in prefix
    revisit_cold_mrr = {k: [] for k in ks}

    for batch in data_loader:
        # Move batch to device
        batch = {key: val.to(device) if isinstance(val, torch.Tensor) else val
                 for key, val in batch.items()}
        targets  = batch["targets"]   # (B,)
        node_ids = batch["node_ids"]  # (B, max_nodes) — unique items in prefix

        with torch.no_grad():
            scores = model_score_fn(batch)  # (B, n_items+1)

        # Set padding (index 0) to -inf so it never ranks top
        scores[:, 0] = float("-inf")

        # Rank: argsort descending → position of target
        # topk is faster than full sort for large n_items
        max_k = max(ks)
        _, top_indices = scores.topk(max_k, dim=1, largest=True, sorted=True)
        # top_indices: (B, max_k) — 0-th col = rank-1 item

        B = targets.shape[0]
        for b in range(B):
            target  = targets[b].item()
            is_cold = target in cold_items

            # Strict cold: cold target did NOT appear in the session prefix
            # node_ids[b] contains unique items in session[:-1]; 0 = padding
            is_revisit = is_cold and bool((node_ids[b] == target).any().item())
            is_strict_cold = is_cold and not is_revisit

            # Find rank of target (1-indexed)
            rank_positions = (top_indices[b] == target).nonzero(as_tuple=True)[0]
            if len(rank_positions) == 0:
                rank = max_k + 1  # not in top-K
            else:
                rank = rank_positions[0].item() + 1  # 1-indexed

            for k in ks:
                hit  = 1.0 if rank <= k else 0.0
                mrr  = (1.0 / rank) if rank <= k else 0.0
                ndcg = (1.0 / np.log2(rank + 1)) if rank <= k else 0.0

                all_hr[k].append(hit)
                all_mrr[k].append(mrr)
                all_ndcg[k].append(ndcg)

                if is_cold:
                    cold_hr[k].append(hit)
                    cold_mrr[k].append(mrr)
                    cold_ndcg[k].append(ndcg)
                else:
                    warm_hr[k].append(hit)

                if is_strict_cold:
                    strict_cold_hr[k].append(hit)
                    strict_cold_mrr[k].append(mrr)

                if is_revisit:
                    revisit_cold_hr[k].append(hit)
                    revisit_cold_mrr[k].append(mrr)

    # Aggregate
    results = {}
    for k in ks:
        results[f"HR@{k}"]   = float(np.mean(all_hr[k]))   if all_hr[k]   else 0.0
        results[f"MRR@{k}"]  = float(np.mean(all_mrr[k]))  if all_mrr[k]  else 0.0
        results[f"NDCG@{k}"] = float(np.mean(all_ndcg[k])) if all_ndcg[k] else 0.0
        results[f"Cold_HR@{k}"]   = float(np.mean(cold_hr[k]))   if cold_hr[k]   else 0.0
        results[f"Cold_MRR@{k}"]  = float(np.mean(cold_mrr[k]))  if cold_mrr[k]  else 0.0
        results[f"Cold_NDCG@{k}"] = float(np.mean(cold_ndcg[k])) if cold_ndcg[k] else 0.0
        results[f"Warm_HR@{k}"]   = float(np.mean(warm_hr[k]))   if warm_hr[k]   else 0.0
        results[f"StrictCold_HR@{k}"]   = float(np.mean(strict_cold_hr[k]))  if strict_cold_hr[k]  else 0.0
        results[f"StrictCold_MRR@{k}"]  = float(np.mean(strict_cold_mrr[k])) if strict_cold_mrr[k] else 0.0
        results[f"RevisitCold_HR@{k}"]  = float(np.mean(revisit_cold_hr[k])) if revisit_cold_hr[k] else 0.0
        results[f"RevisitCold_MRR@{k}"] = float(np.mean(revisit_cold_mrr[k]))if revisit_cold_mrr[k]else 0.0

    results["n_overall"]      = len(all_hr[ks[0]])
    results["n_cold"]         = len(cold_hr[ks[0]])
    results["n_strict_cold"]  = len(strict_cold_hr[ks[0]])
    results["n_revisit_cold"] = len(revisit_cold_hr[ks[0]])

    return results


def format_results(results: dict, prefix: str = "") -> str:
    """Pretty-print evaluation results."""
    lines = []
    if prefix:
        lines.append(prefix)
    lines.append(
        f"  Overall      | HR@10={results.get('HR@10',0):.4f}  HR@20={results.get('HR@20',0):.4f}"
        f"  MRR@10={results.get('MRR@10',0):.4f}  MRR@20={results.get('MRR@20',0):.4f}"
        f"  NDCG@10={results.get('NDCG@10',0):.4f}  NDCG@20={results.get('NDCG@20',0):.4f}"
        f"  (n={results.get('n_overall',0):,})"
    )
    lines.append(
        f"  Warm         | HR@10={results.get('Warm_HR@10',0):.4f}  HR@20={results.get('Warm_HR@20',0):.4f}"
        f"  (n_warm={results.get('n_overall',0)-results.get('n_cold',0):,})"
    )
    lines.append(
        f"  Cold         | HR@10={results.get('Cold_HR@10',0):.4f}  HR@20={results.get('Cold_HR@20',0):.4f}"
        f"  MRR@10={results.get('Cold_MRR@10',0):.4f}  MRR@20={results.get('Cold_MRR@20',0):.4f}"
        f"  NDCG@10={results.get('Cold_NDCG@10',0):.4f}  NDCG@20={results.get('Cold_NDCG@20',0):.4f}"
        f"  (n={results.get('n_cold',0):,})"
    )
    if results.get("n_strict_cold", 0) > 0:
        lines.append(
            f"  StrictCold   | HR@10={results.get('StrictCold_HR@10',0):.4f}  HR@20={results.get('StrictCold_HR@20',0):.4f}"
            f"  MRR@10={results.get('StrictCold_MRR@10',0):.4f}  MRR@20={results.get('StrictCold_MRR@20',0):.4f}"
            f"  (n={results.get('n_strict_cold',0):,}  first-time cold only)"
        )
    if results.get("n_revisit_cold", 0) > 0:
        lines.append(
            f"  RevisitCold  | HR@10={results.get('RevisitCold_HR@10',0):.4f}  HR@20={results.get('RevisitCold_HR@20',0):.4f}"
            f"  MRR@10={results.get('RevisitCold_MRR@10',0):.4f}  MRR@20={results.get('RevisitCold_MRR@20',0):.4f}"
            f"  (n={results.get('n_revisit_cold',0):,}  revisit cold)"
        )
    return "\n".join(lines)
