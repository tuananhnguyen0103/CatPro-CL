"""
core.py — CORE Baseline (CORE-ave)
------------------------------------
Hou et al., "CORE: Simple and Effective Session-based Recommendation
within Consistent Representation Space", SIGIR 2022.

Official code : https://github.com/RUCAIBox/CORE  (core_ave.py)
Datasets in paper: Diginetica, Nowplaying, RetailRocket, Tmall, Yoochoose

Model: CORE-ave (average-pooling variant, no Transformer dependency)

Key ideas from official core_ave.py:
  1. Representation-Consistent Encoder (RCE):
       z_s = L2_norm( mean(item_emb[v] for v in session) )
  2. Robust Distance Measuring (RDM):
       all_item_emb = L2_norm(item_emb.weight)
  3. Score = dot(z_s, z_item) / temperature   (cosine similarity / T)
  4. Loss = CrossEntropy(scores, target)

Our adaptation:
  - Use graph batch (node_ids + seq_idx + mask) to recover item IDs
    without changing the data loader
  - Full-ranking evaluation with same cold split as CatPro-CL
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catprocl.data_loader import get_dataloaders
from evaluation.evaluator import evaluate, format_results


# ── CORE-ave Model ─────────────────────────────────────────────────────────────
class COREModel(nn.Module):
    """
    CORE-ave: Consistent Representation for Session-based Recommendation.

    Faithfully reproduces core_ave.py from https://github.com/RUCAIBox/CORE
    Session emb and item embs share the SAME L2-normalised representation space.

    Differences from original (for fair comparison):
      - Accepts graph-format batch (node_ids, seq_idx, mask) instead of plain
        item_seq; reconstructs item IDs from graph batch internally.
      - Uses our full-ranking evaluator + cold-split protocol.
    """

    def __init__(self, n_items: int, d: int = 128, temperature: float = 0.07,
                 sess_dropout: float = 0.1, item_dropout: float = 0.1):
        super().__init__()
        self.d           = d
        self.temperature = temperature

        # Embedding table: 0=padding, 1..n_items=items
        self.item_emb  = nn.Embedding(n_items + 1, d, padding_idx=0)
        self.sess_drop = nn.Dropout(sess_dropout)
        self.item_drop = nn.Dropout(item_dropout)

        # Uniform init matching official _reset_parameters
        stdv = 1.0 / math.sqrt(d)
        for p in self.parameters():
            p.data.uniform_(-stdv, stdv)

    def forward(self, batch: dict) -> torch.Tensor:
        """
        RCE: Representation-Consistent Encoder.
        Returns z_s (B, d) — L2-normalised masked-mean session embedding.
        """
        node_ids = batch["node_ids"]  # (B, N)
        seq_idx  = batch["seq_idx"]   # (B, L)
        mask     = batch["mask"]      # (B, L) 1=valid 0=padding

        # Recover item IDs in session order
        seq_item_ids = torch.gather(node_ids, 1, seq_idx.clamp(min=0))  # (B, L)

        # Item embeddings with session dropout
        x = self.item_emb(seq_item_ids)  # (B, L, d)
        x = self.sess_drop(x)

        # Masked mean — official ave_net: alpha = mask / sum(mask)
        lengths = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)  # (B, 1)
        alpha   = mask.float() / lengths                           # (B, L)
        z_s     = torch.sum(alpha.unsqueeze(-1) * x, dim=1)        # (B, d)

        return F.normalize(z_s, dim=-1)   # L2-normalise

    def all_item_embeddings(self) -> torch.Tensor:
        """
        RDM: Robust Distance Measuring.
        Returns L2-normalised item embeddings (n_items+1, d).
        item_dropout off during eval (model.eval() handles this).

        Fix: cold items never appear in training → embeddings stay near init
        but Adam weight_decay can push them toward ~0 over many epochs.
        F.normalize(~0_vector) = 0/0 = NaN; on CUDA torch.topk ranks NaN at
        position 1 → Cold_HR inflated to ~32%.
        Solution: replace NaN with 0 so cold item scores stay near 0.
        """
        emb      = self.item_drop(self.item_emb.weight)
        emb_norm = F.normalize(emb, dim=-1)
        emb_norm = torch.nan_to_num(emb_norm, nan=0.0)   # ← NaN fix
        return emb_norm


# ── Training loop ──────────────────────────────────────────────────────────────
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CORE-ave | dataset={args.dataset} | seed={args.seed} | device={device}")
    print(f"  Ref: https://github.com/RUCAIBox/CORE  (SIGIR 2022)")

    train_loader, val_loader, test_loader, data = get_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size,
        num_workers=args.num_workers, maxlen=args.maxlen,
    )
    n_items    = data["n_items"]
    cold_items = data["cold_items"]
    print(f"  n_items={n_items:,}  cold_items={len(cold_items):,}")

    torch.manual_seed(args.seed)
    model = COREModel(
        n_items=n_items, d=args.d,
        temperature=args.temperature,
        sess_dropout=args.sess_dropout,
        item_dropout=args.item_dropout,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)

    best_val_hr20 = 0.0
    best_epoch    = 0
    best_test     = {}
    patience_cnt  = 0
    epoch_logs    = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Ep {epoch:3d}", ncols=100, leave=False)
        for batch in pbar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            targets = batch["targets"]

            z_s      = model(batch)                                    # (B, d) normalised
            item_emb = model.all_item_embeddings()                     # (n+1, d) normalised
            scores   = z_s @ item_emb.T / model.temperature            # (B, n+1) cosine/T
            loss     = F.cross_entropy(scores, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        pbar.close()
        scheduler.step()

        # ── Eval ──────────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            item_emb = model.all_item_embeddings()

            def score_fn(b):
                z = model(b)
                return z @ item_emb.T / model.temperature

            val_res = evaluate(score_fn, val_loader, item_emb, cold_items,
                               ks=[10, 20], device=device)

        val_hr20 = val_res["HR@20"]
        elapsed  = time.time() - t0
        epoch_logs.append({
            "epoch":        epoch,
            "loss":         total_loss / len(train_loader),
            "epoch_time_s": elapsed,
            **{f"val_{k}": v for k, v in val_res.items()},
        })
        print(f"Epoch {epoch:3d} | loss={total_loss/len(train_loader):.4f} | "
              f"Val HR@20={val_hr20:.4f} | {elapsed:.1f}s")

        if val_hr20 > best_val_hr20:
            best_val_hr20 = val_hr20
            best_epoch    = epoch
            patience_cnt  = 0
            with torch.no_grad():
                test_res = evaluate(score_fn, test_loader, item_emb, cold_items,
                                    ks=[10, 20], device=device)
            best_test = test_res
            print(format_results(test_res, prefix="  → New best! Test:"))
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"Early stop at epoch {epoch}  (patience={args.patience})")
                break

    print(f"\nBest epoch={best_epoch}  Val HR@20={best_val_hr20:.4f}")
    print(format_results(best_test, prefix="Final Test:"))

    os.makedirs(args.output_dir, exist_ok=True)
    out = {
        "dataset":    args.dataset,
        "ablation":   "CORE",
        "seed":       args.seed,
        "best_epoch": best_epoch,
        "test":       best_test,
        "epoch_logs": epoch_logs,
        "config":     vars(args),
    }
    fname = os.path.join(args.output_dir, f"{args.dataset}_CORE_seed{args.seed}.json")
    with open(fname, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved → {fname}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",       required=True)
    p.add_argument("--output_dir",     default="~/results")
    p.add_argument("--dataset",        default="retailrocket")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--d",              type=int,   default=128)
    p.add_argument("--batch_size",     type=int,   default=100)
    p.add_argument("--epochs",         type=int,   default=30)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--lr_dc",          type=float, default=0.1)
    p.add_argument("--lr_dc_step",     type=int,   default=3)
    p.add_argument("--weight_decay",   type=float, default=1e-5)
    p.add_argument("--clip",           type=float, default=5.0)
    p.add_argument("--patience",       type=int,   default=10)
    p.add_argument("--num_workers",    type=int,   default=4)
    p.add_argument("--maxlen",         type=int,   default=0,
                   help="Truncate sessions to last N items (0=full)")
    p.add_argument("--temperature",    type=float, default=0.07,
                   help="Cosine similarity temperature (official default=0.07)")
    p.add_argument("--sess_dropout",   type=float, default=0.1)
    p.add_argument("--item_dropout",   type=float, default=0.1)
    args = p.parse_args()

    args.output_dir = os.path.expanduser(args.output_dir)
    train(args)
