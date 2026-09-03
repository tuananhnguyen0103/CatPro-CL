"""
letitgo.py — LetItGo Baseline
-------------------------------
Paper: "Let It Go? Not Quite: Addressing Item Cold Start in Sequential
        Recommendations with Content-Based Initialization" (RecSys 2025)
GitHub: github.com/ArtemF42/let-it-go

CORE IDEA (adapted for SBR):
  Item embeddings are structured as:
      h_item[i] = cat_emb[cat(i)] + delta[i]

  - cat_emb : shared category embedding (content baseline)
  - delta   : small per-item correction, init to 0, regularized to stay small

  Cold items (0 training interactions):
      h_cold[i] = cat_emb[cat(i)]   ← delta=0 since item never seen

  Because ALL warm items are trained in (cat_emb + delta) space, the model
  learns a recommendation function that works in category/content space →
  cold inference is naturally aligned with the trained collaborative space.

DIFFERENCE vs M2TRec:
  M2TRec: cat_emb[c] is directly the cold item embedding (no warm alignment)
  LetItGo: warm items ALSO use cat_emb[cat] + delta → content space is
            aligned with collaborative space throughout training → better cold inference

DIFFERENCE vs NirGNN:
  NirGNN: TransferMLP(cat_proto) → cold item repr (extra transfer network)
  LetItGo: direct cat_emb init; delta regularized; simpler, no extra module

OUR ADAPTATION:
  - SASRec-style Transformer backbone (same as M2TRec)
  - Content = category embedding (only metadata in our datasets)
  - item_emb[i] = cat_emb[cat(i)] + delta[i]
  - Loss = CE_rec + lambda_delta * mean(||delta[warm_ids]||²)

Usage:
  python baselines/letitgo.py \\
      --data_dir ~/data/retailrocket_unified/cold_20 \\
      --output_dir ~/results_v6/data/retailrocket/fullen/5_LetItGo \\
      --dataset retailrocket_fullen --seed 42
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


# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_item_seq(batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract padded global item-ID sequence from graph batch.
    Returns (item_seq, mask): both (B, L), item_seq[b,t]=0 at padding.
    """
    node_ids = batch["node_ids"]          # (B, max_nodes)
    seq_idx  = batch["seq_idx"]           # (B, L)
    mask     = batch["mask"]              # (B, L) float

    item_seq = node_ids.gather(1, seq_idx.clamp(min=0))  # (B, L)
    item_seq = item_seq * mask.long()
    return item_seq, mask


# ── SASRec Layer ──────────────────────────────────────────────────────────────
class SASRecLayer(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d * 4, d), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        causal = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        a, _   = self.attn(x, x, x, attn_mask=causal, key_padding_mask=pad_mask)
        x = self.norm1(x + a)
        x = self.norm2(x + self.ff(x))
        return x


# ── LetItGo Model ─────────────────────────────────────────────────────────────
class LetItGoModel(nn.Module):
    """
    SASRec backbone + (cat_emb + delta) item embeddings.

    Registers item2cat as a buffer so it moves to device automatically.
    """

    def __init__(self, n_items: int, n_cats: int, item2cat_t: torch.Tensor,
                 d: int = 128, n_layers: int = 2, n_heads: int = 2,
                 maxlen: int = 50, dropout: float = 0.1):
        super().__init__()
        self.d       = d
        self.maxlen  = maxlen
        self.n_items = n_items

        # Content embedding — shared per category
        self.cat_emb    = nn.Embedding(n_cats + 1, d, padding_idx=0)

        # Per-item delta — initialized to 0, regularized small during training
        self.item_delta = nn.Embedding(n_items + 1, d, padding_idx=0)
        nn.init.zeros_(self.item_delta.weight)

        # Positional embedding (learned)
        self.pos_emb = nn.Embedding(maxlen + 1, d, padding_idx=0)

        # Transformer layers
        self.layers  = nn.ModuleList([SASRecLayer(d, n_heads, dropout) for _ in range(n_layers)])
        self.norm    = nn.LayerNorm(d)
        self.drop    = nn.Dropout(dropout)

        # item2cat lookup: (n_items+1,) long tensor — registered as buffer
        self.register_buffer("item2cat_t", item2cat_t)

    # ── Embedding helpers ──────────────────────────────────────────────────
    def _item_repr(self, item_ids: torch.Tensor) -> torch.Tensor:
        """h[i] = cat_emb[cat(i)] + delta[i]   — shape: (*item_ids.shape, d)"""
        cat_ids = self.item2cat_t[item_ids]
        return self.cat_emb(cat_ids) + self.item_delta(item_ids)

    def all_item_embeddings(self) -> torch.Tensor:
        """
        Full (n_items+1, d) embedding matrix for scoring.
        Cold items (delta never updated): h = cat_emb[cat(i)], delta = 0.
        """
        all_ids = torch.arange(self.n_items + 1, device=self.item2cat_t.device)
        return self._item_repr(all_ids)

    # ── Session encoder ────────────────────────────────────────────────────
    def encode(self, batch: dict) -> torch.Tensor:
        """
        Returns session repr (B, d) using causal SASRec over item sequence.
        """
        item_seq, mask = extract_item_seq(batch)   # (B, L)
        B, L = item_seq.shape

        # Truncate to maxlen
        if L > self.maxlen:
            item_seq = item_seq[:, -self.maxlen:]
            mask     = mask[:, -self.maxlen:]
            L        = self.maxlen

        h = self._item_repr(item_seq)               # (B, L, d)

        # Learned positional embedding (1-indexed, 0=pad)
        lengths = (mask.cumsum(dim=1) * mask.long()).long()  # (B, L) positions
        h = h + self.pos_emb(lengths.clamp(max=self.maxlen))
        h = self.drop(h)

        pad_mask = (mask == 0)                      # True where padding
        for layer in self.layers:
            h = layer(h, pad_mask)
        h = self.norm(h)                            # (B, L, d)

        # Last valid position as session representation
        last_pos = mask.sum(dim=1).long().clamp(min=1) - 1   # (B,)
        return h[torch.arange(B, device=h.device), last_pos]  # (B, d)

    def forward(self, batch: dict) -> torch.Tensor:
        """Returns (B, n_items+1) logit scores."""
        s      = self.encode(batch)               # (B, d)
        all_e  = self.all_item_embeddings()       # (n_items+1, d)
        return s @ all_e.T                        # (B, n_items+1)


# ── Training ──────────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | LetItGo | Dataset: {args.dataset} | Seed: {args.seed}")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    train_loader, val_loader, test_loader, data_dict = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=-1, maxlen=args.maxlen,
    )
    n_items    = data_dict["n_items"]
    n_cats     = data_dict["n_cats"]
    cold_items = data_dict["cold_items"]
    item2cat   = data_dict["item2cat"]

    print(f"n_items={n_items} | n_cats={n_cats} | cold_items={len(cold_items)}")

    # Build item2cat lookup tensor
    item2cat_t = torch.zeros(n_items + 1, dtype=torch.long)
    for iid, cid in item2cat.items():
        if 1 <= int(iid) <= n_items:
            item2cat_t[int(iid)] = int(cid)

    # Warm item IDs (for delta regularization)
    warm_ids_list = [i for i in range(1, n_items + 1) if i not in cold_items]
    warm_ids_t    = torch.tensor(warm_ids_list, dtype=torch.long, device=device)

    model = LetItGoModel(
        n_items=n_items, n_cats=n_cats, item2cat_t=item2cat_t,
        d=args.d, n_layers=args.n_layers, n_heads=args.n_heads,
        maxlen=args.maxlen if args.maxlen > 0 else 50,
        dropout=args.dropout,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_hr20 = -1.0
    best_epoch    = 0
    patience_cnt  = 0
    best_state    = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for batch in tqdm(train_loader, desc=f"Ep{epoch}", leave=False):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            targets = batch["targets"]   # (B,)

            optimizer.zero_grad()

            # Recommendation loss
            scores  = model(batch)                    # (B, n_items+1)
            scores[:, 0] = float("-inf")              # suppress pad
            loss_rec = F.cross_entropy(scores, targets)

            # Delta regularization: keep delta small (LetItGo constraint)
            delta_warm = model.item_delta(warm_ids_t)  # (n_warm, d)
            loss_delta = delta_warm.pow(2).mean()

            loss = loss_rec + args.lambda_delta * loss_delta
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / max(len(train_loader), 1)

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            item_W = model.all_item_embeddings()

            def score_fn(b):
                return model(b)

            val_metrics = evaluate(score_fn, val_loader, item_W, cold_items,
                                   ks=[10, 20], device=device)

        hr20 = val_metrics.get("HR@20", 0.0)
        elapsed = time.time() - t0
        print(f"Ep{epoch:3d} | loss={avg_loss:.4f} | val HR@20={hr20:.4f} | {elapsed:.1f}s")

        if hr20 > best_val_hr20:
            best_val_hr20 = hr20
            best_epoch    = epoch
            patience_cnt  = 0
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"Early stop at epoch {epoch} (patience={args.patience})")
                break

    # ── Test ──────────────────────────────────────────────────────────────
    print(f"\nBest epoch={best_epoch} | val HR@20={best_val_hr20:.4f}")
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    with torch.no_grad():
        item_W_test = model.all_item_embeddings()

        def score_fn_test(b):
            return model(b)

        test_metrics = evaluate(score_fn_test, test_loader, item_W_test, cold_items,
                                ks=[10, 20], device=device)

    print("\n=== TEST RESULTS ===")
    print(format_results(test_metrics))

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.dataset}_LetItGo_seed{args.seed}.json")
    result = {
        "dataset":    args.dataset,
        "ablation":   "LetItGo",
        "seed":       args.seed,
        "best_epoch": best_epoch,
        "test":       test_metrics,
        "config": {
            "d":             args.d,
            "n_heads":       args.n_heads,
            "n_layers":      args.n_layers,
            "dropout":       args.dropout,
            "lambda_delta":  args.lambda_delta,
            "epochs":        args.epochs,
            "batch_size":    args.batch_size,
            "lr":            args.lr,
            "wd":            args.wd,
            "patience":      args.patience,
        },
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="LetItGo: Content-init + Trainable Delta for Cold-Start SBR")
    p.add_argument("--data_dir",      required=True)
    p.add_argument("--output_dir",    required=True)
    p.add_argument("--dataset",       required=True)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--maxlen",        type=int,   default=0,    help="0=full session")
    p.add_argument("--d",             type=int,   default=128)
    p.add_argument("--n_heads",       type=int,   default=2)
    p.add_argument("--n_layers",      type=int,   default=2)
    p.add_argument("--dropout",       type=float, default=0.1)
    p.add_argument("--lambda_delta",  type=float, default=0.01, help="Delta regularization weight")
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--batch_size",    type=int,   default=100)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--wd",            type=float, default=1e-5)
    p.add_argument("--patience",      type=int,   default=5)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
