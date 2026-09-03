"""
m2trec.py — M2TRec Baseline (our adaptation)
---------------------------------------------
Original paper: "M2TRec: Metadata-informed Masked-Transformer for Cold-start
Session-based Recommendations" (RecSys 2022, Ding et al.)
Official code  : not publicly available — re-implemented from paper description.

ORIGINAL M2TRec IDEA:
  - Transformer encoder over item sequence
  - Cold items: use metadata embedding (category / brand / price) as item repr
  - Masked training: randomly mask warm items → predict them from metadata context
    (self-supervised, teaches model that metadata ~ item identity)

OUR ADAPTATION (category-only):
  Category is the only item metadata in our datasets (Diginetica, RetailRocket).
  1. Transformer encoder over linear session sequence.
  2. Category embedding table: cat_emb[c] ∈ R^d — learned alongside item_emb.
  3. Training:
     L_rec  : CE on next-item prediction (standard SBR)
     L_mask : randomly mask 15% of warm positions → replace with cat_emb →
              predict original item (mimics M2TRec masked metadata training)
     L_total = L_rec + λ * L_mask
  4. Cold inference: for cold item i, h_i = cat_emb[cat(i)]
     (same principle as original: metadata embedding IS the cold item repr)

WHY M2TRec DIFFERS FROM CatPro-CL (A7):
  - M2TRec: cat_emb[c] is learned DIRECTLY as cold item repr (no aggregation).
  - CatPro-CL A7: prototype_c = EMA average of warm items in category c,
    aligned via InfoNCE contrastive loss. The prototype is a DERIVED, aggregated
    representation of actual interactions — not just a learnable parameter.
  This makes M2TRec a clean ablation baseline: same cold-start principle,
  without EMA aggregation and without contrastive alignment.

Input batch format (from SessionGraphCollator):
  node_ids : (B, max_nodes) — unique item IDs in session prefix, 0=pad
  seq_idx  : (B, max_seq)   — local indices into node_ids (sequence order)
  mask     : (B, max_seq)   — 1=valid, 0=pad
  targets  : (B,)           — target item ID to predict

Usage:
    python baselines/m2trec.py \\
        --data_dir ~/data/diginetica_maxlen6/cold_20 \\
        --output_dir ~/results_baseline/diginetica_maxlen6 \\
        --dataset diginetica_maxlen6 --seed 42
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
    Extract padded item sequence (global IDs) from graph batch.

    batch["node_ids"] : (B, max_nodes) global item IDs, 0=pad
    batch["seq_idx"]  : (B, max_seq)   local indices into node_ids
    batch["mask"]     : (B, max_seq)   1=valid, 0=pad

    Returns:
      item_seq : (B, max_seq) global item IDs in sequence order, 0=pad
      mask     : (B, max_seq) float32
    """
    node_ids = batch["node_ids"]   # (B, N)
    seq_idx  = batch["seq_idx"]    # (B, L)
    mask     = batch["mask"]       # (B, L)

    B, N = node_ids.shape
    L    = seq_idx.shape[1]

    # Gather global IDs: node_ids[b, seq_idx[b, t]] for all b, t
    seq_idx_exp = seq_idx.clamp(min=0).unsqueeze(-1)          # (B, L, 1)
    node_ids_exp = node_ids.unsqueeze(1).expand(B, L, N)      # (B, L, N)
    item_seq = node_ids.gather(1, seq_idx.clamp(min=0))        # (B, L)

    # Zero out padded positions (seq_idx padding points to node 0, mask=0)
    item_seq = item_seq * mask.long()
    return item_seq, mask


# ── Positional Encoding ───────────────────────────────────────────────────────
class SinusoidalPE(nn.Module):
    def __init__(self, d: int, max_len: int = 300, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


# ── M2TRec Model ─────────────────────────────────────────────────────────────
class M2TRecModel(nn.Module):
    """
    Metadata-informed Masked Transformer for Session-Based Recommendation.

    Embedding tables:
      item_emb  : (n_items+1, d)  — warm item embeddings (idx 0 = pad)
      cat_emb   : (n_cats+1,  d)  — category embeddings  (idx 0 = pad)
        For cold items at inference: h_i = cat_emb[cat(i)]

    Architecture: Transformer encoder (batch_first) → last valid position → out_proj
    """

    def __init__(
        self,
        n_items: int,
        n_cats:  int,
        d:       int   = 128,
        n_heads: int   = 2,
        n_layers:int   = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_items = n_items
        self.n_cats  = n_cats
        self.d       = d

        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)
        self.cat_emb  = nn.Embedding(n_cats  + 1, d, padding_idx=0)
        self.pos_enc  = SinusoidalPE(d, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=d * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.layer_norm  = nn.LayerNorm(d)
        self.out_proj    = nn.Linear(d, d, bias=False)
        self.dropout     = nn.Dropout(dropout)

        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.cat_emb.weight,  std=0.02)
        self.item_emb.weight.data[0].fill_(0)
        self.cat_emb.weight.data[0].fill_(0)

    def encode(self, batch: dict) -> torch.Tensor:
        """
        Encode a batch (graph format) into session embeddings.
        Returns h_s: (B, d)
        """
        item_seq, mask = extract_item_seq(batch)     # (B, L), (B, L)
        device = item_seq.device

        x = self.item_emb(item_seq)                  # (B, L, d)
        x = self.pos_enc(x)

        pad_mask = (item_seq == 0)                    # (B, L), True=ignore
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        x = self.layer_norm(x)

        # Last valid position
        lengths = mask.long().sum(dim=1) - 1         # (B,) 0-indexed last valid
        lengths = lengths.clamp(min=0)
        h_last  = x[torch.arange(x.size(0), device=device), lengths]  # (B, d)

        return self.out_proj(h_last)                 # (B, d)

    def masked_cat_loss(
        self,
        item_seq:    torch.Tensor,    # (B, L) global item IDs, 0=pad
        item2cat_t:  torch.Tensor,    # (n_items+1,) cat ID per item
        warm_mask_t: torch.Tensor,    # (n_items+1,) bool — True=warm
        mask_prob:   float = 0.15,
    ) -> torch.Tensor:
        """
        L_mask: randomly replace 15% of warm positions with cat_emb,
        then predict the original item from the masked representation.
        Teaches: cat_emb[c] ≈ item_emb[i] for i ∈ category c.
        """
        B, L  = item_seq.shape
        device = item_seq.device

        # Eligible: non-pad AND warm
        is_warm   = warm_mask_t[item_seq]             # (B, L) bool
        is_nonpad = (item_seq != 0)
        eligible  = is_warm & is_nonpad               # (B, L)

        rand = torch.rand(B, L, device=device)
        to_mask = eligible & (rand < mask_prob)       # (B, L) bool

        if to_mask.sum() == 0:
            return item_seq.new_zeros(1, dtype=torch.float).squeeze()

        # Build input with masked positions replaced by cat embedding
        cat_seq = item2cat_t[item_seq]               # (B, L) cat IDs
        x_item  = self.item_emb(item_seq)            # (B, L, d)
        x_cat   = self.cat_emb(cat_seq)              # (B, L, d)
        x = torch.where(to_mask.unsqueeze(-1), x_cat, x_item)
        x = self.pos_enc(x)

        pad_mask = (item_seq == 0)
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        x = self.layer_norm(x)

        # Only compute loss at masked positions
        h_masked  = x[to_mask]                        # (N_mask, d)
        h_masked  = self.out_proj(h_masked)           # (N_mask, d)
        tgt_items = item_seq[to_mask]                 # (N_mask,)

        logits = h_masked @ self.item_emb.weight.T   # (N_mask, n_items+1)
        logits[:, 0] = float("-inf")                  # suppress pad
        return F.cross_entropy(logits, tgt_items)


# ── Build item matrix for evaluation ─────────────────────────────────────────
@torch.no_grad()
def build_item_matrix(
    model:       M2TRecModel,
    item2cat_t:  torch.Tensor,         # (n_items+1,)
    cold_items:  set,
    n_items:     int,
    device:      torch.device,
) -> torch.Tensor:
    """
    Returns (n_items+1, d):
      warm items → item_emb[i]
      cold items → cat_emb[cat(i)]   (M2TRec cold-start principle)
    """
    W = model.item_emb.weight.clone()    # (n_items+1, d)
    cat_W = model.cat_emb.weight         # (n_cats+1, d)

    cold_ids = torch.tensor(
        [i for i in cold_items if 1 <= i <= n_items],
        dtype=torch.long, device=device
    )
    if cold_ids.numel() > 0:
        cold_cats = item2cat_t[cold_ids]  # (n_cold,)
        W[cold_ids] = cat_W[cold_cats]

    return W


# ── Training loop ─────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | M2TRec | Dataset: {args.dataset} | Seed: {args.seed}")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Data (returns train_loader, val_loader, test_loader, data_dict)
    train_loader, val_loader, test_loader, data_dict = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=-1, maxlen=args.maxlen,
    )
    n_items    = data_dict["n_items"]
    n_cats     = data_dict["n_cats"]
    cold_items = data_dict["cold_items"]
    item2cat   = data_dict["item2cat"]   # {item_id (int): cat_id (int)}

    print(f"n_items={n_items} | n_cats={n_cats} | cold_items={len(cold_items)}")

    # Lookup tensors
    item2cat_t = torch.zeros(n_items + 1, dtype=torch.long, device=device)
    for iid, cid in item2cat.items():
        if 1 <= int(iid) <= n_items:
            item2cat_t[int(iid)] = int(cid)

    warm_mask_t = torch.ones(n_items + 1, dtype=torch.bool, device=device)
    for ci in cold_items:
        if ci <= n_items:
            warm_mask_t[ci] = False
    warm_mask_t[0] = False  # pad

    # Model
    model = M2TRecModel(
        n_items=n_items, n_cats=n_cats,
        d=args.d, n_heads=args.n_heads, n_layers=args.n_layers,
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
        total_rec = total_mask = 0.0
        t0 = time.time()

        for batch in tqdm(train_loader, desc=f"Ep{epoch}", leave=False):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            targets = batch["targets"]                     # (B,)

            # Extract sequence for M2TRec
            item_seq, mask = extract_item_seq(batch)      # (B, L)

            optimizer.zero_grad()

            # ── L_rec: recommendation loss ────────────────────────────────
            h_s   = model.encode(batch)                   # (B, d)
            item_W = model.item_emb.weight                # (n_items+1, d)
            logits = h_s @ item_W.T                       # (B, n_items+1)
            logits[:, 0] = float("-inf")
            L_rec  = F.cross_entropy(logits, targets)

            # ── L_mask: masked category training ─────────────────────────
            L_mask = model.masked_cat_loss(
                item_seq, item2cat_t, warm_mask_t, mask_prob=args.mask_prob
            )

            loss = L_rec + args.lambda_mask * L_mask
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_rec  += L_rec.item()
            total_mask += L_mask.item() if isinstance(L_mask, torch.Tensor) else L_mask

        scheduler.step()
        n_batches = max(len(train_loader), 1)
        epoch_t   = time.time() - t0

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            item_W_full = build_item_matrix(model, item2cat_t, cold_items, n_items, device)

            def score_fn(b):
                h = model.encode(b)
                return h @ item_W_full.T

            val_metrics = evaluate(
                score_fn, val_loader, item_W_full, cold_items,
                ks=[10, 20], device=device,
            )

        val_hr20 = val_metrics.get("HR@20", 0.0)
        print(
            f"Ep{epoch:2d} | rec={total_rec/n_batches:.4f} "
            f"mask={total_mask/n_batches:.4f} | "
            f"Val HR@10={val_metrics.get('HR@10',0):.4f} "
            f"HR@20={val_hr20:.4f} "
            f"ColdHR@20={val_metrics.get('Cold_HR@20',0):.4f} | "
            f"{epoch_t:.1f}s"
        )

        if val_hr20 > best_val_hr20:
            best_val_hr20 = val_hr20
            best_epoch    = epoch
            patience_cnt  = 0
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"  Early stop at epoch {epoch} (patience={args.patience})")
                break

    # ── Test ─────────────────────────────────────────────────────────────────
    print(f"\nBest epoch={best_epoch} | val HR@20={best_val_hr20:.4f}")
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    with torch.no_grad():
        item_W_full = build_item_matrix(model, item2cat_t, cold_items, n_items, device)

        def score_fn_test(b):
            h = model.encode(b)
            return h @ item_W_full.T

        test_metrics = evaluate(
            score_fn_test, test_loader, item_W_full, cold_items,
            ks=[10, 20], device=device,
        )

    print("\n=== TEST RESULTS ===")
    print(format_results(test_metrics))

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir, f"{args.dataset}_M2TRec_seed{args.seed}.json"
    )
    result = {
        "dataset":    args.dataset,
        "ablation":   "M2TRec",
        "seed":       args.seed,
        "best_epoch": best_epoch,
        "test":       test_metrics,
        "config": {
            "d":           args.d,
            "n_heads":     args.n_heads,
            "n_layers":    args.n_layers,
            "dropout":     args.dropout,
            "mask_prob":   args.mask_prob,
            "lambda_mask": args.lambda_mask,
            "epochs":      args.epochs,
            "batch_size":  args.batch_size,
            "lr":          args.lr,
            "wd":          args.wd,
            "patience":    args.patience,
        },
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="M2TRec: Metadata-informed Masked Transformer")
    p.add_argument("--data_dir",    required=True,  help="Path to cold_20/ folder")
    p.add_argument("--output_dir",  required=True,  help="Where to save JSON result")
    p.add_argument("--dataset",     required=True,  help="Dataset name tag in output file")
    # Model
    p.add_argument("--d",           type=int,   default=128,  help="Embedding dim")
    p.add_argument("--n_heads",     type=int,   default=2,    help="Transformer heads")
    p.add_argument("--n_layers",    type=int,   default=2,    help="Transformer layers")
    p.add_argument("--dropout",     type=float, default=0.1)
    # Training
    p.add_argument("--mask_prob",   type=float, default=0.15, help="Fraction of positions to mask")
    p.add_argument("--lambda_mask", type=float, default=0.5,  help="Weight of L_mask")
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=100)
    p.add_argument("--maxlen",      type=int,   default=0,
                   help="Truncate sessions to last N items (0=full session)")
    p.add_argument("--lr",          type=float, default=0.001)
    p.add_argument("--wd",          type=float, default=1e-5)
    p.add_argument("--grad_clip",   type=float, default=5.0)
    p.add_argument("--patience",    type=int,   default=5)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
