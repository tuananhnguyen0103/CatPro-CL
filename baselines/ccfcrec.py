"""
ccfcrec.py — CCFCRec Baseline (adapted for SBR)
-------------------------------------------------
Paper: "Contrastive Collaborative Filtering for Cold-Start Item Recommendation"
       (WWW 2023), Zhou et al.
GitHub: github.com/zzhin/CCFCRec

ORIGINAL CCFCRec IDEA:
  Two CF modules operating on items:
    1. Content CF module   : content_features → item embedding via MLP
    2. Co-occurrence CF module : item_id → embedding (learned from interactions)
  InfoNCE contrastive loss aligns content-CF embeddings with co-occurrence-CF
  embeddings for the same item → content-CF learns to approximate collaborative
  signal even for cold items.
  Cold inference: use content-CF module only (no ID embedding needed).

OUR ADAPTATION FOR SBR:
  - "User" = session (anonymous user)
  - "Co-occurrence" = items appearing in the same session (session co-occurrence)
  - Content feature = category embedding (only metadata available)
  - Session scoring: mean pooling of item embeddings from session prefix
  - Training losses:
      L_rec    = CE(session_repr @ item_emb.T, target)  [recommendation]
      L_contra = InfoNCE(content_emb, coo_emb)          [alignment]
      L_total  = L_rec + lambda_c * L_contra

  Cold inference: content-CF module provides cold item embedding
  The alignment training ensures content-CF space ≈ co-occurrence-CF space.

NOTATION:
  content_emb[i] = MLP(cat_emb[cat(i)])    — content-CF representation
  coo_emb[i]     = item_emb_coo[i]         — co-occurrence-CF representation
  At test: warm items → coo_emb; cold items → content_emb

Usage:
  python baselines/ccfcrec.py \\
      --data_dir ~/data/retailrocket_unified/cold_20 \\
      --output_dir ~/results_v6/data/retailrocket/fullen/6_CCFCRec \\
      --dataset retailrocket_fullen --seed 42
"""

import argparse
import json
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
    node_ids = batch["node_ids"]
    seq_idx  = batch["seq_idx"]
    mask     = batch["mask"]
    item_seq = node_ids.gather(1, seq_idx.clamp(min=0))
    return item_seq * mask.long(), mask


def info_nce_loss(q: torch.Tensor, k: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """
    Symmetric InfoNCE between q and k.
    q, k: (N, d) — N pairs of (content_emb, coo_emb) for same items.
    Positive pairs: q[i] ↔ k[i]; negatives: all other j≠i in batch.
    """
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    logits = (q @ k.T) / tau          # (N, N)
    labels = torch.arange(len(q), device=q.device)
    loss_qk = F.cross_entropy(logits, labels)
    loss_kq = F.cross_entropy(logits.T, labels)
    return (loss_qk + loss_kq) / 2


# ── CCFCRec Model ─────────────────────────────────────────────────────────────
class CCFCRecModel(nn.Module):
    """
    Two CF modules:
      content-CF : cat_emb → MLP → item_repr   (works for ALL items incl. cold)
      coo-CF     : item_emb_coo[item_id]        (only for warm items)

    Session representation: mean pooling over item embeddings.
    """

    def __init__(self, n_items: int, n_cats: int, item2cat_t: torch.Tensor,
                 d: int = 128, n_mlp_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d       = d
        self.n_items = n_items

        # ── Content-CF module ─────────────────────────────────────────────
        self.cat_emb = nn.Embedding(n_cats + 1, d, padding_idx=0)
        layers = []
        for _ in range(n_mlp_layers):
            layers += [nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(d, d))
        self.content_mlp = nn.Sequential(*layers)

        # ── Co-occurrence-CF module ───────────────────────────────────────
        self.item_emb_coo = nn.Embedding(n_items + 1, d, padding_idx=0)

        self.register_buffer("item2cat_t", item2cat_t)

    # ── Item embeddings ────────────────────────────────────────────────────
    def content_emb(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Content-CF embedding for any item (warm or cold)."""
        cat_ids = self.item2cat_t[item_ids]
        return self.content_mlp(self.cat_emb(cat_ids))

    def all_content_emb(self) -> torch.Tensor:
        """(n_items+1, d) content-CF embeddings for all items."""
        all_ids = torch.arange(self.n_items + 1, device=self.item2cat_t.device)
        return self.content_emb(all_ids)

    def build_eval_embeddings(self, cold_items_set: set) -> torch.Tensor:
        """
        Build (n_items+1, d) matrix for test evaluation:
          warm items → coo_emb  (collaborative signal)
          cold items → content_emb (content-CF, only signal available)
        """
        W    = self.item_emb_coo.weight.clone()   # (n_items+1, d)
        cont = self.all_content_emb()             # (n_items+1, d)

        cold_ids = torch.tensor(
            [i for i in cold_items_set if 1 <= i <= self.n_items],
            dtype=torch.long, device=W.device
        )
        if cold_ids.numel() > 0:
            W[cold_ids] = cont[cold_ids]
        return W

    # ── Session encoder ────────────────────────────────────────────────────
    def encode_session(self, item_seq: torch.Tensor, mask: torch.Tensor,
                       use_content: bool = False) -> torch.Tensor:
        """
        Mean-pool item embeddings over valid session positions.
        item_seq: (B, L) global item IDs; mask: (B, L) float.
        Returns: (B, d) session representation.
        """
        if use_content:
            embs = self.content_emb(item_seq)            # (B, L, d)
        else:
            embs = self.item_emb_coo(item_seq)           # (B, L, d)

        mask_e = mask.unsqueeze(-1).float()              # (B, L, 1)
        pooled = (embs * mask_e).sum(dim=1)              # (B, d)
        denom  = mask_e.sum(dim=1).clamp(min=1e-9)      # (B, 1)
        return pooled / denom                            # (B, d)

    def forward(self, batch: dict, item_W: torch.Tensor) -> torch.Tensor:
        """
        Returns (B, n_items+1) scores for recommendation.
        Uses coo_emb for session encoding during training.
        """
        item_seq, mask = extract_item_seq(batch)
        s = self.encode_session(item_seq, mask, use_content=False)   # (B, d)
        return s @ item_W.T                                           # (B, n_items+1)

    def contrastive_loss(self, warm_ids: torch.Tensor, tau: float) -> torch.Tensor:
        """
        InfoNCE between content-CF and coo-CF embeddings for warm items.
        warm_ids: (N,) warm item IDs sampled from batch.
        """
        q = self.content_emb(warm_ids)               # (N, d) content-CF
        k = self.item_emb_coo(warm_ids)              # (N, d) coo-CF
        return info_nce_loss(q, k, tau)


# ── Training ──────────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | CCFCRec | Dataset: {args.dataset} | Seed: {args.seed}")

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

    # Warm item IDs for contrastive sampling
    warm_set = set(range(1, n_items + 1)) - cold_items

    model = CCFCRecModel(
        n_items=n_items, n_cats=n_cats, item2cat_t=item2cat_t,
        d=args.d, n_mlp_layers=args.n_mlp_layers, dropout=args.dropout,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_hr20 = -1.0
    best_epoch    = 0
    patience_cnt  = 0
    best_state    = None

    warm_ids_all = torch.tensor(list(warm_set), dtype=torch.long)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for batch in tqdm(train_loader, desc=f"Ep{epoch}", leave=False):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            targets = batch["targets"]   # (B,)

            optimizer.zero_grad()

            # ── Recommendation loss (using coo_emb) ───────────────────────
            item_W   = model.item_emb_coo.weight              # (n_items+1, d)
            scores   = model(batch, item_W)                   # (B, n_items+1)
            scores[:, 0] = float("-inf")
            loss_rec = F.cross_entropy(scores, targets)

            # ── Contrastive loss (content-CF vs coo-CF) ───────────────────
            # Sample warm items from current batch's node_ids
            node_ids = batch["node_ids"].flatten()
            warm_in_batch = node_ids[(node_ids > 0)]
            warm_in_batch = warm_in_batch[
                torch.tensor([i.item() in warm_set for i in warm_in_batch], device=device)
            ]
            # Deduplicate and cap size
            warm_sample = warm_in_batch.unique()[:args.n_contra_sample]

            if warm_sample.numel() < 2:
                # Fall back to random warm sample if batch has too few warm items
                idx = torch.randint(0, len(warm_ids_all), (args.n_contra_sample,))
                warm_sample = warm_ids_all[idx].to(device)

            loss_c = model.contrastive_loss(warm_sample, tau=args.tau)

            loss = loss_rec + args.lambda_c * loss_c
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / max(len(train_loader), 1)

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            item_W_eval = model.build_eval_embeddings(cold_items)

            def score_fn(b):
                return model(b, item_W_eval)

            val_metrics = evaluate(score_fn, val_loader, item_W_eval, cold_items,
                                   ks=[10, 20], device=device)

        hr20    = val_metrics.get("HR@20", 0.0)
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
        item_W_test = model.build_eval_embeddings(cold_items)

        def score_fn_test(b):
            return model(b, item_W_test)

        test_metrics = evaluate(score_fn_test, test_loader, item_W_test, cold_items,
                                ks=[10, 20], device=device)

    print("\n=== TEST RESULTS ===")
    print(format_results(test_metrics))

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.dataset}_CCFCRec_seed{args.seed}.json")
    result = {
        "dataset":    args.dataset,
        "ablation":   "CCFCRec",
        "seed":       args.seed,
        "best_epoch": best_epoch,
        "test":       test_metrics,
        "config": {
            "d":               args.d,
            "n_mlp_layers":    args.n_mlp_layers,
            "dropout":         args.dropout,
            "tau":             args.tau,
            "lambda_c":        args.lambda_c,
            "n_contra_sample": args.n_contra_sample,
            "epochs":          args.epochs,
            "batch_size":      args.batch_size,
            "lr":              args.lr,
            "wd":              args.wd,
            "patience":        args.patience,
        },
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="CCFCRec: Contrastive CF for Cold-Start (adapted for SBR)")
    p.add_argument("--data_dir",         required=True)
    p.add_argument("--output_dir",       required=True)
    p.add_argument("--dataset",          required=True)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--maxlen",           type=int,   default=0)
    p.add_argument("--d",                type=int,   default=128)
    p.add_argument("--n_mlp_layers",     type=int,   default=2,   help="Layers in content MLP")
    p.add_argument("--dropout",          type=float, default=0.1)
    p.add_argument("--tau",              type=float, default=0.1,  help="InfoNCE temperature")
    p.add_argument("--lambda_c",         type=float, default=0.1,  help="Contrastive loss weight")
    p.add_argument("--n_contra_sample",  type=int,   default=64,   help="Warm items per contrastive batch")
    p.add_argument("--epochs",           type=int,   default=30)
    p.add_argument("--batch_size",       type=int,   default=100)
    p.add_argument("--lr",               type=float, default=1e-3)
    p.add_argument("--wd",               type=float, default=1e-5)
    p.add_argument("--patience",         type=int,   default=5)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
