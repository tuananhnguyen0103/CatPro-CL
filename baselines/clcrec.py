"""
clcrec.py — CLCRec Baseline (adapted for SBR)
----------------------------------------------
Paper: "Contrastive Learning for Cold-Start Recommendation" (ACM MM 2021)
       Wei et al., National University of Singapore.
GitHub: github.com/weiyinwei/CLCRec

ORIGINAL CLCRec IDEA:
  Reformulates cold-start from information-theoretic view:
    Maximize I(content_features; collaborative_signals)
  Lower-bounded by:
    I(collab_emb_user; collab_emb_item) + I(collab_emb_item; content_emb_item)

  Concretely, two contrastive objectives:
    L1: InfoNCE between user_collab_emb and item_collab_emb (user-item alignment)
    L2: InfoNCE between item_collab_emb and item_content_emb (content-collab alignment)

  Cold inference: content encoder maps cold item features → collab-aligned space.

DIFFERENCE vs CCFCRec:
  CCFCRec: contrasts content-CF vs co-occurrence-CF (both CF-style)
  CLCRec:  contrasts user_collab vs item_collab (L1) AND item_collab vs content (L2)
           → explicit user-item alignment + content-collaborative alignment

OUR ADAPTATION FOR SBR:
  - "User" = session; user embedding = mean-pool over session items (collab)
  - Content feature = category embedding
  - item_collab_emb = standard embedding table (learned from sessions)
  - item_content_emb = MLP(cat_emb[cat(item)])

  Training losses:
    L_rec = CE(session_repr @ item_collab.T, target)     [recommendation]
    L1    = InfoNCE(session_repr, item_collab[target])   [user↔item]
    L2    = InfoNCE(item_collab[warm], item_content[warm]) [collab↔content]
    L     = L_rec + lambda1 * L1 + lambda2 * L2

  Cold inference: item_content_emb (content → collab-aligned space via L2)

Usage:
  python baselines/clcrec.py \\
      --data_dir ~/data/retailrocket_unified/cold_20 \\
      --output_dir ~/results_v6/data/retailrocket/fullen/7_CLCRec \\
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


def info_nce(anchor: torch.Tensor, positive: torch.Tensor, tau: float) -> torch.Tensor:
    """
    InfoNCE loss.
    anchor:   (N, d)
    positive: (N, d)
    Positives: anchor[i] ↔ positive[i]; all others are negatives.
    """
    anchor   = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    logits   = (anchor @ positive.T) / tau      # (N, N)
    labels   = torch.arange(len(anchor), device=anchor.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


# ── CLCRec Model ──────────────────────────────────────────────────────────────
class CLCRecModel(nn.Module):
    """
    Two item representation spaces:
      collab  : item_emb_collab[item_id]        — collaborative (learned from sessions)
      content : MLP(cat_emb[cat(item)])          — content-based (generalizes to cold items)

    Session repr: mean-pool of item_emb_collab over session prefix.
    Cold inference: content space (aligned to collab via L2 contrastive).
    """

    def __init__(self, n_items: int, n_cats: int, item2cat_t: torch.Tensor,
                 d: int = 128, n_mlp_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d       = d
        self.n_items = n_items

        # ── Collaborative embedding ────────────────────────────────────────
        self.item_emb_collab = nn.Embedding(n_items + 1, d, padding_idx=0)

        # ── Content encoder ────────────────────────────────────────────────
        self.cat_emb = nn.Embedding(n_cats + 1, d, padding_idx=0)
        layers = []
        for _ in range(n_mlp_layers):
            layers += [nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(d, d))
        self.content_encoder = nn.Sequential(*layers)

        self.register_buffer("item2cat_t", item2cat_t)

    # ── Representations ────────────────────────────────────────────────────
    def content_repr(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Content-based representation (usable for cold items)."""
        cat_ids = self.item2cat_t[item_ids]
        return self.content_encoder(self.cat_emb(cat_ids))

    def session_repr(self, item_seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Mean-pool collaborative embeddings over valid session positions.
        item_seq: (B, L); mask: (B, L) float.
        Returns: (B, d).
        """
        embs   = self.item_emb_collab(item_seq)           # (B, L, d)
        mask_e = mask.unsqueeze(-1).float()
        pooled = (embs * mask_e).sum(dim=1)               # (B, d)
        denom  = mask_e.sum(dim=1).clamp(min=1e-9)
        return pooled / denom

    def build_eval_embeddings(self, cold_items_set: set) -> torch.Tensor:
        """
        (n_items+1, d): warm → collab_emb, cold → content_repr.
        """
        W    = self.item_emb_collab.weight.clone()
        all_ids = torch.arange(self.n_items + 1, device=W.device)
        cont = self.content_repr(all_ids)                 # (n_items+1, d)

        cold_ids = torch.tensor(
            [i for i in cold_items_set if 1 <= i <= self.n_items],
            dtype=torch.long, device=W.device
        )
        if cold_ids.numel() > 0:
            W[cold_ids] = cont[cold_ids]
        return W

    def forward(self, batch: dict, item_W: torch.Tensor) -> torch.Tensor:
        """(B, n_items+1) recommendation scores."""
        item_seq, mask = extract_item_seq(batch)
        s = self.session_repr(item_seq, mask)             # (B, d)
        return s @ item_W.T                               # (B, n_items+1)


# ── Training ──────────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | CLCRec | Dataset: {args.dataset} | Seed: {args.seed}")

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

    item2cat_t = torch.zeros(n_items + 1, dtype=torch.long)
    for iid, cid in item2cat.items():
        if 1 <= int(iid) <= n_items:
            item2cat_t[int(iid)] = int(cid)

    warm_set = set(range(1, n_items + 1)) - cold_items

    model = CLCRecModel(
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
            targets = batch["targets"]     # (B,)

            optimizer.zero_grad()

            item_seq, mask = extract_item_seq(batch)

            # ── L_rec: recommendation ─────────────────────────────────────
            item_W   = model.item_emb_collab.weight
            s        = model.session_repr(item_seq, mask)     # (B, d)
            scores   = s @ item_W.T                           # (B, n_items+1)
            scores[:, 0] = float("-inf")
            loss_rec = F.cross_entropy(scores, targets)

            # ── L1: user↔item alignment (InfoNCE on session vs target item) ─
            target_collab = model.item_emb_collab(targets)    # (B, d)
            loss_l1 = info_nce(s, target_collab, tau=args.tau)

            # ── L2: collab↔content alignment for warm items ───────────────
            node_ids    = batch["node_ids"].flatten()
            warm_batch  = node_ids[(node_ids > 0)]
            warm_batch  = warm_batch[
                torch.tensor([i.item() in warm_set for i in warm_batch], device=device)
            ].unique()[:args.n_contra_sample]

            if warm_batch.numel() < 2:
                idx        = torch.randint(0, len(warm_ids_all), (args.n_contra_sample,))
                warm_batch = warm_ids_all[idx].to(device)

            collab_w = model.item_emb_collab(warm_batch)      # (N, d)
            content_w = model.content_repr(warm_batch)         # (N, d)
            loss_l2   = info_nce(collab_w, content_w, tau=args.tau)

            loss = loss_rec + args.lambda1 * loss_l1 + args.lambda2 * loss_l2
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
    out_path = os.path.join(args.output_dir, f"{args.dataset}_CLCRec_seed{args.seed}.json")
    result = {
        "dataset":    args.dataset,
        "ablation":   "CLCRec",
        "seed":       args.seed,
        "best_epoch": best_epoch,
        "test":       test_metrics,
        "config": {
            "d":               args.d,
            "n_mlp_layers":    args.n_mlp_layers,
            "dropout":         args.dropout,
            "tau":             args.tau,
            "lambda1":         args.lambda1,
            "lambda2":         args.lambda2,
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
    p = argparse.ArgumentParser(description="CLCRec: Contrastive Learning for Cold-Start (adapted for SBR)")
    p.add_argument("--data_dir",         required=True)
    p.add_argument("--output_dir",       required=True)
    p.add_argument("--dataset",          required=True)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--maxlen",           type=int,   default=0)
    p.add_argument("--d",                type=int,   default=128)
    p.add_argument("--n_mlp_layers",     type=int,   default=2)
    p.add_argument("--dropout",          type=float, default=0.1)
    p.add_argument("--tau",              type=float, default=0.1,  help="InfoNCE temperature")
    p.add_argument("--lambda1",          type=float, default=0.1,  help="L1 user-item weight")
    p.add_argument("--lambda2",          type=float, default=0.1,  help="L2 collab-content weight")
    p.add_argument("--n_contra_sample",  type=int,   default=64)
    p.add_argument("--epochs",           type=int,   default=30)
    p.add_argument("--batch_size",       type=int,   default=100)
    p.add_argument("--lr",               type=float, default=1e-3)
    p.add_argument("--wd",               type=float, default=1e-5)
    p.add_argument("--patience",         type=int,   default=5)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
