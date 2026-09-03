"""
nirgnn.py — NirGNN Baseline (adaptation with optional 2-level hierarchy)
-------------------------------------------------------------------------
Original paper: "Dual Intent Enhanced Graph Neural Network for Session-based
New Item Recommendation" (NirGNN), Ee1s, WWW 2023.
Official code : https://github.com/Ee1s/NirGNN

WHY THIS IS AN ADAPTATION (not a direct port):
  The official NirGNN (WWW 2023) requires product taxonomy attributes:
    - taxo1, taxo2, taxo3   (3-level taxonomy hierarchy)
    - ca1, ca2              (category attributes)
  It also uses Beta distribution + Bhattacharyya distance for zero-shot
  new-item representation learning. These require taxonomy data not always
  available in RecSys benchmarks.

OUR ADAPTATION (TransferMLP + optional hierarchy):
  Core idea — same as NirGNN: learn to transfer warm-item representations
  to cold items via category structure.
  1. Train SR-GNN on warm sessions (cold items excluded from training).
  2. At test time: build cold item embeddings via TransferMLP:
       h_cold_i = TransferMLP( mean(warm_items_in_same_cat) )
  3. Optional --use_hierarchy: if cat_parent.json is available and the
     cold item's leaf-category has NO warm items, fall back to the mean of
     warm items across sibling categories (sharing the same parent).
     This implements the hierarchical fallback principle from NirGNN.

  cat_parent.json format: {"cat_id": parent_cat_id_or_null, ...}
  → If all parents are null (RetailRocket / Diginetica), --use_hierarchy is
    a no-op (falls through to TransferMLP only, same as without the flag).
  → If hierarchy exists (CellPhones: 31 cats with real parents), sibling
    transfer kicks in for orphan cold categories.

LOSS:
  L = CE(z_s, target) + β * L_transfer
  where L_transfer = ||h_item - TransferMLP(cat_proto)||²

Usage:
    python baselines/nirgnn.py --data_dir $DATA --output_dir ~/results [--use_hierarchy]
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
from catprocl.cold_inference import build_eval_item_embeddings
from evaluation.evaluator import evaluate, format_results


# ── Transfer GNN: cat prototype → item representation ─────────────────────────
class TransferMLP(nn.Module):
    """
    Lightweight 2-layer MLP that maps category prototype to item embedding.
    Used at test time to infer cold item representations.
    Input  : h_cat (d,) — mean embedding of warm items in category
    Output : h_cold (d,) — predicted cold item embedding
    """

    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, d),
        )

    def forward(self, h_cat: torch.Tensor) -> torch.Tensor:
        return self.net(h_cat)


# ── Main model ────────────────────────────────────────────────────────────────
class NirGNNModel(nn.Module):
    """
    SR-GNN backbone + TransferMLP for cold inference.
    """

    def __init__(self, n_items: int, d: int = 128, n_steps: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.d       = d
        self.n_steps = n_steps

        # Shared item embedding (warm items only trained; cold items inferred)
        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)

        # GNN layers (SR-GNN)
        self.W_in  = nn.Linear(d, d, bias=True)
        self.W_out = nn.Linear(d, d, bias=True)
        self.gru   = nn.GRUCell(2 * d, d)

        # Soft-attention readout
        self.W1 = nn.Linear(d, d, bias=False)
        self.W2 = nn.Linear(d, d, bias=False)
        self.q  = nn.Linear(d, 1, bias=False)
        self.W3 = nn.Linear(2 * d, d, bias=False)

        # Transfer module for cold inference
        self.transfer = TransferMLP(d, dropout=dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_emb.weight, std=0.1)
        for name, p in self.named_parameters():
            if "item_emb" in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, batch: dict) -> torch.Tensor:
        """Returns z_s (B, d)."""
        node_ids = batch["node_ids"]
        A_in     = batch["A_in"]
        A_out    = batch["A_out"]
        seq_idx  = batch["seq_idx"]
        mask     = batch["mask"]

        B, N = node_ids.shape
        L    = seq_idx.shape[1]

        x = self.item_emb(node_ids)
        h = x
        for _ in range(self.n_steps):
            agg_in  = torch.bmm(A_in, h)
            agg_out = torch.bmm(A_out, h)
            a       = torch.cat([self.W_in(agg_in), self.W_out(agg_out)], dim=-1)
            h_flat  = self.gru(a.view(B * N, 2 * self.d), h.view(B * N, self.d))
            h       = h_flat.view(B, N, self.d)

        seq_idx_exp = seq_idx.unsqueeze(-1).expand(B, L, self.d)
        seq_h       = torch.gather(h, 1, seq_idx_exp)

        lengths  = mask.sum(dim=1).long()
        last_idx = (lengths - 1).clamp(min=0)
        h_last   = seq_h.gather(1, last_idx.view(B, 1, 1).expand(B, 1, self.d)).squeeze(1)

        alpha = self.q(torch.sigmoid(self.W1(seq_h) + self.W2(h_last).unsqueeze(1)))
        alpha = alpha * mask.unsqueeze(-1)
        s_g   = (alpha * seq_h).sum(dim=1)

        return self.W3(torch.cat([s_g, h_last], dim=-1))

    def forward(self, batch: dict) -> torch.Tensor:
        return self.encode(batch)

    def all_item_embeddings(self) -> torch.Tensor:
        return self.item_emb.weight  # (n_items+1, d)

    @torch.no_grad()
    def build_cold_embeddings(
        self,
        item2cat: dict,
        cat2items_warm: dict,
        cold_items: set,
        device: str,
        cat_parent: dict = None,   # {cat_id: parent_cat_id or None} — optional hierarchy
    ) -> torch.Tensor:
        """
        Build item embedding matrix with cold items replaced by TransferMLP output.

        Level 1: cold item's leaf-category has warm items
          → h_cold = TransferMLP(mean(warm_items_in_same_cat))

        Level 2 (only if --use_hierarchy and hierarchy exists):
          cold item's leaf-category has NO warm items but parent is known
          → gather all warm items from sibling categories (same parent)
          → h_cold = TransferMLP(mean(warm_items_in_sibling_cats))

        Items with no match at either level retain their random init embedding.
        """
        self.eval()
        emb = self.item_emb.weight.clone()   # (n_items+1, d)

        # ── Build parent → {sibling leaf cats} mapping ────────────────────
        parent2sibs: dict = {}
        if cat_parent:
            for cat_id, parent in cat_parent.items():
                if parent is not None:
                    parent2sibs.setdefault(parent, set()).add(cat_id)

        # ── Level-1 prototypes: TransferMLP(mean(warm_in_cat)) ────────────
        cat_protos: dict = {}
        for cat_id, warm_ids in cat2items_warm.items():
            ids_t = torch.tensor(warm_ids, dtype=torch.long, device=device)
            proto = emb[ids_t].mean(0)
            cat_protos[cat_id] = self.transfer(proto)

        # ── Build parent-level prototypes for Level-2 fallback ─────────────
        parent_protos: dict = {}
        if cat_parent and parent2sibs:
            for parent, sibling_cats in parent2sibs.items():
                sib_warm = []
                for sib in sibling_cats:
                    if sib in cat2items_warm:
                        sib_warm.extend(cat2items_warm[sib])
                if sib_warm:
                    ids_t = torch.tensor(sib_warm, dtype=torch.long, device=device)
                    proto = emb[ids_t].mean(0)
                    parent_protos[parent] = self.transfer(proto)

        # ── Assign cold item embeddings ────────────────────────────────────
        for item_id in cold_items:
            cat_id = item2cat.get(item_id, None)
            if cat_id is None:
                continue

            if cat_id in cat_protos:
                # Level 1: same leaf category has warm items
                emb[item_id] = cat_protos[cat_id]
            elif cat_parent:
                # Level 2: sibling-category fallback via parent
                parent = cat_parent.get(cat_id, None)
                if parent is not None and parent in parent_protos:
                    emb[item_id] = parent_protos[parent]

        return emb


# ── Transfer loss ─────────────────────────────────────────────────────────────
def compute_transfer_loss(
    model: NirGNNModel,
    item2cat: dict[int, int],
    cat2items_warm: dict[int, list[int]],
    device: str,
    n_sample_cats: int = 32,
) -> torch.Tensor:
    """
    Reconstruction loss: for randomly sampled warm items, predict their
    embedding from the category prototype.

    For each sampled category c:
      - split warm items into proto_set and target_set
      - h_cat  = mean(item_emb[proto_set])
      - h_pred = TransferMLP(h_cat)
      - L      = mean ||h_pred - item_emb[target_item]||²

    This trains TransferMLP to map category means → individual item embeddings.
    """
    if not cat2items_warm:
        return torch.tensor(0.0, device=device)

    cats = list(cat2items_warm.keys())
    # Sample categories that have at least 2 items
    valid_cats = [c for c in cats if len(cat2items_warm[c]) >= 2]
    if not valid_cats:
        return torch.tensor(0.0, device=device)

    n_sample = min(n_sample_cats, len(valid_cats))
    perm     = torch.randperm(len(valid_cats))[:n_sample].tolist()
    sampled  = [valid_cats[i] for i in perm]

    losses = []
    for cat_id in sampled:
        warm_ids = cat2items_warm[cat_id]
        # Hold one item out as target
        tgt_idx  = torch.randint(len(warm_ids), (1,)).item()
        tgt_id   = warm_ids[tgt_idx]
        proto_ids = [w for i, w in enumerate(warm_ids) if i != tgt_idx]

        if not proto_ids:
            continue

        proto_t  = torch.tensor(proto_ids, dtype=torch.long, device=device)
        h_cat    = model.item_emb(proto_t).mean(0)                 # (d,)
        h_pred   = model.transfer(h_cat)                           # (d,)
        h_target = model.item_emb(torch.tensor([tgt_id], device=device)).squeeze(0)
        losses.append(F.mse_loss(h_pred, h_target.detach()))

    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


# ── Training loop ─────────────────────────────────────────────────────────────
def load_cat_parent(data_dir: str) -> dict:
    """
    Load cat_parent.json → {int(cat_id): int(parent_id) or None}.
    Returns empty dict if file not found.
    """
    import os
    path = os.path.join(data_dir, "cat_parent.json")
    # Also try one level up (unified → cold_20 subdir case)
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(data_dir), "cat_parent.json")
    if not os.path.exists(path):
        return {}
    raw = json.load(open(path))
    return {int(k): (int(v) if v is not None else None) for k, v in raw.items()}


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"NirGNN | dataset={args.dataset} | seed={args.seed} | device={device}")
    print(f"  use_hierarchy={args.use_hierarchy}")

    train_loader, val_loader, test_loader, data = get_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size,
        num_workers=args.num_workers, maxlen=args.maxlen,
    )
    n_items        = data["n_items"]
    cold_items     = data["cold_items"]
    item2cat       = data["item2cat"]
    cat2items_warm = data["cat2items_warm"]
    print(f"  n_items={n_items:,}  cold_items={len(cold_items):,}")
    print(f"  n_cats_with_warm={len(cat2items_warm):,}")

    # Load optional category hierarchy
    cat_parent = None
    if args.use_hierarchy:
        cat_parent = load_cat_parent(args.data_dir)
        n_with_parent = sum(1 for v in cat_parent.values() if v is not None)
        print(f"  cat_parent loaded: {len(cat_parent)} cats, {n_with_parent} with real parent")

    torch.manual_seed(args.seed)
    model = NirGNNModel(
        n_items=n_items, d=args.d, n_steps=args.n_steps, dropout=args.dropout,
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
        total_loss     = 0.0
        total_transfer = 0.0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Ep {epoch:3d}", ncols=100, leave=False)
        for batch in pbar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            targets = batch["targets"]

            z_s      = model(batch)
            item_emb = model.all_item_embeddings()
            scores   = z_s @ item_emb.T
            loss_rec = F.cross_entropy(scores, targets)

            # Transfer reconstruction loss
            loss_trans = compute_transfer_loss(
                model, item2cat, cat2items_warm, device,
                n_sample_cats=args.n_sample_cats,
            )
            loss = loss_rec + args.beta * loss_trans

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            total_loss     += loss_rec.item()
            total_transfer += loss_trans.item()
            pbar.set_postfix(rec=f"{loss_rec.item():.4f}", trans=f"{loss_trans.item():.4f}")

        pbar.close()
        scheduler.step()

        # ── Eval with cold-replaced embeddings ────────────────────────────
        model.eval()
        with torch.no_grad():
            cold_emb = model.build_cold_embeddings(
                item2cat, cat2items_warm, cold_items, device,
                cat_parent=cat_parent,
            )

            def score_fn(b):
                z_s = model.encode(b)
                return z_s @ cold_emb.T

            val_res = evaluate(score_fn, val_loader, cold_emb, cold_items,
                               ks=[10, 20], device=device)

        val_hr20 = val_res["HR@20"]
        elapsed  = time.time() - t0
        epoch_logs.append({
            "epoch":        epoch,
            "loss":         total_loss / len(train_loader),
            "transfer":     total_transfer / len(train_loader),
            "epoch_time_s": elapsed,
            **{f"val_{k}": v for k, v in val_res.items()},
        })
        print(f"Epoch {epoch:3d} | rec={total_loss/len(train_loader):.4f} "
              f"trans={total_transfer/len(train_loader):.4f} | "
              f"Val HR@20={val_hr20:.4f} | {elapsed:.1f}s")

        if val_hr20 > best_val_hr20:
            best_val_hr20 = val_hr20
            best_epoch    = epoch
            patience_cnt  = 0
            with torch.no_grad():
                # Re-build cold embeddings with final model weights
                cold_emb_test = model.build_cold_embeddings(
                    item2cat, cat2items_warm, cold_items, device,
                    cat_parent=cat_parent,
                )

                def score_fn_test(b):
                    z_s = model.encode(b)
                    return z_s @ cold_emb_test.T

                test_res = evaluate(score_fn_test, test_loader, cold_emb_test, cold_items,
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
        "dataset":       args.dataset,
        "ablation":      "NirGNN",
        "seed":          args.seed,
        "best_epoch":    best_epoch,
        "use_hierarchy": args.use_hierarchy,
        "test":          best_test,
        "epoch_logs":    epoch_logs,
        "config":        vars(args),
    }
    fname = os.path.join(args.output_dir, f"{args.dataset}_NirGNN_seed{args.seed}.json")
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
    p.add_argument("--n_steps",        type=int,   default=1)
    p.add_argument("--dropout",        type=float, default=0.1)
    p.add_argument("--batch_size",     type=int,   default=100)
    p.add_argument("--epochs",         type=int,   default=30)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--lr_dc",          type=float, default=0.1)
    p.add_argument("--lr_dc_step",     type=int,   default=3)
    p.add_argument("--weight_decay",   type=float, default=1e-5)
    p.add_argument("--clip",           type=float, default=5.0)
    p.add_argument("--patience",       type=int,   default=5)
    p.add_argument("--num_workers",    type=int,   default=4)
    p.add_argument("--maxlen",         type=int,   default=0,
                   help="Truncate sessions to last N items (0=full session)")
    p.add_argument("--beta",           type=float, default=0.1,
                   help="Weight for transfer reconstruction loss")
    p.add_argument("--n_sample_cats",  type=int,   default=32,
                   help="Num categories sampled per batch for transfer loss")
    p.add_argument("--use_hierarchy",  action="store_true", default=False,
                   help="Load cat_parent.json and use 2-level hierarchy fallback "
                        "for cold items whose leaf-category has no warm items. "
                        "No-op if cat_parent.json has all-null parents (RR/Digi).")
    args = p.parse_args()

    args.output_dir = os.path.expanduser(args.output_dir)
    train(args)
