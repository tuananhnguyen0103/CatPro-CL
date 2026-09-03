"""
nclsbr.py — NCL-SBR Baseline
------------------------------
Lin et al., "Improving Graph Collaborative Filtering with
Neighborhood-enriched Contrastive Learning", WWW 2022 — adapted for SBR.

SBR adaptation: NCL-SBR uses UNSUPERVISED K-means prototypes at the
SESSION level (vs our CatPro-CL which uses CATEGORY-SUPERVISED prototypes
at the ITEM level). Key comparison to show category supervision is superior.

Architecture:
  - Backbone: SR-GNN (same as A1)
  - Prototypes: K-means on SESSION embeddings from previous epoch
                K = n_cats (fair comparison with A4/A5/A6)
  - CL loss  : InfoNCE between z_s and its nearest cluster prototype
  - Loss     : CE(z_s, target) + λ * InfoNCE(z_s, proto_k)

Note on cold: prototypes only cover warm sessions (cold items not in train),
so cold performance depends solely on the backbone.

Usage:
    python baselines/nclsbr.py --data_dir $DATA --output_dir ~/results
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


# ── Spherical K-means on session embeddings ──────────────────────────────────
@torch.no_grad()
def session_kmeans(
    session_embs: torch.Tensor,   # (M, d) — collected from one epoch
    n_clusters: int,
    n_iters: int = 30,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Spherical K-means on session embeddings.
    Returns:
        centroids   : (K, d) normalised
        assignments : (M,)  cluster_id per session
    """
    M, d = session_embs.shape
    z    = F.normalize(session_embs.to(device), dim=1)

    # Random init from actual sessions
    k    = min(n_clusters, M)
    perm = torch.randperm(M, device=device)[:k]
    C    = z[perm].clone()  # (k, d) already normalised

    if k < n_clusters:
        pad = F.normalize(torch.randn(n_clusters - k, d, device=device), dim=1)
        C   = torch.cat([C, pad], dim=0)

    asgn = torch.zeros(M, dtype=torch.long, device=device)
    for it in range(n_iters):
        # Assign
        sims     = z @ C.T         # (M, K)
        new_asgn = sims.argmax(1)  # (M,)
        if it > 0 and torch.equal(new_asgn, asgn):
            break
        asgn = new_asgn

        # Update centroids
        C_new = torch.zeros_like(C)
        for c in range(n_clusters):
            mask = asgn == c
            if mask.sum() > 0:
                C_new[c] = F.normalize(z[mask].mean(0), dim=0)
            else:
                C_new[c] = C[c]
        C = C_new

    return C, asgn  # (K, d), (M,)


# ── Model ─────────────────────────────────────────────────────────────────────
class NCLSBRModel(nn.Module):
    """SR-GNN backbone (identical to A1/A4)."""

    def __init__(self, n_items: int, d: int = 128, n_steps: int = 1):
        super().__init__()
        self.d = d
        self.n_steps = n_steps

        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)

        self.W_in  = nn.Linear(d, d, bias=True)
        self.W_out = nn.Linear(d, d, bias=True)
        self.gru   = nn.GRUCell(2 * d, d)

        self.W1 = nn.Linear(d, d, bias=False)
        self.W2 = nn.Linear(d, d, bias=False)
        self.q  = nn.Linear(d, 1, bias=False)
        self.W3 = nn.Linear(2 * d, d, bias=False)

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
        return self.item_emb.weight


# ── InfoNCE loss (session vs prototype) ──────────────────────────────────────
def infonce_session_proto(
    z_s: torch.Tensor,           # (B, d) session embeddings
    protos: torch.Tensor,        # (K, d) cluster prototypes (normalised)
    assignments: torch.Tensor,   # (B,) cluster id per session in this batch
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    InfoNCE: pull z_s toward its assigned prototype, push from others.
    Positive: protos[assignments[b]]
    Negatives: all other K-1 prototypes (in-batch negatives across batch)
    """
    z_norm = F.normalize(z_s, dim=1)        # (B, d)
    # scores against all K prototypes
    logits = z_norm @ protos.T / temperature  # (B, K)
    return F.cross_entropy(logits, assignments.to(z_s.device))


# ── Training loop ─────────────────────────────────────────────────────────────
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"NCL-SBR | dataset={args.dataset} | seed={args.seed} | device={device}")

    train_loader, val_loader, test_loader, data = get_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size,
        num_workers=args.num_workers, maxlen=args.maxlen,
    )
    n_items    = data["n_items"]
    n_cats     = data["n_cats"]       # K = n_cats for fair comparison
    cold_items = data["cold_items"]
    print(f"  n_items={n_items:,}  n_cats={n_cats}  cold_items={len(cold_items):,}")

    torch.manual_seed(args.seed)
    model = NCLSBRModel(n_items=n_items, d=args.d, n_steps=args.n_steps).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)

    # K-means state — initialised after epoch 1
    protos: torch.Tensor | None = None     # (K, d) on device
    session_assignments: list[int] = []    # tracks cluster per training session (flat)

    best_val_hr20 = 0.0
    best_epoch    = 0
    best_test     = {}
    patience_cnt  = 0
    epoch_logs    = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss  = 0.0
        total_cl    = 0.0
        t0 = time.time()

        # Collect (session_emb, assignment) from this epoch for next epoch's K-means
        emb_buf: list[torch.Tensor] = []

        batch_idx = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch:3d}", ncols=100, leave=False)
        for batch in pbar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            targets = batch["targets"]
            B       = targets.shape[0]

            z_s      = model(batch)                          # (B, d)
            item_emb = model.all_item_embeddings()
            scores   = z_s @ item_emb.T                     # (B, n+1)
            loss_rec = F.cross_entropy(scores, targets)
            loss     = loss_rec

            # ── NCL contrastive loss (available from epoch 2 onward) ────────
            if protos is not None and args.lambda_cl > 0:
                # Look up pre-computed assignments for this batch
                start = batch_idx * args.batch_size
                end   = start + B
                if end <= len(session_assignments):
                    asgn_batch = torch.tensor(
                        session_assignments[start:end], dtype=torch.long, device=device
                    )
                    cl_loss = infonce_session_proto(z_s, protos, asgn_batch, args.temperature)
                    loss    = loss_rec + args.lambda_cl * cl_loss
                    total_cl += cl_loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            total_loss += loss_rec.item()
            pbar.set_postfix(loss=f"{loss_rec.item():.4f}")

            # Collect embeddings for K-means (detach)
            emb_buf.append(z_s.detach().cpu())
            batch_idx += 1

        pbar.close()
        scheduler.step()

        # ── Update K-means after each epoch ────────────────────────────────
        all_embs    = torch.cat(emb_buf, dim=0)             # (M, d)
        protos, asgn = session_kmeans(all_embs, n_cats, device=device)
        session_assignments = asgn.cpu().tolist()

        # ── Eval ──────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            item_emb = model.all_item_embeddings()

            def score_fn(b):
                return model(b) @ item_emb.T

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
        cl_str   = f" CL={total_cl/max(batch_idx,1):.4f}" if protos is not None else ""
        print(f"Epoch {epoch:3d} | loss={total_loss/len(train_loader):.4f}{cl_str} | "
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
        "ablation":   "NCL-SBR",
        "seed":       args.seed,
        "best_epoch": best_epoch,
        "test":       best_test,
        "epoch_logs": epoch_logs,
        "config":     vars(args),
    }
    fname = os.path.join(args.output_dir, f"{args.dataset}_NCLSBR_seed{args.seed}.json")
    with open(fname, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved → {fname}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",     required=True)
    p.add_argument("--output_dir",   default="~/results")
    p.add_argument("--dataset",      default="retailrocket")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--d",            type=int,   default=128)
    p.add_argument("--n_steps",      type=int,   default=1)
    p.add_argument("--batch_size",   type=int,   default=100)
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--lr_dc",        type=float, default=0.1)
    p.add_argument("--lr_dc_step",   type=int,   default=3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--clip",         type=float, default=5.0)
    p.add_argument("--patience",     type=int,   default=10)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--maxlen",       type=int,   default=0,
                   help="Truncate sessions to last N items (0=full)")
    p.add_argument("--lambda_cl",    type=float, default=0.1,
                   help="Weight for session-level NCL contrastive loss")
    p.add_argument("--temperature",  type=float, default=0.1,
                   help="InfoNCE temperature")
    args = p.parse_args()

    args.output_dir = os.path.expanduser(args.output_dir)
    train(args)
