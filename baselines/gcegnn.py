"""
gcegnn.py — GCE-GNN Baseline
------------------------------
Wang et al., "Global Context Enhanced Graph Neural Networks for
Session-based Recommendation", SIGIR 2020.
Official code: https://github.com/CCIIPLab/GCE-GNN

Architecture (copied verbatim from official aggregator.py + model.py):
  LocalAggregator  : GAT with 4 attention vectors for 4 edge types
                     (self-loop=1, forward=2, backward=3, bidirectional=4)
  GlobalAggregator : Neighbor-sampling GNN with session-context attention
  Readout          : Position embeddings + soft-attention (compute_scores)

Adaptation notes:
  - Model classes (LocalAggregator, GlobalAggregator, CombineGraph)
    are taken verbatim from official code.
  - Data pipeline unchanged: we use our graph-batch format
    (node_ids, A_in, A_out, seq_idx, mask) and convert A_in/A_out
    → 4-type integer adj via make_adj_4type().
  - Global adj_all table is built offline from sessions_train.txt via
    build_global_adj_table(), replicating official handle_adj().
  - Training loop, optimizer, eval: our standard protocol.
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catprocl.data_loader import get_dataloaders
from evaluation.evaluator import evaluate, format_results


# ════════════════════════════════════════════════════════════════════════
# Official aggregator classes (verbatim from CCIIPLab/GCE-GNN/aggregator.py)
# ════════════════════════════════════════════════════════════════════════
class LocalAggregator(nn.Module):
    """
    Local graph attention with 4 edge-type-specific attention vectors.
    adj : (B, N, N) long tensor with values 0/1/2/3/4.
          0 = no edge, 1 = self-loop, 2 = forward, 3 = backward, 4 = bi-directional.
    """

    def __init__(self, dim, alpha, dropout=0.0, name=None):
        super().__init__()
        self.dim     = dim
        self.dropout = dropout

        self.a_0 = Parameter(torch.Tensor(dim, 1))
        self.a_1 = Parameter(torch.Tensor(dim, 1))
        self.a_2 = Parameter(torch.Tensor(dim, 1))
        self.a_3 = Parameter(torch.Tensor(dim, 1))
        self.bias = Parameter(torch.Tensor(dim))

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, hidden, adj, mask_item=None):
        h          = hidden
        batch_size = h.shape[0]
        N          = h.shape[1]

        a_input = (h.repeat(1, 1, N).view(batch_size, N * N, self.dim)
                   * h.repeat(1, N, 1)).view(batch_size, N, N, self.dim)

        e_0 = self.leakyrelu(torch.matmul(a_input, self.a_0)).squeeze(-1)
        e_1 = self.leakyrelu(torch.matmul(a_input, self.a_1)).squeeze(-1)
        e_2 = self.leakyrelu(torch.matmul(a_input, self.a_2)).squeeze(-1)
        e_3 = self.leakyrelu(torch.matmul(a_input, self.a_3)).squeeze(-1)

        mask_neg = -9e15 * torch.ones_like(e_0)
        alpha = torch.where(adj.eq(1), e_0, mask_neg)
        alpha = torch.where(adj.eq(2), e_1, alpha)
        alpha = torch.where(adj.eq(3), e_2, alpha)
        alpha = torch.where(adj.eq(4), e_3, alpha)
        alpha = torch.softmax(alpha, dim=-1)

        return torch.matmul(alpha, h)


class GlobalAggregator(nn.Module):
    """
    Neighbor-sampling aggregator with session-context attention weighting.
    Verbatim from official GlobalAggregator except for minor style cleanup.
    """

    def __init__(self, dim, dropout, act=torch.relu, name=None):
        super().__init__()
        self.dropout = dropout
        self.act     = act
        self.dim     = dim

        self.w_1  = Parameter(torch.Tensor(dim + 1, dim))
        self.w_2  = Parameter(torch.Tensor(dim, 1))
        self.w_3  = Parameter(torch.Tensor(2 * dim, dim))
        self.bias = Parameter(torch.Tensor(dim))

    def forward(self, self_vectors, neighbor_vector, batch_size,
                masks, neighbor_weight, extra_vector=None):
        if extra_vector is not None:
            # (B, N, n_sample, d) element-wise with session context (B, N, 1, d)
            ctx = extra_vector.unsqueeze(2).expand_as(neighbor_vector)
            alpha_in = torch.cat([ctx * neighbor_vector,
                                  neighbor_weight.unsqueeze(-1)], dim=-1)
            alpha = F.leaky_relu(torch.matmul(alpha_in, self.w_1), negative_slope=0.2)
            alpha = torch.matmul(alpha, self.w_2).squeeze(-1)  # (B, N, n_sample)
            alpha = torch.softmax(alpha, dim=-1).unsqueeze(-1)
            neighbor_vector = torch.sum(alpha * neighbor_vector, dim=-2)  # (B, N, d)
        else:
            neighbor_vector = neighbor_vector.mean(dim=-2)                # (B, N, d)

        output = torch.cat([self_vectors, neighbor_vector], dim=-1)       # (B, N, 2d)
        output = F.dropout(output, self.dropout, training=self.training)
        output = torch.matmul(output, self.w_3)
        output = output.view(batch_size, -1, self.dim)
        return self.act(output)


# ════════════════════════════════════════════════════════════════════════
# Global adjacency table (replicates official handle_adj + build process)
# ════════════════════════════════════════════════════════════════════════
def build_global_adj_table(data_dir: str, n_items: int, n_sample: int = 12):
    """
    Build adj_all (n_items+1, n_sample) and num (n_items+1, n_sample) from
    training sessions, matching official GCE-GNN handle_adj() logic.
    """
    print("Building global adjacency table …")
    adj_dict = defaultdict(list)
    num_dict = defaultdict(lambda: defaultdict(int))

    with open(os.path.join(data_dir, "sessions_train.txt")) as f:
        for line in f:
            items = list(map(int, line.strip().split()))
            for i in range(len(items) - 1):
                u, v = items[i], items[i + 1]
                if u != v and 1 <= u <= n_items and 1 <= v <= n_items:
                    num_dict[u][v] += 1

    for u, v_cnt in num_dict.items():
        adj_dict[u] = list(v_cnt.keys())
        num_dict[u] = [v_cnt[v] for v in adj_dict[u]]

    adj_entity = np.zeros([n_items + 1, n_sample], dtype=np.int64)
    num_entity = np.zeros([n_items + 1, n_sample], dtype=np.float32)

    for item in range(1, n_items + 1):
        nbrs   = adj_dict[item]
        counts = num_dict[item] if isinstance(num_dict[item], list) else list(num_dict[item])
        n_nbr  = len(nbrs)
        if n_nbr == 0:
            continue
        if n_nbr >= n_sample:
            idx = np.random.choice(n_nbr, size=n_sample, replace=False)
        else:
            idx = np.random.choice(n_nbr, size=n_sample, replace=True)
        adj_entity[item] = [nbrs[i] for i in idx]
        num_entity[item] = [counts[i] for i in idx]

    print(f"  Global table: {sum(len(v) for v in adj_dict.values()):,} edges, "
          f"{len(adj_dict):,} source nodes")
    return adj_entity, num_entity


def make_adj_4type(A_in: torch.Tensor, A_out: torch.Tensor) -> torch.Tensor:
    """
    Convert our A_in (B,N,N) + A_out (B,N,N) float tensors to the
    official 4-type integer adj matching GCE-GNN's encoding:
      1 = self-loop
      2 = A_out edge only (u→v, one-way)
      3 = A_in edge only  (v→u, one-way)
      4 = both directions (bidirectional)
    """
    B, N, _ = A_in.shape
    adj = torch.zeros(B, N, N, dtype=torch.long, device=A_in.device)

    # Self-loops
    eye = torch.eye(N, dtype=torch.bool, device=A_in.device).unsqueeze(0)
    adj[eye.expand(B, -1, -1)] = 1

    fwd = (A_out > 0) & ~eye.expand(B, -1, -1)   # A_out but not self-loop
    bwd = (A_in  > 0) & ~eye.expand(B, -1, -1)   # A_in  but not self-loop

    # Bidirectional: both fwd[i][j] and fwd[j][i]
    bidir = fwd & fwd.transpose(1, 2)
    adj[bidir] = 4

    # Forward only (u→v but not v→u)
    adj[fwd & ~fwd.transpose(1, 2)] = 2

    # Backward only (v→u but not u→v)  — from the perspective of A_in
    adj[bwd & ~bwd.transpose(1, 2)] = 3

    return adj


# ════════════════════════════════════════════════════════════════════════
# Main model (adapted from official CombineGraph)
# ════════════════════════════════════════════════════════════════════════
class GCEGNNModel(nn.Module):
    """
    GCE-GNN model: LocalAggregator + GlobalAggregator + position-embedding readout.
    Adapted from official CombineGraph to work with our graph-batch format.
    """

    def __init__(self, n_items: int, d: int = 128, n_iter: int = 1,
                 n_sample: int = 12, alpha: float = 0.2,
                 dropout_local: float = 0.0, dropout_global: float = 0.5,
                 dropout_gcn: float = 0.0):
        super().__init__()
        self.d        = d
        self.n_items  = n_items
        self.n_iter   = n_iter
        self.n_sample = n_sample
        self.dropout_local  = dropout_local
        self.dropout_global = dropout_global

        # ── Official aggregators ──────────────────────────────────────────
        self.local_agg  = LocalAggregator(d, alpha, dropout=0.0)
        self.global_agg = nn.ModuleList([
            GlobalAggregator(d, dropout_gcn, act=torch.relu)
            for _ in range(n_iter)
        ])

        # ── Item + Position embeddings ────────────────────────────────────
        self.embedding     = nn.Embedding(n_items + 1, d, padding_idx=0)
        self.pos_embedding = nn.Embedding(200, d)   # positional encoding (official)

        # ── Readout parameters (official compute_scores) ──────────────────
        self.w_1   = Parameter(torch.Tensor(2 * d, d))
        self.w_2   = Parameter(torch.Tensor(d, 1))
        self.glu1  = nn.Linear(d, d)
        self.glu2  = nn.Linear(d, d, bias=False)

        self.leakyrelu = nn.LeakyReLU(alpha)

        # ── Global adj table (populated after build) ──────────────────────
        self.register_buffer("adj_all", torch.zeros(1, n_sample, dtype=torch.long))
        self.register_buffer("num_all", torch.zeros(1, n_sample))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.d)
        for w in self.parameters():
            w.data.uniform_(-stdv, stdv)

    def set_global_adj(self, adj_all: np.ndarray, num_all: np.ndarray):
        """Load pre-built global adj table (n_items+1, n_sample) into buffers."""
        self.adj_all = torch.from_numpy(adj_all).long()
        self.num_all = torch.from_numpy(num_all).float()

    def _sample(self, nodes: torch.Tensor):
        """Neighbor lookup: nodes (B, K) → (B, K*n_sample), (B, K*n_sample)."""
        flat = nodes.reshape(-1)
        return self.adj_all[flat], self.num_all[flat]

    def compute_scores(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Official GCE-GNN readout (compute_scores):
          - Position embeddings for each position
          - Session mean (mask-average) as context
          - Soft attention via GLU + w_2
          Returns: (B, n_items) score logits
        """
        mask_f = mask.float().unsqueeze(-1)       # (B, L, 1)
        B, L, _ = hidden.shape

        pos_emb = self.pos_embedding.weight[:L]   # (L, d)
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)  # (B, L, d)

        # Session mean
        hs = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        hs = hs.unsqueeze(1).expand(-1, L, -1)   # (B, L, d)

        nh   = torch.tanh(torch.matmul(torch.cat([pos_emb, hidden], dim=-1), self.w_1))
        nh   = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2) * mask_f   # (B, L, 1)
        select = (beta * hidden).sum(dim=1)           # (B, d)

        b = self.embedding.weight[1:]             # (n_items, d) — exclude padding
        return torch.matmul(select, b.T)          # (B, n_items)

    def forward(self, batch: dict, adj_4type: torch.Tensor) -> torch.Tensor:
        """
        batch     : dict from get_dataloaders (node_ids, seq_idx, mask, ...)
        adj_4type : (B, N, N) long with values 0-4 (from make_adj_4type)
        Returns   : (B, n_items) score logits (NOT embeddings)
        """
        node_ids = batch["node_ids"]   # (B, N) unique nodes
        seq_idx  = batch["seq_idx"]    # (B, L) alias indices into node_ids
        mask     = batch["mask"]       # (B, L) valid positions

        B, N = node_ids.shape
        L    = seq_idx.shape[1]

        # ── 1. Item embeddings ─────────────────────────────────────────────
        h = self.embedding(node_ids)   # (B, N, d)

        # ── 2. Local aggregation (official GAT 4-type) ─────────────────────
        h_local = self.local_agg(h, adj_4type)   # (B, N, d)

        # ── 3. Global aggregation (neighbor-sampling, official) ─────────────
        # Hop-0 item embeddings for global context (session mean)
        mask_node = (node_ids > 0).float().unsqueeze(-1)
        item_emb_mean = (h * mask_node).sum(1) / mask_node.sum(1).clamp(min=1.0)
        # session_info: repeat for each node position
        session_ctx = item_emb_mean.unsqueeze(1).expand(B, N, self.d)  # (B, N, d)

        item_neighbors   = [node_ids]    # (B, N)
        weight_neighbors = []
        support_size     = N

        for i in range(1, self.n_iter + 1):
            nbr, wt = self._sample(item_neighbors[-1])
            support_size *= self.n_sample
            item_neighbors.append(nbr.view(B, support_size))
            weight_neighbors.append(wt.view(B, support_size))

        entity_vectors = [self.embedding(idx) for idx in item_neighbors]

        for n_hop in range(self.n_iter):
            ev_next = []
            shape   = [B, -1, self.n_sample, self.d]
            for hop in range(self.n_iter - n_hop):
                agg = self.global_agg[n_hop]
                vec = agg(
                    self_vectors    = entity_vectors[hop],
                    neighbor_vector = entity_vectors[hop + 1].view(shape),
                    batch_size      = B,
                    masks           = None,
                    neighbor_weight = weight_neighbors[hop].view(B, -1, self.n_sample),
                    extra_vector    = session_ctx if hop == 0 else None,
                )
                ev_next.append(vec)
            entity_vectors = ev_next

        h_global = entity_vectors[0].view(B, N, self.d)   # (B, N, d)

        # ── 4. Combine local + global ──────────────────────────────────────
        h_local  = F.dropout(h_local,  self.dropout_local,  training=self.training)
        h_global = F.dropout(h_global, self.dropout_global, training=self.training)
        output   = h_local + h_global                       # (B, N, d)

        # ── 5. Readout: map node embeddings → sequence, compute scores ─────
        seq_exp = seq_idx.clamp(min=0).unsqueeze(-1).expand(B, L, self.d)
        seq_h   = torch.gather(output, 1, seq_exp)          # (B, L, d)

        return self.compute_scores(seq_h, mask)             # (B, n_items)

    def all_item_embeddings(self) -> torch.Tensor:
        return self.embedding.weight   # (n_items+1, d)


# ════════════════════════════════════════════════════════════════════════
# Training loop
# ════════════════════════════════════════════════════════════════════════
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"GCE-GNN | dataset={args.dataset} | seed={args.seed} | device={device}")
    print(f"  Ref: https://github.com/CCIIPLab/GCE-GNN  (SIGIR 2020)")

    train_loader, val_loader, test_loader, data = get_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size,
        num_workers=args.num_workers, maxlen=args.maxlen,
    )
    n_items    = data["n_items"]
    cold_items = data["cold_items"]
    print(f"  n_items={n_items:,}  cold_items={len(cold_items):,}  maxlen={args.maxlen or 'full'}")

    # Build global adj table (official approach)
    adj_all_np, num_all_np = build_global_adj_table(
        args.data_dir, n_items, args.n_sample
    )

    torch.manual_seed(args.seed)
    model = GCEGNNModel(
        n_items        = n_items,
        d              = args.d,
        n_iter         = args.n_iter,
        n_sample       = args.n_sample,
        alpha          = args.alpha,
        dropout_local  = args.dropout_local,
        dropout_global = args.dropout_global,
        dropout_gcn    = args.dropout_gcn,
    ).to(device)
    model.set_global_adj(adj_all_np, num_all_np)
    # Move buffers to device after loading
    model.adj_all = model.adj_all.to(device)
    model.num_all = model.num_all.to(device)

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
            targets = batch["targets"]                         # (B,)

            # Build 4-type adj on the fly
            adj_4type = make_adj_4type(batch["A_in"], batch["A_out"])  # (B,N,N) long

            # forward returns (B, n_items) logits (items 1..n_items)
            scores = model(batch, adj_4type)                   # (B, n_items)

            # target IDs are 1-indexed; scores[:, 0] corresponds to item 1
            # targets are 1..n_items; shift to 0-indexed for CE
            loss = F.cross_entropy(scores, targets - 1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        pbar.close()
        scheduler.step()

        # ── Eval (use our standard evaluator) ────────────────────────────
        model.eval()
        with torch.no_grad():
            item_emb = model.all_item_embeddings()   # (n_items+1, d)

            def score_fn(b):
                b_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in b.items()}
                adj_4t = make_adj_4type(b_dev["A_in"], b_dev["A_out"])
                # GCEGNNModel.forward returns (B, n_items) using embedding.weight[1:]
                # but evaluate() calls score_fn and expects (B, n_items+1) OR we
                # match via item_emb provided separately.
                # Return (B, n_items+1): prepend col of -inf for padding index 0
                logits = model(b_dev, adj_4t)   # (B, n_items)
                pad_col = torch.full((logits.size(0), 1), float("-inf"), device=device)
                return torch.cat([pad_col, logits], dim=1)  # (B, n_items+1)

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
        "ablation":   "GCE-GNN",
        "seed":       args.seed,
        "best_epoch": best_epoch,
        "test":       best_test,
        "epoch_logs": epoch_logs,
        "config":     vars(args),
    }
    fname = os.path.join(args.output_dir, f"{args.dataset}_GCEGNN_seed{args.seed}.json")
    with open(fname, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved → {fname}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",        required=True)
    p.add_argument("--output_dir",      default="~/results")
    p.add_argument("--dataset",         default="retailrocket")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--d",               type=int,   default=128)
    p.add_argument("--n_iter",          type=int,   default=1,
                   help="Number of GNN hops for global aggregation")
    p.add_argument("--n_sample",        type=int,   default=12,
                   help="Number of neighbor samples per node (global)")
    p.add_argument("--alpha",           type=float, default=0.2,
                   help="LeakyReLU negative slope")
    p.add_argument("--dropout_local",   type=float, default=0.0)
    p.add_argument("--dropout_global",  type=float, default=0.5)
    p.add_argument("--dropout_gcn",     type=float, default=0.0)
    p.add_argument("--batch_size",      type=int,   default=100)
    p.add_argument("--epochs",          type=int,   default=30)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--lr_dc",           type=float, default=0.1)
    p.add_argument("--lr_dc_step",      type=int,   default=3)
    p.add_argument("--weight_decay",    type=float, default=1e-5)
    p.add_argument("--clip",            type=float, default=5.0)
    p.add_argument("--patience",        type=int,   default=10)
    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--maxlen",          type=int,   default=0,
                   help="Truncate sessions to last N items (0=full)")
    args = p.parse_args()

    args.output_dir = os.path.expanduser(args.output_dir)
    train(args)
