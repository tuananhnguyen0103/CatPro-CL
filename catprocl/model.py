"""
model.py — SR-GNN Backbone
--------------------------
GatedSessionGNN  : GGNN propagation trên session graph (A_in, A_out)
SRGNNEncoder     : Soft-attention readout → session embedding z_s
                   Dùng chung cho Ablation A1 → A5 (không thay đổi)

Reference: Wu et al., "Session-based Recommendation with Graph Neural Networks", AAAI 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Gated Graph Neural Network (1 propagation step) ──────────────────────────
class GatedSessionGNN(nn.Module):
    """
    Propagate node embeddings on the session graph via gated update.
    Follows SR-GNN Eq. (1)-(3).

    A_in  : normalized in-adjacency  (B, N, N)
    A_out : normalized out-adjacency (B, N, N)
    """

    def __init__(self, d: int, n_steps: int = 1):
        super().__init__()
        self.n_steps = n_steps

        # Input transformation (concat in+out neighbor aggregation)
        self.W_in  = nn.Linear(d, d, bias=True)
        self.W_out = nn.Linear(d, d, bias=True)

        # GRU-style gated update
        self.gru = nn.GRUCell(2 * d, d)

    def forward(
        self,
        x: torch.Tensor,      # (B, N, d) — initial node embeddings
        A_in: torch.Tensor,   # (B, N, N)
        A_out: torch.Tensor,  # (B, N, N)
    ) -> torch.Tensor:        # (B, N, d) — updated node embeddings

        B, N, d = x.shape
        h = x

        for _ in range(self.n_steps):
            # Aggregate neighbors
            agg_in  = torch.bmm(A_in,  h)   # (B, N, d)
            agg_out = torch.bmm(A_out, h)   # (B, N, d)

            # Linear transformation
            a_in  = self.W_in(agg_in)        # (B, N, d)
            a_out = self.W_out(agg_out)      # (B, N, d)
            a     = torch.cat([a_in, a_out], dim=-1)  # (B, N, 2d)

            # GRU update (process all nodes in batch at once)
            h_flat = h.view(B * N, d)
            a_flat = a.view(B * N, 2 * d)
            h_flat = self.gru(a_flat, h_flat)
            h = h_flat.view(B, N, d)

        return h  # (B, N, d)


# ─── SR-GNN Encoder ───────────────────────────────────────────────────────────
class SRGNNEncoder(nn.Module):
    """
    Full SR-GNN: embedding lookup → GNN propagation → soft-attention readout.

    Returns z_s (session embedding) shaped (B, d).
    Call all_item_embeddings() to get the item matrix for scoring.

    Ablation usage:
      A1: use as-is, score = z_s @ item_emb.T
      A2-A5: same encoder; prototype bank and CL loss are external modules
    """

    def __init__(self, n_items: int, d: int = 128, n_steps: int = 1):
        super().__init__()
        self.d = d

        # Item embedding table (0 = padding, excluded from grad via padding_idx)
        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)

        self.gnn = GatedSessionGNN(d, n_steps)

        # Soft-attention weights (SR-GNN Eq. 6)
        self.W1 = nn.Linear(d, d, bias=False)
        self.W2 = nn.Linear(d, d, bias=False)
        self.q  = nn.Linear(d, 1, bias=False)

        # Final session representation (concat global + local)
        self.W3 = nn.Linear(2 * d, d, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_emb.weight, std=0.1)
        for name, p in self.named_parameters():
            if "item_emb" in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch: dict from SessionGraphCollator with keys:
                node_ids : (B, N) int64 — global item IDs (0 = pad)
                A_in     : (B, N, N) float32
                A_out    : (B, N, N) float32
                seq_idx  : (B, L) int64 — local indices into node_ids (0 = pad)
                mask     : (B, L) float32 — 1 = valid, 0 = pad

        Returns:
            z_s      : (B, d) session embeddings
            h_nodes  : (B, N, d) node embeddings after GNN (for prototype update)
        """
        node_ids = batch["node_ids"]   # (B, N)
        A_in     = batch["A_in"]       # (B, N, N)
        A_out    = batch["A_out"]      # (B, N, N)
        seq_idx  = batch["seq_idx"]    # (B, L)
        mask     = batch["mask"]       # (B, L)

        B, N = node_ids.shape
        L    = seq_idx.shape[1]

        # 1. Embedding lookup for each node
        x = self.item_emb(node_ids)  # (B, N, d)

        # 2. GNN propagation
        h_nodes = self.gnn(x, A_in, A_out)  # (B, N, d)

        # 3. Gather sequence embeddings using local indices
        #    seq_idx: (B, L) → expand to (B, L, d)
        seq_idx_exp = seq_idx.unsqueeze(-1).expand(B, L, self.d)  # (B, L, d)
        seq_h = torch.gather(h_nodes, 1, seq_idx_exp)             # (B, L, d)

        # 4. Local representation: last item in sequence
        #    Find actual last position using mask
        lengths = mask.sum(dim=1).long()                          # (B,)
        last_idx = (lengths - 1).clamp(min=0)                    # (B,)
        last_idx_exp = last_idx.view(B, 1, 1).expand(B, 1, self.d)
        h_last = seq_h.gather(1, last_idx_exp).squeeze(1)        # (B, d)

        # 5. Global representation: soft-attention over sequence
        #    alpha_i = q * sigmoid(W1 * h_i + W2 * h_last)
        alpha = self.q(
            torch.sigmoid(
                self.W1(seq_h) + self.W2(h_last).unsqueeze(1)  # (B, L, d)
            )
        )  # (B, L, 1)
        alpha = alpha * mask.unsqueeze(-1)  # zero-out padding positions
        s_g = (alpha * seq_h).sum(dim=1)   # (B, d) — global repr

        # 6. Final session embedding
        z_s = self.W3(torch.cat([s_g, h_last], dim=-1))  # (B, d)

        return z_s, h_nodes

    def all_item_embeddings(self) -> torch.Tensor:
        """Return the full item embedding matrix (n_items+1, d). Row 0 = padding."""
        return self.item_emb.weight  # (n_items+1, d)
