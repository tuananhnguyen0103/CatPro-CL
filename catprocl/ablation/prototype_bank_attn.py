"""
prototype_bank_attn.py — Attention-Weighted Prototype Bank (A12)
-----------------------------------------------------------------
CORE-inspired: thay vì mean đều (CORE-ave), dùng dot-product attention
để weight items trong category trước khi cập nhật prototype.

Cụ thể:
    q     = proto[k]                        ← prototype hiện tại làm query
    α_i   = softmax(h_i · q / √d)          ← attention score
    update = Σ α_i * h_i                    ← weighted sum (thay vì mean)
    proto[k] ← momentum * proto[k] + (1-momentum) * update

Khi proto[k] chưa init (norm≈0): α_i = 1/n (uniform) → tương đương mean.
Sau khi init: items gần prototype được weight cao hơn → ổn định hơn, ít nhiễu hơn.

Không thêm parameters học được. Interface giống hệt PrototypeBank.
"""

import torch
import torch.nn.functional as F


class AttentionPrototypeBank:
    """
    Attention-weighted EMA prototype bank (A12).

    Args:
        n_cats   : số categories
        d        : embedding dimension
        momentum : EMA momentum (default 0.99)
        device   : cuda / cpu
    """

    def __init__(
        self,
        n_cats: int,
        d: int,
        momentum: float = 0.99,
        device: str = "cuda",
    ):
        self.n_cats   = n_cats
        self.d        = d
        self.momentum = momentum
        self.device   = device
        self.mode     = "attn"       # để train.py nhận biết

        # slot 0 unused; slots 1..n_cats là 1-indexed categories
        self.protos      = torch.zeros(n_cats + 1, d, device=device)
        self.initialized = False

    # ─── Attention-weighted mean ───────────────────────────────────────────────
    @staticmethod
    @torch.no_grad()
    def _attn_mean(h: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """
        Tính attention-weighted mean của h (N, d) với query (d,).
        Khi query ≈ 0 (chưa init): trả về mean thường.
        """
        if h.shape[0] == 1:
            return h.squeeze(0)

        q_norm = query.norm()
        if q_norm < 1e-6:          # chưa init → uniform weights
            return h.mean(dim=0)

        # dot-product attention: score = h · q / √d
        scores = (h @ query) / (h.shape[-1] ** 0.5)     # (N,)
        alpha  = torch.softmax(scores, dim=0)            # (N,)
        return (alpha.unsqueeze(1) * h).sum(dim=0)       # (d,)

    # ─── Warm start ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def warm_start(
        self,
        item_emb: torch.Tensor,
        item2cat: dict,
        cat2items_warm: dict,
    ) -> None:
        """
        Init prototype:
        Pass 1 → mean (query=0)
        Pass 2 → attention-weighted với mean làm query
        """
        for c in range(1, self.n_cats + 1):
            items = cat2items_warm.get(c, [])
            if not items:
                continue
            idx = torch.tensor(items, dtype=torch.long, device=self.device)
            idx = idx.clamp(0, item_emb.shape[0] - 1)
            h_k = item_emb[idx].detach()

            # Pass 1: mean làm query ban đầu
            proto_init = h_k.mean(dim=0)
            # Pass 2: attention với mean làm query
            self.protos[c] = self._attn_mean(h_k, proto_init)

        self.initialized = True

    # ─── EMA update sau mỗi batch ─────────────────────────────────────────────
    @torch.no_grad()
    def update(
        self,
        h_items: torch.Tensor,   # (M, d) warm item embeddings trong batch
        cat_ids: torch.Tensor,   # (M,)  category id tương ứng
    ) -> None:
        if h_items.shape[0] == 0:
            return

        for c in cat_ids.unique():
            c_idx = c.item()
            if c_idx < 1 or c_idx > self.n_cats:
                continue

            mask  = (cat_ids == c)
            h_k   = h_items[mask]                              # (N, d)
            query = self.protos[c_idx]                         # (d,)
            attn_mean = self._attn_mean(h_k, query)            # (d,)

            self.protos[c_idx] = (
                self.momentum * self.protos[c_idx]
                + (1.0 - self.momentum) * attn_mean
            )

    # ─── Helpers ──────────────────────────────────────────────────────────────
    def get(self, cat_idx) -> torch.Tensor:
        return self.protos[cat_idx]

    def to(self, device: str) -> "AttentionPrototypeBank":
        self.protos = self.protos.to(device)
        self.device = device
        return self

    def state_dict(self) -> dict:
        return {
            "protos":      self.protos,
            "initialized": self.initialized,
            "mode":        self.mode,
        }

    def load_state_dict(self, state: dict) -> None:
        self.protos      = state["protos"].to(self.device)
        self.initialized = state["initialized"]
