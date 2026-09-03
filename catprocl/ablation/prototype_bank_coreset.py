"""
prototype_bank_coreset.py — Coreset EMA Prototype Bank (Ablation A10)
----------------------------------------------------------------------
Giống PrototypeBank(mode="ema") nhưng mỗi batch update chỉ dùng
top-K item có discriminative margin score cao nhất cho mỗi category.

Discriminative margin score (dùng prototype hiện tại làm centroid):
    score(i, k) = cos(h_i, b_k) − max_{k'≠k} cos(h_i, b_{k'})

- score cao  → item gần tâm category mình (giữ lại)
- score thấp → item bị kéo sang category khác (loại bỏ)

Khi chưa warm_start (protos = 0):
    Fallback sang EMA thuần — dùng tất cả items trong batch.
    (Tránh scoring vô nghĩa khi prototype vẫn là zero vector.)

Sau warm_start():
    - n_c > top_k : chọn top-K item, update EMA
    - n_c ≤ top_k : dùng tất cả (không cần lọc)

Interface hoàn toàn tương thích với PrototypeBank để train.py dùng
được mà không cần đổi code evaluation / cold_inference.
"""

import torch
import torch.nn.functional as F


class CoresetPrototypeBank:
    """
    Args:
        n_cats   : số category (prototypes 1-indexed: slot 1..n_cats, slot 0 unused)
        d        : embedding dimension
        momentum : EMA momentum (default 0.99)
        device   : "cuda" hoặc "cpu"
        top_k    : số item tối đa per category dùng cho mỗi EMA update (default 5)
    """

    def __init__(
        self,
        n_cats: int,
        d: int,
        momentum: float = 0.99,
        device: str = "cuda",
        top_k: int = 5,
    ):
        self.n_cats   = n_cats
        self.d        = d
        self.momentum = momentum
        self.device   = device
        self.top_k    = top_k

        # Prototype vectors — size n_cats+1; slot 0 = padding (unused)
        self.protos = torch.zeros(n_cats + 1, d, device=device)
        self.initialized = False

    # ── Warm start ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def warm_start(
        self,
        item_emb: torch.Tensor,          # (n_items+1, d)
        item2cat: dict,                  # {item_id: cat_id}
        cat2items_warm: dict,            # {cat_id: [warm_item_ids]}
    ) -> None:
        """
        Khởi tạo prototype = mean embedding của warm items.
        Gọi 1 lần sau epoch 1 (giống PrototypeBank).
        """
        for c in range(1, self.n_cats + 1):
            items = cat2items_warm.get(c, [])
            if not items:
                continue
            idx = torch.tensor(items, dtype=torch.long, device=self.device)
            idx = idx.clamp(0, item_emb.shape[0] - 1)
            self.protos[c] = item_emb[idx].mean(dim=0).detach()
        self.initialized = True

    # ── Update ────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def update(
        self,
        h_items: torch.Tensor,   # (M, d) — warm item embeddings trong batch
        cat_ids: torch.Tensor,   # (M,)   — category ID (1-indexed)
    ) -> None:
        """
        Chọn top-K items theo discriminative margin, rồi EMA update.

        Khi chưa initialized: fallback sang EMA thuần (không lọc).
        """
        if h_items.shape[0] == 0:
            return

        if not self.initialized:
            # ── Fallback: standard EMA (trước warm_start) ────────────────────
            for c in cat_ids.unique():
                c_idx = c.item()
                if c_idx < 1 or c_idx > self.n_cats:
                    continue
                mask  = (cat_ids == c)
                mean_c = h_items[mask].mean(dim=0)
                self.protos[c_idx] = (
                    self.momentum * self.protos[c_idx]
                    + (1.0 - self.momentum) * mean_c
                )
            return

        # ── Coreset selection (sau warm_start) ────────────────────────────────
        # Pre-normalize một lần cho cả batch
        h_norm = F.normalize(h_items, dim=1)      # (M, d)
        P_norm = F.normalize(self.protos, dim=1)  # (n_cats+1, d)

        for c in cat_ids.unique():
            c_idx = c.item()
            if c_idx < 1 or c_idx > self.n_cats:
                continue

            mask     = (cat_ids == c)
            h_c      = h_items[mask]      # (n_c, d)
            h_c_norm = h_norm[mask]       # (n_c, d)
            n_c      = h_c.shape[0]

            if n_c > self.top_k:
                # ── Discriminative margin ─────────────────────────────────────
                # sim_own : cos(h_i, b_k)
                sim_own = (h_c_norm * P_norm[c_idx]).sum(dim=1)   # (n_c,)

                # sim_all : cos(h_i, b_{k'}) cho mọi k'
                sim_all = h_c_norm @ P_norm.T                      # (n_c, n_cats+1)
                sim_all[:, c_idx] = float('-inf')                  # loại own category
                sim_all[:, 0]     = float('-inf')                  # loại padding slot
                sim_max_other = sim_all.max(dim=1).values          # (n_c,)

                # margin = gần tâm mình - gần tâm khác (cao = tốt)
                margin   = sim_own - sim_max_other                 # (n_c,)
                topk_idx = margin.topk(self.top_k).indices
                h_selected = h_c[topk_idx]
            else:
                # n_c ≤ top_k: dùng tất cả
                h_selected = h_c

            mean_c = h_selected.mean(dim=0)
            self.protos[c_idx] = (
                self.momentum * self.protos[c_idx]
                + (1.0 - self.momentum) * mean_c
            )

    # ── Accessors ─────────────────────────────────────────────────────────────
    def get(self, cat_idx) -> torch.Tensor:
        """Lấy prototype cho 1 category (int) hoặc nhiều (Tensor)."""
        return self.protos[cat_idx]

    def to(self, device: str) -> "CoresetPrototypeBank":
        self.protos = self.protos.to(device)
        self.device = device
        return self

    def state_dict(self) -> dict:
        return {
            "protos":      self.protos,
            "initialized": self.initialized,
            "top_k":       self.top_k,
            "momentum":    self.momentum,
        }

    def load_state_dict(self, state: dict) -> None:
        self.protos      = state["protos"].to(self.device)
        self.initialized = state["initialized"]
        if "top_k" in state:
            self.top_k = state["top_k"]
        if "momentum" in state:
            self.momentum = state["momentum"]
