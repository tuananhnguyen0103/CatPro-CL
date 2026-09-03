"""
prototype_bank_v2.py — Adaptive Category Prototype Bank (Ablation A9)
----------------------------------------------------------------------
Cải tiến so với prototype_bank.py (dùng cho A3/A4/A7):

1. Adaptive EMA momentum
   m_c = clip(1 - 1 / max(n_warm_global[c], 1), min_momentum, max_momentum)
   - Category thưa (n_warm=1)  → m=0.50 (update nhanh, bám signal ngay)
   - Category vừa  (n_warm=10) → m=0.90
   - Category dày  (n_warm=100)→ m=0.99 (ổn định, giống A7)
   Thay vì momentum cố định 0.99 cho tất cả — A7 bị stale với sparse categories.

2. Support count tracking
   n_warm_global[c] = tổng warm items của category c đã seen trong training.
   Dùng để: (a) tính adaptive momentum, (b) lọc support_mask trong loss A9.

3. Duck-typing với PrototypeBank
   Interface giống hệt: .protos, .get(), .update(), .warm_start(),
   .state_dict(), .load_state_dict() — tương thích với cold_inference.py.
"""

import torch


class AdaptivePrototypeBank:
    """
    Adaptive EMA prototype bank — Ablation A9.

    Args:
        n_cats        : số category (1-indexed, slot 0 unused)
        d             : embedding dimension
        device        : cuda | cpu
        min_momentum  : momentum tối thiểu khi category rất thưa (default 0.50)
        max_momentum  : momentum tối đa khi category dày đặc  (default 0.99)
        min_support   : ngưỡng để category được tính là "reliable" cho InfoNCE
    """

    def __init__(
        self,
        n_cats: int,
        d: int,
        device: str = "cuda",
        min_momentum: float = 0.50,
        max_momentum: float = 0.99,
        min_support: int = 10,
    ):
        self.n_cats        = n_cats
        self.d             = d
        self.device        = device
        self.min_momentum  = min_momentum
        self.max_momentum  = max_momentum
        self.min_support   = min_support
        self.mode          = "ema"   # duck-type compat với PrototypeBank.mode
        self.initialized   = False

        # Prototype vectors: (n_cats+1, d), slot 0 unused (padding category)
        self.protos = torch.zeros(n_cats + 1, d, device=device)

        # n_warm_global[c] = số warm items của cat c đã thấy qua toàn bộ training
        self.n_warm_global = torch.zeros(n_cats + 1, dtype=torch.long, device=device)

    # ── Adaptive momentum cho category c ─────────────────────────────────────
    def _momentum_for(self, c_idx: int) -> float:
        """
        m_c = clip(1 - 1/n_warm, min_momentum, max_momentum)
        n_warm=1  → m=0.00 → clip → 0.50
        n_warm=10 → m=0.90
        n_warm=50 → m=0.98
        n_warm=100→ m=0.99 = max_momentum
        """
        n = max(int(self.n_warm_global[c_idx].item()), 1)
        m = 1.0 - 1.0 / n
        return max(self.min_momentum, min(self.max_momentum, m))

    # ── Warm start ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def warm_start(
        self,
        item_emb: torch.Tensor,           # (n_items+1, d)
        item2cat: dict,                   # {item_id: cat_id}
        cat2items_warm: dict,             # {cat_id: [warm_item_ids]}
    ) -> None:
        """
        Khởi tạo prototype = mean embedding của warm items trong category.
        Đồng thời seed n_warm_global từ số lượng warm items thực tế.
        """
        for c in range(1, self.n_cats + 1):
            items = cat2items_warm.get(c, [])
            if not items:
                continue
            idx = torch.tensor(items, dtype=torch.long, device=self.device)
            idx = idx.clamp(0, item_emb.shape[0] - 1)
            self.protos[c]        = item_emb[idx].mean(dim=0).detach()
            self.n_warm_global[c] = len(items)   # seed count từ dataset
        self.initialized = True

    # ── EMA update với adaptive momentum ──────────────────────────────────────
    @torch.no_grad()
    def update(
        self,
        h_items: torch.Tensor,   # (M, d) — embeddings warm items trong batch
        cat_ids: torch.Tensor,   # (M,)   — category ID (1-indexed)
    ) -> None:
        if h_items.shape[0] == 0:
            return

        for c in cat_ids.unique():
            c_idx = c.item()
            if c_idx < 1 or c_idx > self.n_cats:
                continue
            mask   = (cat_ids == c)
            mean_c = h_items[mask].mean(dim=0)

            # Tích lũy count trước khi tính momentum
            self.n_warm_global[c_idx] += mask.sum().long()

            # Adaptive momentum (dựa trên count MỚI sau khi cộng)
            m = self._momentum_for(c_idx)
            self.protos[c_idx] = m * self.protos[c_idx] + (1.0 - m) * mean_c

    # ── Support mask ──────────────────────────────────────────────────────────
    def support_mask(self, min_support: int | None = None) -> torch.Tensor:
        """
        Returns boolean tensor (n_cats+1,):
          True  nếu n_warm_global[c] >= min_support  → reliable, dùng InfoNCE
          False nếu không đủ → bỏ qua trong loss

        slot 0 luôn False (padding category).
        """
        thresh = min_support if min_support is not None else self.min_support
        mask   = self.n_warm_global >= thresh
        mask[0] = False   # slot 0 luôn invalid
        return mask

    # ── Get prototype(s) ──────────────────────────────────────────────────────
    def get(self, cat_idx) -> torch.Tensor:
        return self.protos[cat_idx]

    def to(self, device: str) -> "AdaptivePrototypeBank":
        self.protos        = self.protos.to(device)
        self.n_warm_global = self.n_warm_global.to(device)
        self.device        = device
        return self

    def state_dict(self) -> dict:
        return {
            "protos":        self.protos,
            "n_warm_global": self.n_warm_global,
            "initialized":   self.initialized,
        }

    def load_state_dict(self, state: dict) -> None:
        self.protos        = state["protos"].to(self.device)
        self.n_warm_global = state.get(
            "n_warm_global",
            torch.zeros(self.n_cats + 1, dtype=torch.long, device=self.device),
        ).to(self.device)
        self.initialized   = state["initialized"]
