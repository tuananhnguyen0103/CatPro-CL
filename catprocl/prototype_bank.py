"""
prototype_bank.py — Category Prototype Bank (EMA hoặc Fixed)
-------------------------------------------------------------
Duy trì 1 prototype vector p_c cho mỗi category c.

mode="ema"   (A3, A4, A7): cập nhật EMA sau mỗi batch.
    p_c ← momentum * p_c + (1 - momentum) * mean({h_i : cat(i)=c})
mode="fixed" (A5):         tính mean toàn epoch, gán 1 lần cuối epoch.
    Gọi finalize_epoch() sau khi kết thúc epoch.

NOT nn.Parameter → không tham gia backprop.
"""

import torch


class PrototypeBank:
    """
    Category prototype bank.

    Args:
        n_cats    : số lượng category
        d         : embedding dimension
        momentum  : EMA momentum (chỉ dùng khi mode="ema", default 0.99)
        device    : cuda hoặc cpu
        mode      : "ema" (EMA per-batch) hoặc "fixed" (epoch mean, A5)
    """

    def __init__(
        self,
        n_cats: int,
        d: int,
        momentum: float = 0.99,
        device: str = "cuda",
        mode: str = "ema",      # "ema" | "fixed"
    ):
        self.n_cats   = n_cats
        self.d        = d
        self.momentum = momentum
        self.device   = device
        self.mode     = mode

        # Prototype vectors — size n_cats+1: slot 0 unused, slots 1..n_cats = 1-indexed cats
        self.protos = torch.zeros(n_cats + 1, d, device=device)
        self.initialized = False

        # Fixed-mode accumulators (chỉ dùng khi mode="fixed")
        if mode == "fixed":
            self._acc_sum   = torch.zeros(n_cats + 1, d, device=device)
            self._acc_count = torch.zeros(n_cats + 1,    device=device)

    # ── Warm start: tính mean embedding của mỗi category từ warm items ────────
    @torch.no_grad()
    def warm_start(
        self,
        item_emb: torch.Tensor,      # (n_items+1, d) — từ model.all_item_embeddings()
        item2cat: dict[int, int],     # {item_id: cat_id}
        cat2items_warm: dict[int, list[int]],  # {cat_id: [warm_item_ids]}
    ) -> None:
        """
        Gán prototype ban đầu = mean embedding của warm items trong category.
        Gọi 1 lần sau epoch 0 hoặc sau warm_start của training.
        """
        # cat_ids are 1-indexed (1..n_cats); slot 0 stays zero (unused)
        for c in range(1, self.n_cats + 1):
            items = cat2items_warm.get(c, [])
            if not items:
                continue
            idx = torch.tensor(items, dtype=torch.long, device=self.device)
            idx = idx.clamp(0, item_emb.shape[0] - 1)
            self.protos[c] = item_emb[idx].mean(dim=0).detach()

        self.initialized = True

    # ── Update — gọi SAU optimizer.step() ─────────────────────────────────────
    @torch.no_grad()
    def update(
        self,
        h_items: torch.Tensor,   # (M, d) — embeddings của warm items trong batch
        cat_ids: torch.Tensor,   # (M,)   — category ID của từng item
    ) -> None:
        """
        mode="ema"  : cập nhật EMA prototype ngay sau mỗi batch.
        mode="fixed": tích lũy sum/count; gọi finalize_epoch() cuối epoch.
        """
        if h_items.shape[0] == 0:
            return

        for c in cat_ids.unique():
            c_idx = c.item()
            if c_idx < 1 or c_idx > self.n_cats:   # 1-indexed: valid range [1, n_cats]
                continue
            mask   = (cat_ids == c)
            mean_c = h_items[mask].mean(dim=0)

            if self.mode == "ema":
                self.protos[c_idx] = (
                    self.momentum * self.protos[c_idx]
                    + (1.0 - self.momentum) * mean_c
                )
            else:  # fixed
                count = mask.sum().float()
                self._acc_sum[c_idx]   += h_items[mask].sum(dim=0)
                self._acc_count[c_idx] += count

    # ── Fixed mode: gọi 1 lần cuối epoch ─────────────────────────────────────
    @torch.no_grad()
    def finalize_epoch(self) -> None:
        """
        Chỉ dùng cho mode="fixed" (A5).
        Tính mean toàn epoch → gán vào protos, rồi reset accumulators.
        """
        if self.mode != "fixed":
            return
        valid = self._acc_count > 0
        self.protos[valid] = (
            self._acc_sum[valid] / self._acc_count[valid].unsqueeze(1)
        )
        # Reset
        self._acc_sum.zero_()
        self._acc_count.zero_()

    # ── Get prototype(s) ──────────────────────────────────────────────────────
    def get(self, cat_idx: int | torch.Tensor) -> torch.Tensor:
        """
        Lấy prototype cho 1 category (int) hoặc nhiều categories (Tensor).
        Returns: (d,) hoặc (M, d)
        """
        if isinstance(cat_idx, int):
            return self.protos[cat_idx]
        return self.protos[cat_idx]

    def to(self, device: str) -> "PrototypeBank":
        self.protos = self.protos.to(device)
        self.device = device
        return self

    def state_dict(self) -> dict:
        d = {"protos": self.protos, "initialized": self.initialized, "mode": self.mode}
        if self.mode == "fixed":
            d["_acc_sum"]   = self._acc_sum
            d["_acc_count"] = self._acc_count
        return d

    def load_state_dict(self, state: dict) -> None:
        self.protos      = state["protos"].to(self.device)
        self.initialized = state["initialized"]
        if self.mode == "fixed" and "_acc_sum" in state:
            self._acc_sum   = state["_acc_sum"].to(self.device)
            self._acc_count = state["_acc_count"].to(self.device)
