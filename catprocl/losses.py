"""
losses.py — Loss functions cho CatPro-CL
-----------------------------------------
item_prototype_infonce : Item-Prototype InfoNCE loss (Ablation A3, A4, A5)
combined_loss          : L = L_rec + lambda * L_proto
"""

import torch
import torch.nn.functional as F

from catprocl.prototype_bank import PrototypeBank
from catprocl.kmeans_bank import KMeansPrototypeBank


# ─── Recommendation loss (cross-entropy trên full item set) ───────────────────
def rec_loss(
    z_s: torch.Tensor,        # (B, d) session embeddings
    item_emb: torch.Tensor,   # (n_items+1, d) item embedding matrix
    targets: torch.Tensor,    # (B,) target item IDs (1-indexed)
) -> torch.Tensor:
    """
    L_rec = CrossEntropy(z_s @ item_emb.T, targets)
    Loại bỏ row 0 (padding) khỏi scoring.
    """
    # Score: (B, n_items+1) — bỏ dim 0 (padding)
    logits = z_s @ item_emb.T  # (B, n_items+1)
    return F.cross_entropy(logits, targets)


# ─── Item-Prototype InfoNCE ────────────────────────────────────────────────────
def item_prototype_infonce(
    h_items: torch.Tensor,                            # (M, d) item embeddings (warm only)
    cat_ids: torch.Tensor,                            # (M,)  category/cluster IDs
    proto_bank: PrototypeBank | KMeansPrototypeBank,  # prototype bank
    tau: float = 0.07,                                # temperature
) -> torch.Tensor:
    """
    Kéo h_i gần prototype cùng category, đẩy xa prototype category khác.

    Xử lý 2 loại bank:
      PrototypeBank  : protos là (n_cats+1, d), cat_ids 1-indexed (1..n_cats)
                       → slice protos[1:], shift cat_ids - 1 cho cross_entropy
      KMeansProtoBk  : protos là (n_clusters, d), cluster_ids 0-indexed (0..K-1)
                       → dùng protos trực tiếp, không shift

    Nếu M=0 (không có warm item trong batch), trả về 0.
    """
    if h_items.shape[0] == 0:
        return torch.tensor(0.0, device=h_items.device, requires_grad=True)

    # Lọc item không có category hợp lệ (cat_id = 0 → item2cat thiếu hoặc dataset cũ)
    if not isinstance(proto_bank, KMeansPrototypeBank):
        valid = cat_ids > 0
        h_items  = h_items[valid]
        cat_ids  = cat_ids[valid]
        if h_items.shape[0] == 0:
            return torch.tensor(0.0, device=h_items.device, requires_grad=True)

    h_norm = F.normalize(h_items, dim=1)   # (M, d)

    if isinstance(proto_bank, KMeansPrototypeBank):
        # K-means: 0-indexed clusters, protos shape (K, d)
        P_norm  = F.normalize(proto_bank.protos, dim=1)   # (K, d)
        targets = cat_ids                                  # 0-indexed
    else:
        # PrototypeBank: 1-indexed cats, protos shape (n_cats+1, d), slot 0 unused
        P_norm  = F.normalize(proto_bank.protos[1:], dim=1)  # (n_cats, d)
        targets = cat_ids - 1                                 # shift to 0-indexed

    sim  = (h_norm @ P_norm.T) / tau        # (M, n_cats or K)
    loss = F.cross_entropy(sim, targets)
    return loss


# ─── Session-Prototype NCE (A13) ──────────────────────────────────────────────
def session_prototype_nce(
    z_s: torch.Tensor,             # (B, d) session embeddings
    target_cats: torch.Tensor,     # (B,) category IDs của targets (1-indexed)
    proto_bank,                    # PrototypeBank với protos (n_cats+1, d)
    tau: float = 0.1,              # temperature (riêng cho prototype NCE)
) -> torch.Tensor:
    """
    L_proto_nce: session embedding phải gần prototype của đúng category target.
    NCE over n_cats (~941) — dạy model "biết" category prototype tại train time
    → khi test cold item = prototype, model đã quen → Cold HR tăng.

    Positive: proto_bank.protos[cat(target)]
    Negatives: tất cả prototype còn lại (941 total — manageable)

    Chỉ gọi sau khi proto_bank.initialized = True (sau warm_start epoch 1).
    """
    valid = target_cats > 0
    if not valid.any():
        return torch.tensor(0.0, device=z_s.device, requires_grad=True)

    z_v    = z_s[valid]              # (B', d)
    cats_v = target_cats[valid]      # (B',) 1-indexed

    z_norm = F.normalize(z_v, dim=1)                     # (B', d)
    P_norm = F.normalize(proto_bank.protos[1:], dim=1)   # (n_cats, d) — bỏ slot 0

    sim    = (z_norm @ P_norm.T) / tau    # (B', n_cats)
    labels = cats_v - 1                  # 0-indexed

    return F.cross_entropy(sim, labels)


# ─── Combined loss ─────────────────────────────────────────────────────────────
def combined_loss(
    z_s: torch.Tensor,
    item_emb: torch.Tensor,
    targets: torch.Tensor,
    h_warm: torch.Tensor,
    cat_warm: torch.Tensor,
    proto_bank: PrototypeBank,
    lambda_proto: float = 0.1,
    tau: float = 0.07,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    L = L_rec + lambda_proto * L_proto

    Returns: (total_loss, l_rec, l_proto)
    """
    l_rec   = rec_loss(z_s, item_emb, targets)
    l_proto = item_prototype_infonce(h_warm, cat_warm, proto_bank, tau)
    total   = l_rec + lambda_proto * l_proto
    return total, l_rec, l_proto
