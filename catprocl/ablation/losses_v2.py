"""
losses_v2.py — Loss functions cải tiến cho CatPro-CL v2
---------------------------------------------------------
Thay đổi so với losses.py:
  - item_prototype_infonce_hard_neg : InfoNCE với Hard Negative Mining
    Thay vì dùng tất cả prototypes làm negatives (dễ, gradient yếu),
    chỉ dùng top-n_hard_neg prototype gần nhất (khó nhất) làm negatives.
    → Prototype alignment chính xác hơn, categories gần nhau bị đẩy xa rõ hơn.

  - combined_loss_v2 : L = L_rec + effective_lambda * L_proto_hard
    effective_lambda được tính bên ngoài (warmup schedule trong train_v2.py)
    và truyền vào thay vì dùng lambda_proto cố định.
"""

import torch
import torch.nn.functional as F

from catprocl.prototype_bank import PrototypeBank
from catprocl.kmeans_bank import KMeansPrototypeBank
from catprocl.losses import rec_loss  # dùng lại rec_loss gốc, không thay đổi


# ─── Hard Negative InfoNCE ────────────────────────────────────────────────────
def item_prototype_infonce_hard_neg(
    h_items: torch.Tensor,                            # (M, d)
    cat_ids: torch.Tensor,                            # (M,)
    proto_bank: PrototypeBank | KMeansPrototypeBank,
    tau: float = 0.07,
    n_hard_neg: int = 20,                             # số hard negatives dùng
) -> torch.Tensor:
    """
    InfoNCE với Hard Negative Mining.

    Với mỗi item i thuộc category c:
      - Positive : prototype[c]
      - Negatives: top-n_hard_neg prototype có cosine similarity CAO NHẤT
                   với h_i (tức là khó phân biệt nhất)

    So với InfoNCE thường (dùng tất cả n_cats negatives):
      - Gradient mạnh hơn vì negatives thực sự khó
      - Không bị dominated bởi easy negatives (category xa)
      - Buộc prototype các category gần nhau phải tách biệt

    Returns: scalar loss
    """
    if h_items.shape[0] == 0:
        return torch.tensor(0.0, device=h_items.device, requires_grad=True)

    h_norm = F.normalize(h_items, dim=1)  # (M, d)

    if isinstance(proto_bank, KMeansPrototypeBank):
        P_norm  = F.normalize(proto_bank.protos, dim=1)  # (K, d)
        targets = cat_ids                                 # 0-indexed
    else:
        P_norm  = F.normalize(proto_bank.protos[1:], dim=1)  # (n_cats, d)
        targets = cat_ids - 1                                 # shift to 0-indexed

    n_cats = P_norm.shape[0]
    M      = h_norm.shape[0]

    # Fallback: nếu n_hard_neg >= n_cats-1 thì không có nghĩa gì, dùng standard
    if n_hard_neg >= n_cats - 1:
        sim  = (h_norm @ P_norm.T) / tau
        return F.cross_entropy(sim, targets)

    # ── Vectorized hard negative selection ───────────────────────────────────
    # sim_full: (M, n_cats) — cosine similarity giữa mọi item và mọi prototype
    sim_full = h_norm @ P_norm.T  # (M, n_cats)

    # Positive similarities: (M,)
    pos_sims = sim_full[torch.arange(M, device=h_items.device), targets]

    # Mask positive category ra để tìm hard negatives
    sim_neg = sim_full.clone()
    sim_neg[torch.arange(M, device=h_items.device), targets] = float("-inf")

    # Top-n_hard_neg highest similarity negatives: (M, n_hard_neg)
    topk_neg_sims, _ = sim_neg.topk(n_hard_neg, dim=1)

    # Build logits: [positive, neg_1, neg_2, ..., neg_K] — shape (M, 1+n_hard_neg)
    logits = torch.cat([pos_sims.unsqueeze(1), topk_neg_sims], dim=1) / tau

    # Target = 0 (positive luôn ở vị trí đầu tiên)
    target_labels = torch.zeros(M, dtype=torch.long, device=h_items.device)

    return F.cross_entropy(logits, target_labels)


# ─── Combined loss v2 ─────────────────────────────────────────────────────────
def combined_loss_v2(
    z_s: torch.Tensor,
    item_emb: torch.Tensor,
    targets: torch.Tensor,
    h_warm: torch.Tensor,
    cat_warm: torch.Tensor,
    proto_bank: PrototypeBank,
    effective_lambda: float,         # đã tính warmup bên ngoài
    tau: float = 0.07,
    n_hard_neg: int = 20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    L = L_rec + effective_lambda * L_proto_hard

    effective_lambda = 0 trong warmup epochs → chỉ train SR-GNN
    effective_lambda tăng dần sau warmup → thêm dần InfoNCE hard neg

    Returns: (total_loss, l_rec, l_proto)
    """
    l_rec = rec_loss(z_s, item_emb, targets)

    if effective_lambda == 0.0 or h_warm.shape[0] == 0:
        l_proto = torch.tensor(0.0, device=z_s.device)
        return l_rec, l_rec, l_proto

    l_proto = item_prototype_infonce_hard_neg(
        h_warm, cat_warm, proto_bank, tau=tau, n_hard_neg=n_hard_neg
    )
    total = l_rec + effective_lambda * l_proto
    return total, l_rec, l_proto
