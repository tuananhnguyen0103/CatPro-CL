"""
losses_a9.py — Adaptive InfoNCE Loss cho Ablation A9
-----------------------------------------------------
Cải tiến so với losses.py (A4/A7):

  item_prototype_infonce_adaptive():
    - Dùng TẤT CẢ categories trong InfoNCE (giống A7), KHÔNG hard-gate sparse cats.
    - Lý do: Sparse categories vẫn cần InfoNCE signal dù noisy; loại bỏ hoàn toàn
      khiến prototype không được align với embedding space → cold-start tệ hơn.
    - Sự khác biệt với A7: prototype bank là AdaptivePrototypeBank với momentum
      thích nghi theo số warm items của category (sparse → update nhanh hơn).

  combined_loss_a9():
    - L = L_rec + lambda_proto * L_proto  (giống A7, khác prototype bank)

Ghi chú về support_mask():
  Vẫn có thể gọi proto_bank.support_mask() để LOG xem bao nhiêu category
  đủ support, nhưng KHÔNG dùng để filter trong loss.
"""

import torch
import torch.nn.functional as F

from catprocl.prototype_bank_v2 import AdaptivePrototypeBank
from catprocl.losses import rec_loss   # dùng lại rec_loss gốc


def item_prototype_infonce_adaptive(
    h_items: torch.Tensor,            # (M, d) — warm item embeddings trong batch
    cat_ids: torch.Tensor,            # (M,)   — category IDs (1-indexed)
    proto_bank: AdaptivePrototypeBank,
    tau: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE chuẩn — KHÔNG filter sparse categories.

    Tất cả warm items đều tham gia InfoNCE. Adaptive momentum ở prototype_bank_v2
    đảm bảo sparse categories có prototype được cập nhật nhanh hơn (momentum thấp hơn)
    thay vì bị loại hoàn toàn.

    Interface giống item_prototype_infonce() trong losses.py nhưng nhận
    AdaptivePrototypeBank thay vì PrototypeBank.
    """
    if h_items.shape[0] == 0:
        return torch.tensor(0.0, device=h_items.device, requires_grad=True)

    # Lọc item không có category hợp lệ (cat_id = 0 → dataset cũ hoặc item2cat thiếu)
    valid = cat_ids > 0
    h_items  = h_items[valid]
    cat_ids  = cat_ids[valid]
    if h_items.shape[0] == 0:
        return torch.tensor(0.0, device=h_items.device, requires_grad=True)

    h_norm = F.normalize(h_items, dim=1)                      # (M, d)
    P_norm = F.normalize(proto_bank.protos[1:], dim=1)        # (n_cats, d) — bỏ slot 0

    targets = cat_ids - 1                                     # shift sang 0-indexed

    sim  = (h_norm @ P_norm.T) / tau                          # (M, n_cats)
    loss = F.cross_entropy(sim, targets)
    return loss


def combined_loss_a9(
    z_s: torch.Tensor,
    item_emb: torch.Tensor,
    targets: torch.Tensor,
    h_warm: torch.Tensor,
    cat_warm: torch.Tensor,
    proto_bank: AdaptivePrototypeBank,
    lambda_proto: float = 0.1,
    tau: float = 0.07,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    L = L_rec + lambda_proto * L_proto_adaptive

    Identical với combined_loss() trong losses.py nhưng dùng adaptive prototype bank.
    Returns: (total_loss, l_rec, l_proto)
    """
    l_rec   = rec_loss(z_s, item_emb, targets)
    l_proto = item_prototype_infonce_adaptive(h_warm, cat_warm, proto_bank, tau)
    total   = l_rec + lambda_proto * l_proto
    return total, l_rec, l_proto
