"""
cold_inference.py — Cold Item Embedding Replacement at Eval Time
-----------------------------------------------------------------
Hai chiến lược cold inference (so sánh A7 vs A8):

  A7 — build_eval_item_embeddings():
    h_cold = proto_bank[cat(i)]     (EMA prototype, đã smoothed)

  A8 — build_eval_item_embeddings_ecat():
    h_cold = mean(item_emb[warm_items_in_cat(i)])   (raw category mean,
                                                      tính tươi từ model)
"""

import torch

from catprocl.prototype_bank import PrototypeBank


@torch.no_grad()
def build_eval_item_embeddings(
    item_emb: torch.Tensor,           # (n_items+1, d) — từ model.all_item_embeddings()
    proto_bank: PrototypeBank,
    item2cat: dict[int, int],
    cold_items: set[int],
    perturbation_std: float = 0.0,    # >0 để tránh tie (default: 0 = tắt)
) -> torch.Tensor:
    """
    Trả về bản sao item_emb với cold rows được thay bằng prototype.

    perturbation_std: thêm noise N(0, std) vào prototype để phân biệt
                      các cold items cùng category khi ranking.
                      Paper chưa quy định → để 0.0 (pure prototype) làm default.
                      Nếu Cold HR = 0, thử perturbation_std = 0.01.
    """
    emb = item_emb.clone()  # không sửa in-place vào model parameters

    for item_id in cold_items:
        cat_id = item2cat.get(item_id)
        if cat_id is None:
            continue  # item không có category → giữ nguyên embedding gốc
        proto = proto_bank.get(cat_id).clone()
        if perturbation_std > 0.0:
            proto = proto + torch.randn_like(proto) * perturbation_std
        emb[item_id] = proto

    return emb  # (n_items+1, d)


@torch.no_grad()
def build_eval_item_embeddings_ecat(
    item_emb: torch.Tensor,           # (n_items+1, d) — từ model.all_item_embeddings()
    cat2items_warm: dict[int, list[int]],  # {cat_id: [warm_item_ids]}
    item2cat: dict[int, int],
    cold_items: set[int],
    perturbation_std: float = 0.0,
) -> torch.Tensor:
    """
    A8: thay embedding cold item bằng mean hiện tại của warm items cùng category.
    Không dùng EMA prototype — đây là "raw category mean" e_cat.

    So sánh với A7 (EMA prototype):
      A7 → protos[c] = EMA-smoothed trung bình nhiều epoch (stable)
      A8 → mean(item_emb[warm_items_in_c]) = snapshot tức thời (noisy hơn)

    Mục đích: kiểm tra xem EMA smoothing có cần thiết không.
    """
    emb = item_emb.clone()

    # Pre-compute category mean từ current item embeddings
    cat_means: dict[int, torch.Tensor] = {}
    for cat_id, warm_ids in cat2items_warm.items():
        if not warm_ids:
            continue
        idx = torch.tensor(warm_ids, dtype=torch.long, device=item_emb.device)
        idx = idx.clamp(0, item_emb.shape[0] - 1)
        cat_means[cat_id] = item_emb[idx].mean(dim=0).detach()

    for item_id in cold_items:
        cat_id = item2cat.get(item_id)
        if cat_id is None or cat_id not in cat_means:
            continue
        e_cat = cat_means[cat_id].clone()
        if perturbation_std > 0.0:
            e_cat = e_cat + torch.randn_like(e_cat) * perturbation_std
        emb[item_id] = e_cat

    return emb  # (n_items+1, d)
