"""
train.py — Training entrypoint for CatPro-CL
=============================================
Paper: "Category-Prototype Alignment for Cold-Start Session-Based
        Recommendation via Contrastive Learning" (IEEE Access, 2026)

Quick start (from the repo root):
    python catprocl/train.py --config configs/catprocl_retailrocket_a11.yaml --ablation A11 --seed 42

Override any config key on the command line:
    python catprocl/train.py --config configs/catprocl_diginetica_a11.yaml \\
        --ablation A11 --seed 0 --maxlen 4 \\
        --lambda_proto 0.5 --mask_prob 0.05 --temperature 0.1

Ablation variants:
    A1  : SR-GNN backbone only (no prototype bank, no contrastive loss)
    A3  : + EMA prototype bank (updated but not used in any loss)
    A4  : + EMA prototype bank + item-prototype InfoNCE loss (no cold inference)
    A5  : + Fixed prototype (epoch mean, no EMA) + InfoNCE loss
    A6  : + K-means prototype (K = n_cats) + InfoNCE loss
    A7  : A4 + cold inference at eval time (h_cold = category prototype)
    A8  : cold inference using raw category mean embedding
    A9  : Adaptive EMA prototype bank (support-aware momentum)
    A10 : Coreset-based prototype bank
    A11 : A7 + Prototype Simulation Masking (PSM) ← FULL MODEL (paper default)
    A12 : Attention-weighted prototype bank + PSM
    A13 : A11 + session-prototype NCE auxiliary loss
    A14 : Session-EMA prototype (proto updated from session embeddings)
    A15 : Additive masked reconstruction loss variant

Full-model hyperparameters (A11): lambda_proto=0.5, mask_prob=0.05, temperature=0.1
All hyperparameters are read from the YAML config; no need to edit this file.
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

# Add repo root to sys.path so that `catprocl.*` and `evaluation.*` are importable
# when running from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catprocl.data_loader import get_dataloaders
from catprocl.model import SRGNNEncoder
from catprocl.prototype_bank import PrototypeBank
from catprocl.losses import rec_loss, item_prototype_infonce, combined_loss, session_prototype_nce
# Ablation-specific modules (A6/A9/A10/A12/A13/A15):
from catprocl.ablation.prototype_bank_v2 import AdaptivePrototypeBank
from catprocl.ablation.prototype_bank_coreset import CoresetPrototypeBank
from catprocl.ablation.prototype_bank_attn import AttentionPrototypeBank
from catprocl.ablation.kmeans_bank import KMeansPrototypeBank
from catprocl.ablation.losses_a9 import combined_loss_a9
from catprocl.cold_inference import (
    build_eval_item_embeddings,
    build_eval_item_embeddings_ecat,
)
from evaluation.evaluator import evaluate, format_results


# ─── Seed ─────────────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─── Load config ──────────────────────────────────────────────────────────────
def load_config(config_path: str, overrides: dict) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # Command-line overrides (flat key=value)
    for k, v in overrides.items():
        if v is not None:
            _set_nested(cfg, k, v)
    return cfg


def _set_nested(cfg: dict, key: str, value) -> None:
    """Set cfg[a][b][c] = value given key='a.b.c' or 'c' (top-level)."""
    parts = key.split(".")
    d = cfg
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


# ─── Score function builder (cho evaluator — không phụ thuộc model type) ─────
def make_score_fn(model, item_emb_eval: torch.Tensor, device: str):
    """
    Trả về hàm nhận batch → (B, n_items+1) scores.
    item_emb_eval có thể là bản cold-replaced khi test A5.
    """
    def score_fn(batch):
        z_s, _ = model(batch)                   # (B, d)
        scores = z_s @ item_emb_eval.T          # (B, n_items+1)
        return scores
    return score_fn


# ─── Main training loop ───────────────────────────────────────────────────────
def train(cfg: dict) -> dict:
    # ── Setup ─────────────────────────────────────────────────────────────────
    seed      = cfg.get("seed", 42)
    ablation  = cfg.get("ablation", "A7")
    device    = cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = "cpu"

    set_seed(seed)
    print(f"Device: {device} | Ablation: {ablation} | Seed: {seed}")

    # ── Data ──────────────────────────────────────────────────────────────────
    data_cfg    = cfg["data"]
    train_cfg   = cfg["train"]
    model_cfg   = cfg["model"]
    catprocl_cfg = cfg.get("catprocl", {})
    eval_cfg    = cfg.get("eval", {})

    data_dir   = data_cfg["data_dir"]
    batch_size = train_cfg["batch_size"]

    # maxlen: CLI arg > config > 0 (full session)
    _maxlen_cfg = data_cfg.get("maxlen", 0)
    _maxlen_arg = args.maxlen if hasattr(args, "maxlen") and args.maxlen is not None else None
    maxlen = _maxlen_arg if _maxlen_arg is not None else _maxlen_cfg

    train_loader, val_loader, test_loader, data = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=(device == "cuda"),
        maxlen=maxlen,
    )

    # Sanity check cold_items
    n_cold = len(data["cold_items"])
    expected = data_cfg.get("expected_cold")
    print(f"Data loaded | n_items={data['n_items']:,} | cold_items={n_cold:,}"
          f" | train_batches={len(train_loader):,} | val_batches={len(val_loader):,}")
    if expected and n_cold != expected:
        print(f"  ⚠ WARNING: expected cold_items={expected:,}, got {n_cold:,}")

    n_items     = data["n_items"]
    cold_items  = data["cold_items"]
    item2cat    = data["item2cat"]
    cat2items_w = data["cat2items_warm"]
    n_cats      = data["n_cats"]
    ks          = eval_cfg.get("ks", [10, 20])

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SRGNNEncoder(
        n_items=n_items,
        d=model_cfg.get("hidden_size", 128),
        n_steps=model_cfg.get("n_gnn_steps", 1),
    ).to(device)

    # ── Two-Stage: load pretrained backbone (A7) nếu có ──────────────────────
    pretrain_ckpt = cfg.get("pretrain_ckpt", None)
    if pretrain_ckpt:
        pretrain_ckpt = os.path.expanduser(pretrain_ckpt)
    is_two_stage = bool(pretrain_ckpt and os.path.isfile(pretrain_ckpt))
    if is_two_stage:
        ckpt_data = torch.load(pretrain_ckpt, map_location=device)
        model.load_state_dict(ckpt_data["model"])
        print(f"[Two-Stage] Loaded pretrain backbone: {pretrain_ckpt}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.get("lr", 0.001),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )
    if is_two_stage:
        print(f"[Two-Stage] Fine-tune LR={train_cfg.get('lr', 0.001)} | ablation={ablation}")

    # AMP: FP16 matmul trên tensor cores → ~1.5-2x speedup cho z_s @ item_emb.T
    use_amp = (device == "cuda") and cfg.get("use_amp", True)
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── Ablation flags ────────────────────────────────────────────────────────
    # See module docstring for full legend. A11 is the paper's full model.

    VALID_ABLATIONS = {"A1", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14", "A15"}
    if ablation not in VALID_ABLATIONS:
        raise ValueError(
            f"Ablation '{ablation}' không hợp lệ. "
            f"Hợp lệ: {sorted(VALID_ABLATIONS)}. "
            f"(A2 chưa implement — cần thay đổi kiến trúc model)"
        )

    use_ema_proto      = ablation in ("A3", "A4", "A7", "A11", "A13", "A14", "A15")  # EMA PrototypeBank
    use_fixed_proto    = ablation in ("A5",)                # Fixed epoch-mean PrototypeBank
    use_kmeans_proto   = ablation in ("A6",)                # K-means PrototypeBank
    use_adaptive_proto = ablation in ("A9",)                # Adaptive EMA PrototypeBank (A9)
    use_coreset_proto  = ablation in ("A10",)               # Coreset EMA PrototypeBank (A10)
    use_attn_proto     = ablation in ("A12",)               # Attention-Weighted PrototypeBank (A12)
    any_proto_bank     = (use_ema_proto or use_fixed_proto or use_adaptive_proto
                          or use_coreset_proto or use_attn_proto)
    use_cl             = ablation in ("A4", "A5", "A6", "A7", "A9", "A10", "A11", "A12", "A13", "A14", "A15")  # item InfoNCE
    use_cold_proto     = ablation in ("A7", "A9", "A10", "A11", "A12", "A13", "A14", "A15")  # cold inference
    use_cold_ecat      = ablation in ("A8",)                # cold inference w/ category mean
    use_cold           = use_cold_proto or use_cold_ecat
    # A11/A12: Prototype Simulation Masking — thay target embedding bằng prototype khi train
    use_mask_proto     = ablation in ("A11", "A12")
    # A13: Session-Prototype NCE — thêm loss căn chỉnh z_s với prototype category target
    use_proto_nce      = ablation in ("A13",)
    # A14: Session-EMA Prototype — cập nhật proto bằng z_s (session emb) thay vì h_item
    # Không thay đổi loss, không masking → warm quality giữ nguyên như A7
    # proto[c] ← EMA(z_s của sessions dự đoán item ∈ c) → proto sống trong z_s space
    use_session_proto_update = ablation in ("A14",)
    # A15: Additive Masked Rec Loss — giữ nguyên L_rec (item discrimination) và CỘNG THÊM
    # L_rec_masked để học cold alignment, không conflict như A11 (thay thế hoàn toàn)
    # L = L_rec + λ_mask × L_rec_masked + λ_proto × L_item_proto
    # L_rec_masked: softmax(-) với positive = z_s @ proto[cat(target)],
    #               denominator = logsumexp(z_s @ item_emb.T) — dùng chung với L_rec
    use_masked_rec = ablation in ("A15",)

    momentum           = catprocl_cfg.get("ema_momentum", 0.99)
    lambda_proto       = catprocl_cfg.get("lambda_proto", 0.1)
    tau                = catprocl_cfg.get("temperature", 0.1)   # document §3.5.6: τ=0.1
    perturb_std        = catprocl_cfg.get("perturbation_std", 0.0)
    min_support        = catprocl_cfg.get("min_support", 10)    # A9
    min_momentum       = catprocl_cfg.get("min_momentum", 0.50) # A9
    coreset_k          = catprocl_cfg.get("coreset_k", 5)       # A10
    mask_prob          = catprocl_cfg.get("mask_prob", 0.2)     # A11/A12
    lambda_proto_nce   = catprocl_cfg.get("lambda_proto_nce", 0.1)   # A13: session-proto NCE weight
    tau_proto          = catprocl_cfg.get("tau_proto", 0.1)           # A13: temperature cho proto NCE
    lambda_masked      = catprocl_cfg.get("lambda_masked", 0.3)       # A15: additive masked rec loss weight
    d                  = model_cfg.get("hidden_size", 128)

    # ── Khởi tạo Prototype Bank ────────────────────────────────────────────────
    proto_bank  = None
    kmeans_bank = None

    if use_ema_proto:
        proto_bank = PrototypeBank(
            n_cats=n_cats, d=d, momentum=momentum, device=device, mode="ema"
        )
    elif use_fixed_proto:
        proto_bank = PrototypeBank(
            n_cats=n_cats, d=d, momentum=momentum, device=device, mode="fixed"
        )
    elif use_adaptive_proto:
        proto_bank = AdaptivePrototypeBank(
            n_cats=n_cats, d=d, device=device,
            min_momentum=min_momentum, max_momentum=momentum,
            min_support=min_support,
        )
        print(f"[A9] AdaptivePrototypeBank | min_momentum={min_momentum} "
              f"max_momentum={momentum} min_support={min_support}")
    elif use_coreset_proto:
        proto_bank = CoresetPrototypeBank(
            n_cats=n_cats, d=d, momentum=momentum,
            device=device, top_k=coreset_k,
        )
        print(f"[A10] CoresetPrototypeBank | top_k={coreset_k} "
              f"momentum={momentum}")
    elif use_attn_proto:
        proto_bank = AttentionPrototypeBank(
            n_cats=n_cats, d=d, momentum=momentum, device=device,
        )
        print(f"[A12] AttentionPrototypeBank | momentum={momentum} "
              f"mask_prob={mask_prob} lambda_proto={lambda_proto}")
    elif use_kmeans_proto:
        kmeans_bank = KMeansPrototypeBank(
            n_clusters=n_cats, d=d, device=device
        )

    # Bank tham chiếu dùng cho InfoNCE loss (PrototypeBank hoặc KMeansBank)
    active_bank = proto_bank if any_proto_bank else kmeans_bank

    if use_proto_nce:
        print(f"[A13] Session-Prototype NCE | lambda_proto_nce={lambda_proto_nce} "
              f"tau_proto={tau_proto} | item-NCE: lambda={lambda_proto} tau={tau}")
    if use_session_proto_update:
        print(f"[A14] Target-Item-EMA Prototype | momentum={momentum} lambda={lambda_proto} tau={tau}"
              f" — proto updated from item_emb[target] (raw lookup), NO masking, NO extra loss")
    if use_masked_rec:
        print(f"[A15] Additive Masked Rec Loss | lambda_masked={lambda_masked} lambda_proto={lambda_proto} tau={tau}"
              f" — L = L_rec + {lambda_masked}×L_rec_masked + {lambda_proto}×L_item_proto"
              f" | L_rec_masked: pos=z_s@proto[cat], denom=logsumexp(z_s@item_emb.T)")

    # ── Pre-compute item2cat tensor + warm_mask (1 lần, không rebuild mỗi epoch) ─
    item2cat_tensor = torch.zeros(n_items + 1, dtype=torch.long, device=device)
    warm_mask_global = torch.zeros(n_items + 1, dtype=torch.bool, device=device)
    for item_id, cat_id in item2cat.items():
        item2cat_tensor[item_id] = cat_id
        if item_id not in cold_items:
            warm_mask_global[item_id] = True

    warm_item_ids = torch.where(warm_mask_global)[0]  # (n_warm,) dùng cho K-means

    # A11/A12: pre-compute cold category mask — True = category có ít nhất 1 cold item
    # Chỉ mask targets thuộc cold categories → CE sạch trên non-cold categories
    cold_cat_mask = torch.zeros(n_cats + 1, dtype=torch.bool, device=device)
    for cold_id in cold_items:
        c = item2cat.get(cold_id, 0)
        if 0 < c <= n_cats:
            cold_cat_mask[c] = True
    n_cold_cats = cold_cat_mask.sum().item()

    print(f"Building item2cat tensor and warm mask ...")
    warm_count = warm_item_ids.shape[0]
    if use_mask_proto:
        print(f"  cold_cat_mask: {n_cold_cats}/{n_cats} categories have cold items "
              f"→ masking restricted to these categories only")

    # Two-Stage: warm-start prototype bank ngay từ đầu (backbone đã tốt rồi)
    if is_two_stage and proto_bank is not None and not proto_bank.initialized:
        init_emb = model.all_item_embeddings().detach()
        proto_bank.warm_start(init_emb, item2cat, cat2items_w)
        print(f"[Two-Stage] Proto bank warm-started immediately from pretrained backbone")

    print(f"Ready | warm_items={warm_count:,} | Starting training ...")

    # ── A6: K-means init trước epoch 1 (dùng embeddings ban đầu) ─────────────
    if use_kmeans_proto and kmeans_bank is not None:
        print(f"[A6] Running initial K-means (K={n_cats}, n_warm={warm_count:,}) ...")
        init_emb = model.all_item_embeddings().detach()
        kmeans_bank.fit(init_emb, warm_item_ids, n_items)
        print(f"[A6] K-means init done.")

    # ── Training ──────────────────────────────────────────────────────────────
    n_epochs   = train_cfg.get("epochs", 30)
    patience   = train_cfg.get("patience", 10)
    grad_clip  = train_cfg.get("grad_clip", 5.0)
    eval_every = eval_cfg.get("eval_every", 1)

    best_val_hr = 0.0
    best_epoch  = 0
    no_improve  = 0
    best_ckpt   = None

    # Output dirs  (expanduser để ~ trong config được expand đúng)
    result_dir = os.path.expanduser(cfg.get("output_dir", "../results"))
    ckpt_dir   = os.path.expanduser(cfg.get("checkpoint_dir", "../checkpoints"))
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    all_epoch_logs = []
    epoch_times: list[float] = []   # để tính ETA

    # ── Epoch progress bar (outer) ─────────────────────────────────────────────
    epoch_bar = tqdm(
        range(1, n_epochs + 1),
        desc=f"[{ablation}|seed{seed}]",
        unit="epoch",
        dynamic_ncols=True,
        colour="blue",
    )

    for epoch in epoch_bar:
        model.train()
        t0         = time.time()
        total_loss = 0.0
        n_batches  = 0

        # ── Batch progress bar (inner) ─────────────────────────────────────
        batch_bar = tqdm(
            train_loader,
            desc=f"  Epoch {epoch:3d}",
            unit="batch",
            leave=False,          # xóa sau khi epoch xong — không làm rối terminal
            dynamic_ncols=True,
            colour="green",
        )

        for batch in batch_bar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            targets = batch["targets"]   # (B,)

            # Forward (wrapped in AMP autocast)
            with torch.cuda.amp.autocast(enabled=use_amp):
                z_s, h_nodes = model(batch)              # (B,d), (B,N,d)
                item_emb = model.all_item_embeddings()   # (n_items+1, d)

                # ── Collect warm item embeddings từ batch ─────────────────────
                node_ids_flat = batch["node_ids"].view(-1)
                h_nodes_flat  = h_nodes.view(-1, model.d)

                warm_mask_batch = warm_mask_global[node_ids_flat]
                warm_node_ids   = node_ids_flat[warm_mask_batch]
                h_warm          = h_nodes_flat[warm_mask_batch]

                # cat_warm: dùng item2cat cho tất cả ablations ngoại trừ A6
                # A6: dùng kmeans cluster assignment
                if use_kmeans_proto and kmeans_bank is not None and kmeans_bank.initialized:
                    cat_warm = kmeans_bank.item_clusters[warm_node_ids]
                else:
                    cat_warm = item2cat_tensor[warm_node_ids]

                # ── A11/A12: Prototype Simulation Masking ─────────────────────
                # Sau warm_start (epoch≥2), thay embedding của một số warm targets
                # bằng prototype của category chúng → model học score items qua prototype
                # → không còn distribution mismatch với cold inference lúc test
                #
                # Cold-category-only masking: chỉ mask targets thuộc categories
                # có ít nhất 1 cold item → CE sạch trên non-cold categories
                # → Overall tăng vì 87% category không bị nhiễu
                if use_mask_proto and proto_bank is not None \
                        and proto_bank.initialized and mask_prob > 0:
                    item_emb_for_loss = item_emb.clone()
                    target_cats  = item2cat_tensor[targets]         # (B,)
                    warm_tgt     = warm_mask_global[targets]        # (B,) bool
                    valid_cat    = target_cats > 0                  # (B,) bool
                    is_cold_cat  = cold_cat_mask[target_cats]       # (B,) bool ← chỉ cold categories
                    do_mask      = (
                        torch.rand(targets.shape[0], device=device) < mask_prob
                    ) & warm_tgt & valid_cat & is_cold_cat
                    if do_mask.any():
                        items_to_mask = targets[do_mask].unique()
                        cats_for_mask = item2cat_tensor[items_to_mask]
                        # detach: prototype không backprop qua đây
                        item_emb_for_loss[items_to_mask] = \
                            proto_bank.protos[cats_for_mask].detach()
                else:
                    item_emb_for_loss = item_emb

                # ── Loss ──────────────────────────────────────────────────────
                bank = active_bank   # PrototypeBank (A3/A4/A5/A7/A11) hoặc KMeansBank (A6)

                if use_cl and bank is not None:
                    if use_adaptive_proto:
                        # A9: support-gated InfoNCE (sparse categories bị lọc)
                        loss, l_rec, l_proto = combined_loss_a9(
                            z_s, item_emb_for_loss, targets,
                            h_warm, cat_warm, bank,
                            lambda_proto=lambda_proto, tau=tau,
                        )
                    else:
                        # A4, A5, A6, A7, A11: standard InfoNCE
                        loss, l_rec, l_proto = combined_loss(
                            z_s, item_emb_for_loss, targets,
                            h_warm, cat_warm, bank,
                            lambda_proto=lambda_proto, tau=tau,
                        )
                else:
                    # A1: chỉ L_rec
                    # A3: có proto bank nhưng không dùng trong loss
                    loss    = rec_loss(z_s, item_emb_for_loss, targets)
                    l_rec   = loss
                    l_proto = torch.tensor(0.0, device=device)

                # A13: thêm Session-Prototype NCE (chỉ sau khi proto_bank initialized)
                if use_proto_nce and proto_bank is not None and proto_bank.initialized:
                    target_cats_batch = item2cat_tensor[targets]   # (B,)
                    l_session_proto = session_prototype_nce(
                        z_s, target_cats_batch, proto_bank, tau=tau_proto,
                    )
                    loss = loss + lambda_proto_nce * l_session_proto

                # A15: Additive Masked Rec Loss (đúng cách — clone item_emb, thay target slot)
                # L_rec_masked = CE(z_s @ item_emb_masked.T, targets)
                # item_emb_masked[target_i] = proto[cat(target_i)]  ← thay đúng slot
                # → denominator tự bao gồm proto → self-regulating, không collapse
                # Grad chỉ qua z_s (item_emb_masked được build under no_grad)
                if use_masked_rec and proto_bank is not None and proto_bank.initialized:
                    target_cats_a15 = item2cat_tensor[targets]   # (B,)
                    valid_a15 = target_cats_a15 > 0
                    if valid_a15.any():
                        with torch.no_grad():
                            item_emb_m = item_emb.clone()           # (n_items+1, d) — no grad
                            valid_tgt_a15  = targets[valid_a15]
                            valid_cats_a15 = target_cats_a15[valid_a15]
                            item_emb_m[valid_tgt_a15] = proto_bank.protos[valid_cats_a15]
                        l_masked = rec_loss(z_s, item_emb_m, targets)
                        loss     = loss + lambda_masked * l_masked

            # ── Backward ──────────────────────────────────────────────────
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            # ── Update prototype bank SAU optimizer.step() ────────────────
            if proto_bank is not None and h_warm.shape[0] > 0:
                # A3, A4, A7 (ema): EMA update mỗi batch
                # A5 (fixed): tích lũy, sẽ finalize_epoch() cuối epoch
                if not use_session_proto_update:
                    # A7 và các ablation khác: update từ h_warm (item-level GNN embeddings)
                    proto_bank.update(h_warm.detach(), item2cat_tensor[warm_node_ids].detach())
                else:
                    # A14: update proto bằng item_emb[warm_target] — raw lookup embedding
                    # của chính item được dự đoán (target), thay vì GNN output của tất cả nodes.
                    # Khác A7 (h_warm = GNN output của warm NODES trong prefix),
                    # A14 chỉ focus vào item TARGET → proto cụ thể hơn cho từng category.
                    # Cùng space với item_emb dùng trong scoring → không bị scale mismatch.
                    warm_tgt_mask = warm_mask_global[targets]         # (B,) bool
                    if warm_tgt_mask.any():
                        warm_targets  = targets[warm_tgt_mask]        # (B',) warm target IDs
                        cats_tgt      = item2cat_tensor[warm_targets]  # (B',)
                        valid_cat     = cats_tgt > 0
                        if valid_cat.any():
                            # Raw lookup embedding của target item (cùng space với item_emb scoring)
                            tgt_emb = item_emb[warm_targets[valid_cat]].detach()  # (B'', d)
                            proto_bank.update(tgt_emb, cats_tgt[valid_cat])

            total_loss += loss.item()
            n_batches  += 1

            # Update inner bar với loss hiện tại
            batch_bar.set_postfix(loss=f"{loss.item():.4f}", refresh=False)

        batch_bar.close()

        avg_loss    = total_loss / max(n_batches, 1)
        elapsed     = time.time() - t0
        epoch_times.append(elapsed)

        # ── ETA calculation ────────────────────────────────────────────────
        avg_epoch_time = sum(epoch_times[-5:]) / len(epoch_times[-5:])  # rolling 5
        epochs_left    = n_epochs - epoch
        eta_sec        = avg_epoch_time * epochs_left
        eta_str        = str(timedelta(seconds=int(eta_sec)))

        # ── Post-epoch updates ─────────────────────────────────────────────
        current_item_emb = model.all_item_embeddings().detach()

        # A3/A4/A7 (ema) + A10 (coreset) + A12 (attn): warm start sau epoch 1
        if epoch == 1 and proto_bank is not None \
                and not proto_bank.initialized \
                and getattr(proto_bank, "mode", "ema") in ("ema", "coreset", "attn"):
            proto_bank.warm_start(current_item_emb, item2cat, cat2items_w)

        # A5 (fixed): finalize epoch mean → gán prototype
        if use_fixed_proto and proto_bank is not None:
            proto_bank.finalize_epoch()
            if not proto_bank.initialized:
                proto_bank.initialized = True

        # A6 (kmeans): refit K-means trên embeddings mới nhất
        if use_kmeans_proto and kmeans_bank is not None:
            kmeans_bank.fit(current_item_emb, warm_item_ids, n_items)

        # ── Evaluation ────────────────────────────────────────────────────
        if epoch % eval_every == 0:
            model.eval()

            item_emb_val = model.all_item_embeddings().detach()
            score_fn_val = make_score_fn(model, item_emb_val, device)
            val_results  = evaluate(score_fn_val, val_loader, item_emb_val,
                                    cold_items, ks=ks, device=device)
            val_hr = val_results[f"HR@{max(ks)}"]

            # Update outer epoch bar
            epoch_bar.set_postfix(
                loss=f"{avg_loss:.4f}",
                val_hr=f"{val_hr:.4f}",
                best=f"{best_val_hr:.4f}",
                ETA=eta_str,
            )

            log_entry = {"epoch": epoch, "loss": avg_loss,
                         "epoch_time_s": elapsed,
                         **{f"val_{k}": v for k, v in val_results.items()}}
            all_epoch_logs.append(log_entry)

            if val_hr > best_val_hr:
                best_val_hr = val_hr
                best_epoch  = epoch
                no_improve  = 0
                best_ckpt = os.path.join(ckpt_dir,
                    f"{data_cfg['dataset']}_{ablation}_seed{seed}_best.pt")
                torch.save({
                    "epoch":       epoch,
                    "model":       model.state_dict(),
                    "optimizer":   optimizer.state_dict(),
                    "proto_bank":  proto_bank.state_dict()  if proto_bank  else None,
                    "kmeans_bank": kmeans_bank.state_dict() if kmeans_bank else None,
                    "val_hr":      best_val_hr,
                    "cfg":         cfg,
                }, best_ckpt)
                tqdm.write(f"  ✓ Epoch {epoch:3d} | loss={avg_loss:.4f} | "
                           f"Val HR@{max(ks)}={val_hr:.4f} | "
                           f"time={elapsed:.1f}s | ETA={eta_str}  ← best")
            else:
                no_improve += 1
                tqdm.write(f"  · Epoch {epoch:3d} | loss={avg_loss:.4f} | "
                           f"Val HR@{max(ks)}={val_hr:.4f} | "
                           f"time={elapsed:.1f}s | ETA={eta_str} "
                           f"(no improve {no_improve}/{patience})")
                if no_improve >= patience:
                    tqdm.write(f"\nEarly stopping at epoch {epoch} "
                               f"(best epoch={best_epoch}, "
                               f"best Val HR@{max(ks)}={best_val_hr:.4f})")
                    break
        else:
            epoch_bar.set_postfix(loss=f"{avg_loss:.4f}", ETA=eta_str)

    # ── Test evaluation (load best checkpoint) ────────────────────────────────
    print(f"\nLoading best checkpoint (epoch={best_epoch}) for test evaluation ...")
    if best_ckpt and os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        if proto_bank and ckpt.get("proto_bank"):
            proto_bank.load_state_dict(ckpt["proto_bank"])
        if kmeans_bank and ckpt.get("kmeans_bank"):
            kmeans_bank.load_state_dict(ckpt["kmeans_bank"])

    model.eval()
    item_emb_base = model.all_item_embeddings().detach()

    # ── Test item embeddings: áp dụng cold inference nếu cần ─────────────────
    if use_cold_proto and proto_bank is not None:
        # CatPro: thay embedding cold items bằng EMA prototype
        item_emb_test = build_eval_item_embeddings(
            item_emb_base, proto_bank, item2cat, cold_items,
            perturbation_std=perturb_std,
        )
    elif use_cold_ecat:
        # A8: thay embedding cold items bằng mean warm items trong category
        item_emb_test = build_eval_item_embeddings_ecat(
            item_emb_base, cat2items_w, item2cat, cold_items,
            perturbation_std=perturb_std,
        )
    else:
        # A1, A3, A4, A5, A6: giữ nguyên embeddings
        item_emb_test = item_emb_base

    score_fn_test = make_score_fn(model, item_emb_test, device)
    test_results  = evaluate(score_fn_test, test_loader, item_emb_test,
                             cold_items, ks=ks, device=device)

    print("\n" + "=" * 60)
    print(f"TEST RESULTS — {data_cfg['dataset']} | {ablation} | seed={seed}")
    print(format_results(test_results))
    print("=" * 60)

    # ── Save results to JSON ──────────────────────────────────────────────────
    result = {
        "dataset":    data_cfg["dataset"],
        "ablation":   ablation,
        "seed":       seed,
        "best_epoch": best_epoch,
        "test":       test_results,
        "epoch_logs": all_epoch_logs,
        "config":     cfg,
    }
    base_name    = f"{data_cfg['dataset']}_{ablation}_seed{seed}"
    result_fname = os.path.join(result_dir, f"{base_name}.json")
    if os.path.exists(result_fname):
        v = 2
        while os.path.exists(os.path.join(result_dir, f"{base_name}_v{v}.json")):
            v += 1
        result_fname = os.path.join(result_dir, f"{base_name}_v{v}.json")
    with open(result_fname, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved → {result_fname}")

    return test_results


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CatPro-CL")
    parser.add_argument("--config",           required=True, help="Path to YAML config")
    parser.add_argument("--ablation",         default=None,  help="Override ablation (A1/A3-A8)")
    parser.add_argument("--seed",             type=int, default=None, help="Override seed")
    parser.add_argument("--device",           default=None,  help="Override device (cuda/cpu)")
    parser.add_argument("--data_dir",         default=None,  help="Override data directory")
    parser.add_argument("--perturbation_std", type=float, default=None,
                        help="Override catprocl.perturbation_std (e.g. 0.05 to break ties in A7)")
    parser.add_argument("--lambda_proto",     type=float, default=None,
                        help="Override catprocl.lambda_proto (e.g. 0.01, 0.05, 0.1)")
    parser.add_argument("--temperature",      type=float, default=None,
                        help="Override catprocl.temperature (InfoNCE tau)")
    parser.add_argument("--patience",         type=int,   default=None,
                        help="Override train.patience")
    parser.add_argument("--top_k",            type=int,   default=None,
                        help="Override catprocl.coreset_k (A10: top-K items per category)")
    parser.add_argument("--ema_momentum",     type=float, default=None,
                        help="Override catprocl.ema_momentum (e.g. 0.95, 0.9)")
    parser.add_argument("--mask_prob",        type=float, default=None,
                        help="Override catprocl.mask_prob (A11: prob thay target emb bằng prototype)")
    parser.add_argument("--output_dir",       default=None,
                        help="Override output_dir (thư mục lưu JSON kết quả)")
    parser.add_argument("--checkpoint_dir",   default=None,
                        help="Override checkpoint_dir (thư mục lưu .pt checkpoint)")
    parser.add_argument("--pretrain_ckpt",    default=None,
                        help="Two-stage: path tới A7 checkpoint để load backbone trước khi fine-tune")
    parser.add_argument("--finetune_lr",      type=float, default=None,
                        help="Two-stage: LR nhỏ hơn cho fine-tune stage (e.g. 0.0001)")
    parser.add_argument("--maxlen",           type=int,   default=None,
                        help="Truncate sessions to last N items (0=full, None=read from config)")
    args = parser.parse_args()

    overrides = {
        "ablation":                     args.ablation,
        "seed":                         args.seed,
        "device":                       args.device,
        "data.data_dir":                args.data_dir,
        "catprocl.perturbation_std":    args.perturbation_std,
        "catprocl.lambda_proto":        args.lambda_proto,
        "catprocl.temperature":         args.temperature,
        "train.patience":               args.patience,
        "catprocl.coreset_k":           args.top_k,
        "catprocl.ema_momentum":        args.ema_momentum,
        "catprocl.mask_prob":           args.mask_prob,
        "output_dir":                   args.output_dir,
        "checkpoint_dir":               args.checkpoint_dir,
        "pretrain_ckpt":                args.pretrain_ckpt,
        "train.lr":                     args.finetune_lr,
    }
    cfg = load_config(args.config, overrides)
    train(cfg)
