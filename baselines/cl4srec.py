"""
cl4srec.py — CL4SRec Baseline
-------------------------------
Xie et al., "Contrastive Learning for Sequential Recommendation", ICDE 2022.

Official code : https://github.com/RUCAIBox/RecBole-DA
  recbole/model/sequential_recommender/cl4srec.py

Architecture (SASRec-style Transformer + contrastive augmentation):
  - Item embeddings + position embeddings (0..max_len-1, ALL positions)
  - 2-layer Transformer with causal + padding attention mask
  - Session rep = output at LAST VALID position (right-padded format)
  - Three augmentation operators: crop / mask / reorder
  - Symmetric InfoNCE between two independently augmented views
  - Loss = CE(z_main, target) + λ * InfoNCE(z_aug1, z_aug2)

Critical implementation notes from official code:
  - RIGHT-padding (items at 0..L-1, padding at L..max_len-1)
  - position_ids = arange(0..max_len-1) for ALL positions (not just valid)
  - gather_indexes(output, seq_len - 1) to get last valid position
  - Combined additive attention mask: causal AND padding, value -10000 for masked

Our adaptation:
  - Standalone Transformer (no RecBole dependency)
  - Same augmentation as official (crop/mask/reorder with eta=0.6/0.3/0.6)
  - Full-ranking eval + cold-split protocol
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.evaluator import format_results


# ── Data (sequence format, right-padded) ──────────────────────────────────────
def load_sessions(path: str) -> list:
    sessions = []
    with open(path) as f:
        for line in f:
            items = list(map(int, line.strip().split()))
            if len(items) >= 2:
                sessions.append(items)
    return sessions


class SeqDataset(Dataset):
    def __init__(self, sessions: list, max_len: int = 50):
        self.sessions = sessions
        self.max_len  = max_len

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        sess   = self.sessions[idx]
        target = sess[-1]
        items  = sess[:-1][-self.max_len:]  # truncate to max_len
        return items, target


def collate_seq(batch, max_len: int = 50):
    """
    RIGHT-pad sequences to max_len (matching RecBole's convention).
    Items at positions 0..L-1, padding (0) at L..max_len-1.
    """
    seqs, targets = zip(*batch)
    B = len(seqs)

    seq_tensor = torch.zeros(B, max_len, dtype=torch.long)   # 0=padding
    len_tensor = torch.zeros(B, dtype=torch.long)

    for i, s in enumerate(seqs):
        L = min(len(s), max_len)
        if L > 0:
            seq_tensor[i, :L] = torch.tensor(s[:L], dtype=torch.long)
        len_tensor[i] = max(L, 1)  # at least 1

    return {
        "seq":     seq_tensor,                                     # (B, max_len)
        "seq_len": len_tensor,                                     # (B,)
        "targets": torch.tensor(targets, dtype=torch.long),
    }


# ── Augmentation (matches official CL4SRec in RecBole-DA) ────────────────────
def item_crop(seq: list, seq_len: int, eta: float = 0.6) -> tuple:
    """Keep eta fraction of the valid sequence (random contiguous window)."""
    num_left = math.floor(seq_len * eta)
    num_left = max(num_left, 1)
    crop_begin = random.randint(0, seq_len - num_left)
    cropped = [0] * len(seq)
    end = min(crop_begin + num_left, len(seq))
    cropped[:num_left] = seq[crop_begin:end]
    return cropped, num_left


def item_mask(seq: list, seq_len: int, n_items: int, gamma: float = 0.3) -> tuple:
    """Randomly replace gamma fraction of valid items with mask token (n_items)."""
    num_mask = math.floor(seq_len * gamma)
    num_mask = max(num_mask, 1)
    mask_index = random.sample(range(seq_len), k=min(num_mask, seq_len))
    masked = list(seq)
    for i in mask_index:
        masked[i] = n_items   # mask token = n_items (same as RecBole)
    return masked, seq_len


def item_reorder(seq: list, seq_len: int, beta: float = 0.6) -> tuple:
    """Shuffle a random contiguous sub-sequence of length beta * seq_len."""
    num_reorder = math.floor(seq_len * beta)
    num_reorder = max(num_reorder, 1)
    reorder_begin = random.randint(0, seq_len - num_reorder)
    reordered = list(seq)
    shuffle_index = list(range(reorder_begin, reorder_begin + num_reorder))
    random.shuffle(shuffle_index)
    for i, j in enumerate(range(reorder_begin, reorder_begin + num_reorder)):
        reordered[j] = seq[shuffle_index[i]]
    return reordered, seq_len


def augment_one(seq: list, seq_len: int, n_items: int) -> tuple:
    """Apply one random augmentation operator (official: crop/mask/reorder)."""
    if seq_len <= 1:
        return list(seq), max(seq_len, 1)
    op = random.randint(0, 2)
    if op == 0:
        return item_crop(seq, seq_len)
    elif op == 1:
        return item_mask(seq, seq_len, n_items)
    else:
        return item_reorder(seq, seq_len)


def make_aug_batch(raw_seqs: list, seq_lens: list, n_items: int,
                   max_len: int, device) -> tuple:
    """
    Augment a batch of right-padded sequences and return new tensors.
    raw_seqs : list of list[int] (right-padded, length max_len)
    seq_lens : list[int]
    """
    B = len(raw_seqs)
    seq_t = torch.zeros(B, max_len, dtype=torch.long)
    len_t = torch.zeros(B, dtype=torch.long)
    for i, (s, l) in enumerate(zip(raw_seqs, seq_lens)):
        aug, aug_len = augment_one(s[:l], l, n_items)
        L = min(len(aug), max_len)
        if L > 0:
            seq_t[i, :L] = torch.tensor(aug[:L], dtype=torch.long)
        len_t[i] = max(L, 1)
    return seq_t.to(device), len_t.to(device)


# ── Transformer (standalone, matches RecBole's SASRec TransformerEncoder) ──────
class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with combined causal+padding additive mask."""

    def __init__(self, d: int, n_heads: int, attn_dropout: float = 0.1):
        super().__init__()
        assert d % n_heads == 0
        self.d      = d
        self.h      = n_heads
        self.d_k    = d // n_heads
        self.w_q    = nn.Linear(d, d)
        self.w_k    = nn.Linear(d, d)
        self.w_v    = nn.Linear(d, d)
        self.out    = nn.Linear(d, d)
        self.drop   = nn.Dropout(attn_dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        H, dk   = self.h, self.d_k

        q = self.w_q(x).view(B, L, H, dk).transpose(1, 2)  # (B, H, L, dk)
        k = self.w_k(x).view(B, L, H, dk).transpose(1, 2)
        v = self.w_v(x).view(B, L, H, dk).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dk)  # (B, H, L, L)
        scores = scores + attn_mask                           # additive mask (-10000)

        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)

        z = (attn @ v).transpose(1, 2).contiguous().view(B, L, self.d)
        return self.out(z)


class FeedForwardBlock(nn.Module):
    def __init__(self, d: int, inner_size: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, inner_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_size, d),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Single Transformer layer: self-attn + FFN, both with post-norm (SASRec style)."""

    def __init__(self, d: int, n_heads: int, inner_size: int,
                 attn_dropout: float = 0.1, hidden_dropout: float = 0.1,
                 layer_norm_eps: float = 1e-12):
        super().__init__()
        self.attn  = MultiHeadSelfAttention(d, n_heads, attn_dropout)
        self.ff    = FeedForwardBlock(d, inner_size, hidden_dropout)
        self.norm1 = nn.LayerNorm(d, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d, eps=layer_norm_eps)
        self.drop  = nn.Dropout(hidden_dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        z = self.attn(x, attn_mask)
        x = self.norm1(x + self.drop(z))
        x = self.norm2(x + self.ff(x))
        return x


class CL4SRecEncoder(nn.Module):
    """
    CL4SRec model: SASRec Transformer + contrastive augmentation.

    Architecture matches official RecBole-DA CL4SRec:
      - item_embedding: Embedding(n_items + 1, d)  [0=padding, n_items=mask_token]
      - position_embedding: Embedding(max_seq_length, d)  [0..max_len-1]
      - Causal + padding additive mask (combined, -10000 for masked)
      - gather_indexes at position seq_len - 1
    """

    def __init__(self, n_items: int, d: int = 128, n_layers: int = 2,
                 n_heads: int = 4, inner_size: int = 256, max_len: int = 50,
                 attn_dropout: float = 0.1, hidden_dropout: float = 0.1):
        super().__init__()
        self.d       = d
        self.n_items = n_items
        self.max_len = max_len

        # n_items + 1 to accommodate mask token at index n_items
        # (RecBole: item_mask = n_items, items 1..n_items, padding 0)
        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)
        # Position embeddings: indices 0..max_len-1 (all positions get embeddings)
        self.pos_emb  = nn.Embedding(max_len, d)
        self.LayerNorm = nn.LayerNorm(d, eps=1e-12)
        self.drop      = nn.Dropout(hidden_dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d, n_heads, inner_size, attn_dropout, hidden_dropout)
            for _ in range(n_layers)
        ])

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Matches official CL4SRec _init_weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _get_attention_mask(self, item_seq: torch.Tensor) -> torch.Tensor:
        """
        Build combined causal + padding additive mask (B, 1, L, L).
        Matches official get_attention_mask() in RecBole CL4SRec.
        Value 0.0 = can attend; -10000.0 = masked (padding or future).
        """
        # Padding mask: 1 for valid positions, 0 for padding
        pad_mask = (item_seq > 0).long()                   # (B, L)
        ext_mask = pad_mask.unsqueeze(1).unsqueeze(2)      # (B, 1, 1, L)

        # Causal mask: lower-triangular = can attend (1), upper = cannot (0)
        L      = item_seq.size(1)
        causal = torch.triu(torch.ones(1, 1, L, L, device=item_seq.device), diagonal=1)
        causal = (causal == 0).long()                      # lower tri = 1

        # Combined: valid position AND causal constraint → can attend
        combined = ext_mask * causal                       # (B, 1, L, L)

        # Convert to additive float mask
        dtype = next(self.parameters()).dtype
        combined = combined.to(dtype=dtype)
        return (1.0 - combined) * -10000.0                 # (B, 1, L, L)

    def encode(self, seq: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        """
        seq     : (B, max_len) right-padded item IDs
        seq_len : (B,) number of valid items per sequence
        Returns : (B, d) session representation at last valid position
        """
        B, L = seq.shape

        # Position IDs: 0..L-1 for ALL positions (official RecBole approach)
        pos_ids = torch.arange(L, dtype=torch.long, device=seq.device)
        pos_ids = pos_ids.unsqueeze(0).expand(B, -1)       # (B, L)

        # Input embedding (item + position, then LayerNorm + dropout)
        x = self.item_emb(seq) + self.pos_emb(pos_ids)     # (B, L, d)
        x = self.drop(self.LayerNorm(x))

        # Attention mask
        attn_mask = self._get_attention_mask(seq)           # (B, 1, L, L)

        for block in self.blocks:
            x = block(x, attn_mask)

        # gather_indexes(output, seq_len - 1): last valid position
        # With RIGHT-padding items are at 0..seq_len-1
        gather_idx = (seq_len - 1).clamp(min=0)            # (B,)
        gather_idx = gather_idx.view(B, 1, 1).expand(B, 1, self.d)
        z = x.gather(dim=1, index=gather_idx).squeeze(1)   # (B, d)
        return z

    def forward(self, seq: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        return self.encode(seq, seq_len)

    def all_item_embeddings(self) -> torch.Tensor:
        """Return item embeddings for scoring: rows 0..n_items (exclude mask token)."""
        return self.item_emb.weight  # (n_items+1, d), target IDs in 1..n_items


# ── InfoNCE (symmetric, in-batch negatives, official info_nce with sim='dot') ──
def infonce_pair(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1):
    """
    Symmetric InfoNCE matching official CL4SRec info_nce(sim='dot').
    z1, z2 : (B, d) session embeddings for two augmented views.
    """
    B  = z1.size(0)
    z  = torch.cat([z1, z2], dim=0)                # (2B, d)
    sim = (z @ z.T) / temperature                   # (2B, 2B)

    # Mask self-similarity (diagonal)
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    mask[torch.arange(B), torch.arange(B, 2 * B)] = True
    mask[torch.arange(B, 2 * B), torch.arange(B)] = True
    negative_samples = sim[~mask].view(2 * B, -1)   # (2B, 2B-2)

    # Positive pairs: (i, i+B) and (i+B, i)
    pos_i_j = torch.diag(sim, B)                    # (B,)
    pos_j_i = torch.diag(sim, -B)                   # (B,)
    positive_samples = torch.cat([pos_i_j, pos_j_i]).view(2 * B, 1)

    labels  = torch.zeros(2 * B, dtype=torch.long, device=z.device)
    logits  = torch.cat([positive_samples, negative_samples], dim=1)  # (2B, 2B-1)
    return F.cross_entropy(logits, labels)


# ── Full-ranking eval (sequence format, matches our evaluator protocol) ────────
def evaluate_seq(
    model: CL4SRecEncoder,
    test_sessions: list,
    cold_items: set,
    ks: list,
    device: str,
    batch_size: int = 100,
    max_len: int = 50,
) -> dict:
    model.eval()
    item_emb = model.all_item_embeddings().to(device)

    hits   = {k: 0 for k in ks}
    rr     = {k: 0.0 for k in ks}
    c_hits = {k: 0 for k in ks}
    c_rr   = {k: 0.0 for k in ks}
    n_all  = 0
    n_cold = 0

    with torch.no_grad():
        for start in range(0, len(test_sessions), batch_size):
            batch_sess = test_sessions[start: start + batch_size]
            inp_list, tgt_list = [], []
            for sess in batch_sess:
                tgt_list.append(sess[-1])
                inp_list.append(sess[:-1][-max_len:])

            batch_dict = collate_seq(list(zip(inp_list, tgt_list)), max_len=max_len)
            seq     = batch_dict["seq"].to(device)
            seq_len = batch_dict["seq_len"].to(device)
            targets = batch_dict["targets"].to(device)

            z_s    = model.encode(seq, seq_len)          # (B, d)
            scores = z_s @ item_emb.T                    # (B, n_items+1)
            scores[:, 0] = float("-inf")                 # exclude padding index

            max_k = max(ks)
            _, top_idx = scores.topk(max_k, dim=1, largest=True, sorted=True)

            for b in range(targets.size(0)):
                tgt     = targets[b].item()
                is_cold = tgt in cold_items
                n_all  += 1
                if is_cold:
                    n_cold += 1

                pos = (top_idx[b] == tgt).nonzero(as_tuple=True)[0]
                rank = (pos[0].item() + 1) if len(pos) > 0 else (max_k + 1)

                for k in ks:
                    if rank <= k:
                        hits[k]   += 1
                        rr[k]     += 1.0 / rank
                        if is_cold:
                            c_hits[k] += 1
                            c_rr[k]   += 1.0 / rank

    res = {}
    for k in ks:
        res[f"HR@{k}"]       = hits[k]   / n_all  if n_all  > 0 else 0.0
        res[f"MRR@{k}"]      = rr[k]     / n_all  if n_all  > 0 else 0.0
        res[f"Cold_HR@{k}"]  = c_hits[k] / n_cold if n_cold > 0 else 0.0
        res[f"Cold_MRR@{k}"] = c_rr[k]   / n_cold if n_cold > 0 else 0.0
    res["n_overall"] = n_all
    res["n_cold"]    = n_cold
    return res


# ── Training loop ──────────────────────────────────────────────────────────────
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CL4SRec | dataset={args.dataset} | seed={args.seed} | device={device}")
    print(f"  Ref: https://github.com/RUCAIBox/RecBole-DA  (ICDE 2022)")

    with open(os.path.join(args.data_dir, "meta.json")) as f:
        meta       = json.load(f)
    n_items        = meta["n_items"]
    cold_items     = set(meta.get("cold_items", []))

    train_sessions = load_sessions(os.path.join(args.data_dir, "sessions_train.txt"))
    val_sessions   = load_sessions(os.path.join(args.data_dir, "sessions_val.txt"))
    test_sessions  = load_sessions(os.path.join(args.data_dir, "sessions_test.txt"))
    print(f"  n_items={n_items:,}  cold={len(cold_items):,}  "
          f"train={len(train_sessions):,}  val={len(val_sessions):,}  "
          f"test={len(test_sessions):,}")

    train_ds = SeqDataset(train_sessions, max_len=args.max_len)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_seq(b, max_len=args.max_len),
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = CL4SRecEncoder(
        n_items=n_items, d=args.d, n_layers=args.n_layers,
        n_heads=args.n_heads, inner_size=args.inner_size,
        max_len=args.max_len,
        attn_dropout=args.attn_dropout, hidden_dropout=args.hidden_dropout,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)

    best_val_hr20 = 0.0
    best_epoch    = 0
    best_test     = {}
    patience_cnt  = 0
    epoch_logs    = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss  = 0.0
        total_cl    = 0.0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Ep {epoch:3d}", ncols=100, leave=False)
        for batch in pbar:
            seq     = batch["seq"].to(device)      # (B, max_len) right-padded
            seq_len = batch["seq_len"].to(device)  # (B,)
            targets = batch["targets"].to(device)  # (B,)
            B       = seq.size(0)

            # ── Main next-item prediction ────────────────────────────────────
            z_s    = model.encode(seq, seq_len)                # (B, d)
            item_e = model.all_item_embeddings()               # (n+1, d)
            scores = z_s @ item_e.T                            # (B, n+1)
            loss_rec = F.cross_entropy(scores, targets)

            # ── Contrastive: two augmented views (official augment method) ───
            raw_seqs = [seq[b, :seq_len[b].item()].cpu().tolist() for b in range(B)]
            raw_lens = seq_len.cpu().tolist()

            seq1, len1 = make_aug_batch(raw_seqs, raw_lens, n_items, args.max_len, device)
            seq2, len2 = make_aug_batch(raw_seqs, raw_lens, n_items, args.max_len, device)

            z1 = model.encode(seq1, len1)                      # (B, d)
            z2 = model.encode(seq2, len2)                      # (B, d)

            cl_loss = infonce_pair(z1, z2, temperature=args.temperature)
            loss    = loss_rec + args.lambda_cl * cl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            total_loss += loss_rec.item()
            total_cl   += cl_loss.item()
            pbar.set_postfix(rec=f"{loss_rec.item():.4f}", cl=f"{cl_loss.item():.4f}")

        pbar.close()
        scheduler.step()

        # ── Eval ──────────────────────────────────────────────────────────────
        val_res  = evaluate_seq(model, val_sessions, cold_items, [10, 20],
                                device=device, batch_size=args.batch_size,
                                max_len=args.max_len)
        val_hr20 = val_res["HR@20"]
        elapsed  = time.time() - t0
        epoch_logs.append({
            "epoch":        epoch,
            "loss":         total_loss / len(train_loader),
            "cl":           total_cl / len(train_loader),
            "epoch_time_s": elapsed,
            **{f"val_{k}": v for k, v in val_res.items()},
        })
        print(f"Epoch {epoch:3d} | rec={total_loss/len(train_loader):.4f} "
              f"CL={total_cl/len(train_loader):.4f} | "
              f"Val HR@20={val_hr20:.4f} | {elapsed:.1f}s")

        if val_hr20 > best_val_hr20:
            best_val_hr20 = val_hr20
            best_epoch    = epoch
            patience_cnt  = 0
            test_res = evaluate_seq(model, test_sessions, cold_items, [10, 20],
                                    device=device, batch_size=args.batch_size,
                                    max_len=args.max_len)
            best_test = test_res
            print(format_results(test_res, prefix="  → New best! Test:"))
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"Early stop at epoch {epoch}  (patience={args.patience})")
                break

    print(f"\nBest epoch={best_epoch}  Val HR@20={best_val_hr20:.4f}")
    print(format_results(best_test, prefix="Final Test:"))

    os.makedirs(args.output_dir, exist_ok=True)
    out = {
        "dataset":    args.dataset,
        "ablation":   "CL4SRec",
        "seed":       args.seed,
        "best_epoch": best_epoch,
        "test":       best_test,
        "epoch_logs": epoch_logs,
        "config":     vars(args),
    }
    fname = os.path.join(args.output_dir, f"{args.dataset}_CL4SRec_seed{args.seed}.json")
    with open(fname, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved → {fname}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",       required=True)
    p.add_argument("--output_dir",     default="~/results")
    p.add_argument("--dataset",        default="retailrocket")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--d",              type=int,   default=128)
    p.add_argument("--n_layers",       type=int,   default=2)
    p.add_argument("--n_heads",        type=int,   default=4)
    p.add_argument("--inner_size",     type=int,   default=256,
                   help="FFN inner dimension (official: 2*d=256)")
    p.add_argument("--max_len",        type=int,   default=50)
    p.add_argument("--attn_dropout",   type=float, default=0.1)
    p.add_argument("--hidden_dropout", type=float, default=0.1)
    p.add_argument("--batch_size",     type=int,   default=100)
    p.add_argument("--epochs",         type=int,   default=30)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--lr_dc",          type=float, default=0.1)
    p.add_argument("--lr_dc_step",     type=int,   default=3)
    p.add_argument("--weight_decay",   type=float, default=1e-5)
    p.add_argument("--clip",           type=float, default=5.0)
    p.add_argument("--patience",       type=int,   default=10)
    p.add_argument("--num_workers",    type=int,   default=4)
    p.add_argument("--lambda_cl",      type=float, default=0.1,
                   help="Weight for contrastive loss (official lmd)")
    p.add_argument("--temperature",    type=float, default=0.1,
                   help="InfoNCE temperature (official tau=0.1)")
    args = p.parse_args()

    args.output_dir = os.path.expanduser(args.output_dir)
    train(args)
