"""
dataset.py
----------
PyTorch Dataset for SR-GNN style session graphs.
Loads from pre-computed .pkl cache — fast, no graph-building overhead.

Each sample in the dataset is one SessionGraph dict.
Collation (padding + batching) is handled by SessionGraphCollator.
"""

import json
import os
import pickle
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# ─── Dataset ──────────────────────────────────────────────────────────────────
class SessionGraphDataset(Dataset):
    """
    Wraps a list of pre-computed SessionGraph dicts.
    __getitem__ returns one graph; collation handles batching.
    """

    def __init__(self, graphs: list[dict]):
        self.graphs = graphs

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> dict:
        return self.graphs[idx]


# ─── Collator ─────────────────────────────────────────────────────────────────
class SessionGraphCollator:
    """
    Pads and batches a list of SessionGraph dicts into tensors.

    Output batch keys:
      node_ids  : (B, max_nodes) int64 — padded global item IDs (0 = pad)
      A_in      : (B, max_nodes, max_nodes) float32
      A_out     : (B, max_nodes, max_nodes) float32
      seq_idx   : (B, max_seq)   int64 — local indices into node_ids (0 = pad)
      mask      : (B, max_seq)   float32 — 1 for valid, 0 for pad
      targets   : (B,)           int64 — global item IDs to predict
      n_nodes   : (B,)           int64
    """

    def __call__(self, batch: list[dict]) -> dict:
        max_nodes = max(g["n_nodes"]   for g in batch)
        max_seq   = max(len(g["seq_idx"]) for g in batch)
        B         = len(batch)

        node_ids_pad = np.zeros((B, max_nodes), dtype=np.int64)
        A_in_pad     = np.zeros((B, max_nodes, max_nodes), dtype=np.float32)
        A_out_pad    = np.zeros((B, max_nodes, max_nodes), dtype=np.float32)
        seq_idx_pad  = np.zeros((B, max_seq),   dtype=np.int64)
        mask_pad     = np.zeros((B, max_seq),   dtype=np.float32)
        targets      = np.zeros(B,              dtype=np.int64)

        for i, g in enumerate(batch):
            n  = g["n_nodes"]
            sq = len(g["seq_idx"])

            node_ids_pad[i, :n]   = g["node_ids"]
            A_in_pad[i, :n, :n]   = g["A_in"]
            A_out_pad[i, :n, :n]  = g["A_out"]
            seq_idx_pad[i, :sq]   = g["seq_idx"]
            mask_pad[i, :sq]      = 1.0
            targets[i]            = g["target"]

        return {
            "node_ids": torch.from_numpy(node_ids_pad),
            "A_in":     torch.from_numpy(A_in_pad),
            "A_out":    torch.from_numpy(A_out_pad),
            "seq_idx":  torch.from_numpy(seq_idx_pad),
            "mask":     torch.from_numpy(mask_pad),
            "targets":  torch.from_numpy(targets),
            "n_nodes":  torch.tensor([g["n_nodes"] for g in batch], dtype=torch.int64),
        }
