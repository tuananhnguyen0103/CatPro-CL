"""
kmeans_bank.py — K-means Prototype Bank cho Ablation A6
---------------------------------------------------------
Thay category-supervised prototype bằng K-means cluster prototype
với K = n_cats (để so sánh fair với NCL-style clustering).

Pipeline:
  1. Gọi fit() sau mỗi epoch với warm item embeddings.
  2. Trong batch loop: dùng item_cluster_tensor thay item2cat_tensor
     để lookup cluster_id cho InfoNCE loss.
  3. Duck-typing với PrototypeBank: có .protos, .get(), .state_dict().

Spherical K-means (cosine similarity) phù hợp hơn Euclidean
cho embedding space đã normalize.
"""

import torch
import torch.nn.functional as F


class KMeansPrototypeBank:
    """
    Spherical K-means prototype bank cho Ablation A6.

    Args:
        n_clusters : số cluster = n_cats (để so sánh fair)
        d          : embedding dimension
        n_iters    : số vòng lặp K-means (default 30)
        device     : cuda hoặc cpu
    """

    def __init__(
        self,
        n_clusters: int,
        d: int,
        n_iters: int = 30,
        device: str = "cuda",
    ):
        self.n_clusters = n_clusters
        # PrototypeBank alias: n_cats dùng bởi InfoNCE loss
        self.n_cats     = n_clusters
        self.d          = d
        self.n_iters    = n_iters
        self.device     = device

        # Cluster centroids — shape (n_clusters, d)
        self.protos = torch.zeros(n_clusters, d, device=device)
        # item_id → cluster_id — shape (n_items+1,), khởi tạo bằng -1
        self.item_clusters: torch.Tensor | None = None
        self.initialized = False

    # ── Fit: chạy K-means trên warm item embeddings ────────────────────────────
    @torch.no_grad()
    def fit(
        self,
        item_emb: torch.Tensor,          # (n_items+1, d) từ model.all_item_embeddings()
        warm_item_ids: torch.Tensor,     # (n_warm,) — index của warm items
        n_items: int,                    # tổng số items (không tính padding)
    ) -> None:
        """
        Chạy spherical K-means, cập nhật self.protos và self.item_clusters.
        Gọi 1 lần trước epoch 1 (init) và 1 lần sau mỗi epoch.
        """
        warm_embs = item_emb[warm_item_ids]     # (n_warm, d)
        n_warm    = warm_embs.shape[0]

        # Normalize để dùng cosine similarity
        warm_norm = F.normalize(warm_embs, dim=1)

        # ── Khởi tạo centroid ngẫu nhiên từ warm items ────────────────────────
        k = min(self.n_clusters, n_warm)
        perm      = torch.randperm(n_warm, device=self.device)[:k]
        centroids = warm_norm[perm].clone()  # (k, d) — đã normalize

        # Nếu k < n_clusters (rất hiếm), pad bằng random unit vector
        if k < self.n_clusters:
            pad = F.normalize(
                torch.randn(self.n_clusters - k, self.d, device=self.device), dim=1
            )
            centroids = torch.cat([centroids, pad], dim=0)

        # ── K-means iterations ─────────────────────────────────────────────────
        assignments = torch.zeros(n_warm, dtype=torch.long, device=self.device)
        for _ in range(self.n_iters):
            # Assign: cosine similarity → argmax
            sims        = warm_norm @ centroids.T          # (n_warm, K)
            new_assign  = sims.argmax(dim=1)               # (n_warm,)

            # Convergence check
            if torch.equal(new_assign, assignments) and _ > 0:
                break
            assignments = new_assign

            # Update centroids = mean của assigned embeddings, rồi re-normalize
            new_centroids = torch.zeros_like(centroids)
            for c in range(self.n_clusters):
                mask = assignments == c
                if mask.sum() > 0:
                    new_centroids[c] = F.normalize(
                        warm_norm[mask].mean(dim=0), dim=0
                    )
                else:
                    new_centroids[c] = centroids[c]   # empty cluster → keep
            centroids = new_centroids

        # ── Lưu kết quả ───────────────────────────────────────────────────────
        # protos: de-normalize không cần thiết — InfoNCE cũng normalize lại
        # Ta lưu centroids (đã normalize) làm prototypes
        self.protos = centroids  # (n_clusters, d) — normalized unit vectors

        # Xây item_cluster_tensor: (n_items+1,) — mặc định cluster 0 cho unknown
        self.item_clusters = torch.zeros(
            n_items + 1, dtype=torch.long, device=self.device
        )
        self.item_clusters[warm_item_ids] = assignments
        self.initialized = True

    # ── Duck-typing với PrototypeBank ─────────────────────────────────────────
    def get(self, cluster_idx: int | torch.Tensor) -> torch.Tensor:
        return self.protos[cluster_idx]

    def to(self, device: str) -> "KMeansPrototypeBank":
        self.protos = self.protos.to(device)
        if self.item_clusters is not None:
            self.item_clusters = self.item_clusters.to(device)
        self.device = device
        return self

    def state_dict(self) -> dict:
        return {
            "protos":        self.protos,
            "item_clusters": self.item_clusters,
            "initialized":   self.initialized,
        }

    def load_state_dict(self, state: dict) -> None:
        self.protos        = state["protos"].to(self.device)
        self.item_clusters = state["item_clusters"].to(self.device) if state["item_clusters"] is not None else None
        self.initialized   = state["initialized"]
