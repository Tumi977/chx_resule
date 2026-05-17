"""DenseDGCNN encoder: DGCNN with skip-concat between layers.

Compared to the plain stacked DGCNN encoder:
    layer t input = concat(x, h_1, ..., h_{t-1})  (DenseNet-style)
    final feature = concat(h_1, h_2, h_3, h_4) -> Conv1d -> embedding_dim

This was shown by Mao et al. (PD-LTS, CVPR'24) to give better multi-scale
behavior than plain stacked EdgeConv at similar parameter budget.

Drop-in replacement for DGCNNEncoder — same execute(x) -> (B, N, embedding_dim)
shape contract.
"""
from __future__ import annotations
import jittor as jt
from jittor import nn

from .dgcnn import knn_idx, gather_neighbors


class DenseEdgeConvLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, k: int = 16):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv2d(in_dim * 2, out_dim, 1, bias=False),
            nn.BatchNorm(out_dim),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_dim, out_dim, 1, bias=False),
            nn.BatchNorm(out_dim),
            nn.LeakyReLU(0.2),
        )

    def execute(self, x: jt.Var, idx: jt.Var = None) -> jt.Var:
        """x: (B, N, C) -> (B, N, out)."""
        if idx is None:
            idx = knn_idx(x, self.k)
        nbrs = gather_neighbors(x, idx)                  # (B, N, k, C)
        x_exp = x.unsqueeze(2).expand_as(nbrs)
        edge = jt.concat([x_exp, nbrs - x_exp], dim=-1)  # (B, N, k, 2C)
        edge = edge.permute(0, 3, 1, 2)                  # (B, 2C, N, k)
        h = self.mlp(edge).max(dim=-1)                    # (B, out, N)
        return h.transpose(0, 2, 1)                      # (B, N, out)


class DenseDGCNNEncoder(nn.Module):
    """DGCNN with dense skip connections.

    Default 4 layers each width=growth=64 -> total feat=256, fused to
    embedding_dim. A bit beefier than plain DGCNN (~0.7M vs 0.38M params)
    but should learn richer geometry.
    """
    def __init__(
        self,
        n_layers: int = 4,
        growth: int = 64,
        k: int = 16,
        embedding_dim: int = 256,
    ):
        super().__init__()
        self.k = k

        self.layers = nn.ModuleList()
        in_d = 3
        for _ in range(n_layers):
            self.layers.append(DenseEdgeConvLayer(in_d, growth, k=k))
            in_d += growth

        total = in_d                                     # 3 + n_layers * growth
        self.fuse = nn.Sequential(
            nn.Conv1d(total, embedding_dim, 1, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.LeakyReLU(0.2),
            nn.Conv1d(embedding_dim, embedding_dim, 1, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.LeakyReLU(0.2),
        )
        self.embedding_dim = embedding_dim

    def execute(self, x: jt.Var) -> jt.Var:
        feats = [x]
        for layer in self.layers:
            cat_in = jt.concat(feats, dim=-1)            # dense skip
            out = layer(cat_in)                          # (B, N, growth) — fresh KNN every layer
            feats.append(out)
        cat_all = jt.concat(feats, dim=-1)               # (B, N, total)
        cat_all = cat_all.transpose(0, 2, 1)             # (B, total, N)
        return self.fuse(cat_all).transpose(0, 2, 1)     # (B, N, embedding_dim)
