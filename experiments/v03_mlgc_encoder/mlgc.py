"""MLGC: Multi-Level Graph Convolution encoder (PD-LTS style).

Reference: Mao et al. "Denoising Point Clouds in Latent Space via Graph
Convolution and Invertible Neural Network", CVPR 2024.

Key idea — "DenseEdgeConv" with skip-concat:
    Each layer's input is the concatenation of the original coordinates and
    *all* previous layers' outputs (DenseNet-style). This drastically reuses
    features across depths and lets the model fuse multi-scale information
    without explicitly running multiple branches with different k.

In PD-LTS, "PointNet-style augmented input" h_a is computed once per shape
to enrich the raw 3D coordinates with global context. We adopt a small,
cheap variant: linear projection (3 -> 12).

Compared to the DGCNN encoder in encoders/dgcnn.py:
    - DGCNN: layer_t input = layer_{t-1} output (no skip, single feature path)
    - MLGC : layer_t input = concat(x, h_1, ..., h_{t-1})  (dense skip)

Output channel = embedding_dim (default 256), same as DGCNNEncoder, so
this module is a drop-in replacement.
"""
from __future__ import annotations
import jittor as jt
from jittor import nn

from .dgcnn import knn_idx, gather_neighbors


class DenseEdgeConvBlock(nn.Module):
    """One EdgeConv layer with channel mixing inside.

    Args:
        in_dim:    input feature dim (passed via dense skip)
        growth:    out channels for this layer
        k:         number of neighbors
        dynamic:   if True, rebuild KNN graph in feature space; else use idx
    """
    def __init__(self, in_dim: int, growth: int, k: int = 16, dynamic: bool = True):
        super().__init__()
        self.k = k
        self.dynamic = dynamic
        self.mlp = nn.Sequential(
            nn.Conv2d(in_dim * 2, growth, 1, bias=False),
            nn.BatchNorm(growth),
            nn.LeakyReLU(0.2),
            nn.Conv2d(growth, growth, 1, bias=False),
            nn.BatchNorm(growth),
            nn.LeakyReLU(0.2),
        )

    def execute(self, x: jt.Var, idx: jt.Var = None):
        """x: (B, N, C). Returns (B, N, growth)."""
        if idx is None or self.dynamic:
            idx = knn_idx(x, self.k)
        nbrs = gather_neighbors(x, idx)              # (B, N, k, C)
        x_exp = x.unsqueeze(2).expand_as(nbrs)
        edge = jt.concat([x_exp, nbrs - x_exp], dim=-1)  # (B, N, k, 2C)
        edge = edge.permute(0, 3, 1, 2)              # (B, 2C, N, k)
        h = self.mlp(edge).max(dim=-1)               # (B, growth, N)
        return h.transpose(0, 2, 1)                  # (B, N, growth)


class MLGCEncoder(nn.Module):
    """PD-LTS-style dense-connected graph encoder.

    Args:
        n_layers:        depth of dense block (default 4, matching DGCNN base)
        growth:          feature width of each layer (default 24, PD-LTS light)
        k:               KNN neighborhood size
        embedding_dim:   final fused feature dim (matches DGCNNEncoder default)
    """
    def __init__(
        self,
        n_layers: int = 4,
        growth: int = 24,
        k: int = 16,
        embedding_dim: int = 256,
        in_proj_dim: int = 12,
    ):
        super().__init__()
        self.k = k

        # Lightweight input enrichment: 3 -> in_proj_dim shared MLP
        self.in_proj = nn.Sequential(
            nn.Linear(3, in_proj_dim),
            nn.LeakyReLU(0.2),
        )

        self.layers = nn.ModuleList()
        in_d = 3 + in_proj_dim
        self.in_dim_seq = []
        for _ in range(n_layers):
            self.in_dim_seq.append(in_d)
            self.layers.append(DenseEdgeConvBlock(in_d, growth, k=k, dynamic=True))
            in_d += growth

        # Fuse all layer outputs (concatenated) into embedding_dim.
        # The total channel count is 3 + in_proj_dim + n_layers * growth.
        total = in_d
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
        """x: (B, N, 3) -> (B, N, embedding_dim)."""
        # initial enriched features (raw + projection)
        proj = self.in_proj(x)                       # (B, N, in_proj_dim)
        h = jt.concat([x, proj], dim=-1)             # (B, N, 3 + p)

        feats = [h]
        for layer in self.layers:
            cat_in = jt.concat(feats, dim=-1)        # (B, N, sum)
            out = layer(cat_in)                      # (B, N, growth)
            feats.append(out)
        cat_all = jt.concat(feats, dim=-1)           # (B, N, total)

        cat_all = cat_all.transpose(0, 2, 1)         # (B, total, N)
        fused = self.fuse(cat_all).transpose(0, 2, 1)  # (B, N, embedding_dim)
        return fused
