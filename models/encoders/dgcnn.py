"""DGCNN/EdgeConv encoder used by P0 baseline.

Faithful re-implementation of the encoder in starter_code (4 dynamic-EdgeConv
layers, k=16, embedding 256), expressed compactly. Inputs (B, N, 3); output
(B, N, embedding_dim).

Later P-stages (P3) will replace this with PD-LTS-style DenseEdgeConv with
dense connections, kept in models/encoders/mlgc.py.
"""
from __future__ import annotations
import jittor as jt
from jittor import nn


def knn_idx(x: jt.Var, k: int) -> jt.Var:
    """KNN by L2 in feature space. x: (B, N, C). Returns idx (B, N, k)."""
    inner = -2 * jt.matmul(x, x.transpose(0, 2, 1))
    sq = (x ** 2).sum(-1, keepdims=True)
    dist = sq + sq.transpose(0, 2, 1) + inner  # (B, N, N)
    _, idx = jt.topk(-dist, k=k, dim=-1)
    return idx


def gather_neighbors(x: jt.Var, idx: jt.Var) -> jt.Var:
    """x: (B, N, C); idx: (B, N, k). Returns (B, N, k, C)."""
    B, N, C = x.shape
    k = idx.shape[-1]
    base = jt.arange(B).reshape(B, 1, 1) * N
    flat_idx = (idx + base).reshape(-1)  # (B*N*k,)
    out = x.reshape(-1, C)[flat_idx].reshape(B, N, k, C)
    return out


class EdgeConv(nn.Module):
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

    def execute(self, x: jt.Var, idx: jt.Var = None):
        # x: (B, N, C). Build dynamic graph in feature space if idx not given.
        if idx is None:
            idx = knn_idx(x, self.k)
        nbrs = gather_neighbors(x, idx)  # (B, N, k, C)
        x_exp = x.unsqueeze(2).expand_as(nbrs)
        edge = jt.concat([x_exp, nbrs - x_exp], dim=-1)  # (B, N, k, 2C)
        # to (B, 2C, N, k) for Conv2d
        edge = edge.permute(0, 3, 1, 2)
        h = self.mlp(edge)  # (B, out, N, k)
        h = h.max(dim=-1)  # (B, out, N)
        return h.transpose(0, 2, 1)  # (B, N, out)


class DGCNNEncoder(nn.Module):
    """Starter-style encoder: 4 dynamic EdgeConv layers + concat + 1x1 conv.

    Output channel = embedding_dim.
    """

    def __init__(self, k: int = 16, dims=(64, 64, 128, 256), embedding_dim: int = 256):
        super().__init__()
        self.k = k
        self.layers = nn.ModuleList()
        in_d = 3
        for d in dims:
            self.layers.append(EdgeConv(in_d, d, k=k))
            in_d = d
        self.fuse = nn.Sequential(
            nn.Conv1d(sum(dims), embedding_dim, 1, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.LeakyReLU(0.2),
        )
        self.embedding_dim = embedding_dim

    def execute(self, x: jt.Var) -> jt.Var:
        feats = []
        h = x
        for layer in self.layers:
            h = layer(h)  # dynamic KNN inside each layer
            feats.append(h)
        cat = jt.concat(feats, dim=-1)  # (B, N, sum_dims)
        cat = cat.transpose(0, 2, 1)
        out = self.fuse(cat).transpose(0, 2, 1)  # (B, N, embedding_dim)
        return out
