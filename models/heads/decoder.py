"""MLP displacement decoder (P0). Predicts (B, N, 3) displacement from (B, N, F).
"""
from __future__ import annotations
import jittor as jt
from jittor import nn


class MLPDecoder(nn.Module):
    def __init__(self, in_dim: int = 256, hidden: int = 128, out_dim: int = 3, n_blocks: int = 3):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_blocks):
            layers += [nn.Linear(d, hidden), nn.LeakyReLU(0.2)]
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def execute(self, feat: jt.Var) -> jt.Var:
        # feat: (B, N, F) -> (B, N, 3)
        B, N, F = feat.shape
        out = self.net(feat.reshape(-1, F)).reshape(B, N, -1)
        return out
