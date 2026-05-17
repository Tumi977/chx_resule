"""FiLM (Feature-wise Linear Modulation) over a scalar noise level.

Given features f ∈ R^F and a noise level σ ∈ R, produces
    γ(σ), β(σ) ∈ R^F  -- two MLPs from σ
    f' = γ(σ) ⊙ f + β(σ)

Usage:
    film = FiLM(feat_dim=256, hidden=64)
    feat_modulated = film(feat, sigma_per_batch)
    # feat: (B, N, F), sigma: (B,) or (B, 1)
    # output: (B, N, F)
"""
from __future__ import annotations
import jittor as jt
from jittor import nn


class FiLM(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 64):
        super().__init__()
        self.feat_dim = feat_dim
        self.gamma_mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, feat_dim),
        )
        self.beta_mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, feat_dim),
        )
        # init γ near 1 and β near 0 so that initial behavior ≈ identity
        # (last linear: weights small, bias = ones for γ, zeros for β)
        for m in (self.gamma_mlp[-1], self.beta_mlp[-1]):
            nn.init.gauss_(m.weight, 0.0, 1e-3)
        nn.init.constant_(self.gamma_mlp[-1].bias, 1.0)
        nn.init.constant_(self.beta_mlp[-1].bias, 0.0)

    def execute(self, feat: jt.Var, sigma: jt.Var) -> jt.Var:
        """feat: (B, N, F); sigma: (B,) or (B, 1) -> (B, N, F)."""
        B = feat.shape[0]
        s = sigma.reshape(B, 1).float32()                         # (B, 1)
        gamma = self.gamma_mlp(s).reshape(B, 1, self.feat_dim)    # (B, 1, F)
        beta = self.beta_mlp(s).reshape(B, 1, self.feat_dim)
        return feat * gamma + beta
