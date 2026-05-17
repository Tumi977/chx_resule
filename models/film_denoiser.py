"""Denoiser variant with FiLM σ-conditioning between encoder and decoder.

Same forward-shape contract as the plain Denoiser:
    execute(x, sigma) -> displacement
"""
from __future__ import annotations
import jittor as jt
from jittor import nn

from .denoiser import build_encoder
from .heads.decoder import MLPDecoder
from .film import FiLM


class FiLMDenoiser(nn.Module):
    def __init__(
        self,
        k: int = 16,
        embedding_dim: int = 256,
        dec_hidden: int = 128,
        encoder_kind: str = "mlgc",
        encoder_kwargs: dict | None = None,
        film_hidden: int = 64,
    ):
        super().__init__()
        ek = encoder_kwargs or {}
        ek.setdefault("k", k)
        ek.setdefault("embedding_dim", embedding_dim)
        self.encoder = build_encoder(encoder_kind, **ek)
        self.film = FiLM(feat_dim=embedding_dim, hidden=film_hidden)
        self.decoder = MLPDecoder(in_dim=embedding_dim, hidden=dec_hidden, out_dim=3)

    def execute(self, x: jt.Var, sigma: jt.Var) -> jt.Var:
        """x: (B, N, 3); sigma: (B,). Returns predicted displacement (B, N, 3)."""
        feat = self.encoder(x)
        feat = self.film(feat, sigma)
        return self.decoder(feat)


class MultiStageFiLMDenoiser(nn.Module):
    """T independent FiLMDenoiser stages with σ conditioning per step."""

    def __init__(
        self,
        n_stages: int = 4,
        k: int = 16,
        embedding_dim: int = 256,
        dec_hidden: int = 128,
        encoder_kind: str = "mlgc",
        encoder_kwargs: dict | None = None,
        film_hidden: int = 64,
    ):
        super().__init__()
        self.n_stages = n_stages
        self.stages = nn.ModuleList([
            FiLMDenoiser(
                k=k, embedding_dim=embedding_dim, dec_hidden=dec_hidden,
                encoder_kind=encoder_kind, encoder_kwargs=encoder_kwargs,
                film_hidden=film_hidden,
            )
            for _ in range(n_stages)
        ])

    def execute_stage(self, tau: int, x: jt.Var, sigma: jt.Var) -> jt.Var:
        return self.stages[tau](x, sigma)
