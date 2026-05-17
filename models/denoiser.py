"""Single-stage denoiser: encoder -> decoder -> displacement.

Encoder backbone is selectable:
    - 'dgcnn': baseline DGCNNEncoder (P0/P1/P2 default)
    - 'mlgc' : MLGC dense-EdgeConv (P3 onward)
"""
from __future__ import annotations
import jittor as jt
from jittor import nn

from .encoders.dgcnn import DGCNNEncoder
from .encoders.mlgc import MLGCEncoder
from .encoders.dense_dgcnn import DenseDGCNNEncoder
from .heads.decoder import MLPDecoder


def build_encoder(kind: str, **kwargs):
    if kind == "dgcnn":
        return DGCNNEncoder(
            k=kwargs.get("k", 16),
            embedding_dim=kwargs.get("embedding_dim", 256),
        )
    if kind == "mlgc":
        return MLGCEncoder(
            n_layers=kwargs.get("n_layers", 4),
            growth=kwargs.get("growth", 24),
            k=kwargs.get("k", 16),
            embedding_dim=kwargs.get("embedding_dim", 256),
            in_proj_dim=kwargs.get("in_proj_dim", 12),
        )
    if kind == "dense_dgcnn":
        return DenseDGCNNEncoder(
            n_layers=kwargs.get("n_layers", 4),
            growth=kwargs.get("growth", 64),
            k=kwargs.get("k", 16),
            embedding_dim=kwargs.get("embedding_dim", 256),
        )
    raise ValueError(f"unknown encoder kind: {kind}")


class Denoiser(nn.Module):
    def __init__(
        self,
        k: int = 16,
        embedding_dim: int = 256,
        dec_hidden: int = 128,
        encoder_kind: str = "dgcnn",
        encoder_kwargs: dict | None = None,
    ):
        super().__init__()
        ek = encoder_kwargs or {}
        ek.setdefault("k", k)
        ek.setdefault("embedding_dim", embedding_dim)
        self.encoder = build_encoder(encoder_kind, **ek)
        # Use the actual encoder embedding_dim (encoder_kwargs may override)
        actual_dim = ek["embedding_dim"]
        self.decoder = MLPDecoder(in_dim=actual_dim, hidden=dec_hidden, out_dim=3)

    def execute(self, x: jt.Var) -> jt.Var:
        feat = self.encoder(x)
        disp = self.decoder(feat)
        return disp
