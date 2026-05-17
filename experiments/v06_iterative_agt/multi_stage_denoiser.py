"""P2 multi-stage denoiser: T independent ItM modules.

Each ItM is an independent Denoiser (not weight-shared, per IterativePFN
ablation). At inference, modules run in series; at training, AGT supervises
each step independently with σ_τ-scaled targets.

The Denoiser type can be swapped (DGCNN encoder for P2; MLGC encoder for P3).
"""
from __future__ import annotations
from typing import List

import jittor as jt
from jittor import nn

from .denoiser import Denoiser


class MultiStageDenoiser(nn.Module):
    """T independent Denoiser instances; supports per-step σ conditioning later."""

    def __init__(
        self,
        n_stages: int = 4,
        k: int = 16,
        embedding_dim: int = 256,
        dec_hidden: int = 128,
        encoder_kind: str = "dgcnn",
        encoder_kwargs: dict | None = None,
    ):
        super().__init__()
        self.n_stages = n_stages
        self.stages = nn.ModuleList([
            Denoiser(
                k=k,
                embedding_dim=embedding_dim,
                dec_hidden=dec_hidden,
                encoder_kind=encoder_kind,
                encoder_kwargs=encoder_kwargs,
            )
            for _ in range(n_stages)
        ])

    def execute_stage(self, tau: int, x: jt.Var) -> jt.Var:
        """Run module τ on input x. Returns predicted displacement (B, M, 3)."""
        return self.stages[tau](x)

    def execute(self, x: jt.Var) -> List[jt.Var]:
        """Inference: run all T stages in series; return list of intermediates.

        Args:
            x: (B, M, 3) — initial noisy input
        Returns:
            list of (B, M, 3) — [x, x+d_1, ..., x+d_1+...+d_T]
        """
        states = [x]
        cur = x
        for tau in range(self.n_stages):
            d = self.stages[tau](cur)
            cur = cur + d
            states.append(cur)
        return states
