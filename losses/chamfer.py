"""Symmetric Chamfer Distance (squared, mean over points).

CD(A, B) = mean_a min_b ||a - b||^2 + mean_b min_a ||a - b||^2

Unlike `dcd.py` which adds a density-aware penalty, this is the plain
bidirectional Chamfer used as a direct distribution-level regularizer.
Differentiable; complementary to point-wise frame-DSM:
    - frame-DSM aligns paired-point displacements (deterministic)
    - chamfer aligns the cloud distributions (one-to-many tolerant)

For paired data, frame-DSM = 0 implies chamfer = 0; but adding chamfer as
auxiliary helps when paired correspondence is approximate (eg post-noise
NN drift) or when the network output drifts away from the GT manifold.
"""
from __future__ import annotations
import jittor as jt


def chamfer_loss(pred: jt.Var, gt: jt.Var) -> jt.Var:
    """Bidirectional symmetric Chamfer (squared, mean).

    Args:
        pred: (B, N, 3)
        gt:   (B, M, 3)
    Returns:
        scalar loss
    """
    # diff[b,i,j] = pred[b,i] - gt[b,j]  shape (B, N, M, 3)
    diff = pred.unsqueeze(2) - gt.unsqueeze(1)
    d2 = (diff ** 2).sum(-1)               # (B, N, M)
    # min over j (for each pred -> nearest gt)
    d_pg, _ = jt.topk(-d2, k=1, dim=2)     # (B, N, 1)
    d_pg = -d_pg.squeeze(-1)                # (B, N)
    # min over i (for each gt -> nearest pred)
    d_gp, _ = jt.topk(-d2, k=1, dim=1)     # (B, 1, M)
    d_gp = -d_gp.squeeze(1)                 # (B, M)
    return d_pg.mean() + d_gp.mean()
