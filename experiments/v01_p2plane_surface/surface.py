"""P1 surface-aligned losses (point-to-plane + repulsion).

These complement the frame-DSM loss in P0 by adding two surface-quality terms:

    L_p2plane = mean( ((pred - clean_NN) · normal_clean_NN)^2 )
        Penalizes the *normal* component of the residual error — directly
        proxies the official P2S metric (point-to-mesh-surface distance).
        Tangential drift along the surface is unpenalized, so the network is
        free to redistribute points on the manifold.

    L_repulsion = mean( exp(-||x_i - x_j||^2 / h^2) ) for j in KNN_k(i)
        Discourages point clustering after denoising, which would manifest as
        a spike in CD (multi-to-one collapse).
"""
from __future__ import annotations
import jittor as jt


def p2plane_loss(
    pred_disp: jt.Var,
    pc_noisy: jt.Var,
    pc_clean: jt.Var,
    normals_clean: jt.Var,
) -> jt.Var:
    """Point-to-plane (P2S surrogate) loss.

    Since pc_noisy[i] is point-wise paired with pc_clean[i], the closest GT
    point for the predicted location is, in the limit of small noise, exactly
    pc_clean[i]. We therefore project the *predicted residual* onto the GT
    surface normal at i:

        residual_i  = (pc_noisy[i] + pred_disp[i]) - pc_clean[i]
        L          = mean( (residual_i . normal_i)^2 )

    Args:
        pred_disp:    (B, M, 3)
        pc_noisy:     (B, M, 3)
        pc_clean:     (B, M, 3)
        normals_clean:(B, M, 3) per-point unit normals at pc_clean[i]
    """
    pred_pos = pc_noisy + pred_disp
    residual = pred_pos - pc_clean                     # (B, M, 3)
    n = normals_clean                                  # (B, M, 3) assumed unit
    proj = (residual * n).sum(dim=-1)                  # (B, M)
    return (proj ** 2).mean()


def repulsion_loss(
    pred_pos: jt.Var,
    k: int = 5,
    h: float = 0.03,
) -> jt.Var:
    """Encourage uniform spread among predicted points.

    For each point, pull-down RBF energy with its k nearest neighbors:
        E = mean_i mean_{j in KNN_k(i)} exp(-||x_i - x_j||^2 / h^2)
    Smaller h = stronger short-range repulsion. h = 0.03 ~ 1.5x typical
    spacing in our patch (50000 pts / unit-sphere -> ~0.02 spacing).

    Args:
        pred_pos: (B, M, 3) predicted denoised positions
        k:        number of nearest neighbors to include (excluding self)
        h:        bandwidth
    """
    B, M, _ = pred_pos.shape
    diff = pred_pos.unsqueeze(2) - pred_pos.unsqueeze(1)        # (B, M, M, 3)
    d2 = (diff ** 2).sum(-1)                                    # (B, M, M)
    # exclude self by setting diagonal to large
    eye = jt.init.eye(M).unsqueeze(0).broadcast([B, M, M]) * 1e6
    d2 = d2 + eye
    # k smallest distances (negate for topk=largest of negatives)
    nb_d2, _ = jt.topk(-d2, k=k, dim=-1)                        # (B, M, k)
    nb_d2 = -nb_d2
    energy = jt.exp(-nb_d2 / (h * h))                           # (B, M, k)
    return energy.mean()
