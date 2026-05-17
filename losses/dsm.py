"""Frame-level DSM loss (Score-Denoise Luo & Hu ICCV21 style) -- Bug Fix #1.

The starter computes its loss only on `num_train_points` randomly chosen anchor
points, throwing away the score signal at all other points in the patch. The
original Score-Denoise paper computes it on a *frame* of `frame_k` neighbors
around each of `n_anchor` anchors, supervising n_anchor*frame_k point displacements.
This raises the supervision density by ~32x at no extra forward-pass cost.

For a supervised setting where pc_noisy and pc_clean are point-wise paired
(our dataset is — index i in pc_noisy corresponds to index i in pc_clean):

    target_disp[i] = mean(clean_k nearest pc_clean points around pc_noisy[i]) - pc_noisy[i]

Because the pairing is exact, the simplest form (clean_k = 1) gives
target_disp[i] = pc_clean[i] - pc_noisy[i]. We average over a small clean_k > 1
to smooth out individual point noise — exactly as the original paper.
"""
from __future__ import annotations
import jittor as jt


def frame_dsm_loss(
    pred_disp: jt.Var,
    pc_noisy: jt.Var,
    pc_clean: jt.Var,
    n_anchor: int = 128,
    frame_k: int = 32,
    clean_k: int = 4,
    dsm_sigma: float = 0.01,
) -> jt.Var:
    """Compute frame-level DSM loss.

    Args:
        pred_disp: (B, M, 3) predicted displacement at every patch point
        pc_noisy:  (B, M, 3)
        pc_clean:  (B, M, 3) point-wise paired with pc_noisy
    """
    B, M, _ = pc_noisy.shape

    # 1. anchors
    anchor_idx = jt.randperm(M)[:n_anchor]                     # (n_anchor,)

    # 2. frame_k neighbors of each anchor in pc_noisy (Euclidean)
    anchors = pc_noisy[:, anchor_idx, :]                       # (B, n_anchor, 3)
    diff = anchors.unsqueeze(2) - pc_noisy.unsqueeze(1)        # (B, n_anchor, M, 3)
    dist = (diff ** 2).sum(-1)                                 # (B, n_anchor, M)
    _, frame_idx = jt.topk(-dist, k=frame_k, dim=-1)           # (B, n_anchor, frame_k)

    # gather frame points: pred_disp and pc_noisy at frame indices
    flat_fi = frame_idx.reshape(B, n_anchor * frame_k)         # (B, n_anchor*frame_k)
    flat_fi_expanded = flat_fi.unsqueeze(-1).expand(B, n_anchor * frame_k, 3)

    frame_pts_noisy = jt.gather(pc_noisy, 1, flat_fi_expanded) # (B, n_anchor*frame_k, 3)
    frame_disp_pred = jt.gather(pred_disp, 1, flat_fi_expanded)

    # 3. for each frame point, find clean_k nearest in pc_clean
    diff_c = frame_pts_noisy.unsqueeze(2) - pc_clean.unsqueeze(1)  # (B, F, M, 3) where F=n_anchor*frame_k
    dist_c = (diff_c ** 2).sum(-1)                                  # (B, F, M)
    _, clean_idx = jt.topk(-dist_c, k=clean_k, dim=-1)              # (B, F, clean_k)

    F = n_anchor * frame_k
    flat_ci = clean_idx.reshape(B, F * clean_k).unsqueeze(-1).expand(B, F * clean_k, 3)
    clean_nbrs = jt.gather(pc_clean, 1, flat_ci).reshape(B, F, clean_k, 3)
    clean_mean = clean_nbrs.mean(dim=2)                              # (B, F, 3)

    target = clean_mean - frame_pts_noisy                            # (B, F, 3)
    diff_pred = frame_disp_pred - target
    loss = (diff_pred ** 2).sum(-1).mean() / dsm_sigma
    return loss


def simple_displacement_loss(
    pred_disp: jt.Var, pc_noisy: jt.Var, pc_clean: jt.Var
) -> jt.Var:
    target = pc_clean - pc_noisy
    return ((pred_disp - target) ** 2).sum(-1).mean()
