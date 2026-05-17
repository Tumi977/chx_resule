"""Density-aware Chamfer Distance (DCD).

Wu et al. "Density-aware Chamfer Distance as a Comprehensive Metric for
Point Cloud Completion", NeurIPS 2021. Directly addresses the many-to-one
collapse problem of standard Chamfer:

    For each pred_i, count how many other pred points are mapped to the
    same gt nearest neighbor. Down-weight contributions from such crowded
    assignments — the network is incentivized to *spread* its predictions.

For two point clouds A=pred ∈ R^{N×3}, B=gt ∈ R^{M×3}:

    For each a_i: j* = argmin_j ||a_i - b_j||^2
    For each b_j: count n_j = #{i : j*(i) = j}
    weight_i = 1 - exp(-alpha * n_{j*(i)} / N)

    L_a→b = mean_i weight_i * (1 - exp(-||a_i - b_{j*(i)}||^2))
    L_b→a = symmetric (count crowdedness from other side)
    L_dcd = (L_a→b + L_b→a) / 2

The (1 - exp(-d^2)) form bounds the loss, making it robust to outliers.
"""
from __future__ import annotations
import jittor as jt


def _knn1_index(query: jt.Var, ref: jt.Var) -> jt.Var:
    """For each point in query, return idx of its nearest in ref. (B,N)."""
    # query: (B, N, 3), ref: (B, M, 3)
    diff = query.unsqueeze(2) - ref.unsqueeze(1)         # (B, N, M, 3)
    dist = (diff ** 2).sum(-1)                           # (B, N, M)
    _, idx = jt.topk(-dist, k=1, dim=-1)                 # (B, N, 1)
    return idx.squeeze(-1)                                # (B, N)


def _gather_per_batch(ref: jt.Var, idx: jt.Var) -> jt.Var:
    """Gather ref points by per-batch index. ref:(B,M,3) idx:(B,N) -> (B,N,3)."""
    idx_exp = idx.unsqueeze(-1).expand(idx.shape[0], idx.shape[1], 3)
    return jt.gather(ref, 1, idx_exp)


def _crowd_count(idx: jt.Var, M: int) -> jt.Var:
    """Count how many query points were assigned to each ref point.

    Returns (B, N) — for each query, the n_j of its assigned ref point.
    Uses bincount via scatter_add.
    """
    B, N = idx.shape
    counts = jt.zeros((B, M))
    ones = jt.ones((B, N))
    counts = counts.scatter_(1, idx, ones, reduce="add")    # (B, M)
    crowd = jt.gather(counts, 1, idx)                       # (B, N)
    return crowd


def dcd_loss(pred: jt.Var, gt: jt.Var, alpha: float = 200.0, n_lambda: float = 1.0) -> jt.Var:
    """Density-aware Chamfer Distance.

    Args:
        pred: (B, N, 3)
        gt:   (B, M, 3)
        alpha: scaling on density term inside the exp; larger = more
               aggressive penalty on crowded assignments. NeurIPS'21
               default α≈200 for unit-sphere scale.
        n_lambda: extra normalizer on n/N before exp. 1.0 keeps the
                  paper's form; smaller relaxes the penalty.
    """
    B, N, _ = pred.shape
    _, M, _ = gt.shape

    # pred -> gt
    j_star = _knn1_index(pred, gt)                          # (B, N)
    nn_pts = _gather_per_batch(gt, j_star)                  # (B, N, 3)
    d2_pg = ((pred - nn_pts) ** 2).sum(-1)                  # (B, N)
    n_j = _crowd_count(j_star, M).float32()                 # (B, N)
    w_pg = 1.0 - jt.exp(-alpha * n_lambda * n_j / N)
    L_pg = (w_pg * (1.0 - jt.exp(-d2_pg))).mean()

    # gt -> pred (symmetric)
    i_star = _knn1_index(gt, pred)                          # (B, M)
    nn_pts2 = _gather_per_batch(pred, i_star)               # (B, M, 3)
    d2_gp = ((gt - nn_pts2) ** 2).sum(-1)
    m_i = _crowd_count(i_star, N).float32()
    w_gp = 1.0 - jt.exp(-alpha * n_lambda * m_i / M)
    L_gp = (w_gp * (1.0 - jt.exp(-d2_gp))).mean()

    return 0.5 * (L_pg + L_gp)
