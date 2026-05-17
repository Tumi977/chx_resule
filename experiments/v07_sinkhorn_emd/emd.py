"""Sinkhorn EMD loss — Earth Mover's Distance via Sinkhorn iteration.

Problem with Chamfer:
    CD(A,B) = mean_i min_j ||a_i - b_j||² + mean_j min_i ||a_i - b_j||²
    Allows many-to-one assignment: multiple a points can map to the same b
    point (point cloud collapse) without being penalized.

EMD addresses this with a one-to-one assignment:
    EMD(A,B) = min_{π: A→B bijection} mean_i ||a_i - π(a_i)||²
    Used as the main supervision in PD-LTS (CVPR'24) which achieves SOTA on
    PU-Net 50K @ 1-2% noise.

Exact EMD requires the Hungarian algorithm (O(N³)) which is intractable.
Sinkhorn approximates it with entropic regularization:

    Sinkhorn(A,B; ε) = min_{π: doubly stochastic} <π, C> + ε·H(π)

where C[i,j] = ||a_i - b_j||² is the cost matrix and H(π) is entropy.
Standard Sinkhorn alternates row/column normalization on K = exp(-C/ε):

    a_iter ←  μ / (K · b_iter)        (μ, ν are marginals = uniform)
    b_iter ← ν / (K^T · a_iter)

After T iterations, π = diag(a) · K · diag(b) approximates the optimal
transport plan, and <π, C> approximates EMD.

Implementation notes:
    - We use log-domain Sinkhorn for numerical stability with small ε.
    - We choose ε ≈ 0.01 (relative to unit-sphere scale).
    - 50 iterations is sufficient for our patch_size=1024 (well-converged).
"""
from __future__ import annotations
import jittor as jt


def _logsumexp(x: jt.Var, dim: int) -> jt.Var:
    """Numerically stable log-sum-exp; reduces given dim, returns shape with that dim removed."""
    x_max = x.max(dim=dim, keepdims=True)
    out = jt.log((x - x_max).exp().sum(dim=dim, keepdims=True) + 1e-30) + x_max
    # squeeze the reduced dim manually (jittor squeeze only works on size-1 dims, which is what we have)
    shape = list(out.shape)
    shape.pop(dim)
    return out.reshape(shape)


def sinkhorn_emd(
    pred: jt.Var,
    gt: jt.Var,
    eps: float = 0.01,
    n_iters: int = 50,
) -> jt.Var:
    """Differentiable Sinkhorn-approximated EMD.

    Args:
        pred: (B, N, 3)
        gt:   (B, N, 3) — same N (uniform marginals require N == M)
        eps:  entropic regularization strength; smaller = closer to true EMD
              but slower to converge and less stable
        n_iters: Sinkhorn iterations
    Returns:
        scalar mean EMD over batch
    """
    B, N, _ = pred.shape
    M = gt.shape[1]
    assert N == M, f"Sinkhorn EMD requires same N; got {N} vs {M}"

    # Cost matrix C[b, i, j] = ||pred_i - gt_j||^2
    diff = pred.unsqueeze(2) - gt.unsqueeze(1)            # (B, N, M, 3)
    C = (diff ** 2).sum(-1)                                # (B, N, M)

    # Log-domain Sinkhorn for numerical stability
    log_K = -C / eps                                       # (B, N, M)

    # Uniform marginals
    log_mu = float(-jt.log(jt.array(float(N))).item())     # scalar log(1/N)
    log_nu = float(-jt.log(jt.array(float(M))).item())

    log_a = jt.zeros((B, N))                               # init
    log_b = jt.zeros((B, M))

    for _ in range(n_iters):
        # log_a[b,i] = log_mu - logsumexp_j(log_K[b,i,j] + log_b[b,j])
        log_a = log_mu - _logsumexp(log_K + log_b.unsqueeze(1), dim=2)  # reduce j -> shape (B,N)
        # log_b[b,j] = log_nu - logsumexp_i(log_K[b,i,j] + log_a[b,i])
        log_b = log_nu - _logsumexp(log_K + log_a.unsqueeze(2), dim=1)  # reduce i -> shape (B,M)

    # Transport plan π = exp(log_a + log_K + log_b)
    log_pi = log_a.unsqueeze(2) + log_K + log_b.unsqueeze(1)  # (B, N, M)
    pi = log_pi.exp()

    # EMD ≈ Σ_ij π_ij · C_ij  (mean over all entries -> per-pair contribution,
    # then we want sum over i,j and mean over batch.)
    cost_per = (pi * C)                                    # (B, N, M)
    # sum over (N, M) = (B,)
    emd_per = cost_per.sum(dims=(1, 2))                    # (B,)
    return emd_per.mean()


def sinkhorn_emd_lite(
    pred: jt.Var,
    gt: jt.Var,
    eps: float = 0.05,
    n_iters: int = 20,
) -> jt.Var:
    """Lite version with larger eps + fewer iters; faster but less accurate.

    Use for warm-up or as a regularizer, not as main loss.
    """
    return sinkhorn_emd(pred, gt, eps=eps, n_iters=n_iters)
