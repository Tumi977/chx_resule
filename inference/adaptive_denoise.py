"""σ-adaptive inference for multi-stage denoisers.

Idea (cf. Adaptive Score-Based Denoising, CGF 2025):
    Run the FIRST stage as a probe; the displacement-norm distribution
    encodes how much noise is left in each region. Use its variance to
    estimate the input noise level σ̂, then pick how many further stages
    to actually run.

    Mapping (heuristic, calibrated on validation):
        σ̂ < 0.006   -> n_iter = 1
        0.006-0.012  -> n_iter = 2
        0.012-0.018  -> n_iter = 3
        > 0.018      -> n_iter = 4

This is a pure-inference change; training is unchanged.
"""
from __future__ import annotations
import math
import numpy as np
import jittor as jt
from scipy.spatial import cKDTree

from .patch_denoise import fps_seeds_np


def estimate_sigma_from_disp(displacement: np.ndarray) -> float:
    """displacement: (M, 3). Returns σ̂ ~= sqrt(E[|d|^2]/3) ~ Gaussian noise std."""
    sq = (displacement ** 2).sum(axis=1)  # (M,)
    return float(np.sqrt(sq.mean() / 3.0))


def adaptive_n_iter(sigma_hat: float, breakpoints=(0.006, 0.012, 0.018)) -> int:
    if sigma_hat < breakpoints[0]:
        return 1
    if sigma_hat < breakpoints[1]:
        return 2
    if sigma_hat < breakpoints[2]:
        return 3
    return 4


@jt.no_grad()
def adaptive_patch_denoise(
    multistage_model,
    pcl_noisy: np.ndarray,
    n_stages_max: int = 4,
    patch_size: int = 1024,
    seed_k: int = 6,
    batch_patches: int = 8,
    use_film: bool = False,
    sigma_for_film: float = None,   # if known; else estimated globally
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """Patch-based denoise with σ-adaptive iteration count.

    Returns (denoised_pc, info_dict).
    """
    assert pcl_noisy.ndim == 2 and pcl_noisy.shape[-1] == 3
    N = pcl_noisy.shape[0]
    n_patches = max(1, int(math.ceil(seed_k * N / patch_size)))

    if rng is None:
        rng = np.random.default_rng(0)

    seed_idx = fps_seeds_np(pcl_noisy, n_patches, rng=rng)
    seeds = pcl_noisy[seed_idx]
    tree = cKDTree(pcl_noisy)
    dists, nn_idx = tree.query(seeds, k=patch_size, workers=-1)

    bandwidth = (dists[:, -1:] / 3.0).clip(1e-6, None)
    weights_pm = np.exp(-(dists / bandwidth) ** 2)

    pat_centered = pcl_noisy[nn_idx] - seeds[:, None, :]   # (P, M, 3)

    # ------- step 1: probe with stage 0 only, estimate σ̂ -------
    disp_step1 = np.zeros_like(pat_centered)
    for i in range(0, n_patches, batch_patches):
        chunk = pat_centered[i : i + batch_patches]
        x = jt.array(chunk.astype(np.float32))
        if use_film:
            sig_init = float(sigma_for_film) if sigma_for_film else 0.01
            B = x.shape[0]
            sigma_jt = jt.array(np.full(B, sig_init, dtype=np.float32))
            d = multistage_model.execute_stage(0, x, sigma_jt).numpy()
        else:
            d = multistage_model.execute_stage(0, x).numpy()
        disp_step1[i : i + batch_patches] = d

    sigma_hat = estimate_sigma_from_disp(disp_step1.reshape(-1, 3))
    n_iter = min(adaptive_n_iter(sigma_hat), n_stages_max)

    # ------- continue with stages [1..n_iter-1] -------
    cur = pat_centered + disp_step1   # already applied stage 0
    for tau in range(1, n_iter):
        for i in range(0, n_patches, batch_patches):
            chunk = cur[i : i + batch_patches]
            x = jt.array(chunk.astype(np.float32))
            if use_film:
                B = x.shape[0]
                # rough σ at this stage assuming geometric decay δ=2
                sig_t = max(1e-4, sigma_hat / (2.0 ** tau))
                sigma_jt = jt.array(np.full(B, sig_t, dtype=np.float32))
                d = multistage_model.execute_stage(tau, x, sigma_jt).numpy()
            else:
                d = multistage_model.execute_stage(tau, x).numpy()
            cur[i : i + batch_patches] = chunk + d

    pat_denoised_world = cur + seeds[:, None, :]

    # ------- gaussian fusion to N points -------
    flat_idx = nn_idx.reshape(-1)
    flat_w = weights_pm.reshape(-1)
    flat_pts = pat_denoised_world.reshape(-1, 3)

    sum_pts = np.zeros_like(pcl_noisy, dtype=np.float64)
    sum_w = np.zeros((N,), dtype=np.float64)
    np.add.at(sum_pts, flat_idx, flat_w[:, None] * flat_pts.astype(np.float64))
    np.add.at(sum_w, flat_idx, flat_w)

    uncovered = sum_w < 1e-12
    if uncovered.any():
        sum_pts[uncovered] = pcl_noisy[uncovered].astype(np.float64)
        sum_w[uncovered] = 1.0

    out = (sum_pts / sum_w[:, None]).astype(np.float32)
    info = {"sigma_hat": sigma_hat, "n_iter": n_iter}
    return out, info
