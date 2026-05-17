"""Patch-based denoising at inference (Bug Fix #2 + #3).

Bug fixes vs starter_code/src/model/vm.py:
  - hard-pick fusion -> Gaussian-weighted soft fusion (radius r/3)
  - guarantee output point count == input point count (no missing points)

Pipeline:
    1) FPS seeds on noisy cloud (n_seeds = ceil(seed_k * N / patch_size))
    2) KNN patch_size around each seed (cKDTree on CPU is fast enough)
    3) center each patch at seed; run model -> patch_denoised (still seed-centered)
    4) un-center; for each original point, accumulate weighted contributions
       from every patch covering it: w = exp(-d2 / (r/3)^2), where d2 is dist
       to its patch's seed and r is the patch radius.
"""
from __future__ import annotations
import math

import jittor as jt
import numpy as np
from scipy.spatial import cKDTree


def fps_seeds_np(points: np.ndarray, n_seeds: int, rng: np.random.Generator | None = None) -> np.ndarray:
    n = points.shape[0]
    if rng is None:
        rng = np.random.default_rng(0)
    seeds = np.empty(n_seeds, dtype=np.int64)
    seeds[0] = rng.integers(0, n)
    dists = np.full(n, np.inf, dtype=np.float32)
    for i in range(n_seeds):
        cur = points[seeds[i - 1]] if i > 0 else points[seeds[0]]
        new_d = ((points - cur) ** 2).sum(axis=1)
        dists = np.minimum(dists, new_d)
        if i + 1 < n_seeds:
            seeds[i + 1] = int(np.argmax(dists))
    return seeds


@jt.no_grad()
def patch_based_denoise(
    model,
    pcl_noisy: np.ndarray,
    patch_size: int = 1024,
    seed_k: int = 6,
    batch_patches: int = 8,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Denoise a single full point cloud (N, 3) -> (N, 3).

    `model` is a callable taking jt.Var (B, M, 3) and returning displacement
    of the same shape.
    """
    assert pcl_noisy.ndim == 2 and pcl_noisy.shape[-1] == 3
    N = pcl_noisy.shape[0]
    n_patches = max(1, int(math.ceil(seed_k * N / patch_size)))

    seed_idx = fps_seeds_np(pcl_noisy, n_patches, rng=rng)
    seeds = pcl_noisy[seed_idx]  # (P, 3)

    tree = cKDTree(pcl_noisy)
    dists, nn_idx = tree.query(seeds, k=patch_size, workers=-1)  # (P, M), (P, M)

    # Gaussian fusion bandwidth = r/3 where r = patch radius (max dist within patch)
    patch_radius = dists[:, -1:]  # (P, 1) Euclidean dist (cKDTree returns Euclidean)
    bandwidth = patch_radius / 3.0  # (P, 1)
    # weights for each (patch, member) point
    weights_pm = np.exp(-(dists / np.clip(bandwidth, 1e-6, None)) ** 2)  # (P, M)

    # Run model on patches in mini-batches
    pat_centered = pcl_noisy[nn_idx] - seeds[:, None, :]  # (P, M, 3)
    disp_all = np.zeros_like(pat_centered)  # (P, M, 3)

    for i in range(0, n_patches, batch_patches):
        chunk = pat_centered[i : i + batch_patches]  # (b, M, 3)
        x = jt.array(chunk.astype(np.float32))
        d = model(x).numpy()  # (b, M, 3)
        disp_all[i : i + batch_patches] = d

    # patch denoised in world coords:
    pat_denoised_world = (pat_centered + disp_all) + seeds[:, None, :]  # (P, M, 3)

    # Aggregate to original N points using soft Gaussian weights.
    # accumulator[k, :] += w_pm * pat_denoised_world[p, m]
    # weights[k] += w_pm  for k = nn_idx[p, m]
    flat_idx = nn_idx.reshape(-1)             # (P*M,)
    flat_w = weights_pm.reshape(-1)           # (P*M,)
    flat_pts = pat_denoised_world.reshape(-1, 3)  # (P*M, 3)

    sum_pts = np.zeros_like(pcl_noisy, dtype=np.float64)  # (N, 3)
    sum_w = np.zeros((N,), dtype=np.float64)
    np.add.at(sum_pts, flat_idx, flat_w[:, None] * flat_pts.astype(np.float64))
    np.add.at(sum_w, flat_idx, flat_w)

    # Bug Fix #3: any uncovered point falls back to noisy-input itself
    uncovered = sum_w < 1e-12
    if uncovered.any():
        sum_pts[uncovered] = pcl_noisy[uncovered].astype(np.float64)
        sum_w[uncovered] = 1.0

    out = (sum_pts / sum_w[:, None]).astype(np.float32)
    assert out.shape == pcl_noisy.shape, f"shape mismatch: {out.shape} vs {pcl_noisy.shape}"
    return out
