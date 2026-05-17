"""Data loading for V3 point-cloud denoiser.

Source:
- dataset_train_pc/shapenet/<synset>/<model>/clean.npz
  keys: points (150000,3), normals (150000,3), center (3,), scale ()
  Already normalized to unit sphere.

For training we:
1) randomly subsample N_PC=N_PC_TRAIN points (default 50000) from the 150k
2) gaussian-noise σ ~ U[noise_min, noise_max]
3) extract M patches of patch_size points, each centered at an FPS seed of the
   noisy cloud and KNN-cropped (use cKDTree, much faster than jittor FPS Python loop)

For validation we deterministically subsample 50000 points and add fixed-σ noise.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.spatial import cKDTree

import jittor as jt
from jittor.dataset import Dataset

DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt/ssd4t/data/chx/graphics")
PC_ROOT = os.path.join(DATA_ROOT, "dataset_train_pc", "shapenet")


def load_datalist(name: str) -> List[str]:
    """Read datalist file (relative paths starting with 'shapenet/...')."""
    path = os.path.join(DATA_ROOT, "starter_code", "datalist", f"{name}.txt")
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def load_clean_npz(rel_path: str):
    """rel_path is 'shapenet/<synset>/<model>'. Returns (points, normals)."""
    p = os.path.join(DATA_ROOT, "dataset_train_pc", rel_path, "clean.npz")
    d = np.load(p)
    return d["points"], d["normals"]


def random_sample(points: np.ndarray, normals: np.ndarray, n: int, rng: np.random.Generator):
    if points.shape[0] <= n:
        return points.copy(), normals.copy()
    idx = rng.choice(points.shape[0], n, replace=False)
    return points[idx], normals[idx]


def fps_seeds(points: np.ndarray, n_seeds: int, rng: np.random.Generator) -> np.ndarray:
    """Greedy farthest-point sampling using numpy. Returns seed indices."""
    n = points.shape[0]
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


def extract_patches(
    pts_noisy: np.ndarray,
    pts_clean: np.ndarray,
    normals_clean: np.ndarray,
    n_patches: int,
    patch_size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample n_patches patches via FPS-on-noisy seeds + KNN.

    Returns (P, M, 3) noisy / clean / normals_clean each centered at the seed,
    and (P,) seed indices.
    """
    seed_idx = fps_seeds(pts_noisy, n_patches, rng)
    seeds = pts_noisy[seed_idx]  # (P, 3)
    tree = cKDTree(pts_noisy)
    _, nn_idx = tree.query(seeds, k=patch_size, workers=-1)  # (P, M)
    pat_noisy = pts_noisy[nn_idx]  # (P, M, 3)
    pat_clean = pts_clean[nn_idx]
    pat_normal = normals_clean[nn_idx]
    # center each patch at its seed (in NOISY coords; clean uses same offset)
    pat_noisy_c = pat_noisy - seeds[:, None, :]
    pat_clean_c = pat_clean - seeds[:, None, :]
    return pat_noisy_c, pat_clean_c, pat_normal, seed_idx


@dataclass
class TrainConfig:
    datalist: str = "train"
    n_pc_train: int = 50000  # subsample from 150k
    patch_size: int = 1024
    n_patches: int = 4  # patches per shape
    noise_min: float = 0.005
    noise_max: float = 0.020
    rotate_p: float = 0.5  # random z-rotation prob
    # multi-scale + dropout (for "force-local" data augmentation)
    patch_size_choices: tuple = ()       # if non-empty, override patch_size every epoch
    point_dropout_p: float = 0.0         # 0 disables; otherwise [0, 0.5]
    point_dropout_min: float = 0.10      # min fraction kept-min (so kept ≥ this)
    point_dropout_max: float = 0.30      # max fraction dropped
    # extra geometric augmentation
    scale_p: float = 0.0                 # 0 disables; otherwise prob of scaling
    scale_min: float = 0.9
    scale_max: float = 1.1
    rotate_full_so3: bool = False        # if True, full 3D rotation (else z-only)


@dataclass
class ValidateConfig:
    datalist: str = "validate"
    n_pc_val: int = 50000
    noise_sigma: float = 0.010
    seed_offset: int = 12345  # for reproducible noise


class TrainPatchDataset(Dataset):
    """Per-iteration unit is one 'shape patch-bundle' of n_patches patches.

    Each `__getitem__` returns a dict of jittor tensors:
        pc_noisy:  (P, M, 3)
        pc_clean:  (P, M, 3)
        pc_normal: (P, M, 3)
        sigma:     (P,)
    """

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.items = load_datalist(cfg.datalist)
        # filter out ones missing on disk
        self.items = [
            r for r in self.items
            if os.path.exists(os.path.join(DATA_ROOT, "dataset_train_pc", r, "clean.npz"))
        ]
        self.set_attrs(total_len=len(self.items))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        cfg = self.cfg
        # epoch-aware seed: combines current numpy global state and idx so each
        # __getitem__ is deterministic-given-fork, but varies across workers/epochs.
        rng = np.random.default_rng(np.random.SeedSequence([idx, np.random.randint(0, 2**31 - 1)]))
        pts, nrm = load_clean_npz(self.items[idx])
        pts, nrm = random_sample(pts, nrm, cfg.n_pc_train, rng)

        # optional z-rotation (cheap, keeps unit-sphere)
        if rng.random() < cfg.rotate_p:
            theta = rng.uniform(0, 2 * np.pi)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
            pts = pts @ R.T
            nrm = nrm @ R.T

        # optional isotropic scaling (extra geometry augmentation)
        if cfg.scale_p > 0 and rng.random() < cfg.scale_p:
            s = rng.uniform(cfg.scale_min, cfg.scale_max)
            pts = pts * s
            # normals are unit vectors, scale-invariant; do not change them

        sigma = rng.uniform(cfg.noise_min, cfg.noise_max)
        noise = rng.normal(0, sigma, size=pts.shape).astype(np.float32)
        pts_noisy = pts + noise

        # Pick patch_size: either fixed or random per-shape (collate handles
        # mixed sizes by truncating each sample to the batch-min size).
        if cfg.patch_size_choices:
            patch_size = int(rng.choice(cfg.patch_size_choices))
        else:
            patch_size = cfg.patch_size

        pat_n, pat_c, pat_nrm, _ = extract_patches(
            pts_noisy.astype(np.float32),
            pts.astype(np.float32),
            nrm.astype(np.float32),
            cfg.n_patches,
            patch_size,
            rng,
        )

        # Optional point dropout: drop a random fraction of each patch's points,
        # then re-pad with replacement of remaining points so the tensor shape
        # stays (P, patch_size, 3). The network sees an effectively smaller,
        # "incomplete" patch — strong regularizer for partial-input robustness.
        if cfg.point_dropout_p > 0 and rng.random() < cfg.point_dropout_p:
            keep_frac = 1.0 - rng.uniform(cfg.point_dropout_min, cfg.point_dropout_max)
            n_keep = max(64, int(patch_size * keep_frac))
            for p in range(pat_n.shape[0]):
                keep_idx = rng.choice(patch_size, n_keep, replace=False)
                # repeat to fill back to patch_size
                rep = rng.choice(n_keep, patch_size - n_keep, replace=True)
                full_idx = np.concatenate([keep_idx, keep_idx[rep]])
                rng.shuffle(full_idx)
                pat_n[p] = pat_n[p, full_idx]
                pat_c[p] = pat_c[p, full_idx]
                pat_nrm[p] = pat_nrm[p, full_idx]

        sigma_arr = np.full((cfg.n_patches,), sigma, dtype=np.float32)

        return {
            "pc_noisy": pat_n,        # (P, M, 3)
            "pc_clean": pat_c,        # (P, M, 3)
            "pc_normal": pat_nrm,     # (P, M, 3)
            "sigma": sigma_arr,       # (P,)
        }

    def collate_batch(self, batch):
        # batch: list of dict-of-numpy with shapes (P, M, 3) etc.
        # If patch_size varies across batch entries (multi-scale training),
        # truncate each sample's M to the batch minimum so jt.stack works.
        sizes = [b["pc_noisy"].shape[1] for b in batch]
        m_min = min(sizes)
        out = {}
        for k in batch[0].keys():
            arrs = []
            for b in batch:
                arr = b[k]
                if arr.ndim >= 2 and arr.shape[1] != m_min and k in ("pc_noisy", "pc_clean", "pc_normal"):
                    arr = arr[:, :m_min, :]
                arrs.append(arr)
            arr_stack = np.stack(arrs, axis=0)              # (B, P, ...)
            arr_stack = arr_stack.reshape(-1, *arr_stack.shape[2:])  # (B*P, ...)
            out[k] = jt.array(arr_stack)
        return out


def make_validation_set(cfg: ValidateConfig):
    """Pre-sample once; returns list of (rel_path, pts_noisy, pts_clean, normals)."""
    items = load_datalist(cfg.datalist)
    items = [
        r for r in items
        if os.path.exists(os.path.join(DATA_ROOT, "dataset_train_pc", r, "clean.npz"))
    ]
    out = []
    for i, rel in enumerate(items):
        rng = np.random.default_rng(cfg.seed_offset + i)
        pts, nrm = load_clean_npz(rel)
        pts, nrm = random_sample(pts, nrm, cfg.n_pc_val, rng)
        noise = rng.normal(0, cfg.noise_sigma, size=pts.shape).astype(np.float32)
        pts_noisy = pts + noise
        out.append((rel, pts_noisy.astype(np.float32), pts.astype(np.float32), nrm.astype(np.float32)))
    return out


if __name__ == "__main__":
    print("Train datalist size:", len(load_datalist("train")))
    print("Validate datalist size:", len(load_datalist("validate")))
    cfg = TrainConfig()
    ds = TrainPatchDataset(cfg)
    print("Dataset items:", len(ds))
    item = ds[0]
    for k, v in item.items():
        print(f"  {k}: {v.shape} dtype={v.dtype}")
