"""Test-Time Augmentation (TTA) inference, plus iteration count K and seed-ensemble.

Pure inference-time improvements over `patch_based_denoise`:

1. **K iterations**: run model K times in series (each iteration is one full
   patch_based_denoise pass) — StraightPCF's `--niters` mechanic.
   For our σ ∈ [0.005, 0.020], K=2 is the sweet spot.

2. **Rotation TTA**: rotate input by R ∈ {I, R_z(90°), R_z(180°), R_z(270°)},
   denoise each, rotate back, average.
   The model is not rotation-equivariant; averaging removes axis-aligned bias.

3. **Seed ensemble**: run patch_based_denoise with M different FPS-seed RNGs
   and average. Different seeds → different patches → different stitching.

All combine multiplicatively in cost but are independent in theory:
    final = mean over (R, seed, K-th iter) of denoised(R⁻¹ x)

Usage:
    bash scripts/jt_python.sh inference/predict_tta.py \\
        --ckpt ckpts/p1_p2plane_only/best.pkl \\
        --K 2 --rot_tta 4 --seed_ensemble 3 \\
        --val_subset 100  # for local val
        [--testset]       # to run on official 200-shape testset and emit zip
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Callable, List

import numpy as np
import jittor as jt

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from models.denoiser import Denoiser
from inference.patch_denoise import patch_based_denoise
from datasets.pc_dataset import ValidateConfig, make_validation_set
from eval.local_eval import evaluate_one, aggregate, fmt_summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--encoder_kind", default="dgcnn", choices=["dgcnn", "mlgc", "dense_dgcnn"])
    p.add_argument("--encoder_dim", type=int, default=256)
    p.add_argument("--encoder_k", type=int, default=16)
    p.add_argument("--mlgc_growth", type=int, default=24)
    p.add_argument("--mlgc_layers", type=int, default=4)
    p.add_argument("--dense_growth", type=int, default=64)
    p.add_argument("--dense_layers", type=int, default=4)

    # TTA knobs
    p.add_argument("--K", type=int, default=2, help="iterations of full patch denoise")
    p.add_argument("--rot_tta", type=int, default=4,
                   help="rotation TTA: number of z-axis rotations evenly spaced (1/2/4)")
    p.add_argument("--seed_ensemble", type=int, default=1,
                   help="number of different FPS-seed runs to average")
    p.add_argument("--patch_size", type=int, default=1024)
    p.add_argument("--seed_k", type=int, default=6)
    p.add_argument("--batch_patches", type=int, default=8)

    # mode
    p.add_argument("--testset", action="store_true",
                   help="run on official testset and produce result.zip")
    p.add_argument("--val_subset", type=int, default=100)
    p.add_argument("--out_zip", default="outputs/result_tta.zip")
    p.add_argument("--out_dir", default="outputs/tta_predictions")
    return p.parse_args()


def build_model(args):
    enc_kwargs = {"k": args.encoder_k, "embedding_dim": args.encoder_dim}
    if args.encoder_kind == "mlgc":
        enc_kwargs.update({"growth": args.mlgc_growth, "n_layers": args.mlgc_layers})
    elif args.encoder_kind == "dense_dgcnn":
        enc_kwargs.update({"growth": args.dense_growth, "n_layers": args.dense_layers})
    m = Denoiser(
        k=args.encoder_k, embedding_dim=args.encoder_dim,
        encoder_kind=args.encoder_kind, encoder_kwargs=enc_kwargs,
    )
    return m


def rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def denoise_with_tta(model_fwd: Callable, pc_noisy: np.ndarray, args) -> np.ndarray:
    """Combine K iterations + rotation-TTA + seed-ensemble.

    Order of ops (from outer to inner loop):
        for r in rotations:                # rot_tta
            x_rot = R @ pc_noisy
            for s in 0..seed_ensemble-1:   # different FPS seeds
                rng = np.random.default_rng(s)
                cur = x_rot
                for k in 0..K-1:           # K iterations
                    cur = patch_based_denoise(cur, rng=rng_for_this_seed)
                accum_pred[r, s] = R^T @ cur
        out = mean over r,s of accum_pred

    To avoid mistakes that mess up point-correspondence, we never permute points
    or change shape — every transform is a per-point linear/identity map.
    """
    N = pc_noisy.shape[0]

    # rotation set (z-axis only, since most ShapeNet models are upright)
    if args.rot_tta == 1:
        thetas = [0.0]
    elif args.rot_tta == 2:
        thetas = [0.0, np.pi]
    elif args.rot_tta == 4:
        thetas = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
    elif args.rot_tta == 8:
        thetas = [k * np.pi / 4 for k in range(8)]
    else:
        raise ValueError(f"unsupported rot_tta={args.rot_tta}")

    accum = np.zeros_like(pc_noisy, dtype=np.float64)
    n_runs = 0

    for theta in thetas:
        R = rot_z(theta)
        x_rot = pc_noisy @ R.T  # rotate

        for s in range(args.seed_ensemble):
            rng = np.random.default_rng(s + 12345)
            cur = x_rot.astype(np.float32)
            for _k in range(args.K):
                cur = patch_based_denoise(
                    model_fwd, cur,
                    patch_size=args.patch_size,
                    seed_k=args.seed_k,
                    batch_patches=args.batch_patches,
                    rng=rng,
                )
            # rotate back
            cur_back = cur @ R  # since R^T inverts a rotation matrix
            accum += cur_back.astype(np.float64)
            n_runs += 1

    out = (accum / n_runs).astype(np.float32)
    assert out.shape == pc_noisy.shape, f"shape mismatch: {out.shape} vs {pc_noisy.shape}"
    return out


def list_test_models(test_root: str) -> List[tuple]:
    out = []
    for syn in sorted(os.listdir(test_root)):
        d_syn = os.path.join(test_root, syn)
        if not os.path.isdir(d_syn):
            continue
        for mdl in sorted(os.listdir(d_syn)):
            npy = os.path.join(d_syn, mdl, "noisy.npy")
            if os.path.exists(npy):
                rel = f"shapenet/{syn}/{mdl}"
                out.append((rel, npy))
    return out


def main():
    args = parse_args()
    jt.flags.use_cuda = 1
    print(f"loading ckpt: {args.ckpt}")
    print(f"TTA config: K={args.K}, rot_tta={args.rot_tta}, seed_ensemble={args.seed_ensemble}")
    print(f"  -> {args.K * args.rot_tta * args.seed_ensemble} forward passes per shape")

    model = build_model(args)
    model.load_state_dict(jt.load(args.ckpt))
    model.eval()

    if args.testset:
        run_testset(model, args)
    else:
        run_local_val(model, args)


def run_local_val(model, args):
    vcfg = ValidateConfig(n_pc_val=50000, noise_sigma=0.010)
    val_set = make_validation_set(vcfg)
    val_set = val_set[: args.val_subset]
    print(f"# val shapes: {len(val_set)}")

    results = []
    t0 = time.time()
    for i, (rel, pc_noisy, pc_clean, _) in enumerate(val_set):
        pc_pred = denoise_with_tta(model, pc_noisy.astype(np.float32), args)
        r = evaluate_one(rel, pc_noisy, pc_clean, pc_pred)
        results.append(r)
        if (i + 1) % 5 == 0 or i == 0:
            dt = time.time() - t0
            eta = dt / (i + 1) * (len(val_set) - i - 1)
            print(f"  [{i+1}/{len(val_set)}] {rel} score={r['score']:.2f} cd={r['cd_score']:.2f} p2s={r['p2s_score']:.2f} eta={eta/60:.1f}min")
    agg = aggregate(results)
    print("\n" + fmt_summary(agg))
    print(f"TOTAL TIME: {time.time()-t0:.1f}s")


def run_testset(model, args):
    DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt/ssd4t/data/chx/graphics")
    TEST_NOISY_ROOT = os.path.join(DATA_ROOT, "dataset_test_noisy", "shapenet")
    items = list_test_models(TEST_NOISY_ROOT)
    print(f"# test shapes: {len(items)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.out_zip).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for i, (rel, npy_path) in enumerate(items):
        pc_noisy = np.load(npy_path).astype(np.float32)
        pc_pred = denoise_with_tta(model, pc_noisy, args)
        assert pc_pred.shape == pc_noisy.shape, f"shape mismatch on {rel}"
        target = out_dir / "dataset_test_noisy" / rel / "denoised.npy"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(target), pc_pred.astype(np.float32))
        if (i + 1) % 10 == 0 or i == 0:
            dt = time.time() - t0
            eta = dt / (i + 1) * (len(items) - i - 1)
            print(f"  [{i+1}/{len(items)}] {rel} dt={dt:.0f}s eta={eta/60:.1f}min")

    src = out_dir / "dataset_test_noisy"
    print(f"zipping {src} -> {args.out_zip}")
    with zipfile.ZipFile(args.out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in src.rglob("*"):
            if p.is_file():
                zf.write(p, str(p.relative_to(out_dir)))
    print(f"done; zip size = {os.path.getsize(args.out_zip)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
