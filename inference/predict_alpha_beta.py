"""Inference-time (alpha, beta) calibration:

    raw    = model(noisy)              # one forward pass
    delta  = raw - noisy
    base   = noisy + alpha * delta     # alpha-scaled output

    # beta = Laplacian smoothing weight (in [0, 1])
    # smooth_i = mean of base[j] over j in KNN_k(base, i)
    # result = (1 - beta) * base + beta * smooth

This is a pure-postprocessing search:
  - one model forward per shape
  - KNN computed *once per (shape, alpha)* using cKDTree
  - all (alpha, beta) combinations are then cheap arithmetic

Usage:
    bash scripts/jt_python.sh inference/predict_alpha_beta.py \\
        --ckpt ckpts/_archive_72.16_p3_mlgc_big/best.pkl \\
        --encoder_kind mlgc --mlgc_growth 128 --mlgc_layers 6 --encoder_dim 384 \\
        --mode search \\
        --alphas "1.025,1.05,1.075,1.10" \\
        --betas "0.0,0.15,0.30,0.45,0.60" \\
        --knn_k 8 \\
        --val_subset 100

    # Apply best (alpha, beta) to test set
    bash scripts/jt_python.sh inference/predict_alpha_beta.py \\
        --ckpt ckpts/_archive_72.16_p3_mlgc_big/best.pkl \\
        --encoder_kind mlgc --mlgc_growth 128 --mlgc_layers 6 --encoder_dim 384 \\
        --mode submit --alpha 1.075 --beta 0.45 --knn_k 8 \\
        --out_zip outputs/result_a1.075_b0.45.zip
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import jittor as jt
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from models.denoiser import Denoiser
from inference.patch_denoise import patch_based_denoise
from datasets.pc_dataset import ValidateConfig, make_validation_set
from eval.local_eval import evaluate_one, aggregate, fmt_summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--mode", choices=["search", "submit", "single"], default="search")

    p.add_argument("--alphas", default="1.025,1.05,1.075,1.10",
                   help="comma-separated alpha values to search")
    p.add_argument("--betas",  default="0.0,0.15,0.30,0.45,0.60",
                   help="comma-separated beta (Laplacian smoothing weight) values")
    p.add_argument("--knn_k", type=int, default=8,
                   help="K for Laplacian smoothing neighborhood")
    p.add_argument("--alpha", type=float, default=1.075)
    p.add_argument("--beta",  type=float, default=0.45)

    p.add_argument("--encoder_kind", default="mlgc", choices=["dgcnn", "mlgc", "dense_dgcnn"])
    p.add_argument("--encoder_dim", type=int, default=384)
    p.add_argument("--encoder_k", type=int, default=16)
    p.add_argument("--mlgc_growth", type=int, default=128)
    p.add_argument("--mlgc_layers", type=int, default=6)
    p.add_argument("--dec_hidden", type=int, default=128)

    p.add_argument("--patch_size", type=int, default=1024)
    p.add_argument("--seed_k", type=int, default=6)
    p.add_argument("--batch_patches", type=int, default=8)

    p.add_argument("--val_subset", type=int, default=100)
    p.add_argument("--out_zip", default="outputs/result_alpha_beta.zip")
    p.add_argument("--out_dir", default="outputs/alpha_beta_predictions")
    return p.parse_args()


def build_model(args):
    enc_kwargs = {"k": args.encoder_k, "embedding_dim": args.encoder_dim}
    if args.encoder_kind == "mlgc":
        enc_kwargs["growth"] = args.mlgc_growth
        enc_kwargs["n_layers"] = args.mlgc_layers
    return Denoiser(
        k=args.encoder_k, embedding_dim=args.encoder_dim,
        dec_hidden=args.dec_hidden,
        encoder_kind=args.encoder_kind, encoder_kwargs=enc_kwargs,
    )


def laplacian_smooth(points: np.ndarray, k: int) -> np.ndarray:
    """For each point, return the mean of its k nearest neighbors (excluding self).

    Args:
        points: (N, 3) numpy
        k: number of neighbors (excluding self)
    Returns:
        smooth: (N, 3) — mean of K-NN(excluding self)
    """
    n = points.shape[0]
    tree = cKDTree(points)
    # query k+1 because the first NN of each point is itself
    _, idx = tree.query(points, k=k + 1, workers=-1)
    # idx[:, 0] is self; use idx[:, 1:] for K external neighbors
    nbrs = points[idx[:, 1:]]            # (N, k, 3)
    return nbrs.mean(axis=1)              # (N, 3)


def list_test_models():
    DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt/ssd4t/data/chx/graphics")
    TEST_NOISY_ROOT = os.path.join(DATA_ROOT, "dataset_test_noisy", "shapenet")
    out = []
    for syn in sorted(os.listdir(TEST_NOISY_ROOT)):
        d_syn = os.path.join(TEST_NOISY_ROOT, syn)
        if not os.path.isdir(d_syn): continue
        for mdl in sorted(os.listdir(d_syn)):
            npy = os.path.join(d_syn, mdl, "noisy.npy")
            if os.path.exists(npy):
                rel = f"shapenet/{syn}/{mdl}"
                out.append((rel, npy))
    return out


def search_alpha_beta(model, val_set, alphas, betas, knn_k, args):
    """For each (alpha, beta), score on val_set.

    Optimization:
      - run model only ONCE per shape
      - for each alpha, compute base = noisy + alpha*delta + Laplacian smooth
        (the smoothing depends on base, not on beta)
      - for each (alpha, beta), result = (1-beta)*base + beta*smooth (cheap)
    """
    print(f"\n=== α×β search on {len(val_set)} validation shapes ===")
    print(f"alphas={alphas}, betas={betas}, knn_k={knn_k}\n")

    print(f"Step 1: model forward (×{len(val_set)})...")
    raw = []
    t0 = time.time()
    for i, (rel, pc_noisy, pc_clean, _) in enumerate(val_set):
        pc_pred = patch_based_denoise(
            model, pc_noisy.astype(np.float32),
            patch_size=args.patch_size, seed_k=args.seed_k,
            batch_patches=args.batch_patches,
        )
        raw.append((rel, pc_noisy, pc_clean, pc_pred))
        if (i + 1) % 10 == 0 or i == 0:
            dt = time.time() - t0
            eta = dt / (i + 1) * (len(val_set) - i - 1)
            print(f"  [{i+1}/{len(val_set)}] dt={dt:.0f}s eta={eta:.0f}s")
    print(f"  -> raw inference total: {time.time()-t0:.1f}s\n")

    # Pre-compute smooths per (shape, alpha) once
    print(f"Step 2: precomputing Laplacian KNN(k={knn_k}) for each α ({len(alphas)} alphas)...")
    t0 = time.time()
    cache = {}  # (shape_idx, alpha) -> (base, smooth)
    for shape_idx, (rel, pc_noisy, pc_clean, pc_raw) in enumerate(raw):
        delta = pc_raw - pc_noisy
        for alpha in alphas:
            base = pc_noisy + alpha * delta
            smooth = laplacian_smooth(base, k=knn_k)
            cache[(shape_idx, alpha)] = (base, smooth)
        if (shape_idx + 1) % 20 == 0:
            print(f"  [{shape_idx+1}/{len(val_set)}]")
    print(f"  -> KNN precomputation total: {time.time()-t0:.1f}s\n")

    # Score each (alpha, beta)
    print(f"Step 3: scoring {len(alphas)*len(betas)} combinations...")
    grid = {}                                  # (alpha, beta) -> agg
    for alpha in alphas:
        for beta in betas:
            scores = []
            for shape_idx, (rel, pc_noisy, pc_clean, pc_raw) in enumerate(raw):
                base, smooth = cache[(shape_idx, alpha)]
                pc_pred = (1.0 - beta) * base + beta * smooth
                pc_pred = pc_pred.astype(np.float32)
                r = evaluate_one(rel, pc_noisy, pc_clean, pc_pred)
                scores.append(r)
            agg = aggregate(scores)
            grid[(alpha, beta)] = agg
            print(f"  α={alpha:.4f}  β={beta:.4f}  CD={agg['cd_score']:.2f}  P2S={agg['p2s_score']:.2f}  Total={agg['score']:.4f}")

    print("\n=== Summary (sorted by Total, top 10) ===")
    sorted_grid = sorted(grid.items(), key=lambda kv: -kv[1]['score'])
    print(f"{'rank':>4}  {'alpha':>7}  {'beta':>6}  {'CD':>6}  {'P2S':>6}  {'Total':>7}")
    print("  " + "-" * 46)
    for rank, ((a, b), agg) in enumerate(sorted_grid[:10], 1):
        print(f"{rank:>4}  {a:>7.4f}  {b:>6.4f}  {agg['cd_score']:>6.2f}  {agg['p2s_score']:>6.2f}  {agg['score']:>7.4f}")

    best_a, best_b = sorted_grid[0][0]
    best_score = sorted_grid[0][1]['score']
    raw_score = grid.get((1.075, 0.0)) or grid.get((alphas[0], 0.0))
    raw_score_v = raw_score['score'] if raw_score else 0
    print(f"\n  BEST: α={best_a:.4f}  β={best_b:.4f}  Total={best_score:.4f}")
    if (1.0, 0.0) in grid:
        print(f"  vs raw (α=1.0 β=0): Δ = {best_score - grid[(1.0, 0.0)]['score']:+.4f}")
    print(f"  vs α={alphas[0]} β=0:  Δ = {best_score - raw_score_v:+.4f}")
    return best_a, best_b


def submit(model, alpha: float, beta: float, knn_k: int, args):
    items = list_test_models()
    print(f"# test models: {len(items)}, applying α={alpha} β={beta} k={knn_k}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.out_zip).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for i, (rel, npy_path) in enumerate(items):
        pc_noisy = np.load(npy_path).astype(np.float32)
        pc_raw = patch_based_denoise(
            model, pc_noisy,
            patch_size=args.patch_size, seed_k=args.seed_k,
            batch_patches=args.batch_patches,
        )
        delta = pc_raw - pc_noisy
        base = pc_noisy + alpha * delta
        if beta > 0:
            smooth = laplacian_smooth(base.astype(np.float32), k=knn_k)
            pc_pred = (1.0 - beta) * base + beta * smooth
        else:
            pc_pred = base
        pc_pred = pc_pred.astype(np.float32)
        assert pc_pred.shape == pc_noisy.shape
        target = out_dir / "dataset_test_noisy" / rel / "denoised.npy"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(target), pc_pred)
        if (i + 1) % 10 == 0 or i == 0:
            dt = time.time() - t0
            eta = dt / (i + 1) * (len(items) - i - 1)
            print(f"  [{i+1}/{len(items)}] {rel}  dt={dt:.0f}s eta={eta:.0f}s")

    src = out_dir / "dataset_test_noisy"
    print(f"zipping {src} -> {args.out_zip}")
    with zipfile.ZipFile(args.out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in src.rglob("*"):
            if p.is_file():
                zf.write(p, str(p.relative_to(out_dir)))
    print(f"done; zip size = {os.path.getsize(args.out_zip)/1e6:.1f} MB")
    print(f"  used (α, β, k) = ({alpha}, {beta}, {knn_k})")


def main():
    args = parse_args()
    jt.flags.use_cuda = 1
    print(f"Loading ckpt {args.ckpt}")
    model = build_model(args)
    model.load_state_dict(jt.load(args.ckpt))
    model.eval()
    print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M  encoder={args.encoder_kind}\n")

    if args.mode == "search":
        alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
        betas = [float(b) for b in args.betas.split(",") if b.strip()]
        vcfg = ValidateConfig(n_pc_val=50000, noise_sigma=0.010)
        val_set = make_validation_set(vcfg)
        val_set = val_set[: args.val_subset]
        print(f"Validation set: {len(val_set)} shapes")
        search_alpha_beta(model, val_set, alphas, betas, args.knn_k, args)
    elif args.mode == "submit":
        submit(model, args.alpha, args.beta, args.knn_k, args)
    elif args.mode == "single":
        vcfg = ValidateConfig(n_pc_val=50000, noise_sigma=0.010)
        val_set = make_validation_set(vcfg)[: args.val_subset]
        results = []
        for rel, pc_noisy, pc_clean, _ in val_set:
            pc_raw = patch_based_denoise(model, pc_noisy.astype(np.float32),
                                         patch_size=args.patch_size, seed_k=args.seed_k,
                                         batch_patches=args.batch_patches)
            base = pc_noisy + args.alpha * (pc_raw - pc_noisy)
            if args.beta > 0:
                sm = laplacian_smooth(base.astype(np.float32), args.knn_k)
                pred = (1 - args.beta) * base + args.beta * sm
            else:
                pred = base
            results.append(evaluate_one(rel, pc_noisy, pc_clean, pred.astype(np.float32)))
        agg = aggregate(results)
        print(fmt_summary(agg))


if __name__ == "__main__":
    main()
