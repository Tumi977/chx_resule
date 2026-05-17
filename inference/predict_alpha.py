"""Inference-time alpha calibration:
    denoised = noisy + alpha * (raw_denoised - noisy)

This is a post-hoc, training-free correction. Search alpha on the local
validation set, then apply the best one to the testset submission.

Usage:
    # 1) Search alpha on local val
    bash scripts/jt_python.sh inference/predict_alpha.py \\
        --ckpt ckpts/_archive_72.16_p3_mlgc_big/best.pkl \\
        --encoder_kind mlgc --mlgc_growth 128 --mlgc_layers 6 --encoder_dim 384 \\
        --mode search --alphas "0.95,1.00,1.05,1.075,1.10,1.125,1.15,1.20" \\
        --val_subset 100

    # 2) Apply chosen alpha to test set
    bash scripts/jt_python.sh inference/predict_alpha.py \\
        --ckpt ckpts/_archive_72.16_p3_mlgc_big/best.pkl \\
        --encoder_kind mlgc --mlgc_growth 128 --mlgc_layers 6 --encoder_dim 384 \\
        --mode submit --alpha 1.10 \\
        --out_zip outputs/result_alpha1.10.zip
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

    # alpha config
    p.add_argument("--alphas", default="0.95,1.00,1.05,1.075,1.10,1.125,1.15,1.20",
                   help="comma-separated alpha values to search (search mode)")
    p.add_argument("--alpha", type=float, default=1.10,
                   help="single alpha to apply (submit/single mode)")

    # arch
    p.add_argument("--encoder_kind", default="mlgc", choices=["dgcnn", "mlgc", "dense_dgcnn"])
    p.add_argument("--encoder_dim", type=int, default=384)
    p.add_argument("--encoder_k", type=int, default=16)
    p.add_argument("--mlgc_growth", type=int, default=128)
    p.add_argument("--mlgc_layers", type=int, default=6)
    p.add_argument("--dec_hidden", type=int, default=128)

    # patch denoise config
    p.add_argument("--patch_size", type=int, default=1024)
    p.add_argument("--seed_k", type=int, default=6)
    p.add_argument("--batch_patches", type=int, default=8)

    # mode-specific
    p.add_argument("--val_subset", type=int, default=100)
    p.add_argument("--out_zip", default="outputs/result_alpha.zip")
    p.add_argument("--out_dir", default="outputs/alpha_predictions")
    return p.parse_args()


def build_model(args):
    enc_kwargs = {"k": args.encoder_k, "embedding_dim": args.encoder_dim}
    if args.encoder_kind == "mlgc":
        enc_kwargs["growth"] = args.mlgc_growth
        enc_kwargs["n_layers"] = args.mlgc_layers
    m = Denoiser(
        k=args.encoder_k, embedding_dim=args.encoder_dim,
        dec_hidden=args.dec_hidden,
        encoder_kind=args.encoder_kind, encoder_kwargs=enc_kwargs,
    )
    return m


def list_test_models() -> list:
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


def search_alpha(model, val_set, alphas: list, args):
    """Run model ONCE per shape, get raw denoised; then apply each alpha cheaply.

    Trick: since alpha only scales (raw - noisy), we don't need to re-run the
    model for each alpha. Just compute raw once, then test all alphas in
    seconds.
    """
    print(f"\n=== α search on {len(val_set)} validation shapes ===\n")

    # 1) Run model once, store (rel, noisy, clean, raw_denoised)
    print(f"Step 1: computing raw denoised for {len(val_set)} shapes...")
    raw_results = []
    t0 = time.time()
    for i, (rel, pc_noisy, pc_clean, _) in enumerate(val_set):
        pc_pred = patch_based_denoise(
            model, pc_noisy.astype(np.float32),
            patch_size=args.patch_size,
            seed_k=args.seed_k,
            batch_patches=args.batch_patches,
        )
        raw_results.append((rel, pc_noisy, pc_clean, pc_pred))
        if (i + 1) % 10 == 0 or i == 0:
            dt = time.time() - t0
            eta = dt / (i + 1) * (len(val_set) - i - 1)
            print(f"  [{i+1}/{len(val_set)}] dt={dt:.0f}s eta={eta:.0f}s")
    print(f"  -> raw inference total: {time.time()-t0:.1f}s\n")

    # 2) For each alpha, compute denoised = noisy + alpha*(raw - noisy)
    print(f"Step 2: scoring {len(alphas)} alpha values (cheap, no model calls)...\n")

    results_by_alpha = {}
    for alpha in alphas:
        scores = []
        for rel, pc_noisy, pc_clean, pc_raw in raw_results:
            delta = pc_raw - pc_noisy
            pc_pred = pc_noisy + alpha * delta
            r = evaluate_one(rel, pc_noisy, pc_clean, pc_pred)
            scores.append(r)
        agg = aggregate(scores)
        results_by_alpha[alpha] = agg
        print(f"  α={alpha:.4f} | CD={agg['cd_score']:.2f}  P2S={agg['p2s_score']:.2f}  Total={agg['score']:.4f}")

    print("\n=== Summary (sorted by Total) ===")
    sorted_by_score = sorted(results_by_alpha.items(), key=lambda x: -x[1]['score'])
    print(f"{'rank':>4}  {'alpha':>7}  {'CD':>6}  {'P2S':>6}  {'Total':>7}")
    print("  " + "-" * 38)
    for rank, (alpha, agg) in enumerate(sorted_by_score, 1):
        print(f"{rank:>4}  {alpha:>7.4f}  {agg['cd_score']:>6.2f}  {agg['p2s_score']:>6.2f}  {agg['score']:>7.4f}")
    best_alpha = sorted_by_score[0][0]
    best_score = sorted_by_score[0][1]['score']
    print(f"\n  BEST: α={best_alpha:.4f}  Total={best_score:.4f}")
    print(f"  vs raw (α=1.0): Δ = {best_score - results_by_alpha.get(1.00, results_by_alpha.get(1.0))['score']:+.4f}")
    return best_alpha


def submit_with_alpha(model, alpha: float, args):
    items = list_test_models()
    print(f"# test models: {len(items)}, applying α={alpha}")
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
        pc_pred = pc_noisy + alpha * delta
        assert pc_pred.shape == pc_noisy.shape
        target = out_dir / "dataset_test_noisy" / rel / "denoised.npy"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(target), pc_pred.astype(np.float32))
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
    print(f"alpha used: {alpha}")


def main():
    args = parse_args()
    jt.flags.use_cuda = 1
    print(f"Loading ckpt {args.ckpt}")
    model = build_model(args)
    model.load_state_dict(jt.load(args.ckpt))
    model.eval()
    print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M  encoder={args.encoder_kind}")

    if args.mode == "search":
        alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
        vcfg = ValidateConfig(n_pc_val=50000, noise_sigma=0.010)
        val_set = make_validation_set(vcfg)
        val_set = val_set[: args.val_subset]
        print(f"Validation set: {len(val_set)} shapes")
        search_alpha(model, val_set, alphas, args)
    elif args.mode == "submit":
        submit_with_alpha(model, args.alpha, args)
    elif args.mode == "single":
        # single validation pass with one alpha (sanity)
        vcfg = ValidateConfig(n_pc_val=50000, noise_sigma=0.010)
        val_set = make_validation_set(vcfg)[: args.val_subset]
        results = []
        for rel, pc_noisy, pc_clean, _ in val_set:
            pc_raw = patch_based_denoise(model, pc_noisy.astype(np.float32),
                                         patch_size=args.patch_size, seed_k=args.seed_k,
                                         batch_patches=args.batch_patches)
            delta = pc_raw - pc_noisy
            pc_pred = pc_noisy + args.alpha * delta
            results.append(evaluate_one(rel, pc_noisy, pc_clean, pc_pred))
        agg = aggregate(results)
        print(fmt_summary(agg))


if __name__ == "__main__":
    main()
