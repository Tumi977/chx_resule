"""Run inference on the official test set and produce result.zip in the
required submission format.

Submission layout (per official PDF p4):

  result.zip
    └── dataset_test_noisy/
          └── shapenet/
                └── <synset_id>/
                      └── <model_id>/
                            └── denoised.npy   # shape (N, 3) float32, N == input N

Usage:
    bash scripts/jt_python.sh inference/predict_testset.py \
        --ckpt ckpts/p2_agt_multistage/best.pkl \
        --out_zip outputs/results_p2.zip \
        [--multi_stage --n_stages 4]
        [--encoder_kind mlgc --growth 32]
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
from models.multi_stage_denoiser import MultiStageDenoiser
from inference.patch_denoise import patch_based_denoise


DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt/ssd4t/data/chx/graphics")
TEST_NOISY_ROOT = os.path.join(DATA_ROOT, "dataset_test_noisy", "shapenet")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out_dir", default="outputs/test_predictions")
    p.add_argument("--out_zip", default="outputs/result.zip")
    p.add_argument("--patch_size", type=int, default=1024)
    p.add_argument("--batch_patches", type=int, default=8)
    p.add_argument("--seed_k", type=int, default=6)
    p.add_argument("--multi_stage", action="store_true",
                   help="ckpt is MultiStageDenoiser; otherwise single Denoiser")
    p.add_argument("--n_stages", type=int, default=4)
    p.add_argument("--encoder_kind", default="dgcnn", choices=["dgcnn", "mlgc"])
    p.add_argument("--encoder_dim", type=int, default=256)
    p.add_argument("--encoder_k", type=int, default=16)
    p.add_argument("--mlgc_growth", type=int, default=24)
    p.add_argument("--mlgc_layers", type=int, default=4)
    return p.parse_args()


def list_test_models():
    """Returns list of (rel_path 'shapenet/<syn>/<model>', noisy_npy_path)."""
    out = []
    for syn in sorted(os.listdir(TEST_NOISY_ROOT)):
        d_syn = os.path.join(TEST_NOISY_ROOT, syn)
        if not os.path.isdir(d_syn):
            continue
        for mdl in sorted(os.listdir(d_syn)):
            d_mdl = os.path.join(d_syn, mdl)
            npy = os.path.join(d_mdl, "noisy.npy")
            if os.path.exists(npy):
                rel = f"shapenet/{syn}/{mdl}"
                out.append((rel, npy))
    return out


def build_model(args):
    enc_kwargs = {
        "k": args.encoder_k,
        "embedding_dim": args.encoder_dim,
    }
    if args.encoder_kind == "mlgc":
        enc_kwargs["growth"] = args.mlgc_growth
        enc_kwargs["n_layers"] = args.mlgc_layers
    if args.multi_stage:
        m = MultiStageDenoiser(
            n_stages=args.n_stages,
            k=args.encoder_k,
            embedding_dim=args.encoder_dim,
            encoder_kind=args.encoder_kind,
            encoder_kwargs=enc_kwargs,
        )
    else:
        m = Denoiser(
            k=args.encoder_k,
            embedding_dim=args.encoder_dim,
            encoder_kind=args.encoder_kind,
            encoder_kwargs=enc_kwargs,
        )
    return m


def make_callable(model, multi_stage: bool, n_stages: int):
    if not multi_stage:
        return model
    @jt.no_grad()
    def f(x):
        cur = x
        for tau in range(n_stages):
            d = model.execute_stage(tau, cur)
            cur = cur + d
        return cur - x
    return f


def main():
    args = parse_args()
    jt.flags.use_cuda = 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.out_zip).parent.mkdir(parents=True, exist_ok=True)

    print(f"loading ckpt: {args.ckpt}")
    model = build_model(args)
    sd = jt.load(args.ckpt)
    model.load_state_dict(sd)
    model.eval()

    f = make_callable(model, args.multi_stage, args.n_stages)

    items = list_test_models()
    print(f"# test models: {len(items)}")

    t0 = time.time()
    for i, (rel, npy_path) in enumerate(items):
        pc_noisy = np.load(npy_path).astype(np.float32)
        pc_pred = patch_based_denoise(
            f, pc_noisy,
            patch_size=args.patch_size,
            seed_k=args.seed_k,
            batch_patches=args.batch_patches,
        )
        # strict alignment guard
        assert pc_pred.shape == pc_noisy.shape, f"shape mismatch on {rel}: {pc_pred.shape} vs {pc_noisy.shape}"
        out_path = out_dir / "dataset_test_noisy" / rel.replace("shapenet/", "shapenet/")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # actually we want dataset_test_noisy/shapenet/<syn>/<model>/denoised.npy
        target = out_dir / "dataset_test_noisy" / rel / "denoised.npy"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(target), pc_pred.astype(np.float32))
        if (i + 1) % 10 == 0 or i == 0:
            dt = time.time() - t0
            eta = dt / (i + 1) * (len(items) - i - 1)
            print(f"[{i+1}/{len(items)}] {rel}  shape={pc_pred.shape}  dt={dt:.1f}s eta={eta:.1f}s")

    # zip everything under out_dir/dataset_test_noisy/...
    src_root = out_dir / "dataset_test_noisy"
    print(f"zipping {src_root} -> {args.out_zip}")
    with zipfile.ZipFile(args.out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in src_root.rglob("*"):
            if p.is_file():
                arcname = p.relative_to(out_dir)  # keeps 'dataset_test_noisy/...' prefix
                zf.write(p, str(arcname))
    print(f"done. result zip: {args.out_zip}, size = {os.path.getsize(args.out_zip)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
