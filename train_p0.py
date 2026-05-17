"""P0 training entry.

Goals (P0):
- Establish working pipeline end-to-end
- Apply 3 bug-fixes vs starter:
    #1 frame-level DSM loss instead of degenerate per-anchor loss
    #2 Gaussian soft fusion at inference (in inference/patch_denoise.py)
    #3 strict point-count alignment (in inference/patch_denoise.py)
- Establish local-validation anchor score
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import jittor as jt
import numpy as np
from jittor import nn

# Make project package imports work when called as `python train_p0.py` directly.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from datasets.pc_dataset import TrainConfig, TrainPatchDataset, ValidateConfig, make_validation_set
from models.denoiser import Denoiser
from losses.dsm import frame_dsm_loss, simple_displacement_loss
from inference.patch_denoise import patch_based_denoise
from eval.local_eval import evaluate_one, aggregate, fmt_summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", default="p0_baseline")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=4)  # patches per step actually = batch_size * n_patches
    p.add_argument("--n_patches", type=int, default=4)
    p.add_argument("--patch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--noise_min", type=float, default=0.005)
    p.add_argument("--noise_max", type=float, default=0.020)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--use_simple_loss", action="store_true",
                   help="use plain displacement L2 instead of frame-DSM (debug/ablation)")
    p.add_argument("--frame_n_anchor", type=int, default=128)
    p.add_argument("--frame_k", type=int, default=32)
    p.add_argument("--clean_k", type=int, default=4)
    p.add_argument("--dsm_sigma", type=float, default=0.01)
    p.add_argument("--val_every", type=int, default=5)
    p.add_argument("--val_subset", type=int, default=20,
                   help="number of validation shapes to use during training (full eval at end)")
    p.add_argument("--encoder_dim", type=int, default=256)
    p.add_argument("--encoder_k", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry_run", action="store_true",
                   help="run 5 train iterations + 1 val shape and exit")
    return p.parse_args()


def main():
    args = parse_args()
    jt.set_global_seed(args.seed)
    np.random.seed(args.seed)
    jt.flags.use_cuda = 1

    out_dir = HERE / "outputs" / args.exp_name
    ckpt_dir = HERE / "ckpts" / args.exp_name
    log_dir = HERE / "logs" / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.log"
    log_f = open(log_path, "a", buffering=1)

    def log(msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")

    log(f"args: {vars(args)}")

    # ---- data ----
    tcfg = TrainConfig(
        n_pc_train=50000,
        patch_size=args.patch_size,
        n_patches=args.n_patches,
        noise_min=args.noise_min,
        noise_max=args.noise_max,
    )
    train_ds = TrainPatchDataset(tcfg)
    train_ds.set_attrs(batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    vcfg = ValidateConfig(n_pc_val=50000, noise_sigma=0.010)
    val_set = make_validation_set(vcfg)
    if args.dry_run:
        val_set = val_set[:1]
    log(f"train shapes: {len(train_ds)}, val shapes: {len(val_set)} (subset {args.val_subset})")

    # ---- model ----
    model = Denoiser(k=args.encoder_k, embedding_dim=args.encoder_dim, dec_hidden=128)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model params: {n_params/1e6:.2f}M")

    optimizer = nn.Adam(model.parameters(), lr=args.lr)
    # Cosine schedule from lr -> lr_min over total epochs.
    def get_lr(epoch_idx: int) -> float:
        prog = epoch_idx / max(1, args.epochs - 1)
        return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + np.cos(np.pi * prog))

    # ---- training loop ----
    global_step = 0
    best_score = -1.0

    for epoch in range(args.epochs):
        # cosine LR
        cur_lr = get_lr(epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr
        log(f"epoch {epoch} starting (lr={cur_lr:.2e})")

        model.train()
        t0 = time.time()
        ep_losses = []
        for batch in train_ds:
            pc_noisy = batch["pc_noisy"]    # (B*P, M, 3)
            pc_clean = batch["pc_clean"]    # (B*P, M, 3)

            pred = model(pc_noisy)
            if args.use_simple_loss:
                loss = simple_displacement_loss(pred, pc_noisy, pc_clean)
            else:
                loss = frame_dsm_loss(
                    pred, pc_noisy, pc_clean,
                    n_anchor=args.frame_n_anchor,
                    frame_k=args.frame_k,
                    clean_k=args.clean_k,
                    dsm_sigma=args.dsm_sigma,
                )
            optimizer.step(loss)

            ep_losses.append(loss.item())
            if global_step % 200 == 0:
                log(f"epoch {epoch} step {global_step} loss {loss.item():.6f}")
            global_step += 1

            if args.dry_run and global_step >= 5:
                break

        log(f"epoch {epoch} done in {time.time()-t0:.1f}s, mean_loss={float(np.mean(ep_losses)):.6f}")

        # ---- mid-train validation on a small subset ----
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1 or args.dry_run:
            score = run_validation(model, val_set[: args.val_subset if not args.dry_run else 1],
                                   patch_size=args.patch_size, log=log)
            if score > best_score:
                best_score = score
                jt.save(model.state_dict(), str(ckpt_dir / "best.pkl"))
                log(f"** new best score {score:.2f}, saved ckpt")
            jt.save(model.state_dict(), str(ckpt_dir / "last.pkl"))

        if args.dry_run:
            log("dry_run finished")
            break

    log(f"training done. best subset score = {best_score:.2f}")

    # ---- final full validation on best ckpt ----
    if not args.dry_run and (ckpt_dir / "best.pkl").exists():
        log("running FULL validation on best ckpt...")
        model.load_state_dict(jt.load(str(ckpt_dir / "best.pkl")))
        score = run_validation(model, val_set, patch_size=args.patch_size, log=log)
        log(f"FULL validation score = {score:.2f}")


def run_validation(model, val_set, patch_size: int, log):
    model.eval()
    results = []
    t0 = time.time()
    for i, (rel, pc_noisy, pc_clean, _) in enumerate(val_set):
        pc_pred = patch_based_denoise(
            model, pc_noisy.astype(np.float32), patch_size=patch_size, batch_patches=8
        )
        r = evaluate_one(rel, pc_noisy, pc_clean, pc_pred)
        results.append(r)
        if (i + 1) % 5 == 0 or i == 0:
            log(f"  val {i+1}/{len(val_set)} score={r['score']:.2f} cd={r['cd_score']:.2f} p2s={r['p2s_score']:.2f}")
    agg = aggregate(results)
    log("\n" + fmt_summary(agg))
    log(f"validation took {time.time()-t0:.1f}s")
    return agg["score"]


if __name__ == "__main__":
    main()
