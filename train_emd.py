"""train_emd.py — Sinkhorn EMD as the main supervision (PD-LTS-style).

Replaces frame-DSM as the primary loss with EMD, which:
    - is one-to-one bijection (no many-to-one collapse, the CD weak spot)
    - is what PD-LTS uses for SOTA on PU-Net 50K @ 1-2% noise

We keep p2plane as auxiliary (it complements EMD by directly tightening P2S).

Note: EMD is O(N²) memory + O(N² · n_iters) compute. For N=1024 patch and
batch_size=4, this is ~16M cells × 50 iter ≈ 800M ops per step — costly but
tolerable on 4090D. We use eps=0.01 and n_iters=50 as a balance.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from datasets.pc_dataset import TrainConfig, TrainPatchDataset, ValidateConfig, make_validation_set
from models.denoiser import Denoiser
from losses.dsm import frame_dsm_loss
from losses.surface import p2plane_loss, repulsion_loss
from losses.emd import sinkhorn_emd
from inference.patch_denoise import patch_based_denoise
from eval.local_eval import evaluate_one, aggregate, fmt_summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", default="p1_emd")
    p.add_argument("--init_ckpt", default="ckpts/p0_baseline/best.pkl")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--n_patches", type=int, default=2,
                   help="lower than usual since EMD is O(N²) memory")
    p.add_argument("--patch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--noise_min", type=float, default=0.005)
    p.add_argument("--noise_max", type=float, default=0.020)
    p.add_argument("--num_workers", type=int, default=4)
    # losses
    p.add_argument("--lambda_dsm", type=float, default=0.5,
                   help="frame-DSM as auxiliary; main loss is EMD now")
    p.add_argument("--lambda_p2plane", type=float, default=30.0)
    p.add_argument("--lambda_emd", type=float, default=10.0,
                   help="EMD main loss")
    p.add_argument("--emd_eps", type=float, default=0.01)
    p.add_argument("--emd_iters", type=int, default=50)
    # val
    p.add_argument("--val_every", type=int, default=3)
    p.add_argument("--val_subset", type=int, default=20)
    # arch
    p.add_argument("--encoder_dim", type=int, default=256)
    p.add_argument("--encoder_k", type=int, default=16)
    p.add_argument("--encoder_kind", default="dgcnn")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    jt.set_global_seed(args.seed)
    np.random.seed(args.seed)
    jt.flags.use_cuda = 1

    out_dir = HERE / "outputs" / args.exp_name
    ckpt_dir = HERE / "ckpts" / args.exp_name
    log_dir = HERE / "logs" / args.exp_name
    for d in (out_dir, ckpt_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.log"
    log_f = open(log_path, "a", buffering=1)

    def log(msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")

    log(f"args: {vars(args)}")

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
    log(f"train shapes: {len(train_ds)}, val shapes: {len(val_set)}")

    model = Denoiser(
        k=args.encoder_k,
        embedding_dim=args.encoder_dim,
        dec_hidden=128,
        encoder_kind=args.encoder_kind,
    )
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model params: {n_params/1e6:.2f}M")

    if args.init_ckpt:
        ip = (HERE / args.init_ckpt) if not os.path.isabs(args.init_ckpt) else Path(args.init_ckpt)
        if ip.exists():
            model.load_state_dict(jt.load(str(ip)))
            log(f"warm-started from {ip}")
        else:
            log(f"WARNING: init {ip} not found, starting from scratch")

    optimizer = nn.Adam(model.parameters(), lr=args.lr)

    def get_lr(epoch_idx):
        prog = epoch_idx / max(1, args.epochs - 1)
        return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + np.cos(np.pi * prog))

    global_step = 0
    best_score = -1.0

    for epoch in range(args.epochs):
        cur_lr = get_lr(epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr
        log(f"epoch {epoch} starting (lr={cur_lr:.2e})")

        model.train()
        t0 = time.time()
        ep_l_dsm, ep_l_p2p, ep_l_emd = [], [], []

        for batch in train_ds:
            pc_noisy = batch["pc_noisy"]
            pc_clean = batch["pc_clean"]
            pc_normal = batch["pc_normal"]

            pred_disp = model(pc_noisy)
            pred_pos = pc_noisy + pred_disp

            l_dsm = frame_dsm_loss(pred_disp, pc_noisy, pc_clean,
                                    n_anchor=128, frame_k=32, clean_k=4, dsm_sigma=0.01)
            l_p2p = p2plane_loss(pred_disp, pc_noisy, pc_clean, pc_normal)
            l_emd = sinkhorn_emd(pred_pos, pc_clean, eps=args.emd_eps, n_iters=args.emd_iters)

            loss = (
                args.lambda_dsm * l_dsm
                + args.lambda_p2plane * l_p2p
                + args.lambda_emd * l_emd
            )
            optimizer.step(loss)

            ep_l_dsm.append(l_dsm.item())
            ep_l_p2p.append(l_p2p.item())
            ep_l_emd.append(l_emd.item())

            if global_step % 100 == 0:
                log(f"epoch {epoch} step {global_step} | dsm {l_dsm.item():.4f} p2p {l_p2p.item():.6f} emd {l_emd.item():.6f}")
            global_step += 1
            if args.dry_run and global_step >= 5:
                break

        log(
            f"epoch {epoch} done in {time.time()-t0:.1f}s | "
            f"dsm {float(np.mean(ep_l_dsm)):.4f} "
            f"p2p {float(np.mean(ep_l_p2p)):.6f} "
            f"emd {float(np.mean(ep_l_emd)):.6f}"
        )

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

    log(f"training done. best subset = {best_score:.2f}")
    if not args.dry_run and (ckpt_dir / "best.pkl").exists():
        log("running FULL validation on best ckpt...")
        model.load_state_dict(jt.load(str(ckpt_dir / "best.pkl")))
        score = run_validation(model, val_set, patch_size=args.patch_size, log=log)
        log(f"FULL validation score = {score:.2f}")


def run_validation(model, val_set, patch_size, log):
    model.eval()
    results = []
    t0 = time.time()
    for i, (rel, pc_noisy, pc_clean, _) in enumerate(val_set):
        pc_pred = patch_based_denoise(model, pc_noisy.astype(np.float32), patch_size=patch_size, batch_patches=8)
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
