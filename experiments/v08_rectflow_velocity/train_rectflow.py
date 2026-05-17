"""train_rectflow.py — rectified-flow velocity field training (StraightPCF).

Key idea (StraightPCF, CVPR'24 §3):
    Instead of predicting a *displacement* d = clean - noisy at the noisy
    point, train a *velocity field* v(x_t, t) where t ∈ [0,1] interpolates
    between the noisy and clean states:

        x_t = (1-t)·noisy + t·clean
        target velocity = clean - noisy   (constant along the straight line!)

    This is rectified flow: the velocity is constant along each trajectory,
    and the network learns "for any state x_t along this line, what's the
    direction back to clean".

Why this is better than displacement prediction:
    - Network sees ALL intermediate states, not just t=0
    - Training distribution is much wider (one shape -> many (x_t, target)
      tuples), acting as strong regularization
    - At inference we run 1-step Euler from noisy → noisy + v(noisy, t=0)
      OR multi-step: x_{k+1} = x_k + (1/K)·v(x_k)

Implementation note:
    Our data has noisy = clean + ε so noisy and clean are paired. We compute:

        t ~ U(0, 1)    per-batch
        x_t = (1-t)·pc_noisy + t·pc_clean
        target = pc_clean - pc_noisy
        pred = model(x_t)
        loss = ||pred - target||²

    pred is the network's velocity prediction at state x_t.
    Surface losses (p2plane) compute against pred_pos = pc_noisy + pred (i.e.
    one Euler step from noisy with the predicted velocity).
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
from losses.surface import p2plane_loss
from inference.patch_denoise import patch_based_denoise
from eval.local_eval import evaluate_one, aggregate, fmt_summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", default="p1_rectflow")
    p.add_argument("--init_ckpt", default="ckpts/p0_baseline/best.pkl")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--n_patches", type=int, default=4)
    p.add_argument("--patch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--noise_min", type=float, default=0.005)
    p.add_argument("--noise_max", type=float, default=0.020)
    p.add_argument("--num_workers", type=int, default=4)
    # rect-flow specific
    p.add_argument("--t_dist", default="uniform", choices=["uniform", "low_t", "logit"],
                   help="distribution of t in (0,1); uniform is default")
    # losses
    p.add_argument("--lambda_velocity", type=float, default=1.0)
    p.add_argument("--lambda_p2plane", type=float, default=30.0)
    p.add_argument("--dsm_sigma", type=float, default=0.01)
    # val
    p.add_argument("--val_every", type=int, default=5)
    p.add_argument("--val_subset", type=int, default=20)
    # inference
    p.add_argument("--inference_steps", type=int, default=1,
                   help="K-step Euler integration at inference; 1 = single shot")
    # arch
    p.add_argument("--encoder_dim", type=int, default=256)
    p.add_argument("--encoder_k", type=int, default=16)
    p.add_argument("--encoder_kind", default="dgcnn")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def sample_t(B: int, kind: str) -> jt.Var:
    """Sample t ∈ (0, 1) per-batch."""
    if kind == "uniform":
        return jt.rand((B, 1, 1))
    elif kind == "low_t":
        # bias to small t (more weight near noisy state where most gradient lives)
        u = jt.rand((B, 1, 1))
        return u ** 2
    elif kind == "logit":
        # Stable Diffusion-style logit-normal sampling around 0.5
        z = jt.randn((B, 1, 1))
        return jt.sigmoid(z)
    else:
        return jt.rand((B, 1, 1))


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
        ep_l_v, ep_l_p2p = [], []

        for batch in train_ds:
            pc_noisy = batch["pc_noisy"]
            pc_clean = batch["pc_clean"]
            pc_normal = batch["pc_normal"]

            B = pc_noisy.shape[0]
            t = sample_t(B, args.t_dist)                        # (B, 1, 1)

            # Rectified-flow inputs
            x_t = (1.0 - t) * pc_noisy + t * pc_clean           # (B, M, 3)
            target_v = pc_clean - pc_noisy                       # constant velocity
            pred_v = model(x_t)                                  # network output

            # Velocity loss
            l_v = (((pred_v - target_v) ** 2).sum(-1)).mean() / args.dsm_sigma

            # Surface loss: apply pred_v as one Euler step from pc_noisy
            pred_pos_from_noisy = pc_noisy + pred_v
            # we want surface alignment between predicted endpoint and clean
            l_p2p = p2plane_loss(pred_v, pc_noisy, pc_clean, pc_normal)

            loss = args.lambda_velocity * l_v + args.lambda_p2plane * l_p2p
            optimizer.step(loss)

            ep_l_v.append(l_v.item())
            ep_l_p2p.append(l_p2p.item())

            if global_step % 200 == 0:
                log(f"epoch {epoch} step {global_step} | v {l_v.item():.4f} p2p {l_p2p.item():.6f} t_mean {t.numpy().mean():.3f}")
            global_step += 1
            if args.dry_run and global_step >= 5:
                break

        log(
            f"epoch {epoch} done in {time.time()-t0:.1f}s | "
            f"v {float(np.mean(ep_l_v)):.4f} p2p {float(np.mean(ep_l_p2p)):.6f}"
        )

        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1 or args.dry_run:
            score = run_validation(model, val_set[: args.val_subset if not args.dry_run else 1],
                                   patch_size=args.patch_size,
                                   inference_steps=args.inference_steps, log=log)
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
        score = run_validation(model, val_set, patch_size=args.patch_size,
                               inference_steps=args.inference_steps, log=log)
        log(f"FULL validation score = {score:.2f}")


def make_kstep_callable(model, K: int):
    """K-step Euler integration: x_{k+1} = x_k + (1/K)·v(x_k)."""
    @jt.no_grad()
    def f(x):
        cur = x
        for _ in range(K):
            v = model(cur)
            cur = cur + v / K
        return cur - x  # net displacement
    return f


def run_validation(model, val_set, patch_size, inference_steps, log):
    model.eval()
    f = make_kstep_callable(model, K=inference_steps)
    results = []
    t0 = time.time()
    for i, (rel, pc_noisy, pc_clean, _) in enumerate(val_set):
        pc_pred = patch_based_denoise(f, pc_noisy.astype(np.float32), patch_size=patch_size, batch_patches=8)
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
