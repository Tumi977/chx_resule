"""train_p2_v2.py — Multi-stage AGT training with FIXED loss.

Key change vs train_p2.py:
    Each stage's input is generated explicitly:
        x_τ_input = pc_clean + σ_τ · ξ
        target_τ  = -σ_τ · ξ
    No NN-projection (which degenerates to identity in paired data).

Each stage τ is an independent Denoiser learning a noise-level-specific
denoiser. At inference, run stages serially (handles cascade of decreasing σ).

Surface losses (p2plane, repulsion) applied on the FINAL stage prediction.
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
from models.multi_stage_denoiser import MultiStageDenoiser
from losses.agt import schedule_sigmas, make_stage_input, stage_loss
from losses.surface import p2plane_loss, repulsion_loss
from inference.patch_denoise import patch_based_denoise
from eval.local_eval import evaluate_one, aggregate, fmt_summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", default="p2_agt_T4_v2")
    p.add_argument("--init_each_stage_from", default="ckpts/p0_baseline/best.pkl",
                   help="ckpt to copy into each ItM as warm-start; '' = scratch")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--n_patches", type=int, default=4)
    p.add_argument("--patch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--noise_min", type=float, default=0.005)
    p.add_argument("--noise_max", type=float, default=0.020)
    p.add_argument("--num_workers", type=int, default=4)
    # AGT
    p.add_argument("--n_stages", type=int, default=4)
    p.add_argument("--sigma_delta", type=float, default=2.0)
    p.add_argument("--normalize_loss_by_sigma2", action="store_true",
                   help="divide each stage loss by σ_τ² (scale-invariant)")
    # Surface losses
    p.add_argument("--lambda_p2plane", type=float, default=30.0)
    p.add_argument("--lambda_repulsion", type=float, default=0.0)
    # Inference
    p.add_argument("--inference_n_stages", type=int, default=0,
                   help="how many stages to run at inference (0 = all)")
    # val
    p.add_argument("--val_every", type=int, default=5)
    p.add_argument("--val_subset", type=int, default=20)
    # arch
    p.add_argument("--encoder_dim", type=int, default=256)
    p.add_argument("--encoder_k", type=int, default=16)
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

    model = MultiStageDenoiser(
        n_stages=args.n_stages,
        k=args.encoder_k,
        embedding_dim=args.encoder_dim,
        dec_hidden=128,
    )
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model params: {n_params/1e6:.2f}M (n_stages={args.n_stages})")

    if args.init_each_stage_from:
        ip = (HERE / args.init_each_stage_from) if not os.path.isabs(args.init_each_stage_from) else Path(args.init_each_stage_from)
        if ip.exists():
            sd = jt.load(str(ip))
            for tau in range(args.n_stages):
                model.stages[tau].load_state_dict(sd)
            log(f"warm-started ALL {args.n_stages} stages from {ip}")
        else:
            log(f"WARNING: init ckpt {ip} not found, starting from scratch")

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
        ep_losses = [[] for _ in range(args.n_stages)]
        ep_l_p2p = []

        for batch in train_ds:
            pc_clean = batch["pc_clean"]                   # (B, M, 3)
            pc_normal = batch["pc_normal"]
            sigma0 = batch["sigma"]                        # (B,)

            # σ schedule per-batch
            B = pc_clean.shape[0]
            sig_sched = jt.array(np.stack([
                schedule_sigmas(s, args.n_stages, args.sigma_delta) for s in sigma0.numpy()
            ], axis=0))                                    # (B, T)

            # Memory-efficient training: each stage backwards independently.
            # Since stages have INDEPENDENT weights and AGT loss for stage τ
            # only depends on stage_τ params, we can backprop and free graph
            # one stage at a time. This avoids 4× peak memory.
            stage_loss_vals = []
            pred_last = None
            x_input_last = None

            for tau in range(args.n_stages):
                sigma_tau = sig_sched[:, tau]              # (B,)
                # FIXED: explicit input generation, deterministic target
                x_input, target = make_stage_input(pc_clean, sigma_tau)
                pred_disp = model.execute_stage(tau, x_input)

                sig_for_norm = sigma_tau if args.normalize_loss_by_sigma2 else None
                l = stage_loss(pred_disp, target, sigma_tau=sig_for_norm)

                # Last stage: fold in p2plane loss in same backward
                if tau == args.n_stages - 1:
                    l_p2p = p2plane_loss(pred_disp, x_input, pc_clean, pc_normal)
                    total_for_this_stage = l + args.lambda_p2plane * l_p2p
                else:
                    total_for_this_stage = l

                # Independent step per stage — frees forward graph immediately.
                optimizer.step(total_for_this_stage)

                stage_loss_vals.append(l.item())
                ep_losses[tau].append(l.item())
                if tau == args.n_stages - 1:
                    ep_l_p2p.append(l_p2p.item())
                    last_p2p_for_log = l_p2p.item()
                pred_last = pred_disp
                x_input_last = x_input

            if global_step % 200 == 0:
                msg = f"epoch {epoch} step {global_step} | "
                msg += " ".join(f"L{tau}={stage_loss_vals[tau]:.4f}" for tau in range(args.n_stages))
                msg += f" p2p={last_p2p_for_log:.6f}"
                log(msg)
            global_step += 1
            if args.dry_run and global_step >= 5:
                break

        msg = f"epoch {epoch} done in {time.time()-t0:.1f}s | "
        msg += " ".join(f"L{tau}={float(np.mean(ep_losses[tau])):.4f}" for tau in range(args.n_stages))
        msg += f" p2p={float(np.mean(ep_l_p2p)):.6f}"
        log(msg)

        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1 or args.dry_run:
            n_inf = args.inference_n_stages or args.n_stages
            score = run_validation(model, val_set[: args.val_subset if not args.dry_run else 1],
                                   patch_size=args.patch_size, n_stages=n_inf, log=log)
            if score > best_score:
                best_score = score
                jt.save(model.state_dict(), str(ckpt_dir / "best.pkl"))
                log(f"** new best score {score:.2f}, saved ckpt")
            jt.save(model.state_dict(), str(ckpt_dir / "last.pkl"))
        if args.dry_run:
            log("dry_run finished")
            break

    log(f"training done. best subset score = {best_score:.2f}")
    if not args.dry_run and (ckpt_dir / "best.pkl").exists():
        log("running FULL validation on best ckpt...")
        model.load_state_dict(jt.load(str(ckpt_dir / "best.pkl")))
        n_inf = args.inference_n_stages or args.n_stages
        score = run_validation(model, val_set, patch_size=args.patch_size, n_stages=n_inf, log=log)
        log(f"FULL validation score = {score:.2f}")


def make_serial_callable(model, n_stages):
    """Wrap multistage model into f(x) -> displacement, running stages in series."""
    @jt.no_grad()
    def f(x):
        cur = x
        for tau in range(n_stages):
            d = model.execute_stage(tau, cur)
            cur = cur + d
        return cur - x
    return f


def run_validation(model, val_set, patch_size, n_stages, log):
    model.eval()
    f = make_serial_callable(model, n_stages)
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
