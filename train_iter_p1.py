"""K-step shared-weight iterative training (IterativePFN-style).

Same single-stage Denoiser model; at training time we unroll K iterations
with SHARED weights and AGT supervision per step:

    σ_τ = σ_0 · δ^(-τ),    δ = K^(1/(K-1))   roughly halving per step
    Y_τ = clean + σ_τ · ξ_τ,                  random Gaussian residual
    For τ = 0..K-1:
        pred_τ = model(x_τ)            (shared weights)
        target_τ_i = NN(x_τ_i, Y_τ) - x_τ_i
        L_τ = ||pred_τ - target_τ||^2
        x_{τ+1} = (x_τ + pred_τ).detach()      stop-grad between steps
    Total = Σ_τ L_τ + λ_p2p · L_p2p(final) + λ_rep · L_rep(final)

Inference: simply run the model K times in a row.

Why this beats plain P1: the network at every step sees an input distribution
that matches what it produced last step. P1 only ever sees "raw noisy" inputs
during training, so its 2nd-iteration generalization is an extrapolation. Here
it's the training distribution by construction.
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
from losses.surface import p2plane_loss, repulsion_loss
from losses.dcd import dcd_loss
from losses.agt import schedule_sigmas, agt_step_loss
from inference.patch_denoise import patch_based_denoise
from eval.local_eval import evaluate_one, aggregate, fmt_summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", default="p1_iter_K3")
    p.add_argument("--init_ckpt", default="ckpts/p1_p2plane_only/best.pkl",
                   help="warm-start single-stage Denoiser; '' to disable")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--n_patches", type=int, default=4)
    p.add_argument("--patch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--noise_min", type=float, default=0.005)
    p.add_argument("--noise_max", type=float, default=0.020)
    p.add_argument("--num_workers", type=int, default=4)
    # iteration
    p.add_argument("--K", type=int, default=3, help="iteration steps")
    p.add_argument("--sigma_delta", type=float, default=2.0)
    # losses
    p.add_argument("--lambda_p2plane", type=float, default=30.0)
    p.add_argument("--lambda_repulsion", type=float, default=0.0)
    p.add_argument("--lambda_dcd", type=float, default=0.0)
    p.add_argument("--dcd_alpha", type=float, default=200.0)
    # val
    p.add_argument("--val_every", type=int, default=5)
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
    log(f"Model params: {n_params/1e6:.2f}M (single-stage, run K={args.K} times)")

    if args.init_ckpt:
        ip = (HERE / args.init_ckpt) if not os.path.isabs(args.init_ckpt) else Path(args.init_ckpt)
        if ip.exists():
            model.load_state_dict(jt.load(str(ip)))
            log(f"warm-started from {ip}")
        else:
            log(f"WARNING: init_ckpt {ip} not found, starting from scratch")

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
        ep_losses = [[] for _ in range(args.K)]
        ep_l_p2p = []

        for batch in train_ds:
            pc_noisy = batch["pc_noisy"]      # (B, M, 3)
            pc_clean = batch["pc_clean"]
            pc_normal = batch["pc_normal"]
            sigma0 = batch["sigma"]

            B = pc_noisy.shape[0]
            sig_sched = jt.array(np.stack([
                schedule_sigmas(s, args.K, args.sigma_delta) for s in sigma0.numpy()
            ], axis=0))

            x_cur = pc_noisy
            losses_steps = []
            pred_last_disp = None
            x_last = None
            for tau in range(args.K):
                sigma_tau = sig_sched[:, tau]
                xi = jt.randn_like(pc_clean)
                y_tau = pc_clean + sigma_tau.reshape(B, 1, 1) * xi

                pred_disp = model(x_cur)                             # SHARED weights
                l = agt_step_loss(pred_disp, x_cur, y_tau)
                losses_steps.append(l)
                ep_losses[tau].append(l.item())

                x_cur = (x_cur + pred_disp).detach()                 # stop-grad
                pred_last_disp = pred_disp
                x_last = x_cur if pred_last_disp is None else x_cur  # x_cur after this step

            # final-step P2plane on the last predicted positions
            final_pos = pc_noisy + pred_last_disp.detach() if False else x_last
            # but we want the gradient to flow into pred_last_disp; instead take
            # x_input_last + pred_last_disp BEFORE we detached
            # → reconstruct it here:
            # (we already detached x_cur, so use the bookkeeping below)

            # Cleaner: just compute p2p on the *non-detached* last-step output:
            # last x_input was the input to ItM at τ=K-1
            # we detached AFTER summing pred — so we need to apply p2p with grad
            # Re-run the last step ungated:
            #   actually pred_last_disp DOES still hold gradient (it was the pred,
            #   not the detached x_cur).
            l_p2p = p2plane_loss(pred_last_disp, pc_noisy if args.K == 1 else (x_last - pred_last_disp).detach(),
                                 pc_clean, pc_normal)
            ep_l_p2p.append(l_p2p.item())

            total_loss = sum(losses_steps) + args.lambda_p2plane * l_p2p
            optimizer.step(total_loss)

            if global_step % 200 == 0:
                msg = f"epoch {epoch} step {global_step} | "
                msg += " ".join(f"L{tau}={losses_steps[tau].item():.4f}" for tau in range(args.K))
                msg += f" p2p={l_p2p.item():.6f}"
                log(msg)
            global_step += 1
            if args.dry_run and global_step >= 5:
                break

        msg = f"epoch {epoch} done in {time.time()-t0:.1f}s | "
        msg += " ".join(f"L{tau}={float(np.mean(ep_losses[tau])):.4f}" for tau in range(args.K))
        msg += f" p2p={float(np.mean(ep_l_p2p)):.6f}"
        log(msg)

        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1 or args.dry_run:
            score = run_validation(model, val_set[: args.val_subset if not args.dry_run else 1],
                                   patch_size=args.patch_size, K=args.K, log=log)
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
        score = run_validation(model, val_set, patch_size=args.patch_size, K=args.K, log=log)
        log(f"FULL validation score = {score:.2f}")


def make_iter_callable(model, K: int):
    @jt.no_grad()
    def f(x):
        cur = x
        for _ in range(K):
            d = model(cur)
            cur = cur + d
        return cur - x
    return f


def run_validation(model, val_set, patch_size: int, K: int, log):
    model.eval()
    f = make_iter_callable(model, K)
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
