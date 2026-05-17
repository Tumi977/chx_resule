"""P2 training: AGT-style multi-step supervision + rectified-flow t-aug,
with T independent ItM modules (multi-stage denoiser).

Per-batch loop (for one mini-batch of patches):
    pc_noisy0 -- inserted as x_0
    For τ = 0..T-1:
        sigma_τ = sigma_0 / δ^τ
        ξ_τ ~ N(0, I)
        Y_τ = pc_clean + σ_τ ξ_τ
        (rect-flow) X̃_τ = (1-t_τ)·x_τ + t_τ·Y_τ,  t_τ ~ U(0,1)
        pred_τ = ItM_τ(X̃_τ)
        target_τ = NN(X̃_τ, Y_τ) - X̃_τ
        L_τ = ||pred_τ - target_τ||^2
        x_{τ+1} = X̃_τ + pred_τ.detach()      (stop-grad to next step input;
                                              each ItM trained independently)

Final loss = sum_τ L_τ  (+ p2plane + repulsion at last step, optional)

Inference: serial unroll without rect-flow mixing.
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
from losses.agt import schedule_sigmas, agt_step_loss, rect_flow_mix
from losses.surface import p2plane_loss, repulsion_loss
from inference.patch_denoise import patch_based_denoise
from eval.local_eval import evaluate_one, aggregate, fmt_summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", default="p2_agt_multistage")
    p.add_argument("--init_each_stage_from", default="ckpts/p1_surface/best.pkl",
                   help="ckpt to copy into each ItM as warm-start; '' = scratch")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=4)
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
    p.add_argument("--rect_flow_p", type=float, default=1.0)
    # losses
    p.add_argument("--lambda_p2plane", type=float, default=0.5)
    p.add_argument("--lambda_repulsion", type=float, default=0.05)
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
    log(f"train shapes: {len(train_ds)}, val shapes: {len(val_set)} (subset {args.val_subset})")

    model = MultiStageDenoiser(
        n_stages=args.n_stages,
        k=args.encoder_k,
        embedding_dim=args.encoder_dim,
        dec_hidden=128,
    )
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model params: {n_params/1e6:.2f}M (n_stages={args.n_stages})")

    # warm-start each stage independently from given ckpt (single-stage Denoiser)
    if args.init_each_stage_from:
        ip = (HERE / args.init_each_stage_from) if not os.path.isabs(args.init_each_stage_from) else Path(args.init_each_stage_from)
        if ip.exists():
            sd = jt.load(str(ip))
            for tau in range(args.n_stages):
                model.stages[tau].load_state_dict(sd)
            log(f"warm-started ALL {args.n_stages} stages from {ip}")
        else:
            log(f"WARNING: init ckpt {ip} not found, training from scratch")

    optimizer = nn.Adam(model.parameters(), lr=args.lr)

    def get_lr(epoch_idx: int) -> float:
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
        ep_losses_per_stage = [[] for _ in range(args.n_stages)]
        ep_l_p2p, ep_l_rep = [], []

        for batch in train_ds:
            pc_noisy = batch["pc_noisy"]    # (B*P, M, 3)
            pc_clean = batch["pc_clean"]
            pc_normal = batch["pc_normal"]
            sigma0 = batch["sigma"]          # (B*P,)

            # σ schedule per-batch
            B = pc_noisy.shape[0]
            sig_sched = jt.array(np.stack([
                schedule_sigmas(s, args.n_stages, args.sigma_delta) for s in sigma0.numpy()
            ], axis=0))   # (B, T)

            x_cur = pc_noisy
            losses_steps = []
            pred_last_disp = None
            x_input_last = None

            for tau in range(args.n_stages):
                sigma_tau = sig_sched[:, tau]                          # (B,)
                # build Y_τ = clean + σ_τ * ξ
                xi = jt.randn_like(pc_clean)
                y_tau = pc_clean + sigma_tau.reshape(B, 1, 1) * xi

                # rect-flow input mix
                if args.rect_flow_p > 0:
                    x_input, _ = rect_flow_mix(x_cur, y_tau, p=args.rect_flow_p)
                else:
                    x_input = x_cur

                pred_disp = model.execute_stage(tau, x_input)
                l = agt_step_loss(pred_disp, x_input, y_tau)
                losses_steps.append(l)
                ep_losses_per_stage[tau].append(l.item())

                # advance state without grad to next stage (each ItM trained ind.)
                x_cur = (x_input + pred_disp).detach()
                pred_last_disp = pred_disp
                x_input_last = x_input

            # P1-style surface losses on the FINAL stage's prediction
            l_p2p = p2plane_loss(pred_last_disp, x_input_last, pc_clean, pc_normal)
            # repulsion on final predicted positions
            final_pos = x_input_last + pred_last_disp
            l_rep = repulsion_loss(final_pos, k=5, h=0.03)

            total_loss = sum(losses_steps) + args.lambda_p2plane * l_p2p + args.lambda_repulsion * l_rep
            optimizer.step(total_loss)

            ep_l_p2p.append(l_p2p.item())
            ep_l_rep.append(l_rep.item())

            if global_step % 200 == 0:
                msg = f"epoch {epoch} step {global_step} | "
                msg += " ".join(f"L{tau}={losses_steps[tau].item():.4f}" for tau in range(args.n_stages))
                msg += f" p2p={l_p2p.item():.6f} rep={l_rep.item():.4f}"
                log(msg)
            global_step += 1
            if args.dry_run and global_step >= 5:
                break

        msg = f"epoch {epoch} done in {time.time()-t0:.1f}s | "
        msg += " ".join(f"L{tau}={float(np.mean(ep_losses_per_stage[tau])):.4f}" for tau in range(args.n_stages))
        msg += f" p2p={float(np.mean(ep_l_p2p)):.6f} rep={float(np.mean(ep_l_rep)):.4f}"
        log(msg)

        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1 or args.dry_run:
            score = run_validation(model, val_set[: args.val_subset if not args.dry_run else 1],
                                   patch_size=args.patch_size, n_stages=args.n_stages, log=log)
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
        score = run_validation(model, val_set, patch_size=args.patch_size, n_stages=args.n_stages, log=log)
        log(f"FULL validation score = {score:.2f}")


def make_serial_callable(model: MultiStageDenoiser, n_stages: int):
    """Wrap a multistage model into f(x) -> (final pred displacement) for the
    existing patch_based_denoise. Internally runs all T stages serially.
    """
    @jt.no_grad()
    def f(x):
        cur = x
        for tau in range(n_stages):
            d = model.execute_stage(tau, cur)
            cur = cur + d
        return cur - x  # net displacement
    return f


def run_validation(model, val_set, patch_size: int, n_stages: int, log):
    model.eval()
    results = []
    t0 = time.time()
    f = make_serial_callable(model, n_stages)
    for i, (rel, pc_noisy, pc_clean, _) in enumerate(val_set):
        pc_pred = patch_based_denoise(
            f, pc_noisy.astype(np.float32), patch_size=patch_size, batch_patches=8
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
