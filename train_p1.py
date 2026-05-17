"""P1 training: P0 model + P1 surface losses (p2plane + repulsion).

Differences vs train_p0.py:
- Total loss = L_dsm + λ_p2plane · L_p2plane + λ_repulsion · L_repulsion
- pc_normal pulled from batch (already provided by dataloader)
- Hot-starts from P0 best.pkl unless --from_scratch.
- repulsion uses *predicted positions* x' = x + d
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
from losses.dcd import dcd_loss
from losses.chamfer import chamfer_loss
from inference.patch_denoise import patch_based_denoise
from eval.local_eval import evaluate_one, aggregate, fmt_summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", default="p1_surface")
    p.add_argument("--init_ckpt", default="ckpts/p0_baseline/best.pkl",
                   help="path to P0 best ckpt to warm-start; pass '' to disable")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--n_patches", type=int, default=4)
    p.add_argument("--patch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--noise_min", type=float, default=0.005)
    p.add_argument("--noise_max", type=float, default=0.020)
    p.add_argument("--num_workers", type=int, default=4)
    # losses
    p.add_argument("--lambda_dsm", type=float, default=1.0)
    p.add_argument("--lambda_p2plane", type=float, default=0.5)
    p.add_argument("--lambda_repulsion", type=float, default=0.05)
    p.add_argument("--lambda_dcd", type=float, default=0.0,
                   help="density-aware Chamfer (NeurIPS'21); 0 disables")
    p.add_argument("--dcd_alpha", type=float, default=200.0)
    p.add_argument("--lambda_chamfer", type=float, default=0.0,
                   help="bidirectional symmetric Chamfer; 0 disables")
    p.add_argument("--frame_n_anchor", type=int, default=128)
    p.add_argument("--frame_k", type=int, default=32)
    p.add_argument("--clean_k", type=int, default=4)
    p.add_argument("--dsm_sigma", type=float, default=0.01)
    p.add_argument("--repulsion_k", type=int, default=5)
    p.add_argument("--repulsion_h", type=float, default=0.03)
    # val
    p.add_argument("--val_every", type=int, default=5)
    p.add_argument("--val_subset", type=int, default=20)
    # arch
    p.add_argument("--encoder_dim", type=int, default=256)
    p.add_argument("--encoder_k", type=int, default=16)
    p.add_argument("--encoder_kind", default="dgcnn", choices=["dgcnn", "mlgc", "dense_dgcnn"])
    p.add_argument("--mlgc_growth", type=int, default=24)
    p.add_argument("--mlgc_layers", type=int, default=4)
    p.add_argument("--dense_growth", type=int, default=64)
    p.add_argument("--dense_layers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry_run", action="store_true")
    # data augmentation (force-local: multi-scale patch + point dropout)
    p.add_argument("--patch_size_choices", type=str, default="",
                   help="comma-separated list, e.g. '512,1024,1536'; empty = use --patch_size only")
    p.add_argument("--point_dropout_p", type=float, default=0.0,
                   help="probability of applying point-dropout per sample")
    p.add_argument("--point_dropout_min", type=float, default=0.10)
    p.add_argument("--point_dropout_max", type=float, default=0.30)
    # extra geometric augmentation
    p.add_argument("--scale_p", type=float, default=0.0,
                   help="probability of isotropic scaling")
    p.add_argument("--scale_min", type=float, default=0.9)
    p.add_argument("--scale_max", type=float, default=1.1)
    # LR warmup
    p.add_argument("--warmup_epochs", type=int, default=0,
                   help="linear LR warmup from lr_min to lr over first N epochs")
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

    tcfg = TrainConfig(
        n_pc_train=50000,
        patch_size=args.patch_size,
        n_patches=args.n_patches,
        noise_min=args.noise_min,
        noise_max=args.noise_max,
        patch_size_choices=tuple(int(s) for s in args.patch_size_choices.split(",") if s.strip()) if args.patch_size_choices else (),
        point_dropout_p=args.point_dropout_p,
        point_dropout_min=args.point_dropout_min,
        point_dropout_max=args.point_dropout_max,
        scale_p=args.scale_p,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
    )
    train_ds = TrainPatchDataset(tcfg)
    train_ds.set_attrs(batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    vcfg = ValidateConfig(n_pc_val=50000, noise_sigma=0.010)
    val_set = make_validation_set(vcfg)
    if args.dry_run:
        val_set = val_set[:1]
    log(f"train shapes: {len(train_ds)}, val shapes: {len(val_set)} (subset {args.val_subset})")

    model = Denoiser(
        k=args.encoder_k,
        embedding_dim=args.encoder_dim,
        dec_hidden=128,
        encoder_kind=args.encoder_kind,
        encoder_kwargs={
            "k": args.encoder_k,
            "embedding_dim": args.encoder_dim,
            **({"growth": args.mlgc_growth, "n_layers": args.mlgc_layers}
               if args.encoder_kind == "mlgc" else {}),
            **({"growth": args.dense_growth, "n_layers": args.dense_layers}
               if args.encoder_kind == "dense_dgcnn" else {}),
        },
    )
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model params: {n_params/1e6:.2f}M (encoder={args.encoder_kind})")

    # warm-start
    if args.init_ckpt:
        init_path = (HERE / args.init_ckpt) if not os.path.isabs(args.init_ckpt) else Path(args.init_ckpt)
        if init_path.exists():
            model.load_state_dict(jt.load(str(init_path)))
            log(f"warm-started from {init_path}")
        else:
            log(f"WARNING: init_ckpt {init_path} not found, training from scratch")

    optimizer = nn.Adam(model.parameters(), lr=args.lr)

    def get_lr(epoch_idx: int) -> float:
        # Linear warmup from lr_min to lr over the first warmup_epochs.
        if args.warmup_epochs > 0 and epoch_idx < args.warmup_epochs:
            return args.lr_min + (args.lr - args.lr_min) * (epoch_idx / args.warmup_epochs)
        # Then cosine decay over the remaining epochs.
        remaining = max(1, args.epochs - args.warmup_epochs - 1)
        prog = (epoch_idx - args.warmup_epochs) / remaining
        prog = max(0.0, min(1.0, prog))
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
        ep_l_dsm, ep_l_p2p, ep_l_rep, ep_l_dcd, ep_l_cd = [], [], [], [], []
        for batch in train_ds:
            pc_noisy = batch["pc_noisy"]
            pc_clean = batch["pc_clean"]
            pc_normal = batch["pc_normal"]

            pred_disp = model(pc_noisy)

            l_dsm = frame_dsm_loss(
                pred_disp, pc_noisy, pc_clean,
                n_anchor=args.frame_n_anchor,
                frame_k=args.frame_k,
                clean_k=args.clean_k,
                dsm_sigma=args.dsm_sigma,
            )
            l_p2p = p2plane_loss(pred_disp, pc_noisy, pc_clean, pc_normal)
            l_rep = repulsion_loss(pc_noisy + pred_disp, k=args.repulsion_k, h=args.repulsion_h)
            if args.lambda_dcd > 0:
                l_dcd = dcd_loss(pc_noisy + pred_disp, pc_clean, alpha=args.dcd_alpha)
            else:
                l_dcd = jt.zeros(1)[0]
            if args.lambda_chamfer > 0:
                l_cd = chamfer_loss(pc_noisy + pred_disp, pc_clean)
            else:
                l_cd = jt.zeros(1)[0]

            loss = (
                args.lambda_dsm * l_dsm
                + args.lambda_p2plane * l_p2p
                + args.lambda_repulsion * l_rep
                + args.lambda_dcd * l_dcd
                + args.lambda_chamfer * l_cd
            )
            optimizer.step(loss)

            ep_l_dsm.append(l_dsm.item())
            ep_l_p2p.append(l_p2p.item())
            ep_l_rep.append(l_rep.item())
            ep_l_dcd.append(l_dcd.item())
            ep_l_cd.append(l_cd.item())

            if global_step % 200 == 0:
                log(f"epoch {epoch} step {global_step} | dsm {l_dsm.item():.4f} p2p {l_p2p.item():.6f} rep {l_rep.item():.4f} dcd {l_dcd.item():.4f} cd {l_cd.item():.6f}")
            global_step += 1
            if args.dry_run and global_step >= 5:
                break

        log(
            f"epoch {epoch} done in {time.time()-t0:.1f}s | "
            f"dsm {float(np.mean(ep_l_dsm)):.4f} p2p {float(np.mean(ep_l_p2p)):.6f} "
            f"rep {float(np.mean(ep_l_rep)):.4f} dcd {float(np.mean(ep_l_dcd)):.4f} "
            f"cd {float(np.mean(ep_l_cd)):.6f}"
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

    log(f"training done. best subset score = {best_score:.2f}")

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
