"""Time a few training iterations to estimate epoch time and pick training config."""
from __future__ import annotations
import sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import jittor as jt
import numpy as np
from jittor import nn

from datasets.pc_dataset import TrainConfig, TrainPatchDataset
from models.denoiser import Denoiser
from losses.dsm import frame_dsm_loss

jt.flags.use_cuda = 1
jt.set_global_seed(0)

# Quick: batch=4 shapes * n_patches=4 = 16 patches/step
ds = TrainPatchDataset(TrainConfig(n_patches=4, patch_size=1024))
ds.set_attrs(batch_size=4, shuffle=True, num_workers=2)

model = Denoiser(k=16, embedding_dim=256, dec_hidden=128)
print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
opt = nn.Adam(model.parameters(), lr=1e-4)

print("warm-up 3 iters...")
N_WARM, N_TIME = 3, 20
times = []
it = iter(ds)
for i in range(N_WARM + N_TIME):
    batch = next(it)
    pn = batch["pc_noisy"]
    pc = batch["pc_clean"]
    t0 = time.time()
    pred = model(pn)
    loss = frame_dsm_loss(pred, pn, pc)
    opt.step(loss)
    jt.sync_all(True)
    dt = time.time() - t0
    if i >= N_WARM:
        times.append(dt)
    print(f"  iter {i}: {dt:.3f}s loss={loss.item():.4f}")

mean_step = float(np.mean(times))
print(f"\nmean step time: {mean_step:.3f}s")
n_iters_per_epoch = len(ds) // 4 + 1
total_per_epoch = n_iters_per_epoch * mean_step
print(f"iters / epoch  : {n_iters_per_epoch}")
print(f"epoch time est : {total_per_epoch:.0f}s = {total_per_epoch/60:.1f}min")
print(f"60 epochs est  : {total_per_epoch*60/3600:.1f}h")
print(f"30 epochs est  : {total_per_epoch*30/3600:.1f}h")
