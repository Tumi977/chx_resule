"""Quick correctness checks for losses/agt.py."""
from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import jittor as jt
import numpy as np

from losses.agt import schedule_sigmas, make_agt_target_jt, agt_step_loss, rect_flow_mix

jt.flags.use_cuda = 1
np.random.seed(0)

# (1) σ schedule: 4 steps, σ_0=0.02, δ=2 -> 0.02, 0.01, 0.005, 0.0025
s = schedule_sigmas(0.02, 4, 2.0)
print(f"σ schedule (T=4, δ=2)   : {s.tolist()}")
assert np.allclose(s, [0.02, 0.01, 0.005, 0.0025])

# (2) make_agt_target: when x_prev == clean, target should ~= 0 + sigma*xi
B, M = 2, 256
clean = jt.array(np.random.randn(B, M, 3).astype(np.float32))
sigma = jt.array(np.array([0.01, 0.02], dtype=np.float32))
y_tau, target_disp, nn_idx = make_agt_target_jt(clean, clean, sigma)
# x_prev == clean, so for each i, NN(clean[i], y_tau) is *most likely* y_tau[i]
# (paired identity NN). target_disp[i] = y_tau[i] - clean[i] = sigma*xi[i].
# Check that |target_disp| has the right magnitude:
mag = (target_disp ** 2).sum(-1).sqrt().mean(dim=-1).numpy()  # per-batch
expected = sigma.numpy() * np.sqrt(3)  # E[|xi|] for 3D gaussian sigma=σ
print(f"|target_disp| per batch : {mag} ; expected ~{expected}")

# (3) agt_step_loss is 0 when pred_disp == target_disp
loss_perfect = agt_step_loss(target_disp, clean, y_tau).item()
print(f"agt_step_loss perfect   : {loss_perfect:.2e}  (expect ~0)")

# (4) rect_flow_mix: t ~ U(0,1); shape preserved
x_in, t = rect_flow_mix(clean, y_tau, p=1.0)
print(f"rect_flow x shape       : {x_in.shape}  t shape: {t.shape}")
print(f"t stats                 : min={t.min().item():.3f} max={t.max().item():.3f}")
print(f"x_in == clean when t=0  : (impossible to test exactly, but t.mean ≈ 0.5 expected)")

print("\nALL CHECKS DONE")
