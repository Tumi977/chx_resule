"""Quick unit-test for losses/surface.py (run after P0 launch, doesn't need GPU long).

Checks:
- p2plane is 0 when pred is exactly clean
- p2plane increases monotonically when pred drifts in normal direction
- p2plane stays small when pred drifts in tangent direction (proves the loss
  truly proxies P2S, not chamfer)
- repulsion is finite, smaller for spread-out points than for clustered ones
"""
from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import jittor as jt
import numpy as np

from losses.surface import p2plane_loss, repulsion_loss

jt.flags.use_cuda = 1
np.random.seed(0)

B, M = 2, 256
pc_clean = jt.array(np.random.randn(B, M, 3).astype(np.float32))
normals = jt.array(np.array([[1.0, 0.0, 0.0]] * (B * M), dtype=np.float32).reshape(B, M, 3))
pc_noisy = pc_clean + jt.array(np.random.randn(B, M, 3).astype(np.float32) * 0.01)

# (1) zero-residual test: disp such that pred = clean -> loss = 0
disp_perfect = pc_clean - pc_noisy
l0 = p2plane_loss(disp_perfect, pc_noisy, pc_clean, normals).item()
print(f"p2plane zero-residual : {l0:.2e}  (expect ~0)")

# (2) normal-direction drift -> grows quadratically
for s in [0.0, 0.01, 0.05, 0.1]:
    drift = jt.array(np.array([[s, 0.0, 0.0]] * (B * M), dtype=np.float32).reshape(B, M, 3))
    disp = (pc_clean - pc_noisy) + drift  # ends at clean + s*n
    l = p2plane_loss(disp, pc_noisy, pc_clean, normals).item()
    print(f"p2plane normal drift {s:.2f}: {l:.6f}  (expect ~{s*s:.4f})")

# (3) tangent-direction drift -> stays ~0 (since residual is along y)
disp_tangent = (pc_clean - pc_noisy) + jt.array(np.array([[0.0, 0.1, 0.0]] * (B * M), dtype=np.float32).reshape(B, M, 3))
l_t = p2plane_loss(disp_tangent, pc_noisy, pc_clean, normals).item()
print(f"p2plane tangent drift  : {l_t:.6e}  (expect ~0)")

# (4) repulsion: cluster vs spread
spread = jt.array(np.random.randn(1, 64, 3).astype(np.float32) * 0.5)
cluster = jt.array(np.random.randn(1, 64, 3).astype(np.float32) * 0.005)  # tightly packed
r_spread = repulsion_loss(spread, k=5, h=0.05).item()
r_cluster = repulsion_loss(cluster, k=5, h=0.05).item()
print(f"repulsion spread       : {r_spread:.6e}")
print(f"repulsion cluster      : {r_cluster:.6e}  (expect >> spread)")

print("\nALL CHECKS DONE")
