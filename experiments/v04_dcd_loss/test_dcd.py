import sys
sys.path.insert(0, '.')
import jittor as jt
import numpy as np
from losses.dcd import dcd_loss

jt.flags.use_cuda = 1

# Test 1: pred = gt -> 0
np.random.seed(0)
gt = jt.array(np.random.randn(2, 1024, 3).astype(np.float32) * 0.3)
pred = gt.clone()
print(f"DCD pred=gt:   {dcd_loss(pred, gt).item():.6f}  (expect ~0)")

# Test 2: pred is gt + small Gaussian -> small loss
pred = gt + jt.array(np.random.randn(2, 1024, 3).astype(np.float32) * 0.01)
print(f"DCD pred=gt+ε: {dcd_loss(pred, gt).item():.6f}  (expect small)")

# Test 3: pred all collapsed to single point -> large + crowded penalty
pred = jt.zeros_like(gt)
pred = pred + gt[:, :1, :]   # everyone collapsed to first gt point
print(f"DCD collapsed: {dcd_loss(pred, gt).item():.6f}  (expect HIGH due to crowd weight ~1)")

# Test 4: pred just permuted gt -> 0
perm = np.random.permutation(1024)
pred = gt[:, perm, :]
print(f"DCD permute:   {dcd_loss(pred, gt).item():.6f}  (expect ~0)")

print("DCD loss tests done")
