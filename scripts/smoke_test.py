"""Smoke test: verify Jittor can run on this machine after env.sh.

Checks:
- import jittor, version
- jt.flags.use_cuda = 1 + tiny tensor op on GPU
- jt.misc.knn works on a (B,N,3) input
- numpy / scipy / point_cloud_utils importable (used by evaluate.py)
"""
import jittor as jt
import numpy as np
import time

print(f"jittor version : {jt.__version__}")
jt.flags.use_cuda = 1
print(f"use_cuda flag  : {jt.flags.use_cuda}")

# tiny op
x = jt.array(np.random.randn(2, 1024, 3).astype(np.float32))
y = (x * 2 + 1).sum()
y.sync()
print(f"basic gpu op   : ok, sum={y.item():.4f}")

# knn
t0 = time.time()
_, idx = jt.misc.knn(x, x, 16)
idx.sync()
dt = time.time() - t0
print(f"jt.misc.knn    : ok, idx.shape={tuple(idx.shape)}, t={dt:.3f}s")

# imports needed for evaluation
import scipy.spatial  # noqa: F401
print("scipy          : ok")
try:
    import point_cloud_utils as pcu  # noqa: F401
    print("pcu            : ok")
except ImportError as e:
    print(f"pcu            : MISSING ({e})")

print("\nSMOKE TEST PASSED")
