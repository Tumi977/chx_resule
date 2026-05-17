"""Quick sanity tests for Sinkhorn EMD."""
import sys; sys.path.insert(0, '.')
import jittor as jt; jt.flags.use_cuda = 1
import numpy as np
from losses.emd import sinkhorn_emd

def test(name, pred, gt, **kw):
    try:
        v = sinkhorn_emd(pred, gt, **kw)
        print(f"{name:25s}: {float(v.numpy()):.6f}")
    except Exception as e:
        print(f"{name:25s}: ERROR {type(e).__name__}: {str(e)[:80]}")

np.random.seed(0)
gt = jt.array(np.random.randn(2, 256, 3).astype(np.float32) * 0.3)

test("pred = gt", gt, gt, eps=0.01, n_iters=20)

pred = gt + jt.array(np.random.randn(2, 256, 3).astype(np.float32) * 0.02)
test("pred = gt + ε(0.02)", pred, gt, eps=0.01, n_iters=20)

perm = np.random.permutation(256)
pred = jt.array(gt.numpy()[:, perm, :])
test("pred = gt permuted", pred, gt, eps=0.01, n_iters=20)

# collapse - use jt array, not expand
pred_np = np.tile(gt.numpy()[:, :1, :], (1, 256, 1))
pred = jt.array(pred_np)
test("pred all collapsed", pred, gt, eps=0.01, n_iters=20)

# gradient
pred_train = jt.array(gt.numpy())
pred_train.start_grad()
gt_target = gt + jt.array(np.random.randn(2, 256, 3).astype(np.float32) * 0.05)
loss = sinkhorn_emd(pred_train, gt_target, eps=0.01, n_iters=20)
g = jt.grad(loss, pred_train)
g_norm_val = float((g ** 2).sum().sqrt().numpy())
print(f"grad norm                : {g_norm_val:.4f}")
print("ALL EMD TESTS DONE")
