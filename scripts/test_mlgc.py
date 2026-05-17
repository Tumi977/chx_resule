"""Sanity test for MLGC encoder: shape check + parameter count + forward time."""
from __future__ import annotations
import sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import jittor as jt
import numpy as np

from models.encoders.dgcnn import DGCNNEncoder
from models.encoders.mlgc import MLGCEncoder

jt.flags.use_cuda = 1
np.random.seed(0)

B, N = 4, 1024
x = jt.array(np.random.randn(B, N, 3).astype(np.float32))

print("=" * 60)
print("DGCNN baseline")
m1 = DGCNNEncoder(k=16, embedding_dim=256)
n1 = sum(p.numel() for p in m1.parameters())
y1 = m1(x); y1.sync()
t0 = time.time()
for _ in range(3):
    y1 = m1(x); y1.sync()
print(f"  params : {n1/1e6:.3f} M")
print(f"  out    : {tuple(y1.shape)}")
print(f"  3-fwds : {(time.time()-t0):.3f}s")

print("=" * 60)
print("MLGC encoder (n_layers=4, growth=24, in_proj=12)")
m2 = MLGCEncoder(n_layers=4, growth=24, k=16, embedding_dim=256, in_proj_dim=12)
n2 = sum(p.numel() for p in m2.parameters())
y2 = m2(x); y2.sync()
t0 = time.time()
for _ in range(3):
    y2 = m2(x); y2.sync()
print(f"  params : {n2/1e6:.3f} M")
print(f"  out    : {tuple(y2.shape)}")
print(f"  3-fwds : {(time.time()-t0):.3f}s")

print("=" * 60)
print("MLGC encoder (n_layers=4, growth=32, in_proj=12)  // a bit heavier")
m3 = MLGCEncoder(n_layers=4, growth=32, k=16, embedding_dim=256, in_proj_dim=12)
n3 = sum(p.numel() for p in m3.parameters())
y3 = m3(x); y3.sync()
print(f"  params : {n3/1e6:.3f} M")

print("\nALL OK")
