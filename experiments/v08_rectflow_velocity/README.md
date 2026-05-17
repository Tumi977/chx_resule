# v08 — Rectified-Flow Velocity Field

## 一句话

**实现 StraightPCF 的真正训练范式:不预测位移,预测沿"noisy → clean 直线"的恒定速度场。训练时随机采样中间状态 X_t = (1-t)·noisy + t·clean,target 永远是 (clean - noisy)。**

## 论文方法

| 方法 | 论文出处 | 在本实验中的角色 |
|---|---|---|
| **Rectified Flow** | Liu et al., ICLR 2023 | 数学基础,直线 ODE 的常数 velocity 场 |
| **StraightPCF velocity prediction** | de Silva Edirimuni et al., CVPR 2024 §3 | 应用到点云降噪;**这是我们项目里"误以为已实现"实际没实现的**论文方法 |
| **Random t sampling** | StraightPCF 的核心 | t ~ U(0,1) 让网络见所有插值状态 |
| **K-step Euler 推理** | 同上 §4.3 | 推理时按 σ 分段决定 niters,我们用 K=1 |

## 与"位移预测"的根本区别

| | 位移预测(P0/P1)| Rectified-flow velocity(v08)|
|---|---|---|
| 训练输入 | 总是 pc_noisy | **X_t = (1-t)·noisy + t·clean,t∼U(0,1)** |
| 训练 target | clean - noisy | **clean - noisy** (一样!但状态不同) |
| 网络看到的状态分布 | 只有 t=0 | 沿直线 t=[0,1] 全部 |
| 推理 | 一步 x ← x + d | K-step Euler: x_{k+1} = x_k + (1/K)·v(x_k) |

**关键洞察**:**target 是"常数速度"**——沿同一条 noisy → clean 直线,任何点 X_t 的速度都等于 (clean - noisy)。所以网络要学的是一个"速度场" v(x_t):**对任何 x_t,告诉我应该往哪个方向走**。

这个范式有两个隐性优势:

1. **训练分布扩大 1+K 倍**:每个 patch 不只是 (noisy, clean) 一对训练样本,而是沿直线的所有插值状态都是有效输入
2. **训练时见过所有"中间状态"**:推理时如果多步走,后续步骤的输入分布(已经被部分降噪的点)是训练分布的一部分,**没有 distribution shift**

## 实现细节

```python
# Sample t ~ U(0, 1) per batch
t = jt.rand((B, 1, 1))                    # broadcastable

# Rectified-flow inputs
x_t = (1 - t) * pc_noisy + t * pc_clean  # 中间状态
target_v = pc_clean - pc_noisy            # 常数速度
pred_v = model(x_t)                       # 网络预测

# Loss
l_v = ||pred_v - target_v||² / σ          # 主损失
l_p2p = p2plane(pred_v, pc_noisy, pc_clean, normal)  # 用从 noisy 单步预测做 P2S
loss = l_v + λ_p2p · l_p2p
```

t 分布选项:
- `uniform`:t ∼ U(0,1) ← StraightPCF 默认
- `low_t`:t ∼ U(0,1)²,偏向小 t(noisy 附近)
- `logit`:logit-normal,偏向 t=0.5(SD-style)

## 推理

```python
# K-step Euler integration
def f(x):
    cur = x
    for _ in range(K):
        v = model(cur)
        cur = cur + v / K
    return cur - x   # net displacement
```

K=1 等价于一步预测;K>1 是真正的 ODE 积分。

## 配置(p1_rectflow)

```
model           : DGCNN 256/128
params          : 0.38M
init            : ckpts/p0_baseline/best.pkl  warm-start
epochs          : 30
batch_size      : 4 × n_patches=4
lr              : 5e-5 → 1e-6
t_dist          : uniform
inference_steps : 1  (K=1, 与训练分布一致)
loss            : 1.0·v + 30·p2plane
```

## 与 v00/v01(位移预测)的训练动力学差异

```
v00 (位移预测):
  step 0 loss : 0.0049   (warm-start 后已经收敛在最优点)
  
v08 (rect-flow):
  step 0 loss : 0.0396   (warm-start 后 t∼U(0,1) 让网络见到新分布,loss 大 8×)
  这说明: P0 ckpt 在 t=0 上已经训好,但在 t>0 的中间状态上几乎没见过
         → 网络重新学"对所有 t 鲁棒"的速度场
```

这个 loss 差距正是 rect-flow 训练的"重新学习"信号。

## 结果

🟢 **运行中**(0:55,ep 0 跑了一半)。

## 文件

- 训练脚本:`train_rectflow.py`
- ckpts:`ckpts/p1_rectflow/best.pkl`(运行中)

## 历史意义

**填补了项目最大的"以为做了实际没做"的论文方法**。我之前误以为"位移预测和 velocity 预测本质一样",事实上它们的训练分布根本不同。这是项目中最重要的方法学修正之一。
