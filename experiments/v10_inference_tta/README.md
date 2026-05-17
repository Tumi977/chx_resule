# v10 — Inference-Time Test-Time Augmentation (TTA)

## 一句话

**纯推理改造:K 次串行迭代 + N 个旋转的 forward 平均 + 多 seed FPS ensemble。证明在我们 σ ∈ [0.005, 0.020] 设定下:K=1 + rot=4 持平 P0,K=2 反向掉分。**

## 论文方法

| 方法 | 论文出处 | 在本实验中的角色 |
|---|---|---|
| **K-step iterative refinement** | StraightPCF §4.3 (`--niters` 参数) | 推理时跑 K 次完整 patch_denoise,逐步收敛 |
| **Rotation TTA** | DGCNN §5.2 / 多数点云方法的标准增强 | z-axis 旋转 4 次推理后平均(模型不是旋转等变) |
| **FPS Seed Ensemble** | 我们自己的扩展 | 不同 RNG seed 让 FPS 选不同 patch 中心,平均结果 |

## 流程

```python
for r in rotations:                  # rot_tta {1, 2, 4, 8}
    R = rot_z(theta_r)
    x_rot = pc_noisy @ R.T
    
    for s in 0..seed_ensemble-1:     # 不同 FPS seed
        rng = default_rng(s + 12345)
        cur = x_rot
        
        for k in 0..K-1:             # K-step 串行迭代
            cur = patch_based_denoise(cur, rng=rng)
        
        accum[r,s] = cur @ R          # 旋转回去
        
out = mean over (r, s)               # 平均融合
```

每个 shape 的 forward 数量:`K × rot_tta × seed_ensemble`。

## 子实验

| 实验 | K | rot | seed | forward/shape | FULL val | vs P0 |
|---|---|---|---|---|---|---|
| `tta_p1_K2_rot4` | **2** | 4 | 1 | 8 | **68.09** | **-1.79** ❌ |
| `tta_p0_K1_rot4` | **1** | 4 | 1 | 4 | **70.92** | -0.02(打平) |

## 关键发现:**K=2 在 σ ∈ [0.005, 0.020] 上是过迭代**

| 指标 | P0 无 TTA | TTA K=2 rot=4 |
|---|---|---|
| CD score | ~57 | 52.89 (**-3.11**) |
| P2S score | ~85 | 83.30 (-0.46) |
| Total | 70.94 | 68.09 (**-1.79**) |

CD 大幅退步,说明:
1. K=2 第二次迭代把已经在表面附近的点继续往 GT 推 → "内陷"(inward bias)
2. 4 次旋转推理结果点位**不严格一致**,平均后每个点位置略糊

**StraightPCF 论文里 K=2 适用于 σ=2-3%,K=1 适用于 σ=1%**。我们 σ ∈ [0.005, 0.020] 主要在 1% 附近,**K=1 是正确选择**。

## 实测结论

- **K=1 + rot=4**:与 P0 等同(70.92 vs 70.94),**没掉分但也没涨**
- **K=2**:严重劣化
- **rotation TTA 不能补 paired 数据上的过训练**(P0 已经训得很好,旋转 TTA 不再是新信息)

## 论文方法的边界条件

TTA 在以下条件下有效:
1. 模型欠训练(P0 60 epoch 已经满)
2. 模型不旋转等变 + 训练时旋转增强不够(我们训练有 z 旋转增强 0.5 prob,TTA 信息已经覆盖)
3. 噪声大于训练分布(我们 val σ=0.01 完全在训练分布内)

**全部不满足** → TTA 没收益。

## 配置

```python
inference/predict_tta.py:
    --K              : 1 or 2 (number of full patch_denoise rounds)
    --rot_tta        : 1, 2, 4, or 8 (z-axis rotations)
    --seed_ensemble  : N different FPS seeds to average
    --testset        : output result.zip for official testset
```

## 文件

- 推理脚本:`inference/predict_tta.py`
- 测试集打包:`inference/predict_testset.py`
- log:`logs/tta_p0_K1_rot4.log`,`logs/tta_p1_K2_rot4/run.log`

## 历史意义

**确认了 TTA 不是涨分路径**。这是一个明确的"路径关闭"信号:
- 不再做 TTA 实验
- 资源转向训练改进(论文方法 v06/v07/v08)
- 提交时不需要 TTA(单 forward 就够)
