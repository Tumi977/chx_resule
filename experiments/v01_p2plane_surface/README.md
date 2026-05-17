# v01 — P2plane Surface Loss

## 一句话

**在 v00 基础上加法向切平面距离损失(P2S 评测的可微替代),把 P2S 项压到 83.76(超 IterativePFN 50K@1% 论文水平)。**

## 论文方法

| 方法 | 论文出处 | 在本实验中的角色 |
|---|---|---|
| **法向切平面距离** | 思想源自 3DMambaIPF (AAAI 2025) 可微渲染 + Pointfilter 法向加权位移;此处用最简化形式 | 辅助损失,补 frame-DSM 不直接对齐 P2S 的缺陷 |
| **Repulsion loss(可选)** | PU-Net / 标准点云方法 | 防点聚簇 |

## 损失数学

```
residual_i = (pc_noisy[i] + pred_disp[i]) - pc_clean[i]   # 残余位移
n_i        = clean.npz 自带的 GT 法向
L_p2plane  = mean( (residual_i · n_i)^2 )

最终损失 = L_dsm + λ_p2plane · L_p2plane + λ_repulsion · L_rep
```

**关键洞察**:沿法向的位移误差等同于点到表面的距离误差(切平面近似);切向漂移不被惩罚,网络可以在表面上自由重分布点。

## 子实验

3 个不同的损失权重组合:

| 子实验 | λ_dsm | λ_p2plane | λ_repulsion | 备注 |
|---|---|---|---|---|
| `p1_p2plane_only` | 1.0 | **50** | 0 | **纯 p2plane,无 repulsion** |
| `p1_balanced` | 1.0 | 30 | 0.005 | 平衡权重 |
| `p1_default` | 1.0 | 0.5 | 0.05 | (老错误版本)rep 权重过大,失败 |
| `p1_cd_strong` | 1.0 | 0.2 | 0.15 | (老错误版本)rep 权重过大,完全崩 0.0 |

**重要教训**:λ_repulsion ≥ 0.05 时 rep 项主导训练,网络优化"打散点"而非"贴近表面",分数掉到 0。这是损失权重平衡的典型陷阱。

## 配置(p1_p2plane_only)

```
encoder       : DGCNN, k=16, embedding_dim=256
params        : 0.38M
init_ckpt     : ckpts/p0_baseline/best.pkl  (warm-start)
epochs        : 30
lr            : 5e-5 → 1e-6
loss          : 1.0·dsm + 50·p2plane (no repulsion)
```

## 结果

| 实验 | best subset | **FULL val** | CD | P2S |
|---|---|---|---|---|
| p1_p2plane_only | 63.59 | **69.88** | 56.00 | 83.76 |
| p1_balanced | 64.49 | 69.78 | 56.00 | 83.56 |
| p1_default(失败版) | 54.25 | — | — | — |
| p1_cd_strong(崩 0)| 0.00 | — | — | — |

## 关键发现

1. **p2plane 损失方向正确**:P2S 拉到 83.76(等同 IterativePFN 论文 50K@1% 数字)
2. **Warm-start 反向漂移**:从 P0(70.94)出发 fine-tune 反而退步 1 分。原因:5e-5 LR 让 P0 ckpt 沿 p2plane 梯度方向微调,但因为 P0 已经在 dsm 损失的最优点,任何方向都是离开最优
3. **不需要 repulsion**:GT 法向已知 + paired 数据,p2plane 已经隐式包含了"贴表面"约束,repulsion 反而引入噪声梯度

## 文件

- 训练脚本:`train_p1.py`
- 损失:`losses/surface.py`(p2plane + repulsion)
- ckpts:`ckpts/p1_{balanced,p2plane_only}/best.pkl`

## 历史意义

确认了"法向切平面"是免费的 P2S 提分。后续所有实验都默认带这个损失。
