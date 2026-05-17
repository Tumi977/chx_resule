# v00 — Score-Denoise Baseline (P0)

## 一句话

**最强 baseline:修复了 starter_code 的三个 bug 后,纯 frame-DSM 监督的 DGCNN 降噪器,本地 99-shape FULL val 70.94。**

## 论文方法

| # | 方法 | 论文出处 | 在本实验中的角色 |
|---|---|---|---|
| 1 | **Frame-level DSM 监督** | Score-Denoise (Luo & Hu, ICCV 2021) | 主损失。训练时随机选 128 anchor 各取 32 邻居,对 4096 个 frame 点全部监督位移,而不是只监督 128 个 anchor |
| 2 | **Dynamic Edge Convolution** | DGCNN (Wang et al., TOG 2019) | Encoder backbone:4 层 EdgeConv,每层 KNN k=16 |
| 3 | **Patch-based 训练** | StraightPCF (CVPR 2024) / IterativePFN (CVPR 2023) 的标准做法 | 每个 batch 取 1024 点 patch,FPS 中心 + KNN 近邻 |
| 4 | **Gaussian-weighted patch stitching** | IterativePFN §3.4 | 推理时 r_s = r/3 的高斯加权融合,替代 starter 的 hard-pick |

## 修复的 3 个 bug(对比 starter_code)

| Bug | 原因 | 修复 |
|---|---|---|
| 1. 退化 DSM | starter 只在 num_train_points 个随机点上算 loss | frame-level 全 frame 监督 (32×) |
| 2. Hard-pick patch 融合 | starter 每点选权重最高的单个 patch | Gaussian 软融合 (r_s=r/3) |
| 3. 49926 vs 50000 输出 | 某些点未被任何 patch 覆盖时丢失 | 强制点对齐:未覆盖点用 noisy 输入填充 |

## 配置

```
encoder       : DGCNN, k=16, embedding_dim=256, dec_hidden=128
params        : 0.38M
patch_size    : 1024
batch_size    : 4 (× n_patches=4 = 16 effective)
epochs        : 60
lr            : 1e-4 → 1e-6 cosine
noise σ       : Uniform[0.005, 0.020]
loss          : frame_dsm only (no surface)
```

## 结果

| 指标 | 值 |
|---|---|
| **FULL validation (99 shapes)** | **70.94** |
| best subset score (20 shapes) | 65.42 |
| CD score | ≈57 |
| P2S score | ≈85 |
| 训练时长 | 21h 24min |

## 文件

- 训练脚本:`train_p0.py`(复制)
- 损失:`losses/dsm.py`(frame-DSM)
- 推理:`inference/patch_denoise.py`(Gaussian 融合)
- ckpt:`ckpts/p0_baseline/best.pkl`
- log:`logs/p0_baseline/train.log`

## 分析

**这是迄今为止最稳定的高分实验**。所有后续 fine-tune(p1_balanced, p1_p2plane_only)从 P0 出发反而退步 1 分(69.78-69.88),原因是 warm-start LR 5e-5 把 P0 ckpt 微调漂走。**P0 60 epoch 充分训练这件事比任何 fine-tune 都重要**。

## 历史意义

- **超过历史在线提交分 70.36(+0.58)**
- 是后续所有实验的"零点"基线
- 修 bug 这一步贡献最大(58.13 ep4 → 65.42 ep24 → 70.94 final)
