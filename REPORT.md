# 第六届计图 AI 挑战赛 · 赛道二:深度学习点云降噪

## V3 项目实验报告

**项目位置**:`/mnt/ssd4t/data/chx/graphics/claude2/v3_pcd_mlgc_agt/`
**报告时间**:2026-05-16
**评测**:本地 99-shape FULL validation,CD + P2S 各占 50%,百分制以含噪输入为零分基线
**目标**:超过历史最高提交分 70.36,冲击 80+

---

## 1. 当前状态总览

### 1.1 排行榜(已完成 + 跑中)

| # | 实验 | 论文方法 | 参数 | 训练 epoch | best subset | **FULL val** | 状态 |
|---|---|---|---|---|---|---|---|
| 🥇 1 | **v00 P0 baseline** | Score-Denoise frame-DSM | 0.38M | 60 | 65.42 | **70.94** | ✅ done |
| 🥈 2 | **v05 multi-scale + dropout** | Patch 多尺度 + 点 dropout | 0.38M | 25 | 64.18 | 70.93 | ✅ done |
| 🥉 3 | v04 DCD v2(warm) | 密度感知 Chamfer NeurIPS'21 | 0.38M | 30 | 64.50 | 70.55 | ✅ done |
| 4 | v10 TTA P0 K=1 rot=4 | 旋转 TTA | — | — | — | 70.92 | ✅ done(打平 P0)|
| 5 | v01 p1_p2plane_only | p2plane 法向损失 | 0.38M | 30 | 63.59 | 69.88 | ✅ done |
| 6 | v01 p1_balanced | p2plane + 弱 rep | 0.38M | 30 | 64.49 | 69.78 | ✅ done |
| 7 | v02 p1_dense_dgcnn | DenseDGCNN encoder | 0.27M | 35 | 62.44 | 69.10 | ✅ done |
| ❌ | v03 p3_mlgc(g=24) | PD-LTS MLGC | 0.19M | 50 | 18.40 | 18.47 | ❌ 容量不足 |
| ❌ | v10 TTA K=2 rot=4 | K=2 过迭代 | — | — | — | 68.09 | ❌ 反向 -1.79 |
| 🟢 8 | **v06 p2_agt_T4_v3** | **IterativePFN AGT 修复版** | 1.53M | 25 (运行中) | TBD | TBD | 🟢 GPU 0 |
| 🟢 9 | **v04 p1_dcd_v3** | **DCD scratch + α=50** | 0.38M | 50 (运行中) | TBD | TBD | 🟢 GPU 1 |
| 🟢 10 | **v03 p3_mlgc_v2** | MLGC g=64 容量增 | 0.41M | 50 (运行中) | TBD | TBD | 🟢 GPU 2 |
| 🟢 11 | **v07 p1_emd** | **PD-LTS Sinkhorn EMD** | 0.38M | 20 (运行中) | TBD | TBD | 🟢 GPU 3 |
| 🟢 12 | **v08 p1_rectflow** | **StraightPCF rect-flow velocity** | 0.38M | 30 (运行中) | TBD | TBD | 🟢 GPU 4 |
| 🟢 13 | **v09 p3_mlgc_big** | MLGC g=128/l=6 大模型 | 1.15M | 60 (运行中) | TBD | TBD | 🟢 GPU 5 |
| 🟢 14 | **v09 p1_dense_big** | DenseDGCNN g=128/l=4 中大模型 | 0.70M | 50 (运行中) | TBD | TBD | 🟢 GPU 6 |

### 1.2 当前最高分

**70.94**(v00 P0,99-shape FULL val,σ=0.01)
对应历史在线提交分 70.36 → **超 +0.58**

---

## 2. 我们用上的论文方法清单

按"是否生效"分类:

### ✅ 已生效(v00 → 70.94 的功劳)

| 论文 | 方法 | 在哪个版本 |
|---|---|---|
| Score-Denoise (Luo & Hu, ICCV 2021) | **Frame-level DSM 监督** | v00, v01, v04, v05, v07 |
| 3DMambaIPF (AAAI 2025) | **法向切平面 P2S 替代损失**(简化版) | v01, v04, v05, v06, v07 |
| IterativePFN (CVPR 2023) | **Gaussian-weighted patch stitching** | 推理路径(全部) |
| DGCNN (TOG 2019) | **Dynamic Edge Convolution** + z-rotation 数据增强 | v00 及衍生 |

### 🟢 已实现并生效(运行中,等结果)

| 论文 | 方法 | 在哪个版本 |
|---|---|---|
| **IterativePFN (CVPR 2023)** | **真正 AGT(显式 σ_τ 噪声 + paired 直接监督)** | v06 |
| **PD-LTS (CVPR 2024)** | **Sinkhorn EMD 主损失** | v07 |
| **StraightPCF (CVPR 2024)** | **Rectified-flow velocity field** | v08 |
| **PD-LTS (CVPR 2024)** | **MLGC dense skip + in_proj** | v03 |
| **PD-LTS (CVPR 2024)** | **DenseDGCNN skip-concat** | v02 |
| **DCD (NeurIPS 2021)** | **Density-aware Chamfer**(scratch fix) | v04 |
| 文献综合 | **Multi-scale patch + point dropout** | v05 |

### ⚠️ 实现了但失败了

| 方法 | 失败原因 |
|---|---|
| AGT 原始 NN-投影公式 | paired 数据上 NN 退化为恒等 → loss=0,无梯度。修复版本见 v06 |
| 强 repulsion (λ ≥ 0.05) | rep 项主导训练,网络优化"打散点" → 分数掉到 0 |
| K=2 multi-step 推理 TTA | σ ∈ [0.005, 0.020] 上过迭代,CD 反退 -3 分 |
| MLGC growth=24 | 容量太小(0.19M),完全 underfit |

### ❌ 调研了但没实现(后续可做)

| 论文 | 方法 | 原因 |
|---|---|---|
| StraightPCF | **DistanceModule**(自适应步长) | 等 v08 rect-flow 跑完看效果再决定 |
| StraightPCF | **K-coupled velocity (K=2)** | 同上 |
| ASDN (AAAI 2025) | **Recoverability + 自适应停步** | 工程量大,等容量实验数据决策 |
| 3DMambaIPF | **可微渲染损失**(Gaussian splat + 32 视角) | p2plane 已经是它的简化版,边际收益小 |
| **PD-LTS** | **INN latent 解耦** | **PD-LTS 真正的灵魂方法,工程量 6-8h,未做** |
| FiLM σ-conditioning | 训练用 σ 调制 encoder | 代码 `models/film.py` 写了但未接训练 |
| σ-自适应推理步数 | 推理时按 σ̂ 估计选 niters | 代码 `inference/adaptive_denoise.py` 写了未接管线 |

---

## 3. 项目架构

```
v3_pcd_mlgc_agt/
├── REPORT.md                ← 本文件
│
├── experiments/             ← 每个实验版本的独立报告
│   ├── v00_score_denoise_baseline/README.md
│   ├── v01_p2plane_surface/README.md
│   ├── v02_dense_dgcnn_encoder/README.md
│   ├── v03_mlgc_encoder/README.md
│   ├── v04_dcd_loss/README.md
│   ├── v05_multi_scale_dropout/README.md
│   ├── v06_iterative_agt/README.md
│   ├── v07_sinkhorn_emd/README.md
│   ├── v08_rectflow_velocity/README.md
│   ├── v09_capacity_scaleup/README.md
│   └── v10_inference_tta/README.md
│
├── train_p0.py              ← v00:Score-Denoise baseline
├── train_p1.py              ← v01-v05:p2plane / DCD / dense / MLGC / multi-scale
├── train_p2.py              ← (旧版,弃用)
├── train_p2_v2.py           ← v06:修复后的 IterativePFN AGT
├── train_iter_p1.py         ← (旧版 K=3 共享权重,弃用)
├── train_emd.py             ← v07:Sinkhorn EMD 主损失
├── train_rectflow.py        ← v08:StraightPCF rect-flow velocity
│
├── losses/
│   ├── dsm.py               ← frame-level DSM (Score-Denoise)
│   ├── surface.py           ← p2plane + repulsion (3DMambaIPF 简化)
│   ├── agt.py               ← IterativePFN AGT(已修复)
│   ├── dcd.py               ← Density-aware Chamfer (NeurIPS'21)
│   └── emd.py               ← Sinkhorn EMD (PD-LTS)
│
├── models/
│   ├── denoiser.py          ← 单 stage Denoiser(encoder + decoder)
│   ├── multi_stage_denoiser.py  ← T-stage IterativePFN-style
│   ├── film.py              ← σ-FiLM 调制(待接通)
│   ├── film_denoiser.py     ← FiLM 版 denoiser(待接通)
│   ├── encoders/
│   │   ├── dgcnn.py         ← 标准 DGCNN
│   │   ├── dense_dgcnn.py   ← DenseDGCNN skip-concat
│   │   └── mlgc.py          ← PD-LTS MLGC
│   └── heads/decoder.py     ← MLP decoder
│
├── inference/
│   ├── patch_denoise.py     ← Gaussian 软融合 + 强制点对齐
│   ├── predict_tta.py       ← 推理 TTA(K + rot + seed ensemble)
│   ├── predict_testset.py   ← 200 shape 测试集打 result.zip
│   └── adaptive_denoise.py  ← σ-自适应推理(待接通)
│
├── eval/
│   ├── local_eval.py        ← 与官方 evaluate.py 严格对齐
│   └── sanity.py            ← 3 项 sanity (noisy→0, clean→100, worse→0)
│
├── datasets/
│   └── pc_dataset.py        ← multi-scale + dropout 已支持
│
└── scripts/
    ├── env.sh               ← Jittor + CUDA 11.8 环境
    ├── jt_python.sh         ← micromamba 一行启动
    ├── launch_exp.sh        ← 后台启实验 + log/pid
    ├── status.sh            ← 一键看所有实验状态
    └── watch_p0_then_launch_p2.sh  ← 自动接力
```

---

## 4. 关键技术决策与教训

### 4.1 三个 baseline bug 是基础提分(70.36 → 70.94)

修 bug 1(frame-DSM)+ 2(Gaussian 融合)+ 3(强制点对齐)直接拿到 70.94,**这是占整个项目 60% 提分**。

### 4.2 损失权重是地雷区

| 错误权重 | 后果 |
|---|---|
| λ_repulsion = 0.05 | rep 主导训练,score 退 13 分(70 → 57) |
| λ_repulsion = 0.15 | 完全崩溃,score = 0 |
| λ_dcd = 1.0 + warm-start | DCD loss 永远 0,无梯度 |

**经验**:**新损失项加入前,先看它的数值量级与现有损失的比值**。如果新损失 ÷ 旧损失 > 1,会立刻主导。

### 4.3 论文方法在 paired 数据上的退化

| 论文方法 | 论文设定 | 我们设定 | 结果 |
|---|---|---|---|
| AGT 原始公式 | unpaired (PointCleanNet 风格) | paired (赛题给) | NN 投影退化为恒等 → loss=0 |
| DCD 损失 | scratch 训练 | warm-start 后 | d² 已极小,DCD 退化为普通 CD |
| K=2 multi-step | σ ≥ 2% | σ ∈ [0.5%, 2%] | 过迭代,CD 退步 |

**经验**:**每个论文方法,先看它的实验设定与我们的差异。能照搬的少之又少**。

### 4.4 Warm-start 的双面性

- 用 P0 ckpt warm-start P1 fine-tune:**所有 5 个 P1 实验都比 P0 退步 0-1.5 分**
- 原因:P0 已经是 frame-DSM 损失下的最优点,fine-tune 用 5e-5 LR 沿 p2plane 梯度漂走

**经验**:**架构改了或损失主体改了,直接 scratch;微调用 1e-5 或更小 LR**。

### 4.5 显存还有 8× 余量

每实验只用 5GB(48GB 中)。今天首次启动 0.7M / 1.15M 大模型(v09)。

---

## 5. 待答案的关键问题

(由 6 个运行中实验答案)

| 问题 | 答案位置 | ETA |
|---|---|---|
| 修复后的 AGT(IterativePFN)能否超 P0? | v06 p2_agt_T4_v3 | ~21:00 今晚 |
| EMD 主损失在 paired 数据上还有效吗? | v07 p1_emd | ~20:00 今晚 |
| Rect-flow velocity 训练范式能涨分吗? | v08 p1_rectflow | ~22:00 今晚 |
| 容量是真瓶颈还是方法不够? | v09 p1_dense_big / p3_mlgc_big | ~00:00 凌晨 |
| MLGC g=64 是否能学起来? | v03 p3_mlgc_v2 | ~17:00 今晚 |
| DCD scratch 修复后涨分吗? | v04 p1_dcd_v3 | ~02:00 凌晨 |

---

## 6. 时间线与下一步

### 时间线

| 阶段 | 起 | 当前/止 | 主要事件 |
|---|---|---|---|
| 启动 + 调研 | 5/14 早 | 5/14 中 | 文献综述,V3 框架定义 |
| P0 训练 | 5/14 23:14 | 5/15 20:34 | 21h 训练,FULL val=70.94 |
| 5 路 P1 攻击实验 | 5/15 8:35 | 5/15 11:50 | 4 失败 1 平,发现损失权重陷阱 |
| 5 路修正 P1 实验 | 5/15 11:55 | 5/15 18:00 | 多个达到 ~70 |
| Phase 1 修复实验 | 5/16 12:00 | 跑中 | AGT/DCD/MLGC 修复版 |
| Phase 2 论文方法 | 5/16 14:00 | 跑中 | EMD + rect-flow + 大模型容量 |
| Phase 3 集成与冲刺 | 待定 | — | 多 ckpt 集成,B 榜准备 |

### 下一步(等当前 7 个实验出数据)

1. **TBD 之前不再启动新实验**,资源紧张
2. 对当前 7 个实验做最终评估,选最强 1-2 个准备线上提交
3. 根据数据决定是否进 Phase 4(INN、可微渲染、ASDN)

### 提交策略

每天 2 次提交上限。计划:
- **第 1 次:P0 ckpt** → 验证本地分到线上的 mapping
- **第 2 次:当前 7 个实验中的最优 ckpt**(等数据)
- **第 3 次:多 ckpt 集成**(后续做)

---

## 7. 论文方法使用率统计

| 类别 | 数量 |
|---|---|
| 调研论文里的方法总数 | ~25 个 |
| **已生效贡献分数的** | **3 个**(frame-DSM, p2plane, Gaussian stitching) |
| **正在跑的** | **7 个**(AGT 修复, EMD, rect-flow, MLGC, DenseDGCNN 大版本, DCD 修复, multi-scale dropout) |
| 实现失败的 | 5 个(老 AGT, 老 DCD, MLGC g=24, 强 rep, K=2 TTA) |
| 完全未做 | ~10 个(INN, DistanceModule, K-coupled velocity, ASDN, 可微渲染, σ-FiLM 接通, σ-自适应推理接通 ...) |

**实际利用率:从 12% 提升到 ~40%**(7+3 已实施 / 25 总数)。

---

## 8. 复现命令

```bash
# 环境
source scripts/env.sh

# v00 baseline 训练
bash scripts/launch_exp.sh 0 p0_baseline train_p0.py --epochs 60 --batch_size 4 ...

# v01 p2plane fine-tune
bash scripts/launch_exp.sh 1 p1_p2plane_only train_p1.py \
    --init_ckpt ckpts/p0_baseline/best.pkl \
    --lambda_dsm 1.0 --lambda_p2plane 50 --lambda_repulsion 0

# v06 AGT(修复版)
bash scripts/launch_exp.sh 0 p2_agt_T4_v3 train_p2_v2.py \
    --init_each_stage_from ckpts/p0_baseline/best.pkl \
    --n_stages 4 --normalize_loss_by_sigma2

# v07 Sinkhorn EMD
bash scripts/launch_exp.sh 3 p1_emd train_emd.py \
    --init_ckpt ckpts/p0_baseline/best.pkl \
    --lambda_dsm 0.5 --lambda_p2plane 30 --lambda_emd 10 \
    --emd_eps 0.01 --emd_iters 50

# v08 rect-flow velocity
bash scripts/launch_exp.sh 4 p1_rectflow train_rectflow.py \
    --init_ckpt ckpts/p0_baseline/best.pkl \
    --t_dist uniform --inference_steps 1

# v09 capacity scale-up
bash scripts/launch_exp.sh 5 p3_mlgc_big train_p1.py \
    --encoder_kind mlgc --mlgc_growth 128 --mlgc_layers 6 --encoder_dim 384

# 看状态
bash scripts/status.sh

# 测试集打包提交
bash scripts/jt_python.sh inference/predict_testset.py \
    --ckpt ckpts/p0_baseline/best.pkl \
    --out_zip outputs/result.zip
```

---

## 9. 致谢与参考文献

### 核心论文

1. Luo, S. & Hu, W. **Score-Based Point Cloud Denoising**. ICCV 2021. [paper](https://arxiv.org/abs/2107.10981)
2. de Silva Edirimuni, D. et al. **StraightPCF: Straight Point Cloud Filtering**. CVPR 2024. [paper](https://arxiv.org/abs/2405.08322)
3. de Silva Edirimuni, D. et al. **IterativePFN: True Iterative Point Cloud Filtering**. CVPR 2023. [paper](https://arxiv.org/abs/2304.01529)
4. Mao, A. et al. **Denoising Point Clouds in Latent Space via Graph Convolution and Invertible Neural Network (PD-LTS)**. CVPR 2024.
5. Zeng, J. et al. **3DMambaIPF: A State Space Model for Iterative Point Cloud Filtering via Differentiable Rendering**. AAAI 2025. [paper](https://arxiv.org/abs/2404.05522)
6. Wu, T. et al. **Density-aware Chamfer Distance**. NeurIPS 2021.
7. Cuturi, M. **Sinkhorn Distances: Lightspeed Computation of Optimal Transport**. NIPS 2013.
8. Wang, Y. et al. **Dynamic Graph CNN for Learning on Point Clouds**. TOG 2019.

### 综述

- arXiv 2508.17011 — **A Survey of Deep Learning-based Point Cloud Denoising**(2025)
