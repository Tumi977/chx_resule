# v09 — Capacity Scale-up

## 一句话

**显存只用 5GB(48GB 中),所以测试模型容量是不是天花板。两个大模型(0.70M / 1.15M)scratch 训练,验证"baseline 的 0.38M 是真容量瓶颈"假设。**

## 思路

之前所有实验都用 0.38M 参数(DGCNN 256/128)。论文里:
- IterativePFN: 4 stage × ItM = 1.5M
- PD-LTS light: 0.68M(含 INN)
- PD-LTS main: 几 M

**我们一直没测试更大容量模型对我们任务的实际影响**。

## 实验设计

| GPU | 实验 | encoder | 参数 | epochs | scratch? |
|---|---|---|---|---|---|
| 6 | `p1_dense_big` | DenseDGCNN g=128/l=4 | **0.70M** | 50 | ✓ |
| 5 | `p3_mlgc_big` | MLGC g=128/l=6 | **1.15M** | 60 | ✓ |

## 论文方法(同 v02/v03)

- DenseDGCNN skip-concat (PD-LTS-style)
- MLGC dense connection + in_proj 输入丰富 (PD-LTS §3.2)

## 配置共同点

```
batch_size    : 4 × n_patches=4 = 16 effective
patch_size    : 1024
init          : scratch (因为架构变了不能 warm)
lr            : 1e-4 → 1e-6 cosine
loss          : 1.0·dsm + 30·p2plane (与 v00/v01 同)
val_every     : 5 epochs
val_subset    : 20 shapes
```

## 时序

| 实验 | 单 epoch 时长 | 总训练时长 |
|---|---|---|
| p1_dense_big (0.70M) | ~12 min | ~10 hours (50 ep) |
| p3_mlgc_big (1.15M) | ~16 min | ~16 hours (60 ep) |

## 结果

| 实验 | step 0 dsm | best subset | FULL val | 状态 |
|---|---|---|---|---|
| p1_dense_big | 1.09 | TBD | TBD | 🟢 ep 0,GPU 6 |
| p3_mlgc_big | 1.64 | TBD | TBD | 🟢 ep 0,GPU 5 |

(scratch 起步 → step 0 loss ≈ 1-2,健康)

## 假设与预期

| 假设 | 验证标准 |
|---|---|
| **0.38M 是容量瓶颈** | 大模型 FULL > 70.94 → ✓ |
| **0.38M 已饱和,问题在方法** | 大模型 FULL ≤ 70.94 → 容量不是关键,要靠 v06/v07/v08 等论文方法 |

## 与其他实验的关系

- v02 (DenseDGCNN g=64,0.27M):**容量太小**
- v09 (DenseDGCNN g=128,0.70M):**中等容量**
- v03 (MLGC g=64,0.41M):**与 baseline 同级**
- v09 (MLGC g=128/l=6,1.15M):**大容量**

这个矩阵让我们能 ablate:**架构 × 容量** 是否 monotonic 影响分数。

## 文件

- 训练脚本:`train_p1.py --encoder_kind {dense_dgcnn,mlgc} --{dense,mlgc}_growth 128 ...`
- ckpts:`ckpts/p1_dense_big/best.pkl`,`ckpts/p3_mlgc_big/best.pkl`(都运行中)

## 历史意义

**第一次系统性测试容量影响**。如果两个大模型都超 70.94,后续所有冲分实验默认用 ≥0.7M 参数。如果不超,**说明 0.38M 已经能支撑 paired 数据的所有信息,问题在算法**。
