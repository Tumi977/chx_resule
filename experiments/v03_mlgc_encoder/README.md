# v03 — MLGC Encoder (PD-LTS-style)

## 一句话

**实现 PD-LTS 的 Multi-Level Graph Convolution encoder,带额外的 PointNet-style 输入丰富层。第一版容量太小完全失败,第二版 g=64 接近 70,第三版 g=128/l=6 大模型实验中。**

## 论文方法

| 方法 | 论文出处 | 在本实验中的角色 |
|---|---|---|
| **MLGC dense skip** | PD-LTS (Mao et al., CVPR 2024) | 主架构;每层 input = concat(原坐标 + projection + 之前所有层输出) |
| **In-projection (3 → 12 dim)** | PD-LTS §3.2 | 用一个 MLP 把原始 (x,y,z) 映射到 12 维特征,再喂给第一层 EdgeConv |
| **Final fusion conv** | PD-LTS §3.2 | 把所有层的输出 concat 后用 Conv1d 压成 embedding_dim |

## 与 DenseDGCNN(v02)的区别

| | v02 DenseDGCNN | v03 MLGC |
|---|---|---|
| 输入丰富 | 直接用原 3D 坐标 | 原坐标 + 12 维 in_proj 特征 |
| 第一层输入 | 3 维 | 15 维 |
| dense 拼接 | 第 t 层输入 = (x, h_1, ..., h_{t-1}) | 第 t 层输入 = (x, proj, h_1, ..., h_{t-1}) |

**这是 PD-LTS 的真正配方**(论文里叫 MLGC = Multi-Level Graph Conv)。

## 子实验

| 子实验 | growth | n_layers | emb_dim | params | epochs | 起点 | 状态 |
|---|---|---|---|---|---|---|---|
| `p3_mlgc` | 24 | 4 | 256 | 0.19M | 50 | scratch | ❌ 容量太小,best 18.40 |
| `p3_mlgc_v2` | 64 | 4 | 256 | 0.41M | 50 | scratch | 🟢 跑中(03h+) |
| `p3_mlgc_big` | **128** | **6** | **384** | **1.15M** | 60 | scratch | 🟢 跑中(0:30) |

## 配置(p3_mlgc_big,当前最大版本)

```
encoder         : mlgc, growth=128, n_layers=6, embedding_dim=384, in_proj_dim=12
dec_hidden      : 128
params          : 1.15M  (3x baseline DGCNN)
batch_size      : 4 × n_patches=4
epochs          : 60
lr              : 1e-4 → 1e-6
loss            : 1.0·dsm + 30·p2plane
```

## 结果

| 实验 | last ep | best subset | FULL val | CD | P2S |
|---|---|---|---|---|---|
| p3_mlgc(g=24) | 49 | 18.40 | **18.47** | 16.57 | 20.37 |
| p3_mlgc_v2(g=64) | 跑中 | 57.10 | TBD | TBD | TBD |
| p3_mlgc_big(g=128) | 跑中 | TBD | TBD | TBD | TBD |

## 关键发现

1. **g=24 失败**(0.19M):**MLGC 容量必须够大才能学到东西**,太小完全 underfit
2. **PD-LTS 论文 0.68M 的 light 版本也是有 INN 的**;光 MLGC + MLP decoder ≠ PD-LTS
3. **g=128/l=6 是接近 PD-LTS 主版本的容量**(1.15M),60 epoch 跑完才能下结论
4. 与 DenseDGCNN 对比有意义:**两者都是 dense skip,MLGC 多了 in_proj**;预期 v03 略强 v02

## 文件

- 训练脚本:`train_p1.py --encoder_kind mlgc`
- 模型:`models/encoders/mlgc.py`
- ckpts:`ckpts/p3_mlgc{,_v2,_big}/best.pkl`
