# v02 — DenseDGCNN Encoder

## 一句话

**把 DGCNN 改成 DenseNet 风格 skip-concat encoder(每层输入是之前所有层的拼接),验证多尺度感受野能否超过 P0。结果:不能。**

## 论文方法

| 方法 | 论文出处 | 在本实验中的角色 |
|---|---|---|
| **DenseEdgeConv 跳跃连接** | PD-LTS (Mao et al., CVPR 2024) | 取代 plain stacked DGCNN,每层 input = concat(x, h_1, ..., h_{t-1}) |
| **Dynamic KNN per layer** | DGCNN (TOG 2019) | 每层用当前特征空间重建图(每层 k=16) |

## 模型对比

| Encoder | 层数 | 单层输入 | 总参数 |
|---|---|---|---|
| DGCNN (v00) | 4 | 仅上一层输出 | 0.38M |
| **DenseDGCNN (v02)** | 4 | 原坐标 + 之前所有层 | 0.27-0.70M |

## 子实验

| 子实验 | growth | n_layers | embedding_dim | params | 训练 epoch |
|---|---|---|---|---|---|
| `p1_dense_dgcnn` | 64 | 4 | 256 | 0.27M | 35 |
| `p1_dense_big` | 128 | 4 | 384 | **0.70M** | 50(运行中) |

## 配置(p1_dense_dgcnn)

```
encoder        : dense_dgcnn, growth=64, n_layers=4
params         : 0.27M (比 DGCNN 0.38M 还少!)
init_ckpt      : "" (scratch)
epochs         : 35
loss           : 1.0·dsm + 50·p2plane
patch_size     : 1024
```

## 结果

| 实验 | best subset | **FULL val** | CD | P2S | vs P0 |
|---|---|---|---|---|---|
| p1_dense_dgcnn | 62.44 | **69.10** | 55.38 | 82.81 | -1.84 |
| p1_dense_big | TBD | TBD | TBD | TBD | TBD |

## 关键发现

1. **Dense skip 没有想象中那么强**:0.27M 参数从零起步,35 epoch 只到 69.10,**比 P0 同 epoch 数低 ~1 分**
2. **参数量是真的瓶颈**:dense skip 让前几层 input dim 小(3 → 67 → 131 → 195),实际可学参数比 DGCNN 还少
3. **PD-LTS 论文里 dense skip 之所以有效,是因为搭配了 INN(可逆神经网络)做 latent 解耦** — 我们没用 INN,光 encoder 不够

## 历史意义

- 确认 dense skip **单独使用** 不能突破 P0
- 提示真正路径:**或者参数量大幅放大(p1_dense_big 验证中)**,**或者补 INN 灵魂方法**(Phase 4.13 待办)

## 文件

- 训练脚本:`train_p1.py --encoder_kind dense_dgcnn`
- 模型:`models/encoders/dense_dgcnn.py`
- ckpts:`ckpts/p1_dense_dgcnn/best.pkl`(已 done)
- ckpts:`ckpts/p1_dense_big/best.pkl`(运行中)
