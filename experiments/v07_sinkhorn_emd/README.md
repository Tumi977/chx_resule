# v07 — Sinkhorn EMD Loss

## 一句话

**实现 PD-LTS 用作 SOTA 主损失的 Earth Mover's Distance(EMD,via Sinkhorn 50-iter 近似),替代 frame-DSM 作为主监督。EMD 强制 one-to-one 匹配,从根本上消除点云塌缩。**

## 论文方法

| 方法 | 论文出处 | 在本实验中的角色 |
|---|---|---|
| **Sinkhorn EMD** | Cuturi (NIPS 2013) + 在点云中应用见 PD-LTS (CVPR 2024) §3 | 主损失,one-to-one 匹配 |
| **Log-domain Sinkhorn iteration** | 同上,数值稳定版本 | 防小 ε 时数值溢出 |
| **frame-DSM(辅)** | Score-Denoise (ICCV 2021) | 辅助损失,加快收敛 |
| **p2plane(辅)** | 同 v01 | 辅助损失 |

## EMD vs Chamfer 的根本区别

```
Chamfer(A, B) = mean_a min_b ||a-b||² + mean_b min_a ||a-b||²
              ↑ 不是 bijection: 多个 a 可以映到同一 b

EMD(A, B)     = min_{π: A↔B 双射} mean_i ||a_i - π(a_i)||²
              ↑ 严格 bijection: 一个 b 只能匹配一个 a
```

**点云塌缩**(N 个 pred 都映到同一 GT 点)在 Chamfer 下损失为 0,但 EMD 下会很大。这是 PD-LTS 拿到 1-2% noise 档 SOTA 的关键之一。

## Sinkhorn 实现

精确 EMD 需要 Hungarian 算法 O(N³),不可行。Sinkhorn 用熵正则化近似:

```
Sinkhorn(A, B; ε) = min_{π doubly stochastic} <π, C> + ε·H(π)
```

其中 C[i,j] = ||a_i - b_j||² 是成本矩阵,H 是熵。迭代算法:

```
1. log_K = -C / ε                              # 核矩阵
2. Iterate T times:
     log_a = log(1/N) - logsumexp_j(log_K + log_b)   # 行边际归一化
     log_b = log(1/N) - logsumexp_i(log_K + log_a)   # 列边际归一化
3. π = exp(log_a + log_K + log_b)              # 最优传输计划
4. EMD ≈ Σ_ij π_ij · C_ij
```

## 配置(p1_emd)

```
model           : DGCNN 256/128
params          : 0.38M
init            : ckpts/p0_baseline/best.pkl  warm-start
epochs          : 20
batch_size      : 4 × n_patches=2 (× 2 instead of 4 due to EMD memory cost)
lr              : 5e-5 → 1e-6
emd_eps         : 0.01
emd_iters       : 50
loss            : 0.5·frame_dsm + 30·p2plane + 10·EMD
val_every       : 3 epochs (more frequent due to short total)
```

## 单元测试结果

| 测试 | 期望 | 实测 |
|---|---|---|
| `pred = gt` | ≈ 0 | 0.0071(熵正则项,正常) |
| `pred = gt + ε(0.02)` | 略大 | 0.0081 ✓ |
| `pred = gt 排列后` | ≈ 0 | 0.0071 ✓(双射识别 permutation) |
| **`pred 全部塌缩到 1 点`** | **HIGH** | **0.6390(90× 大)** ✓ |
| 反向传播梯度 | 非零 | 0.0065 ✓ |

**塌缩 case 90× 大** = EMD 真正禁止多对一,这是用它的核心理由。

## 计算成本

EMD 是 O(N²) 内存 + O(N² × n_iters) 计算:

```
N=1024 patch: 16 × 1024 × 1024 × 50 = 800M ops/step
单步耗时   : ~0.4s (vs frame-DSM 0.18s)
单 epoch   : ≈ 8 min  (vs frame-DSM 3 min)
```

为此把 n_patches 从 4 降到 2,batch 总数减半。

## 结果

🟢 **运行中**(0:55,ep 0 跑了一半)。step 0 emd=0.008(warm-start 后)。

## 文件

- 训练脚本:`train_emd.py`
- 损失:`losses/emd.py` + 单元测试 `scripts/test_emd.py`
- ckpts:`ckpts/p1_emd/best.pkl`(运行中)

## 历史意义

**首次实现真正的 PD-LTS SOTA 主损失**。如果它能超过 P0(70.94),验证了"EMD 比 frame-DSM 强"的论文论断;如果不行,说明 paired 数据 + p2plane 已经隐式做到了 EMD 的同等约束,EMD 边际收益小。
