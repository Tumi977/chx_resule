# v04 — Density-aware Chamfer Distance (DCD)

## 一句话

**实现 NeurIPS 2021 的 Density-aware Chamfer Distance,用密度感知的 one-to-one 软分配代替普通 Chamfer 的 multi-to-one 塌缩。第一版 warm-start 后 loss=0(无梯度),第二版 scratch 得到健康信号。**

## 论文方法

| 方法 | 论文出处 | 在本实验中的角色 |
|---|---|---|
| **Density-aware Chamfer Distance** | DCD (Wu et al., NeurIPS 2021) | 主损失修正:用 `(1 - exp(-d²)) · (1 - exp(-α·n/N))` 代替原 CD,惩罚多对一塌缩 |
| **Frame-DSM + p2plane** | 同 v00 / v01 | 辅助损失 |

## 损失公式

```
对每个 pred 点 a_i:
    j*(i) = argmin_j ||a_i - b_j||²        # 找 GT 中最近邻
    d²_i  = ||a_i - b_{j*(i)}||²          # 距离平方
    n_j   = #{i: j*(i) = j}                # b_j 被多少个 pred 指向
    weight_i = 1 - exp(-α · n_{j*(i)} / N) # 拥挤度权重 ∈ [0, 1]
    L_i   = weight_i · (1 - exp(-d²_i))   # 越拥挤 + 越远 → 越大

L_DCD = mean over (pred → gt) + mean over (gt → pred)   # 双向
```

**关键洞察**:
- 普通 Chamfer 不惩罚多对一(N 个 pred 都映到同一 GT 点 → 0 损失)
- DCD 通过 `weight_i` 软-惩罚拥挤分配
- `(1 - exp(-d²))` 把 loss 限制在 [0, 1],对离群点 robust

## 子实验

| 子实验 | init | α | λ_dcd | 状态 | 诊断 |
|---|---|---|---|---|---|
| `p1_dcd_first_attempt` | P1 ckpt | 200 | 1.0 | 死(早) | 我把 init 选错了 |
| `p1_dcd_v2_zero_loss` | P0 ckpt | 200 | 2.0 | done | DCD 始终 0.0000 → **warm-start 后 d² 已极小,DCD 退化为普通 CD** |
| `p1_dcd_v3` | **scratch** | **50** | 2.0 | 🟢 ep 13/50 | dcd loss 从 0.0055 起步,正常 |

**关键修复**:
1. `init_ckpt = ""` → scratch,让 DCD 在欠训练模型上发挥作用
2. `α = 50`(不是 200)→ 拥挤度权重不会过早饱和

## 配置(p1_dcd_v3)

```
encoder         : DGCNN 256/128
params          : 0.38M
init            : scratch
epochs          : 50
lr              : 1e-4 → 1e-6
loss            : 1.0·dsm + 30·p2plane + 2.0·DCD(α=50)
```

## 结果

| 实验 | best subset | FULL val | CD | P2S |
|---|---|---|---|---|
| p1_dcd_v2_zero_loss | 64.50 | 70.55 | 56.60 | 84.50 |
| p1_dcd_v3 | 跑中 | TBD | TBD | TBD |

## 关键发现

1. **v2 失败的根本原因不是公式本身**,而是**应用时机**:warm-start 模型的 d² 已经极小,DCD 项数值接近 0,实际等同于普通 frame-DSM 训练
2. **DCD 论文的实验是从 scratch 开始的**,所以早期梯度大,后期自然衰减;这正是它设计的工作模式
3. v3 起步 dcd=0.0055,5 epoch 后降到 0.0001 → 已经几乎完成它的"反塌缩"工作。**DCD 是一个早期监督**,不是长期主损失

## 文件

- 训练脚本:`train_p1.py --lambda_dcd 2.0 --dcd_alpha 50`
- 损失:`losses/dcd.py`
- ckpts:`ckpts/p1_dcd_v{2,3}/best.pkl`

## 历史意义

诊断了 DCD 的"应用时机"陷阱:**好损失也会因为 warm-start 时机而失效**。这是同一篇论文方法在不同启动条件下的两种命运。
