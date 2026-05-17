# v06 — Iterative AGT (Multi-Stage Denoising)

## 一句话

**实现 IterativePFN 的 Adaptive Ground Truth multi-stage 训练框架。第一版用 NN 投影目标在 paired 数据上完全失效(loss=0),修复为"显式 σ_τ 噪声 + paired 直接监督"后训练信号正常。**

## 论文方法

| 方法 | 论文出处 | 在本实验中的角色 |
|---|---|---|
| **Multi-stage iterative denoising (T=4)** | IterativePFN (Sediri et al., CVPR 2023) | T 个独立权重 ItM(Iteration Module),依次处理减小的噪声水平 |
| **Adaptive Ground Truth (AGT)** | 同上,§3.2 | 每个 stage τ 见到的输入和监督是不同噪声水平 σ_τ = σ_0 · δ^(-τ) |
| **σ-geometric schedule** | 同上 | δ=2,T=4 → σ_T = σ_0 / 8(8× 衰减) |
| **Rectified-flow t-aug**(尝试过)| StraightPCF (CVPR 2024) | 训练时把当前状态向 GT 插值,扩展训练分布。**但被证明对 paired 数据不必要** |

## 核心 bug 与修复

### 原始(IterativePFN 论文做法,我们 v1 复制)

```python
y_τ = pc_clean + σ_τ · ξ                    # σ_τ 衰减的"残余噪声 GT"
target = NN(x̃, y_τ) - x̃                     # 找 y_τ 中的最近邻(投影)
```

这个公式在 IterativePFN 的 unpaired 数据上有效,但在我们的赛题(noisy 和 clean **point-wise paired**)上**完全退化**:

- **paired 数据 + σ_τ ≪ 点间距 → NN(x̃[i], y_τ) ≈ y_τ[i] 自己**
- target ≈ y_τ[i] - x̃[i] = (pc_clean[i] + σ_τ·ξ[i]) - pc_clean[i] = σ_τ·ξ[i]
- **网络被要求去预测一个随机噪声向量**,梯度信号是纯噪声 → loss 在 0 附近震荡,4000 步后 L0=L1=L2=L3 = 0.0000

### 修复(我们 v2/v3 的真正做法)

**思想**:抛弃 NN 投影,直接利用 paired 关系 + 显式生成 σ_τ 噪声。

```python
For τ in 0..T-1:
    σ_τ = σ_0 / δ^τ                                  # σ_τ 衰减
    ξ_τ ~ N(0, I)
    x_τ_input = pc_clean + σ_τ · ξ_τ                # 生成 σ_τ 噪声样本
    target_τ  = pc_clean - x_τ_input = -σ_τ · ξ_τ    # 确定性 GT
    pred_τ    = ItM_τ(x_τ_input)
    L_τ       = ||pred_τ - target_τ||² / σ_τ²        # σ² 归一化使所有 stage 等权重
```

每个 stage 学一个**针对自己 σ_τ 的去噪器**;推理时串行级联(高 σ → 低 σ)。

**这是 IterativePFN 论文的训练 intent 在 paired 数据上的正确表达**:他们论文里用 NN 投影是因为 unpaired 数据没有现成对应关系,**有 pairing 时直接监督更纯粹**。

## OOM 问题与第二次修复

T=4 multi-stage 在 batch=2 + n_patches=4 时 8 patch × 4 stage 的 KNN forward graph 都保留 → 显存 47GB+ OOM。

**修复**:**每个 stage 训练完立即 backward + zero_grad**(独立权重让这个 trick 安全):

```python
for tau in range(n_stages):
    x_input, target = make_stage_input(pc_clean, sigma_tau)
    pred = model.execute_stage(tau, x_input)
    l = stage_loss(pred, target)
    if tau == n_stages - 1:
        l += λ_p2plane · p2plane_loss(pred, x_input, pc_clean, normal)
    optimizer.step(l)   # ← 立即释放该 stage 的 forward graph
```

## 子实验时间线

| 子实验 | AGT 公式 | OOM 修复 | 起点 | 状态 |
|---|---|---|---|---|
| `p2_agt_multistage` | 旧 NN 投影 | 否 | warm | dry-run only |
| `p2_agt_T4_OOM_failed` | 旧 | 否 | warm | OOM 死 |
| `p1_iter_K3_first_attempt` | 旧 | 否 | warm | killed,确认 loss=0 |
| `p1_iter_K3_v2` | 旧 | 否 | scratch | killed,确认 loss=0 |
| `p2_agt_T4_v2_oom` | **新(显式 σ_τ)** | 否 | warm | OOM 死 |
| **`p2_agt_T4_v3`** | **新(显式 σ_τ)** | **是(per-stage backward)** | warm | 🟢 跑中(0:55) |

## 配置(p2_agt_T4_v3,当前)

```
model         : MultiStageDenoiser(T=4 × DGCNN 256/128) = 1.53M
init          : ckpts/p0_baseline/best.pkl  warm-start ALL 4 stages
epochs        : 25
batch_size    : 2 × n_patches=4 = 8 effective
lr            : 5e-5 → 1e-6
n_stages      : 4
sigma_delta   : 2.0  (σ_T = σ_0 / 8)
σ² normalize  : true (L_τ /= σ_τ²)
loss          : Σ_τ stage_loss + 30·p2plane(last stage only)
backward      : per-stage independent step
```

## 结果

| 实验 | step 0 loss | best | FULL | 备注 |
|---|---|---|---|---|
| 旧版本(NN 投影)| L0-L3 都 ≈ 0.0000 | — | — | 无梯度 |
| `p2_agt_T4_v2_oom` | L0-L3 ≈ 2-6 | — | OOM | 修对了公式但 OOM |
| **`p2_agt_T4_v3`** | **L0=2.32, L1=2.49, L2=3.21, L3=6.43** | 跑中 | TBD | **每个 stage 真正分化** |

## 关键发现

1. **AGT 在 paired 数据上需要重新设计公式**:照搬论文必死
2. **σ² 归一化让所有 stage 等权重训练**:否则 L0(大 σ)主导,L3(小 σ)永远学不到细节
3. **per-stage backward 是工程必需**:T=4 一起 backward 显存爆;独立 stage(IterativePFN 强调"独立权重")允许这个 trick

## 文件

- 训练脚本:`train_p2_v2.py`
- 损失:`losses/agt.py`(已重写,旧版做向后兼容 shim)
- 模型:`models/multi_stage_denoiser.py`
- ckpts:`ckpts/p2_agt_T4_v3/best.pkl`(运行中)

## 历史意义

**确认了"照搬论文公式可以失效"的根本风险**。从 NN 投影 bug 到显式 σ_τ 噪声修复,是这个项目最大的方法论教训。
