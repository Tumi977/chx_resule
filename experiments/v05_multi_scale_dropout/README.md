# v05 — Multi-Scale + Point Dropout

## 一句话

**实施"被迫学局部"假设(用户提出):patch_size 多尺度采样 + 训练时随机丢弃 10-30% 点,迫使网络从不完整局部数据学几何。本地分 70.93,几乎打平 P0。**

## 思想来源

**用户的洞察**:数据集类别单一,网络可能过度依赖整体形状先验。如果训练时把 patch 拆得更小、更不完整,网络会被迫只学局部几何(曲率、平面、边缘),提升泛化。

文献佐证:
- **Patch 训练本身**就是这种思想的标准做法(StraightPCF / IterativePFN / PD-LTS 都不让网络看完整物体)
- **patch 尺度多样化**:StraightPCF 训练时 patch ∈ [512, 2048] 随机
- **点 dropout**:模拟扫描丢点,Pointfilter / DROPS 在用

## 实现方法

### 1. Multi-scale patch_size

```python
# 在 dataset 内每 32 个 idx 切换一次 patch_size
group = (idx // 32) % len(patch_size_choices)
patch_size = patch_size_choices[group]   # ∈ {768, 1024, 1536}
```

每次 dataloader 取 32 个连续 idx 是同一 patch_size,batch 内一致(避免 collate 失败)。

### 2. Point dropout

```python
# 50% 概率应用 dropout
if rng.random() < 0.5:
    keep_frac = 1.0 - rng.uniform(0.10, 0.30)   # 保留 70-90%
    n_keep = int(patch_size * keep_frac)
    keep_idx = rng.choice(patch_size, n_keep, replace=False)
    # 重复填充回 patch_size 维度(保持 batch 一致)
    rep = rng.choice(n_keep, patch_size - n_keep, replace=True)
    full_idx = concat([keep_idx, keep_idx[rep]])
    pat[p] = pat[p, full_idx]
```

被 drop 的点位置用现有点填充(保 patch_size 不变),网络看到的"信息量"等同于 0.7-0.9× 实际尺寸。

## 配置(p1_multi_scale_dropout)

```
encoder         : DGCNN 256/128
params          : 0.38M
init            : ckpts/p0_baseline/best.pkl  (warm-start)
epochs          : 25
lr              : 5e-5 → 1e-6
loss            : 1.0·dsm + 30·p2plane (no rep)
patch_choices   : {768, 1024, 1536}
dropout_p       : 0.5 (50% probability per shape)
dropout_min/max : 10% / 30% drop fraction
```

## 结果

| 实验 | best subset | **FULL val** | CD | P2S | vs P0 |
|---|---|---|---|---|---|
| p1_multi_scale_dropout | 64.18 | **70.93** | 56.96 | 84.91 | -0.01 |

## 关键发现

1. **p2plane warm-start 实验里,这是最接近 P0 的**:70.93 vs P0 70.94,差 0.01 分(基本打平)
2. **CD 项最高(56.96)** in 所有 P1 fine-tune 实验:multi_scale + dropout **的确帮助 CD**(其他 p1_balanced/p1_p2plane_only 是 56.00)
3. **P2S 项最高(84.91)** in 所有 P1 fine-tune 实验:被迫学局部 → 表面贴合更好
4. 但仍未超过 P0:warm-start LR 漂移问题没解决

## 用户假设的验证结果

| 假设 | 验证结果 |
|---|---|
| "数据单一导致泛化差" | **未明确证伪也未证实**:这一项与 P0 几乎相等,差 0.01 |
| "被迫学局部能提鲁棒性" | **CD 确实从 56.00 → 56.96(+0.96)**,P2S 从 83.76 → 84.91(+1.15)。**两个分项都涨,只是 warm-start 漂移把总分拉平了** |

如果**从 scratch 训练 60 epoch 加 multi_scale + dropout**,**预期会超过 P0**(因为没有 warm-start 漂移问题)。这是一个未来值得做的实验。

## 文件

- 训练脚本:`train_p1.py --patch_size_choices "768,1024,1536" --point_dropout_p 0.5 --point_dropout_min 0.10 --point_dropout_max 0.30`
- 数据加载改造:`datasets/pc_dataset.py`(TrainConfig 新增字段 + `__getitem__` 内的 dropout)
- ckpts:`ckpts/p1_multi_scale_dropout/best.pkl`
