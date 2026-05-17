# 72.16 baseline ckpt archive (frozen 2026-05-17 19:07)

## 这是 V3 项目当前的最优 ckpt

**FULL val score = 72.16**(本地 99-shape validation,σ=0.01)

## 来源

- 实验:`p3_mlgc_big`
- 训练时间:2026-05-16 15:03 → 2026-05-17 10:17(~19h,60 epoch)
- 对应原始日志:`logs/p3_mlgc_big/train.log`

## 文件

- `best.pkl`(md5 b3aa12e67b126a74ccb61d76a34eedc0):验证集 best subset score 期间保存的权重(对应 ep 30 附近)
- `last.pkl`(md5 d16a125698949dee03b726502eb7970d):60 epoch 训练完毕的最终权重

**FULL val 的 72.16 是用 best.pkl 算出来的**,不是 last.pkl。

## 模型结构(加载所需)

```
encoder_kind   = "mlgc"
mlgc_growth    = 128
mlgc_layers    = 6
encoder_dim    = 384       # 不是默认 256
encoder_k      = 16
dec_hidden     = 128       # MLP decoder hidden
n_params       = 1.15M
```

## 训练配方(不可丢)

```bash
bash scripts/launch_exp.sh <gpu> <name> train_p1.py \
    --init_ckpt "" \
    --encoder_kind mlgc --mlgc_growth 128 --mlgc_layers 6 --encoder_dim 384 \
    --epochs 60 \
    --batch_size 4 --n_patches 4 --patch_size 1024 \
    --lr 1e-4 --lr_min 1e-6 \
    --noise_min 0.005 --noise_max 0.020 \
    --num_workers 4 \
    --lambda_dsm 1.0 --lambda_p2plane 30.0 \
    --val_every 5 --val_subset 20
```

## 评测细节

- CD score:57.99 / 100
- P2S score:86.33 / 100
- Total: **72.16 / 100**
- 评测在 99 shapes 验证集上,σ=0.01 高斯噪声

## 提交命令(直接用)

```bash
cd /mnt/ssd4t/data/chx/graphics/claude2/v3_pcd_mlgc_agt && \
bash scripts/jt_python.sh inference/predict_testset.py \
    --ckpt ckpts/_archive_72.16_p3_mlgc_big/best.pkl \
    --encoder_kind mlgc \
    --mlgc_growth 128 --mlgc_layers 6 \
    --encoder_dim 384 \
    --out_zip outputs/result_72.16.zip \
    --testset
```

## 不要动这个目录

`ckpts/p3_mlgc_big/` 是原始训练目录,以防误删。
`ckpts/_archive_72.16_p3_mlgc_big/` 是本次冷冻副本,**任何后续训练都不会覆盖到这里**。
