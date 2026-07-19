# CUDA W8A8 优化记录

本文记录 2026-07-15 在 `worldmodel.c` CUDA resident runtime 上完成的一条实验性 W8A8 推理路径，包括失败的第一版、后续融合、性能和质量结果，以及它与 TurboDiffusion 官方实现的差异。

## 结论先行

这条路径现在已经覆盖 WorldDiT 每层的 QKV、attention out projection、controller MLP 和主 MLP，并在 RTX 4090 D 48 GiB 上得到以下结果：

- Waypoint-1.5-1B（每 chunk 512 tokens）：Transformer 从 131.461 ms 降到 78.846 ms，约 `1.67x`；包含 VAE 的 RGB 吞吐从 29.10 提升到 46.81 FPS，约 `1.61x`。
- Waypoint-1.5-1B-360P（每 chunk 128 tokens）：自动排除不划算的 attention out projection 后，Transformer 从 36.608 ms 降到 32.924 ms，约 `1.11x`；RGB 吞吐从 98.94 提升到 109.27 FPS，约 `1.10x`。
- 全量 W8A8 时，6.56 GiB FP32/FP16 fallback device 权重不再分配，2.25 GiB 初始化专用条件投影在预计算后释放；扣除新增约 1.09 GiB INT8 权重后，相对原权重常驻方式净减少约 7.72 GiB，另有 20 MiB（main）或 5 MiB（360P）共享 scratch。

但它**不是默认路径，也不是可直接部署的质量结论**。启用时程序会明确打印 `experimental row/channel-wise PTQ` 警告。短程误差较小，32-chunk 自回归 rollout 的精确 latent 和逐像素指标会随时间明显分叉；目前缺少动作一致性、FVD/LPIPS 和足量人工盲评。因此默认 FP32/FP16 路径保持不变，W8A8 必须通过环境变量显式开启。

## 实现的量化规则

当前实现是对称、动态激活量化加静态权重量化：

### 激活：每 token/row 一个 scale

对于输入矩阵的一行 `A[r, :]`：

```text
sA[r]    = max(abs(A[r, :])) / 127
qA[r, k] = clamp(round(A[r, k] / sA[r]), -127, 127)
```

全零行约定 scale 为 1，量化值仍全为 0。scale 使用 FP32，量化值使用有符号 INT8。

### 权重：每 output channel 一个 scale

对于按输出通道组织的权重行 `W[n, :]`：

```text
sW[n]    = max(abs(W[n, :])) / 127
qW[n, k] = clamp(round(W[n, k] / sW[n]), -127, 127)
```

权重在加载时量化一次。QKV 仍保留原有 Q、K、V 拼接布局。

### GEMM 与反量化

CUTLASS Tensor Core 执行 `s8 x s8 -> s32`：

```text
acc[r, n] = sum_k int32(qA[r, k]) * int32(qW[n, k])
y[r, n]   = float(acc[r, n]) * sA[r] * sW[n]
```

使用的主 kernel 是 row-major A、column-major B、row-major INT32 输出，针对 SM80+ Tensor Core 的 `16x8x32` 指令形状。独立 probe 对真实模型的 9 组 GEMM shape 做过检查：采样 INT32 dot product 与 CPU 参考的 mismatch 为 0，反量化外积 scale 误差为 0，本次最终运行的最小线性输出 SQNR 为 37.37 dB。这只能证明量化边界和 GEMM 实现一致，不等于长程模型质量已经通过。

## 覆盖范围与融合边界

共享 A8、row-scale 和 INT32 scratch 在每层重复使用，避免按层常驻中间缓冲。各分支的实际数据流如下。

Runtime 初始化时会先检查 GPU 为 SM80+，并用实际 device pointer 对所有启用的 CUTLASS GEMM shape/layout 做 `can_implement` preflight；失败会在生成开始前明确终止。

| 分支 | 融合后的路径 |
|---|---|
| QKV | AdaRMSNorm + 动态 A8 → INT8 QKV GEMM → 反量化直接融合进 Q/K RMSNorm、RoPE 和 V 写出 |
| Attention out | attention FP32 输出动态 A8 → INT8 out GEMM → 反量化 + gate + residual |
| Controller | RMSNorm + A8 → fc1 → 反量化 + bias + SiLU + 下一层 A8 → fc2 → 反量化 + residual |
| MLP | AdaRMSNorm + A8 → fc1 → 反量化 + SiLU + 下一层 A8 → fc2 → 反量化 + gate + residual |

第二个线性层前的 A8 量化与 fc1 反量化/SiLU 放在同一 kernel 中；fc2 后的反量化与 gate/residual 放在同一 kernel 中。QKV 也不再把完整 FP32 投影结果写回显存后再做 Q/K 处理。

启用 W8A8 out projection 时，attention 不能直接留下原有的 half 输出，因为下一个边界需要从 FP32 attention 结果求动态 row max。对于 360P 的 `M=128` 小矩阵，这一成本超过 INT8 GEMM 的收益，所以自动配置会保留 FP16 out projection；main 的 `M=512` 则仍启用。

## 优化过程：第一版为什么变慢

### 1. 先做可验证的两阶段基线

第一版有意采用最容易检查的结构：

```text
单独量化 kernel → INT8 GEMM 写 INT32 矩阵 → 单独反量化 kernel
```

它让 probe 可以逐元素验证 INT32 和 scale，但在 resident runtime 中增加了 kernel launch、完整 INT32 中间矩阵的写回/读回，以及归一化、激活、残差之间的新边界。

在 360P 满 cache、最后 4 个 chunk 的同一测试口径下：

| 版本 | W8A8 ops | Transformer ms | 相对 FP 基线 |
|---|---|---:|---:|
| FP32/FP16 基线 | 无 | 36.608 | `1.00x` |
| 两阶段第一版 | all | 38.516 | `0.95x` |

单独打开 MLP、QKV 或 out 同样没有带来收益。这个结果说明“把乘法改成 INT8”本身不够；边界流量和 launch 成本必须一起消掉。

### 2. 融合量化、反量化与逐元素操作

后续按上一节的数据流逐项融合。360P 的一次分段 profile 也暴露了 out projection 的 crossover：融合后 MLP fc1 记录从 7.282 ms 降到 3.718 ms，MLP fc2 从 9.600 ms 降到 7.524 ms；但 out GEMM 分段反而从 2.864 ms 升到 3.592 ms。profile 的 cache/attention 段受异步归属和系统噪声影响，不能相加当作总耗时，这里只用它定位局部方向。

融合后的 360P `all` 为 34.944 ms，仅约 `1.05x`；排除 out 后为 32.924 ms，达到约 `1.11x`。因此 `WORLD_W8A8=1` 会根据 token 数做简单选择：`T <= 256` 时启用 `mlp,qkv,ctrl`，否则启用全部四类操作。

### 3. 消除 fallback 分配，并释放初始化专用权重

每层 `cond_proj_weight` 只用于初始化时预计算 conditioning modulation，24 层合计约 2.25 GiB；预计算完成后无论是否启用 W8A8 都可释放。

全部相关大线性权重同时保留 FP32 与 FP16 时约为 6.56 GiB，INT8-only 约为 1.09 GiB，因此去掉 fallback 的线性权重净省约 5.47 GiB。加上 2.25 GiB 初始化专用条件投影，总净减少约 7.72 GiB。早期实现先分配三种 dtype，再在初始化末尾释放 fallback，虽然降低了常驻显存，却使初始化峰值更高；最终实现对选中的 W8A8 层直接跳过 FP32/FP16 device 分配。

按默认自动策略，main 日志为：

```text
W8A8 shared scratch: 20.00 MiB (A8 4.00, INT32 16.00)
released 2.25 GiB of init-only/fallback layer weights; skipped 6.56 GiB of W8A8 FP fallback allocations
```

360P 自动保留 out fallback，日志为 5.00 MiB scratch、释放 2.25 GiB、跳过 6.00 GiB fallback 分配。以上是分配记账，不是 `nvidia-smi` 采样到的进程峰值；CUDA allocator、host 侧原始 checkpoint、attention/cache 和驱动工作区会让实际进程显存不同。

## 性能结果

测试环境：NVIDIA GeForce RTX 4090 D 48 GiB、driver 610.62、CUDA 12.9、Windows Release build。每次运行 24 层、4 denoising steps、`--cache-window 8`；一个 resident runtime 连续运行 12 个 chunk，报告最后 4 个满 cache chunk 的均值。每个 chunk 产生 4 个 RGB frame。

| 模型 | 配置 | Transformer ms | Total ms | RGB FPS | Transformer 加速 | 端到端加速 |
|---|---|---:|---:|---:|---:|---:|
| Waypoint-1.5-1B | FP32/FP16 | 131.461 | 137.470 | 29.10 | `1.00x` | `1.00x` |
| Waypoint-1.5-1B | W8A8 all | 78.846 | 85.465 | 46.81 | `1.67x` | `1.61x` |
| Waypoint-1.5-1B-360P | FP32/FP16 | 36.608 | 40.441 | 98.94 | `1.00x` | `1.00x` |
| Waypoint-1.5-1B-360P | W8A8 auto（无 out） | 32.924 | 36.615 | 109.27 | `1.11x` | `1.10x` |
| Waypoint-1.5-1B-360P | W8A8 all | 34.944 | 38.702 | 103.40 | `1.05x` | `1.05x` |

另一次 main 32-chunk 饱和运行的 Transformer 中位数为 131.661 ms（基线）和 80.299 ms（all W8A8），约 `1.64x`，与 12-chunk 结果方向一致。GPU 时钟、温度和后台负载会引起明显波动，因此表格只表示这台机器这组运行，不应外推成所有 Ada GPU 的固定比例。

## 质量检查与局限

### 单 chunk

在 360P、相同 seed/control 的单 chunk 对比中，W8A8 all 输出全部 finite：

| 指标 | 数值 |
|---|---:|
| latent cosine | 0.999824 |
| relative L2 | 0.01888 |
| MAE | 0.01662 |
| max absolute error | 0.190 |

这说明单次 forward 没有数值崩溃，但不能预测闭环自回归误差。

### 32-chunk 自回归 rollout

main 模型连续运行 32 个 chunk，即 128 个 RGB frame。完整 W8A8 all 对同 seed 的 FP32/FP16 轨迹得到：

| 精确 latent 指标（逐 chunk） | 结果 |
|---|---:|
| cosine，median / p5 / min / last | 0.9342 / 0.8164 / 0.7957 / 0.8331 |
| relative L2，median / p95 / max / last | 0.3660 / 0.5857 / 0.6217 / 0.5547 |
| cosine，前 8 chunk 均值 / 后 8 chunk 均值 | 0.9788 / 0.8371 |

将 32 个 latent chunk 分别经同一 VAE 解码后，对 128 对 RGB frame 做逐像素配对比较：

| RGB 指标 | 结果 |
|---|---:|
| PSNR，mean / median / p5 / last | 23.89 / 23.38 / 19.71 / 19.26 dB |
| PSNR，前四分之一 / 后四分之一均值 | 28.56 / 20.13 dB |
| SSIM，mean / median / p5 / last | 0.5820 / 0.5209 / 0.4465 / 0.4415 |
| SSIM，前四分之一 / 后四分之一均值 | 0.8004 / 0.4536 |
| RGB MAE，mean / median / last | 11.46 / 11.53 / 19.78 |

人工查看最终二进制生成的第 31 和第 127 帧时，两条轨迹仍都是结构合理且高度相近的游戏场景，武器、HUD、地形和视角大体一致；纹理、轮廓和局部位置已经有可见差异。这符合闭环世界模型对微小数值扰动高度敏感的特点，但不能用“系统会混沌分叉”来豁免质量验证：这些结果没有通过严格的 paired exact-state 门槛，也没有证明感知质量、控制响应或任务成功率等价。

作为误差/速度折中探索，仅量化最后 6 层 `[18,24)` 时，32-chunk 结果有所改善：all ops 的 latent cosine median/p5/last 为 0.9939/0.9316/0.9312，Transformer 中位数约 117.42 ms（相对同组 131.66 ms 基线约 `1.12x`）；排除 QKV、仅 `mlp,out,ctrl` 时为 0.9970/0.9395/0.8999 和约 118.68 ms。样本只有一条固定 rollout，尚不足以把这个配置定为“质量安全默认值”。

目前至少还缺少：

- 多 seed、多动作序列和更长 horizon；
- 与控制输入相关的轨迹/事件一致性指标；
- LPIPS、FVD 或特征空间视频指标；
- 对失败率、视觉伪影和可玩性的盲评；
- 不同 GPU、稳定时钟和重复运行的置信区间。

## 开关与环境变量

### W8A8 开关

| 变量 | 含义 |
|---|---|
| `WORLD_W8A8` | 未设置、`0`、`off`、`none`：关闭。`1`：自动选择，main 为 all，`T <= 256` 时排除 out。也可直接写 `all` 或逗号分隔的 ops。 |
| `WORLD_W8A8_OPS` | 非空时覆盖 `WORLD_W8A8` 中的 ops，但仍需 `WORLD_W8A8` 已启用。可选 token：`mlp`、`qkv`、`out`、`ctrl`/`controller`、`all`。 |
| `WORLD_W8A8_LAYER_BEGIN` | 首个量化层，默认 0；负值会 clamp 到 0。 |
| `WORLD_W8A8_LAYER_END` | 量化层的 exclusive end，默认本次实际运行层数；超过层数会 clamp。最终范围必须满足 `end > begin`。 |
| `WORLD_W8A8_DROP_FALLBACK` | 默认 1：不为已有 INT8 替代的线性层分配 FP32/FP16 device 副本。设为 0 保留高精度副本以便诊断，但当前 INT8 dispatch 失败仍会明确终止，不会自动重跑高精度路径。 |

辅助验证变量：

| 变量 | 含义 |
|---|---|
| `WORLD_TRANSFORMER_PROFILE=1` | 打开 Transformer 分段 CUDA timing。profile 段不应机械相加为总耗时。 |
| `WORLD_DUMP_RUNTIME_LATENT=PATH` | resident runtime 每个 chunk 将当前 latent 以 FP32 追加到文件；frame ordinal 0 会覆盖旧文件。用于固定 seed 的长程对比。 |

常见配置示例：

```powershell
# 关闭，使用默认 FP32/FP16 路径
Remove-Item Env:WORLD_W8A8,Env:WORLD_W8A8_OPS,Env:WORLD_W8A8_LAYER_BEGIN,Env:WORLD_W8A8_LAYER_END,Env:WORLD_W8A8_DROP_FALLBACK -ErrorAction SilentlyContinue

# 自动：main=all，360P=mlp,qkv,ctrl
$env:WORLD_W8A8 = '1'

# 强制所有四类线性层
$env:WORLD_W8A8 = 'all'

# 只选部分操作
$env:WORLD_W8A8 = 'mlp,qkv,ctrl'

# 仅量化最后 6 层，并保留 fallback 供调试
$env:WORLD_W8A8 = 'all'
$env:WORLD_W8A8_LAYER_BEGIN = '18'
$env:WORLD_W8A8_LAYER_END = '24'
$env:WORLD_W8A8_DROP_FALLBACK = '0'
```

## 构建与复现

### 构建和数值 probe

```powershell
cmake --build build-win-cudnn --config Release `
  --target worldmodel_raylib worldmodel_cuda --parallel

python test_w8a8_quantization.py

cmake -S tests/w8a8 -B build-w8a8 -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-w8a8 --config Release --parallel
.\build-w8a8\Release\worldmodel_w8a8_probe.exe --case all --samples 256
```

### 12-chunk 性能复现

以下先运行 1 个预热 chunk，再生成 11 个，取日志中最后 4 条 `live timing` 的均值。基线与 W8A8 应在相同 GPU 时钟/功耗状态下交错重复运行。

```powershell
$exe = '.\build-win-cudnn\Release\worldmodel_raylib.exe'

# Main baseline
Remove-Item Env:WORLD_W8A8,Env:WORLD_W8A8_OPS,Env:WORLD_W8A8_LAYER_BEGIN,Env:WORLD_W8A8_LAYER_END,Env:WORLD_W8A8_DROP_FALLBACK -ErrorAction SilentlyContinue
& $exe --model-dir .\Waypoint-1.5-1B --steps 4 --cache-window 8 `
  --warmup 1 --seed 1234 --headless-button 32 --headless-mouse 0.2 0 `
  --headless-smoke --headless-generate 11 2>&1 `
  | Tee-Object bench-fp16-main.log

# Main W8A8 all
$env:WORLD_W8A8 = 'all'
& $exe --model-dir .\Waypoint-1.5-1B --steps 4 --cache-window 8 `
  --warmup 1 --seed 1234 --headless-button 32 --headless-mouse 0.2 0 `
  --headless-smoke --headless-generate 11 2>&1 `
  | Tee-Object bench-w8a8-main.log

# 360P baseline；程序会使用 sibling main 模型的 VAE
Remove-Item Env:WORLD_W8A8 -ErrorAction SilentlyContinue
& $exe --model-dir .\Waypoint-1.5-1B-360P --steps 4 --cache-window 8 `
  --warmup 1 --seed 1234 --headless-button 32 --headless-mouse 0.2 0 `
  --headless-smoke --headless-generate 11 2>&1 `
  | Tee-Object bench-fp16-360.log

# 360P auto，日志应显示 mlp,qkv,ctrl，不含 out
$env:WORLD_W8A8 = '1'
& $exe --model-dir .\Waypoint-1.5-1B-360P --steps 4 --cache-window 8 `
  --warmup 1 --seed 1234 --headless-button 32 --headless-mouse 0.2 0 `
  --headless-smoke --headless-generate 11 2>&1 `
  | Tee-Object bench-w8a8-360.log
```

### 32-chunk latent 与 RGB 对比

每次运行前清理会影响 ops/layer range 的旧环境变量，确保只比较目标配置：

```powershell
$exe = '.\build-win-cudnn\Release\worldmodel_raylib.exe'

Remove-Item Env:WORLD_W8A8,Env:WORLD_W8A8_OPS,Env:WORLD_W8A8_LAYER_BEGIN,Env:WORLD_W8A8_LAYER_END,Env:WORLD_W8A8_DROP_FALLBACK -ErrorAction SilentlyContinue
$env:WORLD_DUMP_RUNTIME_LATENT = "$PWD\quality-fp16-main-32.f32"
& $exe --model-dir .\Waypoint-1.5-1B --steps 4 --cache-window 8 `
  --warmup 1 --seed 1234 --headless-button 32 --headless-mouse 0.2 0 `
  --headless-smoke --headless-generate 31

$env:WORLD_W8A8 = 'all'
$env:WORLD_DUMP_RUNTIME_LATENT = "$PWD\quality-w8a8-main-32.f32"
& $exe --model-dir .\Waypoint-1.5-1B --steps 4 --cache-window 8 `
  --warmup 1 --seed 1234 --headless-button 32 --headless-mouse 0.2 0 `
  --headless-smoke --headless-generate 31

Remove-Item Env:WORLD_DUMP_RUNTIME_LATENT -ErrorAction SilentlyContinue

# 一个 latent chunk 解码出 4 个 RGB frame，因此 --frames 32 最终写出 128 帧。
$cli = '.\build-win-cudnn\Release\worldmodel_cuda.exe'
& $cli --model-dir .\Waypoint-1.5-1B --vae-only `
  --latent .\quality-fp16-main-32.f32 --frames 32 --out .\quality-fp16.ppm
& $cli --model-dir .\Waypoint-1.5-1B --vae-only `
  --latent .\quality-w8a8-main-32.f32 --frames 32 --out .\quality-w8a8.ppm
```

latent 的 cosine/relative-L2 可直接用 NumPy 复算：

```powershell
@'
import numpy as np
a = np.fromfile('quality-fp16-main-32.f32', np.float32).reshape(32, -1)
b = np.fromfile('quality-w8a8-main-32.f32', np.float32).reshape(32, -1)
cos = np.sum(a*b, 1) / (np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1))
rel = np.linalg.norm(a-b, axis=1) / np.linalg.norm(a, axis=1)
print('finite:', np.isfinite(b).all())
print('cos median/p5/min/last:', np.median(cos), np.percentile(cos, 5), cos.min(), cos[-1])
print('rel median/p95/max/last:', np.median(rel), np.percentile(rel, 95), rel.max(), rel[-1])
'@ | python -
```

RGB SSIM 数据使用逐帧、全 RGB channel 的 `skimage.metrics.structural_similarity(..., channel_axis=2, data_range=255)`；PSNR 和 MAE 则从同一批 uint8 PPM 逐帧计算。生成的 `.f32`、PPM 和 benchmark log 都是临时验证产物，不应提交到仓库。

## 与 TurboDiffusion 官方 W8A8 的差异

当前代码实现的是本项目自己的 **per-row activation / per-output-channel weight PTQ**，不是 TurboDiffusion 官方 W8A8 kernel 的移植，也没有使用其量化 checkpoint。不能把 TurboDiffusion 论文或仓库的速度/质量数字当作本实现的验证结果。

TurboDiffusion 一手实现显示：

- [官方仓库与运行说明](https://github.com/thu-ml/TurboDiffusion)使用 `--quant_linear` 和量化 checkpoint 路径；仓库当前也把 autoregressive model support 列在 roadmap 中。
- [`Int8Linear` 实现](https://github.com/thu-ml/TurboDiffusion/blob/abb9d0944b941de3c03e55e37933d743551db21f/turbodiffusion/ops/core.py#L391-L432)的权重 scale shape 是 `ceil(out/128) x ceil(in/128)`。
- [官方量化 kernel](https://github.com/thu-ml/TurboDiffusion/blob/abb9d0944b941de3c03e55e37933d743551db21f/turbodiffusion/ops/quant/quant.hpp#L32-L98)对 A 和 W 都采用二维 `128 x 128` blockwise max-abs 量化：`scale=max(amax,1e-8)/128`，量化值饱和到 `[-128,127]`，而不是本项目的整行/整输出通道 `/127` 规则。
- [官方 GEMM kernel](https://github.com/thu-ml/TurboDiffusion/blob/abb9d0944b941de3c03e55e37933d743551db21f/turbodiffusion/ops/gemm/kernel.hpp#L390-L426)沿 K 维逐个 128 block 做 `S8 x S8 -> S32`，随即乘相应 A/W block scale 并累加到 FP32，最终写 FP16/BF16。这样每个 K block 可以使用不同 scale。
- [模型替换逻辑](https://github.com/thu-ml/TurboDiffusion/blob/abb9d0944b941de3c03e55e37933d743551db21f/turbodiffusion/inference/modify_model.py#L56-L73)替换 `model.blocks` 内除 `proj_l` 外的 linear；Norm 内部保持 FP32，bias 在 GEMM 后单独相加。官方 W8A8 没有使用 SmoothQuant、旋转、数据校准或量化感知训练，权重是直接 max-abs PTQ，激活是独立 kernel 动态量化；其 FusedNorm 也是单独优化，不是本实现这种 Norm→A8 融合。
- [论文 Figure 4](https://arxiv.org/html/2512.16093#S0.F4)中从 3182 s 到 2783 s、约 `1.14x` 的一项是 **W8A8 与 FusedNorm 合并加入** 的结果，不能把全部收益归给 W8A8。后面的主要数量级加速来自 rCM 与 SageSLA。
- 论文入口为 [TurboDiffusion: Accelerating Video Diffusion Models by 100-200 Times](https://arxiv.org/abs/2512.16093)。论文只支持理解其整体 W8A8 方向，具体数值规则应以代码和对应 checkpoint 为准。

两者的关键区别如下：

| 项目 | 本实现 | TurboDiffusion 官方实现 |
|---|---|---|
| Activation scale | 每 token row、跨完整 K | 每 `128 x 128` 二维 block |
| Weight scale | 每 output channel、跨完整 K | 每 `128 x 128` 二维 block |
| K 维反量化 | 完整 INT32 dot 后乘一对 scale | 每个 K=128 partial dot 立即按 block scale 转 FP32 累加 |
| 输出/边界 | 针对 WorldDiT 手工融合 QKV、SiLU、gate、residual | 通用量化 linear kernel，配套官方模型路径 |
| checkpoint | 原始 Waypoint 权重直接 PTQ | 官方说明提供/使用对应 quantized checkpoint；具体模型支持以仓库为准 |
| 自回归验证 | 本文的 32-chunk Waypoint 实验 | 官方仓库当前仍将 autoregressive support 列入 roadmap |

官方 blockwise 规则理论上能减少某个 token 或输出通道内不同 K 区段动态范围差异造成的误差，但会增加 scale 数量和 GEMM 内的 FP32 partial accumulation。是否能改善 Waypoint 的长程质量且仍保留当前速度，必须实际移植和测量，不能只凭粒度推断。

### 官方 kernel 的移植可行性探针

本轮还在仓库外的临时目录做了最小移植实验，未把第三方源码并入当前提交。TurboDiffusion commit `abb9d094` 的 8 个核心 header（约 1,191 行，[Apache-2.0](https://github.com/thu-ml/TurboDiffusion/blob/abb9d0944b941de3c03e55e37933d743551db21f/LICENSE)）去掉 PyTorch wrapper 后，可以用本项目现有 CUTLASS 3.5.1、CUDA 12.9 和 SM89 直接编译；RTX 4090 D 上 `128x128 block quant -> INT8 GEMM -> FP16` 与 CPU 按量化整数和 block scale 计算的 FP16 参考逐元素一致，`max_abs=0`。

只计 half activation quant 与 GEMM、权重量化不计入稳态的 microbench 为：

| M | QKV | out | MLP fc1 | MLP fc2 |
|---:|---:|---:|---:|---:|
| 128 | 0.0316 ms | 0.0355 ms | 0.0349 ms | 0.1041 ms |
| 512 | 0.0631 ms | 0.0358 ms | 0.0938 ms | 0.1165 ms |

这只证明“官方 kernel 能移植”，不代表它可机械替换当前 runtime。该 kernel 约使用 98.4 KiB dynamic shared memory、239 registers/thread，每个 SM 最多驻留一个 CTA；360P 的 out/fc2 只有约 16 个 CTA，欠占用明显。更重要的是它输出 FP16，而当前加速来自 Norm→A8、fc1→SiLU→A8、fc2→residual 的跨边界融合；完整替换必须重新设计这些接口后再做端到端 A/B。

## 后续路线

建议按以下顺序继续：

1. 移植或独立实现 TurboDiffusion 风格的 `128 x 128` blockwise A/W quant 和 K-block FP32 accumulation，先用现有 9-shape probe 与当前 row/channel 方案做同 shape、同输入对比。
2. 保留当前 WorldDiT 边界融合；不要退回完整 INT32 中间矩阵的两阶段实现。重点解决 blockwise scale 如何进入 QKV、SiLU 和 residual epilogue。
3. 离线生成 INT8 权重和 scale 文件，避免每次启动从原权重重新量化；随后评估能否不再加载 FP32 副本，从而降低初始化峰值和启动时间。
4. 做 layer/op sensitivity sweep，至少覆盖多 seed、多动作序列；用每层量化误差和 rollout 指标选出高精度保留层，而不是仅凭 GEMM 占比。
5. 比较全模型、最后 N 层、仅 MLP/controller、QKV 保留高精度等 Pareto 点，同时记录 Transformer ms、端到端 FPS、净显存、LPIPS/FVD、动作一致性和人工失败率。
6. 若纯 PTQ 仍不能通过长程门槛，再考虑 activation clipping 校准、quant-aware finetune 或蒸馏。TurboDiffusion 的量化 checkpoint 经验不能直接替代 Waypoint 自身校准。
7. 性能测量使用锁定/记录 GPU clocks、交错基线、至少 3 次重复和中位数/置信区间；质量门槛在速度配置成为默认值之前固定下来。

当前最合理的产品定位是：`WORLD_W8A8` 作为显式的高吞吐实验选项，默认路径继续保持原精度。下一阶段的核心不是再堆一个 INT8 GEMM，而是把官方 blockwise 数值策略、现有 WorldDiT 融合和长程自回归质量验证同时闭环。
