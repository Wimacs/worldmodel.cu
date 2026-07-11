# Vulkan Kernel Optimization Notes

这份文档记录 Vulkan 推理按版本推进的 kernel 优化。目标和 CUDA 路径一致：先用真实完整管线 profile 找最大耗时，再做一项有边界的实现、对拍和端到端 A/B；不围绕单个 shader 无休止调参数。

## 基线

固定测试配置：

- NVIDIA GeForce RTX 4090 D
- Waypoint-1.5-1B-360P，24 layers，4 denoise steps
- `cache-window=8`，连续生成 9 个 chunk，最后一个 chunk 已经历满 cache
- 每个 chunk 解码 4 个 RGB frame

旧默认路径使用 F32 activation/F32 weight linear shader。满 cache timestamp profile 为 `306.460 ms/chunk`：

- MLP fc2：`104.693 ms/120`
- MLP fc1：`63.140 ms/120`
- QKV：`33.108 ms/120`
- attention out：`18.176 ms/120`
- attention split-K part/reduce：`28.357 ms/120`
- 其余主要是 F32/NCHW VAE

因此第一版必须先处理 GEMM。此时 GEMM 族约占 dispatch 总时延的 72%，先做 VAE 或继续调 attention 都不是最大收益顺序。

## v1: Cooperative Matrix 默认路径

Vulkan runtime 已经有经过 probe 的 `GL_KHR_cooperative_matrix` shader，但过去只有显式设置 `WORLD_VULKAN_LINEAR_WF16_COOPMAT=1` 才会使用。v1 在设备同时支持 `shaderFloat16`、16-bit storage 和 cooperative matrix 时默认启用这条路径；`WORLD_VULKAN_LINEAR_WF16_COOPMAT=0` 保留旧 F32 回退。

默认路径包括：

- QKV：F32 activation、FP16 weight、FP32 accumulator，N32 cooperative-matrix tile。
- attention out/control：同样使用 N32 cooperative-matrix tile，并融合 gated residual。
- MLP fc1：GEMM epilogue 直接执行 SiLU 并写 FP16 hidden。
- MLP fc2：直接消费 FP16 hidden，GEMM epilogue 融合 gate 和 residual。

这对应 CUDA 中已经证明有效的三条经验：权重常驻 FP16、Tensor Core/Cooperative Matrix 主路径、消掉 GEMM 周围的 dtype bridge 和独立 epilogue kernel。

本版没有顺手默认启用以下实验：

- FP16 KV cache：当前满 cache attention A/B 几乎没有端到端收益，保留 `WORLD_VULKAN_KV_CACHE_F16=1`。
- MLP fc1 的 FP16 input：额外 cast 后 fc1 时间没有改善，保留 `WORLD_VULKAN_MLP_FC1_F16X=1`。
- GEMM split-K：当前 16x8 cooperative-matrix shader已经产生足够多 workgroup；真实 fc2 autotune 中 N8 是 `0.2131 ms`、N16 `0.2799 ms`、N32 `0.2296 ms`、WG4 `0.1988 ms`，没有 CUDA 大 threadblock 下的并行度不足问题。不能机械照搬 CUDA split-K。

### 结果

满 cache timestamp profile：

- 旧默认：`306.460 ms/chunk`
- v1 默认：`151.411 ms/chunk`
- dispatch 延迟降低 `50.6%`，等效吞吐提升约 `102.4%`

正常无 profile 的完整 resident runtime：

- 旧默认：`0.307 s/chunk`，约 `13.0 RGB FPS`
- v1 默认：`0.152 s/chunk`，约 `26.3 RGB FPS`
- 端到端吞吐提升约 `102%`

新默认在 profile/非 profile 两次运行中导出的 4 张 PPM 逐字节一致。与显式 F32 回退相比，4 张图平均像素绝对误差约 `0.035/255`，最大误差 `6/255`，符合 weight FP16 量化路径；各 cooperative-matrix、fused SiLU 和 gated residual probe 继续作为算子级 correctness gate。

## 下一版

v1 后满 cache 主要耗时变为：

- attention split-K part/reduce：约 `28.3 ms`
- MLP fc2：约 `24.2 ms`
- MLP fc1：约 `22.2 ms`
- QKV + attention out：约 `24.2 ms`
- F32/NCHW VAE：约 `33 ms`

下一版先分别建立 cooperative-matrix FMHA 和 FP16/NHWC VAE 的最小 probe。两者都必须先对拍，再由完整 profile 决定谁进入 runtime；不根据 CUDA 的百分比直接假设 Vulkan 会得到相同收益。

## v2: VAE 3x3 cooperative-matrix convolution

CUDA 的 FP16/NHWC VAE 不能直接照搬成一个 NCHW tensor-op kernel。第一版 Vulkan probe 保持 F32/NCHW activation 和 output，只把 OIHW weight 量化为 FP16，并让一个 subgroup 用 `16x16x8` cooperative matrix 计算 8 个输出通道。3x3 和 1x1 的 CPU reference 都在相同位置量化 activation/weight，两个 probe 都是 `max_abs=0`；但完整 VAE 的卷积时间从约 `32.7 ms` 增加到 `44.9 ms`。原因是每 8 个输出通道都重复一次带 padding 的 NCHW implicit-im2col gather。

v2 把输出 tile 扩成 32 通道：同一个 subgroup 维护四个 `16x8` FP32 accumulator，同一块 activation gather 在 32 个输出通道间复用。真实 VAE 的内部输出通道都是 64/128/256，能完整覆盖这条路径。N32 的 3x3 和 1x1 probe 同样为 `max_abs=0`。

连续 11 次 512x256、4-frame 的 VAE-only profile（去掉前三次后取后 8 次平均）：

- 旧 F32/NCHW C4：GPU dispatch `37.45 ms/chunk`，墙钟 `37.95 ms/chunk`
- N32 cooperative matrix：GPU dispatch `20.50 ms/chunk`，墙钟 `20.96 ms/chunk`
- VAE GPU 时间下降 `45.3%`，独立 VAE 吞吐约从 `105` 提到 `191 RGB fps`

1x1 在高分辨率下没有 padding/gather 负担，旧 C4 仍更快，因此 runtime 只把内部 3x3 卷积切到 N32 cooperative matrix；1x1 和最终 12-channel RGBA 卷积保留旧路径。这个混合策略的连续 VAE-only profile 稳定在 GPU dispatch `18.36 ms/chunk`、墙钟 `18.98 ms/chunk`，比把 1x1 也交给 cooperative matrix 再省约 `1.5 ms`。支持 `shaderFloat16`、16-bit storage 和 `VK_KHR_cooperative_matrix` 的设备默认启用，`WORLD_VULKAN_VAE_WF16_COOPMAT=0` 可回退。

完整 24-layer、4-step、`cache-window=8` 连续生成到满 cache 后做 `new -> fallback -> new` 夹测。最后 4 个 chunk 的 fallback 墙钟均值为 `154.87 ms/chunk`、`25.83 RGB fps`；两次默认新路径分别为 `136.31` 和 `136.12 ms/chunk`，平均约 `29.37 RGB fps`。端到端延迟下降约 `12.0%`，吞吐提升约 `13.7%`。同一 latent 的最终 RGB 与 F32 路径平均绝对像素差为 `0.031-0.034/255`，最大 `3/255`，目视一致。

这条路径仍保留 F32/NCHW activation 边界，所以不是 CUDA FP16/NHWC VAE 的终点。若后续 profile 再次指向 VAE，下一步应把 repeat/conv/ReLU/concat/upsample/tgrow 整条流改成 FP16/NHWC；不能继续只缩小 cooperative-matrix tile。

## v3: Native Sparse GQA FMHA

旧 attention split-K kernel 仍然按 query head 读取 K/V，并在每个 split 内用标量 FMA 完成 QK 和 PV。v3 针对当前模型固定的 `D=64` 和 128-token frame block 实现原生 sparse GQA FMHA：

- cache index collector 在原 dispatch 中同时生成有序物理 block table；ring wrap 后也不需要 gather K/V。
- 一个 subgroup 负责 16 个 folded GQA query 和一个 128-token 稀疏块。`Hq/Hkv` 个 query head 折叠到同一个 KV head，K/V 不复制到 query-head 维度。
- QK 和 PV 使用 `GL_KHR_cooperative_matrix`；Q、K、V 和 block-local probability 为 FP16，累加、softmax 统计和 split merge 为 FP32。
- score 和 probability 只存在于 shared memory。每个 sparse block 仅写出 64 维 accumulator 及 `(m, l)`，再由 reduce kernel按 online-softmax 公式合并。

当前 shader 固定 `QUERY_TILE=16`、`SPARSE_BLOCK=128`。实验过单 workgroup 处理 64 个 query 的四 subgroup 版本，但 shared memory 增至约 74 KiB，满 cache part 时间由 `10.966 ms/120` 退化为 `12.087 ms/120`，因此没有保留。支持 shader FP16、16-bit storage 和 cooperative matrix 的 `D=64` GQA 模型默认启用；`WORLD_VULKAN_SPARSE_GQA_FMHA=0` 可回退到旧 attention。

### Correctness

独立 probe 使用 `B=1, Hq=4, Hkv=2, Tq=32, Tk=384, D=64`，并故意按 `{2, 0}` 读取非连续物理块，覆盖 GQA 映射、稀疏 block table 和 ring wrap：

- cache index/block probe：write/read 两阶段均为 `mismatches=0`。
- Vulkan 对逐块 CPU online-softmax reference：`max_abs=1.397e-9`，`mean_abs=7.903e-11`。
- `test_vulkan_sparse_gqa_fmha.py` 读取 probe 原始张量并重建 PyTorch blockwise GQA reference，对拍通过。

完整 24-layer、4-step、`cache-window=8` 连续生成 11 个 chunk，所有 latent 均为 finite。相对旧路径，最后一个 chunk 的 latent 平均绝对误差约 `4.16e-4`、最大误差 `1.31e-2`；最终 RGB 平均绝对像素误差约 `0.054-0.057/255`、最大 `4/255`，目视一致。

### Performance

满 cache timestamp profile：

- 旧 split-K part/reduce：`28.260 + 0.994 = 29.254 ms/120`。
- sparse GQA FMHA part/reduce：`10.966 + 0.764 = 11.730 ms/120`。
- attention bucket 延迟降低 `59.9%`，完整 GPU dispatch 从 `136.087 ms` 降到 `117.024 ms`，降低 `14.0%`。

正常无 profile 的最后 4 个满 cache chunk：旧路径为 `135.81 ms/chunk`、`29.45 RGB FPS`；新路径为 `116.87 ms/chunk`、`34.23 RGB FPS`。端到端延迟降低 `14.0%`，吞吐提升 `16.2%`。

## v4: CTA-tiled GEMM and FP16 boundaries

旧 cooperative-matrix GEMM 由一个 subgroup 独立负责 `16x8/16/32` 输出块。它能使用 tensor core，但同一个 A tile 会被多个 subgroup 重复读取，且 MLP 的 RMSNorm、SiLU、FC2 和 residual 之间仍有 F32 中间张量。v4 为当前 360p 模型新增一个直接面向真实 shape 的 CTA mainloop：

- 固定 `CTA_M=64`，编译 `CTA_N=64/128`、`CTA_K=32/64` 变体；分别由 8 或 16 个 subgroup 协作。
- A/B 使用 `f16vec4` 合并加载，权重启动时转成 K-major FP16；A/B 同时进入 shared memory，再由 subgroup cooperative matrix 消费。
- K32 提供单 stage 和双 stage ping-pong 变体。Vulkan 没有这里可用的异步 copy，所以 p2 是 software pipeline：先把当前 cooperative-matrix fragment 读入寄存器，再填充下一块 shared stage，并在换 stage 前同步。
- epilogue 包含 F32 output、fused SiLU-to-FP16、fused gated residual，以及 split-K partial。split-K reduce 在一次 pass 中完成 partial sum、gate 和 residual。
- attention/MLP Ada RMSNorm、controller RMSNorm、controller add+SiLU 和 sparse FMHA reduce 可直接写 FP16，GEMM 不再需要独立 F32-to-FP16 cast。

FC2 的 `K=8192` 默认使用 split-K 4；split-K 2 仍作为候选。`T=128, D=2048` 下 N64/split4 产生 256 个 CTA，避免 unsplit FC2 只有 64 个 CTA 时的并行度不足。partial workspace 为 `4 * T * D * sizeof(float) = 4 MiB`，attention out 与 MLP FC2 按执行顺序复用。

### Autotune

`worldmodel_vulkan_gemm_autotune --epilogue model` 只搜索五个实际 shape：

| label | M | N | K | epilogue |
|---|---:|---:|---:|---|
| qkv | 128 | 4096 | 2048 | none |
| attn_out | 128 | 2048 | 2048 | gated residual |
| ctrl_fc | 128 | 2048 | 2048 | none |
| mlp_fc1 | 128 | 8192 | 2048 | SiLU-to-FP16 |
| mlp_fc2 | 128 | 2048 | 8192 | gated residual |

默认测量模拟模型的 streaming-weight 情况：每次 timestamp 前用 96 MiB workspace 连续做 64 次读写以拉起 GPU 时钟并驱逐 L2，每个候选取 5 次 cold dispatch 的中位数。cache v2 记录 `tile_m/n/k`、packed layout、split-K 和 shader 名；runtime 会按 shape 选择 stage1/p2 与 split 数，未命中则使用经过完整流水线验证的默认值。

RTX 4090 D 的一次稳定 cold-cache 搜索得到：

- qkv：CTA M64N64K64，`0.1033 ms`
- attn_out：CTA M64N64K32 split-K 4 stage1，含 reduce `0.0479 ms`
- ctrl_fc：旧 N8 subgroup kernel，`0.0603 ms`
- mlp_fc1：CTA M64N64K32 p2，`0.1320 ms`
- mlp_fc2：CTA M64N64K32 p2 split-K 4，含 reduce `0.1456 ms`

QKV 是一个重要的整模型例外。独立 cold probe 偏向 CTA，但 24-layer 满 cache A/B 中 CTA 为 `11.784 ms/120`，N32 subgroup 为 `11.218 ms/120`；完整 dispatch 分别为 `94.940` 和 `94.057 ms`。两条路径的四张 PPM 逐字节一致，因此保留 `WORLD_VULKAN_QKV_CTA=1` 作为实验开关，默认采用整模型更快的 N32。

`WORLD_VULKAN_GEMM_CTA=0` 可整体回退旧 GEMM，`WORLD_VULKAN_HALF_GEMM_BOUNDARY=0` 只关闭 FP16 边界，`WORLD_VULKAN_MLP_FC2_CTA_SPLITK=2|4` 可固定 FC2 split 数。设置 `WORLD_VULKAN_GEMM_AUTOTUNE_CACHE=PATH` 后，cache 中的 stage1/p2 和 split-K 选择优先于默认值。

### Correctness

`test_vulkan_gemm_cta.py` 会运行所有 packed CTA stage1/p2、none/SiLU/gated、split-K 2/4 变体，导出原始输入/权重/输出并与 PyTorch matmul 对拍；同时测试 FP16 RMSNorm、Ada RMSNorm 和 add+SiLU。当前 CTA probe 的 `max_abs=0`、`mean_abs=0`，FP16 boundary 三项同样为 0。`test_vulkan_sparse_gqa_fmha.py` 额外检查 FMHA FP16 reduce，结果与 `reference.half()` 完全一致。

完整 QKV CTA/N32 A/B 的四张最终图 SHA256 一致。此前 FP16 boundary 对 F32 回退的完整图像平均绝对像素误差约 `0.039/255`、最大 `2/255`。

### Performance

360p、24 layer、4 step、`cache-window=8` 满 cache timestamp profile：

- v3 起点：`118.194 ms` GPU dispatch。
- v4 默认：`94.057 ms` GPU dispatch，延迟降低 `20.4%`，等效吞吐提升 `25.7%`。
- MLP FC1：`22.130 -> 15.430 ms/120`，降低 `30.3%`。
- MLP FC2：`23.622 -> 16.366 + 0.572 = 16.938 ms/120`，降低 `28.3%`。
- attention out：`11.667 -> 4.458 + 0.570 = 5.028 ms/120`，降低 `56.9%`。
- controller FC1/FC2：`7.591 -> 4.851 ms/80`，降低 `36.1%`。

双 stage 相对第一版单 stage 的完整 dispatch 从 `95.708` 降至 `94.057 ms`；主要收益在 FC1（约 `8.8%`），FC2 因 split-K 已有足够并行度，p2 只再改善约 `1.5%`。这也是保留 autotune 选择而不强制所有 shape 使用 ping-pong 的原因。

默认无 profile 连续生成 11 个 chunk，满 cache 最后 4 个 chunk 平均为 `94.68 ms/chunk`、`42.25 RGB FPS`，最后一个 chunk 为 `95.02 ms`、`42.10 RGB FPS`。相对 v3 的 `116.87 ms/chunk`、`34.23 RGB FPS`，端到端延迟降低 `19.0%`，吞吐提升 `23.4%`。
