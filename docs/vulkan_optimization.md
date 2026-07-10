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
