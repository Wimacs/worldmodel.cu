# CUDA CUTLASS Kernel Notes

这份文档记录 CUDA 路径改成 CUTLASS 后的 kernel 设计和后续优化方法。目标很明确：

- 不链接 cuBLAS/cuDNN 或其它 CUDA 数学库，只链接 CUDA runtime。
- GEMM/linear 优先用 CUTLASS device kernel。
- 每个新 kernel 都要有 PyTorch 对拍；先保数值，再做 profile 和 autotune。
- 参考 Triton 优化思路，但实现落在 CUTLASS 模板、tile、epilogue 和调度上。

## 当前落地状态

CUDA standalone 的 `row_major_linear()` 已经改为 CUTLASS GEMM：

- 输入 `x[M,K]` 是 row-major。
- 权重 `w[N,K]` 仍保持 checkpoint/export 的 row-major 布局。
- GEMM 逻辑是 `y[M,N] = x[M,K] * w[N,K]^T`。
- CUTLASS 把 `w[N,K]` 解释为 `B[K,N]` column-major，因此不用预转置，也不用多拷贝一份权重。

FP16-weight fast path：

- runtime 仍先把 activation 从 FP32 cast 到 FP16 scratch。
- weight 预先保存为 FP16。
- resident runtime 默认使用 CUTLASS tensor-op GEMM 做 `half x half -> float accumulator -> float output`。
- tensor-op kernel 使用 CUTLASS Sm80 tests 覆盖过的 `128x128x32` threadblock，`64x64x32` warp，`16x8x16` instruction，4 stages。
- probe 已覆盖 resident 主要真实 shape：`(1,2048,8192)`、`(512,2048,2048)`、`(512,2048,4096)`、`(512,2048,8192)`、`(512,8192,2048)`。
- MLP fc2 的真实 360p 形状是 `M=128,K=8192,N=2048`，默认 tensor-op GEMM 只有 `1x16` 个 CTA，profile 显示它是 transformer 最大单块。新增 CUTLASS serial split-K 专用路径，默认只在小 M、大 K 形状自动用 `split_k_slices=4`；`WORLD_MLP_FC2_SPLITK=1` 可关闭，显式设置 `2/4/8/...` 可覆盖。
- 360p、24 layer、4 step、`WORLD_VAE_FP16_NHWC=1`、`cache-window=8` 的 profile 对照：默认无 split-K 时 `mlp_fc2=26.413ms/120`、`total=80.033ms`；`splitK=2` 为 `14.241ms/120`、`66.900ms`；`splitK=4` 为 `8.637ms/120`、`62.249ms`；`splitK=8` 为 `9.976ms/120`、`63.275ms`。正常非 profile smoke 默认启发式为 `total=53.422ms`、`chunk_fps=18.719`、`rgb_fps=74.875`。所以这一版选择 4，不继续做无边界 tile 搜索。
- 下一版 profile 后，除 MLP fc2 外的新大头变成 qkv/out/ctrl/fc1 GEMM。`worldmodel_cuda_gemm_probe --bench` 增加了少量真实 shape tile probe，仅比较 `128x128x32`、`64x128x32`、`128x64x32`、`64x64x32`。复测结果：`M=128,K=2048,N=2048` 从 base `0.0560ms` 到 `64x64=0.0168ms`；`N=4096` 从 `0.0561ms` 到 `0.0299ms`；`N=8192` 从 `0.0576ms` 到 `0.0472ms`；`K=8192,N=2048` 从 `0.2141ms` 到 `0.0601ms`。runtime 因此默认对小 M token GEMM 启用 `64x64x32` tile，但 MLP fc2 仍优先用 split-K，因为真实 profile 中 split-K 略快。
- 360p、24 layer、4 step、`WORLD_VAE_FP16_NHWC=1`、`cache-window=8`，默认启用 `64x64x32` 小 M tile + MLP fc2 split-K 后，profile 为 `qkv_gemm=4.131ms/120`、`attn_out_gemm=2.756ms/120`、`ctrl=2.005ms/40`、`mlp_fc1=6.807ms/120`、`mlp_fc2=8.686ms/120`、`total=49.880ms`。正常非 profile smoke 为 `total=41.994ms`、`chunk_fps=23.813`、`rgb_fps=95.251`。`WORLD_FP16_GEMM_TILE=base` 可回退旧 tile。
- 上一版沿 MLP 链路做了一次小的 fusion：`fc1` 仍用 CUTLASS 输出 float，随后用 `silu_f32_to_f16_kernel` 一次完成 SiLU 和 half cast，`fc2` 的 CUTLASS serial split-K 直接读取 half activation。这样省掉旧路径中 `SiLU(float)->float hidden` 后又在 fc2 内部 `float hidden->half scratch` 的第二次全量读写。新增 PyTorch 对拍覆盖 `silu_to_half` 和 half-input split-K。360p profile 变为 `mlp_silu=0.629ms/120`、`mlp_fc2=8.159ms/120`、`total=48.872ms`；正常非 profile smoke 为 `total=41.613ms`、`chunk_fps=24.031`、`rgb_fps=96.123`。
- 当前版把这个思路推进到 CUTLASS epilogue：新增 `LinearCombinationSilu` output op，让 fc1 GEMM 直接输出 `half(SiLU(acc))` 到 fc2 的 half activation buffer。runtime 默认启用，`WORLD_MLP_FC1_SILU_EPILOGUE=0` 可回退上一版的独立 kernel。PyTorch 对拍新增 `row_major_linear_fp16_tensorop_m64n64_silu_half`，覆盖真实 fc1 shape `M=128,K=2048,N=8192`。360p profile 为 `mlp_fc1=6.824ms/120`、`mlp_silu=0.000ms/0`、`mlp_fc2=8.200ms/120`、`total=48.185ms`；正常非 profile smoke 为 `total=40.793ms`、`chunk_fps=24.514`、`rgb_fps=98.056`。回退 `WORLD_MLP_FC1_SILU_EPILOGUE=0` 的非 profile smoke 为 `total=41.303ms`、`rgb_fps=96.845`。这版到此停止，下一步应该转向 attention，而不是继续压 MLP 这几个小数点。
- CMake/NVCC 独立探针 `worldmodel_cuda_gemm_probe --tensorop` 现在通过；此前 `ldsm RowMajor` 失败的原因是 CMake 实际编译成了 `sm_52`，导致 CUTLASS SM80 `ldsm` 实现没有启用。
- CMake 默认 `CMAKE_CUDA_ARCHITECTURES=89` 已移到 `enable_language(CUDA)` 之前设置，并在 configure 时打印实际架构。
- `WORLD_FP16_GEMM=simt` 可以强制回退旧的 FP16 SIMT GEMM；`WORLD_FP16_GEMM=0` 可以强制回退到 FP32 SIMT GEMM，用于定位问题。

独立探针命令：

```sh
./build/worldmodel_cuda_gemm_probe --small
./build/worldmodel_cuda_gemm_probe --small --tensorop
./build/worldmodel_cuda_gemm_probe --tensorop
```

当前预期是三条都通过。如果这里重新出现 `ldsm RowMajor`，优先检查 configure 输出里的 `CUDA architectures` 是否为 `80+`。

FP32 path：

- standalone 当前用 CUTLASS SIMT FP32 GEMM，作为不依赖 cuBLAS/cuDNN 的稳定 baseline。
- TF32 tensor-op 还没有接入 runtime；后续应先加一个独立对拍入口验证真实 shape，再决定是否接到 runtime。
- PyTorch extension 的 `patchify_cutlass` 也使用 CUTLASS SIMT FP32 GEMM，避免 TF32 误差影响严格对拍。

VAE：

- cuDNN 路径已经移除。
- 默认 VAE decode 仍是 F32/NCHW 主路径。
- 1x1 conv 已接入 CUTLASS GEMM baseline：每个 frame 把 NCHW 平面视作 `B = [Cin, H*W]` row-major，权重视作 `A = [Cout, Cin]` row-major，输出直接落到该 frame 的 `[Cout, H*W]` NCHW 平面。
- 3x3 conv 已接入 tiled im2col + CUTLASS GEMM baseline：每次 materialize `K = Cin*3*3` 和最多 16384 个 spatial column，做 `weight[Cout,K] x cols[K,tile] -> out[Cout,tile]`。这样避免完整 im2col 在后段高分辨率层爆显存，同时把 tile/GEMM launch 数降下来。
- `WORLD_VAE_1X1_GEMM=0` 可以回退旧 direct 1x1 conv，用于性能和数值定位。
- `WORLD_VAE_3X3_GEMM=0` 可以回退旧 direct 3x3 conv，用于性能和数值定位。
- `WORLD_VAE_3X3_TILE_COLS=N` 可以覆盖 3x3 tiled GEMM 的 spatial tile，默认 `16384`；workspace 分配失败时会自动对半降低，最低到 `1024`。
- `WORLD_VAE_3X3_BATCH_COLS=1` 可以试验把 frame 维并入 3x3 tile columns。这个路径有 PyTorch 对拍，但默认关闭，因为当前 profile 中它只把 tile 数从 `496` 降到 `446`，却增加了 NCHW scatter 和更慢的 im2col，VAE 从 `72.391ms` 变成 `77.269ms`。
- `WORLD_VAE_PROFILE=1` 会在 VAE conv 内部打开 CUDA event 分段计时，输出 direct、1x1 GEMM、3x3 im2col/GEMM/bias 时间。这个模式会同步每段 kernel，只用于诊断，不用于正常 FPS。
- 当前 1-layer/4-step headless smoke 对照：默认 tile 16384 `vae=69.233ms`；旧 tile 4096 profile-mode `vae=116.244ms`；全 direct conv `vae=294.243ms`。
- 当前 profiling 结果：默认 tile 16384 per-frame path 是 `1x1_gemm=1.314ms/16 launches`，`3x3_im2col=24.346ms/496 tiles`，`3x3_gemm=34.252ms/496 tiles`，`3x3_bias=1.486ms`；旧 tile 4096 是 `1784 tiles`、`3x3_gemm=65.235ms`。下一步大头仍是 3x3 GEMM，但主要 launch 数已经降了一轮。
- CUTLASS implicit-GEMM 3x3 NHWC/KRSC probe 已加入 PyTorch extension：`taehv_conv3x3_cutlass_implicit_nhwc`。它使用 F32 SIMT、same padding、alignment=1，先保证任意小 shape 能严格对拍。
- 小型 CUDA event probe 显示 implicit NHWC core 确实能绕开 im2col 物化：`N=4,Cin=64,Cout=64,H=64,W=128` 时，现有 batched im2col path `0.220ms`，implicit NHWC core `0.115ms`，临时 `NCHW->NHWC->NCHW` 包裹后 `0.157ms`；`Cout=128` 时分别是 `0.282ms`、`0.197ms`、`0.269ms`。
- 单层 probe 结论：不把“每层临时转 NHWC 再转回 NCHW”的实现接入默认 runtime；需要用连续 NHWC island 或完整 NHWC/FP16 路径，让 layout 转换只发生在边界。
- 连续 NHWC island probe 已加入 PyTorch extension：`taehv_conv3x3_cutlass_implicit_nhwc_pair`，执行 `conv3x3 -> ReLU -> conv3x3`，中间不离开 NHWC。小型 probe 显示 `N=4,Cin=64,Cmid=64,H=64,W=128` 时，两层 im2col path 是 `0.456ms`，NHWC pair core 是 `0.246ms`，边界转换一次后是 `0.279ms`；`Cin=128` 时分别是 `0.674ms`、`0.341ms`、`0.421ms`；`H=128,W=256` 时分别是 `1.896ms`、`0.800ms`、`1.016ms`。
- runtime FP16/NHWC VAE conv 已接入实验开关 `WORLD_VAE_FP16_NHWC=1`。加载 VAE 时会保留旧 OIHW F32 权重，同时额外预打包 KRSC half 权重；`taehv_run_conv_h_nhwc()` 使用 CUTLASS tensor-op implicit conv，accumulator 为 float，输出 half，bias 暂时单独 half NHWC kernel。
- half implicit conv 使用 A/B alignment=8。原因是 CUTLASS SM80 tensor-op implicit conv 使用 `cp.async`，alignment=1 会退成 2-byte 搬运并触发 `Size is not supported` 编译错误；当前 VAE activation/filter 的 `Cin` 都是 8 的倍数，A/B alignment=8 能覆盖真实 shape。output epilogue 仍用 element-per-access=1，以兼容最后 `Cout=12`。
- 实测 headless smoke：1024x512、1 layer、1 step，旧 F32/NCHW VAE `68.705ms`，`WORLD_VAE_FP16_NHWC=1` 后 VAE `12.613ms`；1024x512、1 layer、4 step 为 `12.854ms`；360p、24 layer、4 step 完整闭环为 transformer `66.851ms`、VAE `3.716ms`、total `71.825ms`。
- 这版刻意不做 bias/ReLU fusion 和 tile 搜索。下一版应先判断视觉/数值是否接受 half VAE，再考虑 fused bias/ReLU 或针对真实 VAE shape 增加少量专用 CUTLASS 变体。

Attention：

- cuBLAS attention prototype 已经移除。
- 当前默认是项目内置 indexed warp fallback。
- `WORLD_FLASH_ATTN=1` 可以打开已有 online-softmax tiled prototype。
- 后续要么继续优化自写 fused attention，要么用 CUTLASS/CuTe 写专门的 QK/softmax/AV fused kernel；不要回退到 cuBLAS。

## 从 Triton 优化方案学什么

Triton kernel 的常见优化点不是语言本身，而是以下几件事：

1. 问题 shape specialization：为真实 shape 编译少量专门 kernel，而不是一个通吃 kernel。
2. tile search：枚举 `BLOCK_M/BLOCK_N/BLOCK_K/num_warps/num_stages`。
3. 数据复用：A/B tile 进 shared memory 或寄存器，尽量让 K 维加载复用。
4. epilogue fusion：GEMM 后的 bias、SiLU、gate、residual 尽量在输出阶段完成。
5. Split-K/stream-K：当 M/N 太小导致并行度不足时，把 K 拆开提高 occupancy。
6. layout 先行：避免运行时 transpose；用 RowMajor/ColumnMajor 解释已有内存。
7. profile-driven：只相信 Nsight Compute / CUDA event / PyTorch 对拍后的真实数据。

在 CUTLASS 里对应到：

- `ThreadblockShape` 对应 Triton 的 `BLOCK_M/BLOCK_N/BLOCK_K`。
- `WarpShape` 和 `InstructionShape` 控制 tensor core 使用方式。
- `Stages` 对应 pipeline depth。
- `AlignmentA/B` 对应 vectorized load 粒度。
- `EpilogueOutputOp` 或 visitor epilogue 对应 Triton 的 fused epilogue。
- `GemmSplitKParallel`/stream-K kernel 对应 Triton Split-K 思路。

## GEMM 优化计划

迭代原则：

- 每一版只推进一个明确目标，例如“FP16 resident GEMM tensor-op 化”或“MLP fc1+SiLU epilogue fusion”。
- 不在主线里无休止枚举 tile；先用 CUTLASS 已验证的形状或少量真实 shape probe 建立正确性。
- 只有端到端 profile 指向某个 kernel 仍是最大瓶颈时，才进入下一轮 tile/autotune。

第一阶段：shape inventory

- 打印 runtime 中每类 GEMM 的 `(M,K,N,dtype,epilogue)`。
- 重点 shape 通常是：
  - token linear: `M=T`, `K=D`, `N=D / qkv / mlp_hidden`
  - MLP fc1/fc2: `T x D -> mlp_hidden`, `T x mlp_hidden -> D`
  - scheduler/control: `M=1`, 小 batch GEMV-like
  - patchify/unpatchify: 小 K 或小 N 的特殊 GEMM

第二阶段：候选 CUTLASS kernel

- token GEMM 优先候选：
  - `128x128x64`, stages 3/4
  - `128x64x64`, stages 3/4
  - `64x128x64`, stages 3/4
  - `64x64x64`, stages 4, 用于较小 N
- FP16 tensor core 需要单独处理 layout：当前 checkpoint 权重是 `w[N,K]` row-major，虽然可以被当成 `B[K,N]` column-major 解释，但 CUTLASS half tensor-op 对这种组合并不总是可用。真正高性能版本应在 load 阶段预打包权重到 tensor-op friendly layout，并让 activation scratch 使用匹配 layout。
- 小 `M=1` 的 conditioning/control GEMM 不一定适合大 tile；需要单独 GEMV-like kernel 或 CUTLASS Gemv。
- `K` 很大且 `M/N` 小时再测 Split-K；否则 partial buffer + reduce 会拖慢。

第三阶段：fusion

- QKV projection: 已经把 Q/K/V weight 拼成一个 `N = D + 2*kv_dim` 的 GEMM，这是正确方向。
- MLP fc1 + SiLU: 下一步用 CUTLASS epilogue 输出 FP16 或直接 SiLU，减少一个 global read/write。
- gated residual: 对 attn out_proj 和 mlp fc2 可以把 gate/residual 放进 epilogue，目标是少一个 kernel launch 和一次读写。
- control branch: `fc1_x + fc1_c + SiLU` 可融合到 epilogue 或紧随 GEMM 的 lightweight kernel。

第四阶段：autotune

- 写 `world_cuda_gemm_autotune`，对真实 shape 枚举 CUTLASS typedef。
- 每个候选先跑 PyTorch/CPU correctness probe，再用 CUDA event 计时。
- 输出 cache，例如：
  - `M K N dtype epilogue -> kernel_id tile stages alignment split_k`
- runtime 按 shape 查表；没有命中就走保守默认。

## Attention 优化计划

当前 indexed attention 的问题是 GQA + sparse cache indices + online softmax，不是一个普通 batched GEMM。优化方向：

- 保留 fused online softmax，避免 materialize `scores[H,T,Nkv]`。
- 对 `D=64` 专门化，一个 warp 处理一个 query row，或一个 block 处理多个 query row。
- K/V tile 放 shared memory，按 `Nkv` 分块；保持 max/sum online update。
- GQA 不要复制 K/V；用 head group 映射直接读对应 KV head。
- 对小 cache window，warp kernel 可能比大 block flash kernel 更快；要按 `Nkv` 分桶选择。
- 对大 cache window，再考虑 split-K style partial softmax：每个 K block 输出 `(m, l, acc)`，reduce kernel 合并。

## VAE conv 优化计划

内置 direct conv 正确但慢。当前已完成 1x1 per-frame GEMM、3x3 tiled im2col GEMM baseline，以及实验性的 runtime FP16/NHWC implicit conv 全路径。后续还有两个可选路线：

- CUTLASS implicit-GEMM conv2d：适合 3x3/1x1、channel 比较规整的层。
- 显式 im2row + CUTLASS GEMM：实现简单，容易对拍和 profile，但会多写一个 im2row buffer。

建议顺序：

1. 1x1 conv 已改成 per-frame CUTLASS GEMM，不需要 im2row buffer。
2. 3x3 conv 已改成 tiled im2col + CUTLASS GEMM baseline。
3. frame-batched tile 实验证明“简单合并 frame 维”不值得进默认。
4. CUTLASS implicit-GEMM NHWC 单层和 pair probe 已证明 core 有收益，且连续 NHWC island 能 amortize 边界转换。
5. runtime FP16/NHWC conv 已接入实验开关：补 KRSC half 权重预打包，实现 `taehv_run_conv_h_nhwc()`，并补 half implicit conv PyTorch 对拍。
6. 再考虑 bias/ReLU fusion，以及真实 VAE shape 的少量专用 tensor-op/TF32 变体。

## 对拍规则

每加一个 CUTLASS kernel，都必须补测试：

- PyTorch extension 暴露一个最小入口。
- `test_worldmodel_kernels.py` 用固定 seed 构造输入。
- reference 尽量使用 PyTorch 原生算子。
- FP32 SIMT 对拍用严格阈值。
- TF32/FP16 tensor-op 对拍要和实际量化路径一致，例如 `x.half().float() @ w.half().float().T`。
- 性能优化前先跑 correctness；性能结果不能替代 correctness。

## Profiling checklist

每次调 tile 前先看：

- kernel 时间占比：`nsys profile` 找最耗时 kernel。
- tensor core 利用率：Nsight Compute 看 `sm__pipe_tensor`。
- memory throughput：确认是不是 bandwidth-bound。
- occupancy/register：大 tile 是否寄存器溢出。
- shared memory bank conflict。
- launch count：可融合 epilogue 时不要只调 tile。
- end-to-end FPS：单 kernel 变快但 chunk 变慢时，以 end-to-end 为准。
