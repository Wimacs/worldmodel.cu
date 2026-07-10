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
- resident runtime 当前仍使用 CUTLASS SIMT GEMM 做 `half x half -> float accumulator -> float output`，这是稳定默认路径。
- 本迭代新增了 `row_major_linear_fp16_tensorop` PyTorch extension probe，使用 CUTLASS Sm80 tests 覆盖过的 `128x128x32` threadblock，`64x64x32` warp，`16x8x16` instruction，4 stages。
- probe 已覆盖 resident 主要真实 shape：`(1,2048,8192)`、`(512,2048,2048)`、`(512,2048,4096)`、`(512,2048,8192)`、`(512,8192,2048)`。
- 本迭代还新增了 CMake/NVCC 独立探针 `worldmodel_cuda_gemm_probe`。默认 SIMT 路径通过，`--tensorop` 在最小 shape 上也会稳定复现 CUTLASS `ldsm RowMajor` unsupported path。
- 这说明问题不是 resident runtime 的权重数据或调用链独有问题，而是当前 CMake/NVCC 编译环境下这组 RowMajor x ColumnMajor tensor-op kernel 配置不可直接用。
- 因此这一版不启用 runtime tensor-op；下一版应该先对齐 PyTorch extension 与 CMake target 的 arch/编译宏/kernel config，或在 load 阶段预打包权重到 tensor-op friendly layout，再接默认路径。
- `WORLD_FP16_GEMM=0` 可以强制回退到 FP32 SIMT GEMM，用于定位问题。

独立探针命令：

```sh
./build/worldmodel_cuda_gemm_probe --small
./build/worldmodel_cuda_gemm_probe --small --tensorop
```

当前预期是第一条通过，第二条失败并打印 `ldsm RowMajor`。这条失败测试不是回归，而是为了固定下一版要解决的边界条件。

FP32 path：

- standalone 当前用 CUTLASS SIMT FP32 GEMM，作为不依赖 cuBLAS/cuDNN 的稳定 baseline。
- TF32 tensor-op 还没有接入 runtime；后续应先加一个独立对拍入口验证真实 shape，再决定是否接到 runtime。
- PyTorch extension 的 `patchify_cutlass` 也使用 CUTLASS SIMT FP32 GEMM，避免 TF32 误差影响严格对拍。

VAE：

- cuDNN 路径已经移除。
- 当前 VAE decode 只走内置 F32/NCHW direct conv。
- 后续性能优化应该用 CUTLASS conv/implicit-GEMM 或显式 im2row + CUTLASS GEMM，不再接 cuDNN。

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

内置 direct conv 正确但慢。下一步有两个可选路线：

- CUTLASS implicit-GEMM conv2d：适合 3x3/1x1、channel 比较规整的层。
- 显式 im2row + CUTLASS GEMM：实现简单，容易对拍和 profile，但会多写一个 im2row buffer。

建议顺序：

1. 先把 1x1 conv 改成 CUTLASS GEMM。
2. 再做 3x3 conv 的 im2row + CUTLASS GEMM baseline。
3. 如果 im2row 带宽太重，再换 CUTLASS implicit-GEMM conv。
4. 最后考虑 NHWC/FP16 全路径和 bias/ReLU fusion。

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
