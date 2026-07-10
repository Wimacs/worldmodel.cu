# worldmodel.cu

这是 Waypoint 1.5 的纯 C/CUDA/Vulkan 运行时实验仓库。目标是把 PyTorch 侧实时 WorldDiT 管线逐步迁到可独立运行的 CUDA/Vulkan 后端，并持续用 PyTorch reference 对拍。

当前 CUDA 路径不链接 cuBLAS/cuDNN，只链接 CUDA runtime；GEMM/linear 走 CUTLASS。CUTLASS kernel 现状、layout 取舍、Triton 优化思路映射和后续 autotune 计划见 [`docs/cuda_cutlass_kernels.md`](docs/cuda_cutlass_kernels.md)。

## 1. 环境

必需：

- Git
- CMake 3.24+
- CUDA Toolkit
- Python 3
- Windows: Visual Studio 2022 Build Tools，并使用 x64 developer shell

可选：

- Vulkan SDK: 只在编译 Vulkan 后端时需要，`glslc` 要能在 `PATH` 里找到
- CUTLASS checkout: 可通过 `WORLD_CUTLASS_DIR` 或 `CUTLASS_DIR` 指定；否则 CMake 会找 `3rd/cutlass`、常见 sibling checkout，最后用 FetchContent 拉取

## 2. Clone

```sh
git clone --recurse-submodules https://github.com/Wimacs/worldmodel.cu.git
cd worldmodel.cu
```

如果已经 clone 过但没拉子模块：

```sh
git submodule update --init --recursive
```

交互窗口依赖 `3rd/raylib`，所以这个目录必须存在。

## 3. 下载权重

安装 Hugging Face CLI：

```sh
python -m pip install -U huggingface_hub
```

下载主模型：

```sh
hf download Overworld/Waypoint-1.5-1B --local-dir Waypoint-1.5-1B
```

如果要跑 360p Vulkan 路径，再下载：

```sh
hf download Overworld/Waypoint-1.5-1B-360P --local-dir Waypoint-1.5-1B-360P
```

权重目录大致应该长这样：

```text
worldmodel.cu/
  Waypoint-1.5-1B/
    config.yaml
    model.safetensors
    transformer/diffusion_pytorch_model.safetensors
    vae/diffusion_pytorch_model.safetensors
  Waypoint-1.5-1B-360P/
    config.yaml
    model.safetensors
```

注意：当前 CUDA standalone F32 smoke path 使用 `Waypoint-1.5-1B` 的 F32 transformer 权重；360p checkpoint 是 BF16，主要用于 Vulkan 360p 路径。

## 4. 编译

### Windows CUDA

在 "x64 Native Tools Command Prompt for VS 2022" 里运行。CUDA-only 构建不需要 Vulkan SDK：

```bat
cmake -S . -B build-win -G "Visual Studio 17 2022" -A x64 ^
  -DWORLD_ENABLE_CUDA=ON ^
  -DWORLD_ENABLE_VULKAN=OFF ^
  -DCMAKE_CUDA_ARCHITECTURES=89

cmake --build build-win --config Release --parallel
```

`CMAKE_CUDA_ARCHITECTURES` 按显卡改。RTX 30 系列常用 `86`，RTX 40 系列常用 `89`。不传这个参数时，项目默认是 `89`。

生成文件：

```text
build-win/Release/worldmodel_raylib.exe
build-win/Release/worldmodel_cuda.exe
```

### Windows Vulkan

Vulkan 构建需要安装 Vulkan SDK，并保证 `glslc.exe` 在 `PATH` 中，或 `VULKAN_SDK` 指向 SDK 根目录：

```bat
cmake -S . -B build-win-vulkan -G "Visual Studio 17 2022" -A x64 ^
  -DWORLD_ENABLE_CUDA=OFF ^
  -DWORLD_ENABLE_VULKAN=ON

cmake --build build-win-vulkan --config Release --parallel
```

生成文件：

```text
build-win-vulkan/Release/worldmodel_raylib_vulkan.exe
build-win-vulkan/Release/worldmodel_vulkan_probe.exe
build-win-vulkan/Release/worldmodel_vulkan_gemm_autotune.exe
```

### Windows CUDA + Vulkan

如果同一台机器上 CUDA Toolkit 和 Vulkan SDK 都已经配置好，可以一次构建两个后端：

```bat
cmake -S . -B build-win-all -G "Visual Studio 17 2022" -A x64 ^
  -DWORLD_ENABLE_CUDA=ON ^
  -DWORLD_ENABLE_VULKAN=ON ^
  -DCMAKE_CUDA_ARCHITECTURES=89

cmake --build build-win-all --config Release --parallel
```

### Linux CUDA

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DWORLD_ENABLE_CUDA=ON \
  -DWORLD_ENABLE_VULKAN=OFF

cmake --build build -j
```

生成文件：

```text
build/worldmodel_raylib
build/worldmodel_cuda
```

### Linux Vulkan

```sh
cmake -S . -B build-vulkan -DCMAKE_BUILD_TYPE=Release \
  -DWORLD_ENABLE_CUDA=OFF \
  -DWORLD_ENABLE_VULKAN=ON

cmake --build build-vulkan -j
```

## 5. 运行

### CUDA 交互窗口

Windows：

```bat
.\build-win\Release\worldmodel_raylib.exe --model-dir .\Waypoint-1.5-1B --fast-realtime --mouse-scale 1.0
```

Linux：

```sh
./build/worldmodel_raylib --model-dir ./Waypoint-1.5-1B --fast-realtime --mouse-scale 1.0
```

控制：

- `W` / `Space`: 前进，送入模型的 PyTorch button id 是 `32`
- `A` / `D` / `S`: 左 / 右 / 后退，送入模型的 PyTorch button id 分别是 `65` / `68` / `83`
- `Shift`: 冲刺
- 鼠标: 转向
- 鼠标左右键: 按钮输入
- `Esc`: 退出

推荐先用 PyTorch 导出一个 seed latent，减少随机初始画面：

```sh
python export_seed_latent.py --model-dir ./Waypoint-1.5-1B --out ./world_seed_latent.f32
./build/worldmodel_raylib --model-dir ./Waypoint-1.5-1B --seed-latent ./world_seed_latent.f32 --steps 4 --cache-window 8 --mouse-scale 1.0
```

### Headless Smoke

Windows：

```bat
.\build-win\Release\worldmodel_raylib.exe --model-dir .\Waypoint-1.5-1B --fast-realtime --headless-smoke
```

Linux：

```sh
./build/worldmodel_raylib --model-dir ./Waypoint-1.5-1B --fast-realtime --headless-smoke
```

需要写出调试图时加：

```sh
--headless-out world_runtime.ppm
```

### Standalone CUDA 出图/生成

Windows：

```bat
.\build-win\Release\worldmodel_cuda.exe --model-dir .\Waypoint-1.5-1B --steps 4 --frames 1 --out .\world_full.ppm
```

Linux：

```sh
./build/worldmodel_cuda --model-dir ./Waypoint-1.5-1B --steps 4 --frames 1 --out ./world_full.ppm
```

`--out` 会加载 VAE 权重并写 PPM；不传 `--out` 时只跑 latent/transformer path。

### Vulkan 360p Smoke

Vulkan 360p 路径使用 `Waypoint-1.5-1B-360P` 的 transformer 权重，同时复用 `Waypoint-1.5-1B` 的 VAE 权重。

Windows：

```bat
.\build-win-vulkan\Release\worldmodel_raylib_vulkan.exe ^
  --model-dir .\Waypoint-1.5-1B-360P ^
  --vae-weights .\Waypoint-1.5-1B\vae\diffusion_pytorch_model.safetensors ^
  --layers 4 ^
  --steps 4 ^
  --cache-window 8 ^
  --headless-smoke
```

Linux：

```sh
./build-vulkan/worldmodel_raylib_vulkan \
  --model-dir ./Waypoint-1.5-1B-360P \
  --vae-weights ./Waypoint-1.5-1B/vae/diffusion_pytorch_model.safetensors \
  --layers 4 \
  --steps 4 \
  --cache-window 8 \
  --headless-smoke
```

Vulkan shader probe：

```sh
./build-vulkan/worldmodel_vulkan_probe
```

## 6. 常用参数

- `--fast-realtime`: 低延迟预设，会用较少 steps 和较小 KV cache
- `--steps 4 --cache-window 8`: 质量更好，但更慢
- `--cache-window 1`: 仅用于调试，通常没有可控历史
- `--warmup N`: 用临时 runtime 预热 N 个 chunk，随后丢弃临时 KV/latent 状态；正式窗口从新的 resident runtime 开始
- `--mouse-scale X`: 鼠标 delta 缩放，默认 `1.0`；最终送入模型的 `CtrlInput.mouse` 会 clamp 到 PyTorch 侧使用的 `[-1, 1]` 范围
- `--frame-idx N`: 设置第一个 latent frame 的 temporal RoPE/cache 位置
- `--frames N`: 同一进程生成 N 个 latent frame，并保留 KV cache 历史
- `--cache-pass`: 每个 frame 生成后追加 sigma=0 的 cache 写入 pass
- `--control FILE` / `--control-seq FILE`: 输入控制向量或控制序列
- `--latent FILE`: 载入外部 little-endian float32 latent
- `--vae-only --latent FILE --out FILE`: 只调试 TAEHV decode

raylib 前端会采样 WASD、Space、Shift、鼠标左右键、鼠标 delta 和滚轮，映射到 PyTorch `examples/gen_sample.py` 里使用的 controller layout。CUDA generation 在 worker thread 里跑，主线程继续轮询输入；收到新 decoded chunk 后按顺序播放一次，然后停在最后一帧等待下一个 chunk。`WORLD_CONTROL_DEBUG=1` 会打印每个 chunk 实际送入模型的 mouse/button id/wheel；按下物理 `W` 时应看到 `buttons={32}`，不是键盘扫描码 `87`。

## 7. CUDA/CUTLASS 状态

- CUDA 目标只链接 `CUDA::cudart`，不链接 cuBLAS/cuDNN。
- FP32 fallback 仍走 CUTLASS SIMT baseline，保证真实权重可 load 和完整 4-step 可运行。
- resident FP16-weight path 默认走 CUTLASS tensor-op half x half -> float accumulator；`WORLD_FP16_GEMM=simt` 可强制回退旧 FP16 SIMT，`WORLD_FP16_GEMM=0` 可强制回退 FP32 SIMT。
- PyTorch extension tensor-op probe 和 CMake/NVCC 独立探针 `worldmodel_cuda_gemm_probe --tensorop` 都覆盖 resident 主要 GEMM shape。
- CMake configure 会打印 `CUDA architectures`；tensor-op 需要 `80+`，项目默认是 `89`。
- 后续高性能路线按版本推进：先 profile 确认最大瓶颈，再做少量真实 shape CUTLASS probe，最后才接入 runtime；不在主线里无休止搜索 tile。
- `WORLD_TRANSFORMER_PROFILE=1` 会打印 transformer 分段 CUDA event timing。360p 24 layer/4 step 当前最大块是 MLP fc2；小 M、大 K 形状默认启用 `WORLD_MLP_FC2_SPLITK=4` 的 CUTLASS serial split-K，`WORLD_MLP_FC2_SPLITK=1` 可关闭，`2/8/...` 可手动覆盖。
- 360p token GEMM 默认对小 M 形状启用 CUTLASS `64x64x32` tensor-op tile；`WORLD_FP16_GEMM_TILE=base` 可回退旧 `128x128x32` tile，`./build/worldmodel_cuda_gemm_probe --bench` 可复测候选 tile。
- MLP `fc1 -> SiLU -> fc2` 默认使用 CUTLASS fc1 SiLU-to-half epilogue，fc2 split-K 直接消费 half activation；`WORLD_MLP_FC1_SILU_EPILOGUE=0` 可回退到独立 `SiLU + cast` kernel。
- D=64 cache attention 默认走项目内 indexed warp fallback；`WORLD_FLASH_ATTN=1` 可启用 online-softmax tiled prototype，`WORLD_ATTN_D64_Q4_SHARED=1` 可启用 4-row shared-KV probe。两者当前真实 profile 都慢于默认 fallback，保留为诊断开关。
- `WORLD_ATTN_D64_HALF_CACHE=1` 会把 K/V cache 存为 FP16，并走 half2 warp attention；`WORLD_ATTN_D64_HALF_FLASH=1` 可进一步启用 half-cache group-flash probe。group-flash 已有 PyTorch 对拍，但当前 360p 满 cache profile 慢于 half2 warp path，因此只作为下一轮 attention 设计的对照，不默认启用。
- `WORLD_ATTN_D64_CUTLASS=1` 会启用 opt-in CUTLASS materialized QK/AV attention probe，并强制 FP16 KV cache；它显著加速 360p 满 cache attention，但会额外 materialize scores/probs，scratch 默认限制为 2048 MiB，可用 `WORLD_ATTN_D64_CUTLASS_MAX_SCRATCH_MIB` 覆盖。
- `WORLD_ATTN_D64_CUTLASS_GROUPED=1` 会启用 grouped-M GQA materialized CUTLASS 变体，减少 K/V compact 和 GQA B 矩阵重复读取；当前端到端收益很小，作为 opt-in 对照保留。
- VAE decode 默认仍是 F32/NCHW 主路径；1x1 conv 默认走 per-frame CUTLASS GEMM，3x3 conv 默认走 tiled im2col + CUTLASS GEMM，默认 `WORLD_VAE_3X3_TILE_COLS=16384`。`WORLD_VAE_1X1_GEMM=0` / `WORLD_VAE_3X3_GEMM=0` 可分别回退旧 direct conv。
- `WORLD_VAE_FP16_NHWC=1` 可启用实验性的 VAE FP16/NHWC 全路径：VAE 权重在 load 阶段额外预打包成 KRSC half，runtime 用 CUTLASS tensor-op implicit conv，bias 仍是单独 half kernel。当前保持默认关闭，方便和 F32/NCHW 对照及定位画面问题。
- `WORLD_VAE_3X3_BATCH_COLS=1` 可试验 frame-batched 3x3 tiles；当前 profile 显示它慢于默认 per-frame path，所以默认关闭。
- PyTorch extension 里已有 CUTLASS implicit-GEMM 3x3 NHWC/KRSC 单层、half 单层和 pair probe；half 单层测试按 runtime 的 half-output-then-bias 舍入路径对齐。
- `WORLD_VAE_PROFILE=1` 会同步 CUDA event 并打印 VAE conv 分段时间，只用于 profile，不用于正常 FPS。

## 8. 测试

CUDA/Vulkan build：

```sh
cmake -S . -B build -DWORLD_ENABLE_CUDA=ON -DWORLD_ENABLE_VULKAN=ON
cmake --build build -j
```

CUDA extension 对拍：

```sh
python test_worldmodel_kernels.py
```

CMake/NVCC CUTLASS GEMM 探针：

```sh
./build/worldmodel_cuda_gemm_probe --small
./build/worldmodel_cuda_gemm_probe --small --tensorop
./build/worldmodel_cuda_gemm_probe --tensorop
```

如果这里出现 `ldsm RowMajor`，先检查 CMake configure 输出里的 `CUDA architectures` 是否为 `80+`。

真实 F32 权重 4-step smoke：

```sh
./build/worldmodel_cuda --model-dir ./Waypoint-1.5-1B --seed 1 --noise uniform --sigma 1.0 --steps 4 --frames 1
```

更完整的 standalone/PyTorch parity：

```sh
python test_standalone_probe.py
python generate_smoke.py --model-dir ./Waypoint-1.5-1B --steps 4 --output smoke_latent.pt
```
