# worldmodel.cu

这是 Waypoint 1.5 权重的 CUDA/Vulkan 运行时。这个 README 只讲怎么把项目跑起来。

## 1. 环境

必需：

- Git
- CMake 3.24+
- CUDA Toolkit
- Python 3
- Windows: Visual Studio 2022 Build Tools，并使用 x64 developer shell

可选：

- Vulkan SDK: 只在编译 Vulkan 后端时需要，`glslc` 要能在 `PATH` 里找到
- cuDNN 9: VAE decode 会更快；没有也可以编译和运行

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

## 4. 编译

### Windows CUDA

在 "x64 Native Tools Command Prompt for VS 2022" 里运行：

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

### 可选 cuDNN

cuDNN 不是必须的。装好后重新跑 CMake，项目会从 `CUDNN_ROOT`、`CUDNN_PATH` 或 Python 包里尝试自动发现：

```sh
python -m pip install nvidia-cudnn-cu12
```

### 可选 Vulkan

Windows：

```bat
cmake -S . -B build-win-vulkan -G "Visual Studio 17 2022" -A x64 ^
  -DWORLD_ENABLE_CUDA=OFF ^
  -DWORLD_ENABLE_VULKAN=ON

cmake --build build-win-vulkan --config Release --parallel
```

Linux：

```sh
cmake -S . -B build-vulkan -DCMAKE_BUILD_TYPE=Release \
  -DWORLD_ENABLE_CUDA=OFF \
  -DWORLD_ENABLE_VULKAN=ON

cmake --build build-vulkan -j
```

## 5. 运行

### 快速交互 CUDA

Windows：

```bat
.\build-win\Release\worldmodel_raylib.exe --model-dir .\Waypoint-1.5-1B --fast-realtime --mouse-scale 0.1
```

Linux：

```sh
./build/worldmodel_raylib --model-dir ./Waypoint-1.5-1B --fast-realtime --mouse-scale 0.1
```

控制：

- `WASD`: 移动
- `Space`: 跳
- `Shift`: 冲刺
- 鼠标: 转向
- 鼠标左右键: 按钮输入
- `Esc`: 退出

### 推荐：带初始图的交互运行

先装 Python 依赖：

```sh
python -m pip install torch diffusers pillow numpy safetensors
```

导出 seed latent：

```bat
python export_seed_latent.py --model-dir .\Waypoint-1.5-1B --out .\world_seed_latent.f32
```

Linux 用：

```sh
python export_seed_latent.py --model-dir ./Waypoint-1.5-1B --out ./world_seed_latent.f32
```

Windows 运行：

```bat
.\build-win\Release\worldmodel_raylib.exe ^
  --model-dir .\Waypoint-1.5-1B ^
  --seed-latent .\world_seed_latent.f32 ^
  --steps 4 ^
  --cache-window 8 ^
  --mouse-scale 0.1
```

Linux 运行：

```sh
./build/worldmodel_raylib \
  --model-dir ./Waypoint-1.5-1B \
  --seed-latent ./world_seed_latent.f32 \
  --steps 4 \
  --cache-window 8 \
  --mouse-scale 0.1
```

不传 `--seed-latent` 也能启动，但会从随机 latent 开始，更适合调试，不太适合作为可控世界 rollout。

### 无窗口 smoke test

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

### standalone CUDA 出图

Windows：

```bat
.\build-win\Release\worldmodel_cuda.exe --model-dir .\Waypoint-1.5-1B --steps 4 --frames 1 --out .\world_full.ppm
```

Linux：

```sh
./build/worldmodel_cuda --model-dir ./Waypoint-1.5-1B --steps 4 --frames 1 --out ./world_full.ppm
```

### Vulkan 360p smoke test

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

Windows 路径：

```bat
.\build-win-vulkan\Release\worldmodel_vulkan_probe.exe
```

## 常用参数

- `--fast-realtime`: 低延迟预设，会用较少 steps 和较小 KV cache
- `--steps 4 --cache-window 8`: 质量更好，但更慢
- `--cache-window 1`: 仅用于调试，通常没有可控历史
- `--mouse-scale 0.1`: 鼠标太灵敏时继续调低
- 找不到 raylib 时，先跑 `git submodule update --init --recursive`
- 运行时缺 CUDA DLL 时，从 CUDA 已加入 `PATH` 的 shell 启动
