# worldmodel.cu

An experimental standalone C/CUDA/Vulkan runtime for Waypoint 1.5.

The CUDA backend uses CUTLASS for GEMM, attention, and convolution kernels. It links only the CUDA runtime, without cuBLAS or cuDNN. CUDA kernels are tested against PyTorch references. The Vulkan backend is experimental.

## Requirements

- CMake 3.24+
- CUDA Toolkit for the CUDA backend
- Vulkan SDK and `glslc` for the Vulkan backend
- Python 3 for weight downloads and parity tests
- Visual Studio 2022 Build Tools on Windows

## Setup

```sh
git clone --recurse-submodules https://github.com/Wimacs/worldmodel.cu.git
cd worldmodel.cu
python -m pip install -U huggingface_hub
hf download Overworld/Waypoint-1.5-1B --local-dir Waypoint-1.5-1B
```

## Build

Linux CUDA:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DWORLD_ENABLE_CUDA=ON
cmake --build build -j
```

Linux Vulkan:

```sh
cmake -S . -B build-vulkan -DCMAKE_BUILD_TYPE=Release \
  -DWORLD_ENABLE_CUDA=OFF -DWORLD_ENABLE_VULKAN=ON
cmake --build build-vulkan -j
```

Windows CUDA, from an x64 Visual Studio developer shell:

```bat
cmake -S . -B build-win -G "Visual Studio 17 2022" -A x64 ^
  -DWORLD_ENABLE_CUDA=ON -DWORLD_ENABLE_VULKAN=OFF ^
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-win --config Release --parallel
```

## Run

Interactive CUDA runtime:

```sh
./build/worldmodel_raylib ./Waypoint-1.5-1B/model.safetensors ./image.png
```

Drop another image onto the window to reset the world from that frame.

Headless smoke test:

```sh
./build/worldmodel_raylib --model-dir ./Waypoint-1.5-1B \
  --steps 4 --cache-window 8 --headless-smoke
```

Use WASD and the mouse in the interactive window. Run an executable without arguments to see its available options.

## Tests

```sh
python test_worldmodel_kernels.py
python test_vae_encoder_parity.py
python test_standalone_probe.py
python test_vulkan_gemm_cta.py
python test_vulkan_sparse_gqa_fmha.py
./build/worldmodel_cuda_gemm_probe --tensorop
./build-vulkan/worldmodel_vulkan_probe --taehv
./build-vulkan/worldmodel_vulkan_probe --cache-indices
./build-vulkan/worldmodel_vulkan_probe --half-boundary
./build-vulkan/worldmodel_vulkan_probe --sparse-fmha
```

Autotune the five model GEMM shapes and load the resulting cache at runtime:

```sh
./build-vulkan/worldmodel_vulkan_gemm_autotune \
  --config ../Waypoint-1.5-1B-360P/config.yaml \
  --input f16 --epilogue model --out world_vulkan_gemm.cache
WORLD_VULKAN_GEMM_AUTOTUNE_CACHE=world_vulkan_gemm.cache \
  ./build-vulkan/worldmodel_raylib_vulkan \
  --model-dir ../Waypoint-1.5-1B-360P \
  --vae-weights ../Waypoint-1.5-1B/vae/diffusion_pytorch_model.safetensors
```

Implementation and optimization notes are in [docs/cuda_cutlass_kernels.md](docs/cuda_cutlass_kernels.md) and [docs/vulkan_optimization.md](docs/vulkan_optimization.md).
