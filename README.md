# worldmodel.c

<p align="center">
  <a href="docs/assets/worldmodel-demo.mp4">
    <img src="docs/assets/worldmodel-demo-preview.webp" alt="Interactive world generation demo running at 30 FPS" width="100%">
  </a>
</p>

<p align="center"><sub>Interactive world generation at 30 FPS — click the preview to watch the full video.</sub></p>

An experimental standalone C/CUDA runtime for [Waypoint 1.5](https://raw.githubusercontent.com/Overworldai/Biome/feat/paper/research/WP1_5_Paper.pdf).

The runtime uses CUTLASS for GEMM, attention, and convolution kernels. It links only the CUDA runtime, without cuBLAS or cuDNN. CUDA kernels are tested against PyTorch references. The former CUDA/Vulkan implementation is preserved on the `legacy-cuda-vulkan` branch.

## Performance

Measured on an NVIDIA GeForce RTX 4090 D (48 GiB), NVIDIA driver 610.62, using Windows Release builds. Every run uses all 24 layers, 4 denoising steps, and `--cache-window 8`. A single resident runtime generates 12 consecutive chunks; RGB FPS is the average of the final four full-cache chunks, with four RGB frames emitted per chunk. Peak VRAM is the maximum `nvidia-smi memory.used` increase over the pre-launch baseline, sampled every 100 ms.

| Model | Output | Backend | RGB FPS | Peak VRAM |
|---|---:|---|---:|---:|
| [Waypoint-1.5-1B](https://huggingface.co/Overworld/Waypoint-1.5-1B) | 1024x512 | CUDA | 32.3 | 14.0 GiB |
| [Waypoint-1.5-1B-360P](https://huggingface.co/Overworld/Waypoint-1.5-1B-360P) | 512x256 | CUDA | 102.4 | 11.2 GiB |

These figures measure backend throughput rather than the interactive window's display refresh cap. Results vary with drivers, clocks, and background GPU usage.

## Requirements

- CMake 3.24+
- CUDA Toolkit
- Python 3 for weight downloads and parity tests
- Visual Studio 2022 Build Tools on Windows

## Setup

```sh
git clone --recurse-submodules https://github.com/Wimacs/worldmodel.c.git
cd worldmodel.c
python -m pip install -U huggingface_hub
hf download Overworld/Waypoint-1.5-1B --local-dir Waypoint-1.5-1B
hf download Overworld/Waypoint-1.5-1B-360P --local-dir Waypoint-1.5-1B-360P
```

## Build

Linux CUDA:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Windows CUDA, from an x64 Visual Studio developer shell:

```bat
cmake -S . -B build-win -G "Visual Studio 17 2022" -A x64 ^
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

### Experimental CUDA W8A8

The resident CUDA runtime has an opt-in row/channel-wise W8A8 path for WorldDiT QKV, attention output, controller, and MLP linears:

```powershell
$env:WORLD_W8A8 = '1'
.\build-win\Release\worldmodel_raylib.exe --model-dir .\Waypoint-1.5-1B `
  --steps 4 --cache-window 8 --headless-smoke
```

This is an experimental post-training quantization path, not a TurboDiffusion-equivalent kernel, and long autoregressive rollouts can diverge from the FP32/FP16 trajectory. The implementation, failed first attempt, benchmarks, quality checks, memory accounting, controls, and exact TurboDiffusion comparison are recorded in [docs/cuda_w8a8_optimization.md](docs/cuda_w8a8_optimization.md).

## Tests

```sh
ctest --test-dir build --output-on-failure
python -m pytest tools/tests
```

CUDA profilers and standalone diagnostics are built with `WORLD_BUILD_TOOLS=ON`, which is the default:

```sh
./build/tools/worldmodel_cuda_gemm_probe --tensorop
./build/tools/worldmodel_w8a8_probe --case quick
```

The `tools/` directory contains tests, profilers, and reference generators. Implementation and optimization notes are in [docs/cuda_cutlass_kernels.md](docs/cuda_cutlass_kernels.md) and [docs/cuda_w8a8_optimization.md](docs/cuda_w8a8_optimization.md).
