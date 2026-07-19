# Standalone W8A8 probe

This probe validates the proposed WorldModel linear-layer quantization boundary without changing
the resident runtime:

- activation: dynamic symmetric signed INT8, one FP32 scale per token/row;
- weight: static symmetric signed INT8, one FP32 scale per output channel;
- GEMM: CUTLASS Tensor Core `s8 x s8 -> s32`;
- output: `float(accumulator) * activation_scale[row] * weight_scale[column]`.

The probe checks sampled INT32 dot products exactly against a host reference, verifies the outer
scale dequantization, reports SQNR against the original FP32 linear, tests the all-zero-row scale
convention, and covers every real Waypoint token-GEMM shape for both 360P and the main model.

The separate INT32 output and dequantization kernel are intentional: they provide a simple numeric
baseline. They are not the final performance design. A runtime implementation should fuse scale
application into a coordinate-aware epilogue or the downstream consumer to avoid extra standalone
activation/residual passes.

This probe validates the runtime's row/channel-wise PTQ rule. It is not a reproduction of
TurboDiffusion's official 128x128 blockwise quantizer and per-K-block FP32 accumulation; see
[`docs/cuda_w8a8_optimization.md`](../../docs/cuda_w8a8_optimization.md) for the exact comparison.

## Build and run

Windows, from a PowerShell or developer shell:

```powershell
cmake -S tests/w8a8 -B build-w8a8 -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-w8a8 --config Release --parallel
./build-w8a8/Release/worldmodel_w8a8_probe.exe --case quick
./build-w8a8/Release/worldmodel_w8a8_probe.exe --case all --samples 256
./build-w8a8/Release/worldmodel_w8a8_probe.exe --case fc2_main --bench 100
```

Linux:

```sh
cmake -S tests/w8a8 -B build-w8a8 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-w8a8 -j
./build-w8a8/worldmodel_w8a8_probe --case quick
```

Pass `-DWORLD_CUTLASS_DIR=/path/to/cutlass` when CUTLASS is not in a location already searched by
the probe. With no local checkout, CMake fetches the same CUTLASS 3.5.1 version as the main build.

The CPU-side policy tests have no CUDA requirement and can be run directly:

```sh
python test_w8a8_quantization.py
```

`--case quick` runs the small exact sanity case plus the 360P QKV and MLP fc2 shapes. `--case all`
also runs out projection and MLP fc1/fc2 for `M=128` and `M=512`. A custom aligned shape can be
tested with `--m M --k K --n N`.
