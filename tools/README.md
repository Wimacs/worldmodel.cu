# Tools

Production runtime code lives under `src/`. This directory contains entry points used to validate,
profile, and generate reference data without adding those concerns to the runtime API.

- `tests/`: C validation tests and CUDA/PyTorch parity tests.
- `profile/`: CUDA runtime, GEMM, attention, and W8A8 profiling tools.
- `reference/`: offline PyTorch/reference-data generators.

Production parity uses two thin PyTorch bindings: `tests/cuda_ops_torch.cu` compiles and calls
`src/world_cuda_ops.cu` for Transformer operators, while `tests/cuda_vae_ops_torch.cu` compiles
and calls `src/world_cuda_vae_ops.cu` for VAE operators. The VAE binding synchronizes PyTorch's
current stream before each composite call and the production operators before returning because
the backend-private VAE API intentionally uses its implicit per-thread default stream.

`tests/cuda_experimental_torch.cu` is limited to prototypes without a production API: standalone
OrthoRoPE, written-mask attention, raw contiguous-BMHK CUTLASS FMHA, and the FP32 NHWC
implicit-GEMM VAE convolution. The two-layer FP32 NHWC test composes that one convolution
prototype with `torch.relu`; it does not carry a second ReLU or production VAE kernel copy.

Typical commands:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DWORLD_BUILD_TOOLS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
python -m pytest tools/tests
./build/tools/worldmodel_cuda_gemm_probe --tensorop
```

Set `WORLD_BUILD_TOOLS=OFF` for a production-only build. C tests remain controlled separately by
CMake's standard `BUILD_TESTING` option.
