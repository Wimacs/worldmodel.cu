# worldmodel.cu

Small CUDA-first implementation area for the WorldModel runtime.

The current code has two tracks:

- CUDA extension kernels with PyTorch parity tests.
- A standalone C+CUDA executable that loads Waypoint config/safetensors without
  PyTorch and starts the generation path.

Implemented CUDA ops:

- `silu`
- `rms_norm`
- `ada_rms_norm`
- `ortho_rope`
- `qkv_rms_rope`
- `masked_attention`
- `kv_cache_upsert`
- `patchify`
- `unpatchify`

Run parity tests:

```sh
python test_worldmodel_kernels.py
```

The test file is also pytest-compatible:

```sh
python -m pytest -q test_worldmodel_kernels.py
```

Build the standalone no-PyTorch executable:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Run the current standalone generation probe:

```sh
./build/worldmodel_cuda --model-dir ../Waypoint-1.5-1B
```

This executable is plain C+CUDA and currently links only CUDA runtime and
cuBLAS. It parses `config.yaml`, reads `transformer/diffusion_pytorch_model.safetensors`,
loads real `patchify.weight` and layer-0 `q_proj.weight`, then runs:

```text
random latent -> patchify CUDA kernel -> layer0 q_proj cuBLAS GEMM
```

Run a real-weight latent generation smoke test:

```sh
python generate_smoke.py --model-dir ../Waypoint-1.5-1B --steps 4 --output smoke_latent.pt
```

The smoke runner loads Waypoint safetensors, initializes KV caches, samples a
latent tensor, and runs the scheduler through the WorldDiT path. It also has an
optional final cache write pass:

```sh
python generate_smoke.py --model-dir ../Waypoint-1.5-1B --steps 1 --cache-pass
```

Notes:

- Kernels currently target float32 parity first.
- `worldmodel_cuda` is the no-PyTorch path. At this milestone it verifies
  config parsing, safetensors loading, device allocation, patchify, and the
  first real transformer projection.
- `generate_smoke.py` is intentionally hybrid for now: World-specific CUDA
  kernels are used for patch/token layout, QKV+RoPE, KV cache, and attention;
  linear layers still use PyTorch/cuBLAS while the dedicated GEMM path is built.
- `qkv_rms_rope` fuses QKV split, Q/K RMSNorm, World OrthoRoPE, and V layout.
- `masked_attention` is an online-softmax GQA written-mask kernel. It is a
  correctness-oriented bridge toward the real ring-cache/block-mask attention.
- `kv_cache_upsert` mirrors the Python ring-cache/tail-frame update semantics.
- `patchify` and `unpatchify` fuse the WorldModel layout transforms with their
  Conv2d/Linear math.
