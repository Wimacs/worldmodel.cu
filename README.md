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

Run the current standalone transformer probe:

```sh
./build/worldmodel_cuda --model-dir ../Waypoint-1.5-1B
```

This executable is plain C+CUDA and currently links only CUDA runtime and
cuBLAS. It parses `config.yaml`, reads `transformer/diffusion_pytorch_model.safetensors`,
loads the real transformer weights, and by default runs all 24 WorldDiT layers
through the latent-token path, then converts the final tokens back to a latent
velocity tensor:

```text
sigma embedding -> denoise MLP
random latent -> patchify
24x (cond head -> AdaRMSNorm -> Q/K/V -> RMS+OrthoRoPE -> current-frame GQA attention
     -> out projection -> gated residual add -> optional zero-control ctrl fusion
     -> MLP AdaRMSNorm -> DiT MLP -> gated residual add)
out_norm modulation -> token RMS+SiLU -> unpatchify -> latent_out [32,32,64]
```

Use `--layers 1` to run only the fully instrumented layer-0 parity path. Pass
`--dump-prefix /tmp/world` to write binary f32 dumps such as
`/tmp/world.latent_out.f32`.

Run the standalone executable parity test against PyTorch reference math:

```sh
python test_standalone_probe.py
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
  config parsing, safetensors loading, device allocation, denoise conditioning,
  patchify, arrayized layer weight loading, 24-layer transformer token forward,
  value residual, current-frame GQA attention, optional zero-control ctrl
  fusion, DiT MLP, final out_norm modulation, and unpatchify back to latent.
  `test_standalone_probe.py` checks both the fully dumped layer-0 path and a
  two-layer transformer + latent output path against PyTorch reference math.
- `generate_smoke.py` is intentionally hybrid for now: World-specific CUDA
  kernels are used for patch/token layout, QKV+RoPE, KV cache, and attention;
  linear layers still use PyTorch/cuBLAS while the dedicated GEMM path is built.
- `qkv_rms_rope` fuses QKV split, Q/K RMSNorm, World OrthoRoPE, and V layout.
- `masked_attention` is an online-softmax GQA written-mask kernel. It is a
  correctness-oriented bridge toward the real ring-cache/block-mask attention.
- `kv_cache_upsert` mirrors the Python ring-cache/tail-frame update semantics.
- `patchify` and `unpatchify` fuse the WorldModel layout transforms with their
  Conv2d/Linear math.
