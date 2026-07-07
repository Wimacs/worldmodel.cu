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

Run the current standalone image probe:

```sh
./build/worldmodel_cuda --model-dir ../Waypoint-1.5-1B --steps 4 --out /tmp/world_full.ppm --dump-prefix /tmp/world_full
```

This executable is plain C+CUDA and currently links only CUDA runtime and
cuBLAS. It parses `config.yaml`, reads `transformer/diffusion_pytorch_model.safetensors`,
loads the real transformer weights, and runs the scheduler through the WorldDiT
path. The standalone transformer copies all requested layer weights to GPU once
per run, then reuses those resident weights across scheduler steps. If `--out`
is provided, it also reads
`vae/diffusion_pytorch_model.safetensors`, decodes the final latent with the
TAEHV decoder, writes the first decoded RGB frame to the requested PPM path,
and writes the full decoded 4-frame chunk as sibling files such as
`/tmp/world_full.0.ppm` through `/tmp/world_full.3.ppm`. The default is one
scheduler step for quick parity checks; `--steps 4` follows the current config
schedule `1 -> 0.9 -> 0.75 -> 0.3 -> 0`.
Each step runs all requested WorldDiT layers, converts the final tokens back to
a latent velocity tensor, and updates the latent on GPU. The decode path then
expands the final latent to a `1024x512` RGB frame:

```text
sigma embedding -> denoise MLP
random latent -> patchify
24x (cond head -> AdaRMSNorm -> Q/K/V -> RMS+OrthoRoPE -> current-frame GQA attention
     -> out projection -> gated residual add -> optional zero-control ctrl fusion
     -> MLP AdaRMSNorm -> DiT MLP -> gated residual add)
out_norm modulation -> token RMS+SiLU -> unpatchify -> latent_out [32,32,64]
latent += (next_sigma - sigma) * latent_out
TAEHV decode -> pixel shuffle -> 4x RGB uint8 PPM [1024,512,3]
```

Use `--layers 1` to run only the fully instrumented layer-0 parity path. Pass
`--dump-prefix /tmp/world` to write binary f32 dumps such as
`/tmp/world.latent_out.f32` and `/tmp/world.latent_final.f32`.

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
  patchify, arrayized layer weight loading, resident GPU layer weights,
  24-layer transformer token forward,
  value residual, current-frame GQA attention, optional zero-control ctrl
  fusion, DiT MLP, final out_norm modulation, unpatchify back to latent velocity,
  scheduler latent updates through the config sigma schedule, F16 VAE weight
  conversion, TAEHV direct conv/MemBlock/TGrow/upsample decode, pixel shuffle,
  and 4-frame PPM output.
  `test_standalone_probe.py` checks both the fully dumped layer-0 path and a
  two-layer transformer + latent output + one-step scheduler update + VAE PPM
  decode path against PyTorch reference math for all 4 decoded frames.
- `generate_smoke.py` is intentionally hybrid for now: World-specific CUDA
  kernels are used for patch/token layout, QKV+RoPE, KV cache, and attention;
  linear layers still use PyTorch/cuBLAS while the dedicated GEMM path is built.
- `qkv_rms_rope` fuses QKV split, Q/K RMSNorm, World OrthoRoPE, and V layout.
- `masked_attention` is an online-softmax GQA written-mask kernel. It is a
  correctness-oriented bridge toward the real ring-cache/block-mask attention.
- `kv_cache_upsert` mirrors the Python ring-cache/tail-frame update semantics.
- `patchify` and `unpatchify` fuse the WorldModel layout transforms with their
  Conv2d/Linear math.
- `taehv_conv2d`, `taehv_concat_past`, `taehv_upsample2`, and
  `taehv_tgrow_reshape` have focused CUDA extension parity tests. The standalone
  VAE decoder uses the same formulas in plain C+CUDA.
