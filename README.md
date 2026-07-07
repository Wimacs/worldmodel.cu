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
./build/worldmodel_cuda --model-dir ../Waypoint-1.5-1B --steps 4 --frames 2 --frame-idx 3 --cache-pass --control-seq controls_seq.f32 --out /tmp/world_full.ppm --dump-prefix /tmp/world_full
```

Initial latent noise defaults to `--noise normal`, using the standalone C
LCG plus Box-Muller transform to produce standard normal samples. Pass
`--noise uniform` to reproduce the earlier uniform `[-1, 1]` parity fixtures.
Pass `--latent latent.f32` to load external little-endian float32 latent values
instead of sampling noise; the file must contain
`frames * channels * height * patch_h * width * patch_w` values, with one
latent frame after another.
Pass `--control controls.f32` to provide one little-endian float32 controller
vector of length `n_buttons + 3`; it is broadcast to every generated frame. Pass
`--control-seq controls_seq.f32` to provide `frames * (n_buttons + 3)` float32
values, one controller vector per generated latent frame. If omitted, the
executable uses zeros.
Pass `--frame-idx N` to set the temporal RoPE position and per-layer cache
bucket for the first generated latent frame. Pass `--frames N` to generate N
latent frames in one process while keeping the per-layer KV caches alive between
frames; `--frames > 1` automatically enables `--cache-pass`. Pass `--cache-pass`
to run one final sigma=0 unfrozen transformer pass after each frame's scheduler;
this writes the final latent's K/V entries into the local/global ring caches.
With `--dump-prefix`, the standalone executable also writes
`/tmp/world_full.cache_written_counts.f32` with one float count per layer so
cache writes can be parity-checked.

This executable is plain C+CUDA and currently links only CUDA runtime and
cuBLAS. It parses `config.yaml`, reads `transformer/diffusion_pytorch_model.safetensors`,
loads the real transformer weights, and runs the scheduler through the WorldDiT
path. The standalone transformer copies all requested layer weights to GPU once
per run, then reuses those resident weights across scheduler steps. If `--out`
is provided, it also reads
`vae/diffusion_pytorch_model.safetensors`, decodes the final latent with the
TAEHV decoder, writes the first decoded RGB frame to the requested PPM path,
and writes the full decoded 4-frame chunk as sibling files such as
`/tmp/world_full.0.ppm` through `/tmp/world_full.3.ppm` for the first latent
frame. Multi-frame rollouts continue the decoded frame numbering, e.g.
`--frames 2` also writes `/tmp/world_full.4.ppm` through
`/tmp/world_full.7.ppm`. The default is one scheduler step for quick parity
checks; `--steps 4` follows the current config schedule
`1 -> 0.9 -> 0.75 -> 0.3 -> 0`.
Each step runs all requested WorldDiT layers, converts the final tokens back to
a latent velocity tensor, and updates the latent on GPU. The decode path then
expands the final latent to a `1024x512` RGB frame:

```text
sigma embedding -> denoise MLP
random latent -> patchify
24x (cond head -> AdaRMSNorm -> Q/K/V -> RMS+OrthoRoPE -> KV cache indexed GQA attention
     -> out projection -> gated residual add -> optional controller ctrl fusion
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
  normal or uniform seeded latent noise, external latent loading, controller
  input loading, controller embedding, per-layer local/global KV ring-cache
  allocation, frozen tail upsert, indexed GQA attention, 24-layer transformer
  token forward, value residual, optional ctrl fusion with
  `fc1_x + fc1_c`, DiT MLP, final out_norm modulation, unpatchify back to latent velocity,
  scheduler latent updates through the config sigma schedule, F16 VAE weight
  conversion, resident VAE decoder weights/scratch buffers, TAEHV direct
  conv/MemBlock/TGrow/upsample decode, pixel shuffle, and 4-frame PPM output.
  Multi-frame standalone rollout recomputes the controller embedding from the
  current frame's control vector before each denoise/cache pass pair.
  `test_standalone_probe.py` checks the standalone normal-noise fixture,
  external latent loading, the fully dumped layer-0 path, and a two-layer
  transformer + latent output + one-step scheduler update + VAE PPM decode path
  against PyTorch reference math for all 4 decoded frames.
- `generate_smoke.py` is intentionally hybrid for now: World-specific CUDA
  kernels are used for patch/token layout, QKV+RoPE, KV cache, and attention;
  linear layers still use PyTorch/cuBLAS while the dedicated GEMM path is built.
- In the standalone transformer path, each layer's Q/K/V projection weights are
  copied to GPU as one resident concatenated matrix, so QKV projection is one
  cuBLAS GEMM before `qkv_rms_rope` fuses QKV split, Q/K RMSNorm, World
  OrthoRoPE, and V layout.
- The standalone layer conditioning heads also keep the six attention/MLP
  scale-bias-gate projection matrices as one resident GPU matrix, reducing the
  per-layer conditioning projection from six cuBLAS GEMMs to one.
- Controller conditioning vectors are projected once per generated frame for
  each control-conditioned layer, then reused across all scheduler and cache
  passes for that frame.
- Indexed cache attention now consumes the cache index count directly from GPU
  memory, avoiding a per-layer host round trip between index collection and
  attention launch.
- The standalone transformer alternates two token buffers between layers and
  aliases the no-controller path, removing per-layer device-to-device token
  copies.
- `worldmodel_cuda` now uses per-layer ring caches and indexed GQA attention in
  the standalone transformer path. It supports a final unfrozen cache write pass
  and a simple multi-frame rollout loop with cache history persisting across
  generated latent frames. The current Waypoint-1.5-1B config has
  `prompt_conditioning: null`, so this checkpoint does not have prompt
  cross-attention weights to load in the standalone path.
- `masked_attention` is an online-softmax GQA written-mask kernel kept for
  focused extension parity coverage.
- `kv_cache_upsert` mirrors the Python ring-cache/tail-frame update semantics.
- `patchify` and `unpatchify` fuse the WorldModel layout transforms with their
  Conv2d/Linear math.
- `taehv_conv2d`, `taehv_concat_past`, `taehv_upsample2`, and
  `taehv_tgrow_reshape` have focused CUDA extension parity tests. The standalone
  VAE decoder uses the same formulas in plain C+CUDA.
