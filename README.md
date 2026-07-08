# worldmodel.cu

Small CUDA-first implementation area for the WorldModel runtime, now with a
Vulkan backend being ported incrementally.

The current code has two tracks:

- CUDA extension kernels with PyTorch parity tests.
- A standalone C+CUDA executable that loads Waypoint config/safetensors without
  PyTorch and starts the generation path.
- A C+Vulkan resident runtime scaffold that shares the raylib frontend and
  dispatches compute shaders from `shaders/vulkan/`.

Implemented CUDA ops:

- `silu`
- `rms_norm`
- `ada_rms_norm`
- `ortho_rope`
- `qkv_rms_rope`
- `masked_attention`
- `kv_cache_upsert`
- `cache_frame_indices`
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

CUDA and Vulkan targets are controlled independently:

```sh
cmake -S . -B build -DWORLD_ENABLE_CUDA=ON -DWORLD_ENABLE_VULKAN=ON
cmake --build build -j
```

The Vulkan target currently builds `worldmodel_raylib_vulkan` and compiles
the GLSL files in `shaders/vulkan/` to SPIR-V. The runtime creates a Vulkan
device, resident compute pipelines, and returns RGB frames through the same
raylib/headless path. When real model weights are supplied, the Vulkan raylib
path now runs a resident latent scheduler slice every frame:
`control_input -> linear_f32.comp -> silu_f32.comp -> linear_f32.comp ->
rms_norm_f32.comp`, followed by
`patchify_f32.comp -> out_norm_silu_f32.comp -> unpatchify_orig_f32.comp ->
latent_update_f32.comp -> latent_to_rgba.comp`.
At initialization it also precomputes the CUDA-style scheduler conditioning path
`noise_embed -> denoise MLP -> out_norm modulation table`, plus per-layer
`cond + bias -> silu -> 6x layer modulation table`, for all requested scheduler
passes. This wires the loaded control embedding, denoise/out-norm, layer
conditioning, and patch/unpatch weights into the interactive runtime. The DiT
block loop is now resident for all requested layers:
`ada_rms_norm_f32.comp -> linear_f32.comp(QKV) -> qkv_rms_rope_f32.comp ->
kv_cache_upsert_copy_f32.comp -> cache_frame_indices.comp ->
indexed_attention_f32.comp -> linear_f32.comp(attn out_proj) ->
gated_residual_add_f32.comp -> rms_norm_f32.comp(ctrl) ->
linear_f32.comp(ctrl fc1_x/fc1_c/fc2) -> add_channel_silu_f32.comp ->
add_f32.comp -> ada_rms_norm_f32.comp(MLP) -> linear_f32.comp(dit_mlp fc1/fc2)
-> silu_f32.comp -> gated_residual_add_f32.comp`, with per-layer KV cache
offsets, local/global cache parameters, token ping-pong, optional control
fusion per layer, and CUDA-style value residual via `lerp_inplace_f32.comp`.
The current Vulkan runtime then applies the precomputed out-norm modulation,
unpatches to latent velocity, updates the resident latent through the configured
sigma schedule, and runs the final unfrozen cache pass. The path is still FP32
with naive `linear_f32.comp`, and VAE decode is still not ported to Vulkan.
Without weights, it falls back to the simple `fill_rgba.comp` scaffold for
lightweight probes.

`worldmodel_vulkan_probe` currently runs CPU parity checks for `linear_f32.comp`,
elementwise `silu_f32.comp`, fused `add_bias_silu_f32.comp`,
`add_channel_silu_f32.comp`, `add_f32.comp`, `out_norm_silu_f32.comp`,
`latent_update_f32.comp`, `lerp_inplace_f32.comp`, rowwise `rms_norm_f32.comp`, fused control embedding,
denoise/out-norm scheduler conditioning and layer
modulation precompute, `gated_residual_add_f32.comp`, the runtime layer0
QKV/cache/attention/out-projection/control-fusion/DiT-MLP/scheduler-update slice,
`ada_rms_norm_f32.comp`, `ortho_rope_f32.comp`, fused `qkv_rms_rope_f32.comp`,
`masked_attention_f32.comp`, the KV-cache helpers, `indexed_attention_f32.comp`,
and the patch/unpatch latent-token boundary.

```sh
./build/worldmodel_raylib_vulkan \
  --model-dir ../Waypoint-1.5-1B-360P \
  --vae-weights ../Waypoint-1.5-1B/vae/diffusion_pytorch_model.safetensors \
  --layers 4 \
  --steps 4 \
  --cache-window 8 \
  --headless-smoke
```

```sh
./build/worldmodel_vulkan_probe
```

If `3rd/raylib` is present, the build also produces `worldmodel_raylib`, a
raylib frontend that loads the transformer/VAE weights into a resident CUDA
runtime, runs a warmup chunk, then opens a window and streams decoded RGB frames
directly into a raylib texture without writing image files:

```sh
python export_seed_latent.py \
  --model-dir ../Waypoint-1.5-1B \
  --out /tmp/world_seed_latent.f32

./build/worldmodel_raylib \
  --model-dir ../Waypoint-1.5-1B \
  --steps 4 \
  --cache-window 8 \
  --mouse-scale 0.1 \
  --seed-latent /tmp/world_seed_latent.f32
```

The seed latent mirrors the Python runtime's `append_frame()` startup path: it
encodes a starter image with the checkpoint VAE, runs one cache-only pass to
anchor visual history, and then generates subsequent frames from live controls.
Running without `--seed-latent` starts from random latent noise and is useful for
parity/debugging, but it is not a controllable world rollout.

For terminal-only validation of the same resident runtime path:

```sh
./build/worldmodel_raylib \
  --model-dir ../Waypoint-1.5-1B \
  --steps 4 \
  --cache-window 8 \
  --seed-latent /tmp/world_seed_latent.f32 \
  --headless-smoke
```

Add `--headless-out /tmp/world_runtime.ppm` to the smoke command when you want
debug PPM frames for the latest resident-runtime chunk. This is only a debugging
path; the interactive raylib loop still renders directly to the framebuffer.

For a lower-latency interactive mode, use the fast realtime preset. It uses one
scheduler step and clamps both local/global KV cache windows to two frame chunks:

```sh
./build/worldmodel_raylib --model-dir ../Waypoint-1.5-1B --fast-realtime
```

`--cache-window N` can be used independently to trade temporal history for
latency. It clamps both local and global KV cache windows; non-fast global
windows keep the checkpoint's `global_pinned_dilation` and are rounded up when
needed so the dilated cache buckets stay valid. `--cache-window 1` is a
debug-only minimum-latency mode: the current ring slot is masked during
attention, so no previous-frame history is visible and rollout looks like a
fresh sample every frame. Use `--cache-window 2` or higher for interactive
control. On an RTX 4090 D, `--steps 4 --cache-window 8 --warmup 24
--headless-smoke` stabilizes around 235-254 ms per decoded 4-frame RGB chunk,
about 15.7-17.0 RGB fps. D=64 cache attention uses cuBLAS by default for
contiguous `pinned_dilation=1` cache layers, while sparse/global layers keep the
indexed warp fallback. Set `WORLD_CUBLAS_ATTN=0` to force the older all-warp
attention fallback, `WORLD_CUBLAS_ATTN_GQA=1` to try pointer-batched GQA cuBLAS,
or `WORLD_FLASH_ATTN=1` to try the online-softmax indexed attention prototype.
The experimental paths match the masked attention tests, but are not the
default on the RTX 4090 D because they are slower in the current kernels.

The raylib frontend samples WASD, Space, Shift, left/right mouse buttons, mouse
delta, and mouse wheel into the PyTorch controller layout
`[mouse_x, mouse_y, buttons..., scroll]`, using Owl-Control/Windows virtual key
codes for the button indices. CUDA generation runs on a worker thread while the
raylib main thread keeps polling input. When a new decoded chunk arrives, the
frontend presents its frames once in order and then holds on the last frame until
the next chunk arrives. Mouse deltas are multiplied by `--mouse-scale`, which
defaults to `0.1`; lower it to `0.05` if camera motion feels too strong.

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
Pass `--vae-only --latent latent.f32 --out out.ppm` to debug only the TAEHV
decoder, bypassing the transformer path entirely.
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

This executable is plain C+CUDA and links CUDA runtime/cuBLAS, with optional
cuDNN acceleration for the VAE decoder when cuDNN is available at configure time.
The VAE decoder prebuilds cuDNN conv plans, uses pinned host RGB transfer in
the resident realtime path, and defaults to an FP16/NHWC cuDNN path when cuDNN
is available. Set `WORLD_VAE_FP16_NHWC=0` to force the older F32/NCHW decoder
path.
It parses `config.yaml`, reads `transformer/diffusion_pytorch_model.safetensors`,
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

- Kernels currently target float32 parity first, with resident realtime fast
  paths for FP16-weight GEMM, cuBLAS-backed D=64 indexed cache attention, and
  optional FP16/NHWC cuDNN VAE convolutions. `WORLD_FLASH_ATTN=1` enables a
  fused online-softmax attention prototype for parity/perf experiments.
- `worldmodel_cuda` is the no-PyTorch path. At this milestone it verifies
  config parsing, safetensors loading, device allocation, denoise conditioning,
  patchify, arrayized layer weight loading, resident GPU layer weights,
  normal or uniform seeded latent noise, external latent loading, controller
  input loading, controller embedding, per-layer local/global KV ring-cache
  allocation, frozen tail upsert, indexed GQA attention, 24-layer transformer
  token forward, value residual, optional ctrl fusion with
  `fc1_x + fc1_c`, DiT MLP, final out_norm modulation, unpatchify back to latent velocity,
  scheduler latent updates through the config sigma schedule, F16 VAE weight
  conversion, resident VAE decoder weights/scratch buffers, TAEHV
  FP16/NHWC cuDNN-accelerated or built-in F32/NCHW direct
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
- Standalone cache index collection now expands whole-frame cache slots directly
  from `written` flags, preserving ascending index order while removing the
  separate per-token mask kernel and mask buffer.
- The standalone transformer alternates two token buffers between layers and
  aliases the no-controller path, removing per-layer device-to-device token
  copies.
- When value residuals are enabled, layer 0 writes V directly into the resident
  residual buffer, avoiding the separate first-layer V copy.
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
