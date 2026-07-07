# worldmodel.cu

Small CUDA-first implementation area for the WorldModel runtime.

The current code intentionally starts with standalone operator kernels and
PyTorch parity tests before adding full model weight loading or scheduling.

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
python worldmodel.cu/test_worldmodel_kernels.py
```

The test file is also pytest-compatible:

```sh
python -m pytest -q worldmodel.cu/test_worldmodel_kernels.py
```

Notes:

- Kernels currently target float32 parity first.
- `qkv_rms_rope` fuses QKV split, Q/K RMSNorm, World OrthoRoPE, and V layout.
- `masked_attention` is an online-softmax GQA written-mask kernel. It is a
  correctness-oriented bridge toward the real ring-cache/block-mask attention.
- `kv_cache_upsert` mirrors the Python ring-cache/tail-frame update semantics.
- `patchify` and `unpatchify` fuse the WorldModel layout transforms with their
  Conv2d/Linear math.
