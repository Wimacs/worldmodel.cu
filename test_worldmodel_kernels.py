import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import CUDA_HOME, load

try:
    import pytest
except ModuleNotFoundError:
    class _PyTestShim:
        def fixture(self, *args, **kwargs):
            def deco(fn):
                return fn
            return deco

        def skip(self, msg):
            raise RuntimeError(msg)

    pytest = _PyTestShim()


ROOT = Path(__file__).resolve().parent


def find_cutlass_include():
    candidates = []
    env_dir = os.environ.get("WORLD_CUTLASS_DIR") or os.environ.get("CUTLASS_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.extend(
        [
            ROOT / "3rd" / "cutlass",
            ROOT.parents[1] / "flux2" / "third_party" / "cutlass",
            ROOT.parents[1] / "flux.cu" / "third_party" / "cutlass",
        ]
    )
    for candidate in candidates:
        include = candidate / "include"
        if (include / "cutlass" / "cutlass.h").exists():
            return include
    return None


def load_wm_cuda():
    if not torch.cuda.is_available() or CUDA_HOME is None:
        pytest.skip("CUDA and nvcc are required for worldmodel.cu parity tests")
    cutlass_include = find_cutlass_include()
    if cutlass_include is None:
        pytest.skip("CUTLASS include directory is required for worldmodel.cu parity tests")

    build_dir = ROOT / ".torch_extensions"
    build_dir.mkdir(exist_ok=True)
    return load(
        name="worldmodel_cuda_ext",
        sources=[str(ROOT / "worldmodel_kernels.cu")],
        build_directory=str(build_dir),
        extra_include_paths=[str(cutlass_include)],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=bool(os.environ.get("WORLD_CU_VERBOSE_BUILD")),
    )


@pytest.fixture(scope="session")
def wm_cuda():
    return load_wm_cuda()


def _world_rope_tables(d_head: int, height: int, width: int, device: torch.device):
    assert d_head % 8 == 0
    d_xy = d_head // 8
    d_t = d_head // 4
    nyq = 0.8
    max_freq = min(height, width) * nyq
    n = (d_xy + 1) // 2
    xy = (torch.linspace(1.0, max_freq / 2, n, dtype=torch.float32, device=device) * torch.pi)
    xy = xy.repeat_interleave(2)[:d_xy].contiguous()

    theta = 10000.0
    inv_t = 1.0 / (theta ** (torch.arange(0, d_t, 2, dtype=torch.float32, device=device) / d_t))
    inv_t = inv_t.repeat_interleave(2).contiguous()
    return xy, inv_t


def _ref_ortho_rope(x, x_pos, y_pos, t_pos, xy, inv_t, width, height):
    x_coord = (2.0 * x_pos.float() + 1.0) / width - 1.0
    y_coord = (2.0 * y_pos.float() + 1.0) / height - 1.0
    freqs = torch.cat(
        (
            x_coord[:, None] * xy[None, :],
            y_coord[:, None] * xy[None, :],
            t_pos.float()[:, None] * inv_t[None, :],
        ),
        dim=-1,
    )
    cos = freqs.cos()[None, None]
    sin = freqs.sin()[None, None]
    x0, x1 = x.float().unfold(-1, 2, 2).unbind(-1)
    return torch.cat((x0 * cos - x1 * sin, x1 * cos + x0 * sin), dim=-1)


def test_silu_matches_torch(wm_cuda):
    torch.manual_seed(1)
    x = torch.randn(4099, device="cuda", dtype=torch.float32) * 4
    y = wm_cuda.silu(x)
    ref = F.silu(x)
    torch.testing.assert_close(y, ref, rtol=1e-6, atol=1e-6)


def test_row_major_linear_fp16_matches_half_rounded_reference(wm_cuda):
    torch.manual_seed(11)
    x = torch.randn(32, 256, device="cuda", dtype=torch.float32) * 0.5
    w = torch.randn(384, 256, device="cuda", dtype=torch.float32) * 0.5

    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        y = wm_cuda.row_major_linear_fp16(x.contiguous(), w.contiguous())
        ref = x.half().float().matmul(w.half().float().t())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32

    torch.testing.assert_close(y, ref, rtol=2e-3, atol=2e-3)


def test_row_major_linear_fp16_tensorop_matches_real_shapes(wm_cuda):
    torch.manual_seed(111)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for m, k, n in (
            (1, 2048, 8192),
            (512, 2048, 2048),
            (512, 2048, 4096),
            (512, 2048, 8192),
            (512, 8192, 2048),
        ):
            x = torch.randn(m, k, device="cuda", dtype=torch.float32) * 0.25
            w = torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.25
            y = wm_cuda.row_major_linear_fp16_tensorop(x.contiguous(), w.contiguous())
            ref = x.half().float().matmul(w.half().float().t())
            torch.testing.assert_close(y, ref, rtol=2e-3, atol=2e-3)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


def test_rms_norm_matches_torch(wm_cuda):
    torch.manual_seed(2)
    x = torch.randn(5, 17, 128, device="cuda", dtype=torch.float32)
    y = wm_cuda.rms_norm(x, 1.0e-6)
    ref = F.rms_norm(x, (x.shape[-1],), eps=1.0e-6)
    torch.testing.assert_close(y, ref, rtol=2e-6, atol=2e-6)


def test_ada_rms_norm_matches_world_formula(wm_cuda):
    torch.manual_seed(3)
    b, n, m, d = 2, 3, 5, 96
    x = torch.randn(b, n * m, d, device="cuda", dtype=torch.float32)
    scale = torch.randn(b, n, d, device="cuda", dtype=torch.float32) * 0.1
    bias = torch.randn(b, n, d, device="cuda", dtype=torch.float32) * 0.1

    y = wm_cuda.ada_rms_norm(x, scale, bias, 1.0e-6)
    x4 = x.view(b, n, m, d)
    ref = F.rms_norm(x4, (d,), eps=1.0e-6) * (1.0 + scale[:, :, None, :]) + bias[:, :, None, :]
    ref = ref.reshape_as(x)
    torch.testing.assert_close(y, ref, rtol=3e-6, atol=3e-6)


def test_ortho_rope_matches_world_formula(wm_cuda):
    torch.manual_seed(4)
    b, h, t, d = 2, 3, 11, 128
    height, width = 18, 20
    x = torch.randn(b, h, t, d, device="cuda", dtype=torch.float32)
    idx = torch.arange(t, device="cuda", dtype=torch.long)
    y_pos = (idx % height).contiguous()
    x_pos = ((idx * 3) % width).contiguous()
    t_pos = (idx * 7).contiguous()
    xy, inv_t = _world_rope_tables(d, height, width, x.device)

    y = wm_cuda.ortho_rope(x, x_pos, y_pos, t_pos, xy, inv_t, width, height)
    ref = _ref_ortho_rope(x, x_pos, y_pos, t_pos, xy, inv_t, width, height)
    torch.testing.assert_close(y, ref, rtol=2e-5, atol=2e-5)


def test_qkv_rms_rope_matches_split_reference(wm_cuda):
    torch.manual_seed(5)
    b, t, n_heads, n_kv_heads, d = 2, 13, 6, 2, 128
    height, width = 16, 19
    total = n_heads + 2 * n_kv_heads
    qkv = torch.randn(b, t, total * d, device="cuda", dtype=torch.float32)
    idx = torch.arange(t, device="cuda", dtype=torch.long)
    y_pos = (idx % height).contiguous()
    x_pos = ((idx * 5) % width).contiguous()
    t_pos = (idx * 2).contiguous()
    xy, inv_t = _world_rope_tables(d, height, width, qkv.device)

    q, k, v = wm_cuda.qkv_rms_rope(qkv, x_pos, y_pos, t_pos, xy, inv_t, n_heads, n_kv_heads, width, height, 1.0e-6)

    q_raw, k_raw, v_raw = qkv.split((n_heads * d, n_kv_heads * d, n_kv_heads * d), dim=-1)
    q_ref = q_raw.view(b, t, n_heads, d).permute(0, 2, 1, 3).contiguous()
    k_ref = k_raw.view(b, t, n_kv_heads, d).permute(0, 2, 1, 3).contiguous()
    v_ref = v_raw.view(b, t, n_kv_heads, d).permute(0, 2, 1, 3).contiguous()

    q_ref = _ref_ortho_rope(F.rms_norm(q_ref, (d,), eps=1.0e-6), x_pos, y_pos, t_pos, xy, inv_t, width, height)
    k_ref = _ref_ortho_rope(F.rms_norm(k_ref, (d,), eps=1.0e-6), x_pos, y_pos, t_pos, xy, inv_t, width, height)

    torch.testing.assert_close(q, q_ref, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(k, k_ref, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(v, v_ref, rtol=0, atol=0)


def test_masked_attention_matches_torch_gqa_written_mask(wm_cuda):
    torch.manual_seed(6)
    b, hq, hkv, tq, tk, d = 2, 6, 2, 7, 11, 64
    q = torch.randn(b, hq, tq, d, device="cuda", dtype=torch.float32)
    k = torch.randn(b, hkv, tk, d, device="cuda", dtype=torch.float32)
    v = torch.randn(b, hkv, tk, d, device="cuda", dtype=torch.float32)
    written = torch.tensor([1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1], device="cuda", dtype=torch.bool)
    scale = d ** -0.5

    y = wm_cuda.masked_attention(q, k, v, written, scale)

    group = hq // hkv
    k_gqa = k.repeat_interleave(group, dim=1)
    v_gqa = v.repeat_interleave(group, dim=1)
    scores = torch.einsum("bhtd,bhkd->bhtk", q, k_gqa) * scale
    scores = scores.masked_fill(~written.view(1, 1, 1, tk), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    ref = torch.einsum("bhtk,bhkd->bhtd", probs, v_gqa)

    torch.testing.assert_close(y, ref, rtol=3e-5, atol=3e-5)


def test_indexed_attention_matches_masked_attention_reference(wm_cuda):
    torch.manual_seed(11)
    b, hq, hkv, tq, tk, d = 1, 8, 2, 9, 17, 64
    q = torch.randn(b, hq, tq, d, device="cuda", dtype=torch.float32)
    k = torch.randn(b, hkv, tk, d, device="cuda", dtype=torch.float32)
    v = torch.randn(b, hkv, tk, d, device="cuda", dtype=torch.float32)
    indices = torch.tensor([0, 3, 4, 9, 16], device="cuda", dtype=torch.long)
    scale = d ** -0.5

    y = wm_cuda.indexed_attention(q, k, v, indices, scale)

    group = hq // hkv
    k_gqa = k.repeat_interleave(group, dim=1)
    v_gqa = v.repeat_interleave(group, dim=1)
    scores = torch.einsum("bhtd,bhkd->bhtk", q, k_gqa[:, :, indices, :]) * scale
    probs = torch.softmax(scores, dim=-1)
    ref = torch.einsum("bhtn,bhnd->bhtd", probs, v_gqa[:, :, indices, :])

    torch.testing.assert_close(y, ref, rtol=3e-5, atol=3e-5)


def test_indexed_attention_flash_matches_masked_attention_reference(wm_cuda):
    torch.manual_seed(19)
    b, hq, hkv, tq, tk, d = 2, 8, 2, 11, 23, 64
    q = torch.randn(b, hq, tq, d, device="cuda", dtype=torch.float32)
    k = torch.randn(b, hkv, tk, d, device="cuda", dtype=torch.float32)
    v = torch.randn(b, hkv, tk, d, device="cuda", dtype=torch.float32)
    indices = torch.tensor([0, 2, 5, 7, 11, 16, 22], device="cuda", dtype=torch.long)
    scale = d ** -0.5

    y = wm_cuda.indexed_attention_flash(q, k, v, indices, scale)

    group = hq // hkv
    k_gqa = k.repeat_interleave(group, dim=1)
    v_gqa = v.repeat_interleave(group, dim=1)
    scores = torch.einsum("bhtd,bhkd->bhtk", q, k_gqa[:, :, indices, :]) * scale
    probs = torch.softmax(scores, dim=-1)
    ref = torch.einsum("bhtn,bhnd->bhtd", probs, v_gqa[:, :, indices, :])

    torch.testing.assert_close(y, ref, rtol=3e-5, atol=3e-5)


def _ref_kv_cache_upsert(cache_k, cache_v, written, k, v, frame_idx, ring_length, pinned_dilation, frozen):
    t = k.shape[2]
    bucket = (frame_idx + (pinned_dilation - 1)) // pinned_dilation
    num_buckets = (ring_length // t) // pinned_dilation
    base = (bucket % num_buckets) * t
    ring_idx = torch.arange(base, base + t, device=k.device)
    tail_idx = torch.arange(ring_length, ring_length + t, device=k.device)
    write_step = frame_idx % pinned_dilation == 0

    mask_written = written.clone()
    if write_step:
        mask_written[ring_idx] = False

    cache_k[:, :, tail_idx, :] = k
    cache_v[:, :, tail_idx, :] = v
    if not frozen:
        dst = ring_idx if write_step else tail_idx
        cache_k[:, :, dst, :] = k
        cache_v[:, :, dst, :] = v
        written[dst] = True

    return mask_written


def test_kv_cache_upsert_matches_frozen_write_step(wm_cuda):
    torch.manual_seed(7)
    b, h, t, d = 1, 2, 4, 8
    ring_length = 16
    capacity = ring_length + t
    cache_k = torch.randn(b, h, capacity, d, device="cuda", dtype=torch.float32)
    cache_v = torch.randn(b, h, capacity, d, device="cuda", dtype=torch.float32)
    written = torch.zeros(capacity, device="cuda", dtype=torch.bool)
    written[0:4] = True
    written[ring_length:] = True
    k = torch.randn(b, h, t, d, device="cuda", dtype=torch.float32)
    v = torch.randn(b, h, t, d, device="cuda", dtype=torch.float32)

    ref_k = cache_k.clone()
    ref_v = cache_v.clone()
    ref_written = written.clone()
    ref_mask = _ref_kv_cache_upsert(ref_k, ref_v, ref_written, k, v, frame_idx=4, ring_length=ring_length, pinned_dilation=1, frozen=True)

    mask = wm_cuda.kv_cache_upsert(cache_k, cache_v, written, k, v, 4, ring_length, 1, True)
    torch.testing.assert_close(cache_k, ref_k, rtol=0, atol=0)
    torch.testing.assert_close(cache_v, ref_v, rtol=0, atol=0)
    torch.testing.assert_close(written, ref_written, rtol=0, atol=0)
    torch.testing.assert_close(mask, ref_mask, rtol=0, atol=0)


def test_kv_cache_upsert_matches_unfrozen_pinned_dilation(wm_cuda):
    torch.manual_seed(8)
    b, h, t, d = 1, 3, 4, 16
    ring_length = 32
    capacity = ring_length + t
    cache_k = torch.zeros(b, h, capacity, d, device="cuda", dtype=torch.float32)
    cache_v = torch.zeros(b, h, capacity, d, device="cuda", dtype=torch.float32)
    written = torch.zeros(capacity, device="cuda", dtype=torch.bool)
    written[0:8] = True
    written[ring_length:] = True
    k = torch.randn(b, h, t, d, device="cuda", dtype=torch.float32)
    v = torch.randn(b, h, t, d, device="cuda", dtype=torch.float32)

    ref_k = cache_k.clone()
    ref_v = cache_v.clone()
    ref_written = written.clone()
    ref_mask = _ref_kv_cache_upsert(ref_k, ref_v, ref_written, k, v, frame_idx=5, ring_length=ring_length, pinned_dilation=2, frozen=False)

    mask = wm_cuda.kv_cache_upsert(cache_k, cache_v, written, k, v, 5, ring_length, 2, False)
    torch.testing.assert_close(cache_k, ref_k, rtol=0, atol=0)
    torch.testing.assert_close(cache_v, ref_v, rtol=0, atol=0)
    torch.testing.assert_close(written, ref_written, rtol=0, atol=0)
    torch.testing.assert_close(mask, ref_mask, rtol=0, atol=0)


def test_cache_frame_indices_matches_mask_nonzero_reference(wm_cuda):
    t = 4
    slots = 9
    capacity = t * slots
    written = torch.zeros(capacity, device="cuda", dtype=torch.bool)
    for slot in (0, 2, 5, 8):
        written[slot * t : (slot + 1) * t] = True

    base = 2 * t
    indices, count = wm_cuda.cache_frame_indices(written, t, base, True)
    mask = written.clone()
    mask[base : base + t] = False
    ref = torch.nonzero(mask, as_tuple=False).flatten()
    n = int(count.cpu()[0])
    assert n == ref.numel()
    torch.testing.assert_close(indices[:n], ref, rtol=0, atol=0)

    indices, count = wm_cuda.cache_frame_indices(written, t, 0, False)
    ref = torch.nonzero(written, as_tuple=False).flatten()
    n = int(count.cpu()[0])
    assert n == ref.numel()
    torch.testing.assert_close(indices[:n], ref, rtol=0, atol=0)


def test_patchify_matches_torch_conv2d_layout(wm_cuda):
    torch.manual_seed(9)
    b, c, h, w = 2, 5, 8, 10
    d, ph, pw = 7, 2, 2
    x = torch.randn(b, c, h, w, device="cuda", dtype=torch.float32)
    weight = torch.randn(d, c, ph, pw, device="cuda", dtype=torch.float32)

    y = wm_cuda.patchify(x, weight)
    ref = F.conv2d(x, weight, bias=None, stride=(ph, pw))
    ref = ref.permute(0, 2, 3, 1).reshape(b, (h // ph) * (w // pw), d).contiguous()
    torch.testing.assert_close(y, ref, rtol=2e-5, atol=2e-5)


def test_patchify_cutlass_matches_torch_conv2d_layout(wm_cuda):
    torch.manual_seed(17)
    b, c, h, w = 2, 4, 8, 12
    d, ph, pw = 7, 2, 3
    x = torch.randn(b, c, h, w, device="cuda", dtype=torch.float32)
    weight = torch.randn(d, c, ph, pw, device="cuda", dtype=torch.float32)

    y = wm_cuda.patchify_cutlass(x, weight)
    ref = F.conv2d(x, weight, bias=None, stride=(ph, pw))
    ref = ref.permute(0, 2, 3, 1).reshape(b, (h // ph) * (w // pw), d).contiguous()
    torch.testing.assert_close(y, ref, rtol=2e-5, atol=2e-5)


def test_unpatchify_matches_torch_linear_layout(wm_cuda):
    torch.manual_seed(10)
    b, c, h, w = 2, 4, 6, 8
    ph, pw = 2, 2
    hp, wp = h // ph, w // pw
    d = 9
    out_dim = c * ph * pw
    tokens = torch.randn(b, hp * wp, d, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_dim, d, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_dim, device="cuda", dtype=torch.float32)

    y = wm_cuda.unpatchify(tokens, weight, bias, c, h, w, ph, pw)
    ref = F.linear(tokens, weight, bias)
    ref = ref.view(b, hp, wp, c, ph, pw).permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w).contiguous()
    torch.testing.assert_close(y, ref, rtol=2e-5, atol=2e-5)


def test_taehv_conv2d_matches_torch_same_padding(wm_cuda):
    torch.manual_seed(12)
    n, cin, cout, h, w = 3, 5, 7, 9, 11
    x = torch.randn(n, cin, h, w, device="cuda", dtype=torch.float32)
    weight = torch.randn(cout, cin, 3, 3, device="cuda", dtype=torch.float32) * 0.2
    bias = torch.randn(cout, device="cuda", dtype=torch.float32) * 0.1

    y = wm_cuda.taehv_conv2d(x, weight, bias)
    ref = F.conv2d(x, weight, bias=bias, padding=1)
    torch.testing.assert_close(y, ref, rtol=2e-5, atol=2e-5)


def test_taehv_conv1x1_cutlass_matches_torch(wm_cuda):
    torch.manual_seed(121)
    n, cin, cout, h, w = 4, 6, 9, 8, 13
    x = torch.randn(n, cin, h, w, device="cuda", dtype=torch.float32)
    weight = torch.randn(cout, cin, 1, 1, device="cuda", dtype=torch.float32) * 0.2
    bias = torch.randn(cout, device="cuda", dtype=torch.float32) * 0.1

    y = wm_cuda.taehv_conv1x1_cutlass(x, weight, bias)
    ref = F.conv2d(x, weight, bias=bias)
    torch.testing.assert_close(y, ref, rtol=2e-5, atol=2e-5)


def test_taehv_conv3x3_cutlass_matches_torch_same_padding(wm_cuda):
    torch.manual_seed(122)
    n, cin, cout, h, w = 3, 5, 8, 7, 10
    x = torch.randn(n, cin, h, w, device="cuda", dtype=torch.float32)
    weight = torch.randn(cout, cin, 3, 3, device="cuda", dtype=torch.float32) * 0.2
    bias = torch.randn(cout, device="cuda", dtype=torch.float32) * 0.1

    y = wm_cuda.taehv_conv3x3_cutlass(x, weight, bias)
    ref = F.conv2d(x, weight, bias=bias, padding=1)
    torch.testing.assert_close(y, ref, rtol=2e-5, atol=2e-5)


def test_taehv_conv3x3_cutlass_batched_matches_torch_same_padding(wm_cuda):
    torch.manual_seed(123)
    n, cin, cout, h, w = 4, 5, 8, 7, 10
    x = torch.randn(n, cin, h, w, device="cuda", dtype=torch.float32)
    weight = torch.randn(cout, cin, 3, 3, device="cuda", dtype=torch.float32) * 0.2
    bias = torch.randn(cout, device="cuda", dtype=torch.float32) * 0.1

    y = wm_cuda.taehv_conv3x3_cutlass_batched(x, weight, bias, 97)
    ref = F.conv2d(x, weight, bias=bias, padding=1)
    torch.testing.assert_close(y, ref, rtol=2e-5, atol=2e-5)


def test_taehv_conv3x3_cutlass_implicit_nhwc_matches_torch_same_padding(wm_cuda):
    torch.manual_seed(124)
    n, cin, cout, h, w = 2, 5, 7, 8, 11
    x = torch.randn(n, cin, h, w, device="cuda", dtype=torch.float32)
    weight = torch.randn(cout, cin, 3, 3, device="cuda", dtype=torch.float32) * 0.2
    bias = torch.randn(cout, device="cuda", dtype=torch.float32) * 0.1

    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    weight_krsc = weight.permute(0, 2, 3, 1).contiguous()
    y = wm_cuda.taehv_conv3x3_cutlass_implicit_nhwc(x_nhwc, weight_krsc, bias)
    y_nchw = y.permute(0, 3, 1, 2).contiguous()
    ref = F.conv2d(x, weight, bias=bias, padding=1)
    torch.testing.assert_close(y_nchw, ref, rtol=2e-5, atol=2e-5)


def test_taehv_conv3x3_cutlass_implicit_nhwc_pair_matches_torch(wm_cuda):
    torch.manual_seed(125)
    n, cin, cmid, cout, h, w = 2, 5, 8, 7, 8, 11
    x = torch.randn(n, cin, h, w, device="cuda", dtype=torch.float32)
    weight0 = torch.randn(cmid, cin, 3, 3, device="cuda", dtype=torch.float32) * 0.2
    bias0 = torch.randn(cmid, device="cuda", dtype=torch.float32) * 0.1
    weight1 = torch.randn(cout, cmid, 3, 3, device="cuda", dtype=torch.float32) * 0.2
    bias1 = torch.randn(cout, device="cuda", dtype=torch.float32) * 0.1

    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    weight0_krsc = weight0.permute(0, 2, 3, 1).contiguous()
    weight1_krsc = weight1.permute(0, 2, 3, 1).contiguous()
    y = wm_cuda.taehv_conv3x3_cutlass_implicit_nhwc_pair(x_nhwc, weight0_krsc, bias0, weight1_krsc, bias1)
    y_nchw = y.permute(0, 3, 1, 2).contiguous()
    ref = F.conv2d(F.relu(F.conv2d(x, weight0, bias=bias0, padding=1)), weight1, bias=bias1, padding=1)
    torch.testing.assert_close(y_nchw, ref, rtol=2e-5, atol=2e-5)


def test_taehv_concat_past_matches_reference(wm_cuda):
    torch.manual_seed(13)
    x = torch.randn(5, 4, 6, 7, device="cuda", dtype=torch.float32)
    y = wm_cuda.taehv_concat_past(x)
    past = torch.cat((torch.zeros_like(x[:1]), x[:-1]), dim=0)
    ref = torch.cat((x, past), dim=1).contiguous()
    torch.testing.assert_close(y, ref, rtol=0, atol=0)


def test_taehv_upsample2_matches_nearest(wm_cuda):
    torch.manual_seed(14)
    x = torch.randn(3, 5, 4, 7, device="cuda", dtype=torch.float32)
    y = wm_cuda.taehv_upsample2(x)
    ref = F.interpolate(x, scale_factor=2, mode="nearest")
    torch.testing.assert_close(y, ref, rtol=0, atol=0)


def test_taehv_tgrow_reshape_matches_torch_view(wm_cuda):
    torch.manual_seed(15)
    n, c, h, w, stride = 4, 6, 5, 7, 2
    x = torch.randn(n, c * stride, h, w, device="cuda", dtype=torch.float32)
    y = wm_cuda.taehv_tgrow_reshape(x, stride)
    ref = x.reshape(-1, c, h, w).contiguous()
    torch.testing.assert_close(y, ref, rtol=0, atol=0)


if __name__ == "__main__":
    ext = load_wm_cuda()
    for test in (
        test_silu_matches_torch,
        test_row_major_linear_fp16_matches_half_rounded_reference,
        test_row_major_linear_fp16_tensorop_matches_real_shapes,
        test_rms_norm_matches_torch,
        test_ada_rms_norm_matches_world_formula,
        test_ortho_rope_matches_world_formula,
        test_qkv_rms_rope_matches_split_reference,
        test_masked_attention_matches_torch_gqa_written_mask,
        test_indexed_attention_matches_masked_attention_reference,
        test_indexed_attention_flash_matches_masked_attention_reference,
        test_kv_cache_upsert_matches_frozen_write_step,
        test_kv_cache_upsert_matches_unfrozen_pinned_dilation,
        test_cache_frame_indices_matches_mask_nonzero_reference,
        test_patchify_matches_torch_conv2d_layout,
        test_patchify_cutlass_matches_torch_conv2d_layout,
        test_unpatchify_matches_torch_linear_layout,
        test_taehv_conv2d_matches_torch_same_padding,
        test_taehv_conv1x1_cutlass_matches_torch,
        test_taehv_conv3x3_cutlass_matches_torch_same_padding,
        test_taehv_conv3x3_cutlass_batched_matches_torch_same_padding,
        test_taehv_conv3x3_cutlass_implicit_nhwc_matches_torch_same_padding,
        test_taehv_conv3x3_cutlass_implicit_nhwc_pair_matches_torch,
        test_taehv_concat_past_matches_reference,
        test_taehv_upsample2_matches_nearest,
        test_taehv_tgrow_reshape_matches_torch_view,
    ):
        test(ext)
        print(f"{test.__name__}: ok")
