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


ROOT = Path(__file__).resolve().parents[2]


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


def find_cutlass_example_include(cutlass_include: Path):
    example_dir = cutlass_include.parent / "examples" / "41_fused_multi_head_attention"
    return example_dir if (example_dir / "kernel_forward.h").exists() else None


def load_wm_cuda():
    if not torch.cuda.is_available() or CUDA_HOME is None:
        pytest.skip("CUDA and nvcc are required for worldmodel.c parity tests")
    cutlass_include = find_cutlass_include()
    if cutlass_include is None:
        pytest.skip("CUTLASS include directory is required for worldmodel.c parity tests")
    include_paths = [str(ROOT / "src"), str(cutlass_include)]
    cutlass_fmha_include = find_cutlass_example_include(cutlass_include)
    if cutlass_fmha_include is not None:
        include_paths.append(str(cutlass_fmha_include))

    build_dir = ROOT / ".torch_extensions" / "ops"
    build_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name="worldmodel_cuda_ops_ext",
        sources=[
            str(ROOT / "tools" / "tests" / "cuda_ops_torch.cu"),
            str(ROOT / "src" / "world_cuda_ops.cu"),
        ],
        build_directory=str(build_dir),
        extra_include_paths=include_paths,
        extra_cuda_cflags=["-O3", "--use_fast_math", "--default-stream=per-thread"],
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

def test_silu_to_half_matches_torch_half(wm_cuda):
    torch.manual_seed(10)
    x = torch.randn(128, 8192, device="cuda", dtype=torch.float32) * 3
    y = wm_cuda.silu_to_half(x.contiguous())
    ref = F.silu(x).half()
    torch.testing.assert_close(y, ref, rtol=1e-3, atol=4e-3)

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

def test_row_major_linear_fp16_tensorop_splitk_matches_mlp_fc2_shape(wm_cuda):
    torch.manual_seed(112)
    m, k, n = 128, 8192, 2048
    x = torch.randn(m, k, device="cuda", dtype=torch.float32) * 0.125
    w = torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.125

    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        y = wm_cuda.row_major_linear_fp16_tensorop_splitk(x.contiguous(), w.contiguous(), 4)
        ref = x.half().float().matmul(w.half().float().t())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32

    torch.testing.assert_close(y, ref, rtol=3e-3, atol=3e-3)

def test_row_major_linear_fp16_tensorop_m64n64_matches_small_m_shapes(wm_cuda):
    torch.manual_seed(113)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for m, k, n in (
            (128, 2048, 2048),
            (128, 2048, 4096),
            (128, 2048, 8192),
            (128, 8192, 2048),
        ):
            x = torch.randn(m, k, device="cuda", dtype=torch.float32) * 0.125
            w = torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.125
            y = wm_cuda.row_major_linear_fp16_tensorop_m64n64(x.contiguous(), w.contiguous())
            ref = x.half().float().matmul(w.half().float().t())
            torch.testing.assert_close(y, ref, rtol=3e-3, atol=3e-3)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32

def test_row_major_linear_fp16_tensorop_m64n64_silu_half_matches_reference(wm_cuda):
    torch.manual_seed(115)
    m, k, n = 128, 2048, 8192
    x = torch.randn(m, k, device="cuda", dtype=torch.float32) * 0.125
    w = torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.125

    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        y = wm_cuda.row_major_linear_fp16_tensorop_m64n64_silu_half(x.contiguous(), w.contiguous())
        ref = F.silu(x.half().float().matmul(w.half().float().t())).half()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32

    torch.testing.assert_close(y, ref, rtol=3e-3, atol=4e-3)

def test_row_major_linear_fp16_input_tensorop_m64n64_variants_match_reference(wm_cuda):
    torch.manual_seed(128)
    m, k = 128, 2048
    x = (torch.randn(m, k, device="cuda", dtype=torch.float32) * 0.125).half()

    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for n in (2048, 4096):
            w = (torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.125).half()
            y = wm_cuda.row_major_linear_fp16_input_tensorop_m64n64(x.contiguous(), w.contiguous())
            ref = x.float().matmul(w.float().t())
            torch.testing.assert_close(y, ref, rtol=3e-3, atol=3e-3)

        w_silu = (torch.randn(8192, k, device="cuda", dtype=torch.float32) * 0.125).half()
        y_silu = wm_cuda.row_major_linear_fp16_input_tensorop_m64n64_silu_half(
            x.contiguous(), w_silu.contiguous()
        )
        ref_silu = F.silu(x.float().matmul(w_silu.float().t())).half()
        torch.testing.assert_close(y_silu, ref_silu, rtol=3e-3, atol=4e-3)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32

def test_row_major_linear_fp16_input_tensorop_splitk_matches_half_reference(wm_cuda):
    torch.manual_seed(114)
    m, k, n = 128, 8192, 2048
    x = (torch.randn(m, k, device="cuda", dtype=torch.float32) * 0.125).half()
    w = (torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.125).half()

    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        y = wm_cuda.row_major_linear_fp16_input_tensorop_splitk(x.contiguous(), w.contiguous(), 4)
        ref = x.float().matmul(w.float().t())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32

    torch.testing.assert_close(y, ref, rtol=3e-3, atol=3e-3)

def test_row_major_linear_fp16_input_tensorop_splitk_parallel_matches_half_reference(wm_cuda):
    torch.manual_seed(117)
    m, k, n = 128, 8192, 2048
    x = (torch.randn(m, k, device="cuda", dtype=torch.float32) * 0.125).half()
    w = (torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.125).half()

    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        y = wm_cuda.row_major_linear_fp16_input_tensorop_splitk_parallel(x.contiguous(), w.contiguous(), 4)
        ref = x.float().matmul(w.float().t())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32

    torch.testing.assert_close(y, ref, rtol=3e-3, atol=3e-3)

def test_mlp_fused_silu_half_then_splitk_matches_reference(wm_cuda):
    torch.manual_seed(116)
    m, d, hidden = 128, 2048, 8192
    x = torch.randn(m, d, device="cuda", dtype=torch.float32) * 0.125
    w1 = torch.randn(hidden, d, device="cuda", dtype=torch.float32) * 0.125
    w2 = torch.randn(d, hidden, device="cuda", dtype=torch.float32) * 0.125

    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        h = wm_cuda.row_major_linear_fp16_tensorop_m64n64_silu_half(x.contiguous(), w1.contiguous())
        w2_h = w2.half().contiguous()
        y = wm_cuda.row_major_linear_fp16_input_tensorop_splitk(h.contiguous(), w2_h, 4)
        h_ref = F.silu(x.half().float().matmul(w1.half().float().t())).half()
        ref = h_ref.float().matmul(w2_h.float().t())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32

    torch.testing.assert_close(y, ref, rtol=3e-3, atol=4e-3)

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

def test_ada_rms_norm_half_matches_half_rounded_world_formula(wm_cuda):
    torch.manual_seed(129)
    b, n, m, d = 2, 3, 5, 96
    x = torch.randn(b, n * m, d, device="cuda", dtype=torch.float32)
    scale = torch.randn(b, n, d, device="cuda", dtype=torch.float32) * 0.1
    bias = torch.randn(b, n, d, device="cuda", dtype=torch.float32) * 0.1

    y = wm_cuda.ada_rms_norm_half(x, scale, bias, 1.0e-6)
    x4 = x.view(b, n, m, d)
    ref = F.rms_norm(x4, (d,), eps=1.0e-6) * (1.0 + scale[:, :, None, :]) + bias[:, :, None, :]
    ref = ref.reshape_as(x).half()

    assert y.dtype == torch.float16
    torch.testing.assert_close(y, ref, rtol=3e-3, atol=3e-3)

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

def test_indexed_attention_half_kv_matches_half_reference(wm_cuda):
    torch.manual_seed(23)
    b, hq, hkv, tq, tk, d = 2, 8, 2, 13, 31, 64
    q = torch.randn(b, hq, tq, d, device="cuda", dtype=torch.float32)
    k = torch.randn(b, hkv, tk, d, device="cuda", dtype=torch.float32).half().contiguous()
    v = torch.randn(b, hkv, tk, d, device="cuda", dtype=torch.float32).half().contiguous()
    indices = torch.tensor([0, 1, 4, 8, 13, 21, 30], device="cuda", dtype=torch.long)
    scale = d ** -0.5

    y = wm_cuda.indexed_attention_half_kv(q, k, v, indices, scale)

    group = hq // hkv
    k_gqa = k.float().repeat_interleave(group, dim=1)
    v_gqa = v.float().repeat_interleave(group, dim=1)
    scores = torch.einsum("bhtd,bhkd->bhtk", q, k_gqa[:, :, indices, :]) * scale
    probs = torch.softmax(scores, dim=-1)
    ref = torch.einsum("bhtn,bhnd->bhtd", probs, v_gqa[:, :, indices, :])

    torch.testing.assert_close(y, ref, rtol=8e-4, atol=8e-4)

def test_indexed_attention_half_kv_flash_matches_half_reference(wm_cuda):
    torch.manual_seed(24)
    b, hq, hkv, tq, tk, d = 2, 8, 2, 17, 37, 64
    q = torch.randn(b, hq, tq, d, device="cuda", dtype=torch.float32)
    k = torch.randn(b, hkv, tk, d, device="cuda", dtype=torch.float32).half().contiguous()
    v = torch.randn(b, hkv, tk, d, device="cuda", dtype=torch.float32).half().contiguous()
    indices = torch.tensor([0, 1, 3, 5, 8, 13, 21, 30, 36], device="cuda", dtype=torch.long)
    scale = d ** -0.5

    y = wm_cuda.indexed_attention_half_kv_flash(q, k, v, indices, scale)

    group = hq // hkv
    k_gqa = k.float().repeat_interleave(group, dim=1)
    v_gqa = v.float().repeat_interleave(group, dim=1)
    scores = torch.einsum("bhtd,bhkd->bhtk", q, k_gqa[:, :, indices, :]) * scale
    probs = torch.softmax(scores, dim=-1)
    ref = torch.einsum("bhtn,bhnd->bhtd", probs, v_gqa[:, :, indices, :])

    torch.testing.assert_close(y, ref, rtol=8e-4, atol=8e-4)

def test_indexed_attention_half_kv_cutlass_matches_materialized_half_reference(wm_cuda):
    torch.manual_seed(25)
    b, hq, hkv, tq, tk, d = 2, 8, 2, 16, 48, 64
    q = torch.randn(b, hq, tq, d, device="cuda", dtype=torch.float32)
    k = torch.randn(b, hkv, tk, d, device="cuda", dtype=torch.float32).half().contiguous()
    v = torch.randn(b, hkv, tk, d, device="cuda", dtype=torch.float32).half().contiguous()
    indices = torch.tensor([0, 1, 3, 5, 8, 13, 21, 30, 36, 39, 40, 41, 42, 43, 46, 47], device="cuda", dtype=torch.long)
    scale = d ** -0.5

    y = wm_cuda.indexed_attention_half_kv_cutlass(q, k, v, indices, scale)

    group = hq // hkv
    q_h = q.half().float()
    k_gqa = k.float().repeat_interleave(group, dim=1)
    v_gqa = v.float().repeat_interleave(group, dim=1)
    scores = torch.einsum("bhtd,bhkd->bhtk", q_h, k_gqa[:, :, indices, :]) * scale
    probs_h = torch.softmax(scores, dim=-1).half()
    ref = torch.einsum("bhtn,bhnd->bhtd", probs_h.float(), v_gqa[:, :, indices, :])

    torch.testing.assert_close(y, ref, rtol=2e-3, atol=2e-3)

def test_indexed_attention_half_kv_cutlass_grouped_matches_materialized_half_reference(wm_cuda):
    torch.manual_seed(26)
    b, hq, hkv, tq, tk, d = 2, 8, 2, 16, 48, 64
    q = torch.randn(b, hq, tq, d, device="cuda", dtype=torch.float32)
    k = torch.randn(b, hkv, tk, d, device="cuda", dtype=torch.float32).half().contiguous()
    v = torch.randn(b, hkv, tk, d, device="cuda", dtype=torch.float32).half().contiguous()
    indices = torch.tensor([0, 1, 3, 5, 8, 13, 21, 30, 36, 39, 40, 41, 42, 43, 46, 47], device="cuda", dtype=torch.long)
    scale = d ** -0.5

    y = wm_cuda.indexed_attention_half_kv_cutlass_grouped(q, k, v, indices, scale)

    group = hq // hkv
    q_h = q.half().float()
    k_gqa = k.float().repeat_interleave(group, dim=1)
    v_gqa = v.float().repeat_interleave(group, dim=1)
    scores = torch.einsum("bhtd,bhkd->bhtk", q_h, k_gqa[:, :, indices, :]) * scale
    probs_h = torch.softmax(scores, dim=-1).half()
    ref = torch.einsum("bhtn,bhnd->bhtd", probs_h.float(), v_gqa[:, :, indices, :])

    torch.testing.assert_close(y, ref, rtol=2e-3, atol=2e-3)

def test_indexed_attention_half_kv_fmha_gqa_matches_half_reference(wm_cuda):
    torch.manual_seed(127)
    b, hq, hkv, tq, tk, d = 2, 8, 2, 16, 48, 64
    q = torch.randn(b, hq, tq, d, device="cuda", dtype=torch.float32)
    k = torch.randn(b, hkv, tk, d, device="cuda", dtype=torch.float32).half().contiguous()
    v = torch.randn(b, hkv, tk, d, device="cuda", dtype=torch.float32).half().contiguous()
    indices = torch.tensor([0, 1, 3, 5, 8, 13, 21, 30, 36, 39, 40, 41, 42, 43, 46, 47], device="cuda", dtype=torch.long)
    scale = d ** -0.5

    y = wm_cuda.indexed_attention_half_kv_fmha_gqa(q, k, v, indices, scale)

    group = hq // hkv
    q_h = q.half().float()
    k_gqa = k.float().repeat_interleave(group, dim=1)
    v_gqa = v.float().repeat_interleave(group, dim=1)
    scores = torch.einsum("bhtd,bhkd->bhtk", q_h, k_gqa[:, :, indices, :]) * scale
    probs = torch.softmax(scores, dim=-1)
    ref = torch.einsum("bhtn,bhnd->bhtd", probs, v_gqa[:, :, indices, :]).half().float()

    torch.testing.assert_close(y, ref, rtol=4e-3, atol=4e-3)

def test_indexed_attention_half_kv_fmha_gqa_half_output_matches_reference(wm_cuda):
    torch.manual_seed(130)
    b, hq, hkv, tq, tk, d = 2, 8, 2, 16, 48, 64
    q = torch.randn(b, hq, tq, d, device="cuda", dtype=torch.float32)
    k = torch.randn(b, hkv, tk, d, device="cuda", dtype=torch.float32).half().contiguous()
    v = torch.randn(b, hkv, tk, d, device="cuda", dtype=torch.float32).half().contiguous()
    indices = torch.tensor([0, 1, 3, 5, 8, 13, 21, 30, 36, 39, 40, 41, 42, 43, 46, 47], device="cuda", dtype=torch.long)
    scale = d ** -0.5

    y = wm_cuda.indexed_attention_half_kv_fmha_gqa_half_output(q, k, v, indices, scale)

    group = hq // hkv
    q_h = q.half().float()
    k_gqa = k.float().repeat_interleave(group, dim=1)
    v_gqa = v.float().repeat_interleave(group, dim=1)
    scores = torch.einsum("bhtd,bhkd->bhtk", q_h, k_gqa[:, :, indices, :]) * scale
    probs = torch.softmax(scores, dim=-1)
    ref = torch.einsum("bhtn,bhnd->bhtd", probs, v_gqa[:, :, indices, :]).half()

    assert y.dtype == torch.float16
    torch.testing.assert_close(y, ref, rtol=4e-3, atol=4e-3)

def _expand_sparse_block_ids(block_ids, block_size=128):
    offsets = torch.arange(block_size, device=block_ids.device, dtype=block_ids.dtype)
    return (block_ids[..., None] * block_size + offsets).flatten(-2)

def test_sparse_attention_half_kv_fmha_gqa_matches_batched_block_reference(wm_cuda):
    torch.manual_seed(131)
    b, hq, hkv, tq, tk, d = 2, 8, 2, 16, 1024, 64
    q = torch.randn(b, hq, tq, d, device="cuda", dtype=torch.float32)
    k = torch.randn(b, hkv, tk, d, device="cuda", dtype=torch.float32).half().contiguous()
    v = torch.randn(b, hkv, tk, d, device="cuda", dtype=torch.float32).half().contiguous()
    block_ids = torch.tensor([[0, 3, 6], [1, 4, 7]], device="cuda", dtype=torch.int32)
    scale = d ** -0.5

    y = wm_cuda.sparse_attention_half_kv_fmha_gqa(q, k, v, block_ids, scale)

    indices = _expand_sparse_block_ids(block_ids).long()
    gather_index = indices[:, None, :, None].expand(b, hkv, indices.size(1), d)
    k_selected = torch.gather(k.float(), 2, gather_index)
    v_selected = torch.gather(v.float(), 2, gather_index)
    group = hq // hkv
    scores = torch.einsum(
        "bhtd,bhkd->bhtk",
        q.half().float(),
        k_selected.repeat_interleave(group, dim=1),
    ) * scale
    probs = torch.softmax(scores, dim=-1)
    ref = torch.einsum(
        "bhtn,bhnd->bhtd",
        probs,
        v_selected.repeat_interleave(group, dim=1),
    ).half().float()

    torch.testing.assert_close(y, ref, rtol=4e-3, atol=4e-3)

def test_sparse_attention_half_kv_fmha_gqa_matches_compact_bridge(wm_cuda):
    torch.manual_seed(132)
    b, hq, hkv, tq, tk, d = 1, 32, 16, 128, 1024, 64
    q = torch.randn(b, hq, tq, d, device="cuda", dtype=torch.float32)
    k = torch.randn(b, hkv, tk, d, device="cuda", dtype=torch.float32).half().contiguous()
    v = torch.randn(b, hkv, tk, d, device="cuda", dtype=torch.float32).half().contiguous()
    block_ids = torch.tensor([0, 2, 5, 7], device="cuda", dtype=torch.int32)
    indices = _expand_sparse_block_ids(block_ids).long().contiguous()
    scale = d ** -0.5

    y = wm_cuda.sparse_attention_half_kv_fmha_gqa_half_output(q, k, v, block_ids, scale)
    bridge = wm_cuda.indexed_attention_half_kv_fmha_gqa_half_output(q, k, v, indices, scale)

    assert y.dtype == torch.float16
    torch.testing.assert_close(y, bridge, rtol=0, atol=0)

def bench_indexed_attention_half_kv_variants(wm_cuda, warmup=10, iters=40):
    torch.manual_seed(125)
    b, hq, hkv, tq, tk, d = 1, 32, 16, 128, 1024, 64
    q = torch.randn(b, hq, tq, d, device="cuda", dtype=torch.float32)
    k = torch.randn(b, hkv, tk, d, device="cuda", dtype=torch.float32).half().contiguous()
    v = torch.randn(b, hkv, tk, d, device="cuda", dtype=torch.float32).half().contiguous()
    indices = torch.arange(tk, device="cuda", dtype=torch.long)
    block_ids = torch.arange(tk // 128, device="cuda", dtype=torch.int32)
    scale = d ** -0.5

    def time_ms(fn):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        stop.record()
        torch.cuda.synchronize()
        return start.elapsed_time(stop) / iters

    timings = {
        "half_warp": time_ms(lambda: wm_cuda.indexed_attention_half_kv(q, k, v, indices, scale)),
        "half_group_flash": time_ms(lambda: wm_cuda.indexed_attention_half_kv_flash(q, k, v, indices, scale)),
        "half_cutlass_materialized": time_ms(lambda: wm_cuda.indexed_attention_half_kv_cutlass(q, k, v, indices, scale)),
        "half_cutlass_grouped": time_ms(lambda: wm_cuda.indexed_attention_half_kv_cutlass_grouped(q, k, v, indices, scale)),
        "half_fmha_gqa_bridge": time_ms(lambda: wm_cuda.indexed_attention_half_kv_fmha_gqa(q, k, v, indices, scale)),
        "half_fmha_gqa_half_output": time_ms(
            lambda: wm_cuda.indexed_attention_half_kv_fmha_gqa_half_output(q, k, v, indices, scale)
        ),
        "half_sparse_fmha_gqa": time_ms(
            lambda: wm_cuda.sparse_attention_half_kv_fmha_gqa_half_output(q, k, v, block_ids, scale)
        ),
    }
    for name, ms in timings.items():
        print(f"{name}: {ms:.4f} ms")
    return timings

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

def test_kv_cache_upsert_half_matches_half_reference(wm_cuda):
    torch.manual_seed(9)
    b, h, t, d = 1, 3, 4, 64
    ring_length = 32
    capacity = ring_length + t
    cache_k = torch.zeros(b, h, capacity, d, device="cuda", dtype=torch.float16)
    cache_v = torch.zeros(b, h, capacity, d, device="cuda", dtype=torch.float16)
    written = torch.zeros(capacity, device="cuda", dtype=torch.bool)
    written[0:8] = True
    written[ring_length:] = True
    k = torch.randn(b, h, t, d, device="cuda", dtype=torch.float32)
    v = torch.randn(b, h, t, d, device="cuda", dtype=torch.float32)

    ref_k = cache_k.clone()
    ref_v = cache_v.clone()
    ref_written = written.clone()
    ref_mask = _ref_kv_cache_upsert(ref_k, ref_v, ref_written, k.half(), v.half(), frame_idx=6, ring_length=ring_length, pinned_dilation=2, frozen=False)

    mask = wm_cuda.kv_cache_upsert_half(cache_k, cache_v, written, k, v, 6, ring_length, 2, False)
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
    weight = torch.randn(d, c, ph, pw, device="cuda", dtype=torch.float32)
    bias = torch.randn(c, device="cuda", dtype=torch.float32)

    y = wm_cuda.unpatchify(tokens, weight, bias, c, h, w, ph, pw)
    linear_weight = weight.permute(1, 2, 3, 0).reshape(out_dim, d).contiguous()
    ref = F.linear(tokens, linear_weight)
    ref = ref.view(b, hp, wp, c, ph, pw)
    ref = ref + bias.view(1, 1, 1, c, 1, 1)
    ref = ref.permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w).contiguous()
    torch.testing.assert_close(y, ref, rtol=2e-5, atol=2e-5)
