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


def load_wm_cuda():
    if not torch.cuda.is_available() or CUDA_HOME is None:
        pytest.skip("CUDA and nvcc are required for worldmodel.cu parity tests")

    build_dir = ROOT / ".torch_extensions"
    build_dir.mkdir(exist_ok=True)
    return load(
        name="worldmodel_cuda_ext",
        sources=[str(ROOT / "worldmodel_kernels.cu")],
        build_directory=str(build_dir),
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


if __name__ == "__main__":
    ext = load_wm_cuda()
    for test in (
        test_silu_matches_torch,
        test_rms_norm_matches_torch,
        test_ada_rms_norm_matches_world_formula,
        test_ortho_rope_matches_world_formula,
        test_qkv_rms_rope_matches_split_reference,
        test_masked_attention_matches_torch_gqa_written_mask,
    ):
        test(ext)
        print(f"{test.__name__}: ok")
