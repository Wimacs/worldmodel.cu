"""CUDA prototypes that intentionally have no production operator API."""

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
    include_paths = [str(ROOT), str(cutlass_include)]
    cutlass_fmha_include = find_cutlass_example_include(cutlass_include)
    if cutlass_fmha_include is not None:
        include_paths.append(str(cutlass_fmha_include))

    build_dir = ROOT / ".torch_extensions" / "experimental"
    build_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name="worldmodel_cuda_experimental_ext",
        sources=[str(ROOT / "tools" / "tests" / "cuda_experimental_torch.cu")],
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


def test_cutlass_fmha_bmhk_fp16_matches_torch_half_reference(wm_cuda):
    torch.manual_seed(126)
    for b, m, n, h, d in [(1, 64, 64, 2, 64), (1, 128, 256, 8, 64)]:
        q = torch.randn(b, m, h, d, device="cuda", dtype=torch.float32).half().contiguous()
        k = torch.randn(b, n, h, d, device="cuda", dtype=torch.float32).half().contiguous()
        v = torch.randn(b, n, h, d, device="cuda", dtype=torch.float32).half().contiguous()
        scale = d ** -0.5

        y = wm_cuda.cutlass_fmha_bmhk_fp16(q, k, v, scale)
        scores = torch.einsum("bmhd,bnhd->bhmn", q.float(), k.float()) * scale
        probs = torch.softmax(scores, dim=-1)
        ref = torch.einsum("bhmn,bnhd->bmhd", probs, v.float()).half()
        torch.testing.assert_close(y.float(), ref.float(), rtol=3e-3, atol=3e-3)


def test_taehv_conv3x3_cutlass_implicit_nhwc_matches_torch_same_padding(wm_cuda):
    torch.manual_seed(124)
    n, cin, cout, h, w = 2, 5, 7, 8, 11
    x = torch.randn(n, cin, h, w, device="cuda", dtype=torch.float32)
    weight = torch.randn(cout, cin, 3, 3, device="cuda", dtype=torch.float32) * 0.2
    bias = torch.randn(cout, device="cuda", dtype=torch.float32) * 0.1

    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    weight_krsc = weight.permute(0, 2, 3, 1).contiguous()
    y = wm_cuda.taehv_conv3x3_cutlass_implicit_nhwc(x_nhwc, weight_krsc, bias)
    ref = F.conv2d(x, weight, bias=bias, padding=1)
    torch.testing.assert_close(
        y.permute(0, 3, 1, 2).contiguous(), ref, rtol=2e-5, atol=2e-5
    )


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
    hidden = wm_cuda.taehv_conv3x3_cutlass_implicit_nhwc(
        x_nhwc, weight0_krsc, bias0
    )
    y = wm_cuda.taehv_conv3x3_cutlass_implicit_nhwc(
        torch.relu(hidden), weight1_krsc, bias1
    )
    ref = F.conv2d(
        F.relu(F.conv2d(x, weight0, bias=bias0, padding=1)),
        weight1,
        bias=bias1,
        padding=1,
    )
    torch.testing.assert_close(
        y.permute(0, 3, 1, 2).contiguous(), ref, rtol=2e-5, atol=2e-5
    )
