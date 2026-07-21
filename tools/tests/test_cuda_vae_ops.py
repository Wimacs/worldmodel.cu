"""PyTorch parity tests for the production VAE CUDA operator API."""

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


def load_wm_cuda_vae():
    if not torch.cuda.is_available() or CUDA_HOME is None:
        pytest.skip("CUDA and nvcc are required for worldmodel.c VAE parity tests")
    cutlass_include = find_cutlass_include()
    if cutlass_include is None:
        pytest.skip("CUTLASS include directory is required for worldmodel.c VAE parity tests")

    build_dir = ROOT / ".torch_extensions" / "vae_ops"
    build_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name="worldmodel_cuda_vae_ops_ext",
        sources=[
            str(ROOT / "tools" / "tests" / "cuda_vae_ops_torch.cu"),
            str(ROOT / "src" / "world_cuda_vae_ops.cu"),
        ],
        build_directory=str(build_dir),
        extra_include_paths=[str(ROOT / "src"), str(cutlass_include)],
        extra_cuda_cflags=["-O3", "--use_fast_math", "--default-stream=per-thread"],
        verbose=bool(os.environ.get("WORLD_CU_VERBOSE_BUILD")),
    )


@pytest.fixture(scope="session")
def wm_cuda_vae():
    return load_wm_cuda_vae()


def test_taehv_conv_direct_nchw_f32_matches_torch_same_padding(wm_cuda_vae):
    torch.manual_seed(12)
    n, cin, cout, h, w = 3, 5, 7, 9, 11
    x = torch.randn(n, cin, h, w, device="cuda", dtype=torch.float32)
    weight = torch.randn(cout, cin, 3, 3, device="cuda", dtype=torch.float32) * 0.2
    bias = torch.randn(cout, device="cuda", dtype=torch.float32) * 0.1

    y = wm_cuda_vae.conv_direct_nchw_f32(x, weight, bias)
    ref = F.conv2d(x, weight, bias=bias, padding=1)
    torch.testing.assert_close(y, ref, rtol=2e-5, atol=2e-5)


def test_taehv_conv1x1_gemm_nchw_f32_matches_torch(wm_cuda_vae):
    torch.manual_seed(121)
    n, cin, cout, h, w = 4, 6, 9, 8, 13
    x = torch.randn(n, cin, h, w, device="cuda", dtype=torch.float32)
    weight = torch.randn(cout, cin, 1, 1, device="cuda", dtype=torch.float32) * 0.2
    bias = torch.randn(cout, device="cuda", dtype=torch.float32) * 0.1

    y = wm_cuda_vae.conv1x1_gemm_nchw_f32(x, weight, bias)
    ref = F.conv2d(x, weight, bias=bias)
    torch.testing.assert_close(y, ref, rtol=2e-5, atol=2e-5)


def test_taehv_conv3x3_gemm_nchw_f32_matches_torch_same_padding(wm_cuda_vae):
    torch.manual_seed(122)
    n, cin, cout, h, w = 3, 5, 8, 7, 10
    x = torch.randn(n, cin, h, w, device="cuda", dtype=torch.float32)
    weight = torch.randn(cout, cin, 3, 3, device="cuda", dtype=torch.float32) * 0.2
    bias = torch.randn(cout, device="cuda", dtype=torch.float32) * 0.1

    y = wm_cuda_vae.conv3x3_gemm_nchw_f32(x, weight, bias)
    ref = F.conv2d(x, weight, bias=bias, padding=1)
    torch.testing.assert_close(y, ref, rtol=2e-5, atol=2e-5)


def test_taehv_conv3x3_gemm_batched_nchw_f32_matches_torch_same_padding(wm_cuda_vae):
    torch.manual_seed(123)
    n, cin, cout, h, w = 4, 5, 8, 7, 10
    x = torch.randn(n, cin, h, w, device="cuda", dtype=torch.float32)
    weight = torch.randn(cout, cin, 3, 3, device="cuda", dtype=torch.float32) * 0.2
    bias = torch.randn(cout, device="cuda", dtype=torch.float32) * 0.1

    y = wm_cuda_vae.conv3x3_gemm_batched_nchw_f32(x, weight, bias, 97)
    ref = F.conv2d(x, weight, bias=bias, padding=1)
    torch.testing.assert_close(y, ref, rtol=2e-5, atol=2e-5)


def test_taehv_conv3x3_nhwc_f16_matches_reference(wm_cuda_vae):
    torch.manual_seed(126)
    n, cin, cout, h, w = 2, 8, 12, 9, 10
    x = (torch.randn(n, cin, h, w, device="cuda", dtype=torch.float32) * 0.5).half()
    weight = (torch.randn(cout, cin, 3, 3, device="cuda", dtype=torch.float32) * 0.1).half()
    bias = (torch.randn(cout, device="cuda", dtype=torch.float32) * 0.05).half()

    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    weight_krsc = weight.permute(0, 2, 3, 1).contiguous()
    y = wm_cuda_vae.conv_nhwc_f16(x_nhwc, weight_krsc, bias)
    ref_conv = F.conv2d(x.float(), weight.float(), bias=None, padding=1).half()
    ref = (ref_conv + bias.view(1, -1, 1, 1)).half()
    torch.testing.assert_close(
        y.permute(0, 3, 1, 2).float(), ref.float(), rtol=2e-2, atol=2e-2
    )


def test_taehv_concat_past_nchw_f32_matches_reference(wm_cuda_vae):
    torch.manual_seed(13)
    x = torch.randn(5, 4, 6, 7, device="cuda", dtype=torch.float32)
    y = wm_cuda_vae.concat_past_nchw_f32(x)
    past = torch.cat((torch.zeros_like(x[:1]), x[:-1]), dim=0)
    ref = torch.cat((x, past), dim=1).contiguous()
    torch.testing.assert_close(y, ref, rtol=0, atol=0)


def test_taehv_conv_stride2_nchw_f32_matches_reference(wm_cuda_vae):
    torch.manual_seed(127)
    for h, w in ((12, 14), (11, 13)):
        x = torch.randn(2, 5, h, w, device="cuda", dtype=torch.float32) * 0.25
        weight = torch.randn(7, 5, 3, 3, device="cuda", dtype=torch.float32) * 0.1
        bias = torch.randn(7, device="cuda", dtype=torch.float32) * 0.05
        y = wm_cuda_vae.conv_stride2_nchw_f32(x, weight, bias)
        ref = F.conv2d(x, weight, bias=bias, stride=2, padding=1)
        torch.testing.assert_close(y, ref, rtol=2e-5, atol=2e-5)


def test_taehv_upsample2_nchw_f32_matches_nearest(wm_cuda_vae):
    torch.manual_seed(14)
    x = torch.randn(3, 5, 4, 7, device="cuda", dtype=torch.float32)
    y = wm_cuda_vae.upsample2_nchw_f32(x)
    ref = F.interpolate(x, scale_factor=2, mode="nearest")
    torch.testing.assert_close(y, ref, rtol=0, atol=0)


def test_taehv_tgrow_reshape_nchw_f32_matches_torch_view(wm_cuda_vae):
    torch.manual_seed(15)
    n, c, h, w, stride = 4, 6, 5, 7, 2
    x = torch.randn(n, c * stride, h, w, device="cuda", dtype=torch.float32)
    y = wm_cuda_vae.tgrow_reshape_nchw_f32(x, stride)
    ref = x.reshape(-1, c, h, w).contiguous()
    torch.testing.assert_close(y, ref, rtol=0, atol=0)
