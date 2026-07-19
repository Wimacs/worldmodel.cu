import math

import torch


def _symmetric_quantize_rows(x: torch.Tensor):
    """Reference policy used by the standalone CUTLASS W8A8 probe."""
    assert x.ndim == 2
    amax = x.abs().amax(dim=1)
    scale = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
    quantized = torch.round(x / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return quantized, scale


def _w8a8_linear_reference(x: torch.Tensor, weight: torch.Tensor):
    qx, x_scale = _symmetric_quantize_rows(x)
    qw, weight_scale = _symmetric_quantize_rows(weight)
    accumulators = qx.to(torch.int32) @ qw.to(torch.int32).T
    output = accumulators.float() * x_scale[:, None] * weight_scale[None, :]
    return output, qx, qw, x_scale, weight_scale, accumulators


def test_w8a8_zero_rows_have_unit_scale_and_zero_codes():
    x = torch.zeros(3, 128, dtype=torch.float32)
    x[1] = torch.linspace(-2.0, 2.0, 128)
    quantized, scale = _symmetric_quantize_rows(x)

    assert scale[0].item() == 1.0
    assert scale[2].item() == 1.0
    assert torch.count_nonzero(quantized[0]).item() == 0
    assert torch.count_nonzero(quantized[2]).item() == 0
    assert quantized.min().item() >= -127
    assert quantized.max().item() <= 127


def test_w8a8_outer_scale_dequantization_matches_elementwise_formula():
    torch.manual_seed(0x57A8)
    x = torch.randn(17, 128, dtype=torch.float32)
    weight = torch.randn(33, 128, dtype=torch.float32)
    output, qx, qw, x_scale, weight_scale, accumulators = _w8a8_linear_reference(x, weight)

    for row, col in ((0, 0), (3, 7), (16, 32)):
        exact_accumulator = sum(
            int(qx[row, inner]) * int(qw[col, inner]) for inner in range(x.shape[1])
        )
        assert exact_accumulator == int(accumulators[row, col])
        expected = float(exact_accumulator) * float(x_scale[row]) * float(weight_scale[col])
        assert math.isclose(float(output[row, col]), expected, rel_tol=1e-6, abs_tol=1e-6)


def test_w8a8_reference_quality_with_row_scale_variation_and_outliers():
    torch.manual_seed(1234)
    m, k, n = 32, 256, 64
    x = torch.randn(m, k, dtype=torch.float32)
    weight = torch.randn(n, k, dtype=torch.float32)
    x *= torch.exp2(torch.linspace(-1.5, 1.5, m))[:, None]
    weight *= torch.exp2(torch.linspace(-1.0, 1.0, n))[:, None]
    x[1, k // 3] *= 10.0
    weight[1, k // 5] *= -10.0

    output, *_ = _w8a8_linear_reference(x, weight)
    reference = x @ weight.T
    error = output - reference
    sqnr_db = 10.0 * torch.log10(reference.square().sum() / error.square().sum())

    assert torch.isfinite(output).all()
    assert float(sqnr_db) >= 30.0


def test_waypoint_maximum_k_has_int32_accumulator_headroom():
    # Symmetric s8 uses [-127, 127]. Waypoint's largest linear reduction is K=8192.
    worst_case = 127 * 127 * 8192
    assert worst_case == 132_128_768
    assert worst_case < torch.iinfo(torch.int32).max


if __name__ == "__main__":
    tests = [
        test_w8a8_zero_rows_have_unit_scale_and_zero_codes,
        test_w8a8_outer_scale_dequantization_matches_elementwise_formula,
        test_w8a8_reference_quality_with_row_scale_variation_and_outliers,
        test_waypoint_maximum_k_has_int32_accumulator_headroom,
    ]
    for test in tests:
        test()
    print(f"W8A8 quantization reference tests passed: {len(tests)}")
