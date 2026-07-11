import os
from pathlib import Path
import subprocess
import tempfile

import torch


M = 64
N = 128
K = 128


def read_tensor(path, dtype, shape):
    data = Path(path).read_bytes()
    return torch.frombuffer(bytearray(data), dtype=dtype).clone().reshape(shape)


def test_vulkan_cta_gemm_matches_torch():
    root = Path(__file__).resolve().parent
    autotune = Path(
        os.environ.get(
            "WORLD_VULKAN_GEMM_AUTOTUNE_BIN",
            root / "build-vulkan" / "worldmodel_vulkan_gemm_autotune",
        )
    )
    if not autotune.exists():
        raise FileNotFoundError(f"Vulkan GEMM autotuner not found: {autotune}")

    with tempfile.TemporaryDirectory() as temp_dir:
        prefix = Path(temp_dir) / "cta"
        env = os.environ.copy()
        env["WORLD_VULKAN_GEMM_PROBE_DUMP"] = str(prefix)
        subprocess.run(
            [
                str(autotune),
                "--shape",
                str(M),
                str(N),
                str(K),
                "--input",
                "f16",
                "--epilogue",
                "all",
                "--out",
                str(Path(temp_dir) / "autotune.cache"),
            ],
            cwd=root,
            env=env,
            check=True,
        )

        outputs = sorted(Path(temp_dir).glob("cta.gemm_cta_*.out_*.bin"))
        assert outputs, "CTA probe produced no output dumps"
        saw_none = False
        saw_silu = False
        saw_gated = False
        saw_split4 = False
        saw_ping_pong = False

        for output_path in outputs:
            output_name = output_path.name
            if output_name.endswith(".out_f16.bin"):
                stem = str(output_path)[: -len(".out_f16.bin")]
                got = read_tensor(output_path, torch.float16, (M, N)).float()
            else:
                stem = str(output_path)[: -len(".out_f32.bin")]
                got = read_tensor(output_path, torch.float32, (M, N))

            x = read_tensor(f"{stem}.x_f16.bin", torch.float16, (M, K)).float()
            weight = read_tensor(
                f"{stem}.w_nk_f16.bin", torch.float16, (N, K)
            ).float()
            ref = x @ weight.T

            if "_silu" in output_name:
                ref = torch.nn.functional.silu(ref).half().float()
                saw_silu = True
            elif "_gated" in output_name or "_splitk" in output_name:
                residual = read_tensor(
                    f"{stem}.residual_f32.bin", torch.float32, (M, N)
                )
                gate = read_tensor(f"{stem}.gate_f32.bin", torch.float32, (N,))
                ref = residual + ref * gate
                saw_gated = True
                saw_split4 |= ".split4." in output_name
            else:
                saw_none = True

            saw_ping_pong |= "_p2.comp" in output_name
            diff = (got - ref).abs()
            assert diff.max().item() <= 5.0e-3, output_name
            assert diff.mean().item() <= 5.0e-4, output_name

        assert saw_none and saw_silu and saw_gated
        assert saw_split4 and saw_ping_pong


def test_vulkan_half_boundary_ops_match_torch():
    root = Path(__file__).resolve().parent
    probe = Path(
        os.environ.get(
            "WORLD_VULKAN_PROBE_BIN",
            root / "build-vulkan" / "worldmodel_vulkan_probe",
        )
    )
    if not probe.exists():
        raise FileNotFoundError(f"Vulkan probe executable not found: {probe}")

    rows = 4
    d_model = 256
    ada_n = 2
    eps = 1.0e-6
    with tempfile.TemporaryDirectory() as temp_dir:
        prefix = Path(temp_dir) / "half_boundary"
        env = os.environ.copy()
        env["WORLD_VULKAN_HALF_BOUNDARY_PROBE_DUMP"] = str(prefix)
        subprocess.run(
            [str(probe), "--half-boundary"],
            cwd=root,
            env=env,
            check=True,
        )

        x = read_tensor(f"{prefix}.x_f32.bin", torch.float32, (rows, d_model))
        weight = read_tensor(
            f"{prefix}.weight_f32.bin", torch.float32, (d_model,)
        )
        scale = read_tensor(
            f"{prefix}.scale_f32.bin", torch.float32, (ada_n, d_model)
        )
        bias = read_tensor(
            f"{prefix}.bias_f32.bin", torch.float32, (ada_n, d_model)
        )
        channel = read_tensor(
            f"{prefix}.channel_f32.bin", torch.float32, (d_model,)
        )
        inv = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)

        rms_ref = (x * inv * weight).half().float()
        ada_index = torch.arange(rows) // (rows // ada_n)
        ada_ref = (
            x * inv * (1.0 + scale[ada_index]) + bias[ada_index]
        ).half().float()
        add_ref = torch.nn.functional.silu(x + channel).half().float()

        rms_got = read_tensor(
            f"{prefix}.rms_out_f16.bin", torch.float16, (rows, d_model)
        ).float()
        ada_got = read_tensor(
            f"{prefix}.ada_out_f16.bin", torch.float16, (rows, d_model)
        ).float()
        add_got = read_tensor(
            f"{prefix}.add_out_f16.bin", torch.float16, (rows, d_model)
        ).float()

        for got, ref in ((rms_got, rms_ref), (ada_got, ada_ref), (add_got, add_ref)):
            diff = (got - ref).abs()
            assert diff.max().item() <= 1.0e-3
            assert diff.mean().item() <= 1.0e-4


if __name__ == "__main__":
    test_vulkan_cta_gemm_matches_torch()
    test_vulkan_half_boundary_ops_match_torch()
    print("Vulkan CTA GEMM and FP16 boundary PyTorch parity: ok")
