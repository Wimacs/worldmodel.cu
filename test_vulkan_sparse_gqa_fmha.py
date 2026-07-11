import os
from pathlib import Path
import subprocess
import tempfile

import torch


B = 1
HQ = 4
HKV = 2
TQ = 32
TK = 384
D = 64
BLOCK_SIZE = 128
SCALE = 1.0 / 8.0


def read_tensor(path, dtype, shape):
    data = Path(path).read_bytes()
    return torch.frombuffer(bytearray(data), dtype=dtype).clone().reshape(shape)


def torch_sparse_gqa_reference(q, k, v, block_ids):
    group = HQ // HKV
    out = torch.empty((B, TQ, HQ, D), dtype=torch.float32)
    q = q.half().float()
    k = k.float()
    v = v.float()

    for b in range(B):
        for hk in range(HKV):
            folded_q = torch.cat(
                [q[b, hk * group + g] for g in range(group)], dim=0
            )
            merged = torch.zeros((group * TQ, D), dtype=torch.float32)
            merged_m = torch.full((group * TQ,), -torch.inf, dtype=torch.float32)
            merged_l = torch.zeros((group * TQ,), dtype=torch.float32)

            for physical_block in block_ids.tolist():
                start = int(physical_block) * BLOCK_SIZE
                kb = k[b, hk, start : start + BLOCK_SIZE]
                vb = v[b, hk, start : start + BLOCK_SIZE]
                scores = (folded_q @ kb.T) * SCALE
                block_m = scores.max(dim=1).values
                probs = torch.exp(scores - block_m[:, None])
                block_l = probs.sum(dim=1)
                block_acc = probs.half().float() @ vb

                new_m = torch.maximum(merged_m, block_m)
                alpha = torch.exp(merged_m - new_m)
                beta = torch.exp(block_m - new_m)
                merged = merged * alpha[:, None] + block_acc * beta[:, None]
                merged_l = merged_l * alpha + block_l * beta
                merged_m = new_m

            merged /= merged_l[:, None]
            for g in range(group):
                hq = hk * group + g
                out[b, :, hq] = merged[g * TQ : (g + 1) * TQ]
    return out


def test_vulkan_sparse_gqa_fmha_matches_torch():
    root = Path(__file__).resolve().parent
    probe = Path(
        os.environ.get(
            "WORLD_VULKAN_PROBE_BIN",
            root / "build-vulkan" / "worldmodel_vulkan_probe",
        )
    )
    if not probe.exists():
        raise FileNotFoundError(f"Vulkan probe executable not found: {probe}")

    with tempfile.TemporaryDirectory() as temp_dir:
        prefix = Path(temp_dir) / "sparse_gqa"
        env = os.environ.copy()
        env["WORLD_VULKAN_SPARSE_PROBE_DUMP"] = str(prefix)
        subprocess.run(
            [str(probe), "--sparse-fmha"],
            cwd=root,
            env=env,
            check=True,
        )

        q = read_tensor(f"{prefix}.q_f32.bin", torch.float32, (B, HQ, TQ, D))
        k = read_tensor(f"{prefix}.k_f16.bin", torch.float16, (B, HKV, TK, D))
        v = read_tensor(f"{prefix}.v_f16.bin", torch.float16, (B, HKV, TK, D))
        block_ids = read_tensor(f"{prefix}.blocks_u32.bin", torch.uint32, (2,))
        got = read_tensor(f"{prefix}.out_f32.bin", torch.float32, (B, TQ, HQ, D))
        got_f16 = read_tensor(
            f"{prefix}.out_f16.bin", torch.float16, (B, TQ, HQ, D)
        ).float()
        ref = torch_sparse_gqa_reference(q, k, v, block_ids)

        diff = (got - ref).abs()
        assert diff.max().item() <= 2.0e-3
        assert diff.mean().item() <= 2.0e-4
        diff_f16 = (got_f16 - ref.half().float()).abs()
        assert diff_f16.max().item() <= 1.0e-3
        assert diff_f16.mean().item() <= 1.0e-4


if __name__ == "__main__":
    test_vulkan_sparse_gqa_fmha_matches_torch()
    print("Vulkan sparse GQA FMHA PyTorch parity: ok")
