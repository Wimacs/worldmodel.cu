import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors import safe_open

try:
    import pytest
except ModuleNotFoundError:
    class _PyTestShim:
        def skip(self, msg):
            raise RuntimeError(msg)

    pytest = _PyTestShim()


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT.parent / "Waypoint-1.5-1B"
WEIGHTS_PATH = MODEL_DIR / "transformer" / "diffusion_pytorch_model.safetensors"


def _load_config():
    with open(MODEL_DIR / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("mlp_ratio", 4)
    cfg.setdefault("n_kv_heads", cfg["n_heads"])
    return cfg


def _build_executable():
    subprocess.run(["cmake", "-S", str(ROOT), "-B", str(ROOT / "build"), "-DCMAKE_BUILD_TYPE=Release"], check=True)
    subprocess.run(["cmake", "--build", str(ROOT / "build"), "-j"], check=True)
    exe = ROOT / "build" / "worldmodel_cuda"
    if not exe.exists():
        raise RuntimeError(f"missing executable: {exe}")
    return exe


def _lcg_latent(shape, seed):
    state = seed or 1
    out = np.empty(int(np.prod(shape)), dtype=np.float32)
    for i in range(out.size):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        u = ((state >> 8) & 0x00FFFFFF) / 16777216.0
        out[i] = 2.0 * u - 1.0
    return out.reshape(shape)


def _read_dump(prefix, name, shape):
    path = Path(f"{prefix}.{name}.f32")
    arr = np.fromfile(path, dtype=np.float32)
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise AssertionError(f"{path.name}: expected {expected} floats, got {arr.size}")
    return torch.from_numpy(arr.reshape(shape).copy())


def _load_required_tensors(names, device):
    out = {}
    with safe_open(str(WEIGHTS_PATH), framework="pt", device="cpu") as f:
        for name in names:
            out[name] = f.get_tensor(name).to(device=device, dtype=torch.float32).contiguous()
    return out


def _noise_embedding(sigma, device):
    half = 256
    freq = torch.logspace(0, -1, steps=half, base=10000.0, dtype=torch.float32, device=device)
    phase = torch.tensor([sigma * 1000.0], dtype=torch.float32, device=device)[:, None] * freq[None, :]
    return torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1) * math.sqrt(2.0)


def _world_rope_tables(d_head, height, width, device):
    d_xy = d_head // 8
    d_t = d_head // 4
    max_freq = min(height, width) * 0.8
    n = (d_xy + 1) // 2
    xy = torch.linspace(1.0, max_freq / 2, n, dtype=torch.float32, device=device) * math.pi
    xy = xy.repeat_interleave(2)[:d_xy].contiguous()
    inv_t = 1.0 / (10000.0 ** (torch.arange(0, d_t, 2, dtype=torch.float32, device=device) / d_t))
    return xy, inv_t.repeat_interleave(2).contiguous()


def _ref_ortho_rope(x, x_pos, y_pos, t_pos, xy, inv_t, width, height):
    d_head = x.shape[-1]
    d_xy = d_head // 8
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
    assert freqs.shape[-1] == d_head // 2
    assert xy.numel() == d_xy
    cos = freqs.cos()[None, None]
    sin = freqs.sin()[None, None]
    x0, x1 = x.unfold(-1, 2, 2).unbind(-1)
    return torch.cat((x0 * cos - x1 * sin, x1 * cos + x0 * sin), dim=-1)


def _assert_close(name, actual_cpu, expected, rtol=3.0e-4, atol=3.0e-4):
    actual = actual_cpu.to(device=expected.device, dtype=torch.float32)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=name)
    diff = (actual - expected).abs().max().item()
    print(f"{name}: ok max_abs={diff:.6g}")


def test_standalone_layer0_probe_matches_pytorch():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if not WEIGHTS_PATH.exists():
        pytest.skip(f"missing Waypoint weights: {WEIGHTS_PATH}")

    cfg = _load_config()
    exe = _build_executable()
    seed = 1234
    sigma = 1.0
    device = torch.device("cuda")

    with tempfile.TemporaryDirectory(prefix="world_layer0_probe_") as td:
        prefix = str(Path(td) / "dump")
        subprocess.run(
            [
                str(exe),
                "--model-dir",
                str(MODEL_DIR),
                "--seed",
                str(seed),
                "--sigma",
                str(sigma),
                "--dump-prefix",
                prefix,
            ],
            check=True,
            cwd=str(ROOT),
        )

        c = cfg["channels"]
        d_model = cfg["d_model"]
        n_heads = cfg["n_heads"]
        n_kv_heads = cfg["n_kv_heads"]
        ph, pw = int(cfg["patch"][0]), int(cfg["patch"][1])
        height, width = cfg["height"], cfg["width"]
        h, w = height * ph, width * pw
        tpf = height * width
        d_head = d_model // n_heads
        kv_dim = n_kv_heads * d_head
        hidden = d_model * cfg["mlp_ratio"]

        names = [
            "patchify.weight",
            "denoise_step_emb.mlp.fc1.weight",
            "denoise_step_emb.mlp.fc2.weight",
            "transformer.blocks.0.mlp_cond_head.bias_in",
            "transformer.blocks.0.attn_cond_head.cond_proj.0.weight",
            "transformer.blocks.0.attn_cond_head.cond_proj.1.weight",
            "transformer.blocks.0.attn_cond_head.cond_proj.2.weight",
            "transformer.blocks.0.attn.q_proj.weight",
            "transformer.blocks.0.attn.k_proj.weight",
            "transformer.blocks.0.attn.v_proj.weight",
            "transformer.blocks.0.attn.out_proj.weight",
            "transformer.blocks.0.mlp_cond_head.cond_proj.0.weight",
            "transformer.blocks.0.mlp_cond_head.cond_proj.1.weight",
            "transformer.blocks.0.mlp_cond_head.cond_proj.2.weight",
            "transformer.blocks.0.ctrl_mlpfusion.fc1_x.weight",
            "transformer.blocks.0.ctrl_mlpfusion.fc2.weight",
            "transformer.blocks.0.dit_mlp.fc1.weight",
            "transformer.blocks.0.dit_mlp.fc2.weight",
        ]
        state = _load_required_tensors(names, device)

        latent = torch.from_numpy(_lcg_latent((1, c, h, w), seed)).to(device)
        noise = _noise_embedding(sigma, device)
        cond = F.linear(
            F.silu(F.linear(noise, state["denoise_step_emb.mlp.fc1.weight"])),
            state["denoise_step_emb.mlp.fc2.weight"],
        ).reshape(d_model)
        cond_act = F.silu(cond + state["transformer.blocks.0.mlp_cond_head.bias_in"])
        s0 = F.linear(cond_act, state["transformer.blocks.0.attn_cond_head.cond_proj.0.weight"])
        b0 = F.linear(cond_act, state["transformer.blocks.0.attn_cond_head.cond_proj.1.weight"])
        g0 = F.linear(cond_act, state["transformer.blocks.0.attn_cond_head.cond_proj.2.weight"])
        s1 = F.linear(cond_act, state["transformer.blocks.0.mlp_cond_head.cond_proj.0.weight"])
        b1 = F.linear(cond_act, state["transformer.blocks.0.mlp_cond_head.cond_proj.1.weight"])
        g1 = F.linear(cond_act, state["transformer.blocks.0.mlp_cond_head.cond_proj.2.weight"])

        tokens = F.conv2d(latent, state["patchify.weight"], bias=None, stride=(ph, pw))
        tokens = tokens.permute(0, 2, 3, 1).reshape(1, tpf, d_model).contiguous()
        norm = F.rms_norm(tokens, (d_model,), eps=1.0e-6) * (1.0 + s0) + b0
        q_raw = F.linear(norm, state["transformer.blocks.0.attn.q_proj.weight"]).reshape(tpf, d_model)
        k_raw = F.linear(norm, state["transformer.blocks.0.attn.k_proj.weight"]).reshape(tpf, kv_dim)
        v_raw = F.linear(norm, state["transformer.blocks.0.attn.v_proj.weight"]).reshape(tpf, kv_dim)

        idx = torch.arange(tpf, device=device, dtype=torch.long)
        y_pos = idx.div(width, rounding_mode="floor").contiguous()
        x_pos = idx.remainder(width).contiguous()
        t_pos = torch.zeros_like(idx)
        xy, inv_t = _world_rope_tables(d_head, height, width, device)
        q = q_raw.view(1, tpf, n_heads, d_head).permute(0, 2, 1, 3).contiguous()
        k = k_raw.view(1, tpf, n_kv_heads, d_head).permute(0, 2, 1, 3).contiguous()
        v = v_raw.view(tpf, n_kv_heads, d_head).permute(1, 0, 2).contiguous()
        q = _ref_ortho_rope(F.rms_norm(q, (d_head,), eps=1.0e-6), x_pos, y_pos, t_pos, xy, inv_t, width, height)[0]
        k = _ref_ortho_rope(F.rms_norm(k, (d_head,), eps=1.0e-6), x_pos, y_pos, t_pos, xy, inv_t, width, height)[0]
        group = n_heads // n_kv_heads
        k_gqa = k.repeat_interleave(group, dim=0)
        v_gqa = v.repeat_interleave(group, dim=0)
        scores = torch.einsum("htd,hkd->htk", q, k_gqa) * (d_head ** -0.5)
        probs = torch.softmax(scores, dim=-1)
        attn_heads = torch.einsum("htk,hkd->htd", probs, v_gqa)
        attn = attn_heads.permute(1, 0, 2).reshape(tpf, d_model).contiguous()
        attn_out = F.linear(attn, state["transformer.blocks.0.attn.out_proj.weight"])
        tokens_after_attn = tokens.reshape(tpf, d_model) + attn_out * g0
        ctrl_hidden = F.linear(
            F.rms_norm(tokens_after_attn, (d_model,), eps=1.0e-6),
            state["transformer.blocks.0.ctrl_mlpfusion.fc1_x.weight"],
        )
        ctrl_out = F.linear(F.silu(ctrl_hidden), state["transformer.blocks.0.ctrl_mlpfusion.fc2.weight"])
        tokens_after_ctrl = tokens_after_attn + ctrl_out
        mlp_in = F.rms_norm(tokens_after_ctrl, (d_model,), eps=1.0e-6) * (1.0 + s1) + b1
        mlp_out = F.linear(
            F.silu(F.linear(mlp_in, state["transformer.blocks.0.dit_mlp.fc1.weight"])),
            state["transformer.blocks.0.dit_mlp.fc2.weight"],
        )
        tokens_after_mlp = tokens_after_ctrl + mlp_out * g1

        assert hidden == state["denoise_step_emb.mlp.fc1.weight"].shape[0]
        _assert_close("latent", _read_dump(prefix, "latent", (1, c, h, w)), latent)
        _assert_close("tokens", _read_dump(prefix, "tokens", (1, tpf, d_model)), tokens)
        _assert_close("cond", _read_dump(prefix, "cond", (d_model,)), cond)
        _assert_close("s0", _read_dump(prefix, "s0", (d_model,)), s0)
        _assert_close("b0", _read_dump(prefix, "b0", (d_model,)), b0)
        _assert_close("g0", _read_dump(prefix, "g0", (d_model,)), g0)
        _assert_close("s1", _read_dump(prefix, "s1", (d_model,)), s1)
        _assert_close("b1", _read_dump(prefix, "b1", (d_model,)), b1)
        _assert_close("g1", _read_dump(prefix, "g1", (d_model,)), g1)
        _assert_close("norm", _read_dump(prefix, "norm", (1, tpf, d_model)), norm)
        _assert_close("q_raw", _read_dump(prefix, "q_raw", (tpf, d_model)), q_raw)
        _assert_close("k_raw", _read_dump(prefix, "k_raw", (tpf, kv_dim)), k_raw)
        _assert_close("v_raw", _read_dump(prefix, "v_raw", (tpf, kv_dim)), v_raw)
        _assert_close("q", _read_dump(prefix, "q", (n_heads, tpf, d_head)), q, rtol=5.0e-4, atol=5.0e-4)
        _assert_close("k", _read_dump(prefix, "k", (n_kv_heads, tpf, d_head)), k, rtol=5.0e-4, atol=5.0e-4)
        _assert_close("v", _read_dump(prefix, "v", (n_kv_heads, tpf, d_head)), v)
        _assert_close("attn", _read_dump(prefix, "attn", (tpf, d_model)), attn, rtol=5.0e-4, atol=5.0e-4)
        _assert_close("attn_out", _read_dump(prefix, "attn_out", (tpf, d_model)), attn_out, rtol=5.0e-4, atol=5.0e-4)
        _assert_close(
            "tokens_after_attn",
            _read_dump(prefix, "tokens_after_attn", (tpf, d_model)),
            tokens_after_attn,
            rtol=6.0e-4,
            atol=6.0e-4,
        )
        _assert_close("ctrl_out", _read_dump(prefix, "ctrl_out", (tpf, d_model)), ctrl_out, rtol=6.0e-4, atol=6.0e-4)
        _assert_close(
            "tokens_after_ctrl",
            _read_dump(prefix, "tokens_after_ctrl", (tpf, d_model)),
            tokens_after_ctrl,
            rtol=7.0e-4,
            atol=7.0e-4,
        )
        _assert_close("mlp_in", _read_dump(prefix, "mlp_in", (tpf, d_model)), mlp_in, rtol=8.0e-4, atol=8.0e-4)
        _assert_close("mlp_out", _read_dump(prefix, "mlp_out", (tpf, d_model)), mlp_out, rtol=8.0e-4, atol=8.0e-4)
        _assert_close(
            "tokens_after_mlp",
            _read_dump(prefix, "tokens_after_mlp", (tpf, d_model)),
            tokens_after_mlp,
            rtol=1.0e-3,
            atol=1.0e-3,
        )


if __name__ == "__main__":
    test_standalone_layer0_probe_matches_pytorch()
