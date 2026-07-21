import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors import safe_open
from safetensors.torch import load_file

try:
    import pytest
except ModuleNotFoundError:
    class _PyTestShim:
        def skip(self, msg):
            raise RuntimeError(msg)

    pytest = _PyTestShim()


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT.parent / "Waypoint-1.5-1B"
WEIGHTS_PATH = MODEL_DIR / "transformer" / "diffusion_pytorch_model.safetensors"
VAE_DIR = MODEL_DIR / "vae"
VAE_WEIGHTS_PATH = VAE_DIR / "diffusion_pytorch_model.safetensors"


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


def _lcg_normal_latent(shape, seed):
    state = seed or 1
    out = np.empty(int(np.prod(shape)), dtype=np.float32)
    i = 0
    while i < out.size:
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        u1 = np.float32(((state >> 8) & 0x00FFFFFF) / 16777216.0)
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        u2 = np.float32(((state >> 8) & 0x00FFFFFF) / 16777216.0)
        if u1 < np.float32(1.0e-7):
            u1 = np.float32(1.0e-7)
        mag = np.float32(math.sqrt(float(np.float32(-2.0) * np.float32(math.log(float(u1))))))
        phase = np.float32(np.float32(2.0 * math.pi) * u2)
        out[i] = np.float32(mag * np.float32(math.cos(float(phase))))
        if i + 1 < out.size:
            out[i + 1] = np.float32(mag * np.float32(math.sin(float(phase))))
        i += 2
    return out.reshape(shape)


def _fixture_latent(shape):
    idx = np.arange(int(np.prod(shape)), dtype=np.float32)
    data = 0.5 * np.sin(idx * np.float32(0.013)) + 0.25 * np.cos(idx * np.float32(0.031))
    return data.astype(np.float32).reshape(shape)


def _control_vector(cfg):
    n = int(cfg["n_buttons"]) + 3
    idx = np.arange(n, dtype=np.float32)
    return (0.25 * np.sin(idx * 0.17) + 0.05 * np.cos(idx * 0.07)).astype(np.float32)


def _read_dump(prefix, name, shape):
    path = Path(f"{prefix}.{name}.f32")
    arr = np.fromfile(path, dtype=np.float32)
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise AssertionError(f"{path.name}: expected {expected} floats, got {arr.size}")
    return torch.from_numpy(arr.reshape(shape).copy())


def _read_ppm(path):
    data = Path(path).read_bytes()
    magic, width, height, maxval, body = data.split(None, 4)
    if magic != b"P6" or maxval != b"255":
        raise AssertionError(f"unsupported PPM header in {path}")
    width = int(width)
    height = int(height)
    arr = np.frombuffer(body, dtype=np.uint8)
    expected = width * height * 3
    if arr.size != expected:
        raise AssertionError(f"{path}: expected {expected} bytes, got {arr.size}")
    return arr.reshape(height, width, 3).copy()


def _assert_rgb_close(name, actual_rgb, ref_rgb, max_atol, mean_atol):
    diff = np.abs(actual_rgb.astype(np.int16) - ref_rgb.astype(np.int16))
    if diff.max() > max_atol or diff.mean() > mean_atol:
        raise AssertionError(f"{name} max_diff={diff.max()} mean_diff={diff.mean():.6f}")
    print(f"{name}: ok max_abs={diff.max()} mean_abs={diff.mean():.6g}")


def _frame_path(path, frame_idx):
    p = Path(path)
    return p.with_name(f"{p.stem}.{frame_idx}{p.suffix}")


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


def test_standalone_normal_noise_dump_matches_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if not WEIGHTS_PATH.exists():
        pytest.skip(f"missing Waypoint weights: {WEIGHTS_PATH}")

    cfg = _load_config()
    exe = _build_executable()
    seed = 2468
    device = torch.device("cuda")

    with tempfile.TemporaryDirectory(prefix="world_normal_noise_probe_") as td:
        prefix = str(Path(td) / "dump")
        subprocess.run(
            [
                str(exe),
                "--model-dir",
                str(MODEL_DIR),
                "--seed",
                str(seed),
                "--noise",
                "normal",
                "--sigma",
                "1.0",
                "--layers",
                "1",
                "--dump-prefix",
                prefix,
            ],
            check=True,
            cwd=str(ROOT),
        )

        c = cfg["channels"]
        ph, pw = int(cfg["patch"][0]), int(cfg["patch"][1])
        h, w = cfg["height"] * ph, cfg["width"] * pw
        expected = torch.from_numpy(_lcg_normal_latent((1, c, h, w), seed)).to(device)
        _assert_close("normal_noise_latent", _read_dump(prefix, "latent", (1, c, h, w)), expected, rtol=0, atol=2.0e-6)


def test_standalone_external_latent_dump_matches_input():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if not WEIGHTS_PATH.exists():
        pytest.skip(f"missing Waypoint weights: {WEIGHTS_PATH}")

    cfg = _load_config()
    exe = _build_executable()
    device = torch.device("cuda")

    with tempfile.TemporaryDirectory(prefix="world_external_latent_probe_") as td:
        prefix = str(Path(td) / "dump")
        c = cfg["channels"]
        ph, pw = int(cfg["patch"][0]), int(cfg["patch"][1])
        h, w = cfg["height"] * ph, cfg["width"] * pw
        latent_np = _fixture_latent((1, c, h, w))
        latent_path = Path(td) / "latent.f32"
        latent_np.tofile(latent_path)

        subprocess.run(
            [
                str(exe),
                "--model-dir",
                str(MODEL_DIR),
                "--latent",
                str(latent_path),
                "--sigma",
                "1.0",
                "--layers",
                "1",
                "--dump-prefix",
                prefix,
            ],
            check=True,
            cwd=str(ROOT),
        )

        expected = torch.from_numpy(latent_np).to(device)
        _assert_close("external_latent", _read_dump(prefix, "latent", (1, c, h, w)), expected, rtol=0, atol=0)


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
        control_np = _control_vector(cfg)
        control_path = Path(td) / "control.f32"
        control_np.tofile(control_path)
        subprocess.run(
            [
                str(exe),
                "--model-dir",
                str(MODEL_DIR),
                "--seed",
                str(seed),
                "--noise",
                "uniform",
                "--sigma",
                str(sigma),
                "--layers",
                "1",
                "--control",
                str(control_path),
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
            "ctrl_emb.mlp.fc1.weight",
            "ctrl_emb.mlp.fc2.weight",
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
            "transformer.blocks.0.ctrl_mlpfusion.fc1_c.weight",
            "transformer.blocks.0.ctrl_mlpfusion.fc2.weight",
            "transformer.blocks.0.dit_mlp.fc1.weight",
            "transformer.blocks.0.dit_mlp.fc2.weight",
        ]
        state = _load_required_tensors(names, device)
        control = torch.from_numpy(control_np).to(device).view(1, 1, -1)
        ctrl_emb = F.linear(
            F.silu(F.linear(control, state["ctrl_emb.mlp.fc1.weight"])),
            state["ctrl_emb.mlp.fc2.weight"],
        )
        ctrl_cond = F.linear(
            F.rms_norm(ctrl_emb.reshape(d_model), (d_model,), eps=1.0e-6),
            state["transformer.blocks.0.ctrl_mlpfusion.fc1_c.weight"],
        )

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
        ctrl_hidden = ctrl_hidden + ctrl_cond
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


def test_standalone_two_layer_transformer_matches_pytorch():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if not WEIGHTS_PATH.exists():
        pytest.skip(f"missing Waypoint weights: {WEIGHTS_PATH}")
    if not VAE_WEIGHTS_PATH.exists():
        pytest.skip(f"missing Waypoint VAE weights: {VAE_WEIGHTS_PATH}")

    cfg = _load_config()
    exe = _build_executable()
    seed = 1234
    sigma = 1.0
    layers = 2
    frame_idx = 3
    device = torch.device("cuda")

    with tempfile.TemporaryDirectory(prefix="world_transformer2_probe_") as td:
        prefix = str(Path(td) / "dump")
        out_path = str(Path(td) / "out.ppm")
        control_np = _control_vector(cfg)
        control_path = Path(td) / "control.f32"
        control_np.tofile(control_path)
        subprocess.run(
            [
                str(exe),
                "--model-dir",
                str(MODEL_DIR),
                "--seed",
                str(seed),
                "--noise",
                "uniform",
                "--sigma",
                str(sigma),
                "--layers",
                str(layers),
                "--steps",
                "1",
                "--frame-idx",
                str(frame_idx),
                "--cache-pass",
                "--control",
                str(control_path),
                "--dump-prefix",
                prefix,
                "--out",
                out_path,
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

        names = [
            "patchify.weight",
            "denoise_step_emb.mlp.fc1.weight",
            "denoise_step_emb.mlp.fc2.weight",
            "ctrl_emb.mlp.fc1.weight",
            "ctrl_emb.mlp.fc2.weight",
            "out_norm.fc.weight",
            "unpatchify.weight",
            "unpatchify.bias",
        ]
        for layer in range(layers):
            p = f"transformer.blocks.{layer}"
            names.extend(
                [
                    f"{p}.mlp_cond_head.bias_in",
                    f"{p}.attn_cond_head.cond_proj.0.weight",
                    f"{p}.attn_cond_head.cond_proj.1.weight",
                    f"{p}.attn_cond_head.cond_proj.2.weight",
                    f"{p}.attn.q_proj.weight",
                    f"{p}.attn.k_proj.weight",
                    f"{p}.attn.v_proj.weight",
                    f"{p}.attn.out_proj.weight",
                    f"{p}.attn.v_lamb",
                    f"{p}.mlp_cond_head.cond_proj.0.weight",
                    f"{p}.mlp_cond_head.cond_proj.1.weight",
                    f"{p}.mlp_cond_head.cond_proj.2.weight",
                    f"{p}.dit_mlp.fc1.weight",
                    f"{p}.dit_mlp.fc2.weight",
                ]
            )
            if layer % int(cfg["ctrl_conditioning_period"]) == 0:
                names.extend(
                    [
                        f"{p}.ctrl_mlpfusion.fc1_x.weight",
                        f"{p}.ctrl_mlpfusion.fc1_c.weight",
                        f"{p}.ctrl_mlpfusion.fc2.weight",
                    ]
                )
        state = _load_required_tensors(names, device)
        control = torch.from_numpy(control_np).to(device).view(1, 1, -1)
        ctrl_emb = F.linear(
            F.silu(F.linear(control, state["ctrl_emb.mlp.fc1.weight"])),
            state["ctrl_emb.mlp.fc2.weight"],
        ).reshape(d_model)
        ctrl_norm = F.rms_norm(ctrl_emb, (d_model,), eps=1.0e-6)

        latent = torch.from_numpy(_lcg_latent((1, c, h, w), seed)).to(device)
        cond = F.linear(
            F.silu(F.linear(_noise_embedding(sigma, device), state["denoise_step_emb.mlp.fc1.weight"])),
            state["denoise_step_emb.mlp.fc2.weight"],
        )
        tokens = F.conv2d(latent, state["patchify.weight"], bias=None, stride=(ph, pw))
        tokens = tokens.permute(0, 2, 3, 1).reshape(tpf, d_model).contiguous()

        idx = torch.arange(tpf, device=device, dtype=torch.long)
        y_pos = idx.div(width, rounding_mode="floor").contiguous()
        x_pos = idx.remainder(width).contiguous()
        fps_div = int(cfg["inference_fps"]) // int(cfg["temporal_compression"])
        frame_timestamp = frame_idx * (int(cfg["base_fps"]) // fps_div)
        t_pos = torch.full_like(idx, frame_timestamp)
        xy, inv_t = _world_rope_tables(d_head, height, width, device)
        v_first = None
        caches = []
        period = int(cfg["global_attn_period"])
        offset = int(cfg["global_attn_offset"]) % period
        for layer in range(layers):
            is_global = ((layer - offset) % period) == 0
            window = int(cfg["global_window"] if is_global else cfg["local_window"])
            ring_length = window * tpf
            capacity = ring_length + tpf
            cache_k = torch.zeros((n_kv_heads, capacity, d_head), device=device, dtype=torch.float32)
            cache_v = torch.zeros_like(cache_k)
            written = torch.zeros((capacity,), device=device, dtype=torch.bool)
            written[ring_length:] = True
            pinned = int(cfg["global_pinned_dilation"] if is_global else 1)
            caches.append((cache_k, cache_v, written, ring_length, pinned))

        for layer in range(layers):
            p = f"transformer.blocks.{layer}"
            cond_act = F.silu(cond.reshape(d_model) + state[f"{p}.mlp_cond_head.bias_in"])
            s0 = F.linear(cond_act, state[f"{p}.attn_cond_head.cond_proj.0.weight"])
            b0 = F.linear(cond_act, state[f"{p}.attn_cond_head.cond_proj.1.weight"])
            g0 = F.linear(cond_act, state[f"{p}.attn_cond_head.cond_proj.2.weight"])
            s1 = F.linear(cond_act, state[f"{p}.mlp_cond_head.cond_proj.0.weight"])
            b1 = F.linear(cond_act, state[f"{p}.mlp_cond_head.cond_proj.1.weight"])
            g1 = F.linear(cond_act, state[f"{p}.mlp_cond_head.cond_proj.2.weight"])

            norm = F.rms_norm(tokens, (d_model,), eps=1.0e-6) * (1.0 + s0) + b0
            q_raw = F.linear(norm, state[f"{p}.attn.q_proj.weight"])
            k_raw = F.linear(norm, state[f"{p}.attn.k_proj.weight"])
            v_raw = F.linear(norm, state[f"{p}.attn.v_proj.weight"])
            q = q_raw.view(1, tpf, n_heads, d_head).permute(0, 2, 1, 3).contiguous()
            k = k_raw.view(1, tpf, n_kv_heads, d_head).permute(0, 2, 1, 3).contiguous()
            v = v_raw.view(1, tpf, n_kv_heads, d_head).permute(0, 2, 1, 3).contiguous()
            q = _ref_ortho_rope(F.rms_norm(q, (d_head,), eps=1.0e-6), x_pos, y_pos, t_pos, xy, inv_t, width, height)[0]
            k = _ref_ortho_rope(F.rms_norm(k, (d_head,), eps=1.0e-6), x_pos, y_pos, t_pos, xy, inv_t, width, height)[0]
            v = v[0]
            if v_first is None:
                v_first = v
            else:
                v = torch.lerp(v, v_first, state[f"{p}.attn.v_lamb"].float())

            cache_k, cache_v, written, ring_length, pinned = caches[layer]
            bucket = (frame_idx + (pinned - 1)) // pinned
            num_buckets = (ring_length // tpf) // pinned
            base = (bucket % num_buckets) * tpf
            ring_idx = torch.arange(base, base + tpf, device=device)
            tail_idx = torch.arange(ring_length, ring_length + tpf, device=device)
            write_step = frame_idx % pinned == 0
            mask_written = written.clone()
            if write_step:
                mask_written[ring_idx] = False
            cache_k[:, tail_idx, :] = k
            cache_v[:, tail_idx, :] = v
            indices = torch.nonzero(mask_written, as_tuple=False).flatten()

            group = n_heads // n_kv_heads
            k_indexed = cache_k[:, indices, :].repeat_interleave(group, dim=0)
            v_indexed = cache_v[:, indices, :].repeat_interleave(group, dim=0)
            scores = torch.einsum("htd,hnd->htn", q, k_indexed) * (d_head ** -0.5)
            probs = torch.softmax(scores, dim=-1)
            attn_heads = torch.einsum("htn,hnd->htd", probs, v_indexed)
            attn = attn_heads.permute(1, 0, 2).reshape(tpf, d_model).contiguous()
            tokens = tokens + F.linear(attn, state[f"{p}.attn.out_proj.weight"]) * g0

            if f"{p}.ctrl_mlpfusion.fc1_x.weight" in state:
                ctrl_hidden = F.linear(F.rms_norm(tokens, (d_model,), eps=1.0e-6), state[f"{p}.ctrl_mlpfusion.fc1_x.weight"])
                ctrl_hidden = ctrl_hidden + F.linear(ctrl_norm, state[f"{p}.ctrl_mlpfusion.fc1_c.weight"])
                tokens = tokens + F.linear(F.silu(ctrl_hidden), state[f"{p}.ctrl_mlpfusion.fc2.weight"])

            mlp_in = F.rms_norm(tokens, (d_model,), eps=1.0e-6) * (1.0 + s1) + b1
            mlp_out = F.linear(
                F.silu(F.linear(mlp_in, state[f"{p}.dit_mlp.fc1.weight"])),
                state[f"{p}.dit_mlp.fc2.weight"],
            )
            tokens = tokens + mlp_out * g1

        mod = F.linear(F.silu(cond), state["out_norm.fc.weight"]).reshape(2, d_model)
        final_tokens = F.silu(F.rms_norm(tokens, (d_model,), eps=1.0e-6) * (1.0 + mod[0]) + mod[1])
        unpatch_w = state["unpatchify.weight"].permute(1, 2, 3, 0).reshape(c * ph * pw, d_model).contiguous()
        unpatch_b = state["unpatchify.bias"][:, None, None].expand(c, ph, pw).reshape(c * ph * pw).contiguous()
        latent_out = F.linear(final_tokens, unpatch_w, unpatch_b)
        latent_out = latent_out.view(height, width, c, ph, pw).permute(2, 0, 3, 1, 4).reshape(c, h, w).contiguous()
        dsigma = float(cfg["scheduler_sigmas"][1]) - sigma
        latent_final = (latent[0] + latent_out * dsigma).contiguous()

        _assert_close(
            "transformer_tokens_2_layers",
            _read_dump(prefix, "transformer_tokens", (tpf, d_model)),
            tokens,
            rtol=1.2e-3,
            atol=1.2e-3,
        )
        _assert_close(
            "latent_final_2_layers",
            _read_dump(prefix, "latent_final", (c, h, w)),
            latent_final,
            rtol=1.2e-3,
            atol=1.2e-3,
        )
        _assert_close(
            "latent_out_2_layers",
            _read_dump(prefix, "latent_out", (c, h, w)),
            latent_out,
            rtol=1.2e-3,
            atol=1.2e-3,
        )
        cache_counts = _read_dump(prefix, "cache_written_counts", (layers,))
        expected_counts = []
        for layer in range(layers):
            is_global = ((layer - offset) % period) == 0
            pinned = int(cfg["global_pinned_dilation"] if is_global else 1)
            expected_counts.append(float(tpf * (2 if frame_idx % pinned == 0 else 1)))
        torch.testing.assert_close(cache_counts, torch.tensor(expected_counts), rtol=0, atol=0)
        print(f"cache_written_counts: ok {cache_counts.tolist()}")

        sys.path.insert(0, str(VAE_DIR))
        from ae_model import ChunkedStreamingTAEHV, _apply_parallel

        vae = ChunkedStreamingTAEHV.from_config(str(VAE_DIR / "config.json"))
        vae.load_state_dict(load_file(str(VAE_WEIGHTS_PATH), device="cpu"), strict=True)
        vae.eval().to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            z4 = latent_final[None, None].repeat(1, 4, 1, 1, 1)
            decoded = _apply_parallel(vae.decoder, z4)
            n, tt, cc, hh, ww = decoded.shape
            decoded = vae._postprocess_output_frames(decoded.reshape(n * tt, cc, hh, ww))
            decoded = decoded.view(n, tt, 3, hh * 2, ww * 2)
            ref_rgb = (
                decoded[:, -4:]
                .squeeze(0)
                .permute(0, 2, 3, 1)[..., :3]
                .clamp(0, 1)
                .mul(255)
                .round()
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
        actual_rgb = _read_ppm(out_path)
        _assert_rgb_close("vae_decode_ppm_fp16_nhwc", actual_rgb, ref_rgb[0], max_atol=8, mean_atol=0.25)
        for frame_idx in range(4):
            actual_frame = _read_ppm(_frame_path(out_path, frame_idx))
            _assert_rgb_close(
                f"vae_decode_frame{frame_idx}_ppm_fp16_nhwc",
                actual_frame,
                ref_rgb[frame_idx],
                max_atol=8,
                mean_atol=0.25,
            )

        latent_final_path = Path(td) / "latent_final.f32"
        latent_final.detach().cpu().numpy().astype(np.float32).tofile(latent_final_path)
        f32_out_path = str(Path(td) / "out_f32.ppm")
        f32_env = os.environ.copy()
        f32_env["WORLD_VAE_FP16_NHWC"] = "0"
        subprocess.run(
            [
                str(exe),
                "--model-dir",
                str(MODEL_DIR),
                "--vae-only",
                "--latent",
                str(latent_final_path),
                "--out",
                f32_out_path,
            ],
            check=True,
            cwd=str(ROOT),
            env=f32_env,
        )
        _assert_rgb_close("vae_decode_ppm_f32_nchw", _read_ppm(f32_out_path), ref_rgb[0], max_atol=4, mean_atol=0.2)
        for frame_idx in range(4):
            _assert_rgb_close(
                f"vae_decode_frame{frame_idx}_ppm_f32_nchw",
                _read_ppm(_frame_path(f32_out_path, frame_idx)),
                ref_rgb[frame_idx],
                max_atol=4,
                mean_atol=0.2,
            )


def test_vae_streaming_two_latents_matches_pytorch():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if not VAE_WEIGHTS_PATH.exists():
        pytest.skip(f"missing VAE weights: {VAE_WEIGHTS_PATH}")

    cfg = _load_config()
    exe = _build_executable()
    device = torch.device("cuda")
    c = cfg["channels"]
    ph, pw = int(cfg["patch"][0]), int(cfg["patch"][1])
    h, w = cfg["height"] * ph, cfg["width"] * pw

    with tempfile.TemporaryDirectory(prefix="world_vae_stream_probe_") as td:
        td = Path(td)
        latents_np = _lcg_normal_latent((2, c, h, w), 2468)
        latent_path = td / "latents.f32"
        latents_np.tofile(latent_path)

        sys.path.insert(0, str(VAE_DIR))
        from ae_model import ChunkedStreamingTAEHV

        vae = ChunkedStreamingTAEHV.from_config(str(VAE_DIR / "config.json"))
        vae.load_state_dict(load_file(str(VAE_WEIGHTS_PATH), device="cpu"), strict=True)
        vae.eval().to(device=device, dtype=torch.float32)
        refs = []
        with torch.inference_mode():
            for i in range(2):
                latent = torch.from_numpy(latents_np[i : i + 1]).to(device)
                refs.append(vae.decode(latent).cpu().numpy())
        ref_rgb = np.concatenate(refs, axis=0)

        for label, env_value, max_atol, mean_atol in [
            ("f32_nchw", "0", 4, 0.2),
            ("fp16_nhwc", "1", 8, 0.35),
        ]:
            out_path = td / f"out_{label}.ppm"
            env = os.environ.copy()
            env["WORLD_VAE_FP16_NHWC"] = env_value
            subprocess.run(
                [
                    str(exe),
                    "--model-dir",
                    str(MODEL_DIR),
                    "--vae-only",
                    "--frames",
                    "2",
                    "--latent",
                    str(latent_path),
                    "--out",
                    str(out_path),
                ],
                check=True,
                cwd=str(ROOT),
                env=env,
            )
            for frame_idx in range(8):
                _assert_rgb_close(
                    f"vae_stream_{label}_frame{frame_idx}",
                    _read_ppm(_frame_path(out_path, frame_idx)),
                    ref_rgb[frame_idx],
                    max_atol=max_atol,
                    mean_atol=mean_atol,
                )


def test_standalone_two_frame_cache_rollout_matches_pytorch():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if not WEIGHTS_PATH.exists():
        pytest.skip(f"missing Waypoint weights: {WEIGHTS_PATH}")

    cfg = _load_config()
    exe = _build_executable()
    seed = 4321
    sigma = 1.0
    layers = 2
    start_frame_idx = 3
    frames = 2
    device = torch.device("cuda")

    with tempfile.TemporaryDirectory(prefix="world_rollout2_probe_") as td:
        prefix = str(Path(td) / "dump")
        control_np = _control_vector(cfg)
        idx_np = np.arange(control_np.size, dtype=np.float32)
        control_seq_np = np.stack(
            [
                control_np,
                (-0.35 * control_np + 0.08 * np.sin(idx_np * 0.11)).astype(np.float32),
            ],
            axis=0,
        ).astype(np.float32)
        control_path = Path(td) / "control_seq.f32"
        control_seq_np.tofile(control_path)
        subprocess.run(
            [
                str(exe),
                "--model-dir",
                str(MODEL_DIR),
                "--seed",
                str(seed),
                "--noise",
                "uniform",
                "--sigma",
                str(sigma),
                "--layers",
                str(layers),
                "--steps",
                "1",
                "--frames",
                str(frames),
                "--frame-idx",
                str(start_frame_idx),
                "--cache-pass",
                "--control-seq",
                str(control_path),
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

        names = [
            "patchify.weight",
            "denoise_step_emb.mlp.fc1.weight",
            "denoise_step_emb.mlp.fc2.weight",
            "ctrl_emb.mlp.fc1.weight",
            "ctrl_emb.mlp.fc2.weight",
            "out_norm.fc.weight",
            "unpatchify.weight",
            "unpatchify.bias",
        ]
        for layer in range(layers):
            p = f"transformer.blocks.{layer}"
            names.extend(
                [
                    f"{p}.mlp_cond_head.bias_in",
                    f"{p}.attn_cond_head.cond_proj.0.weight",
                    f"{p}.attn_cond_head.cond_proj.1.weight",
                    f"{p}.attn_cond_head.cond_proj.2.weight",
                    f"{p}.attn.q_proj.weight",
                    f"{p}.attn.k_proj.weight",
                    f"{p}.attn.v_proj.weight",
                    f"{p}.attn.out_proj.weight",
                    f"{p}.attn.v_lamb",
                    f"{p}.mlp_cond_head.cond_proj.0.weight",
                    f"{p}.mlp_cond_head.cond_proj.1.weight",
                    f"{p}.mlp_cond_head.cond_proj.2.weight",
                    f"{p}.dit_mlp.fc1.weight",
                    f"{p}.dit_mlp.fc2.weight",
                ]
            )
            if layer % int(cfg["ctrl_conditioning_period"]) == 0:
                names.extend(
                    [
                        f"{p}.ctrl_mlpfusion.fc1_x.weight",
                        f"{p}.ctrl_mlpfusion.fc1_c.weight",
                        f"{p}.ctrl_mlpfusion.fc2.weight",
                    ]
                )
        state = _load_required_tensors(names, device)

        control_seq = torch.from_numpy(control_seq_np).to(device)
        ctrl_norms = []
        for frame_ordinal in range(frames):
            control = control_seq[frame_ordinal].view(1, 1, -1)
            ctrl_emb = F.linear(
                F.silu(F.linear(control, state["ctrl_emb.mlp.fc1.weight"])),
                state["ctrl_emb.mlp.fc2.weight"],
            ).reshape(d_model)
            ctrl_norms.append(F.rms_norm(ctrl_emb, (d_model,), eps=1.0e-6))

        idx = torch.arange(tpf, device=device, dtype=torch.long)
        y_pos = idx.div(width, rounding_mode="floor").contiguous()
        x_pos = idx.remainder(width).contiguous()
        xy, inv_t = _world_rope_tables(d_head, height, width, device)
        fps_div = int(cfg["inference_fps"]) // int(cfg["temporal_compression"])
        frame_stride = int(cfg["base_fps"]) // fps_div

        period = int(cfg["global_attn_period"])
        offset = int(cfg["global_attn_offset"]) % period
        caches = []
        for layer in range(layers):
            is_global = ((layer - offset) % period) == 0
            window = int(cfg["global_window"] if is_global else cfg["local_window"])
            ring_length = window * tpf
            capacity = ring_length + tpf
            cache_k = torch.zeros((n_kv_heads, capacity, d_head), device=device, dtype=torch.float32)
            cache_v = torch.zeros_like(cache_k)
            written = torch.zeros((capacity,), device=device, dtype=torch.bool)
            written[ring_length:] = True
            pinned = int(cfg["global_pinned_dilation"] if is_global else 1)
            caches.append((cache_k, cache_v, written, ring_length, pinned))

        def run_reference(latent, step_sigma, frame_ordinal, current_frame_idx, frozen, emit_velocity):
            cond = F.linear(
                F.silu(F.linear(_noise_embedding(step_sigma, device), state["denoise_step_emb.mlp.fc1.weight"])),
                state["denoise_step_emb.mlp.fc2.weight"],
            )
            tokens = F.conv2d(latent, state["patchify.weight"], bias=None, stride=(ph, pw))
            tokens = tokens.permute(0, 2, 3, 1).reshape(tpf, d_model).contiguous()
            t_pos = torch.full_like(idx, current_frame_idx * frame_stride)
            v_first = None

            for layer in range(layers):
                p = f"transformer.blocks.{layer}"
                cond_act = F.silu(cond.reshape(d_model) + state[f"{p}.mlp_cond_head.bias_in"])
                s0 = F.linear(cond_act, state[f"{p}.attn_cond_head.cond_proj.0.weight"])
                b0 = F.linear(cond_act, state[f"{p}.attn_cond_head.cond_proj.1.weight"])
                g0 = F.linear(cond_act, state[f"{p}.attn_cond_head.cond_proj.2.weight"])
                s1 = F.linear(cond_act, state[f"{p}.mlp_cond_head.cond_proj.0.weight"])
                b1 = F.linear(cond_act, state[f"{p}.mlp_cond_head.cond_proj.1.weight"])
                g1 = F.linear(cond_act, state[f"{p}.mlp_cond_head.cond_proj.2.weight"])

                norm = F.rms_norm(tokens, (d_model,), eps=1.0e-6) * (1.0 + s0) + b0
                q_raw = F.linear(norm, state[f"{p}.attn.q_proj.weight"])
                k_raw = F.linear(norm, state[f"{p}.attn.k_proj.weight"])
                v_raw = F.linear(norm, state[f"{p}.attn.v_proj.weight"])
                q = q_raw.view(1, tpf, n_heads, d_head).permute(0, 2, 1, 3).contiguous()
                k = k_raw.view(1, tpf, n_kv_heads, d_head).permute(0, 2, 1, 3).contiguous()
                v = v_raw.view(1, tpf, n_kv_heads, d_head).permute(0, 2, 1, 3).contiguous()
                q = _ref_ortho_rope(F.rms_norm(q, (d_head,), eps=1.0e-6), x_pos, y_pos, t_pos, xy, inv_t, width, height)[0]
                k = _ref_ortho_rope(F.rms_norm(k, (d_head,), eps=1.0e-6), x_pos, y_pos, t_pos, xy, inv_t, width, height)[0]
                v = v[0]
                if v_first is None:
                    v_first = v
                else:
                    v = torch.lerp(v, v_first, state[f"{p}.attn.v_lamb"].float())

                cache_k, cache_v, written, ring_length, pinned = caches[layer]
                bucket = (current_frame_idx + (pinned - 1)) // pinned
                num_buckets = (ring_length // tpf) // pinned
                base = (bucket % num_buckets) * tpf
                ring_idx = torch.arange(base, base + tpf, device=device)
                tail_idx = torch.arange(ring_length, ring_length + tpf, device=device)
                write_step = current_frame_idx % pinned == 0
                mask_written = written.clone()
                if write_step:
                    mask_written[ring_idx] = False
                cache_k[:, tail_idx, :] = k
                cache_v[:, tail_idx, :] = v
                if write_step and not frozen:
                    cache_k[:, ring_idx, :] = k
                    cache_v[:, ring_idx, :] = v
                    written[ring_idx] = True
                indices = torch.nonzero(mask_written, as_tuple=False).flatten()

                group = n_heads // n_kv_heads
                k_indexed = cache_k[:, indices, :].repeat_interleave(group, dim=0)
                v_indexed = cache_v[:, indices, :].repeat_interleave(group, dim=0)
                scores = torch.einsum("htd,hnd->htn", q, k_indexed) * (d_head ** -0.5)
                probs = torch.softmax(scores, dim=-1)
                attn_heads = torch.einsum("htn,hnd->htd", probs, v_indexed)
                attn = attn_heads.permute(1, 0, 2).reshape(tpf, d_model).contiguous()
                tokens = tokens + F.linear(attn, state[f"{p}.attn.out_proj.weight"]) * g0

                if f"{p}.ctrl_mlpfusion.fc1_x.weight" in state:
                    ctrl_hidden = F.linear(F.rms_norm(tokens, (d_model,), eps=1.0e-6), state[f"{p}.ctrl_mlpfusion.fc1_x.weight"])
                    ctrl_hidden = ctrl_hidden + F.linear(ctrl_norms[frame_ordinal], state[f"{p}.ctrl_mlpfusion.fc1_c.weight"])
                    tokens = tokens + F.linear(F.silu(ctrl_hidden), state[f"{p}.ctrl_mlpfusion.fc2.weight"])

                mlp_in = F.rms_norm(tokens, (d_model,), eps=1.0e-6) * (1.0 + s1) + b1
                mlp_out = F.linear(
                    F.silu(F.linear(mlp_in, state[f"{p}.dit_mlp.fc1.weight"])),
                    state[f"{p}.dit_mlp.fc2.weight"],
                )
                tokens = tokens + mlp_out * g1

            if not emit_velocity:
                return tokens, None, None

            mod = F.linear(F.silu(cond), state["out_norm.fc.weight"]).reshape(2, d_model)
            final_tokens = F.silu(F.rms_norm(tokens, (d_model,), eps=1.0e-6) * (1.0 + mod[0]) + mod[1])
            unpatch_w = state["unpatchify.weight"].permute(1, 2, 3, 0).reshape(c * ph * pw, d_model).contiguous()
            unpatch_b = state["unpatchify.bias"][:, None, None].expand(c, ph, pw).reshape(c * ph * pw).contiguous()
            latent_out = F.linear(final_tokens, unpatch_w, unpatch_b)
            latent_out = latent_out.view(height, width, c, ph, pw).permute(2, 0, 3, 1, 4).reshape(c, h, w).contiguous()
            latent_final = (latent[0] + latent_out * (float(cfg["scheduler_sigmas"][1]) - step_sigma)).contiguous()
            return tokens, latent_out, latent_final

        final_tokens = None
        final_latent_out = None
        final_latent = None
        for frame_ordinal in range(frames):
            current_frame_idx = start_frame_idx + frame_ordinal
            latent = torch.from_numpy(_lcg_latent((1, c, h, w), seed + frame_ordinal)).to(device)
            final_tokens, final_latent_out, final_latent = run_reference(
                latent, sigma, frame_ordinal, current_frame_idx, True, True
            )
            run_reference(final_latent[None], 0.0, frame_ordinal, current_frame_idx, False, False)

        _assert_close(
            "rollout_transformer_tokens",
            _read_dump(prefix, "transformer_tokens", (tpf, d_model)),
            final_tokens,
            rtol=1.5e-3,
            atol=1.5e-3,
        )
        _assert_close(
            "rollout_latent_out",
            _read_dump(prefix, "latent_out", (c, h, w)),
            final_latent_out,
            rtol=1.5e-3,
            atol=1.5e-3,
        )
        _assert_close(
            "rollout_latent_final",
            _read_dump(prefix, "latent_final", (c, h, w)),
            final_latent,
            rtol=1.5e-3,
            atol=1.5e-3,
        )

        cache_counts = _read_dump(prefix, "cache_written_counts", (layers,))
        expected_counts = []
        for layer in range(layers):
            is_global = ((layer - offset) % period) == 0
            window = int(cfg["global_window"] if is_global else cfg["local_window"])
            pinned = int(cfg["global_pinned_dilation"] if is_global else 1)
            num_buckets = window // pinned
            written_bases = set()
            for current_frame_idx in range(start_frame_idx, start_frame_idx + frames):
                if current_frame_idx % pinned == 0:
                    bucket = (current_frame_idx + (pinned - 1)) // pinned
                    written_bases.add(bucket % num_buckets)
            expected_counts.append(float(tpf * (1 + len(written_bases))))
        torch.testing.assert_close(cache_counts, torch.tensor(expected_counts), rtol=0, atol=0)
        print(f"rollout_cache_written_counts: ok {cache_counts.tolist()}")


if __name__ == "__main__":
    test_standalone_normal_noise_dump_matches_reference()
    test_standalone_external_latent_dump_matches_input()
    test_standalone_layer0_probe_matches_pytorch()
    test_standalone_two_layer_transformer_matches_pytorch()
    test_vae_streaming_two_latents_matches_pytorch()
    test_standalone_two_frame_cache_rollout_matches_pytorch()
