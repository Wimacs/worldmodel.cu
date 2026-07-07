import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file

from test_worldmodel_kernels import load_wm_cuda


@dataclass
class LayerCache:
    k: torch.Tensor
    v: torch.Tensor
    written: torch.Tensor
    ring_length: int
    pinned_dilation: int


def _as_pair(x):
    return int(x[0]), int(x[1])


def _load_config(model_dir: Path):
    with open(model_dir / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    defaults = {
        "gated_attn": False,
        "n_kv_heads": cfg.get("n_heads"),
        "patch": [1, 1],
        "prompt_conditioning": None,
        "rope_nyquist_frac": 0.8,
        "rope_theta": 10000.0,
        "value_residual": False,
    }
    defaults.update(cfg)
    return defaults


def _default_weights_path(model_dir: Path) -> Path:
    f32 = model_dir / "transformer" / "diffusion_pytorch_model.safetensors"
    return f32 if f32.exists() else model_dir / "model.safetensors"


def _linear(x, weight):
    return F.linear(x, weight)


def _mlp(x, w1, w2):
    return _linear(F.silu(_linear(x, w1)), w2)


def _world_rope_tables(d_head: int, height: int, width: int, device):
    assert d_head % 8 == 0
    d_xy = d_head // 8
    d_t = d_head // 4
    max_freq = min(height, width) * 0.8
    n = (d_xy + 1) // 2
    xy = (torch.linspace(1.0, max_freq / 2, n, dtype=torch.float32, device=device) * torch.pi)
    xy = xy.repeat_interleave(2)[:d_xy].contiguous()
    theta = 10000.0
    inv_t = 1.0 / (theta ** (torch.arange(0, d_t, 2, dtype=torch.float32, device=device) / d_t))
    return xy, inv_t.repeat_interleave(2).contiguous()


def _noise_conditioner(sigma, state, cfg):
    half = 256
    freq = torch.logspace(0, -1, steps=half, base=10000.0, dtype=torch.float32, device=sigma.device)
    s = sigma.reshape(-1).float() * 1000.0
    phase = s[:, None] * freq[None, :]
    emb = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1) * math.sqrt(2.0)
    emb = _mlp(emb, state["denoise_step_emb.mlp.fc1.weight"], state["denoise_step_emb.mlp.fc2.weight"])
    return emb.view(*sigma.shape, cfg["d_model"])


def _controller_embedding(state, cfg, device):
    ctrl = torch.zeros((1, 1, cfg["n_buttons"] + 3), device=device, dtype=torch.float32)
    return _mlp(ctrl, state["ctrl_emb.mlp.fc1.weight"], state["ctrl_emb.mlp.fc2.weight"])


def _cond_head(cond, state, layer):
    p = f"transformer.blocks.{layer}"
    bias = state[f"{p}.mlp_cond_head.bias_in"]
    h = F.silu(cond + bias)
    return (
        _linear(h, state[f"{p}.attn_cond_head.cond_proj.0.weight"]),
        _linear(h, state[f"{p}.attn_cond_head.cond_proj.1.weight"]),
        _linear(h, state[f"{p}.attn_cond_head.cond_proj.2.weight"]),
        _linear(h, state[f"{p}.mlp_cond_head.cond_proj.0.weight"]),
        _linear(h, state[f"{p}.mlp_cond_head.cond_proj.1.weight"]),
        _linear(h, state[f"{p}.mlp_cond_head.cond_proj.2.weight"]),
    )


def _ctrl_fusion(x, ctrl_emb, state, layer):
    p = f"transformer.blocks.{layer}.ctrl_mlpfusion"
    x_norm = F.rms_norm(x, (x.shape[-1],), eps=1.0e-6)
    c_norm = F.rms_norm(ctrl_emb, (ctrl_emb.shape[-1],), eps=1.0e-6)
    h = _linear(x_norm.view(1, 1, -1, x.shape[-1]), state[f"{p}.fc1_x.weight"])
    h = h + _linear(c_norm, state[f"{p}.fc1_c.weight"]).unsqueeze(2)
    y = _linear(F.silu(h), state[f"{p}.fc2.weight"])
    return y.flatten(1, 2)


def _unpatch_weight_and_bias(state, cfg):
    ph, pw = _as_pair(cfg["patch"])
    c = cfg["channels"]
    w = state["unpatchify.weight"]
    if w.dim() == 4:
        w = w.permute(1, 2, 3, 0).reshape(c * ph * pw, cfg["d_model"]).contiguous()
    b = state["unpatchify.bias"]
    if b.numel() != c * ph * pw:
        b = b[:, None, None].expand(c, ph, pw).reshape(-1).contiguous()
    return w, b


def _make_caches(cfg, device):
    caches = []
    tpf = cfg["height"] * cfg["width"]
    d_head = cfg["d_model"] // cfg["n_heads"]
    period = cfg["global_attn_period"]
    off = cfg["global_attn_offset"] % period
    for layer in range(cfg["n_layers"]):
        is_global = ((layer - off) % period) == 0
        window = cfg["global_window"] if is_global else cfg["local_window"]
        ring_length = window * tpf
        capacity = ring_length + tpf
        k = torch.zeros((1, cfg["n_kv_heads"], capacity, d_head), device=device, dtype=torch.float32)
        v = torch.zeros_like(k)
        written = torch.zeros((capacity,), device=device, dtype=torch.bool)
        written[ring_length:] = True
        pinned = cfg["global_pinned_dilation"] if is_global else 1
        caches.append(LayerCache(k, v, written, ring_length, pinned))
    return caches


def _prepare_weights(state, cfg):
    for k, v in list(state.items()):
        if v.dtype != torch.float32:
            state[k] = v.float()
        elif not v.is_contiguous():
            state[k] = v.contiguous()

    for layer in range(cfg["n_layers"]):
        p = f"transformer.blocks.{layer}.attn"
        state[f"{p}.qkv_proj.weight"] = torch.cat(
            (state[f"{p}.q_proj.weight"], state[f"{p}.k_proj.weight"], state[f"{p}.v_proj.weight"]),
            dim=0,
        ).contiguous()

    state["unpatchify.remap.weight"], state["unpatchify.remap.bias"] = _unpatch_weight_and_bias(state, cfg)


@torch.inference_mode()
def world_forward(x, sigma, frame_idx, cfg, state, caches, ext, frozen=True):
    device = x.device
    ph, pw = _as_pair(cfg["patch"])
    height = cfg["height"]
    width = cfg["width"]
    d_model = cfg["d_model"]
    d_head = d_model // cfg["n_heads"]
    tpf = height * width
    frame_timestamp = frame_idx * (cfg["base_fps"] // (cfg["inference_fps"] // cfg["temporal_compression"]))

    cond = _noise_conditioner(torch.tensor([[sigma]], device=device, dtype=torch.float32), state, cfg)
    ctrl_emb = _controller_embedding(state, cfg, device)

    tokens = ext.patchify(x, state["patchify.weight"])
    idx = torch.arange(tpf, device=device, dtype=torch.long)
    y_pos = idx.div(width, rounding_mode="floor").contiguous()
    x_pos = idx.remainder(width).contiguous()
    t_pos = torch.full((tpf,), int(frame_timestamp), device=device, dtype=torch.long)
    xy, inv_t = _world_rope_tables(d_head, height, width, device)

    v_first = None
    scale = d_head ** -0.5
    for layer in range(cfg["n_layers"]):
        if layer % 4 == 0:
            print(f"  layer {layer:02d}", flush=True)

        s0, b0, g0, s1, b1, g1 = _cond_head(cond, state, layer)
        residual = tokens
        h = ext.ada_rms_norm(tokens, s0, b0, 1.0e-6)

        p = f"transformer.blocks.{layer}"
        qkv = _linear(h, state[f"{p}.attn.qkv_proj.weight"]).contiguous()
        q, k_cur, v_cur = ext.qkv_rms_rope(
            qkv, x_pos, y_pos, t_pos, xy, inv_t, cfg["n_heads"], cfg["n_kv_heads"], width, height, 1.0e-6
        )

        if cfg.get("value_residual", False):
            if v_first is None:
                v_first = v_cur
            else:
                lamb = state[f"{p}.attn.v_lamb"].float()
                v_cur = torch.lerp(v_cur, v_first, lamb)

        cache = caches[layer]
        mask_written = ext.kv_cache_upsert(
            cache.k, cache.v, cache.written, k_cur.contiguous(), v_cur.contiguous(),
            int(frame_idx), cache.ring_length, cache.pinned_dilation, bool(frozen)
        )
        indices = torch.nonzero(mask_written, as_tuple=False).flatten().contiguous()
        attn = ext.indexed_attention(q.contiguous(), cache.k, cache.v, indices, scale)
        attn = attn.permute(0, 2, 1, 3).reshape(1, tpf, d_model).contiguous()
        attn = _linear(attn, state[f"{p}.attn.out_proj.weight"])
        tokens = residual + attn * g0

        if f"{p}.ctrl_mlpfusion.fc1_x.weight" in state:
            tokens = tokens + _ctrl_fusion(tokens, ctrl_emb, state, layer)

        mlp_in = ext.ada_rms_norm(tokens, s1, b1, 1.0e-6)
        mlp_out = _mlp(mlp_in, state[f"{p}.dit_mlp.fc1.weight"], state[f"{p}.dit_mlp.fc2.weight"])
        tokens = tokens + mlp_out * g1

    mod = _linear(F.silu(cond), state["out_norm.fc.weight"])
    a, b = mod.chunk(2, dim=-1)
    tokens = F.silu(F.rms_norm(tokens, (d_model,), eps=1.0e-6) * (1.0 + a) + b)
    return ext.unpatchify(
        tokens.contiguous(),
        state["unpatchify.remap.weight"],
        state["unpatchify.remap.bias"],
        cfg["channels"],
        height * ph,
        width * pw,
        ph,
        pw,
    )


def main():
    parser = argparse.ArgumentParser(description="Load Waypoint weights and start a hybrid CUDA latent generation smoke run.")
    parser.add_argument("--model-dir", type=Path, default=Path("../Waypoint-1.5-1B"))
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=1, help="Number of scheduler steps to run. Use 1 for a quick start smoke.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cache-pass", action="store_true", help="Run the sigma=0 unfrozen pass to persist the final latent into KV cache.")
    parser.add_argument("--output", type=Path, default=Path("smoke_latent.pt"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    cfg = _load_config(args.model_dir)
    weights_path = args.weights or _default_weights_path(args.model_dir)

    print(f"loading extension from {Path(__file__).resolve().parent}", flush=True)
    ext = load_wm_cuda()

    print(f"loading weights: {weights_path}", flush=True)
    state = load_file(str(weights_path), device=str(device))
    _prepare_weights(state, cfg)
    print(f"loaded {len(state)} tensors", flush=True)

    caches = _make_caches(cfg, device)
    ph, pw = _as_pair(cfg["patch"])
    x = torch.randn((1, cfg["channels"], cfg["height"] * ph, cfg["width"] * pw), device=device, dtype=torch.float32)

    sigmas = [float(s) for s in cfg["scheduler_sigmas"]]
    max_steps = min(args.steps, len(sigmas) - 1)
    print(f"starting generation: latent={tuple(x.shape)} steps={max_steps}", flush=True)
    for step in range(max_steps):
        sigma = sigmas[step]
        dsig = sigmas[step + 1] - sigmas[step]
        print(f"step {step + 1}/{max_steps}: sigma={sigma:.6f} dsigma={dsig:.6f}", flush=True)
        v = world_forward(x, sigma, frame_idx=0, cfg=cfg, state=state, caches=caches, ext=ext, frozen=True)
        x = (x + dsig * v).contiguous()
        print(
            "  latent stats after step: "
            f"mean={x.mean().item():.6f} std={x.std().item():.6f} "
            f"min={x.min().item():.6f} max={x.max().item():.6f}",
            flush=True,
        )

    if args.cache_pass:
        print("cache pass: sigma=0, frozen=False", flush=True)
        _ = world_forward(x, 0.0, frame_idx=0, cfg=cfg, state=state, caches=caches, ext=ext, frozen=False)

    torch.save({"latent": x.detach().cpu(), "steps": max_steps, "seed": args.seed}, args.output)
    print(f"saved latent smoke output: {args.output}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
