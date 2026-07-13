import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

try:
    import pytest
except ModuleNotFoundError:
    pytest = None


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
DEFAULT_MODEL = WORKSPACE / "Waypoint-1.5-1B-360P" / "model.safetensors"
DEFAULT_VAE = WORKSPACE / "Waypoint-1.5-1B" / "vae"
DEFAULT_IMAGE = WORKSPACE / "lingbot-world-v2" / "examples" / "02" / "image.jpg"


def _fixture_path(env_name, default):
    return Path(os.environ.get(env_name, default))


def test_native_cuda_vae_encoder_matches_pytorch(tmp_path):
    if not torch.cuda.is_available():
        if pytest is not None:
            pytest.skip("CUDA is required")
        raise RuntimeError("CUDA is required")
    model_path = _fixture_path("WORLD_TEST_MODEL_WEIGHTS", DEFAULT_MODEL)
    vae_dir = _fixture_path("WORLD_TEST_VAE_DIR", DEFAULT_VAE)
    image_path = _fixture_path("WORLD_TEST_IMAGE", DEFAULT_IMAGE)
    for path in (model_path, vae_dir / "diffusion_pytorch_model.safetensors", image_path):
        if not path.exists():
            if pytest is not None:
                pytest.skip(f"missing integration fixture: {path}")
            raise RuntimeError(f"missing integration fixture: {path}")

    subprocess.run(
        ["cmake", "--build", str(ROOT / "build"), "-j2", "--target", "worldmodel_raylib"],
        check=True,
    )
    executable = ROOT / "build" / "worldmodel_raylib"
    input_dump = tmp_path / "vae_input.f32"
    latent_dump = tmp_path / "vae_latent.f32"
    output_path = tmp_path / "frame.ppm"
    env = os.environ.copy()
    env["WORLD_DUMP_VAE_INPUT"] = str(input_dump)
    env["WORLD_DUMP_VAE_LATENT"] = str(latent_dump)
    subprocess.run(
        [
            str(executable),
            str(model_path),
            str(image_path),
            "--vae-weights",
            str(vae_dir / "diffusion_pytorch_model.safetensors"),
            "--layers",
            "1",
            "--steps",
            "1",
            "--warmup",
            "0",
            "--headless-smoke",
            "--headless-reset-check",
            "--headless-out",
            str(output_path),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )

    with open(model_path.parent / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    patch_h, patch_w = config.get("patch", (2, 2))
    latent_h = int(config["height"]) * int(patch_h)
    latent_w = int(config["width"]) * int(patch_w)
    channels = int(config["channels"])
    image_h = latent_h * 16
    image_w = latent_w * 16

    rgb = np.fromfile(input_dump, dtype=np.float32).reshape(image_h, image_w, 3)
    actual = np.fromfile(latent_dump, dtype=np.float32).reshape(channels, latent_h, latent_w)

    sys.path.insert(0, str(vae_dir))
    from ae_model import ChunkedStreamingTAEHV

    model = ChunkedStreamingTAEHV.from_pretrained(
        str(vae_dir), local_files_only=True, torch_dtype=torch.float32
    ).to("cuda").eval()
    frames = (
        torch.from_numpy(rgb)
        .to("cuda")
        .permute(2, 0, 1)[None, None]
        .repeat(1, 4, 1, 1, 1)
        .contiguous()
    )
    with torch.inference_mode():
        expected = model._streaming_encode_step(frames).squeeze(0).squeeze(0).float().cpu().numpy()

    error = np.abs(actual - expected)
    cosine = np.dot(actual.ravel(), expected.ravel()) / (
        np.linalg.norm(actual.ravel()) * np.linalg.norm(expected.ravel())
    )
    print(
        f"VAE encoder parity: max_abs={error.max():.8g} "
        f"mean_abs={error.mean():.8g} cosine={cosine:.10f}"
    )
    assert error.max() < 3.0e-3
    assert error.mean() < 4.0e-4
    assert cosine > 0.999999

    vulkan_build = ROOT / "build-vulkan"
    if not (vulkan_build / "CMakeCache.txt").exists():
        print("Vulkan encoder parity skipped: build-vulkan is not configured")
        return
    subprocess.run(
        [
            "cmake",
            "--build",
            str(vulkan_build),
            "-j2",
            "--target",
            "worldmodel_raylib_vulkan",
        ],
        check=True,
    )
    vulkan_input_dump = tmp_path / "vae_input_vulkan.f32"
    vulkan_latent_dump = tmp_path / "vae_latent_vulkan.f32"
    vulkan_env = os.environ.copy()
    vulkan_env["WORLD_DUMP_VAE_INPUT"] = str(vulkan_input_dump)
    vulkan_env["WORLD_DUMP_VAE_LATENT"] = str(vulkan_latent_dump)
    subprocess.run(
        [
            str(vulkan_build / "worldmodel_raylib_vulkan"),
            str(model_path),
            str(image_path),
            "--vae-weights",
            str(vae_dir / "diffusion_pytorch_model.safetensors"),
            "--layers",
            "1",
            "--steps",
            "1",
            "--warmup",
            "0",
            "--headless-smoke",
            "--headless-reset-check",
        ],
        cwd=ROOT,
        env=vulkan_env,
        check=True,
    )
    vulkan_rgb = np.fromfile(vulkan_input_dump, dtype=np.float32).reshape(image_h, image_w, 3)
    vulkan_actual = np.fromfile(vulkan_latent_dump, dtype=np.float32).reshape(
        channels, latent_h, latent_w
    )
    assert np.array_equal(vulkan_rgb, rgb)
    vulkan_error = np.abs(vulkan_actual - expected)
    vulkan_cosine = np.dot(vulkan_actual.ravel(), expected.ravel()) / (
        np.linalg.norm(vulkan_actual.ravel()) * np.linalg.norm(expected.ravel())
    )
    cuda_vulkan_error = np.abs(vulkan_actual - actual)
    print(
        f"Vulkan VAE encoder parity: max_abs={vulkan_error.max():.8g} "
        f"mean_abs={vulkan_error.mean():.8g} cosine={vulkan_cosine:.10f} "
        f"cuda_max_abs={cuda_vulkan_error.max():.8g}"
    )
    assert vulkan_error.max() < 3.0e-3
    assert vulkan_error.mean() < 4.0e-4
    assert vulkan_cosine > 0.999999
    assert cuda_vulkan_error.max() < 1.0e-5


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="world_vae_parity_") as directory:
        test_native_cuda_vae_encoder_matches_pytorch(Path(directory))
