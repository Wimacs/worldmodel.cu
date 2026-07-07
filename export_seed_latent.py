#!/usr/bin/env python3
import argparse
import pathlib
import sys
import urllib.request
from io import BytesIO

import numpy as np
import torch
from PIL import Image


def load_rgb_image(source: str, width: int, height: int) -> np.ndarray:
    if source.startswith("http://") or source.startswith("https://"):
        with urllib.request.urlopen(source, timeout=60) as res:
            data = res.read()
        img = Image.open(BytesIO(data))
    else:
        img = Image.open(source)
    img = img.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a Waypoint starter image to a TAEHV latent .f32 file.")
    parser.add_argument("--model-dir", default="../Waypoint-1.5-1B")
    parser.add_argument(
        "--image",
        default="https://huggingface.co/spaces/Overworld/waypoint-1-small/resolve/main/starter_18.png",
    )
    parser.add_argument("--out", default="/tmp/world_seed_latent.f32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=512)
    args = parser.parse_args()

    model_dir = pathlib.Path(args.model_dir).resolve()
    sys.path.insert(0, str(model_dir))
    from vae import ChunkedStreamingTAEHV

    image = load_rgb_image(args.image, args.width, args.height)
    frames = torch.from_numpy(np.repeat(image[None], 4, axis=0))
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    vae = ChunkedStreamingTAEHV.from_pretrained(str(model_dir / "vae")).to(device=args.device, dtype=dtype).eval()
    with torch.inference_mode():
        latent = vae.encode(frames).float().cpu().contiguous()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    latent.numpy().tofile(out)
    print(f"wrote {out} shape={tuple(latent.shape)} elems={latent.numel()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
