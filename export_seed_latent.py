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
    parser.add_argument(
        "--warmup-chunks",
        type=int,
        default=5,
        help="number of repeated 4-frame chunks to feed the streaming encoder; exports the last latent",
    )
    args = parser.parse_args()
    if args.warmup_chunks <= 0:
        raise ValueError("--warmup-chunks must be >= 1")

    model_dir = pathlib.Path(args.model_dir).resolve()
    sys.path.insert(0, str(model_dir))
    from vae import ChunkedStreamingTAEHV

    image = load_rgb_image(args.image, args.width, args.height)
    frames = torch.from_numpy(np.repeat(image[None], 4, axis=0))
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    vae = ChunkedStreamingTAEHV.from_pretrained(str(model_dir / "vae")).to(device=args.device, dtype=dtype).eval()
    with torch.inference_mode():
        latent = None
        for _ in range(args.warmup_chunks):
            latent = vae.encode(frames)
        latent = latent.float().cpu().contiguous()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    latent.numpy().tofile(out)
    print(
        f"wrote {out} shape={tuple(latent.shape)} elems={latent.numel()} "
        f"warmup_chunks={args.warmup_chunks}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
