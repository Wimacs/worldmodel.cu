#!/usr/bin/env python3
"""Run the resident CUDA runtime with its built-in per-stage timers enabled."""

import argparse
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def find_executable(build_dir: Path, name: str, config: str):
    suffixes = (".exe", "") if os.name == "nt" else ("", ".exe")
    for directory in (build_dir / config, build_dir):
        for suffix in suffixes:
            candidate = directory / f"{name}{suffix}"
            if candidate.is_file():
                return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build")
    parser.add_argument("--config", default="Release", help="multi-config CMake configuration")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--vae-weights", type=Path)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--cache-window", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path, help="also save the complete process output")
    parser.add_argument("runtime_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    build_dir = args.build_dir.resolve()
    executable = find_executable(build_dir, "worldmodel_raylib", args.config)
    if executable is None:
        parser.error(
            f"missing worldmodel_raylib under {build_dir} or {build_dir / args.config}"
        )

    command = [
        str(executable),
        "--model-dir",
        str(args.model_dir.resolve()),
        "--steps",
        str(args.steps),
        "--cache-window",
        str(args.cache_window),
        "--warmup",
        str(args.warmup),
        "--headless-smoke",
    ]
    if args.vae_weights:
        command.extend(("--vae-weights", str(args.vae_weights.resolve())))
    if args.runtime_args and args.runtime_args[0] == "--":
        args.runtime_args = args.runtime_args[1:]
    command.extend(args.runtime_args)

    env = os.environ.copy()
    env["WORLD_TRANSFORMER_PROFILE"] = "1"
    env["WORLD_VAE_PROFILE"] = "1"
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(result.stdout, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result.stdout, encoding="utf-8")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
