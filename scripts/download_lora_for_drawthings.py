#!/usr/bin/env python3
"""Download a LoRA from Hugging Face Hub and (optionally) copy to Draw Things.

This is meant for macOS users of Draw Things. Draw Things can import LoRAs from a local
file; this script helps you fetch the .safetensors artifact and place it somewhere convenient.

Notes:
- Draw Things stores user files under:
    ~/Library/Containers/com.liuliu.draw-things/Data/Documents/
  and the Downloads folder is:
    ~/Library/Containers/com.liuliu.draw-things/Data/Documents/Downloads
- This script does not modify Draw Things config; you still import in-app.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, get_token


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download LoRA safetensors and optionally copy into Draw Things.")
    p.add_argument("--repo_id", required=True, help="Hugging Face model repo id (user-or-org/repo)")
    p.add_argument(
        "--filename",
        default="",
        help="Exact filename in the repo to download. If empty, tries to auto-pick a *.safetensors file.",
    )
    p.add_argument("--revision", default=None, help="Optional git revision/branch/tag")
    p.add_argument(
        "--out_dir",
        default=str(Path.cwd() / "downloads"),
        help="Where to place the downloaded file (default: ./downloads)",
    )
    p.add_argument(
        "--rename_to",
        default="",
        help="Optional output filename (keeps original name if empty).",
    )
    p.add_argument(
        "--copy_to_drawthings_downloads",
        action="store_true",
        help="Also copy the resulting file into Draw Things Downloads folder on macOS.",
    )
    return p.parse_args()


def drawthings_downloads_dir() -> Path:
    return Path.home() / "Library/Containers/com.liuliu.draw-things/Data/Documents/Downloads"


def pick_safetensors(api: HfApi, repo_id: str) -> str:
    info = api.model_info(repo_id)
    candidates = [s.rfilename for s in info.siblings if s.rfilename.lower().endswith(".safetensors")]
    # Prefer the common Diffusers LoRA artifact name if present.
    for preferred in ("pytorch_lora_weights.safetensors", "lora.safetensors"):
        for c in candidates:
            if c.endswith(preferred):
                return c
    if not candidates:
        raise SystemExit(
            f"No .safetensors files found in {repo_id}. "
            "Upload your LoRA weights first (e.g. pytorch_lora_weights.safetensors)."
        )
    return sorted(candidates)[0]


def main() -> None:
    args = parse_args()

    token = os.getenv("HF_TOKEN") or get_token()
    api = HfApi(token=token) if token else HfApi()

    filename = args.filename.strip()
    if not filename:
        filename = pick_safetensors(api, args.repo_id)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded_path = hf_hub_download(
        repo_id=args.repo_id,
        filename=filename,
        revision=args.revision,
        token=token,
    )

    src = Path(downloaded_path)
    out_name = args.rename_to.strip() or src.name
    dst = out_dir / out_name
    shutil.copy2(src, dst)

    print(f"Downloaded: {args.repo_id}/{filename}")
    print(f"Saved to:   {dst}")

    if args.copy_to_drawthings_downloads:
        dt_dir = drawthings_downloads_dir()
        if not dt_dir.exists():
            raise SystemExit(f"Draw Things Downloads folder not found: {dt_dir}")
        dt_dst = dt_dir / dst.name
        shutil.copy2(dst, dt_dst)
        print(f"Copied to Draw Things Downloads: {dt_dst}")
        print("Open Draw Things -> LoRA dropdown -> Manage... -> Import -> Local file.")


if __name__ == "__main__":
    main()

