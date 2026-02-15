#!/usr/bin/env python3
"""Create (or update) a Hugging Face Space and upload app files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, get_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or update HF Space")
    parser.add_argument("--space_id", required=True, help="user-or-org/space-name")
    parser.add_argument("--space_dir", required=True, help="Local folder with app.py and README.md")
    parser.add_argument("--token", default=None, help="HF token (or HF_TOKEN env var)")
    parser.add_argument(
        "--base_model",
        default="Tongyi-MAI/Z-Image-Turbo",
        help="BASE_MODEL Space variable value",
    )
    parser.add_argument("--lora_repo", default=None, help="Optional LORA_REPO Space variable value")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = args.token or os.getenv("HF_TOKEN") or get_token()
    if not token:
        raise SystemExit("Missing HF token. Run `hf auth login` or use --token/HF_TOKEN.")

    space_dir = Path(args.space_dir)
    if not space_dir.exists() or not space_dir.is_dir():
        raise SystemExit(f"Space folder not found: {space_dir}")

    api = HfApi(token=token)
    api.create_repo(repo_id=args.space_id, repo_type="space", space_sdk="gradio", exist_ok=True)

    api.upload_folder(
        repo_id=args.space_id,
        repo_type="space",
        folder_path=str(space_dir),
        commit_message="Update Space app",
        ignore_patterns=[".DS_Store", "__pycache__", "*.pyc"],
    )

    if hasattr(api, "add_space_variable"):
        api.add_space_variable(repo_id=args.space_id, key="BASE_MODEL", value=args.base_model)
        print(f"Set Space variable BASE_MODEL={args.base_model}")
    else:
        print("HfApi.add_space_variable is not available in this huggingface_hub version.")
        print(f"Set variable manually in Space settings: BASE_MODEL={args.base_model}")

    if args.lora_repo:
        if hasattr(api, "add_space_variable"):
            api.add_space_variable(repo_id=args.space_id, key="LORA_REPO", value=args.lora_repo)
            print(f"Set Space variable LORA_REPO={args.lora_repo}")
        else:
            print(f"Set variable manually in Space settings: LORA_REPO={args.lora_repo}")

    print(f"Space uploaded: https://huggingface.co/spaces/{args.space_id}")


if __name__ == "__main__":
    main()
