#!/usr/bin/env python3
"""Upload a folder to Hugging Face Hub (dataset/model/space)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, get_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload folder to Hugging Face Hub")
    parser.add_argument("--repo_id", required=True, help="user-or-org/repo-name")
    parser.add_argument(
        "--repo_type",
        required=True,
        choices=["model", "dataset", "space"],
        help="Hub repo type",
    )
    parser.add_argument("--folder_path", required=True, help="Local folder to upload")
    parser.add_argument("--token", default=None, help="HF token (or use HF_TOKEN env var)")
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private (recommended when licensing/permissions are not yet confirmed)",
    )
    parser.add_argument("--commit_message", default="Upload from galicia-deepspeed-flux", help="Commit message")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = args.token or os.getenv("HF_TOKEN") or get_token()
    if not token:
        raise SystemExit("Missing HF token. Run `hf auth login` or use --token/HF_TOKEN.")

    folder = Path(args.folder_path)
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type, private=args.private, exist_ok=True)

    api.upload_folder(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        folder_path=str(folder),
        commit_message=args.commit_message,
        ignore_patterns=[".DS_Store", "__pycache__", "*.pyc"],
    )

    print(f"Uploaded {folder} -> {args.repo_type}:{args.repo_id}")


if __name__ == "__main__":
    main()
