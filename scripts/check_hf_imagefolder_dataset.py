#!/usr/bin/env python3
"""Inspect a HF imagefolder dataset repo and print a compact JSON summary."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, get_token, hf_hub_download

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Hugging Face imagefolder dataset contents")
    parser.add_argument("--repo_id", required=True, help="Dataset repo id (user/repo)")
    parser.add_argument("--token", default=None, help="HF token (or HF_TOKEN env / cached login)")
    return parser.parse_args()


def _count_metadata_lines(repo_id: str, split: str, token: str | None) -> dict[str, int | bool]:
    metadata_name = f"{split}/metadata.jsonl"
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=metadata_name,
            token=token,
        )
    except Exception:
        return {
            "metadata_exists": False,
            "metadata_rows": 0,
            "non_empty_text_rows": 0,
        }

    total = 0
    non_empty = 0
    with Path(local_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            total += 1
            if str(row.get("text", "")).strip():
                non_empty += 1
    return {
        "metadata_exists": True,
        "metadata_rows": total,
        "non_empty_text_rows": non_empty,
    }


def main() -> None:
    args = parse_args()
    token = args.token or os.getenv("HF_TOKEN") or get_token()

    api = HfApi(token=token)
    files = api.list_repo_files(args.repo_id, repo_type="dataset")

    summary: dict[str, object] = {
        "repo_id": args.repo_id,
        "total_files": len(files),
        "splits": {},
    }

    for split in ("train", "validation"):
        split_images = [
            f
            for f in files
            if f.startswith(f"{split}/") and f.lower().endswith(IMAGE_EXTS)
        ]
        split_summary: dict[str, int | bool] = {"image_files": len(split_images)}
        split_summary.update(_count_metadata_lines(args.repo_id, split, token))
        summary["splits"][split] = split_summary

    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
