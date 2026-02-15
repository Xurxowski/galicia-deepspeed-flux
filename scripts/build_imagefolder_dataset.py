#!/usr/bin/env python3
"""Build Hugging Face imagefolder dataset with metadata.jsonl captions.

Input structure (raw_dir):
  raw_dir/
    horreo/*.jpg
    cruceiro/*.jpg

Output structure (dataset_dir):
  dataset_dir/
    train/
      000001.jpg
      metadata.jsonl
    validation/
      000101.jpg
      metadata.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

CAPTIONS = {
    "horreo": "ethnographic photo of a galician horreo, stone and wood architecture, natural light",
    "cruceiro": "ethnographic photo of a galician cruceiro, carved stone cross monument, natural light",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build imagefolder dataset with metadata.jsonl")
    parser.add_argument("--raw_dir", required=True, help="Input dir containing label folders")
    parser.add_argument("--dataset_dir", required=True, help="Output dataset root")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def list_images(folder: Path) -> list[Path]:
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_split(items: list[tuple[Path, str]], split_dir: Path) -> None:
    ensure_clean_dir(split_dir)
    metadata_path = split_dir / "metadata.jsonl"

    with metadata_path.open("w", encoding="utf-8") as f:
        for idx, (src, label) in enumerate(items, start=1):
            out_name = f"{idx:06d}.jpg"
            out_path = split_dir / out_name
            shutil.copy2(src, out_path)
            row = {"file_name": out_name, "text": CAPTIONS.get(label, label)}
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    raw_root = Path(args.raw_dir)
    dataset_root = Path(args.dataset_dir)
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "validation"

    all_items: list[tuple[Path, str]] = []

    for label_dir in sorted(raw_root.iterdir()):
        if not label_dir.is_dir():
            continue

        label = label_dir.name.lower()
        images = list_images(label_dir)
        if not images:
            continue

        random.shuffle(images)
        all_items.extend((img, label) for img in images)

    if not all_items:
        raise SystemExit(f"No images found in {raw_root}")

    random.shuffle(all_items)
    val_count = int(len(all_items) * args.val_ratio)
    val_items = all_items[:val_count]
    train_items = all_items[val_count:]

    write_split(train_items, train_dir)
    write_split(val_items, val_dir)

    print(
        json.dumps(
            {
                "dataset_dir": str(dataset_root),
                "train_count": len(train_items),
                "validation_count": len(val_items),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
