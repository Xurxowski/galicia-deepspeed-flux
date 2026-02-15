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
from collections import defaultdict
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

CAPTION_PROFILES = {
    "default": {
        "horreo": "ethnographic photo of a galician horreo, stone and wood architecture, natural light",
        "cruceiro": "ethnographic photo of a galician cruceiro, carved stone cross monument, natural light",
    },
    "horreo_focus": {
        "horreo": (
            "ethnographic documentary photo of a traditional galician horreo, raised granary on stone pillars "
            "(pegollos), elongated elevated chamber with ventilation slats, rural Galicia, no modern house"
        ),
        "cruceiro": "ethnographic photo of a galician cruceiro, carved stone cross monument, natural light",
    },
    "cruceiro_focus": {
        "horreo": "ethnographic photo of a galician horreo, stone and wood architecture, natural light",
        "cruceiro": (
            "ethnographic documentary photo of a galician cruceiro, carved granite cross on stone pedestal, "
            "historic village context, Galicia, no modern sculpture"
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build imagefolder dataset with metadata.jsonl")
    parser.add_argument("--raw_dir", required=True, help="Input dir containing label folders")
    parser.add_argument("--dataset_dir", required=True, help="Output dataset root")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--caption_profile",
        default="default",
        choices=sorted(CAPTION_PROFILES.keys()),
        help="Caption profile for better concept specificity",
    )
    parser.add_argument(
        "--include_labels",
        nargs="+",
        default=None,
        help="Only include these labels (example: horreo)",
    )
    parser.add_argument(
        "--exclude_labels",
        nargs="+",
        default=[],
        help="Exclude these labels",
    )
    parser.add_argument(
        "--per_label_limit",
        type=int,
        default=0,
        help="Optional max images per label (0 = no limit)",
    )
    return parser.parse_args()


def list_images(folder: Path) -> list[Path]:
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_split(items: list[tuple[Path, str]], split_dir: Path, captions: dict[str, str]) -> None:
    ensure_clean_dir(split_dir)
    metadata_path = split_dir / "metadata.jsonl"

    with metadata_path.open("w", encoding="utf-8") as f:
        for idx, (src, label) in enumerate(items, start=1):
            out_name = f"{idx:06d}.jpg"
            out_path = split_dir / out_name
            shutil.copy2(src, out_path)
            row = {"file_name": out_name, "text": captions.get(label, label)}
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    raw_root = Path(args.raw_dir)
    dataset_root = Path(args.dataset_dir)
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "validation"
    captions = CAPTION_PROFILES[args.caption_profile]
    include_labels = set(l.lower() for l in args.include_labels) if args.include_labels else None
    exclude_labels = set(l.lower() for l in args.exclude_labels)

    all_items: list[tuple[Path, str]] = []
    counts_by_label: dict[str, int] = defaultdict(int)

    for label_dir in sorted(raw_root.iterdir()):
        if not label_dir.is_dir():
            continue

        label = label_dir.name.lower()
        if include_labels is not None and label not in include_labels:
            continue
        if label in exclude_labels:
            continue

        images = list_images(label_dir)
        if not images:
            continue

        random.shuffle(images)
        if args.per_label_limit > 0:
            images = images[: args.per_label_limit]

        all_items.extend((img, label) for img in images)
        counts_by_label[label] += len(images)

    if not all_items:
        raise SystemExit(f"No images found in {raw_root}")

    random.shuffle(all_items)
    val_count = int(len(all_items) * args.val_ratio)
    val_items = all_items[:val_count]
    train_items = all_items[val_count:]

    write_split(train_items, train_dir, captions=captions)
    write_split(val_items, val_dir, captions=captions)

    print(
        json.dumps(
            {
                "dataset_dir": str(dataset_root),
                "train_count": len(train_items),
                "validation_count": len(val_items),
                "caption_profile": args.caption_profile,
                "labels": counts_by_label,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
