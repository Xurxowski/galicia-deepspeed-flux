#!/usr/bin/env python3
"""Build HF imagefolder dataset from local image+txt sidecar pairs.

Expected input:
  - A folder for horreos (image files + same-name .txt captions)
  - A folder for cruceiros (image files + same-name .txt captions)

Output:
  dataset_dir/
    train/
      horreo_000001.jpg
      ...
      metadata.jsonl
    validation/
      cruceiro_000001.jpg
      ...
      metadata.jsonl
    local_pairs_manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build imagefolder dataset from local image+txt pairs")
    parser.add_argument("--horreos_dir", required=True, help="Path to Horreos folder")
    parser.add_argument("--cruceiros_dir", required=True, help="Path to Cruceiros folder")
    parser.add_argument("--dataset_dir", required=True, help="Output dataset root")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--allow_missing_caption",
        action="store_true",
        help="Keep image even if .txt sidecar is missing (uses fallback caption by label)",
    )
    return parser.parse_args()


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def read_caption(path: Path) -> str:
    # Most files should be UTF-8; fallback avoids hard failures on legacy encodings.
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")
    return " ".join(text.split()).strip()


def list_label_items(
    folder: Path,
    label: str,
    *,
    allow_missing_caption: bool,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    items: list[dict[str, str]] = []
    skipped = defaultdict(int)

    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Folder not found or not a directory: {folder}")

    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    fallback_caption = (
        "ethnographic documentary photo of a traditional galician horreo"
        if label == "horreo"
        else "ethnographic documentary photo of a galician cruceiro"
    )

    for image_path in files:
        txt_path = image_path.with_suffix(".txt")
        if txt_path.exists():
            caption = read_caption(txt_path)
            if not caption:
                skipped["empty_caption"] += 1
                if allow_missing_caption:
                    caption = fallback_caption
                else:
                    continue
        else:
            skipped["missing_caption"] += 1
            if not allow_missing_caption:
                continue
            caption = fallback_caption

        items.append(
            {
                "image_path": str(image_path),
                "caption_path": str(txt_path) if txt_path.exists() else "",
                "label": label,
                "caption": caption,
            }
        )

    return items, dict(skipped)


def choose_val_count(n_items: int, val_ratio: float) -> int:
    if n_items <= 1:
        return 0
    raw = int(round(n_items * val_ratio))
    if n_items >= 10:
        raw = max(raw, 1)
    return min(raw, n_items - 1)


def write_split(items: list[dict[str, str]], split_dir: Path) -> int:
    ensure_clean_dir(split_dir)
    metadata_path = split_dir / "metadata.jsonl"
    written = 0

    with metadata_path.open("w", encoding="utf-8") as meta:
        for idx, item in enumerate(items, start=1):
            src = Path(item["image_path"])
            ext = src.suffix.lower() if src.suffix.lower() in IMAGE_EXTS else ".jpg"
            out_name = f"{item['label']}_{idx:06d}{ext}"
            shutil.copy2(src, split_dir / out_name)
            row = {"file_name": out_name, "text": item["caption"]}
            meta.write(json.dumps(row, ensure_ascii=True) + "\n")
            written += 1

    return written


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    dataset_root = Path(args.dataset_dir)
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "validation"
    dataset_root.mkdir(parents=True, exist_ok=True)

    horreo_items, horreo_skipped = list_label_items(
        Path(args.horreos_dir),
        "horreo",
        allow_missing_caption=args.allow_missing_caption,
    )
    cruceiro_items, cruceiro_skipped = list_label_items(
        Path(args.cruceiros_dir),
        "cruceiro",
        allow_missing_caption=args.allow_missing_caption,
    )

    if not horreo_items and not cruceiro_items:
        raise SystemExit("No valid image+caption pairs found.")

    per_label = {"horreo": horreo_items, "cruceiro": cruceiro_items}
    train_items: list[dict[str, str]] = []
    val_items: list[dict[str, str]] = []

    for label, label_items in per_label.items():
        random.shuffle(label_items)
        n_val = choose_val_count(len(label_items), args.val_ratio)
        val_items.extend(label_items[:n_val])
        train_items.extend(label_items[n_val:])

    random.shuffle(train_items)
    random.shuffle(val_items)

    n_train = write_split(train_items, train_dir)
    n_val = write_split(val_items, val_dir)

    manifest_path = dataset_root / "local_pairs_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as mf:
        for item in train_items + val_items:
            mf.write(json.dumps(item, ensure_ascii=True) + "\n")

    summary = {
        "dataset_dir": str(dataset_root),
        "train_count": n_train,
        "validation_count": n_val,
        "label_counts": {
            "horreo": len(horreo_items),
            "cruceiro": len(cruceiro_items),
        },
        "skipped": {
            "horreo": horreo_skipped,
            "cruceiro": cruceiro_skipped,
        },
        "manifest": str(manifest_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
