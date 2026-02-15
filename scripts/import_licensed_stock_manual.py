#!/usr/bin/env python3
"""Import manually downloaded stock images into raw training folder.

This script is for assets you already downloaded with proper licenses.
It does not scrape stock websites.

Usage:
  python scripts/import_licensed_stock_manual.py \
    --input_dir /tmp/stock_downloads/horreos \
    --out_dir /tmp/galicia_raw \
    --label horreo \
    --source shutterstock \
    --license_reference "invoice-2026-02-15"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import manually licensed stock images")
    parser.add_argument("--input_dir", required=True, help="Folder containing downloaded licensed images")
    parser.add_argument("--out_dir", required=True, help="Raw dataset root (label subfolder will be created)")
    parser.add_argument("--label", required=True, choices=["horreo", "cruceiro"], help="Target label")
    parser.add_argument("--source", required=True, help="Source name (shutterstock/adobe/dreamstime/manual)")
    parser.add_argument("--license_reference", required=True, help="Invoice/order/license reference id")
    parser.add_argument("--copy", action="store_true", help="Copy files (default).")
    parser.add_argument("--move", action="store_true", help="Move files instead of copy")
    return parser.parse_args()


def list_images(folder: Path) -> list[Path]:
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def safe_stem(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()[:20]


def main() -> None:
    args = parse_args()

    if args.copy and args.move:
        raise SystemExit("Use only one mode: --copy or --move")

    transfer_mode = "move" if args.move else "copy"

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input folder not found: {input_dir}")

    out_root = Path(args.out_dir)
    label_dir = out_root / args.label.lower()
    label_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / "licensed_sources_manifest.jsonl"
    images = list_images(input_dir)
    if not images:
        raise SystemExit(f"No images found in: {input_dir}")

    imported = 0

    for src in images:
        src_key = str(src.resolve())
        out_name = f"{args.source.lower()}_{safe_stem(src_key)}{src.suffix.lower()}"
        dst = label_dir / out_name

        if dst.exists():
            continue

        if transfer_mode == "move":
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(src, dst)

        row = {
            "file_name": str(dst.relative_to(out_root)),
            "label": args.label.lower(),
            "source": args.source.lower(),
            "license_reference": args.license_reference,
            "original_file": str(src),
            "transfer_mode": transfer_mode,
        }
        with manifest_path.open("a", encoding="utf-8") as mf:
            mf.write(json.dumps(row, ensure_ascii=True) + "\n")

        imported += 1

    print(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "out_dir": str(out_root),
                "label": args.label.lower(),
                "source": args.source.lower(),
                "license_reference": args.license_reference,
                "transfer_mode": transfer_mode,
                "imported": imported,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
