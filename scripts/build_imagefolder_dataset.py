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
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Downsample every included label to the minimum label count (helps multi-concept balance)",
    )
    parser.add_argument(
        "--concepts_file",
        default=None,
        help="Optional JSON file with per-label captions/tokens for scalable multi-concept training",
    )
    parser.add_argument(
        "--use_concept_tokens",
        action="store_true",
        help="Prepend concept token to caption when available",
    )
    parser.add_argument(
        "--strict_concepts",
        action="store_true",
        help="Fail if a label does not exist in concepts file",
    )
    return parser.parse_args()


def list_images(folder: Path) -> list[Path]:
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_concepts(path: Path) -> dict[str, dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "labels" in data and isinstance(data["labels"], dict):
        labels_obj = data["labels"]
    elif isinstance(data, dict):
        labels_obj = data
    else:
        raise ValueError(f"Invalid concepts file format: {path}")

    concepts: dict[str, dict[str, str]] = {}
    for raw_label, raw_cfg in labels_obj.items():
        label = str(raw_label).lower()
        if isinstance(raw_cfg, str):
            concepts[label] = {"caption": raw_cfg, "token": ""}
            continue
        if not isinstance(raw_cfg, dict):
            continue
        caption = str(raw_cfg.get("caption", "")).strip()
        token = str(raw_cfg.get("token", "")).strip()
        concepts[label] = {"caption": caption, "token": token}
    return concepts


def build_caption_map(
    labels: set[str],
    *,
    caption_profile: str,
    concepts_file: str | None,
    use_concept_tokens: bool,
    strict_concepts: bool,
) -> tuple[dict[str, str], dict[str, str]]:
    captions = dict(CAPTION_PROFILES[caption_profile])
    tokens: dict[str, str] = {}

    if concepts_file:
        concepts = load_concepts(Path(concepts_file))
        missing: list[str] = []
        for label in labels:
            cfg = concepts.get(label)
            if cfg is None:
                if strict_concepts:
                    missing.append(label)
                continue
            caption = cfg.get("caption", "").strip() or captions.get(label, label)
            token = cfg.get("token", "").strip()
            if use_concept_tokens and token and token not in caption:
                caption = f"{token} {caption}".strip()
            captions[label] = caption
            tokens[label] = token
        if missing:
            raise SystemExit(f"Missing labels in concepts file: {', '.join(sorted(missing))}")

    return captions, tokens


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
    include_labels = set(l.lower() for l in args.include_labels) if args.include_labels else None
    exclude_labels = set(l.lower() for l in args.exclude_labels)

    all_items: list[tuple[Path, str]] = []
    counts_by_label: dict[str, int] = defaultdict(int)
    images_by_label: dict[str, list[Path]] = {}

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

        images_by_label[label] = images

    if not images_by_label:
        raise SystemExit(f"No images found in {raw_root}")

    if args.balance and len(images_by_label) > 1:
        min_count = min(len(v) for v in images_by_label.values() if v)
        for label in list(images_by_label.keys()):
            images_by_label[label] = images_by_label[label][:min_count]

    for label, images in images_by_label.items():
        all_items.extend((img, label) for img in images)
        counts_by_label[label] = len(images)

    if not all_items:
        raise SystemExit(f"No images found in {raw_root}")

    labels_in_dataset = set(counts_by_label.keys())
    captions, tokens = build_caption_map(
        labels_in_dataset,
        caption_profile=args.caption_profile,
        concepts_file=args.concepts_file,
        use_concept_tokens=args.use_concept_tokens,
        strict_concepts=args.strict_concepts,
    )

    random.shuffle(all_items)
    val_count = int(len(all_items) * args.val_ratio)
    val_items = all_items[:val_count]
    train_items = all_items[val_count:]

    write_split(train_items, train_dir, captions=captions)
    write_split(val_items, val_dir, captions=captions)

    # Copy traceability manifests (if any) into dataset root for later attribution.
    dataset_root.mkdir(parents=True, exist_ok=True)
    for manifest in sorted(raw_root.glob("*.jsonl")):
        if manifest.name in {"metadata.jsonl"}:
            continue
        try:
            shutil.copy2(manifest, dataset_root / manifest.name)
        except Exception:
            pass

    print(
        json.dumps(
            {
                "dataset_dir": str(dataset_root),
                "train_count": len(train_items),
                "validation_count": len(val_items),
                "caption_profile": args.caption_profile,
                "labels": counts_by_label,
                "concept_tokens": tokens,
                "balanced": bool(args.balance),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
