#!/usr/bin/env python3
"""Collect images from Pixabay API.

Usage:
  export PIXABAY_API_KEY=...
  python scripts/collect_pixabay_images.py \
    --out_dir /tmp/galicia_raw \
    --queries "horreo gallego" "horreo galicia" \
    --per_query 200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Iterable

import requests
from PIL import Image
from tqdm import tqdm

PIXABAY_API_URL = "https://pixabay.com/api/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect images from Pixabay API")
    parser.add_argument(
        "--queries",
        nargs="+",
        default=["horreo gallego", "horreo galicia", "cruceiro galicia"],
        help="Search queries",
    )
    parser.add_argument("--out_dir", required=True, help="Output root directory")
    parser.add_argument("--per_query", type=int, default=200, help="Target images per query")
    parser.add_argument("--per_page", type=int, default=100, help="API results per page (max 200)")
    parser.add_argument("--max_pages", type=int, default=10, help="Maximum pages per query")
    parser.add_argument("--min_side", type=int, default=512, help="Minimum width/height")
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout")
    parser.add_argument("--sleep_sec", type=float, default=0.4, help="Pause between API calls")
    return parser.parse_args()


def label_from_query(query: str) -> str:
    q = query.lower()
    if "cruce" in q:
        return "cruceiro"
    return "horreo"


def safe_stem(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()[:20]


def ext_from_url(url: str) -> str:
    lower = url.lower()
    if lower.endswith(".png"):
        return ".png"
    if lower.endswith(".webp"):
        return ".webp"
    return ".jpg"


def pixabay_search(api_key: str, query: str, page: int, per_page: int, timeout: float) -> list[dict]:
    params = {
        "key": api_key,
        "q": query,
        "lang": "es",
        "image_type": "photo",
        "safesearch": "true",
        "per_page": min(max(per_page, 3), 200),
        "page": max(page, 1),
    }
    response = requests.get(PIXABAY_API_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    return payload.get("hits", [])


def iter_image_urls(hit: dict) -> Iterable[str]:
    for key in ("largeImageURL", "webformatURL", "previewURL"):
        url = hit.get(key)
        if isinstance(url, str) and url.startswith("http"):
            yield url


def download_and_validate_image(url: str, timeout: float, min_side: int) -> Image.Image | None:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        image.load()
        if min(image.width, image.height) < min_side:
            return None
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    api_key = os.getenv("PIXABAY_API_KEY")
    if not api_key:
        raise SystemExit("Missing PIXABAY_API_KEY environment variable")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    source_log = out_root / "sources_pixabay.jsonl"
    summary: dict[str, int] = {}

    for query in args.queries:
        label = label_from_query(query)
        label_dir = out_root / label
        label_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        seen_hits: set[str] = set()

        for page in range(1, args.max_pages + 1):
            if saved >= args.per_query:
                break

            hits = pixabay_search(
                api_key=api_key,
                query=query,
                page=page,
                per_page=args.per_page,
                timeout=args.timeout,
            )
            if not hits:
                break

            for hit in tqdm(hits, desc=f"{query} page {page}", leave=False):
                if saved >= args.per_query:
                    break

                hit_id = str(hit.get("id", ""))
                if not hit_id or hit_id in seen_hits:
                    continue
                seen_hits.add(hit_id)

                urls = list(iter_image_urls(hit))
                if not urls:
                    continue

                image = None
                selected_url = None
                for url in urls:
                    image = download_and_validate_image(url=url, timeout=args.timeout, min_side=args.min_side)
                    if image is not None:
                        selected_url = url
                        break

                if image is None or selected_url is None:
                    continue

                filename = f"pixabay_{safe_stem(selected_url)}{ext_from_url(selected_url)}"
                out_path = label_dir / filename
                if out_path.exists():
                    continue

                image.save(out_path, quality=95)
                saved += 1

                with source_log.open("a", encoding="utf-8") as logf:
                    row = {
                        "file_name": str(out_path.relative_to(out_root)),
                        "query": query,
                        "label": label,
                        "source": "pixabay",
                        "hit_id": hit.get("id"),
                        "pageURL": hit.get("pageURL"),
                        "imageURL": selected_url,
                        "tags": hit.get("tags"),
                        "user": hit.get("user"),
                    }
                    logf.write(json.dumps(row, ensure_ascii=True) + "\n")

            time.sleep(args.sleep_sec)

        summary[query] = saved
        print(f"[done] query='{query}' label='{label}' saved={saved}")

    print(json.dumps({"out_dir": str(out_root), "saved_per_query": summary}, indent=2))


if __name__ == "__main__":
    main()
