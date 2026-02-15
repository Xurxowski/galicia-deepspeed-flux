#!/usr/bin/env python3
"""Collect images from Wikimedia Commons search results.

Usage:
  python scripts/collect_wikimedia_commons.py \
    --out_dir /tmp/galicia_raw \
    --queries "horreo galicia" "cruceiro galicia" \
    --per_query 120
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image
from tqdm import tqdm

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "galicia-deepspeed-flux/1.0 (dataset curation; contact: user)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Wikimedia Commons images")
    parser.add_argument(
        "--queries",
        nargs="*",
        default=["horreo galicia", "cruceiro galicia"],
        help="Commons search queries",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=[],
        help="Commons categories to crawl (example: 'Category:H\u00f3rreos in Galicia')",
    )
    parser.add_argument("--out_dir", required=True, help="Output root directory")
    parser.add_argument("--per_query", type=int, default=120, help="Target images per query")
    parser.add_argument("--max_requests", type=int, default=20, help="Max API calls per query")
    parser.add_argument(
        "--thumb_width",
        type=int,
        default=1024,
        help="Download Commons thumbnails at this width (0 = original file URL)",
    )
    parser.add_argument("--min_side", type=int, default=512, help="Minimum width/height")
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout")
    parser.add_argument("--sleep_sec", type=float, default=0.4, help="Pause between API calls")
    return parser.parse_args()


def label_from_query(query: str) -> str:
    q = query.lower()
    if any(token in q for token in ("cruce", "cross", "wayside")):
        return "cruceiro"
    return "horreo"


def ext_from_url(url: str) -> str:
    lower = url.lower()
    if lower.endswith(".png"):
        return ".png"
    if lower.endswith(".webp"):
        return ".webp"
    return ".jpg"


def commons_search(
    query: str,
    timeout: float,
    *,
    thumb_width: int,
    continuation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,
        "gsrlimit": 50,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "iiextmetadatafilter": "LicenseShortName|LicenseUrl|UsageTerms|Attribution|Artist|Credit|ImageDescription",
    }
    if thumb_width > 0:
        params["iiurlwidth"] = int(thumb_width)
    if continuation:
        params.update(continuation)

    response = requests.get(
        COMMONS_API,
        params=params,
        timeout=timeout,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    return response.json()


def commons_categorymembers(
    category: str,
    timeout: float,
    *,
    thumb_width: int,
    continuation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    title = category.strip()
    if not title.lower().startswith("category:"):
        title = f"Category:{title}"

    params: dict[str, Any] = {
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "generator": "categorymembers",
        "gcmtitle": title,
        "gcmnamespace": 6,
        "gcmlimit": 50,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "iiextmetadatafilter": "LicenseShortName|LicenseUrl|UsageTerms|Attribution|Artist|Credit|ImageDescription",
    }
    if thumb_width > 0:
        params["iiurlwidth"] = int(thumb_width)
    if continuation:
        params.update(continuation)

    response = requests.get(
        COMMONS_API,
        params=params,
        timeout=timeout,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    return response.json()


def iter_image_items(payload: dict[str, Any]):
    pages = payload.get("query", {}).get("pages", [])
    for page in pages:
        title = page.get("title")
        page_id = page.get("pageid")
        for item in page.get("imageinfo", []):
            url = item.get("url")
            if isinstance(url, str) and url.startswith("http"):
                yield {
                    "url": url,
                    "thumburl": item.get("thumburl"),
                    "thumbwidth": item.get("thumbwidth"),
                    "thumbheight": item.get("thumbheight"),
                    "descriptionurl": item.get("descriptionurl"),
                    "extmetadata": item.get("extmetadata", {}),
                    "title": title,
                    "pageid": page_id,
                }


def safe_stem(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8", errors="ignore")).hexdigest()[:20]


def download_and_validate(url: str, timeout: float, min_side: int) -> Image.Image | None:
    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
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
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)

    source_log = root / "sources_wikimedia_commons.jsonl"
    summary: dict[str, int] = {}
    seen_global: set[str] = set()

    work_items: list[tuple[str, str]] = []
    # categories first (more precise), then free-text search queries
    work_items.extend((c, "category") for c in args.categories)
    work_items.extend((q, "search") for q in args.queries)

    for query, mode in work_items:
        label = label_from_query(query)
        label_dir = root / label
        label_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        continuation: dict[str, Any] | None = None

        for _ in range(args.max_requests):
            if saved >= args.per_query:
                break

            if mode == "category":
                payload = commons_categorymembers(
                    category=query,
                    timeout=args.timeout,
                    thumb_width=args.thumb_width,
                    continuation=continuation,
                )
            else:
                payload = commons_search(
                    query=query,
                    timeout=args.timeout,
                    thumb_width=args.thumb_width,
                    continuation=continuation,
                )
            items = list(iter_image_items(payload))
            if not items:
                break

            for item in tqdm(items, desc=f"{query}", leave=False):
                if saved >= args.per_query:
                    break
                original_url = item.get("url")
                if not isinstance(original_url, str):
                    continue

                download_url = item.get("thumburl") if args.thumb_width > 0 else None
                if not isinstance(download_url, str) or not download_url.startswith("http"):
                    download_url = original_url

                if original_url in seen_global:
                    continue
                seen_global.add(original_url)

                image = download_and_validate(url=download_url, timeout=args.timeout, min_side=args.min_side)
                if image is None:
                    continue

                filename = f"wikimedia_{safe_stem(original_url)}{ext_from_url(download_url)}"
                out_path = label_dir / filename
                if out_path.exists():
                    continue

                image.save(out_path, quality=95)
                saved += 1

                ext = item.get("extmetadata") if isinstance(item.get("extmetadata"), dict) else {}
                row = {
                    "file_name": str(out_path.relative_to(root)),
                    "query": query,
                    "label": label,
                    "source": "wikimedia_commons",
                    "title": item.get("title"),
                    "pageid": item.get("pageid"),
                    "description_url": item.get("descriptionurl"),
                    "image_url": original_url,
                    "download_url": download_url,
                    "thumbwidth": item.get("thumbwidth"),
                    "thumbheight": item.get("thumbheight"),
                    "license_short_name": (ext.get("LicenseShortName") or {}).get("value"),
                    "license_url": (ext.get("LicenseUrl") or {}).get("value"),
                    "usage_terms": (ext.get("UsageTerms") or {}).get("value"),
                    "attribution": (ext.get("Attribution") or {}).get("value"),
                    "artist": (ext.get("Artist") or {}).get("value"),
                    "credit": (ext.get("Credit") or {}).get("value"),
                    "image_description": (ext.get("ImageDescription") or {}).get("value"),
                }
                with source_log.open("a", encoding="utf-8") as logf:
                    logf.write(json.dumps(row, ensure_ascii=True) + "\n")

            continuation = payload.get("continue")
            if not continuation:
                break

            time.sleep(args.sleep_sec)

        summary[query] = saved
        print(f"[done] query='{query}' label='{label}' saved={saved}")

    print(json.dumps({"out_dir": str(root), "saved_per_query": summary}, indent=2))


if __name__ == "__main__":
    main()
