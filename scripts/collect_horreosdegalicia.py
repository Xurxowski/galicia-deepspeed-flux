#!/usr/bin/env python3
"""Collect horreo photos from horreosdegalicia.com with conservative crawling.

IMPORTANT:
- Use this only if you have explicit permission to use images for ML training.
- The script is blocked unless --confirm_rights I_HAVE_PERMISSION is provided.

Example:
  python scripts/collect_horreosdegalicia.py \
    --out_dir /tmp/galicia_raw \
    --max_images 800 \
    --sleep_sec 0.8 \
    --confirm_rights I_HAVE_PERMISSION
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import deque
from io import BytesIO
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from PIL import Image

DEFAULT_START_URLS = [
    "https://horreosdegalicia.com/",
]

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
USER_AGENT = "galicia-deepspeed-flux/1.0 (research dataset curation)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect horreo photos from horreosdegalicia.com")
    parser.add_argument("--out_dir", required=True, help="Raw dataset root directory")
    parser.add_argument("--max_images", type=int, default=800, help="Maximum images to download")
    parser.add_argument("--max_pages", type=int, default=8000, help="Maximum HTML pages to crawl")
    parser.add_argument("--min_side", type=int, default=512, help="Minimum width/height")
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout")
    parser.add_argument("--sleep_sec", type=float, default=0.8, help="Delay between requests")
    parser.add_argument("--start_urls", nargs="+", default=DEFAULT_START_URLS, help="Optional crawl seeds")
    parser.add_argument(
        "--sitemap_url",
        default="https://horreosdegalicia.com/sitemap.xml",
        help="Optional sitemap URL to seed crawling (set empty string to disable)",
    )
    parser.add_argument(
        "--confirm_rights",
        default="",
        help="Set to I_HAVE_PERMISSION to confirm rights for training usage",
    )
    return parser.parse_args()


def normalize_url(base: str, value: str) -> str | None:
    if not value:
        return None
    url = urljoin(base, value)
    url, _ = urldefrag(url)
    p = urlparse(url)
    if p.scheme not in {"http", "https"}:
        return None
    return url


def same_domain(url: str, domain: str = "horreosdegalicia.com") -> bool:
    host = urlparse(url).netloc.lower()
    return host == domain or host.endswith(f".{domain}")


def looks_like_html(url: str) -> bool:
    path = urlparse(url).path.lower()
    if path.endswith((".html", ".htm")):
        return True
    # WordPress often uses extension-less permalinks.
    name = Path(path).name
    if not name:
        return True
    return "." not in name


def extract_attrs(html: str) -> list[str]:
    pattern = r"(?:href|src)\s*=\s*['\"]([^'\"]+)['\"]"
    return re.findall(pattern, html, flags=re.IGNORECASE)


def extract_sitemap_locs(xml_text: str) -> list[str]:
    # Simple extraction (works for most sitemap.xml files and sitemap indexes).
    return re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", xml_text, flags=re.IGNORECASE)


def image_candidate(url: str) -> bool:
    path = urlparse(url).path
    lower = path.lower()
    if not lower.endswith(IMAGE_EXTS):
        return False
    # WordPress uploads are usually under wp-content/uploads
    if "/wp-content/uploads/" in lower:
        return True
    # Fallback: allow other image paths but filter common UI assets.
    if any(token in lower for token in ("logo", "icon", "favicon", "sprite", "avatar", "banner", "header")):
        return False
    return True


def safe_name(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8", errors="ignore")).hexdigest()[:24]


def fetch_text(url: str, timeout: float) -> str | None:
    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
        response.encoding = response.apparent_encoding or response.encoding
        return response.text
    except Exception:
        return None


def fetch_html(url: str, timeout: float) -> str | None:
    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
        ctype = response.headers.get("Content-Type", "").lower()
        if "text/html" not in ctype and "application/xhtml+xml" not in ctype:
            return None
        response.encoding = response.apparent_encoding or response.encoding
        return response.text
    except Exception:
        return None


def download_image(url: str, timeout: float, min_side: int) -> Image.Image | None:
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

    if args.confirm_rights != "I_HAVE_PERMISSION":
        raise SystemExit(
            "Blocked: provide --confirm_rights I_HAVE_PERMISSION only if you have explicit training rights."
        )

    out_root = Path(args.out_dir)
    label_dir = out_root / "horreo"
    label_dir.mkdir(parents=True, exist_ok=True)
    source_log = out_root / "sources_horreosdegalicia.jsonl"

    queue: deque[str] = deque()
    seen_pages: set[str] = set()
    seen_images: set[str] = set()

    sitemap_url = (args.sitemap_url or "").strip()
    if sitemap_url:
        xml_text = fetch_text(sitemap_url, timeout=args.timeout)
        time.sleep(args.sleep_sec)
        if xml_text:
            for loc in extract_sitemap_locs(xml_text):
                full = normalize_url(sitemap_url, loc)
                if full and same_domain(full) and looks_like_html(full):
                    queue.append(full)

    if not queue:
        for seed in args.start_urls:
            full = normalize_url(seed, seed)
            if full:
                queue.append(full)

    saved = 0

    while queue and len(seen_pages) < args.max_pages and saved < args.max_images:
        page_url = queue.popleft()
        if page_url in seen_pages:
            continue
        seen_pages.add(page_url)

        if not same_domain(page_url):
            continue

        html = fetch_html(page_url, timeout=args.timeout)
        time.sleep(args.sleep_sec)
        if not html:
            continue

        attrs = extract_attrs(html)

        for raw in attrs:
            if saved >= args.max_images:
                break

            full = normalize_url(page_url, raw)
            if not full or not same_domain(full):
                continue

            if looks_like_html(full) and full not in seen_pages:
                queue.append(full)
                continue

            if not image_candidate(full):
                continue
            if full in seen_images:
                continue

            seen_images.add(full)
            image = download_image(full, timeout=args.timeout, min_side=args.min_side)
            time.sleep(args.sleep_sec)
            if image is None:
                continue

            out_name = f"horreosdegalicia_{safe_name(full)}.jpg"
            out_path = label_dir / out_name
            if out_path.exists():
                continue

            image.save(out_path, quality=95)
            saved += 1

            row = {
                "file_name": str(out_path.relative_to(out_root)),
                "label": "horreo",
                "source": "horreosdegalicia.com",
                "page_url": page_url,
                "image_url": full,
            }
            with source_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

            if saved % 100 == 0:
                print(f"saved={saved} pages={len(seen_pages)} queue={len(queue)}")

    summary = {
        "out_dir": str(out_root),
        "label": "horreo",
        "saved": saved,
        "pages_crawled": len(seen_pages),
        "queue_remaining": len(queue),
        "max_images": args.max_images,
        "max_pages": args.max_pages,
        "sitemap_url": sitemap_url,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

