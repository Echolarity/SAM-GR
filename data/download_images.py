"""Fast parallel download of one (MAIN) image per surviving product.

Determines the final item set via the *same* k-core filtering used by
``possess_data.py``, then downloads only those products' first (MAIN) image in
parallel, so we never fetch images for items that will be dropped later.

Usage:
  python data/download_images.py --category Industrial_and_Scientific \
      --reviews_file <reviews.jsonl> --meta_file <meta.jsonl> \
      --img_dir <out> --user_k 2 --item_k 2 --workers 64
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from possess_data import k_core_filtering, load_metadata_and_images, load_reviews  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--category", type=str, default="Industrial_and_Scientific")
    p.add_argument("--reviews_file", type=str, required=True)
    p.add_argument("--meta_file", type=str, required=True)
    p.add_argument("--img_dir", type=str, required=True)
    p.add_argument("--user_k", type=int, default=2)
    p.add_argument("--item_k", type=int, default=2)
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--quality", type=str, default="large", choices=["hi_res", "large", "thumb"])
    p.add_argument("--limit", type=int, default=0, help="debug: only download this many images")
    return p.parse_args()


def pick_url(img_info, quality="large"):
    order = {"hi_res": ["hi_res", "large", "thumb"], "large": ["large", "hi_res", "thumb"], "thumb": ["thumb", "large", "hi_res"]}[quality]
    for k in order:
        url = img_info.get(k)
        if url:
            return url
    return None


def fetch_one(asin, url, out_dir, timeout=20, retries=3):
    path = os.path.join(out_dir, f"{asin}_MAIN.jpg")
    if os.path.exists(path):
        return asin, "exists"
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout, stream=True)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt + random.uniform(0, 1))
                continue
            r.raise_for_status()
            tmp = path + ".part"
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
            os.replace(tmp, path)
            return asin, "ok"
        except Exception:
            if attempt == retries - 1:
                return asin, f"fail"
            time.sleep(1 + random.uniform(0, 1))
    return asin, "fail"


def main():
    args = parse_args()
    os.makedirs(args.img_dir, exist_ok=True)

    # 1. read meta (full) -> id_title + per-asin MAIN image url
    print("== 读取 meta ==")
    metadata, id_title, _ = load_metadata_and_images(args.meta_file, args.img_dir, max_lines=None)
    asin2url = {}
    for meta in metadata:
        asin = meta.get("parent_asin") or meta.get("asin")
        for img_info in meta.get("images", []):
            if img_info.get("variant") == "MAIN":
                url = pick_url(img_info, args.quality)
                if url:
                    asin2url[asin] = url
                break

    # 2. read reviews (filtered) + k-core -> surviving item set
    print("== 读取 reviews ==")
    raw_reviews = load_reviews(args.reviews_file, set(id_title.keys()))
    print("== k-core 过滤 ==")
    f_reviews, user_counts, item_counts = k_core_filtering(raw_reviews, id_title, args.user_k, args.item_k)
    surviving = [a for a in item_counts.keys() if a in asin2url]
    print(f"surviving items: {len(surviving)} (with image url: {len(surviving)})")

    if args.limit:
        surviving = surviving[: args.limit]

    # 3. parallel download
    missing = [a for a in surviving if not os.path.exists(os.path.join(args.img_dir, f"{a}_MAIN.jpg"))]
    print(f"需要下载: {len(missing)}")

    ok = exists = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(fetch_one, a, asin2url[a], args.img_dir): a for a in missing}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="下载图片"):
            _, status = fut.result()
            if status == "ok":
                ok += 1
            elif status == "exists":
                exists += 1
            else:
                fail += 1

    print(f"完成: ok={ok} exists={exists} fail={fail}")
    print(f"图片目录: {os.path.abspath(args.img_dir)}")


if __name__ == "__main__":
    main()
