"""Process Amazon (2023 jsonl) reviews+metadata into MIMIGenRec multimodal inputs.

Produces, under <output_path>/<CATEGORY>/:
  * <CATEGORY>.item.json   -> {item_id: {title, description, brand, categories, image_paths}}
                              item_id is a *continuous integer* index (string key).
  * <CATEGORY>.train.inter / .valid.inter / .test.inter  (SASRec dense format)
  * <CATEGORY>_images/      -> downloaded MAIN product images
  * <CATEGORY>.review.json / .user2id / .item2id

The item_id ordering is the single source of truth: `multimodal_amazon_text2emb.py`
indexes the collab feature matrix with `int(item_id)`, so item.json keys MUST be
contiguous integers 0..N-1.
"""

import argparse
import collections
import html
import json
import os
import re
import subprocess

import requests
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# 基础辅助函数
# --------------------------------------------------------------------------- #


def clean_text(text):
    """Remove HTML tags, decode entities, collapse whitespace (amazon18 style)."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", str(text))
    text = html.unescape(text)
    text = text.replace("&quot;", '"').replace("&amp;", "&")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def check_path(path):
    os.makedirs(path, exist_ok=True)


def write_json_file(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_remap_index(index_map, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for original, mapped in index_map.items():
            f.write(f"{original}\t{mapped}\n")


# --------------------------------------------------------------------------- #
# 原始数据定位与图像下载
# --------------------------------------------------------------------------- #


def locate_raw_file(args, name_forms, ext_forms, env_hint=None):
    """Find a raw file by trying several candidate paths/names."""
    candidates = []
    for name in name_forms:
        for ext in ext_forms:
            candidates.append(os.path.join(args.raw_dir, f"{name}{ext}"))
            candidates.append(f"{name}{ext}")
    if env_hint:
        candidates.append(os.environ.get(env_hint, ""))
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def download_amazon18_if_needed(args):
    """Best-effort download of the classic Amazon18 files when jsonl missing."""
    base = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2"
    targets = {
        "reviews": f"{base}/categoryFilesSmall/{args.category}_5.json.gz",
        "meta": f"{base}/metaFiles2/meta_{args.category}.json.gz",
    }
    for key, url in targets.items():
        out = os.path.join(args.raw_dir, os.path.basename(url))
        if os.path.exists(out) or os.path.exists(out.replace(".gz", "")):
            continue
        try:
            subprocess.run(["wget", "-q", "-O", out, url], check=True, timeout=600)
        except Exception:
            print(f"[warn] 下载失败(可忽略, 将使用本地文件): {url}")


def download_product_images(meta_data, save_dir):
    """Download MAIN product images (graceful: skip on any failure)."""
    os.makedirs(save_dir, exist_ok=True)
    downloaded = set()
    for product in tqdm(meta_data, desc="Downloading images"):
        parent_asin = product.get("parent_asin", "unknown_asin")
        for img_info in product.get("images", []):
            variant = img_info.get("variant", "default")
            if "MAIN" not in variant:
                continue
            img_url = img_info.get("hi_res") or img_info.get("large") or img_info.get("thumb")
            if not img_url or img_url in downloaded:
                continue
            img_path = os.path.join(save_dir, f"{parent_asin}_{variant}.jpg")
            if os.path.exists(img_path):
                downloaded.add(img_url)
                continue
            try:
                r = requests.get(img_url, timeout=10, stream=True)
                r.raise_for_status()
                with open(img_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                downloaded.add(img_url)
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------- #


def _read_jsonl_lines(path, max_lines=None):
    import gzip as _gzip
    lines = []
    if str(path).endswith(".gz"):
        fh = _gzip.open(path, "rt", encoding="utf-8")
    else:
        fh = open(path, "r", encoding="utf-8")
    with fh as f:
        if max_lines:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                if line.strip():
                    lines.append(json.loads(line.strip()))
        else:
            for line in f:
                if line.strip():
                    lines.append(json.loads(line.strip()))
    return lines


def load_metadata_and_images(meta_file, img_dir, max_lines=None):
    if not meta_file or not os.path.exists(meta_file):
        print(f"Metadata file not found: {meta_file}")
        return [], {}, set()

    metadata = _read_jsonl_lines(meta_file, max_lines=max_lines)

    # 预索引图片目录一次 (避免逐 item 扫描 O(N*M))
    asin2imgs = {}
    if img_dir and os.path.exists(img_dir):
        for img_name in os.listdir(img_dir):
            if "MAIN" not in img_name:
                continue
            asin = img_name.split("_")[0]
            asin2imgs.setdefault(asin, []).append(os.path.abspath(os.path.join(img_dir, img_name)))

    id_title = {}
    remove_items = set()
    processed = []

    for meta in tqdm(metadata, desc="Filtering metadata & attaching images"):
        asin = meta.get("parent_asin") or meta.get("asin")
        title = clean_text(meta.get("title", ""))
        if not title or "<span id" in title:
            remove_items.add(asin)
            continue

        meta["title"] = title
        meta["image_paths"] = asin2imgs.get(asin, [])

        if len(title) > 1 and len(title.split(" ")) <= 50:
            id_title[asin] = title
            processed.append(meta)
        else:
            remove_items.add(asin)

    return processed, id_title, remove_items


def load_reviews(reviews_file, sample_asins):
    if not reviews_file or not os.path.exists(reviews_file):
        print(f"Reviews file not found: {reviews_file}")
        return []

    reviews = []
    import gzip as _gzip
    fh = _gzip.open(reviews_file, "rt", encoding="utf-8") if str(reviews_file).endswith(".gz") else open(reviews_file, "r", encoding="utf-8")
    with fh as f:
        for line in tqdm(f, desc="Loading reviews"):
            r = json.loads(line.strip())
            asin = r.get("parent_asin") or r.get("asin")
            if asin in sample_asins:
                # 新格式 -> 旧格式字段对齐
                r["unixReviewTime"] = int(r.get("timestamp", 0) / 1000)  # ms -> s
                r["reviewerID"] = r.get("user_id")
                r["asin"] = asin
                r["overall"] = r.get("rating")
                r["reviewText"] = r.get("text", "")
                r["summary"] = r.get("title", "")
                reviews.append(r)
    return reviews


# --------------------------------------------------------------------------- #
# 过滤与转换
# --------------------------------------------------------------------------- #


def k_core_filtering(reviews, id_title, user_k=5, item_k=5):
    remove_users = set()
    remove_items = set()

    for review in reviews:
        if review["asin"] not in id_title:
            remove_items.add(review["asin"])

    while True:
        new_reviews = []
        flag = False
        user_counts = collections.defaultdict(int)
        item_counts = collections.defaultdict(int)

        for review in reviews:
            if review["reviewerID"] in remove_users or review["asin"] in remove_items:
                continue
            user_counts[review["reviewerID"]] += 1
            item_counts[review["asin"]] += 1
            new_reviews.append(review)

        for user, count in user_counts.items():
            if count < user_k:
                remove_users.add(user)
                flag = True
        for item, count in item_counts.items():
            if count < item_k:
                remove_items.add(item)
                flag = True

        if not flag:
            break
        reviews = new_reviews

    print(f"K-core: users={len(user_counts)} items={len(item_counts)} reviews={len(new_reviews)}")
    return new_reviews, user_counts, item_counts


def convert_inters2dict(reviews):
    user2index, item2index = {}, {}
    user_reviews = collections.defaultdict(list)

    for review in reviews:
        user_reviews[review["reviewerID"]].append(review)

    for user in user_reviews:
        user_reviews[user].sort(key=lambda x: int(x["unixReviewTime"]))

    for user in user_reviews:
        if user not in user2index:
            user2index[user] = len(user2index)
        for review in user_reviews[user]:
            item = review["asin"]
            if item not in item2index:
                item2index[item] = len(item2index)

    return user2index, item2index, user_reviews


def generate_interaction_list(user_reviews, user2index, item2index, max_hist=50):
    interaction_list = []
    for user_original, records in tqdm(user_reviews.items(), desc="Creating interaction sequences"):
        user_id = user2index[user_original]
        item_ids = [item2index[r["asin"]] for r in records]
        timestamps = [r["unixReviewTime"] for r in records]
        for i in range(1, len(item_ids)):
            st = max(i - max_hist, 0)
            interaction_list.append([user_id, item_ids[st:i], item_ids[i], timestamps[i]])

    interaction_list.sort(key=lambda x: int(x[-1]))
    return interaction_list


def convert_to_atomic_files(interaction_list, dataset_name, out_dir):
    total = len(interaction_list)
    train_end = int(total * 0.8)
    valid_end = int(total * 0.9)

    splits = {
        "train": interaction_list[:train_end],
        "valid": interaction_list[train_end:valid_end],
        "test": interaction_list[valid_end:],
    }
    for name, data in splits.items():
        path = os.path.join(out_dir, f"{dataset_name}.{name}.inter")
        with open(path, "w", encoding="utf-8") as f:
            f.write("user_id:token\titem_id_list:token_seq\titem_id:token\n")
            for inter in data:
                u = inter[0]
                hist = " ".join(map(str, inter[1]))
                target = inter[2]
                f.write(f"{u}\t{hist}\t{target}\n")
    return splits["train"], splits["valid"], splits["test"]


# --------------------------------------------------------------------------- #
# 特征文件
# --------------------------------------------------------------------------- #


def create_item_features(metadata, item2index):
    item2feature = {}
    asin_to_meta = {m.get("parent_asin") or m.get("asin"): m for m in metadata}

    for item_asin, item_id in item2index.items():
        meta = asin_to_meta.get(item_asin)
        if meta is None:
            continue

        cats = meta.get("categories", [])
        if isinstance(cats, list):
            if len(cats) > 0 and isinstance(cats[0], list):
                cats = [c for sub in cats for c in sub]
            categories = ",".join(str(c).strip() for c in cats)
        else:
            categories = str(cats)

        desc = meta.get("description", "")
        if isinstance(desc, list):
            desc = " ".join(clean_text(d) for d in desc)
        else:
            desc = clean_text(desc)

        item2feature[item_id] = {
            "title": clean_text(meta.get("title", "")),
            "description": desc,
            "brand": meta.get("store", meta.get("brand", "")),
            "categories": categories.strip(),
            "image_paths": meta.get("image_paths", []),
        }
    return item2feature


def create_review_features(reviews, user2index, item2index):
    review_data = {}
    for r in tqdm(reviews, desc="Load reviews feature"):
        user, item = r["reviewerID"], r["asin"]
        if user in user2index and item in item2index:
            uid = user2index[user]
            iid = item2index[item]
            unique_key = f"{uid}_{iid}_{r['unixReviewTime']}"
            review_data[unique_key] = {
                "review": clean_text(r.get("reviewText", "")),
                "summary": clean_text(r.get("summary", "")),
            }
    return review_data


# --------------------------------------------------------------------------- #
# CLI / 主流程
# --------------------------------------------------------------------------- #


def parse_args():
    p = argparse.ArgumentParser(description="Possess Amazon data for MIMIGenRec multimodal pipeline")
    p.add_argument("--category", type=str, default="Industrial_and_Scientific")
    p.add_argument("--sample_size", type=int, default=500, help="0/negative => full data")
    p.add_argument("--user_k", type=int, default=1)
    p.add_argument("--item_k", type=int, default=1)
    p.add_argument("--output_path", type=str, default="./data/Amazon")
    p.add_argument("--raw_dir", type=str, default="./data", help="raw jsonl/gz files dir")
    p.add_argument("--reviews_file", type=str, default="", help="explicit reviews jsonl/gz path")
    p.add_argument("--meta_file", type=str, default="", help="explicit meta jsonl/gz path")
    p.add_argument("--img_dir", type=str, default="", help="pre-downloaded images dir")
    p.add_argument("--download_images", action="store_true", help="download MAIN images from urls")
    p.add_argument("--download_amazon18", action="store_true", help="wget classic amazon18 files if jsonl absent")
    args = p.parse_args()
    if args.sample_size <= 0:
        args.sample_size = None
    return args


def main():
    args = parse_args()

    if args.download_amazon18:
        download_amazon18_if_needed(args)

    reviews_file = args.reviews_file or locate_raw_file(
        args, [args.category, f"{args.category}"], [".jsonl", ".jsonl.gz", "_5.json.gz", "_5.json"]
    )
    meta_file = args.meta_file or locate_raw_file(args, [f"meta_{args.category}"], [".jsonl", ".jsonl.gz", ".json.gz", ".json"])
    img_dir = args.img_dir or os.path.join(args.raw_dir, f"{args.category}_images")

    if not reviews_file:
        raise FileNotFoundError(
            f"找不到 reviews 文件。请提供 --reviews_file 或把 {args.category}.jsonl 放到 --raw_dir 下。"
        )
    if not meta_file:
        raise FileNotFoundError(
            f"找不到 meta 文件。请提供 --meta_file 或把 meta_{args.category}.jsonl 放到 --raw_dir 下。"
        )

    out_dir = os.path.join(args.output_path, args.category)
    check_path(out_dir)

    print(f"reviews_file: {reviews_file}")
    print(f"meta_file:    {meta_file}")
    print(f"img_dir:      {img_dir}")

    print("== 1. metadata 加载与图像筛选 ==")
    metadata, id_title, _ = load_metadata_and_images(meta_file, img_dir, max_lines=args.sample_size)

    if args.download_images:
        print("== 2. 下载 MAIN 商品图 ==")
        download_product_images(metadata, img_dir)

    print("== 3. 加载关联 reviews ==")
    sample_asins = set(id_title.keys())
    raw_reviews = load_reviews(reviews_file, sample_asins)

    print("== 4. K-core 过滤 ==")
    f_reviews, _, _ = k_core_filtering(raw_reviews, id_title, user_k=args.user_k, item_k=args.item_k)

    print("== 5. 序列构造与 8:1:1 划分 ==")
    user2index, item2index, user_reviews = convert_inters2dict(f_reviews)
    interaction_list = generate_interaction_list(user_reviews, user2index, item2index)
    train, valid, test = convert_to_atomic_files(interaction_list, args.category, out_dir)
    print(f"train={len(train)} valid={len(valid)} test={len(test)}")

    print("== 6. 生成特征与映射文件 ==")
    item_features = create_item_features(metadata, item2index)
    review_features = create_review_features(f_reviews, user2index, item2index)

    write_json_file(item_features, os.path.join(out_dir, f"{args.category}.item.json"))
    write_json_file(review_features, os.path.join(out_dir, f"{args.category}.review.json"))
    write_remap_index(user2index, os.path.join(out_dir, f"{args.category}.user2id"))
    write_remap_index(item2index, os.path.join(out_dir, f"{args.category}.item2id"))

    n_with_img = sum(1 for v in item_features.values() if v.get("image_paths"))
    print(f"\n完成: {len(item_features)} items ({n_with_img} 有图), 输出目录 {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
