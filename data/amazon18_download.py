#!/usr/bin/env python
"""下载 2018 Amazon (McAuley amazon_v2) 数据集: 5-core 评论 + 商品 meta + 筛选后的商品首图.

用法:
    python data/amazon18_download.py                      # 用 data/amazon18.conf
    python data/amazon18_download.py --config <conf>      # 指定配置
    python data/amazon18_download.py --categories A B     # 临时覆盖类别
    python data/amazon18_download.py --help

每个类别输出到 <DATA_ROOT>/<CATEGORY>/ :
    {CATEGORY}_5.json.gz / {CATEGORY}_5.json      # 5-core 评论
    meta_{CATEGORY}.json.gz / meta_{CATEGORY}.json# 商品 meta (含图 URL)
    images/{asin}_MAIN.jpg                        # k-core 后每个商品的首图 (仅一张)
    {CATEGORY}.filtered.json                      # 筛选结果清单

特性: 断点续传(已存在的文件/图片自动跳过), 失败重试, 逐类别处理(一类失败不影响其它).
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import random
import shutil
import sys
import time

import requests
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))

# McAuley amazon_v2 官方地址
REV_BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall"
META_BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2"

DEFAULT_CONFIG = os.path.join(HERE, "amazon18.conf")


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
def load_config(path):
    cfg = {}
    if not os.path.exists(path):
        print(f"[warn] 配置文件不存在: {path}, 使用内置默认值")
    else:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip().strip('"').strip("'")
    return cfg


# --------------------------------------------------------------------------- #
# 下载 / 解压
# --------------------------------------------------------------------------- #
def _content_length(url, timeout=30):
    for _ in range(3):
        try:
            r = requests.head(url, timeout=timeout, allow_redirects=True)
            if r.status_code == 200 and "content-length" in r.headers:
                return int(r.headers["content-length"])
        except Exception:
            time.sleep(1 + random.random())
    return None


def _single_download(url, dst, retries=5):
    tmp = dst + ".part"
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f, tqdm(
                    total=int(r.headers.get("content-length", 0)), unit="B", unit_scale=True,
                    desc=f"    {os.path.basename(dst)}"
                ) as bar:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        bar.update(len(chunk))
            os.replace(tmp, dst)
            return True
        except Exception as e:
            print(f"  [warn] {type(e).__name__}, 重试 ({attempt+1}/{retries})")
            time.sleep(2 + random.random())
    return False


def _fetch_range(url, part, start, end, retries=10, timeout=90):
    expect = end - start + 1
    for _ in range(retries):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have >= expect:
            return True
        try:
            with requests.get(url, headers={"Range": f"bytes={start + have}-{end}"},
                               stream=True, timeout=timeout) as r:
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"status {r.status_code}")
                if r.status_code == 200 and have > 0:
                    have = 0  # 服务端忽略了 Range, 重头写
                with open(part, "ab" if have > 0 else "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        f.write(chunk)
            if os.path.getsize(part) >= expect:
                return True
        except Exception:
            pass
        time.sleep(0.5 + random.random())
    return os.path.exists(part) and os.path.getsize(part) >= expect


def download_file(url, dst, conns=16, retries=4):
    """多连接 Range 下载; 断点续传(每个分块独立续传)."""
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"  [skip] 已存在: {os.path.basename(dst)}")
        return True
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    size = _content_length(url)
    if not size:
        print(f"  [warn] 取不到大小, 转单连接: {os.path.basename(dst)}")
        return _single_download(url, dst, retries)

    tmpdir = dst + ".parts"
    os.makedirs(tmpdir, exist_ok=True)
    n = max(1, min(conns, (size + (256 << 10) - 1) // (256 << 10)))
    chunk = (size + n - 1) // n
    ranges = [(i * chunk, min((i + 1) * chunk - 1, size - 1)) for i in range(n)]
    print(f"  [get ] {os.path.basename(dst)}  ({size/1e6:.1f} MB, {n} 连接, Range 续传)")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def complete():
        return all(
            os.path.exists(os.path.join(tmpdir, f"{i}.part"))
            and os.path.getsize(os.path.join(tmpdir, f"{i}.part")) == (e - s + 1)
            for i, (s, e) in enumerate(ranges)
        )

    for wave in range(retries):
        if complete():
            break
        with ThreadPoolExecutor(max_workers=n) as pool:
            futs = [pool.submit(_fetch_range, url, os.path.join(tmpdir, f"{i}.part"), s, e)
                    for i, (s, e) in enumerate(ranges)]
            with tqdm(total=n, desc=f"    {os.path.basename(dst)}") as bar:
                for _ in as_completed(futs):
                    bar.update(1)

    if not complete():
        print(f"  [fail] {os.path.basename(dst)} 有分块未完成, 保留 .parts 供续传")
        return False

    with open(dst, "wb") as out:
        for i in range(n):
            with open(os.path.join(tmpdir, f"{i}.part"), "rb") as f:
                shutil.copyfileobj(f, out)
    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"  [done] {os.path.basename(dst)} ({os.path.getsize(dst)/1e6:.1f} MB)")
    return True


def gunzip_to(gz_path, json_path):
    if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
        return True
    if not os.path.exists(gz_path):
        return False
    with gzip.open(gz_path, "rb") as fi, open(json_path, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    print(f"  [unzip] {os.path.basename(json_path)}")
    return True


def iter_json(path):
    """兼容 json-lines 与 json-array 两种格式."""
    with open(path, encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
            yield from data
        else:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


# --------------------------------------------------------------------------- #
# k-core 筛选 (迭代, 用户与商品分别 >= user_k / item_k)
# --------------------------------------------------------------------------- #
def k_core(pairs, user_k, item_k):
    user_items = collections.defaultdict(set)
    for u, a in pairs:
        if u and a:
            user_items[u].add(a)

    while True:
        item_users = collections.defaultdict(set)
        for u, items in user_items.items():
            for a in items:
                item_users[a].add(u)
        bad_items = {a for a, us in item_users.items() if len(us) < item_k}
        bad_users = {u for u, items in user_items.items() if len(items) < user_k}
        if not bad_items and not bad_users:
            break
        for u in list(user_items.keys()):
            if u in bad_users:
                del user_items[u]
            elif bad_items:
                user_items[u] -= bad_items
        if not user_items:
            break

    items = set()
    for s in user_items.values():
        items |= s
    return items, len(user_items)


def first_image(meta):
    """取商品第一张图 URL (兼容 image 列表 / 字符串 / imUrl)."""
    img = meta.get("image")
    if isinstance(img, list) and img:
        return img[0]
    if isinstance(img, str) and img:
        return img
    return meta.get("imUrl") or None


# --------------------------------------------------------------------------- #
# 图片下载
# --------------------------------------------------------------------------- #
def fetch_image(asin, url, out_dir, retries=3, timeout=20):
    path = os.path.join(out_dir, f"{asin}_MAIN.jpg")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return "exists"
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout, stream=True, allow_redirects=True)
            if r.status_code in (429,) or r.status_code >= 500:
                time.sleep(2 ** attempt + random.uniform(0, 1))
                continue
            if r.status_code == 404:
                return "404"
            r.raise_for_status()
            tmp = path + ".part"
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
            os.replace(tmp, path)
            return "ok"
        except Exception:
            if attempt == retries - 1:
                return "fail"
            time.sleep(1 + random.uniform(0, 1))
    return "fail"


# --------------------------------------------------------------------------- #
# 单个类别
# --------------------------------------------------------------------------- #
def process_category(cat, cfg):
    root = os.path.join(cfg["DATA_ROOT"], cat)
    img_dir = os.path.join(root, "images")
    os.makedirs(img_dir, exist_ok=True)

    rev_gz = os.path.join(root, f"{cat}_5.json.gz")
    rev_js = os.path.join(root, f"{cat}_5.json")
    meta_gz = os.path.join(root, f"meta_{cat}.json.gz")
    meta_js = os.path.join(root, f"meta_{cat}.json")

    print(f"\n========== [{cat}] ==========")
    print("-- 1. 下载 reviews + meta --")
    ok_rev = download_file(f"{REV_BASE}/{cat}_5.json.gz", rev_gz, conns=cfg["CONNS"])
    ok_meta = download_file(f"{META_BASE}/meta_{cat}.json.gz", meta_gz, conns=cfg["CONNS"])
    if not ok_rev or not ok_meta:
        print(f"[skip] {cat}: reviews/meta 下载失败, 跳过")
        return None
    gunzip_to(rev_gz, rev_js)
    gunzip_to(meta_gz, meta_js)

    print("-- 2. k-core 筛选 --")
    pairs = [(r.get("reviewerID"), r.get("asin")) for r in iter_json(rev_js)]
    items, n_users = k_core(pairs, cfg["USER_K"], cfg["ITEM_K"])
    n_inter = sum(1 for u, a in pairs if u and a and a in items)
    print(f"  k-core(user_k={cfg['USER_K']}, item_k={cfg['ITEM_K']}): "
          f"items={len(items)} users={n_users} interactions={n_inter}")

    print("-- 3. 读取 meta 首图 URL --")
    asin2url = {}
    for m in tqdm(iter_json(meta_js), desc="  scan meta"):
        a = m.get("asin")
        if a in items and a not in asin2url:
            u = first_image(m)
            if u:
                asin2url[a] = u
    print(f"  有图的商品: {len(asin2url)} / {len(items)}")

    todo = sorted(asin2url.keys())
    if cfg["MAX_ITEMS"]:
        todo = todo[: cfg["MAX_ITEMS"]]
    print(f"-- 4. 下载首图 ({len(todo)} 张, workers={cfg['WORKERS']}) --")

    stat = collections.Counter()
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=cfg["WORKERS"]) as pool:
        futs = {pool.submit(fetch_image, a, asin2url[a], img_dir): a for a in todo}
        for f in tqdm(as_completed(futs), total=len(futs), desc="  images"):
            stat[f.result()] += 1

    manifest = {
        "category": cat,
        "user_k": cfg["USER_K"],
        "item_k": cfg["ITEM_K"],
        "num_items": len(items),
        "num_users": n_users,
        "num_interactions": n_inter,
        "num_items_with_image": len(asin2url),
        "downloaded": dict(stat),
        "items": sorted(items),
    }
    with open(os.path.join(root, f"{cat}.filtered.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[ok] {cat}: ok={stat['ok']} exists={stat['exists']} fail={stat['fail']} 404={stat['404']}")
    return manifest


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="下载 2018 Amazon 数据集 + 筛选后首图")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--categories", nargs="*", default=None, help="覆盖配置里的类别列表")
    p.add_argument("--user_k", type=int, default=None)
    p.add_argument("--item_k", type=int, default=None)
    p.add_argument("--workers", type=int, default=None, help="图片下载线程数")
    p.add_argument("--conns", type=int, default=None, help="数据文件(评论/meta)多连接下载数")
    p.add_argument("--data_root", default=None)
    p.add_argument("--max_items", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    conf = load_config(args.config)

    cats = args.categories if args.categories is not None else conf.get("CATEGORIES", "").split()
    if not cats:
        print("[error] 没有要下载的类别 (检查 --categories 或配置里的 CATEGORIES)")
        sys.exit(1)

    cfg = {
        "USER_K": args.user_k if args.user_k is not None else int(conf.get("USER_K", 5)),
        "ITEM_K": args.item_k if args.item_k is not None else int(conf.get("ITEM_K", 5)),
        "WORKERS": args.workers if args.workers is not None else int(conf.get("WORKERS", 32)),
        "CONNS": args.conns if args.conns is not None else int(conf.get("CONNS", 16)),
        "DATA_ROOT": args.data_root or conf.get("DATA_ROOT", "data/amazon18"),
        "MAX_ITEMS": args.max_items if args.max_items is not None else int(conf.get("MAX_ITEMS", 0)),
    }
    # 相对路径基于项目根 (脚本上一级目录)
    if not os.path.isabs(cfg["DATA_ROOT"]):
        cfg["DATA_ROOT"] = os.path.join(os.path.dirname(HERE), cfg["DATA_ROOT"])

    print("=" * 60)
    print("2018 Amazon 数据下载")
    print(f"  categories : {cats}")
    print(f"  user_k/item_k : {cfg['USER_K']}/{cfg['ITEM_K']}")
    print(f"  workers : {cfg['WORKERS']}   数据下载连接数 : {cfg['CONNS']}")
    print(f"  data_root : {cfg['DATA_ROOT']}")
    print(f"  max_items : {cfg['MAX_ITEMS'] or '全部'}")
    print("=" * 60)

    results = []
    for cat in cats:
        try:
            r = process_category(cat, cfg)
            if r:
                results.append(r)
        except Exception as e:
            print(f"[error] {cat} 处理异常: {type(e).__name__}: {e}")

    print("\n==================== 汇总 ====================")
    for r in results:
        print(f"  {r['category']:<28} items={r['num_items']:<7} users={r['num_users']:<8} "
              f"inter={r['num_interactions']:<8} imgs={r['num_items_with_image']}")
    print("完成 ✔")


if __name__ == "__main__":
    main()
