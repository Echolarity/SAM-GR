#!/usr/bin/env python3
"""Toys_and_Games / Sports_and_Outdoors: 完整性校验 -> 全量 5-core(与 Industry 一致)
-> 超过 Industry 规模时按热度裁剪 Top-K -> 输出每商品一张 MAIN 图的下载清单.

输出 (每类别):
  {cat}_core_top{K}.jsonl.gz   过滤后 reviews (供 possess 用 core 1/1)
  {cat}_images_todo.jsonl      待下载清单 {"parent_asin":..., "urls":[large,hi_res,thumb]}
  {cat}_filtered.json          manifest 统计

用法:
  python3 cats_prepare.py --category Toys_and_Games \
      --reviews_file data/Toys_and_Games.jsonl.gz --meta_file data/meta_Toys_and_Games.jsonl.gz \
      --user_k 5 --item_k 5 --top_k 30000
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import subprocess
import time
from datetime import datetime

try:
    import orjson

    def parse(line):
        return orjson.loads(line)

    def dump(o) -> str:
        return orjson.dumps(o).decode("utf-8")

except ImportError:
    def parse(line):
        return json.loads(line)

    def dump(o) -> str:
        return json.dumps(o, ensure_ascii=False)


HERE = os.path.dirname(os.path.abspath(__file__))


def log(msg):
    print(f"[{datetime.now().strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def open_gz_lines(path, pigz_threads=10, max_lines=None):
    proc = subprocess.Popen(["pigz", "-dc", "-p", str(pigz_threads), path],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            text=True, bufsize=1 << 20)
    try:
        for i, line in enumerate(proc.stdout):
            if max_lines is not None and i >= max_lines:
                break
            yield line
    finally:
        proc.terminate()
        proc.wait()


def check_gzip(path):
    log(f"== gzip 完整性: {path} ==")
    t0 = time.time()
    rc = subprocess.run(["pigz", "-t", path], capture_output=True, text=True).returncode
    log(f"   {'OK' if rc == 0 else 'FAILED'} (rc={rc}, {time.time()-t0:.0f}s)")
    return rc == 0


def scan_meta(meta_file, max_lines=None):
    log(f"== 扫描 meta: {os.path.basename(meta_file)} ==")
    t0 = time.time()
    total = bad = no_parent = no_title = 0
    id_title, asin2url = {}, {}
    for line in open_gz_lines(meta_file, max_lines=max_lines):
        total += 1
        try:
            m = parse(line)
        except Exception:
            bad += 1
            continue
        asin = m.get("parent_asin") or m.get("asin")
        if not asin:
            no_parent += 1
            continue
        title = (m.get("title") or "").strip()
        if not title:
            no_title += 1
            continue
        if 1 < len(title) and len(title.split(" ")) <= 50 and "<span id" not in title:
            id_title[asin] = title
        for img in m.get("images") or []:
            if img.get("variant") != "MAIN":
                continue
            urls = [img.get(q) for q in ("large", "hi_res", "thumb") if img.get(q)]
            if urls:
                asin2url[asin] = urls
            break
    log(f"   meta 行数 {total} 坏JSON {bad} 缺asin {no_parent} 缺title {no_title}; "
        f"有效title {len(id_title)} 有MAIN图 {len(asin2url)} ({time.time()-t0:.0f}s)")
    return id_title, asin2url


def scan_reviews(reviews_file, id_title, max_lines=None):
    log(f"== 扫描 reviews: {os.path.basename(reviews_file)} ==")
    t0 = time.time()
    total = bad = not_in_meta = 0
    user_items = collections.defaultdict(list)
    item_users = collections.defaultdict(list)
    for line in open_gz_lines(reviews_file, max_lines=max_lines):
        total += 1
        try:
            r = parse(line)
        except Exception:
            bad += 1
            continue
        user = r.get("user_id")
        item = r.get("parent_asin") or r.get("asin")
        if not user or not item:
            continue
        if item not in id_title:
            not_in_meta += 1
            continue
        user_items[user].append(item)
        item_users[item].append(user)
    inter = sum(len(v) for v in user_items.values())
    log(f"   reviews 行数 {total} 坏JSON {bad} 商品不在meta {not_in_meta}; "
        f"用户 {len(user_items)} 商品 {len(item_users)} 交互 {inter} ({time.time()-t0:.0f}s)")
    return user_items, item_users, total


def kcore_peel(user_items, item_users, user_k, item_k):
    log(f"== k-core ({user_k}/{item_k}) 迭代删除 ==")
    t0 = time.time()
    deg_u = {u: len(v) for u, v in user_items.items()}
    deg_i = {a: len(v) for a, v in item_users.items()}
    rem_u = {u for u, d in deg_u.items() if d < user_k}
    rem_i = {a for a, d in deg_i.items() if d < item_k}
    q_u, q_i = list(rem_u), list(rem_i)
    while q_u or q_i:
        while q_u:
            u = q_u.pop()
            for a in user_items[u]:
                if a in rem_i:
                    continue
                deg_i[a] -= 1
                if deg_i[a] < item_k:
                    rem_i.add(a)
                    q_i.append(a)
        while q_i:
            a = q_i.pop()
            for u in item_users[a]:
                if u in rem_u:
                    continue
                deg_u[u] -= 1
                if deg_u[u] < user_k:
                    rem_u.add(u)
                    q_u.append(u)
    keep_u = set(deg_u) - rem_u
    keep_i = set(deg_i) - rem_i
    log(f"   保留 users={len(keep_u)} items={len(keep_i)} ({time.time()-t0:.0f}s)")
    return keep_u, keep_i


def write_outputs(cat, reviews_file, user_items, keep_u, keep_i, asin2url, top_k, user_k, item_k):
    n_inter = sum(1 for u in keep_u for a in user_items[u] if a in keep_i)
    log(f"== 过滤结果: users={len(keep_u)} items={len(keep_i)} interactions={n_inter} ==")

    items = sorted(keep_i)
    trimmed = False
    if top_k > 0 and len(items) > top_k:
        freq = collections.Counter(a for u in keep_u for a in user_items[u] if a in keep_i)
        top = {a for a, _ in freq.most_common(top_k)}
        items = sorted(top)
        trimmed = True
        log(f"== 超出 top_k={top_k}, 裁剪为 {len(items)} 个热门商品 ==")
    keep_i = set(items)
    n_users_final = sum(1 for u in keep_u if any(a in keep_i for a in user_items[u]))

    out_gz = os.path.join(HERE, f"{cat}_core.jsonl.gz")
    n_w = 0
    with gzip.open(out_gz, "wt", encoding="utf-8", compresslevel=1) as f:
        for line in open_gz_lines(reviews_file):
            try:
                r = parse(line)
            except Exception:
                continue
            user = r.get("user_id")
            item = r.get("parent_asin") or r.get("asin")
            if item in keep_i and user:
                f.write(dump(r))
                f.write("\n")
                n_w += 1
    log(f"   写出过滤 reviews: {n_w} 条 -> {os.path.basename(out_gz)}")

    n_img = 0
    todo_path = os.path.join(HERE, f"{cat}_images_todo.jsonl")
    with open(todo_path, "w", encoding="utf-8") as f:
        for a in items:
            if a in asin2url:
                f.write(json.dumps({"parent_asin": a, "urls": asin2url[a]}, ensure_ascii=False) + "\n")
                n_img += 1
    log(f"   待下载图片: {n_img}/{len(items)} -> {os.path.basename(todo_path)}")

    manifest = {
        "category": cat,
        "user_k": user_k, "item_k": item_k,
        "top_k_trimmed": trimmed,
        "users": n_users_final,
        "items": len(items),
        "interactions": n_w,
        "items_with_image": n_img,
    }
    with open(os.path.join(HERE, f"{cat}_filtered.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log(f"   清单 -> {cat}_filtered.json")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    ap.add_argument("--reviews_file", required=True)
    ap.add_argument("--meta_file", required=True)
    ap.add_argument("--user_k", type=int, default=5)
    ap.add_argument("--item_k", type=int, default=5)
    ap.add_argument("--top_k", type=int, default=30000, help="超过该物品数则按热度裁剪; 0=不裁剪")
    ap.add_argument("--limit_meta", type=int, default=0)
    ap.add_argument("--limit_reviews", type=int, default=0)
    args = ap.parse_args()

    log("=" * 70)
    log(f"{args.category}: 完整性 -> 5-core -> top{args.top_k}裁剪 -> 图片清单")
    log("=" * 70)

    gz_ok = check_gzip(args.reviews_file) & check_gzip(args.meta_file)
    id_title, asin2url = scan_meta(args.meta_file, max_lines=args.limit_meta or None)
    user_items, item_users, _ = scan_reviews(args.reviews_file, id_title, max_lines=args.limit_reviews or None)
    keep_u, keep_i = kcore_peel(user_items, item_users, args.user_k, args.item_k)

    if args.limit_reviews or args.limit_meta:
        log("[debug 模式] 跳过写文件")
        return

    write_outputs(args.category, args.reviews_file, user_items, keep_u, keep_i, asin2url,
                  args.top_k, args.user_k, args.item_k)
    log(f"gz_ok={gz_ok}  全部完成 ✔")


if __name__ == "__main__":
    main()
