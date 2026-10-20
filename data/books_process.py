#!/usr/bin/env python3
"""Books (2023 jsonl) 数据: 完整性校验 -> 5-core 过滤 -> 每商品下载一张 MAIN 图.

步骤:
  1. gzip 完整性校验 (pigz -t)
  2. 流式读 meta: JSON 合法性 / 字段完整性统计, 建立 id_title 与 asin->MAIN url
  3. 流式读 reviews: 统计 + 双向邻接表
  4. 精确迭代 k-core (user_k=5, item_k=5, 参考 possess_data.py)
  5. 写出 5-core 过滤后的 reviews (Books_5core.jsonl.gz) + 统计清单
  6. aiohttp 高并发下载每商品一张 MAIN 图 (large -> hi_res -> thumb 兜底),
     断点续传, 实时进度 log

用法:
  python3 books_process.py \
      --reviews_file Books.jsonl.gz --meta_file meta_Books.jsonl.gz \
      --user_k 5 --item_k 5 --img_dir Books_images --concurrency 150
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import gzip
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime

try:
    import orjson

    def parse_line(line: str):
        return orjson.loads(line)

    def dump_obj(obj) -> str:
        return orjson.dumps(obj).decode("utf-8")

except ImportError:
    def parse_line(line: str):
        return json.loads(line)

    def dump_obj(obj) -> str:
        return json.dumps(obj, ensure_ascii=False)


HERE = os.path.dirname(os.path.abspath(__file__))


def log(msg):
    print(f"[{datetime.now().strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def open_gz_lines(path, pigz_threads=8, max_lines=None):
    """流式读 gz 文件 (pigz 多线程解压), 逐行产出 str."""
    proc = subprocess.Popen(
        ["pigz", "-dc", "-p", str(pigz_threads), path],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1 << 20,
    )
    try:
        for i, line in enumerate(proc.stdout):
            if max_lines is not None and i >= max_lines:
                break
            yield line
    finally:
        proc.terminate()
        proc.wait()


# --------------------------------------------------------------------------- #
# 1. gzip 完整性
# --------------------------------------------------------------------------- #
def check_gzip(path):
    log(f"== gzip 完整性校验: {path} ==")
    t0 = time.time()
    rc = subprocess.run(["pigz", "-t", path], capture_output=True, text=True)
    ok = rc.returncode == 0
    log(f"   {'OK' if ok else 'FAILED'}: pigz -t rc={rc.returncode} ({time.time()-t0:.1f}s)")
    if not ok:
        log(f"   {rc.stderr.strip()[:500]}")
    return ok


# --------------------------------------------------------------------------- #
# 2. meta 扫描
# --------------------------------------------------------------------------- #
def scan_meta(meta_file, max_lines=None):
    log(f"== 扫描 meta: {meta_file} ==")
    t0 = time.time()
    total = bad_json = no_parent = no_title = bad_title = 0
    n_with_main = 0
    qual = collections.Counter()
    id_title = {}
    asin2url = {}
    bad_lines = []

    for line in open_gz_lines(meta_file, max_lines=max_lines):
        total += 1
        try:
            m = parse_line(line)
        except Exception:
            bad_json += 1
            if len(bad_lines) < 5:
                bad_lines.append(line[:200])
            continue
        asin = m.get("parent_asin") or m.get("asin")
        if not asin:
            no_parent += 1
            continue
        title = (m.get("title") or "").strip()
        if not title:
            no_title += 1
            continue
        if len(title) > 1 and len(title.split(" ")) <= 50 and "<span id" not in title:
            id_title[asin] = title
        else:
            bad_title += 1
            continue
        for img in m.get("images") or []:
            if img.get("variant") != "MAIN":
                continue
            urls = []
            for q in ("large", "hi_res", "thumb"):
                u = img.get(q)
                if u:
                    urls.append(u)
                    qual[q] += 1
            if urls:
                asin2url[asin] = urls
                n_with_main += 1
            break

    dt = time.time() - t0
    log(f"   meta 行数: {total}  坏JSON行: {bad_json}  (耗时 {dt:.1f}s, {total/max(dt,1e-9):.0f} 行/s)")
    log(f"   缺 parent_asin: {no_parent}  缺 title: {no_title}  标题非法(过滤): {bad_title}")
    log(f"   有效 title: {len(id_title)}  有 MAIN 图: {n_with_main}  URL质量分布: {dict(qual)}")
    if bad_lines:
        log(f"   坏行示例: {bad_lines[0][:150]}")
    return id_title, asin2url, {
        "meta_lines": total, "bad_json": bad_json, "no_parent": no_parent,
        "no_title": no_title, "bad_title": bad_title,
        "valid_titles": len(id_title), "with_main_image": n_with_main,
    }


# --------------------------------------------------------------------------- #
# 3. reviews 扫描 + 邻接表
# --------------------------------------------------------------------------- #
def scan_reviews(reviews_file, id_title, max_lines=None):
    log(f"== 扫描 reviews: {reviews_file} ==")
    t0 = time.time()
    total = bad_json = no_user = no_item = not_in_meta = 0
    user_items = collections.defaultdict(list)
    item_users = collections.defaultdict(list)
    bad_lines = []

    for line in open_gz_lines(reviews_file, max_lines=max_lines):
        total += 1
        try:
            r = parse_line(line)
        except Exception:
            bad_json += 1
            if len(bad_lines) < 5:
                bad_lines.append(line[:200])
            continue
        user = r.get("user_id")
        item = r.get("parent_asin") or r.get("asin")
        if not user:
            no_user += 1
            continue
        if not item:
            no_item += 1
            continue
        if item not in id_title:
            not_in_meta += 1
            continue
        user_items[user].append(item)
        item_users[item].append(user)

    dt = time.time() - t0
    n_inter = sum(len(v) for v in user_items.values())
    log(f"   reviews 行数: {total}  坏JSON行: {bad_json}  (耗时 {dt:.1f}s, {total/max(dt,1e-9):.0f} 行/s)")
    log(f"   缺 user_id: {no_user}  缺 parent_asin: {no_item}  商品不在meta: {not_in_meta}")
    log(f"   参与计算的用户: {len(user_items)}  商品: {len(item_users)}  交互: {n_inter}")
    if bad_lines:
        log(f"   坏行示例: {bad_lines[0][:150]}")
    return user_items, item_users, {
        "review_lines": total, "bad_json": bad_json, "no_user": no_user,
        "no_item": no_item, "not_in_meta": not_in_meta,
        "users_covered": len(user_items), "items_covered": len(item_users),
    }


# --------------------------------------------------------------------------- #
# 4. 精确 k-core (参考 possess_data.k_core_filtering 的迭代删除语义)
# --------------------------------------------------------------------------- #
def kcore_peel(user_items, item_users, user_k, item_k):
    log(f"== k-core 过滤 (user_k={user_k}, item_k={item_k}) ==")
    t0 = time.time()
    deg_u = {u: len(v) for u, v in user_items.items()}
    deg_i = {a: len(v) for a, v in item_users.items()}
    rem_u = {u for u, d in deg_u.items() if d < user_k}
    rem_i = {a for a, d in deg_i.items() if d < item_k}
    q_u, q_i = list(rem_u), list(rem_i)
    edge_visits = 0

    while q_u or q_i:
        while q_u:
            u = q_u.pop()
            for a in user_items[u]:
                edge_visits += 1
                if a in rem_i:
                    continue
                deg_i[a] -= 1
                if deg_i[a] < item_k:
                    rem_i.add(a)
                    q_i.append(a)
        while q_i:
            a = q_i.pop()
            for u in item_users[a]:
                edge_visits += 1
                if u in rem_u:
                    continue
                deg_u[u] -= 1
                if deg_u[u] < user_k:
                    rem_u.add(u)
                    q_u.append(u)

    keep_u = set(deg_u) - rem_u
    keep_i = set(deg_i) - rem_i
    log(f"   删除用户 {len(rem_u)}, 删除商品 {len(rem_i)}, 遍历边 {edge_visits} ({time.time()-t0:.1f}s)")
    return keep_u, keep_i


# --------------------------------------------------------------------------- #
# 5. 写出过滤后 reviews + 清单
# --------------------------------------------------------------------------- #
def write_filtered(reviews_file, keep_u, keep_i, out_gz, max_lines=None):
    log(f"== 写出 5-core 过滤后 reviews -> {out_gz} ==")
    t0 = time.time()
    written = 0
    bad = 0
    with gzip.open(out_gz, "wt", encoding="utf-8", compresslevel=1) as f:
        for line in open_gz_lines(reviews_file, max_lines=max_lines):
            try:
                r = parse_line(line)
            except Exception:
                bad += 1
                continue
            user = r.get("user_id")
            item = r.get("parent_asin") or r.get("asin")
            if user in keep_u and item in keep_i:
                f.write(dump_obj(r))
                f.write("\n")
                written += 1
                if written % 100000 == 0:
                    log(f"   已写入 {written} 条 ({time.time()-t0:.0f}s)")
    log(f"   完成: 写出 {written} 条, 解析失败 {bad}, 耗时 {time.time()-t0:.1f}s")
    return written


# --------------------------------------------------------------------------- #
# 6. 图片下载 (aiohttp 高并发)
# --------------------------------------------------------------------------- #
def image_ok(path, min_bytes=500):
    try:
        if os.path.getsize(path) < min_bytes:
            return False
        with open(path, "rb") as f:
            return f.read(2) == b"\xff\xd8"
    except OSError:
        return False


async def _fetch_one(asin, urls, out_dir, session, sem, stats):
    path = os.path.join(out_dir, f"{asin}_MAIN.jpg")
    if image_ok(path):
        return "exists"
    for attempt in range(2):
        for url in urls:
            try:
                async with sem:
                    async with session.get(
                        url,
                        timeout=aiohttp.ClientTimeout(total=15, connect=3, sock_read=10),
                        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AmazonImages/1.0"},
                    ) as r:
                        if r.status == 200:
                            data = await r.content.read()
                            if len(data) >= 500 and data[:2] == b"\xff\xd8":
                                loop = asyncio.get_running_loop()
                                await loop.run_in_executor(None, _write_bytes, path, data)
                                return "ok"
                            stats["notjpeg"] += 1
                        elif r.status in (404, 403, 410):
                            continue  # 换下一个质量档
                        elif r.status == 429 or r.status >= 500:
                            await asyncio.sleep(1.5 + random.random())
                        else:
                            pass
            except asyncio.TimeoutError:
                continue
            except Exception:
                pass
        await asyncio.sleep(0.3 + random.random())
    return "fail"


def _write_bytes(path, data):
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


async def download_images(asin2url, surviving_items, out_dir, concurrency, limit=0):
    import aiohttp

    os.makedirs(out_dir, exist_ok=True)
    todo = [a for a in sorted(surviving_items) if a in asin2url and not image_ok(os.path.join(out_dir, f"{a}_MAIN.jpg"))]
    if limit:
        todo = todo[:limit]
    log(f"== 下载图片: 存活商品 {len(surviving_items)}, 有URL {len([a for a in surviving_items if a in asin2url])}, "
        f"待下载 {len(todo)}, 并发 {concurrency} -> {out_dir} ==")

    q: asyncio.Queue = asyncio.Queue()
    for a in todo:
        await q.put(a)

    stats = collections.Counter()
    sem = asyncio.Semaphore(concurrency)
    t0 = time.time()
    last = [t0, 0]
    failed = []
    lock = asyncio.Lock()

    async def worker():
        while True:
            try:
                asin = q.get_nowait()
            except asyncio.QueueEmpty:
                return
            status = await _fetch_one(asin, asin2url[asin], out_dir, session, sem, stats)
            async with lock:
                stats[status] += 1
                if status == "fail":
                    failed.append(asin)
                done = stats["ok"] + stats["exists"] + stats["fail"]
                if done % 500 == 0 or done == len(todo):
                    dt = time.time() - last[0]
                    rate = (done - last[1]) / dt if dt > 0 else 0
                    last[0], last[1] = time.time(), done
                    log(f"   {done}/{len(todo)} ({(time.time()-t0)/60:.1f} min, {rate:.1f} 张/s) "
                        f"ok={stats['ok']} exists={stats['exists']} fail={stats['fail']} notjpeg={stats['notjpeg']}")

    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=concurrency * 2, ttl_dns_cache=600, keepalive_timeout=60),
    ) as session:
        workers = [asyncio.create_task(worker()) for _ in range(min(concurrency, max(1, len(todo))))]
        await asyncio.gather(*workers)

    log(f"   下载完成: ok={stats['ok']} exists={stats['exists']} fail={stats['fail']} "
        f"notjpeg={stats['notjpeg']} 总耗时 {(time.time()-t0)/60:.1f} min")
    if failed:
        fail_path = os.path.join(os.path.dirname(out_dir), "Books_images_failed.jsonl")
        with open(fail_path, "w", encoding="utf-8") as f:
            for a in failed:
                f.write(json.dumps({"parent_asin": a, "urls": asin2url[a]}, ensure_ascii=False) + "\n")
        log(f"   失败清单({len(failed)}): {fail_path}")
    return dict(stats)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reviews_file", default=os.path.join(HERE, "Books.jsonl.gz"))
    p.add_argument("--meta_file", default=os.path.join(HERE, "meta_Books.jsonl.gz"))
    p.add_argument("--user_k", type=int, default=5)
    p.add_argument("--item_k", type=int, default=5)
    p.add_argument("--img_dir", default=os.path.join(HERE, "Books_images"))
    p.add_argument("--filtered_out", default=os.path.join(HERE, "Books_5core.jsonl.gz"))
    p.add_argument("--concurrency", type=int, default=150)
    p.add_argument("--skip_images", action="store_true")
    p.add_argument("--limit_reviews", type=int, default=0, help="debug: 只处理前 N 行 reviews")
    p.add_argument("--limit_meta", type=int, default=0, help="debug: 只处理前 N 行 meta")
    p.add_argument("--limit_images", type=int, default=0, help="debug: 只下载前 N 张图")
    return p.parse_args()


def main():
    args = parse_args()
    log("=" * 70)
    log("Books 数据处理: 完整性校验 -> 5-core -> 单图下载")
    log(f"  reviews: {args.reviews_file}")
    log(f"  meta:    {args.meta_file}")
    log(f"  k-core:  {args.user_k}/{args.item_k}")
    log(f"  images:  {args.img_dir}  concurrency={args.concurrency}")
    log("=" * 70)

    gz_ok = check_gzip(args.reviews_file) & check_gzip(args.meta_file)

    id_title, asin2url, meta_stats = scan_meta(args.meta_file, max_lines=args.limit_meta or None)
    user_items, item_users, rev_stats = scan_reviews(
        args.reviews_file, id_title, max_lines=args.limit_reviews or None
    )

    keep_u, keep_i = kcore_peel(user_items, item_users, args.user_k, args.item_k)

    n_inter = sum(1 for u in keep_u for a in user_items[u] if a in keep_i)
    log(f"== k-core 结果: users={len(keep_u)} items={len(keep_i)} interactions={n_inter} ==")

    if not args.limit_reviews and not args.limit_meta:
        written = write_filtered(args.reviews_file, keep_u, keep_i, args.filtered_out)
    else:
        written = None
        log("  [debug 模式] 跳过写出过滤文件")

    manifest = {
        "category": "Books",
        "gzip_ok": gz_ok,
        "user_k": args.user_k,
        "item_k": args.item_k,
        "meta": meta_stats,
        "reviews": rev_stats,
        "kcore": {"users": len(keep_u), "items": len(keep_i), "interactions": n_inter,
                  "filtered_reviews_written": written},
        "surviving_items_with_image": len([a for a in keep_i if a in asin2url]),
    }
    with open(os.path.join(HERE, "Books_filtered.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log(f"  清单已写: {os.path.join(HERE, 'Books_filtered.json')}")

    if not args.skip_images:
        if args.limit_images:
            asyncio.run(download_images(asin2url, list(keep_i), args.img_dir,
                                        args.concurrency, limit=args.limit_images))
        else:
            asyncio.run(download_images(asin2url, list(keep_i), args.img_dir, args.concurrency))

    log("全部完成 ✔")


if __name__ == "__main__":
    main()
