#!/usr/bin/env python3
"""重试下载 Books 商品图 (DNS 高可用 + 断点续传 + 实时日志).

从 Books_images_failed.jsonl 读取待下载清单, 高并发下载每商品一张 MAIN 图到
Books_images/{parent_asin}_MAIN.jpg.

健壮性设计 (针对上次全挂的 DNS 故障):
  * 绕过 systemd-resolved, 直连公共 DNS (8.8.8.8 / 114.114.114.114 / 223.5.5.5)
  * 每张图最多 4 轮尝试, 404/403 自动换下一质量档 URL
  * 连续失败过多时自动暂停 60s 再继续 (扛 DNS/网络抖动), 不会整体退出
  * 已存在且合法 (JPEG, >=500B) 的图片自动跳过
  * 最终失败清单写到 Books_images_failed2.jsonl

用法:
  python3 books_images_retry.py --failed_list Books_images_failed.jsonl \
      --out_dir Books_images --concurrency 150
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import socket
import subprocess
import time
from datetime import datetime

import aiohttp

HERE = os.path.dirname(os.path.abspath(__file__))

NAMESERVERS = ["8.8.8.8", "114.114.114.114", "223.5.5.5"]


def log(msg):
    print(f"[{datetime.now().strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def _dig_a(host, ns):
    out = subprocess.run(["dig", "+short", "+time=3", "+tries=2", "A", host, f"@{ns}"],
                         capture_output=True, text=True, timeout=20).stdout
    ips, cnames = [], []
    for x in out.splitlines():
        x = x.strip().rstrip(".")
        if not x:
            continue
        try:
            socket.inet_aton(x)
            ips.append(x)
        except OSError:
            cnames.append(x)
    return ips, cnames


def resolve_host(host):
    """dig 公共 DNS 解析 IPv4 (沿 CNAME 链), 失败再退回系统解析."""
    ips = set()
    for ns in NAMESERVERS:
        try:
            cur_ips, cnames = _dig_a(host, ns)
            ips.update(cur_ips)
            for c in cnames[:3]:
                sub_ips, sub_cnames = _dig_a(c, ns)
                ips.update(sub_ips)
                for c2 in sub_cnames[:2]:
                    ips.update(_dig_a(c2, ns)[0])
        except Exception:
            continue
    if ips:
        return sorted(ips)
    try:
        infos = socket.getaddrinfo(host, 443, family=socket.AF_INET)
        return sorted({i[4][0] for i in infos})
    except OSError:
        return []


class CachedResolver(aiohttp.abc.AbstractResolver):
    """按预解析的 host->IP 映射直连, 不经过 systemd-resolved."""

    def __init__(self, mapping):
        self.mapping = mapping

    async def resolve(self, host, port=0, family=socket.AF_INET):
        ips = self.mapping.get(host) or self.mapping.get(host.rstrip("."))
        if not ips:
            raise OSError(f"no cached IPs for {host}")
        return [
            {
                "hostname": host,
                "host": ip,
                "port": port,
                "family": socket.AF_INET,
                "proto": socket.IPPROTO_TCP,
                "flags": socket.AI_NUMERICHOST,
            }
            for ip in ips
        ]

    async def close(self):
        pass


def image_ok(path, min_bytes=500):
    try:
        if os.path.getsize(path) < min_bytes:
            return False
        with open(path, "rb") as f:
            return f.read(2) == b"\xff\xd8"
    except OSError:
        return False


def _write_bytes(path, data):
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _jpeg_url(url):
    """改写 URL 强制返回 JPEG (去 FMwebp), 同时保留尺寸参数."""
    url = re.sub(r"_QL(\d+)_FMwebp_", r"_QL\1_", url)
    url = url.replace("_FMwebp_", "_").replace("FMwebp", "")
    return url


async def _fetch_one(entry, out_dir, session, sem):
    asin = entry["parent_asin"]
    urls = [_jpeg_url(u) for u in entry["urls"]]
    path = os.path.join(out_dir, f"{asin}_MAIN.jpg")
    if image_ok(path):
        return "exists"
    for attempt in range(4):
        for url in urls:
            try:
                async with sem:
                    async with session.get(
                        url,
                        timeout=aiohttp.ClientTimeout(total=15, connect=4, sock_read=10),
                        headers={
                            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AmazonImages/1.0",
                            "Accept": "image/jpeg,image/pjpeg,image/*;q=0.6,*/*;q=0.3",
                        },
                    ) as r:
                        if r.status == 200:
                            data = await r.content.read()
                            if len(data) >= 500 and data[:2] == b"\xff\xd8":
                                await asyncio.get_running_loop().run_in_executor(
                                    None, _write_bytes, path, data
                                )
                                return "ok"
                        elif r.status in (404, 403, 410):
                            continue
                        elif r.status == 429 or r.status >= 500:
                            await asyncio.sleep(1.0 + random.random() * 2)
                        else:
                            pass
            except (asyncio.TimeoutError, aiohttp.ClientConnectorError, aiohttp.ServerDisconnectedError):
                pass
            except Exception:
                pass
            await asyncio.sleep(0.15 + random.random() * 0.3)
        await asyncio.sleep(0.5 * (attempt + 1))
    return "fail"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--failed_list", default=os.path.join(HERE, "Books_images_failed.jsonl"))
    ap.add_argument("--out_dir", default=os.path.join(HERE, "Books_images"))
    ap.add_argument("--concurrency", type=int, default=150)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pause_after_fails", type=int, default=300, help="连续失败该数量后暂停60s")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log(f"读取清单: {args.failed_list}")
    entries = [json.loads(line) for line in open(args.failed_list, encoding="utf-8") if line.strip()]
    log(f"清单共 {len(entries)} 条")

    todo = [e for e in entries if not image_ok(os.path.join(args.out_dir, f"{e['parent_asin']}_MAIN.jpg"))]
    if args.limit:
        todo = todo[: args.limit]
    log(f"待下载: {len(todo)}  并发: {args.concurrency}  -> {os.path.abspath(args.out_dir)}")

    q: asyncio.Queue = asyncio.Queue()
    for e in todo:
        await q.put(e)

    hosts = sorted({u.split("/")[2] for e in todo for u in e["urls"] if "://" in u})
    log(f"  涉及 host: {hosts}")
    host2ips = {}
    for h in hosts:
        ips = []
        for _ in range(24):  # DNS 抖动时反复重试, 最长约 2 分钟
            ips = resolve_host(h)
            if ips:
                break
            time.sleep(5)
        host2ips[h] = ips
        log(f"    {h} -> {ips[:4]}{'...' if len(ips) > 4 else ''} ({len(ips)} 个IP)")
    if not any(host2ips.values()):
        log("[error] 所有 host 都无法解析, 退出")
        return

    resolver = CachedResolver(host2ips)
    log(f"  预解析 host -> IP: { {h: ips[:2] for h, ips in host2ips.items()} }")
    connector = aiohttp.TCPConnector(limit=args.concurrency * 2, resolver=resolver,
                                     ttl_dns_cache=60, keepalive_timeout=60,
                                     family=socket.AF_INET)
    sem = asyncio.Semaphore(args.concurrency)
    stats = {"ok": 0, "exists": 0, "fail": 0}
    fail_streak = 0
    failed_entries = []
    lock = asyncio.Lock()
    t0 = time.time()
    last = [t0, 0]

    async def worker():
        nonlocal fail_streak
        while True:
            try:
                entry = q.get_nowait()
            except asyncio.QueueEmpty:
                return
            status = await _fetch_one(entry, args.out_dir, session, sem)
            async with lock:
                stats[status] += 1
                if status == "fail":
                    fail_streak += 1
                    failed_entries.append(entry)
                else:
                    fail_streak = 0
                done = sum(stats.values())
                if done % 500 == 0 or done == len(todo):
                    dt = time.time() - last[0]
                    rate = (done - last[1]) / dt if dt > 0 else 0
                    last[0], last[1] = time.time(), done
                    log(f"   {done}/{len(todo)} ({(time.time()-t0)/60:.1f} min, {rate:.1f} 张/s) "
                        f"ok={stats['ok']} exists={stats['exists']} fail={stats['fail']} "
                        f"连续失败={fail_streak}")
                if fail_streak >= args.pause_after_fails:
                    fail_streak = 0
                    log("   连续失败过多, 暂停 60s (DNS/网络恢复中)...")
                    await asyncio.sleep(60)
                    log("   继续下载")

    async def refresh_dns():
        while True:
            await asyncio.sleep(600)
            for h in hosts:
                ips = resolve_host(h)
                if ips:
                    host2ips[h] = ips
                    log(f"   [DNS刷新] {h} -> {len(ips)} 个IP")

    async with aiohttp.ClientSession(connector=connector) as session:
        n_workers = min(args.concurrency, max(1, len(todo)))
        workers = [asyncio.create_task(worker()) for _ in range(n_workers)]
        refresher = asyncio.create_task(refresh_dns())
        await asyncio.gather(*workers)
        refresher.cancel()

    log(f"完成: ok={stats['ok']} exists={stats['exists']} fail={stats['fail']} "
        f"总耗时 {(time.time()-t0)/60:.1f} min")
    if failed_entries:
        fail_path = args.failed_list.replace("_images_todo.jsonl", "_images_failed.jsonl")
        if fail_path == args.failed_list:
            fail_path = os.path.join(HERE, "images_failed.jsonl")
        with open(fail_path, "w", encoding="utf-8") as f:
            for e in failed_entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        log(f"仍失败 {len(failed_entries)} 条 -> {fail_path}")
    log("全部完成")


if __name__ == "__main__":
    asyncio.run(main())
