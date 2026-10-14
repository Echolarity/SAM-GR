# -*- coding: utf-8 -*-
"""
本地(Windows + VPN) 下载 2018 Amazon 数据集: reviews(5-core) + meta
适合在能快速访问外网的机器上跑, 再把 amazon18/ 目录拷回服务器。

用法:
    pip install requests tqdm      # tqdm 可选, 不装也能跑
    python download_amazon18_local.py

依赖: requests (必需), tqdm (可选)
特性: 多连接分块下载(快), 断点续传(每个分块独立), 失败自动重试, 已存在自动跳过
"""

import os
import sys
import time
import gzip
import shutil
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    sys.exit("缺少依赖: 请先运行  pip install requests")

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None


# ========================= 配置区 (直接改这里) =========================
CATEGORIES = [
    "Industrial_and_Scientific",
    "Office_Products",
    "Toys_and_Games",
    # 想加别的就加进来, 例如:
    # "Sports_and_Outdoors",
    # "All_Beauty",
]
OUT_DIR    = "./amazon18"   # 输出目录 (脚本会在此下建 <类别>/ 子目录)
CONNS      = 32             # 每个数据文件的多连接数 (VPN 快的话可调到 64)
TIMEOUT    = 60             # 单次请求超时(秒)
RETRIES    = 8              # 每个分块的最大重试轮数
DECOMPRESS = True           # 下完自动解压 .gz -> .json
# =====================================================================

REV_BASE  = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall"
META_BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2"


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def head_size(url):
    """取文件大小 (用于分块)."""
    for _ in range(3):
        try:
            r = requests.head(url, timeout=20, allow_redirects=True)
            if r.status_code == 200 and "content-length" in r.headers:
                return int(r.headers["content-length"])
        except Exception:
            time.sleep(1 + random.random())
    return None


def fetch_range(url, part, start, end, timeout):
    """下载 [start,end] 区间到 part, 支持续传. 返回是否完成."""
    expect = end - start + 1
    have = os.path.getsize(part) if os.path.exists(part) else 0
    if have >= expect:
        return True
    try:
        hdr = {"Range": f"bytes={start + have}-{end}"}
        with requests.get(url, headers=hdr, stream=True, timeout=timeout) as r:
            if r.status_code not in (200, 206):
                return False
            if r.status_code == 200 and have > 0:
                have = 0  # 服务端忽略 Range, 重头写
            with open(part, "ab" if have > 0 else "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
    except Exception:
        return False
    return os.path.exists(part) and os.path.getsize(part) >= expect


def download_file(url, dst, conns=32, retries=8, timeout=60):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"  [跳过] 已存在 {os.path.basename(dst)} ({human(os.path.getsize(dst))})")
        return True

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    name = os.path.basename(dst)
    size = head_size(url)

    # --- 单连接回退 (取不到大小时) ---
    if not size:
        print(f"  [下载] {name}  (单连接)")
        tmp = dst + ".part"
        for _ in range(retries):
            try:
                with requests.get(url, stream=True, timeout=timeout) as r:
                    r.raise_for_status()
                    with open(tmp, "wb") as f:
                        for c in r.iter_content(chunk_size=1 << 20):
                            f.write(c)
                os.replace(tmp, dst)
                print(f"  [完成] {name}  ({human(os.path.getsize(dst))})")
                return True
            except Exception:
                time.sleep(2 + random.random())
        print(f"  [失败] {name}")
        return False

    # --- 多连接分块下载 ---
    tmpdir = dst + ".parts"
    os.makedirs(tmpdir, exist_ok=True)
    n = max(1, min(conns, (size + (256 << 10) - 1) // (256 << 10)))
    chunk = (size + n - 1) // n
    ranges = [(i * chunk, min((i + 1) * chunk - 1, size - 1)) for i in range(n)]
    print(f"  [下载] {name}  {human(size)}  分 {n} 块并发")

    def complete():
        return all(
            os.path.exists(os.path.join(tmpdir, f"{i}.part"))
            and os.path.getsize(os.path.join(tmpdir, f"{i}.part")) == (e - s + 1)
            for i, (s, e) in enumerate(ranges)
        )

    t0 = time.time()
    for _wave in range(retries):
        if complete():
            break
        with ThreadPoolExecutor(max_workers=n) as pool:
            futs = [
                pool.submit(fetch_range, url, os.path.join(tmpdir, f"{i}.part"), s, e, timeout)
                for i, (s, e) in enumerate(ranges)
            ]
            it = as_completed(futs)
            if _tqdm:
                it = _tqdm(it, total=n, desc=f"    {name}", leave=False)
            for _ in it:
                pass
        done = sum(
            1 for i, (s, e) in enumerate(ranges)
            if os.path.exists(os.path.join(tmpdir, f"{i}.part"))
            and os.path.getsize(os.path.join(tmpdir, f"{i}.part")) == (e - s + 1)
        )
        print(f"    进度 {done}/{n} 块")

    if not complete():
        print(f"  [未完成] {name}  (已保留 .parts, 重跑本脚本可续传)")
        return False

    with open(dst, "wb") as out:
        for i in range(n):
            with open(os.path.join(tmpdir, f"{i}.part"), "rb") as f:
                shutil.copyfileobj(f, out)
    shutil.rmtree(tmpdir, ignore_errors=True)

    dt = time.time() - t0
    sp = os.path.getsize(dst) / dt / 1024 if dt else 0
    print(f"  [完成] {name}  {human(os.path.getsize(dst))}  用时 {dt:.0f}s  ({sp:.0f} kB/s)")
    return True


def decompress(gz, js):
    if os.path.exists(js) and os.path.getsize(js) > 0:
        return
    with gzip.open(gz, "rb") as fi, open(js, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    print(f"  [解压] {os.path.basename(js)}")


def main():
    print("=" * 64)
    print("2018 Amazon 数据集下载 (reviews + meta)")
    print("  类别 :", ", ".join(CATEGORIES))
    print("  输出 :", os.path.abspath(OUT_DIR))
    print("  连接 :", CONNS, " 解压:", DECOMPRESS)
    print("=" * 64)

    results = {}
    for cat in CATEGORIES:
        print(f"\n===== {cat} =====")
        d = os.path.join(OUT_DIR, cat)
        os.makedirs(d, exist_ok=True)

        rev_gz = os.path.join(d, f"{cat}_5.json.gz")
        meta_gz = os.path.join(d, f"meta_{cat}.json.gz")

        ok_rev = download_file(f"{REV_BASE}/{cat}_5.json.gz", rev_gz, CONNS, RETRIES, TIMEOUT)
        ok_meta = download_file(f"{META_BASE}/meta_{cat}.json.gz", meta_gz, CONNS, RETRIES, TIMEOUT)

        if DECOMPRESS and ok_rev and ok_meta:
            decompress(rev_gz, os.path.join(d, f"{cat}_5.json"))
            decompress(meta_gz, os.path.join(d, f"meta_{cat}.json"))

        results[cat] = ok_rev and ok_meta

    print("\n" + "=" * 64)
    print("汇总:")
    for c, ok in results.items():
        print(f"  {c:<28} {'OK' if ok else '失败(可重跑续传)'}")
    print("=" * 64)
    print("把整个 'amazon18' 目录拷到服务器项目下 (MIMIGenRec/data/amazon18) 即可.")
    print("服务器端会跳过已存在文件, 只补图片.")


if __name__ == "__main__":
    main()
