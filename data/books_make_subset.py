#!/usr/bin/env python3
"""从全局 5-core 的 Books_5core.jsonl.gz 按物品热度取 Top-K 子集.

输出:
  Books_top{K}.jsonl.gz   保留的 review 行 (原格式)
  Books_top{K}.manifest.json  统计 (items/users/interactions)

用法: python3 books_make_subset.py --k 1500
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import subprocess

try:
    import orjson

    parse = orjson.loads

    def dump(o) -> str:
        return orjson.dumps(o).decode("utf-8")

except ImportError:
    parse = json.loads

    def dump(o) -> str:
        return json.dumps(o, ensure_ascii=False)


HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "Books_5core.jsonl.gz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=1500)
    args = ap.parse_args()

    item_freq = collections.Counter()
    print("[1/3] 统计 item 频次 (流式)", flush=True)
    proc = subprocess.Popen(["pigz", "-dc", "-p", "12", SRC],
                            stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
    for line in proc.stdout:
        try:
            r = parse(line)
        except Exception:
            continue
        it = r.get("parent_asin") or r.get("asin")
        if it:
            item_freq[it] += 1
    proc.wait()

    top = {i for i, _ in item_freq.most_common(args.k)}
    print(f"[2/3] Top-{args.k} 物品已选 (最小频次 {item_freq.most_common(args.k)[-1][1]})", flush=True)

    out_gz = os.path.join(HERE, f"Books_top{args.k}.jsonl.gz")
    n_w = n_inter = 0
    users = set()
    with gzip.open(out_gz, "wt", encoding="utf-8", compresslevel=1) as f:
        proc = subprocess.Popen(["pigz", "-dc", "-p", "12", SRC],
                                stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
        for line in proc.stdout:
            try:
                r = parse(line)
            except Exception:
                continue
            it = r.get("parent_asin") or r.get("asin")
            if it in top:
                f.write(dump(r))
                f.write("\n")
                n_w += 1
                users.add(r.get("user_id"))
        proc.wait()

    manifest = {
        "k": args.k,
        "items": len(top),
        "interactions": n_w,
        "users": len(users),
        "min_item_freq": item_freq.most_common(args.k)[-1][1],
    }
    with open(os.path.join(HERE, f"Books_top{args.k}.manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[3/3] 写出 {n_w} 条 -> {out_gz}", flush=True)
    print(manifest, flush=True)


if __name__ == "__main__":
    main()
