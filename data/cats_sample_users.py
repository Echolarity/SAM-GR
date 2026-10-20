#!/usr/bin/env python3
"""从 {CAT}_core.jsonl.gz 随机采样 p 比例的用户, 输出 {CAT}_core_sampled.jsonl.gz.

目的: 把交互量压到 Industry 量级(~50万), 保持 30K 物品全部保留.
用法: python3 cats_sample_users.py --category Sports_and_Outdoors --p 0.05 --seed 42
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import random
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    ap.add_argument("--p", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_reviews", type=int, default=5, help="只从评论数>=N的用户中抽样")
    args = ap.parse_args()

    src = os.path.join(HERE, f"{args.category}_core.jsonl.gz")
    dst = os.path.join(HERE, f"{args.category}_core_sampled.jsonl.gz")

    user_counts = collections.Counter()
    proc = subprocess.Popen(["pigz", "-dc", "-p", "12", src],
                            stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
    for line in proc.stdout:
        try:
            r = parse(line)
        except Exception:
            continue
        u = r.get("user_id")
        if u:
            user_counts[u] += 1
    proc.wait()

    eligible = [u for u, c in user_counts.items() if c >= args.min_reviews]
    rng = random.Random(args.seed)
    kept_users = {u for u in eligible if rng.random() < args.p}
    exp_inter = sum(user_counts[u] for u in kept_users)
    print(f"总用户 {len(user_counts)}, 评论>=N 用户 {len(eligible)}, 抽样 {len(kept_users)} 人, "
          f"预期交互 {exp_inter}", flush=True)

    n_w = 0
    items = set()
    with gzip.open(dst, "wt", encoding="utf-8", compresslevel=1) as f:
        proc = subprocess.Popen(["pigz", "-dc", "-p", "12", src],
                                stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
        for line in proc.stdout:
            try:
                r = parse(line)
            except Exception:
                continue
            if r.get("user_id") in kept_users:
                f.write(dump(r))
                f.write("\n")
                n_w += 1
                items.add(r.get("parent_asin") or r.get("asin"))
        proc.wait()
    print(f"写出 {n_w} 条, 物品 {len(items)} -> {dst}", flush=True)


if __name__ == "__main__":
    main()
