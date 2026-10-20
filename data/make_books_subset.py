import subprocess, sys, os, zlib
try:
    import orjson
    loads = orjson.loads
except ImportError:
    import json
    loads = json.loads

REV = "data/Books_5core.jsonl.gz"
META = "data/meta_Books.jsonl.gz"
MOD = 20  # keep 1/20 of users

def stream(path):
    p = subprocess.Popen(["pigz", "-dc", "-p", "8", path], stdout=subprocess.PIPE, text=True, bufsize=1<<20)
    for line in p.stdout:
        yield line
    p.stdout.close(); p.wait()

items = set(); n = 0; kept = 0
with open("data/Books_sub_items.txt", "w") as fi, __import__("gzip").open("data/Books_sub_reviews.jsonl.gz", "wt") as fo:
    for line in stream(REV):
        n += 1
        try:
            d = loads(line)
        except Exception:
            continue
        u = d.get("user_id", "")
        if zlib.crc32(u.encode()) % MOD != 0:
            continue
        a = d.get("parent_asin") or d.get("asin")
        items.add(a)
        fi.write(a + "\n")
        fo.write(line if line.endswith("\n") else line + "\n")
        kept += 1
print(f"reviews: total={n} kept={kept} items={len(items)}", flush=True)

meta_kept = 0
with __import__("gzip").open("data/Books_sub_meta.jsonl.gz", "wt") as fo:
    for line in stream(META):
        try:
            d = loads(line)
        except Exception:
            continue
        if d.get("parent_asin") in items:
            fo.write(line if line.endswith("\n") else line + "\n")
            meta_kept += 1
print(f"meta kept={meta_kept}", flush=True)
