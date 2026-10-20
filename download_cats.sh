#!/usr/bin/env bash
# =============================================================================
# Sports_and_Outdoors + Toys_and_Games 连续处理:
#   1. gzip 完整性校验 -> 全量 5-core(与 Industry 一致) -> 超 30k 物品按热度裁剪
#   2. 输出每商品一张 MAIN 图的下载清单
#   3. 高速下载图片 (aiohttp 200 并发 + 公共DNS预解析IP直连 + 强制JPEG + 断点续传)
# 输出:
#   data/{CAT}_core.jsonl.gz           过滤后 reviews
#   data/{CAT}_images_todo.jsonl       下载清单
#   data/{CAT}_images/                 {asin}_MAIN.jpg
#   data/{CAT}_filtered.json           统计
# 用法: bash nohup_run.sh run cats_images_dl bash download_cats.sh
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

for CAT in Sports_and_Outdoors Toys_and_Games; do
  echo "############################################################"
  echo "########## [$CAT] 开始 ##########"
  echo "############################################################"

  echo ">>> 阶段1: 完整性校验 + 5-core + 图片清单"
  python3 -u data/cats_prepare.py \
    --category "${CAT}" \
    --reviews_file "data/${CAT}.jsonl.gz" \
    --meta_file "data/meta_${CAT}.jsonl.gz" \
    --user_k 5 --item_k 5 --top_k 30000 \
    || { echo "[$CAT] prepare 失败, 终止"; exit 1; }

  echo ">>> 阶段2: 高速下载 MAIN 图"
  python3 -u data/images_retry.py \
    --failed_list "data/${CAT}_images_todo.jsonl" \
    --out_dir "data/${CAT}_images" \
    --concurrency 200 \
    || { echo "[$CAT] 图片下载失败, 终止"; exit 1; }

  echo "########## [$CAT] 完成 ##########"
done

echo "ALL_DONE ✔"
