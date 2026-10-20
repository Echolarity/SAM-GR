#!/usr/bin/env bash
# Toys_and_Games 阶段5-7 (RQ-VAE 已完成), 在 GPU2/3 空闲后运行.
set -uo pipefail
cd "$(dirname "$0")"

PATH="/data/sht/conda/mimigenrec/bin:$PATH" \
CATEGORY=Toys_and_Games \
MODAL_CHOICE=csga_moe_collab_v2 \
USER_K=1 ITEM_K=1 \
ONLY_TASK1=1 \
Optimization=dsz2_csga_omoe_v2 \
SID_MERGE_RATIO=0.15 \
CUDA_VISIBLE=2,3 \
MONO_CUDA=2 \
EVAL_CUDA_LIST='2 3' \
START_STAGE=5 \
bash main.sh
echo "TOYS_DONE rc=$?"
