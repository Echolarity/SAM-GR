#!/usr/bin/env bash
# Sports_and_Outdoors 阶段3-7 (SASRec 已完成, emb 已保存), GPU2/3.
set -uo pipefail
cd "$(dirname "$0")"

PATH="/data/sht/conda/mimigenrec/bin:$PATH" \
CATEGORY=Sports_and_Outdoors \
MODAL_CHOICE=csga_moe_collab_v2 \
USER_K=1 ITEM_K=1 \
ONLY_TASK1=1 \
Optimization=dsz2_csga_omoe_v2 \
SID_MERGE_RATIO=0.15 \
CUDA_VISIBLE=2,3 \
MONO_CUDA=2 \
EVAL_CUDA_LIST='2 3' \
START_STAGE=3 \
bash main.sh
echo "SPORTS_DONE rc=$?"
