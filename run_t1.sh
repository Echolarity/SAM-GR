#!/usr/bin/env bash
# =============================================================================
# A1+A2 实验驱动: 仅 task1 (history->SID) + 统一 train/test 模板 + 2 epoch
#   * 使用新实验名 (…-sft-dsz2-t1) 以保留原 dsz2 基线, 便于对比
#   * 不改动模型结构与评估流程, 仅按顺序调用既有 sft.sh / evaluate.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export CATEGORY="Industrial_and_Scientific"
export Optimization="dsz2"
export SFT_YAML="examples/train_full/${CATEGORY}/new_industry_rec_full_sft_0.5b_${Optimization}_t1.yaml"
export EVAL_EXP_NAME="newsaves/qwen2.5-0.5b/full/${CATEGORY}-sft-${Optimization}-t1"

echo "=== [t1] SFT 开始: ${SFT_YAML} ==="
bash sft.sh

echo "=== [t1] 评估开始: ${EVAL_EXP_NAME} ==="
bash evaluate.sh

echo "=== [t1] 全部完成 ==="
