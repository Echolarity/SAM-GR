#!/usr/bin/env bash
set -euo pipefail

# 从环境变量安全导入 (为空时告警并回落默认值)
get_env_or_default() {
    local name="$1" default="$2"
    local val="${!name:-}"
    if [ -z "$val" ]; then
        echo "[warn] 环境变量 $name 未设置, 使用默认值: $default" >&2
        echo "$default"
    else
        echo "$val"
    fi
}

export DISABLE_VERSION_CHECK=1
export CUDA_VISIBLE_DEVICES="$(get_env_or_default CUDA_VISIBLE 0,1,2,3)"
export CATEGORY="$(get_env_or_default CATEGORY Industrial_and_Scientific)"
export Optimization="$(get_env_or_default Optimization dsz3)"
export SFT_YAML="$(get_env_or_default SFT_YAML examples/train_full/${CATEGORY}/new_industry_rec_full_sft_0.5b_${Optimization}.yaml)"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WANDB_PROJECT="MIMIGenRec-SFT"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

# NCCL 环境 (启用 SHM/P2P 以获得最高通信带宽; 之前禁用导致走网络极慢)
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_IB_DISABLE=1
export LC_ALL=en_US.UTF-8
export LC_CTYPE=en_US.UTF-8
export PYTORCH_ALLOC_CONF=expandable_segments:True

if [ ! -f "$SFT_YAML" ]; then
    echo "[error] SFT yaml 不存在: $SFT_YAML" >&2
    exit 1
fi

echo "SFT: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES yaml=$SFT_YAML"
set -x
llamafactory-cli train "$SFT_YAML"
