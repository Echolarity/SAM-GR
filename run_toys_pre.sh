#!/usr/bin/env bash
# Toys_and_Games 前期阶段 (2-4): SASRec -> text2emb -> RQ-VAE, 在 GPU1 上跑,
# 与 Sports(阶段3-7, GPU2/3) 并行。之后由 run_toys_post.sh 接力阶段5-7。
set -uo pipefail
cd "$(dirname "$0")"
export PATH="/data/sht/conda/mimigenrec/bin:$PATH"

export CATEGORY=Toys_and_Games
export MODAL_CHOICE=csga_moe_collab_v2
export MODEL_NAME=Qwen3VL
export TEXT2EMB_MODEL=/data02/qy/models/Qwen3-VL-8B-Instruct
export MOE_EPOCHS=30
export EMB_BATCH_SIZE=4
export TEXT_WEIGHT="-1.0"
export COMBINE_ORIG_TEXT=true
export CUDA_VISIBLE=1
export MONO_CUDA=1
ROOT="./data/Amazon/Toys_and_Games"

export_nccl_env() {
  export NCCL_DEBUG=INFO
  export TORCH_NCCL_BLOCKING_WAIT=1
  export NCCL_CUMEM_HOST_ENABLE=0
  export NCCL_SHM_DISABLE=1
  export NCCL_P2P_DISABLE=1
  export NCCL_IB_DISABLE=1
}

echo "== Toys 阶段2 SASRec =="
if [ -f ./sasrec_models/Toys_and_Games/sasrec_item_emb.npy ]; then
  echo "[skip] SASRec emb 已存在"
else
  mkdir -p ./sasrec_models
  CUDA_VISIBLE_DEVICES="${MONO_CUDA}" bash scripts/rec_zoo/train_sasrec.sh \
    --data_dir "${ROOT}" --epoch 10000 --batch_size 128 --hidden_factor 128 \
    --seq_size 50 --lr 0.001 --early_stop 50 --save_dir ./sasrec_models/Toys_and_Games
fi

echo "== Toys 阶段3 text2emb (单卡) =="
export_nccl_env
RANDOM_PORT=$((20000 + RANDOM % 40000))
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE}" torchrun \
  --nproc_per_node=1 --master_port="${RANDOM_PORT}" \
  rq/text2emb/multimodal_amazon_text2emb.py \
  --dataset "${CATEGORY}" --root "${ROOT}" \
  --plm_checkpoint "${TEXT2EMB_MODEL}" --plm_name "${MODEL_NAME}" \
  --mode "${MODAL_CHOICE}" --batch_size "${EMB_BATCH_SIZE}" \
  --max_sent_len 1024 --moe_epochs "${MOE_EPOCHS}" \
  --text_weight "${TEXT_WEIGHT}" --combine_orig_text \
  --collab_emb_path ./sasrec_models/Toys_and_Games/sasrec_item_emb.npy

echo "== Toys 阶段4 RQ-VAE =="
mkdir -p "./rq_output/${CATEGORY}"
CUDA_VISIBLE_DEVICES="${MONO_CUDA}" python rq/rqvae.py \
  --data_path "${ROOT}/${CATEGORY}.emb-${MODEL_NAME}-${MODAL_CHOICE}.npy" \
  --ckpt_dir "./rq_output/${CATEGORY}" \
  --lr 1e-3 --epochs 5000 --batch_size 20480 \
  --num_emb_list 256 256 256 256 --sk_epsilons 0.0 0.0 0.0 0.0

echo "TOYS_PRE_DONE"
