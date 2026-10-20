#!/usr/bin/env bash
# =============================================================================
# MIMI (多模态 MIMIGenRec) 一键编排脚本
#   阶段1  possess_data           -> item.json + *.inter + 图片
#   阶段2  SASRec (collab 模式)    -> sasrec_item_emb.npy
#   阶段3  multimodal_text2emb     -> *.emb-Qwen3VL-<MODAL>.npy
#   阶段4  RQ-VAE (4 层 codebook)  -> best_collision_model.pth
#   阶段5  generate_indices        -> *.index.json
#   阶段6  preprocess (sid_num=4)  -> sft/ + rl/ + new_tokens.json
#   阶段7  SFT + 评估
#
# 断点续跑: 设 START_STAGE=N 可从第 N 阶段开始 (前面阶段跳过, 但会校验其产物存在)
#   例: START_STAGE=7 bash nohup_run.sh run pipeline bash main.sh
# =============================================================================
set -euo pipefail

# --------------------------------------------------------------------------- #
# 顶部配置区
# --------------------------------------------------------------------------- #
export CATEGORY="${CATEGORY:-Industrial_and_Scientific}"
export POSSESS_DATA_OUTPATH="./data/Amazon"
export RAW_DATA_DIR="${RAW_DATA_DIR:-./data}"   # 原始 jsonl 数据目录 (与参考项目一致: 直接放 data/)
export IMG_DIR="${IMG_DIR:-${RAW_DATA_DIR}/${CATEGORY}_images}"
export CUDA_VISIBLE="${CUDA_VISIBLE:-0,1,2,3}"      # 多卡 (text2emb / SFT)
export MONO_CUDA="${MONO_CUDA:-0}"                  # 单卡 (rqvae / generate_indices)

export Optimization="${Optimization:-dsz2}"
export SFT_YAML="examples/train_full/${CATEGORY}/new_industry_rec_full_sft_0.5b_${Optimization}.yaml"
export EVAL_EXP_NAME="newsaves/qwen2.5-0.5b/full/${CATEGORY}-sft-${Optimization}"

# 核心模态控制器: multimodal, text_only, image_only, concat, image2text,
#   image2text_supp, moe_decouple, contrastive_adapter,
#   multimodal_collab, concat_collab, moe_collab
export MODAL_CHOICE="${MODAL_CHOICE:-text_only}"

export TEXT_WEIGHT="-1.0"         # >=0 时退化为加权均值/加权拼接
export COMBINE_ORIG_TEXT="true"   # image2text 是否拼接原文
export MOE_EPOCHS=30
export EMB_BATCH_SIZE="${EMB_BATCH_SIZE:-4}"

export SASREC_EPOCHS=10000
export SASREC_BATCH_SIZE="${SASREC_BATCH_SIZE:-128}"
export SASREC_HIDDEN="${SASREC_HIDDEN:-128}"
export SASREC_SEQ="${SASREC_SEQ:-50}"
export SASREC_LR="${SASREC_LR:-0.001}"
export SASREC_EARLY_STOP="${SASREC_EARLY_STOP:-50}"
export SASREC_SAVE_DIR="./sasrec_models/${CATEGORY}"

export SAMPLE_SIZE="${SAMPLE_SIZE:--1}"   # -1 = 全量
export USER_K="${USER_K:-5}"
export ITEM_K="${ITEM_K:-5}"
export TASK4_SAMPLE=-1
export SEED=42

export TEXT2EMB_MODEL="${TEXT2EMB_MODEL:-/data02/qy/models/Qwen3-VL-8B-Instruct}"
export MODEL_NAME="Qwen3VL"

export OUTPUT_DIR="./data/${CATEGORY}"

# ---- 断点续跑: 从第几个阶段开始 (1..7); 默认 1=从头 ----
export START_STAGE="${START_STAGE:-1}"

# ---- 阶段6 数据构建: 1=只生成 task1 (history->SID) 消融数据, 0=全部任务(默认) ----
export ONLY_TASK1="${ONLY_TASK1:-0}"

# ---- 阶段5 可控 SID 合并: 将比例 SID_MERGE_RATIO 的低频物品并入更热门的协同邻居 (0=关闭) ----
export SID_MERGE_RATIO="${SID_MERGE_RATIO:-0}"

# --------------------------------------------------------------------------- #
# 颜色 / 计时 / 检查
# --------------------------------------------------------------------------- #
red()    { echo -e "\033[31m$@\033[0m"; }
green()  { echo -e "\033[32m$@\033[0m"; }
yellow() { echo -e "\033[33m$@\033[0m"; }
blue()   { echo -e "\033[34m$@\033[0m"; }

check_path() {
    if [ ! -e "$1" ]; then
        red "错误: 路径不存在: $1"
        exit 1
    fi
}

# 是否运行第 N 阶段
should_run() { [ "${START_STAGE}" -le "$1" ]; }

STEP_START=0
step_begin() {
    STEP_START=$(date +%s)
    blue "========================================"
    blue ">> $1"
    blue "========================================"
}
step_end() {
    local end=$(date +%s)
    green "<< $1 完成 (耗时 $((end - STEP_START))s)"
}
skip() { yellow "跳过 [$1] (START_STAGE=${START_STAGE})"; }

# NCCL 防卡死环境
export_nccl_env() {
    export NCCL_DEBUG=INFO
    export TORCH_NCCL_BLOCKING_WAIT=1
    export NCCL_CUMEM_HOST_ENABLE=0
    export NCCL_SHM_DISABLE=1
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1
    export LC_ALL=en_US.UTF-8
    export LC_CTYPE=en_US.UTF-8
}

# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
main() {
    ROOT="${POSSESS_DATA_OUTPATH}/${CATEGORY}"
    NPY_FILE="${ROOT}/${CATEGORY}.emb-${MODEL_NAME}-${MODAL_CHOICE}.npy"
    RQVAE_CKPT="./rq_output/${CATEGORY}/best_collision_model.pth"
    INDEX_FILE="${ROOT}/${CATEGORY}.index.json"

    blue "START_STAGE=${START_STAGE}  MODAL_CHOICE=${MODAL_CHOICE}  USER_K=${USER_K}/ITEM_K=${ITEM_K}"

    # ========== 阶段1: possess_data ==========
    if should_run 1; then
        step_begin "阶段1 possess_data (${CATEGORY})"
        python ./data/possess_data.py \
            --category "${CATEGORY}" \
            --sample_size "${SAMPLE_SIZE}" \
            --user_k "${USER_K}" \
            --item_k "${ITEM_K}" \
            --output_path "${POSSESS_DATA_OUTPATH}" \
            --raw_dir "${RAW_DATA_DIR}" \
            --img_dir "${IMG_DIR}"
        check_path "${ROOT}/${CATEGORY}.item.json"
        step_end "阶段1"
    else
        skip "阶段1 possess_data"
        check_path "${ROOT}/${CATEGORY}.item.json"
    fi

    # ========== 阶段2: SASRec collab 特征 ==========
    if should_run 2; then
        if [[ "$MODAL_CHOICE" == *collab* ]]; then
            step_begin "阶段2 SASRec (collab 特征)"
            mkdir -p "${SASREC_SAVE_DIR}"
            CUDA_VISIBLE_DEVICES="${MONO_CUDA}" bash scripts/rec_zoo/train_sasrec.sh \
                --data_dir "${ROOT}" \
                --epoch "${SASREC_EPOCHS}" \
                --batch_size "${SASREC_BATCH_SIZE}" \
                --hidden_factor "${SASREC_HIDDEN}" \
                --seq_size "${SASREC_SEQ}" \
                --lr "${SASREC_LR}" \
                --early_stop "${SASREC_EARLY_STOP}" \
                --save_dir "${SASREC_SAVE_DIR}"
            check_path "${SASREC_SAVE_DIR}/sasrec_item_emb.npy"
            step_end "阶段2"
        else
            yellow "跳过阶段2 (非 collab 模式)"
        fi
    else
        skip "阶段2 SASRec"
    fi

    # ========== 阶段3: multimodal text2emb ==========
    if should_run 3; then
        step_begin "阶段3 multimodal text2emb (mode=${MODAL_CHOICE})"
        export_nccl_env
        check_path "${TEXT2EMB_MODEL}"

        NPROC=$(echo "${CUDA_VISIBLE}" | tr ',' ' ' | wc -w)
        RANDOM_PORT=$((20000 + RANDOM % 40000))

        CMD_ARGS=(
            --dataset "${CATEGORY}"
            --root "${ROOT}"
            --plm_checkpoint "${TEXT2EMB_MODEL}"
            --plm_name "${MODEL_NAME}"
            --mode "${MODAL_CHOICE}"
            --batch_size "${EMB_BATCH_SIZE}"
            --max_sent_len 1024
            --moe_epochs "${MOE_EPOCHS}"
            --text_weight "${TEXT_WEIGHT}"
        )
        [ "${COMBINE_ORIG_TEXT}" = "true" ] && CMD_ARGS+=(--combine_orig_text)
        if [[ "$MODAL_CHOICE" == *collab* ]]; then
            CMD_ARGS+=(--collab_emb_path "${SASREC_SAVE_DIR}/sasrec_item_emb.npy")
        fi

        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE}" torchrun \
            --nproc_per_node="${NPROC}" \
            --master_port="${RANDOM_PORT}" \
            rq/text2emb/multimodal_amazon_text2emb.py "${CMD_ARGS[@]}"

        check_path "${NPY_FILE}"
        green "嵌入文件: $(du -h "${NPY_FILE}" | cut -f1)"
        step_end "阶段3"
    else
        skip "阶段3 text2emb"
        check_path "${NPY_FILE}"
    fi

    # ========== 阶段4: RQ-VAE (4 层 codebook) ==========
    if should_run 4; then
        step_begin "阶段4 RQ-VAE (4 层 codebook)"
        mkdir -p "./rq_output/${CATEGORY}"
        CUDA_VISIBLE_DEVICES="${MONO_CUDA}" python rq/rqvae.py \
            --data_path "${NPY_FILE}" \
            --ckpt_dir "./rq_output/${CATEGORY}" \
            --lr 1e-3 \
            --epochs 5000 \
            --batch_size 20480 \
            --num_emb_list 256 256 256 256 \
            --sk_epsilons 0.0 0.0 0.0 0.0
        check_path "${RQVAE_CKPT}"
        step_end "阶段4"
    else
        skip "阶段4 RQ-VAE"
        check_path "${RQVAE_CKPT}"
    fi

    # ========== 阶段5: generate_indices ==========
    if should_run 5; then
        step_begin "阶段5 generate_indices"
        rm -f "${ROOT}"/*.index.json 2>/dev/null || true
        local merge_args=()
        if [ "${SID_MERGE_RATIO}" != "0" ] && [[ "$MODAL_CHOICE" == *collab* ]]; then
            merge_args=(--collab_emb_path "${SASREC_SAVE_DIR}/sasrec_item_emb.npy" \
                        --train_inter "${ROOT}/${CATEGORY}.train.inter" \
                        --merge_ratio "${SID_MERGE_RATIO}")
        fi
        CUDA_VISIBLE_DEVICES="${MONO_CUDA}" python rq/generate_indices.py \
            --dataset "${CATEGORY}" \
            --ckpt_path "${RQVAE_CKPT}" \
            --output_dir "${ROOT}" \
            --device "cuda:0" \
            "${merge_args[@]}"
        check_path "${INDEX_FILE}"
        step_end "阶段5"
    else
        skip "阶段5 generate_indices"
        check_path "${INDEX_FILE}"
    fi

    # ========== 阶段6: preprocess (sid_num=4) ==========
    if should_run 6; then
        step_begin "阶段6 preprocess SFT/RL (sid_num=4)"
        mkdir -p "${OUTPUT_DIR}"
        ITEM_JSON="${ROOT}/${CATEGORY}.item.json"
        GEN_ITEM_JSON="${ROOT}/${CATEGORY}.item-${MODAL_CHOICE}.json"

        run_preprocess() {
            local extra=()
            [ "${ONLY_TASK1}" = "1" ] && extra+=(--only_task1 True)
            python preprocess_data_sft_rl.py \
                --data_dir "${POSSESS_DATA_OUTPATH}" \
                --category "${CATEGORY}" \
                --output_dir "${OUTPUT_DIR}" \
                --seq_sample "${TASK4_SAMPLE}" \
                --seed "${SEED}" \
                --data_source "${CATEGORY}" \
                --sid_num 4 \
                "${extra[@]}"
        }

        # image2text 数据调包/复原
        if [[ "$MODAL_CHOICE" == "image2text" || "$MODAL_CHOICE" == "image2text_supp" ]] && [[ -f "$GEN_ITEM_JSON" ]]; then
            cp "${ITEM_JSON}" "${ITEM_JSON}.bak"
            cp "${GEN_ITEM_JSON}" "${ITEM_JSON}"
            run_preprocess
            mv "${ITEM_JSON}.bak" "${ITEM_JSON}"
            green "已复原 item.json"
        else
            run_preprocess
        fi
        step_end "阶段6"
    else
        skip "阶段6 preprocess"
        check_path "${OUTPUT_DIR}/sft/train.json"
        check_path "${OUTPUT_DIR}/new_tokens.json"
    fi

    # ---- (可选) codebook 先验注入; 实验时手动打开并改用 inject_*.yaml ----
    # python inject_priors.py --model_id Qwen/Qwen2.5-0.5B-Instruct ...

    # ========== 阶段7: SFT + 评估 ==========
    if should_run 7; then
        step_begin "阶段7 SFT + 评估"
        check_path "${SFT_YAML}"
        check_path "${OUTPUT_DIR}/new_tokens.json"
        bash sft.sh
        bash evaluate.sh
        step_end "阶段7"
    else
        skip "阶段7 SFT + 评估"
    fi

    blue "全部流程结束 ✔  输出目录: ${OUTPUT_DIR}"
}

main
