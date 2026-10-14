#!/usr/bin/env bash
# =============================================================================
# MIMIGenRec 一键顺序运行脚本
#   依次执行: 环境准备 -> (可选)数据预处理 -> SFT -> RL(GRPO) -> 评估
#   所有参数集中配置于 run_config.sh, 本脚本只需 source 它。
#   用法: bash run_all.sh
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
source "$HERE/run_config.sh"

log()  { echo -e "\n\033[1;36m[run_all]\033[0m $*"; }
fail() { echo -e "\033[1;31m[ERROR]\033[0m $*"; exit 1; }

# -----------------------------------------------------------------------------
# 1. 环境准备
# -----------------------------------------------------------------------------
log "激活 conda 环境: ${CONDA_ENV}"
CONDA_BASE="$(conda info --base 2>/dev/null || echo /home/sht/miniconda3)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTHON_BIN="$(command -v python)"
export HF_ENDPOINT WANDB_PROJECT WANDB_MODE WANDB_API_KEY DISABLE_VERSION_CHECK

log "安装 llamafactory (editable, 不重复装依赖) ..."
"$PYTHON_BIN" -m pip install -e . --no-deps -q

log "检查关键依赖 ..."
"$PYTHON_BIN" - <<'PY'
import importlib
for p in ["torch", "transformers", "trl", "accelerate", "peft", "datasets", "deepspeed"]:
    m = importlib.import_module(p)
    print(f"  {p:12s} {getattr(m, '__version__', '?')}")
import torch
print(f"  cuda_available = {torch.cuda.is_available()}  devices = {torch.cuda.device_count()}")
PY

# -----------------------------------------------------------------------------
# 2. (可选) 数据预处理
# -----------------------------------------------------------------------------
if [ "${RUN_DATA_PREP}" = "1" ]; then
  log "数据预处理: 下载并处理 Amazon 数据 (${CATEGORY})"
  bash data/amazon18_data_process_${CATEGORY}.sh || fail "数据预处理失败"
  bash preprocess_data_sft_rl.sh || fail "SFT/RL 数据生成失败"
else
  [ -f "data/${CATEGORY}/sft/train.json" ] || fail "缺少预处理数据 data/${CATEGORY}/sft/train.json (可置 RUN_DATA_PREP=1 重新生成)"
  [ -f "data/${CATEGORY}/id2sid.json"   ] || fail "缺少 id2sid.json"
  log "跳过数据预处理 (已存在预处理结果)"
fi

# -----------------------------------------------------------------------------
# 3. SFT 训练
# -----------------------------------------------------------------------------
if [ "${RUN_SFT}" = "1" ]; then
  DS_JSON="examples/deepspeed/ds_z0_config.json"
  [ "$DSZ" = "dsz2" ] && DS_JSON="examples/deepspeed/ds_z2_config.json"
  [ "$DSZ" = "dsz3" ] && DS_JSON="examples/deepspeed/ds_z3_config.json"

  SFT_YAML="configs/generated_sft_${CATEGORY}_${MODEL_SIZE}_${DSZ}.yaml"
  mkdir -p configs
  log "生成 SFT 配置 -> ${SFT_YAML}"

  if [ "${QUICK_TEST}" = "1" ]; then
    MAX_STEPS="${SFT_MAX_STEPS:-30}"
    cat > "$SFT_YAML" <<EOF
model_name_or_path: ${MODEL_NAME}
trust_remote_code: true
add_tokens_list: 'data/${CATEGORY}/new_tokens.json'
stage: sft
do_train: true
finetuning_type: full
deepspeed: ${DS_JSON}
dataset: ${CATEGORY}_train
eval_dataset: ${CATEGORY}_valid
cutoff_len: ${SFT_CUTOFF_LEN}
preprocessing_num_workers: 8
dataloader_num_workers: 2
output_dir: ${SFT_OUTPUT_DIR}
logging_steps: 1
save_steps: 15
save_strategy: steps
save_total_limit: 2
overwrite_output_dir: true
report_to: wandb
run_name: ${CATEGORY}-qwen2.5-${MODEL_SIZE}-sft-${DSZ}-quick
per_device_train_batch_size: ${SFT_PER_DEVICE_BATCH}
per_device_eval_batch_size: ${SFT_PER_DEVICE_BATCH}
gradient_accumulation_steps: ${SFT_GRAD_ACCUM}
learning_rate: ${SFT_LR}
max_steps: ${MAX_STEPS}
lr_scheduler_type: cosine
warmup_steps: ${SFT_WARMUP_STEPS}
bf16: true
ddp_timeout: 180000000
eval_strategy: steps
eval_steps: 15
EOF
  else
    cat > "$SFT_YAML" <<EOF
model_name_or_path: ${MODEL_NAME}
trust_remote_code: true
add_tokens_list: 'data/${CATEGORY}/new_tokens.json'
stage: sft
do_train: true
finetuning_type: full
deepspeed: ${DS_JSON}
dataset: ${CATEGORY}_train
eval_dataset: ${CATEGORY}_valid
cutoff_len: ${SFT_CUTOFF_LEN}
preprocessing_num_workers: 8
dataloader_num_workers: 2
output_dir: ${SFT_OUTPUT_DIR}
logging_steps: 1
save_steps: 0.05
plot_loss: true
overwrite_output_dir: true
save_only_model: false
save_strategy: steps
load_best_model_at_end: true
save_total_limit: 1
report_to: wandb
run_name: ${CATEGORY}-qwen2.5-${MODEL_SIZE}-sft-${DSZ}
per_device_train_batch_size: ${SFT_PER_DEVICE_BATCH}
per_device_eval_batch_size: ${SFT_PER_DEVICE_BATCH}
gradient_accumulation_steps: ${SFT_GRAD_ACCUM}
learning_rate: ${SFT_LR}
num_train_epochs: ${SFT_EPOCHS}
lr_scheduler_type: cosine
warmup_steps: ${SFT_WARMUP_STEPS}
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null
early_stopping_steps: 3
metric_for_best_model: loss
eval_strategy: steps
eval_steps: 0.05
EOF
  fi

  log "启动 SFT 训练 (GPU: ${CUDA_VISIBLE_DEVICES})"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" llamafactory-cli train "$SFT_YAML" || fail "SFT 训练失败"
  [ -d "$SFT_OUTPUT_DIR" ] || fail "SFT 输出目录不存在: $SFT_OUTPUT_DIR"
  log "SFT 完成: $SFT_OUTPUT_DIR"
fi

# -----------------------------------------------------------------------------
# 4. RL (GRPO) 训练
# -----------------------------------------------------------------------------
if [ "${RUN_RL}" = "1" ]; then
  [ -d "$RL_MODEL_PATH" ] || fail "RL 需要 SFT checkpoint: $RL_MODEL_PATH (先运行 SFT)"
  RL_EPOCHS_EFF="$RL_EPOCHS"
  RL_EVAL_EFF="$RL_EVAL_STEP"
  RL_SAVE_STEPS_EFF="0.1"
  [ "${QUICK_TEST}" = "1" ] && RL_EPOCHS_EFF="0.002" && RL_EVAL_EFF="100000" && RL_SAVE_STEPS_EFF="1.0"

  log "启动 RL(GRPO) 训练"
  accelerate launch \
    --config_file "$RL_DS_CONFIG" \
    --num_processes "$NUM_PROCESSES" \
    --main_process_port "$MAIN_PORT" \
    trl_trainer.py \
    --model "$RL_MODEL_PATH" \
    --data_dir "$RL_DATA_DIR" \
    --index_path "$RL_INDEX_PATH" \
    --output_dir "$RL_OUTPUT_DIR" \
    --num_beams "$RL_NUM_BEAMS" \
    --per_device_train_batch_size "$RL_TRAIN_BATCH" \
    --per_device_eval_batch_size "$RL_EVAL_BATCH" \
    --gradient_accumulation_steps "$RL_GRAD_ACCUM" \
    --num_train_epochs "$RL_EPOCHS_EFF" \
    --learning_rate "$RL_LR" \
    --eval_step "$RL_EVAL_EFF" \
    --max_completion_length "$RL_MAX_COMPLETION_LENGTH" \
    --beta "$RL_BETA" \
    --save_steps "$RL_SAVE_STEPS_EFF" \
    --save_total_limit "$RL_SAVE_TOTAL_LIMIT" \
    --report_to wandb || fail "RL 训练失败"
  log "RL 完成: $RL_OUTPUT_DIR"
fi

# -----------------------------------------------------------------------------
# 5. 评估
# -----------------------------------------------------------------------------
if [ "${RUN_EVAL}" = "1" ]; then
  [ -d "$EVAL_EXP_NAME" ] || fail "待评估模型不存在: $EVAL_EXP_NAME"

  EVAL_TEST="$EVAL_TEST_DATA"
  if [ "${QUICK_TEST}" = "1" ]; then
    # 快速测试: 只取前 20 条样本
    EVAL_TEST="configs/quick_test.json"
    "$PYTHON_BIN" - "$EVAL_TEST_DATA" "$EVAL_TEST" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
json.dump(data[:20], open(sys.argv[2], "w"), ensure_ascii=False)
print(f"quick test subset: {len(data[:20])} samples")
PY
  fi

  log "启动评估 (GPU: ${EVAL_GPU})"
  mkdir -p "$EVAL_OUTPUT_DIR"
  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "$PYTHON_BIN" -u evaluate.py \
    --model_name_or_path "$EVAL_EXP_NAME" \
    --test_data_path "$EVAL_TEST" \
    --result_json_path "$EVAL_OUTPUT_DIR/final_result.json" \
    --index_path "$EVAL_INDEX_PATH" \
    --batch_size "$EVAL_BATCH_SIZE" \
    --max_new_tokens "$EVAL_MAX_NEW_TOKENS" \
    --num_beams "$EVAL_NUM_BEAMS" \
    --temperature "$EVAL_TEMPERATURE" \
    --do_sample "$EVAL_DO_SAMPLE" \
    --length_penalty "$EVAL_LENGTH_PENALTY" \
    --compute_metrics_flag True || fail "评估失败"

  log "评估完成, 结果: $EVAL_OUTPUT_DIR/final_result.json"
  [ -f "$EVAL_OUTPUT_DIR/metrics.json" ] && cat "$EVAL_OUTPUT_DIR/metrics.json" || true
fi

log "全部阶段完成 ✔"
