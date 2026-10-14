#!/usr/bin/env bash
# =============================================================================
# MIMIGenRec 集中配置文件 (single source of truth)
# 所有可调参数集中在此文件；run_all.sh 会 source 本文件并按顺序执行各阶段。
# =============================================================================

# -----------------------------------------------------------------------------
# 0. 环境 / 路径
# -----------------------------------------------------------------------------
export CONDA_ENV="${CONDA_ENV:-mimigenrec}"   # 使用的 conda 环境名
export PYTHON_BIN="/data/sht/conda/${CONDA_ENV}/bin/python"
export PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -----------------------------------------------------------------------------
# 1. 数据集 / 模型选择
# -----------------------------------------------------------------------------
# 可选类别: Industrial_and_Scientific / Office_Products / Toys_and_Games
export CATEGORY="${CATEGORY:-Industrial_and_Scientific}"

# 模型规模: 0.5b / 1.5b / 3b  (0.5b、1.5b 已本地缓存)
export MODEL_SIZE="${MODEL_SIZE:-0.5b}"
# deepspeed 阶段: dsz0 / dsz2 / dsz3  (对应 examples/deepspeed/ds_z{0,2,3}_config.json)
export DSZ="${DSZ:-dsz0}"

case "$MODEL_SIZE" in
  0.5b) export MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct" ;;
  1.5b) export MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct" ;;
  3b)   export MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"    ;;
  *)    export MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct" ;;
esac

# -----------------------------------------------------------------------------
# 2. 本地算力 (4 张 GPU)
# -----------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export NUM_PROCESSES=4
export MAIN_PORT=29503

# -----------------------------------------------------------------------------
# 3. 阶段开关
# -----------------------------------------------------------------------------
export RUN_DATA_PREP="${RUN_DATA_PREP:-0}"       # 1=重新下载/预处理原始 Amazon 数据 (默认关闭: 已含预处理结果)
export RUN_SFT="${RUN_SFT:-1}"
export RUN_RL="${RUN_RL:-1}"
export RUN_EVAL="${RUN_EVAL:-1}"
export QUICK_TEST="${QUICK_TEST:-0}"          # 1=快速冒烟测试 (极少量 step, 验证整条链路)

# -----------------------------------------------------------------------------
# 4. SFT 超参
# -----------------------------------------------------------------------------
export SFT_OUTPUT_DIR="saves/qwen2.5-${MODEL_SIZE}/full/${CATEGORY}-sft-${DSZ}"
export SFT_CUTOFF_LEN=512
export SFT_PER_DEVICE_BATCH=32
export SFT_GRAD_ACCUM=8
export SFT_EPOCHS=10.0
export SFT_LR=3.0e-4
export SFT_WARMUP_STEPS=20
export SFT_MAX_STEPS=""                        # 置为非空则覆盖 epochs (用于快速测试)

# -----------------------------------------------------------------------------
# 5. RL (GRPO) 超参
# -----------------------------------------------------------------------------
export RL_MODEL_PATH="${SFT_OUTPUT_DIR}"       # SFT 得到的 checkpoint
export RL_DATA_DIR="data/${CATEGORY}/rl"
export RL_INDEX_PATH="data/${CATEGORY}/id2sid.json"
export RL_OUTPUT_DIR="rl_outputs/${CATEGORY}-qwen2.5-${MODEL_SIZE}-instruct-grpo"
export RL_NUM_BEAMS=16
export RL_TRAIN_BATCH=64
export RL_EVAL_BATCH=64
export RL_GRAD_ACCUM=2
export RL_EPOCHS=2
export RL_LR=1e-5
export RL_BETA=1e-3
export RL_EVAL_STEP=20
export RL_MAX_COMPLETION_LENGTH=128
export RL_SAVE_TOTAL_LIMIT=1
export RL_DS_CONFIG="config/zero2.yaml"

# -----------------------------------------------------------------------------
# 6. 评估超参
# -----------------------------------------------------------------------------
export EVAL_EXP_NAME="${SFT_OUTPUT_DIR}"       # 待评估模型路径
export EVAL_TEST_DATA="data/${CATEGORY}/sft/test.json"
export EVAL_INDEX_PATH="data/${CATEGORY}/id2sid.json"
export EVAL_OUTPUT_DIR="results/$(basename ${EVAL_EXP_NAME})"
export EVAL_BATCH_SIZE=8
export EVAL_MAX_NEW_TOKENS=256
export EVAL_NUM_BEAMS=50
export EVAL_TEMPERATURE=1.0
export EVAL_DO_SAMPLE=False
export EVAL_LENGTH_PENALTY=0.0
export EVAL_GPU="0"                            # 单卡评估即可

# -----------------------------------------------------------------------------
# 7. 日志 / 网络
# -----------------------------------------------------------------------------
export HF_ENDPOINT="https://hf-mirror.com"
export HF_OFFLINE="${HF_OFFLINE:-1}"          # 1=离线(模型已本地缓存, 不联网) / 0=联网下载
if [ "$HF_OFFLINE" = "1" ]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
fi
export WANDB_PROJECT="MIMIGenRec"
export WANDB_MODE="offline"                    # offline=不联网 / online=上传 wandb
export WANDB_API_KEY=""
export DISABLE_VERSION_CHECK=1
