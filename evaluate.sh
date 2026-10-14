#!/usr/bin/env bash
set -euo pipefail

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

CATEGORY="$(get_env_or_default CATEGORY Industrial_and_Scientific)"
Optimization="$(get_env_or_default Optimization dsz3)"
exp_name="$(get_env_or_default EVAL_EXP_NAME newsaves/qwen2.5-0.5b/full/${CATEGORY}-sft-${Optimization})"
cuda_list="$(get_env_or_default EVAL_CUDA_LIST '0 1 2 3')"

test_data_path="data/${CATEGORY}/sft/test.json"
temp_dir="temp/eval-${CATEGORY}"
output_dir="results/${exp_name#*/}"

index_path="data/${CATEGORY}/id2sid.json"
batch_size="${EVAL_BATCH_SIZE:-8}"
max_new_tokens=256
num_beams=50
temperature=1.0
do_sample=False
length_penalty=0.0
seed=42

exp_name_clean=$(basename "$exp_name")
echo "=========================================="
echo "Evaluation: $exp_name_clean"
echo "Test data: $test_data_path"
echo "=========================================="

[[ -f "$test_data_path" ]] || { echo "Error: 缺少 $test_data_path"; exit 1; }
[[ -f "$index_path" ]] || { echo "Error: 缺少 $index_path"; exit 1; }
[[ -d "$exp_name" ]] || { echo "Error: 缺少模型 $exp_name"; exit 1; }

rm -rf "$temp_dir"
mkdir -p "$temp_dir" "$output_dir"

# 单卡模式
if [[ -z "$cuda_list" ]] || [[ "$cuda_list" == "0" ]]; then
    echo "Single GPU evaluation..."
    CUDA_VISIBLE_DEVICES=0 python -u evaluate.py \
        --model_name_or_path "$exp_name" \
        --test_data_path "$test_data_path" \
        --result_json_path "$temp_dir/result.json" \
        --index_path "$index_path" \
        --batch_size "$batch_size" \
        --max_new_tokens "$max_new_tokens" \
        --num_beams "$num_beams" \
        --temperature "$temperature" \
        --do_sample "$do_sample" \
        --length_penalty "$length_penalty" \
        --seed "$seed" \
        --compute_metrics_flag True
    cp "$temp_dir/result.json" "$output_dir/final_result.json"
    [[ -f "$temp_dir/metrics.json" ]] && cp "$temp_dir/metrics.json" "$output_dir/metrics.json"
    rm -rf "$temp_dir"
    echo "Evaluation done!"
    exit 0
fi

# 多卡并行评估
echo "Multi-GPU parallel evaluation: $cuda_list"
if [[ ! -f "split_json.py" ]]; then
    echo "Warning: split_json.py 缺失, 回落单卡"
    CUDA_VISIBLE_DEVICES=0 python -u evaluate.py \
        --model_name_or_path "$exp_name" \
        --test_data_path "$test_data_path" \
        --result_json_path "$temp_dir/result.json" \
        --index_path "$index_path" \
        --batch_size "$batch_size" \
        --max_new_tokens "$max_new_tokens" \
        --num_beams "$num_beams" \
        --temperature "$temperature" \
        --do_sample "$do_sample" \
        --length_penalty "$length_penalty" \
        --seed "$seed" \
        --compute_metrics_flag True
    cp "$temp_dir/result.json" "$output_dir/final_result.json"
    [[ -f "$temp_dir/metrics.json" ]] && cp "$temp_dir/metrics.json" "$output_dir/metrics.json"
    rm -rf "$temp_dir"
    exit 0
fi

python split_json.py --input_path "$test_data_path" --output_path "$temp_dir" --cuda_list "$cuda_list"

split_idx=0
for i in $cuda_list; do
    if [[ -f "$temp_dir/${i}.json" ]]; then
        echo "Starting evaluation on GPU $i (seed=$((seed + split_idx)))..."
        CUDA_VISIBLE_DEVICES=$i python -u evaluate.py \
            --model_name_or_path "$exp_name" \
            --test_data_path "$temp_dir/${i}.json" \
            --result_json_path "$temp_dir/${i}_result.json" \
            --index_path "$index_path" \
            --batch_size "$batch_size" \
            --max_new_tokens "$max_new_tokens" \
            --num_beams "$num_beams" \
            --temperature "$temperature" \
            --do_sample "$do_sample" \
            --length_penalty "$length_penalty" \
            --seed "$((seed + split_idx))" \
            --compute_metrics_flag False &
    fi
    split_idx=$((split_idx + 1))
done

wait

echo "Merging results..."
python merge_json.py --input_path "$temp_dir" --output_path "$output_dir/final_result.json" --cuda_list "$cuda_list" --compute_metrics False
[[ -f "$output_dir/final_result.json" ]] || { echo "Error: Merge failed"; exit 1; }

# 汇总指标
python -u evaluate.py \
    --result_json_path "$output_dir/final_result.json" \
    --metrics_only True

rm -rf "$temp_dir"
echo "Evaluation done!"
