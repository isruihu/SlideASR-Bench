#!/bin/bash
# 当任何命令失败时立即退出脚本
set -e
cd "$(dirname "$0")" || exit 1
cd ./slide_asr_evaluation

# 检查调用脚本时是否提供了输入文件参数
if [ "$#" -ne 1 ]; then
    echo "错误: 请提供一个输入文件作为参数。"
    echo "用法: $0 /path/to/your_input_file.jsonl"
    exit 1
fi

# 检查输入文件是否存在
if [ ! -f "$1" ]; then
    echo "错误: 输入文件不存在: $1"
    exit 1
fi

# --- 定义文件名和路径变量 ---
# 第一个参数作为输入文件的完整路径
INPUT_FILE="$1"
# 使用 dirname 获取输入文件所在的目录路径
# 例如，如果输入是 "data/results/test.jsonl", INPUT_DIR 会是 "data/results"
INPUT_DIR=$(dirname "${INPUT_FILE}")
# 使用 basename 获取不带后缀的文件名
# 例如，如果输入是 "data/results/test.jsonl", BASENAME 会是 "test"
BASENAME=$(basename "${INPUT_FILE}" .jsonl)

# 将目录路径和新的文件名组合成完整的输出路径
PREPARED_RESULTS="${INPUT_DIR}/1.prepared_${BASENAME}.txt"
METRICS_RESULTS="${INPUT_DIR}/2.metric_${BASENAME}.txt"
OVERALL_RESULTS="${INPUT_DIR}/3.result_${BASENAME}.csv"

# --- 步骤 1: 准备 ASR 结果 ---
echo "Step 1: Preparing ASR results from ${INPUT_FILE}..."
python3 prepare_asr_result_for_metrics.py "${INPUT_FILE}" "${PREPARED_RESULTS}"

# --- 步骤 2: 计算指标 ---
echo "Step 2: Calculating metrics for ${PREPARED_RESULTS}..."
python3 calculate_metrics.py "${PREPARED_RESULTS}" "${METRICS_RESULTS}"

# --- 步骤 3: 聚合与展示结果 ---
echo "Step 3: Aggregating and viewing final metrics from ${METRICS_RESULTS}..."
python3 view_metric_result.py "${METRICS_RESULTS}" "${OVERALL_RESULTS}"
