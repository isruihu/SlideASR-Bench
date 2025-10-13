#!/bin/bash
cd "$(dirname "$0")" || exit 1
MODEL="$1"
ROOT=$(realpath ../eval_results) 

tasks=(
  "SlideASR-S:${ROOT}/SlideASR-S/${MODEL}.jsonl"
  "SlideASR-R:${ROOT}/SlideASR-R/${MODEL}.jsonl"
)

for item in "${tasks[@]}"; do
  # 用 Bash 字符串处理拆成 name 和 file
  name=${item%%:*}
  file=${item#*:}

  if [[ -f $file ]]; then
    echo "##### eval $name #####"
    bash evaluate_slide_asr_bench.sh "$file"
  else
    echo "Skip $name：$file does not exists."
  fi
done
