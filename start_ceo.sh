#!/usr/bin/env bash
# CEO 用の起動スクリプト。
# --init_service を渡さないとモデルが読み込まれず、生成時に
# "Model not fully initialized" で失敗する（既定値が false のため）。
set -euo pipefail
cd "$(dirname "$0")"

lsof -ti tcp:7860 | xargs -r kill -9 2>/dev/null || true
sleep 1

ACESTEP_LM_BACKEND=mlx exec uv run acestep \
  --port 7860 \
  --server-name 127.0.0.1 \
  --language ja \
  --init_service true \
  --config_path acestep-v15-xl-sft \
  --init_llm true \
  --lm_model_path acestep-5Hz-lm-4B \
  --backend mlx \
  --batch_size 1
