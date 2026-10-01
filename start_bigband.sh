#!/usr/bin/env bash
# ビッグバンド編曲プリセット付きの起動スクリプト。
#   ./start_bigband.sh ballad   # バラード（4拍子・62BPM）
#   ./start_bigband.sh waltz    # ジャズワルツ（3拍子・120BPM）
#
# プリセットは presets/<name>.env を環境変数として読み込み、Gradio の
# キャプション / LM ネガティブプロンプト / BPM / キー / 拍子 の初期値になる。
# 実装は acestep/ui/gradio/interfaces/user_defaults.py。
set -euo pipefail
cd "$(dirname "$0")"

PRESET="${1:-ballad}"
PRESET_FILE="presets/${PRESET}.env"

if [[ ! -f "$PRESET_FILE" ]]; then
  echo "プリセットが見つかりません: $PRESET_FILE" >&2
  echo "利用可能: $(ls presets/*.env 2>/dev/null | xargs -n1 basename 2>/dev/null | sed 's/\.env$//' | tr '\n' ' ')" >&2
  exit 1
fi

# 原曲ベース編曲（Remix = cover タスク）専用 UI。効かないフォームは全て隠す。
# 0 を渡せば素の全モード UI に戻る。
export ACESTEP_UI_REMIX_ONLY="${ACESTEP_UI_REMIX_ONLY:-1}"

set -a
# shellcheck disable=SC1090
source "$PRESET_FILE"
set +a

echo "プリセット: $PRESET"
echo "  caption : ${ACESTEP_DEFAULT_CAPTION_FILE:-（既定）}"
echo "  negative: ${ACESTEP_DEFAULT_NEGATIVE_PROMPT_FILE:-（既定）}"
echo "  bpm=${ACESTEP_DEFAULT_BPM:-auto} timesig=${ACESTEP_DEFAULT_TIMESIG:-auto} key=${ACESTEP_DEFAULT_KEYSCALE:-auto}"
echo "  remix専用UI: ${ACESTEP_UI_REMIX_ONLY}"

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
