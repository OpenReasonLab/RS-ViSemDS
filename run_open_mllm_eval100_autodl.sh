#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
MODEL_ALIAS="${MODEL_ALIAS:-${1:-}}"
PREPARE_EXAMPLES="${PREPARE_EXAMPLES:-0}"
REMOTECLIP_CHECKPOINT="${REMOTECLIP_CHECKPOINT:-$PROJECT_ROOT/checkpoints/RemoteCLIP-ViT-B-32.pt}"
LIMIT="${LIMIT:-}"

case "$MODEL_ALIAS" in
  llama32_11b)
    RUNNER="run_llama32_11b_aid_nwpu_all.py"
    MODEL_ENV_VAR="LLAMA32_11B_MODEL"
    DEFAULT_MODEL_PATH="/root/autodl-tmp/models/Llama-3.2-11B-Vision-Instruct"
    ;;
  gemma3_12b)
    RUNNER="run_gemma3_12b_aid_nwpu_all.py"
    MODEL_ENV_VAR="GEMMA3_12B_MODEL"
    DEFAULT_MODEL_PATH="/root/autodl-tmp/models/Gemma 3-12B"
    ;;
  qwen25vl_7b)
    RUNNER="run_qwen25vl_7b_aid_nwpu_all.py"
    MODEL_ENV_VAR="QWEN25VL_7B_MODEL"
    DEFAULT_MODEL_PATH="/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct"
    ;;
  qwen3vl_8b)
    RUNNER="run_qwen3vl_8b_aid_nwpu_all.py"
    MODEL_ENV_VAR="QWEN3VL_8B_MODEL"
    DEFAULT_MODEL_PATH="/root/autodl-tmp/models/Qwen3-VL-8B"
    ;;
  internvl35_8b)
    RUNNER="run_internvl35_8b_aid_nwpu_all.py"
    MODEL_ENV_VAR="INTERNVL35_8B_MODEL"
    DEFAULT_MODEL_PATH="/root/autodl-tmp/models/InternVL3.5-8B"
    ;;
  internvl35_14b)
    RUNNER="run_internvl35_14b_aid_nwpu_all.py"
    MODEL_ENV_VAR="INTERNVL35_14B_MODEL"
    DEFAULT_MODEL_PATH="/root/autodl-tmp/models/InternVL3.5-14B"
    ;;
  *)
    echo "Usage: MODEL_ALIAS=<alias> bash run_open_mllm_eval100_autodl.sh" >&2
    echo "Aliases: llama32_11b gemma3_12b qwen25vl_7b qwen3vl_8b internvl35_8b internvl35_14b" >&2
    exit 2
    ;;
esac

MODEL_PATH="${MLLM_MODEL_PATH:-}"
if [[ -z "$MODEL_PATH" ]]; then
  MODEL_PATH="${!MODEL_ENV_VAR-}"
fi
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"

cd "$PROJECT_ROOT"
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model directory not found: $MODEL_PATH" >&2
  exit 1
fi
if [[ ! -d data_raw/AID_dataset || ! -d data_raw/NWPU-RESISC45 ]]; then
  echo "Place both datasets under $PROJECT_ROOT/data_raw first." >&2
  exit 1
fi

if [[ "$PREPARE_EXAMPLES" == "1" ]]; then
  if [[ ! -f "$REMOTECLIP_CHECKPOINT" ]]; then
    echo "RemoteCLIP checkpoint not found: $REMOTECLIP_CHECKPOINT" >&2
    exit 1
  fi
  "$PYTHON_BIN" "$SCRIPT_DIR/prepare_eval100_protocol.py" \
    --datasets aid nwpu_fg_urban \
    --strategies random knn \
    --shots 1 3 5 10 \
    --skip-manifests \
    --remoteclip-cache checkpoints \
    --remoteclip-checkpoint "$REMOTECLIP_CHECKPOINT" \
    --feature-batch-size 64 \
    --feature-num-workers 4
fi

runner_args=(
  --datasets aid nwpu_fg_urban
  --shots 1 3 5 10
  --model "$MODEL_PATH"
  --backend transformers
  --prompt-mode minimal
  --max-tokens 256
  --torch-dtype bfloat16
  --device-map auto
  --python "$PYTHON_BIN"
)
if [[ -n "$LIMIT" ]]; then
  runner_args+=(--limit "$LIMIT")
fi

exec "$PYTHON_BIN" -u "$SCRIPT_DIR/$RUNNER" "${runner_args[@]}"
