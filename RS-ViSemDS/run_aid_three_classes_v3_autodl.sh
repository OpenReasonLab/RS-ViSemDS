#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
MODEL_PATH="${INTERNVL35_14B_MODEL:-/root/autodl-tmp/models/InternVL3.5-14B}"
REMOTECLIP_CHECKPOINT="${REMOTECLIP_CHECKPOINT:-$PROJECT_ROOT/checkpoints/RemoteCLIP-ViT-B-32.pt}"
RUN_ROOT="RS-ViSemDS/ablations/aid_three_classes/v3_weight_602020_reference_prompt"
SELECTION_DIR="$RUN_ROOT/selection"
RESULT_DIR="$RUN_ROOT/results_internvl35_14b"

cd "$PROJECT_ROOT"
test -x "$PYTHON_BIN"
test -d "$MODEL_PATH"
test -f "$REMOTECLIP_CHECKPOINT"

echo "V3: alpha=0.6 beta=0.2 gamma=0.2; prompt=reference_guided_v1"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

if [[ ! -f "$SELECTION_DIR/examples_rs_visemds_shot_3.csv" ]]; then
  "$PYTHON_BIN" RS-ViSemDS/build_rs_visemds_examples.py \
    --dataset aid \
    --manifest-dir manifests/aid_eval100_seed42 \
    --out-dir "$SELECTION_DIR" \
    --r 3 --k 3 \
    --alpha 0.6 --beta 0.2 --gamma 0.2 \
    --remoteclip-checkpoint "$REMOTECLIP_CHECKPOINT" \
    --feature-batch-size 64 \
    --feature-num-workers 0
fi

exec "$PYTHON_BIN" RS-ViSemDS/run_rs_visemds_mllm.py \
  --dataset aid \
  --manifest-dir manifests/aid_eval100_seed42 \
  --selected-examples-csv "$SELECTION_DIR/examples_rs_visemds_shot_3.csv" \
  --model "$MODEL_PATH" \
  --out-dir "$RESULT_DIR" \
  --prompt-mode reference_guided_v1 \
  --target-classes Center Church Commercial \
  --torch-dtype bfloat16 \
  --device-map auto \
  --max-tokens 256 \
  --resume
