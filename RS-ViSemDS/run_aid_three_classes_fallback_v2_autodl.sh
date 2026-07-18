#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/remote_sensing_project/strict_fewshot_baselines}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
MODEL_PATH="${INTERNVL35_14B_MODEL:-/root/autodl-tmp/models/InternVL3.5-14B}"
SELECTION_DIR="RS-ViSemDS/ablations/aid_three_classes/v3_weight_602020_reference_prompt/selection"
RUN_ROOT="RS-ViSemDS/ablations/aid_three_classes/v5_weight_602020_reference_fallback_v2"
RESULT_DIR="$RUN_ROOT/results_internvl35_14b"

test -x "$PYTHON_BIN"
test -d "$MODEL_PATH"
test -f "$SELECTION_DIR/examples_rs_visemds_shot_3.csv"
cd "$PROJECT_ROOT"

echo "V5: alpha=0.6 beta=0.2 gamma=0.2; prompt=reference_fallback_v2"
echo "Reusing the frozen V3 selection file; only prompt instructions and fallback descriptions change."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

exec "$PYTHON_BIN" RS-ViSemDS/run_rs_visemds_mllm.py \
  --dataset aid \
  --manifest-dir manifests/aid_eval100_seed42 \
  --selected-examples-csv "$SELECTION_DIR/examples_rs_visemds_shot_3.csv" \
  --model "$MODEL_PATH" \
  --out-dir "$RESULT_DIR" \
  --prompt-mode reference_fallback_v2 \
  --target-classes Center Church Commercial \
  --torch-dtype bfloat16 \
  --device-map auto \
  --max-tokens 256 \
  --resume
