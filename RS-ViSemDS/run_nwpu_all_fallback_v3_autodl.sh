#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/remote_sensing_project/strict_fewshot_baselines}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
MODEL_PATH="${INTERNVL35_14B_MODEL:-/root/autodl-tmp/models/InternVL3.5-14B}"
SELECTION_DIR="RS-ViSemDS/ablations/nwpu_all/weight_602020/selection"
RUN_ROOT="RS-ViSemDS/ablations/nwpu_all/weight_602020/reference_fallback_v3_all_classes"
RESULT_DIR="$RUN_ROOT/results_internvl35_14b"

test -x "$PYTHON_BIN"
test -d "$MODEL_PATH"
test -f "$SELECTION_DIR/examples_rs_visemds_shot_3.csv"
cd "$PROJECT_ROOT"

echo "NWPU all classes: alpha=0.6 beta=0.2 gamma=0.2; prompt=reference_fallback_v3"
echo "Reusing the frozen 800-target NWPU selection file."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

exec "$PYTHON_BIN" RS-ViSemDS/run_rs_visemds_mllm.py \
  --dataset nwpu_fg_urban \
  --manifest-dir manifests/nwpu_eval100_seed42 \
  --selected-examples-csv "$SELECTION_DIR/examples_rs_visemds_shot_3.csv" \
  --model "$MODEL_PATH" \
  --out-dir "$RESULT_DIR" \
  --prompt-mode reference_fallback_v3 \
  --target-classes dense_residential medium_residential sparse_residential mobile_home_park commercial_area industrial_area parking_lot railway_station \
  --torch-dtype bfloat16 \
  --device-map auto \
  --max-tokens 256 \
  --resume
