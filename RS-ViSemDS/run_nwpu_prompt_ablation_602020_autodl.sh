#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-both}"
PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/remote_sensing_project/strict_fewshot_baselines}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
MODEL_PATH="${INTERNVL35_14B_MODEL:-/root/autodl-tmp/models/InternVL3.5-14B}"
REMOTECLIP_CHECKPOINT="${REMOTECLIP_CHECKPOINT:-$PROJECT_ROOT/RemoteCLIP/models--chendelong--RemoteCLIP/snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38/RemoteCLIP-ViT-B-32.pt}"
RUN_ROOT="RS-ViSemDS/ablations/nwpu_all/weight_602020"
SELECTION_DIR="$RUN_ROOT/selection"

case "$MODE" in
  guided|no_description|both) ;;
  *)
    echo "Usage: $0 [guided|no_description|both]" >&2
    exit 2
    ;;
esac

test -x "$PYTHON_BIN"
test -d "$MODEL_PATH"
test -f "$REMOTECLIP_CHECKPOINT"
cd "$PROJECT_ROOT"

echo "NWPU prompt ablation: alpha=0.6 beta=0.2 gamma=0.2; mode=$MODE"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

if [[ ! -f "$SELECTION_DIR/examples_rs_visemds_shot_3.csv" ]]; then
  "$PYTHON_BIN" RS-ViSemDS/build_rs_visemds_examples.py \
    --dataset nwpu_fg_urban \
    --manifest-dir manifests/nwpu_eval100_seed42 \
    --out-dir "$SELECTION_DIR" \
    --r 3 --k 3 \
    --alpha 0.6 --beta 0.2 --gamma 0.2 \
    --remoteclip-checkpoint "$REMOTECLIP_CHECKPOINT" \
    --feature-batch-size 64 \
    --feature-num-workers 0
fi

run_prompt() {
  local prompt_mode="$1"
  local output_name="$2"
  "$PYTHON_BIN" RS-ViSemDS/run_rs_visemds_mllm.py \
    --dataset nwpu_fg_urban \
    --manifest-dir manifests/nwpu_eval100_seed42 \
    --selected-examples-csv "$SELECTION_DIR/examples_rs_visemds_shot_3.csv" \
    --model "$MODEL_PATH" \
    --out-dir "$RUN_ROOT/$output_name/results_internvl35_14b" \
    --prompt-mode "$prompt_mode" \
    --torch-dtype bfloat16 \
    --device-map auto \
    --max-tokens 256 \
    --resume
}

if [[ "$MODE" == "guided" || "$MODE" == "both" ]]; then
  run_prompt reference_guided_v1 with_category_descriptions
fi

if [[ "$MODE" == "no_description" || "$MODE" == "both" ]]; then
  run_prompt reference_only_v1 without_category_descriptions
fi

echo "NWPU prompt ablation finished."
