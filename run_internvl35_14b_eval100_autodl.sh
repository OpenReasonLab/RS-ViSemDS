#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/remote_sensing_project/strict_fewshot_baselines}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${INTERNVL35_14B_MODEL:-/root/autodl-tmp/models/InternVL3.5-14B}"
REMOTECLIP_CHECKPOINT="${REMOTECLIP_CHECKPOINT:-$PROJECT_ROOT/checkpoints/RemoteCLIP-ViT-B-32.pt}"
PREPARE_EXAMPLES="${PREPARE_EXAMPLES:-1}"
LIMIT="${LIMIT:-}"

cd "$PROJECT_ROOT"

if [[ ! -f "$REMOTECLIP_CHECKPOINT" ]]; then
  discovered_checkpoint="$(find "$PROJECT_ROOT" -type f -name 'RemoteCLIP-ViT-B-32.pt' -not -path '*/.Trash-*/*' -print -quit 2>/dev/null || true)"
  if [[ -n "$discovered_checkpoint" ]]; then
    REMOTECLIP_CHECKPOINT="$discovered_checkpoint"
  fi
fi

LOG_DIR="logs_eval100_seed42"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/internvl35_14b_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Project: $PROJECT_ROOT"
echo "Python: $PYTHON_BIN"
echo "Model: $MODEL_PATH"
echo "RemoteCLIP: $REMOTECLIP_CHECKPOINT"
echo "Log: $LOG_FILE"

required_files=(
  "eval100_protocol.py"
  "prepare_eval100_protocol.py"
  "prepare_manifest.py"
  "build_examples.py"
  "run_zero_shot_mllm.py"
  "run_random_fewshot_mllm.py"
  "run_knn_totalshot_mllm.py"
  "run_internvl35_14b_aid_nwpu_all.py"
  "configs/aid.json"
  "configs/nwpu_fg_urban.json"
)
for path in "${required_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $PROJECT_ROOT/$path" >&2
    exit 1
  fi
done

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "InternVL3.5-14B model directory not found: $MODEL_PATH" >&2
  exit 1
fi
if [[ ! -d "data_raw/AID_dataset" ]]; then
  echo "AID dataset not found: $PROJECT_ROOT/data_raw/AID_dataset" >&2
  exit 1
fi
if [[ ! -d "data_raw/NWPU-RESISC45" ]]; then
  echo "NWPU dataset not found: $PROJECT_ROOT/data_raw/NWPU-RESISC45" >&2
  exit 1
fi

"$PYTHON_BIN" -c "import importlib.util, torch, torchvision, timm, transformers, open_clip; print('torch=', torch.__version__); print('transformers=', transformers.__version__); print('flash_attn=', importlib.util.find_spec('flash_attn') is not None); print('cuda=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
nvidia-smi

if [[ "$PREPARE_EXAMPLES" == "1" ]]; then
  if [[ ! -f "$REMOTECLIP_CHECKPOINT" ]]; then
    echo "RemoteCLIP checkpoint not found. Refusing to build kNN examples with random weights." >&2
    echo "Set REMOTECLIP_CHECKPOINT to the existing RemoteCLIP-ViT-B-32.pt file." >&2
    exit 1
  fi
  prepare_args=(
    --datasets aid nwpu_fg_urban
    --strategies random knn
    --shots 1 3 5 10
    --remoteclip-cache checkpoints
    --feature-batch-size 64
    --feature-num-workers 4
  )
  prepare_args+=(--remoteclip-checkpoint "$REMOTECLIP_CHECKPOINT")
  "$PYTHON_BIN" prepare_eval100_protocol.py "${prepare_args[@]}"
else
  echo "Skipping manifest/example preparation (PREPARE_EXAMPLES=$PREPARE_EXAMPLES)."
fi

runner_args=(
  --datasets aid nwpu_fg_urban
  --shots 1 3 5 10
  --model "$MODEL_PATH"
  --out-root "InternVL3.5-14B/results_eval100_seed42"
  --backend transformers
  --prompt-mode minimal
  --max-tokens 256
  --torch-dtype bfloat16
  --device-map auto
)
if [[ -n "$LIMIT" ]]; then
  runner_args+=(--limit "$LIMIT")
fi

"$PYTHON_BIN" run_internvl35_14b_aid_nwpu_all.py "${runner_args[@]}"

echo "InternVL3.5-14B eval100 suite completed."
echo "Results: $PROJECT_ROOT/InternVL3.5-14B/results_eval100_seed42"
