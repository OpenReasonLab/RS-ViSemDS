#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
MODEL_PATH="${INTERNVL35_14B_MODEL:-/root/autodl-tmp/models/InternVL3.5-14B}"
REMOTECLIP_CHECKPOINT="${REMOTECLIP_CHECKPOINT:-$PROJECT_ROOT/checkpoints/RemoteCLIP-ViT-B-32.pt}"

case "$MODE" in
  smoke)
    EXTRA_ARGS=(--limit 2)
    ;;
  full)
    EXTRA_ARGS=()
    ;;
  dry-run)
    EXTRA_ARGS=(--limit 2 --dry-run)
    ;;
  *)
    echo "Usage: $0 [smoke|full|dry-run]" >&2
    exit 2
    ;;
esac

cd "$PROJECT_ROOT"
test -x "$PYTHON_BIN"
test -d "$MODEL_PATH"
test -f "$REMOTECLIP_CHECKPOINT"

echo "mode=$MODE"
echo "project=$PROJECT_ROOT"
echo "model=$MODEL_PATH"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

exec "$PYTHON_BIN" RS-ViSemDS/run_rs_visemds_all.py \
  --datasets aid nwpu_fg_urban \
  --models internvl35_14b \
  --model-path "internvl35_14b=$MODEL_PATH" \
  --r 3 \
  --k 3 \
  --alpha 0.6 \
  --beta 0.2 \
  --gamma 0.2 \
  --prompt-mode manuscript_v1 \
  --remoteclip-checkpoint "$REMOTECLIP_CHECKPOINT" \
  --feature-batch-size 64 \
  --feature-num-workers 0 \
  --max-tokens 256 \
  "${EXTRA_ARGS[@]}"
