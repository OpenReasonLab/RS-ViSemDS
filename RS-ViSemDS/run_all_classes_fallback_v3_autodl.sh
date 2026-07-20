#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_ROOT"

AID_RUN_ROOT="RS-ViSemDS/ablations/aid_all/weight_602020_reference_fallback_v3"
NWPU_RUN_ROOT="RS-ViSemDS/ablations/nwpu_all/weight_602020/reference_fallback_v3_all_classes"
mkdir -p "$AID_RUN_ROOT" "$NWPU_RUN_ROOT"

echo "[$(date -Iseconds)] Starting AID all-class reference_fallback_v3 experiment."
bash RS-ViSemDS/run_aid_all_fallback_v3_autodl.sh > "$AID_RUN_ROOT/run.log" 2>&1
echo "[$(date -Iseconds)] AID completed; starting NWPU all-class experiment."
bash RS-ViSemDS/run_nwpu_all_fallback_v3_autodl.sh > "$NWPU_RUN_ROOT/run.log" 2>&1
echo "[$(date -Iseconds)] All reference_fallback_v3 experiments completed."
