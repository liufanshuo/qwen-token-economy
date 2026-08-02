#!/usr/bin/env bash
set -euo pipefail

PROTOCOL="${1:-}"

case "$PROTOCOL" in
  direct|cot|self_consistency_3|self_consistency_5|self_refine_1|self_refine_2)
    ;;
  *)
    echo "Invalid protocol: $PROTOCOL" >&2
    exit 2
    ;;
esac

source /home/qwen-token-exp/token_economy/.env.sh
source "$PROJECT_ROOT/.venv/bin/activate"
cd "$PROJECT_ROOT"

mkdir -p logs results/raw results/errors

echo "===== GSM8K main task ====="
echo "protocol=$PROTOCOL"
echo "hostname=$(hostname)"
echo "started_at=$(date -Is)"
echo "python=$(which python)"

python src/run_experiment.py \
  --config configs/gsm8k_main.yaml \
  --protocol "$PROTOCOL" \
  2>&1 | tee -a "logs/gsm8k_main__${PROTOCOL}.log"

status=${PIPESTATUS[0]}

echo "finished_at=$(date -Is)"
echo "exit_status=$status"

exit "$status"
