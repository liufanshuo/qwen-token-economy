#!/usr/bin/env bash
set -euo pipefail

BUNDLE="${1:-}"
if [[ -z "$BUNDLE" ]]; then
  echo "Usage: $0 <bundle-name>" >&2
  exit 2
fi

source /home/qwen-token-exp/token_economy/.env.sh
source "$PROJECT_ROOT/.venv/bin/activate"
cd "$PROJECT_ROOT"

mkdir -p logs results/raw results/errors results/status results/summary results/figures

python -m py_compile \
  src/run_experiment.py \
  src/analyze_results.py \
  scripts/run_experiment_bundle.py

printf 'bundle=%s\n' "$BUNDLE"
printf 'hostname=%s\n' "$(hostname)"
printf 'started_at=%s\n' "$(date -Is)"
printf 'python=%s\n' "$(which python)"

python scripts/run_experiment_bundle.py \
  --bundle "$BUNDLE" \
  --n-boot 10000

printf 'finished_at=%s\n' "$(date -Is)"