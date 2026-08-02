#!/usr/bin/env bash
set -euo pipefail

source /home/qwen-token-exp/token_economy/.env.sh
source "$PROJECT_ROOT/.venv/bin/activate"
cd "$PROJECT_ROOT"

CONFIGS=(
  configs/gsm8k_main.yaml
  configs/math500_smoke.yaml
  configs/math500_main.yaml
  configs/gsm8k_sc_seed20260731.yaml
  configs/gsm8k_sc_seed20260801.yaml
  configs/math500_sc_seed20260731.yaml
  configs/math500_sc_seed20260801.yaml
)

python -m py_compile \
  src/run_experiment.py \
  src/analyze_results.py \
  src/analyze_multiseed.py \
  scripts/create_final_matrix.py \
  scripts/run_experiment_bundle.py

for config in "${CONFIGS[@]}"; do
  python src/analyze_results.py \
    --root . \
    --config "$config" \
    --n-boot 10000
done

python src/analyze_multiseed.py \
  --root . \
  --matrix configs/multiseed_matrix.yaml

python - <<'PY'
from pathlib import Path
import json
import yaml

root = Path("/home/qwen-token-exp/token_economy")
configs = [
    "configs/gsm8k_main.yaml",
    "configs/math500_smoke.yaml",
    "configs/math500_main.yaml",
    "configs/gsm8k_sc_seed20260731.yaml",
    "configs/gsm8k_sc_seed20260801.yaml",
    "configs/math500_sc_seed20260731.yaml",
    "configs/math500_sc_seed20260801.yaml",
]

total_rows = 0
for rel in configs:
    cfg = yaml.safe_load((root / rel).read_text(encoding="utf-8"))
    experiment = cfg["experiment_name"]
    seed = int(cfg["run_seed"])
    expected = int(cfg["n_samples"])
    for protocol in cfg["protocols"]:
        path = (
            root / "results/raw" /
            f"{experiment}__{protocol}__seed{seed}.jsonl"
        )
        if not path.is_file():
            raise SystemExit(f"Missing: {path}")
        ids = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("error") is not None:
                raise SystemExit(f"Error row in: {path}")
            ids.append(int(row["sample_id"]))
        if len(ids) != expected or len(set(ids)) != expected:
            raise SystemExit(
                f"Bad rows: {path}, rows={len(ids)}, unique={len(set(ids))}"
            )
        total_rows += len(ids)
        print("OK", path.name, "rows=", len(ids))

print("Final raw-result rows including MATH smoke:", total_rows)
print("FINAL EXPERIMENT MATRIX PASSED.")
PY