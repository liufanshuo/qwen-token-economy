from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return value


def read_seed_protocol(
    *,
    root: Path,
    dataset: str,
    experiment_name: str,
    seed: int,
    protocol: str,
    expected_n: int,
) -> dict:
    path = (
        root
        / "results"
        / "raw"
        / f"{experiment_name}__{protocol}__seed{seed}.jsonl"
    )
    if not path.is_file():
        raise FileNotFoundError(path)

    rows: list[dict] = []
    seen_ids: set[int] = set()

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON at {path}:{line_number}"
                ) from exc

            if record.get("error") is not None:
                raise ValueError(f"Error row at {path}:{line_number}")
            if record.get("dataset") != dataset:
                raise ValueError(f"Dataset mismatch at {path}:{line_number}")
            if record.get("experiment_name") != experiment_name:
                raise ValueError(f"Experiment mismatch at {path}:{line_number}")
            if record.get("protocol") != protocol:
                raise ValueError(f"Protocol mismatch at {path}:{line_number}")
            if int(record.get("run_seed")) != seed:
                raise ValueError(f"Seed mismatch at {path}:{line_number}")

            sample_id = int(record["sample_id"])
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample_id={sample_id} in {path}")
            seen_ids.add(sample_id)

            calls = record.get("calls")
            if not isinstance(calls, list) or not calls:
                raise ValueError(f"Missing calls at {path}:{line_number}")
            call_total = sum(int(call["total_tokens"]) for call in calls)
            if int(record["total_tokens"]) != call_total:
                raise ValueError(f"Token mismatch at {path}:{line_number}")

            rows.append(record)

    if len(rows) != expected_n:
        raise ValueError(
            f"Expected {expected_n} rows but found {len(rows)}: {path}"
        )

    return {
        "dataset": dataset,
        "experiment_name": experiment_name,
        "seed": seed,
        "protocol": protocol,
        "n": len(rows),
        "accuracy": float(np.mean([bool(row["correct"]) for row in rows])),
        "mean_input_tokens": float(
            np.mean([int(row["input_tokens"]) for row in rows])
        ),
        "mean_output_tokens": float(
            np.mean([int(row["output_tokens"]) for row in rows])
        ),
        "mean_total_tokens": float(
            np.mean([int(row["total_tokens"]) for row in rows])
        ),
        "mean_end_to_end_latency_sec": float(
            np.mean([
                float(row["end_to_end_latency_sec"])
                for row in rows
            ])
        ),
        "parse_error_rate": float(
            np.mean([bool(row["parse_error"]) for row in rows])
        ),
        "truncation_rate": float(
            np.mean([bool(row["truncated"]) for row in rows])
        ),
    }


def aggregate_seed_metrics(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    metrics = [
        "accuracy",
        "mean_input_tokens",
        "mean_output_tokens",
        "mean_total_tokens",
        "mean_end_to_end_latency_sec",
        "parse_error_rate",
        "truncation_rate",
    ]

    for protocol, group in per_seed.groupby("protocol", sort=False):
        row: dict[str, object] = {
            "protocol": protocol,
            "seed_count": len(group),
        }
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_seed_mean"] = float(values.mean())
            row[f"{metric}_seed_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
            row[f"{metric}_seed_min"] = float(values.min())
            row[f"{metric}_seed_max"] = float(values.max())
        rows.append(row)

    return pd.DataFrame(rows)


def protocol_deltas(per_seed: pd.DataFrame) -> pd.DataFrame:
    accuracy = per_seed.pivot(
        index="seed",
        columns="protocol",
        values="accuracy",
    )
    tokens = per_seed.pivot(
        index="seed",
        columns="protocol",
        values="mean_total_tokens",
    )

    required = {"self_consistency_3", "self_consistency_5"}
    if not required.issubset(accuracy.columns):
        raise ValueError("Missing SC@3 or SC@5 in multiseed data")

    return pd.DataFrame({
        "seed": accuracy.index.astype(int),
        "accuracy_delta_sc5_minus_sc3": (
            accuracy["self_consistency_5"]
            - accuracy["self_consistency_3"]
        ).to_numpy(),
        "mean_total_tokens_delta_sc5_minus_sc3": (
            tokens["self_consistency_5"]
            - tokens["self_consistency_3"]
        ).to_numpy(),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--matrix",
        default="configs/multiseed_matrix.yaml",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    matrix_path = Path(args.matrix)
    if not matrix_path.is_absolute():
        matrix_path = root / matrix_path
    matrix = load_yaml(matrix_path.resolve())

    protocols = list(matrix["protocols"])
    datasets = matrix["datasets"]
    if not isinstance(datasets, dict):
        raise TypeError("datasets must be a mapping")

    summary_dir = root / "results" / "summary"
    figure_dir = root / "results" / "figures"
    summary_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    for dataset, spec in datasets.items():
        expected_n = int(spec["n_samples"])
        runs = list(spec["runs"])
        rows: list[dict] = []

        for run in runs:
            seed = int(run["seed"])
            experiment_name = str(run["experiment_name"])
            for protocol in protocols:
                rows.append(
                    read_seed_protocol(
                        root=root,
                        dataset=str(dataset),
                        experiment_name=experiment_name,
                        seed=seed,
                        protocol=protocol,
                        expected_n=expected_n,
                    )
                )

        per_seed = pd.DataFrame(rows)
        aggregate = aggregate_seed_metrics(per_seed)
        deltas = protocol_deltas(per_seed)

        per_seed_path = summary_dir / f"{dataset}_multiseed__per_seed.csv"
        aggregate_path = summary_dir / f"{dataset}_multiseed__aggregate.csv"
        delta_path = summary_dir / f"{dataset}_multiseed__sc5_vs_sc3.csv"
        per_seed.to_csv(per_seed_path, index=False)
        aggregate.to_csv(aggregate_path, index=False)
        deltas.to_csv(delta_path, index=False)

        ordered = aggregate.set_index("protocol").loc[protocols].reset_index()
        positions = np.arange(len(ordered))
        plt.figure(figsize=(8, 5))
        plt.errorbar(
            positions,
            ordered["accuracy_seed_mean"],
            yerr=ordered["accuracy_seed_std"],
            fmt="o",
            capsize=4,
        )
        plt.xticks(positions, ordered["protocol"], rotation=20)
        plt.ylabel("Accuracy across run seeds")
        plt.title(f"{dataset}: Self-Consistency seed variability")
        plt.tight_layout()
        plt.savefig(
            figure_dir / f"{dataset}_multiseed__accuracy.png",
            dpi=200,
        )
        plt.close()

        print(f"\n===== {dataset} multiseed =====")
        print(per_seed.to_string(index=False))
        print(aggregate.to_string(index=False))
        print(deltas.to_string(index=False))
        print("Saved:", per_seed_path)
        print("Saved:", aggregate_path)
        print("Saved:", delta_path)


if __name__ == "__main__":
    main()
    