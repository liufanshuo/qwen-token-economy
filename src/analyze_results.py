from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import binomtest


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    if len(values) == 0:
        raise ValueError("Cannot bootstrap an empty array.")

    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    chunk_size = 1000

    for start in range(0, n_boot, chunk_size):
        stop = min(start + chunk_size, n_boot)
        indices = rng.integers(
            0,
            len(values),
            size=(stop - start, len(values)),
        )
        means[start:stop] = values[indices].mean(axis=1)

    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def paired_bootstrap_delta_ci(
    direct: np.ndarray,
    other: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    if len(direct) != len(other) or len(direct) == 0:
        raise ValueError("Paired arrays must have the same nonzero length.")

    differences = other.astype(float) - direct.astype(float)
    return bootstrap_mean_ci(
        differences,
        n_boot=n_boot,
        seed=seed,
    )


def exact_mcnemar_pvalue(
    direct: np.ndarray,
    other: np.ndarray,
) -> tuple[int, int, float]:
    direct_only = int(
        np.sum((direct == 1) & (other == 0))
    )
    other_only = int(
        np.sum((direct == 0) & (other == 1))
    )
    discordant = direct_only + other_only

    if discordant == 0:
        return direct_only, other_only, 1.0

    p_value = float(
        binomtest(
            other_only,
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    )
    return direct_only, other_only, p_value


def pareto_mask(
    tokens: np.ndarray,
    accuracy: np.ndarray,
) -> np.ndarray:
    keep = np.ones(len(tokens), dtype=bool)

    for index in range(len(tokens)):
        for competitor in range(len(tokens)):
            if index == competitor:
                continue

            no_more_tokens = tokens[competitor] <= tokens[index]
            no_less_accuracy = (
                accuracy[competitor] >= accuracy[index]
            )
            strictly_better = (
                tokens[competitor] < tokens[index]
                or accuracy[competitor] > accuracy[index]
            )
            if no_more_tokens and no_less_accuracy and strictly_better:
                keep[index] = False
                break

    return keep


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("Config must contain one YAML mapping.")

    required = {
        "experiment_name",
        "dataset",
        "n_samples",
        "run_seed",
        "protocols",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Config is missing keys: {missing}")

    protocols = list(config["protocols"])
    if len(protocols) != len(set(protocols)):
        raise ValueError("Config contains duplicate protocols.")

    return config


def read_experiment_rows(
    root: Path,
    config: dict,
) -> pd.DataFrame:
    experiment_name = str(config["experiment_name"])
    dataset_name = str(config["dataset"])
    run_seed = int(config["run_seed"])
    expected_protocols = list(config["protocols"])
    rows: list[dict] = []

    for protocol in expected_protocols:
        path = (
            root
            / "results"
            / "raw"
            / (
                f"{experiment_name}__{protocol}"
                f"__seed{run_seed}.jsonl"
            )
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing result file for {protocol}: {path}"
            )

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
                    raise ValueError(
                        f"Raw result contains an error row at "
                        f"{path}:{line_number}"
                    )
                if record.get("experiment_name") != experiment_name:
                    raise ValueError(
                        f"Unexpected experiment_name at "
                        f"{path}:{line_number}"
                    )
                if record.get("dataset") != dataset_name:
                    raise ValueError(
                        f"Unexpected dataset at {path}:{line_number}"
                    )
                if record.get("protocol") != protocol:
                    raise ValueError(
                        f"Unexpected protocol at {path}:{line_number}"
                    )
                if int(record.get("run_seed")) != run_seed:
                    raise ValueError(
                        f"Unexpected run_seed at {path}:{line_number}"
                    )

                sample_id = int(record["sample_id"])
                if sample_id in seen_ids:
                    raise ValueError(
                        f"Duplicate sample_id={sample_id} in {path}"
                    )
                seen_ids.add(sample_id)

                calls = record.get("calls")
                if not isinstance(calls, list) or not calls:
                    raise ValueError(
                        f"Missing calls at {path}:{line_number}"
                    )

                call_input = sum(
                    int(call["input_tokens"]) for call in calls
                )
                call_output = sum(
                    int(call["output_tokens"]) for call in calls
                )
                call_total = sum(
                    int(call["total_tokens"]) for call in calls
                )

                if int(record["input_tokens"]) != call_input:
                    raise ValueError(
                        f"Input-token mismatch at {path}:{line_number}"
                    )
                if int(record["output_tokens"]) != call_output:
                    raise ValueError(
                        f"Output-token mismatch at {path}:{line_number}"
                    )
                if int(record["total_tokens"]) != call_total:
                    raise ValueError(
                        f"Total-token mismatch at {path}:{line_number}"
                    )
                if call_total != call_input + call_output:
                    raise ValueError(
                        f"Call token arithmetic mismatch at "
                        f"{path}:{line_number}"
                    )

                rows.append({
                    "protocol": protocol,
                    "sample_id": sample_id,
                    "run_seed": run_seed,
                    "correct": int(bool(record["correct"])),
                    "input_tokens": int(record["input_tokens"]),
                    "output_tokens": int(record["output_tokens"]),
                    "total_tokens": int(record["total_tokens"]),
                    "generation_latency_sec": float(
                        record["generation_latency_sec"]
                    ),
                    "end_to_end_latency_sec": float(
                        record["end_to_end_latency_sec"]
                    ),
                    "num_calls": int(record["num_calls"]),
                    "peak_gpu_memory_mb": float(
                        record["peak_gpu_memory_mb"]
                    ),
                    "parse_error": int(
                        bool(record["parse_error"])
                    ),
                    "truncated": int(
                        bool(record["truncated"])
                    ),
                    "extraction_method": str(
                        record.get("extraction_method", "unknown")
                    ),
                })

    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        raise ValueError("No experiment rows were loaded.")

    expected_n = int(config["n_samples"])
    counts = dataframe.groupby("protocol").size().to_dict()
    bad_counts = {
        protocol: int(counts.get(protocol, 0))
        for protocol in expected_protocols
        if int(counts.get(protocol, 0)) != expected_n
    }
    if bad_counts:
        raise ValueError(
            f"Protocol row counts do not equal n_samples={expected_n}: "
            f"{bad_counts}"
        )

    sample_sets = {
        protocol: set(
            group["sample_id"].astype(int).tolist()
        )
        for protocol, group in dataframe.groupby("protocol")
    }
    reference_protocol = expected_protocols[0]
    reference_ids = sample_sets[reference_protocol]

    mismatches = {
        protocol: {
            "missing_vs_reference": sorted(
                reference_ids - sample_ids
            ),
            "extra_vs_reference": sorted(
                sample_ids - reference_ids
            ),
        }
        for protocol, sample_ids in sample_sets.items()
        if sample_ids != reference_ids
    }
    if mismatches:
        raise ValueError(
            "Protocols do not use identical sample IDs:\n"
            + json.dumps(mismatches, ensure_ascii=False, indent=2)
        )

    return dataframe


def build_summary(
    dataframe: pd.DataFrame,
    config: dict,
    *,
    n_boot: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    protocol_order = list(config["protocols"])
    summary_rows: list[dict] = []

    for protocol in protocol_order:
        group = dataframe.loc[
            dataframe["protocol"] == protocol
        ].copy()
        accuracy = float(group["correct"].mean())
        low, high = bootstrap_mean_ci(
            group["correct"].to_numpy(dtype=float),
            n_boot=n_boot,
            seed=int(config["run_seed"]),
        )
        correct_count = int(group["correct"].sum())

        summary_rows.append({
            "protocol": protocol,
            "n": len(group),
            "correct_count": correct_count,
            "accuracy": accuracy,
            "accuracy_ci_low": low,
            "accuracy_ci_high": high,
            "mean_input_tokens": group["input_tokens"].mean(),
            "mean_output_tokens": group["output_tokens"].mean(),
            "mean_total_tokens": group["total_tokens"].mean(),
            "median_total_tokens": group["total_tokens"].median(),
            "p95_total_tokens": group["total_tokens"].quantile(0.95),
            "tokens_per_correct": (
                group["total_tokens"].sum() / correct_count
                if correct_count > 0
                else np.inf
            ),
            "mean_generation_latency_sec": (
                group["generation_latency_sec"].mean()
            ),
            "p95_generation_latency_sec": (
                group["generation_latency_sec"].quantile(0.95)
            ),
            "mean_end_to_end_latency_sec": (
                group["end_to_end_latency_sec"].mean()
            ),
            "p95_end_to_end_latency_sec": (
                group["end_to_end_latency_sec"].quantile(0.95)
            ),
            "mean_calls": group["num_calls"].mean(),
            "mean_peak_gpu_memory_mb": (
                group["peak_gpu_memory_mb"].mean()
            ),
            "max_peak_gpu_memory_mb": (
                group["peak_gpu_memory_mb"].max()
            ),
            "parse_error_rate": group["parse_error"].mean(),
            "truncation_rate": group["truncated"].mean(),
            "numeric_fallback_rate": (
                group["extraction_method"]
                .eq("numeric_fallback")
                .mean()
            ),
        })

    summary = pd.DataFrame(summary_rows)
    direct_row = summary.loc[
        summary["protocol"] == "direct"
    ]
    if direct_row.empty:
        raise ValueError("The config has no Direct baseline.")

    direct_tokens = float(
        direct_row.iloc[0]["mean_total_tokens"]
    )
    direct_accuracy = float(
        direct_row.iloc[0]["accuracy"]
    )

    summary["relative_token_cost"] = (
        summary["mean_total_tokens"] / direct_tokens
    )
    summary["accuracy_delta_vs_direct"] = (
        summary["accuracy"] - direct_accuracy
    )
    summary["additional_tokens_vs_direct"] = (
        summary["mean_total_tokens"] - direct_tokens
    )
    summary[
        "accuracy_points_per_additional_1k_tokens_vs_direct"
    ] = np.where(
        summary["additional_tokens_vs_direct"] > 0,
        (
            summary["accuracy_delta_vs_direct"]
            / summary["additional_tokens_vs_direct"]
            * 1000
            * 100
        ),
        np.nan,
    )

    summary["pareto_efficient"] = pareto_mask(
        summary["mean_total_tokens"].to_numpy(dtype=float),
        summary["accuracy"].to_numpy(dtype=float),
    )

    pivot = dataframe.pivot(
        index="sample_id",
        columns="protocol",
        values="correct",
    ).sort_index()
    direct = pivot["direct"].to_numpy(dtype=int)
    paired_rows: list[dict] = []

    for protocol_index, protocol in enumerate(protocol_order):
        other = pivot[protocol].to_numpy(dtype=int)
        delta = float(np.mean(other - direct))
        delta_low, delta_high = paired_bootstrap_delta_ci(
            direct,
            other,
            n_boot=n_boot,
            seed=int(config["run_seed"]) + protocol_index,
        )
        direct_only, other_only, p_value = exact_mcnemar_pvalue(
            direct,
            other,
        )
        paired_rows.append({
            "protocol": protocol,
            "paired_accuracy_delta_vs_direct": delta,
            "paired_delta_ci_low": delta_low,
            "paired_delta_ci_high": delta_high,
            "direct_correct_other_wrong": direct_only,
            "direct_wrong_other_correct": other_only,
            "mcnemar_exact_pvalue": p_value,
        })

    paired = pd.DataFrame(paired_rows)
    summary = summary.merge(
        paired,
        on="protocol",
        how="left",
        validate="one_to_one",
    )
    return summary, paired


def save_outputs(
    root: Path,
    dataframe: pd.DataFrame,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    config: dict,
) -> None:
    experiment_name = str(config["experiment_name"])
    dataset_name = str(config["dataset"])
    protocol_order = list(config["protocols"])

    summary_dir = root / "results" / "summary"
    figure_dir = root / "results" / "figures"
    summary_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        summary_dir / f"{experiment_name}__summary.csv",
        index=False,
    )
    paired.to_csv(
        summary_dir / f"{experiment_name}__paired_vs_direct.csv",
        index=False,
    )
    dataframe.to_csv(
        summary_dir / f"{experiment_name}__per_item.csv",
        index=False,
    )

    ordered_summary = (
        summary.set_index("protocol")
        .loc[protocol_order]
        .reset_index()
    )
    positions = np.arange(len(ordered_summary))

    y_error = np.vstack([
        (
            ordered_summary["accuracy"]
            - ordered_summary["accuracy_ci_low"]
        ),
        (
            ordered_summary["accuracy_ci_high"]
            - ordered_summary["accuracy"]
        ),
    ])
    plt.figure(figsize=(10, 6))
    plt.bar(
        positions,
        ordered_summary["accuracy"],
        yerr=y_error,
        capsize=4,
    )
    plt.xticks(
        positions,
        ordered_summary["protocol"],
        rotation=30,
        ha="right",
    )
    plt.ylabel("Accuracy")
    plt.title(f"{dataset_name}: Accuracy by reasoning protocol")
    plt.tight_layout()
    plt.savefig(
        figure_dir / f"{experiment_name}__accuracy_bar.png",
        dpi=200,
    )
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.bar(
        positions,
        ordered_summary["mean_input_tokens"],
        label="Input tokens",
    )
    plt.bar(
        positions,
        ordered_summary["mean_output_tokens"],
        bottom=ordered_summary["mean_input_tokens"],
        label="Output tokens",
    )
    plt.xticks(
        positions,
        ordered_summary["protocol"],
        rotation=30,
        ha="right",
    )
    plt.ylabel("Mean tokens per item")
    plt.title(f"{dataset_name}: Mean input and output tokens")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figure_dir / f"{experiment_name}__token_stacked_bar.png",
        dpi=200,
    )
    plt.close()

    tokens = summary["mean_total_tokens"].to_numpy(dtype=float)
    accuracy = summary["accuracy"].to_numpy(dtype=float)
    plt.figure(figsize=(9, 6))
    plt.scatter(tokens, accuracy)
    for _, row in summary.iterrows():
        plt.annotate(
            row["protocol"],
            (row["mean_total_tokens"], row["accuracy"]),
            xytext=(5, 5),
            textcoords="offset points",
        )
    frontier = summary.loc[
        summary["pareto_efficient"]
    ].sort_values("mean_total_tokens")
    plt.plot(
        frontier["mean_total_tokens"],
        frontier["accuracy"],
    )
    plt.xlabel("Mean total tokens per item")
    plt.ylabel("Accuracy")
    plt.title(
        f"{dataset_name}: Empirical accuracy-token Pareto frontier"
    )
    plt.tight_layout()
    plt.savefig(
        figure_dir / f"{experiment_name}__accuracy_vs_tokens.png",
        dpi=200,
    )
    plt.close()

    values = [
        dataframe.loc[
            dataframe["protocol"] == protocol,
            "total_tokens",
        ].to_numpy()
        for protocol in protocol_order
    ]
    plt.figure(figsize=(10, 6))
    plt.boxplot(
        values,
        labels=protocol_order,
        showfliers=False,
    )
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Total tokens per item")
    plt.title(f"{dataset_name}: Token distribution")
    plt.tight_layout()
    plt.savefig(
        figure_dir / f"{experiment_name}__token_boxplot.png",
        dpi=200,
    )
    plt.close()

    plt.figure(figsize=(9, 6))
    for protocol in protocol_order:
        group = dataframe.loc[
            dataframe["protocol"] == protocol
        ]
        plt.scatter(
            group["total_tokens"],
            group["end_to_end_latency_sec"],
            s=12,
            alpha=0.5,
            label=protocol,
        )
    plt.xlabel("Total tokens per item")
    plt.ylabel("End-to-end latency (seconds)")
    plt.title(f"{dataset_name}: Tokens vs end-to-end latency")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(
        figure_dir / f"{experiment_name}__tokens_vs_latency.png",
        dpi=200,
    )
    plt.close()

    valid = ordered_summary.dropna(
        subset=[
            "accuracy_points_per_additional_1k_tokens_vs_direct"
        ]
    )
    plt.figure(figsize=(10, 6))
    plt.bar(
        valid["protocol"],
        valid[
            "accuracy_points_per_additional_1k_tokens_vs_direct"
        ],
    )
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(
        "Accuracy points per additional 1K tokens vs Direct"
    )
    plt.title(
        f"{dataset_name}: Marginal token efficiency vs Direct"
    )
    plt.tight_layout()
    plt.savefig(
        figure_dir / f"{experiment_name}__marginal_efficiency.png",
        dpi=200,
    )
    plt.close()

    print(summary.to_string(index=False))
    print(
        "Saved summary:",
        summary_dir / f"{experiment_name}__summary.csv",
    )
    print("Saved figures:", figure_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()

    if args.n_boot <= 0:
        raise ValueError("--n-boot must be positive.")

    root = Path(args.root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config_path = config_path.resolve()

    config = load_config(config_path)
    dataframe = read_experiment_rows(root, config)
    summary, paired = build_summary(
        dataframe,
        config,
        n_boot=args.n_boot,
    )
    save_outputs(
        root,
        dataframe,
        summary,
        paired,
        config,
    )


if __name__ == "__main__":
    main()