from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import StrMethodFormatter


PROTOCOL_ORDER = [
    "direct",
    "cot",
    "self_consistency_3",
    "self_consistency_5",
    "self_refine_1",
    "self_refine_2",
]

PROTOCOL_LABELS = {
    "direct": "Direct",
    "cot": "CoT",
    "self_consistency_3": "SC@3",
    "self_consistency_5": "SC@5",
    "self_refine_1": "SR@1",
    "self_refine_2": "SR@2",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Plot GSM8K total-token consumption divided by the number "
            "of correct answers for all six reasoning protocols."
        )
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=(
            root
            / "results"
            / "summary"
            / "gsm8k_main_qwen25_7b__summary.csv"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=(
            root
            / "results"
            / "figures"
            / "gsm8k_main_qwen25_7b__tokens_per_correct_bar"
        ),
    )
    return parser.parse_args()


def load_gsm8k_summary(path: Path) -> pd.DataFrame:
    summary = pd.read_csv(path)
    required_columns = {
        "protocol",
        "n",
        "correct_count",
        "mean_total_tokens",
        "tokens_per_correct",
    }
    missing_columns = sorted(required_columns - set(summary.columns))
    if missing_columns:
        raise ValueError(
            f"Summary is missing required columns: {missing_columns}"
        )

    protocols = summary["protocol"].astype(str).tolist()
    if set(protocols) != set(PROTOCOL_ORDER) or len(protocols) != 6:
        raise ValueError(
            "Expected exactly the six GSM8K protocols; found: "
            f"{protocols}"
        )

    ordered = (
        summary.set_index("protocol").loc[PROTOCOL_ORDER].reset_index()
    )
    if (ordered["correct_count"] <= 0).any():
        bad = ordered.loc[
            ordered["correct_count"] <= 0, "protocol"
        ].tolist()
        raise ValueError(
            "Cannot divide by zero correct answers for protocols: "
            f"{bad}"
        )

    # Token counts were summed as integers in the raw analysis. Recover the
    # protocol totals from the stored per-item mean and sample count, then use
    # the requested definition: all tokens / all correct answers.
    ordered["total_tokens"] = np.rint(
        ordered["mean_total_tokens"] * ordered["n"]
    ).astype(np.int64)
    ordered["computed_tokens_per_correct"] = (
        ordered["total_tokens"] / ordered["correct_count"]
    )

    if not np.allclose(
        ordered["computed_tokens_per_correct"],
        ordered["tokens_per_correct"],
        rtol=1e-12,
        atol=1e-9,
    ):
        raise ValueError(
            "Recomputed total_tokens / correct_count does not match "
            "the summary's tokens_per_correct column."
        )

    return ordered


def create_figure(summary: pd.DataFrame) -> plt.Figure:
    labels = [PROTOCOL_LABELS[p] for p in summary["protocol"]]
    values = summary["computed_tokens_per_correct"].to_numpy()
    positions = np.arange(len(labels))

    fig, ax = plt.subplots(
        figsize=(10, 6),
        constrained_layout=True,
    )
    bars = ax.bar(
        positions,
        values,
        width=0.68,
        color="#2C7FB8",
        edgecolor="#1A1A1A",
        linewidth=0.8,
    )

    ax.bar_label(
        bars,
        labels=[f"{value:,.1f}" for value in values],
        padding=4,
        fontsize=10,
    )
    ax.set_xticks(positions, labels)
    ax.set_xlabel("Reasoning protocol")
    ax.set_ylabel("Total tokens / correct answers")
    ax.set_title("GSM8K: Amortized token consumption per correct answer")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.set_ylim(0, values.max() * 1.14)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


def main() -> None:
    args = parse_args()
    summary = load_gsm8k_summary(args.summary.resolve())
    figure = create_figure(summary)

    output_prefix = args.output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    svg_path = output_prefix.with_suffix(".svg")
    figure.savefig(png_path, dpi=300, facecolor="white")
    figure.savefig(svg_path, facecolor="white")
    plt.close(figure)

    print(
        summary[
            [
                "protocol",
                "total_tokens",
                "correct_count",
                "computed_tokens_per_correct",
            ]
        ].to_string(index=False)
    )
    print(f"Saved PNG: {png_path}")
    print(f"Saved SVG: {svg_path}")


if __name__ == "__main__":
    main()
