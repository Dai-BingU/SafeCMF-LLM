#!/usr/bin/env python3
"""Plot compact distribution-concentration summaries for the revised manuscript."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    analysis: dict[str, Any] = json.loads(args.analysis.read_text(encoding="utf-8"))
    distributions = analysis["label_and_output_distributions"]
    levels = [("exact", "Evidence ID"), ("treatment", "Treatment"), ("mechanism", "Mechanism")]

    reference_top10 = [100 * distributions[key]["test_reference"]["top10_share"] for key, _ in levels]
    prediction_top10 = [100 * distributions[key]["model_prediction"]["top10_share"] for key, _ in levels]
    reference_unique = [distributions[key]["test_reference"]["unique_values"] for key, _ in levels]
    prediction_unique = [distributions[key]["model_prediction"]["unique_values"] for key, _ in levels]
    labels = [label for _, label in levels]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), constrained_layout=True)
    x = np.arange(len(labels))
    width = 0.34
    colors = ("#0072B2", "#D55E00")

    panels = [
        (axes[0], reference_top10, prediction_top10, "(a) Top-10 share", "Share of assignments (%)", 60),
        (axes[1], reference_unique, prediction_unique, "(b) Unique labels", "Number of unique labels", 200),
    ]

    for ax, reference, prediction, title, ylabel, upper in panels:
        ref_bars = ax.bar(x - width / 2, reference, width, color=colors[0], label="Reference labels")
        pred_bars = ax.bar(x + width / 2, prediction, width, color=colors[1], label="Model outputs")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, upper)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bars, is_percent in ((ref_bars, title.startswith("(a)")), (pred_bars, title.startswith("(a)"))):
            for bar in bars:
                value = bar.get_height()
                label = f"{value:.1f}" if is_percent else f"{int(value)}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + upper * 0.025,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )

    axes[0].legend(frameon=False, loc="upper left")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
