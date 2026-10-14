#!/usr/bin/env python3
"""Create the grouped-bar version of the model-performance figure."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    "Exact Evidence\nTop-1",
    "Exact Evidence\nHit@3",
    "Mechanism\nHit@3",
    "Exact Evidence\nJaccard",
    "Empty-set Decision\nAccuracy",
]

SERIES = [
    ("Qwen3-14B Base", [19.25, 38.49, 46.44, 16.58, 0.00], "#9ecae1", "//"),
    ("Qwen3-14B Fine-tuned", [44.35, 71.13, 78.24, 38.22, 95.08], "#2171b5", ""),
    ("Llama3.1-8B Base", [24.27, 51.05, 60.67, 18.93, 0.00], "#fdd0a2", "//"),
    ("Llama3.1-8B Fine-tuned", [50.63, 72.80, 78.66, 38.34, 96.72], "#e6550d", ""),
    ("Qwen3-32B Base", [25.52, 67.78, 77.41, 22.73, 0.00], "#a1d99b", "//"),
    ("Qwen3-32B Fine-tuned", [53.14, 76.15, 83.26, 44.18, 95.08], "#238b45", ""),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("out/figures"))
    output_dir = parser.parse_args().out_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    x = np.arange(len(METRICS))
    width = 0.125
    offsets = (np.arange(len(SERIES)) - (len(SERIES) - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(11.6, 6.25))
    for offset, (label, values, color, hatch) in zip(offsets, SERIES):
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=label,
            color=color,
            edgecolor="#333333",
            linewidth=0.55,
            hatch=hatch,
            zorder=3,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                max(value + 1.2, 1.4),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=7.1,
                color="#222222",
                clip_on=False,
            )

    ax.set_ylabel("Performance (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS)
    ax.set_ylim(0, 108)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=3,
        frameon=False,
        columnspacing=1.3,
        handlelength=2.1,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.17, top=0.82)

    fig.savefig(output_dir / "figure6_grouped_bar_preview.png", dpi=600, facecolor="white")
    fig.savefig(output_dir / "figure6_grouped_bar_preview.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
