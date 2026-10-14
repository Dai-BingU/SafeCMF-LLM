#!/usr/bin/env python3
"""Plot concrete countermeasure frequencies for training, test, and model outputs."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmfrec.countermeasure_family import countermeasure_text_key  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def label_ids(row: dict[str, Any]) -> list[str]:
    recommendations = row.get("topk") or (row.get("teacher_output") or {}).get("recommendations") or []
    ranked = sorted(recommendations, key=lambda item: int(item.get("rank") or 999))
    return [str(item["evidence_ids"][0]) for item in ranked if item.get("evidence_ids")]


def treatment_values(ids: list[str], evidence: dict[str, dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for evidence_id in ids:
        card = evidence.get(evidence_id)
        if not card:
            continue
        value = countermeasure_text_key(card)
        if value:
            values.append(value)
    return values


def readable_label(value: str) -> str:
    replacements = {
        "hfst": "HFST",
        "vsl": "VSL",
        "stop": "STOP",
    }
    words = [replacements.get(word, word.capitalize()) for word in value.split()]
    return textwrap.fill(" ".join(words), width=38)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--test-labels", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--evidence-store", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--data-out", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=15)
    args = parser.parse_args()

    train_rows = read_jsonl(args.train_labels)
    test_rows = read_jsonl(args.test_labels)
    prediction_rows = read_jsonl(args.predictions)
    evidence = {str(row["evidence_id"]): row for row in read_jsonl(args.evidence_store)}

    train_ids = [evidence_id for row in train_rows for evidence_id in label_ids(row)]
    test_ids = [evidence_id for row in test_rows for evidence_id in label_ids(row)]
    prediction_ids = [
        str(evidence_id)
        for row in prediction_rows
        for evidence_id in (row.get("predicted_evidence_ids") or [])
    ]

    distributions = {
        "Training reference": Counter(treatment_values(train_ids, evidence)),
        "Test reference": Counter(treatment_values(test_ids, evidence)),
        "Model outputs": Counter(treatment_values(prediction_ids, evidence)),
    }
    totals = {name: sum(counts.values()) for name, counts in distributions.items()}
    all_values = set().union(*(counts.keys() for counts in distributions.values()))

    def share(name: str, value: str) -> float:
        return 100.0 * distributions[name][value] / totals[name] if totals[name] else 0.0

    selected = sorted(
        all_values,
        key=lambda value: max(share(name, value) for name in distributions),
        reverse=True,
    )[: args.top_n]
    selected = sorted(selected, key=lambda value: max(share(name, value) for name in distributions))

    series_names = list(distributions)
    series_values = {name: [share(name, value) for value in selected] for name in series_names}

    if args.data_out:
        def make_rows(values: list[str]) -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            for value in values:
                row: dict[str, Any] = {
                    "countermeasure_key": value,
                    "countermeasure": " ".join(readable_label(value).splitlines()),
                }
                for name in series_names:
                    prefix = name.lower().replace(" ", "_")
                    row[f"{prefix}_count"] = distributions[name][value]
                    row[f"{prefix}_share"] = share(name, value) / 100.0
                rows.append(row)
            return rows

        top_rows = make_rows(list(reversed(selected)))
        all_ranked = sorted(
            all_values,
            key=lambda value: max(share(name, value) for name in distributions),
            reverse=True,
        )
        all_rows = make_rows(all_ranked)
        args.data_out.parent.mkdir(parents=True, exist_ok=True)
        args.data_out.write_text(
            json.dumps(
                {
                    "selection_rule": (
                        "Top 15 countermeasure types ranked by the maximum recommendation share observed "
                        "across training references, test references, and model outputs."
                    ),
                    "totals": totals,
                    "rows": top_rows,
                    "all_rows": all_rows,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.7), constrained_layout=True)
    y = np.arange(len(selected))
    height = 0.24
    colors = ("#0072B2", "#009E73", "#D55E00")
    offsets = (-height, 0.0, height)

    for name, color, offset in zip(series_names, colors, offsets):
        bars = ax.barh(y + offset, series_values[name], height, color=color, label=name)
        for bar in bars:
            value = bar.get_width()
            if value >= 0.45:
                ax.text(
                    value + 0.07,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.1f}",
                    va="center",
                    ha="left",
                    fontsize=6.8,
                )

    ax.set_yticks(y, [readable_label(value) for value in selected])
    ax.set_xlabel("Share of recommendations (%)")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    max_value = max(max(values) for values in series_values.values())
    ax.set_xlim(0, max_value * 1.22)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
