#!/usr/bin/env python3
"""Plot rank-frequency distributions for test references and model outputs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmfrec.countermeasure_family import countermeasure_text_key, mechanism_family  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def label_ids(row: dict[str, Any]) -> list[str]:
    recommendations = row.get("topk") or (row.get("teacher_output") or {}).get("recommendations") or []
    ranked = sorted(recommendations, key=lambda item: int(item.get("rank") or 999))
    return [str(item["evidence_ids"][0]) for item in ranked if item.get("evidence_ids")]


def project(
    ids: list[str], evidence: dict[str, dict[str, Any]], fn: Callable[[dict[str, Any]], str]
) -> list[str]:
    return [value for evidence_id in ids if evidence_id in evidence and (value := fn(evidence[evidence_id]))]


def ranked_shares(values: list[str]) -> tuple[list[int], list[float]]:
    counts = Counter(values)
    total = sum(counts.values())
    ordered = sorted(counts.values(), reverse=True)
    return list(range(1, len(ordered) + 1)), [100.0 * value / total for value in ordered]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--evidence-store", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    labels = {row["qid"]: row for row in read_jsonl(args.labels)}
    predictions = {row["qid"]: row for row in read_jsonl(args.predictions)}
    evidence = {str(row["evidence_id"]): row for row in read_jsonl(args.evidence_store)}
    if set(labels) != set(predictions):
        raise ValueError("Reference and prediction QID sets do not match")

    qids = sorted(labels)
    reference_ids = [evidence_id for qid in qids for evidence_id in label_ids(labels[qid])]
    prediction_ids = [
        str(evidence_id)
        for qid in qids
        for evidence_id in (predictions[qid].get("predicted_evidence_ids") or [])
    ]

    levels: list[tuple[str, Callable[[list[str]], list[str]]]] = [
        ("Evidence ID", lambda ids: ids),
        ("Treatment", lambda ids: project(ids, evidence, countermeasure_text_key)),
        ("Mechanism", lambda ids: project(ids, evidence, mechanism_family)),
    ]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), constrained_layout=True)
    colors = {"Reference labels": "#0072B2", "Model outputs": "#D55E00"}

    for index, (title, transform) in enumerate(levels):
        ax = axes[index]
        for label, ids in (("Reference labels", reference_ids), ("Model outputs", prediction_ids)):
            ranks, shares = ranked_shares(transform(ids))
            ax.plot(ranks, shares, color=colors[label], linewidth=1.6, label=label)
        ax.set_title(f"({chr(97 + index)}) {title}")
        ax.set_xlabel("Frequency rank")
        ax.set_yscale("log")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if index == 0:
            ax.set_ylabel("Assignment share (%)")
            ax.legend(frameon=False, loc="upper right")

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
