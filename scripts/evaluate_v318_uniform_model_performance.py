#!/usr/bin/env python3
"""Evaluate model predictions with the unified v3.18 reporting metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def index_unique(path: Path, key: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in read_jsonl(path):
        value = str(row.get(key) or "")
        if not value or value in result:
            raise ValueError(f"Missing or duplicate {key} in {path}")
        result[value] = row
    if not result:
        raise ValueError(f"Empty input: {path}")
    return result


def rate(numerator: float, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def reference_ids(row: dict[str, Any]) -> list[str]:
    ranked = sorted(
        row.get("topk") or (row.get("teacher_output") or {}).get("recommendations") or [],
        key=lambda item: int(item.get("rank") or 999),
    )
    return [str(item["evidence_ids"][0]) for item in ranked if item.get("evidence_ids")]


def jaccard(left: list[str], right: list[str]) -> float:
    a, b = set(left), set(right)
    return len(a & b) / len(a | b) if a or b else 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--evidence-store", type=Path, required=True)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root
    if project_root.exists() and str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from cmfrec.countermeasure_family import mechanism_family

    predictions = index_unique(args.predictions, "qid")
    labels = index_unique(args.labels, "qid")
    evidence = index_unique(args.evidence_store, "evidence_id")
    if set(predictions) != set(labels):
        raise ValueError(f"Prediction/label QID mismatch: {len(predictions)}/{len(labels)}")

    nonempty = [qid for qid, row in labels.items() if reference_ids(row)]
    empty = [qid for qid, row in labels.items() if not reference_ids(row)]
    exact_top1 = exact_hit3 = mechanism_hit3 = 0
    exact_jaccard = 0.0

    for qid in nonempty:
        expected = reference_ids(labels[qid])
        predicted = [str(value) for value in predictions[qid].get("predicted_evidence_ids") or []]
        exact_top1 += bool(predicted and predicted[0] == expected[0])
        exact_hit3 += bool(set(predicted) & set(expected))
        exact_jaccard += jaccard(predicted, expected)
        expected_mechanisms = {
            mechanism_family(evidence[eid]) for eid in expected if eid in evidence
        }
        predicted_mechanisms = {
            mechanism_family(evidence[eid]) for eid in predicted if eid in evidence
        }
        mechanism_hit3 += bool(expected_mechanisms & predicted_mechanisms)

    summary = {
        "queries": len(labels),
        "nonempty_reference_queries": len(nonempty),
        "empty_reference_queries": len(empty),
        "exact_evidence_top1": rate(exact_top1, len(nonempty)),
        "exact_evidence_hit_at_3": rate(exact_hit3, len(nonempty)),
        "mechanism_hit_at_3": rate(mechanism_hit3, len(nonempty)),
        "exact_evidence_jaccard": rate(exact_jaccard, len(nonempty)),
        "empty_set_decision_accuracy": rate(sum(
            not predictions[qid].get("predicted_evidence_ids") for qid in empty
        ), len(empty)),
        "json_valid_rate": sum(not row.get("json_error") for row in predictions.values()) / len(labels),
        "all_ids_in_candidates_rate": sum(
            bool(row.get("all_ids_in_frozen_candidates")) for row in predictions.values()
        ) / len(labels),
        "duplicate_rows": sum(bool(row.get("duplicate_ids")) for row in predictions.values()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
