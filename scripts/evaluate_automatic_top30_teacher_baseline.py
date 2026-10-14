#!/usr/bin/env python3
"""Validate and evaluate automatic Top-30 teacher-baseline predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmfrec.countermeasure_family import mechanism_family  # noqa: E402


SPLIT = ROOT / "data/release/v3_18_final_split_2635_300"
DEFAULT_INPUTS = SPLIT / "ablation_candidates/full.top30.inputs.label_free.jsonl"
DEFAULT_PREDICTIONS = SPLIT / "ablation_candidates/automatic_top30_teacher_baseline.gpt55.predictions.jsonl"
DEFAULT_LABELS = SPLIT / "reference_labels.test_300.evaluation_only.jsonl"
DEFAULT_EVIDENCE = ROOT / "data/evidence_store.jsonl"
DEFAULT_OUT = SPLIT / "ablation_candidates/automatic_top30_teacher_baseline.gpt55.metrics.json"
QWEN_METRICS = SPLIT / "ablation_candidates/module_ablation.final_metrics.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def reference_ids(row: dict[str, Any]) -> list[str]:
    ranked = sorted(
        row.get("topk") or (row.get("teacher_output") or {}).get("recommendations") or [],
        key=lambda item: int(item.get("rank") or 999),
    )
    return [str(item["evidence_ids"][0]) for item in ranked if item.get("evidence_ids")]


def project(ids: list[str], evidence: dict[str, dict[str, Any]], fn: Callable[[dict[str, Any]], str]) -> list[str]:
    return [value for eid in ids if eid in evidence and (value := fn(evidence[eid]))]


def evidence_family(row: dict[str, Any]) -> str:
    return mechanism_family(
        {
            "countermeasure": row.get("countermeasure"),
            "countermeasure_category": row.get("category"),
            "countermeasure_subcategory": row.get("subcategory"),
        }
    )


def jaccard(left: list[str], right: list[str]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if a or b else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--evidence-store", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    inputs = {str(row["qid"]): row for row in read_jsonl(args.inputs)}
    predictions = {str(row["qid"]): row for row in read_jsonl(args.predictions)}
    if len(predictions) != len(read_jsonl(args.predictions)):
        raise ValueError("Duplicate qid found in predictions")
    if set(predictions) != set(inputs):
        missing = sorted(set(inputs) - set(predictions))[:10]
        extra = sorted(set(predictions) - set(inputs))[:10]
        raise ValueError(f"Prediction/input mismatch: missing={missing}, extra={extra}")

    structure_rows: list[dict[str, Any]] = []
    for qid, pred in predictions.items():
        allowed = {str(eid) for eid in inputs[qid].get("candidate_evidence_ids") or []}
        raw_ids = pred.get("predicted_evidence_ids")
        if raw_ids is None:
            raw_ids = pred.get("predicted_ids")
        ids = [str(eid) for eid in raw_ids or []]
        issues = []
        if len(ids) > 3:
            issues.append("more_than_three_ids")
        if len(ids) != len(set(ids)):
            issues.append("duplicate_ids")
        if any(eid not in allowed for eid in ids):
            issues.append("id_outside_candidate_pool")
        if pred.get("json_error"):
            issues.append("json_or_api_error")
        structure_rows.append({"qid": qid, "ids": ids, "issues": issues})

    labels = {str(row["qid"]): row for row in read_jsonl(args.labels)}
    evidence = {str(row["evidence_id"]): row for row in read_jsonl(args.evidence_store)}
    if set(labels) != set(inputs):
        raise ValueError("Label/input qid mismatch")

    rows: list[dict[str, Any]] = []
    nonempty_rows: list[dict[str, Any]] = []
    empty_rows: list[dict[str, Any]] = []
    for qid in sorted(labels):
        expected = reference_ids(labels[qid])
        raw_ids = predictions[qid].get("predicted_evidence_ids")
        if raw_ids is None:
            raw_ids = predictions[qid].get("predicted_ids")
        predicted = [str(eid) for eid in raw_ids or []]
        metrics = {
            "exact": {
                "top1": bool(expected and predicted and expected[0] == predicted[0]),
                "hit_at_3": bool(set(expected) & set(predicted)),
                "jaccard": jaccard(expected, predicted),
                "expected": expected,
                "predicted": predicted,
            },
            "mechanism": {
                "hit_at_3": bool(
                    set(project(expected, evidence, evidence_family))
                    & set(project(predicted, evidence, evidence_family))
                ),
                "expected": project(expected, evidence, evidence_family),
                "predicted": project(predicted, evidence, evidence_family),
            },
        }
        row = {"qid": qid, "expected_ids": expected, "predicted_ids": predicted, "metrics": metrics}
        rows.append(row)
        if expected:
            nonempty_rows.append(row)
        else:
            empty_rows.append(row)

    if len(nonempty_rows) != 239 or len(empty_rows) != 61:
        raise ValueError(f"Unexpected reference split: nonempty={len(nonempty_rows)}, empty={len(empty_rows)}")

    teacher = {
        "exact_evidence_top1": sum(row["metrics"]["exact"]["top1"] for row in nonempty_rows) / len(nonempty_rows),
        "exact_evidence_hit_at_3": sum(row["metrics"]["exact"]["hit_at_3"] for row in nonempty_rows) / len(nonempty_rows),
        "mechanism_hit_at_3": sum(row["metrics"]["mechanism"]["hit_at_3"] for row in nonempty_rows) / len(nonempty_rows),
        "exact_evidence_jaccard": sum(row["metrics"]["exact"]["jaccard"] for row in nonempty_rows) / len(nonempty_rows),
        "empty_set_decision_accuracy": sum(not row["predicted_ids"] for row in empty_rows) / len(empty_rows),
    }
    qwen_data = json.loads(QWEN_METRICS.read_text(encoding="utf-8"))
    qwen_full = qwen_data["variants"]["full"]
    qwen = {
        "exact_evidence_top1": qwen_full["metrics"]["exact"]["top1"],
        "exact_evidence_hit_at_3": qwen_full["metrics"]["exact"]["hit_at_3"],
        "mechanism_hit_at_3": qwen_full["metrics"]["mechanism"]["hit_at_3"],
        "exact_evidence_jaccard": qwen_full["metrics"]["exact"]["jaccard"],
        "empty_set_decision_accuracy": qwen_full["empty_set_accuracy"],
    }
    summary = {
        "paths": {
            "inputs": str(args.inputs),
            "predictions": str(args.predictions),
            "labels": str(args.labels),
            "evidence_store": str(args.evidence_store),
            "qwen_metrics": str(QWEN_METRICS),
            "output": str(args.out),
        },
        "sha256": {
            "inputs": sha256(args.inputs),
            "predictions": sha256(args.predictions),
            "labels": sha256(args.labels),
        },
        "input_confirmation": {
            "rows": len(inputs),
            "candidate_count_min": min(int(row.get("candidate_count") or 0) for row in inputs.values()),
            "candidate_count_max": max(int(row.get("candidate_count") or 0) for row in inputs.values()),
            "candidate_cards": sum(int(row.get("candidate_count") or 0) for row in inputs.values()),
            "qwen_full_metrics_source": "module_ablation.final_metrics.json variants.full",
            "qwen_full_expected_percent": [53.14, 76.15, 83.26, 44.18, 95.08],
        },
        "structure_check": {
            "rows": len(structure_rows),
            "rows_with_issues": sum(bool(row["issues"]) for row in structure_rows),
            "issue_counts": dict(sorted(Counter(issue for row in structure_rows for issue in row["issues"]).items())),
            "prediction_count_distribution": dict(
                sorted(Counter(len(row["ids"]) for row in structure_rows).items())
            ),
        },
        "evaluation_denominators": {
            "nonempty_reference_queries": len(nonempty_rows),
            "empty_reference_queries": len(empty_rows),
        },
        "metrics": {
            "automatic_top30_teacher_baseline_gpt55": teacher,
            "finetuned_qwen3_32b_same_automatic_top30_pool": qwen,
        },
        "comparison_percent": [
            {
                "metric": "Exact Evidence Top-1",
                "automatic_top30_teacher_baseline_gpt55": round(teacher["exact_evidence_top1"] * 100, 2),
                "finetuned_qwen3_32b_same_automatic_top30_pool": round(qwen["exact_evidence_top1"] * 100, 2),
            },
            {
                "metric": "Exact Evidence Hit@3",
                "automatic_top30_teacher_baseline_gpt55": round(teacher["exact_evidence_hit_at_3"] * 100, 2),
                "finetuned_qwen3_32b_same_automatic_top30_pool": round(qwen["exact_evidence_hit_at_3"] * 100, 2),
            },
            {
                "metric": "Mechanism Hit@3",
                "automatic_top30_teacher_baseline_gpt55": round(teacher["mechanism_hit_at_3"] * 100, 2),
                "finetuned_qwen3_32b_same_automatic_top30_pool": round(qwen["mechanism_hit_at_3"] * 100, 2),
            },
            {
                "metric": "Exact Evidence Jaccard",
                "automatic_top30_teacher_baseline_gpt55": round(teacher["exact_evidence_jaccard"] * 100, 2),
                "finetuned_qwen3_32b_same_automatic_top30_pool": round(qwen["exact_evidence_jaccard"] * 100, 2),
            },
            {
                "metric": "Empty-set Decision Accuracy",
                "automatic_top30_teacher_baseline_gpt55": round(teacher["empty_set_decision_accuracy"] * 100, 2),
                "finetuned_qwen3_32b_same_automatic_top30_pool": round(qwen["empty_set_decision_accuracy"] * 100, 2),
            },
        ],
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("input_confirmation", "structure_check", "evaluation_denominators", "comparison_percent")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
