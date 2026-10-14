#!/usr/bin/env python3
"""Evaluate completed locked-test predictions after label-free inference."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmfrec.countermeasure_family import broad_mechanism_family, countermeasure_text_key, mechanism_family


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def reference_ids(row: dict[str, Any]) -> list[str]:
    ranked = sorted(
        row.get("topk") or (row.get("teacher_output") or {}).get("recommendations") or [],
        key=lambda item: int(item.get("rank") or 999),
    )
    return [str(item["evidence_ids"][0]) for item in ranked if item.get("evidence_ids")]


def project(ids: list[str], evidence: dict[str, dict[str, Any]], fn: Callable[[dict[str, Any]], str]) -> list[str]:
    return [value for eid in ids if eid in evidence and (value := fn(evidence[eid]))]


def jaccard(left: list[str], right: list[str]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if a or b else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--evidence-store", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    predictions = {str(row["qid"]): row for row in read_jsonl(args.predictions)}
    labels = {str(row["qid"]): row for row in read_jsonl(args.labels)}
    evidence = {str(row["evidence_id"]): row for row in read_jsonl(args.evidence_store)}
    if set(predictions) != set(labels):
        raise ValueError(f"Prediction/label mismatch: predictions={len(predictions)}, labels={len(labels)}")

    levels = {
        "exact": lambda ids: ids,
        "treatment": lambda ids: project(ids, evidence, countermeasure_text_key),
        "mechanism": lambda ids: project(ids, evidence, mechanism_family),
        "broad_mechanism": lambda ids: project(ids, evidence, broad_mechanism_family),
    }
    rows: list[dict[str, Any]] = []
    for qid in sorted(labels):
        expected = reference_ids(labels[qid])
        predicted = [str(x) for x in predictions[qid].get("predicted_evidence_ids") or []]
        metrics: dict[str, Any] = {}
        for name, transform in levels.items():
            exp, pred = transform(expected), transform(predicted)
            metrics[name] = {
                "top1": bool(exp and pred and exp[0] == pred[0]),
                "hit_at_3": bool(set(exp) & set(pred)),
                "jaccard": jaccard(exp, pred),
                "expected": exp,
                "predicted": pred,
            }
        rows.append({"qid": qid, "expected_ids": expected, "predicted_ids": predicted, "metrics": metrics})

    summary: dict[str, Any] = {"n": len(rows)}
    for name in levels:
        summary[name] = {
            "top1": sum(row["metrics"][name]["top1"] for row in rows) / len(rows),
            "hit_at_3": sum(row["metrics"][name]["hit_at_3"] for row in rows) / len(rows),
            "jaccard": sum(row["metrics"][name]["jaccard"] for row in rows) / len(rows),
        }
    summary["format"] = {
        "json_valid": sum(not predictions[qid].get("json_error") for qid in labels) / len(labels),
        "all_ids_in_candidates": sum(bool(predictions[qid].get("all_ids_in_frozen_candidates")) for qid in labels) / len(labels),
        "rows_with_duplicate_ids": sum(bool(predictions[qid].get("duplicate_ids")) for qid in labels),
        "prediction_count_distribution": dict(sorted(Counter(int(predictions[qid].get("prediction_count") or 0) for qid in labels).items())),
    }
    summary["latency_seconds"] = {
        "mean": sum(float(predictions[qid].get("latency_seconds") or 0.0) for qid in labels) / len(labels),
        "total": sum(float(predictions[qid].get("latency_seconds") or 0.0) for qid in labels),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
