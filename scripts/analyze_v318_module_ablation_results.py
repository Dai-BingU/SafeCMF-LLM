#!/usr/bin/env python3
"""Combine candidate and ranker metrics for the v3.18 module ablation table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmfrec.countermeasure_family import (  # noqa: E402
    broad_mechanism_family,
    countermeasure_text_key,
    mechanism_family,
)
from cmfrec.free_text import infer_query_from_text  # noqa: E402
from cmfrec.mechanism_recall import candidate_context_mismatch, hidden_precondition_mismatch  # noqa: E402


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
    return len(a & b) / len(a | b)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--evidence-store", type=Path, required=True)
    parser.add_argument("--candidate-metrics", type=Path, required=True)
    parser.add_argument("--prediction", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    labels = {str(row["qid"]): row for row in read_jsonl(args.labels)}
    queries = {str(row["qid"]): row for row in read_jsonl(args.queries)}
    evidence = {str(row["evidence_id"]): row for row in read_jsonl(args.evidence_store)}
    candidate_metrics = json.loads(args.candidate_metrics.read_text(encoding="utf-8"))
    levels = {
        "exact": lambda ids: ids,
        "treatment": lambda ids: project(ids, evidence, countermeasure_text_key),
        "mechanism": lambda ids: project(ids, evidence, mechanism_family),
        "broad_mechanism": lambda ids: project(ids, evidence, broad_mechanism_family),
    }
    result: dict[str, Any] = {"test_queries": len(labels), "variants": {}}

    for spec in args.prediction:
        name, sep, raw_path = spec.partition("=")
        if not sep:
            raise ValueError(f"Invalid --prediction {spec!r}")
        predictions = {str(row["qid"]): row for row in read_jsonl(Path(raw_path))}
        if set(predictions) != set(labels):
            raise ValueError(f"Prediction QID mismatch for {name}: {len(predictions)}/{len(labels)}")

        nonempty = [qid for qid, row in labels.items() if reference_ids(row)]
        empty = [qid for qid, row in labels.items() if not reference_ids(row)]
        metrics: dict[str, Any] = {}
        for level, transform in levels.items():
            top1 = hits = 0
            jac = 0.0
            for qid in nonempty:
                expected = transform(reference_ids(labels[qid]))
                predicted = transform([str(value) for value in predictions[qid].get("predicted_evidence_ids") or []])
                top1 += bool(expected and predicted and expected[0] == predicted[0])
                hits += bool(set(expected) & set(predicted))
                jac += jaccard(expected, predicted)
            metrics[level] = {
                "top1": top1 / len(nonempty),
                "hit_at_3": hits / len(nonempty),
                "jaccard": jac / len(nonempty),
            }

        selected_conflicts = 0
        selected_total = 0
        conflict_queries = 0
        for qid, prediction in predictions.items():
            user_text = str(queries[qid].get("user_text") or "")
            query = infer_query_from_text(user_text).query
            row_conflicts = 0
            for evidence_id in [str(value) for value in prediction.get("predicted_evidence_ids") or []]:
                if evidence_id not in evidence:
                    continue
                card = evidence[evidence_id]
                row_conflicts += bool(
                    hidden_precondition_mismatch(card, user_text)
                    or candidate_context_mismatch(card, user_text, query)
                )
            selected_conflicts += row_conflicts
            selected_total += len(prediction.get("predicted_evidence_ids") or [])
            conflict_queries += row_conflicts > 0

        result["variants"][name] = {
            "candidate": candidate_metrics["variants"][name],
            "ranker_nonempty_queries": len(nonempty),
            "metrics": metrics,
            "empty_set_accuracy": sum(not predictions[qid].get("predicted_evidence_ids") for qid in empty) / len(empty),
            "selected_engineering_conflict_rate": selected_conflicts / selected_total if selected_total else 0.0,
            "queries_with_selected_engineering_conflict_rate": conflict_queries / len(labels),
            "json_valid_rate": sum(not row.get("json_error") for row in predictions.values()) / len(labels),
            "all_ids_in_candidates_rate": sum(
                bool(row.get("all_ids_in_frozen_candidates")) for row in predictions.values()
            ) / len(labels),
            "duplicate_rows": sum(bool(row.get("duplicate_ids")) for row in predictions.values()),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
