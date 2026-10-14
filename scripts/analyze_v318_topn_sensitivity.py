#!/usr/bin/env python3
"""Evaluate frozen Top-10/20/30 candidate and ranker sensitivity on v3.18."""

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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def references(row: dict[str, Any]) -> list[list[str]]:
    ranked = sorted(
        row.get("topk") or (row.get("teacher_output") or {}).get("recommendations") or [],
        key=lambda item: int(item.get("rank") or 999),
    )
    return [[str(value) for value in item.get("evidence_ids") or []] for item in ranked]


def project(ids: list[str], evidence: dict[str, dict[str, Any]], fn: Callable[[dict[str, Any]], str] | None) -> list[str]:
    if fn is None:
        return ids
    return [value for eid in ids if eid in evidence and (value := fn(evidence[eid]))]


def jac(left: list[str], right: list[str]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def parse_specs(specs: list[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for spec in specs:
        raw_k, sep, raw_path = spec.partition("=")
        if not sep:
            raise ValueError(f"Invalid specification {spec!r}")
        result[int(raw_k)] = Path(raw_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--evidence-store", type=Path, required=True)
    parser.add_argument("--input", action="append", required=True, help="K=INPUT_JSONL")
    parser.add_argument("--prediction", action="append", default=[], help="K=PREDICTIONS_JSONL")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    labels = {str(row["qid"]): row for row in read_jsonl(args.labels)}
    evidence = {str(row["evidence_id"]): row for row in read_jsonl(args.evidence_store)}
    inputs = parse_specs(args.input)
    prediction_paths = parse_specs(args.prediction)
    levels: dict[str, Callable[[dict[str, Any]], str] | None] = {
        "exact": None,
        "treatment": countermeasure_text_key,
        "mechanism": mechanism_family,
        "broad_mechanism": broad_mechanism_family,
    }
    result: dict[str, Any] = {
        "test_queries": len(labels),
        "nonempty_queries": sum(bool(references(row)) for row in labels.values()),
        "empty_reference_queries": sum(not references(row) for row in labels.values()),
        "top_n": {},
    }

    for k, input_path in sorted(inputs.items()):
        rows = {str(row["qid"]): row for row in read_jsonl(input_path)}
        if set(rows) != set(labels):
            raise ValueError(f"Input QID mismatch for Top-{k}")
        group_hits = {level: 0 for level in levels}
        groups_total = 0
        for qid, label in labels.items():
            candidate_ids = [str(value) for value in rows[qid].get("candidate_evidence_ids") or []]
            for group in references(label):
                groups_total += 1
                for level, fn in levels.items():
                    group_hits[level] += bool(set(project(group, evidence, fn)) & set(project(candidate_ids, evidence, fn)))
        item: dict[str, Any] = {
            "candidate_count_mean": sum(len(row.get("candidate_evidence_ids") or []) for row in rows.values()) / len(rows),
            "reference_groups": groups_total,
            "candidate_group_recall": {level: group_hits[level] / groups_total for level in levels},
        }

        if k in prediction_paths:
            predictions = {str(row["qid"]): row for row in read_jsonl(prediction_paths[k])}
            if set(predictions) != set(labels):
                raise ValueError(f"Prediction QID mismatch for Top-{k}: {len(predictions)}/{len(labels)}")
            nonempty = [qid for qid, row in labels.items() if references(row)]
            empty = [qid for qid, row in labels.items() if not references(row)]
            output_metrics: dict[str, Any] = {}
            for level, fn in levels.items():
                top1 = hit = 0
                jaccard_sum = 0.0
                for qid in nonempty:
                    expected = project([group[0] for group in references(labels[qid]) if group], evidence, fn)
                    predicted = project(
                        [str(value) for value in predictions[qid].get("predicted_evidence_ids") or []], evidence, fn
                    )
                    top1 += bool(expected and predicted and expected[0] == predicted[0])
                    hit += bool(set(expected) & set(predicted))
                    jaccard_sum += jac(expected, predicted)
                output_metrics[level] = {
                    "top1": top1 / len(nonempty),
                    "hit_at_3": hit / len(nonempty),
                    "jaccard": jaccard_sum / len(nonempty),
                }
            item["ranker_nonempty_metrics"] = output_metrics
            item["empty_set_accuracy"] = sum(
                not predictions[qid].get("predicted_evidence_ids") for qid in empty
            ) / len(empty)
            item["json_valid_rate"] = sum(not row.get("json_error") for row in predictions.values()) / len(labels)
            item["all_ids_in_candidates_rate"] = sum(
                bool(row.get("all_ids_in_frozen_candidates")) for row in predictions.values()
            ) / len(labels)
            item["mean_input_tokens"] = sum(float(row.get("input_tokens") or 0) for row in predictions.values()) / len(labels)
            item["mean_latency_seconds"] = sum(float(row.get("latency_seconds") or 0) for row in predictions.values()) / len(labels)
        result["top_n"][str(k)] = item

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
