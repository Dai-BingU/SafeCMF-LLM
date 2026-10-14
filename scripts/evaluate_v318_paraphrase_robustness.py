#!/usr/bin/env python3
"""Evaluate v3.18 paraphrase robustness after all label-free predictions are frozen."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmfrec.countermeasure_family import mechanism_family  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def index(rows: list[dict[str, Any]], key: str = "qid") -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key) or "")
        if not value or value in result:
            raise ValueError(f"Missing or duplicate {key}: {value!r}")
        result[value] = row
    return result


def candidate_ids(row: dict[str, Any]) -> list[str]:
    values = row.get("teacher_candidate_evidence_ids") or row.get("candidate_evidence_ids") or []
    return [str(value) for value in values][:30]


def predicted_ids(row: dict[str, Any]) -> list[str]:
    values = row.get("predicted_evidence_ids")
    if values is None:
        values = row.get("predicted_ids")
    return [str(value) for value in values or []][:3]


def reference_ids(row: dict[str, Any]) -> list[str]:
    ranked = sorted(
        row.get("topk") or (row.get("teacher_output") or {}).get("recommendations") or [],
        key=lambda item: int(item.get("rank") or 999),
    )
    return [str(item["evidence_ids"][0]) for item in ranked if item.get("evidence_ids")]


def jaccard(left: list[str], right: list[str]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def mechanism_set(ids: list[str], evidence: dict[str, dict[str, Any]]) -> set[str]:
    result: set[str] = set()
    for evidence_id in ids:
        row = evidence.get(evidence_id)
        if not row:
            continue
        value = mechanism_family(
            {
                "countermeasure": row.get("countermeasure"),
                "countermeasure_category": row.get("countermeasure_category") or row.get("category"),
                "countermeasure_subcategory": row.get("countermeasure_subcategory") or row.get("subcategory"),
            }
        )
        if value:
            result.add(value)
    return result


def performance_summary(
    rows: list[dict[str, Any]], evidence: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    nonempty = [row for row in rows if row["reference_ids"]]
    empty = [row for row in rows if not row["reference_ids"]]
    return {
        "nonempty_queries": len(nonempty),
        "empty_queries": len(empty),
        "exact_evidence_top1": (
            mean(
                bool(row["predicted_ids"])
                and row["reference_ids"][0] == row["predicted_ids"][0]
                for row in nonempty
            )
            if nonempty
            else None
        ),
        "exact_evidence_hit_at_3": (
            mean(bool(set(row["reference_ids"]) & set(row["predicted_ids"])) for row in nonempty)
            if nonempty
            else None
        ),
        "mechanism_hit_at_3": (
            mean(
                bool(
                    mechanism_set(row["reference_ids"], evidence)
                    & mechanism_set(row["predicted_ids"], evidence)
                )
                for row in nonempty
            )
            if nonempty
            else None
        ),
        "exact_evidence_jaccard": (
            mean(jaccard(row["reference_ids"], row["predicted_ids"]) for row in nonempty)
            if nonempty
            else None
        ),
        "empty_set_decision_accuracy": (
            mean(not row["predicted_ids"] for row in empty) if empty else None
        ),
    }


def summarize(
    rows: list[dict[str, Any]], evidence: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    nonempty = [row for row in rows if row["reference_ids"]]
    empty = [row for row in rows if not row["reference_ids"]]
    reference_total = sum(len(set(row["reference_ids"])) for row in rows)
    reference_retrieved = sum(
        len(set(row["reference_ids"]) & set(row["variant_candidates"])) for row in rows
    )
    return {
        "pairs": len(rows),
        "candidate_set_jaccard_mean": mean(row["candidate_jaccard"] for row in rows),
        "candidate_set_jaccard_median": median(row["candidate_jaccard"] for row in rows),
        "identical_candidate_sets": sum(row["candidate_jaccard"] == 1.0 for row in rows),
        "reference_evidence_recall_at_30": (
            reference_retrieved / reference_total if reference_total else None
        ),
        "recommendation_set_jaccard_mean": mean(row["recommendation_jaccard"] for row in rows),
        "recommendation_set_jaccard_median": median(row["recommendation_jaccard"] for row in rows),
        "identical_recommendation_sets": sum(row["recommendation_jaccard"] == 1.0 for row in rows),
        "nonempty_reference_pairs": len(nonempty),
        "exact_evidence_hit_at_3": performance_summary(
            [
                {
                    "reference_ids": row["reference_ids"],
                    "predicted_ids": row["variant_predictions"],
                }
                for row in rows
            ],
            evidence,
        )["exact_evidence_hit_at_3"],
        "empty_reference_pairs": len(empty),
        "empty_set_decision_accuracy": (
            mean(not row["variant_predictions"] for row in empty) if empty else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-queries", type=Path, required=True)
    parser.add_argument("--variant-queries", type=Path, required=True)
    parser.add_argument("--base-candidates", type=Path, required=True)
    parser.add_argument("--variant-candidates", type=Path, required=True)
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--variant-predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--evidence-store", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--details-out", type=Path, required=True)
    args = parser.parse_args()

    base_queries = index(read_jsonl(args.base_queries))
    variant_queries = index(read_jsonl(args.variant_queries))
    base_candidates = index(read_jsonl(args.base_candidates))
    variant_candidates = index(read_jsonl(args.variant_candidates))
    base_predictions_all = index(read_jsonl(args.base_predictions))
    variant_predictions = index(read_jsonl(args.variant_predictions))
    labels_all = index(read_jsonl(args.labels))
    evidence = index(read_jsonl(args.evidence_store), "evidence_id")

    if len(base_queries) != 30 or len(variant_queries) != 150:
        raise ValueError(f"Expected 30 bases and 150 variants: {len(base_queries)}/{len(variant_queries)}")
    if set(base_candidates) != set(base_queries):
        raise ValueError("Base query/candidate QID mismatch")
    if set(variant_candidates) != set(variant_queries):
        raise ValueError("Variant query/candidate QID mismatch")
    if set(variant_predictions) != set(variant_queries):
        raise ValueError("Variant prediction/input QID mismatch")
    if not set(base_queries) <= set(base_predictions_all) or not set(base_queries) <= set(labels_all):
        raise ValueError("Base predictions or labels do not cover the 30 selected QIDs")

    issues: list[str] = []
    for qid, row in variant_predictions.items():
        ids = predicted_ids(row)
        allowed = set(candidate_ids(variant_candidates[qid]))
        if row.get("json_error"):
            issues.append(f"{qid}:json_error")
        if len(ids) != len(set(ids)):
            issues.append(f"{qid}:duplicate_ids")
        if any(value not in allowed for value in ids):
            issues.append(f"{qid}:outside_candidate_pool")
    if issues:
        raise ValueError(f"Prediction structure issues ({len(issues)}): {issues[:10]}")

    details: list[dict[str, Any]] = []
    for qid in sorted(variant_queries):
        variant = variant_queries[qid]
        base_qid = str(variant.get("variant_of_qid") or "")
        if base_qid not in base_queries:
            raise ValueError(f"Unknown base QID for {qid}: {base_qid}")
        base_pool = candidate_ids(base_candidates[base_qid])
        variant_pool = candidate_ids(variant_candidates[qid])
        base_pred = predicted_ids(base_predictions_all[base_qid])
        variant_pred = predicted_ids(variant_predictions[qid])
        refs = reference_ids(labels_all[base_qid])
        details.append(
            {
                "qid": qid,
                "base_qid": base_qid,
                "paraphrase_style": variant.get("paraphrase_style"),
                "reference_ids": refs,
                "base_candidates": base_pool,
                "variant_candidates": variant_pool,
                "base_predictions": base_pred,
                "variant_predictions": variant_pred,
                "candidate_jaccard": jaccard(base_pool, variant_pool),
                "recommendation_jaccard": jaccard(base_pred, variant_pred),
            }
        )

    base_nonempty = []
    base_empty = []
    base_performance_rows: list[dict[str, Any]] = []
    base_reference_total = 0
    base_reference_retrieved = 0
    for qid in sorted(base_queries):
        refs = reference_ids(labels_all[qid])
        pred = predicted_ids(base_predictions_all[qid])
        pool = candidate_ids(base_candidates[qid])
        base_performance_rows.append({"reference_ids": refs, "predicted_ids": pred})
        base_reference_total += len(set(refs))
        base_reference_retrieved += len(set(refs) & set(pool))
        (base_nonempty if refs else base_empty).append((refs, pred))

    by_style: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        by_style[str(row["paraphrase_style"])].append(row)

    overall = summarize(details, evidence)
    base_hit = (
        mean(bool(set(refs) & set(pred)) for refs, pred in base_nonempty) if base_nonempty else None
    )
    base_empty_acc = mean(not pred for _, pred in base_empty) if base_empty else None
    overall["original_exact_evidence_hit_at_3"] = base_hit
    overall["exact_evidence_hit_at_3_change_percentage_points"] = (
        100 * (overall["exact_evidence_hit_at_3"] - base_hit)
        if base_hit is not None and overall["exact_evidence_hit_at_3"] is not None
        else None
    )
    overall["original_empty_set_decision_accuracy"] = base_empty_acc
    original_recall = (
        base_reference_retrieved / base_reference_total if base_reference_total else None
    )
    overall["original_reference_evidence_recall_at_30"] = original_recall
    overall["reference_evidence_recall_at_30_change_percentage_points"] = (
        100 * (overall["reference_evidence_recall_at_30"] - original_recall)
        if original_recall is not None
        else None
    )

    paraphrase_performance_rows = [
        {
            "reference_ids": row["reference_ids"],
            "predicted_ids": row["variant_predictions"],
        }
        for row in details
    ]

    result = {
        "design": {
            "base_queries": len(base_queries),
            "variants_per_query": 5,
            "paraphrase_variants": len(variant_queries),
            "candidate_pool_size": 30,
            "labels_used_during_generation_or_inference": False,
        },
        "prediction_structure": {"rows": len(variant_predictions), "issues": 0},
        "main_performance_metrics": {
            "original_queries": performance_summary(base_performance_rows, evidence),
            "paraphrased_queries": performance_summary(paraphrase_performance_rows, evidence),
        },
        "overall": overall,
        "by_style": {name: summarize(rows, evidence) for name, rows in sorted(by_style.items())},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.details_out.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in details),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
