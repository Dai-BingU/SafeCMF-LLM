#!/usr/bin/env python3
"""Classify representative failures in the final v3.18 test evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmfrec.countermeasure_family import countermeasure_text_key, mechanism_family  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def reference_ids(row: dict[str, Any]) -> list[str]:
    recommendations = row.get("topk") or (row.get("teacher_output") or {}).get("recommendations") or []
    ranked = sorted(recommendations, key=lambda item: int(item.get("rank") or 999))
    return [str(item["evidence_ids"][0]) for item in ranked if item.get("evidence_ids")]


def detail(evidence_id: str, evidence: dict[str, dict[str, Any]]) -> dict[str, Any]:
    card = evidence.get(evidence_id, {})
    return {
        "evidence_id": evidence_id,
        "countermeasure": card.get("countermeasure", ""),
        "treatment": countermeasure_text_key(card) if card else "",
        "mechanism": mechanism_family(card) if card else "",
        "conditions": card.get("conditions", {}),
        "quality": card.get("quality", {}).get("star_quality_rating"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--evidence-store", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    queries = {str(row["qid"]): row for row in read_jsonl(args.queries)}
    labels = {str(row["qid"]): row for row in read_jsonl(args.labels)}
    candidates = {str(row["qid"]): row for row in read_jsonl(args.candidates)}
    predictions = {str(row["qid"]): row for row in read_jsonl(args.predictions)}
    evidence = {str(row["evidence_id"]): row for row in read_jsonl(args.evidence_store)}

    categories: dict[str, list[dict[str, Any]]] = {
        "top200_retrieval_miss": [],
        "top30_truncation": [],
        "reranking_failure_all_references_in_top30": [],
        "nonempty_exact_failure": [],
        "exact_failure_but_treatment_overlap": [],
        "exact_failure_but_mechanism_overlap": [],
        "empty_reference_overrecommendation": [],
    }

    for qid, label in labels.items():
        refs = reference_ids(label)
        candidate_row = candidates[qid]
        master = [str(value) for value in candidate_row.get("candidate_evidence_ids") or []]
        top30 = [str(value) for value in candidate_row.get("teacher_candidate_evidence_ids") or []][:30]
        preds = [str(value) for value in predictions[qid].get("predicted_evidence_ids") or []]
        master_rank = {value: index + 1 for index, value in enumerate(master)}
        top30_rank = {value: index + 1 for index, value in enumerate(top30)}

        ref_treatments = {countermeasure_text_key(evidence[value]) for value in refs if value in evidence}
        pred_treatments = {countermeasure_text_key(evidence[value]) for value in preds if value in evidence}
        ref_mechanisms = {mechanism_family(evidence[value]) for value in refs if value in evidence}
        pred_mechanisms = {mechanism_family(evidence[value]) for value in preds if value in evidence}

        row = {
            "qid": qid,
            "query": queries[qid].get("user_text", ""),
            "query_completeness": queries[qid].get("query_completeness", ""),
            "references": [
                {
                    **detail(value, evidence),
                    "top200_rank": master_rank.get(value),
                    "top30_rank": top30_rank.get(value),
                }
                for value in refs
            ],
            "predictions": [detail(value, evidence) for value in preds],
        }

        if refs and any(value not in master_rank for value in refs):
            categories["top200_retrieval_miss"].append(row)
        if refs and any(value in master_rank and value not in top30_rank for value in refs):
            categories["top30_truncation"].append(row)
        if refs and all(value in top30_rank for value in refs) and not (set(refs) & set(preds)):
            categories["reranking_failure_all_references_in_top30"].append(row)
        if refs and not (set(refs) & set(preds)):
            categories["nonempty_exact_failure"].append(row)
            if ref_treatments & pred_treatments:
                categories["exact_failure_but_treatment_overlap"].append(row)
            if ref_mechanisms & pred_mechanisms:
                categories["exact_failure_but_mechanism_overlap"].append(row)
        if not refs and preds:
            categories["empty_reference_overrecommendation"].append(row)

    result = {
        "counts": {name: len(rows) for name, rows in categories.items()},
        "categories": categories,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
