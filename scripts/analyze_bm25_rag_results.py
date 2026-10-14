#!/usr/bin/env python3
"""Produce subgroup and integrity analysis for the final BM25 RAG baseline."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from analyze_v318_final_results import evaluate_group, label_ids, read_jsonl, write_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = ROOT / "data" / "release" / "v3_18_final_split_2635_300"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--evidence-store", type=Path, default=ROOT / "data" / "evidence_store.jsonl")
    args = parser.parse_args()

    release = args.release_dir
    rag = release / "bm25_rag_baseline"
    inputs = {row["qid"]: row for row in read_jsonl(rag / "test_300.bm25_top30.rag_inputs.label_free.jsonl")}
    labels = {row["qid"]: row for row in read_jsonl(release / "reference_labels.test_300.evaluation_only.jsonl")}
    predictions_rows = {
        row["qid"]: row
        for row in read_jsonl(rag / "out" / "v318_final_test_300.bm25_rag.predictions.jsonl")
    }
    evidence = {row["evidence_id"]: row for row in read_jsonl(args.evidence_store)}
    if not (len(inputs) == len(labels) == len(predictions_rows) == 300):
        raise ValueError("Expected 300 unique rows in inputs, labels, and predictions")
    if not (set(inputs) == set(labels) == set(predictions_rows)):
        raise ValueError("Input, label, and prediction qid sets differ")

    references = {qid: label_ids(row) for qid, row in labels.items()}
    predictions = {
        qid: [str(evidence_id) for evidence_id in row.get("predicted_evidence_ids") or []]
        for qid, row in predictions_rows.items()
    }
    qids = sorted(labels)
    base_qids = [qid for qid in qids if qid.startswith("v3_query_")]
    incomplete_qids = [qid for qid in qids if qid.startswith("v3_incomplete_")]

    outside_rows: list[dict[str, Any]] = []
    for qid in qids:
        allowed = set(str(value) for value in inputs[qid]["retrieved_evidence_ids"])
        outside = [value for value in predictions[qid] if value not in allowed]
        if outside:
            outside_rows.append(
                {
                    "qid": qid,
                    "user_text": inputs[qid]["user_text"],
                    "outside_ids": outside,
                    "predicted_ids": predictions[qid],
                    "bm25_top30_ids": inputs[qid]["retrieved_evidence_ids"],
                }
            )

    output = {
        "protocol": {
            "knowledge_base": "8,844 complete CMF Evidence Cards",
            "retrieval": "BM25 Okapi Top-30 (k1=1.5, b=0.75)",
            "generator": "base Qwen3-32B without task fine-tuning",
            "proposed_retrieval_modules_used": False,
            "labels_available_during_retrieval_or_generation": False,
        },
        "groups": {
            "overall": evaluate_group(qids, references, predictions, evidence),
            "base": evaluate_group(base_qids, references, predictions, evidence),
            "incomplete": evaluate_group(incomplete_qids, references, predictions, evidence),
        },
        "format": {
            "n": len(qids),
            "json_valid_n": sum(not predictions_rows[qid].get("json_error") for qid in qids),
            "all_ids_in_bm25_top30_n": sum(
                bool(predictions_rows[qid].get("all_ids_in_frozen_candidates")) for qid in qids
            ),
            "duplicate_id_rows": sum(bool(predictions_rows[qid].get("duplicate_ids")) for qid in qids),
            "outside_candidate_rows": outside_rows,
            "prediction_count_distribution": dict(
                sorted(Counter(len(predictions[qid]) for qid in qids).items())
            ),
            "mean_input_tokens": sum(predictions_rows[qid]["input_tokens"] for qid in qids) / len(qids),
            "max_input_tokens": max(predictions_rows[qid]["input_tokens"] for qid in qids),
            "mean_latency_seconds": sum(predictions_rows[qid]["latency_seconds"] for qid in qids) / len(qids),
        },
    }
    out_path = rag / "v318_final_bm25_rag_analysis.json"
    write_json(out_path, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
