#!/usr/bin/env python3
"""Analyze final v3.18 predictions, retrieval baseline, and label revisions."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def label_ids(row: dict[str, Any]) -> list[str]:
    recommendations = row.get("topk") or (row.get("teacher_output") or {}).get("recommendations") or []
    ranked = sorted(recommendations, key=lambda item: int(item.get("rank") or 999))
    return [str(item["evidence_ids"][0]) for item in ranked if item.get("evidence_ids")]


def project(
    ids: list[str], evidence: dict[str, dict[str, Any]], fn: Callable[[dict[str, Any]], str]
) -> list[str]:
    return [value for evidence_id in ids if evidence_id in evidence and (value := fn(evidence[evidence_id]))]


def jaccard(left: list[str], right: list[str]) -> float:
    left_set, right_set = set(left), set(right)
    if not left_set and not right_set:
        return 1.0
    return len(left_set & right_set) / len(left_set | right_set)


def safe_rate(numerator: int | float, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def evaluate_group(
    qids: list[str],
    references: dict[str, list[str]],
    predictions: dict[str, list[str]],
    evidence: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    levels: dict[str, Callable[[list[str]], list[str]]] = {
        "exact": lambda ids: ids,
        "treatment": lambda ids: project(ids, evidence, countermeasure_text_key),
        "mechanism": lambda ids: project(ids, evidence, mechanism_family),
        "broad_mechanism": lambda ids: project(ids, evidence, broad_mechanism_family),
    }
    nonempty = [qid for qid in qids if references[qid]]
    empty = [qid for qid in qids if not references[qid]]
    summary: dict[str, Any] = {
        "n": len(qids),
        "reference_nonempty_n": len(nonempty),
        "reference_empty_n": len(empty),
        "reference_count_distribution": dict(sorted(Counter(len(references[qid]) for qid in qids).items())),
        "prediction_count_distribution": dict(sorted(Counter(len(predictions[qid]) for qid in qids).items())),
        "exact_set_agreement": safe_rate(
            sum(set(references[qid]) == set(predictions[qid]) for qid in qids), len(qids)
        ),
        "empty_decision_accuracy": safe_rate(sum(not predictions[qid] for qid in empty), len(empty)),
    }
    for level_name, transform in levels.items():
        rows: dict[str, tuple[list[str], list[str]]] = {
            qid: (transform(references[qid]), transform(predictions[qid])) for qid in qids
        }

        def aggregate(subset: list[str]) -> dict[str, float | None]:
            return {
                "top1": safe_rate(
                    sum(bool(rows[qid][0] and rows[qid][1] and rows[qid][0][0] == rows[qid][1][0]) for qid in subset),
                    len(subset),
                ),
                "hit_at_3": safe_rate(
                    sum(bool(set(rows[qid][0]) & set(rows[qid][1])) for qid in subset), len(subset)
                ),
                "jaccard": safe_rate(sum(jaccard(*rows[qid]) for qid in subset), len(subset)),
            }

        summary[level_name] = {"all": aggregate(qids), "reference_nonempty": aggregate(nonempty)}
    return summary


def classify_failures(
    qids: list[str], references: dict[str, list[str]], predictions: dict[str, list[str]]
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for qid in qids:
        expected, predicted = references[qid], predictions[qid]
        expected_set, predicted_set = set(expected), set(predicted)
        if not expected and predicted:
            failure_type = "overrecommendation_on_empty_reference"
        elif expected and not predicted:
            failure_type = "empty_prediction_on_nonempty_reference"
        elif expected and predicted and not (expected_set & predicted_set):
            failure_type = "no_exact_hit"
        elif expected and predicted and expected[0] != predicted[0]:
            failure_type = "top1_order_mismatch"
        elif expected_set != predicted_set:
            failure_type = "partial_set_mismatch"
        else:
            continue
        counts[failure_type] += 1
        rows.append(
            {
                "qid": qid,
                "failure_type": failure_type,
                "expected_ids": expected,
                "predicted_ids": predicted,
            }
        )
    return dict(sorted(counts.items())), rows


def revision_summary(initial: dict[str, list[str]], final: dict[str, list[str]], qids: list[str]) -> dict[str, Any]:
    ordered_changed = [qid for qid in qids if initial[qid] != final[qid]]
    set_changed = [qid for qid in qids if set(initial[qid]) != set(final[qid])]
    reorder_only = [
        qid for qid in qids if initial[qid] != final[qid] and set(initial[qid]) == set(final[qid])
    ]
    count_changed = [qid for qid in qids if len(initial[qid]) != len(final[qid])]
    added = sum(len(set(final[qid]) - set(initial[qid])) for qid in qids)
    removed = sum(len(set(initial[qid]) - set(final[qid])) for qid in qids)
    return {
        "n": len(qids),
        "ordered_output_revised_n": len(ordered_changed),
        "ordered_output_revision_rate": safe_rate(len(ordered_changed), len(qids)),
        "evidence_set_revised_n": len(set_changed),
        "evidence_set_revision_rate": safe_rate(len(set_changed), len(qids)),
        "reorder_only_n": len(reorder_only),
        "recommendation_count_changed_n": len(count_changed),
        "evidence_ids_added": added,
        "evidence_ids_removed": removed,
    }


def frequency_summary(values: list[str]) -> dict[str, Any]:
    counts = Counter(values)
    total = sum(counts.values())
    ordered = counts.most_common()
    return {
        "total_assignments": total,
        "unique_values": len(counts),
        "top1_share": safe_rate(sum(count for _, count in ordered[:1]), total),
        "top10_share": safe_rate(sum(count for _, count in ordered[:10]), total),
        "top20_share": safe_rate(sum(count for _, count in ordered[:20]), total),
        "hhi": sum((count / total) ** 2 for count in counts.values()) if total else None,
        "top10": [{"value": value, "count": count, "share": count / total} for value, count in ordered[:10]],
    }


def jensen_shannon(left: list[str], right: list[str]) -> float | None:
    left_counts, right_counts = Counter(left), Counter(right)
    left_total, right_total = sum(left_counts.values()), sum(right_counts.values())
    if not left_total or not right_total:
        return None
    keys = set(left_counts) | set(right_counts)
    divergence = 0.0
    for key in keys:
        p = left_counts[key] / left_total
        q = right_counts[key] / right_total
        midpoint = (p + q) / 2
        if p:
            divergence += 0.5 * p * math.log2(p / midpoint)
        if q:
            divergence += 0.5 * q * math.log2(q / midpoint)
    return divergence


def distribution_analysis(
    train_references: dict[str, list[str]],
    test_references: dict[str, list[str]],
    test_predictions: dict[str, list[str]],
    evidence: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    levels: dict[str, Callable[[list[str]], list[str]]] = {
        "exact": lambda ids: ids,
        "treatment": lambda ids: project(ids, evidence, countermeasure_text_key),
        "mechanism": lambda ids: project(ids, evidence, mechanism_family),
        "broad_mechanism": lambda ids: project(ids, evidence, broad_mechanism_family),
    }
    result: dict[str, Any] = {}
    for name, transform in levels.items():
        train_values = [value for ids in train_references.values() for value in transform(ids)]
        test_values = [value for ids in test_references.values() for value in transform(ids)]
        prediction_values = [value for ids in test_predictions.values() for value in transform(ids)]
        result[name] = {
            "training_reference": frequency_summary(train_values),
            "test_reference": frequency_summary(test_values),
            "model_prediction": frequency_summary(prediction_values),
            "js_divergence_train_vs_test_reference": jensen_shannon(train_values, test_values),
            "js_divergence_test_reference_vs_prediction": jensen_shannon(test_values, prediction_values),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release-dir", type=Path, default=ROOT / "data/release/v3_18_final_split_2635_300"
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Prediction JSONL to analyze. Defaults to v318_final_test_300.predictions.jsonl in the release directory.",
    )
    parser.add_argument("--evidence-store", type=Path, default=ROOT / "data/evidence_store.jsonl")
    parser.add_argument(
        "--initial-base",
        type=Path,
        default=ROOT / "data/reconstructed_v3/teacher_labels_1935.v3_9.query_only.initial.frozen.jsonl",
    )
    parser.add_argument(
        "--initial-incomplete",
        type=Path,
        default=ROOT / "data/reconstructed_v3/teacher_labels_1000.v3_17.initial.jsonl",
    )
    parser.add_argument(
        "--final-labels",
        type=Path,
        default=ROOT / "data/reconstructed_v3/train_dev_teacher_labels_2935.v3_18.open_audit_revised.final.jsonl",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    release = args.release_dir
    output_path = args.out or release / "v318_final_cpu_analysis.json"
    queries = {row["qid"]: row for row in read_jsonl(release / "queries.test_300.jsonl")}
    label_rows = {row["qid"]: row for row in read_jsonl(release / "reference_labels.test_300.evaluation_only.jsonl")}
    prediction_path = args.predictions or release / "v318_final_test_300.predictions.jsonl"
    prediction_rows = {row["qid"]: row for row in read_jsonl(prediction_path)}
    candidates = {
        row["qid"]: row for row in read_jsonl(release / "candidates.test_300.top200_master.jsonl")
    }
    evidence = {str(row["evidence_id"]): row for row in read_jsonl(args.evidence_store)}
    qids = sorted(queries)
    if set(qids) != set(label_rows) or set(qids) != set(prediction_rows) or set(qids) != set(candidates):
        raise ValueError("Test Query, label, prediction, and candidate QID sets do not match")

    references = {qid: label_ids(label_rows[qid]) for qid in qids}
    predictions = {
        qid: [str(value) for value in prediction_rows[qid].get("predicted_evidence_ids") or []] for qid in qids
    }
    retrieval_predictions = {
        qid: [str(value) for value in candidates[qid].get("teacher_candidate_evidence_ids") or []][:3]
        for qid in qids
    }
    base_qids = [qid for qid in qids if queries[qid].get("query_completeness") != "controlled_incomplete"]
    incomplete_qids = [qid for qid in qids if queries[qid].get("query_completeness") == "controlled_incomplete"]
    missingness_groups = {
        name: [qid for qid in incomplete_qids if queries[qid].get("missingness_type") == name]
        for name in sorted({str(queries[qid].get("missingness_type")) for qid in incomplete_qids})
    }

    final_groups = {
        "overall": evaluate_group(qids, references, predictions, evidence),
        "base": evaluate_group(base_qids, references, predictions, evidence),
        "incomplete": evaluate_group(incomplete_qids, references, predictions, evidence),
        "incomplete_by_missingness": {
            name: evaluate_group(group_qids, references, predictions, evidence)
            for name, group_qids in missingness_groups.items()
        },
    }
    retrieval_groups = {
        "overall": evaluate_group(qids, references, retrieval_predictions, evidence),
        "base": evaluate_group(base_qids, references, retrieval_predictions, evidence),
        "incomplete": evaluate_group(incomplete_qids, references, retrieval_predictions, evidence),
    }
    failure_counts, failure_rows = classify_failures(qids, references, predictions)

    initial_rows = read_jsonl(args.initial_base) + read_jsonl(args.initial_incomplete)
    initial = {row["qid"]: label_ids(row) for row in initial_rows}
    final_all = {row["qid"]: label_ids(row) for row in read_jsonl(args.final_labels)}
    all_revision_qids = sorted(set(initial) & set(final_all))
    if len(all_revision_qids) != 2935:
        raise ValueError(f"Expected 2,935 comparable initial/final labels, found {len(all_revision_qids)}")
    revision = {
        "overall": revision_summary(initial, final_all, all_revision_qids),
        "base": revision_summary(initial, final_all, [qid for qid in all_revision_qids if qid.startswith("v3_query_")]),
        "incomplete": revision_summary(
            initial, final_all, [qid for qid in all_revision_qids if qid.startswith("v3_incomplete_")]
        ),
    }
    initial_teacher_predictions = {qid: initial[qid] for qid in qids}
    final_candidate_sets = {
        qid: {str(value) for value in candidates[qid].get("teacher_candidate_evidence_ids") or []}
        for qid in qids
    }
    teacher_assignment_n = sum(len(initial_teacher_predictions[qid]) for qid in qids)
    teacher_outside_ids = {
        qid: [
            evidence_id
            for evidence_id in initial_teacher_predictions[qid]
            if evidence_id not in final_candidate_sets[qid]
        ]
        for qid in qids
    }
    teacher_outside_ids = {qid: values for qid, values in teacher_outside_ids.items() if values}
    initial_teacher_baseline = {
        "overall": evaluate_group(qids, references, initial_teacher_predictions, evidence),
        "base": evaluate_group(base_qids, references, initial_teacher_predictions, evidence),
        "incomplete": evaluate_group(incomplete_qids, references, initial_teacher_predictions, evidence),
        "final_top30_compliance": {
            "rows_all_ids_in_final_top30": len(qids) - len(teacher_outside_ids),
            "row_compliance_rate": safe_rate(len(qids) - len(teacher_outside_ids), len(qids)),
            "assignments_all_ids_in_final_top30": teacher_assignment_n
            - sum(len(values) for values in teacher_outside_ids.values()),
            "assignment_compliance_rate": safe_rate(
                teacher_assignment_n - sum(len(values) for values in teacher_outside_ids.values()),
                teacher_assignment_n,
            ),
            "rows_with_ids_outside_final_top30": teacher_outside_ids,
        },
        "interpretation": (
            "Preserved initial teacher outputs predate the final train/test split and author adjudication. "
            "They used the same at-most-Top-30 Evidence-ID output policy, but subsequent candidate-pool "
            "corrections mean that a small number of selected IDs are not in the final Top-30 pools."
        ),
    }
    train_label_rows = {
        row["qid"]: row for row in read_jsonl(release / "reference_labels.train_2635.jsonl")
    }
    train_references = {qid: label_ids(row) for qid, row in train_label_rows.items()}
    distributions = distribution_analysis(train_references, references, predictions, evidence)

    result = {
        "protocol": {
            "test_n": len(qids),
            "base_n": len(base_qids),
            "incomplete_n": len(incomplete_qids),
            "prediction_file": str(prediction_path),
            "retrieval_baseline": "first three IDs from the frozen ordered Top-30 retrieval pool",
        },
        "finetuned_model": final_groups,
        "retrieval_top3_baseline": retrieval_groups,
        "finetuned_failure_counts": failure_counts,
        "finetuned_failure_rows": failure_rows,
        "teacher_to_author_revision": revision,
        "preserved_initial_teacher_test_baseline": initial_teacher_baseline,
        "label_and_output_distributions": distributions,
    }
    write_json(output_path, result)
    print(json.dumps({key: value for key, value in result.items() if key != "finetuned_failure_rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
