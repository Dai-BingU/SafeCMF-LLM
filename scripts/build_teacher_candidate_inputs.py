#!/usr/bin/env python3
"""Build blind teacher-labeling inputs from retrieved candidate IDs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cmfrec.free_text import infer_query_from_text  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def compact_card(ev: dict[str, Any], rank: int) -> dict[str, Any]:
    cond = ev.get("conditions") or {}
    effect = ev.get("effect") or {}
    quality = ev.get("quality") or {}
    citation = ev.get("citation") or {}
    raw = ev.get("raw") or {}
    condition_keys = [
        "crash_type",
        "severity_kabco",
        "roadway_type",
        "area_type",
        "intersection_related",
        "traffic_control_type",
        "intersection_type",
        "intersection_geometry",
        "roadway_division_type",
        "crash_time_of_day",
        "crash_weather",
        "min_speed_limit",
        "max_speed_limit",
        "speed_unit",
        "min_num_lanes",
        "max_num_lanes",
        "traffic_volume_unit",
        "min_traffic_volume_non_intersection",
        "max_traffic_volume_non_intersection",
        "min_major_road_traffic_volume",
        "max_major_road_traffic_volume",
        "min_minor_road_traffic_volume",
        "max_minor_road_traffic_volume",
    ]
    conditions = {key: cond.get(key) for key in condition_keys if cond.get(key) not in (None, "", [])}
    if raw.get("Prior Condition") not in (None, ""):
        conditions["prior_condition"] = raw.get("Prior Condition")
    return {
        "retrieval_rank": rank,
        "evidence_id": str(ev.get("evidence_id")),
        "countermeasure": ev.get("countermeasure"),
        "category": ev.get("countermeasure_category"),
        "subcategory": ev.get("countermeasure_subcategory"),
        "conditions": conditions,
        "effect": {key: effect.get(key) for key in ["cmf", "crf"] if effect.get(key) is not None},
        "quality": {
            key: quality.get(key)
            for key in ["star_quality_rating", "total_quality_points"]
            if quality.get(key) is not None
        },
        "citation": {
            key: citation.get(key)
            for key in ["study_title", "publication_year", "methodology", "state_province", "country"]
            if citation.get(key) not in (None, "")
        },
    }


def teacher_prompt(payload: dict[str, Any]) -> str:
    return (
        "You are a transportation safety engineer selecting evidence for a crash-context query.\n"
        "Return valid JSON only.\n"
        "Select zero to three ranked evidence IDs, using only INPUT_JSON.candidates.\n"
        "Treat retrieval rank as a clue, not as ground truth.\n"
        "Classify every selected item as direct or conditional; reject explicit context conflicts.\n"
        "Direct means that the query states the decisive applicability conditions. Conditional means that a critical condition is unknown, not contradicted; list every missing critical condition.\n"
        "Rank all direct matches before conditional matches, and never select conditional evidence merely to fill three slots.\n"
        "Reject evidence with facility, traffic-control, crash-type, geometry, or hidden prior-condition conflicts.\n"
        "Reject risk-increasing, exposure-only, or unsafe reverse-interpreted cards as positive recommendations.\n"
        "Do not select negative-quality or one-star evidence. Select two-star evidence only when no better direct same-mechanism alternative exists, and mark it low quality.\n"
        "Prefer direct context applicability and stronger evidence quality over a large CRF.\n"
        "Avoid duplicate or near-duplicate mechanisms unless they are meaningful alternative designs.\n"
        "It is acceptable to return fewer than three recommendations.\n"
        "For every selection, provide a short applicability reason and any prerequisite that must be verified.\n"
        "Output schema: {\"recommendations\":[{\"rank\":1,\"evidence_ids\":[\"...\"],"
        "\"applicability_status\":\"direct|conditional\",\"missing_critical_conditions\":[\"...\"],"
        "\"low_quality\":false,\"reason\":\"...\",\"prerequisites\":[\"...\"]}],"
        "\"rejected_top_candidates\":[{\"evidence_id\":\"...\","
        "\"reason\":\"...\"}]}\n"
        "INPUT_JSON:\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def visible_context_from_text(user_text: str) -> dict[str, Any]:
    """Return only deterministic fields recoverable from the actual query text."""
    query = infer_query_from_text(user_text).query
    fields = [
        "crash_type", "roadway_type", "area_type", "facility_type", "intersection_related",
        "traffic_control_type", "intersection_geometry", "min_speed_limit", "max_speed_limit",
        "num_lanes", "traffic_volume_aadt", "major_road_volume_aadt", "minor_road_volume_aadt",
    ]
    return {field: getattr(query, field) for field in fields if getattr(query, field) not in (None, "", (), [])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--evidence-store", type=Path, default=Path("data/evidence_store.jsonl"))
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument(
        "--query-only-context",
        action="store_true",
        help="Expose only context deterministically recoverable from user_text; required for robustness variants.",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.top_n not in {10, 20, 30}:
        raise SystemExit("--top-n must be one of 10, 20, or 30 under reconstruction-v2.1")

    queries = {str(row["qid"]): row for row in read_jsonl(args.queries)}
    candidates = {str(row["qid"]): row for row in read_jsonl(args.candidates)}
    evidence = {str(row["evidence_id"]): row for row in read_jsonl(args.evidence_store)}
    if set(queries) != set(candidates):
        raise SystemExit(f"query/candidate qid mismatch: {len(queries)} vs {len(candidates)}")

    rows: list[dict[str, Any]] = []
    for qid in queries:
        query = queries[qid]
        candidate_source = (
            candidates[qid].get("teacher_candidate_evidence_ids")
            or candidates[qid].get("candidate_evidence_ids")
            or []
        )
        candidate_ids = [str(x) for x in candidate_source][: args.top_n]
        missing = [evidence_id for evidence_id in candidate_ids if evidence_id not in evidence]
        if missing:
            raise SystemExit(f"{qid}: missing evidence IDs {missing[:10]}")
        cards = [compact_card(evidence[evidence_id], rank) for rank, evidence_id in enumerate(candidate_ids, start=1)]
        exposed_context = (
            visible_context_from_text(str(query.get("user_text") or ""))
            if args.query_only_context
            else (query.get("target_context") or {})
        )
        payload = {
            "qid": qid,
            "user_text": query.get("user_text"),
            "target_context": exposed_context,
            "top_k": 3,
            "candidates": cards,
        }
        rows.append(
            {
                "qid": qid,
                "variant_of_qid": query.get("variant_of_qid"),
                "scenario_group_id": query.get("scenario_group_id"),
                "query_completeness": query.get("query_completeness"),
                "mechanism_information_level": query.get("mechanism_information_level"),
                "missingness_type": query.get("missingness_type"),
                "missing_visible_fields": query.get("missing_visible_fields") or [],
                "split": query.get("split"),
                "target_distribution": query.get("target_distribution"),
                "audit_stratum": query.get("audit_stratum"),
                "generation_provenance": query.get("generation_provenance") or {},
                "teacher_input": payload,
                "prompt": teacher_prompt(payload),
                "teacher_output": None,
                "label_status": "awaiting_teacher",
                "blindness_audit": {
                    "source_evidence_id_exposed": False,
                    "cluster_supporting_evidence_ids_exposed": False,
                    "candidate_ids_only": True,
                },
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "count": len(rows), "top_n": args.top_n}, indent=2))


if __name__ == "__main__":
    main()
