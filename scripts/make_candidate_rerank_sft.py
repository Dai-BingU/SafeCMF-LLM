#!/usr/bin/env python3
"""Build candidate-rerank SFT data from reviewed query/label files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmfrec.context_tags import infer_context_tags_from_user_text
from cmfrec.facility import infer_facility_type_from_user_text
from cmfrec.mechanism_recall import candidate_context_mismatch
from cmfrec.scoring import Query, score_row
from scripts.demo_end_to_end_compare import _build_score_row


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def to_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except Exception:
        return None


def first_eid(item: dict[str, Any]) -> str:
    ids = item.get("evidence_ids") or []
    if ids:
        return str(ids[0])
    return str(item.get("cm_id") or item.get("evidence_id") or "")


def expected_ids(label: dict[str, Any], top_k: int) -> list[str]:
    out: list[str] = []
    for item in (label.get("topk") or [])[:top_k]:
        eid = first_eid(item)
        if eid and eid not in out:
            out.append(eid)
    return out


def query_from_context(user_text: str, ctx: dict[str, Any]) -> Query:
    facility_type = ctx.get("facility_type")
    if not facility_type:
        facility_type = infer_facility_type_from_user_text(
            user_text, intersection_related=str(ctx.get("intersection_related") or "")
        )
    return Query(
        crash_type=ctx.get("crash_type"),
        severity=ctx.get("severity_kabco") or ctx.get("severity"),
        roadway_type=ctx.get("roadway_type"),
        area_type=ctx.get("area_type"),
        facility_type=facility_type,
        intersection_related=ctx.get("intersection_related"),
        traffic_control_type=ctx.get("traffic_control_type"),
        intersection_geometry=ctx.get("intersection_geometry") or ctx.get("intersection_type"),
        min_speed_limit=to_float(ctx.get("min_speed_limit")),
        max_speed_limit=to_float(ctx.get("max_speed_limit")),
        num_lanes=to_float(ctx.get("max_num_lanes") or ctx.get("min_num_lanes") or ctx.get("num_lanes")),
        traffic_volume_aadt=to_float(
            ctx.get("avg_traffic_volume_non_intersection")
            or ctx.get("max_traffic_volume_non_intersection")
            or ctx.get("min_traffic_volume_non_intersection")
        ),
        major_road_volume_aadt=to_float(
            ctx.get("avg_major_road_traffic_volume")
            or ctx.get("max_major_road_traffic_volume")
            or ctx.get("min_major_road_traffic_volume")
        ),
        minor_road_volume_aadt=to_float(
            ctx.get("avg_minor_road_traffic_volume")
            or ctx.get("max_minor_road_traffic_volume")
            or ctx.get("min_minor_road_traffic_volume")
        ),
        context_tags=infer_context_tags_from_user_text(user_text),
    )


def make_card(ev: dict[str, Any], score: float | None = None) -> dict[str, Any]:
    c = ev.get("conditions") or {}
    e = ev.get("effect") or {}
    q = ev.get("quality") or {}
    conditions = {
        "crash_type": c.get("crash_type"),
        "severity_kabco": c.get("severity_kabco"),
        "area_type": c.get("area_type"),
        "facility_type": c.get("facility_type"),
        "intersection_related": c.get("intersection_related"),
        "traffic_control_type": c.get("traffic_control_type"),
        "intersection_geometry": c.get("intersection_geometry"),
        "roadway_type": c.get("roadway_type"),
        "min_speed_limit": c.get("min_speed_limit"),
        "max_speed_limit": c.get("max_speed_limit"),
        "min_num_lanes": c.get("min_num_lanes"),
        "max_num_lanes": c.get("max_num_lanes"),
        "min_traffic_volume_non_intersection": c.get("min_traffic_volume_non_intersection"),
        "max_traffic_volume_non_intersection": c.get("max_traffic_volume_non_intersection"),
        "min_major_road_traffic_volume": c.get("min_major_road_traffic_volume"),
        "max_major_road_traffic_volume": c.get("max_major_road_traffic_volume"),
        "min_minor_road_traffic_volume": c.get("min_minor_road_traffic_volume"),
        "max_minor_road_traffic_volume": c.get("max_minor_road_traffic_volume"),
    }
    conditions = {k: v for k, v in conditions.items() if v not in (None, "", [], "Not specified")}
    effect = {k: v for k, v in {"cmf": e.get("cmf"), "crf": e.get("crf")}.items() if v not in (None, "")}
    card = {
        "evidence_id": str(ev.get("evidence_id")),
        "countermeasure": ev.get("countermeasure"),
    }
    if ev.get("countermeasure_category"):
        card["category"] = ev.get("countermeasure_category")
    if ev.get("countermeasure_subcategory"):
        card["subcategory"] = ev.get("countermeasure_subcategory")
    if conditions:
        card["conditions"] = conditions
    if effect:
        card["effect"] = effect
    if q.get("star_quality_rating") not in (None, ""):
        card["star"] = q.get("star_quality_rating")
    if score is not None:
        card["retrieval_score"] = round(float(score), 4)
    return card


def candidate_cards(
    evidences: list[dict[str, Any]],
    score_rows: list[dict[str, str]],
    query: Query,
    top_n: int,
    *,
    user_text: str = "",
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    scored: list[tuple[float, dict[str, Any]]] = []
    scores_by_id: dict[str, float] = {}
    for ev, row in zip(evidences, score_rows):
        if user_text and candidate_context_mismatch(ev, user_text, query):
            continue
        s = score_row(row, query)
        if s is None:
            continue
        score = float(s.total_score)
        eid = str(ev.get("evidence_id"))
        scores_by_id[eid] = score
        scored.append((score, ev))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [make_card(ev, score) for score, ev in scored[:top_n]], scores_by_id


def trim_context(ctx: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "crash_type",
        "severity_kabco",
        "area_type",
        "facility_type",
        "intersection_related",
        "traffic_control_type",
        "intersection_geometry",
        "intersection_type",
        "roadway_type",
        "roadway_division_type",
        "min_speed_limit",
        "max_speed_limit",
        "min_num_lanes",
        "max_num_lanes",
        "avg_traffic_volume_non_intersection",
        "min_traffic_volume_non_intersection",
        "max_traffic_volume_non_intersection",
        "avg_major_road_traffic_volume",
        "min_major_road_traffic_volume",
        "max_major_road_traffic_volume",
        "avg_minor_road_traffic_volume",
        "min_minor_road_traffic_volume",
        "max_minor_road_traffic_volume",
        "traffic_volume_unit",
    ]
    return {k: ctx.get(k) for k in keys if ctx.get(k) not in (None, "", [])}


def prompt(user_text: str, target_context: dict[str, Any], candidates: list[dict[str, Any]], top_k: int) -> str:
    payload = {
        "user_text": user_text,
        "target_context": target_context,
        "candidates": candidates,
        "top_k": top_k,
    }
    return (
        "You are a transportation safety engineer.\n"
        "Task: Choose the best ranked countermeasures from INPUT_JSON.candidates.\n"
        "Constraints:\n"
        "- Output valid JSON only.\n"
        "- You MUST choose evidence_ids only from INPUT_JSON.candidates.\n"
        "- Do not invent evidence_ids, CMF values, CRF values, star ratings, or citations.\n"
        "- Remove duplicate or near-duplicate countermeasures unless no clean alternative exists.\n"
        "- It is allowed to return fewer than top_k recommendations if remaining candidates do not fit.\n"
        "- Prefer mechanism and site-context fit over high CRF alone.\n"
        "- Output shape: {\"recommendations\":[{\"rank\":1,\"evidence_ids\":[\"...\"]}]}\n"
        "INPUT_JSON:\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + "\n"
    )


def completion(eids: list[str]) -> str:
    return json.dumps(
        {"recommendations": [{"rank": i + 1, "evidence_ids": [eid]} for i, eid in enumerate(eids)]},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=Path, nargs="+", required=True)
    ap.add_argument("--labels", type=Path, nargs="+", required=True)
    ap.add_argument("--evidence-store", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summary-out", type=Path, required=True)
    ap.add_argument("--candidate-top-n", type=int, default=40)
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    queries: dict[str, dict[str, Any]] = {}
    for path in args.queries:
        for row in read_jsonl(path):
            queries[str(row["qid"])] = row

    labels: dict[str, dict[str, Any]] = {}
    for path in args.labels:
        for row in read_jsonl(path):
            labels[str(row["qid"])] = row

    if set(queries) != set(labels):
        raise SystemExit(f"qid mismatch queries={len(queries)} labels={len(labels)}")

    evidences = read_jsonl(args.evidence_store)
    ev_by_id = {str(ev.get("evidence_id")): ev for ev in evidences}
    score_rows = [_build_score_row(ev) for ev in evidences]

    rows: list[dict[str, Any]] = []
    stats = {
        "records": 0,
        "candidate_top_n": int(args.candidate_top_n),
        "top_k": int(args.top_k),
        "all_expected_in_raw_candidates": 0,
        "top1_expected_in_raw_candidates": 0,
        "expected_forced_into_candidates": 0,
        "missing_expected_evidence": 0,
        "candidate_len_counts": {},
    }

    for qid in sorted(queries):
        qrow = queries[qid]
        lrow = labels[qid]
        user_text = str(qrow.get("user_text") or "")
        ctx = qrow.get("target_context") or {}
        exp = expected_ids(lrow, int(args.top_k))
        if not exp:
            continue
        if any(eid not in ev_by_id for eid in exp):
            stats["missing_expected_evidence"] += 1
            continue

        cards, score_by_id = candidate_cards(
            evidences,
            score_rows,
            query_from_context(user_text, ctx),
            int(args.candidate_top_n),
            user_text=user_text,
        )
        raw_ids = [str(c["evidence_id"]) for c in cards]
        raw_set = set(raw_ids)
        if exp[0] in raw_set:
            stats["top1_expected_in_raw_candidates"] += 1
        if all(eid in raw_set for eid in exp):
            stats["all_expected_in_raw_candidates"] += 1

        final_cards = list(cards)
        final_ids = set(raw_ids)
        forced = False
        for eid in exp:
            if eid in final_ids:
                continue
            final_cards.append(make_card(ev_by_id[eid], score_by_id.get(eid)))
            final_ids.add(eid)
            forced = True
        if forced:
            stats["expected_forced_into_candidates"] += 1

        # Keep candidate list compact but always target-complete.
        max_final = int(args.candidate_top_n) + int(args.top_k)
        final_cards = final_cards[:max_final]
        key = str(len(final_cards))
        stats["candidate_len_counts"][key] = stats["candidate_len_counts"].get(key, 0) + 1

        rows.append(
            {
                "id": qid,
                "prompt": prompt(user_text, trim_context(ctx), final_cards, int(args.top_k)),
                "completion": completion(exp),
                "meta": {
                    "expected_ids": exp,
                    "raw_candidate_head": raw_ids[:10],
                    "all_expected_in_raw_candidates": all(eid in raw_set for eid in exp),
                    "forced_expected_into_candidates": forced,
                },
            }
        )

    stats["records"] = len(rows)
    denom = max(1, len(rows))
    stats["top1_expected_raw_candidate_rate"] = stats["top1_expected_in_raw_candidates"] / denom
    stats["all_expected_raw_candidate_rate"] = stats["all_expected_in_raw_candidates"] / denom
    stats["forced_expected_rate"] = stats["expected_forced_into_candidates"] / denom

    write_jsonl(args.out, rows)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
