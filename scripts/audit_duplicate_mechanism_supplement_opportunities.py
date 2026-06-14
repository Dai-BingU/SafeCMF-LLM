#!/usr/bin/env python3
"""Audit mechanism-level supplement opportunities after display-family dedupe.

This is intentionally conservative. It is not a blind auto-apply script: it
lists distinct engineering mechanisms that may be used to fill Top-K after
near-duplicate recommendations are removed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmfrec.countermeasure_family import mechanism_family
from scripts.audit_display_family_duplicate_topk import display_family
from scripts.make_display_family_duplicate_review_plan import effect, read_jsonl, star


DEFAULT_PLAN = ROOT / "out/qa_labeler/final_audit_current/display_family_duplicates/strict_duplicate_review_plan.jsonl"
DEFAULT_CANDIDATES = ROOT / "out/qa_labeler/final_audit_current/display_family_duplicates/strict_duplicate_candidates_top30.jsonl"
DEFAULT_EVIDENCE = ROOT / "data/evidence_store.facility_v4.driveway_formula_all.jsonl"
DEFAULT_OUT_JSONL = ROOT / "out/qa_labeler/final_audit_current/display_family_duplicates/strict_duplicate_mechanism_opportunities.jsonl"
DEFAULT_OUT_MD = ROOT / "out/qa_labeler/final_audit_current/display_family_duplicates/strict_duplicate_mechanism_opportunities.md"
DEFAULT_OUT_SUMMARY = ROOT / "out/qa_labeler/final_audit_current/display_family_duplicates/mechanism_opportunity_summary.json"


def norm(value: object) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def get_context(candidates: dict[str, dict[str, Any]], qid: str) -> dict[str, Any]:
    row = candidates.get(qid) or {}
    parsed = row.get("parsed_context") or {}
    return parsed.get("context") or {}


def candidate_ids(candidates: dict[str, dict[str, Any]], qid: str) -> list[str]:
    return [str(x) for x in (candidates.get(qid) or {}).get("candidate_evidence_ids") or []]


def bucket_for(ev: dict[str, Any]) -> str:
    cm = norm(ev.get("countermeasure"))
    fam, _level = display_family(ev)
    mech = mechanism_family(ev)

    if fam == "median_barrier":
        return "median_barrier"
    if "median width" in cm or re.search(r"\bmedian\b.*\bconversion\b", cm):
        return "median_width_or_cross_section"
    if "median shoulder" in cm:
        return "median_shoulder"
    if "flatten side slope" in cm or "flatten side slopes" in cm or "sideslope" in cm:
        return "sideslope_or_clear_zone"
    if fam == "roadside_guardrail" or "guardrail" in cm or "barrier" in cm and "median" not in cm:
        return "roadside_barrier_or_guardrail"

    if fam == "rumble_strips":
        if "centerline" in cm:
            return "centerline_rumble_strips"
        if "shoulder" in cm or "edge line" in cm or "edgeline" in cm:
            return "shoulder_or_edge_rumble_strips"
        return "rumble_strips"
    if fam == "shoulder_improvement" or "widen shoulder" in cm or "pave shoulder" in cm:
        return "shoulder_widening_or_paving"
    if mech == "lane_departure_safety_edge" or "safety edge" in cm:
        return "safety_edge"
    if "widen narrow pavement" in cm or "widen pavement" in cm:
        return "lane_or_pavement_widening"

    if fam == "pavement_friction" or "high friction" in cm or "hfst" in cm:
        return "pavement_friction"
    if "open graded asphalt" in cm or "ogac" in cm or "resurface" in cm:
        return "resurfacing_or_drainage_surface"
    if "wet reflective" in cm or "wet-reflective" in cm:
        return "wet_reflective_markings"

    if fam == "lighting_or_illumination":
        return "lighting_or_illumination"
    if mech.startswith("signal_visibility") or "signal head" in cm or "backplate" in cm or "lens" in cm:
        return "traffic_signal_visibility"
    if "raised pavement marker" in cm or "pavement marker" in cm:
        return "pavement_markers_or_delineation"
    if "wider edge line" in cm or "wider line" in cm or "wider pavement marking" in cm:
        return "wider_pavement_markings"
    if "chevron" in cm or "curve warning" in cm or "curve sign" in cm or "curve marking" in cm:
        return "curve_delineation"

    if fam == "new_signalization" or "install a traffic signal" in cm:
        return "new_signalization"
    if fam == "roundabout_conversion" or "roundabout" in cm:
        return "roundabout_conversion"
    if "all way stop" in cm or "all-way stop" in cm:
        return "all_way_stop_control"
    if "icws" in cm or "conflict warning" in cm:
        return "intersection_conflict_warning"
    if "flashing beacon" in cm or "advance warning" in cm:
        return "intersection_warning_or_beacon"
    if "left turn lane" in cm:
        return "left_turn_lane"
    if "offset left" in cm or "left turn offset" in cm:
        return "left_turn_offset"
    if "j turn" in cm or "j-turn" in cm or "median u turn" in cm or "median u-turn" in cm:
        return "j_turn_or_median_u_turn"
    if "right turn lane" in cm:
        return "right_turn_lane"

    if "speed" in cm or "automated enforcement" in cm:
        return "speed_management"
    if "bicycle lane" in cm or "cycle track" in cm or "bicycle track" in cm:
        return "bicycle_facility"
    if "pedestrian hybrid beacon" in cm or "rrfb" in cm or "crosswalk" in cm:
        return "pedestrian_crossing_treatment"

    return f"other:{fam}"


def query_flags(query: str, context: dict[str, Any]) -> set[str]:
    q = norm(query)
    flags: set[str] = set()
    crash = norm(context.get("crash_type"))
    facility = norm(context.get("facility_type"))
    control = norm(context.get("traffic_control_type"))
    area = norm(context.get("area_type"))
    geometry = norm(context.get("intersection_geometry"))

    if "segment" in q or facility == "segment":
        flags.add("segment")
    if "intersection" in q or "at grade" in facility or context.get("intersection_related") == "yes":
        flags.add("intersection")
    if "interchange" in q or "ramp" in q or "interchange" in facility:
        flags.add("interchange_or_ramp")
    if "rural" in q or area == "rural":
        flags.add("rural")
    if "urban" in q or area == "urban":
        flags.add("urban")
    if "freeway" in q or "expressway" in q or "interstate" in q:
        flags.add("freeway_or_expressway")
    if "divided" in q or "median" in q:
        flags.add("divided_or_median")
    if "undivided" in q or "two lane" in q or "two-lane" in q:
        flags.add("two_lane_undivided")
    if "curve" in q or "horizontal" in q:
        flags.add("curve")
    if "wet" in q or "skid" in q or "traction" in q or "wet road" in crash:
        flags.add("wet_or_skid")
    if "night" in q or "dark" in q or "nighttime" in crash:
        flags.add("night")
    if "run off" in q or "run-off" in q or "runoff" in q or "lane departure" in q:
        flags.add("runoff_or_lane_departure")
    if "head on" in q or "head-on" in q or "opposite direction" in q or "opposing direction" in q:
        flags.add("head_on_or_opposing")
    if "cross median" in q or "cross-median" in q:
        flags.add("cross_median")
    if "fixed object" in q or "roadside" in q:
        flags.add("roadside")
    if "stop" in q or "stop" in control:
        flags.add("stop_controlled")
    if "all way" in q or "all-way" in q:
        flags.add("all_way_stop")
    if "signal" in q or "signalized" in q or "signalized" in control:
        flags.add("signalized")
    if "angle" in q or "angle" in crash:
        flags.add("angle")
    if "rear end" in q or "rear-end" in q or "rear end" in crash:
        flags.add("rear_end")
    if "left turn" in q or "left-turn" in q:
        flags.add("left_turn")
    if "pedestrian" in q:
        flags.add("pedestrian")
    if "bicycle" in q or "bike" in q:
        flags.add("bicycle")
    if "4 leg" in q or "4-leg" in q or "4 leg" in geometry:
        flags.add("four_leg")
    max_speed = context.get("max_speed_limit")
    try:
        if max_speed is not None and float(max_speed) >= 40:
            flags.add("high_speed_intersection")
    except (TypeError, ValueError):
        pass
    if "high speed" in q or "high-speed" in q:
        flags.add("high_speed_intersection")
    max_major = context.get("max_major_road_traffic_volume") or context.get("avg_major_road_traffic_volume")
    max_minor = context.get("max_minor_road_traffic_volume") or context.get("avg_minor_road_traffic_volume")
    try:
        if max_major is not None and float(max_major) >= 15000:
            flags.add("high_volume_intersection")
    except (TypeError, ValueError):
        pass
    try:
        if max_minor is not None and float(max_minor) >= 10000:
            flags.add("high_volume_intersection")
    except (TypeError, ValueError):
        pass
    if "moderate to high" in q or "tens of thousands" in q:
        flags.add("high_volume_intersection")
    return flags


def bucket_allowed(bucket: str, flags: set[str], ev: dict[str, Any]) -> tuple[bool, str]:
    """Return whether a bucket is a conservative supplement for this query."""
    cm = norm(ev.get("countermeasure"))

    if "cross_median" in flags:
        allowed = {
            "median_barrier",
            "median_width_or_cross_section",
            "median_shoulder",
            "sideslope_or_clear_zone",
        }
        return bucket in allowed, "cross-median scenarios should stay within median protection/cross-section mechanisms"

    if "two_lane_undivided" in flags and bucket in {"median_barrier", "median_width_or_cross_section", "j_turn_or_median_u_turn"}:
        return False, "median treatments conflict with ordinary two-lane undivided context"

    if "wet_or_skid" in flags:
        if "intersection" in flags:
            allowed = {"pavement_friction", "resurfacing_or_drainage_surface"}
            return bucket in allowed, "wet-skid intersection should use surface-friction/drainage mechanisms"
        allowed = {"pavement_friction", "resurfacing_or_drainage_surface", "wet_reflective_markings"}
        if "curve" in flags:
            allowed |= {"curve_delineation", "speed_management", "sideslope_or_clear_zone"}
        return bucket in allowed, "wet-road segment supplement"

    if "runoff_or_lane_departure" in flags:
        allowed = {
            "shoulder_or_edge_rumble_strips",
            "shoulder_widening_or_paving",
            "safety_edge",
            "roadside_barrier_or_guardrail",
            "sideslope_or_clear_zone",
            "pavement_friction",
            "lane_or_pavement_widening",
        }
        if "curve" in flags:
            allowed |= {"curve_delineation", "speed_management"}
        if "night" in flags:
            allowed |= {"lighting_or_illumination", "pavement_markers_or_delineation", "wider_pavement_markings"}
        return bucket in allowed, "run-off-road/lane-departure supplement"

    if "head_on_or_opposing" in flags:
        allowed = {"centerline_rumble_strips", "lane_or_pavement_widening", "safety_edge"}
        if "divided_or_median" in flags:
            allowed |= {"median_barrier", "median_width_or_cross_section", "median_shoulder"}
        return bucket in allowed, "head-on/opposing-direction supplement"

    if "night" in flags:
        if "signalized" in flags:
            allowed = {"lighting_or_illumination", "traffic_signal_visibility", "pavement_markers_or_delineation"}
        elif "intersection" in flags:
            allowed = {"lighting_or_illumination"}
        else:
            allowed = {
                "lighting_or_illumination",
                "pavement_markers_or_delineation",
                "wider_pavement_markings",
            }
            if "curve" in flags:
                allowed.add("curve_delineation")
        return bucket in allowed, "nighttime visibility supplement"

    if "intersection" in flags and "stop_controlled" in flags:
        if "all_way_stop" in flags:
            allowed = {"roundabout_conversion", "new_signalization", "intersection_warning_or_beacon"}
        else:
            allowed = {
                "new_signalization",
                "roundabout_conversion",
                "all_way_stop_control",
                "intersection_conflict_warning",
                "intersection_warning_or_beacon",
                "left_turn_lane",
                "right_turn_lane",
                "left_turn_offset",
                "j_turn_or_median_u_turn",
            }
        if "high_speed_intersection" in flags or "high_volume_intersection" in flags:
            allowed.discard("all_way_stop_control")
        if "rear_end" in flags:
            allowed -= {"intersection_conflict_warning"}
        return bucket in allowed, "stop-controlled intersection supplement"

    if "intersection" in flags and "signalized" in flags:
        allowed = {"traffic_signal_visibility", "lighting_or_illumination"}
        if "left_turn" in flags or "angle" in flags:
            allowed |= {"left_turn_lane", "left_turn_offset", "j_turn_or_median_u_turn"}
        if "rear_end" in flags:
            allowed |= {"left_turn_lane"}
        if "pedestrian" in flags:
            allowed |= {"pedestrian_crossing_treatment"}
        if "bicycle" in flags:
            allowed |= {"bicycle_facility"}
        return bucket in allowed, "signalized intersection supplement"

    if "pedestrian" in flags:
        return bucket in {"pedestrian_crossing_treatment", "lighting_or_illumination"}, "pedestrian supplement"
    if "bicycle" in flags:
        return bucket in {"bicycle_facility", "pedestrian_crossing_treatment"}, "bicycle/active-transport supplement"

    if "speed" in cm and ("speed" not in flags):
        return False, "speed treatment only when speed context is explicit"
    return False, "no conservative supplement rule triggered"


def score_candidate(ev: dict[str, Any], rank: int, bucket: str) -> float:
    bucket_bonus = {
        "median_barrier": 4,
        "pavement_friction": 4,
        "centerline_rumble_strips": 4,
        "shoulder_or_edge_rumble_strips": 4,
        "safety_edge": 3,
        "traffic_signal_visibility": 3,
        "roundabout_conversion": 3,
        "new_signalization": 3,
        "all_way_stop_control": 3,
        "intersection_warning_or_beacon": 2,
        "curve_delineation": 2,
        "wet_reflective_markings": 2,
    }.get(bucket, 1)
    return bucket_bonus + star(ev) * 2 + min(max(effect(ev, "crf"), -50), 100) / 20 - rank * 0.04


def opportunity_card(eid: str, ev: dict[str, Any], bucket: str, reason: str, rank: int, score: float) -> dict[str, Any]:
    fam, level = display_family(ev)
    return {
        "evidence_id": str(eid),
        "countermeasure": ev.get("countermeasure"),
        "display_family": fam,
        "duplicate_level": level,
        "mechanism_family": mechanism_family(ev),
        "mechanism_bucket": bucket,
        "rank_in_candidate_pool": rank,
        "cmf": (ev.get("effect") or {}).get("cmf"),
        "crf": (ev.get("effect") or {}).get("crf"),
        "star": (ev.get("quality") or {}).get("star_quality_rating"),
        "reason": reason,
        "score": round(score, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--evidence-store", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT_JSONL)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    plans = read_jsonl(args.plan)
    candidates = {str(row["qid"]): row for row in read_jsonl(args.candidates)}
    evidence = {str(row.get("evidence_id")): row for row in read_jsonl(args.evidence_store) if row.get("evidence_id")}

    out_rows: list[dict[str, Any]] = []
    summary = Counter()
    by_action: dict[str, Counter[str]] = defaultdict(Counter)
    by_dup_family: dict[str, Counter[str]] = defaultdict(Counter)
    bucket_counts = Counter()

    for row in plans:
        qid = str(row["qid"])
        context = get_context(candidates, qid)
        flags = sorted(query_flags(row.get("query") or "", context))
        selected_ids = {str(c.get("evidence_id")) for c in row.get("suggested_topk") or []}
        selected_buckets = {bucket_for(evidence.get(eid) or {}) for eid in selected_ids}
        selected_display_families = {c.get("family") for c in row.get("suggested_topk") or []}
        duplicate_families = set(row.get("duplicate_families") or [])

        opportunities: list[dict[str, Any]] = []
        rejected_seen = Counter()
        for rank, eid in enumerate(candidate_ids(candidates, qid), start=1):
            if eid in selected_ids:
                continue
            ev = evidence.get(eid) or {}
            if not ev:
                continue
            display_fam, _level = display_family(ev)
            bucket = bucket_for(ev)
            allowed, reason = bucket_allowed(bucket, set(flags), ev)
            if not allowed:
                rejected_seen[reason] += 1
                continue
            if display_fam in selected_display_families or display_fam in duplicate_families:
                rejected_seen["same display family already represented"] += 1
                continue
            if bucket in selected_buckets:
                rejected_seen["same mechanism bucket already represented"] += 1
                continue
            # Avoid weak filler. Formula-style rows can pass if the star rating is usable.
            if star(ev) < 3 and effect(ev, "crf") < 25:
                rejected_seen["weak effect/quality for supplement"] += 1
                continue
            score = score_candidate(ev, rank, bucket)
            opportunities.append(opportunity_card(eid, ev, bucket, reason, rank, score))

        opportunities.sort(key=lambda item: item["score"], reverse=True)
        # Keep at most one evidence card per mechanism bucket for readability.
        compact: list[dict[str, Any]] = []
        used_buckets: set[str] = set()
        for item in opportunities:
            if item["mechanism_bucket"] in used_buckets:
                continue
            compact.append(item)
            used_buckets.add(item["mechanism_bucket"])
            if len(compact) >= 5:
                break

        current_len = len(row.get("suggested_topk") or [])
        if current_len >= 3:
            status = "already_top3_review_only" if compact else "already_top3_no_extra"
        elif compact:
            status = "can_supplement_to_top3"
        else:
            status = "no_safe_supplement_found"

        summary[status] += 1
        summary[f"current_top{current_len}"] += 1
        by_action[row["action"]][status] += 1
        for family in row.get("duplicate_families") or []:
            by_dup_family[family][status] += 1
        for item in compact[:3]:
            bucket_counts[item["mechanism_bucket"]] += 1

        out_rows.append(
            {
                "qid": qid,
                "dataset": row.get("dataset"),
                "query": row.get("query"),
                "current_action": row.get("action"),
                "duplicate_families": row.get("duplicate_families"),
                "current_suggested_topk": row.get("suggested_topk"),
                "context_flags": flags,
                "status": status,
                "recommended_next_step": (
                    "consider adding the first listed distinct mechanism"
                    if status == "can_supplement_to_top3"
                    else "keep current shorter Top-K unless manually revised"
                    if status == "no_safe_supplement_found"
                    else "review only; current Top-K already has three items"
                ),
                "supplement_opportunities": compact,
                "rejection_summary": rejected_seen.most_common(5),
            }
        )

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.out_jsonl.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in out_rows) + "\n",
        encoding="utf-8",
    )

    summary_obj = {
        "total_rows": len(out_rows),
        "status_counts": summary.most_common(),
        "by_current_action": {k: v.most_common() for k, v in sorted(by_action.items())},
        "by_duplicate_family": {k: v.most_common() for k, v in sorted(by_dup_family.items())},
        "top_supplement_buckets": bucket_counts.most_common(30),
        "out_jsonl": str(args.out_jsonl),
        "out_md": str(args.out_md),
    }
    args.out_summary.write_text(json.dumps(summary_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Strict Duplicate Mechanism Supplement Opportunities",
        "",
        f"Total rows: {len(out_rows)}",
        "",
        "## Status Counts",
        "",
    ]
    for key, value in summary.most_common():
        if key.startswith("current_top"):
            continue
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Top Supplement Mechanism Buckets", ""])
    for bucket, value in bucket_counts.most_common(20):
        lines.append(f"- {bucket}: {value}")

    lines.extend(["", "## Rows That Can Be Supplemented To Top-3", ""])
    supplement_rows = [row for row in out_rows if row["status"] == "can_supplement_to_top3"]
    for row in supplement_rows[:80]:
        lines.append(f"### {row['qid']} ({row['current_action']})")
        lines.append("")
        lines.append(f"Query: {row['query']}")
        lines.append("")
        lines.append("Current Top-K:")
        for item in row["current_suggested_topk"] or []:
            lines.append(f"- R{item.get('rank')} #{item.get('evidence_id')} {item.get('countermeasure')} [{item.get('family')}]")
        lines.append("Possible supplements:")
        for item in row["supplement_opportunities"][:3]:
            lines.append(
                f"- #{item['evidence_id']} {item['countermeasure']} "
                f"[{item['mechanism_bucket']}; CRF={item['crf']}; Star={item['star']}; cand_rank={item['rank_in_candidate_pool']}]"
            )
        lines.append("")

    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary_obj, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
