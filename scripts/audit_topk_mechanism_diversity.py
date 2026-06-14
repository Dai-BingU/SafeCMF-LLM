#!/usr/bin/env python3
"""Audit Top-K mechanism diversity for the current recommendation dataset.

This is a diagnostic script only. It does not edit data. The goal is to find
queries where the displayed Top-K list may over-concentrate on one engineering
pathway when the query is broad enough to deserve a more diverse treatment set.
"""

from __future__ import annotations

import csv
import json
import re
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cmfrec.countermeasure_family import mechanism_family  # noqa: E402


MAIN_Q = ROOT / "data/queries.training_3501.naturalized_dedup_v2.rpm_roadside_curve_negcurv_groupfix_v769.pathway_support_global_v1.jsonl"
MAIN_L = ROOT / "data/labels.reco.training_3501.dedup_v2.rpm_roadside_curve_negcurv_groupfix_v769.pathway_support_global_v1.jsonl"
REPAIR_Q = ROOT / "data/repair/queries.source_preserving_repairs_v700.pathway_support_global_v1.jsonl"
REPAIR_L = ROOT / "data/repair/labels.source_preserving_repairs_v700.pathway_support_global_v1.jsonl"
EVIDENCE = ROOT / "data/evidence_store.facility_v1.jsonl"

OUT_DIR = ROOT / "out/mechanism_diversity_audit_current"
SUMMARY_OUT = OUT_DIR / "summary.json"
CSV_OUT = OUT_DIR / "high_risk_cases.csv"
MD_OUT = OUT_DIR / "high_risk_cases.md"
ALL_OUT = OUT_DIR / "all_cases.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def norm(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def primary_slot_from_mechanism(family: str, text: str = "") -> str:
    """Map detailed evidence mechanisms to engineering pathway slots."""
    t = norm(text)
    f = family

    if f in {"lane_departure_centerline_rumble", "lane_departure_shoulder_rumble", "lane_departure_rumble_combined"}:
        return "lane_departure_prevention_rumble"
    if f in {"lane_departure_wider_markings", "night_visibility_pavement_markers", "curve_guidance_delineation"}:
        return "delineation_or_warning"
    if f in {"lane_departure_safety_edge", "shoulder_improvement_pave", "shoulder_improvement_pave_deteriorated", "shoulder_improvement_widen"}:
        return "recovery_support_shoulder_edge"
    if f in {"roadside_guardrail", "roadside_embankment_guardrail", "roadside_side_slope_guardrail", "roadside_utility_pole_guardrail", "median_barrier"}:
        return "barrier_or_roadside_protection"
    if f in {"roadside_sideslope_improvement", "roadside_lateral_clearance", "roadside_fixed_object_removal"}:
        return "roadside_hazard_reduction"
    if f == "pavement_friction":
        return "pavement_friction"
    if f in {"speed_enforcement", "speed_management_posted_limit", "speed_management_variable_speed_limit", "curve_speed_management"}:
        return "speed_management"

    if f in {"intersection_control_signalization", "intersection_control_all_way_stop", "intersection_control_roundabout"}:
        return "intersection_control_change"
    if f in {"intersection_warning_flashing_beacon", "intersection_stop_sign_visibility_signing", "intersection_advance_guidance_signing"}:
        return "intersection_warning_visibility"
    if f in {"intersection_left_turn_lane", "intersection_left_turn_offset"}:
        return "turn_lane_or_offset_geometry"
    if f == "intersection_left_turn_phasing":
        return "left_turn_signal_phasing"
    if f in {"intersection_displaced_left_turn_or_cfi", "intersection_median_u_turn", "intersection_restricted_crossing_u_turn"}:
        return "intersection_geometric_reconfiguration"

    if f in {"active_transport_uncontrolled_crossing_beacon", "active_transport_rural_crosswalk_warning"}:
        return "ped_bike_crossing_beacon_warning"
    if f in {"active_transport_uncontrolled_crossing_refuge", "active_transport_uncontrolled_crossing_traffic_calming"}:
        return "ped_bike_crossing_refuge_or_calming"
    if f in {"active_transport_pedestrian_signal", "active_transport_pedestrian_signal_timing"}:
        return "pedestrian_signal_timing_or_display"
    if f in {"active_transport_bicycle_lane", "active_transport_bicycle_track"}:
        return "bicycle_facility"

    if f.startswith("access_management_") or f in {"lane_reconfiguration_road_diet", "access_management_twlttl"}:
        return "access_or_corridor_management"
    if f == "lighting":
        return "lighting"
    if f.startswith("winter_weather_"):
        return "winter_weather_surface_management"
    if f == "toll_plaza_electronic_or_open_road_tolling":
        return "toll_plaza_operations"
    if f == "frontage_road_one_way_conversion":
        return "frontage_road_operations"
    if f == "passing_or_climbing_lane":
        return "passing_or_climbing_lane"
    if f == "signal_visibility":
        return "signal_visibility"
    if f == "signal_operations_adaptive_control":
        return "signal_operations"
    if f == "signal_operations_red_light_camera":
        return "red_light_or_speed_camera"
    if "guardrail" in t or "barrier" in t:
        return "barrier_or_roadside_protection"
    return f


SCENARIO_SLOT_HINTS: dict[str, set[str]] = {
    "run_off_road_or_lane_departure": {
        "lane_departure_prevention_rumble",
        "delineation_or_warning",
        "recovery_support_shoulder_edge",
        "barrier_or_roadside_protection",
        "roadside_hazard_reduction",
        "pavement_friction",
        "speed_management",
    },
    "cross_median": {
        "barrier_or_roadside_protection",
        "recovery_support_shoulder_edge",
        "roadside_hazard_reduction",
        "delineation_or_warning",
        "speed_management",
    },
    "pedestrian": {
        "ped_bike_crossing_beacon_warning",
        "ped_bike_crossing_refuge_or_calming",
        "pedestrian_signal_timing_or_display",
        "lighting",
        "intersection_warning_visibility",
    },
    "bicycle": {
        "bicycle_facility",
        "ped_bike_crossing_beacon_warning",
        "ped_bike_crossing_refuge_or_calming",
        "pedestrian_signal_timing_or_display",
        "turn_lane_or_offset_geometry",
    },
    "left_turn": {
        "left_turn_signal_phasing",
        "turn_lane_or_offset_geometry",
        "intersection_geometric_reconfiguration",
        "intersection_control_change",
        "intersection_warning_visibility",
    },
    "stop_control_angle": {
        "intersection_control_change",
        "intersection_warning_visibility",
        "turn_lane_or_offset_geometry",
        "intersection_geometric_reconfiguration",
    },
    "wet_road": {
        "pavement_friction",
        "delineation_or_warning",
        "recovery_support_shoulder_edge",
        "speed_management",
    },
    "nighttime": {
        "lighting",
        "delineation_or_warning",
        "intersection_warning_visibility",
        "signal_visibility",
        "lane_departure_prevention_rumble",
    },
    "speed_related": {
        "speed_management",
        "delineation_or_warning",
        "pavement_friction",
        "intersection_warning_visibility",
    },
    "access_management": {
        "access_or_corridor_management",
        "turn_lane_or_offset_geometry",
        "intersection_control_change",
        "intersection_geometric_reconfiguration",
    },
}


def classify_scenarios(query: dict[str, Any]) -> set[str]:
    ctx = query.get("target_context") or {}
    text = norm(query.get("user_text"))
    crash = norm(ctx.get("crash_type"))
    traffic_control = norm(ctx.get("traffic_control_type"))
    inter = norm(ctx.get("intersection_related"))

    tags: set[str] = set()
    if any(x in f" {text} {crash} " for x in ["run-off", "run off", "roadway-departure", "lane-departure", "lane departure", "fixed-object", "fixed object", "single-vehicle", "single vehicle"]):
        tags.add("run_off_road_or_lane_departure")
    if "cross median" in text or "cross-median" in text or "median-departure" in text or "cross median" in crash:
        tags.add("cross_median")
    if "pedestrian" in text or "vehicle/pedestrian" in crash:
        tags.add("pedestrian")
    if "bicycle" in text or "bike" in text or "vehicle/bicycle" in crash:
        tags.add("bicycle")
    if "left-turn" in text or "left turn" in text or "left turn" in crash:
        tags.add("left_turn")
    if ("angle" in text or "angle" in crash) and ("stop" in traffic_control or "stop-controlled" in text or "stop controlled" in text):
        tags.add("stop_control_angle")
    if "wet" in text or "wet road" in crash or "rain" in text:
        tags.add("wet_road")
    if "night" in text or "dark" in text or norm(ctx.get("crash_time_of_day")) == "nighttime":
        tags.add("nighttime")
    if "speed" in text or "speed-related" in text or "speed related" in crash:
        tags.add("speed_related")
    if (
        "access-related" in text
        or "access related" in text
        or "driveway" in text
        or "access point" in text
        or "median opening" in text
        or "twltl" in text
        or "two-way left turn" in text
    ):
        tags.add("access_management")
        # Do not let the word "multimodal" alone explode this into ped/bike/runoff/speed.
        if "multimodal access" in text:
            tags.discard("pedestrian")
            tags.discard("bicycle")
            tags.discard("run_off_road_or_lane_departure")
            tags.discard("speed_related")
            tags.discard("left_turn")
    if not tags:
        tags.add("other")
    return tags


def is_narrow_query(query: dict[str, Any]) -> bool:
    text = norm(query.get("user_text"))
    narrow_terms = [
        "where pavement-edge",
        "where pavement edge",
        "edge drop",
        "drop-off",
        "limited or no lighting",
        "lighting is limited",
        "lighting is missing",
        "uncontrolled crossing",
        "existing permissive",
        "permissive left-turn",
        "permissive left turn",
        "median opening",
        "frontage road",
        "toll plaza",
        "railroad",
    ]
    return any(term in text for term in narrow_terms)


def load_evidence(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        out[str(row.get("evidence_id"))] = row
    return out


def representative_ids(label: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for item in label.get("topk") or []:
        ev = [str(x) for x in (item.get("evidence_ids") or [])]
        if ev:
            ids.append(ev[0])
        elif item.get("cm_id"):
            ids.append(str(item.get("cm_id")))
    return ids


def all_support_ids(label: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for item in label.get("topk") or []:
        ev = [str(x) for x in (item.get("evidence_ids") or [])]
        ids.extend(ev)
    return ids


def audit_case(dataset: str, query: dict[str, Any], label: dict[str, Any], evidence: dict[str, dict[str, Any]]) -> dict[str, Any]:
    qid = query["qid"]
    rep_ids = representative_ids(label)
    slots: list[str] = []
    mechanisms: list[str] = []
    names: list[str] = []
    for eid in rep_ids:
        ev = evidence.get(eid, {"countermeasure": eid})
        mech = mechanism_family(ev)
        mechanisms.append(mech)
        slots.append(primary_slot_from_mechanism(mech, ev.get("countermeasure", "")))
        names.append(ev.get("countermeasure", ""))

    topk_n = len(rep_ids)
    distinct_slots = len(set(slots))
    distinct_mechanisms = len(set(mechanisms))
    scenarios = classify_scenarios(query)
    narrow = is_narrow_query(query)

    expected_slots: set[str] = set()
    for tag in scenarios:
        expected_slots |= SCENARIO_SLOT_HINTS.get(tag, set())
    relevant_slots = [slot for slot in slots if not expected_slots or slot in expected_slots]
    repeated_slots = [slot for slot, count in Counter(slots).items() if count > 1]
    repeated_mechanisms = [mech for mech, count in Counter(mechanisms).items() if count > 1]

    flags: list[str] = []
    risk = 0

    if topk_n >= 3 and distinct_slots == 1 and not narrow:
        flags.append("all_topk_same_slot_for_broad_query")
        risk += 40
    elif topk_n >= 3 and distinct_slots <= 2 and not narrow:
        flags.append("limited_slot_diversity_for_broad_query")
        risk += 18
    if repeated_mechanisms:
        flags.append("repeated_mechanism_family")
        risk += min(25, 8 * len(repeated_mechanisms))
    if repeated_slots and not narrow:
        flags.append("repeated_engineering_slot")
        risk += min(25, 8 * len(repeated_slots))
    if expected_slots and topk_n >= 2 and len(relevant_slots) == 0 and "other" not in scenarios:
        flags.append("no_expected_slot_match")
        risk += 30
    if expected_slots and topk_n >= 3 and len(set(relevant_slots)) <= 1 and len(scenarios) >= 2 and not narrow:
        flags.append("multi_scenario_low_relevant_diversity")
        risk += 25
    if topk_n == 3 and len(all_support_ids(label)) == 3 and repeated_mechanisms:
        flags.append("duplicates_not_grouped_as_supporting_evidence")
        risk += 10

    # Some low diversity is acceptable for highly specialized operational samples.
    if narrow and risk:
        risk = max(0, risk - 20)
        flags.append("narrow_query_relaxed")

    return {
        "dataset": dataset,
        "qid": qid,
        "query": query.get("user_text", ""),
        "source_evidence_id": query.get("source_evidence_id"),
        "scenarios": sorted(scenarios),
        "is_narrow_query": narrow,
        "topk_n": topk_n,
        "representative_ids": rep_ids,
        "countermeasures": names,
        "mechanisms": mechanisms,
        "slots": slots,
        "distinct_slots": distinct_slots,
        "distinct_mechanisms": distinct_mechanisms,
        "repeated_slots": repeated_slots,
        "repeated_mechanisms": repeated_mechanisms,
        "expected_slots": sorted(expected_slots),
        "flags": flags,
        "risk_score": risk,
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "risk_score",
        "dataset",
        "qid",
        "query",
        "scenarios",
        "is_narrow_query",
        "topk_n",
        "representative_ids",
        "countermeasures",
        "slots",
        "mechanisms",
        "distinct_slots",
        "repeated_slots",
        "repeated_mechanisms",
        "flags",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row.get(k), ensure_ascii=False) if isinstance(row.get(k), (list, dict)) else row.get(k) for k in fields})


def write_markdown(rows: list[dict[str, Any]], path: Path, limit: int = 120) -> None:
    lines = [
        "# Top-K Mechanism Diversity Audit",
        "",
        "This report lists high-risk cases where the displayed Top-K may not cover sufficiently distinct engineering pathways.",
        "",
        f"Showing top {min(limit, len(rows))} of {len(rows)} high-risk cases.",
        "",
    ]
    for row in rows[:limit]:
        lines += [
            f"## {row['qid']} ({row['dataset']}) - risk {row['risk_score']}",
            "",
            f"**Query**: {row['query']}",
            "",
            f"**Scenarios**: {', '.join(row['scenarios'])}",
            f"**Flags**: {', '.join(row['flags'])}",
            "",
        ]
        for i, (eid, name, slot, mech) in enumerate(zip(row["representative_ids"], row["countermeasures"], row["slots"], row["mechanisms"]), start=1):
            lines.append(f"- Rank {i}: #{eid} {name} | slot=`{slot}` | family=`{mech}`")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def rel_or_str(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-queries", type=Path, default=MAIN_Q)
    parser.add_argument("--main-labels", type=Path, default=MAIN_L)
    parser.add_argument("--repair-queries", type=Path, default=REPAIR_Q)
    parser.add_argument("--repair-labels", type=Path, default=REPAIR_L)
    parser.add_argument("--evidence-store", type=Path, default=EVIDENCE)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    out_dir = args.out_dir
    summary_out = out_dir / "summary.json"
    csv_out = out_dir / "high_risk_cases.csv"
    md_out = out_dir / "high_risk_cases.md"
    all_out = out_dir / "all_cases.jsonl"

    evidence = load_evidence(args.evidence_store)
    main_queries = read_jsonl(args.main_queries)
    repair_queries = read_jsonl(args.repair_queries)
    qrows = {**{r["qid"]: r for r in main_queries}, **{r["qid"]: r for r in repair_queries}}
    datasets = {r["qid"]: "main" for r in main_queries}
    datasets.update({r["qid"]: "repair" for r in repair_queries})
    labels = read_jsonl(args.main_labels) + read_jsonl(args.repair_labels)

    rows = [audit_case(datasets.get(label["qid"], "unknown"), qrows[label["qid"]], label, evidence) for label in labels]
    rows.sort(key=lambda r: (-r["risk_score"], r["qid"]))

    high_risk = [r for r in rows if r["risk_score"] >= 40]
    medium_risk = [r for r in rows if 20 <= r["risk_score"] < 40]
    flagged = [r for r in rows if r["risk_score"] > 0]

    out_dir.mkdir(parents=True, exist_ok=True)
    all_out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    write_csv(high_risk + medium_risk, csv_out)
    write_markdown(high_risk + medium_risk, md_out)

    summary = {
        "input": {
            "main_queries": rel_or_str(args.main_queries),
            "main_labels": rel_or_str(args.main_labels),
            "repair_queries": rel_or_str(args.repair_queries),
            "repair_labels": rel_or_str(args.repair_labels),
        },
        "counts": {
            "total_rows": len(rows),
            "flagged_rows": len(flagged),
            "high_risk_ge_40": len(high_risk),
            "medium_risk_20_39": len(medium_risk),
            "zero_risk": sum(1 for r in rows if r["risk_score"] == 0),
        },
        "risk_by_scenario": {
            tag: {
                "high": sum(1 for r in high_risk if tag in r["scenarios"]),
                "medium": sum(1 for r in medium_risk if tag in r["scenarios"]),
                "any": sum(1 for r in flagged if tag in r["scenarios"]),
            }
            for tag in sorted({tag for r in rows for tag in r["scenarios"]})
        },
        "flag_counts": dict(Counter(flag for row in rows for flag in row["flags"])),
        "top_repeated_slots": dict(Counter(slot for row in rows for slot in row["repeated_slots"]).most_common(20)),
        "outputs": {
            "all_cases": rel_or_str(all_out),
            "high_risk_csv": rel_or_str(csv_out),
            "high_risk_markdown": rel_or_str(md_out),
        },
    }
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
