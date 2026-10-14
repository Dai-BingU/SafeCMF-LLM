#!/usr/bin/env python3
"""Build evidence-independent road-safety scenario clusters for reconstructed-v3.

This stage deliberately does not read the CMF evidence store, candidate pools,
old queries, or old labels.  The only inherited artifact is the aggregate
facility x crash-pattern distribution reported in the manuscript.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MATRIX = {
    "At-grade intersection": [250, 0, 102, 64, 165, 74, 154, 32, 8, 8, 14],
    "Street/corridor segment": [18, 20, 45, 9, 9, 22, 0, 39, 5, 29, 36],
    "Rural highway/road segment": [0, 67, 7, 44, 2, 16, 0, 7, 38, 4, 31],
    "Freeway/expressway segment": [7, 54, 0, 51, 6, 13, 0, 0, 18, 9, 48],
    "Ramp/interchange": [13, 4, 0, 12, 1, 31, 0, 0, 0, 12, 31],
    "Horizontal curve segment": [8, 73, 0, 12, 0, 6, 0, 0, 10, 2, 2],
    "Other road segment": [10, 28, 57, 13, 12, 2, 4, 30, 16, 4, 17],
}

MECHANISMS = [
    "Angle",
    "Run-off-road / lane departure",
    "Vehicle/pedestrian",
    "Nighttime",
    "Rear-end",
    "Multiple vehicle / mixed",
    "Left-turn",
    "Vehicle/bicycle",
    "Wet/friction",
    "Speed-related",
    "Other crash mechanisms",
]

CRASH_PATTERNS = {
    "Angle": ["right-angle crashes", "side-impact crashes", "crossing-path crashes"],
    "Run-off-road / lane departure": ["run-off-road crashes", "lane-departure crashes"],
    "Vehicle/pedestrian": ["pedestrian crashes", "vehicle-pedestrian conflicts", "crashes involving people crossing"],
    "Nighttime": ["crashes after dark", "nighttime crashes", "crashes during low-light conditions"],
    "Rear-end": ["rear-end crashes", "queue-related rear-end crashes", "crashes involving sudden stopping"],
    "Multiple vehicle / mixed": ["multivehicle crashes", "sideswipe and lane-change crashes", "mixed crash patterns"],
    "Left-turn": ["left-turn crashes", "crashes involving left-turning vehicles", "opposing-through and left-turn crashes"],
    "Vehicle/bicycle": ["bicycle crashes", "vehicle-bicycle conflicts", "crashes involving people cycling"],
    "Wet/friction": ["wet-weather crashes", "skidding crashes during rain", "loss-of-control crashes on wet pavement"],
    "Speed-related": ["speed-related crashes", "high-severity crashes associated with high approach speeds", "crashes in a high-speed operating environment"],
    "Other crash mechanisms": ["fixed-object and other crashes", "mixed crash patterns", "crashes without a dominant collision type"],
}

# Observable facts are scene inputs, not treatment names or CMF applicability text.
OBSERVATIONS = {
    "Angle": [
        "crashes concentrate on one approach",
        "drivers have a restricted view of cross traffic",
        "the crossing distance is long",
        "the intersection is difficult to recognize on approach",
        "conflicts increase during peak traffic",
        "vehicles enter from the minor approach with little warning",
        "the approaches meet at a skewed angle",
        "crashes occur near the far side of the intersection",
    ],
    "Run-off-road / lane departure": [
        "departures concentrate along the outside edge",
        "roadside recovery space is limited",
        "the pavement edge is difficult to distinguish",
        "drivers sometimes overrun the pavement edge",
        "departures increase on the downgrade",
        "the shoulder is narrow or discontinuous",
        "lane keeping becomes difficult during low visibility",
    ],
    "Vehicle/pedestrian": [
        "people frequently cross at the same location",
        "driver yielding is low near the crossing",
        "the crossing distance is long",
        "parked or queued vehicles restrict visibility",
        "pedestrian activity peaks near transit stops",
        "children or older pedestrians frequently use the site",
        "many crossings occur between intersections",
        "turning vehicles cross the main pedestrian path",
    ],
    "Nighttime": [
        "the roadway path is difficult to follow after dark",
        "pedestrians and roadside objects have low contrast at night",
        "signs and lane boundaries are difficult to see at night",
        "glare affects drivers approaching the site",
        "the site is much less recognizable at night than in daylight",
        "crashes cluster in unlit portions of the site",
        "drivers have difficulty seeing the pavement edge after dark",
        "nighttime crashes concentrate near a conflict point",
    ],
    "Rear-end": [
        "queues regularly extend beyond the expected storage area",
        "drivers encounter stopped traffic with little advance view",
        "braking is frequent during peak periods",
        "traffic speeds vary sharply through the site",
        "the back of the queue moves substantially between cycles",
        "closely spaced access points create repeated slowing",
        "downstream congestion causes intermittent stopping",
        "rear-end crashes concentrate on one approach",
    ],
    "Multiple vehicle / mixed": [
        "lane changes are frequent during heavy traffic",
        "drivers merge across several traffic streams",
        "a lane ends near the conflict area",
        "traffic movements overlap within a short distance",
        "speed differences are large between adjacent lanes",
        "crashes concentrate where traffic enters and exits",
        "drivers make late lane changes near the site",
        "congestion produces irregular merging and braking",
    ],
    "Left-turn": [
        "turning drivers have a restricted view of opposing traffic",
        "left-turn queues extend into a through lane",
        "crashes concentrate during periods of heavy opposing flow",
        "the turning path is difficult to judge",
        "opposing traffic approaches at relatively high speed",
        "left-turning vehicles wait in a constrained space",
        "the intersection alignment complicates the turn",
        "crashes cluster on one left-turn movement",
    ],
    "Vehicle/bicycle": [
        "vehicle and bicycle paths cross at the site",
        "usable road space for cyclists narrows abruptly",
        "turning vehicles cross a frequent bicycle movement",
        "parked vehicles restrict the view of cyclists",
        "the bicycle route becomes discontinuous near the site",
        "cyclists merge with faster traffic over a short distance",
        "driveway traffic crosses the main bicycle path",
        "bicycle conflicts concentrate during peak travel periods",
    ],
    "Wet/friction": [
        "crashes rise during rain and drivers report a slick surface",
        "water collects in wheel paths during storms",
        "skidding occurs during braking or turning",
        "the pavement surface is visibly polished",
        "loss-of-control crashes concentrate after rainfall begins",
        "drainage is slow near the conflict area",
        "wet-weather crashes occur on a downgrade",
        "the surface provides inconsistent grip when wet",
    ],
    "Speed-related": [
        "observed speeds are high near the conflict area",
        "drivers enter the site faster than the surrounding context suggests",
        "speed variation is large between vehicles",
        "the site follows a transition from a high-speed road",
        "serious crashes concentrate during light traffic",
        "drivers accelerate before clearing the conflict area",
        "approach speed leaves little time to react",
        "the operating speed remains high despite roadside activity",
    ],
    "Other crash mechanisms": [
        "crashes concentrate near a roadside access point",
        "large vehicles are involved in a noticeable share of crashes",
        "fixed objects are close to the traveled way",
        "the traffic pattern changes abruptly near the site",
        "crashes involve several different movements",
        "the site has an unusual alignment or transition",
        "drivers appear uncertain about the intended path",
        "crashes recur within one short section",
    ],
}

FACILITY_CONTEXTS = {
    "At-grade intersection": {
        "areas": ["urban", "suburban", "rural"],
        "controls": ["signalized", "stop-controlled", "yield-controlled", "uncontrolled"],
        "geometries": ["three-leg", "four-leg", "skewed four-leg"],
        "forms": ["two-lane roads", "multilane arterial roads", "a divided arterial and a side road", "collector roads"],
    },
    "Street/corridor segment": {
        "areas": ["urban", "suburban"],
        "controls": ["not applicable"],
        "geometries": ["segment"],
        "forms": ["two-lane street", "multilane arterial", "divided boulevard", "commercial corridor", "residential collector"],
    },
    "Rural highway/road segment": {
        "areas": ["rural"],
        "controls": ["not applicable"],
        "geometries": ["segment"],
        "forms": ["two-lane undivided road", "multilane divided highway", "undivided four-lane road", "rolling-terrain highway", "bridge approach"],
    },
    "Freeway/expressway segment": {
        "areas": ["urban", "suburban", "rural"],
        "controls": ["limited access"],
        "geometries": ["basic segment", "lane-drop area", "bridge approach", "freeway transition"],
        "forms": ["four-lane freeway", "six-lane freeway", "eight-lane expressway", "freeway with a narrow median-side shoulder"],
    },
    "Ramp/interchange": {
        "areas": ["urban", "suburban", "rural"],
        "controls": ["free-flow", "yield-controlled merge", "signal-controlled ramp terminal"],
        "geometries": ["entrance ramp", "exit ramp", "loop ramp", "interchange weaving area", "ramp terminal"],
        "forms": ["diamond interchange", "cloverleaf interchange", "system interchange", "service interchange"],
    },
    "Horizontal curve segment": {
        "areas": ["urban", "suburban", "rural"],
        "controls": ["not applicable"],
        "geometries": ["isolated curve", "reverse-curve section", "downgrade curve", "curve near a side-road junction"],
        "forms": ["two-lane road", "multilane undivided road", "divided highway", "mountain road"],
    },
    "Other road segment": {
        "areas": ["urban", "suburban", "rural"],
        "controls": ["not applicable"],
        "geometries": ["segment", "transition area"],
        "forms": ["school-zone road", "local residential street", "bridge segment", "tunnel approach", "railroad-crossing approach", "work-zone approach", "transit corridor", "parking-access road"],
    },
}

SPECIAL_OBSERVATIONS = {
    ("Freeway/expressway segment", "Angle"): [
        "side-impact crashes concentrate where the freeway cross section changes",
        "drivers encounter a crossing-path conflict near the freeway transition",
        "side-impact crashes recur where traffic leaves the limited-access section",
        "the transition creates an unexpected crossing movement",
    ],
    ("Ramp/interchange", "Angle"): [
        "crossing-path conflicts concentrate at the ramp terminal",
        "drivers entering from the ramp have a restricted view of cross traffic",
        "side-impact crashes recur where the ramp meets the surface road",
        "the ramp terminal is difficult to recognize on approach",
    ],
    ("Horizontal curve segment", "Angle"): [
        "crossing-path crashes recur at a side-road junction within the curve",
        "drivers leaving the side road have a restricted view around the curve",
        "the curve obscures vehicles approaching the side-road junction",
        "side-impact crashes concentrate just beyond the curve apex",
    ],
}


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def valid_combo(facility: str, mechanism: str, combo: tuple[str, str, str, str, str, str]) -> bool:
    area, control, geometry, form, crash_pattern, observation = combo
    if facility == "At-grade intersection":
        if control == "uncontrolled" and "queue" in observation:
            return False
        if mechanism == "Left-turn" and control == "yield-controlled":
            return False
        if mechanism == "Rear-end" and control not in {"signalized", "stop-controlled"}:
            return False
    if facility == "Ramp/interchange" and mechanism in {"Angle", "Rear-end"}:
        if mechanism == "Angle":
            return geometry == "ramp terminal" and control in {"yield-controlled merge", "signal-controlled ramp terminal"}
        if geometry not in {"interchange weaving area", "exit ramp", "ramp terminal"} and control == "free-flow":
            return False
    if facility == "Ramp/interchange":
        # Interchange form, ramp geometry, and control must be compatible;
        # marginally balancing fields from a Cartesian product is not enough.
        if form == "system interchange" and control != "free-flow":
            return False
        if geometry == "interchange weaving area" and control != "free-flow":
            return False
        if form == "diamond interchange" and geometry == "loop ramp":
            return False
        if form == "diamond interchange" and geometry == "interchange weaving area":
            return False
        if form == "cloverleaf interchange" and geometry == "ramp terminal":
            return False
        if geometry == "ramp terminal" and control == "free-flow":
            return False
        if geometry != "ramp terminal" and control == "signal-controlled ramp terminal":
            return False
    if facility == "Freeway/expressway segment" and mechanism == "Angle":
        return geometry == "freeway transition"
    if facility == "Freeway/expressway segment" and "roadside access point" in observation:
        return False
    if facility == "Horizontal curve segment" and mechanism == "Angle":
        return geometry == "curve near a side-road junction"
    if mechanism == "Vehicle/pedestrian" and facility == "Rural highway/road segment":
        return form == "bridge approach" or "cross" in observation
    if mechanism == "Vehicle/bicycle" and facility == "Rural highway/road segment":
        return form in {"two-lane undivided road", "bridge approach"}
    if mechanism == "Vehicle/pedestrian" and facility == "Other road segment":
        if form in {"tunnel approach", "railroad-crossing approach"}:
            return False
    if mechanism == "Other crash mechanisms":
        if crash_pattern == "fixed-object and other crashes":
            return observation in {
                "fixed objects are close to the traveled way",
                "crashes recur within one short section",
                "the site has an unusual alignment or transition",
            }
        if crash_pattern == "mixed crash patterns":
            return observation not in {"fixed objects are close to the traveled way"}
    return True


def candidates_for(facility: str, mechanism: str) -> list[tuple[str, str, str, str, str, str]]:
    context = FACILITY_CONTEXTS[facility]
    candidates = [
        combo
        for combo in itertools.product(
            context["areas"],
            context["controls"],
            context["geometries"],
            context["forms"],
            CRASH_PATTERNS[mechanism],
            SPECIAL_OBSERVATIONS.get((facility, mechanism), OBSERVATIONS[mechanism]),
        )
        if valid_combo(facility, mechanism, combo)
    ]
    candidates.sort(key=lambda combo: stable_int(f"v3:{facility}:{mechanism}:{combo}"))
    return candidates


def round_robin_select(candidates: list[tuple[str, str, str, str, str, str]], quota: int) -> list[tuple[str, str, str, str, str, str]]:
    """Prefer broad marginal coverage instead of adjacent Cartesian products."""
    if len(candidates) < quota:
        raise RuntimeError(f"Only {len(candidates)} independent combinations for quota {quota}")
    selected: list[tuple[str, str, str, str, str, str]] = []
    counts = [Counter() for _ in range(6)]
    remaining = list(candidates)
    while len(selected) < quota:
        best = min(
            remaining,
            key=lambda item: (
                sum(counts[index][value] for index, value in enumerate(item)),
                max(counts[index][value] for index, value in enumerate(item)),
                stable_int(f"pick:{len(selected)}:{item}"),
            ),
        )
        selected.append(best)
        remaining.remove(best)
        for index, value in enumerate(best):
            counts[index][value] += 1
    return selected


def select_cell_scenarios(
    facility: str,
    mechanism: str,
    quota: int,
) -> list[tuple[str, str, str, str, str, str]]:
    """Select broad scenarios while keeping skewed intersections a long-tail case.

    Geometry is a taxonomy category, not an empirical prevalence distribution.
    Equal round-robin balancing previously made skewed four-leg intersections
    roughly one third of all at-grade queries. Cap them near ten percent within
    each non-empty at-grade facility x mechanism cell instead.
    """
    candidates = candidates_for(facility, mechanism)
    if facility != "At-grade intersection":
        return round_robin_select(candidates, quota)
    skewed = [item for item in candidates if item[2] == "skewed four-leg"]
    regular = [item for item in candidates if item[2] != "skewed four-leg"]
    skew_quota = min(len(skewed), max(1, round(quota * 0.10)))
    regular_quota = quota - skew_quota
    picked = [
        *round_robin_select(regular, regular_quota),
        *round_robin_select(skewed, skew_quota),
    ]
    return sorted(picked, key=lambda item: stable_int(f"cell-order:{facility}:{mechanism}:{item}"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/reconstructed_v3/scenario_clusters_1935.v3_6.frozen.jsonl"))
    parser.add_argument("--audit", type=Path, default=Path("data/reconstructed_v3/scenario_clusters_1935.v3_6.audit.json"))
    parser.add_argument("--report", type=Path, default=Path("data/reconstructed_v3/scenario_clusters_1935.v3_6.audit.md"))
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    sequence = 0
    for facility, quotas in MATRIX.items():
        for mechanism, quota in zip(MECHANISMS, quotas):
            if not quota:
                continue
            picked = select_cell_scenarios(facility, mechanism, quota)
            for area, control, geometry, form, crash_pattern, observation in picked:
                sequence += 1
                scenario_id = f"v3_scene_{sequence:04d}"
                family_signature = "|".join((facility, mechanism, area, control, geometry, form))
                family_id = f"v3_family_{hashlib.sha256(family_signature.encode('utf-8')).hexdigest()[:12]}"
                split_score = stable_int(family_id) % 10
                rows.append(
                    {
                        "scenario_group_id": scenario_id,
                        "split_group_id": family_id,
                        "facility_bucket": facility,
                        "primary_mechanism_bucket": mechanism,
                        "area_type": area,
                        "traffic_control": control,
                        "geometry": geometry,
                        "road_form": form,
                        "crash_pattern": crash_pattern,
                        "observable_condition": observation,
                        "split": "dev" if split_score == 0 else "train",
                        "scenario_source": {
                            "type": "independent_expert_taxonomy",
                            "inputs_used": [
                                "manuscript Figure 5 aggregate distribution",
                                "facility and crash-pattern taxonomy",
                                "observable roadway and crash circumstances",
                            ],
                            "inputs_explicitly_not_used": [
                                "CMF evidence cards",
                                "candidate retrieval results",
                                "previous query text",
                                "previous Top-K labels",
                            ],
                        },
                        "freeze_status": "scenario_frozen_before_retrieval",
                    }
                )

    write_jsonl(args.output, rows)
    signatures = {
        (row["facility_bucket"], row["primary_mechanism_bucket"], row["area_type"], row["traffic_control"], row["geometry"], row["road_form"], row["crash_pattern"], row["observable_condition"])
        for row in rows
    }
    cell_counts = Counter((row["facility_bucket"], row["primary_mechanism_bucket"]) for row in rows)
    expected = {(facility, mechanism): quota for facility, quotas in MATRIX.items() for mechanism, quota in zip(MECHANISMS, quotas) if quota}
    family_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        family_splits[row["split_group_id"]].add(row["split"])
    audit = {
        "release": "reconstructed-v3.6-independent-scenarios",
        "record_count": len(rows),
        "unique_scenario_signature_count": len(signatures),
        "duplicate_scenario_signature_count": len(rows) - len(signatures),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "split_group_count": len(family_splits),
        "split_groups_crossing_train_dev": sum(len(splits) > 1 for splits in family_splits.values()),
        "facility_counts": dict(Counter(row["facility_bucket"] for row in rows)),
        "mechanism_counts": dict(Counter(row["primary_mechanism_bucket"] for row in rows)),
        "area_counts": dict(Counter(row["area_type"] for row in rows)),
        "skewed_four_leg_count": sum(row["geometry"] == "skewed four-leg" for row in rows),
        "at_grade_intersection_count": sum(row["facility_bucket"] == "At-grade intersection" for row in rows),
        "cell_counts_match_figure5_target": dict(cell_counts) == expected,
        "provenance_policy": "No Evidence Card, retrieval result, old query, or old label was read during scenario construction.",
        "output_sha256": sha256_file(args.output),
    }
    audit["skewed_share_of_at_grade"] = round(
        audit["skewed_four_leg_count"] / audit["at_grade_intersection_count"], 4
    )
    audit["pass"] = (
        len(rows) == 1935
        and len(signatures) == len(rows)
        and audit["cell_counts_match_figure5_target"]
        and audit["split_groups_crossing_train_dev"] == 0
        and 0 < audit["skewed_share_of_at_grade"] <= 0.12
    )
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    sample = random.Random(20260823).sample(rows, 20)
    lines = [
        "# reconstructed-v3 独立场景簇审计",
        "",
        f"- 场景数：{len(rows)}",
        f"- 唯一结构化场景：{len(signatures)}",
        f"- 重复结构化场景：{len(rows) - len(signatures)}",
        f"- 相似场景家族数：{len(family_splits)}",
        f"- 跨 train/dev 的场景家族：{audit['split_groups_crossing_train_dev']}",
        f"- 与 Figure 5 设施×事故类型配额一致：{audit['cell_counts_match_figure5_target']}",
        f"- 文件 SHA-256：`{audit['output_sha256']}`",
        "- 生成阶段未读取 CMF Evidence Card、候选池、旧 Query 或旧标签。",
        "",
        "## 20 条随机场景",
        "",
        "| ID | 设施 | 事故模式 | 区域/控制/形态 | 可观察情况 |",
        "|---|---|---|---|---|",
    ]
    for row in sample:
        context = f"{row['area_type']} / {row['traffic_control']} / {row['geometry']} / {row['road_form']}"
        lines.append(f"| {row['scenario_group_id']} | {row['facility_bucket']} | {row['crash_pattern']} | {context} | {row['observable_condition']} |")
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
