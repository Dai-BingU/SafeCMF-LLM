#!/usr/bin/env python3
"""Generate concise user queries from the frozen v3 independent scenarios."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PARENT_SHA256 = "9FB6D29CFCD70B37F779B3ECAC9BFB0F4634BCED03B7BC4E0980B68CDF3A7682"

QUESTIONS = [
    "What changes should be considered?",
    "What could help reduce the problem?",
    "How should the agency respond?",
    "What safety improvements would be appropriate?",
    "Which countermeasures should be evaluated?",
    "What practical improvements could reduce these crashes?",
    "What options should the agency examine?",
    "How could this recurring problem be addressed?",
    "What would be worth trying at this location?",
    "Which improvements make the most sense here?",
    "What should be investigated first?",
    "How might the site be made safer?",
    "What practical changes could help?",
    "Which safety measures fit this situation?",
]

FORBIDDEN = [
    "evidence-supported", "evidence based", "evidence-based", "evidence card",
    "evidence id", "cmf", "crf", "clearinghouse", "database", "retrieval",
    "rumble strip", "high friction surface", "safety edge", "guardrail",
    "roundabout", "camera", "beacon", "bike lane", "bicycle lane",
    "road diet", "raised median", "signal timing", "lighting",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text)


def normalized(text: str) -> str:
    return " ".join(word.lower() for word in words(text))


def with_article(phrase: str) -> str:
    return ("an " if phrase[:1].lower() in "aeiou" else "a ") + phrase


def site_phrase(row: dict[str, Any], rng: random.Random) -> str:
    area = row["area_type"]
    facility = row["facility_bucket"]
    control = row["traffic_control"]
    geometry = row["geometry"]
    form = row["road_form"]
    if facility == "At-grade intersection":
        road_relations = {
            "two-lane roads": (
                "where two-lane roads meet",
                "between two-lane roads",
                "formed by two-lane roads",
            ),
            "multilane arterial roads": (
                "where multilane arterial roads meet",
                "between multilane arterial roads",
                "formed by multilane arterial roads",
            ),
            "a divided arterial and a side road": (
                "where a side road meets a divided arterial",
                "between a divided arterial and a side road",
                "formed by a side road and a divided arterial",
            ),
            "collector roads": (
                "where collector roads meet",
                "between collector roads",
                "formed by collector roads",
            ),
        }[form]
        relation_index = rng.randrange(len(road_relations))
        relation = road_relations[relation_index]
        variants = [
            f"{with_article(f'{area} {geometry} {control} intersection')} {relation}",
            f"{with_article(f'{control} {geometry} intersection')} {relation} in {with_article(f'{area} area')}",
            f"{with_article(f'{area} {control} {geometry} junction')} {relation}",
            f"{with_article(f'{geometry} {control} junction')} {relation} in {with_article(f'{area} setting')}",
        ]
        return rng.choice(variants)
    if facility == "Street/corridor segment":
        variants = {
            "two-lane street": [
                with_article(f"{area} two-lane street"),
                f"a two-lane street in {with_article(f'{area} area')}",
                with_article(f"{area} two-lane corridor"),
            ],
            "multilane arterial": [
                with_article(f"{area} multilane arterial"),
                f"a multilane arterial in {with_article(f'{area} area')}",
                with_article(f"{area} arterial corridor"),
            ],
            "divided boulevard": [
                with_article(f"{area} divided boulevard"),
                f"a divided boulevard in {with_article(f'{area} area')}",
                with_article(f"{area} divided-road corridor"),
            ],
            "commercial corridor": [
                with_article(f"{area} commercial corridor"),
                f"a commercial corridor in {with_article(f'{area} area')}",
                with_article(f"{area} commercial street"),
            ],
            "residential collector": [
                with_article(f"{area} residential collector"),
                f"a residential collector in {with_article(f'{area} area')}",
                with_article(f"{area} neighborhood collector"),
            ],
        }[form]
        return rng.choice(variants)
    if facility == "Rural highway/road segment":
        variants = {
            "two-lane undivided road": [
                "a rural two-lane undivided road",
                "a segment of a rural two-lane undivided road",
                "a two-lane undivided road in a rural area",
            ],
            "multilane divided highway": [
                "a rural multilane divided highway",
                "a segment of a rural multilane divided highway",
                "a multilane divided highway in a rural area",
            ],
            "undivided four-lane road": [
                "a rural undivided four-lane road",
                "a segment of a rural undivided four-lane road",
                "an undivided four-lane road in a rural area",
            ],
            "rolling-terrain highway": [
                "a rural highway through rolling terrain",
                "a rolling-terrain highway in a rural area",
                "a segment of rural highway in rolling terrain",
            ],
            "bridge approach": [
                "a rural bridge approach",
                "the approach to a bridge on a rural road",
                "a bridge approach in a rural area",
            ],
        }[form]
        return rng.choice(variants)
    if facility == "Freeway/expressway segment":
        if geometry == "basic segment":
            return rng.choice([
                f"a basic segment of {with_article(f'{area} {form}')}",
                with_article(f"{area} {form} segment"),
                f"a basic freeway section in {with_article(f'{area} area')} on {with_article(form)}",
            ])
        geometry_display = {
            "lane-drop area": "lane-drop area",
            "bridge approach": "bridge approach",
            "freeway transition": "freeway transition",
        }[geometry]
        return rng.choice([
            f"{with_article(geometry_display)} on {with_article(f'{area} {form}')}",
            f"{with_article(f'{area} {form}')} near {with_article(geometry_display)}",
            f"{with_article(geometry_display)} along {with_article(f'{area} {form}')}",
        ])
    if facility == "Ramp/interchange":
        control_adjective = {
            "free-flow": "free-flow",
            "yield-controlled merge": "yield-controlled",
            "signal-controlled ramp terminal": "signal-controlled",
        }[control]
        control_noun = {
            "free-flow": "free-flow operation",
            "yield-controlled merge": "yield control",
            "signal-controlled ramp terminal": "signal control",
        }[control]
        return rng.choice([
            f"{with_article(f'{control_adjective} {geometry}')} within {with_article(f'{area} {form}')}",
            f"{with_article(f'{area} {geometry}')} at {with_article(form)} under {control_noun}",
            f"{with_article(geometry)} within {with_article(f'{area} {form}')} operating under {control_noun}",
        ])
    if facility == "Horizontal curve segment":
        return rng.choice([
            f"{with_article(geometry)} on {with_article(f'{area} {form}')}",
            f"{with_article(f'{area} {form}')} through {with_article(geometry)}",
            f"{with_article(geometry)} along {with_article(f'{area} {form}')}",
        ])
    if geometry == "segment":
        return rng.choice([
            with_article(f"{area} {form}"),
            f"{with_article(form)} in {with_article(f'{area} area')}",
        ])
    return rng.choice([
        f"{with_article(geometry)} on {with_article(f'{area} {form}')}",
        f"{with_article(f'{area} {form}')} near {with_article(geometry)}",
        f"{with_article(geometry)} along {with_article(f'{area} {form}')}",
    ])


def visible_key(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        row["facility_bucket"], row["primary_mechanism_bucket"], row["area_type"],
        row["traffic_control"], row["geometry"], row["road_form"], row["crash_pattern"],
    )


def assign_levels(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Assign about 35% concise queries without creating a hidden-context duplicate."""
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cell[(row["facility_bucket"], row["primary_mechanism_bucket"])].append(row)
    result: dict[str, str] = {}
    for cell, group in by_cell.items():
        target = round(len(group) * 0.35)
        used: set[tuple[str, ...]] = set()
        ordered = sorted(group, key=lambda row: stable_int(f"level:{row['scenario_group_id']}"))
        for row in ordered:
            key = visible_key(row)
            if len(used) < target and key not in used:
                result[row["scenario_group_id"]] = "concise"
                used.add(key)
            else:
                result[row["scenario_group_id"]] = "observable"
    return result


def generate(row: dict[str, Any], level: str, salt: int) -> str:
    rng = random.Random(stable_int(f"query-v3.7-natural:{row['scenario_group_id']}:{salt}"))
    site = site_phrase(row, rng)
    crashes = row["crash_pattern"]
    observation = row["observable_condition"]
    question = rng.choice(QUESTIONS)
    concise_templates = [
        "{crashes_cap} recur at {site}. {question}",
        "Recent crash records show repeated {crashes} at {site}. {question}",
        "At {site}, {crashes} remain a concern. {question}",
        "What could reduce recurring {crashes} at {site}?",
        "Several {crashes} have occurred at {site}. {question}",
        "How should recurring {crashes} at {site} be addressed?",
        "{site_cap} has an ongoing problem with {crashes}. {question}",
        "What might help with the pattern of {crashes} at {site}?",
        "Safety staff have identified recurring {crashes} at {site}. {question}",
        "How could the risk of {crashes} at {site} be reduced?",
        "There is a recurring pattern of {crashes} at {site}. {question}",
        "Which improvements could address the {crashes} occurring at {site}?",
    ]
    observable_templates = [
        "{crashes_cap} recur at {site}, where {observation}. {question}",
        "At {site}, {crashes} recur. Records indicate that {observation}. {question}",
        "Crash records show repeated {crashes} at {site}. Investigators also note that {observation}. {question}",
        "What could reduce {crashes} at {site} when {observation}?",
        "{crashes_cap} have become a concern at {site}. Since {observation}, what improvements should be examined?",
        "{site_cap} has recurring {crashes}, and {observation}. {question}",
        "How should {crashes} at {site} be addressed, given that {observation}?",
        "Reports describe recurring {crashes} at {site}. They also indicate that {observation}. {question}",
        "The main concern at {site} is recurring {crashes}. Field observations show that {observation}. {question}",
        "What changes might reduce {crashes} at {site}, especially since {observation}?",
        "A safety review of {site} found recurring {crashes} and noted that {observation}. {question}",
        "At {site}, the record includes repeated {crashes}. The available observations suggest that {observation}. {question}",
    ]
    template = rng.choice(concise_templates if level == "concise" else observable_templates)
    return template.format(
        crashes=crashes,
        crashes_cap=crashes[:1].upper() + crashes[1:],
        site=site,
        site_cap=site[:1].upper() + site[1:],
        observation=observation,
        question=question,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=Path, default=Path("data/reconstructed_v3/scenario_clusters_1935.v3_6.frozen.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/reconstructed_v3/queries_1935.v3_7.pre_retrieval.jsonl"))
    parser.add_argument("--audit", type=Path, default=Path("data/reconstructed_v3/queries_1935.v3_7.audit.json"))
    parser.add_argument("--sample", type=Path, default=Path("data/reconstructed_v3/queries_1935.v3_7.sample.md"))
    args = parser.parse_args()

    actual_sha = sha256_file(args.scenarios)
    if actual_sha != PARENT_SHA256:
        raise RuntimeError(f"Frozen scenario SHA-256 changed: {actual_sha}")
    scenarios = read_jsonl(args.scenarios)
    levels = assign_levels(scenarios)
    output: list[dict[str, Any]] = []
    used_text: set[str] = set()
    for row in scenarios:
        level = levels[row["scenario_group_id"]]
        for salt in range(100):
            text = generate(row, level, salt)
            text_length = len(words(text))
            if 10 <= text_length <= 42 and normalized(text) not in used_text:
                used_text.add(normalized(text))
                break
        else:
            raise RuntimeError(f"Could not generate unique text for {row['scenario_group_id']}")
        visible = {
            "facility_bucket": row["facility_bucket"],
            "primary_mechanism_bucket": row["primary_mechanism_bucket"],
            "area_type": row["area_type"],
            "traffic_control": row["traffic_control"],
            "geometry": row["geometry"],
            "road_form": row["road_form"],
            "crash_pattern": row["crash_pattern"],
        }
        if level == "observable":
            visible["observable_condition"] = row["observable_condition"]
        output.append(
            {
                "qid": row["scenario_group_id"].replace("scene", "query"),
                "scenario_group_id": row["scenario_group_id"],
                "split_group_id": row["split_group_id"],
                "user_text": text,
                "visible_context": visible,
                "information_level": level,
                "query_completeness": "label_critical_context_present",
                "split": row["split"],
                "target_distribution": {
                    "facility_bucket": row["facility_bucket"],
                    "primary_mechanism_bucket": row["primary_mechanism_bucket"],
                },
                "generation_provenance": {
                    "release": "reconstructed-v3.7-natural-query",
                    "frozen_scenario_sha256": PARENT_SHA256,
                    "generated_before_retrieval": True,
                    "evidence_store_used": False,
                    "old_query_or_label_used": False,
                },
            }
        )

    write_jsonl(args.output, output)
    lengths = [len(words(row["user_text"])) for row in output]
    forbidden_hits = [
        {"qid": row["qid"], "term": term}
        for row in output for term in FORBIDDEN if term in row["user_text"].lower()
    ]
    semicolon_qids = [row["qid"] for row in output if ";" in row["user_text"]]
    mechanical_joining_qids = [row["qid"] for row in output if " joining " in row["user_text"].lower()]
    visible_signatures = [tuple(sorted(row["visible_context"].items())) for row in output]
    family_splits: dict[str, set[str]] = defaultdict(set)
    for row in output:
        family_splits[row["split_group_id"]].add(row["split"])
    audit = {
        "release": "reconstructed-v3.7-natural-query",
        "record_count": len(output),
        "unique_normalized_query_count": len(used_text),
        "unique_visible_context_count": len(set(visible_signatures)),
        "information_level_counts": dict(Counter(row["information_level"] for row in output)),
        "split_counts": dict(Counter(row["split"] for row in output)),
        "split_group_count": len(family_splits),
        "split_groups_crossing_train_dev": sum(len(splits) > 1 for splits in family_splits.values()),
        "length_words": {"minimum": min(lengths), "maximum": max(lengths), "mean": round(sum(lengths) / len(lengths), 2)},
        "forbidden_treatment_or_research_cue_hits": forbidden_hits,
        "semicolon_query_count": len(semicolon_qids),
        "semicolon_qids": semicolon_qids,
        "mechanical_intersection_joining_query_count": len(mechanical_joining_qids),
        "mechanical_intersection_joining_qids": mechanical_joining_qids,
        "skewed_four_leg_query_count": sum(
            row["visible_context"].get("geometry") == "skewed four-leg" for row in output
        ),
        "scenario_sha256": actual_sha,
        "output_sha256": sha256_file(args.output),
    }
    audit["pass"] = (
        len(output) == 1935
        and len(used_text) == len(output)
        and audit["unique_visible_context_count"] == len(output)
        and min(lengths) >= 10
        and max(lengths) <= 42
        and not forbidden_hits
        and not semicolon_qids
        and not mechanical_joining_qids
        and audit["split_groups_crossing_train_dev"] == 0
    )
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    sample = random.Random(20260823).sample(output, 30)
    lines = [
        "# reconstructed-v3：检索前 Query 随机样本",
        "",
        "这些 Query 只从已冻结的独立道路场景生成，生成时尚未运行 CMF 检索。",
        "",
        "| ID | 层级 | Query |",
        "|---|---|---|",
    ]
    for row in sample:
        lines.append(f"| {row['qid']} | {row['information_level']} | {row['user_text']} |")
    args.sample.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
