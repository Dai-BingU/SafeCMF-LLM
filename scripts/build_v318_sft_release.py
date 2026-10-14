#!/usr/bin/env python3
"""Build the frozen v3.18 Top-30 SFT release without rerunning retrieval.

The builder intentionally consumes the already-frozen Query, candidate, and
teacher-label files. It never reconstructs candidates, never injects a label
into a candidate pool, and retains zero-recommendation examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "reconstructed_v3"
DEFAULT_OUT = ROOT / "data" / "release" / "v3_18"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{lineno}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def index_unique(rows: list[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = str(row.get("qid") or "")
        if not qid:
            raise ValueError(f"Missing qid in {path}")
        if qid in out:
            raise ValueError(f"Duplicate qid {qid!r} in {path}")
        out[qid] = row
    return out


def normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def selected_ids(label: dict[str, Any]) -> list[str]:
    recommendations = ((label.get("teacher_output") or {}).get("recommendations") or [])
    ranked = sorted(recommendations, key=lambda item: int(item.get("rank") or 999))
    ids: list[str] = []
    for item in ranked:
        evidence_ids = item.get("evidence_ids") or []
        if len(evidence_ids) != 1:
            raise ValueError(
                f"Each recommendation must contain exactly one evidence ID: {label.get('qid')}"
            )
        evidence_id = str(evidence_ids[0])
        if evidence_id in ids:
            raise ValueError(f"Duplicate selected evidence ID {evidence_id}: {label.get('qid')}")
        ids.append(evidence_id)
    if len(ids) > 3:
        raise ValueError(f"Top-K exceeds 3: {label.get('qid')}")
    return ids


def compact_conditions(conditions: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "crash_type",
        "severity_kabco",
        "roadway_type",
        "area_type",
        "facility_type",
        "intersection_related",
        "traffic_control_type",
        "intersection_type",
        "intersection_geometry",
        "roadway_division_type",
        "street_type",
        "crash_time_of_day",
        "crash_weather",
        "min_speed_limit",
        "max_speed_limit",
        "speed_unit",
        "min_num_lanes",
        "max_num_lanes",
        "min_traffic_volume_non_intersection",
        "max_traffic_volume_non_intersection",
        "min_major_road_traffic_volume",
        "max_major_road_traffic_volume",
        "min_minor_road_traffic_volume",
        "max_minor_road_traffic_volume",
        "traffic_volume_unit",
    ]
    return {
        key: conditions.get(key)
        for key in keys
        if conditions.get(key) not in (None, "", [], "Not specified")
    }


def compact_card(evidence: dict[str, Any]) -> dict[str, Any]:
    effect = evidence.get("effect") or {}
    quality = evidence.get("quality") or {}
    card: dict[str, Any] = {
        "evidence_id": str(evidence["evidence_id"]),
        "countermeasure": evidence.get("countermeasure"),
    }
    if evidence.get("countermeasure_category"):
        card["category"] = evidence["countermeasure_category"]
    if evidence.get("countermeasure_subcategory"):
        card["subcategory"] = evidence["countermeasure_subcategory"]
    conditions = compact_conditions(evidence.get("conditions") or {})
    if conditions:
        card["conditions"] = conditions
    compact_effect = {
        key: effect.get(key)
        for key in ("cmf", "crf")
        if effect.get(key) not in (None, "")
    }
    if compact_effect:
        card["effect"] = compact_effect
    if quality.get("star_quality_rating") not in (None, ""):
        card["star"] = quality["star_quality_rating"]
    return card


def make_prompt(user_text: str, cards: list[dict[str, Any]]) -> str:
    payload = {"user_text": user_text, "candidates": cards, "top_k": 3}
    return (
        "You are a transportation safety engineer.\n"
        "Choose zero to three ranked CMF Evidence Cards from INPUT_JSON.candidates.\n"
        "Rules:\n"
        "- Use only information stated in user_text and the displayed Evidence Cards.\n"
        "- Output valid JSON only.\n"
        "- Select evidence_ids only from INPUT_JSON.candidates.\n"
        "- Do not invent crash mechanisms, evidence IDs, effects, ratings, or sources.\n"
        "- Reject hard facility, control, geometry, direction, or treatment conflicts.\n"
        "- A card may be selected conditionally when the crash direction is explicit and only site or implementation prerequisites are missing.\n"
        "- Do not select duplicate or near-duplicate treatments.\n"
        "- Return fewer than three recommendations, including zero, when evidence is insufficient.\n"
        "- Output shape: {\"recommendations\":[{\"rank\":1,\"evidence_ids\":[\"...\"]}]}\n"
        "INPUT_JSON:\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + "\n"
    )


def make_completion(ids: list[str]) -> str:
    return json.dumps(
        {
            "recommendations": [
                {"rank": rank, "evidence_ids": [evidence_id]}
                for rank, evidence_id in enumerate(ids, 1)
            ]
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_DATA / "train_dev_queries_2935.v3_18.open_audit_revised.final.jsonl",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=DEFAULT_DATA
        / "train_dev_candidates_2935.v3_18.open_audit_revised.final.top200_master.jsonl",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_DATA / "train_dev_teacher_labels_2935.v3_18.open_audit_revised.final.jsonl",
    )
    parser.add_argument("--evidence-store", type=Path, default=ROOT / "data" / "evidence_store.jsonl")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    query_rows = read_jsonl(args.queries)
    candidate_rows = read_jsonl(args.candidates)
    label_rows = read_jsonl(args.labels)
    evidence_rows = read_jsonl(args.evidence_store)

    queries = index_unique(query_rows, args.queries)
    candidates = index_unique(candidate_rows, args.candidates)
    labels = index_unique(label_rows, args.labels)
    evidence = {str(row["evidence_id"]): row for row in evidence_rows}

    if not (set(queries) == set(candidates) == set(labels)):
        raise ValueError(
            "QID sets differ: "
            f"queries={len(queries)}, candidates={len(candidates)}, labels={len(labels)}"
        )
    if len(queries) != 2935:
        raise ValueError(f"Expected 2,935 v3.18 records, found {len(queries)}")

    seen_text: dict[str, str] = {}
    split_groups: dict[str, set[str]] = defaultdict(set)
    reviewed_rows: list[dict[str, Any]] = []
    sft_rows: list[dict[str, Any]] = []
    topk_counts: Counter[int] = Counter()
    split_counts: Counter[str] = Counter()

    for qid in sorted(queries):
        query = queries[qid]
        candidate = candidates[qid]
        label = labels[qid]
        user_text = str(query.get("user_text") or "").strip()
        if not user_text:
            raise ValueError(f"Empty user_text: {qid}")
        if str(candidate.get("user_text") or "").strip() != user_text:
            raise ValueError(f"Query/candidate text mismatch: {qid}")

        norm = normalized_text(user_text)
        if norm in seen_text:
            raise ValueError(f"Duplicate normalized Query text: {seen_text[norm]} and {qid}")
        seen_text[norm] = qid

        split = str(query.get("split") or "")
        if split not in {"train", "dev"}:
            raise ValueError(f"Unexpected split {split!r}: {qid}")
        group = str(query.get("split_group_id") or "")
        if not group:
            raise ValueError(f"Missing split_group_id: {qid}")
        split_groups[group].add(split)

        candidate_ids = [str(value) for value in candidate.get("teacher_candidate_evidence_ids") or []]
        if not 1 <= len(candidate_ids) <= 30:
            raise ValueError(f"Teacher candidate size outside 1..30: {qid} ({len(candidate_ids)})")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"Duplicate candidate Evidence ID: {qid}")
        missing_cards = [evidence_id for evidence_id in candidate_ids if evidence_id not in evidence]
        if missing_cards:
            raise ValueError(f"Missing Evidence Cards for {qid}: {missing_cards}")

        expected_ids = selected_ids(label)
        outside = [evidence_id for evidence_id in expected_ids if evidence_id not in candidate_ids]
        if outside:
            raise ValueError(f"Selected IDs outside frozen Top-30 for {qid}: {outside}")

        cards = [compact_card(evidence[evidence_id]) for evidence_id in candidate_ids]
        prompt = make_prompt(user_text, cards)
        completion = make_completion(expected_ids)
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]

        reviewed_rows.append(
            {
                "qid": qid,
                "scenario_group_id": query.get("scenario_group_id"),
                "split_group_id": group,
                "split": split,
                "user_text": user_text,
                "visible_context": query.get("visible_context") or {},
                "information_level": query.get("information_level"),
                "query_completeness": query.get("query_completeness"),
                "candidate_evidence_ids": candidate_ids,
                "topk": (label.get("teacher_output") or {}).get("recommendations") or [],
                "label_status": label.get("label_status"),
                "teacher_provenance": label.get("teacher_provenance") or {},
            }
        )
        sft_rows.append(
            {
                "id": qid,
                "qid": qid,
                "split": split,
                "split_group_id": group,
                "prompt": prompt,
                "completion": completion,
                "messages": messages,
                "meta": {
                    "expected_ids": expected_ids,
                    "candidate_evidence_ids": candidate_ids,
                    "candidate_count": len(candidate_ids),
                    "top_k": len(expected_ids),
                    "query_completeness": query.get("query_completeness"),
                    "source_release": "v3.18-author-accepted-final-freeze",
                    "prompt_uses_visible_query_only": True,
                },
            }
        )
        topk_counts[len(expected_ids)] += 1
        split_counts[split] += 1

    leaking_groups = {group: sorted(splits) for group, splits in split_groups.items() if len(splits) > 1}
    if leaking_groups:
        raise ValueError(f"Split-group leakage detected: {len(leaking_groups)} groups")
    if split_counts != Counter({"train": 2592, "dev": 343}):
        raise ValueError(f"Unexpected split counts: {dict(split_counts)}")
    if topk_counts != Counter({0: 593, 1: 362, 2: 632, 3: 1348}):
        raise ValueError(f"Unexpected Top-K counts: {dict(topk_counts)}")

    out_dir: Path = args.out_dir
    reviewed_path = out_dir / "reviewed_rerank_examples.jsonl"
    all_sft_path = out_dir / "sft_top30_examples.jsonl"
    train_path = out_dir / "sft_top30_examples.train.jsonl"
    dev_path = out_dir / "sft_top30_examples.dev.jsonl"
    manifest_path = out_dir / "manifest.json"

    write_jsonl(reviewed_path, reviewed_rows)
    write_jsonl(all_sft_path, sft_rows)
    write_jsonl(train_path, (row for row in sft_rows if row["split"] == "train"))
    write_jsonl(dev_path, (row for row in sft_rows if row["split"] == "dev"))

    manifest = {
        "release": "v3.18-author-accepted-final-freeze",
        "records": len(sft_rows),
        "split_counts": dict(sorted(split_counts.items())),
        "topk_counts": {str(key): topk_counts[key] for key in range(4)},
        "candidate_pool": "frozen at-most-Top-30 distinct treatments",
        "zero_label_examples_retained": True,
        "candidate_retrieval_rerun": False,
        "selected_ids_forced_into_candidates": False,
        "prompt_context": "user_text plus frozen displayed Evidence Cards only",
        "locked_test_included": False,
        "source_files": {
            str(args.queries.relative_to(ROOT)): sha256(args.queries),
            str(args.candidates.relative_to(ROOT)): sha256(args.candidates),
            str(args.labels.relative_to(ROOT)): sha256(args.labels),
            str(args.evidence_store.relative_to(ROOT)): sha256(args.evidence_store),
        },
        "output_files": {
            reviewed_path.name: {"records": len(reviewed_rows), "sha256": sha256(reviewed_path)},
            all_sft_path.name: {"records": len(sft_rows), "sha256": sha256(all_sft_path)},
            train_path.name: {"records": split_counts["train"], "sha256": sha256(train_path)},
            dev_path.name: {"records": split_counts["dev"], "sha256": sha256(dev_path)},
        },
        "integrity_checks": {
            "duplicate_qids": 0,
            "duplicate_normalized_queries": 0,
            "split_group_leakage": 0,
            "selected_ids_outside_frozen_top30": 0,
            "duplicate_candidate_ids": 0,
            "topk_above_three": 0,
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
