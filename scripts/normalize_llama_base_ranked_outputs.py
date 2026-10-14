#!/usr/bin/env python3
"""Normalize verbose Llama base-model outputs into ranked Evidence-ID lists.

This script is label-free. It only reads the inference inputs and raw model
outputs, extracts IDs that the model explicitly selected, verifies them against
the corresponding candidate pool, and writes a normalized prediction JSONL.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def ordered_unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def ids_from_recommendations(value: dict[str, Any]) -> list[str]:
    recommendations = value.get("recommendations")
    if not isinstance(recommendations, list):
        return []
    ranked = sorted(
        (item for item in recommendations if isinstance(item, dict)),
        key=lambda item: int(item.get("rank") or 999),
    )
    result: list[str] = []
    for item in ranked:
        evidence_ids = item.get("evidence_ids")
        if isinstance(evidence_ids, list) and evidence_ids:
            result.append(str(evidence_ids[0]))
    return result


def extract_ranked_ids(raw: str, allowed: set[str]) -> tuple[list[str], str]:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    decoder = json.JSONDecoder()
    decoded: list[dict[str, Any]] = []

    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            decoded.append(value)

    # Prefer the requested top-level schema whenever the model completed it.
    for value in decoded:
        ids = ids_from_recommendations(value)
        if ids:
            valid = [eid for eid in ordered_unique(ids) if eid in allowed]
            return valid[:3], "recommendations_json"
        if isinstance(value.get("recommendations"), list):
            return [], "empty_recommendations_json"

    # Some Llama outputs emit one JSON object per rank instead of a wrapper.
    rank_items: list[tuple[int, str]] = []
    for value in decoded:
        evidence_ids = value.get("evidence_ids")
        if not isinstance(evidence_ids, list) or not evidence_ids:
            continue
        try:
            rank = int(value.get("rank") or 999)
        except (TypeError, ValueError):
            rank = 999
        rank_items.append((rank, str(evidence_ids[0])))
    if rank_items:
        ids = [eid for _, eid in sorted(rank_items) if eid in allowed]
        return ordered_unique(ids)[:3], "rank_objects_json"

    # Another observed form is a numbered list containing the complete
    # Evidence Card object for each selected rank.
    numbered_cards = re.findall(
        r'(?im)^\s*\d+[.)]\s*\{\s*"evidence_id"\s*:\s*"?(\d+)',
        cleaned,
    )
    valid = [eid for eid in ordered_unique(numbered_cards) if eid in allowed]
    if valid:
        return valid[:3], "numbered_evidence_card_json"

    # Markdown/YAML-style ranked selections, for example
    # "1. **Evidence ID:** 123" or "1. evidence_id: 123".
    selection_markers = list(
        re.finditer(
            r"(?:here\s+are\s+the\s+selected\s+evidence\s+cards|"
            r"the\s+selected\s+cards\s+are|"
            r"selected\s+the\s+top\s+\d+\s+ranked\s+cmf\s+evidence\s+cards|"
            r"select\s+the\s+top\s+\d+\s+ranked\s+cmf\s+evidence\s+cards)\s*:",
            cleaned,
            flags=re.I,
        )
    )
    selection_text = cleaned[selection_markers[-1].end() :] if selection_markers else cleaned
    markdown_ids = re.findall(
        r"(?im)^\s*\d+[.)]\s*(?:\*\*)?Evidence[ _]ID\s*:\s*(?:\*\*)?\s*(\d+)",
        selection_text,
    )
    valid = [eid for eid in ordered_unique(markdown_ids) if eid in allowed]
    if valid:
        return valid[:3], "ranked_markdown_text"

    # Verbose base-model answers commonly state selections as
    # "(Evidence ID: 123)" before a requested JSON block.
    explicit = re.findall(
        r"\(\s*Evidence\s*ID\s*[:#]?\s*[`'\"]?(\d+)[`'\"]?\s*\)",
        cleaned,
        flags=re.I,
    )
    if not explicit:
        explicit = re.findall(
            r"(?im)^\s*(?:\d+[.)]|[-*])\s+.*?Evidence\s*ID\s*[:#]?\s*[`'\"]?(\d+)",
            cleaned,
        )
    valid = [eid for eid in ordered_unique(explicit) if eid in allowed]
    if valid:
        return valid[:3], "explicit_ranked_text"

    if re.search(
        r"\b(?:no suitable|no applicable|no recommendation|recommendations?\s*:\s*none)\b",
        cleaned,
        flags=re.I,
    ):
        return [], "explicit_empty_decision"
    return [], "unresolved"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--raw-predictions", type=Path, required=True)
    parser.add_argument(
        "--replacement-predictions",
        type=Path,
        action="append",
        default=[],
        help="Optional label-free rerun outputs that replace matching qids.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    input_rows = {str(row["qid"]): row for row in read_jsonl(args.inputs)}
    predictions_by_qid = {
        str(row["qid"]): row for row in read_jsonl(args.raw_predictions)
    }
    for replacement_path in args.replacement_predictions:
        for row in read_jsonl(replacement_path):
            predictions_by_qid[str(row["qid"])] = row
    predictions = [predictions_by_qid[qid] for qid in input_rows]
    normalized: list[dict[str, Any]] = []
    method_counts: dict[str, int] = {}
    unresolved_qids: list[str] = []

    for row in predictions:
        qid = str(row["qid"])
        input_row = input_rows[qid]
        allowed = {str(value) for value in input_row["candidate_evidence_ids"]}
        ids, method = extract_ranked_ids(str(row.get("raw_output") or ""), allowed)
        method_counts[method] = method_counts.get(method, 0) + 1
        if method == "unresolved":
            unresolved_qids.append(qid)
        updated = dict(row)
        updated.update(
            {
                "predicted_evidence_ids": ids,
                "json_error": "" if method != "unresolved" else "unresolved_ranked_output",
                "all_ids_in_frozen_candidates": all(eid in allowed for eid in ids),
                "duplicate_ids": False,
                "prediction_count": len(ids),
                "normalization_method": method,
            }
        )
        normalized.append(updated)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in normalized
        ),
        encoding="utf-8",
    )
    audit = {
        "input_rows": len(input_rows),
        "prediction_rows": len(predictions),
        "unique_qids": len({str(row["qid"]) for row in predictions}),
        "normalization_method_counts": method_counts,
        "unresolved_count": len(unresolved_qids),
        "unresolved_qids": unresolved_qids,
    }
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
