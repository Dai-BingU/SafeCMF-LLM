#!/usr/bin/env python3
"""Audit candidate recall and engineering conflicts for v3.18 module ablations."""

from __future__ import annotations

import argparse
import json
import sys
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
from cmfrec.free_text import infer_query_from_text  # noqa: E402
from cmfrec.mechanism_recall import (  # noqa: E402
    candidate_context_mismatch,
    hidden_precondition_mismatch,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def reference_groups(row: dict[str, Any]) -> list[list[str]]:
    ranked = sorted(
        row.get("topk") or (row.get("teacher_output") or {}).get("recommendations") or [],
        key=lambda item: int(item.get("rank") or 999),
    )
    return [[str(value) for value in item.get("evidence_ids") or []] for item in ranked]


def projected_set(
    ids: list[str], evidence: dict[str, dict[str, Any]], fn: Callable[[dict[str, Any]], str] | None
) -> set[str]:
    if fn is None:
        return set(ids)
    return {value for eid in ids if eid in evidence and (value := fn(evidence[eid]))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--evidence-store", type=Path, required=True)
    parser.add_argument("--variant", action="append", required=True, help="NAME=PATH; repeat for each variant")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    labels = {str(row["qid"]): row for row in read_jsonl(args.labels)}
    evidence = {str(row["evidence_id"]): row for row in read_jsonl(args.evidence_store)}
    variants: dict[str, Path] = {}
    for spec in args.variant:
        name, sep, raw_path = spec.partition("=")
        if not sep or not name or not raw_path:
            raise ValueError(f"Invalid --variant {spec!r}; expected NAME=PATH")
        variants[name] = Path(raw_path)

    levels: dict[str, Callable[[dict[str, Any]], str] | None] = {
        "exact": None,
        "treatment": countermeasure_text_key,
        "mechanism": mechanism_family,
        "broad_mechanism": broad_mechanism_family,
    }
    result: dict[str, Any] = {
        "test_queries": len(labels),
        "nonempty_queries": sum(bool(reference_groups(row)) for row in labels.values()),
        "empty_reference_queries": sum(not reference_groups(row) for row in labels.values()),
        "variants": {},
    }

    for name, path in variants.items():
        rows = {str(row["qid"]): row for row in read_jsonl(path)}
        if set(rows) != set(labels):
            raise ValueError(f"QID mismatch for {name}: candidates={len(rows)} labels={len(labels)}")

        hits = {level: 0 for level in levels}
        groups_total = 0
        all_hits = {level: 0 for level in levels}
        nonempty_total = 0
        conflict_cards = 0
        cards_total = 0
        queries_with_conflict = 0
        candidate_counts: list[int] = []

        for qid, label in labels.items():
            row = rows[qid]
            candidate_ids = [str(value) for value in row.get("teacher_candidate_evidence_ids") or []][:30]
            candidate_counts.append(len(candidate_ids))
            groups = reference_groups(label)
            if groups:
                nonempty_total += 1
                per_query = {level: [] for level in levels}
                for group in groups:
                    groups_total += 1
                    for level, fn in levels.items():
                        gold = projected_set(group, evidence, fn)
                        retrieved = projected_set(candidate_ids, evidence, fn)
                        matched = bool(gold & retrieved)
                        hits[level] += matched
                        per_query[level].append(matched)
                for level in levels:
                    all_hits[level] += all(per_query[level])

            user_text = str(row.get("user_text") or "")
            query = infer_query_from_text(user_text).query
            row_conflicts = 0
            for evidence_id in candidate_ids:
                card = evidence[evidence_id]
                conflict = hidden_precondition_mismatch(card, user_text) or candidate_context_mismatch(
                    card, user_text, query
                )
                row_conflicts += bool(conflict)
            conflict_cards += row_conflicts
            cards_total += len(candidate_ids)
            queries_with_conflict += row_conflicts > 0

        result["variants"][name] = {
            "candidate_count_min": min(candidate_counts),
            "candidate_count_max": max(candidate_counts),
            "candidate_count_mean": sum(candidate_counts) / len(candidate_counts),
            "reference_groups": groups_total,
            "group_recall_at_30": {level: hits[level] / groups_total for level in levels},
            "all_reference_groups_recalled_at_30": {
                level: all_hits[level] / nonempty_total for level in levels
            },
            "engineering_conflict_card_rate": conflict_cards / cards_total,
            "queries_with_engineering_conflict_rate": queries_with_conflict / len(labels),
            "engineering_conflict_cards": conflict_cards,
            "candidate_cards": cards_total,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
