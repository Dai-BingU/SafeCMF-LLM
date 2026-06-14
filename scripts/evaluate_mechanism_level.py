#!/usr/bin/env python3
"""Summarize recommender eval results at countermeasure/mechanism level.

Exact evidence IDs are useful for debugging source-preserving examples, but
they are too strict for judging whether a recommendation is engineering-fit.
This script adds softer metrics:
  - exact countermeasure text match
  - mechanism-family match
  - broad mechanism-family match
  - high-signal bad-fit flags
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cmfrec.countermeasure_family import (
    bad_fit_flags,
    broad_mechanism_family,
    countermeasure_text_key,
    jaccard,
    mechanism_family,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_query_meta(paths: list[Path]) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            qid = str(row.get("qid") or "")
            if not qid:
                continue
            meta[qid] = {
                "source_evidence_id": str(row.get("source_evidence_id") or "") or None,
                "repair_tag": (row.get("repair") or {}).get("tag"),
                "original_qid": row.get("original_qid"),
            }
    return meta


def ids(cards: list[dict[str, Any]]) -> list[str]:
    return [str(c.get("evidence_id")) for c in cards if c.get("evidence_id") is not None]


def values(cards: list[dict[str, Any]], fn) -> list[str]:
    return [fn(c) for c in cards if fn(c)]


def top1_match(expected: list[str], actual: list[str]) -> bool:
    return bool(expected and actual and expected[0] == actual[0])


def any_overlap(expected: list[str], actual: list[str]) -> bool:
    return bool(set(expected) & set(actual))


def summarize_row(row: dict[str, Any], query_meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    expected_cards = row.get("expected_cards") or []
    adapter_cards = row.get("adapter_cards") or []
    scorer_cards = row.get("scorer_cards") or []
    user_text = str(row.get("user_text") or "")

    expected_ids = ids(expected_cards) or [str(x) for x in row.get("expected_ids") or []]
    adapter_ids = ids(adapter_cards) or [str(x) for x in row.get("adapter_ids") or []]
    scorer_ids = ids(scorer_cards) or [str(x) for x in row.get("scorer_ids") or []]

    expected_text = values(expected_cards, countermeasure_text_key)
    adapter_text = values(adapter_cards, countermeasure_text_key)
    scorer_text = values(scorer_cards, countermeasure_text_key)

    expected_family = values(expected_cards, mechanism_family)
    adapter_family = values(adapter_cards, mechanism_family)
    scorer_family = values(scorer_cards, mechanism_family)

    expected_broad = values(expected_cards, broad_mechanism_family)
    adapter_broad = values(adapter_cards, broad_mechanism_family)
    scorer_broad = values(scorer_cards, broad_mechanism_family)

    adapter_flags = [
        {
            "evidence_id": str(card.get("evidence_id")),
            "countermeasure": card.get("countermeasure"),
            "flags": bad_fit_flags(user_text, card),
        }
        for card in adapter_cards
    ]
    adapter_flags = [x for x in adapter_flags if x["flags"]]

    qid = str(row.get("qid") or "")
    meta = query_meta.get(qid) or {}
    source_eid = meta.get("source_evidence_id")
    source_strict = bool(meta.get("repair_tag") or "__repair_" in qid)

    return {
        "qid": qid,
        "user_text": user_text,
        "expected_ids": expected_ids,
        "adapter_ids": adapter_ids,
        "scorer_ids": scorer_ids,
        "exact_id_top1": top1_match(expected_ids, adapter_ids),
        "exact_id_jaccard": jaccard(expected_ids, adapter_ids),
        "text_top1": top1_match(expected_text, adapter_text),
        "text_jaccard": jaccard(expected_text, adapter_text),
        "mechanism_top1": top1_match(expected_family, adapter_family),
        "mechanism_jaccard": jaccard(expected_family, adapter_family),
        "broad_mechanism_top1": top1_match(expected_broad, adapter_broad),
        "broad_mechanism_jaccard": jaccard(expected_broad, adapter_broad),
        "scorer_mechanism_jaccard": jaccard(expected_family, scorer_family),
        "expected_mechanisms": expected_family,
        "adapter_mechanisms": adapter_family,
        "expected_broad_mechanisms": expected_broad,
        "adapter_broad_mechanisms": adapter_broad,
        "adapter_bad_fit_flags": adapter_flags,
        "source_evidence_id": source_eid,
        "source_exact_present": (source_eid in adapter_ids) if source_eid else None,
        "source_strict": source_strict,
        "source_preserving_alarm": bool(source_strict and source_eid and source_eid not in adapter_ids),
        "repair_tag": meta.get("repair_tag"),
        "expected_cards": expected_cards,
        "adapter_cards": adapter_cards,
    }


def avg(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(r.get(key) or 0.0) for r in rows) / len(rows)


def count_true(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for r in rows if r.get(key) is True)


def write_md(path: Path, rows: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.extend(["# Mechanism-Level Recommender Eval", "", "## Metrics", "", "```json"])
    lines.append(json.dumps(metrics, ensure_ascii=False, indent=2))
    lines.extend(["```", ""])

    lines.append("## Review Rows")
    lines.append("")
    for r in rows:
        flags = sum(len(x["flags"]) for x in r["adapter_bad_fit_flags"])
        lines.append(f"### {r['qid']}")
        lines.append("")
        lines.append(f"**Query:** {r['user_text']}")
        lines.append("")
        lines.append(
            "**Scores:** "
            f"exact_id_jaccard={r['exact_id_jaccard']:.3f}, "
            f"text_jaccard={r['text_jaccard']:.3f}, "
            f"mechanism_jaccard={r['mechanism_jaccard']:.3f}, "
            f"broad_jaccard={r['broad_mechanism_jaccard']:.3f}, "
            f"bad_fit_flags={flags}"
        )
        if r.get("source_evidence_id"):
            lines.append(
                f"**Source reference:** source={r['source_evidence_id']}, "
                f"strict={r['source_strict']}, present={r['source_exact_present']}"
            )
        lines.append("")
        lines.append(
            f"**Expected mechanisms:** `{', '.join(r['expected_mechanisms'])}`"
        )
        lines.append(
            f"**Adapter mechanisms:** `{', '.join(r['adapter_mechanisms'])}`"
        )
        lines.append("")
        for title, cards in [("Expected", r["expected_cards"]), ("Adapter", r["adapter_cards"])]:
            lines.append(f"**{title}**")
            lines.append("")
            lines.append("| Rank | Evidence | Mechanism | Countermeasure | CRF | Star |")
            lines.append("|---:|---|---|---|---:|---:|")
            for idx, card in enumerate(cards, start=1):
                cm = str(card.get("countermeasure") or "").replace("|", "\\|")
                lines.append(
                    f"| {idx} | {card.get('evidence_id')} | {mechanism_family(card)} | {cm} | "
                    f"{card.get('crf')} | {card.get('star')} |"
                )
            lines.append("")
        if r["adapter_bad_fit_flags"]:
            lines.append("**Bad-Fit Flags**")
            lines.append("")
            for flag_row in r["adapter_bad_fit_flags"]:
                lines.append(
                    f"- `{flag_row['evidence_id']}` {flag_row['countermeasure']}: "
                    f"`{', '.join(flag_row['flags'])}`"
                )
            lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-json", type=Path, required=True)
    ap.add_argument("--queries", type=Path, nargs="*", default=[])
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    args = ap.parse_args()

    raw_rows = read_json(args.eval_json)
    query_meta = load_query_meta(args.queries)
    rows = [summarize_row(row, query_meta) for row in raw_rows]

    metrics = {
        "n": len(rows),
        "exact_id_top1": count_true(rows, "exact_id_top1"),
        "text_top1": count_true(rows, "text_top1"),
        "mechanism_top1": count_true(rows, "mechanism_top1"),
        "broad_mechanism_top1": count_true(rows, "broad_mechanism_top1"),
        "exact_id_jaccard_avg": avg(rows, "exact_id_jaccard"),
        "text_jaccard_avg": avg(rows, "text_jaccard"),
        "mechanism_jaccard_avg": avg(rows, "mechanism_jaccard"),
        "broad_mechanism_jaccard_avg": avg(rows, "broad_mechanism_jaccard"),
        "scorer_mechanism_jaccard_avg": avg(rows, "scorer_mechanism_jaccard"),
        "rows_with_bad_fit_flags": sum(1 for r in rows if r["adapter_bad_fit_flags"]),
        "source_reference_rows": sum(1 for r in rows if r.get("source_evidence_id")),
        "source_strict_rows": sum(1 for r in rows if r.get("source_strict")),
        "source_preserving_alarms": sum(1 for r in rows if r.get("source_preserving_alarm")),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({"metrics": metrics, "rows": rows}, ensure_ascii=False, indent=2) + "\n")
    write_md(args.out_md, rows, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
