#!/usr/bin/env python3
"""Parallel builder for candidate-rerank SFT data.

This keeps the output schema identical to make_candidate_rerank_sft.py, but
parallelizes per-query candidate scoring for large top-N rebuilds.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.make_candidate_rerank_sft import (  # noqa: E402
    candidate_cards,
    completion,
    expected_ids,
    make_card,
    prompt,
    query_from_context,
    read_jsonl,
    trim_context,
    write_jsonl,
)
from scripts.demo_end_to_end_compare import _build_score_row  # noqa: E402


_EVIDENCES: list[dict[str, Any]] = []
_EV_BY_ID: dict[str, dict[str, Any]] = {}
_SCORE_ROWS: list[dict[str, str]] = []
_TOP_N = 20
_TOP_K = 3


def _init_worker(
    evidences: list[dict[str, Any]],
    score_rows: list[dict[str, str]],
    top_n: int,
    top_k: int,
) -> None:
    global _EVIDENCES, _EV_BY_ID, _SCORE_ROWS, _TOP_N, _TOP_K
    _EVIDENCES = evidences
    _EV_BY_ID = {str(ev.get("evidence_id")): ev for ev in evidences}
    _SCORE_ROWS = score_rows
    _TOP_N = int(top_n)
    _TOP_K = int(top_k)


def _build_one(item: tuple[str, dict[str, Any], dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    qid, qrow, lrow = item
    stat = {
        "input": 1,
        "records": 0,
        "all_expected_in_raw_candidates": 0,
        "top1_expected_in_raw_candidates": 0,
        "expected_forced_into_candidates": 0,
        "missing_expected_evidence": 0,
        "candidate_len": None,
    }
    user_text = str(qrow.get("user_text") or "")
    ctx = qrow.get("target_context") or {}
    exp = expected_ids(lrow, _TOP_K)
    if not exp:
        return None, stat
    if any(eid not in _EV_BY_ID for eid in exp):
        stat["missing_expected_evidence"] = 1
        return None, stat

    cards, score_by_id = candidate_cards(
        _EVIDENCES,
        _SCORE_ROWS,
        query_from_context(user_text, ctx),
        _TOP_N,
        user_text=user_text,
    )
    raw_ids = [str(c["evidence_id"]) for c in cards]
    raw_set = set(raw_ids)
    stat["top1_expected_in_raw_candidates"] = int(exp[0] in raw_set)
    stat["all_expected_in_raw_candidates"] = int(all(eid in raw_set for eid in exp))

    final_cards = list(cards)
    final_ids = set(raw_ids)
    forced = False
    for eid in exp:
        if eid in final_ids:
            continue
        final_cards.append(make_card(_EV_BY_ID[eid], score_by_id.get(eid)))
        final_ids.add(eid)
        forced = True
    stat["expected_forced_into_candidates"] = int(forced)

    max_final = _TOP_N + _TOP_K
    final_cards = final_cards[:max_final]
    stat["candidate_len"] = len(final_cards)
    stat["records"] = 1

    row = {
        "id": qid,
        "prompt": prompt(user_text, trim_context(ctx), final_cards, _TOP_K),
        "completion": completion(exp),
        "meta": {
            "expected_ids": exp,
            "raw_candidate_head": raw_ids[:10],
            "all_expected_in_raw_candidates": all(eid in raw_set for eid in exp),
            "forced_expected_into_candidates": forced,
        },
    }
    return row, stat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=Path, nargs="+", required=True)
    ap.add_argument("--labels", type=Path, nargs="+", required=True)
    ap.add_argument("--evidence-store", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summary-out", type=Path, required=True)
    ap.add_argument("--candidate-top-n", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    ap.add_argument("--chunksize", type=int, default=8)
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
    score_rows = [_build_score_row(ev) for ev in evidences]
    items = [(qid, queries[qid], labels[qid]) for qid in sorted(queries)]

    stats: dict[str, Any] = {
        "records": 0,
        "candidate_top_n": int(args.candidate_top_n),
        "top_k": int(args.top_k),
        "workers": int(args.workers),
        "all_expected_in_raw_candidates": 0,
        "top1_expected_in_raw_candidates": 0,
        "expected_forced_into_candidates": 0,
        "missing_expected_evidence": 0,
        "candidate_len_counts": {},
    }
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=int(args.workers),
        initializer=_init_worker,
        initargs=(evidences, score_rows, int(args.candidate_top_n), int(args.top_k)),
    ) as ex:
        for row, stat in ex.map(_build_one, items, chunksize=int(args.chunksize)):
            for key in [
                "all_expected_in_raw_candidates",
                "top1_expected_in_raw_candidates",
                "expected_forced_into_candidates",
                "missing_expected_evidence",
            ]:
                stats[key] += int(stat.get(key) or 0)
            if stat.get("candidate_len") is not None:
                k = str(stat["candidate_len"])
                stats["candidate_len_counts"][k] = stats["candidate_len_counts"].get(k, 0) + 1
            if row is not None:
                rows.append(row)

    rows.sort(key=lambda r: str(r["id"]))
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
