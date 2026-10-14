#!/usr/bin/env python3
"""Build final label-free Top-30 test prompts without opening reference labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_v318_sft_release import compact_card, make_prompt
from cmfrec.countermeasure_family import countermeasure_text_key


DEFAULT_RELEASE = ROOT / "data" / "release" / "v3_18_final_split_2635_300"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def index_unique(rows: list[dict[str, Any]], key: str, source: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key) or "")
        if not value or value in result:
            raise ValueError(f"Missing or duplicate {key}={value!r} in {source}")
        result[value] = row
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=DEFAULT_RELEASE / "queries.test_300.jsonl")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_RELEASE / "candidates.test_300.top200_master.jsonl")
    parser.add_argument("--evidence-store", type=Path, default=ROOT / "data" / "evidence_store.jsonl")
    parser.add_argument("--out", type=Path, default=DEFAULT_RELEASE / "test_300.top30.inputs.label_free.jsonl")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_RELEASE / "test_300.top30.inputs.label_free.manifest.json")
    parser.add_argument("--candidate-limit", type=int, default=30, choices=range(1, 41))
    parser.add_argument(
        "--expected-count",
        type=int,
        default=300,
        help="Required number of query/candidate rows; set explicitly for robustness subsets.",
    )
    args = parser.parse_args()

    queries = index_unique(read_jsonl(args.queries), "qid", args.queries)
    candidates = index_unique(read_jsonl(args.candidates), "qid", args.candidates)
    evidence = index_unique(read_jsonl(args.evidence_store), "evidence_id", args.evidence_store)
    if set(queries) != set(candidates) or len(queries) != int(args.expected_count):
        raise ValueError(
            f"Expected identical {args.expected_count}-QID query/candidate sets: "
            f"{len(queries)}/{len(candidates)}"
        )

    rows: list[dict[str, Any]] = []
    for qid in sorted(queries):
        query = queries[qid]
        candidate = candidates[qid]
        teacher_ids = [str(value) for value in candidate.get("teacher_candidate_evidence_ids") or []]
        ids = teacher_ids[: int(args.candidate_limit)]
        if int(args.candidate_limit) > len(ids):
            seen_treatments = {
                countermeasure_text_key(evidence[evidence_id]) for evidence_id in ids if evidence_id in evidence
            }
            for value in candidate.get("candidate_evidence_ids") or []:
                evidence_id = str(value)
                if evidence_id in ids or evidence_id not in evidence:
                    continue
                treatment = countermeasure_text_key(evidence[evidence_id])
                if treatment and treatment in seen_treatments:
                    continue
                ids.append(evidence_id)
                if treatment:
                    seen_treatments.add(treatment)
                if len(ids) >= int(args.candidate_limit):
                    break
        if not 1 <= len(ids) <= int(args.candidate_limit) or len(ids) != len(set(ids)):
            raise ValueError(f"Invalid frozen Top-{args.candidate_limit} candidates for {qid}: {len(ids)}")
        missing = [evidence_id for evidence_id in ids if evidence_id not in evidence]
        if missing:
            raise ValueError(f"Unknown Evidence IDs for {qid}: {missing}")
        user_text = str(query.get("user_text") or "").strip()
        rows.append(
            {
                "qid": qid,
                "user_text": user_text,
                "query_completeness": query.get("query_completeness"),
                "candidate_evidence_ids": ids,
                "candidate_count": len(ids),
                "prompt": make_prompt(user_text, [compact_card(evidence[evidence_id]) for evidence_id in ids]),
                "input_policy": f"visible_query_and_frozen_top{args.candidate_limit}_only_no_test_labels",
            }
        )

    write_jsonl(args.out, rows)
    manifest = {
        "release": f"v3.18-label-free-{args.expected_count}-query-top{args.candidate_limit}",
        "rows": len(rows),
        "labels_read": False,
        "prompt_matches_v3_18_training": True,
        "candidate_maximum": int(args.candidate_limit),
        "candidate_count_min": min(row["candidate_count"] for row in rows),
        "candidate_count_max": max(row["candidate_count"] for row in rows),
        "query_completeness": dict(sorted(__import__("collections").Counter(str(row.get("query_completeness")) for row in rows).items())),
        "inputs": {
            "queries": {"path": str(args.queries), "sha256": sha256(args.queries)},
            "candidates": {"path": str(args.candidates), "sha256": sha256(args.candidates)},
            "evidence_store": {"path": str(args.evidence_store), "sha256": sha256(args.evidence_store)},
        },
        "output": {"path": str(args.out), "sha256": sha256(args.out)},
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
