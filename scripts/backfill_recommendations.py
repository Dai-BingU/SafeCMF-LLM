#!/usr/bin/env python3
"""Attach evidence facts to ranked model outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--evidence-store", type=Path, default=ROOT / "data/evidence_store.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    evidence = {str(row["evidence_id"]): row for row in map(json.loads, args.evidence_store.read_text(encoding="utf-8").splitlines())}
    results = []
    for line in args.predictions.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ids = [str(eid) for eid in row["predicted_evidence_ids"]]
        if len(ids) > 3 or len(ids) != len(set(ids)):
            raise ValueError(f"Invalid ranked IDs for {row['qid']}")
        recommendations = [{"rank": rank, "evidence": evidence[eid]} for rank, eid in enumerate(ids, 1)]
        results.append({"qid": row["qid"], "recommendations": recommendations})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8")
    print(f"Wrote {len(results)} recommendations to {args.out}")


if __name__ == "__main__":
    main()
