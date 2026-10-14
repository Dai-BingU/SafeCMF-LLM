#!/usr/bin/env python3
"""Convert public examples into training, inference, query, and label files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_v318_sft_release import make_prompt, read_jsonl, write_jsonl


def convert(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    outputs = {name: [] for name in ("sft", "inputs", "labels", "queries")}
    seen = set()
    for row in rows:
        qid = str(row.get("qid") or "")
        if not qid or qid in seen:
            raise ValueError(f"Missing or duplicate qid: {qid!r}")
        seen.add(qid)
        user_text = str(row["input"]["user_text"]).strip()
        if not user_text:
            raise ValueError(f"Empty query: {qid}")
        cards = row["input"]["candidates"]
        ids = [str(card["evidence_id"]) for card in cards]
        if not all(ids) or len(ids) != len(set(ids)):
            raise ValueError(f"Invalid candidate IDs: {qid}")
        target = row["target"]["recommendations"]
        if len(target) > 3 or [t["rank"] for t in target] != list(range(1, len(target) + 1)):
            raise ValueError(f"Invalid recommendation ranks: {qid}")
        selected = [str(eid) for item in target for eid in item["evidence_ids"]]
        if any(len(item["evidence_ids"]) != 1 for item in target):
            raise ValueError(f"Expected one evidence ID per rank: {qid}")
        if len(selected) != len(set(selected)) or not set(selected) <= set(ids):
            raise ValueError(f"Reference IDs are duplicated or outside the candidate pool: {qid}")
        prompt = make_prompt(user_text, cards)
        outputs["sft"].append({"qid": qid, "prompt": prompt, "completion": json.dumps(
            {"recommendations": target}, ensure_ascii=False, separators=(",", ":"))})
        outputs["inputs"].append({"qid": qid, "user_text": user_text, "prompt": prompt,
                                  "candidate_evidence_ids": ids, "candidate_count": len(ids)})
        outputs["labels"].append({"qid": qid, "topk": target})
        outputs["queries"].append({"qid": qid, "user_text": user_text})
    if not rows:
        raise ValueError("No examples found")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, default=ROOT / "data/release/examples100.jsonl")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "out/examples")
    args = parser.parse_args()
    outputs = convert(read_jsonl(args.examples))
    for name, rows in outputs.items():
        write_jsonl(args.out_dir / f"{name}.jsonl", rows)
    print(json.dumps({"examples": len(outputs["sft"]), "output_directory": str(args.out_dir)}))


if __name__ == "__main__":
    main()
