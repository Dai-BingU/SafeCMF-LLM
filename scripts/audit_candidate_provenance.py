#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit retrieval provenance and mechanism-family coverage.")
    ap.add_argument("--candidates", required=True, help="Candidates JSONL from retrieve_candidates.py")
    ap.add_argument("--out-json", required=True, help="Output summary JSON")
    ap.add_argument("--out-md", default=None, help="Optional readable Markdown report")
    ap.add_argument("--top-k", type=int, default=20, help="Audit only first K candidate_evidence_ids per query")
    args = ap.parse_args()

    rows = 0
    candidate_lengths: list[int] = []
    qids_with_provenance = 0
    qids_with_family_slots = 0
    source_counter: Counter[str] = Counter()
    slot_counter: Counter[str] = Counter()
    top_source_counter: Counter[str] = Counter()
    missing_provenance: list[str] = []
    examples_by_slot: dict[str, list[dict[str, object]]] = defaultdict(list)

    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            qid = str(row.get("qid") or "")
            ids = [str(x) for x in (row.get("candidate_evidence_ids") or [])]
            candidate_lengths.append(len(ids))
            provenance = row.get("candidate_provenance") or {}
            if provenance:
                qids_with_provenance += 1
            elif ids:
                missing_provenance.append(qid)

            slots = row.get("family_recall_slots") or []
            if slots:
                qids_with_family_slots += 1
            for slot in slots:
                slot_counter[str(slot)] += 1

            for eid in ids[: int(args.top_k)]:
                srcs = provenance.get(str(eid)) or ["missing"]
                for src in srcs:
                    src_s = str(src)
                    source_counter[src_s] += 1
                    top_source_counter[src_s] += 1
                    if src_s.startswith("family_slot:"):
                        slot = src_s.split(":", 1)[1]
                        if len(examples_by_slot[slot]) < 5:
                            examples_by_slot[slot].append(
                                {
                                    "qid": qid,
                                    "evidence_id": eid,
                                    "rank": ids.index(eid) + 1,
                                    "user_text": row.get("user_text"),
                                }
                            )

    summary = {
        "candidates": args.candidates,
        "top_k": int(args.top_k),
        "rows": rows,
        "candidate_len_min": min(candidate_lengths) if candidate_lengths else 0,
        "candidate_len_avg": (sum(candidate_lengths) / len(candidate_lengths)) if candidate_lengths else 0,
        "candidate_len_max": max(candidate_lengths) if candidate_lengths else 0,
        "qids_with_provenance": qids_with_provenance,
        "qids_missing_provenance": len(missing_provenance),
        "qids_missing_provenance_sample": missing_provenance[:20],
        "qids_with_family_slots": qids_with_family_slots,
        "family_slot_counts": dict(slot_counter.most_common()),
        "top_candidate_source_counts": dict(top_source_counter.most_common()),
        "family_slot_examples": examples_by_slot,
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.out_md:
        os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
        lines: list[str] = []
        lines.append("# Candidate Provenance Audit\n\n")
        lines.append(f"- Candidates: `{args.candidates}`\n")
        lines.append(f"- Rows: `{rows}`\n")
        lines.append(
            f"- Candidate length min/avg/max: `{summary['candidate_len_min']}` / "
            f"`{summary['candidate_len_avg']:.1f}` / `{summary['candidate_len_max']}`\n"
        )
        lines.append(f"- Rows with provenance: `{qids_with_provenance}`\n")
        lines.append(f"- Rows with family slots: `{qids_with_family_slots}`\n\n")
        lines.append("## Top Candidate Source Counts\n\n")
        for src, count in top_source_counter.most_common(30):
            lines.append(f"- `{src}`: {count}\n")
        lines.append("\n## Family Slot Counts\n\n")
        for slot, count in slot_counter.most_common(30):
            lines.append(f"- `{slot}`: {count}\n")
        lines.append("\n## Family Slot Examples\n\n")
        for slot, examples in sorted(examples_by_slot.items()):
            lines.append(f"### `{slot}`\n")
            for ex in examples:
                lines.append(
                    f"- qid=`{ex['qid']}`, rank={ex['rank']}, evidence_id=`{ex['evidence_id']}`: "
                    f"{ex.get('user_text')}\n"
                )
            lines.append("\n")
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write("".join(lines))

    print(f"Rows: {rows}")
    print(f"Output JSON: {args.out_json}")
    if args.out_md:
        print(f"Output MD: {args.out_md}")


if __name__ == "__main__":
    main()
