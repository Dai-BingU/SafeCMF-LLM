#!/usr/bin/env python3
"""Build candidate pools and model prompts from query text."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--evidence-store", type=Path, default=ROOT / "data/evidence_store.jsonl")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/retrieval.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--candidate-limit", type=int, choices=range(1, 41), default=30)
    parser.add_argument("--variant", choices=("full", "no_context", "no_mechanism", "no_filter"), default="full")
    args = parser.parse_args()
    queries = [json.loads(line) for line in args.queries.read_text(encoding="utf-8").splitlines() if line.strip()]
    qids = [str(row.get("qid") or "") for row in queries]
    if not qids or not all(qids) or len(qids) != len(set(qids)):
        raise ValueError("Queries must have unique, nonempty qids")
    if any(not str(row.get("user_text") or "").strip() for row in queries):
        raise ValueError("Each query must contain user_text")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if any(config.get(key) for key in ("use_target_context", "force_include_source_evidence", "llm_parser_base_model", "llm_parser_adapter")):
        raise ValueError("This entry point retrieves from visible query text")
    if args.variant == "no_context":
        config["disable_context_retrieval"] = True
    elif args.variant == "no_filter":
        config["disable_applicability_filtering"] = True
    elif args.variant == "no_mechanism":
        for key in list(config):
            if key.startswith("force_include_") or key.startswith("seed_") or key == "master_diversity_slots":
                config[key] = False if isinstance(config[key], bool) else 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    visible_path = args.out_dir / "queries.jsonl"
    visible_path.write_text("".join(json.dumps({"qid": row["qid"], "user_text": row["user_text"]}, ensure_ascii=False) + "\n" for row in queries), encoding="utf-8")
    command = [sys.executable, str(ROOT / "scripts/retrieve_candidates.py"),
               "--queries", str(visible_path), "--evidence-store", str(args.evidence_store),
               "--out", str(args.out_dir / "candidates.jsonl")]
    for key, value in config.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command.extend((flag, str(value)))
    subprocess.run(command, check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/build_v318_final_label_free_test_inputs.py"),
                    "--queries", str(visible_path), "--candidates", str(args.out_dir / "candidates.jsonl"),
                    "--evidence-store", str(args.evidence_store), "--out", str(args.out_dir / "inputs.jsonl"),
                    "--manifest", str(args.out_dir / "manifest.json"), "--expected-count", str(len(queries)),
                    "--candidate-limit", str(args.candidate_limit)], check=True)


if __name__ == "__main__":
    main()
