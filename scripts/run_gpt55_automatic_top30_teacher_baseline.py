#!/usr/bin/env python3
"""Run a GPT-5.5 teacher baseline on v3.18 automatic Top-30 label-free inputs.

This script reads only the label-free prompt/candidate input during generation.
It appends one JSONL record per completed query so interrupted runs can resume.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = ROOT / "data/release/v3_18_final_split_2635_300/ablation_candidates/full.top30.inputs.label_free.jsonl"
DEFAULT_OUT = (
    ROOT
    / "data/release/v3_18_final_split_2635_300/ablation_candidates/"
    / "automatic_top30_teacher_baseline.gpt55.predictions.jsonl"
)


SYSTEM_MESSAGE = (
    "You are a transportation safety engineer. Return valid JSON only. "
    "Choose zero to three ranked evidence IDs from the displayed candidates. "
    "Do not use IDs outside the candidate pool, do not repeat IDs, and keep the order by rank."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def extract_json_object(text: str) -> tuple[dict[str, Any] | None, str]:
    cleaned = re.sub(r"```(?:json)?|```", "", text, flags=re.I).strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value, ""
    return None, "no_valid_json_object"


def response_text(value: dict[str, Any]) -> str:
    if isinstance(value.get("output_text"), str):
        return value["output_text"]
    parts: list[str] = []
    for item in value.get("output") or []:
        for content in item.get("content") or []:
            if isinstance(content.get("text"), str):
                parts.append(content["text"])
    return "\n".join(parts).strip()


def selected_ids(parsed: dict[str, Any] | None, allowed: set[str]) -> tuple[list[str], list[str], bool]:
    if not parsed:
        return [], [], False
    recommendations = parsed.get("recommendations")
    if not isinstance(recommendations, list):
        return [], [], False
    ranked = sorted(
        [item for item in recommendations if isinstance(item, dict)],
        key=lambda item: int(item.get("rank") or 999),
    )
    ids: list[str] = []
    rejected: list[str] = []
    duplicate_seen = False
    for item in ranked:
        evidence_ids = item.get("evidence_ids")
        if not isinstance(evidence_ids, list) or not evidence_ids:
            continue
        eid = str(evidence_ids[0])
        if eid in ids:
            duplicate_seen = True
            rejected.append(eid)
            continue
        if eid not in allowed:
            rejected.append(eid)
            continue
        if len(ids) < 3:
            ids.append(eid)
    return ids, rejected, duplicate_seen


def call_responses_api(
    prompt: str,
    *,
    model: str,
    api_key: str,
    base_url: str,
    timeout: int,
    max_output_tokens: int,
) -> tuple[str, str]:
    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_MESSAGE}]},
            {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
        ],
        "max_output_tokens": max_output_tokens,
        "temperature": 0,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "evidence_selection",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "recommendations": {
                            "type": "array",
                            "minItems": 0,
                            "maxItems": 3,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "rank": {"type": "integer", "minimum": 1, "maximum": 3},
                                    "evidence_ids": {
                                        "type": "array",
                                        "minItems": 1,
                                        "maxItems": 1,
                                        "items": {"type": "string"},
                                    },
                                },
                                "required": ["rank", "evidence_ids"],
                            },
                        }
                    },
                    "required": ["recommendations"],
                },
                "strict": True,
            }
        },
    }
    url = base_url.rstrip("/") + "/responses"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as handle:
            raw = handle.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:1000]}") from exc
    value = json.loads(raw)
    return response_text(value), raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")

    rows = read_jsonl(args.inputs)
    if args.limit:
        rows = rows[: args.limit]
    completed = {str(row.get("qid")) for row in read_jsonl(args.out)} if args.out.exists() else set()

    for ordinal, row in enumerate(rows, 1):
        qid = str(row["qid"])
        if qid in completed:
            continue
        started = time.perf_counter()
        allowed = {str(eid) for eid in row.get("candidate_evidence_ids") or []}
        raw_output = ""
        raw_response = ""
        json_error = ""
        api_error = ""
        try:
            raw_output, raw_response = call_responses_api(
                str(row["prompt"]),
                model=args.model,
                api_key=api_key,
                base_url=args.base_url,
                timeout=args.timeout,
                max_output_tokens=args.max_output_tokens,
            )
            parsed, json_error = extract_json_object(raw_output)
        except Exception as exc:  # noqa: BLE001
            parsed = None
            json_error = "api_error"
            api_error = str(exc)
        ids, rejected_ids, duplicate_raw_ids = selected_ids(parsed, allowed)
        result = {
            "qid": qid,
            "variant": "automatic_top30_teacher_baseline",
            "model": args.model,
            "predicted_evidence_ids": ids,
            "json_error": json_error,
            "api_error": api_error,
            "all_ids_in_frozen_candidates": all(eid in allowed for eid in ids),
            "duplicate_ids": len(ids) != len(set(ids)),
            "duplicate_raw_ids_removed": duplicate_raw_ids,
            "invalid_or_extra_ids_removed": rejected_ids,
            "prediction_count": len(ids),
            "candidate_count": int(row.get("candidate_count") or len(allowed)),
            "latency_seconds": round(time.perf_counter() - started, 4),
            "raw_output": raw_output,
            "raw_response": raw_response,
            "inference_policy": (
                "gpt55_label_free_automatic_full_top30_candidate_constrained_zero_to_three_ordered"
            ),
            "input_policy": row.get("input_policy"),
        }
        append_jsonl(args.out, result)
        print(
            json.dumps(
                {
                    "row": ordinal,
                    "total": len(rows),
                    "qid": qid,
                    "ids": ids,
                    "json_error": json_error,
                    "api_error": bool(api_error),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if args.sleep:
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
