#!/usr/bin/env python3
"""Run resumable label-free generation for the frozen BM25 RAG inputs."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def extract_object(text: str) -> tuple[dict[str, Any] | None, str]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
            if isinstance(value, dict):
                return value, ""
        except json.JSONDecodeError:
            pass
    return None, "no_valid_json_object"


def selected_ids(value: dict[str, Any] | None) -> list[str]:
    if not value:
        return []
    recommendations = [
        item for item in (value.get("recommendations") or []) if isinstance(item, dict)
    ]
    ranked = sorted(recommendations, key=lambda item: int(item.get("rank") or 999))
    result: list[str] = []
    for item in ranked[:3]:
        ids = item.get("evidence_ids") or []
        if ids:
            result.append(str(ids[0]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    inputs = read_jsonl(args.inputs)
    if args.limit:
        inputs = inputs[: args.limit]
    completed_rows = read_jsonl(args.out)
    completed = {str(row.get("qid")) for row in completed_rows}

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()

    for ordinal, row in enumerate(inputs, 1):
        qid = str(row["qid"])
        if qid in completed:
            continue
        started = time.perf_counter()
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": str(row["prompt"])}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        ).to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        raw = tokenizer.decode(
            generated[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        value, json_error = extract_object(raw)
        ids = selected_ids(value)
        allowed = {str(evidence_id) for evidence_id in row["retrieved_evidence_ids"]}
        result = {
            "qid": qid,
            "predicted_evidence_ids": ids,
            "json_error": json_error,
            "all_ids_in_frozen_candidates": all(evidence_id in allowed for evidence_id in ids),
            "duplicate_ids": len(ids) != len(set(ids)),
            "prediction_count": len(ids),
            "input_tokens": int(encoded["input_ids"].shape[1]),
            "latency_seconds": round(time.perf_counter() - started, 4),
            "raw_output": raw,
            "inference_policy": "base_qwen3_32b_native_chat_greedy_bm25_top30_rag_label_free",
        }
        append_jsonl(args.out, result)
        print(
            json.dumps(
                {
                    "row": ordinal,
                    "total": len(inputs),
                    "qid": qid,
                    "ids": ids,
                    "json_error": json_error,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
