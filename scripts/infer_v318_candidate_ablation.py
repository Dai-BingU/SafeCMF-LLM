#!/usr/bin/env python3
"""Run four v3.18 candidate-module ablations while loading the ranker once."""

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
    ranked = sorted(
        [item for item in value.get("recommendations") or [] if isinstance(item, dict)],
        key=lambda item: int(item.get("rank") or 999),
    )
    result: list[str] = []
    for item in ranked[:3]:
        ids = item.get("evidence_ids") or []
        if ids:
            result.append(str(ids[0]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="NAME=INPUT_JSONL=OUTPUT_JSONL; repeat for each candidate ablation",
    )
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    variants: list[tuple[str, Path, Path]] = []
    for spec in args.variant:
        parts = spec.split("=", 2)
        if len(parts) != 3 or not all(parts):
            raise ValueError(f"Invalid --variant {spec!r}")
        variants.append((parts[0], Path(parts[1]), Path(parts[2])))

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        quantization_config=quant,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    for variant, input_path, output_path in variants:
        inputs = read_jsonl(input_path)
        completed = {str(row.get("qid")) for row in read_jsonl(output_path)}
        for ordinal, row in enumerate(inputs, 1):
            qid = str(row["qid"])
            if qid in completed:
                continue
            started = time.perf_counter()
            encoded = tokenizer(str(row["prompt"]), return_tensors="pt").to("cuda")
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
            allowed = {str(candidate_id) for candidate_id in row["candidate_evidence_ids"]}
            append_jsonl(
                output_path,
                {
                    "qid": qid,
                    "variant": variant,
                    "predicted_evidence_ids": ids,
                    "json_error": json_error,
                    "all_ids_in_frozen_candidates": all(eid in allowed for eid in ids),
                    "duplicate_ids": len(ids) != len(set(ids)),
                    "prediction_count": len(ids),
                    "input_tokens": int(encoded["input_ids"].shape[1]),
                    "latency_seconds": round(time.perf_counter() - started, 4),
                    "raw_output": raw,
                    "inference_policy": "greedy_label_free_top30_same_ranker_candidate_module_ablation",
                },
            )
            print(
                json.dumps(
                    {"variant": variant, "row": ordinal, "total": len(inputs), "qid": qid, "ids": ids},
                    ensure_ascii=False,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
