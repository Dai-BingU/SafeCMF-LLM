#!/usr/bin/env python3
"""Run label-free greedy inference for frozen v3.18 locked-test prompts."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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
        [item for item in (value.get("recommendations") or []) if isinstance(item, dict)],
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
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--native-chat-no-think",
        action="store_true",
        help="Wrap each frozen prompt in the base model's native chat template with thinking disabled.",
    )
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    inputs = read_jsonl(args.inputs)
    if args.limit:
        inputs = inputs[: args.limit]
    completed = {str(row.get("qid")) for row in read_jsonl(args.out)} if args.out.exists() else set()

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer_source = args.adapter or args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        quantization_config=quant,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(base, args.adapter)
        inference_policy = "greedy_label_free_frozen_top30_same_v3_18_prompt_finetuned"
    else:
        model = base
        inference_policy = (
            "greedy_label_free_frozen_top30_native_chat_no_think_base_model"
            if args.native_chat_no_think
            else "greedy_label_free_frozen_top30_same_v3_18_prompt_base_model"
        )
    model.eval()

    for ordinal, row in enumerate(inputs, 1):
        qid = str(row["qid"])
        if qid in completed:
            continue
        started = time.perf_counter()
        prompt = str(row["prompt"])
        if args.native_chat_no_think:
            encoded = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
                return_dict=True,
            ).to("cuda")
        else:
            encoded = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        raw = tokenizer.decode(generated[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        value, json_error = extract_object(raw)
        ids = selected_ids(value)
        allowed = {str(x) for x in row["candidate_evidence_ids"]}
        result = {
            "qid": qid,
            "predicted_evidence_ids": ids,
            "json_error": json_error,
            "all_ids_in_frozen_candidates": all(eid in allowed for eid in ids),
            "duplicate_ids": len(ids) != len(set(ids)),
            "prediction_count": len(ids),
            "input_tokens": int(encoded["input_ids"].shape[1]),
            "latency_seconds": round(time.perf_counter() - started, 4),
            "raw_output": raw,
            "inference_policy": inference_policy,
        }
        append_jsonl(args.out, result)
        print(json.dumps({"row": ordinal, "total": len(inputs), "qid": qid, "ids": ids, "json_error": json_error}), flush=True)


if __name__ == "__main__":
    main()
