#!/usr/bin/env python3
"""Train a causal-LM LoRA/QLoRA adapter on prompt/completion JSONL records.

This entry point masks prompt tokens, so loss is calculated only on the JSON
completion. It does not run retrieval and does not touch the locked test set.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local base-model path")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--dev-file", type=Path, help="Optional development set for periodic evaluation")
    parser.add_argument(
        "--no-dev-eval",
        action="store_true",
        help="Train on the full training set without loading or evaluating a separate development set.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Override epoch-based training when positive; useful for smoke tests.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--allow-truncation",
        action="store_true",
        help="Allow right-truncating prompts. Default is to fail if any record exceeds max length.",
    )
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.no_dev_eval = args.no_dev_eval or args.dev_file is None
    if args.bf16 and args.fp16:
        raise SystemExit("Choose at most one of --bf16 and --fp16")
    if not args.train_file.exists() or (not args.no_dev_eval and not args.dev_file.exists()):
        raise SystemExit(
            "Training/development file not found. Prepare prompt/completion JSONL records first."
        )

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": dtype,
    }
    if args.load_in_4bit:
        compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        model_kwargs["device_map"] = {"": local_rank}

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.config.use_cache = False
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[value.strip() for value in args.lora_target_modules.split(",") if value.strip()],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_files = {"train": str(args.train_file)}
    if not args.no_dev_eval:
        data_files["validation"] = str(args.dev_file)
    dataset = load_dataset("json", data_files=data_files)
    if args.max_train_samples is not None:
        dataset["train"] = dataset["train"].select(
            range(min(args.max_train_samples, len(dataset["train"])))
        )
    if not args.no_dev_eval and args.max_eval_samples is not None:
        dataset["validation"] = dataset["validation"].select(
            range(min(args.max_eval_samples, len(dataset["validation"])))
        )
    truncation_count = 0

    def tokenize_row(row: dict[str, Any]) -> dict[str, list[int]]:
        nonlocal truncation_count
        prompt = str(row["prompt"])
        completion = str(row["completion"])
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        completion_ids = tokenizer(
            completion + (tokenizer.eos_token or ""), add_special_tokens=False
        )["input_ids"]
        if len(completion_ids) >= args.max_seq_length:
            raise ValueError(f"Completion alone exceeds max length for {row.get('qid')}")
        available_prompt = args.max_seq_length - len(completion_ids)
        if len(prompt_ids) > available_prompt:
            if not args.allow_truncation:
                raise ValueError(
                    f"{row.get('qid')} needs {len(prompt_ids) + len(completion_ids)} tokens; "
                    f"increase --max-seq-length above {args.max_seq_length}."
                )
            prompt_ids = prompt_ids[:available_prompt]
            truncation_count += 1
        input_ids = prompt_ids + completion_ids
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": [-100] * len(prompt_ids) + completion_ids,
        }

    tokenized = dataset.map(
        tokenize_row,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing prompt/completion records",
    )
    if truncation_count:
        print(f"WARNING: truncated prompts: {truncation_count}")

    kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "logging_strategy": "steps",
        "save_strategy": "steps",
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "report_to": [] if args.report_to == "none" else [args.report_to],
        "remove_unused_columns": False,
        "seed": args.seed,
        "data_seed": args.seed,
        "optim": "paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
    }
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "warmup_ratio" in parameters:
        kwargs["warmup_ratio"] = args.warmup_ratio
    else:
        # Transformers 5 accepts a fractional value through warmup_steps.
        kwargs["warmup_steps"] = args.warmup_ratio
    eval_strategy = "no" if args.no_dev_eval else "steps"
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = eval_strategy
    else:
        kwargs["evaluation_strategy"] = eval_strategy
    training_args = TrainingArguments(**kwargs)

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized["train"],
        "data_collator": collator,
    }
    if not args.no_dev_eval:
        trainer_kwargs["eval_dataset"] = tokenized["validation"]
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir / "final_adapter"))
    tokenizer.save_pretrained(str(args.output_dir / "final_adapter"))
    if not args.no_dev_eval and not args.skip_final_eval:
        metrics = trainer.evaluate()
        (args.output_dir / "final_eval_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
