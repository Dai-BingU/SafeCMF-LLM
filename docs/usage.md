# Running SafeCMF-LLM

Run commands from the repository root with Python 3.10 or newer. QLoRA training and model inference use an NVIDIA GPU with CUDA and BF16 support.

## Installation

For data preparation, retrieval, and evaluation:

```bash
python -m pip install -r requirements-core.txt
```

For model training and inference, install a CUDA-compatible PyTorch build, then:

```bash
python -m pip install -r requirements.txt
```

## Prepare the examples

```bash
python scripts/prepare_examples.py
```

This creates `out/examples/sft.jsonl` (prompt/completion pairs), `inputs.jsonl` (inference prompts), `queries.jsonl` (query text), and `labels.jsonl` (reference outputs). The prompts use the same template as the final training release.

## Train a model

```bash
python scripts/train_qlora_sft.py --model Qwen/Qwen3-14B --train-file out/examples/sft.jsonl --output-dir out/qwen3-14b
```

Set `--train-file` to your training JSONL file. Each record contains `qid`, `prompt`, and `completion`. An optional `--dev-file` enables development-set evaluation. The adapter is saved under `out/qwen3-14b/final_adapter`.

The defaults match the study's training settings: one epoch, LoRA rank 16, alpha 32, dropout 0.05, learning rate 2e-4, batch size 1, gradient accumulation 16, and seed 42. Loss is computed on completion tokens.

Use `Qwen/Qwen3-32B` or `meta-llama/Meta-Llama-3.1-8B-Instruct` with a separate output directory to train the other backbones.

## Run inference and evaluation

```bash
python scripts/infer_v318_locked_test.py --base-model Qwen/Qwen3-14B --adapter out/qwen3-14b/final_adapter --inputs out/examples/inputs.jsonl --out out/examples/predictions.jsonl
python scripts/evaluate_v318_uniform_model_performance.py --predictions out/examples/predictions.jsonl --labels out/examples/labels.jsonl --evidence-store data/evidence_store.jsonl --out out/examples/metrics.json
python scripts/backfill_recommendations.py --predictions out/examples/predictions.jsonl --out out/examples/recommendations.jsonl
```

For a base-model comparison, omit `--adapter` and add `--native-chat-no-think`. Generation is greedy; `--max-new-tokens` controls the output budget. `normalize_llama_base_ranked_outputs.py` normalizes ranked Llama outputs using the input candidate IDs.

The evaluator reports exact Top-1, exact Hit@3, mechanism Hit@3, Jaccard, empty-set decision accuracy, and output validity. Recommendation metrics use queries with nonempty reference sets; empty-set decisions are evaluated separately.

## Retrieve candidates for new queries

Input JSONL records have this format:

```json
{"qid":"example_001","user_text":"Vehicles frequently run off a rural curve at night. What improvements should be considered?"}
```

```bash
python scripts/build_candidate_inputs.py --queries out/examples/queries.jsonl --out-dir out/retrieval
```

The configuration is in [`configs/retrieval.json`](../configs/retrieval.json). The command builds a master pool and up to 30 candidate cards per query, then writes `out/retrieval/inputs.jsonl` for the inference command above. The lower-level `retrieve_candidates.py` exposes individual scoring, recall, filtering, and diversity options.

## Additional experiments

| Task | Entry points |
|---|---|
| Group-aware train/test splitting | `build_v318_final_train_test_split.py` |
| SFT release construction | `build_v318_sft_release.py` |
| Scenario and query generation | `build_independent_scenarios_v3.py`, `generate_independent_queries_v3.py` |
| Rule-based label proposals | `build_teacher_candidate_inputs.py`, `run_teacher_selection_v39.py` |
| Expert review interface | `build_expert_review_app.py` |
| Candidate/module ablations | `build_candidate_inputs.py --variant no_context`, `--variant no_mechanism`, or `--variant no_filter`; `infer_v318_candidate_ablation.py` |
| Candidate-pool size | `build_candidate_inputs.py --candidate-limit 10` (also 20, 30, 40); `analyze_v318_topn_sensitivity.py` |
| BM25 baseline | `build_bm25_rag_baseline.py`, `infer_bm25_rag_baseline.py` |
| API teacher baseline | `run_gpt55_automatic_top30_teacher_baseline.py` (`OPENAI_API_KEY`, optional `OPENAI_BASE_URL`) |
| Paraphrase robustness | `evaluate_v318_paraphrase_robustness.py` |
| Result analysis and plots | `analyze_v318_*.py`, `plot_v318_*.py`, `plot_model_performance_grouped_bar.py` |

These scripts are in [`scripts/`](../scripts/). Run an entry point with `--help` to see its input and output arguments.
