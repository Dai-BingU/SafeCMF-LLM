# Release Data

This directory documents the stable dataset files used in the
candidate-constrained CMF evidence reranking experiments.

## Public Release Files

```text
reviewed_rerank_examples.jsonl
sft_top20_examples.jsonl
manifest.json
```

The evidence source is stored at:

```text
data/evidence_store.jsonl
```

## Counts

| File | Records |
|---|---:|
| `reviewed_rerank_examples.jsonl` | 2,150 |
| `sft_top20_examples.jsonl` | 2,150 |

The supervised reranking dataset contains 2,150 query-label examples in total.

## File Meaning

- `reviewed_rerank_examples.jsonl`: one row per reviewed crash-context query.
  Each row contains the query text, target context, source evidence ID, reviewed
  Top-K reference labels, and compact metadata.
- `sft_top20_examples.jsonl`: one row per model-ready supervised fine-tuning
  example. Each row contains a `split` field, prompt, and target completion with
  the reviewed evidence IDs.
- `manifest.json`: release metadata.
- Evidence IDs in labels and SFT targets refer to `data/evidence_store.jsonl`.
