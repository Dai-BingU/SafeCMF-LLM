# Release Data

This directory documents the stable aliases for the final dataset files used in
the candidate-constrained CMF evidence reranking experiments.

The full reviewed query-label JSONL files are withheld from the public repository
during manuscript review and are intended to be released after acceptance,
subject to data redistribution requirements.

## Public Release Files

```text
reviewed_rerank_examples.jsonl
sft_top20_examples.jsonl
manifest.json
```

The full evidence source is stored locally at:

```text
data/evidence_store.jsonl
```

## Counts

| File | Records |
|---|---:|
| `reviewed_rerank_examples.jsonl` | 2,150 |
| `sft_top20_examples.jsonl` | 2,150 |

The supervised reranking dataset contains 2,150 query-label examples in total.
These counts describe the internal dataset used in the experiments; the JSONL
files are ignored by Git in the pre-acceptance public repository.

## File Meaning

- `reviewed_rerank_examples.jsonl`: one row per reviewed crash-context query.
  Each row contains the query text, target context, source evidence ID, reviewed
  Top-K reference labels, and compact metadata.
- `sft_top20_examples.jsonl`: one row per model-ready supervised fine-tuning
  example. Each row contains a `split` field, prompt, and target completion with
  the reviewed evidence IDs.
- `manifest.json`: release metadata and source-file provenance.
- Evidence IDs in labels and SFT targets refer to `data/evidence_store.jsonl`.

## Internal Source Files

The combined files were produced from these internal files:

```text
main_queries.jsonl
main_labels.jsonl
repair_queries.jsonl
repair_labels.jsonl
sft_top20_train.jsonl
sft_top20_dev.jsonl
sft_top20_test.jsonl
```

These source files are kept locally for audit/debugging but do not need to be
part of the public release.
