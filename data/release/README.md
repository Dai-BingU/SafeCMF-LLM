# Example data

This directory provides **100 examples sampled from the final 2,635-record training split**. No test records are included. The full study has 2,935 queries (2,635 training / 300 test); the complete query-label dataset and model weights are not included in this release.

## Files

- [`examples100.jsonl`](examples100.jsonl): one JSON object per line, including query, candidate Evidence Cards, and reference output.
- [`manifest.json`](manifest.json): source release, sampling method, counts, selected query IDs, and SHA-256 checksums.
- [`../evidence_store.jsonl`](../evidence_store.jsonl): the existing store of 8,844 CMF Evidence Cards for looking up evidence IDs.

## Record format

| Field | Meaning |
|---|---|
| `qid` | Query identifier |
| `split` | Always `train` |
| `scenario_group_id`, `split_group_id` | Group identifiers retained from the source split |
| `query_completeness`, `missingness_type` | Query completeness and the missing-information category |
| `visible_context` | Structured context available in the query |
| `input.user_text` | Natural-language crash-context query |
| `input.candidates` | Compact candidate Evidence Cards supplied to the model |
| `target.recommendations` | Zero to three ranked reference recommendations, each with `rank` and `evidence_ids` |
| `source_release` | Identifier of the final study release |

Candidate cards include the evidence ID, countermeasure, category, applicability conditions, effect estimates, and star rating. Reference evidence IDs belong to the record's candidate pool. An empty recommendation list represents abstention.

## Sampling

Sampling uses seed `20261014`, stratified by query completeness and reference recommendation count, with unique split groups. The sample contains 65 complete queries and 35 controlled incomplete queries. The counts for zero, one, two, and three reference recommendations are 20, 12, 21, and 47, respectively. Neither query IDs nor split-group IDs overlap the final test set.

The query, candidates, and target are retained from the source training records. This is a demonstration sample, not a benchmark or a complete reproduction package for the paper's reported results.
