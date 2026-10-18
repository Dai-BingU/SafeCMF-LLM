# Example data

This directory provides **one example from the final 2,635-record training split**. The study has 2,935 queries (2,635 training / 300 test).

## Files

- [`example.jsonl`](example.jsonl): one JSON object containing the query, candidate Evidence Cards, and reference output.
- [`manifest.json`](manifest.json): source release, selection method, counts, the selected query ID, and SHA-256 checksums.
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

## Selected example

The selected record, `v3_query_0023`, describes right-angle crashes at a four-leg signalized intersection on a divided arterial in an urban area. It contains a complete query, 30 candidate Evidence Cards, and three ranked reference recommendations.

The query, candidates, and target are retained from the source training record.
