# CMF Evidence-Grounded Countermeasure Recommendation

This repository contains an evidence-grounded framework for traffic safety
countermeasure recommendation. The system converts a natural-language crash
scenario into a constrained pool of CMF evidence cards, then uses a fine-tuned
LLM reranker to select traceable evidence IDs from that pool.

The key design choice is to recommend CMF evidence cards rather than free-form
countermeasure text. Each selected evidence ID can be mapped back to a
countermeasure name, CMF/CRF value, evidence quality, applicability conditions,
and literature source.

## Framework

![Evidence-grounded countermeasure recommendation framework](assets/framework.png)

High-resolution PDF version: [assets/framework.pdf](assets/framework.pdf)

## What Is Included

- Code support for a structured CMF evidence store derived from CMF
  Clearinghouse records.
- Scripts and schemas for constructing the supervised reranking dataset with
  crash-context queries, candidate evidence pools, and reviewed Top-K evidence
  labels.
- Context-aware candidate retrieval and scoring utilities.
- Mechanism-aware recall augmentation and engineering applicability filtering.
- Data builders for candidate-constrained SFT training.
- Evaluation utilities for evidence-level, countermeasure-level, and
  mechanism-level agreement.
- Static demo builders for visualizing retrieval and reranking outputs.

## Repository Layout

```text
cmfrec/
  Core retrieval, scoring, context inference, mechanism recall, and utility code.

scripts/
  Dataset construction, retrieval, SFT formatting, evaluation, audit, and demo
  generation scripts.

data/evidence_store.jsonl
  Expected location for structured CMF evidence cards used as the evidence fact
  table.

data/release/
  Dataset schema and release metadata.

docs/
  Review notes and project documentation.

out/
  Generated review/demo webpages.
```

## Data Availability

The experimental dataset is organized around three components:

- 8,844 CMF evidence cards.
- 2,150 reviewed query-label examples.
- 1,827 training examples, 108 validation examples, and 215 test examples.

The dataset uses the following stable file names:

```text
data/evidence_store.jsonl
data/release/reviewed_rerank_examples.jsonl
data/release/sft_top20_examples.jsonl
data/release/manifest.json
```

`reviewed_rerank_examples.jsonl` contains one row per reviewed crash-context
query, including the query text, target context, source evidence ID, and reviewed
Top-K reference labels. `sft_top20_examples.jsonl` contains one row per
model-ready SFT example, including the train/dev/test split field, compact Top-20
candidate evidence pool, and JSON target output. The separate `evidence_store`
remains the evidence fact table used for citation and CMF/CRF backfill.

## Task Formulation

For each crash-context query `q`, the system constructs a candidate evidence pool
`C(q)` from the CMF evidence store. The reranker outputs an ordered list of up to
three evidence IDs:

```json
{
  "recommendations": [
    {"rank": 1, "evidence_ids": ["..."]},
    {"rank": 2, "evidence_ids": ["..."]},
    {"rank": 3, "evidence_ids": ["..."]}
  ]
}
```

The model is constrained to choose IDs from the provided candidate pool. It does
not generate new CMF values, CRF values, star ratings, or citations. Those fields
are deterministically backfilled from `data/evidence_store.jsonl`.

## Candidate Evidence Construction

Candidate construction has three main stages:

1. Context-based retrieval scoring.
2. Mechanism-aware recall augmentation.
3. Engineering applicability filtering.

The retrieval score is implemented as a multiplicative utility score:

```text
S_ret(q, e) = M(q, e)^lambda * E(e) * W(e)
```

where `M(q, e)` is the context-match score, `E(e)` is the safety-effect utility,
`W(e)` is the evidence-quality weight, and `lambda = 1.0` in the current
implementation.

Mechanism-aware augmentation supplements evidence cards associated with
improvement mechanisms implied by the query context, such as nighttime
visibility, curve delineation, lane-departure control, pedestrian crossing
treatments, bicycle facility treatments, access management, median protection,
speed management, signal visibility, left-turn operation, frontage-road
operation, and toll-plaza operation.

Engineering applicability filtering removes candidates with clear facility,
traffic-control, crash-mechanism, roadway-geometry, hidden-prerequisite, or
negative-effect conflicts.

## Quick Start

Use Python 3.10 or newer. Most retrieval and data inspection scripts rely on the
standard library plus the local `cmfrec` package. Data-dependent scripts require
the dataset files at the paths described above. When running scripts directly,
set `PYTHONPATH=.`:

```bash
cd /path/to/this/repository
PYTHONPATH=. python3 scripts/build_retrieval_demo_web.py
```

If the release data are available locally, inspect them with:

```bash
python3 - <<'PY'
import json
from pathlib import Path

for path in [
    "data/evidence_store.jsonl",
    "data/release/reviewed_rerank_examples.jsonl",
    "data/release/sft_top20_examples.jsonl",
]:
    p = Path(path)
    print(path, sum(1 for _ in p.open()))
PY
```

If the local test data and prediction files are available, build the mentor demo
webpage with:

```bash
PYTHONPATH=. python3 scripts/build_retrieval_demo_web.py
```

The generated demo is written to:

```text
out/mentor_retrieval_demo_current/index.html
```

## Training Notes

The release SFT files are already formatted for candidate-constrained supervised
fine-tuning. The training examples use compact Top-20 evidence cards to control
context length.

Model training can be conducted on GPU machines using open-source LLM backbones
and parameter-efficient fine-tuning.

For GPU training, install the appropriate CUDA-compatible PyTorch build for your
machine, then install typical SFT dependencies such as:

```text
transformers
datasets
accelerate
peft
trl
safetensors
```

## Evaluation

The evaluation framework compares model outputs against reviewed reference
labels at multiple levels:

- Evidence-level agreement: exact CMF evidence ID match.
- Countermeasure-level agreement: same or highly similar countermeasure text.
- Mechanism-level agreement: same safety-improvement mechanism.
- Engineering bad-fit rate: recommendations with clear engineering
  applicability conflicts.

LLM-assisted and independent expert reviews can also be used to score context
fit, mechanism consistency, engineering applicability, and overall usefulness on
a 1-5 scale.
