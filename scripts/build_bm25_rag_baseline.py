#!/usr/bin/env python3
"""Build a plain BM25-to-Qwen RAG baseline for the final 300-query test set.

This script is deliberately independent of the proposed retrieval pipeline. It
uses only the visible query text and a text serialization of the 8,844-card CMF
Evidence Store. It does not use labels, structured query fields, mechanism
augmentation, applicability filtering, or the frozen proposed Top-30 pools.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from build_v318_sft_release import compact_card


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = ROOT / "data" / "release" / "v3_18_final_split_2635_300"
TOKEN_RE = re.compile(r"[a-z0-9]+")
MISSING_VALUES = {
    "",
    "all",
    "none",
    "not specified",
    "no values chosen.",
    "unit unknown",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{lineno}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def useful(value: Any) -> bool:
    if value is None or value == []:
        return False
    return str(value).strip().lower() not in MISSING_VALUES


def serialize_card(card: dict[str, Any]) -> str:
    """Serialize a card as ordinary searchable text without numeric outcomes."""
    parts: list[str] = []
    named_fields = (
        ("countermeasure", "countermeasure"),
        ("countermeasure_category", "category"),
        ("countermeasure_subcategory", "subcategory"),
    )
    for source_key, label in named_fields:
        value = card.get(source_key)
        if useful(value):
            parts.append(f"{label} {value}")

    conditions = card.get("conditions") or {}
    condition_fields = (
        ("crash_type", "crash type"),
        ("severity_kabco", "crash severity"),
        ("roadway_type", "roadway type"),
        ("area_type", "area type"),
        ("facility_type", "facility type"),
        ("intersection_related", "intersection related"),
        ("traffic_control_type", "traffic control"),
        ("intersection_type", "intersection type"),
        ("intersection_geometry", "intersection geometry"),
        ("roadway_division_type", "roadway division"),
        ("street_type", "street type"),
        ("crash_time_of_day", "time of day"),
        ("crash_weather", "weather"),
        ("min_speed_limit", "minimum speed limit"),
        ("max_speed_limit", "maximum speed limit"),
        ("speed_unit", "speed unit"),
        ("min_num_lanes", "minimum lanes"),
        ("max_num_lanes", "maximum lanes"),
    )
    for source_key, label in condition_fields:
        value = conditions.get(source_key)
        if useful(value):
            parts.append(f"{label} {value}")
    return ". ".join(parts)


def evidence_sort_key(evidence_id: str) -> tuple[int, str]:
    match = re.search(r"\d+", evidence_id)
    return (int(match.group()) if match else 10**12, evidence_id)


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.n = len(documents)
        self.lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_index, document in enumerate(documents):
            counts = Counter(tokens(document))
            self.lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                self.postings[term].append((doc_index, frequency))
        self.avgdl = sum(self.lengths) / max(self.n, 1)

    def search(self, query: str, top_k: int, ids: list[str]) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokens(query)):
            posting = self.postings.get(term)
            if not posting:
                continue
            df = len(posting)
            idf = math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))
            for doc_index, frequency in posting:
                length_norm = 1.0 - self.b + self.b * self.lengths[doc_index] / self.avgdl
                numerator = frequency * (self.k1 + 1.0)
                denominator = frequency + self.k1 * length_norm
                scores[doc_index] += idf * numerator / denominator

        ranked = sorted(
            scores.items(),
            key=lambda item: (-item[1], evidence_sort_key(ids[item[0]])),
        )
        if len(ranked) < top_k:
            used = {doc_index for doc_index, _ in ranked}
            for doc_index in sorted(
                (index for index in range(self.n) if index not in used),
                key=lambda index: evidence_sort_key(ids[index]),
            ):
                ranked.append((doc_index, 0.0))
                if len(ranked) == top_k:
                    break
        return ranked[:top_k]


def make_rag_prompt(user_text: str, cards: list[dict[str, Any]]) -> str:
    payload = {"query": user_text, "retrieved_evidence_cards": cards}
    return (
        "You are a transportation safety engineer answering with evidence from a retrieved CMF knowledge base.\n"
        "Recommend zero to three applicable safety countermeasures for the query.\n"
        "Use only the retrieved Evidence Cards and cite each recommendation by its evidence_id.\n"
        "Return fewer than three recommendations, including none, if the retrieved evidence is insufficient.\n"
        "Return valid JSON only in this form: "
        '{"recommendations":[{"rank":1,"evidence_ids":["..."]}]}\n'
        "Do not use duplicate IDs or IDs absent from retrieved_evidence_cards.\n"
        "INPUT_JSON:\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=DEFAULT_RELEASE / "queries.test_300.jsonl")
    parser.add_argument("--evidence-store", type=Path, default=ROOT / "data" / "evidence_store.jsonl")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RELEASE / "bm25_rag_baseline")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args()

    query_rows = read_jsonl(args.queries)
    evidence_rows = read_jsonl(args.evidence_store)
    if len(query_rows) != 300:
        raise ValueError(f"Expected 300 test queries, found {len(query_rows)}")
    if len(evidence_rows) != 8844:
        raise ValueError(f"Expected 8,844 Evidence Cards, found {len(evidence_rows)}")

    qids = [str(row.get("qid") or "") for row in query_rows]
    if not all(qids) or len(set(qids)) != len(qids):
        raise ValueError("Missing or duplicate qids")
    evidence_ids = [str(row.get("evidence_id") or "") for row in evidence_rows]
    if not all(evidence_ids) or len(set(evidence_ids)) != len(evidence_ids):
        raise ValueError("Missing or duplicate evidence IDs")

    documents = [serialize_card(row) for row in evidence_rows]
    if any(not document for document in documents):
        raise ValueError("At least one Evidence Card has an empty text serialization")
    bm25 = BM25(documents, k1=args.k1, b=args.b)

    retrieval_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    zero_score_queries = 0
    for query in query_rows:
        user_text = str(query.get("user_text") or "").strip()
        if not user_text:
            raise ValueError(f"Empty query text: {query.get('qid')}")
        ranked = bm25.search(user_text, args.top_k, evidence_ids)
        if not ranked or ranked[0][1] <= 0.0:
            zero_score_queries += 1
        retrieved_ids = [evidence_ids[index] for index, _ in ranked]
        retrieved_cards = [compact_card(evidence_rows[index]) for index, _ in ranked]
        retrieval_rows.append(
            {
                "qid": query["qid"],
                "user_text": user_text,
                "retrieval_method": "BM25_Okapi",
                "bm25_parameters": {"k1": args.k1, "b": args.b},
                "retrieved": [
                    {
                        "rank": rank,
                        "evidence_id": evidence_ids[index],
                        "score": round(score, 8),
                    }
                    for rank, (index, score) in enumerate(ranked, 1)
                ],
            }
        )
        input_rows.append(
            {
                "qid": query["qid"],
                "user_text": user_text,
                "query_completeness": query.get("query_completeness"),
                "retrieved_evidence_ids": retrieved_ids,
                "retrieved_count": len(retrieved_ids),
                "prompt": make_rag_prompt(user_text, retrieved_cards),
                "input_policy": "visible_query_plus_plain_bm25_top30_no_test_labels",
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    retrieval_path = args.out_dir / "test_300.bm25_top30.retrieval.jsonl"
    inputs_path = args.out_dir / "test_300.bm25_top30.rag_inputs.label_free.jsonl"
    manifest_path = args.out_dir / "manifest.json"
    write_jsonl(retrieval_path, retrieval_rows)
    write_jsonl(inputs_path, input_rows)
    manifest = {
        "release": "v3.18-final-bm25-rag-baseline",
        "query_count": len(query_rows),
        "evidence_card_count": len(evidence_rows),
        "top_k": args.top_k,
        "retrieval": {
            "method": "BM25_Okapi",
            "tokenization": "lowercase ASCII alphanumeric tokens",
            "k1": args.k1,
            "b": args.b,
            "document_fields": [
                "countermeasure",
                "category",
                "subcategory",
                "crash/roadway/area/facility/intersection/control/geometry/time/weather conditions",
            ],
            "mechanism_coverage_augmentation": False,
            "applicability_filtering": False,
            "structured_query_fields_used": False,
            "labels_used": False,
        },
        "generation": {
            "model": "base Qwen3-32B (not fine-tuned)",
            "output": "zero to three cited Evidence IDs",
        },
        "checks": {
            "all_queries_have_top_k_unique_ids": all(
                len(row["retrieved_evidence_ids"]) == args.top_k
                and len(set(row["retrieved_evidence_ids"])) == args.top_k
                for row in input_rows
            ),
            "zero_positive_score_queries": zero_score_queries,
        },
        "files": {
            retrieval_path.name: {"sha256": sha256(retrieval_path)},
            inputs_path.name: {"sha256": sha256(inputs_path)},
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
