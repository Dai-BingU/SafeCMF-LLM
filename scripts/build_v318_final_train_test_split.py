#!/usr/bin/env python3
"""Build the final group-stratified 2,635/300 split from the frozen v3.18 corpus."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_v318_sft_release import compact_card, make_completion, make_prompt


DATA = ROOT / "data" / "reconstructed_v3"
DEFAULT_QUERIES = DATA / "train_dev_queries_2935.v3_18.open_audit_revised.final.jsonl"
DEFAULT_CANDIDATES = DATA / "train_dev_candidates_2935.v3_18.open_audit_revised.final.top200_master.jsonl"
DEFAULT_LABELS = DATA / "train_dev_teacher_labels_2935.v3_18.open_audit_revised.final.jsonl"
DEFAULT_EVIDENCE = ROOT / "data" / "evidence_store.jsonl"
DEFAULT_OUT = ROOT / "data" / "release" / "v3_18_final_split_2635_300"

RELEASE_ID = "v3.18-final-group-stratified-2635-300-seed20260828"
TEST_BASE = 200
TEST_INCOMPLETE = 100
MISSINGNESS_TARGETS = {
    "crash_detail_missing": 35,
    "facility_detail_missing": 30,
    "control_missing": 20,
    "mixed_missing": 15,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def index_unique(rows: list[dict[str, Any]], key: str, source: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key) or "")
        if not value or value in result:
            raise ValueError(f"Missing or duplicate {key}={value!r} in {source}")
        result[value] = row
    return result


def is_incomplete(row: dict[str, Any]) -> bool:
    return str(row.get("query_completeness") or "") == "controlled_incomplete"


def context_value(row: dict[str, Any], key: str) -> str:
    value = (row.get("visible_context") or {}).get(key)
    text = str(value or "not stated").strip().lower()
    return text or "not stated"


def selected_ids(label: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for recommendation in (label.get("teacher_output") or {}).get("recommendations") or []:
        for evidence_id in recommendation.get("evidence_ids") or []:
            value = str(evidence_id)
            if value not in result:
                result.append(value)
    return result


def add_exact_constraint(
    rows: list[np.ndarray], lower: list[float], upper: list[float], coefficients: np.ndarray, target: int
) -> None:
    rows.append(coefficients)
    lower.append(float(target))
    upper.append(float(target))


def solve_split(queries: list[dict[str, Any]], seed: int) -> tuple[set[str], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in queries:
        group = str(row.get("split_group_id") or "")
        if not group:
            raise ValueError(f"Missing split_group_id: {row.get('qid')}")
        groups[group].append(row)
    group_ids = sorted(groups)
    n_groups = len(group_ids)

    base_counts = np.array([sum(not is_incomplete(row) for row in groups[group]) for group in group_ids], dtype=float)
    incomplete_counts = np.array([sum(is_incomplete(row) for row in groups[group]) for group in group_ids], dtype=float)

    # Distribution objectives are based only on visible Query attributes. Labels are not used to choose the test set.
    specs: list[tuple[str, Callable[[dict[str, Any]], bool], Callable[[dict[str, Any]], str], int, int]] = []
    for query_kind, predicate, sample_size, population_size in [
        ("base", lambda row: not is_incomplete(row), TEST_BASE, 1935),
        ("incomplete", is_incomplete, TEST_INCOMPLETE, 1000),
    ]:
        for key in ("facility_bucket", "primary_mechanism_bucket", "area_type", "traffic_control"):
            specs.append((f"{query_kind}:{key}", predicate, lambda row, k=key: context_value(row, k), sample_size, population_size))

    deviation_specs: list[tuple[str, np.ndarray, float, float]] = []
    for dimension, predicate, key_fn, sample_size, population_size in specs:
        categories = sorted({key_fn(row) for row in queries if predicate(row)})
        dimension_weight = 1.0 / max(1, len(categories))
        for category in categories:
            coefficients = np.array(
                [sum(predicate(row) and key_fn(row) == category for row in groups[group]) for group in group_ids],
                dtype=float,
            )
            population_count = int(sum(coefficients))
            target = population_count * sample_size / population_size
            cost = dimension_weight / max(1.0, target)
            deviation_specs.append((f"{dimension}:{category}", coefficients, target, cost))

    n_deviations = len(deviation_specs)
    n_variables = n_groups + n_deviations
    objective = np.zeros(n_variables, dtype=float)
    rng = random.Random(seed)
    for index, group in enumerate(group_ids):
        size = len(groups[group])
        objective[index] = 0.00002 * max(0, size - 1) ** 2 + rng.random() * 0.000001
    for offset, (_, _, _, cost) in enumerate(deviation_specs):
        objective[n_groups + offset] = cost

    constraint_rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []
    add_exact_constraint(
        constraint_rows,
        lower,
        upper,
        np.pad(base_counts, (0, n_deviations)),
        TEST_BASE,
    )
    add_exact_constraint(
        constraint_rows,
        lower,
        upper,
        np.pad(incomplete_counts, (0, n_deviations)),
        TEST_INCOMPLETE,
    )
    for missingness, target in MISSINGNESS_TARGETS.items():
        coefficients = np.array(
            [
                sum(is_incomplete(row) and str(row.get("missingness_type") or "") == missingness for row in groups[group])
                for group in group_ids
            ],
            dtype=float,
        )
        add_exact_constraint(
            constraint_rows,
            lower,
            upper,
            np.pad(coefficients, (0, n_deviations)),
            target,
        )

    for offset, (_, coefficients, target, _) in enumerate(deviation_specs):
        positive = np.zeros(n_variables, dtype=float)
        positive[:n_groups] = coefficients
        positive[n_groups + offset] = -1.0
        constraint_rows.append(positive)
        lower.append(-math.inf)
        upper.append(target)

        negative = np.zeros(n_variables, dtype=float)
        negative[:n_groups] = -coefficients
        negative[n_groups + offset] = -1.0
        constraint_rows.append(negative)
        lower.append(-math.inf)
        upper.append(-target)

    bounds_lower = np.zeros(n_variables, dtype=float)
    bounds_upper = np.concatenate([np.ones(n_groups, dtype=float), np.full(n_deviations, np.inf)])
    integrality = np.concatenate([np.ones(n_groups, dtype=int), np.zeros(n_deviations, dtype=int)])
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(bounds_lower, bounds_upper),
        constraints=LinearConstraint(csr_matrix(np.vstack(constraint_rows)), np.array(lower), np.array(upper)),
        options={"presolve": True, "time_limit": 300.0, "mip_rel_gap": 0.0},
    )
    if result.x is None or result.status not in {0, 1}:
        raise RuntimeError(f"MILP split failed: status={result.status}, message={result.message}")
    selected_groups = {group_ids[index] for index in range(n_groups) if result.x[index] > 0.5}
    selected_rows = [row for group in selected_groups for row in groups[group]]
    if sum(not is_incomplete(row) for row in selected_rows) != TEST_BASE:
        raise ValueError("Final test base count is not 200")
    if sum(is_incomplete(row) for row in selected_rows) != TEST_INCOMPLETE:
        raise ValueError("Final test incomplete count is not 100")
    actual_missingness = Counter(str(row.get("missingness_type") or "") for row in selected_rows if is_incomplete(row))
    if actual_missingness != Counter(MISSINGNESS_TARGETS):
        raise ValueError(f"Missingness quotas differ: {dict(actual_missingness)}")

    solver_audit = {
        "status": int(result.status),
        "message": str(result.message),
        "objective": float(result.fun),
        "mip_gap": None if getattr(result, "mip_gap", None) is None else float(result.mip_gap),
        "groups_total": n_groups,
        "groups_selected": len(selected_groups),
        "largest_selected_group": max(len(groups[group]) for group in selected_groups),
        "selection_attributes": ["query_completeness", "missingness_type", "facility_bucket", "primary_mechanism_bucket", "area_type", "traffic_control"],
        "labels_used_for_selection": False,
    }
    return selected_groups, solver_audit


def distribution(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    if key == "query_type":
        values = ["incomplete" if is_incomplete(row) else "base" for row in rows]
    elif key == "missingness_type":
        values = [str(row.get(key) or "not applicable") for row in rows]
    else:
        values = [context_value(row, key) for row in rows]
    return dict(sorted(Counter(values).items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--evidence-store", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()

    query_rows = read_jsonl(args.queries)
    candidate_rows = read_jsonl(args.candidates)
    label_rows = read_jsonl(args.labels)
    evidence_rows = read_jsonl(args.evidence_store)
    queries = index_unique(query_rows, "qid", args.queries)
    candidates = index_unique(candidate_rows, "qid", args.candidates)
    labels = index_unique(label_rows, "qid", args.labels)
    evidence = index_unique(evidence_rows, "evidence_id", args.evidence_store)
    if not (set(queries) == set(candidates) == set(labels)) or len(queries) != 2935:
        raise ValueError(f"Source QID mismatch: queries={len(queries)}, candidates={len(candidates)}, labels={len(labels)}")

    scenario_to_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in query_rows:
        scenario_to_rows[str(row.get("scenario_group_id") or "")].append(row)
    for row in query_rows:
        if not is_incomplete(row):
            continue
        parent = str(row.get("variant_of_qid") or "")
        if parent not in queries or is_incomplete(queries[parent]):
            raise ValueError(f"Invalid incomplete-query parent for {row['qid']}: {parent}")
        if row.get("scenario_group_id") != queries[parent].get("scenario_group_id"):
            raise ValueError(f"Scenario mismatch for pair {parent}/{row['qid']}")
        if row.get("split_group_id") != queries[parent].get("split_group_id"):
            raise ValueError(f"Split-group mismatch for pair {parent}/{row['qid']}")

    selected_groups, solver_audit = solve_split(query_rows, args.seed)
    split_by_qid = {
        qid: ("test" if str(row.get("split_group_id")) in selected_groups else "train")
        for qid, row in queries.items()
    }
    train_qids = sorted(qid for qid, split in split_by_qid.items() if split == "train")
    test_qids = sorted(qid for qid, split in split_by_qid.items() if split == "test")
    if len(train_qids) != 2635 or len(test_qids) != 300:
        raise ValueError(f"Unexpected split sizes: train={len(train_qids)}, test={len(test_qids)}")

    for field in ("scenario_group_id", "split_group_id"):
        membership: dict[str, set[str]] = defaultdict(set)
        for qid, row in queries.items():
            membership[str(row.get(field) or "")].add(split_by_qid[qid])
        leaking = {group: sorted(splits) for group, splits in membership.items() if len(splits) > 1}
        if leaking:
            raise ValueError(f"{field} leakage: {len(leaking)} groups")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "split_manifest": out_dir / "split_manifest_2935.jsonl",
        "train_queries": out_dir / "queries.train_2635.jsonl",
        "test_queries": out_dir / "queries.test_300.jsonl",
        "train_candidates": out_dir / "candidates.train_2635.top200_master.jsonl",
        "test_candidates": out_dir / "candidates.test_300.top200_master.jsonl",
        "train_labels": out_dir / "reference_labels.train_2635.jsonl",
        "test_labels": out_dir / "reference_labels.test_300.evaluation_only.jsonl",
        "train_reviewed": out_dir / "reviewed_rerank_examples.train_2635.jsonl",
        "train_sft": out_dir / "sft_top30_examples.train_2635.jsonl",
        "audit": out_dir / "split_audit.json",
        "manifest": out_dir / "manifest.json",
    }

    split_manifest: list[dict[str, Any]] = []
    split_queries: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    split_candidates: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    split_labels: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    train_reviewed: list[dict[str, Any]] = []
    train_sft: list[dict[str, Any]] = []
    topk_by_split: dict[str, Counter[int]] = {"train": Counter(), "test": Counter()}

    for qid in sorted(queries):
        query = queries[qid]
        candidate = candidates[qid]
        label = labels[qid]
        split = split_by_qid[qid]
        query_out = copy.deepcopy(query)
        query_out["previous_split"] = query_out.get("split")
        query_out["split"] = split
        query_out["split_release"] = RELEASE_ID
        split_queries[split].append(query_out)
        split_candidates[split].append(candidate)
        split_labels[split].append(label)

        expected_ids = selected_ids(label)
        candidate_ids = [str(value) for value in candidate.get("teacher_candidate_evidence_ids") or []][:30]
        if not 1 <= len(candidate_ids) <= 30 or len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"Invalid Top-30 candidate pool: {qid}")
        if any(evidence_id not in evidence for evidence_id in candidate_ids):
            raise ValueError(f"Candidate Evidence ID missing from store: {qid}")
        if any(evidence_id not in candidate_ids for evidence_id in expected_ids):
            raise ValueError(f"Reference Evidence ID outside Top-30: {qid}")
        topk_by_split[split][len(expected_ids)] += 1

        split_manifest.append(
            {
                "qid": qid,
                "scenario_group_id": query.get("scenario_group_id"),
                "split_group_id": query.get("split_group_id"),
                "query_type": "incomplete" if is_incomplete(query) else "base",
                "variant_of_qid": query.get("variant_of_qid"),
                "missingness_type": query.get("missingness_type"),
                "previous_split": query.get("split"),
                "final_split": split,
                "split_release": RELEASE_ID,
            }
        )

        if split == "train":
            cards = [compact_card(evidence[evidence_id]) for evidence_id in candidate_ids]
            prompt = make_prompt(str(query.get("user_text") or "").strip(), cards)
            completion = make_completion(expected_ids)
            train_reviewed.append(
                {
                    "qid": qid,
                    "scenario_group_id": query.get("scenario_group_id"),
                    "split_group_id": query.get("split_group_id"),
                    "split": "train",
                    "user_text": query.get("user_text"),
                    "visible_context": query.get("visible_context") or {},
                    "information_level": query.get("information_level"),
                    "query_completeness": query.get("query_completeness"),
                    "candidate_evidence_ids": candidate_ids,
                    "topk": (label.get("teacher_output") or {}).get("recommendations") or [],
                    "label_status": label.get("label_status"),
                    "teacher_provenance": label.get("teacher_provenance") or {},
                }
            )
            train_sft.append(
                {
                    "id": qid,
                    "qid": qid,
                    "split": "train",
                    "split_group_id": query.get("split_group_id"),
                    "prompt": prompt,
                    "completion": completion,
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion},
                    ],
                    "meta": {
                        "expected_ids": expected_ids,
                        "candidate_evidence_ids": candidate_ids,
                        "candidate_count": len(candidate_ids),
                        "top_k": len(expected_ids),
                        "query_completeness": query.get("query_completeness"),
                        "source_release": RELEASE_ID,
                        "prompt_uses_visible_query_only": True,
                    },
                }
            )

    write_jsonl(paths["split_manifest"], split_manifest)
    write_jsonl(paths["train_queries"], split_queries["train"])
    write_jsonl(paths["test_queries"], split_queries["test"])
    write_jsonl(paths["train_candidates"], split_candidates["train"])
    write_jsonl(paths["test_candidates"], split_candidates["test"])
    write_jsonl(paths["train_labels"], split_labels["train"])
    write_jsonl(paths["test_labels"], split_labels["test"])
    write_jsonl(paths["train_reviewed"], train_reviewed)
    write_jsonl(paths["train_sft"], train_sft)

    train_rows = [queries[qid] for qid in train_qids]
    test_rows = [queries[qid] for qid in test_qids]
    test_incomplete = [row for row in test_rows if is_incomplete(row)]
    test_base = [row for row in test_rows if not is_incomplete(row)]
    parent_pairs = sum(split_by_qid.get(str(row.get("variant_of_qid") or "")) == "test" for row in test_incomplete)
    paired_base_qids = {str(row.get("variant_of_qid")) for row in test_incomplete}
    audit = {
        "release": RELEASE_ID,
        "seed": args.seed,
        "counts": {
            "all": len(query_rows),
            "train": len(train_rows),
            "test": len(test_rows),
            "train_base": sum(not is_incomplete(row) for row in train_rows),
            "train_incomplete": sum(is_incomplete(row) for row in train_rows),
            "test_base": len(test_base),
            "test_incomplete": len(test_incomplete),
            "test_paired_base_queries": len(paired_base_qids),
            "test_base_without_incomplete_variant": len(test_base) - len(paired_base_qids),
            "test_incomplete_parent_in_test": parent_pairs,
        },
        "missingness_targets": MISSINGNESS_TARGETS,
        "missingness_actual_test": distribution(test_incomplete, "missingness_type"),
        "group_integrity": {
            "scenario_group_cross_split": 0,
            "split_group_cross_split": 0,
            "all_incomplete_parents_in_same_split": True,
        },
        "selection": solver_audit,
        "distribution": {
            key: {
                "all": distribution(query_rows, key),
                "train": distribution(train_rows, key),
                "test": distribution(test_rows, key),
                "test_base": distribution(test_base, key),
                "test_incomplete": distribution(test_incomplete, key),
            }
            for key in ("query_type", "facility_bucket", "primary_mechanism_bucket", "area_type", "traffic_control", "missingness_type")
        },
        "topk_distribution_audit_only_not_used_for_selection": {
            split: {str(k): topk_by_split[split][k] for k in range(4)} for split in ("train", "test")
        },
        "structural_checks": {
            "complete_partition": set(train_qids).isdisjoint(test_qids) and len(train_qids) + len(test_qids) == 2935,
            "duplicate_qids": len(split_by_qid) != 2935,
            "selected_ids_outside_top30": 0,
            "duplicate_candidate_ids": 0,
        },
    }
    write_json(paths["audit"], audit)

    manifest = {
        "release": RELEASE_ID,
        "status": "final_split_frozen_before_retraining",
        "seed": args.seed,
        "split_policy": {
            "classes": ["training", "test"],
            "training_queries": 2635,
            "test_queries": 300,
            "test_base_queries": 200,
            "test_incomplete_queries": 100,
            "group_keys_kept_intact": ["scenario_group_id", "split_group_id"],
            "test_labels_used_for_selection": False,
        },
        "inputs": {
            "queries": {"path": str(args.queries), "sha256": sha256(args.queries)},
            "candidates": {"path": str(args.candidates), "sha256": sha256(args.candidates)},
            "labels": {"path": str(args.labels), "sha256": sha256(args.labels)},
            "evidence_store": {"path": str(args.evidence_store), "sha256": sha256(args.evidence_store)},
        },
        "outputs": {
            name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items() if name != "manifest"
        },
    }
    write_json(paths["manifest"], manifest)
    print(json.dumps({"manifest": str(paths["manifest"]), "audit": audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
