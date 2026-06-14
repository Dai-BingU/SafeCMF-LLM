#!/usr/bin/env python3
"""Build a reproducible stratified train/dev/test split for recommendation data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmfrec.countermeasure_family import broad_mechanism_family, mechanism_family


DEFAULT_MAIN_Q = Path(
    "data/queries.training_3501.naturalized_dedup_v2.rpm_roadside_curve_negcurv_groupfix_v758.spotcheck_round2_top1_supplements.jsonl"
)
DEFAULT_MAIN_L = Path(
    "data/labels.reco.training_3501.dedup_v2.rpm_roadside_curve_negcurv_groupfix_v758.spotcheck_round2_top1_supplements.jsonl"
)
DEFAULT_REPAIR_Q = Path("data/repair/queries.source_preserving_repairs_v689.qcov_c2_00038_repair_lighting_context.jsonl")
DEFAULT_REPAIR_L = Path("data/repair/labels.source_preserving_repairs_v689.qcov_c2_00038_repair_lighting_context.jsonl")
DEFAULT_SFT = Path("data/sft.reco_candidate_rerank.v758_spotcheck_round2.top18_compact.jsonl")
DEFAULT_EVIDENCE = Path("data/evidence_store.facility_v4.driveway_formula_all.jsonl")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + "\n")


def norm_bucket(value: object, *, default: str = "unknown") -> str:
    text = str(value or "").strip().lower()
    if not text or text in {"none", "null", "not specified", "unknown", "all"}:
        return default
    if "urban" in text and "suburban" in text:
        return "urban_suburban"
    if "suburban" in text:
        return "suburban"
    if "urban" in text:
        return "urban"
    if "rural" in text:
        return "rural"
    if "segment" in text:
        return "segment"
    if "intersection" in text:
        return "intersection"
    if "signal" in text:
        return "signalized"
    if "stop" in text:
        return "stop_controlled"
    return text.replace(" ", "_")


def crash_bucket(value: object) -> str:
    text = str(value or "").lower()
    if not text or text in {"none", "not specified", "all"}:
        return "all_or_unknown"
    checks = [
        ("pedbike", ["pedestrian", "bicycle", "bike"]),
        ("runoff", ["run off", "run-off", "runoff", "roadway departure", "lane departure"]),
        ("angle", ["angle"]),
        ("rear_end", ["rear end", "rear-end"]),
        ("head_on", ["head on", "head-on"]),
        ("sideswipe", ["sideswipe"]),
        ("left_turn", ["left turn", "left-turn"]),
        ("cross_median", ["cross median", "cross-median"]),
        ("wet", ["wet"]),
        ("night", ["night"]),
        ("truck", ["truck"]),
    ]
    found = [name for name, terms in checks if any(term in text for term in terms)]
    if not found:
        return "other"
    if len(found) > 2:
        return "mixed"
    return "+".join(found)


def first_evidence_id(label: dict[str, Any]) -> str | None:
    for item in label.get("topk") or []:
        ids = item.get("evidence_ids") or []
        if ids:
            return str(ids[0])
        if item.get("cm_id") is not None:
            return str(item.get("cm_id"))
    return None


def source_bucket(qid: str, query: dict[str, Any]) -> str:
    if "__repair_" in qid or query.get("repair"):
        return "repair"
    if qid.startswith("q_exp_"):
        return "expansion"
    if qid.startswith("q_cov_"):
        return "coverage"
    return "main"


def split_group(rows: list[dict[str, Any]], rng: random.Random, dev_ratio: float, test_ratio: float) -> dict[str, list[dict[str, Any]]]:
    rows = list(rows)
    rng.shuffle(rows)
    n = len(rows)
    if n == 1:
        return {"train": rows, "dev": [], "test": []}
    n_test = max(1, round(n * test_ratio)) if n >= 5 else 0
    n_dev = max(1, round(n * dev_ratio)) if n >= 10 else 0
    if n_test + n_dev >= n:
        n_test = min(n_test, max(0, n - 1))
        n_dev = min(n_dev, max(0, n - n_test - 1))
    test = rows[:n_test]
    dev = rows[n_test : n_test + n_dev]
    train = rows[n_test + n_dev :]
    return {"train": train, "dev": dev, "test": test}


def exact_size_adjust(splits: dict[str, list[dict[str, Any]]], target_dev: int, target_test: int, seed: int) -> None:
    rng = random.Random(seed + 17)

    def move(src: str, dst: str, count: int) -> None:
        if count <= 0:
            return
        rng.shuffle(splits[src])
        moving = splits[src][:count]
        splits[src] = splits[src][count:]
        splits[dst].extend(moving)

    while len(splits["test"]) < target_test and splits["train"]:
        move("train", "test", 1)
    while len(splits["test"]) > target_test:
        move("test", "train", 1)
    while len(splits["dev"]) < target_dev and splits["train"]:
        move("train", "dev", 1)
    while len(splits["dev"]) > target_dev:
        move("dev", "train", 1)


def rebalance_bucket_presence(
    splits: dict[str, list[dict[str, Any]]],
    *,
    key: str,
    ratios: dict[str, float],
    min_total: int,
    seed: int,
) -> None:
    """Ensure non-tiny buckets are represented in dev/test.

    Highly specific stratification creates many singleton groups. Without this
    second pass, rare-but-important aggregate buckets such as expansion or
    coverage examples can accidentally stay entirely in train. This pass moves
    matching train rows into dev/test and swaps out overrepresented rows so
    split sizes stay fixed.
    """
    rng = random.Random(seed)
    all_rows = [row for rows in splits.values() for row in rows]
    total_by_bucket = Counter(str(row.get(key) or "unknown") for row in all_rows)

    def split_count(split: str, bucket: str) -> int:
        return sum(1 for row in splits[split] if str(row.get(key) or "unknown") == bucket)

    def take_from_train(bucket: str) -> dict[str, Any] | None:
        idxs = [i for i, row in enumerate(splits["train"]) if str(row.get(key) or "unknown") == bucket]
        if not idxs:
            return None
        idx = rng.choice(idxs)
        return splits["train"].pop(idx)

    def swap_out(split: str, needed_bucket: str) -> dict[str, Any] | None:
        # Prefer removing a row whose bucket is over its target in this split.
        candidates: list[int] = []
        for i, row in enumerate(splits[split]):
            bucket = str(row.get(key) or "unknown")
            if bucket == needed_bucket:
                continue
            desired = round(total_by_bucket[bucket] * ratios[split])
            if split_count(split, bucket) > desired:
                candidates.append(i)
        if not candidates:
            candidates = [i for i, row in enumerate(splits[split]) if str(row.get(key) or "unknown") != needed_bucket]
        if not candidates:
            return None
        idx = rng.choice(candidates)
        return splits[split].pop(idx)

    for split in ["dev", "test"]:
        for bucket, total in sorted(total_by_bucket.items()):
            if total < min_total:
                continue
            desired = round(total * ratios[split])
            if desired <= 0:
                continue
            while split_count(split, bucket) < desired:
                incoming = take_from_train(bucket)
                if incoming is None:
                    break
                outgoing = swap_out(split, bucket)
                if outgoing is None:
                    splits["train"].append(incoming)
                    break
                splits[split].append(incoming)
                splits["train"].append(outgoing)


def counter(rows: list[dict[str, Any]], key: str) -> Counter:
    return Counter(str(row.get(key) or "unknown") for row in rows)


def top_counts(c: Counter, n: int = 20) -> list[dict[str, Any]]:
    return [{"bucket": k, "count": v} for k, v in c.most_common(n)]


def distribution_table(all_rows: list[dict[str, Any]], splits: dict[str, list[dict[str, Any]]], key: str) -> dict[str, Any]:
    all_counter = counter(all_rows, key)
    out = {"overall": top_counts(all_counter, 30)}
    for name, rows in splits.items():
        out[name] = top_counts(counter(rows, key), 30)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-queries", type=Path, default=DEFAULT_MAIN_Q)
    ap.add_argument("--main-labels", type=Path, default=DEFAULT_MAIN_L)
    ap.add_argument("--repair-queries", type=Path, default=DEFAULT_REPAIR_Q)
    ap.add_argument("--repair-labels", type=Path, default=DEFAULT_REPAIR_L)
    ap.add_argument("--sft", type=Path, default=DEFAULT_SFT)
    ap.add_argument("--evidence-store", type=Path, default=DEFAULT_EVIDENCE)
    ap.add_argument("--out-dir", type=Path, default=Path("data/splits/v690_stratified_85_5_10"))
    ap.add_argument("--seed", type=int, default=20260528)
    ap.add_argument("--dev-ratio", type=float, default=0.05)
    ap.add_argument("--test-ratio", type=float, default=0.10)
    args = ap.parse_args()

    queries = read_jsonl(args.main_queries) + read_jsonl(args.repair_queries)
    labels = read_jsonl(args.main_labels) + read_jsonl(args.repair_labels)
    sft_rows = read_jsonl(args.sft)
    evidences = {str(row.get("evidence_id")): row for row in read_jsonl(args.evidence_store)}

    q_by_id = {str(row["qid"]): row for row in queries}
    l_by_id = {str(row["qid"]): row for row in labels}
    sft_by_id = {str(row["id"]): row for row in sft_rows}
    if set(q_by_id) != set(l_by_id):
        raise SystemExit(f"query/label qid mismatch: {len(q_by_id)} vs {len(l_by_id)}")
    if set(q_by_id) != set(sft_by_id):
        missing = sorted(set(q_by_id) ^ set(sft_by_id))[:20]
        raise SystemExit(f"query/sft id mismatch; examples={missing}")

    annotated: list[dict[str, Any]] = []
    for qid in sorted(q_by_id):
        q = q_by_id[qid]
        label = l_by_id[qid]
        ctx = q.get("target_context") or {}
        first_id = first_evidence_id(label)
        ev = evidences.get(first_id or "") or {}
        card = {
            "countermeasure": ev.get("countermeasure"),
            "countermeasure_category": ev.get("countermeasure_category"),
            "countermeasure_subcategory": ev.get("countermeasure_subcategory"),
        }
        area = norm_bucket(ctx.get("area_type"))
        facility = norm_bucket(ctx.get("facility_type"))
        control = norm_bucket(ctx.get("traffic_control_type"), default="none")
        crash = crash_bucket(ctx.get("crash_type") or q.get("user_text"))
        broad = broad_mechanism_family(card) if first_id else "unknown"
        mech = mechanism_family(card) if first_id else "unknown"
        src = source_bucket(qid, q)
        strat_key = "|".join([area, facility, control, crash, broad, src])
        annotated.append(
            {
                "qid": qid,
                "strat_key": strat_key,
                "area": area,
                "facility": facility,
                "control": control,
                "crash": crash,
                "broad_mechanism": broad,
                "mechanism": mech,
                "source": src,
            }
        )

    total = len(annotated)
    target_test = round(total * float(args.test_ratio))
    target_dev = round(total * float(args.dev_ratio))
    rng = random.Random(args.seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in annotated:
        grouped[row["strat_key"]].append(row)

    splits = {"train": [], "dev": [], "test": []}
    for key in sorted(grouped):
        group_splits = split_group(grouped[key], rng, float(args.dev_ratio), float(args.test_ratio))
        for name in splits:
            splits[name].extend(group_splits[name])
    exact_size_adjust(splits, target_dev=target_dev, target_test=target_test, seed=args.seed)
    ratios = {"train": 1 - args.dev_ratio - args.test_ratio, "dev": args.dev_ratio, "test": args.test_ratio}
    for dim, min_total in [("source", 10), ("facility", 10), ("control", 10), ("area", 10)]:
        rebalance_bucket_presence(splits, key=dim, ratios=ratios, min_total=min_total, seed=args.seed + len(dim))
    exact_size_adjust(splits, target_dev=target_dev, target_test=target_test, seed=args.seed + 101)
    for name in splits:
        splits[name].sort(key=lambda row: row["qid"])

    all_split_qids = [row["qid"] for rows in splits.values() for row in rows]
    if len(all_split_qids) != len(set(all_split_qids)) or set(all_split_qids) != set(q_by_id):
        raise SystemExit("split qids are not a complete partition")

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        qids = [row["qid"] for row in rows]
        write_jsonl(out / f"queries.{name}.jsonl", [q_by_id[qid] for qid in qids])
        write_jsonl(out / f"labels.{name}.jsonl", [l_by_id[qid] for qid in qids])
        write_jsonl(out / f"sft_candidate_rerank_top18_compact.{name}.jsonl", [sft_by_id[qid] for qid in qids])
        write_jsonl(out / f"manifest.{name}.jsonl", rows)

    summary = {
        "split_id": hashlib.sha1((",".join(sorted(q_by_id)) + str(args.seed)).encode()).hexdigest()[:12],
        "seed": args.seed,
        "ratios": ratios,
        "counts": {name: len(rows) for name, rows in splits.items()},
        "inputs": {
            "main_queries": str(args.main_queries),
            "main_labels": str(args.main_labels),
            "repair_queries": str(args.repair_queries),
            "repair_labels": str(args.repair_labels),
            "sft": str(args.sft),
            "evidence_store": str(args.evidence_store),
        },
        "distributions": {
            key: distribution_table(annotated, splits, key)
            for key in ["area", "facility", "control", "crash", "broad_mechanism", "source"]
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"out_dir": str(out), "counts": summary["counts"], "split_id": summary["split_id"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
