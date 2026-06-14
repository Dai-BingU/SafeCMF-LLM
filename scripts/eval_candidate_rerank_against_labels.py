#!/usr/bin/env python3
"""Compare scorer-only vs adapter reranking against reviewed labels.

This evaluates the intended recommender path:
query context -> scorer candidate pool -> adapter selects only from candidates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cmfrec.context_tags import infer_context_tags_from_user_text
from cmfrec.facility import resolve_query_facility_type
from cmfrec.mechanism_recall import (
    access_management_supplement_ids,
    advance_guidance_supplement_ids,
    candidate_context_mismatch,
    cfi_supplement_ids,
    drowsy_driving_supplement_ids,
    frontage_road_supplement_ids,
    hidden_precondition_mismatch,
    managed_lane_supplement_ids,
    median_opening_supplement_ids,
    normalized_countermeasure_key,
    passing_lane_supplement_ids,
    pedestrian_crossing_supplement_ids,
    roadside_mechanism_supplement_ids,
    signalized_left_turn_supplement_ids,
    signal_visibility_supplement_ids,
    shoulder_improvement_supplement_ids,
    speed_management_supplement_ids,
    stop_control_supplement_ids,
    nighttime_segment_supplement_ids,
    toll_plaza_supplement_ids,
    winter_weather_supplement_ids,
)
from cmfrec.scoring import Query, score_row
from scripts.demo_end_to_end_compare import _build_score_row, _extract_json


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def first_eid(item: dict[str, Any]) -> str:
    ids = item.get("evidence_ids") or []
    if ids:
        return str(ids[0])
    return str(item.get("cm_id") or item.get("evidence_id") or "")


def expected_ids(label: dict[str, Any], top_k: int) -> list[str]:
    return [first_eid(x) for x in (label.get("topk") or [])[:top_k] if first_eid(x)]


def to_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except Exception:
        return None


def query_from_context(user_text: str, ctx: dict[str, Any]) -> Query:
    facility_type = resolve_query_facility_type(
        user_text,
        facility_type=ctx.get("facility_type"),
        intersection_related=str(ctx.get("intersection_related") or ""),
    )
    return Query(
        crash_type=ctx.get("crash_type"),
        severity=ctx.get("severity_kabco") or ctx.get("severity"),
        roadway_type=ctx.get("roadway_type"),
        area_type=ctx.get("area_type"),
        facility_type=facility_type,
        intersection_related=ctx.get("intersection_related"),
        traffic_control_type=ctx.get("traffic_control_type"),
        intersection_geometry=ctx.get("intersection_geometry") or ctx.get("intersection_type"),
        min_speed_limit=to_float(ctx.get("min_speed_limit")),
        max_speed_limit=to_float(ctx.get("max_speed_limit")),
        num_lanes=to_float(ctx.get("max_num_lanes") or ctx.get("min_num_lanes") or ctx.get("num_lanes")),
        traffic_volume_aadt=to_float(
            ctx.get("avg_traffic_volume_non_intersection")
            or ctx.get("max_traffic_volume_non_intersection")
            or ctx.get("min_traffic_volume_non_intersection")
        ),
        major_road_volume_aadt=to_float(
            ctx.get("avg_major_road_traffic_volume")
            or ctx.get("max_major_road_traffic_volume")
            or ctx.get("min_major_road_traffic_volume")
        ),
        minor_road_volume_aadt=to_float(
            ctx.get("avg_minor_road_traffic_volume")
            or ctx.get("max_minor_road_traffic_volume")
            or ctx.get("min_minor_road_traffic_volume")
        ),
        context_tags=infer_context_tags_from_user_text(user_text),
    )


def candidate_cards(
    evidences: list[dict[str, Any]],
    score_rows: list[dict[str, str]],
    query: Query,
    top_n: int,
    *,
    user_text: str = "",
    use_mechanism_recall: bool = False,
) -> list[dict[str, Any]]:
    scored: list[tuple[float, dict[str, Any]]] = []
    scores_by_id: dict[str, float] = {}
    for ev, score_row_data in zip(evidences, score_rows):
        if user_text and hidden_precondition_mismatch(ev, user_text):
            continue
        if user_text and candidate_context_mismatch(ev, user_text, query):
            continue
        s = score_row(score_row_data, query)
        if s is None:
            continue
        score = float(s.total_score)
        scores_by_id[str(ev.get("evidence_id"))] = score
        scored.append((score, ev))
    scored.sort(key=lambda t: t[0], reverse=True)

    out: list[dict[str, Any]] = []
    selected: list[tuple[float | None, dict[str, Any]]] = [(score, ev) for score, ev in scored[:top_n]]
    if use_mechanism_recall:
        ev_by_id = {str(ev.get("evidence_id")): ev for ev in evidences}
        row_by_id = {str(ev.get("evidence_id")): row for ev, row in zip(evidences, score_rows)}
        existing = {str(ev.get("evidence_id")) for _, ev in selected}
        supplement_ids = roadside_mechanism_supplement_ids(
            user_text=user_text,
            query=query,
            evidences=evidences,
            score_rows=score_rows,
            existing_ids=existing,
        )
        existing.update(str(eid) for eid in supplement_ids)
        access_ids = access_management_supplement_ids(
            user_text=user_text,
            query=query,
            evidences=evidences,
            score_rows=score_rows,
            existing_ids=existing,
        )
        existing.update(str(eid) for eid in access_ids)
        supplement_ids.extend(access_ids)
        signal_ids = signal_visibility_supplement_ids(
            user_text=user_text,
            query=query,
            evidences=evidences,
            score_rows=score_rows,
            existing_ids=existing,
        )
        existing.update(str(eid) for eid in signal_ids)
        supplement_ids.extend(signal_ids)
        signal_left_ids = signalized_left_turn_supplement_ids(
            user_text=user_text,
            query=query,
            evidences=evidences,
            score_rows=score_rows,
            existing_ids=existing,
        )
        existing.update(str(eid) for eid in signal_left_ids)
        supplement_ids.extend(signal_left_ids)
        stop_ids = stop_control_supplement_ids(
            user_text=user_text,
            query=query,
            evidences=evidences,
            score_rows=score_rows,
            existing_ids=existing,
        )
        existing.update(str(eid) for eid in stop_ids)
        supplement_ids.extend(stop_ids)
        median_ids = median_opening_supplement_ids(
            user_text=user_text,
            query=query,
            evidences=evidences,
            score_rows=score_rows,
            existing_ids=existing,
        )
        existing.update(str(eid) for eid in median_ids)
        supplement_ids.extend(median_ids)
        passing_ids = passing_lane_supplement_ids(
            user_text=user_text,
            query=query,
            evidences=evidences,
            score_rows=score_rows,
            existing_ids=existing,
        )
        existing.update(str(eid) for eid in passing_ids)
        supplement_ids.extend(passing_ids)
        advance_ids = advance_guidance_supplement_ids(
            user_text=user_text,
            query=query,
            evidences=evidences,
            score_rows=score_rows,
            existing_ids=existing,
        )
        existing.update(str(eid) for eid in advance_ids)
        supplement_ids.extend(advance_ids)
        ped_crossing_ids = pedestrian_crossing_supplement_ids(
            user_text=user_text,
            query=query,
            evidences=evidences,
            score_rows=score_rows,
            existing_ids=existing,
        )
        existing.update(str(eid) for eid in ped_crossing_ids)
        supplement_ids.extend(ped_crossing_ids)
        frontage_ids = frontage_road_supplement_ids(
            user_text=user_text,
            query=query,
            evidences=evidences,
            score_rows=score_rows,
            existing_ids=existing,
        )
        existing.update(str(eid) for eid in frontage_ids)
        supplement_ids.extend(frontage_ids)
        managed_lane_ids = managed_lane_supplement_ids(
            user_text=user_text,
            query=query,
            evidences=evidences,
            score_rows=score_rows,
            existing_ids=existing,
        )
        existing.update(str(eid) for eid in managed_lane_ids)
        supplement_ids.extend(managed_lane_ids)
        winter_ids = winter_weather_supplement_ids(
            user_text=user_text,
            query=query,
            evidences=evidences,
            score_rows=score_rows,
            existing_ids=existing,
        )
        existing.update(str(eid) for eid in winter_ids)
        supplement_ids.extend(winter_ids)
        speed_ids = speed_management_supplement_ids(
            user_text=user_text,
            query=query,
            evidences=evidences,
            score_rows=score_rows,
            existing_ids=existing,
        )
        existing.update(str(eid) for eid in speed_ids)
        supplement_ids.extend(speed_ids)
        drowsy_ids = drowsy_driving_supplement_ids(
            user_text=user_text,
            query=query,
            evidences=evidences,
            score_rows=score_rows,
            existing_ids=existing,
        )
        existing.update(str(eid) for eid in drowsy_ids)
        supplement_ids.extend(drowsy_ids)
        cfi_ids = cfi_supplement_ids(
            user_text=user_text,
            query=query,
            evidences=evidences,
            score_rows=score_rows,
            existing_ids=existing,
        )
        existing.update(str(eid) for eid in cfi_ids)
        supplement_ids.extend(cfi_ids)
        shoulder_ids = shoulder_improvement_supplement_ids(
            user_text=user_text,
            query=query,
            evidences=evidences,
            score_rows=score_rows,
            existing_ids=existing,
        )
        existing.update(str(eid) for eid in shoulder_ids)
        supplement_ids.extend(shoulder_ids)
        toll_ids = toll_plaza_supplement_ids(
            user_text=user_text,
            query=query,
            evidences=evidences,
            score_rows=score_rows,
            existing_ids=existing,
        )
        existing.update(str(eid) for eid in toll_ids)
        supplement_ids.extend(toll_ids)
        supplement_ids.extend(
            nighttime_segment_supplement_ids(
                user_text=user_text,
                query=query,
                evidences=evidences,
                score_rows=score_rows,
                existing_ids=existing,
            )
        )
        for eid in supplement_ids:
            eid_s = str(eid)
            if eid_s not in ev_by_id:
                continue
            score = scores_by_id.get(eid_s)
            if score is None and eid_s in row_by_id:
                robust_score = score_row(row_by_id[eid_s], query, match_mode="robust")
                score = None if robust_score is None else float(robust_score.total_score)
            selected.append((score, ev_by_id[eid_s]))

    deduped: list[tuple[float | None, dict[str, Any]]] = []
    seen_keys: set[str] = set()
    for score, ev in selected:
        key = normalized_countermeasure_key(ev)
        if key and key in seen_keys:
            continue
        if key:
            seen_keys.add(key)
        deduped.append((score, ev))
    selected = deduped

    for score, ev in selected:
        c = ev.get("conditions") or {}
        e = ev.get("effect") or {}
        q = ev.get("quality") or {}
        card = {
            "evidence_id": str(ev.get("evidence_id")),
            "countermeasure": ev.get("countermeasure"),
            "crash_type": c.get("crash_type"),
            "area_type": c.get("area_type"),
            "facility_type": c.get("facility_type"),
            "traffic_control_type": c.get("traffic_control_type"),
            "crf": e.get("crf"),
            "cmf": e.get("cmf"),
            "star": q.get("star_quality_rating"),
        }
        if score is not None:
            card["score"] = round(float(score), 4)
        out.append(card)
    return out


def rerank_prompt(user_text: str, target_context: dict[str, Any], candidates: list[dict[str, Any]], top_k: int) -> str:
    payload = {
        "user_text": user_text,
        "target_context": target_context,
        "candidates": candidates,
        "top_k": top_k,
    }
    return (
        "You are a transportation safety engineer.\n"
        "Task: Choose the best ranked countermeasures from INPUT_JSON.candidates.\n"
        "Constraints:\n"
        "- Output valid JSON only.\n"
        "- You MUST choose evidence_ids only from INPUT_JSON.candidates.\n"
        "- Do not invent evidence_ids.\n"
        "- Do not choose duplicate or near-duplicate countermeasures unless no clean alternative exists.\n"
        "- It is allowed to return fewer than top_k recommendations if remaining candidates do not fit.\n"
        "- Prefer mechanism and site-context fit over high CRF alone.\n"
        "- For rural nighttime roadway-segment queries about lane path, roadway alignment, markings, delineation, or pavement-marker visibility, prefer marking/delineation/rumble/curve-guidance treatments; do not choose generic lighting unless the query explicitly says unlit/no lighting/limited lighting/lighting deficiency.\n"
        "- If a nighttime roadway or intersection query explicitly says unlit, no lighting, without lighting, limited lighting, absent lighting, or lighting deficiency, lighting/illumination can be a valid primary recommendation.\n"
        "- For stop-controlled intersections, do not prioritize new signalization unless the query gives adequate volume/warrant context.\n"
        "- For an already signalized intersection, do not choose treatments that install a new traffic signal; use existing-signal timing, phasing, visibility, or geometric treatments instead.\n"
        "- For divided-road median openings or direct-left-turn conflicts, prefer RTUT/RCUT/superstreet/positive-offset left-turn treatments when present.\n"
        "- In that median-opening/direct-left-turn context, do not rank generic warning systems, stop-sign visibility, or flashing-beacon treatments above those geometric conflict treatments unless the query explicitly asks for approach warning/visibility.\n"
        "- For driveway-density/access-point corridor queries, formula/access-management evidence can be useful, but do not add unrelated pedestrian refuge or TWLTL-conversion fillers unless the query states that context.\n"
        "- Output shape: {\"recommendations\":[{\"rank\":1,\"evidence_ids\":[\"...\"]}]}\n"
        "INPUT_JSON:\n"
        + json.dumps(payload, ensure_ascii=False)
        + "\n"
    )


def parse_selected(text: str) -> tuple[list[str], str]:
    obj, err = _extract_json(text)
    if not obj:
        return [], err
    out: list[str] = []
    for rec in obj.get("recommendations") or []:
        if not isinstance(rec, dict):
            continue
        ids = rec.get("evidence_ids") or []
        if ids:
            out.append(str(ids[0]))
    return out, ""


def jaccard(a: list[str], b: list[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def card_by_id(cards: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(c.get("evidence_id")): c for c in cards}


def describe_ids(ids: list[str], cards_by_id: dict[str, dict[str, Any]], ev_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for eid in ids:
        card = cards_by_id.get(eid)
        if not card:
            ev = ev_by_id.get(eid) or {}
            c = ev.get("conditions") or {}
            e = ev.get("effect") or {}
            q = ev.get("quality") or {}
            card = {
                "evidence_id": eid,
                "countermeasure": ev.get("countermeasure"),
                "crash_type": c.get("crash_type"),
                "facility_type": c.get("facility_type"),
                "crf": e.get("crf"),
                "star": q.get("star_quality_rating"),
            }
        rows.append(card)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=Path, nargs="+", required=True)
    ap.add_argument("--labels", type=Path, nargs="+", required=True)
    ap.add_argument("--evidence-store", type=Path, required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--qids", default="", help="Comma-separated qids. If omitted, use first --max rows sorted by qid.")
    ap.add_argument("--max", type=int, default=8)
    ap.add_argument("--candidate-top-n", type=int, default=30)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=140)
    ap.add_argument(
        "--use-mechanism-recall",
        action="store_true",
        help="Append mechanism-specific recall supplements, currently roadside/clear-zone/guardrail families.",
    )
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    queries: dict[str, dict[str, Any]] = {}
    for path in args.queries:
        for row in read_jsonl(path):
            queries[str(row["qid"])] = row
    labels: dict[str, dict[str, Any]] = {}
    for path in args.labels:
        for row in read_jsonl(path):
            labels[str(row["qid"])] = row

    qids = [x.strip() for x in args.qids.split(",") if x.strip()]
    if not qids:
        qids = [qid for qid in sorted(queries) if qid in labels][: int(args.max)]
    else:
        qids = [qid for qid in qids if qid in queries and qid in labels]

    evidences = read_jsonl(args.evidence_store)
    ev_by_id = {str(ev.get("evidence_id")): ev for ev in evidences}
    score_rows = [_build_score_row(ev) for ev in evidences]

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    results: list[dict[str, Any]] = []
    for qid in qids:
        qrow = queries[qid]
        lrow = labels[qid]
        user_text = str(qrow.get("user_text") or "")
        ctx = qrow.get("target_context") or {}
        exp = expected_ids(lrow, int(args.top_k))
        query_obj = query_from_context(user_text, ctx)
        cards = candidate_cards(
            evidences,
            score_rows,
            query_obj,
            int(args.candidate_top_n),
            user_text=user_text,
            use_mechanism_recall=bool(args.use_mechanism_recall),
        )
        scorer_ids = [str(c["evidence_id"]) for c in cards[: int(args.top_k)]]
        prompt = rerank_prompt(user_text, ctx, cards, int(args.top_k))
        inp = tok(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=False,
                temperature=0.0,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
            )
        gen = tok.decode(out[0][inp["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        adapter_ids, json_err = parse_selected(gen)
        cand_ids = {str(c["evidence_id"]) for c in cards}
        c_by_id = card_by_id(cards)
        result = {
            "qid": qid,
            "user_text": user_text,
            "expected_ids": exp,
            "scorer_ids": scorer_ids,
            "adapter_ids": adapter_ids,
            "adapter_json_error": json_err,
            "adapter_ids_in_candidates": all(eid in cand_ids for eid in adapter_ids),
            "adapter_has_duplicate_ids": len(adapter_ids) != len(set(adapter_ids)),
            "scorer_top1_match": bool(exp and scorer_ids and exp[0] == scorer_ids[0]),
            "adapter_top1_match": bool(exp and adapter_ids and exp[0] == adapter_ids[0]),
            "scorer_jaccard": jaccard(exp, scorer_ids),
            "adapter_jaccard": jaccard(exp, adapter_ids),
            "candidate_head": [str(c["evidence_id"]) for c in cards[:10]],
            "expected_cards": describe_ids(exp, c_by_id, ev_by_id),
            "scorer_cards": describe_ids(scorer_ids, c_by_id, ev_by_id),
            "adapter_cards": describe_ids(adapter_ids, c_by_id, ev_by_id),
            "raw_output": gen[:1200],
        }
        results.append(result)
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    metrics = {
        "n": len(results),
        "adapter_json_ok": sum(1 for r in results if not r["adapter_json_error"]),
        "adapter_ids_in_candidates": sum(1 for r in results if r["adapter_ids_in_candidates"]),
        "adapter_duplicate_rows": sum(1 for r in results if r["adapter_has_duplicate_ids"]),
        "scorer_top1_match": sum(1 for r in results if r["scorer_top1_match"]),
        "adapter_top1_match": sum(1 for r in results if r["adapter_top1_match"]),
        "scorer_jaccard_avg": sum(float(r["scorer_jaccard"]) for r in results) / max(1, len(results)),
        "adapter_jaccard_avg": sum(float(r["adapter_jaccard"]) for r in results) / max(1, len(results)),
    }

    lines = ["# Candidate Rerank Eval", "", "## Metrics", "", "```json", json.dumps(metrics, ensure_ascii=False, indent=2), "```", ""]
    for r in results:
        lines.append(f"## {r['qid']}")
        lines.append("")
        lines.append(f"**Query:** {r['user_text']}")
        lines.append("")
        lines.append(f"**Candidate head:** `{', '.join(r['candidate_head'])}`")
        lines.append("")
        lines.append(
            f"**Flags:** json_error=`{r['adapter_json_error']}`, ids_in_candidates=`{r['adapter_ids_in_candidates']}`, duplicate_ids=`{r['adapter_has_duplicate_ids']}`"
        )
        lines.append("")
        for title, key in [("Expected", "expected_cards"), ("Scorer-only", "scorer_cards"), ("Adapter-rerank", "adapter_cards")]:
            lines.append(f"**{title}**")
            lines.append("")
            lines.append("| Rank | Evidence | Countermeasure | CRF | Star | Crash / Facility |")
            lines.append("|---:|---|---|---:|---:|---|")
            for idx, card in enumerate(r[key], start=1):
                cm = str(card.get("countermeasure") or "").replace("|", "\\|")
                lines.append(
                    f"| {idx} | {card.get('evidence_id')} | {cm} | {card.get('crf')} | {card.get('star')} | {card.get('crash_type')} / {card.get('facility_type')} |"
                )
            lines.append("")
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": metrics, "out_json": str(args.out_json), "out_md": str(args.out_md)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
