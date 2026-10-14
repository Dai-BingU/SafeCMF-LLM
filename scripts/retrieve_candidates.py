#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import replace


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cmfrec.free_text import infer_query_from_text  # noqa: E402
from cmfrec.mechanism_recall import (  # noqa: E402
    MechanismRecallConfig,
    access_management_supplement_ids,
    advance_guidance_supplement_ids,
    candidate_context_mismatch,
    frontage_road_supplement_ids,
    hidden_precondition_mismatch,
    interchange_merge_supplement_ids,
    managed_lane_supplement_ids,
    median_opening_supplement_ids,
    passing_lane_supplement_ids,
    pedestrian_crossing_supplement_ids,
    roadside_mechanism_supplement_ids,
    signalized_left_turn_supplement_ids,
    signal_visibility_supplement_ids,
    shoulder_improvement_supplement_ids,
    skew_realignment_supplement_ids,
    stop_control_supplement_ids,
    nighttime_segment_supplement_ids,
    toll_plaza_supplement_ids,
    winter_weather_supplement_ids,
)
from cmfrec.scoring import DEFAULT_TEACHER_CONFIG, Query, score_row  # noqa: E402
from cmfrec.facility import infer_facility_type_for_evidence, resolve_query_facility_type  # noqa: E402
from cmfrec.context_tags import infer_context_tags_from_user_text  # noqa: E402
from cmfrec.countermeasure_family import mechanism_family  # noqa: E402


def _norm(value: object) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _norm_key(value: object) -> str | None:
    v = _norm(value)
    return v.lower() if v else None


def _is_generic(value: str | None) -> bool:
    if not value:
        return True
    v = value.strip().lower()
    return v in {
        "all",
        "not specified",
        "unknown",
        "n/a",
        "na",
        "(blank)",
        "no values chosen.",
        "no values chosen",
    }


def _generic_keys(values: list[str]) -> set[str]:
    return {v for v in values if _is_generic(v)}


def _area_type_aliases(value: str | None) -> list[str]:
    """
    Evidence often uses compound area labels like "Urban and suburban".
    Index those rows under both "urban" and "suburban" so queries that say only
    "urban" can still retrieve them.
    """
    v = _norm_key(value)
    if not v:
        return [""]
    out: list[str] = [v]
    if "urban" in v and "urban" not in out:
        out.append("urban")
    if "suburban" in v and "suburban" not in out:
        out.append("suburban")
    if "rural" in v and "rural" not in out:
        out.append("rural")
    return out


def _norm_countermeasure_text(value: object) -> str:
    s = str(value or "").strip().lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _countermeasure_family_key(ev: dict[str, object]) -> str:
    """
    Coarse countermeasure family key for candidate-pool diversification.

    This is intentionally conservative: exact text de-dup is handled separately;
    family mode only merges very common variants that otherwise occupy many slots
    while conveying nearly the same recommendation to the downstream selector.
    """
    cm = _norm_countermeasure_text(ev.get("countermeasure"))
    cat = _norm_countermeasure_text(ev.get("countermeasure_category"))
    sub = _norm_countermeasure_text(ev.get("countermeasure_subcategory"))
    text = f"{cm} {cat} {sub}"

    if "high friction surface treatment" in text or " hfst " in f" {text} ":
        return "family:hfst"
    if "sidewalk" in text:
        return "family:sidewalk"
    if "crosswalk" in text:
        return "family:crosswalk"
    if "rectangular rapid flashing beacon" in text or " rrfb " in f" {text} ":
        return "family:rrfb"
    if "pedestrian hybrid beacon" in text or " hawk " in f" {text} ":
        return "family:ped_hybrid_beacon"
    if "leading pedestrian interval" in text:
        return "family:lpi"
    if "pedestrian" in text and ("cycle length" in text or "crossing time" in text or "phase" in text):
        return "family:ped_signal_timing"
    if "bicycle lane" in text or "bike lane" in text:
        return "family:bicycle_lanes"
    if "cycle track" in text or "on street cycling" in text:
        return "family:cycle_tracks"
    if "raised bicycle crossing" in text:
        return "family:raised_bicycle_crossing"
    if "roundabout" in text:
        return "family:roundabout_conversion"
    if "centerline rumble" in text and "shoulder rumble" in text:
        return "family:centerline_shoulder_rumble"
    if "centerline rumble" in text:
        return "family:centerline_rumble"
    if "shoulder rumble" in text:
        return "family:shoulder_rumble"
    if "median barrier" in text or "cable median barrier" in text:
        return "family:median_barrier"
    if "pavement friction" in text or "skid resistance" in text:
        return "family:pavement_friction"
    if "lighting" in text or "illumination" in text:
        return "family:lighting"
    if "left turn lane" in text or "left turn phasing" in text or "protected left turn" in text:
        return "family:left_turn_treatment"
    return f"text:{cm}"


def _candidate_dedup_key(ev: dict[str, object], mode: str) -> str | None:
    if mode == "none":
        return None
    if mode == "cm_id":
        cm_id = ev.get("cm_id")
        return f"cm_id:{cm_id}" if cm_id is not None else None
    if mode == "countermeasure":
        return f"text:{_norm_countermeasure_text(ev.get('countermeasure'))}"
    if mode == "family":
        return _countermeasure_family_key(ev)
    raise ValueError(f"Unsupported candidate dedup key: {mode}")


def _dedup_evidence_ids(
    evidence_ids: list[str],
    *,
    evidence_by_id: dict[str, dict[str, object]],
    mode: str,
    limit: int | None = None,
    scores_by_eid: dict[str, float] | None = None,
    quality_tiebreak: bool = False,
    quality_tiebreak_margin: float = 0.15,
) -> list[str]:
    def quality_tuple(eid: str) -> tuple[float, float, float, float]:
        ev = evidence_by_id.get(str(eid)) or {}
        quality = ev.get("quality") or {}
        effect = ev.get("effect") or {}

        def to_float(value: object, default: float = -1.0) -> float:
            try:
                if value is None:
                    return default
                return float(value)
            except Exception:
                return default

        star = to_float(quality.get("star_quality_rating"), default=-1.0)
        n = max(
            to_float(quality.get("num_crashes"), default=-1.0),
            to_float(quality.get("num_crashes_before"), default=-1.0),
        )
        # Lower SE is better. Missing SE sorts after available SE.
        se = min(
            x
            for x in [
                to_float(quality.get("se_adjusted_cmf"), default=1e9),
                to_float(quality.get("se_unadjusted_cmf"), default=1e9),
                to_float(quality.get("se_adjusted_crf"), default=1e9),
                to_float(quality.get("se_unadjusted_crf"), default=1e9),
            ]
            if x is not None
        )
        crf = to_float(effect.get("crf"), default=-1.0)
        if crf < 0:
            cmf = to_float(effect.get("cmf"), default=-1.0)
            crf = abs((1.0 - cmf) * 100.0) if cmf >= 0 else -1.0
        return (star, n, -se, crf)

    def is_better(candidate: str, current: str) -> bool:
        if not quality_tiebreak:
            return False
        if scores_by_eid:
            cand_score = float(scores_by_eid.get(str(candidate), 0.0))
            cur_score = float(scores_by_eid.get(str(current), 0.0))
            # Preserve contextual fit when score gap is meaningful.
            denom = max(abs(cur_score), abs(cand_score), 1e-9)
            if cand_score < cur_score and ((cur_score - cand_score) / denom) > float(quality_tiebreak_margin):
                return False
            if cand_score > cur_score and ((cand_score - cur_score) / denom) > float(quality_tiebreak_margin):
                return True
        return quality_tuple(str(candidate)) > quality_tuple(str(current))

    out: list[str] = []
    seen_eids: set[str] = set()
    key_to_pos: dict[str, int] = {}
    for eid in evidence_ids:
        if not eid or eid in seen_eids:
            continue
        ev = evidence_by_id.get(str(eid))
        key = None if mode == "none" else _candidate_dedup_key(ev or {}, mode)
        if key and key in key_to_pos:
            pos = key_to_pos[key]
            current = out[pos]
            if is_better(str(eid), str(current)):
                seen_eids.discard(current)
                out[pos] = str(eid)
                seen_eids.add(str(eid))
            continue
        out.append(eid)
        seen_eids.add(eid)
        if key:
            key_to_pos[key] = len(out) - 1
        if limit and len(out) >= limit:
            break
    return out


def _append_compatible_backfill(
    current: list[str],
    ranked_fallback: list[str],
    *,
    evidence_by_id: dict[str, dict[str, object]],
    mode: str,
    target_minimum: int,
    hard_limit: int,
) -> tuple[list[str], list[str]]:
    """Fill a short diverse list without disturbing its existing ranking.

    Every fallback ID comes from the already filtered, positively scored pool.
    The staged caller first permits a new exact treatment, then a new CMF row,
    and only then any distinct evidence ID. No random or context-conflicting
    card is introduced merely to reach the teacher-visible minimum.
    """
    out = [str(x) for x in current]
    seen_ids = set(out)
    seen_keys: set[str] = set()
    if mode != "none":
        for eid in out:
            ev = evidence_by_id.get(eid) or {}
            key = _candidate_dedup_key(ev, mode)
            if key:
                seen_keys.add(key)
    added: list[str] = []
    for eid0 in ranked_fallback:
        eid = str(eid0)
        if eid in seen_ids:
            continue
        ev = evidence_by_id.get(eid) or {}
        key = None if mode == "none" else _candidate_dedup_key(ev, mode)
        if key and key in seen_keys:
            continue
        out.append(eid)
        added.append(eid)
        seen_ids.add(eid)
        if key:
            seen_keys.add(key)
        if len(out) >= target_minimum or len(out) >= hard_limit:
            break
    return out, added


def _teacher_diverse_ids(
    evidence_ids: list[str],
    *,
    evidence_by_id: dict[str, dict[str, object]],
    limit: int,
    allow_alternative_evidence: bool = True,
) -> list[str]:
    """Build a diverse teacher prefix without discarding master-pool evidence."""
    ranked = [str(eid) for eid in evidence_ids if str(eid) in evidence_by_id]
    out: list[str] = []
    seen_ids: set[str] = set()
    seen_families: set[str] = set()
    seen_treatments: set[str] = set()

    # First expose one card per engineering mechanism family.
    for eid in ranked:
        family = mechanism_family(evidence_by_id[eid])
        treatment = _norm_countermeasure_text(evidence_by_id[eid].get("countermeasure"))
        if family in seen_families or treatment in seen_treatments:
            continue
        out.append(eid)
        seen_ids.add(eid)
        seen_families.add(family)
        seen_treatments.add(treatment)
        if len(out) >= limit:
            return out

    # Then allow distinct treatments within an already represented family.
    for eid in ranked:
        if eid in seen_ids:
            continue
        treatment = _norm_countermeasure_text(evidence_by_id[eid].get("countermeasure"))
        if treatment in seen_treatments:
            continue
        out.append(eid)
        seen_ids.add(eid)
        seen_treatments.add(treatment)
        if len(out) >= limit:
            return out

    if not allow_alternative_evidence:
        return out

    # Finally retain alternative evidence cards if explicitly requested.
    for eid in ranked:
        if eid in seen_ids:
            continue
        out.append(eid)
        seen_ids.add(eid)
        if len(out) >= limit:
            break
    return out


def _to_query_from_context(ctx: dict[str, object], *, user_text: str | None = None) -> Query:
    # The reconstructed queries intentionally store only visible structured
    # context. Some plainly stated facts (most often crash type) live only in
    # user_text, so use the deterministic text parser solely as a fallback for
    # fields absent from the visible context. This never consults a paired base
    # query or hidden reconstruction metadata.
    inferred = infer_query_from_text(str(user_text or "")).query if user_text else Query()
    facility_type = resolve_query_facility_type(
        str(user_text or ""),
        facility_type=_norm(ctx.get("facility_type")) or inferred.facility_type,
        intersection_related=_norm(ctx.get("intersection_related")) or inferred.intersection_related,
    )
    context_tags = ()
    if user_text:
        context_tags = infer_context_tags_from_user_text(str(user_text))
    return Query(
        crash_type=_norm(ctx.get("crash_type")) or inferred.crash_type,
        severity=_norm(ctx.get("severity_kabco")) or _norm(ctx.get("severity")) or inferred.severity,
        roadway_type=_norm(ctx.get("roadway_type")) or inferred.roadway_type,
        area_type=_norm(ctx.get("area_type")) or inferred.area_type,
        facility_type=facility_type,
        intersection_related=_norm(ctx.get("intersection_related")) or inferred.intersection_related,
        traffic_control_type=_norm(ctx.get("traffic_control_type")) or inferred.traffic_control_type,
        intersection_geometry=_norm(ctx.get("intersection_geometry")) or inferred.intersection_geometry,
        min_speed_limit=(
            ctx.get("min_speed_limit")
            if isinstance(ctx.get("min_speed_limit"), (int, float))
            else inferred.min_speed_limit
        ),
        max_speed_limit=(
            ctx.get("max_speed_limit")
            if isinstance(ctx.get("max_speed_limit"), (int, float))
            else inferred.max_speed_limit
        ),
        num_lanes=(
            ctx.get("num_lanes") if isinstance(ctx.get("num_lanes"), (int, float)) else
            (ctx.get("max_num_lanes") if isinstance(ctx.get("max_num_lanes"), (int, float)) else
             (ctx.get("min_num_lanes") if isinstance(ctx.get("min_num_lanes"), (int, float)) else inferred.num_lanes))
        ),
        traffic_volume_aadt=(
            ctx.get("traffic_volume_aadt") if isinstance(ctx.get("traffic_volume_aadt"), (int, float)) else
            (ctx.get("avg_traffic_volume_non_intersection") if isinstance(ctx.get("avg_traffic_volume_non_intersection"), (int, float)) else
             (ctx.get("max_traffic_volume_non_intersection") if isinstance(ctx.get("max_traffic_volume_non_intersection"), (int, float)) else
              (ctx.get("min_traffic_volume_non_intersection") if isinstance(ctx.get("min_traffic_volume_non_intersection"), (int, float)) else inferred.traffic_volume_aadt)))
        ),
        major_road_volume_aadt=(
            ctx.get("avg_major_road_traffic_volume") if isinstance(ctx.get("avg_major_road_traffic_volume"), (int, float)) else
            (ctx.get("max_major_road_traffic_volume") if isinstance(ctx.get("max_major_road_traffic_volume"), (int, float)) else
             (ctx.get("min_major_road_traffic_volume") if isinstance(ctx.get("min_major_road_traffic_volume"), (int, float)) else inferred.major_road_volume_aadt))
        ),
        minor_road_volume_aadt=(
            ctx.get("avg_minor_road_traffic_volume") if isinstance(ctx.get("avg_minor_road_traffic_volume"), (int, float)) else
            (ctx.get("max_minor_road_traffic_volume") if isinstance(ctx.get("max_minor_road_traffic_volume"), (int, float)) else
             (ctx.get("min_minor_road_traffic_volume") if isinstance(ctx.get("min_minor_road_traffic_volume"), (int, float)) else inferred.minor_road_volume_aadt))
        ),
        countermeasure_category=None,
        countermeasure_subcategory=None,
        min_star=None,
        context_tags=context_tags,
    )


def _load_llm_parser(base_model: str, adapter: str):
    """
    Load a trained parser adapter (QLoRA) for free-text → parsed_context JSON.

    Imports are intentionally local so that the script can still run in a lightweight
    environment when LLM parsing is not enabled.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


def _extract_json_object(text: str) -> dict[str, object]:
    s = (text or "").strip()
    if s.lower().startswith("assistant:"):
        s = s.split(":", 1)[1].strip()
    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("No JSON object found in generation.")
    return json.loads(s[start : end + 1])


def _parse_context_with_llm(tok, model, user_text: str, max_new_tokens: int) -> dict[str, object]:
    import torch

    prompt = (
        f"User: {str(user_text).strip()}\n"
        "Task: Extract the crash/site context from the user text as JSON.\n"
        "Constraints:\n"
        "- Output must be valid JSON.\n"
        "- Do NOT guess missing values; use null when unknown.\n"
        "- Use field names consistent with the evidence_store schema.\n"
    )
    inp = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )
    gen = tok.decode(out[0][inp["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    obj = _extract_json_object(gen)
    ctx = obj.get("parsed_context") if isinstance(obj, dict) else None
    if not isinstance(ctx, dict):
        raise ValueError("JSON did not contain parsed_context object.")
    return {
        "parsed_context": ctx,
        "missing_fields": obj.get("missing_fields") or [],
        "uncertainties": obj.get("uncertainties") or [],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrieve Top-N candidate evidences for each query")
    ap.add_argument("--evidence-store", required=True, help="evidence_store.jsonl")
    ap.add_argument("--queries", required=True, help="queries.jsonl")
    ap.add_argument("--out", default="data/candidates.jsonl", help="Output JSONL path")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--top-n", type=int, default=50, help="Candidate list size per query")
    ap.add_argument(
        "--score-sample",
        type=int,
        default=800,
        help="If candidate pool is larger than this, randomly sample this many for scoring (default: 800)",
    )
    ap.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="If >0, stop after processing this many queries (debug/smoke test)",
    )
    ap.add_argument(
        "--start-query",
        type=int,
        default=0,
        help="Zero-based input row at which to start processing (for deterministic sharding).",
    )
    ap.add_argument(
        "--use-target-context",
        action="store_true",
        help="Use queries.target_context (oracle) instead of parsing user_text",
    )
    ap.add_argument(
        "--llm-parser-base-model",
        default=None,
        help="Use a trained parser adapter (QLoRA) to parse user_text. Provide base model path/name.",
    )
    ap.add_argument(
        "--llm-parser-adapter",
        default=None,
        help="Parser adapter directory path (output of scripts/train_qlora_sft.py --task parser).",
    )
    ap.add_argument("--llm-parser-max-new-tokens", type=int, default=400)
    ap.add_argument(
        "--mix-random",
        type=int,
        default=10,
        help="Add this many random distractor evidences to each candidate list (default: 10)",
    )
    ap.add_argument(
        "--match-mode",
        choices=["strict", "robust"],
        default="strict",
        help="Matching mode for retrieval scoring (strict excludes mismatches; robust keeps mismatches with downweight)",
    )
    ap.add_argument(
        "--disable-context-retrieval",
        action="store_true",
        help=(
            "Ablation: disable query-context indexing and matching for the initial retrieval stage. "
            "Evidence is ranked only by estimated safety effect and evidence quality; other enabled modules remain unchanged."
        ),
    )
    ap.add_argument(
        "--disable-applicability-filtering",
        action="store_true",
        help=(
            "Ablation: disable the standalone hidden-precondition and engineering-context conflict filters. "
            "Basic evidence validity/effect checks and other enabled modules remain unchanged."
        ),
    )
    ap.add_argument(
        "--match-score-power",
        type=float,
        default=None,
        help="Override TeacherConfig.match_score_power for retrieval scoring (e.g., 1.5–2.0 penalizes generic matches)",
    )
    ap.add_argument(
        "--mismatch-multiplier",
        type=float,
        default=None,
        help="Override TeacherConfig.mismatch_multiplier for robust matching (smaller => harsher mismatch penalty)",
    )
    ap.add_argument(
        "--seed-pedestrian-category",
        action="store_true",
        help='If set, vehicle/pedestrian queries will always include evidence rows in category "Pedestrians" in the scoring pool.',
    )
    ap.add_argument(
        "--force-include-ped-top",
        type=int,
        default=0,
        help=(
            'If >0, for vehicle/pedestrian queries force-include the top-N scored evidences from category "Pedestrians" '
            "even if they are not in the overall Top-N. Useful for recall (e.g., RRFB) when overall ranking is dominated "
            "by generic high-effect treatments."
        ),
    )
    ap.add_argument(
        "--seed-targeted-categories",
        action="store_true",
        help=(
            "If set, add scenario-specific recall pools before scoring, e.g. pedestrian, bicycle, "
            "cross-median, nighttime, wet-road/friction, and head-on/sideswipe lane-departure evidence."
        ),
    )
    ap.add_argument(
        "--force-include-targeted-top",
        type=int,
        default=0,
        help=(
            "If >0 with --seed-targeted-categories, prepend the top-N scored evidence rows from matched "
            "scenario-specific recall pools. This improves recall when global ranking misses domain-specific evidence."
        ),
    )
    ap.add_argument(
        "--candidate-dedup-key",
        choices=["none", "cm_id", "countermeasure", "family"],
        default="none",
        help=(
            "De-duplicate the retrieved candidate list before writing it. "
            "'countermeasure' removes repeated exact treatment names; 'family' also collapses common near-duplicate variants."
        ),
    )
    ap.add_argument(
        "--min-candidates",
        type=int,
        default=0,
        help=(
            "Minimum compatible candidates to retain after diversity de-duplication. "
            "Short lists are backfilled only from already filtered, positively scored evidence."
        ),
    )
    ap.add_argument(
        "--teacher-top-n",
        type=int,
        default=0,
        help="If >0, also store a separately diversified teacher_candidate_evidence_ids list.",
    )
    ap.add_argument(
        "--teacher-unique-treatments-only",
        action="store_true",
        help=(
            "Keep at most one Evidence Card per normalized treatment in the teacher pool. "
            "Sparse facilities may therefore have fewer than --teacher-top-n cards; the full master pool is unchanged."
        ),
    )
    ap.add_argument(
        "--master-diversity-slots",
        type=int,
        default=0,
        help=(
            "Reserve this many leading master-pool positions for distinct engineering mechanisms/treatments; "
            "remaining positions retain the highest-ranked compatible evidence cards."
        ),
    )
    ap.add_argument(
        "--dedup-quality-tiebreak",
        action="store_true",
        help=(
            "When de-duplicating repeated countermeasures/families, keep the contextually best evidence first, "
            "but allow higher-quality evidence to replace it when retrieval scores are close."
        ),
    )
    ap.add_argument(
        "--dedup-quality-tiebreak-margin",
        type=float,
        default=0.15,
        help=(
            "Relative score margin for --dedup-quality-tiebreak. A higher-quality duplicate may replace the current "
            "one when its retrieval score is within this relative margin."
        ),
    )
    ap.add_argument(
        "--force-include-family-slots",
        type=int,
        default=0,
        help=(
            "If >0, force-include up to this many scored evidences from each scenario-specific treatment family "
            "(e.g., sidewalk/crosswalk/RRFB for pedestrian segments). This is a recall guard against missing key content."
        ),
    )
    ap.add_argument(
        "--force-include-roadside-mechanism-top",
        type=int,
        default=0,
        help=(
            "If >0, append up to this many roadside/clear-zone/guardrail mechanism candidates for segment "
            "queries mentioning utility poles, side slopes, clear zone, limited recovery area, or fixed roadside objects."
        ),
    )
    ap.add_argument(
        "--force-include-nighttime-segment-mechanism-top",
        type=int,
        default=0,
        help=(
            "If >0, append nighttime segment mechanism candidates. Lighting is high-priority only for explicit "
            "lighting-deficiency or urban/suburban contexts; rural head-on/ROR segments prioritize rumble, markings, "
            "curve delineation, and safety edge."
        ),
    )
    ap.add_argument(
        "--force-include-access-management-mechanism-top",
        type=int,
        default=0,
        help=(
            "If >0, append driveway/access-management mechanism candidates, including formula-based "
            "driveway-density rows that lack a concrete CRF."
        ),
    )
    ap.add_argument(
        "--force-include-signal-visibility-mechanism-top",
        type=int,
        default=0,
        help=(
            "If >0, append signal-display visibility candidates for signalized intersection queries "
            "that mention night or signal visibility."
        ),
    )
    ap.add_argument(
        "--force-include-signalized-left-turn-mechanism-top",
        type=int,
        default=0,
        help=(
            "If >0, append left-turn phasing/offset candidates for existing signalized-intersection "
            "queries that mention left-turn conflicts."
        ),
    )
    ap.add_argument(
        "--force-include-stop-control-mechanism-top",
        type=int,
        default=0,
        help="If >0, append stop-control signing/marking candidates for stop-controlled intersection queries.",
    )
    ap.add_argument(
        "--force-include-skew-realignment-mechanism-top",
        type=int,
        default=0,
        help=(
            "If >0, append compatible skew-angle CMFunction evidence for explicitly skewed, "
            "stop-controlled urban or rural intersections."
        ),
    )
    ap.add_argument(
        "--force-include-median-opening-mechanism-top",
        type=int,
        default=0,
        help="If >0, append RTUT/RCUT/superstreet/offset-left-turn candidates for divided-road median-opening conflicts.",
    )
    ap.add_argument(
        "--force-include-passing-lane-mechanism-top",
        type=int,
        default=0,
        help="If >0, append passing/climbing lane candidates for segment queries that mention limited passing opportunities.",
    )
    ap.add_argument(
        "--force-include-advance-guidance-mechanism-top",
        type=int,
        default=0,
        help="If >0, append advance-guidance/ICWS candidates for intersection queries that mention approach guidance or advance warning.",
    )
    ap.add_argument(
        "--force-include-ped-crossing-mechanism-top",
        type=int,
        default=0,
        help="If >0, append PHB/RRFB/refuge/raised-crosswalk or rural crosswalk warning candidates for pedestrian-crossing queries.",
    )
    ap.add_argument(
        "--force-include-frontage-road-mechanism-top",
        type=int,
        default=0,
        help="If >0, append frontage-road one-way conversion candidates for two-way frontage-road conflict queries.",
    )
    ap.add_argument(
        "--force-include-managed-lane-mechanism-top",
        type=int,
        default=0,
        help="If >0, append HOV-to-HOT candidates for managed-lane/HOV access or weaving queries.",
    )
    ap.add_argument(
        "--force-include-interchange-merge-mechanism-top",
        type=int,
        default=0,
        help="If >0, append acceleration-lane, ramp-metering, merge/diverge, and weaving candidates for ramp queries.",
    )
    ap.add_argument(
        "--force-include-winter-weather-mechanism-top",
        type=int,
        default=0,
        help="If >0, append FAST / snow-ice-control candidates for winter-weather or pavement-icing queries.",
    )
    ap.add_argument(
        "--force-include-shoulder-improvement-mechanism-top",
        type=int,
        default=0,
        help="If >0, append shoulder paving/widening candidates for explicit shoulder-condition lane-departure queries.",
    )
    ap.add_argument(
        "--force-include-toll-plaza-mechanism-top",
        type=int,
        default=0,
        help="If >0, append open-road/all-electronic tolling candidates for traditional mainline toll-plaza queries.",
    )
    ap.add_argument(
        "--force-include-source-evidence",
        action="store_true",
        help=(
            "If set, prepend each query's source_evidence_id to the candidate list when it exists in the evidence store. "
            "This is mainly for coverage-boost or audit sets where the query was generated from a specific evidence row."
        ),
    )
    ap.add_argument(
        "--force-include-source-rank",
        type=int,
        default=1,
        help=(
            "1-based insertion rank for --force-include-source-evidence. Use 40 to keep the source visible to a "
            "--max-candidates 40 LLM teacher without placing it first."
        ),
    )
    ap.add_argument(
        "--include-candidate-provenance",
        action="store_true",
        help=(
            "If set, include candidate_provenance: evidence_id -> retrieval sources "
            "(field_scored, targeted_top:*, family_slot:*, pedestrian_category_top, random_distractor)."
        ),
    )
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cfg = DEFAULT_TEACHER_CONFIG
    if args.match_score_power is not None:
        cfg = replace(cfg, match_score_power=float(args.match_score_power))
    if args.mismatch_multiplier is not None:
        cfg = replace(cfg, mismatch_multiplier=float(args.mismatch_multiplier))

    use_llm_parser = bool(args.llm_parser_base_model and args.llm_parser_adapter)
    if args.use_target_context and use_llm_parser:
        raise SystemExit("Choose only one parsing mode: --use-target-context OR --llm-parser-*")

    llm_tok = llm_model = None
    if use_llm_parser:
        llm_tok, llm_model = _load_llm_parser(args.llm_parser_base_model, args.llm_parser_adapter)

    # Load evidence store and build lightweight indexes.
    evidences: list[dict[str, object]] = []
    score_rows: list[dict[str, str]] = []
    idx: dict[str, dict[str, list[int]]] = {
        "crash_type": defaultdict(list),
        "area_type": defaultdict(list),
        "intersection_related": defaultdict(list),
        "traffic_control_type": defaultdict(list),
        "intersection_geometry": defaultdict(list),
        "countermeasure_category": defaultdict(list),
    }

    with open(args.evidence_store, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            ev = json.loads(line)
            evidences.append(ev)
            c = ev.get("conditions") or {}
            q = ev.get("quality") or {}
            e = ev.get("effect") or {}
            derived_facility_type = infer_facility_type_for_evidence(
                countermeasure=ev.get("countermeasure"),
                conditions=c,
            )

            crash = _norm(c.get("crash_type"))
            if crash and "," in crash:
                for part in [p.strip() for p in crash.split(",") if p.strip()]:
                    idx["crash_type"][_norm_key(part) or ""].append(i)
            idx["crash_type"][_norm_key(crash) or ""].append(i)
            for a in _area_type_aliases(_norm(c.get("area_type"))):
                idx["area_type"][a].append(i)
            idx["intersection_related"][_norm_key(c.get("intersection_related")) or ""].append(i)
            idx["traffic_control_type"][_norm_key(c.get("traffic_control_type")) or ""].append(i)
            geom = _norm(c.get("intersection_geometry"))
            if geom and "," in geom:
                for part in [p.strip() for p in geom.split(",") if p.strip()]:
                    idx["intersection_geometry"][_norm_key(part) or ""].append(i)
            idx["intersection_geometry"][_norm_key(geom) or ""].append(i)
            idx["countermeasure_category"][_norm_key(ev.get("countermeasure_category")) or ""].append(i)

            # Pre-build a minimal row dict compatible with cmfrec.scoring.score_row.
            score_rows.append(
                {
                    "Countermeasure": str(ev.get("countermeasure") or ""),
                    "Countermeasure Category": str(ev.get("countermeasure_category") or ""),
                    "Countermeasure Subcategory": str(ev.get("countermeasure_subcategory") or ""),
                    "CMF": str(e.get("cmf") or ""),
                    "CRF": str(e.get("crf") or ""),
                    "Star Quality Rating": str(q.get("star_quality_rating") or ""),
                    "Adjusted Standard Error of CMF": str(q.get("se_adjusted_cmf") or ""),
                    "Unadjusted Standard Error of CMF": str(q.get("se_unadjusted_cmf") or ""),
                    "Adjusted Standard Error of CRF": str(q.get("se_adjusted_crf") or ""),
                    "Unadjusted Standard Error of CRF": str(q.get("se_unadjusted_crf") or ""),
                    "Number of Crashes": str(q.get("num_crashes") or ""),
                    "Number of Crashes Before": str(q.get("num_crashes_before") or ""),
                    "Crash Type": str(c.get("crash_type") or ""),
                    "KABCO Crash Severity": str(c.get("severity_kabco") or ""),
                    "Roadway Type": str(c.get("roadway_type") or ""),
                    "Area Type": str(c.get("area_type") or ""),
                    "Facility Type": str(c.get("facility_type") or derived_facility_type or ""),
                    "Intersection Related": str(c.get("intersection_related") or ""),
                    "Traffic Control Type": str(c.get("traffic_control_type") or ""),
                    "Intersection Type": str(c.get("intersection_type") or ""),
                    "Intersection Geometry": str(c.get("intersection_geometry") or ""),
                    "Min Speed Limit": str(c.get("min_speed_limit") or ""),
                    "Max Speed Limit": str(c.get("max_speed_limit") or ""),
                    "Min Num Lanes": str(c.get("min_num_lanes") or ""),
                    "Max Num Lanes": str(c.get("max_num_lanes") or ""),
                    "Roadway Division Type": str(c.get("roadway_division_type") or ""),
                    "Minimum Traffic Volume (non-intersection)": str(
                        c.get("min_traffic_volume_non_intersection") or ""
                    ),
                    "Maximum Traffic Volume (non-intersection)": str(
                        c.get("max_traffic_volume_non_intersection") or ""
                    ),
                    "Average Traffic Volume (non-intersection)": str(
                        c.get("avg_traffic_volume_non_intersection") or ""
                    ),
                    "Minimum Major Road Traffic Volume (intersection)": str(
                        c.get("min_major_road_traffic_volume") or ""
                    ),
                    "Maximum Major Road Traffic Volume (intersection)": str(
                        c.get("max_major_road_traffic_volume") or ""
                    ),
                    "Average Major Road Traffic Volume (intersection)": str(
                        c.get("avg_major_road_traffic_volume") or ""
                    ),
                    "Minimum Minor Road Traffic Volume (intersection)": str(
                        c.get("min_minor_road_traffic_volume") or ""
                    ),
                    "Maximum Minor Road Traffic Volume (intersection)": str(
                        c.get("max_minor_road_traffic_volume") or ""
                    ),
                    "Average Minor Road Traffic Volume (intersection)": str(
                        c.get("avg_minor_road_traffic_volume") or ""
                    ),
                }
            )

    evidence_by_id = {str(ev.get("evidence_id")): ev for ev in evidences if ev.get("evidence_id") is not None}

    # Precompute generic buckets (per field) to keep generalized evidence available.
    crash_generic = _generic_keys([k for k in idx["crash_type"].keys() if k])
    area_generic = _generic_keys([k for k in idx["area_type"].keys() if k])
    ctrl_generic = _generic_keys([k for k in idx["traffic_control_type"].keys() if k])
    geom_generic = _generic_keys([k for k in idx["intersection_geometry"].keys() if k])

    all_indices = list(range(len(evidences)))
    ped_category_pool = set(idx["countermeasure_category"].get("pedestrians", []))

    targeted_pools: dict[str, set[int]] = {
        "pedestrian": set(),
        "bicycle": set(),
        "cross_median": set(),
        "nighttime": set(),
        "wet_road": set(),
        "head_on_sideswipe": set(),
    }
    family_pools: dict[str, set[int]] = defaultdict(set)

    def _text_has(text: str, needles: tuple[str, ...]) -> bool:
        return any(needle in text for needle in needles)

    for i, ev in enumerate(evidences):
        c = ev.get("conditions") or {}
        text = " ".join(
            str(x or "")
            for x in [
                ev.get("countermeasure"),
                ev.get("countermeasure_category"),
                ev.get("countermeasure_subcategory"),
                c.get("crash_type"),
                c.get("crash_time_of_day"),
                c.get("crash_weather"),
            ]
        ).lower()

        if _text_has(
            text,
            (
                "pedestrian",
                "crosswalk",
                "sidewalk",
                "rrfb",
                "rapid flashing beacon",
                "hawk",
                "pedestrian hybrid beacon",
                "raised median",
            ),
        ):
            targeted_pools["pedestrian"].add(i)
        if _text_has(text, ("bicycle", "bicyclist", "bike", "cycle track", "cycling")):
            targeted_pools["bicycle"].add(i)
        if _text_has(text, ("cross median", "median barrier", "cable median", "median guardrail")):
            targeted_pools["cross_median"].add(i)
        if _text_has(text, ("nighttime", "night-time", "night time", "lighting", "illumination", "delineation")):
            targeted_pools["nighttime"].add(i)
        if _text_has(
            text,
            (
                "wet road",
                "wet-weather",
                "wet weather",
                "friction",
                "skid",
                "high friction",
                "hfst",
                "ogfc",
                "diamond grinding",
                "chip seal",
            ),
        ):
            targeted_pools["wet_road"].add(i)
        if _text_has(
            text,
            (
                "head on",
                "sideswipe",
                "centerline rumble",
                "centerline and shoulder rumble",
                "raised median",
                "median barrier",
            ),
        ):
            targeted_pools["head_on_sideswipe"].add(i)

        if _text_has(text, ("sidewalk", "walkway", "pedestrian path")):
            family_pools["ped_sidewalk"].add(i)
        if _text_has(text, ("crosswalk", "marked crossing", "high visibility crossing")):
            family_pools["ped_crosswalk"].add(i)
        if _text_has(text, ("rrfb", "rectangular rapid flashing beacon", "rapid flashing beacon")):
            family_pools["ped_rrfb"].add(i)
        if _text_has(text, ("pedestrian hybrid beacon", "hawk")):
            family_pools["ped_hybrid_beacon"].add(i)
        if _text_has(text, ("leading pedestrian interval", "exclusive pedestrian phasing", "barnes dance", "pedestrian countdown", "pedestrian crossing time", "cycle length for pedestrian")):
            family_pools["ped_signal_timing"].add(i)
        if _text_has(text, ("raised median", "median refuge", "pedestrian refuge")):
            family_pools["ped_refuge"].add(i)
        if _text_has(text, ("speed enforcement", "automated speed", "speed camera", "speed restriction", "traffic calming")):
            family_pools["speed_management"].add(i)
        if _text_has(text, ("bicycle lane", "bike lane")):
            family_pools["bike_lane"].add(i)
        if _text_has(text, ("cycle track", "on-street cycling", "on street cycling")):
            family_pools["cycle_track"].add(i)
        if _text_has(text, ("raised bicycle crossing", "bicycle crossing")):
            family_pools["bike_crossing"].add(i)
        if _text_has(text, ("protected left-turn", "protected left turn", "left-turn phasing", "left turn phasing")):
            family_pools["signal_phasing"].add(i)
        if _text_has(text, ("high friction surface treatment", "hfst")):
            family_pools["hfst"].add(i)
        if _text_has(text, ("pavement friction", "skid resistance", "diamond grinding", "chip seal")):
            family_pools["pavement_friction"].add(i)
        if _text_has(text, ("lighting", "illumination")):
            family_pools["lighting"].add(i)
        if _text_has(text, ("centerline rumble", "shoulder rumble", "rumble strips")):
            family_pools["rumble_strips"].add(i)
        if _text_has(text, ("safety edge", "shoulder widening", "widen shoulder", "widen narrow pavement")):
            family_pools["roadside_recovery"].add(i)
        if _text_has(text, ("twltl", "two-way left turn lane", "left turn lane")):
            family_pools["left_turn_lane"].add(i)
        if _text_has(text, ("median barrier", "cable median barrier", "median guardrail")):
            family_pools["median_barrier"].add(i)
        family_pools[f"mechanism:{mechanism_family(ev)}"].add(i)

    def _split_multi(value: str | None) -> list[str]:
        if not value:
            return []
        if "," in value:
            return [p.strip() for p in value.split(",") if p.strip()]
        return [value.strip()]

    def targeted_pool_names(query: Query, user_text: str | None) -> list[str]:
        crash = (query.crash_type or "").lower()
        raw_text = (user_text or "").lower()
        search_text = f"{crash} {raw_text}"
        tags = {str(t).lower() for t in (query.context_tags or ())}
        out: list[str] = []
        if "ped" in search_text:
            out.append("pedestrian")
        if "bicycle" in search_text or "bike" in search_text:
            out.append("bicycle")
        if "cross median" in search_text or "cross-median" in search_text:
            out.append("cross_median")
        if "night" in search_text or "nighttime" in tags:
            out.append("nighttime")
        if "wet" in search_text or "wet_road" in tags:
            out.append("wet_road")
        if "head on" in search_text or "head-on" in search_text or "sideswipe" in search_text:
            out.append("head_on_sideswipe")

        seen_names: set[str] = set()
        deduped: list[str] = []
        for name in out:
            if name not in seen_names:
                deduped.append(name)
                seen_names.add(name)
        return deduped

    def query_for_target_pool(query: Query, pool_name: str) -> Query:
        if pool_name == "pedestrian":
            return replace(query, crash_type="Vehicle/pedestrian")
        if pool_name == "bicycle":
            return replace(query, crash_type="Vehicle/bicycle")
        if pool_name == "cross_median":
            return replace(query, crash_type="Cross median")
        if pool_name == "nighttime":
            return replace(query, crash_type="Nighttime")
        if pool_name == "wet_road":
            return replace(query, crash_type="Wet road")
        if pool_name == "head_on_sideswipe":
            return replace(query, crash_type="Head on,Sideswipe")
        return query

    def family_slot_names(query: Query, user_text: str | None) -> list[str]:
        crash = (query.crash_type or "").lower()
        raw_text = (user_text or "").lower()
        search_text = f"{crash} {raw_text}"
        is_intersection = _norm_key(query.intersection_related) == "yes"
        is_segment = _norm_key(query.intersection_related) == "no"
        is_signalized = _norm_key(query.traffic_control_type) == "signalized"
        out: list[str] = []

        if "ped" in search_text:
            if is_segment:
                out.extend(["ped_sidewalk", "ped_crosswalk", "ped_rrfb", "ped_hybrid_beacon", "ped_refuge", "speed_management", "lighting"])
            elif is_signalized:
                out.extend(["ped_signal_timing", "ped_crosswalk", "ped_refuge", "lighting"])
            else:
                out.extend(["ped_crosswalk", "ped_rrfb", "ped_hybrid_beacon", "ped_signal_timing", "ped_refuge", "speed_management"])
        if "bicycle" in search_text or "bike" in search_text:
            out.extend(["bike_lane", "cycle_track", "speed_management"])
            if is_intersection:
                out.extend(["bike_crossing", "signal_phasing"])
        if "wet" in search_text:
            out.extend(["hfst", "pavement_friction"])
        if "night" in search_text:
            out.extend(["lighting"])
        if "run off road" in search_text or "run-off-road" in search_text or "road departure" in search_text:
            out.extend(["rumble_strips", "roadside_recovery", "hfst", "pavement_friction"])
        if "rear end" in search_text or "rear-end" in search_text:
            if is_segment:
                out.extend(["left_turn_lane", "pavement_friction", "speed_management"])
            elif is_signalized:
                out.extend(["signal_phasing", "left_turn_lane", "pavement_friction"])
        if "cross median" in search_text or "cross-median" in search_text or "head on" in search_text or "head-on" in search_text:
            out.extend(["median_barrier", "rumble_strips"])

        def add_mechanisms(*names: str) -> None:
            out.extend(f"mechanism:{name}" for name in names)

        if "angle" in search_text:
            add_mechanisms(
                "intersection_control_roundabout",
                "intersection_control_signalization",
                "intersection_warning_conflict_system",
                "intersection_advance_guidance_signing",
                "intersection_stop_sign_visibility_signing",
                "intersection_warning_flashing_beacon",
                "signal_visibility",
                "access_management_median_opening",
                "access_management_driveway",
            )
        if "left turn" in search_text or "left-turn" in search_text:
            add_mechanisms(
                "intersection_left_turn_lane",
                "intersection_left_turn_phasing",
                "intersection_left_turn_offset",
                "intersection_restricted_crossing_u_turn",
                "access_management_raised_median",
                "access_management_median_opening",
            )
        if "rear end" in search_text or "rear-end" in search_text:
            add_mechanisms(
                "signal_change_interval",
                "signal_operations_adaptive_control",
                "intersection_left_turn_lane",
                "access_management_driveway",
                "speed_enforcement",
                "pavement_friction",
            )
        if "run off road" in search_text or "run-off-road" in search_text or "lane departure" in search_text:
            add_mechanisms(
                "lane_departure_rumble_combined",
                "lane_departure_centerline_rumble",
                "lane_departure_shoulder_rumble",
                "lane_departure_safety_edge",
                "shoulder_improvement_pave",
                "shoulder_improvement_widen",
                "curve_guidance_delineation",
                "pavement_friction",
                "roadside_guardrail",
                "roadside_fixed_object_removal",
                "roadside_sideslope_improvement",
                "median_barrier",
            )
        if "night" in search_text:
            add_mechanisms("lighting", "night_visibility_pavement_markers", "curve_guidance_delineation", "signal_visibility")
        if "wet" in search_text or "friction" in search_text:
            add_mechanisms("pavement_friction")
        if "ped" in search_text:
            add_mechanisms(
                "active_transport_uncontrolled_crossing_beacon",
                "active_transport_uncontrolled_crossing_refuge",
                "active_transport_uncontrolled_crossing_traffic_calming",
                "active_transport_pedestrian_signal",
                "active_transport_pedestrian_signal_timing",
                "active_transport_sidewalk",
            )
        if "bicycle" in search_text or "bike" in search_text:
            add_mechanisms(
                "active_transport_bicycle_lane",
                "active_transport_bicycle_track",
                "active_transport_bicycle_crossing_speed_control",
            )
        if "speed" in search_text:
            add_mechanisms(
                "speed_enforcement",
                "speed_management_posted_limit",
                "speed_management_variable_speed_limit",
                "speed_management_traffic_calming",
            )
        if "multiple" in search_text or "mixed" in search_text:
            add_mechanisms(
                "access_management_driveway",
                "access_management_raised_median",
                "lane_reconfiguration_road_diet",
                "signal_operations_adaptive_control",
                "speed_enforcement",
            )
        if is_segment and any(term in search_text for term in ("freeway", "expressway", "interstate")):
            add_mechanisms(
                "median_barrier",
                "lane_departure_shoulder_rumble",
                "shoulder_improvement_widen",
                "speed_management_variable_speed_limit",
                "roadside_fixed_object_removal",
            )

        seen: set[str] = set()
        deduped: list[str] = []
        for name in out:
            if name in seen:
                continue
            if not family_pools.get(name):
                continue
            seen.add(name)
            deduped.append(name)
        return deduped

    def candidates_for_field(field: str, value: str | None) -> set[int] | None:
        if not value or _is_generic(value):
            return None
        s: set[int] = set()
        for v0 in _split_multi(value):
            v = v0.strip().lower()
            s.update(idx[field].get(v, []))
        # also add generic/unknown in this field (keeps broader evidence)
        if field == "crash_type":
            for g in crash_generic:
                s.update(idx[field].get(g, []))
        elif field == "area_type":
            for g in area_generic:
                s.update(idx[field].get(g, []))
        elif field == "traffic_control_type":
            for g in ctrl_generic:
                s.update(idx[field].get(g, []))
            s.update(idx[field].get("", []))
        elif field == "intersection_geometry":
            for g in geom_generic:
                s.update(idx[field].get(g, []))
            s.update(idx[field].get("", []))
        else:
            s.update(idx[field].get("", []))
        return s

    written = 0
    with open(args.queries, "r", encoding="utf-8") as fin, open(
        args.out, "w", encoding="utf-8"
    ) as fout:
        for input_index, line in enumerate(fin):
            if input_index < int(args.start_query):
                continue
            q = json.loads(line)
            user_text = q.get("user_text") or ""

            if args.use_target_context:
                ctx = q.get("target_context") or {}
                query = _to_query_from_context(ctx, user_text=str(user_text))
                parsed = {
                    "mode": "visible_target_context_plus_user_text_fallback",
                    "context": ctx,
                    "hidden_or_paired_context_used": False,
                }
            elif use_llm_parser:
                parsed_obj = _parse_context_with_llm(
                    llm_tok,
                    llm_model,
                    str(user_text),
                    max_new_tokens=args.llm_parser_max_new_tokens,
                )
                ctx = parsed_obj["parsed_context"]
                query = _to_query_from_context(ctx, user_text=str(user_text))
                parsed = {
                    "mode": "llm_parser_adapter",
                    "context": ctx,
                    "missing_fields": parsed_obj.get("missing_fields") or [],
                    "uncertainties": parsed_obj.get("uncertainties") or [],
                }
            else:
                inferred = infer_query_from_text(str(user_text))
                query = inferred.query
                parsed = {
                    "mode": "parsed_from_text",
                    "context": {
                        "crash_type": query.crash_type,
                        "area_type": query.area_type,
                        "facility_type": query.facility_type,
                        "intersection_related": query.intersection_related,
                        "traffic_control_type": query.traffic_control_type,
                        "intersection_geometry": query.intersection_geometry,
                        "min_speed_limit": query.min_speed_limit,
                        "max_speed_limit": query.max_speed_limit,
                        "num_lanes": query.num_lanes,
                        "traffic_volume_aadt": query.traffic_volume_aadt,
                    },
                    "notes": inferred.notes,
                }

            # Initial context-aware retrieval. In the ablation, the entire
            # evidence store is ranked by effect and evidence quality only.
            pool: set[int] | None = None
            if not args.disable_context_retrieval:
                for field, val in [
                    ("crash_type", query.crash_type),
                    ("area_type", query.area_type),
                    ("intersection_related", query.intersection_related),
                    ("traffic_control_type", query.traffic_control_type),
                    ("intersection_geometry", query.intersection_geometry),
                ]:
                    cand = candidates_for_field(field, val)
                    if cand is None:
                        continue
                    pool = cand if pool is None else pool.intersection(cand)

            if args.disable_context_retrieval or pool is None or not pool:
                pool_list = all_indices
            else:
                pool_list = list(pool)

            scoring_query = Query() if args.disable_context_retrieval else query
            mechanism_recall_cfg = MechanismRecallConfig(
                apply_applicability_filter=not args.disable_applicability_filtering
            )

            def applicability_conflict(ev: dict) -> bool:
                if args.disable_applicability_filtering or not user_text:
                    return False
                return hidden_precondition_mismatch(ev, str(user_text)) or candidate_context_mismatch(
                    ev, str(user_text), query
                )

            provenance: dict[str, set[str]] = defaultdict(set)

            if args.seed_pedestrian_category:
                qt = (query.crash_type or "").lower()
                if "ped" in qt:  # matches "vehicle/pedestrian" and other pedestrian strings
                    pool_list = list(set(pool_list).union(ped_category_pool))

            matched_target_pools = (
                targeted_pool_names(query, str(user_text)) if args.seed_targeted_categories else []
            )
            matched_family_slots = (
                family_slot_names(query, str(user_text)) if int(args.force_include_family_slots) > 0 else []
            )
            if matched_target_pools:
                target_union: set[int] = set()
                for pool_name in matched_target_pools:
                    pool_indices = targeted_pools.get(pool_name, set())
                    target_union.update(pool_indices)
                    if args.include_candidate_provenance:
                        for ev_idx in pool_indices:
                            eid = evidences[ev_idx].get("evidence_id")
                            if eid is not None:
                                provenance[str(eid)].add(f"targeted_pool:{pool_name}")
                if target_union:
                    pool_list = list(set(pool_list).union(target_union))
            if matched_family_slots:
                family_union: set[int] = set()
                for family_name in matched_family_slots:
                    pool_indices = family_pools.get(family_name, set())
                    family_union.update(pool_indices)
                    if args.include_candidate_provenance:
                        for ev_idx in pool_indices:
                            eid = evidences[ev_idx].get("evidence_id")
                            if eid is not None:
                                provenance[str(eid)].add(f"family_pool:{family_name}")
                if family_union:
                    pool_list = list(set(pool_list).union(family_union))

            # Light scoring to pick a small-ish Top-N pool.
            if args.score_sample > 0 and len(pool_list) > args.score_sample:
                pool_list = random.sample(pool_list, args.score_sample)

            scored: list[tuple[float, int]] = []
            for ev_idx in pool_list:
                if applicability_conflict(evidences[ev_idx]):
                    continue
                s = score_row(score_rows[ev_idx], scoring_query, match_mode=args.match_mode, cfg=cfg)
                if s is None or float(s.total_score) <= 0:
                    continue
                scored.append((s.total_score, ev_idx))

            scored.sort(key=lambda t: t[0], reverse=True)
            score_by_eid = {
                str(evidences[i].get("evidence_id")): float(score)
                for score, i in scored
                if evidences[i].get("evidence_id") is not None
            }
            scored_eids = [evidences[i].get("evidence_id") for _, i in scored]
            top = [str(x) for x in scored_eids if x is not None]
            if args.include_candidate_provenance:
                for eid in top:
                    provenance[str(eid)].add("field_scored")
            top = _dedup_evidence_ids(
                top,
                evidence_by_id=evidence_by_id,
                mode=str(args.candidate_dedup_key),
                limit=int(args.top_n),
                scores_by_eid=score_by_eid,
                quality_tiebreak=bool(args.dedup_quality_tiebreak),
                quality_tiebreak_margin=float(args.dedup_quality_tiebreak_margin),
            )

            # For pedestrian crash queries, force-include high-quality pedestrian-category evidences.
            if args.force_include_ped_top and ped_category_pool:
                qt = (query.crash_type or "").lower()
                if "ped" in qt:
                    ped_scored: list[tuple[float, int]] = []
                    for ev_idx in ped_category_pool:
                        if applicability_conflict(evidences[ev_idx]):
                            continue
                        s = score_row(score_rows[ev_idx], query, match_mode=args.match_mode, cfg=cfg)
                        if s is None or float(s.total_score) <= 0:
                            continue
                        ped_scored.append((s.total_score, int(ev_idx)))
                    ped_scored.sort(key=lambda t: t[0], reverse=True)
                    forced = [
                        str(evidences[i].get("evidence_id"))
                        for _, i in ped_scored[: int(args.force_include_ped_top)]
                        if evidences[i].get("evidence_id") is not None
                    ]
                    if args.include_candidate_provenance:
                        for eid in forced:
                            provenance[str(eid)].add("pedestrian_category_top")
                    top = _dedup_evidence_ids(
                        [*forced, *top],
                        evidence_by_id=evidence_by_id,
                        mode=str(args.candidate_dedup_key),
                        limit=int(args.top_n),
                        scores_by_eid=score_by_eid,
                        quality_tiebreak=bool(args.dedup_quality_tiebreak),
                        quality_tiebreak_margin=float(args.dedup_quality_tiebreak_margin),
                    )

            if args.force_include_targeted_top and matched_target_pools:
                target_scores: dict[int, float] = {}
                for pool_name in matched_target_pools:
                    target_query = query_for_target_pool(query, pool_name)
                    for ev_idx in targeted_pools.get(pool_name, set()):
                        if applicability_conflict(evidences[ev_idx]):
                            continue
                        s = score_row(score_rows[ev_idx], target_query, match_mode=args.match_mode, cfg=cfg)
                        if s is None or float(s.total_score) <= 0:
                            continue
                        target_scores[int(ev_idx)] = max(float(s.total_score), target_scores.get(int(ev_idx), -1e18))
                target_scored = [(score, ev_idx) for ev_idx, score in target_scores.items()]
                target_scored.sort(key=lambda t: t[0], reverse=True)
                forced = [
                    str(evidences[i].get("evidence_id"))
                    for _, i in target_scored[: int(args.force_include_targeted_top)]
                    if evidences[i].get("evidence_id") is not None
                ]
                if args.include_candidate_provenance:
                    forced_set = set(forced)
                    for pool_name in matched_target_pools:
                        target_query = query_for_target_pool(query, pool_name)
                        pool_scored: list[tuple[float, int]] = []
                        for ev_idx in targeted_pools.get(pool_name, set()):
                            if applicability_conflict(evidences[ev_idx]):
                                continue
                            s = score_row(score_rows[ev_idx], target_query, match_mode=args.match_mode, cfg=cfg)
                            if s is None or float(s.total_score) <= 0:
                                continue
                            pool_scored.append((float(s.total_score), int(ev_idx)))
                        pool_scored.sort(key=lambda t: t[0], reverse=True)
                        for _, i in pool_scored[: int(args.force_include_targeted_top)]:
                            eid = evidences[i].get("evidence_id")
                            if eid is not None and str(eid) in forced_set:
                                provenance[str(eid)].add(f"targeted_top:{pool_name}")
                top = _dedup_evidence_ids(
                    [*forced, *top],
                    evidence_by_id=evidence_by_id,
                    mode=str(args.candidate_dedup_key),
                    limit=int(args.top_n),
                    scores_by_eid=score_by_eid,
                    quality_tiebreak=bool(args.dedup_quality_tiebreak),
                    quality_tiebreak_margin=float(args.dedup_quality_tiebreak_margin),
                )

            if int(args.force_include_family_slots) > 0 and matched_family_slots:
                forced_by_family: list[str] = []
                for family_name in matched_family_slots:
                    family_scored: list[tuple[float, int]] = []
                    # Score family slots under the original query. This preserves facility/control constraints
                    # while ensuring each expected mechanism class has a chance to enter the candidate pool.
                    for ev_idx in family_pools.get(family_name, set()):
                        if applicability_conflict(evidences[ev_idx]):
                            continue
                        s = score_row(score_rows[ev_idx], query, match_mode=args.match_mode, cfg=cfg)
                        if s is None or float(s.total_score) <= 0:
                            continue
                        family_scored.append((float(s.total_score), int(ev_idx)))
                    family_scored.sort(key=lambda t: t[0], reverse=True)
                    for _, i in family_scored[: int(args.force_include_family_slots)]:
                        eid = evidences[i].get("evidence_id")
                        if eid is not None:
                            eid_s = str(eid)
                            forced_by_family.append(eid_s)
                            if args.include_candidate_provenance:
                                provenance[eid_s].add(f"family_slot:{family_name}")
                top = _dedup_evidence_ids(
                    [*forced_by_family, *top],
                    evidence_by_id=evidence_by_id,
                    mode=str(args.candidate_dedup_key),
                    limit=int(args.top_n),
                    scores_by_eid=score_by_eid,
                    quality_tiebreak=bool(args.dedup_quality_tiebreak),
                    quality_tiebreak_margin=float(args.dedup_quality_tiebreak_margin),
                )

            if int(args.force_include_roadside_mechanism_top) > 0:
                roadside_forced = roadside_mechanism_supplement_ids(
                    user_text=str(user_text),
                    query=query,
                    evidences=evidences,
                    score_rows=score_rows,
                    existing_ids=set(top),
                    cfg=mechanism_recall_cfg,
                )[: int(args.force_include_roadside_mechanism_top)]
                if args.include_candidate_provenance:
                    for eid in roadside_forced:
                        provenance[str(eid)].add("mechanism_recall:roadside")
                top = _dedup_evidence_ids(
                    [*roadside_forced, *top],
                    evidence_by_id=evidence_by_id,
                    mode=str(args.candidate_dedup_key),
                    limit=int(args.top_n),
                    scores_by_eid=score_by_eid,
                    quality_tiebreak=bool(args.dedup_quality_tiebreak),
                    quality_tiebreak_margin=float(args.dedup_quality_tiebreak_margin),
                )

            if int(args.force_include_nighttime_segment_mechanism_top) > 0:
                night_forced = nighttime_segment_supplement_ids(
                    user_text=str(user_text),
                    query=query,
                    evidences=evidences,
                    score_rows=score_rows,
                    existing_ids=set(top),
                    cfg=mechanism_recall_cfg,
                )[: int(args.force_include_nighttime_segment_mechanism_top)]
                if args.include_candidate_provenance:
                    for eid in night_forced:
                        provenance[str(eid)].add("mechanism_recall:nighttime_segment")
                top = _dedup_evidence_ids(
                    [*night_forced, *top],
                    evidence_by_id=evidence_by_id,
                    mode=str(args.candidate_dedup_key),
                    limit=int(args.top_n),
                    scores_by_eid=score_by_eid,
                    quality_tiebreak=bool(args.dedup_quality_tiebreak),
                    quality_tiebreak_margin=float(args.dedup_quality_tiebreak_margin),
                )

            if int(args.force_include_access_management_mechanism_top) > 0:
                access_forced = access_management_supplement_ids(
                    user_text=str(user_text),
                    query=query,
                    evidences=evidences,
                    score_rows=score_rows,
                    existing_ids=set(top),
                    cfg=mechanism_recall_cfg,
                )[: int(args.force_include_access_management_mechanism_top)]
                if args.include_candidate_provenance:
                    for eid in access_forced:
                        provenance[str(eid)].add("mechanism_recall:access_management")
                top = _dedup_evidence_ids(
                    [*access_forced, *top],
                    evidence_by_id=evidence_by_id,
                    mode=str(args.candidate_dedup_key),
                    limit=int(args.top_n),
                    scores_by_eid=score_by_eid,
                    quality_tiebreak=bool(args.dedup_quality_tiebreak),
                    quality_tiebreak_margin=float(args.dedup_quality_tiebreak_margin),
                )

            if int(args.force_include_signal_visibility_mechanism_top) > 0:
                signal_forced = signal_visibility_supplement_ids(
                    user_text=str(user_text),
                    query=query,
                    evidences=evidences,
                    score_rows=score_rows,
                    existing_ids=set(top),
                    cfg=mechanism_recall_cfg,
                )[: int(args.force_include_signal_visibility_mechanism_top)]
                if args.include_candidate_provenance:
                    for eid in signal_forced:
                        provenance[str(eid)].add("mechanism_recall:signal_visibility")
                top = _dedup_evidence_ids(
                    [*signal_forced, *top],
                    evidence_by_id=evidence_by_id,
                    mode=str(args.candidate_dedup_key),
                    limit=int(args.top_n),
                    scores_by_eid=score_by_eid,
                    quality_tiebreak=bool(args.dedup_quality_tiebreak),
                    quality_tiebreak_margin=float(args.dedup_quality_tiebreak_margin),
                )

            if int(args.force_include_signalized_left_turn_mechanism_top) > 0:
                signal_left_forced = signalized_left_turn_supplement_ids(
                    user_text=str(user_text),
                    query=query,
                    evidences=evidences,
                    score_rows=score_rows,
                    existing_ids=set(top),
                    cfg=mechanism_recall_cfg,
                )[: int(args.force_include_signalized_left_turn_mechanism_top)]
                if args.include_candidate_provenance:
                    for eid in signal_left_forced:
                        provenance[str(eid)].add("mechanism_recall:signalized_left_turn")
                top = _dedup_evidence_ids(
                    [*signal_left_forced, *top],
                    evidence_by_id=evidence_by_id,
                    mode=str(args.candidate_dedup_key),
                    limit=int(args.top_n),
                    scores_by_eid=score_by_eid,
                    quality_tiebreak=bool(args.dedup_quality_tiebreak),
                    quality_tiebreak_margin=float(args.dedup_quality_tiebreak_margin),
                )

            if int(args.force_include_stop_control_mechanism_top) > 0:
                stop_forced = stop_control_supplement_ids(
                    user_text=str(user_text),
                    query=query,
                    evidences=evidences,
                    score_rows=score_rows,
                    existing_ids=set(top),
                    cfg=mechanism_recall_cfg,
                )[: int(args.force_include_stop_control_mechanism_top)]
                if args.include_candidate_provenance:
                    for eid in stop_forced:
                        provenance[str(eid)].add("mechanism_recall:stop_control")
                top = _dedup_evidence_ids(
                    [*stop_forced, *top],
                    evidence_by_id=evidence_by_id,
                    mode=str(args.candidate_dedup_key),
                    limit=int(args.top_n),
                    scores_by_eid=score_by_eid,
                    quality_tiebreak=bool(args.dedup_quality_tiebreak),
                    quality_tiebreak_margin=float(args.dedup_quality_tiebreak_margin),
                )

            if int(args.force_include_median_opening_mechanism_top) > 0:
                median_forced = median_opening_supplement_ids(
                    user_text=str(user_text),
                    query=query,
                    evidences=evidences,
                    score_rows=score_rows,
                    existing_ids=set(top),
                    cfg=mechanism_recall_cfg,
                )[: int(args.force_include_median_opening_mechanism_top)]
                if args.include_candidate_provenance:
                    for eid in median_forced:
                        provenance[str(eid)].add("mechanism_recall:median_opening")
                top = _dedup_evidence_ids(
                    [*median_forced, *top],
                    evidence_by_id=evidence_by_id,
                    mode=str(args.candidate_dedup_key),
                    limit=int(args.top_n),
                    scores_by_eid=score_by_eid,
                    quality_tiebreak=bool(args.dedup_quality_tiebreak),
                    quality_tiebreak_margin=float(args.dedup_quality_tiebreak_margin),
                )

            if int(args.force_include_passing_lane_mechanism_top) > 0:
                passing_forced = passing_lane_supplement_ids(
                    user_text=str(user_text),
                    query=query,
                    evidences=evidences,
                    score_rows=score_rows,
                    existing_ids=set(top),
                    cfg=mechanism_recall_cfg,
                )[: int(args.force_include_passing_lane_mechanism_top)]
                if args.include_candidate_provenance:
                    for eid in passing_forced:
                        provenance[str(eid)].add("mechanism_recall:passing_lane")
                top = _dedup_evidence_ids(
                    [*passing_forced, *top],
                    evidence_by_id=evidence_by_id,
                    mode=str(args.candidate_dedup_key),
                    limit=int(args.top_n),
                    scores_by_eid=score_by_eid,
                    quality_tiebreak=bool(args.dedup_quality_tiebreak),
                    quality_tiebreak_margin=float(args.dedup_quality_tiebreak_margin),
                )

            if int(args.force_include_interchange_merge_mechanism_top) > 0:
                # Re-score the whole store, including IDs already present in
                # the generic list, then move the best ramp-specific families
                # into the teacher-visible prefix. Merely appending an already
                # retrieved acceleration-lane card at rank 55 does not solve a
                # Top-30 truncation failure.
                interchange_forced = interchange_merge_supplement_ids(
                    user_text=str(user_text),
                    query=query,
                    evidences=evidences,
                    score_rows=score_rows,
                    existing_ids=set(),
                    cfg=mechanism_recall_cfg,
                )[: int(args.force_include_interchange_merge_mechanism_top)]
                if args.include_candidate_provenance:
                    for eid in interchange_forced:
                        provenance[str(eid)].add("mechanism_recall:interchange_merge")
                top = _dedup_evidence_ids(
                    [*interchange_forced, *top],
                    evidence_by_id=evidence_by_id,
                    mode=str(args.candidate_dedup_key),
                    limit=int(args.top_n),
                    scores_by_eid=score_by_eid,
                    quality_tiebreak=bool(args.dedup_quality_tiebreak),
                    quality_tiebreak_margin=float(args.dedup_quality_tiebreak_margin),
                )

            extra_mechanism_specs = [
                (
                    "advance_guidance",
                    int(args.force_include_advance_guidance_mechanism_top),
                    advance_guidance_supplement_ids,
                ),
                (
                    "ped_crossing",
                    int(args.force_include_ped_crossing_mechanism_top),
                    pedestrian_crossing_supplement_ids,
                ),
                (
                    "skew_realignment",
                    int(args.force_include_skew_realignment_mechanism_top),
                    skew_realignment_supplement_ids,
                ),
                (
                    "frontage_road",
                    int(args.force_include_frontage_road_mechanism_top),
                    frontage_road_supplement_ids,
                ),
                (
                    "managed_lane",
                    int(args.force_include_managed_lane_mechanism_top),
                    managed_lane_supplement_ids,
                ),
                (
                    "winter_weather",
                    int(args.force_include_winter_weather_mechanism_top),
                    winter_weather_supplement_ids,
                ),
                (
                    "shoulder_improvement",
                    int(args.force_include_shoulder_improvement_mechanism_top),
                    shoulder_improvement_supplement_ids,
                ),
                (
                    "toll_plaza",
                    int(args.force_include_toll_plaza_mechanism_top),
                    toll_plaza_supplement_ids,
                ),
            ]
            for mechanism_name, mechanism_top, mechanism_fn in extra_mechanism_specs:
                if mechanism_top <= 0:
                    continue
                forced = mechanism_fn(
                    user_text=str(user_text),
                    query=query,
                    evidences=evidences,
                    score_rows=score_rows,
                    existing_ids=set(top),
                    cfg=mechanism_recall_cfg,
                )[:mechanism_top]
                if args.include_candidate_provenance:
                    for eid in forced:
                        provenance[str(eid)].add(f"mechanism_recall:{mechanism_name}")
                top = _dedup_evidence_ids(
                    [*forced, *top],
                    evidence_by_id=evidence_by_id,
                    mode=str(args.candidate_dedup_key),
                    limit=int(args.top_n),
                    scores_by_eid=score_by_eid,
                    quality_tiebreak=bool(args.dedup_quality_tiebreak),
                    quality_tiebreak_margin=float(args.dedup_quality_tiebreak_margin),
                )

            if args.force_include_source_evidence:
                source_eid = q.get("source_evidence_id")
                if source_eid is not None and str(source_eid) in evidence_by_id:
                    source_eid_s = str(source_eid)
                    if args.include_candidate_provenance:
                        provenance[source_eid_s].add("source_evidence")
                    source_key = None
                    if str(args.candidate_dedup_key) != "none":
                        source_key = _candidate_dedup_key(
                            evidence_by_id[source_eid_s], str(args.candidate_dedup_key)
                        )
                    top_without_source = []
                    for eid in top:
                        eid_s = str(eid)
                        if eid_s == source_eid_s:
                            continue
                        if source_key is not None:
                            ev = evidence_by_id.get(eid_s)
                            if ev is not None and _candidate_dedup_key(ev, str(args.candidate_dedup_key)) == source_key:
                                continue
                        top_without_source.append(eid_s)
                    insert_at = max(0, min(int(args.force_include_source_rank) - 1, len(top_without_source)))
                    source_inserted = [
                        *top_without_source[:insert_at],
                        source_eid_s,
                        *top_without_source[insert_at:],
                    ]
                    # Keep the exact source row. This is intentionally quality_tiebreak=False:
                    # coverage-boost queries are created from a specific evidence row, so replacing it
                    # with a near-duplicate would defeat the audit/coverage purpose.
                    top = _dedup_evidence_ids(
                        source_inserted,
                        evidence_by_id=evidence_by_id,
                        mode=str(args.candidate_dedup_key),
                        limit=int(args.top_n),
                        scores_by_eid=score_by_eid,
                        quality_tiebreak=False,
                    )

            # Every supplemental path must obey the same visible-text
            # applicability filters as the main scored pool. This final guard
            # prevents a mechanism supplement from reintroducing a card that
            # was correctly rejected earlier.
            if user_text and not args.disable_applicability_filtering:
                top = [
                    str(eid)
                    for eid in top
                    if str(eid) in evidence_by_id
                    and not hidden_precondition_mismatch(evidence_by_id[str(eid)], str(user_text))
                    and not candidate_context_mismatch(evidence_by_id[str(eid)], str(user_text), query)
                ]

            if int(args.master_diversity_slots) > 0:
                master_diverse = _teacher_diverse_ids(
                    [*top, *[str(x) for x in scored_eids if x is not None]],
                    evidence_by_id=evidence_by_id,
                    limit=min(int(args.master_diversity_slots), int(args.top_n)),
                )
                if args.include_candidate_provenance:
                    for eid in master_diverse:
                        provenance[eid].add("master_diversity_slot")
                top = _dedup_evidence_ids(
                    [*master_diverse, *top, *[str(x) for x in scored_eids if x is not None]],
                    evidence_by_id=evidence_by_id,
                    mode="none",
                    limit=int(args.top_n),
                )

            backfill_summary = {
                "requested_minimum": int(args.min_candidates),
                "before_backfill": len(top),
                "added_by_countermeasure": 0,
                "added_by_cm_id": 0,
                "added_by_evidence_id": 0,
            }
            if int(args.min_candidates) > 0 and len(top) < int(args.min_candidates):
                ranked_fallback = [str(x) for x in scored_eids if x is not None]
                for mode, summary_key in (
                    ("countermeasure", "added_by_countermeasure"),
                    ("cm_id", "added_by_cm_id"),
                    ("none", "added_by_evidence_id"),
                ):
                    top, added = _append_compatible_backfill(
                        top,
                        ranked_fallback,
                        evidence_by_id=evidence_by_id,
                        mode=mode,
                        target_minimum=int(args.min_candidates),
                        hard_limit=int(args.top_n),
                    )
                    backfill_summary[summary_key] = len(added)
                    if args.include_candidate_provenance:
                        for eid in added:
                            provenance[eid].add(f"compatible_backfill:{mode}")
                    if len(top) >= int(args.min_candidates):
                        break
            backfill_summary["after_backfill"] = len(top)

            # Add random distractors (avoid duplicates).
            if args.mix_random > 0:
                existing = set(top)
                for _ in range(args.mix_random * 20):
                    ev = evidences[random.choice(all_indices)]
                    if applicability_conflict(ev):
                        continue
                    effect = ev.get("effect") or {}
                    try:
                        if effect.get("crf") is not None and float(effect["crf"]) < 0.0:
                            continue
                        if effect.get("cmf") is not None and float(effect["cmf"]) > 1.0:
                            continue
                    except Exception:
                        continue
                    eid = str(ev.get("evidence_id"))
                    if eid and eid not in existing:
                        top.append(eid)
                        existing.add(eid)
                        if args.include_candidate_provenance:
                            provenance[eid].add("random_distractor")
                    if len(top) >= args.top_n + args.mix_random:
                        break
                top = _dedup_evidence_ids(
                    top,
                    evidence_by_id=evidence_by_id,
                    mode=str(args.candidate_dedup_key),
                    limit=int(args.top_n) + int(args.mix_random),
                    scores_by_eid=score_by_eid,
                    quality_tiebreak=bool(args.dedup_quality_tiebreak),
                    quality_tiebreak_margin=float(args.dedup_quality_tiebreak_margin),
                )

            teacher_ids = (
                _teacher_diverse_ids(
                    top,
                    evidence_by_id=evidence_by_id,
                    limit=int(args.teacher_top_n),
                    allow_alternative_evidence=not bool(args.teacher_unique_treatments_only),
                )
                if int(args.teacher_top_n) > 0
                else []
            )

            rec = {
                "qid": q.get("qid"),
                "user_text": user_text,
                "parsed_context": parsed,
                "targeted_recall_pools": matched_target_pools,
                "family_recall_slots": matched_family_slots,
                "roadside_mechanism_recall": int(args.force_include_roadside_mechanism_top),
                "nighttime_segment_mechanism_recall": int(args.force_include_nighttime_segment_mechanism_top),
                "access_management_mechanism_recall": int(args.force_include_access_management_mechanism_top),
                "signal_visibility_mechanism_recall": int(args.force_include_signal_visibility_mechanism_top),
                "signalized_left_turn_mechanism_recall": int(args.force_include_signalized_left_turn_mechanism_top),
                "stop_control_mechanism_recall": int(args.force_include_stop_control_mechanism_top),
                "skew_realignment_mechanism_recall": int(args.force_include_skew_realignment_mechanism_top),
                "median_opening_mechanism_recall": int(args.force_include_median_opening_mechanism_top),
                "passing_lane_mechanism_recall": int(args.force_include_passing_lane_mechanism_top),
                "interchange_merge_mechanism_recall": int(args.force_include_interchange_merge_mechanism_top),
                "candidate_dedup_key": args.candidate_dedup_key,
                "dedup_quality_tiebreak": bool(args.dedup_quality_tiebreak),
                "dedup_quality_tiebreak_margin": args.dedup_quality_tiebreak_margin,
                "compatible_backfill": backfill_summary,
                "master_diversity_slots": int(args.master_diversity_slots),
                "candidate_evidence_ids": top,
            }
            if int(args.teacher_top_n) > 0:
                rec["teacher_top_n"] = int(args.teacher_top_n)
                rec["teacher_unique_treatments_only"] = bool(args.teacher_unique_treatments_only)
                rec["teacher_candidate_evidence_ids"] = teacher_ids
            if args.include_candidate_provenance:
                rec["candidate_provenance"] = {
                    eid: sorted(provenance.get(str(eid), {"unknown"}))
                    for eid in top
                }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            if args.max_queries and written >= args.max_queries:
                break

    print(f"Queries processed: {written}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
