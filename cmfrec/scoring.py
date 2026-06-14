from __future__ import annotations

import math
from dataclasses import dataclass

from cmfrec.facility import normalize_evidence_facility_type, normalize_facility_type  # type: ignore
from cmfrec.context_tags import (
    infer_required_context_tags,
    normalize_context_tags,
)


@dataclass(frozen=True)
class TeacherConfig:
    # Effect robustness.
    cmf_clip_min: float = 0.05
    cmf_clip_max: float = 5.0
    crf_clip_abs_max: float = 100.0
    effect_clip_min: float = -100.0
    effect_clip_max: float = 95.0

    # Evidence quality weights.
    star_missing_weight: float = 0.12
    se_missing_weight: float = 0.75
    size_missing_weight: float = 0.6

    # Standard-error penalty: se_w = 1 / (1 + se_penalty_scale * se)
    se_penalty_scale: float = 5.0
    # Sample-size normalization: log1p(n)/log1p(size_norm_max), capped at 1.
    size_norm_max: float = 5000.0

    # Guardrail: large effects with weak evidence get extra downweight.
    weak_guard_effect_abs_threshold: float = 80.0
    weak_guard_star_threshold: float = 2.0
    weak_guard_n_threshold: float = 20.0
    weak_guard_downweight: float = 0.6

    # Robust matching: global multiplier applied to per-field mismatch_factor.
    mismatch_multiplier: float = 1.0

    # Reweight matching vs effect: total = (match_score ** match_score_power) * effect * evidence_weight.
    # >1.0 makes weak matches (generic "All"/"Not specified") drop faster, which helps avoid
    # high-effect-but-off-context treatments dominating specialized contexts (e.g., pedestrian crashes).
    match_score_power: float = 1.0

    # Feasibility-aware soft penalties (do NOT hard-filter).
    # If evidence requires a divided/median facility but the query indicates a very low total lane count,
    # downweight (common failure mode: median-related treatments surfacing for 1-lane segments).
    divided_low_lanes_threshold: float = 2.1
    divided_low_lanes_penalty_1lane: float = 0.25
    divided_low_lanes_penalty_2lane: float = 0.55

    # Evidence specificity: downweight evidence rows that list many crash types.
    broad_crash_type_penalty_scale: float = 0.12

    # Pedestrian-specific sanity: median barriers (and similar) are often irrelevant to pedestrian crashes unless
    # the evidence explicitly reports vehicle/pedestrian outcomes. Downweight these to avoid high-effect generic
    # barrier rows dominating pedestrian segment queries.
    ped_barrier_penalty: float = 0.15

    # Already-signalized intersections should not be dominated by "install/convert to signal" treatments.
    # Those are useful when signalization is absent, but they are mechanism-conflicting when the user says the
    # site is already signalized.
    signalized_conversion_penalty: float = 0.12

    # For signalized pedestrian intersections, uncontrolled/midblock beacon treatments can still be relevant
    # only in special layouts, but should rank behind signal timing/phasing treatments unless the evidence itself
    # is signalized-intersection-specific.
    signalized_ped_midblock_beacon_penalty: float = 0.45


DEFAULT_TEACHER_CONFIG = TeacherConfig()


def _to_float(value: str) -> float | None:
    try:
        if value is None:
            return None
        s = str(value).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _norm_text(value: str) -> str:
    return (value or "").strip()


def _is_generic(value: str) -> bool:
    v = _norm_text(value).lower()
    return v in {"", "all", "not specified", "unknown", "n/a", "na"}


@dataclass(frozen=True)
class Query:
    crash_type: str | None = None
    severity: str | None = None
    roadway_type: str | None = None
    area_type: str | None = None
    facility_type: str | None = None
    intersection_related: str | None = None
    traffic_control_type: str | None = None
    intersection_geometry: str | None = None
    min_speed_limit: float | None = None
    max_speed_limit: float | None = None
    num_lanes: float | None = None
    traffic_volume_aadt: float | None = None
    major_road_volume_aadt: float | None = None
    minor_road_volume_aadt: float | None = None
    countermeasure_category: str | None = None
    countermeasure_subcategory: str | None = None
    min_star: float | None = None
    context_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceScore:
    match_score: float
    effect_crf: float | None
    evidence_weight: float
    total_score: float
    reasons: list[str]


def compute_effect_crf(
    row: dict[str, str], cfg: TeacherConfig | None = None
) -> tuple[float | None, list[str]]:
    cfg = cfg or DEFAULT_TEACHER_CONFIG
    reasons: list[str] = []
    cmf = _to_float(row.get("CMF", ""))
    if cmf is not None:
        if cmf < 0:
            return None, reasons
        # Some datasets contain pathological CMF values (e.g., 0). Clip to a plausible range for robustness.
        cmf_clipped = max(cfg.cmf_clip_min, min(cmf, cfg.cmf_clip_max))
        if cmf_clipped != cmf:
            reasons.append("CMF outlier clipped")
        return (1.0 - cmf_clipped) * 100.0, reasons

    crf = _to_float(row.get("CRF", ""))
    if crf is not None:
        # Bound extremes (data can contain outliers).
        cap = float(cfg.crf_clip_abs_max)
        crf_clipped = max(-cap, min(cap, crf))
        if crf_clipped != crf:
            reasons.append("CRF outlier clipped")
        return crf_clipped, reasons

    return None, reasons


def compute_evidence_weight(
    row: dict[str, str], cfg: TeacherConfig | None = None
) -> tuple[float, list[str]]:
    cfg = cfg or DEFAULT_TEACHER_CONFIG
    reasons: list[str] = []

    star = _to_float(row.get("Star Quality Rating", ""))
    if star is None or star <= 0:
        star_w = float(cfg.star_missing_weight)
        reasons.append("Star Quality Rating missing/invalid; evidence weight reduced")
    else:
        star_clamped = max(0.0, min(star, 5.0))
        star_w = 0.2 + 0.8 * (star_clamped / 5.0)

    se = (
        _to_float(row.get("Adjusted Standard Error of CMF", ""))
        or _to_float(row.get("Unadjusted Standard Error of CMF", ""))
        or _to_float(row.get("Adjusted Standard Error of CRF", ""))
        or _to_float(row.get("Unadjusted Standard Error of CRF", ""))
    )
    if se is None or se <= 0:
        se_w = float(cfg.se_missing_weight)
    else:
        # Heuristic: penalize large standard error.
        se_w = 1.0 / (1.0 + float(cfg.se_penalty_scale) * float(se))

    n_crashes = _to_float(row.get("Number of Crashes", "")) or _to_float(
        row.get("Number of Crashes Before", "")
    )
    if n_crashes is None or n_crashes <= 0:
        size_w = float(cfg.size_missing_weight)
    else:
        # Diminishing returns with sample size.
        size_w = min(1.0, math.log1p(n_crashes) / math.log1p(float(cfg.size_norm_max)))

    w = star_w * se_w * size_w
    return w, reasons


def compute_match_score(
    row: dict[str, str],
    query: Query,
    *,
    match_mode: str = "strict",
    cfg: TeacherConfig | None = None,
) -> tuple[float, list[str]]:
    """
    Soft matching:
      - exact match => 1.0
      - generic row value (All/Not specified/blank) => 0.55
      - mismatch => strict: 0.0 (exclude); robust: downweight (keep)
      - query missing for field => neutral (does not affect)
    """
    score = 1.0
    reasons: list[str] = []
    is_robust = (match_mode or "strict").lower() == "robust"
    cfg = cfg or DEFAULT_TEACHER_CONFIG

    def match(
        field_name: str,
        row_key: str,
        query_value: str | None,
        *,
        weight: float,
        generic_factor: float = 0.55,
        unknown_factor: float = 0.75,
        mismatch_factor: float = 0.25,
        allow_mismatch: bool = False,
    ) -> bool:
        nonlocal score
        if not query_value:
            return True
        rv = _norm_text(row.get(row_key, ""))
        qv = _norm_text(query_value)
        if not rv:
            # unknown in evidence row; keep but downweight
            score *= unknown_factor**weight
            reasons.append(f"{field_name}: 证据未注明，适配度下调")
            return True
        # Allow multi-valued strings like "Rear end,Sideswipe" (common in parsed free text).
        rv_parts = [p.strip() for p in rv.split(",") if p.strip()] if "," in rv else [rv]
        qv_parts = [p.strip() for p in qv.split(",") if p.strip()] if "," in qv else [qv]
        rv_set = {p.lower() for p in rv_parts}
        qv_set = {p.lower() for p in qv_parts}
        if rv_set & qv_set:
            return True
        if _is_generic(rv):
            score *= generic_factor**weight
            reasons.append(f"{field_name}: 仅给出泛化条件（{rv}）")
            return True
        # mismatch
        if is_robust and allow_mismatch:
            mf = float(mismatch_factor) * float(cfg.mismatch_multiplier)
            mf = max(0.05, min(1.0, mf))
            score *= mf**weight
            reasons.append(f"{field_name}: mismatch kept (evidence={rv}, query={qv})")
            return True
        score = 0.0
        reasons.append(f"{field_name}: 不匹配（证据={rv}，需求={qv}）")
        return False

    # Context precondition guard: reject specialized treatments (e.g., transit / rail crossing) if the user
    # did not mention the required context. This is a hard constraint (applies to both strict and robust).
    required = infer_required_context_tags(
        countermeasure=row.get("Countermeasure", ""),
        countermeasure_category=row.get("Countermeasure Category", ""),
    )
    if required:
        qtags = set(normalize_context_tags(query.context_tags))
        if not set(required).issubset(qtags):
            score = 0.0
            reasons.append(f"Context: missing required tags (required={list(required)}, query={list(qtags)})")
            return 0.0, reasons

    if not match(
        "Countermeasure Category",
        "Countermeasure Category",
        query.countermeasure_category,
        weight=0.6,
        generic_factor=0.7,
    ):
        return 0.0, reasons
    if not match(
        "Countermeasure Subcategory",
        "Countermeasure Subcategory",
        query.countermeasure_subcategory,
        weight=0.6,
        generic_factor=0.7,
    ):
        return 0.0, reasons

    if not match(
        "Crash Type",
        "Crash Type",
        query.crash_type,
        weight=1.0,
        generic_factor=0.35,
        unknown_factor=0.65,
        allow_mismatch=False,
    ):
        return 0.0, reasons

    # Evidence specificity: if the evidence row lists many crash types, treat it as less specific.
    # This helps prevent very broad, high-effect treatments from dominating when more targeted
    # crash-type-specific evidence exists.
    rv_crash = _norm_text(row.get("Crash Type", ""))
    if rv_crash and ("," in rv_crash) and (not _is_generic(rv_crash)):
        parts = [p.strip().lower() for p in rv_crash.split(",") if p.strip()]
        n_types = len(set(parts))
        if n_types > 1:
            # n=2 => ~0.89, n=6 => ~0.58, n=20 => ~0.30
            scale = float(getattr(cfg, "broad_crash_type_penalty_scale", 0.12) or 0.12)
            penalty = 1.0 / (1.0 + scale * float(n_types - 1))
            score *= penalty
            reasons.append(f"Crash Type: broad evidence downweighted (n_types={n_types})")

    # Pedestrian crash queries: de-emphasize generic median barrier treatments unless the evidence explicitly
    # covers vehicle/pedestrian crashes.
    q_crash_l = _norm_text(query.crash_type or "").lower()
    if "ped" in q_crash_l:
        cm_l = _norm_text(row.get("Countermeasure", "")).lower()
        sub_l = _norm_text(row.get("Countermeasure Subcategory", "")).lower()
        is_median_barrier = ("median barrier" in cm_l) or ("cable median barrier" in cm_l) or ("median barriers" in sub_l)
        if is_median_barrier and ("ped" not in rv_crash.lower()):
            pen = float(getattr(cfg, "ped_barrier_penalty", 0.15) or 0.15)
            pen = max(0.05, min(1.0, pen))
            score *= pen
            reasons.append("Pedestrian: generic median barrier downweighted")
    if not match(
        "Severity",
        "KABCO Crash Severity",
        query.severity,
        weight=0.7,
        generic_factor=0.5,
        mismatch_factor=0.35,
        allow_mismatch=True,
    ):
        return 0.0, reasons

    # Facility type: interchange vs at-grade intersection vs segment.
    # This is a high-impact guard because some evidence rows have very generic intersection fields
    # but the countermeasure is facility-specific (e.g., diamond interchange conversions).
    if query.facility_type:
        rv = normalize_evidence_facility_type(row)
        qv = normalize_facility_type(query.facility_type)
        if qv:
            if not rv:
                score *= 0.75 ** 1.0
                reasons.append("Facility Type: 证据未注明，适配度下调")
            elif rv == qv:
                pass
            else:
                score = 0.0
                reasons.append(f"Facility Type: 不匹配（证据={rv}，需求={qv}）")
                return 0.0, reasons
    if not match(
        "Roadway Type",
        "Roadway Type",
        query.roadway_type,
        weight=0.6,
        generic_factor=0.6,
        mismatch_factor=0.45,
        allow_mismatch=True,
    ):
        return 0.0, reasons
    if not match(
        "Area Type",
        "Area Type",
        query.area_type,
        weight=0.4,
        mismatch_factor=0.45,
        allow_mismatch=True,
    ):
        return 0.0, reasons
    if not match(
        "Intersection Related",
        "Intersection Related",
        query.intersection_related,
        weight=0.4,
        mismatch_factor=0.45,
        allow_mismatch=True,
    ):
        return 0.0, reasons
    if not match(
        "Traffic Control Type",
        "Traffic Control Type",
        query.traffic_control_type,
        weight=0.4,
        mismatch_factor=0.45,
        allow_mismatch=True,
    ):
        return 0.0, reasons

    q_ctrl_l = _norm_text(query.traffic_control_type or "").lower()
    cm_l = _norm_text(row.get("Countermeasure", "")).lower()
    row_ctrl_l = _norm_text(row.get("Traffic Control Type", "")).lower()
    row_fac_l = normalize_evidence_facility_type(row) or ""

    if q_ctrl_l == "signalized":
        converts_to_signal = (
            "install a traffic signal" in cm_l
            or "install traffic signal" in cm_l
            or "convert" in cm_l and "to signal" in cm_l
            or "signalize" in cm_l
            or "signalization" in cm_l
        )
        if converts_to_signal and not any(
            keep in cm_l for keep in ["phasing", "phase", "timing", "cycle length", "leading pedestrian interval"]
        ):
            penalty = float(getattr(cfg, "signalized_conversion_penalty", 0.12) or 0.12)
            penalty = max(0.03, min(1.0, penalty))
            score *= penalty
            reasons.append("Traffic Control Type: already signalized; signal-installation treatment downweighted")

    if (
        "ped" in q_crash_l
        and q_ctrl_l == "signalized"
        and _norm_text(query.intersection_related or "").lower() == "yes"
    ):
        is_midblock_or_unsignalized_beacon = any(
            k in cm_l
            for k in [
                "rrfb",
                "rapid flashing beacon",
                "pedestrian hybrid beacon",
                "hawk",
                "midblock",
            ]
        )
        row_not_signalized_intersection = (
            row_ctrl_l not in {"signalized", "traffic signal", "signalized control"}
            or row_fac_l == "segment"
        )
        if is_midblock_or_unsignalized_beacon and row_not_signalized_intersection:
            penalty = float(getattr(cfg, "signalized_ped_midblock_beacon_penalty", 0.45) or 0.45)
            penalty = max(0.05, min(1.0, penalty))
            score *= penalty
            reasons.append("Pedestrian signalized intersection: midblock/uncontrolled beacon downweighted")

    # Intersection geometry (3-leg / 4-leg / more than 4 legs).
    if query.intersection_geometry:
        rv = _norm_text(row.get("Intersection Geometry", ""))
        qv = _norm_text(query.intersection_geometry)
        if not rv:
            score *= 0.8 ** 0.35
            reasons.append("Intersection Geometry: 证据未注明，适配度下调")
        elif _is_generic(rv) or rv.lower() in {"no values chosen.", "no values chosen"}:
            score *= 0.65 ** 0.35
            reasons.append(f"Intersection Geometry: 仅给出泛化条件（{rv}）")
        else:
            rv_l = rv.lower()
            qv_l = qv.lower()
            if rv_l == qv_l:
                pass
            elif "," in rv_l:
                parts = [p.strip() for p in rv_l.split(",") if p.strip()]
                if qv_l not in parts:
                    if is_robust:
                        score *= 0.45 ** 0.35
                        reasons.append(f"Intersection Geometry: mismatch kept (evidence={rv}, query={qv})")
                    else:
                        score = 0.0
                        reasons.append(f"Intersection Geometry: 不匹配（证据={rv}，需求={qv}）")
                        return 0.0, reasons
            else:
                if is_robust:
                    score *= 0.45 ** 0.35
                    reasons.append(f"Intersection Geometry: mismatch kept (evidence={rv}, query={qv})")
                else:
                    score = 0.0
                    reasons.append(f"Intersection Geometry: 不匹配（证据={rv}，需求={qv}）")
                    return 0.0, reasons

    # Optional numeric match for speed limits.
    row_min = _to_float(row.get("Min Speed Limit", ""))
    row_max = _to_float(row.get("Max Speed Limit", ""))
    if query.min_speed_limit is not None and row_min is not None:
        if row_min > query.min_speed_limit:
            score *= 0.85
            reasons.append("Min Speed Limit: 证据下限高于现场，适配度下调")
    if query.max_speed_limit is not None and row_max is not None:
        if row_max < query.max_speed_limit:
            score *= 0.85
            reasons.append("Max Speed Limit: 证据上限低于现场，适配度下调")

    # Optional numeric match for lanes.
    lanes_min = _to_float(row.get("Min Num Lanes", ""))
    lanes_max = _to_float(row.get("Max Num Lanes", ""))
    if query.num_lanes is not None:
        if lanes_min is not None and lanes_max is not None:
            if query.num_lanes < lanes_min - 0.1 or query.num_lanes > lanes_max + 0.1:
                if is_robust:
                    score *= 0.5
                    reasons.append(
                        f"Num Lanes: mismatch kept (evidence_range={lanes_min}-{lanes_max}, query={query.num_lanes})"
                    )
                else:
                    score = 0.0
                    reasons.append(
                        f"Num Lanes: 不匹配（证据范围={lanes_min}-{lanes_max}，现场={query.num_lanes}）"
                    )
                    return 0.0, reasons
        elif lanes_min is not None and query.num_lanes < lanes_min - 0.1:
            score *= 0.85
            reasons.append("Num Lanes: 现场低于证据下限，适配度下调")
        elif lanes_max is not None and query.num_lanes > lanes_max + 0.1:
            score *= 0.85
            reasons.append("Num Lanes: 现场高于证据上限，适配度下调")

    # Roadway division type (median/divided vs undivided).
    # Many median-related treatments require a divided facility; for low lane-count segments
    # (1–2 lanes total), a divided cross-section is uncommon. Downweight such evidence to avoid
    # high-effect-but-implausible treatments dominating pedestrian segment queries.
    div = _norm_text(row.get("Roadway Division Type", ""))
    if div and query.num_lanes is not None:
        div_l = div.lower()
        if ("divided" in div_l) or ("median" in div_l):
            thr = float(getattr(cfg, "divided_low_lanes_threshold", 2.1) or 2.1)
            if query.num_lanes <= thr:
                if query.num_lanes <= 1.1:
                    penalty = float(getattr(cfg, "divided_low_lanes_penalty_1lane", 0.25) or 0.25)
                else:
                    penalty = float(getattr(cfg, "divided_low_lanes_penalty_2lane", 0.55) or 0.55)
                penalty = max(0.05, min(1.0, penalty))
                score *= penalty
                reasons.append(f"Roadway Division Type: downweighted (evidence={div}, lanes={query.num_lanes})")

    # Optional numeric match for traffic volume (AADT/ADT).
    def volume_penalty(qv: float, vmin: float | None, vmax: float | None, vavg: float | None, label: str) -> None:
        nonlocal score
        if vmin is not None and vmax is not None:
            if qv < vmin * 0.8 or qv > vmax * 1.2:
                score *= 0.8
                reasons.append(f"{label}: 现场流量超出证据范围较多，适配度下调")
            elif qv < vmin or qv > vmax:
                score *= 0.9
                reasons.append(f"{label}: 现场流量略超证据范围，适配度下调")
        elif vavg is not None:
            ratio = qv / vavg if vavg > 0 else 1.0
            if ratio < 0.5 or ratio > 2.0:
                score *= 0.85
                reasons.append(f"{label}: 现场流量与证据均值差异较大，适配度下调")

    if query.traffic_volume_aadt is not None:
        vmin = _to_float(row.get("Minimum Traffic Volume (non-intersection)", ""))
        vmax = _to_float(row.get("Maximum Traffic Volume (non-intersection)", ""))
        vavg = _to_float(row.get("Average Traffic Volume (non-intersection)", ""))
        volume_penalty(query.traffic_volume_aadt, vmin, vmax, vavg, "Traffic Volume (non-intersection)")

    if query.major_road_volume_aadt is not None:
        vmin = _to_float(row.get("Minimum Major Road Traffic Volume (intersection)", ""))
        vmax = _to_float(row.get("Maximum Major Road Traffic Volume (intersection)", ""))
        vavg = _to_float(row.get("Average Major Road Traffic Volume (intersection)", ""))
        volume_penalty(query.major_road_volume_aadt, vmin, vmax, vavg, "Major Road Volume (intersection)")

    if query.minor_road_volume_aadt is not None:
        vmin = _to_float(row.get("Minimum Minor Road Traffic Volume (intersection)", ""))
        vmax = _to_float(row.get("Maximum Minor Road Traffic Volume (intersection)", ""))
        vavg = _to_float(row.get("Average Minor Road Traffic Volume (intersection)", ""))
        volume_penalty(query.minor_road_volume_aadt, vmin, vmax, vavg, "Minor Road Volume (intersection)")

    return score, reasons


def score_row(
    row: dict[str, str],
    query: Query,
    *,
    match_mode: str = "strict",
    cfg: TeacherConfig | None = None,
) -> EvidenceScore | None:
    if query.min_star is not None:
        star = _to_float(row.get("Star Quality Rating", ""))
        if star is None or star < query.min_star:
            return None

    cfg = cfg or DEFAULT_TEACHER_CONFIG
    match_score, match_reasons = compute_match_score(row, query, match_mode=match_mode, cfg=cfg)
    if match_score <= 0:
        return None

    effect, effect_reasons = compute_effect_crf(row, cfg=cfg)
    if effect is None:
        return None

    evidence_w, evidence_reasons = compute_evidence_weight(row, cfg=cfg)

    # Clamp effect (CRF%) as well (some outliers exist).
    effect_clamped = max(cfg.effect_clip_min, min(cfg.effect_clip_max, float(effect)))
    if effect_clamped != effect:
        effect_reasons = [*effect_reasons, "Effect outlier clipped"]
    effect = effect_clamped

    # Extra guardrail: very large effects + weak evidence => downweight more.
    star = _to_float(row.get("Star Quality Rating", ""))
    n_crashes = _to_float(row.get("Number of Crashes", "")) or _to_float(row.get("Number of Crashes Before", ""))
    se = (
        _to_float(row.get("Adjusted Standard Error of CMF", ""))
        or _to_float(row.get("Unadjusted Standard Error of CMF", ""))
        or _to_float(row.get("Adjusted Standard Error of CRF", ""))
        or _to_float(row.get("Unadjusted Standard Error of CRF", ""))
    )
    if (
        abs(effect) >= float(cfg.weak_guard_effect_abs_threshold)
        and ((star is None) or star <= float(cfg.weak_guard_star_threshold))
        and (
            (se is None)
            or se <= 0
            or (n_crashes is not None and n_crashes < float(cfg.weak_guard_n_threshold))
        )
    ):
        evidence_w *= float(cfg.weak_guard_downweight)
        evidence_reasons = [*evidence_reasons, "Large effect with weak evidence; extra downweight"]

    mp = float(getattr(cfg, "match_score_power", 1.0) or 1.0)
    # Guard against weird configs.
    if mp <= 0:
        mp = 1.0
    total = (match_score ** mp) * effect * evidence_w
    return EvidenceScore(
        match_score=match_score,
        effect_crf=effect,
        evidence_weight=evidence_w,
        total_score=total,
        reasons=[*match_reasons, *effect_reasons, *evidence_reasons],
    )
