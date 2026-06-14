from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from cmfrec.scoring import Query, compute_match_score, score_row


def _norm_text(value: object) -> str:
    text = str(value or "").lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _effect_crf(ev: dict[str, Any]) -> float:
    effect = ev.get("effect") or {}
    try:
        if effect.get("crf") is not None:
            return float(effect.get("crf"))
    except Exception:
        pass
    try:
        if effect.get("cmf") is not None:
            return (1.0 - float(effect.get("cmf"))) * 100.0
    except Exception:
        pass
    return -999.0


def _star(ev: dict[str, Any]) -> float:
    try:
        return float((ev.get("quality") or {}).get("star_quality_rating"))
    except Exception:
        return -1.0


def _roadside_family(ev: dict[str, Any]) -> str | None:
    cm = _norm_text(ev.get("countermeasure"))
    cat = _norm_text(ev.get("countermeasure_category"))
    sub = _norm_text(ev.get("countermeasure_subcategory"))
    text = f"{cm} {cat} {sub}"

    if "utility pole" in text and "guardrail" in text:
        return "utility_pole_guardrail"
    if "side slope" in text and "guardrail" in text:
        return "side_slope_guardrail"
    if "guardrail" in text and ("embankment" in text or "roadside barrier" in text):
        return "embankment_guardrail"
    if ("remove" in text or "relocate" in text) and "fixed object" in text:
        return "fixed_object_removal"
    if "sideslope" in text or "side slope" in text or "flatten slope" in text:
        return "sideslope_improvement"
    if "lateral clearance" in text:
        return "lateral_clearance"
    if "safety edge" in text:
        return "safety_edge"
    return None


def _access_family(ev: dict[str, Any]) -> str | None:
    cm = _norm_text(ev.get("countermeasure"))
    cat = _norm_text(ev.get("countermeasure_category"))
    sub = _norm_text(ev.get("countermeasure_subcategory"))
    text = f"{cm} {cat} {sub}"

    if "driveway density" in text:
        return "driveway_density_formula"
    if "absence of access points" in text or "access points" in text and "absence" in text:
        return "access_point_control"
    if "driveway" in text or "access management" in text:
        return "driveway_or_access_management"
    if "median opening" in text:
        return "median_opening_management"
    if "replace twltl" in text and "raised median" in text:
        return "replace_twltl_with_raised_median"
    if "raised median" in text:
        return "raised_median"
    if "twltl" in text or "two way left turn lane" in text:
        return "twltl"
    return None


def _has_formula_effect(ev: dict[str, Any]) -> bool:
    effect = ev.get("effect") or {}
    if effect.get("cmf_formula"):
        return True
    # Some formula CMFs are stored with no scalar CMF/CRF and no formula field,
    # but the countermeasure text itself identifies the regression variable.
    cm = _norm_text(ev.get("countermeasure"))
    return "change in driveway density" in cm or "driveways per mile" in cm


def _access_priorities(user_text: str, query: Query) -> dict[str, float]:
    text = _norm_text(user_text)
    if not text:
        return {}
    access_density_context = any(
        term in text
        for term in [
            "driveway density",
            "frequent driveway",
            "frequent access",
            "many driveway",
            "many closely spaced",
            "closely spaced driveway",
            "closely spaced access",
            "access points",
        ]
    )
    if not access_density_context:
        return {}
    if _is_intersection(query):
        return {}

    divided = "divided" in text or "median" in text
    twltl = "twltl" in text or "two way left turn" in text or "two way left turn lane" in text

    priorities = {
        "driveway_density_formula": 130.0,
        "access_point_control": 120.0,
        "driveway_or_access_management": 110.0,
        "median_opening_management": 90.0,
        "raised_median": 70.0,
        "replace_twltl_with_raised_median": 65.0,
        "twltl": 45.0,
    }
    if divided and not twltl:
        priorities["raised_median"] = 35.0
        priorities["replace_twltl_with_raised_median"] = 20.0
        priorities["twltl"] = 15.0
    if twltl:
        priorities["replace_twltl_with_raised_median"] = 115.0
        priorities["raised_median"] = 90.0
    return priorities


def _stop_control_family(ev: dict[str, Any]) -> str | None:
    cm = _norm_text(ev.get("countermeasure"))
    cat = _norm_text(ev.get("countermeasure_category"))
    sub = _norm_text(ev.get("countermeasure_subcategory"))
    text = f"{cm} {cat} {sub}"

    if "retroreflectivity" in text and "stop" in text:
        return "stop_sign_retroreflectivity"
    if "systemic signing" in text and "stop" in text:
        return "systemic_stop_signing_marking"
    if "stop ahead" in text and "pavement" in text:
        return "stop_ahead_pavement_marking"
    if "flashing led stop" in text or ("led stop sign" in text):
        return "flashing_led_stop_sign"
    if "all way stop" in text or "allway stop" in text:
        return "all_way_stop"
    return None


def _stop_control_priorities(user_text: str, query: Query) -> dict[str, float]:
    text = _norm_text(user_text)
    if not _is_intersection(query):
        return {}
    ctrl = _norm_text(query.traffic_control_type)
    if "stop" not in ctrl and "stop controlled" not in text and "stop-controlled" not in str(user_text).lower():
        return {}
    # This recall is for stop-control operations/visibility, not for queries
    # that explicitly frame a high-volume signal-warrant situation.
    signal_warrant_context = any(term in text for term in ["signal may", "signal warrant", "warrant signal"])
    if signal_warrant_context:
        return {}
    return {
        "stop_sign_retroreflectivity": 130.0,
        "systemic_stop_signing_marking": 120.0,
        "all_way_stop": 115.0,
        "flashing_led_stop_sign": 105.0,
        "stop_ahead_pavement_marking": 95.0,
    }


def _median_opening_family(ev: dict[str, Any]) -> str | None:
    cm = _norm_text(ev.get("countermeasure"))
    cat = _norm_text(ev.get("countermeasure_category"))
    sub = _norm_text(ev.get("countermeasure_subcategory"))
    text = f"{cm} {cat} {sub}"

    if "right turn u turn" in text or "rtut" in text:
        return "right_turn_u_turn"
    if "superstreet" in text or "restricted crossing u turn" in text or "rcut" in text or "j turn" in text:
        return "superstreet_or_rcut"
    if "positive offset" in text or "left turn offset" in text or "offset left" in text:
        return "positive_offset_left_turn"
    if "left turn lane" in text or "left turn lanes" in text:
        return "left_turn_lane"
    if "median u turn" in text or "mut" in text:
        return "median_u_turn"
    return None


def _passing_lane_family(ev: dict[str, Any]) -> str | None:
    cm = _norm_text(ev.get("countermeasure"))
    cat = _norm_text(ev.get("countermeasure_category"))
    sub = _norm_text(ev.get("countermeasure_subcategory"))
    text = f"{cm} {cat} {sub}"

    if "periodic passing lane" in text:
        return "periodic_passing_lane"
    if "passing lane" in text or "climbing lane" in text:
        return "passing_or_climbing_lane"
    return None


def _passing_lane_priorities(user_text: str, query: Query) -> dict[str, float]:
    text = _norm_text(user_text)
    if _is_intersection(query):
        return {}
    if not any(
        term in text
        for term in [
            "limited passing",
            "passing opportunities",
            "risky passing",
            "overtaking",
            "slow moving vehicles",
            "climbing lane",
            "passing lane",
        ]
    ):
        return {}
    return {
        "periodic_passing_lane": 130.0,
        "passing_or_climbing_lane": 120.0,
    }


def _advance_guidance_family(ev: dict[str, Any]) -> str | None:
    cm = _norm_text(ev.get("countermeasure"))
    cat = _norm_text(ev.get("countermeasure_category"))
    sub = _norm_text(ev.get("countermeasure_subcategory"))
    text = f"{cm} {cat} {sub}"

    if "advance freeway guidance" in text or "advance guidance" in text or "guide sign" in text:
        return "advance_guidance_signing"
    if "intersection conflict warning" in text or " icws " in f" {text} ":
        return "intersection_conflict_warning"
    if "flashing beacon" in text and "intersection" in text:
        return "intersection_flashing_beacon"
    return None


def _advance_guidance_priorities(user_text: str, query: Query) -> dict[str, float]:
    text = _norm_text(user_text)
    if not _is_intersection(query):
        return {}
    if not any(
        term in text
        for term in [
            "advance guidance",
            "clearer advance guidance",
            "advance warning",
            "freeway style intersection",
            "freeway style",
            "guide sign",
            "guidance before",
            "far side of the median",
        ]
    ):
        return {}
    return {
        "advance_guidance_signing": 140.0,
        "intersection_conflict_warning": 120.0,
        "intersection_flashing_beacon": 95.0,
    }


def _ped_crossing_family(ev: dict[str, Any]) -> str | None:
    cm = _norm_text(ev.get("countermeasure"))
    cat = _norm_text(ev.get("countermeasure_category"))
    sub = _norm_text(ev.get("countermeasure_subcategory"))
    text = f"{cm} {cat} {sub}"

    if "transverse rumble" in text and ("pedestrian" in text or "crosswalk" in text):
        return "rural_crosswalk_transverse_rumble"
    if "pedestrian hybrid beacon" in text or "hawk" in text:
        return "pedestrian_hybrid_beacon"
    if "rectangular rapid flashing beacon" in text or " rrfb " in f" {text} ":
        return "rrfb"
    if "raised pedestrian crosswalk" in text or "raised crosswalk" in text:
        return "raised_crosswalk"
    if "raised median" in text and ("pedestrian" in text or "crosswalk" in text or "refuge" in text):
        return "pedestrian_refuge_median"
    return None


def _ped_crossing_priorities(user_text: str, query: Query) -> dict[str, float]:
    text = _norm_text(user_text)
    crash = _norm_text(query.crash_type)
    if "pedestrian" not in text and "vehicle pedestrian" not in crash:
        return {}
    crossing_context = any(
        term in text
        for term in ["crossing", "crosswalk", "midblock", "uncontrolled crossing", "pedestrian crossing"]
    )
    if not crossing_context:
        return {}
    rural_low_volume = "rural" in text and any(term in text for term in ["low volume", "low traffic", "low flow"])
    if rural_low_volume:
        return {
            "rural_crosswalk_transverse_rumble": 140.0,
            "rrfb": 95.0,
            "pedestrian_refuge_median": 85.0,
            "raised_crosswalk": 80.0,
            "pedestrian_hybrid_beacon": 75.0,
        }
    return {
        "pedestrian_hybrid_beacon": 130.0,
        "rrfb": 120.0,
        "pedestrian_refuge_median": 110.0,
        "raised_crosswalk": 100.0,
        "rural_crosswalk_transverse_rumble": 80.0,
    }


def _frontage_road_family(ev: dict[str, Any]) -> str | None:
    cm = _norm_text(ev.get("countermeasure"))
    text = f"{cm} {_norm_text(ev.get('countermeasure_category'))} {_norm_text(ev.get('countermeasure_subcategory'))}"
    if "frontage road" in text and "one way" in text:
        return "frontage_road_one_way"
    if "wrong way" in text and "frontage" in text:
        return "frontage_wrong_way_warning"
    return None


def _frontage_road_priorities(user_text: str, query: Query) -> dict[str, float]:
    text = _norm_text(user_text)
    if "frontage road" not in text:
        return {}
    if not any(term in text for term in ["two way", "opposite direction", "opposing direction", "left turn"]):
        return {}
    return {"frontage_road_one_way": 145.0, "frontage_wrong_way_warning": 95.0}


def _managed_lane_family(ev: dict[str, Any]) -> str | None:
    cm = _norm_text(ev.get("countermeasure"))
    text = f"{cm} {_norm_text(ev.get('countermeasure_category'))} {_norm_text(ev.get('countermeasure_subcategory'))}"
    if "convert high occupancy vehicle" in text or (" hov " in f" {text} " and " hot " in f" {text} "):
        return "hov_to_hot"
    return None


def _managed_lane_priorities(user_text: str, query: Query) -> dict[str, float]:
    text = _norm_text(user_text)
    if not any(term in text for term in ["managed lane", "managed lanes", "hov", "hot lane", "hot lanes"]):
        return {}
    return {"hov_to_hot": 145.0}


def _winter_weather_family(ev: dict[str, Any]) -> str | None:
    cm = _norm_text(ev.get("countermeasure"))
    text = f"{cm} {_norm_text(ev.get('countermeasure_category'))} {_norm_text(ev.get('countermeasure_subcategory'))}"
    if "fixed automated spray technology" in text or " fast " in f" {text} ":
        return "fast_anti_icing"
    if "snow" in text or "slush" in text or "ice" in text:
        return "snow_ice_control"
    return None


def _winter_weather_priorities(user_text: str, query: Query) -> dict[str, float]:
    text = _norm_text(user_text)
    if not any(term in text for term in ["winter", "icing", "ice", "snow", "slush", "freezing"]):
        return {}
    return {"fast_anti_icing": 145.0, "snow_ice_control": 125.0}


def _shoulder_improvement_family(ev: dict[str, Any]) -> str | None:
    cm = _norm_text(ev.get("countermeasure"))
    text = f"{cm} {_norm_text(ev.get('countermeasure_category'))} {_norm_text(ev.get('countermeasure_subcategory'))}"
    if "pave deteriorated shoulder" in text:
        return "pave_deteriorated_shoulder"
    if "pave shoulder" in text or "add new paved shoulder" in text:
        return "pave_shoulder"
    if "widen paved shoulder" in text or "widen shoulder" in text:
        return "widen_shoulder"
    return None


def _shoulder_improvement_priorities(user_text: str, query: Query) -> dict[str, float]:
    text = _norm_text(user_text)
    if not any(term in text for term in ["deteriorated shoulder", "narrow shoulder", "paved shoulder", "unpaved shoulder", "shoulder"]):
        return {}
    if _is_intersection(query):
        return {}
    if "deteriorated shoulder" in text:
        return {
            "pave_deteriorated_shoulder": 145.0,
            "pave_shoulder": 130.0,
            "widen_shoulder": 110.0,
        }
    if "narrow" in text or "limited shoulder" in text:
        return {
            "widen_shoulder": 135.0,
            "pave_shoulder": 115.0,
            "pave_deteriorated_shoulder": 90.0,
        }
    return {
        "pave_shoulder": 125.0,
        "widen_shoulder": 115.0,
        "pave_deteriorated_shoulder": 105.0,
    }


def _toll_plaza_family(ev: dict[str, Any]) -> str | None:
    cm = _norm_text(ev.get("countermeasure"))
    text = f"{cm} {_norm_text(ev.get('countermeasure_category'))} {_norm_text(ev.get('countermeasure_subcategory'))}"
    if "open road tolling" in text or " ort " in f" {text} ":
        return "open_road_tolling"
    if "all electric toll" in text or "all electronic toll" in text:
        return "all_electronic_tolling"
    if "tollbooth" in text or "toll plaza" in text:
        return "toll_plaza_conversion"
    return None


def _toll_plaza_priorities(user_text: str, query: Query) -> dict[str, float]:
    text = _norm_text(user_text)
    if not any(term in text for term in ["toll plaza", "tollbooth", "tolling", "mainline toll"]):
        return {}
    return {
        "all_electronic_tolling": 145.0,
        "open_road_tolling": 140.0,
        "toll_plaza_conversion": 125.0,
    }


def _speed_management_family(ev: dict[str, Any]) -> str | None:
    cm = _norm_text(ev.get("countermeasure"))
    text = f"{cm} {_norm_text(ev.get('countermeasure_category'))} {_norm_text(ev.get('countermeasure_subcategory'))}"
    if "automated speed" in text or "speed camera" in text or "speed enforcement" in text:
        return "automated_speed_enforcement"
    if "decreasing posted speed" in text or "posted speed limit" in text or "speed limit" in text:
        return "posted_speed_limit_management"
    return None


def _speed_management_priorities(user_text: str, query: Query) -> dict[str, float]:
    text = _norm_text(user_text)
    if not any(term in text for term in ["posted speed", "speed consistency", "operating speed", "speeding", "speed related"]):
        return {}
    return {"automated_speed_enforcement": 130.0, "posted_speed_limit_management": 120.0}


def _drowsy_family(ev: dict[str, Any]) -> str | None:
    cm = _norm_text(ev.get("countermeasure"))
    text = f"{cm} {_norm_text(ev.get('countermeasure_category'))} {_norm_text(ev.get('countermeasure_subcategory'))}"
    if "rest area" in text or "travel information center" in text:
        return "rest_area"
    return None


def _drowsy_priorities(user_text: str, query: Query) -> dict[str, float]:
    text = _norm_text(user_text)
    if not any(term in text for term in ["drowsy", "fatigue", "fatigued"]):
        return {}
    return {"rest_area": 140.0}


def _cfi_family(ev: dict[str, Any]) -> str | None:
    cm = _norm_text(ev.get("countermeasure"))
    text = f"{cm} {_norm_text(ev.get('countermeasure_category'))} {_norm_text(ev.get('countermeasure_subcategory'))}"
    if "continuous flow intersection" in text or " cfi " in f" {text} ":
        return "continuous_flow_intersection"
    if "displaced left turn" in text:
        return "displaced_left_turn"
    return None


def _cfi_priorities(user_text: str, query: Query) -> dict[str, float]:
    text = _norm_text(user_text)
    if not _is_intersection(query):
        return {}
    if not ("signalized" in text and ("left turn" in text or "queue" in text or "high volume" in text)):
        return {}
    return {"continuous_flow_intersection": 135.0, "displaced_left_turn": 125.0}


def _median_opening_priorities(user_text: str, query: Query) -> dict[str, float]:
    text = _norm_text(user_text)
    trigger = any(
        term in text
        for term in [
            "direct left turn",
            "median opening",
            "median openings",
            "full access minor road",
            "full access intersection",
            "divided four lane",
            "divided 4 lane",
            "divided roadway",
            "principal arterial",
            "crossing or turning",
            "u turn",
            "superstreet",
            "rtut",
        ]
    )
    if not trigger:
        return {}
    if not _is_intersection(query) and not any(
        term in text
        for term in [
            "median opening",
            "median openings",
            "full access minor road",
            "full access intersection",
            "divided roadway corridor",
            "direct left turn",
        ]
    ):
        return {}
    if "direct left turn" in text or "principal arterial" in text:
        return {
            "right_turn_u_turn": 135.0,
            "superstreet_or_rcut": 125.0,
            "positive_offset_left_turn": 110.0,
            "left_turn_lane": 95.0,
            "median_u_turn": 90.0,
        }
    if "limited sight distance" in text or "opposing traffic" in text or "crossing or turning" in text:
        return {
            "positive_offset_left_turn": 135.0,
            "superstreet_or_rcut": 125.0,
            "left_turn_lane": 110.0,
            "right_turn_u_turn": 100.0,
            "median_u_turn": 90.0,
        }
    return {
        "superstreet_or_rcut": 125.0,
        "right_turn_u_turn": 120.0,
        "positive_offset_left_turn": 110.0,
        "left_turn_lane": 100.0,
        "median_u_turn": 90.0,
    }


def _median_opening_min_star(family: str, user_text: str, cfg: MechanismRecallConfig) -> float:
    """Keep low-star but mechanism-specific median-opening treatments visible.

    Some positive-offset/RTUT evidence rows are low-star source-preserving rows
    but are exactly the kind of mechanism users expect to see for high-speed
    divided-road median openings. This only affects recall visibility, not final
    ranking.
    """
    text = _norm_text(user_text)
    exact_median_context = any(
        term in text
        for term in [
            "median opening",
            "direct left turn",
            "limited sight distance",
            "opposing traffic",
            "crossing or turning",
            "divided four lane",
            "divided 4 lane",
            "principal arterial",
        ]
    )
    if exact_median_context and family in {
        "positive_offset_left_turn",
        "right_turn_u_turn",
        "superstreet_or_rcut",
        "median_u_turn",
    }:
        return 1.0
    return float(cfg.min_star)


def _crash_fit_bonus(ev: dict[str, Any], query: Query) -> float:
    ev_crash = _norm_text((ev.get("conditions") or {}).get("crash_type"))
    q_crash = _norm_text(query.crash_type)
    if not ev_crash or not q_crash:
        return 0.0
    if ev_crash == q_crash:
        return 40.0
    ev_parts = {p.strip() for p in ev_crash.split(",") if p.strip()}
    q_parts = {p.strip() for p in q_crash.split(",") if p.strip()}
    if q_parts and q_parts.issubset(ev_parts):
        return 30.0
    if ev_parts and ev_parts.issubset(q_parts):
        return 20.0
    if ev_parts & q_parts:
        return 10.0
    return -20.0


def _roadway_division_fit_bonus(ev: dict[str, Any], user_text: str) -> float:
    text = _norm_text(user_text)
    div = _norm_text((ev.get("conditions") or {}).get("roadway_division_type"))
    if not div:
        return 0.0
    query_divided = "divided" in text or "median" in text
    query_twltl = "twltl" in text or "two way left turn" in text
    if query_divided and "divided by median" in div:
        return 30.0
    if query_divided and "twltl" in div:
        return -25.0
    if query_twltl and "twltl" in div:
        return 30.0
    return 0.0


def _site_context_fit_bonus(ev: dict[str, Any], query: Query, user_text: str) -> float:
    """Reward evidence rows whose site descriptors match explicit query context.

    This sits ahead of CRF for mechanism supplements so a very high-effect but
    off-slice row does not beat a lower-effect row that matches area, severity,
    and lane count.
    """
    text = _norm_text(user_text)
    cond = ev.get("conditions") or {}
    bonus = 0.0

    q_area = _norm_text(query.area_type)
    ev_area = _norm_text(cond.get("area_type"))
    if q_area and ev_area and q_area in ev_area:
        bonus += 25.0
    elif "urban" in text and "urban" in ev_area:
        bonus += 25.0
    elif "rural" in text and "rural" in ev_area:
        bonus += 25.0

    q_sev = _norm_text(query.severity)
    ev_sev = _norm_text(cond.get("severity_kabco"))
    if q_sev and ev_sev and q_sev in ev_sev:
        bonus += 25.0
    elif "fatal" in text and ("fatal" in ev_sev or ev_sev == "k"):
        bonus += 25.0
    elif "property damage" in text and ("property damage" in ev_sev or ev_sev.startswith("o")):
        bonus += 25.0

    q_lanes = query.num_lanes
    try:
        ev_min_lanes = float(cond.get("min_num_lanes")) if cond.get("min_num_lanes") is not None else None
        ev_max_lanes = float(cond.get("max_num_lanes")) if cond.get("max_num_lanes") is not None else None
    except Exception:
        ev_min_lanes = None
        ev_max_lanes = None
    if q_lanes is not None:
        if ev_min_lanes is not None and ev_max_lanes is not None and ev_min_lanes <= q_lanes <= ev_max_lanes:
            bonus += 20.0
        elif ev_min_lanes is not None and ev_max_lanes is None and abs(ev_min_lanes - q_lanes) < 0.01:
            bonus += 15.0
    elif "two lane" in text or "two-lane" in str(user_text).lower():
        if ev_min_lanes == 2 and (ev_max_lanes in {None, 2.0}):
            bonus += 20.0

    return bonus


def _signal_visibility_family(ev: dict[str, Any]) -> str | None:
    cm = _norm_text(ev.get("countermeasure"))
    cat = _norm_text(ev.get("countermeasure_category"))
    sub = _norm_text(ev.get("countermeasure_subcategory"))
    text = f"{cm} {cat} {sub}"

    if "signal lens size" in text or "back plate" in text or "backplate" in text or "reflective tape" in text:
        return "composite_signal_visibility"
    if "visibility of signal heads" in text:
        return "signal_head_visibility"
    if "signal visibility" in text:
        return "composite_signal_visibility"
    if "signal head" in text and ("visibility" in text or "additional" in text):
        return "signal_head_visibility"
    if "reflective tape" in text and "back" in text:
        return "signal_visibility"
    if "light emitting diode" in text or "led" in text and "signal" in text:
        return "led_signal_bulbs"
    if "intersection illuminance" in text or "intersection illumination" in text:
        return "intersection_illumination"
    return None


def _signal_visibility_priorities(user_text: str, query: Query) -> dict[str, float]:
    text = _norm_text(user_text)
    if not _is_intersection(query):
        return {}
    if "signal" not in text and _norm_text(query.traffic_control_type) != "signalized":
        return {}
    signal_visibility_terms = (
        "signal display",
        "signal displays",
        "signal head",
        "signal heads",
        "visibility",
        "visible",
        "noticing",
        "notice",
        "interpreting",
        "recognize",
        "detect",
    )
    night_or_visibility = "night" in text or any(term in text for term in signal_visibility_terms)
    if not night_or_visibility:
        return {}
    return {
        "composite_signal_visibility": 130.0,
        "signal_head_visibility": 120.0,
        "led_signal_bulbs": 105.0,
        "intersection_illumination": 95.0,
    }


def _signalized_left_turn_family(ev: dict[str, Any]) -> str | None:
    cm = _norm_text(ev.get("countermeasure"))
    cat = _norm_text(ev.get("countermeasure_category"))
    sub = _norm_text(ev.get("countermeasure_subcategory"))
    text = f"{cm} {cat} {sub}"

    if "left turn phas" in text or "protected permissive" in text or "protected only" in text:
        return "left_turn_phasing"
    if "offset left" in text or "left turn offset" in text or "positive offset" in text:
        return "left_turn_offset"
    if "left turn lane" in text or "left turn lanes" in text:
        return "left_turn_lane"
    if "yellow" in text and "all red" in text:
        return "signal_change_interval"
    return None


def _signalized_left_turn_priorities(user_text: str, query: Query) -> dict[str, float]:
    text = _norm_text(user_text)
    raw_text = str(user_text or "").lower()
    if not _is_intersection(query):
        return {}
    ctrl = _norm_text(query.traffic_control_type)
    is_signalized = (
        "signalized" in text
        or "signalised" in text
        or "信号灯" in raw_text
        or "信号控制" in raw_text
        or ctrl == "signalized"
    )
    if not is_signalized:
        return {}
    left_turn_context = any(
        term in text
        for term in [
            "left turn",
            "left-turn",
            "left turn phasing",
            "protected permissive",
            "permissive left",
            "offset",
            "turning conflict",
        ]
    ) or "左转" in raw_text
    if not left_turn_context:
        return {}
    return {
        "left_turn_phasing": 135.0,
        "left_turn_offset": 125.0,
        "left_turn_lane": 90.0,
        "signal_change_interval": 70.0,
    }


@dataclass(frozen=True)
class MechanismRecallConfig:
    max_supplements: int = 8
    max_per_family: int = 1
    min_crf: float = 0.0
    min_star: float = 2.0


def _is_intersection(query: Query) -> bool:
    facility = _norm_text(query.facility_type)
    intersection_related = _norm_text(query.intersection_related)
    return "intersection" in facility or intersection_related in {"yes", "true"}


def _trigger_priorities(user_text: str, query: Query) -> dict[str, float]:
    text = _norm_text(user_text)
    if not text:
        return {}

    roadside_terms = (
        "roadside",
        "slope",
        "slopes",
        "side slope",
        "sideslope",
        "clear zone",
        "lateral clearance",
        "utility pole",
        "utility poles",
        "fixed object",
        "fixed objects",
        "embankment",
        "limited recovery",
        "recovery area",
        "recover",
    )
    if not any(term in text for term in roadside_terms):
        return {}

    crash = _norm_text(query.crash_type)
    facility = _norm_text(query.facility_type)
    intersection_related = _norm_text(query.intersection_related)
    if facility and "intersection" in facility:
        return {}
    if intersection_related in {"yes", "true"}:
        return {}
    if crash and not any(x in crash for x in ["run off road", "single vehicle", "fixed object", "all"]):
        return {}

    priorities: dict[str, float] = {}

    if "utility pole" in text or "utility poles" in text:
        priorities.update(
            {
                "utility_pole_guardrail": 120.0,
                "lateral_clearance": 90.0,
                "fixed_object_removal": 85.0,
                "side_slope_guardrail": 75.0,
                "embankment_guardrail": 70.0,
                "sideslope_improvement": 65.0,
                "safety_edge": 45.0,
            }
        )
    if any(term in text for term in ["slope", "slopes", "side slope", "sideslope", "embankment"]):
        priorities.update(
            {
                "sideslope_improvement": max(priorities.get("sideslope_improvement", 0.0), 115.0),
                "side_slope_guardrail": max(priorities.get("side_slope_guardrail", 0.0), 105.0),
                "lateral_clearance": max(priorities.get("lateral_clearance", 0.0), 95.0),
                "embankment_guardrail": max(priorities.get("embankment_guardrail", 0.0), 85.0),
                "safety_edge": max(priorities.get("safety_edge", 0.0), 50.0),
            }
        )
    if any(term in text for term in ["clear zone", "lateral clearance", "limited recovery", "recovery area", "recover"]):
        priorities.update(
            {
                "lateral_clearance": max(priorities.get("lateral_clearance", 0.0), 115.0),
                "sideslope_improvement": max(priorities.get("sideslope_improvement", 0.0), 105.0),
                "safety_edge": max(priorities.get("safety_edge", 0.0), 90.0),
                "side_slope_guardrail": max(priorities.get("side_slope_guardrail", 0.0), 80.0),
                "embankment_guardrail": max(priorities.get("embankment_guardrail", 0.0), 70.0),
            }
        )
    if "fixed object" in text or "fixed objects" in text:
        priorities.update(
            {
                "lateral_clearance": max(priorities.get("lateral_clearance", 0.0), 105.0),
                "fixed_object_removal": max(priorities.get("fixed_object_removal", 0.0), 100.0),
                "utility_pole_guardrail": max(priorities.get("utility_pole_guardrail", 0.0), 95.0),
                "embankment_guardrail": max(priorities.get("embankment_guardrail", 0.0), 80.0),
                "sideslope_improvement": max(priorities.get("sideslope_improvement", 0.0), 70.0),
            }
        )
    return priorities


def _night_family(ev: dict[str, Any]) -> str | None:
    cm = _norm_text(ev.get("countermeasure"))
    cat = _norm_text(ev.get("countermeasure_category"))
    sub = _norm_text(ev.get("countermeasure_subcategory"))
    text = f"{cm} {cat} {sub}"

    if "centerline rumble" in text and "shoulder rumble" in text:
        return "centerline_shoulder_rumble"
    if "centerline rumble" in text:
        return "centerline_rumble"
    if "shoulder rumble" in text or "edgeline rumble" in text or "edge line rumble" in text:
        return "shoulder_or_edgeline_rumble"
    if "wider edgeline" in text or "wider longitudinal" in text or "wide edgeline" in text:
        return "wider_edgeline"
    if (
        "raised pavement marker" in text
        or "inlaid pavement marker" in text
        or "profiled thermoplastic" in text
        or "wet reflective" in text
    ):
        return "night_marking_visibility"
    if "curve warning" in text or "chevron" in text or "curve delineation" in text or "delineator" in text:
        return "curve_delineation"
    if "safety edge" in text:
        return "safety_edge"
    if "lighting" in text or "illumination" in text:
        return "lighting"
    return None


def _night_family_bonus(ev: dict[str, Any], family: str, user_text: str) -> float:
    """Prefer broadly applicable representatives within a mechanism family.

    The supplement is only a recall expansion, but picking a narrow variant can
    still distract the reranker. For example, "shoulder rumble strips on roads
    with existing centerline rumble strips" should not be the representative
    unless the query actually states that prior condition.
    """
    cm = _norm_text(ev.get("countermeasure"))
    text = _norm_text(user_text)
    bonus = 0.0

    if family == "shoulder_or_edgeline_rumble":
        if "existing centerline" in cm and "existing centerline" not in text:
            bonus -= 25.0
        if cm in {"install shoulder rumble strips", "install edgeline rumble strips"}:
            bonus += 20.0
        elif "shoulder rumble strips" in cm and "existing" not in cm:
            bonus += 12.0

    if family == "centerline_shoulder_rumble":
        if "centerline and shoulder rumble strips" in cm:
            bonus += 20.0
        if "existing" in cm and "existing" not in text:
            bonus -= 15.0

    if family == "wider_edgeline":
        if "wider edgeline" in cm or "wider edgelines" in cm:
            bonus += 20.0
        if "longitudinal pavement markings" in cm:
            bonus += 8.0

    if family == "lighting":
        if cm == "install lighting":
            bonus += 20.0
        if "street lighting illuminance" in cm and not any(
            term in text for term in ["street", "urban", "suburban", "illuminance", "uniformity"]
        ):
            bonus -= 10.0

    return bonus


def hidden_precondition_mismatch(ev: dict[str, Any], user_text: str) -> bool:
    """Return True when a candidate has a stated prerequisite absent from the query.

    These rows are not bad evidence, but they are a poor candidate for a generic
    request. Exposing them causes the reranker to pick overly narrow variants
    such as "install shoulder rumble strips on roads with existing centerline
    rumble strips" when the site has no stated existing rumble-strip condition.
    """
    cm = _norm_text(ev.get("countermeasure"))
    text = _norm_text(user_text)
    if "existing centerline rumble" in cm and "existing centerline rumble" not in text:
        return True
    if "existing shoulder rumble" in cm and "existing shoulder rumble" not in text:
        return True
    return False


def candidate_context_mismatch(ev: dict[str, Any], user_text: str, query: Query) -> bool:
    """Return True for candidates that conflict with explicit site context.

    This is an applicability filter for candidate exposure, not a ranking rule.
    The key example is an already-signalized intersection: treatments that
    install a new traffic signal are for unsignalized/stop-controlled sites and
    should not be exposed to the downstream reranker. Signal timing, phasing,
    visibility, and other existing-signal treatments remain eligible.
    """
    cm = _norm_text(ev.get("countermeasure"))
    text = _norm_text(user_text)
    raw_text = str(user_text or "").lower()
    ctrl = _norm_text(query.traffic_control_type)
    facility = _norm_text(query.facility_type)

    query_is_signalized = (
        "signalized" in text
        or "signalised" in text
        or "信号灯" in raw_text
        or "信号控制" in raw_text
        or ctrl == "signalized"
        or ("signal" in ctrl and "install" not in ctrl)
    )
    query_is_intersection = (
        _is_intersection(query)
        or "intersection" in text
        or "路口" in raw_text
        or "交叉口" in raw_text
    )
    installs_new_signal = (
        "install a traffic signal" in cm
        or "install traffic signal" in cm
        or "signalization" in cm
        or "signalisation" in cm
    )
    if query_is_signalized and query_is_intersection and installs_new_signal:
        return True

    median_opening_context = any(
        term in text
        for term in [
            "median opening",
            "full access minor road",
            "full access intersection",
            "direct left turn",
            "divided roadway corridor",
            "divided four lane",
            "divided 4 lane",
            "crossing or turning",
            "restricted crossing",
            "superstreet",
            "rtut",
            "j turn",
        ]
    )
    segment_lane_departure_candidate = any(
        term in cm
        for term in [
            "rumble strip",
            "rumble strips",
            "shoulder rumble",
            "centerline rumble",
            "edgeline rumble",
            "edge line rumble",
            "curve warning",
            "chevron",
            "curve delineation",
            "raised pavement marker",
            "wider edgeline",
            "wider longitudinal",
            "safety edge",
            "high friction surface",
            "hfst",
        ]
    )
    if median_opening_context and segment_lane_departure_candidate:
        return True

    ordinary_opposing_direction_context = any(
        term in text
        for term in [
            "head on",
            "head-on",
            "opposing direction",
            "opposing-direction",
            "opposing traffic",
            "sideswipe",
            "lane separation",
            "centerline",
        ]
    )
    explicit_median_or_divided_context = any(
        term in text
        for term in [
            "median",
            "cross median",
            "cross-median",
            "divided",
            "divided roadway",
            "divided road",
        ]
    )
    if ordinary_opposing_direction_context and not explicit_median_or_divided_context:
        if any(term in cm for term in ["median barrier", "raised median", "cable median barrier"]):
            return True
        if "guardrail" in cm and not any(term in text for term in ["roadside", "fixed object", "embankment", "slope"]):
            return True

    existing_paved_shoulder_context = any(
        term in text
        for term in ["existing paved shoulder", "already paved shoulder", "has paved shoulder", "with paved shoulder"]
    )
    narrow_paved_shoulder_context = "narrow paved shoulder" in text
    if existing_paved_shoulder_context or narrow_paved_shoulder_context:
        if ("pave shoulder" in cm or "add new paved shoulder" in cm) and "widen" not in cm and "deteriorated" not in cm:
            return True

    pedestrian_crossing_context = "pedestrian" in text and any(
        term in text for term in ["crossing", "crosswalk", "midblock", "uncontrolled crossing"]
    )
    if pedestrian_crossing_context and not any(term in text for term in ["wet", "friction", "skid", "pavement surface"]):
        if any(term in cm for term in ["high friction", "hfst", "microsurfacing", "pavement friction"]):
            return True

    winter_weather_context = any(term in text for term in ["winter", "icing", "ice", "snow", "slush", "freezing"])
    if winter_weather_context:
        if any(term in cm for term in ["diverging diamond", "auxiliary lane", "single lane exit ramp"]):
            return True

    toll_plaza_context = any(term in text for term in ["toll plaza", "tollbooth", "tolling", "mainline toll"])
    if toll_plaza_context:
        if not any(term in cm for term in ["toll", "tollbooth", "tolling"]):
            return True

    shoulder_condition_context = any(term in text for term in ["deteriorated shoulder", "narrow shoulder", "paved shoulder", "unpaved shoulder"])
    if shoulder_condition_context and not any(term in text for term in ["clear zone", "utility pole", "embankment", "side slope", "sideslope"]):
        if any(term in cm for term in ["utility pole", "guardrail", "side slope", "sideslope", "lateral clearance"]):
            return True

    speed_management_context = any(term in text for term in ["posted speed", "speed consistency", "operating speed", "speeding", "speed related"])
    if speed_management_context and not any(term in text for term in ["curve", "wet", "friction", "skid", "run off road"]):
        if any(term in cm for term in ["high friction", "hfst", "rumble", "pavement marker", "edgeline"]):
            return True

    drowsy_context = any(term in text for term in ["drowsy", "fatigue", "fatigued"])
    if drowsy_context:
        if any(term in cm for term in ["median barrier", "high friction", "hfst", "posted speed", "speed limit"]):
            return True

    managed_lane_context = any(term in text for term in ["managed lane", "managed lanes", "hov", "hot lane", "hot lanes"])
    if managed_lane_context:
        if any(term in cm for term in ["wildlife crossing", "median barrier", "speed enforcement"]) and not any(
            term in text for term in ["wildlife", "median", "speed"]
        ):
            return True

    # Avoid treating signalized-intersection treatments as segment candidates
    # when the query explicitly says roadway segment and does not mention an
    # intersection or ramp terminal.
    if "segment" in facility and not query_is_intersection:
        ev_facility = _norm_text((ev.get("conditions") or {}).get("facility_type"))
        if "intersection" in ev_facility and any(x in cm for x in ["signal", "left turn phas", "red light"]):
            return True

    return False


def normalized_countermeasure_key(ev: dict[str, Any]) -> str:
    """Stable key for exact duplicate countermeasure text in rerank candidates."""
    return _norm_text(ev.get("countermeasure"))


def _night_priorities(user_text: str, query: Query) -> dict[str, float]:
    text = _norm_text(user_text)
    crash = _norm_text(query.crash_type)
    area = _norm_text(query.area_type)
    facility = _norm_text(query.facility_type)
    if "night" not in text and "night" not in crash:
        return {}
    if _is_intersection(query):
        return {}
    if facility and "segment" not in facility:
        return {}

    is_head_on = any(term in text or term in crash for term in ["head on", "headon", "opposing", "sideswipe"])
    is_ror = any(
        term in text or term in crash
        for term in ["run off road", "runoff road", "roadway departure", "lane departure", "single vehicle"]
    )
    is_curve = any(term in text for term in ["curve", "curved", "horizontal curve", "alignment"])
    lighting_explicit = any(
        term in text
        for term in [
            "unlit",
            "without lighting",
            "no lighting",
            "limited lighting",
            "lighting limited",
            "poor lighting",
            "low lighting",
            "照明",
        ]
    )
    is_urban = "urban" in area or "suburban" in area or "urban" in text or "suburban" in text
    is_rural = "rural" in area or "rural" in text

    priorities: dict[str, float] = {}
    if is_head_on:
        priorities.update(
            {
                "centerline_rumble": 120.0,
                "centerline_shoulder_rumble": 110.0,
                "night_marking_visibility": 85.0,
                "curve_delineation": 95.0 if is_curve else 75.0,
            }
        )
    if is_ror:
        priorities.update(
            {
                "shoulder_or_edgeline_rumble": max(priorities.get("shoulder_or_edgeline_rumble", 0.0), 120.0),
                "centerline_shoulder_rumble": max(priorities.get("centerline_shoulder_rumble", 0.0), 110.0),
                "wider_edgeline": max(priorities.get("wider_edgeline", 0.0), 105.0),
                "night_marking_visibility": max(priorities.get("night_marking_visibility", 0.0), 95.0),
                "curve_delineation": max(priorities.get("curve_delineation", 0.0), 115.0 if is_curve else 80.0),
                "safety_edge": max(priorities.get("safety_edge", 0.0), 75.0),
            }
        )
    if not priorities:
        priorities.update(
            {
                "night_marking_visibility": 90.0,
                "wider_edgeline": 80.0,
                "curve_delineation": 90.0 if is_curve else 65.0,
            }
        )

    if lighting_explicit:
        priorities["lighting"] = 125.0
    elif is_urban:
        priorities["lighting"] = 95.0
    elif not is_rural:
        priorities["lighting"] = 65.0

    return priorities


def nighttime_segment_supplement_ids(
    *,
    user_text: str,
    query: Query,
    evidences: list[dict[str, Any]],
    score_rows: list[dict[str, str]],
    existing_ids: set[str] | None = None,
    cfg: MechanismRecallConfig | None = None,
) -> list[str]:
    """Return extra candidates for nighttime segment crashes.

    Lighting is exposed as a high-priority mechanism only when lighting
    deficiency is explicit or the setting is urban/suburban. For rural segment
    head-on/ROR queries, rumble-strip, pavement-marking, and curve-delineation
    families are prioritized ahead of lighting.
    """
    cfg = cfg or MechanismRecallConfig()
    existing_ids = set(existing_ids or set())
    priorities = _night_priorities(user_text, query)
    if not priorities:
        return []

    best_by_family: dict[str, tuple[tuple[float, float, float, float], str]] = {}
    for ev, row in zip(evidences, score_rows):
        eid = str(ev.get("evidence_id") or "")
        if not eid or eid in existing_ids:
            continue
        if hidden_precondition_mismatch(ev, user_text):
            continue
        family = _night_family(ev)
        if not family or family not in priorities:
            continue
        if _effect_crf(ev) < float(cfg.min_crf) or _star(ev) < float(cfg.min_star):
            continue
        scored = score_row(row, query, match_mode="robust")
        if scored is None:
            continue
        key = (
            priorities[family],
            _night_family_bonus(ev, family, user_text),
            _star(ev),
            _effect_crf(ev),
            float(scored.total_score),
        )
        old = best_by_family.get(family)
        if old is None or key > old[0]:
            best_by_family[family] = (key, eid)

    ranked = sorted(best_by_family.values(), key=lambda item: item[0], reverse=True)
    return [eid for _, eid in ranked[: int(cfg.max_supplements)]]


def access_management_supplement_ids(
    *,
    user_text: str,
    query: Query,
    evidences: list[dict[str, Any]],
    score_rows: list[dict[str, str]],
    existing_ids: set[str] | None = None,
    cfg: MechanismRecallConfig | None = None,
) -> list[str]:
    """Return extra candidates for driveway/access-management corridor queries.

    Formula-based CMFs are allowed here even when they lack a concrete CRF,
    because driveway-density evidence is stored as a CMF formula.
    """
    cfg = cfg or MechanismRecallConfig()
    existing_ids = set(existing_ids or set())
    priorities = _access_priorities(user_text, query)
    if not priorities:
        return []

    best_by_family: dict[str, tuple[tuple[float, float, float, float], str]] = {}
    for ev, row in zip(evidences, score_rows):
        eid = str(ev.get("evidence_id") or "")
        if not eid or eid in existing_ids:
            continue
        family = _access_family(ev)
        if not family or family not in priorities:
            continue
        has_formula = _has_formula_effect(ev)
        crf = _effect_crf(ev)
        if not has_formula and crf < float(cfg.min_crf):
            continue
        if _star(ev) < float(cfg.min_star):
            continue
        match_score, _ = compute_match_score(row, query, match_mode="robust")
        if match_score <= 0:
            continue
        key = (
            priorities[family],
            _crash_fit_bonus(ev, query),
            _roadway_division_fit_bonus(ev, user_text),
            float(match_score),
            _star(ev),
            crf if crf > -900 else 0.0,
        )
        old = best_by_family.get(family)
        if old is None or key > old[0]:
            best_by_family[family] = (key, eid)

    ranked = sorted(best_by_family.values(), key=lambda item: item[0], reverse=True)
    return [eid for _, eid in ranked[: int(cfg.max_supplements)]]


def signal_visibility_supplement_ids(
    *,
    user_text: str,
    query: Query,
    evidences: list[dict[str, Any]],
    score_rows: list[dict[str, str]],
    existing_ids: set[str] | None = None,
    cfg: MechanismRecallConfig | None = None,
) -> list[str]:
    """Expose signal-display visibility treatments for signalized night/intersection queries."""
    cfg = cfg or MechanismRecallConfig()
    existing_ids = set(existing_ids or set())
    priorities = _signal_visibility_priorities(user_text, query)
    if not priorities:
        return []

    best_by_family: dict[str, tuple[tuple[float, float, float, float, float], str]] = {}
    for ev, row in zip(evidences, score_rows):
        eid = str(ev.get("evidence_id") or "")
        if not eid or eid in existing_ids:
            continue
        family = _signal_visibility_family(ev)
        if not family or family not in priorities:
            continue
        if _effect_crf(ev) < float(cfg.min_crf) or _star(ev) < float(cfg.min_star):
            continue
        scored = score_row(row, query, match_mode="robust")
        if scored is None:
            continue
        key = (
            priorities[family],
            _crash_fit_bonus(ev, query),
            _star(ev),
            _effect_crf(ev),
            float(scored.total_score),
        )
        old = best_by_family.get(family)
        if old is None or key > old[0]:
            best_by_family[family] = (key, eid)

    ranked = sorted(best_by_family.values(), key=lambda item: item[0], reverse=True)
    return [eid for _, eid in ranked[: int(cfg.max_supplements)]]


def signalized_left_turn_supplement_ids(
    *,
    user_text: str,
    query: Query,
    evidences: list[dict[str, Any]],
    score_rows: list[dict[str, str]],
    existing_ids: set[str] | None = None,
    cfg: MechanismRecallConfig | None = None,
) -> list[str]:
    """Expose left-turn phasing/offset treatments for existing signalized intersections."""
    cfg = cfg or MechanismRecallConfig()
    existing_ids = set(existing_ids or set())
    priorities = _signalized_left_turn_priorities(user_text, query)
    if not priorities:
        return []

    best_by_family: dict[str, tuple[tuple[float, float, float, float, float], str]] = {}
    for ev, row in zip(evidences, score_rows):
        eid = str(ev.get("evidence_id") or "")
        if not eid or eid in existing_ids:
            continue
        if hidden_precondition_mismatch(ev, user_text) or candidate_context_mismatch(ev, user_text, query):
            continue
        family = _signalized_left_turn_family(ev)
        if not family or family not in priorities:
            continue
        if _effect_crf(ev) < float(cfg.min_crf) or _star(ev) < float(cfg.min_star):
            continue
        scored = score_row(row, query, match_mode="robust")
        if scored is None:
            continue
        key = (
            priorities[family],
            _crash_fit_bonus(ev, query),
            _star(ev),
            _effect_crf(ev),
            float(scored.total_score),
        )
        old = best_by_family.get(family)
        if old is None or key > old[0]:
            best_by_family[family] = (key, eid)

    ranked = sorted(best_by_family.values(), key=lambda item: item[0], reverse=True)
    return [eid for _, eid in ranked[: int(cfg.max_supplements)]]


def passing_lane_supplement_ids(
    *,
    user_text: str,
    query: Query,
    evidences: list[dict[str, Any]],
    score_rows: list[dict[str, str]],
    existing_ids: set[str] | None = None,
    cfg: MechanismRecallConfig | None = None,
) -> list[str]:
    """Expose passing/climbing lane candidates when the query names passing constraints."""
    cfg = cfg or MechanismRecallConfig()
    existing_ids = set(existing_ids or set())
    priorities = _passing_lane_priorities(user_text, query)
    if not priorities:
        return []

    best_by_family: dict[str, tuple[tuple[float, float, float, float, float], str]] = {}
    for ev, row in zip(evidences, score_rows):
        eid = str(ev.get("evidence_id") or "")
        if not eid or eid in existing_ids:
            continue
        if hidden_precondition_mismatch(ev, user_text) or candidate_context_mismatch(ev, user_text, query):
            continue
        family = _passing_lane_family(ev)
        if not family or family not in priorities:
            continue
        if _effect_crf(ev) < float(cfg.min_crf) or _star(ev) < 1.0:
            continue
        scored = score_row(row, query, match_mode="robust")
        if scored is None:
            continue
        key = (
            priorities[family],
            _crash_fit_bonus(ev, query),
            _star(ev),
            _effect_crf(ev),
            float(scored.total_score),
        )
        old = best_by_family.get(family)
        if old is None or key > old[0]:
            best_by_family[family] = (key, eid)

    ranked = sorted(best_by_family.values(), key=lambda item: item[0], reverse=True)
    return [eid for _, eid in ranked[: int(cfg.max_supplements)]]


def _generic_family_supplement_ids(
    *,
    user_text: str,
    query: Query,
    evidences: list[dict[str, Any]],
    score_rows: list[dict[str, str]],
    priorities: dict[str, float],
    family_fn,
    existing_ids: set[str] | None = None,
    cfg: MechanismRecallConfig | None = None,
    min_star: float | None = None,
) -> list[str]:
    cfg = cfg or MechanismRecallConfig()
    existing_ids = set(existing_ids or set())
    if not priorities:
        return []
    best_by_family: dict[str, tuple[tuple[float, float, float, float, float], str]] = {}
    for ev, row in zip(evidences, score_rows):
        eid = str(ev.get("evidence_id") or "")
        if not eid or eid in existing_ids:
            continue
        if hidden_precondition_mismatch(ev, user_text) or candidate_context_mismatch(ev, user_text, query):
            continue
        family = family_fn(ev)
        if not family or family not in priorities:
            continue
        crf = _effect_crf(ev)
        if crf < float(cfg.min_crf):
            continue
        star_floor = float(cfg.min_star if min_star is None else min_star)
        if _star(ev) < star_floor:
            continue
        scored = score_row(row, query, match_mode="robust")
        score_value = 0.0 if scored is None else float(scored.total_score)
        key = (
            priorities[family],
            _crash_fit_bonus(ev, query),
            _site_context_fit_bonus(ev, query, user_text),
            _star(ev),
            crf,
            score_value,
        )
        old = best_by_family.get(family)
        if old is None or key > old[0]:
            best_by_family[family] = (key, eid)
    ranked = sorted(best_by_family.values(), key=lambda item: item[0], reverse=True)
    return [eid for _, eid in ranked[: int(cfg.max_supplements)]]


def advance_guidance_supplement_ids(
    *,
    user_text: str,
    query: Query,
    evidences: list[dict[str, Any]],
    score_rows: list[dict[str, str]],
    existing_ids: set[str] | None = None,
    cfg: MechanismRecallConfig | None = None,
) -> list[str]:
    """Expose approach-guidance/warning treatments when the query asks for advance guidance."""
    return _generic_family_supplement_ids(
        user_text=user_text,
        query=query,
        evidences=evidences,
        score_rows=score_rows,
        existing_ids=existing_ids,
        cfg=cfg,
        priorities=_advance_guidance_priorities(user_text, query),
        family_fn=_advance_guidance_family,
        min_star=1.0,
    )


def pedestrian_crossing_supplement_ids(
    *,
    user_text: str,
    query: Query,
    evidences: list[dict[str, Any]],
    score_rows: list[dict[str, str]],
    existing_ids: set[str] | None = None,
    cfg: MechanismRecallConfig | None = None,
) -> list[str]:
    """Expose controlled/uncontrolled pedestrian-crossing treatments for crossing-specific queries."""
    return _generic_family_supplement_ids(
        user_text=user_text,
        query=query,
        evidences=evidences,
        score_rows=score_rows,
        existing_ids=existing_ids,
        cfg=cfg,
        priorities=_ped_crossing_priorities(user_text, query),
        family_fn=_ped_crossing_family,
        min_star=1.0,
    )


def frontage_road_supplement_ids(
    *,
    user_text: str,
    query: Query,
    evidences: list[dict[str, Any]],
    score_rows: list[dict[str, str]],
    existing_ids: set[str] | None = None,
    cfg: MechanismRecallConfig | None = None,
) -> list[str]:
    """Expose frontage-road operation changes for two-way frontage-road conflict queries."""
    return _generic_family_supplement_ids(
        user_text=user_text,
        query=query,
        evidences=evidences,
        score_rows=score_rows,
        existing_ids=existing_ids,
        cfg=cfg,
        priorities=_frontage_road_priorities(user_text, query),
        family_fn=_frontage_road_family,
        min_star=1.0,
    )


def managed_lane_supplement_ids(
    *,
    user_text: str,
    query: Query,
    evidences: list[dict[str, Any]],
    score_rows: list[dict[str, str]],
    existing_ids: set[str] | None = None,
    cfg: MechanismRecallConfig | None = None,
) -> list[str]:
    """Expose HOV/HOT managed-lane conversion rows for managed-lane weaving/access queries."""
    return _generic_family_supplement_ids(
        user_text=user_text,
        query=query,
        evidences=evidences,
        score_rows=score_rows,
        existing_ids=existing_ids,
        cfg=cfg,
        priorities=_managed_lane_priorities(user_text, query),
        family_fn=_managed_lane_family,
        min_star=1.0,
    )


def winter_weather_supplement_ids(
    *,
    user_text: str,
    query: Query,
    evidences: list[dict[str, Any]],
    score_rows: list[dict[str, str]],
    existing_ids: set[str] | None = None,
    cfg: MechanismRecallConfig | None = None,
) -> list[str]:
    """Expose snow/ice/FAST treatments for winter-weather or icing queries."""
    return _generic_family_supplement_ids(
        user_text=user_text,
        query=query,
        evidences=evidences,
        score_rows=score_rows,
        existing_ids=existing_ids,
        cfg=cfg,
        priorities=_winter_weather_priorities(user_text, query),
        family_fn=_winter_weather_family,
        min_star=1.0,
    )


def shoulder_improvement_supplement_ids(
    *,
    user_text: str,
    query: Query,
    evidences: list[dict[str, Any]],
    score_rows: list[dict[str, str]],
    existing_ids: set[str] | None = None,
    cfg: MechanismRecallConfig | None = None,
) -> list[str]:
    """Expose shoulder paving/widening repair rows for explicit shoulder-condition lane-departure queries."""
    return _generic_family_supplement_ids(
        user_text=user_text,
        query=query,
        evidences=evidences,
        score_rows=score_rows,
        existing_ids=existing_ids,
        cfg=cfg,
        priorities=_shoulder_improvement_priorities(user_text, query),
        family_fn=_shoulder_improvement_family,
        min_star=1.0,
    )


def toll_plaza_supplement_ids(
    *,
    user_text: str,
    query: Query,
    evidences: list[dict[str, Any]],
    score_rows: list[dict[str, str]],
    existing_ids: set[str] | None = None,
    cfg: MechanismRecallConfig | None = None,
) -> list[str]:
    """Expose ORT/all-electronic tolling rows for traditional mainline toll-plaza queries."""
    return _generic_family_supplement_ids(
        user_text=user_text,
        query=query,
        evidences=evidences,
        score_rows=score_rows,
        existing_ids=existing_ids,
        cfg=cfg,
        priorities=_toll_plaza_priorities(user_text, query),
        family_fn=_toll_plaza_family,
        min_star=1.0,
    )


def speed_management_supplement_ids(
    *,
    user_text: str,
    query: Query,
    evidences: list[dict[str, Any]],
    score_rows: list[dict[str, str]],
    existing_ids: set[str] | None = None,
    cfg: MechanismRecallConfig | None = None,
) -> list[str]:
    """Expose speed-enforcement / posted-speed candidates for speed-management queries."""
    return _generic_family_supplement_ids(
        user_text=user_text,
        query=query,
        evidences=evidences,
        score_rows=score_rows,
        existing_ids=existing_ids,
        cfg=cfg,
        priorities=_speed_management_priorities(user_text, query),
        family_fn=_speed_management_family,
        min_star=1.0,
    )


def drowsy_driving_supplement_ids(
    *,
    user_text: str,
    query: Query,
    evidences: list[dict[str, Any]],
    score_rows: list[dict[str, str]],
    existing_ids: set[str] | None = None,
    cfg: MechanismRecallConfig | None = None,
) -> list[str]:
    """Expose rest-area/travel-information-center rows for drowsy-driving freeway queries."""
    return _generic_family_supplement_ids(
        user_text=user_text,
        query=query,
        evidences=evidences,
        score_rows=score_rows,
        existing_ids=existing_ids,
        cfg=cfg,
        priorities=_drowsy_priorities(user_text, query),
        family_fn=_drowsy_family,
        min_star=1.0,
    )


def cfi_supplement_ids(
    *,
    user_text: str,
    query: Query,
    evidences: list[dict[str, Any]],
    score_rows: list[dict[str, str]],
    existing_ids: set[str] | None = None,
    cfg: MechanismRecallConfig | None = None,
) -> list[str]:
    """Expose CFI/DLT candidates for high-volume signalized left-turn/queue conflict queries."""
    return _generic_family_supplement_ids(
        user_text=user_text,
        query=query,
        evidences=evidences,
        score_rows=score_rows,
        existing_ids=existing_ids,
        cfg=cfg,
        priorities=_cfi_priorities(user_text, query),
        family_fn=_cfi_family,
        min_star=1.0,
    )


def stop_control_supplement_ids(
    *,
    user_text: str,
    query: Query,
    evidences: list[dict[str, Any]],
    score_rows: list[dict[str, str]],
    existing_ids: set[str] | None = None,
    cfg: MechanismRecallConfig | None = None,
) -> list[str]:
    """Expose low-cost stop-control signing/marking candidates for stop-controlled intersections."""
    cfg = cfg or MechanismRecallConfig()
    existing_ids = set(existing_ids or set())
    priorities = _stop_control_priorities(user_text, query)
    if not priorities:
        return []

    best_by_family: dict[str, tuple[tuple[float, float, float, float, float], str]] = {}
    for ev, row in zip(evidences, score_rows):
        eid = str(ev.get("evidence_id") or "")
        if not eid or eid in existing_ids:
            continue
        family = _stop_control_family(ev)
        if not family or family not in priorities:
            continue
        if _effect_crf(ev) < float(cfg.min_crf) or _star(ev) < float(cfg.min_star):
            continue
        scored = score_row(row, query, match_mode="robust")
        if scored is None:
            continue
        key = (
            priorities[family],
            _crash_fit_bonus(ev, query),
            _star(ev),
            _effect_crf(ev),
            float(scored.total_score),
        )
        old = best_by_family.get(family)
        if old is None or key > old[0]:
            best_by_family[family] = (key, eid)

    ranked = sorted(best_by_family.values(), key=lambda item: item[0], reverse=True)
    return [eid for _, eid in ranked[: int(cfg.max_supplements)]]


def median_opening_supplement_ids(
    *,
    user_text: str,
    query: Query,
    evidences: list[dict[str, Any]],
    score_rows: list[dict[str, str]],
    existing_ids: set[str] | None = None,
    cfg: MechanismRecallConfig | None = None,
) -> list[str]:
    """Expose RTUT/superstreet/offset-left-turn candidates for divided-road median-opening conflicts."""
    cfg = cfg or MechanismRecallConfig()
    existing_ids = set(existing_ids or set())
    priorities = _median_opening_priorities(user_text, query)
    if not priorities:
        return []

    best_by_family: dict[str, tuple[tuple[float, float, float, float, float], str]] = {}
    for ev, row in zip(evidences, score_rows):
        eid = str(ev.get("evidence_id") or "")
        if not eid or eid in existing_ids:
            continue
        family = _median_opening_family(ev)
        if not family or family not in priorities:
            continue
        min_star = _median_opening_min_star(family, user_text, cfg)
        if _effect_crf(ev) < float(cfg.min_crf) or _star(ev) < min_star:
            continue
        scored = score_row(row, query, match_mode="robust")
        if scored is None:
            continue
        key = (
            priorities[family],
            _crash_fit_bonus(ev, query),
            float(scored.match_score),
            _effect_crf(ev),
            _star(ev),
            float(scored.total_score),
        )
        old = best_by_family.get(family)
        if old is None or key > old[0]:
            best_by_family[family] = (key, eid)

    ranked = sorted(best_by_family.values(), key=lambda item: item[0], reverse=True)
    return [eid for _, eid in ranked[: int(cfg.max_supplements)]]


def roadside_mechanism_supplement_ids(
    *,
    user_text: str,
    query: Query,
    evidences: list[dict[str, Any]],
    score_rows: list[dict[str, str]],
    existing_ids: set[str] | None = None,
    cfg: MechanismRecallConfig | None = None,
) -> list[str]:
    """Return extra candidate evidence IDs for roadside-recovery mechanisms.

    This is a recall supplement, not a rank override: it only exposes relevant
    mechanism families to the downstream reranker when lexical/global scoring is
    dominated by generic high-CRF road-segment treatments.
    """
    cfg = cfg or MechanismRecallConfig()
    existing_ids = set(existing_ids or set())
    priorities = _trigger_priorities(user_text, query)
    if not priorities:
        return []

    best_by_family: dict[str, tuple[tuple[float, float, float, float], str]] = {}
    for ev, row in zip(evidences, score_rows):
        eid = str(ev.get("evidence_id") or "")
        if not eid or eid in existing_ids:
            continue
        family = _roadside_family(ev)
        if not family or family not in priorities:
            continue
        if _effect_crf(ev) < float(cfg.min_crf) or _star(ev) < float(cfg.min_star):
            continue
        scored = score_row(row, query, match_mode="robust")
        if scored is None:
            continue
        # Family priority dominates; scorer and quality break ties within a family.
        key = (
            priorities[family],
            float(scored.total_score),
            _star(ev),
            _effect_crf(ev),
        )
        old = best_by_family.get(family)
        if old is None or key > old[0]:
            best_by_family[family] = (key, eid)

    ranked = sorted(best_by_family.values(), key=lambda item: item[0], reverse=True)
    return [eid for _, eid in ranked[: int(cfg.max_supplements)]]
