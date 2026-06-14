from __future__ import annotations

import re
from typing import Any


def norm_text(value: object) -> str:
    text = str(value or "").lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def countermeasure_text_key(item: dict[str, Any]) -> str:
    """Exact countermeasure-text key, deliberately ignoring evidence_id."""
    return norm_text(item.get("countermeasure"))


def mechanism_family(item: dict[str, Any]) -> str:
    """Map evidence rows/cards to broad engineering mechanism families.

    This is intentionally coarser than evidence_id and usually coarser than the
    exact countermeasure text. It is for evaluation, not for replacing human
    review. Unknown rows fall back to a normalized countermeasure-text key.
    """
    cm = norm_text(item.get("countermeasure"))
    cat = norm_text(item.get("countermeasure_category"))
    sub = norm_text(item.get("countermeasure_subcategory"))
    text = f"{cm} {cat} {sub}"

    # Active transportation and crossings.
    if any(x in text for x in ["pedestrian hybrid beacon", "hawk", "rapid flashing beacon", "rrfb"]):
        return "active_transport_uncontrolled_crossing_beacon"
    if "raised median" in text and any(x in text for x in ["crosswalk", "pedestrian", "refuge"]):
        return "active_transport_uncontrolled_crossing_refuge"
    if "median treatment" in text and any(x in text for x in ["ped", "bike", "bicycle"]):
        return "active_transport_uncontrolled_crossing_refuge"
    if "raised pedestrian crosswalk" in text or "raised crosswalk" in text:
        return "active_transport_uncontrolled_crossing_traffic_calming"
    if "pedestrian countdown" in text or "walk don t walk" in text or "pedestrian signal" in text:
        return "active_transport_pedestrian_signal"
    if "transverse rumble" in text and ("pedestrian" in text or "crosswalk" in text):
        return "active_transport_rural_crosswalk_warning"
    if (
        "pedestrian phase" in text
        or "pedestrian phasing" in text
        or "pedestrian crossing time" in text
        or "cycle length for pedestrian" in text
        or "allow pedestrians more crossing time" in text
        or "exclusive pedestrian" in text
        or "barnes dance" in text
    ):
        return "active_transport_pedestrian_signal_timing"
    if "leading pedestrian interval" in text or "lpi" in text:
        return "active_transport_pedestrian_signal_timing"
    if "bike lane" in text or "bicycle lane" in text:
        return "active_transport_bicycle_lane"
    if "cycle track" in text or "bicycle track" in text:
        return "active_transport_bicycle_track"

    # Intersection traffic control and turn-conflict treatments.
    if "roundabout" in text:
        return "intersection_control_roundabout"
    if "all way stop" in text or "allway stop" in text:
        return "intersection_control_all_way_stop"
    if "install a traffic signal" in text or "install traffic signal" in text or "signalization" in text:
        return "intersection_control_signalization"
    if "flashing beacon" in text and ("stop controlled" in text or "intersection" in text):
        return "intersection_warning_flashing_beacon"
    if "advance freeway guidance" in text or "advance guidance" in text or "guide sign" in text:
        return "intersection_advance_guidance_signing"
    if "double stop sign" in text or "double stop signs" in text:
        return "intersection_stop_sign_visibility_signing"
    if "retroreflectivity" in text and "stop" in text:
        return "intersection_stop_sign_visibility_signing"
    if "systemic signing" in text and "stop" in text:
        return "intersection_stop_sign_visibility_signing"
    if "stop ahead" in text and "pavement" in text:
        return "intersection_stop_sign_visibility_signing"
    if "flashing led stop" in text or "led stop sign" in text:
        return "intersection_stop_sign_visibility_signing"
    if "two way left turn lane" in text or "twltl" in text:
        return "access_management_twlttl"
    if (
        "right turn u turn" in text
        or "rtut" in text
        or "j turn" in text
        or "rcut" in text
        or "restricted crossing u turn" in text
        or "superstreet" in text
    ):
        return "intersection_restricted_crossing_u_turn"
    if "offset left" in text or "left turn offset" in text or "positive offset" in text:
        return "intersection_left_turn_offset"
    if "left turn lane" in text or "left turn lanes" in text:
        return "intersection_left_turn_lane"
    if (
        "left turn phas" in text
        or "protected permissive" in text
        or "protected only" in text
        or "permissive only" in text
        or "permissive with protected" in text
        or "permissive to protected" in text
        or "permitted with protected" in text
        or "permitted to protected" in text
        or "permitted protected to protected" in text
        or "flashing yellow arrow" in text
        or "left turn signal strategies" in text
        or "doghouse" in text
    ):
        return "intersection_left_turn_phasing"
    if "median u turn" in text or "mut" in text:
        return "intersection_median_u_turn"
    if "continuous flow intersection" in text or " cfi " in f" {text} " or "displaced left turn" in text:
        return "intersection_displaced_left_turn_or_cfi"
    if "convert high occupancy vehicle" in text or " hov " in f" {text} " and " hot " in f" {text} ":
        return "managed_lane_hov_to_hot"
    if "adaptive signal" in text:
        return "signal_operations_adaptive_control"
    if "red light camera" in text:
        return "signal_operations_red_light_camera"
    if "signal head" in text or "signal visibility" in text or "backplate" in text or "mast arm" in text:
        return "signal_visibility"
    if "yellow" in text and "all red" in text:
        return "signal_change_interval"

    # Lane departure, roadside, and delineation.
    if "centerline and shoulder rumble" in text or (
        "centerline rumble" in text and "shoulder rumble" in text
    ):
        return "lane_departure_rumble_combined"
    if "centerline rumble" in text:
        return "lane_departure_centerline_rumble"
    if "shoulder rumble" in text or "edgeline rumble" in text or "edge line rumble" in text:
        return "lane_departure_shoulder_rumble"
    if "safety edge" in text:
        return "lane_departure_safety_edge"
    if "pave deteriorated shoulder" in text:
        return "shoulder_improvement_pave_deteriorated"
    if "pave shoulder" in text or "add new paved shoulder" in text:
        return "shoulder_improvement_pave"
    if "widen paved shoulder" in text or "widen shoulder" in text:
        return "shoulder_improvement_widen"
    if "wider edgeline" in text or "wider edgelines" in text or "wider longitudinal" in text:
        return "lane_departure_wider_markings"
    if "raised pavement marker" in text or "inlaid pavement marker" in text or "wet reflective" in text:
        return "night_visibility_pavement_markers"
    if "curve warning" in text or "chevron" in text or "curve delineation" in text:
        return "curve_guidance_delineation"
    if "horizontal curve" in text and "rumble" in text:
        return "curve_lane_departure_treatment"
    if "flatten horizontal curve" in text or "change horizontal alignment" in text:
        return "curve_geometric_realignment"
    if (
        "high friction surface" in text
        or "hfst" in text
        or "improve pavement friction" in text
        or "increased pavement friction" in text
        or "microsurfacing" in text
        or "resurfacing treatment" in text
    ):
        return "pavement_friction"
    if "posted speed limit" in text and "engineering recommendation" in text:
        return "curve_speed_management"

    # Roadside recovery and barriers.
    if "utility pole" in text and "guardrail" in text:
        return "roadside_utility_pole_guardrail"
    if "side slope" in text and "guardrail" in text:
        return "roadside_side_slope_guardrail"
    if "guardrail" in text and ("embankment" in text or "roadside barrier" in text):
        return "roadside_embankment_guardrail"
    if "lateral clearance" in text:
        return "roadside_lateral_clearance"
    if "sideslope" in text or "side slope" in text or "flatten slope" in text:
        return "roadside_sideslope_improvement"
    if ("remove" in text or "relocate" in text) and "fixed object" in text:
        return "roadside_fixed_object_removal"
    if "median barrier" in text or "cable median barrier" in text:
        return "median_barrier"
    if "guardrail" in text:
        return "roadside_guardrail"

    # Segment access management and lane configuration.
    if "raised median" in text:
        return "access_management_raised_median"
    if "driveway density" in text or "driveway" in text:
        return "access_management_driveway"
    if "absence of access points" in text or ("access points" in text and "absence" in text):
        return "access_management_access_point_control"
    if "median opening" in text:
        return "access_management_median_opening"
    if "road diet" in text or "4 to 3" in text or "narrow cross section" in text:
        return "lane_reconfiguration_road_diet"
    if "passing lane" in text or "climbing lane" in text:
        return "passing_or_climbing_lane"

    # Lighting and speed.
    if "lighting" in text or "illumination" in text or "illuminance" in text:
        return "lighting"
    if "speed camera" in text or "automated speed" in text or "speed enforcement" in text:
        return "speed_enforcement"
    if "decreasing posted speed" in text or "posted speed limit" in text:
        return "speed_management_posted_limit"
    if "variable speed limit" in text:
        return "speed_management_variable_speed_limit"
    if "fixed automated spray technology" in text or " fast " in f" {text} ":
        return "winter_weather_anti_icing_fast"
    if "snow" in text or "slush" in text or "ice" in text:
        return "winter_weather_snow_ice_control"
    if "frontage road" in text and "one way" in text:
        return "frontage_road_one_way_conversion"
    if "rest area" in text or "travel information center" in text:
        return "drowsy_driving_rest_area"
    if "tollbooth" in text or "toll plaza" in text or "open road tolling" in text or "all electric toll" in text:
        return "toll_plaza_electronic_or_open_road_tolling"

    return f"cm:{countermeasure_text_key(item)}" if cm else "unknown"


def broad_mechanism_family(item: dict[str, Any]) -> str:
    """A softer family for judging engineering acceptability."""
    family = mechanism_family(item)
    if family.startswith("lane_departure_") or family in {
        "curve_guidance_delineation",
        "curve_lane_departure_treatment",
        "curve_geometric_realignment",
        "curve_speed_management",
        "night_visibility_pavement_markers",
        "pavement_friction",
        "shoulder_improvement_pave_deteriorated",
        "shoulder_improvement_pave",
        "shoulder_improvement_widen",
    }:
        return "lane_departure_visibility_or_control"
    if family.startswith("roadside_"):
        return "roadside_recovery_or_protection"
    if family.startswith("intersection_control_"):
        return "intersection_control_upgrade"
    if family.startswith("intersection_stop_sign_") or family == "intersection_warning_flashing_beacon":
        return "intersection_stop_control_visibility_warning"
    if family == "intersection_advance_guidance_signing":
        return "intersection_stop_control_visibility_warning"
    if family.startswith("intersection_left_turn_"):
        return "intersection_left_turn_treatment"
    if family == "intersection_displaced_left_turn_or_cfi":
        return "intersection_left_turn_treatment"
    if family.startswith("active_transport_"):
        return "active_transport"
    if family.startswith("winter_weather_"):
        return "winter_weather_surface_management"
    if family == "frontage_road_one_way_conversion":
        return "frontage_road_operations"
    if family == "managed_lane_hov_to_hot":
        return "managed_lane_operations"
    if family == "speed_management_posted_limit":
        return "speed_enforcement"
    if family == "drowsy_driving_rest_area":
        return "drowsy_driving_countermeasure"
    if family == "toll_plaza_electronic_or_open_road_tolling":
        return "toll_plaza_operations"
    if family.startswith("access_management_") or family.startswith("lane_reconfiguration_"):
        return "access_management_or_corridor_reconfiguration"
    if family == "passing_or_climbing_lane":
        return "passing_or_climbing_lane"
    return family


def jaccard(left: list[str], right: list[str]) -> float:
    a = {x for x in left if x}
    b = {x for x in right if x}
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def bad_fit_flags(user_text: str, item: dict[str, Any]) -> list[str]:
    """Small set of high-signal automatic warnings for human review."""
    flags: list[str] = []
    q = norm_text(user_text)
    cm = norm_text(item.get("countermeasure"))
    family = mechanism_family(item)

    try:
        crf = float(item.get("crf")) if item.get("crf") is not None else None
    except Exception:
        crf = None
    if crf is not None and crf < 0:
        flags.append("negative_crf")

    if "signalized" in q and family == "intersection_control_signalization":
        flags.append("signal_install_for_already_signalized_query")
    if ("rural" in q and "night" in q and family == "lighting" and not any(
        term in q
        for term in [
            "unlit",
            "without lighting",
            "no lighting",
            "limited lighting",
            "lighting limited",
            "lighting is limited",
            "limited or absent",
            "absent lighting",
            "poor lighting",
        ]
    )):
        flags.append("rural_night_lighting_without_explicit_lighting_deficiency")
    if "existing centerline rumble" in cm and "existing centerline rumble" not in q:
        flags.append("hidden_existing_centerline_rumble_precondition")
    if "existing shoulder rumble" in cm and "existing shoulder rumble" not in q:
        flags.append("hidden_existing_shoulder_rumble_precondition")
    if ("pedestrian" in cm or "crosswalk" in cm or "hawk" in cm or "rrfb" in cm) and (
        "bicycle" in q and "pedestrian" not in q
    ):
        flags.append("pedestrian_crossing_treatment_for_bicycle_only_query")

    return flags
