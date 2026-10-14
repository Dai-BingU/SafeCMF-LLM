#!/usr/bin/env python3
"""Select initial v3.9 teacher labels from the frozen query-only candidate pools.

This is a candidate-constrained labeling stage, not an open audit. It reads only
the teacher inputs produced from visible Query text and the frozen at-most-Top-30
candidate cards. Outputs are Codex-authored systematic initial labels pending
stratified author/expert review; they are not final expert ground truth.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cmfrec.countermeasure_family import broad_mechanism_family, mechanism_family  # noqa: E402
from run_teacher_selection_v28 import (  # noqa: E402
    SPECIFIC_CUE_RULES,
    contains_any,
    context_text,
    evaluate as base_evaluate,
    family_card,
    generic,
    mechanism_phrase,
    norm,
    read_jsonl,
)


GENERIC_ACTION_MISSING = "Confirm the existing condition, treatment absence, and site-specific feasibility."


def stated_lane_count(payload: dict[str, Any]) -> float | None:
    """Return an explicitly stated total lane count, if one is available."""
    ctx = payload.get("target_context") or {}
    value = ctx.get("num_lanes")
    try:
        if value not in (None, ""):
            return float(value)
    except (TypeError, ValueError):
        pass
    text = norm(payload.get("user_text"))
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}
    match = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight) lane\b", text)
    if not match:
        return None
    token = match.group(1)
    return float(words.get(token, int(token) if token.isdigit() else 0)) or None


def lane_count_is_treatment_defining(countermeasure: str) -> bool:
    """Whether a lane-count mismatch changes the treatment itself.

    Portable treatments such as lighting, markings, signs, and friction
    treatments may still be considered with an explicit transferability caveat.
    Cross-section conversions must retain lane count as a hard constraint.
    """
    cm = norm(countermeasure)
    return contains_any(cm, (
        "road diet", "convert 2 lane", "convert two lane", "convert 4 lane", "convert four lane",
        "4 to 3 lanes", "four lane roadways to three", "add a through lane", "add through lane",
        "add one lane", "add two lanes", "number of lanes", "presence of two lane",
        "widen from 2 lanes", "widen from two lanes", "reduce from 4 lanes", "reduce from four lanes",
        "two way left turn lane", "twtl", "on two lane road",
    ))


def crash_group(payload: dict[str, Any]) -> str:
    ctx = payload.get("target_context") or {}
    text = norm(f"{ctx.get('crash_type', '')} {payload.get('user_text', '')}")
    for group, terms in (
        ("pedestrian", ("pedestrian", "walking", "people crossing", "person crossing")),
        ("bicycle", ("bicycle", "cyclist", "bike crash", "people on bicycles", "person on a bicycle")),
        ("left_turn", ("left turn", "left turning")),
        ("right_turn", ("right turn", "right turning")),
        ("rear_end", ("rear end", "rearend")),
        ("run_off_road", ("run off road", "roadside crash", "fixed object", "lane departure")),
        ("head_on", ("head on", "cross median")),
        ("sideswipe", ("sideswipe",)),
        ("angle", ("angle", "side impact", "right angle", "crossing path")),
        ("nighttime", ("nighttime", "after dark", "night crashes", "crashes at night")),
        ("wet", ("wet weather", "wet road", "rain related", "during rain")),
        ("speed", ("speed related", "speeding crash")),
        ("truck", ("truck crash", "heavy vehicle crash")),
        ("broad", ("multivehicle", "multiple vehicle", "different kinds", "different types", "mixed crash")),
    ):
        if contains_any(text, terms):
            return group
    return "other"


def control_group(payload: dict[str, Any]) -> str:
    ctx = payload.get("target_context") or {}
    text = norm(f"{ctx.get('traffic_control_type', '')} {payload.get('user_text', '')}")
    if contains_any(text, ("signalized", "traffic signal", "traffic signals", "signal control")):
        return "signalized"
    if contains_any(text, ("stop controlled", "stop sign", "stop signs")):
        return "stop"
    if contains_any(text, ("yield sign", "yield signs", "yield controlled")):
        return "yield"
    if contains_any(text, ("uncontrolled", "no traffic control")):
        return "uncontrolled"
    return "unknown"


def facility_group(payload: dict[str, Any]) -> str:
    ctx = payload.get("target_context") or {}
    text = norm(f"{ctx.get('facility_type', '')} {payload.get('user_text', '')}")
    if contains_any(text, ("curve", "bend", "reverse curve")) and contains_any(text, ("side road junction", "junction")):
        return "mixed"
    if contains_any(text, ("intersection", "junction")):
        return "intersection"
    if contains_any(text, ("ramp", "interchange")):
        return "interchange"
    return "segment"


def add_reject(item: dict[str, Any], reason: str) -> None:
    if reason not in item["rejects"]:
        item["rejects"].append(reason)


def add_missing(item: dict[str, Any], reason: str, penalty: float = 1.5) -> None:
    if reason not in item["missing"]:
        item["missing"].append(reason)
        item["score"] -= penalty


def evaluate(payload: dict[str, Any], card: dict[str, Any]) -> dict[str, Any]:
    # The query-only parser represents nighttime as a crash pattern. Mirror an
    # explicit user phrase such as "after dark" into the applicability field
    # so a Night card does not ask the author to reconfirm information already
    # present in the visible Query.
    visible_text = norm(payload.get("user_text"))
    effective_payload = payload
    if contains_any(visible_text, ("night", "nighttime", "after dark", "crashes at night")):
        context = dict(payload.get("target_context") or {})
        if context.get("crash_time_of_day") in (None, "", []):
            context["crash_time_of_day"] = "Night"
            effective_payload = {**payload, "target_context": context}

    item = base_evaluate(effective_payload, card)
    payload = effective_payload
    item["transferability_relaxed"] = False
    qtext = context_text(payload)
    cm = norm(card.get("countermeasure"))
    crash = crash_group(payload)
    control = control_group(payload)
    facility = facility_group(payload)
    card_intersection_related = norm((card.get("conditions") or {}).get("intersection_related"))
    card_control = norm((card.get("conditions") or {}).get("traffic_control_type"))
    conditions = card.get("conditions") or {}

    # Natural user wording such as "people crossing the road" is an explicit
    # pedestrian context even though older rule code looked only for the word
    # "pedestrian". Do not reject active-transport evidence as if the road user
    # were absent from the Query.
    if crash == "pedestrian" and item["broad"] == "active_transport":
        item["rejects"] = [
            value for value in item["rejects"]
            if value != "the treatment requires a facility or road-user context not stated in the query"
        ]

    # Pairwise natural-language audit: do not let crash-type metadata turn a
    # semantically different road user or a capacity-expansion bundle into a
    # pedestrian treatment.
    if crash == "pedestrian" and contains_any(cm, ("wildlife crossing", "animal crossing", "deer crossing")):
        add_reject(item, "wildlife-crossing evidence does not address the stated human-pedestrian problem")
    if crash == "pedestrian" and contains_any(cm, ("add a through lane", "add through lane")):
        add_reject(item, "a motor-vehicle capacity expansion is not a defensible pedestrian-safety treatment for this Query")

    # Ordinary feasibility checks are prerequisites, not missing decisive
    # applicability conditions. Keeping them as "missing" made every action
    # card conditional regardless of how well it matched the visible Query.
    generic_missing_count = sum(value == GENERIC_ACTION_MISSING for value in item["missing"])
    item["missing"] = [value for value in item["missing"] if value != GENERIC_ACTION_MISSING]
    item["score"] += 1.75 * generic_missing_count

    # Geographic setting is usually an external-validity limitation rather
    # than a logical impossibility. Preserve the evidence as conditional and
    # make the transferability caveat visible instead of deleting the whole
    # treatment direction (for example, rural evidence for a suburban site).
    area_conflict = "area type conflicts with the stated context"
    if area_conflict in item["rejects"]:
        item["rejects"] = [value for value in item["rejects"] if value != area_conflict]
        query_area = (payload.get("target_context") or {}).get("area_type")
        evidence_area = conditions.get("area_type")
        add_missing(
            item,
            f"Confirm transferability from the evidence area ({evidence_area}) to the stated site ({query_area}).",
            2.0,
        )
        item["transferability_relaxed"] = True

    # Lane count is treated at the Evidence-ID level. A mismatch is hard only
    # when lane count defines the treatment itself (e.g., a road diet). For
    # portable treatments it becomes a conditional external-validity caveat.
    lanes = stated_lane_count(payload)
    min_lanes = conditions.get("min_num_lanes")
    max_lanes = conditions.get("max_num_lanes")
    try:
        min_lanes_f = float(min_lanes) if min_lanes not in (None, "") else None
        max_lanes_f = float(max_lanes) if max_lanes not in (None, "") else None
    except (TypeError, ValueError):
        min_lanes_f = max_lanes_f = None
    lane_conflict = lanes is not None and (
        (min_lanes_f is not None and lanes < min_lanes_f)
        or (max_lanes_f is not None and lanes > max_lanes_f)
    )
    if lane_conflict:
        evidence_range = (
            f"{min_lanes_f:g}–{max_lanes_f:g} lanes" if min_lanes_f is not None and max_lanes_f is not None
            else f"at least {min_lanes_f:g} lanes" if min_lanes_f is not None
            else f"at most {max_lanes_f:g} lanes"
        )
        if lane_count_is_treatment_defining(cm):
            add_reject(item, f"the lane-count-dependent treatment conflicts with the stated {lanes:g}-lane site")
        else:
            add_missing(
                item,
                f"Confirm transferability from the evidence lane range ({evidence_range}) to the stated {lanes:g}-lane site.",
                2.0,
            )
            item["transferability_relaxed"] = True
    elif lanes is not None and (min_lanes_f is not None or max_lanes_f is not None):
        item["score"] += 1.0
    elif lanes is None and (min_lanes_f is not None or max_lanes_f is not None) and lane_count_is_treatment_defining(cm):
        evidence_range = (
            f"{min_lanes_f:g}–{max_lanes_f:g} lanes" if min_lanes_f is not None and max_lanes_f is not None
            else f"at least {min_lanes_f:g} lanes" if min_lanes_f is not None
            else f"at most {max_lanes_f:g} lanes"
        )
        add_missing(item, f"Confirm that the site cross section matches the evidence lane range ({evidence_range}).", 1.5)

    # Broad but real crash descriptions may support broad all-crash evidence.
    # They should not be automatically forced to zero labels. More specific
    # cards require a pairwise audit because a visible facility deficiency
    # (for example, a narrow shoulder) may independently support a treatment.
    if crash == "broad" and facility == "intersection":
        item["rejects"] = [
            value for value in item["rejects"]
            if value != "the query lacks enough crash-mechanism or operating-condition information for evidence selection"
        ]
        item["score"] -= 3.0

    if facility == "intersection" and card_intersection_related == "no":
        add_reject(item, "segment evidence conflicts with the stated intersection context")
    if facility == "segment" and card_intersection_related == "yes":
        # A crash name such as angle or left-turn does not establish that a
        # road-segment Query contains an intersection. The conflict point must
        # be explicit in the visible Query; otherwise intersection-only cards
        # are a facility mismatch, not a conditional option.
        add_reject(item, "intersection-specific evidence is not supported by the visible road-segment Query")

    if card_control and not generic(card_control):
        if control == "signalized" and contains_any(card_control, ("uncontrolled", "stop controlled", "yield")):
            add_reject(item, "the evidence control condition conflicts with the stated signal control")

    signal_operation = contains_any(cm, (
        "signal head", "signal phasing", "left turn phas", "change interval",
        "yellow interval", "all red", "red light camera", "red light indicator",
        "adaptive traffic signal", "signal visibility", "signal backplate",
        "leading pedestrian interval", "pedestrian countdown", "green extension",
    ))
    signal_install = contains_any(cm, (
        "install a traffic signal", "install traffic signal", "install signals",
        "signalization", "convert to signal", "convert intersection to signal",
    ))
    if control in {"stop", "yield", "uncontrolled"} and "signalized" in card_control and not signal_install:
        add_reject(item, "the evidence control condition conflicts with the stated nonsignalized control")
    stop_operation = contains_any(cm, (
        "stop sign", "stop ahead", "stop control", "stop controlled",
    ))
    stop_install = contains_any(cm, (
        "install two way stop controlled",
        "install a stop sign on both minor approaches",
        "install stop control",
    ))
    yield_operation = contains_any(cm, ("yield sign", "yield controlled", "yield control"))

    if signal_operation and control != "signalized" and not signal_install:
        add_reject(item, "signal-operation evidence conflicts with the stated nonsignalized control")
    if stop_operation and control != "stop" and not contains_any(cm, ("install", "convert", "replace yield")):
        add_reject(item, "stop-control operation conflicts with the stated intersection control")
    if stop_install and control == "uncontrolled":
        item["rejects"] = [
            value for value in item["rejects"]
            if value != "traffic-control type conflicts with the stated context"
        ]
        prior = norm(conditions.get("prior_condition"))
        if "no stop signs" in prior and contains_any(qtext, ("no traffic control", "uncontrolled")):
            item["missing"] = [
                value for value in item["missing"]
                if not value.startswith("Verify the required prior/site condition:")
            ]
        add_missing(
            item,
            "Confirm approach priority, sight distance, traffic volumes, and warrants before establishing stop control.",
            1.0,
        )
        # Establishing basic approach priority is a first-line direction at an
        # uncontrolled vehicle intersection. Keep it ahead of more speculative
        # turn-lane or major conversion alternatives when the evidence fits.
        item["score"] += 20.0
    if yield_operation and control not in {"yield", "unknown"} and not contains_any(cm, ("install", "convert")):
        add_reject(item, "yield-control operation conflicts with the stated intersection control")

    if signal_install and control == "signalized":
        add_reject(item, "the Query already states that the intersection is signalized")
    if signal_install and control in {"stop", "yield", "uncontrolled"}:
        add_missing(item, "Confirm that traffic-signal warrants and operational feasibility are satisfied.", 1.0)

    stop_conversion = (
        "convert" in cm
        and "all way stop" in cm
        and contains_any(cm, ("two way", "minor road"))
    )
    if stop_conversion and control != "stop":
        add_reject(item, "the treatment requires an existing two-way or minor-road stop control not stated in the Query")
    elif stop_conversion:
        add_missing(item, "Confirm that the current control is minor-road/two-way stop rather than all-way stop.", 1.0)
        if crash == "rear_end":
            add_reject(item, "all-way stop conversion is not tied to the stated rear-end crash mechanism")
    if "minor road stop control" in cm and "roundabout" in cm:
        if control == "stop":
            add_missing(item, "Confirm that the existing stop control is minor-road rather than all-way stop.", 1.0)
        else:
            add_reject(item, "the roundabout-conversion evidence requires existing minor-road stop control")
    if "convert signalized intersection" in cm and control != "signalized":
        add_reject(item, "the conversion requires an existing signalized intersection not stated in the Query")
    if contains_any(cm, ("replace 8 inch", "replace standard stop sign")):
        add_missing(item, "Confirm that the stated existing device is present and is the version being replaced.", 2.0)

    if contains_any(cm, ("install twltl", "install two way left turn lane", "install two-way left-turn lane")):
        add_missing(item, "Confirm that no TWLTL is currently present and that the available cross section can accommodate it.", 1.5)

    if "change driveway type" in cm and not contains_any(
        qtext, ("commercial driveway", "drive through", "drive-through", "twtl", "two way left turn")
    ):
        add_reject(item, "the driveway-type card requires a specific commercial-driveway and cross-section condition not stated in the Query")
    if contains_any(cm, ("add a through lane on both directions and a raised median", "add through lane")) and not contains_any(
        qtext, ("congestion", "capacity", "through lane", "raised median")
    ):
        add_reject(item, "the capacity-expansion treatment is not supported by the visible Query")

    shoulder_treatment = contains_any(cm, (
        "pave shoulder", "pave deteriorated shoulder", "widen shoulder", "widen paved shoulder",
        "add new paved shoulder",
    ))
    if shoulder_treatment and "shoulder" not in qtext:
        add_reject(item, "a shoulder improvement requires a shoulder condition stated in the Query")
    if contains_any(cm, ("road diet", "convert 4 lane", "convert four lane")) and not contains_any(
        qtext, ("four lane", "4 lane", "four-lane", "4-lane", "undivided")
    ):
        add_reject(item, "the road-diet cross section is not stated in the Query")
    if contains_any(cm, ("passing relief lane", "passing lane", "climbing lane")) and not contains_any(
        qtext, ("passing", "overtaking", "slow vehicle", "upgrade", "steep grade")
    ):
        add_reject(item, "the passing-lane treatment is not tied to a visible passing or grade condition")
    if "widen narrow pavement" in cm and not contains_any(qtext, ("narrow pavement", "narrow roadway", "narrow road")):
        add_reject(item, "pavement widening requires a narrow-road condition stated in the Query")
    if contains_any(cm, ("decrease lane width", "decrease outside paved shoulder width", "reduce lane width")):
        add_reject(item, "reducing lane or shoulder width is not a defensible response to the stated crash problem")
    if re.search(r"lower posted speed from\s+\d+", cm) and not contains_any(qtext, ("posted speed", "speed limit")):
        add_reject(item, "the fixed numeric speed-limit change requires the existing posted speed stated in the Query")

    # Embankment/barrier cards encode mutually different prior-site states.
    # A generic lane-departure Query cannot establish either state by itself.
    if "change barrier along embankment" in cm and not contains_any(qtext, ("barrier", "guardrail")):
        add_reject(item, "the treatment assumes an existing roadside barrier that is not stated in the Query")
    if "new guardrail along embankment" in cm and not contains_any(
        qtext, ("embankment", "drop off", "steep slope", "roadside hazard")
    ):
        add_reject(item, "embankment-specific guardrail evidence requires a roadside hazard stated in the Query")

    if contains_any(cm, ("red light camera", "red light indicator")) and not contains_any(qtext, ("red light", "signal violation")):
        add_missing(item, "Confirm that red-light running materially contributes to the crashes.", 2.0)

    pedestrian_beacon = contains_any(cm, ("pedestrian hybrid beacon", "hawk", "rectangular rapid flashing beacon", "rrfb"))
    if pedestrian_beacon and control == "signalized":
        add_reject(item, "a PHB/RRFB treatment is not supported at the stated signalized intersection")
    elif pedestrian_beacon:
        add_missing(
            item,
            "Confirm pedestrian demand, approach speed, crossing geometry, and beacon warrants.",
            1.0,
        )

    if "high visibility crosswalk" in cm:
        item["missing"] = [
            value for value in item["missing"]
            if not (
                value.startswith("Verify the required prior/site condition:")
                and "aim to increase awareness of pedestrians" in value.lower()
            )
        ]

    left_turn_treatment = item["broad"] == "intersection_left_turn_treatment" or contains_any(cm, (
        "left turn lane", "left turn phas", "flashing yellow arrow", "positive offset",
        "median u turn", "indirect left turn", "left turn prohibition",
    ))
    if left_turn_treatment and crash not in {"left_turn"}:
        if crash == "angle":
            add_reject(item, "a left-turn-specific treatment is not supported without a visible left-turn cue")
        else:
            add_reject(item, "a left-turn-specific treatment is not supported by the stated crash pattern")

    right_turn_treatment = item["broad"] == "intersection_right_turn_treatment" or contains_any(
        cm, ("right turn lane", "right turn channel", "right turn phas")
    )
    if right_turn_treatment and crash not in {"right_turn"}:
        if crash == "angle":
            add_reject(item, "a right-turn-specific treatment is not supported without a visible right-turn cue")
        else:
            add_reject(item, "a right-turn-specific treatment is not supported by the stated crash pattern")

    if contains_any(cm, ("continuous flow intersection", "displaced left turn")):
        geometry = norm((payload.get("target_context") or {}).get("intersection_geometry"))
        if contains_any(geometry, ("3 leg", "three leg")):
            add_reject(item, "the intersection geometry does not support this specialized four-leg design")
        else:
            add_missing(item, "Confirm sufficient right-of-way, demand, and geometric feasibility for the specialized design.", 2.0)

    if contains_any(cm, ("j turn", "restricted crossing u turn", "rcut")):
        if not contains_any(qtext, ("divided", "median")):
            add_reject(item, "the Query does not state the divided-road or median context required for a J-turn treatment")
        else:
            add_missing(item, "Confirm median, access, turning-demand, and right-of-way feasibility for a J-turn.", 1.5)

    if (
        contains_any(cm, ("time limited parking", "parking restriction", "angle parking", "parallel parking"))
        or "parking related" in norm(conditions.get("crash_type"))
    ) and "parking" not in qtext:
        add_reject(item, "parking treatment requires a visible parking condition or parking-related crash cue")

    if "bridge segment" in qtext and contains_any(cm, ("safety edge", "sideslope", "side slope", "flatten slope")):
        add_reject(item, "ordinary roadway-edge or sideslope treatment is not applicable on the stated bridge segment")
    if "bridge segment" in qtext and contains_any(cm, ("rumble strip", "rumble stripe")):
        add_missing(
            item,
            "Confirm that the treatment is placed on a compatible bridge approach/transition rather than an unsuitable bridge deck.",
            1.0,
        )

    access_specific = contains_any(cm, ("driveway", "access point", "access points"))
    if access_specific and not contains_any(qtext, ("driveway", "access point", "access points", "closely spaced access")):
        add_reject(item, "the treatment requires a driveway/access condition not stated in the Query")
    if contains_any(cm, ("u turn", "u-turn")) and not contains_any(qtext, ("u turn", "u-turn")):
        add_reject(item, "the treatment requires a U-turn movement or facility not stated in the Query")

    if "roundabout" in cm and facility != "intersection":
        add_reject(item, "roundabout conversion requires an at-grade intersection context")

    if contains_any(cm, ("diverging diamond interchange", "double crossover diamond", "convert diamond interchange")):
        if "diamond interchange" not in qtext:
            add_reject(item, "the conversion requires an existing diamond interchange, which conflicts with or is absent from the Query")
        else:
            add_missing(item, "Confirm traffic demand, right-of-way, geometric feasibility, and operational warrants for DDI conversion.", 1.0)
    if contains_any(cm, ("convert a type i exit ramp", "type i exit ramp to a type iii")) and not contains_any(
        qtext, ("type i exit ramp", "type 1 exit ramp")
    ):
        add_reject(item, "the conversion requires an existing Type I exit ramp not stated in the Query")

    if contains_any(cm, ("frontage road",)) and not contains_any(qtext, ("frontage road",)):
        add_reject(item, "the treatment requires a frontage-road context not stated in the Query")

    if contains_any(cm, ("railroad", "grade crossing")) and not contains_any(qtext, ("railroad", "rail crossing", "grade crossing")):
        add_reject(item, "the treatment requires a railroad-grade-crossing context")

    # Do not let a specialized condition dominate a broad Query simply because
    # its card has a high star rating.
    if contains_any(cm, ("high friction surface", "improve pavement friction")) and not contains_any(
        qtext, ("wet", "rain", "slick", "skid", "friction", "curve", "bend")
    ):
        add_reject(item, "a friction-related, wet-weather, or curve crash mechanism is not stated in the Query")
    if contains_any(cm, ("install lighting", "roadway lighting", "street lighting", "intersection illumination")) and not contains_any(
        qtext, ("night", "dark", "lighting", "unlit")
    ):
        add_reject(item, "a nighttime or visibility crash mechanism is not stated in the Query")
    if contains_any(cm, ("install lighting", "roadway lighting", "street lighting", "intersection illumination")) and contains_any(
        qtext, ("night", "dark")
    ) and not contains_any(qtext, ("unlit", "no lighting", "poor lighting", "inadequate lighting")):
        add_missing(item, "Confirm that existing interchange lighting is absent or inadequate at the crash location.", 1.0)
    if "interchange lighting" in cm and contains_any(cm, ("full to partial", "partial plus to partial")):
        add_reject(item, "reducing existing interchange lighting is not retained as a safety countermeasure recommendation")

    if contains_any(cm, ("auxiliary lane between an entrance ramp and exit ramp", "auxiliary lane between entrance and exit")):
        paired_ramp_context = "weaving" in qtext or ("entrance ramp" in qtext and "exit ramp" in qtext)
        if not paired_ramp_context:
            add_reject(item, "the auxiliary-lane treatment requires a weaving or paired entrance/exit-ramp context not stated in the Query")
        else:
            add_missing(item, "Confirm an entrance-to-exit weaving segment, available width, and operational need for an auxiliary lane.", 1.0)

    if "ramp meter" in cm:
        ramp_meter_context = contains_any(qtext, ("entrance ramp", "on ramp", "on-ramp", "merge", "weaving"))
        if not ramp_meter_context or ("ramp terminal" in qtext and not ramp_meter_context):
            add_reject(item, "ramp metering requires an entrance-ramp, merge, or weaving context rather than a crossroad ramp terminal")
        else:
            add_missing(item, "Confirm peak-period demand, ramp storage, queue spillback risk, and freeway operational warrants for ramp metering.", 1.0)

    if "fixed object" in qtext and not contains_any(qtext, ("sideswipe", "lane change", "rear end", "multivehicle")):
        if contains_any(cm, ("auxiliary lane between an entrance ramp and exit ramp", "ramp meter", "diverging diamond")):
            add_reject(item, "the traffic-flow treatment is not tied to the stated fixed-object crash pattern")

    conditions = card.get("conditions") or {}
    evidence_crash_type = norm(conditions.get("crash_type"))
    evidence_time = norm(conditions.get("crash_time_of_day"))
    if ("nighttime" in evidence_crash_type or evidence_time == "night") and not contains_any(
        qtext, ("night", "after dark", "low light", "unlit")
    ):
        add_reject(item, "the Evidence Card is nighttime-specific but the Query does not state a nighttime condition")

    family_text = norm(f"{item['family']} {cm}")
    if crash == "nighttime" and not contains_any(
        family_text, ("lighting", "illumination", "visibility", "delineation", "marking", "pavement marker", "reflect", "chevron")
    ):
        add_reject(item, "the treatment mechanism is not tied to the stated nighttime crash pattern")
    if crash == "wet" and not contains_any(
        family_text, ("friction", "drainage", "groov", "pavement", "delineation", "marking", "warning")
    ):
        add_reject(item, "the treatment mechanism is not tied to the stated wet-weather crash pattern")
    if crash == "wet" and contains_any(cm, ("inlaid pavement marker", "raised pavement marker")) and not contains_any(
        qtext, ("night", "dark", "visibility", "delineation", "marking")
    ):
        add_reject(item, "pavement-marker evidence is not tied to the stated wet-weather mechanism without a visibility cue")

    skew_specific_match = contains_any(qtext, ("skewed", "skew intersection")) and contains_any(
        cm, ("realign", "sight distance", "triangle sight", "intersection angle", "two three leg intersections")
    )
    if skew_specific_match:
        item["rejects"] = [
            value for value in item["rejects"]
            if value != "the treatment mechanism is not sufficiently tied to the stated crash pattern"
        ]
        item["score"] += 18.0

    if "black spot treatment programme" in cm:
        add_reject(item, "the card names a general programme rather than a sufficiently specific implementable treatment")
    if contains_any(cm, ("absence of access points", "presence of access points")):
        add_reject(item, "the card describes a comparison condition rather than an implementable countermeasure")

    item["missing"] = list(dict.fromkeys(item["missing"]))[:5]
    item["rejects"] = list(dict.fromkeys(item["rejects"]))
    item["status"] = "conditional" if item["missing"] else "direct"
    item["score"] = round(float(item["score"]), 3)
    return item


def select_row(row: dict[str, Any]) -> dict[str, Any]:
    payload = row["teacher_input"]
    evaluated = [evaluate(payload, card) for card in payload.get("candidates") or []]

    max_star_by_family: dict[str, float] = {}
    for item in evaluated:
        if not item["rejects"]:
            max_star_by_family[item["family"]] = max(max_star_by_family.get(item["family"], 0), item["star"])
    for item in evaluated:
        if item["star"] == 2 and max_star_by_family.get(item["family"], 0) >= 3:
            add_reject(item, "a higher-quality candidate covers the same mechanism")

    eligible = [item for item in evaluated if not item["rejects"]]
    crash = crash_group(payload)
    control = control_group(payload)

    def road_user_priority(item: dict[str, Any]) -> int:
        if crash in {"pedestrian", "bicycle"}:
            return 0 if item["broad"] == "active_transport" else 1
        return 0

    eligible.sort(key=lambda x: (
        road_user_priority(x), x["status"] != "direct", -x["score"], int(x["card"]["retrieval_rank"])
    ))
    selected: list[dict[str, Any]] = []
    used_families: set[str] = set()

    def diversity_key(item: dict[str, Any]) -> str:
        cm = norm(item["card"].get("countermeasure"))
        if "channelization" in cm:
            return "intersection_channelization"
        if "roundabout" in cm:
            return "intersection_roundabout"
        if item["broad"] == "intersection_left_turn_treatment":
            return "intersection_left_turn_treatment"
        if item["broad"] == "intersection_right_turn_treatment":
            return "intersection_right_turn_treatment"
        if contains_any(cm, ("signal head", "signal backplate", "red light indicator")):
            return "signal_conspicuity"
        if contains_any(cm, ("illumination", "lighting")):
            return "lighting"
        if "ramp meter" in cm:
            return "ramp_metering"
        if contains_any(cm, ("rumble strip", "rumble stripe")):
            return "rumble_strip"
        if contains_any(cm, ("reduction in mean speed", "reduce mean speed")):
            return "mean_speed_reduction"
        return item["family"]
    best_by_status: dict[str, float] = {}
    for item in eligible:
        best_by_status.setdefault(item["status"], item["score"])
        # Keep the original tight score window for ordinary candidates. A
        # portable treatment retained only through an explicit geographic or
        # lane-transferability caveat may sit farther below the best card, yet
        # still represent a valuable and distinct treatment direction.
        outside_tight_window = item["score"] < best_by_status[item["status"]] - 5.0
        transferable_exception = item.get("transferability_relaxed") and item["score"] >= 24.0
        uncontrolled_stop_exception = (
            item["family"] == "intersection_control_minor_road_stop"
            and control_group(payload) == "uncontrolled"
            and item["score"] >= 24.0
        )
        if outside_tight_window and not transferable_exception and not uncontrolled_stop_exception:
            continue
        key = diversity_key(item)
        if key in used_families:
            continue
        selected.append(item)
        used_families.add(key)
        if len(selected) == 3:
            break

    recommendations: list[dict[str, Any]] = []
    for rank, item in enumerate(selected, 1):
        card = item["card"]
        cm = str(card.get("countermeasure") or "the selected treatment")
        quality = int(item["star"]) if float(item["star"]).is_integer() else item["star"]
        prerequisites = list(item["missing"])
        if not prerequisites:
            prerequisites = ["Complete routine site-specific feasibility and implementation checks."]
        recommendations.append({
            "rank": rank,
            "evidence_ids": [str(card["evidence_id"])],
            "applicability_status": item["status"],
            "missing_critical_conditions": list(item["missing"]),
            "low_quality": item["star"] <= 2,
            "reason": (
                f"{cm} {mechanism_phrase(item['broad'], cm)} and is compatible with the visible "
                f"Query context; the evidence has a {quality}-star quality rating."
            ),
            "prerequisites": prerequisites,
            "teacher_score": item["score"],
            "candidate_rank": int(card["retrieval_rank"]),
        })

    selected_ids = {rec["evidence_ids"][0] for rec in recommendations}
    rejected: list[dict[str, str]] = []
    for item in sorted(evaluated, key=lambda x: int(x["card"]["retrieval_rank"])):
        eid = str(item["card"]["evidence_id"])
        if eid in selected_ids:
            continue
        if item["rejects"]:
            reason = "; ".join(item["rejects"][:2])
        elif diversity_key(item) in used_families:
            reason = "A stronger selected card already represents this mechanism."
        else:
            reason = "Stronger or more directly applicable candidates were selected."
        rejected.append({"evidence_id": eid, "reason": reason})
        if len(rejected) == 3:
            break

    return {
        "qid": row["qid"],
        "scenario_group_id": row.get("scenario_group_id"),
        "teacher_output": {"recommendations": recommendations, "rejected_top_candidates": rejected},
        "teacher_provenance": {
            "teacher": "Codex-authored v3.9r2 facility-strict transferability-aware candidate-constrained engineering selector",
            "decision_date": "2026-08-24",
            "selection_view": "query_text_query_derived_context_and_frozen_at_most_top30_candidate_cards_only",
            "open_audit_output_visible_to_teacher": False,
            "source_anchor_or_cluster_support_ids_visible_to_teacher": False,
            "locked_test_labels_visible_to_teacher": False,
            "selection_method": "systematic_initial_teacher_labels_with_strict_segment_intersection_separation_and_soft_portable_transferability_pending_author_expert_review",
            "human_expert_review_completed": False,
        },
        "label_status": "initial_teacher_selected_pending_stratified_review",
        "teacher_diagnostics": {
            "eligible_candidates": len(eligible),
            "selected_candidates": len(recommendations),
            "direct_selected": sum(rec["applicability_status"] == "direct" for rec in recommendations),
            "conditional_selected": sum(rec["applicability_status"] == "conditional" for rec in recommendations),
            "specific_observable_cues_matched": sum(
                contains_any(context_text(payload), terms) for terms, _ in SPECIFIC_CUE_RULES
            ),
            "candidate_pool_size": len(payload.get("candidates") or []),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = read_jsonl(args.inputs)
    outputs = [select_row(row) for row in rows]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in outputs) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for row in outputs:
        count = len(row["teacher_output"]["recommendations"])
        counts[str(count)] = counts.get(str(count), 0) + 1
    print(json.dumps({"rows": len(outputs), "recommendations_per_query": counts, "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
