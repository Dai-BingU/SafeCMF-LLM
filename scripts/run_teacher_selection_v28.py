#!/usr/bin/env python3
"""Create cue-sensitive teacher selections from frozen v2.8 Top-30 inputs.

The selector implements the engineering rules in the frozen teacher prompt. Its
outputs are initial Codex-authored systematic prelabels, not completed human
expert review. It never reads allocation anchors or locked-test labels.
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
from cmfrec.countermeasure_family import broad_mechanism_family, mechanism_family  # noqa: E402


GENERIC = {"", "all", "not specified", "unknown", "none", "n a", "no values chosen"}


def norm(value: Any) -> str:
    text = str(value or "").lower().replace("&", " and ").replace("_", " ")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()


def generic(value: Any) -> bool:
    return norm(value) in GENERIC


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def family_card(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "countermeasure": card.get("countermeasure"),
        "countermeasure_category": card.get("category"),
        "countermeasure_subcategory": card.get("subcategory"),
    }


def contains_any(text: str, terms: tuple[str, ...] | list[str] | set[str]) -> bool:
    return any(term in text for term in terms)


def values_overlap(query_value: Any, evidence_value: Any) -> bool:
    q_parts = [norm(part) for part in str(query_value or "").split(",") if norm(part)]
    e = norm(evidence_value)
    if not q_parts or not e or generic(e):
        return True
    aliases = {
        "rear end": {"rear end"}, "run off road": {"run off road", "lane departure"},
        "left turn": {"left turn", "angle"}, "right turn": {"right turn", "angle"},
        "speed related": {"speed related", "speeding"}, "pedestrian": {"pedestrian"},
        "bicycle": {"bicycle", "bike"}, "head on": {"head on"},
        "fixed object": {"fixed object", "run off road"}, "cross median": {"cross median", "head on"},
        "single vehicle": {"single vehicle", "run off road", "fixed object"},
        "multiple vehicle": {"multiple vehicle"}, "sideswipe": {"sideswipe"},
        "angle": {"angle", "left turn", "right turn"},
        "night": {"night", "nighttime"}, "wet": {"wet", "rain"},
        "signalized": {"signalized", "traffic signal"},
        "stop controlled": {"stop controlled", "stop sign"},
        "urban suburban": {"urban", "suburban", "urban suburban"},
        "4 leg": {"4 leg", "four leg"}, "3 leg": {"3 leg", "three leg", "t intersection"},
    }
    q_terms = {term for q in q_parts for term in aliases.get(q, {q})}
    return any(term in e for term in q_terms)


CRASH_FAMILY_TERMS: dict[str, tuple[str, ...]] = {
    "rear end": (
        "signal", "intersection control", "left turn", "access management", "speed", "warning",
        "roundabout", "lane reconfiguration", "lighting", "advance guidance", "queue", "turn lane",
    ),
    "angle": (
        "intersection", "signal", "left turn", "right turn", "access management", "roundabout",
        "stop control", "red light", "lighting", "speed",
    ),
    "left turn": ("left turn", "intersection", "signal", "roundabout", "access management", "median u turn"),
    "right turn": ("right turn", "intersection", "signal", "roundabout", "access management", "active transport"),
    "run off road": ("lane departure", "roadside", "curve", "friction", "shoulder", "speed", "median barrier"),
    "fixed object": ("roadside", "lane departure", "curve", "shoulder", "speed", "guardrail"),
    "single vehicle": ("roadside", "lane departure", "curve", "friction", "shoulder", "speed", "lighting"),
    "head on": ("lane departure", "median barrier", "passing", "curve", "friction", "shoulder", "roadside"),
    "cross median": ("median barrier", "lane departure", "roadside", "friction", "speed"),
    "sideswipe": ("lane departure", "shoulder", "passing", "left turn", "access management", "lane reconfiguration", "marking", "speed"),
    "pedestrian": ("active transport", "pedestrian", "sidewalk", "crossing", "lighting", "speed", "signal", "access management"),
    "bicycle": ("active transport", "bicycle", "bike", "cycle track", "lighting", "speed", "crossing"),
    "speed related": ("speed", "traffic calming", "enforcement", "curve", "roundabout"),
    "parking related": ("parking", "access management", "lane reconfiguration", "active transport"),
    "truck": ("truck", "interchange", "merge", "roadside"),
}


OBSERVED_TERMS: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    (("night", "dark", "lighting", "unlit"), ("lighting", "visibility", "delineation", "marking")),
    (("wet", "rain", "skid", "friction"), ("friction", "wet", "pavement marker", "delineation")),
    (("curve", "bend", "horizontal alignment"), ("curve", "lane departure", "friction", "delineation", "speed")),
    (("queue", "congestion", "platoon"), ("queue", "signal", "turn lane", "access management", "warning", "passing")),
    (("red light", "signal visibility"), ("red light", "signal", "visibility", "change interval")),
    (("driveway", "access point"), ("driveway", "access management", "median", "turn lane")),
    (("shoulder", "edge drop", "drop off"), ("shoulder", "safety edge", "lane departure", "roadside")),
    (("crosswalk", "crossing", "pedestrian"), ("pedestrian", "crossing", "active transport", "sidewalk")),
    (("bicycle", "bike", "cyclist"), ("bicycle", "bike", "cycle", "active transport")),
    (("merge", "weav", "ramp"), ("merge", "interchange", "managed lane", "guidance")),
    (("lane change", "heavy traffic"), ("variable speed", "managed lane", "merge", "lane reconfiguration", "access management")),
    (("sight line", "sight distance", "visibility restricted"), ("offset", "sight distance", "visibility", "left turn")),
    (("speed", "fast"), ("speed", "traffic calming", "enforcement")),
    (("stop sign", "stop controlled"), ("stop control", "stop sign", "intersection warning")),
]


# These rules connect observable scenario facts to plausible treatment
# mechanisms. Query terms describe what a road user or field reviewer could
# realistically report; they never name an Evidence ID or require access to a
# source label. A match only changes the ranking among already retrieved and
# applicability-screened cards.
SPECIFIC_CUE_RULES: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    # Run-off-road / lane departure.
    (("fixed objects are close", "fixed objects close", "objects are close to the traveled way"),
     ("fixed object", "clear zone", "remove", "relocate", "guardrail", "roadside")),
    (("noticeable drop", "edge drop", "drop off", "pavement edge"),
     ("safety edge", "edge", "shoulder")),
    (("usable shoulder narrows", "little recovery space", "shoulder narrows"),
     ("widen shoulder", "pave shoulder", "shoulder", "clear zone", "roadside")),
    (("surface becomes slick", "during rain", "wet surface", "wet curve"),
     ("friction", "high friction", "pavement surface", "skid")),
    (("alignment is difficult to recognize", "curve becomes sharper", "approach a curve faster"),
     ("curve", "chevron", "delineation", "pavement marking", "friction", "speed")),
    (("lane boundary after dark", "lane markings become difficult", "edges are difficult to distinguish",
      "edge line"),
     ("edge line", "edgeline", "pavement marker", "marking", "delineation", "lighting", "rumble")),
    (("drift gradually", "crossing the edge line", "cross the lane edge"),
     ("rumble", "edge line", "edgeline", "marking", "delineation")),
    (("gain speed on a downgrade", "accelerate on a downgrade"),
     ("speed", "warning", "friction", "delineation")),

    # Angle and stop-/signal-controlled intersection conflicts.
    (("unusual angle", "crossing path is difficult", "lane-use path", "path through the intersection is unclear"),
     ("channelization", "pavement marking", "geometry", "roundabout", "delineation")),
    (("side approach see conflicting traffic", "queues near the corner block", "first clear view"),
     ("sight distance", "visibility", "warning", "stop sign", "offset")),
    (("nearby access point", "close to the intersection conflict area"),
     ("access", "driveway", "median", "turn lane")),
    (("hesitate at the control", "crossing gaps become short", "late entries are common",
      "through traffic is already close"),
     ("stop control", "warning", "signal", "roundabout", "gap")),

    # Rear-end and recurring queue scenarios.
    (("traffic-control indication changes", "indication changes"),
     ("change interval", "yellow interval", "all red", "adaptive", "signal")),
    (("lane drop creates", "a lane ends near", "lane ends near"),
     ("lane reconfiguration", "variable speed", "warning", "merge", "lane drop")),
    (("merging traffic produces", "drivers merge late", "acceleration space is limited"),
     ("merge", "variable speed", "ramp", "acceleration lane", "warning")),
    (("turning vehicles slow in a through lane", "block a through lane"),
     ("turn lane", "left turn lane", "right turn lane", "median")),
    (("limited view of stopped vehicles", "stopped traffic becomes visible", "back of the queue",
      "queues extend beyond"),
     ("queue", "advance warning", "warning", "variable speed", "signal")),

    # Pedestrian scenarios.
    (("cross several lanes", "limited refuge", "multiple vehicle lanes"),
     ("refuge", "median", "raised", "crosswalk", "crossing")),
    (("turning vehicles cross the path",),
     ("leading pedestrian", "pedestrian interval", "signal phasing", "protected", "crossing")),
    (("pedestrians late after dark", "hidden by parked", "hidden by queued"),
     ("lighting", "visibility", "parking", "crosswalk")),
    (("walking route ends",),
     ("sidewalk", "walkway", "pedestrian path")),
    (("informal crossing point", "yield inconsistently", "transit stop generates"),
     ("beacon", "rrfb", "raised", "crossing", "yield", "warning")),

    # Left-turn scenarios.
    (("left-turn demand exceeds", "available storage", "left-turning vehicles block"),
     ("left turn lane", "turn lane", "storage", "widen")),
    (("restrict the turning driver's view", "difficulty judging the speed of an approaching"),
     ("offset", "positive offset", "sight distance", "visibility")),
    (("opposing traffic approaches quickly", "turning gaps are short", "heavy opposing flow"),
     ("protected", "left turn phasing", "flashing yellow", "signal", "median u turn")),
    (("turning path is poorly aligned", "late steering corrections"),
     ("channelization", "offset", "geometry", "pavement marking")),

    # Nighttime visibility.
    (("dark areas with little visual guidance", "uneven illumination", "after dark"),
     ("lighting", "illumination", "visibility", "delineation")),
    (("signs and traffic-control features", "blend into the background"),
     ("sign", "backplate", "signal visibility", "lighting")),
    (("crossing area is difficult to detect",),
     ("lighting", "pedestrian", "crossing", "visibility")),

    # Bicycle scenarios.
    (("space for riders narrows", "little lateral clearance", "bicycle route ends abruptly"),
     ("bicycle lane", "bike lane", "cycle track", "shoulder", "separation")),
    (("side road see riders", "riders cross multiple", "turning vehicles cross the path regularly used by riders"),
     ("bicycle crossing", "raised bicycle", "cyclist priority", "bike lane", "cycle track")),
    (("parked vehicles and opening doors",),
     ("parking", "bicycle lane", "bike lane", "buffer")),

    # Wet/friction and drainage.
    (("visibly polished wet surface", "friction is noticeably lower", "becomes slick during rainfall"),
     ("friction", "high friction", "skid", "pavement surface")),
    (("water remains in the wheel paths", "drainage is slow", "wet area remains"),
     ("drainage", "groov", "pavement", "friction")),

    # Mixed-vehicle corridor scenarios.
    (("closely spaced access points", "frequently used access point"),
     ("access", "driveway", "median", "turn lane")),
    (("lane changes are frequent", "late lane-positioning decisions", "abrupt lane changes",
      "change lanes over a short distance"),
     ("variable speed", "managed lane", "lane reconfiguration", "access management", "road diet")),
    (("curbside activity", "parking maneuvers"),
     ("parking", "access", "lane reconfiguration")),
    (("heavy vehicles and passenger cars", "truck and passenger-car speeds"),
     ("truck", "merge", "acceleration lane", "climbing lane")),
]


MECHANISM_PHRASES = {
    "active_transport": "separates or controls vulnerable-road-user movements",
    "lane_departure_visibility_or_control": "reduces roadway-departure likelihood or improves vehicle recovery",
    "roadside_recovery_or_protection": "reduces roadside impact exposure or severity after departure",
    "intersection_control_upgrade": "reduces intersection conflict exposure through a control change",
    "intersection_left_turn_treatment": "reduces turning conflicts through left-turn design or operation",
    "intersection_stop_control_visibility_warning": "improves stop-control conspicuity and driver response",
    "access_management_or_corridor_reconfiguration": "reduces access and lane-changing conflict points",
    "speed_enforcement": "reduces unsafe operating speeds or improves speed compliance",
    "lighting": "improves nighttime visibility",
    "pavement_friction": "improves tire–pavement friction and vehicle control",
    "signal_visibility": "improves signal conspicuity",
    "signal_change_interval": "changes signal clearance timing to reduce signal-related conflicts",
    "median_barrier": "reduces cross-median crash exposure or severity",
}


def context_text(payload: dict[str, Any]) -> str:
    ctx = payload.get("target_context") or {}
    return norm(" ".join([
        str(payload.get("user_text") or ""),
        str(ctx.get("scenario_crash_description") or ""),
        " ".join(str(x) for x in (ctx.get("observed_conditions") or [])),
        str(ctx.get("expert_subcontext") or ""),
    ]))


def effect_is_positive(card: dict[str, Any]) -> bool:
    effect = card.get("effect") or {}
    if effect.get("crf") is not None:
        try:
            return float(effect["crf"]) > 0
        except (TypeError, ValueError):
            return False
    if effect.get("cmf") is not None:
        try:
            return 0 <= float(effect["cmf"]) < 1
        except (TypeError, ValueError):
            return False
    return False


def evaluate(payload: dict[str, Any], card: dict[str, Any]) -> dict[str, Any]:
    ctx = payload.get("target_context") or {}
    cond = card.get("conditions") or {}
    qtext = context_text(payload)
    cm = norm(card.get("countermeasure"))
    fam = mechanism_family(family_card(card))
    broad = broad_mechanism_family(family_card(card))
    family_text = norm(f"{fam} {broad} {cm}")
    diagnostic_family_text = norm(f"{fam} {cm}")
    star = float((card.get("quality") or {}).get("star_quality_rating") or 0)
    rank = int(card.get("retrieval_rank") or 999)
    rejects: list[str] = []
    missing: list[str] = []
    score = 20.0 - 0.22 * rank + 1.5 * star

    if star <= 1:
        rejects.append("evidence quality is one star or unavailable")
    if not effect_is_positive(card):
        rejects.append("the card does not report a positive crash-reduction effect")
    unsafe_direction = contains_any(cm, (
        "increase speed limit", "presence of ", "absence of ", "introduce poor", "reduce sight distance",
        "decrease inside shoulder width", "decrease shoulder width", "reduce shoulder width",
        "narrow lane", "narrow shoulder",
    ))
    # In access-management evidence, "absence of access points" represents the
    # safety condition achieved by closing/consolidating driveways, not an
    # unsafe exposure direction. Keep it eligible when the visible query is
    # explicitly about driveways or access points.
    if "absence of access points" in cm and contains_any(qtext, ("driveway", "access point", "access management")):
        unsafe_direction = False
    if unsafe_direction:
        rejects.append("the card represents a risk factor or unsafe direction rather than a positive treatment")
    if contains_any(cm, (" vs ", " versus ")) and not contains_any(
        cm, ("install", "convert", "improve", "increase", "provide", "implement", "upgrade", "replace", "remove", "set")
    ):
        rejects.append("the card is an exposure comparison rather than an implementable positive treatment")

    facility = norm(ctx.get("facility_type"))
    ir = cond.get("intersection_related")
    if facility == "intersection" and norm(ir) == "no":
        rejects.append("segment evidence conflicts with the intersection query")
    if facility == "segment" and norm(ir) == "yes":
        rejects.append("intersection evidence conflicts with the segment query")

    comparisons = [
        ("traffic_control_type", "traffic_control_type", "traffic-control type"),
        ("area_type", "area_type", "area type"),
        ("intersection_geometry", "intersection_geometry", "intersection geometry"),
        ("crash_time_of_day", "crash_time_of_day", "time of day"),
        ("crash_weather", "crash_weather", "weather/pavement condition"),
        ("roadway_division_type", "roadway_division_type", "roadway division"),
    ]
    for qkey, ekey, label in comparisons:
        qv, ev = ctx.get(qkey), cond.get(ekey)
        if qv not in (None, "", []) and ev not in (None, "", []) and not generic(ev):
            if not values_overlap(qv, ev):
                rejects.append(f"{label} conflicts with the stated context")
            else:
                score += 1.0
        elif qv in (None, "", []) and ev not in (None, "", []) and not generic(ev):
            if label in {"traffic-control type", "intersection geometry", "time of day", "weather/pavement condition"}:
                missing.append(f"Confirm compatible {label}: {ev}.")

    qcrash_raw = ctx.get("crash_type")
    qcrash = norm(qcrash_raw)
    for known_crash in CRASH_FAMILY_TERMS:
        if known_crash in qcrash:
            qcrash = known_crash
            break
    if qcrash in {"", "other"}:
        for inferred in (
            "pedestrian", "bicycle", "run off road", "fixed object", "head on", "cross median",
            "single vehicle", "multiple vehicle", "sideswipe", "rear end", "left turn", "right turn", "angle", "speed related",
            "parking related", "truck",
        ):
            if inferred in qtext.replace("run off road and lane departure", "run off road"):
                qcrash = inferred
                break
    has_problem_signal = qcrash not in {"", "other"} or any(
        contains_any(qtext, query_terms) for query_terms, _ in OBSERVED_TERMS
    )
    if qcrash == "multiple vehicle" and not any(
        contains_any(qtext, query_terms) for query_terms, _ in OBSERVED_TERMS
    ):
        has_problem_signal = False
    if not has_problem_signal:
        rejects.append("the query lacks enough crash-mechanism or operating-condition information for evidence selection")
    ecrash = cond.get("crash_type")
    exact_crash = False
    if qcrash and ecrash not in (None, "", []) and not generic(ecrash):
        exact_crash = values_overlap(qcrash_raw, ecrash)
        if not exact_crash:
            rejects.append("the evidence crash type conflicts with the stated crash pattern")
        else:
            score += 4.0
    elif qcrash and generic(ecrash):
        score -= 1.0

    allowed = CRASH_FAMILY_TERMS.get(qcrash)
    mechanism_match = bool(allowed and contains_any(family_text, allowed))
    if allowed:
        if mechanism_match:
            score += 6.0
        elif exact_crash and qcrash not in {"pedestrian", "bicycle", "speed related", "parking related"}:
            score += 0.5
        else:
            rejects.append("the treatment mechanism is not sufficiently tied to the stated crash pattern")

    # Reward the most diagnosis-specific mechanism over generic treatments that
    # merely have an all-crash CMF at the same facility.
    specific_terms = {
        "left turn": ("left turn", "positive offset"),
        "rear end": ("adaptive", "change interval", "queue", "rear end", "turn lane"),
        "angle": ("red light", "signal visibility", "stop control", "roundabout", "left turn"),
        "pedestrian": ("pedestrian", "crosswalk", "sidewalk", "active transport"),
        "bicycle": ("bicycle", "bike", "cycle", "active transport"),
        "speed related": ("speed", "traffic calming", "enforcement"),
        "parking related": ("parking",),
    }.get(qcrash)
    if specific_terms and contains_any(family_text, specific_terms):
        score += 6.0

    for query_terms, family_terms in OBSERVED_TERMS:
        if contains_any(qtext, query_terms):
            if contains_any(diagnostic_family_text, family_terms):
                score += 6.0
            else:
                score -= 4.0

    # Prefer a treatment that responds to the particular observable cue rather
    # than repeatedly selecting the same generic crash-type trio. Multiple
    # compatible cues may contribute, but cap their aggregate effect so an
    # otherwise weak card cannot win solely through keyword accumulation.
    cue_adjustment = 0.0
    for query_terms, family_terms in SPECIFIC_CUE_RULES:
        if contains_any(qtext, query_terms):
            cue_adjustment += 9.0 if contains_any(diagnostic_family_text, family_terms) else -2.0
    score += max(-4.0, min(18.0, cue_adjustment))

    prior = str(cond.get("prior_condition") or "").strip()
    if prior and not generic(prior):
        missing.append(f"Verify the required prior/site condition: {prior[:180]}.")

    # Treatment wording often carries a decisive prior condition that is absent
    # from the structured fields. Do not call such cards direct matches.
    action_terms = (
        "install ", "convert ", "improve ", "increase ", "provide ", "implement ",
        "upgrade ", "replace ", "remove ", "set ", "flatten ", "widen ", "pave ",
        "lower ", "add ", "modify ", "changing ", "prohibit ", "restrict ", "introduce ",
    )
    if contains_any(cm, action_terms) and not prior:
        missing.append("Confirm the existing condition, treatment absence, and site-specific feasibility.")

    control = norm(ctx.get("traffic_control_type"))
    signal_operation = contains_any(cm, (
        "signal head", "red light camera", "red light indicator", "signal phasing", "left turn phas",
        "change interval", "yellow interval", "all red", "adaptive traffic signal", "signal visibility",
    ))
    stop_operation = contains_any(cm, ("stop sign", "stop controlled", "stop control", "stop ahead"))
    signal_install = contains_any(cm, ("install a traffic signal", "install traffic signal", "signalization"))
    if control == "stop controlled" and signal_operation and not signal_install:
        rejects.append("a signal-operation treatment conflicts with the stated stop-controlled site")
    if control == "signalized" and stop_operation and not contains_any(cm, ("remove signal", "convert")):
        rejects.append("a stop-control treatment conflicts with the stated signalized site")
    if "remove unwarranted signal" in cm and "unwarranted" not in qtext:
        rejects.append("signal removal requires evidence that the existing signal is unwarranted")
    if "convert at grade intersections to diamond interchanges" in cm and contains_any(qtext, ("interchange", "ramp")):
        rejects.append("conversion from an at-grade intersection is inapplicable when the query already describes an interchange/ramp area")

    if contains_any(qtext, ("freeway", "expressway", "interstate")):
        roadway = norm(cond.get("roadway_type"))
        if not generic(roadway) and not contains_any(roadway, ("freeway", "expressway", "interstate")):
            rejects.append("non-freeway evidence conflicts with the stated freeway/expressway segment")
        if contains_any(cm, (
            "two lane road", "two lane highway", "twtl", "two way left turn", "two-way left-turn",
            "passing lane", "climbing lane", "road diet", "four lane roadways to three",
            "4 to 3 lanes", "center turn lane", "add a through lane on both directions and a raised median",
            "widen narrow pavement", "centerline rumble",
        )):
            rejects.append("an arterial/two-lane-road cross-section treatment conflicts with the stated freeway/expressway context")
        prior_norm = norm(cond.get("prior_condition"))
        if contains_any(prior_norm, ("two lane undivided", "total pavement width of 18 to 22 ft")):
            rejects.append("the evidence prior condition describes a narrow undivided roadway rather than a freeway/expressway")

    if contains_any(cm, ("hard shoulder running", "ramp meter", "managed lane")) and not contains_any(
        qtext, ("freeway", "expressway", "interstate", "interchange", "ramp")
    ):
        rejects.append("the treatment requires a freeway, interchange, or ramp context not stated in the query")

    if "speed hump" in cm and not contains_any(qtext, ("local road", "local street", "low speed", "residential")):
        missing.append("Confirm that the site is an appropriate low-speed local or residential street.")
        score -= 5.0

    if "median barrier" in family_text and not contains_any(qtext, ("median", "cross median", "divided")):
        missing.append("Confirm a divided cross section and a material cross-median or median-related crash problem.")
        score -= 4.0

    if (fam.startswith("curve_") or "curve" in cm) and not contains_any(qtext, ("curve", "bend", "horizontal alignment")):
        missing.append("Confirm that the crash pattern is concentrated on a horizontal curve.")
        score -= 6.0

    score -= 1.75 * len(missing)

    # Facility-specific treatments require an explicit cue even when the evidence
    # crash type is generic.
    required_cues = [
        (("railroad", "rail crossing", "grade crossing"), ("railroad", "rail crossing", "grade crossing")),
        (("toll plaza", "tolling"), ("toll",)),
        (("freeway", "ramp", "interchange", "merge"), ("interchange", "ramp", "freeway", "expressway", "merge")),
        (("pedestrian", "crosswalk", "sidewalk"), ("pedestrian", "crosswalk", "walking")),
        (("bicycle", "bike", "cycle track"), ("bicycle", "bike", "cyclist")),
    ]
    for treatment_terms, query_terms in required_cues:
        if contains_any(family_text, treatment_terms) and not contains_any(qtext, query_terms):
            rejects.append("the treatment requires a facility or road-user context not stated in the query")

    # Deduplicate missing-condition statements while preserving order.
    missing = list(dict.fromkeys(missing))[:5]
    status = "conditional" if missing else "direct"
    if status == "direct":
        score += 3.0
    return {
        "card": card,
        "family": fam,
        "broad": broad,
        "star": star,
        "score": round(score, 3),
        "status": status,
        "missing": missing,
        "rejects": list(dict.fromkeys(rejects)),
    }


def mechanism_phrase(broad: str, countermeasure: str) -> str:
    if broad in MECHANISM_PHRASES:
        return MECHANISM_PHRASES[broad]
    for key, phrase in MECHANISM_PHRASES.items():
        if broad.startswith(key):
            return phrase
    return f"provides a distinct safety-treatment mechanism ({countermeasure})"


def select_row(row: dict[str, Any]) -> dict[str, Any]:
    payload = row["teacher_input"]
    evaluated = [evaluate(payload, card) for card in payload.get("candidates") or []]

    # Enforce the two-star rule within a mechanism family.
    max_star_by_family: dict[str, float] = {}
    for item in evaluated:
        if not item["rejects"]:
            max_star_by_family[item["family"]] = max(max_star_by_family.get(item["family"], 0), item["star"])
    for item in evaluated:
        if item["star"] == 2 and max_star_by_family.get(item["family"], 0) >= 3:
            item["rejects"].append("a higher-quality candidate covers the same mechanism")

    eligible = [item for item in evaluated if not item["rejects"]]
    eligible.sort(key=lambda x: (-x["score"], x["status"] != "direct", int(x["card"]["retrieval_rank"])))
    selected: list[dict[str, Any]] = []
    used_families: set[str] = set()
    best_score = eligible[0]["score"] if eligible else None
    for item in eligible:
        # "Up to three" is deliberate: do not fill a slot with a materially
        # weaker alternative simply because it survived the hard filters.
        if best_score is not None and item["score"] < best_score - 6.0:
            continue
        if item["family"] in used_families:
            continue
        selected.append(item)
        used_families.add(item["family"])
        if len(selected) == 3:
            break

    recommendations = []
    for rank, item in enumerate(selected, 1):
        card = item["card"]
        cm = str(card.get("countermeasure") or "the selected treatment")
        quality = int(item["star"]) if float(item["star"]).is_integer() else item["star"]
        reason = (
            f"The treatment {cm} {mechanism_phrase(item['broad'], cm)}; the exposed card is compatible with the stated "
            f"crash/facility context and has a {quality}-star quality rating."
        )
        prerequisites = list(item["missing"])
        if not prerequisites:
            prerequisites = ["Confirm that the treatment is not already present and complete a site-specific feasibility check."]
        recommendations.append({
            "rank": rank,
            "evidence_ids": [str(card["evidence_id"])],
            "applicability_status": item["status"],
            "missing_critical_conditions": item["missing"],
            "low_quality": item["star"] <= 2,
            "reason": reason,
            "prerequisites": prerequisites,
        })

    rejected = []
    selected_ids = {r["evidence_ids"][0] for r in recommendations}
    for item in sorted(evaluated, key=lambda x: int(x["card"]["retrieval_rank"])):
        eid = str(item["card"]["evidence_id"])
        if eid in selected_ids:
            continue
        if item["rejects"]:
            reason = "; ".join(item["rejects"][:2])
        elif item["family"] in used_families:
            reason = "Not selected because it duplicates a mechanism already represented by a stronger candidate."
        else:
            reason = "Not selected because stronger and more directly applicable candidates were available."
        rejected.append({"evidence_id": eid, "reason": reason})
        if len(rejected) == 2:
            break

    return {
        "qid": row["qid"],
        "scenario_group_id": row.get("scenario_group_id"),
        "teacher_output": {"recommendations": recommendations, "rejected_top_candidates": rejected},
        "teacher_provenance": {
            "teacher": "Codex-authored cue-sensitive systematic engineering selector v2.8",
            "decision_date": "2026-08-23",
            "selection_view": "query_text_query_derived_visible_context_and_frozen_top30_candidate_cards_only",
            "source_anchor_or_cluster_support_ids_visible_to_teacher": False,
            "locked_test_labels_visible_to_teacher": False,
            "selection_method": "systematic_initial_prelabel_pending_stratified_human_review",
            "human_expert_review_completed": False,
        },
        "label_status": "initial_teacher_selected_pending_human_review",
        "teacher_diagnostics": {
            "eligible_candidates": len(eligible),
            "selected_candidates": len(recommendations),
            "direct_selected": sum(r["applicability_status"] == "direct" for r in recommendations),
            "conditional_selected": sum(r["applicability_status"] == "conditional" for r in recommendations),
            "specific_observable_cues_matched": sum(
                contains_any(context_text(payload), query_terms) for query_terms, _ in SPECIFIC_CUE_RULES
            ),
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
        n = len(row["teacher_output"]["recommendations"])
        counts[str(n)] = counts.get(str(n), 0) + 1
    print(json.dumps({"rows": len(outputs), "recommendations_per_query": counts, "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
