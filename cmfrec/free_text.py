from __future__ import annotations

import re
from dataclasses import dataclass

from .scoring import Query
from .facility import infer_facility_type_from_user_text
from .context_tags import infer_context_tags_from_user_text


@dataclass(frozen=True)
class InferredQuery:
    query: Query
    notes: list[str]


_RE_MPH = re.compile(r"(?P<val>\d{2,3})\s*(?:mph|mi/h)\b", re.IGNORECASE)
_RE_SPEED_RANGE = re.compile(
    r"(?P<lo>\d{2,3})\s*(?:to|-)\s*(?P<hi>\d{2,3})\s*(?:mph|mi/h)\b",
    re.IGNORECASE,
)

_RE_AADT = re.compile(
    r"(?P<val>\d[\d,]{2,})\s*(?:aadt|adt|vpd|veh/day|vehicles per day|vehicles/day)\b",
    re.IGNORECASE,
)

_RE_MAJOR_AADT = re.compile(
    r"(?:major|main)\s*(?:road|street|approach)?[^\\d]{0,20}(?P<val>\d[\d,]{2,})\s*"
    r"(?:aadt|adt|vpd|veh/day|vehicles per day|vehicles/day)\b",
    re.IGNORECASE,
)

_RE_MINOR_AADT = re.compile(
    r"(?:minor|side)\s*(?:road|street|approach)?[^\\d]{0,20}(?P<val>\d[\d,]{2,})\s*"
    r"(?:aadt|adt|vpd|veh/day|vehicles per day|vehicles/day)\b",
    re.IGNORECASE,
)

_LANE_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
}

_RE_GEOM_4 = re.compile(r"\b(4[-\s]*leg|four[-\s]*leg|four[-\s]*way|cross[-\s]*intersection)\b", re.I)
_RE_GEOM_3 = re.compile(r"\b(3[-\s]*leg|three[-\s]*leg|t[-\s]*intersection|t[-\s]*junction)\b", re.I)
_RE_GEOM_MULTI = re.compile(
    r"\b(more than\s*4\s*legs|multi[-\s]*leg|5[-\s]*leg|6[-\s]*leg|five[-\s]*leg|six[-\s]*leg)\b",
    re.I,
)


def infer_query_from_text(text: str) -> InferredQuery:
    """
    Heuristic extractor for English free-text site context.

    This is intentionally lightweight (no external deps / no LLM). It aims to:
      - normalize common phrases into dataset-aligned field values
      - extract speed limits (mph)
      - infer intersection/control/area/crash type when explicit
    """
    notes: list[str] = []
    t = (text or "").strip()
    tl = t.lower()

    def has_phrase(phrase: str) -> bool:
        p = (phrase or "").strip().lower()
        if not p:
            return False
        return re.search(rf"(?<![a-z0-9]){re.escape(p)}(?![a-z0-9])", tl) is not None

    crash_type: str | None = None
    # Crash type / conflict type heuristics (align to common CMF Clearinghouse labels)
    crash_patterns: list[tuple[str, list[str]]] = [
        ("Rear end", ["rear end", "rear-end", "rearend"]),
        ("Angle", ["angle crash", "angle"]),
        ("Cross median", ["cross median", "cross-median", "crossing the median", "median crossover"]),
        ("Run off road", ["run off road", "run-off-road", "road departure", "off road"]),
        ("Head on", ["head on", "head-on"]),
        ("Sideswipe", ["sideswipe", "side swipe"]),
        ("Fixed object", ["fixed object", "object collision", "tree", "pole"]),
        ("Vehicle/pedestrian", ["pedestrian", "ped"]),
        ("Vehicle/bicycle", ["bicycle", "bicyclist", "cyclist"]),
        ("Nighttime", ["nighttime", "at night", "dark"]),
        ("Dry weather", ["dry weather", "dry road", "clear weather"]),
        ("Wet road", ["wet road", "wet roads", "wet condition", "wet conditions", "wet pavement"]),
        ("Wet weather", ["wet weather", "rain", "snow", "icy", "ice"]),
    ]
    matched_crash_types: list[str] = []
    for label, kws in crash_patterns:
        if any(has_phrase(kw) for kw in kws):
            matched_crash_types.append(label)
    if matched_crash_types:
        crash_type = ",".join(matched_crash_types)
        notes.append(f"Inferred crash_type={crash_type}")

    area_type: str | None = None
    if "rural" in tl:
        area_type = "Rural"
        notes.append("Inferred area_type=Rural")
    elif any(kw in tl for kw in ["village", "countryside", "country side", "remote area", "farm road", "agricultural"]):
        # Common user phrasing that implies rural context.
        area_type = "Rural"
        notes.append("Inferred area_type=Rural (from village/countryside phrasing)")
    elif "urban" in tl:
        area_type = "Urban"
        notes.append("Inferred area_type=Urban")
    elif "suburban" in tl:
        area_type = "Suburban"
        notes.append("Inferred area_type=Suburban")

    intersection_related: str | None = None
    traffic_control_type: str | None = None
    intersection_geometry: str | None = None

    # Detect explicit "non-intersection" first to avoid matching the substring "intersection".
    if any(
        kw in tl
        for kw in [
            "non-intersection",
            "non intersection",
            "nonintersection",
            "midblock",
            "not at an intersection",
            "not at intersection",
            "not intersection-related",
            "not intersection related",
            "not intersection-related",
            "not intersection related",
            "not related to an intersection",
            "not related to intersection",
            "not an intersection",
        ]
    ):
        intersection_related = "no"
        notes.append("Inferred intersection_related=no")
    elif any(kw in tl for kw in ["intersection", "signalized", "stop-controlled", "roundabout"]):
        intersection_related = "yes"
        notes.append("Inferred intersection_related=yes")
    elif "segment" in tl:
        # "segment" alone is ambiguous; treat as non-intersection but with low confidence.
        intersection_related = "no"
        notes.append("Inferred intersection_related=no (from 'segment')")

    if _RE_GEOM_MULTI.search(t):
        intersection_geometry = "More than 4 legs"
        notes.append("Inferred intersection_geometry=More than 4 legs")
        if intersection_related is None:
            intersection_related = "yes"
            notes.append("Inferred intersection_related=yes (from geometry)")
    elif _RE_GEOM_4.search(t):
        intersection_geometry = "4-leg"
        notes.append("Inferred intersection_geometry=4-leg")
        if intersection_related is None:
            intersection_related = "yes"
            notes.append("Inferred intersection_related=yes (from geometry)")
    elif _RE_GEOM_3.search(t):
        intersection_geometry = "3-leg"
        notes.append("Inferred intersection_geometry=3-leg")
        if intersection_related is None:
            intersection_related = "yes"
            notes.append("Inferred intersection_related=yes (from geometry)")

    if any(kw in tl for kw in ["traffic signal", "signalized", "signalised", "green time"]):
        traffic_control_type = "Signalized"
        notes.append("Inferred traffic_control_type=Signalized")
        if intersection_related is None:
            intersection_related = "yes"
            notes.append("Inferred intersection_related=yes (from signalized)")
    elif any(kw in tl for kw in ["stop sign", "two-way stop", "all-way stop", "4-way stop", "stop-controlled"]):
        traffic_control_type = "Stop-controlled"
        notes.append("Inferred traffic_control_type=Stop-controlled")
        if intersection_related is None:
            intersection_related = "yes"
            notes.append("Inferred intersection_related=yes (from stop)")
    elif "roundabout" in tl:
        traffic_control_type = "Roundabout"
        notes.append("Inferred traffic_control_type=Roundabout")
        if intersection_related is None:
            intersection_related = "yes"
            notes.append("Inferred intersection_related=yes (from roundabout)")
    elif any(kw in tl for kw in ["uncontrolled", "no control"]):
        traffic_control_type = "Uncontrolled"
        notes.append("Inferred traffic_control_type=Uncontrolled")

    roadway_type: str | None = None
    if any(kw in tl for kw in ["interstate", "i-"]):
        roadway_type = "Principal Arterial Interstate"
        notes.append("Inferred roadway_type=Principal Arterial Interstate")
    elif "expressway" in tl or "freeway" in tl:
        roadway_type = "Principal Arterial Other Freeways and Expressways"
        notes.append("Inferred roadway_type=Principal Arterial Other Freeways and Expressways")
    elif "minor arterial" in tl:
        roadway_type = "Minor Arterial"
        notes.append("Inferred roadway_type=Minor Arterial")
    elif "principal arterial" in tl or "arterial" in tl:
        roadway_type = "Principal Arterial Other"
        notes.append("Inferred roadway_type=Principal Arterial Other")
    elif "major collector" in tl:
        roadway_type = "Major Collector"
        notes.append("Inferred roadway_type=Major Collector")
    elif "minor collector" in tl:
        roadway_type = "Minor Collector"
        notes.append("Inferred roadway_type=Minor Collector")
    elif "local road" in tl or "local street" in tl:
        roadway_type = "Local"
        notes.append("Inferred roadway_type=Local")

    min_speed_limit: float | None = None
    max_speed_limit: float | None = None
    m_range = _RE_SPEED_RANGE.search(t)
    if m_range:
        min_speed_limit = float(m_range.group("lo"))
        max_speed_limit = float(m_range.group("hi"))
        notes.append(f"Inferred speed range {min_speed_limit}-{max_speed_limit} mph")
    else:
        m = _RE_MPH.search(t)
        if m:
            val = float(m.group("val"))
            min_speed_limit = val
            max_speed_limit = val
            notes.append(f"Inferred speed {val} mph")

    # Traffic volume (AADT/ADT)
    traffic_volume: float | None = None
    major_road_volume: float | None = None
    minor_road_volume: float | None = None
    m_aadt = _RE_AADT.search(t)
    if m_aadt:
        raw = m_aadt.group("val").replace(",", "")
        try:
            traffic_volume = float(raw)
            notes.append(f"Inferred traffic volume {traffic_volume} vpd")
        except Exception:
            pass

    m_major = _RE_MAJOR_AADT.search(t)
    if m_major:
        raw = m_major.group("val").replace(",", "")
        try:
            major_road_volume = float(raw)
            notes.append(f"Inferred major_road_volume_aadt={major_road_volume}")
        except Exception:
            pass

    m_minor = _RE_MINOR_AADT.search(t)
    if m_minor:
        raw = m_minor.group("val").replace(",", "")
        try:
            minor_road_volume = float(raw)
            notes.append(f"Inferred minor_road_volume_aadt={minor_road_volume}")
        except Exception:
            pass

    # Lanes
    num_lanes: float | None = None
    m_lane_num = re.search(r"\b(?P<n>\d)\s*-\s*lane\b|\b(?P<n2>\d)\s*lane", tl)
    if m_lane_num:
        n = m_lane_num.group("n") or m_lane_num.group("n2")
        if n:
            try:
                num_lanes = float(int(n))
                notes.append(f"Inferred num_lanes={int(num_lanes)}")
            except Exception:
                pass
    else:
        m_word = re.search(r"\b(one|two|three|four|five|six|seven|eight)\s*-\s*lane\b", tl)
        if m_word:
            w = m_word.group(1)
            if w in _LANE_WORDS:
                num_lanes = float(_LANE_WORDS[w])
                notes.append(f"Inferred num_lanes={int(num_lanes)}")

    facility_type = infer_facility_type_from_user_text(text, intersection_related=intersection_related)
    if facility_type:
        notes.append(f"Inferred facility_type={facility_type}")

    context_tags = infer_context_tags_from_user_text(text)
    if context_tags:
        notes.append(f"Inferred context_tags={','.join(context_tags)}")

    return InferredQuery(
        query=Query(
            crash_type=crash_type,
            severity=None,
            roadway_type=roadway_type,
            area_type=area_type,
            facility_type=facility_type,
            intersection_related=intersection_related,
            traffic_control_type=traffic_control_type,
            intersection_geometry=intersection_geometry,
            min_speed_limit=min_speed_limit,
            max_speed_limit=max_speed_limit,
            num_lanes=num_lanes,
            traffic_volume_aadt=traffic_volume,
            major_road_volume_aadt=major_road_volume,
            minor_road_volume_aadt=minor_road_volume,
            context_tags=context_tags,
        ),
        notes=notes,
    )
