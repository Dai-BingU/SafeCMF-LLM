from __future__ import annotations

import re
from typing import Any, Iterable


FACILITY_AT_GRADE_INTERSECTION = "at_grade_intersection"
FACILITY_INTERCHANGE = "interchange"
FACILITY_SEGMENT = "segment"

FACILITY_TYPES = {
    FACILITY_AT_GRADE_INTERSECTION,
    FACILITY_INTERCHANGE,
    FACILITY_SEGMENT,
}


_INTERCHANGE_KWS: list[str] = [
    "interchange",
    "diamond interchange",
    "diverging diamond",
    "double crossover diamond",
    "ddi",
    "dcd",
    "ramp",
    "on-ramp",
    "off-ramp",
    "entrance ramp",
    "exit ramp",
    "grade-separated",
    "flyover",
    "cloverleaf",
    "freeway",
    "expressway",
]


def _has_any(text: str, kws: Iterable[str]) -> bool:
    t = (text or "").lower()
    return any(k in t for k in kws)


def infer_facility_type_from_user_text(
    user_text: str,
    *,
    intersection_related: str | None = None,
) -> str | None:
    """
    Infer facility type for the *site* described in user_text.

    - If user explicitly mentions an interchange/ramp/freeway context => interchange.
    - Else fallback to intersection_related:
        - yes => at_grade_intersection
        - no  => segment
    """
    if _has_any(user_text or "", _INTERCHANGE_KWS):
        return FACILITY_INTERCHANGE

    ir = (intersection_related or "").strip().lower()
    if ir == "yes":
        return FACILITY_AT_GRADE_INTERSECTION
    if ir == "no":
        return FACILITY_SEGMENT
    return None


def resolve_query_facility_type(
    user_text: str,
    *,
    facility_type: str | None = None,
    intersection_related: str | None = None,
) -> str | None:
    """Resolve query facility type with a small guard for stale parsed context.

    Some reviewed rows inherited `facility_type=interchange` even though the
    natural-language query is a normal signalized/stop-controlled intersection.
    If the text does not mention an interchange/ramp/freeway context and the row
    is intersection-related, prefer at-grade intersection.
    """
    inferred = infer_facility_type_from_user_text(user_text, intersection_related=intersection_related)
    normalized = normalize_facility_type(facility_type)
    if not normalized:
        return inferred
    if (
        normalized == FACILITY_INTERCHANGE
        and inferred == FACILITY_AT_GRADE_INTERSECTION
        and "intersection" in (user_text or "").lower()
    ):
        return FACILITY_AT_GRADE_INTERSECTION
    return normalized


def normalize_evidence_facility_type(row: dict[str, Any]) -> str | None:
    """Normalize facility type for scoring evidence rows.

    A few evidence rows have `facility_type=interchange` while their parsed
    intersection type says "Roadway/roadway (not interchange related)". Treat
    those as at-grade intersections for matching.
    """
    facility = normalize_facility_type(row.get("Facility Type"))
    intersection_type = str(row.get("Intersection Type") or "").lower()
    if facility == FACILITY_INTERCHANGE and "not interchange" in intersection_type:
        return FACILITY_AT_GRADE_INTERSECTION
    return facility


def infer_facility_type_for_evidence(
    *,
    countermeasure: str | None,
    conditions: dict[str, Any] | None,
) -> str | None:
    """
    Infer facility type for an evidence row.

    This is intentionally conservative and primarily relies on the countermeasure text,
    because many evidence rows have generic/unspecified conditions.
    """
    cm = (countermeasure or "").strip().lower()
    if cm and _has_any(cm, _INTERCHANGE_KWS):
        return FACILITY_INTERCHANGE

    c = conditions or {}
    ir = str(c.get("intersection_related") or "").strip().lower()
    if ir == "yes":
        return FACILITY_AT_GRADE_INTERSECTION
    if ir == "no":
        return FACILITY_SEGMENT
    return None


def normalize_facility_type(value: object) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    s = re.sub(r"\s+", "_", s.lower())
    if s in FACILITY_TYPES:
        return s
    # Accept common variants.
    if s in {"at-grade-intersection", "at_grade", "intersection"}:
        return FACILITY_AT_GRADE_INTERSECTION
    if s in {"road_segment", "roadway_segment", "non_intersection", "segment"}:
        return FACILITY_SEGMENT
    if s in {"interchange"}:
        return FACILITY_INTERCHANGE
    return None
