from __future__ import annotations

import re
from typing import Iterable

_TAG_TRANSIT = "transit"
_TAG_RAIL_CROSSING = "rail_crossing"
_TAG_SCHOOL_ZONE = "school_zone"
_TAG_TOLL_PLAZA = "toll_plaza"
_TAG_INTERCHANGE = "interchange"

ALL_CONTEXT_TAGS = {
    _TAG_TRANSIT,
    _TAG_RAIL_CROSSING,
    _TAG_SCHOOL_ZONE,
    _TAG_TOLL_PLAZA,
    _TAG_INTERCHANGE,
}


def normalize_context_tag(value: object) -> str | None:
    if value is None:
        return None
    s = str(value).strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    return s if s in ALL_CONTEXT_TAGS else None


def normalize_context_tags(values: Iterable[object] | None) -> tuple[str, ...]:
    if not values:
        return ()
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        t = normalize_context_tag(v)
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return tuple(out)


def _contains_keyword(text: str, keyword: str) -> bool:
    if not text or not keyword:
        return False
    # Short acronyms such as DDI/DCD must be token matches; otherwise words
    # like "additional" accidentally trigger interchange context via "ddi".
    if keyword in {"ddi", "dcd"}:
        return re.search(rf"\b{re.escape(keyword)}\b", text) is not None
    return keyword in text


def _contains_any_keyword(text: str, keywords: Iterable[str]) -> bool:
    return any(_contains_keyword(text, keyword) for keyword in keywords)


_TRANSIT_KWS = [
    "transit",
    "tram",
    "streetcar",
    "light rail",
    "lrt",
    "metro",
    "subway",
    "rail platform",
    "platform stop",
    "bus rapid",
]

_RAIL_CROSSING_KWS = [
    "railroad",
    "rail road",
    "railway",
    "train",
    "grade crossing",
    "rail crossing",
    "level crossing",
    "crossing gate",
    "crossbuck",
]

_SCHOOL_KWS = [
    "school",
    "schools",
    "student",
    "students",
    "campus",
    "school zone",
]

_TOLL_KWS = [
    "toll",
    "toll plaza",
    "tollbooth",
    "toll booth",
    "open road toll",
    "ort",
]

_INTERCHANGE_KWS = [
    "interchange",
    "grade-separated",
    "grade separated",
    "freeway",
    "freeways",
    "expressway",
    "expressways",
    "ramp",
    "on-ramp",
    "off-ramp",
    "entrance ramp",
    "exit ramp",
    "merge",
    "diverge",
    "weaving",
    "ddi",
    "dcd",
    "diverging diamond",
    "double crossover diamond",
    "deceleration lane",
    "acceleration lane",
]


def infer_context_tags_from_user_text(user_text: str) -> tuple[str, ...]:
    """
    Infer strong precondition context signals from user text.

    These tags are intended to prevent recommending specialized treatments when
    the user did not mention the necessary context (e.g., transit, rail crossing).
    """
    t = (user_text or "").strip().lower()
    tags: list[str] = []

    def add(tag: str) -> None:
        if tag not in tags:
            tags.append(tag)

    if _contains_any_keyword(t, _TRANSIT_KWS):
        add(_TAG_TRANSIT)
    if _contains_any_keyword(t, _RAIL_CROSSING_KWS):
        add(_TAG_RAIL_CROSSING)
    if _contains_any_keyword(t, _SCHOOL_KWS):
        add(_TAG_SCHOOL_ZONE)
    if _contains_any_keyword(t, _TOLL_KWS):
        add(_TAG_TOLL_PLAZA)
    if _contains_any_keyword(t, _INTERCHANGE_KWS):
        add(_TAG_INTERCHANGE)

    return normalize_context_tags(tags)


def infer_required_context_tags(
    *, countermeasure: str | None, countermeasure_category: str | None
) -> tuple[str, ...]:
    """
    Infer required context tags from evidence metadata.

    This is deliberately conservative: only tag treatments that are obviously
    tied to a specialized context (transit/rail crossing/school/toll).
    """
    cm = (countermeasure or "").strip().lower()
    cat = (countermeasure_category or "").strip().lower()

    tags: list[str] = []

    def add(tag: str) -> None:
        if tag not in tags:
            tags.append(tag)

    if "transit" in cat or any(k in cm for k in ["tram", "streetcar", "light rail", "lrt", "platform"]):
        add(_TAG_TRANSIT)

    if "railroad grade crossing" in cat or "railroad grade crossings" in cat:
        add(_TAG_RAIL_CROSSING)

    if "toll" in cat or "toll" in cm or "toll plaza" in cm:
        add(_TAG_TOLL_PLAZA)

    if "school" in cm or "schools" in cm:
        add(_TAG_SCHOOL_ZONE)

    # Interchange / freeway / grade-separated treatments.
    interchange_kws = [
        "interchange",
        "freeway",
        "expressway",
        "ramp",
        "diverging diamond",
        "double crossover diamond",
        "ddi",
        "dcd",
        "deceleration lane",
        "acceleration lane",
        "grade-separated",
    ]
    if "interchange" in cat or _contains_any_keyword(cm, interchange_kws):
        add(_TAG_INTERCHANGE)

    return normalize_context_tags(tags)
