#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cmfrec.xlsx import read_xlsx_first_sheet  # noqa: E402


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


def _norm_text(value: str) -> str | None:
    v = (value or "").strip()
    return v if v else None


def _as_int(value: str) -> int | None:
    f = _to_float(value)
    if f is None:
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    try:
        return int(round(f))
    except Exception:
        return None


@dataclass(frozen=True)
class EvidenceRow:
    evidence_id: str
    cm_id: str
    countermeasure: str

    countermeasure_category: str | None
    countermeasure_subcategory: str | None

    conditions: dict[str, object]
    effect: dict[str, object]
    quality: dict[str, object]
    citation: dict[str, object]

    source: dict[str, object]
    raw: dict[str, str] | None


def _make_evidence(row: dict[str, str], *, row_index_1based: int, unique_suffix: bool) -> EvidenceRow | None:
    cm_id = _norm_text(row.get("CMF ID", "")) or ""
    if not cm_id:
        return None

    evidence_id = f"{cm_id}::{row_index_1based}" if unique_suffix else cm_id

    countermeasure = _norm_text(row.get("Countermeasure", "")) or ""
    if not countermeasure:
        return None

    # Conditions: keep close to xlsx column names for traceability.
    conditions = {
        "crash_type": _norm_text(row.get("Crash Type", "")),
        "severity_kabco": _norm_text(row.get("KABCO Crash Severity", "")),
        "roadway_type": _norm_text(row.get("Roadway Type", "")),
        "area_type": _norm_text(row.get("Area Type", "")),
        "intersection_related": _norm_text(row.get("Intersection Related", "")),
        "traffic_control_type": _norm_text(row.get("Traffic Control Type", "")),
        "intersection_type": _norm_text(row.get("Intersection Type", "")),
        "intersection_geometry": _norm_text(row.get("Intersection Geometry", "")),
        "min_speed_limit": _to_float(row.get("Min Speed Limit", "")),
        "max_speed_limit": _to_float(row.get("Max Speed Limit", "")),
        "speed_unit": _norm_text(row.get("Speed Unit", "")),
        "min_num_lanes": _to_float(row.get("Min Num Lanes", "")),
        "max_num_lanes": _to_float(row.get("Max Num Lanes", "")),
        "num_lanes_direction": _norm_text(row.get("Num Lanes Direction", "")),
        "num_lanes_comment": _norm_text(row.get("Num Lanes Comment", "")),
        "traffic_volume_unit": _norm_text(row.get("Traffic Volume Unit", "")),
        "min_traffic_volume_non_intersection": _to_float(
            row.get("Minimum Traffic Volume (non-intersection)", "")
        ),
        "max_traffic_volume_non_intersection": _to_float(
            row.get("Maximum Traffic Volume (non-intersection)", "")
        ),
        "avg_traffic_volume_non_intersection": _to_float(
            row.get("Average Traffic Volume (non-intersection)", "")
        ),
        "min_major_road_traffic_volume": _to_float(
            row.get("Minimum Major Road Traffic Volume (intersection)", "")
        ),
        "max_major_road_traffic_volume": _to_float(
            row.get("Maximum Major Road Traffic Volume (intersection)", "")
        ),
        "avg_major_road_traffic_volume": _to_float(
            row.get("Average Major Road Traffic Volume (intersection)", "")
        ),
        "min_minor_road_traffic_volume": _to_float(
            row.get("Minimum Minor Road Traffic Volume (intersection)", "")
        ),
        "max_minor_road_traffic_volume": _to_float(
            row.get("Maximum Minor Road Traffic Volume (intersection)", "")
        ),
        "avg_minor_road_traffic_volume": _to_float(
            row.get("Average Minor Road Traffic Volume (intersection)", "")
        ),
        "roadway_division_type": _norm_text(row.get("Roadway Division Type", "")),
        "street_type": _norm_text(row.get("Street Type", "")),
        "crash_time_of_day": _norm_text(row.get("Crash Time of Day", "")),
        "crash_weather": _norm_text(row.get("Crash Weather", "")),
    }

    cmf = _to_float(row.get("CMF", ""))
    crf = _to_float(row.get("CRF", ""))
    effect_percent = (1.0 - cmf) * 100.0 if cmf is not None else None
    if effect_percent is not None:
        # Keep bounded for training stability.
        effect_percent = max(-100.0, min(100.0, effect_percent))

    effect = {
        "cmf": cmf,
        "crf": crf,
        "effect_percent": effect_percent,
    }

    quality = {
        "star_quality_rating": _to_float(row.get("Star Quality Rating", "")),
        "total_quality_points": _to_float(row.get("Total Quality Points", "")),
        "se_adjusted_cmf": _to_float(row.get("Adjusted Standard Error of CMF", "")),
        "se_unadjusted_cmf": _to_float(row.get("Unadjusted Standard Error of CMF", "")),
        "se_adjusted_crf": _to_float(row.get("Adjusted Standard Error of CRF", "")),
        "se_unadjusted_crf": _to_float(row.get("Unadjusted Standard Error of CRF", "")),
        "num_crashes": _as_int(row.get("Number of Crashes", "")),
        "num_crashes_before": _as_int(row.get("Number of Crashes Before", "")),
        "num_crashes_after": _as_int(row.get("Number of Crashes After", "")),
    }

    citation = {
        "study_title": _norm_text(row.get("Study Title", "")),
        "publication_year": _as_int(row.get("Publication Year", "")),
        "country": _norm_text(row.get("Country", "")),
        "state_province": _norm_text(row.get("State/Province", "")),
        "municipality": _norm_text(row.get("Municipality", "")),
        "methodology": _norm_text(row.get("Type of Study Methodology", "")),
    }

    evidence = EvidenceRow(
        evidence_id=evidence_id,
        cm_id=cm_id,
        countermeasure=countermeasure,
        countermeasure_category=_norm_text(row.get("Countermeasure Category", "")),
        countermeasure_subcategory=_norm_text(row.get("Countermeasure Subcategory", "")),
        conditions=conditions,
        effect=effect,
        quality=quality,
        citation=citation,
        source={"sheet": "Worksheet", "row_index_1based": row_index_1based},
        raw=None,
    )
    return evidence


def main() -> None:
    ap = argparse.ArgumentParser(description="Build evidence_store.jsonl from starratedresults.xlsx")
    ap.add_argument("--xlsx", required=True, help="Path to starratedresults .xlsx")
    ap.add_argument(
        "--out",
        default="data/evidence_store.jsonl",
        help="Output JSONL path (default: data/evidence_store.jsonl)",
    )
    ap.add_argument(
        "--unique-evidence-id",
        action="store_true",
        help="Make evidence_id unique per row using CMF ID + row index (cm_id stays CMF ID).",
    )
    ap.add_argument(
        "--include-raw",
        action="store_true",
        help="Include non-empty raw Excel columns in each evidence (larger output, better traceability).",
    )
    args = ap.parse_args()

    sheet = read_xlsx_first_sheet(args.xlsx)
    rows = sheet.rows

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cm_id_counts: dict[str, int] = {}
    written = 0

    with open(args.out, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows, start=2):  # header row is 1
            ev = _make_evidence(row, row_index_1based=i, unique_suffix=args.unique_evidence_id)
            if ev is None:
                continue
            if args.include_raw:
                raw = {k: (v or "").strip() for k, v in row.items() if (v or "").strip()}
                ev = EvidenceRow(**{**ev.__dict__, "raw": raw})
            cm_id_counts[ev.cm_id] = cm_id_counts.get(ev.cm_id, 0) + 1
            f.write(json.dumps(ev.__dict__, ensure_ascii=False) + "\n")
            written += 1

    duplicated_cm_ids = sum(1 for _, c in cm_id_counts.items() if c > 1)
    print(f"Input rows: {len(rows)}")
    print(f"Written evidences: {written}")
    print(f"Unique cm_id: {len(cm_id_counts)}")
    print(f"cm_id duplicated across rows: {duplicated_cm_ids}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
