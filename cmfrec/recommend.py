from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .free_text import infer_query_from_text
from .scoring import Query, score_row
from .xlsx import read_xlsx_first_sheet


def _norm_countermeasure(value: str) -> str:
    return " ".join((value or "").strip().split())


def _load_query(args: argparse.Namespace) -> Query:
    if args.query_json:
        obj = json.loads(args.query_json)
        return Query(
            crash_type=obj.get("crash_type"),
            severity=obj.get("severity"),
            roadway_type=obj.get("roadway_type"),
            area_type=obj.get("area_type"),
            intersection_related=obj.get("intersection_related"),
            traffic_control_type=obj.get("traffic_control_type"),
            min_speed_limit=obj.get("min_speed_limit"),
            max_speed_limit=obj.get("max_speed_limit"),
            countermeasure_category=obj.get("countermeasure_category"),
            countermeasure_subcategory=obj.get("countermeasure_subcategory"),
            min_star=obj.get("min_star"),
        )

    return Query(
        crash_type=args.crash_type,
        severity=args.severity,
        roadway_type=args.roadway_type,
        area_type=args.area_type,
        intersection_related=args.intersection_related,
        traffic_control_type=args.traffic_control_type,
        min_speed_limit=args.min_speed_limit,
        max_speed_limit=args.max_speed_limit,
        num_lanes=None,
        traffic_volume_aadt=None,
        major_road_volume_aadt=None,
        minor_road_volume_aadt=None,
        countermeasure_category=args.countermeasure_category,
        countermeasure_subcategory=args.countermeasure_subcategory,
        min_star=args.min_star,
    )


def _list_field(rows: list[dict[str, str]], field: str, top: int) -> None:
    counts: dict[str, int] = {}
    for r in rows:
        v = (r.get(field, "") or "").strip() or "(blank)"
        counts[v] = counts.get(v, 0) + 1
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].lower()))
    for v, c in items[:top]:
        print(f"{c:5d}  {v}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="CMF baseline recommender (filter + rank + explain)."
    )
    ap.add_argument("--xlsx", required=True, help="Path to starratedresults .xlsx")
    ap.add_argument("--top-k", type=int, default=10, help="Number of recommendations")
    ap.add_argument(
        "--list-field",
        default=None,
        help="Print top values for a field (e.g. 'Crash Type') and exit",
    )
    ap.add_argument(
        "--list-top", type=int, default=30, help="Top N values for --list-field"
    )

    ap.add_argument("--query-json", default=None, help="Query as JSON string")
    ap.add_argument("--text", default=None, help="English free-text site context")
    ap.add_argument(
        "--show-inferred",
        action="store_true",
        help="When using --text, print inferred fields and notes",
    )
    ap.add_argument("--crash-type", default=None)
    ap.add_argument("--severity", default=None, help="KABCO Crash Severity")
    ap.add_argument("--roadway-type", default=None)
    ap.add_argument("--area-type", default=None)
    ap.add_argument("--intersection-related", default=None)
    ap.add_argument("--traffic-control-type", default=None)
    ap.add_argument("--min-speed-limit", type=float, default=None)
    ap.add_argument("--max-speed-limit", type=float, default=None)
    ap.add_argument("--countermeasure-category", default=None)
    ap.add_argument("--countermeasure-subcategory", default=None)
    ap.add_argument(
        "--min-star",
        type=float,
        default=None,
        help="Filter out evidences with Star Quality Rating < min-star",
    )
    ap.add_argument(
        "--explain",
        action="store_true",
        help="Print reasons for the top evidence behind each recommendation",
    )

    args = ap.parse_args()

    sheet = read_xlsx_first_sheet(args.xlsx)
    rows = sheet.rows

    if args.list_field:
        _list_field(rows, args.list_field, args.list_top)
        return

    query: Query
    if args.text:
        inferred = infer_query_from_text(args.text)
        query = Query(
            crash_type=inferred.query.crash_type,
            severity=inferred.query.severity,
            roadway_type=inferred.query.roadway_type,
            area_type=inferred.query.area_type,
            intersection_related=inferred.query.intersection_related,
            traffic_control_type=inferred.query.traffic_control_type,
            min_speed_limit=inferred.query.min_speed_limit,
            max_speed_limit=inferred.query.max_speed_limit,
            countermeasure_category=args.countermeasure_category,
            countermeasure_subcategory=args.countermeasure_subcategory,
            min_star=args.min_star,
        )
        if args.show_inferred:
            print("Inferred Query:", json.dumps(asdict(query), ensure_ascii=False))
            if inferred.notes:
                print("Inference notes:")
                for n in inferred.notes:
                    print(f"  - {n}")
            print()
    else:
        query = _load_query(args)

    by_cm: dict[str, list[tuple[float, dict[str, str], list[str]]]] = {}
    for row in rows:
        cm = _norm_countermeasure(row.get("Countermeasure", ""))
        if not cm:
            continue
        scored = score_row(row, query)
        if scored is None:
            continue
        by_cm.setdefault(cm, []).append((scored.total_score, row, scored.reasons))

    ranked: list[tuple[float, str, list[tuple[float, dict[str, str], list[str]]]]] = []
    for cm, evidences in by_cm.items():
        evidences_sorted = sorted(evidences, key=lambda t: t[0], reverse=True)
        agg = sum(s for s, _, _ in evidences_sorted[:3])
        ranked.append((agg, cm, evidences_sorted))

    ranked.sort(key=lambda t: t[0], reverse=True)

    print("Query:", json.dumps(asdict(query), ensure_ascii=False))
    print(f"Matched countermeasures: {len(ranked)}")
    print()

    for i, (agg, cm, evidences) in enumerate(ranked[: args.top_k], start=1):
        top_score, top_row, top_reasons = evidences[0]
        cmf_id = top_row.get("CMF ID", "")
        crash_type = top_row.get("Crash Type", "")
        severity = top_row.get("KABCO Crash Severity", "")
        star = top_row.get("Star Quality Rating", "")
        cmf = top_row.get("CMF", "")
        crf = top_row.get("CRF", "")
        study = top_row.get("Study Title", "")
        year = top_row.get("Publication Year", "")

        print(f"{i:02d}. {cm}")
        print(f"    score={agg:.3f}  top_evidence_score={top_score:.3f}")
        print(
            f"    CMF ID={cmf_id}  CMF={cmf or '?'}  CRF={crf or '?'}  Star={star or '?'}"
        )
        print(f"    Crash Type={crash_type or '?'}  Severity={severity or '?'}")
        print(f"    Study={study or '?'} ({year or '?'})")

        if args.explain:
            print("    reasons:")
            for r in top_reasons[:8]:
                print(f"      - {r}")
            if len(evidences) > 1:
                print(f"    other evidences matched: {len(evidences) - 1}")
        print()


if __name__ == "__main__":
    main()
