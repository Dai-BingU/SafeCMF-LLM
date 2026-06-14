#!/usr/bin/env python3
"""Build a presentation-style retrieval/reranking demo webpage.

The page visualizes the current evidence-grounded recommendation pipeline:
query input -> candidate evidence retrieval -> candidate-constrained LLM rerank
-> traceable Top-K CMF recommendations.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


ROOT = Path(".")
EXPERT_APP_DATA = ROOT / "out/expert_review_app_current/review_app_data.json"
EVAL_DATA = ROOT / "out/llm_assisted_eval_full215/qwen3_32b_ft_top18_mechanism_eval_full215.json"
TEST_CANDIDATES = ROOT / "data/splits/v690_stratified_85_5_10/sft_candidate_rerank_top20_compact.test.jsonl"
EVIDENCE_STORE = ROOT / "data/evidence_store.facility_v3.driveway_formula.jsonl"
IMAGE_MAPPING = ROOT / "out/image_hunt_runs/hunt_release_imported_merged/countermeasure_images.mapping.jsonl"
OUT_DIR = ROOT / "out/mentor_retrieval_demo_current"
OUT_HTML = OUT_DIR / "index.html"


FEATURED_QIDS = [
    ("q_exp_bal_00040", "Rural curve, high operating speed"),
    ("q_para_lex_00971", "Railroad-highway grade crossing"),
    ("q_02344_1", "Signalized rear-end queue warning"),
    ("q_03444_1", "Cross-median freeway barrier"),
    ("q_03337_1", "Urban signalized pedestrian crashes"),
    ("q_00326_1", "Rural stop-controlled angle crashes"),
    ("q_para_lex_00266", "Drowsy driving on freeway segments"),
]


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def compact_conditions(conditions: dict[str, Any] | None) -> dict[str, Any]:
    if not conditions:
        return {}
    keys = [
        "crash_type",
        "severity_kabco",
        "area_type",
        "facility_type",
        "roadway_type",
        "intersection_related",
        "traffic_control_type",
        "intersection_geometry",
        "roadway_division_type",
        "min_speed_limit",
        "max_speed_limit",
        "speed_unit",
        "min_num_lanes",
        "max_num_lanes",
        "min_traffic_volume_non_intersection",
        "max_traffic_volume_non_intersection",
        "min_major_road_traffic_volume",
        "max_major_road_traffic_volume",
        "min_minor_road_traffic_volume",
        "max_minor_road_traffic_volume",
        "traffic_volume_unit",
    ]
    return {k: conditions.get(k) for k in keys if conditions.get(k) not in (None, "", [])}


def as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    except Exception:
        return None


def load_evidence_store() -> dict[str, dict[str, Any]]:
    store: dict[str, dict[str, Any]] = {}
    with EVIDENCE_STORE.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            eid = str(obj.get("evidence_id") or obj.get("cm_id"))
            store[eid] = obj
    return store


def load_image_mapping() -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not IMAGE_MAPPING.exists():
        return mapping
    with IMAGE_MAPPING.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            images = obj.get("images") or []
            if not images:
                continue
            countermeasure = obj.get("countermeasure", "")
            src = images[0].get("file")
            if src:
                mapping[normalize_text(countermeasure)] = "../image_hunt_runs/hunt_release_imported_merged/" + src
    return mapping


def first_image(card: dict[str, Any], image_map: dict[str, str]) -> str | None:
    for img in card.get("images") or []:
        src = img.get("src")
        if src:
            return src
    key = normalize_text(card.get("countermeasure", ""))
    return image_map.get(key)


def enrich_card(card: dict[str, Any], evidence: dict[str, dict[str, Any]], image_map: dict[str, str]) -> dict[str, Any]:
    eid = str(card.get("evidence_id") or card.get("cm_id") or "")
    ev = evidence.get(eid, {})
    effect = card.get("effect") or ev.get("effect") or {}
    quality = ev.get("quality") or {}
    conditions = card.get("conditions") or ev.get("conditions") or {}
    citation = ev.get("citation") or {}
    out = {
        "evidence_id": eid,
        "countermeasure": card.get("countermeasure") or ev.get("countermeasure") or "",
        "category": card.get("category") or ev.get("countermeasure_category") or "",
        "subcategory": card.get("subcategory") or ev.get("countermeasure_subcategory") or "",
        "cmf": card.get("cmf", effect.get("cmf")),
        "crf": card.get("crf", effect.get("crf")),
        "star": card.get("star", quality.get("star_quality_rating")),
        "retrieval_score": card.get("retrieval_score"),
        "conditions": compact_conditions(conditions),
        "mechanism": card.get("mechanism"),
        "citation": {
            "study_title": citation.get("study_title"),
            "publication_year": citation.get("publication_year"),
            "methodology": citation.get("methodology"),
            "state_province": citation.get("state_province"),
            "country": citation.get("country"),
        },
        "image": None,
    }
    out["image"] = first_image({**card, **out}, image_map)
    return out


def parse_test_candidate_pools() -> dict[str, dict[str, Any]]:
    pools: dict[str, dict[str, Any]] = {}
    with TEST_CANDIDATES.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt", "")
            if "INPUT_JSON:\n" not in prompt:
                continue
            input_json = json.loads(prompt.split("INPUT_JSON:\n", 1)[1])
            pools[row["id"]] = input_json
    return pools


def pct(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) / float(denominator) * 100, 1)


def build_data() -> dict[str, Any]:
    evidence = load_evidence_store()
    image_map = load_image_mapping()
    expert = json.loads(EXPERT_APP_DATA.read_text())
    eval_obj = json.loads(EVAL_DATA.read_text())
    eval_rows = {row["qid"]: row for row in eval_obj["rows"]}
    candidate_pools = parse_test_candidate_pools()

    rows = eval_obj["rows"]
    n = len(rows)
    mechanism_hit3 = sum(bool(set(r.get("adapter_mechanisms", [])) & set(r.get("expected_mechanisms", []))) for r in rows)
    broad_hit3 = sum(
        bool(set(r.get("adapter_broad_mechanisms", [])) & set(r.get("expected_broad_mechanisms", []))) for r in rows
    )
    exact_hit3 = sum(bool(set(r.get("adapter_ids", [])) & set(r.get("expected_ids", []))) for r in rows)
    metrics = {
        "n": n,
        "exact_top1": pct(eval_obj["metrics"]["exact_id_top1"], n),
        "mechanism_top1": pct(eval_obj["metrics"]["mechanism_top1"], n),
        "broad_top1": pct(eval_obj["metrics"]["broad_mechanism_top1"], n),
        "exact_hit3": pct(exact_hit3, n),
        "mechanism_hit3": pct(mechanism_hit3, n),
        "broad_hit3": pct(broad_hit3, n),
        "bad_fit_rows": eval_obj["metrics"]["rows_with_bad_fit_flags"],
        "exact_jaccard": round(eval_obj["metrics"]["exact_id_jaccard_avg"], 3),
        "mechanism_jaccard": round(eval_obj["metrics"]["mechanism_jaccard_avg"], 3),
        "broad_jaccard": round(eval_obj["metrics"]["broad_mechanism_jaccard_avg"], 3),
    }

    cases = []
    for case in expert["cases"]:
        qid = case["qid"]
        pool = candidate_pools.get(qid, {})
        eval_row = eval_rows.get(qid, {})
        adapter_ids = [str(x) for x in case.get("adapter_ids", [])]
        expected_ids = [str(x) for x in case.get("expected_ids", [])]
        scorer_ids = [str(x) for x in eval_row.get("scorer_ids", [])]

        candidates = []
        for rank, cand in enumerate(pool.get("candidates", []), start=1):
            enriched = enrich_card(cand, evidence, image_map)
            enriched["retrieval_rank"] = rank
            enriched["selected_by_model"] = enriched["evidence_id"] in adapter_ids
            enriched["selected_by_rule_scorer"] = enriched["evidence_id"] in scorer_ids[:3]
            enriched["in_reference"] = enriched["evidence_id"] in expected_ids
            candidates.append(enriched)

        adapter_cards = [enrich_card(c, evidence, image_map) for c in case.get("adapter_cards", [])]
        expected_cards = [enrich_card(c, evidence, image_map) for c in case.get("expected_cards", [])]
        for i, card in enumerate(adapter_cards, start=1):
            card["rank"] = i
        for i, card in enumerate(expected_cards, start=1):
            card["rank"] = i

        cases.append(
            {
                "qid": qid,
                "index": case.get("index"),
                "query": case.get("query") or pool.get("user_text", ""),
                "target_context": pool.get("target_context", {}),
                "candidate_pool_label": "Compact retrieved evidence cards",
                "model": case.get("model", "Qwen3-32B fine-tuned"),
                "adapter_ids": adapter_ids,
                "expected_ids": expected_ids,
                "scorer_ids": scorer_ids[:3],
                "candidates": candidates,
                "recommendations": adapter_cards,
                "references": expected_cards,
                "metrics": {
                    **case.get("metrics", {}),
                    "mechanism_hit3": bool(
                        set(eval_row.get("adapter_mechanisms", [])) & set(eval_row.get("expected_mechanisms", []))
                    ),
                    "broad_mechanism_hit3": bool(
                        set(eval_row.get("adapter_broad_mechanisms", []))
                        & set(eval_row.get("expected_broad_mechanisms", []))
                    ),
                },
            }
        )

    qid_to_index = {c["qid"]: i for i, c in enumerate(cases)}
    featured = [
        {"qid": qid, "label": label, "index": qid_to_index[qid]}
        for qid, label in FEATURED_QIDS
        if qid in qid_to_index
    ]

    return {
        "meta": {
            "title": "Evidence-grounded CMF countermeasure recommendation demo",
            "model": "Qwen3-32B fine-tuned reranker",
            "n_cases": len(cases),
            "evidence_cards": len(evidence),
            "supervised_samples": 2150,
            "candidate_pool": "compact retrieved evidence cards",
            "metrics": metrics,
            "featured": featured,
        },
        "cases": cases,
    }


def html_template(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CMF Evidence-grounded Recommendation Demo</title>
  <style>
    :root {{
      --ink: #13202f;
      --muted: #66758a;
      --line: #d7e0ea;
      --paper: #f7f3e9;
      --panel: rgba(255,255,255,.88);
      --navy: #0d3557;
      --teal: #0e7c7b;
      --green: #1f8a54;
      --gold: #c98316;
      --orange: #d95d22;
      --red: #b43d3d;
      --blue: #2d6cdf;
      --shadow: 0 22px 60px rgba(20, 38, 61, .15);
      --radius: 22px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Avenir Next", "Gill Sans", "Trebuchet MS", sans-serif;
      background:
        radial-gradient(circle at 12% 8%, rgba(36, 119, 153, .18), transparent 28%),
        radial-gradient(circle at 86% 4%, rgba(201, 131, 22, .18), transparent 25%),
        linear-gradient(135deg, #f7f3e9 0%, #edf5f7 54%, #f8efe0 100%);
      min-height: 100vh;
    }}
    .shell {{ max-width: 1560px; margin: 0 auto; padding: 28px; }}
    .panel, .stage, .rec-card, .candidate {{
      background: var(--panel);
      border: 1px solid rgba(13, 53, 87, .12);
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
      border-radius: var(--radius);
    }}
    .pipeline {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0,1fr));
      gap: 12px;
      margin: 18px 0 20px;
    }}
    .stage {{ padding: 15px; min-height: 104px; position: relative; overflow: hidden; }}
    .stage .num {{
      width: 32px; height: 32px; display: inline-grid; place-items: center;
      border-radius: 50%; color: white; background: var(--navy); font-weight: 900;
      margin-right: 8px;
    }}
    .stage h3 {{ margin: 0 0 8px; font-size: 16px; display: flex; align-items: center; }}
    .stage p {{ margin: 0; color: var(--muted); font-size: 13px; line-height: 1.35; }}
    .stage.active {{ outline: 3px solid rgba(14,124,123,.25); transform: translateY(-2px); }}
    .stage.active .num {{ background: var(--orange); }}
    .stage.done {{ background: rgba(255,255,255,.72); }}
    .stage.done .num {{ background: var(--green); }}
    .demo-controls {{
      display: grid;
      grid-template-columns: minmax(420px, 1fr) auto;
      gap: 14px;
      align-items: center;
      margin: -2px 0 18px;
      padding: 14px 16px;
      border-radius: 18px;
      background: rgba(255,255,255,.72);
      border: 1px solid rgba(13,53,87,.12);
      box-shadow: 0 12px 30px rgba(20,38,61,.08);
    }}
    .step-title {{ font-weight: 900; color: var(--navy); margin-bottom: 3px; }}
    .step-narrative {{ color: var(--muted); font-size: 13px; line-height: 1.4; }}
    .demo-actions {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }}
    .step-jump {{ display: flex; gap: 6px; flex-wrap: wrap; margin-top: 9px; }}
    .step-link {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 6px 9px;
      border-radius: 999px;
      background: #eef4ff;
      color: #2855a8;
      border: 1px solid rgba(45,108,223,.22);
      font-size: 11px;
      font-weight: 900;
      text-decoration: none;
    }}
    .step-link.active {{ background: #fff2d8; color: #8a5410; border-color: rgba(201,131,22,.35); }}
    .workspace {{
      display: grid;
      grid-template-columns: 360px minmax(480px, 1fr) 430px;
      gap: 18px;
      align-items: start;
    }}
    .panel {{ padding: 18px; }}
    .panel h2 {{ margin: 0 0 12px; font-size: 18px; }}
    .panel .hint {{ color: var(--muted); font-size: 13px; line-height: 1.45; }}
    .controls {{ display: grid; gap: 12px; }}
    select, textarea, input {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(255,255,255,.82);
      color: var(--ink);
      padding: 12px 13px;
      font: inherit;
      outline: none;
    }}
    textarea {{ min-height: 154px; resize: vertical; line-height: 1.45; }}
    button {{
      border: none;
      border-radius: 14px;
      padding: 12px 16px;
      background: linear-gradient(135deg, var(--navy), #174d78);
      color: white;
      font-weight: 900;
      letter-spacing: .02em;
      cursor: pointer;
      box-shadow: 0 12px 24px rgba(13, 53, 87, .22);
    }}
    button.secondary {{ background: #fff; color: var(--navy); border: 1px solid var(--line); box-shadow: none; }}
    .panel {{ transition: opacity .35s ease, transform .35s ease, filter .35s ease, outline-color .35s ease; }}
    .panel.focus-panel {{ outline: 4px solid rgba(14,124,123,.28); transform: translateY(-2px); }}
    .panel.dim-panel {{ opacity: .48; filter: saturate(.72); }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }}
    .chip {{
      display: inline-flex; gap: 6px; align-items: center;
      border-radius: 999px; padding: 7px 10px;
      background: #eef6f4; color: #0d5d5c;
      border: 1px solid rgba(14,124,123,.18);
      font-size: 12px; font-weight: 800;
    }}
    .chip.gold {{ background: #fff6df; color: #86530a; border-color: rgba(201,131,22,.25); }}
    .chip.blue {{ background: #eef4ff; color: #2855a8; border-color: rgba(45,108,223,.22); }}
    .candidate-toolbar {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; margin-bottom: 12px; }}
    .candidate-list {{ display: grid; gap: 10px; max-height: 760px; overflow: auto; padding-right: 4px; }}
    .candidate {{
      display: grid;
      grid-template-columns: 58px 1fr auto;
      gap: 12px;
      padding: 13px;
      box-shadow: none;
      border-radius: 18px;
      position: relative;
      overflow: hidden;
    }}
    .candidate.selected {{ border-color: rgba(31,138,84,.55); background: linear-gradient(90deg, rgba(31,138,84,.12), rgba(255,255,255,.85)); }}
    .candidate {{ transition: opacity .35s ease, transform .35s ease, filter .35s ease, border-color .35s ease; }}
    .candidate.reference {{ box-shadow: inset 3px 0 0 var(--gold); }}
    .rank-badge {{
      width: 42px; height: 42px; display: grid; place-items: center;
      border-radius: 14px; background: #e8eef4; font-weight: 900; color: var(--navy);
    }}
    .candidate.selected .rank-badge {{ background: var(--green); color: white; }}
    .candidate h4 {{ margin: 0 0 5px; font-size: 14px; line-height: 1.25; }}
    .candidate p {{ margin: 0; color: var(--muted); font-size: 12px; line-height: 1.35; }}
    .scorebar {{ margin-top: 8px; height: 6px; background: #e6ecf2; border-radius: 999px; overflow: hidden; }}
    .scorebar span {{ display: block; height: 100%; background: linear-gradient(90deg, var(--teal), var(--gold)); }}
    .mini-tags {{ display: flex; flex-wrap: wrap; gap: 5px; margin-top: 8px; }}
    .tag {{ font-size: 11px; padding: 4px 7px; border-radius: 999px; background: #f0f3f6; color: #506174; font-weight: 800; }}
    .tag.pick {{ background: #daf3e6; color: #196b42; }}
    .tag.rule {{ background: #fff2d8; color: #8a5410; }}
    .tag.ref {{ background: #e7efff; color: #2855a8; }}
    .rec-list {{ display: grid; gap: 14px; }}
    .rec-card {{ padding: 15px; box-shadow: none; border-radius: 18px; overflow: hidden; }}
    .rec-card {{ transition: opacity .35s ease, transform .35s ease, filter .35s ease; }}
    .rec-top {{ display: grid; grid-template-columns: 86px 1fr; gap: 13px; }}
    .rec-img {{
      width: 86px; height: 78px; border-radius: 14px; object-fit: cover; background: linear-gradient(135deg, #e7edf2, #fff);
      border: 1px solid var(--line);
    }}
    .placeholder-img {{ display: grid; place-items: center; color: #8b99a8; font-weight: 900; font-size: 12px; }}
    .rec-card h3 {{ margin: 0; font-size: 16px; line-height: 1.25; }}
    .evidence-id {{ color: var(--green); font-weight: 900; }}
    .fact-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 8px; margin-top: 12px; }}
    .fact {{ border-radius: 12px; background: #f3f6f8; padding: 9px; }}
    .fact b {{ display:block; font-size: 15px; }}
    .fact span {{ color: var(--muted); font-size: 11px; font-weight: 800; text-transform: uppercase; }}
    details {{ margin-top: 10px; }}
    summary {{ cursor: pointer; color: var(--navy); font-weight: 900; }}
    .study {{ color: var(--muted); font-size: 12px; line-height: 1.4; margin-top: 8px; }}
    .case-metrics {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 8px; margin: 12px 0 14px; }}
    .case-metric {{ padding: 10px; border-radius: 13px; background: #f4f7fa; text-align: center; }}
    .case-metric.ok {{ color: var(--green); background: #eaf7ef; }}
    .case-metric.no {{ color: var(--red); background: #fff0ee; }}
    .case-metric b {{ display: block; font-size: 18px; }}
    .case-metric span {{ font-size: 10px; font-weight: 900; letter-spacing: .04em; text-transform: uppercase; }}
    .small-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    .small-table td {{ border-top: 1px solid var(--line); padding: 7px 0; vertical-align: top; }}
    .small-table td:first-child {{ color: var(--muted); width: 110px; font-weight: 800; }}
    .footer-note {{ margin: 20px 0 4px; color: var(--muted); font-size: 12px; text-align: center; }}
    @keyframes riseIn {{
      from {{ opacity: .08; transform: translateY(14px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes selectedPulse {{
      0%, 100% {{ box-shadow: inset 3px 0 0 var(--gold), 0 0 0 rgba(31,138,84,0); }}
      45% {{ box-shadow: inset 3px 0 0 var(--gold), 0 0 0 7px rgba(31,138,84,.18); }}
    }}
    body.step-1 #candidateList .candidate,
    body.step-1 #caseMetrics,
    body.step-1 #recommendations .rec-card,
    body.step-2 #caseMetrics,
    body.step-2 #recommendations .rec-card,
    body.step-3 #caseMetrics,
    body.step-3 #recommendations .rec-card,
    body.step-4 #caseMetrics,
    body.step-4 #recommendations .rec-card {{
      opacity: .08;
      filter: blur(2px);
      pointer-events: none;
    }}
    body.step-2 #candidateList .candidate {{
      opacity: .38;
      transform: translateY(4px);
    }}
    body.step-3 #candidateList .candidate {{
      animation: riseIn .42s both;
      animation-delay: calc(var(--i) * 30ms);
    }}
    body.step-4 #candidateList .candidate {{
      opacity: .22;
      filter: saturate(.65);
    }}
    body.step-4 #candidateList .candidate.selected {{
      opacity: 1;
      filter: none;
      transform: scale(1.012);
      animation: selectedPulse 1.2s ease-in-out infinite;
    }}
    body.step-5 #recommendations .rec-card {{
      animation: riseIn .45s both;
      animation-delay: calc(var(--i) * 120ms);
    }}
    @media (max-width: 1180px) {{
      .workspace {{ grid-template-columns: 1fr; }}
      .pipeline {{ grid-template-columns: 1fr 1fr; }}
      .demo-controls {{ grid-template-columns: 1fr; }}
      .demo-actions {{ justify-content: flex-start; }}
      .candidate-list {{ max-height: none; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="pipeline" id="pipeline">
      <div class="stage active"><h3><span class="num">1</span>Query input</h3><p>Crash type, facility, traffic control, speed, lanes, AADT.</p></div>
      <div class="stage"><h3><span class="num">2</span>Retrieval scoring</h3><p>Context fit × safety effect × evidence quality.</p></div>
      <div class="stage"><h3><span class="num">3</span>Evidence pool</h3><p>Top compact CMF cards with applicability constraints.</p></div>
      <div class="stage"><h3><span class="num">4</span>LLM reranker</h3><p>Candidate-constrained selection of evidence IDs.</p></div>
      <div class="stage"><h3><span class="num">5</span>Traceable Top-K</h3><p>Recommendations are populated from CMF evidence cards.</p></div>
    </section>
    <section class="demo-controls">
      <div>
        <div class="step-title" id="stepTitle">Step 1 · Query input</div>
        <div class="step-narrative" id="stepNarrative">The user enters a natural-language crash scenario. Key context fields are visible as chips.</div>
        <div class="step-jump">
          <a class="step-link" href="#step-1" data-step="1">1 Query</a>
          <a class="step-link" href="#step-2" data-step="2">2 Score</a>
          <a class="step-link" href="#step-3" data-step="3">3 Pool</a>
          <a class="step-link" href="#step-4" data-step="4">4 Rerank</a>
          <a class="step-link" href="#step-5" data-step="5">5 Output</a>
        </div>
      </div>
      <div class="demo-actions">
        <button class="secondary" id="nextBtn">Next step</button>
        <button class="secondary" id="resetBtn">Reset</button>
      </div>
    </section>

    <main class="workspace">
      <aside class="panel" id="scenarioPanel">
        <h2>Scenario input</h2>
        <p class="hint">Choose a case or search by keyword. Use Next step to reveal retrieval, candidate pool, reranking, and Top-K output.</p>
        <div class="controls">
          <select id="featured"></select>
          <input id="search" placeholder="Search cases, e.g. curve, pedestrian, signalized..." />
          <select id="caseSelect" size="9"></select>
          <textarea id="queryBox"></textarea>
          <button id="runBtn">Show next step</button>
          <button class="secondary" id="copyBtn">Copy query</button>
        </div>
        <div class="chips" id="contextChips"></div>
      </aside>

      <section class="panel" id="poolPanel">
        <div class="candidate-toolbar">
          <div>
            <h2 style="margin-bottom:4px;">Candidate evidence pool</h2>
            <div class="hint" id="poolHint">Context-aware retrieval candidates</div>
          </div>
          <div class="chip gold" id="caseIdChip">Case</div>
        </div>
        <div class="candidate-list" id="candidateList"></div>
      </section>

      <aside class="panel" id="outputPanel">
        <h2>Fine-tuned reranker output</h2>
        <div class="case-metrics" id="caseMetrics"></div>
        <div class="rec-list" id="recommendations"></div>
        <details>
          <summary>Show reference labels / expected evidence IDs</summary>
          <div class="rec-list" id="references" style="margin-top:12px;"></div>
        </details>
      </aside>
    </main>
    <div class="footer-note">Local static demo generated from current test candidate pools, Qwen3-32B fine-tuned outputs, CMF evidence store, and collected countermeasure images.</div>
  </div>

  <script id="demo-data" type="application/json">{payload}</script>
  <script>
    const DATA = JSON.parse(document.getElementById('demo-data').textContent);
    const state = {{ index: 0, filtered: DATA.cases, step: 1, timers: [] }};
    const $ = (id) => document.getElementById(id);
    const fmt = (v, fallback='NA') => (v === null || v === undefined || v === '' ? fallback : v);
    const pct = (v) => v ? '✓' : '—';
    const trunc = (s, n=96) => (s && s.length > n ? s.slice(0, n-1) + '…' : (s || ''));

    function conditionText(c) {{
      if (!c) return '';
      const parts = [];
      for (const [k,v] of Object.entries(c)) {{
        if (v !== null && v !== undefined && v !== '') parts.push(`${{k}}: ${{v}}`);
      }}
      return parts.slice(0, 6).join(' · ');
    }}

    function fillSelectors() {{
      $('featured').innerHTML = '<option value="">Featured cases</option>' + DATA.meta.featured.map(f => `<option value="${{f.index}}">${{f.label}}</option>`).join('');
      filterCases();
    }}

    function filterCases() {{
      const q = $('search').value.toLowerCase().trim();
      state.filtered = DATA.cases.filter(c => !q || c.qid.toLowerCase().includes(q) || c.query.toLowerCase().includes(q));
      $('caseSelect').innerHTML = state.filtered.map((c, i) => `<option value="${{i}}">${{c.qid}} — ${{trunc(c.query, 78)}}</option>`).join('');
      if (!state.filtered[state.index]) state.index = 0;
      $('caseSelect').selectedIndex = state.index;
    }}

    function currentCase() {{ return state.filtered[state.index] || DATA.cases[0]; }}

    function renderChips(c) {{
      const ctx = c.target_context || {{}};
      const chips = [
        ['Crash', ctx.crash_type],
        ['Facility', ctx.facility_type],
        ['Area', ctx.area_type],
        ['Control', ctx.traffic_control_type],
        ['Lanes', [ctx.min_num_lanes, ctx.max_num_lanes].filter(x => x!==undefined && x!==null).join('–')],
        ['AADT', [ctx.min_traffic_volume_non_intersection || ctx.min_major_road_traffic_volume, ctx.max_traffic_volume_non_intersection || ctx.max_major_road_traffic_volume].filter(x => x!==undefined && x!==null).join('–')]
      ].filter(x => x[1]);
      $('contextChips').innerHTML = chips.map(([k,v],i) => `<span class="chip ${{i%3===1?'gold':i%3===2?'blue':''}}">${{k}}: ${{v}}</span>`).join('');
    }}

    function renderCandidates(c) {{
      const candidates = c.candidates || [];
      const maxScore = Math.max(...candidates.map(x => Number(x.retrieval_score || 0)), 1);
      $('poolHint').textContent = `${{c.candidate_pool_label}} · ${{candidates.length}} cards · selected IDs highlighted in green`;
      $('caseIdChip').textContent = c.qid;
      $('candidateList').innerHTML = candidates.map((card, idx) => {{
        const selected = card.selected_by_model ? 'selected' : '';
        const ref = card.in_reference ? 'reference' : '';
        const tags = [
          card.selected_by_model ? '<span class="tag pick">LLM pick</span>' : '',
          card.selected_by_rule_scorer ? '<span class="tag rule">rule scorer</span>' : '',
          card.in_reference ? '<span class="tag ref">reference</span>' : '',
          card.category ? `<span class="tag">${{card.category}}</span>` : ''
        ].filter(Boolean).join('');
        const width = Math.max(4, Math.min(100, Number(card.retrieval_score || 0) / maxScore * 100));
        return `<article class="candidate ${{selected}} ${{ref}}" style="--i:${{idx}}">
          <div class="rank-badge">#${{card.retrieval_rank}}</div>
          <div>
            <h4><span class="evidence-id">${{card.evidence_id}}</span> · ${{card.countermeasure}}</h4>
            <p>${{conditionText(card.conditions)}}</p>
            <div class="mini-tags">${{tags}}</div>
            <div class="scorebar"><span style="width:${{width}}%"></span></div>
          </div>
          <div style="text-align:right; min-width:64px;">
            <b>${{fmt(card.retrieval_score)}}</b>
            <p>score</p>
          </div>
        </article>`;
      }}).join('') || '<p class="hint">No candidate pool found for this case.</p>';
    }}

    function recCard(card, reference=false, idx=0) {{
      const img = card.image ? `<img class="rec-img" src="${{card.image}}" alt="">` : `<div class="rec-img placeholder-img">CMF</div>`;
      const citation = card.citation || {{}};
      return `<article class="rec-card" style="--i:${{idx}}">
        <div class="rec-top">
          ${{img}}
          <div>
            <h3><span class="evidence-id">Rank ${{card.rank || ''}} · #${{card.evidence_id}}</span><br>${{card.countermeasure}}</h3>
            <div class="mini-tags">
              ${{card.mechanism ? `<span class="tag pick">${{card.mechanism}}</span>` : ''}}
              ${{card.category ? `<span class="tag">${{card.category}}</span>` : ''}}
            </div>
          </div>
        </div>
        <div class="fact-grid">
          <div class="fact"><b>${{fmt(card.cmf)}}</b><span>CMF</span></div>
          <div class="fact"><b>${{fmt(card.crf)}}</b><span>CRF</span></div>
          <div class="fact"><b>${{fmt(card.star)}}</b><span>Star</span></div>
        </div>
        <table class="small-table">
          <tr><td>Context</td><td>${{conditionText(card.conditions) || 'See evidence card'}}</td></tr>
          <tr><td>Study</td><td>${{fmt(citation.study_title)}} ${{citation.publication_year ? '('+citation.publication_year+')' : ''}}</td></tr>
        </table>
      </article>`;
    }}

    function renderOutput(c) {{
      const m = c.metrics || {{}};
      const items = [
        ['Exact Top-1', m.exact_id_top1],
        ['Mechanism Top-1', m.mechanism_top1],
        ['Broad Top-1', m.broad_mechanism_top1],
      ];
      $('caseMetrics').innerHTML = items.map(([label, ok]) => `<div class="case-metric ${{ok?'ok':'no'}}"><b>${{pct(ok)}}</b><span>${{label}}</span></div>`).join('');
      $('recommendations').innerHTML = (c.recommendations || []).map((x, i) => recCard(x, false, i)).join('') || '<p class="hint">No recommendations found.</p>';
      $('references').innerHTML = (c.references || []).map((x, i) => recCard(x, true, i)).join('') || '<p class="hint">No reference labels found.</p>';
    }}

    function renderCase() {{
      const c = currentCase();
      $('queryBox').value = c.query;
      renderChips(c);
      renderCandidates(c);
      renderOutput(c);
      applyStep(state.step || 1);
    }}

    const stepTexts = [
      ['Step 1 · Query input', 'The user enters a natural-language crash scenario. Key context fields are visible as chips.'],
      ['Step 2 · Retrieval scoring', 'The retrieval module compares the query context with CMF applicability conditions and assigns candidate scores.'],
      ['Step 3 · Candidate evidence pool', 'A compact evidence pool is constructed. Each card remains traceable to a CMF evidence record.'],
      ['Step 4 · Fine-tuned LLM reranking', 'The reranker selects evidence IDs from the candidate pool only. Selected candidates are highlighted in green.'],
      ['Step 5 · Traceable Top-K recommendation', 'The final IDs are expanded into countermeasure cards with CMF/CRF, evidence quality, context, citation, and images.']
    ];

    function clearTimers() {{
      state.timers.forEach(t => clearTimeout(t));
      state.timers = [];
    }}

    function applyStep(step) {{
      state.step = Math.max(1, Math.min(5, step));
      document.body.classList.remove('step-1', 'step-2', 'step-3', 'step-4', 'step-5');
      document.body.classList.add(`step-${{state.step}}`);
      const stages = Array.from(document.querySelectorAll('.stage'));
      stages.forEach((s, i) => {{
        s.classList.toggle('active', i + 1 === state.step);
        s.classList.toggle('done', i + 1 < state.step);
      }});
      const [title, narrative] = stepTexts[state.step - 1];
      $('stepTitle').textContent = title;
      $('stepNarrative').textContent = narrative;
      const focus = {{
        1: ['scenarioPanel'],
        2: ['poolPanel'],
        3: ['poolPanel'],
        4: ['poolPanel', 'outputPanel'],
        5: ['outputPanel']
      }}[state.step] || [];
      ['scenarioPanel', 'poolPanel', 'outputPanel'].forEach(id => {{
        const el = $(id);
        el.classList.toggle('focus-panel', focus.includes(id));
        el.classList.toggle('dim-panel', !focus.includes(id) && state.step !== 5);
      }});
      $('nextBtn').textContent = state.step >= 5 ? 'Restart steps' : 'Next step';
      document.querySelectorAll('.step-link').forEach(link => {{
        link.classList.toggle('active', Number(link.dataset.step) === state.step);
      }});
    }}

    function nextStep() {{
      clearTimers();
      applyStep(state.step >= 5 ? 1 : state.step + 1);
    }}

    function resetAnimation() {{
      clearTimers();
      applyStep(1);
    }}

    window.demoNextStep = nextStep;
    window.demoResetAnimation = resetAnimation;
    window.demoApplyStep = applyStep;
    globalThis.demoNextStep = nextStep;
    globalThis.demoResetAnimation = resetAnimation;
    globalThis.demoApplyStep = applyStep;

    $('search').addEventListener('input', () => {{ state.index = 0; filterCases(); renderCase(); }});
    $('caseSelect').addEventListener('change', e => {{ state.index = Number(e.target.value || 0); renderCase(); }});
    $('featured').addEventListener('change', e => {{
      const idx = e.target.value;
      if (idx !== '') {{
        const qid = DATA.cases[Number(idx)].qid;
        $('search').value = qid;
        filterCases();
        state.index = 0;
        $('caseSelect').selectedIndex = 0;
        renderCase();
      }}
    }});
    $('runBtn').addEventListener('click', nextStep);
    $('nextBtn').addEventListener('click', nextStep);
    $('resetBtn').addEventListener('click', resetAnimation);
    document.querySelectorAll('.step-link').forEach(link => {{
      link.addEventListener('click', e => {{
        e.preventDefault();
        clearTimers();
        applyStep(Number(link.dataset.step));
      }});
    }});
    $('copyBtn').addEventListener('click', async () => {{
      try {{ await navigator.clipboard.writeText($('queryBox').value); $('copyBtn').textContent='Copied'; setTimeout(()=>$('copyBtn').textContent='Copy query',900); }} catch(e) {{}}
    }});

    fillSelectors();
    const defaultIdx = DATA.meta.featured[0]?.index || 0;
    const defaultQid = DATA.cases[defaultIdx].qid;
    $('search').value = defaultQid;
    filterCases();
    state.index = 0;
    renderCase();
  </script>
</body>
</html>
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = build_data()
    OUT_HTML.write_text(html_template(data), encoding="utf-8")
    summary = {
        "output": str(OUT_HTML),
        "cases": len(data["cases"]),
        "featured_cases": data["meta"]["featured"],
        "metrics": data["meta"]["metrics"],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
