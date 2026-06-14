#!/usr/bin/env python3
"""Build a lightweight expert-review web app for recommendation evaluation."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmfrec.countermeasure_family import mechanism_family


def norm_text(text: str | None) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_image_index(mapping_path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    rows = load_jsonl(mapping_path)
    exact: dict[str, dict[str, Any]] = {}
    compact: list[dict[str, Any]] = []
    for row in rows:
        key = norm_text(row.get("countermeasure"))
        item = {
            "countermeasure": row.get("countermeasure"),
            "category": row.get("category"),
            "subcategory": row.get("subcategory"),
            "cm_uid": row.get("cm_uid"),
            "images": row.get("images", []),
            "norm": key,
        }
        if key and key not in exact:
            exact[key] = item
        compact.append(item)
    return exact, compact


def image_match(countermeasure: str | None, exact: dict[str, dict[str, Any]], all_items: list[dict[str, Any]]) -> dict[str, Any] | None:
    key = norm_text(countermeasure)
    if not key:
        return None
    if key in exact:
        item = exact[key]
        return {"kind": "exact", "score": 1.0, **item}

    best: tuple[float, dict[str, Any] | None] = (0.0, None)
    for item in all_items:
        score = SequenceMatcher(None, key, item["norm"]).ratio()
        if score > best[0]:
            best = (score, item)
    if best[1] is not None and best[0] >= 0.78:
        return {"kind": "fuzzy", "score": round(best[0], 3), **best[1]}
    return None


def compact_card(card: dict[str, Any], exact: dict[str, dict[str, Any]], all_items: list[dict[str, Any]], image_base: str) -> dict[str, Any]:
    match = image_match(card.get("countermeasure"), exact, all_items)
    images: list[dict[str, str]] = []
    if match:
        for img in match.get("images", [])[:3]:
            file = img.get("file")
            if not file:
                continue
            images.append(
                {
                    "src": f"{image_base}/{file}",
                    "caption": img.get("caption") or "",
                    "source": img.get("source") or "",
                    "license": img.get("license") or "",
                }
            )
    return {
        "evidence_id": str(card.get("evidence_id", "")),
        "countermeasure": card.get("countermeasure") or "",
        "crash_type": card.get("crash_type") or "",
        "area_type": card.get("area_type") or "",
        "facility_type": card.get("facility_type") or "",
        "traffic_control_type": card.get("traffic_control_type") or "",
        "cmf": card.get("cmf"),
        "crf": card.get("crf"),
        "star": card.get("star"),
        "retrieval_score": card.get("score"),
        "mechanism": card.get("mechanism") or "",
        "image_match": None if not match else {
            "kind": match.get("kind"),
            "score": match.get("score"),
            "matched_countermeasure": match.get("countermeasure"),
            "category": match.get("category"),
            "subcategory": match.get("subcategory"),
        },
        "images": images,
    }


def first_evidence_id(rec: dict[str, Any]) -> str:
    ids = rec.get("evidence_ids") or []
    return str(ids[0]) if ids else str(rec.get("cm_id") or "")


def evidence_to_eval_card(ev: dict[str, Any], eid: str) -> dict[str, Any]:
    conditions = ev.get("conditions") or {}
    effect = ev.get("effect") or {}
    quality = ev.get("quality") or {}
    return {
        "evidence_id": str(eid),
        "countermeasure": ev.get("countermeasure") or "",
        "crash_type": conditions.get("crash_type") or "",
        "area_type": conditions.get("area_type") or "",
        "facility_type": conditions.get("facility_type") or "",
        "traffic_control_type": conditions.get("traffic_control_type") or "",
        "cmf": effect.get("cmf") if effect.get("cmf") not in [None, ""] else effect.get("cmf_formula"),
        "crf": effect.get("crf"),
        "star": quality.get("star_quality_rating"),
        "mechanism": mechanism_family(ev),
    }


def load_reference_overrides(label_paths: list[Path], evidence: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    overrides: dict[str, dict[str, Any]] = {}
    for path in label_paths:
        if not path.exists():
            raise FileNotFoundError(f"reference label file not found: {path}")
        for row in load_jsonl(path):
            qid = str(row.get("qid") or "")
            if not qid:
                continue
            expected_ids: list[str] = []
            expected_cards: list[dict[str, Any]] = []
            supporting: dict[str, list[str]] = {}
            for rec in row.get("topk") or []:
                eid = first_evidence_id(rec)
                if not eid:
                    continue
                expected_ids.append(eid)
                expected_cards.append(evidence_to_eval_card(evidence.get(eid) or {"evidence_id": eid}, eid))
                support_ids = [str(x) for x in rec.get("supporting_evidence_ids") or [] if str(x)]
                if support_ids:
                    supporting[eid] = support_ids
            overrides[qid] = {
                "expected_ids": expected_ids,
                "expected_cards": expected_cards,
                "expected_mechanisms": [card.get("mechanism") for card in expected_cards],
                "supporting_evidence_ids": supporting,
            }
    return overrides


def build_cases(
    eval_path: Path,
    mapping_path: Path,
    image_base: str,
    reference_label_paths: list[Path] | None = None,
    evidence_store_path: Path | None = None,
) -> dict[str, Any]:
    eval_data = json.loads(eval_path.read_text())
    exact, all_items = build_image_index(mapping_path)
    reference_overrides: dict[str, dict[str, Any]] = {}
    if reference_label_paths:
        if evidence_store_path is None:
            raise ValueError("--evidence-store is required when --reference-labels is used")
        evidence = {str(row.get("evidence_id")): row for row in load_jsonl(evidence_store_path) if row.get("evidence_id")}
        reference_overrides = load_reference_overrides(reference_label_paths, evidence)
    cases: list[dict[str, Any]] = []
    overridden = 0
    for i, row in enumerate(eval_data["rows"], start=1):
        qid = str(row.get("qid") or "")
        override = reference_overrides.get(qid)
        if override:
            expected_source_cards = override["expected_cards"]
            expected_ids = override["expected_ids"]
            expected_mechanisms = override["expected_mechanisms"]
            supporting_evidence_ids = override.get("supporting_evidence_ids") or {}
            overridden += 1
        else:
            expected_source_cards = row.get("expected_cards", [])
            expected_ids = row.get("expected_ids", [])
            expected_mechanisms = row.get("expected_mechanisms", [])
            supporting_evidence_ids = {}
        adapter_cards = [compact_card(c, exact, all_items, image_base) for c in row.get("adapter_cards", [])]
        expected_cards = [compact_card(c, exact, all_items, image_base) for c in expected_source_cards]
        for card, mech in zip(adapter_cards, row.get("adapter_mechanisms", [])):
            card["mechanism"] = mech
        for card, mech in zip(expected_cards, expected_mechanisms):
            card["mechanism"] = mech
            card["supporting_evidence_ids"] = supporting_evidence_ids.get(str(card.get("evidence_id"))) or []
        cases.append(
            {
                "index": i,
                "qid": qid,
                "query": row.get("user_text"),
                "model": "Qwen3-32B fine-tuned",
                "candidate_pool": "Top-18",
                "adapter_ids": row.get("adapter_ids", []),
                "expected_ids": expected_ids,
                "adapter_cards": adapter_cards,
                "expected_cards": expected_cards,
                "bad_fit_flags": row.get("adapter_bad_fit_flags", []),
                "metrics": {
                    "exact_id_top1": row.get("exact_id_top1"),
                    "mechanism_top1": row.get("mechanism_top1"),
                    "broad_mechanism_top1": row.get("broad_mechanism_top1"),
                    "exact_id_jaccard": row.get("exact_id_jaccard"),
                    "mechanism_jaccard": row.get("mechanism_jaccard"),
                    "broad_mechanism_jaccard": row.get("broad_mechanism_jaccard"),
                },
            }
        )
    return {
        "meta": {
            "eval_file": str(eval_path.relative_to(ROOT) if eval_path.is_relative_to(ROOT) else eval_path),
            "image_mapping": str(mapping_path.relative_to(ROOT) if mapping_path.is_relative_to(ROOT) else mapping_path),
            "n_cases": len(cases),
            "model": "Qwen3-32B fine-tuned",
            "candidate_pool": "Top-18",
            "reference_override_files": [
                str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
                for path in (reference_label_paths or [])
            ],
            "reference_overridden_cases": overridden,
        },
        "cases": cases,
    }


APP_HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CMF Recommendation Expert Review</title>
  <style>
    :root {
      --bg: #f6f1e8;
      --panel: #fffaf0;
      --ink: #17202a;
      --muted: #65717e;
      --line: #d9cdb8;
      --blue: #1f5f8b;
      --green: #2e7d55;
      --orange: #b86f22;
      --red: #b13f3f;
      --shadow: 0 18px 48px rgba(52, 42, 25, 0.16);
      --radius: 18px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at 8% 10%, rgba(31, 95, 139, .14), transparent 28rem),
        radial-gradient(circle at 94% 12%, rgba(184, 111, 34, .15), transparent 26rem),
        linear-gradient(135deg, #f9f4ea 0%, #efe3d1 100%);
      font-family: "Avenir Next", "Helvetica Neue", Arial, sans-serif;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 10;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 16px;
      align-items: center;
      padding: 18px 28px;
      background: rgba(246, 241, 232, .88);
      backdrop-filter: blur(16px);
      border-bottom: 1px solid rgba(100, 82, 52, .18);
    }
    h1 { margin: 0; font-size: 24px; letter-spacing: -.02em; }
    .sub { color: var(--muted); font-size: 13px; margin-top: 4px; }
    .toolbar { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
    button, select, input[type="text"] {
      border: 1px solid var(--line);
      background: #fffaf0;
      color: var(--ink);
      border-radius: 999px;
      padding: 9px 13px;
      font: inherit;
    }
    button { cursor: pointer; transition: transform .12s ease, box-shadow .12s ease, background .12s ease; }
    button:hover { transform: translateY(-1px); box-shadow: 0 8px 18px rgba(52, 42, 25, .12); }
    button.primary { background: var(--blue); color: white; border-color: var(--blue); }
    button.green { background: var(--green); color: white; border-color: var(--green); }
    button.warn { background: #fff1db; color: #7b4a15; border-color: #e3ba76; }
    main {
      display: grid;
      grid-template-columns: minmax(260px, 340px) 1fr;
      gap: 20px;
      padding: 22px;
      max-width: 1680px;
      margin: 0 auto;
    }
    aside, .main-panel {
      background: rgba(255, 250, 240, .82);
      border: 1px solid rgba(114, 92, 58, .22);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }
    aside { padding: 16px; height: calc(100vh - 104px); position: sticky; top: 88px; overflow: hidden; display: flex; flex-direction: column; gap: 12px; }
    .progressbar { height: 12px; background: #eadcca; border-radius: 999px; overflow: hidden; }
    .progressbar > div { height: 100%; width: 0; background: linear-gradient(90deg, var(--green), #8bb96d); }
    .case-list { overflow: auto; display: grid; gap: 8px; padding-right: 4px; }
    .case-item {
      border: 1px solid var(--line);
      background: #fffdf7;
      border-radius: 13px;
      padding: 10px;
      cursor: pointer;
    }
    .case-item.active { border-color: var(--blue); box-shadow: 0 0 0 2px rgba(31,95,139,.15); }
    .case-item.reviewed { background: #eff8ef; }
    .case-item .qid { font-weight: 700; font-size: 13px; }
    .case-item .mini { color: var(--muted); font-size: 12px; margin-top: 3px; }
    .main-panel { padding: 22px; min-height: calc(100vh - 104px); }
    .case-head {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 16px;
      align-items: start;
      border-bottom: 1px solid var(--line);
      padding-bottom: 18px;
      margin-bottom: 18px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
      font-weight: 700;
      background: #edf4f8;
      color: #174762;
      border: 1px solid #c8dce8;
      margin: 0 6px 6px 0;
    }
    .query {
      font-family: Georgia, "Times New Roman", serif;
      font-size: 23px;
      line-height: 1.35;
      margin: 10px 0 0;
    }
    .cards { display: grid; gap: 16px; margin-top: 18px; }
    .card {
      display: grid;
      grid-template-columns: minmax(260px, 1fr) minmax(260px, 380px);
      gap: 16px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: #fffdf7;
      padding: 16px;
    }
    .card h3 { margin: 0 0 8px; font-size: 18px; }
    .rank {
      display: inline-grid;
      place-items: center;
      width: 32px; height: 32px;
      margin-right: 8px;
      border-radius: 50%;
      color: white;
      background: var(--blue);
      font-weight: 800;
    }
    .facts { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }
    .fact {
      border-radius: 10px;
      background: #f3eadb;
      padding: 7px 9px;
      font-size: 12px;
      color: #31404c;
    }
    .meta-line { color: var(--muted); font-size: 13px; margin: 5px 0; }
    .image-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 8px; }
    .image-grid img {
      width: 100%; height: 128px; object-fit: cover;
      border-radius: 12px; border: 1px solid #d7cab7; background: #eee;
      cursor: zoom-in;
    }
    .no-image {
      display: grid; place-items: center; min-height: 128px;
      border: 1px dashed #c8bca7; border-radius: 12px; color: var(--muted); font-size: 13px;
      background: #fbf5ea;
    }
    .review-box {
      margin-top: 22px;
      border-top: 1px solid var(--line);
      padding-top: 18px;
    }
    .score-grid { display: grid; grid-template-columns: repeat(4, minmax(170px, 1fr)); gap: 12px; }
    .score-card {
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fffdf7;
    }
    .score-card strong { display: block; margin-bottom: 8px; }
    .score-buttons { display: flex; gap: 6px; }
    .score-buttons button {
      width: 36px; height: 36px; padding: 0; border-radius: 50%;
      font-weight: 800;
    }
    .score-buttons button.selected { background: var(--green); color: white; border-color: var(--green); }
    .flags { display: flex; gap: 8px; flex-wrap: wrap; margin: 16px 0; }
    .flag { border-radius: 999px; border: 1px solid var(--line); padding: 8px 11px; background: #fffdf7; cursor: pointer; user-select: none; }
    .flag input { margin-right: 6px; }
    textarea {
      width: 100%;
      min-height: 96px;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      background: #fffdf7;
      font: inherit;
      resize: vertical;
    }
    details {
      margin-top: 16px;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      background: #fffdf7;
    }
    summary { cursor: pointer; font-weight: 700; }
    .expected { margin-top: 12px; display: grid; gap: 10px; }
    .expected-row { padding: 10px; border-radius: 12px; background: #f3eadb; }
    .modal {
      position: fixed; inset: 0; display: none; place-items: center; z-index: 99;
      background: rgba(12, 18, 24, .72);
      padding: 24px;
    }
    .modal.open { display: grid; }
    .modal img { max-width: min(1200px, 94vw); max-height: 88vh; border-radius: 18px; box-shadow: var(--shadow); background: white; }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      aside { position: relative; top: 0; height: auto; max-height: 50vh; }
      .card { grid-template-columns: 1fr; }
      .score-grid { grid-template-columns: 1fr; }
      header { grid-template-columns: 1fr; }
      .toolbar { justify-content: flex-start; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>CMF Recommendation Expert Review</h1>
      <div class="sub" id="metaLine">Loading...</div>
    </div>
    <div class="toolbar">
      <input id="search" type="text" placeholder="Search QID / query" />
      <select id="filter">
        <option value="all">All cases</option>
        <option value="todo">Not reviewed</option>
        <option value="done">Reviewed</option>
        <option value="low">Overall ≤ 3</option>
      </select>
      <button id="prevBtn">Prev</button>
      <button id="nextBtn">Next</button>
      <input id="expertId" type="text" placeholder="Expert ID" />
      <button class="green" id="exportJson">Export JSON</button>
      <button class="green" id="exportCsv">Export CSV</button>
    </div>
  </header>
  <main>
    <aside>
      <div class="progressbar"><div id="progressFill"></div></div>
      <div class="sub" id="progressText">0 reviewed</div>
      <div class="case-list" id="caseList"></div>
    </aside>
    <section class="main-panel" id="casePanel"></section>
  </main>
  <div class="modal" id="modal"><img alt="countermeasure preview" id="modalImg" /></div>
  <script id="review-data" type="application/json">__DATA__</script>
  <script>
    const DATA = JSON.parse(document.getElementById('review-data').textContent);
    const STORE_KEY = 'cmf_expert_review_v1_' + DATA.meta.model.replace(/\W+/g, '_') + '_' + DATA.meta.n_cases;
    const EXPERT_KEY = STORE_KEY + '_expert_id';
    let expertId = localStorage.getItem(EXPERT_KEY) || '';
    let backendAvailable = false;
    let reviews = JSON.parse(localStorage.getItem(STORE_KEY) || '{}');
    let active = 0;
    const scoreKeys = ['context_fit', 'mechanism_fit', 'engineering_applicability', 'overall_usefulness'];
    const scoreLabels = {
      context_fit: 'Context fit',
      mechanism_fit: 'Mechanism fit',
      engineering_applicability: 'Engineering applicability',
      overall_usefulness: 'Overall usefulness'
    };
    const flagLabels = [
      ['facility_mismatch', 'Facility mismatch'],
      ['mechanism_mismatch', 'Mechanism mismatch'],
      ['traffic_control_mismatch', 'Traffic-control mismatch'],
      ['negative_or_unfavorable_effect', 'Unfavorable safety effect'],
      ['duplicate_recommendation', 'Duplicate / same family'],
      ['missing_key_countermeasure', 'Missing key mechanism'],
      ['ranking_issue', 'Ranking issue'],
      ['image_helpful', 'Image helpful']
    ];

    async function api(path, payload) {
      try {
        const res = await fetch(path, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        if (!res.ok) throw new Error(await res.text());
        backendAvailable = true;
        return await res.json();
      } catch (err) {
        backendAvailable = false;
        return null;
      }
    }
    function save() { localStorage.setItem(STORE_KEY, JSON.stringify(reviews)); renderList(); renderProgress(); }
    function getReview(qid) {
      return reviews[qid] || { scores: {}, flags: {}, notes: '', updated_at: null };
    }
    function setReview(qid, patch) {
      reviews[qid] = { ...getReview(qid), ...patch, updated_at: new Date().toISOString() };
      save();
      const c = DATA.cases.find(x => x.qid === qid);
      api('/api/review', { expert_id: expertId || 'anonymous', case: c, review: reviews[qid] });
    }
    function esc(s) { return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c])); }
    function fmt(v) { return v === null || v === undefined || v === '' ? 'NA' : esc(v); }
    function reviewed(qid) {
      const r = getReview(qid);
      return scoreKeys.every(k => r.scores && r.scores[k]);
    }
    function filteredCases() {
      const q = document.getElementById('search').value.toLowerCase().trim();
      const f = document.getElementById('filter').value;
      return DATA.cases.filter(c => {
        const r = getReview(c.qid);
        const textOk = !q || (c.qid + ' ' + c.query).toLowerCase().includes(q);
        const done = reviewed(c.qid);
        const low = Number(r.scores?.overall_usefulness || 99) <= 3;
        const filterOk = f === 'all' || (f === 'todo' && !done) || (f === 'done' && done) || (f === 'low' && low);
        return textOk && filterOk;
      });
    }
    function renderProgress() {
      const n = DATA.cases.length;
      const done = DATA.cases.filter(c => reviewed(c.qid)).length;
      document.getElementById('progressFill').style.width = (done / n * 100).toFixed(1) + '%';
      document.getElementById('progressText').textContent = `${done}/${n} reviewed (${(done/n*100).toFixed(1)}%)`;
    }
    function renderList() {
      const list = document.getElementById('caseList');
      const cases = filteredCases();
      list.innerHTML = cases.map(c => {
        const cls = ['case-item', c.index === DATA.cases[active].index ? 'active' : '', reviewed(c.qid) ? 'reviewed' : ''].join(' ');
        const r = getReview(c.qid);
        const overall = r.scores?.overall_usefulness ? `Overall ${r.scores.overall_usefulness}` : 'Not reviewed';
        return `<div class="${cls}" data-index="${c.index-1}">
          <div class="qid">#${c.index} · ${esc(c.qid)}</div>
          <div class="mini">${esc(overall)}</div>
        </div>`;
      }).join('');
      list.querySelectorAll('.case-item').forEach(el => el.addEventListener('click', () => {
        active = Number(el.dataset.index);
        render();
      }));
    }
    function cardHtml(card, rank) {
      const match = card.image_match;
      const imgs = card.images?.length
        ? `<div class="image-grid">${card.images.map(img => `<img src="${esc(img.src)}" alt="${esc(card.countermeasure)}" loading="lazy" />`).join('')}</div>`
        : `<div class="no-image">No matched image</div>`;
      const matchLine = match
        ? `${match.kind} image match · ${match.score} · ${esc(match.matched_countermeasure || '')}`
        : 'image match: none';
      return `<article class="card">
        <div>
          <h3><span class="rank">${rank}</span>#${esc(card.evidence_id)} · ${esc(card.countermeasure)}</h3>
          <div class="facts">
            <span class="fact">CMF ${fmt(card.cmf)}</span>
            <span class="fact">CRF ${fmt(card.crf)}</span>
            <span class="fact">Star ${fmt(card.star)}</span>
            <span class="fact">Mechanism ${esc(card.mechanism || 'NA')}</span>
          </div>
          <div class="meta-line">Crash type: ${fmt(card.crash_type)}</div>
          <div class="meta-line">Area / facility / control: ${fmt(card.area_type)} · ${fmt(card.facility_type)} · ${fmt(card.traffic_control_type)}</div>
          <div class="meta-line">Image: ${matchLine}</div>
        </div>
        <div>${imgs}</div>
      </article>`;
    }
    function expectedHtml(c) {
      return `<details>
        <summary>Show reference labels / expected evidence IDs</summary>
        <div class="expected">
          ${c.expected_cards.map((card, i) => {
            const support = card.supporting_evidence_ids && card.supporting_evidence_ids.length
              ? `<br><span class="meta-line">Supporting/equivalent evidence IDs: ${card.supporting_evidence_ids.map(esc).join(', ')}</span>`
              : '';
            return `<div class="expected-row"><strong>Ref ${i+1}: #${esc(card.evidence_id)}</strong> · ${esc(card.countermeasure)}<br><span class="meta-line">Mechanism: ${esc(card.mechanism || 'NA')} · CMF ${fmt(card.cmf)} · CRF ${fmt(card.crf)} · Star ${fmt(card.star)}</span>${support}</div>`;
          }).join('')}
        </div>
      </details>`;
    }
    function renderCase() {
      const c = DATA.cases[active];
      const r = getReview(c.qid);
      const scoreHtml = scoreKeys.map(key => {
        const selected = r.scores?.[key];
        return `<div class="score-card"><strong>${scoreLabels[key]}</strong><div class="score-buttons">
          ${[1,2,3,4,5].map(v => `<button data-score-key="${key}" data-score-val="${v}" class="${selected == v ? 'selected' : ''}">${v}</button>`).join('')}
        </div></div>`;
      }).join('');
      const flagsHtml = flagLabels.map(([key, label]) => `<label class="flag"><input type="checkbox" data-flag="${key}" ${r.flags?.[key] ? 'checked' : ''}>${label}</label>`).join('');
      return `<div class="case-head">
        <div>
          <span class="badge">Case #${c.index}</span>
          <span class="badge">${esc(c.qid)}</span>
          <span class="badge">${esc(c.model)}</span>
          <span class="badge">${esc(c.candidate_pool)}</span>
          <p class="query">${esc(c.query)}</p>
        </div>
        <div class="toolbar">
          <button class="warn" id="markDone">Mark reviewed</button>
        </div>
      </div>
      <h2>Model Top-K Recommendations</h2>
      <div class="cards">${c.adapter_cards.map((card, i) => cardHtml(card, i+1)).join('')}</div>
      ${expectedHtml(c)}
      <div class="review-box">
        <h2>Expert Review</h2>
        <div class="score-grid">${scoreHtml}</div>
        <div class="flags">${flagsHtml}</div>
        <textarea id="notes" placeholder="Short expert comments, ranking concerns, missing key countermeasures...">${esc(r.notes || '')}</textarea>
      </div>`;
    }
    function render() {
      document.getElementById('metaLine').textContent = `${DATA.meta.model} · ${DATA.meta.candidate_pool} · ${DATA.meta.n_cases} cases · image-assisted expert review${backendAvailable ? ' · autosave on' : ' · browser save only until backend is detected'}`;
      document.getElementById('casePanel').innerHTML = renderCase();
      bindCaseEvents();
      renderList();
      renderProgress();
    }
    function bindCaseEvents() {
      const c = DATA.cases[active];
      document.querySelectorAll('[data-score-key]').forEach(btn => {
        btn.addEventListener('click', () => {
          const r = getReview(c.qid);
          const scores = { ...(r.scores || {}) };
          scores[btn.dataset.scoreKey] = Number(btn.dataset.scoreVal);
          setReview(c.qid, { scores });
          render();
        });
      });
      document.querySelectorAll('[data-flag]').forEach(cb => {
        cb.addEventListener('change', () => {
          const r = getReview(c.qid);
          const flags = { ...(r.flags || {}) };
          flags[cb.dataset.flag] = cb.checked;
          setReview(c.qid, { flags });
        });
      });
      document.getElementById('notes').addEventListener('input', e => setReview(c.qid, { notes: e.target.value }));
      document.getElementById('markDone').addEventListener('click', () => {
        const r = getReview(c.qid);
        const scores = { ...(r.scores || {}) };
        for (const k of scoreKeys) if (!scores[k]) scores[k] = 5;
        setReview(c.qid, { scores });
        render();
      });
      document.querySelectorAll('.image-grid img').forEach(img => {
        img.addEventListener('click', () => {
          document.getElementById('modalImg').src = img.src;
          document.getElementById('modal').classList.add('open');
        });
      });
    }
    function move(delta) {
      active = Math.max(0, Math.min(DATA.cases.length - 1, active + delta));
      render();
    }
    function download(filename, content, type='application/json') {
      const blob = new Blob([content], { type });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = filename;
      a.click();
      URL.revokeObjectURL(a.href);
    }
    function exportPayload() {
      return DATA.cases.map(c => ({ qid: c.qid, index: c.index, query: c.query, model: c.model, candidate_pool: c.candidate_pool, adapter_ids: c.adapter_ids, expected_ids: c.expected_ids, review: getReview(c.qid) }));
    }
    function csvEscape(v) { return '"' + String(v ?? '').replaceAll('"', '""') + '"'; }
    document.getElementById('prevBtn').addEventListener('click', () => move(-1));
    document.getElementById('nextBtn').addEventListener('click', () => move(1));
    document.getElementById('search').addEventListener('input', renderList);
    document.getElementById('filter').addEventListener('change', renderList);
    document.getElementById('expertId').value = expertId;
    document.getElementById('expertId').addEventListener('input', e => {
      expertId = e.target.value.trim();
      localStorage.setItem(EXPERT_KEY, expertId);
    });
    document.getElementById('exportJson').addEventListener('click', async () => {
      const payload = { meta: DATA.meta, expert_id: expertId || 'anonymous', exported_at: new Date().toISOString(), reviews: exportPayload() };
      const res = await api('/api/export_json', payload);
      if (res && res.download_url) window.open(res.download_url, '_blank');
      else download('cmf_expert_reviews.json', JSON.stringify(payload, null, 2));
    });
    document.getElementById('exportCsv').addEventListener('click', async () => {
      const rows = [['index','qid','context_fit','mechanism_fit','engineering_applicability','overall_usefulness','flags','notes','adapter_ids','expected_ids']];
      for (const c of DATA.cases) {
        const r = getReview(c.qid);
        rows.push([c.index, c.qid, r.scores?.context_fit || '', r.scores?.mechanism_fit || '', r.scores?.engineering_applicability || '', r.scores?.overall_usefulness || '', Object.entries(r.flags || {}).filter(x => x[1]).map(x => x[0]).join(';'), r.notes || '', c.adapter_ids.join(';'), c.expected_ids.join(';')]);
      }
      const csv = rows.map(row => row.map(csvEscape).join(',')).join('\n');
      const res = await api('/api/export_csv', { expert_id: expertId || 'anonymous', csv });
      if (res && res.download_url) window.open(res.download_url, '_blank');
      else download('cmf_expert_reviews.csv', csv, 'text/csv');
    });
    fetch('/api/reviews').then(r => r.ok ? r.json() : null).then(data => {
      if (data && data.reviews) {
        backendAvailable = true;
        reviews = { ...reviews, ...data.reviews };
        localStorage.setItem(STORE_KEY, JSON.stringify(reviews));
        render();
      }
    }).catch(() => {});
    document.getElementById('modal').addEventListener('click', () => document.getElementById('modal').classList.remove('open'));
    window.addEventListener('keydown', e => {
      if (e.key === 'ArrowLeft') move(-1);
      if (e.key === 'ArrowRight') move(1);
      if (e.key === 'Escape') document.getElementById('modal').classList.remove('open');
    });
    render();
  </script>
</body>
</html>
"""


def build_app(args: argparse.Namespace) -> None:
    eval_path = (ROOT / args.eval).resolve()
    mapping_path = (ROOT / args.mapping).resolve()
    out_dir = (ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    reference_label_paths = [(ROOT / path).resolve() for path in args.reference_labels]
    evidence_store_path = (ROOT / args.evidence_store).resolve() if args.evidence_store else None
    data = build_cases(eval_path, mapping_path, args.image_base, reference_label_paths, evidence_store_path)
    # Keep JSON valid inside a <script type="application/json"> tag without
    # HTML-escaping quotes into entities that JSON.parse cannot read.
    embedded_json = (
        json.dumps(data, ensure_ascii=False)
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    html_text = APP_HTML_TEMPLATE.replace("__DATA__", embedded_json)
    (out_dir / "index.html").write_text(html_text)
    (out_dir / "review_app_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))
    shutil.copy2(mapping_path, out_dir / "image_mapping_used.jsonl")
    print(f"Wrote {out_dir / 'index.html'}")
    print(f"Cases: {data['meta']['n_cases']}")
    with_images = sum(1 for c in data["cases"] for card in c["adapter_cards"] if card["images"])
    total_cards = sum(len(c["adapter_cards"]) for c in data["cases"])
    print(f"Model recommendation cards with matched images: {with_images}/{total_cards}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default="out/llm_assisted_eval_full215/qwen3_32b_ft_top18_mechanism_eval_full215.json")
    parser.add_argument("--mapping", default="out/image_hunt_runs/hunt_release_imported_merged/countermeasure_images.mapping.jsonl")
    parser.add_argument("--out", default="out/expert_review_app_current")
    parser.add_argument("--image-base", default="../image_hunt_runs/hunt_release_imported_merged")
    parser.add_argument("--reference-labels", nargs="*", default=[])
    parser.add_argument("--evidence-store", default="data/evidence_store.facility_v4.driveway_formula_all.jsonl")
    build_app(parser.parse_args())


if __name__ == "__main__":
    main()
