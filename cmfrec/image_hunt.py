"""
Offline countermeasure image collection utilities.

Goal:
  - Create a stable, merge-friendly catalog of countermeasures (deduped by text)
  - Split work into "packs" for multiple annotators
  - Merge pack submissions back into a single mapping
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


def _read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_countermeasure_text(text: str) -> str:
    if text is None:
        return ""
    t = text.strip().lower()
    t = t.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    t = re.sub(r"\s+", " ", t)
    return t


def make_countermeasure_uid(
    countermeasure: str,
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
) -> str:
    """
    Stable ID for merging across machines.
    Uses a normalized key with a short SHA1 digest.
    """
    key = "|".join(
        [
            normalize_countermeasure_text(category or ""),
            normalize_countermeasure_text(subcategory or ""),
            normalize_countermeasure_text(countermeasure or ""),
        ]
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"cm_{digest}"


@dataclass(frozen=True)
class CatalogItem:
    cm_uid: str
    countermeasure: str
    category: Optional[str]
    subcategory: Optional[str]
    example_evidence_ids: Tuple[str, ...]
    freq: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cm_uid": self.cm_uid,
            "countermeasure": self.countermeasure,
            "category": self.category,
            "subcategory": self.subcategory,
            "example_evidence_ids": list(self.example_evidence_ids),
            "freq": self.freq,
        }


def build_countermeasure_catalog(
    *,
    evidence_store_jsonl: str | Path,
    candidates_jsonl: str | Path | None = None,
    top_n: int | None = None,
    min_freq: int = 1,
    shuffle: bool = False,
    seed: int = 7,
    max_example_evidence_ids: int = 8,
) -> List[CatalogItem]:
    """
    Build a deduped catalog by countermeasure text (not evidence_id).

    Frequency:
      - If candidates_jsonl is provided, freq is counted from candidate pools
        (each occurrence of an evidence_id contributes +1 to its countermeasure).
      - Else, freq defaults to 1 per countermeasure.
    """
    # evidence_id -> (countermeasure, category, subcategory)
    evidence_map: Dict[str, Tuple[str, Optional[str], Optional[str]]] = {}
    # normalized countermeasure -> canonical tuple + evidence_ids
    cm_to_eids: Dict[str, List[str]] = defaultdict(list)
    cm_meta: Dict[str, Tuple[str, Optional[str], Optional[str]]] = {}

    for row in _read_jsonl(evidence_store_jsonl):
        evidence_id = str(row.get("evidence_id") or row.get("cm_id") or "").strip()
        countermeasure = (row.get("countermeasure") or "").strip()
        if not evidence_id or not countermeasure:
            continue
        cat = row.get("countermeasure_category")
        sub = row.get("countermeasure_subcategory")
        evidence_map[evidence_id] = (countermeasure, cat, sub)
        norm = normalize_countermeasure_text(countermeasure)
        cm_to_eids[norm].append(evidence_id)
        # Keep the first seen canonical capitalization & meta
        if norm not in cm_meta:
            cm_meta[norm] = (countermeasure, cat, sub)

    freqs: Counter[str] = Counter()
    if candidates_jsonl is not None:
        for q in _read_jsonl(candidates_jsonl):
            for eid in q.get("candidate_evidence_ids", []) or []:
                eid = str(eid)
                meta = evidence_map.get(eid)
                if not meta:
                    continue
                countermeasure, _cat, _sub = meta
                freqs[normalize_countermeasure_text(countermeasure)] += 1
    else:
        for norm in cm_meta.keys():
            freqs[norm] = 1

    items: List[CatalogItem] = []
    for norm, (countermeasure, cat, sub) in cm_meta.items():
        f = int(freqs.get(norm, 0))
        if f < min_freq:
            continue
        cm_uid = make_countermeasure_uid(countermeasure, category=cat, subcategory=sub)
        eids = tuple(cm_to_eids.get(norm, [])[:max_example_evidence_ids])
        items.append(
            CatalogItem(
                cm_uid=cm_uid,
                countermeasure=countermeasure,
                category=cat,
                subcategory=sub,
                example_evidence_ids=eids,
                freq=f,
            )
        )

    # Sort: highest frequency first by default (more "important" / more used)
    items.sort(key=lambda x: (-x.freq, x.category or "", x.subcategory or "", x.countermeasure))

    if top_n is not None:
        items = items[: int(top_n)]

    if shuffle:
        import random

        rng = random.Random(int(seed))
        rng.shuffle(items)

    return items


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

