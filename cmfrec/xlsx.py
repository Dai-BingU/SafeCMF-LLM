from __future__ import annotations

import dataclasses
import zipfile
import xml.etree.ElementTree as ET
from collections.abc import Iterable


_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def _col_letters(cell_ref: str) -> str:
    letters: list[str] = []
    for ch in cell_ref:
        if ch.isalpha():
            letters.append(ch)
        else:
            break
    return "".join(letters)


def _parse_shared_strings(z: zipfile.ZipFile) -> list[str]:
    try:
        data = z.read("xl/sharedStrings.xml")
    except KeyError:
        return []

    root = ET.fromstring(data)
    strings: list[str] = []
    for si in root.findall("main:si", _NS):
        parts: list[str] = []

        direct = si.find("main:t", _NS)
        if direct is not None and direct.text:
            parts.append(direct.text)
        else:
            for run in si.findall("main:r", _NS):
                t = run.find("main:t", _NS)
                if t is not None and t.text:
                    parts.append(t.text)

        strings.append("".join(parts))

    return strings


def _workbook_sheets(z: zipfile.ZipFile) -> list[tuple[str, str]]:
    wb = ET.fromstring(z.read("xl/workbook.xml"))

    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    rid_to_target: dict[str, str] = {}
    for rel in rels.findall(
        "{http://schemas.openxmlformats.org/package/2006/relationships}Relationship"
    ):
        rid = rel.attrib.get("Id")
        target = rel.attrib.get("Target")
        if rid and target:
            rid_to_target[rid] = "xl/" + target.lstrip("/")

    sheets: list[tuple[str, str]] = []
    for sh in wb.findall("main:sheets/main:sheet", _NS):
        name = sh.attrib.get("name")
        rid = sh.attrib.get(f"{{{_NS['rel']}}}id")
        if name and rid and rid in rid_to_target:
            sheets.append((name, rid_to_target[rid]))

    return sheets


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")

    if cell_type == "inlineStr":
        is_el = cell.find("main:is", _NS)
        if is_el is None:
            return ""
        t = is_el.find("main:t", _NS)
        return t.text if t is not None and t.text is not None else ""

    v = cell.find("main:v", _NS)
    if v is None or v.text is None:
        return ""

    if cell_type == "s":
        try:
            return shared_strings[int(v.text)]
        except Exception:
            return ""

    return v.text


def _cols_sorted(cols: Iterable[str]) -> list[str]:
    return sorted(cols, key=lambda s: (len(s), s))


@dataclasses.dataclass(frozen=True)
class Sheet:
    name: str
    headers: list[str]
    rows: list[dict[str, str]]


def read_xlsx_first_sheet(path: str) -> Sheet:
    """
    Read the first worksheet in an .xlsx without external dependencies.

    Returns:
      Sheet(headers=[...], rows=[{header: value_str, ...}, ...])
    """
    with zipfile.ZipFile(path) as z:
        shared_strings = _parse_shared_strings(z)
        sheets = _workbook_sheets(z)
        if not sheets:
            raise ValueError("No worksheets found in xlsx")

        sheet_name, sheet_path = sheets[0]
        root = ET.fromstring(z.read(sheet_path))

        # Read row 1 as headers (first non-empty row).
        first_row_cells: dict[str, str] | None = None
        row_elements = root.findall("main:sheetData/main:row", _NS)
        for row in row_elements:
            cells: dict[str, str] = {}
            for cell in row.findall("main:c", _NS):
                ref = cell.attrib.get("r", "")
                if not ref:
                    continue
                col = _col_letters(ref)
                cells[col] = _cell_value(cell, shared_strings)
            if cells:
                first_row_cells = cells
                break

        if not first_row_cells:
            raise ValueError("Sheet appears empty (no header row found)")

        header_cols = _cols_sorted(first_row_cells.keys())
        headers = [first_row_cells.get(c, "").strip() for c in header_cols]
        col_to_header = {c: headers[i] for i, c in enumerate(header_cols)}

        rows: list[dict[str, str]] = []
        header_seen = False
        for row in row_elements:
            cells: dict[str, str] = {}
            for cell in row.findall("main:c", _NS):
                ref = cell.attrib.get("r", "")
                if not ref:
                    continue
                col = _col_letters(ref)
                val = _cell_value(cell, shared_strings)
                cells[col] = val

            if not cells:
                continue

            if not header_seen:
                # Skip the first non-empty row (header).
                header_seen = True
                continue

            row_dict: dict[str, str] = {}
            for col, header in col_to_header.items():
                if not header:
                    continue
                row_dict[header] = (cells.get(col, "") or "").strip()

            rows.append(row_dict)

        return Sheet(name=sheet_name, headers=headers, rows=rows)

