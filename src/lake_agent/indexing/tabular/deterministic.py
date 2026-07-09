from __future__ import annotations

import csv
import hashlib
import posixpath
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import Iterable
from xml.etree import ElementTree as ET

from lake_agent.domain.indexing_models import (
    ColumnProfile,
    ScalarType,
    TableProfile,
    TabularIndexResult,
)

_SUPPORTED_FORMATS = {"csv", "tsv", "xlsx"}
_DEFAULT_DELIMITERS = {".csv": ",", ".tsv": "\t"}
_XML_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    "docrel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
_CONTEXT_SHEET_NAME_KEYWORDS = {
    "readme",
    "index",
    "contents",
    "overview",
    "description",
    "descriptions",
    "summary",
    "legend",
    "notes",
    "dictionary",
}
_SHEET_HEADER_TOKENS = {"sheet", "worksheet", "tab"}
_DESCRIPTION_HEADER_TOKENS = {"description", "summary", "content", "meaning", "notes"}


@dataclass(frozen=True, slots=True)
class TabularParseOptions:
    preview_row_limit: int = 20
    type_inference_row_limit: int = 200
    categorical_limit: int = 20
    sample_value_limit: int = 10
    sniff_bytes: int = 8192
    header_scan_row_limit: int = 10
    max_sheets: int = 50


class DeterministicTabularParser:
    def __init__(self, options: TabularParseOptions | None = None) -> None:
        self._options = options or TabularParseOptions()

    def parse_file(
        self,
        file_path: str | Path,
        *,
        relative_path: str | None = None,
        source_id: str | None = None,
    ) -> TabularIndexResult:
        path = Path(file_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(path)

        extension = path.suffix.lower().removeprefix(".")
        if extension not in _SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported deterministic tabular format: {extension}")

        normalized_relative_path = relative_path or path.name
        normalized_relative_path = PurePosixPath(
            normalized_relative_path.replace("\\", "/")
        ).as_posix()
        normalized_source_id = source_id or _stable_id(
            normalized_relative_path,
            prefix="source",
        )

        if extension == "csv":
            return self._parse_delimited(
                path,
                relative_path=normalized_relative_path,
                source_id=normalized_source_id,
                file_format="csv",
            )
        if extension == "tsv":
            return self._parse_delimited(
                path,
                relative_path=normalized_relative_path,
                source_id=normalized_source_id,
                file_format="tsv",
            )
        return self._parse_xlsx(
            path,
            relative_path=normalized_relative_path,
            source_id=normalized_source_id,
        )

    def _parse_delimited(
        self,
        path: Path,
        *,
        relative_path: str,
        source_id: str,
        file_format: str,
    ) -> TabularIndexResult:
        warnings: list[str] = []
        delimiter = self._detect_delimiter(path)
        rows = self._read_delimited_rows(path, delimiter)

        if not rows:
            table = self._empty_table(path.stem or path.name, source_id)
            return TabularIndexResult(
                source_id=source_id,
                relative_path=relative_path,
                filename=path.name,
                file_format=file_format,
                tables=[table],
            )

        table = self._build_table_profile(
            table_name=path.stem or path.name,
            rows=rows,
            source_id=source_id,
            sheet_name=None,
            warnings=warnings,
        )
        return TabularIndexResult(
            source_id=source_id,
            relative_path=relative_path,
            filename=path.name,
            file_format=file_format,
            tables=[table],
            parse_warnings=warnings.copy(),
        )

    def _parse_xlsx(
        self,
        path: Path,
        *,
        relative_path: str,
        source_id: str,
    ) -> TabularIndexResult:
        parse_warnings: list[str] = []
        workbook = _WorkbookArchive(path)
        sheet_rows = workbook.read_sheet_rows(self._options.max_sheets)
        tables: list[TableProfile] = []

        for sheet_name, rows in sheet_rows:
            if not rows:
                continue
            table = self._build_table_profile(
                table_name=sheet_name,
                rows=rows,
                source_id=source_id,
                sheet_name=sheet_name,
                warnings=[],
            )
            if table.row_count == 0 and not table.raw_header:
                continue
            tables.append(table)

        if not tables:
            table = self._empty_table(path.stem or path.name, source_id)
            parse_warnings.append("Workbook did not contain any non-empty sheets.")
            tables = [table]

        workbook_sheet_descriptions = _annotate_workbook_context(tables)

        return TabularIndexResult(
            source_id=source_id,
            relative_path=relative_path,
            filename=path.name,
            file_format="xlsx",
            tables=tables,
            parse_warnings=parse_warnings,
            workbook_sheet_descriptions=workbook_sheet_descriptions,
        )

    def _build_table_profile(
        self,
        *,
        table_name: str,
        rows: list[list[str]],
        source_id: str,
        sheet_name: str | None,
        warnings: list[str],
    ) -> TableProfile:
        max_width = max(len(row) for row in rows)
        normalized_rows = [_normalize_row(row, max_width) for row in rows]
        header_index = self._detect_header_index(normalized_rows)

        raw_header: list[str] = []
        context_before_header: list[list[str]] = []
        if header_index is not None:
            context_before_header = _context_before_header(normalized_rows, header_index)
            raw_header = normalized_rows[header_index]
            header = _normalize_headers(raw_header)
            raw_preview_rows = normalized_rows[header_index + 1 :]
            if header_index > 0:
                warnings.append(
                    f"Skipped {header_index} leading row(s) before the detected header."
                )
        else:
            header = [f"column_{index + 1}" for index in range(max_width)]
            raw_preview_rows = normalized_rows
            warnings.append(
                "A reliable header row was not detected; generated column names were used."
            )

        preview_rows = raw_preview_rows[: self._options.preview_row_limit]
        row_count = len(raw_preview_rows)
        profiled_rows = raw_preview_rows[: self._options.type_inference_row_limit]
        columns = self._build_column_profiles(header, profiled_rows)

        return TableProfile(
            table_id=_stable_id(f"{source_id}:{sheet_name or table_name}", prefix="table"),
            table_name=table_name,
            sheet_name=sheet_name,
            header_row_index=header_index,
            context_before_header=context_before_header,
            raw_header=raw_header,
            row_count=row_count,
            column_count=len(header),
            columns=columns,
            preview_rows=preview_rows,
            warnings=warnings,
        )

    def _empty_table(self, table_name: str, source_id: str) -> TableProfile:
        return TableProfile(
            table_id=_stable_id(f"{source_id}:{table_name}", prefix="table"),
            table_name=table_name,
            header_row_index=0,
            row_count=0,
            column_count=0,
            warnings=["The file did not contain any readable rows."],
        )

    def _detect_delimiter(self, path: Path) -> str:
        sample = path.read_text(encoding="utf-8-sig", errors="replace")[
            : self._options.sniff_bytes
        ]
        fallback_delimiter = _DEFAULT_DELIMITERS.get(path.suffix.lower(), ",")
        if not sample.strip():
            return fallback_delimiter

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            return dialect.delimiter
        except csv.Error:
            return fallback_delimiter

    def _read_delimited_rows(self, path: Path, delimiter: str) -> list[list[str]]:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            return [list(row) for row in reader if any(cell.strip() for cell in row)]

    def _build_column_profiles(
        self,
        header: list[str],
        rows: list[list[str]],
    ) -> list[ColumnProfile]:
        profiles: list[ColumnProfile] = []
        total_rows = len(rows)

        for ordinal, column_name in enumerate(header):
            values = [row[ordinal].strip() for row in rows if ordinal < len(row)]
            non_empty_values = [value for value in values if value != ""]
            unique_non_empty = list(dict.fromkeys(non_empty_values))
            inferred_type = _infer_type(non_empty_values)
            nullable = total_rows == 0 or len(non_empty_values) < total_rows
            distinct_ratio = (
                len(set(non_empty_values)) / len(non_empty_values)
                if non_empty_values
                else None
            )
            sample_values = unique_non_empty[: self._options.sample_value_limit]
            categorical_values: list[str] = []
            if unique_non_empty and len(unique_non_empty) <= self._options.categorical_limit:
                categorical_values = unique_non_empty

            profiles.append(
                ColumnProfile(
                    name=column_name,
                    ordinal=ordinal,
                    inferred_type=inferred_type,
                    nullable=nullable,
                    distinct_ratio=distinct_ratio,
                    sample_values=sample_values,
                    categorical_values=categorical_values,
                )
            )
        return profiles

    def _detect_header_index(self, rows: list[list[str]]) -> int | None:
        scan_rows = rows[: self._options.header_scan_row_limit]
        for index, row in enumerate(scan_rows):
            non_empty = [cell.strip() for cell in row if cell.strip()]
            if len(non_empty) < 2:
                continue
            if _looks_like_title_row(row):
                continue
            if _looks_like_sparse_unit_row(row):
                continue

            next_row = _next_non_empty_row(scan_rows, start=index + 1)
            if _looks_like_data_row(non_empty) and next_row is not None:
                next_non_empty = [cell.strip() for cell in next_row if cell.strip()]
                if _looks_like_data_row(next_non_empty):
                    return None

            return index
        return None


class _WorkbookArchive:
    def __init__(self, path: Path) -> None:
        self._path = path

    def read_sheet_rows(self, max_sheets: int) -> list[tuple[str, list[list[str]]]]:
        with zipfile.ZipFile(self._path) as archive:
            workbook_root = _parse_xml(archive.read("xl/workbook.xml"))
            workbook_rels = _relationship_map(archive.read("xl/_rels/workbook.xml.rels"))
            shared_strings = _shared_strings(archive)
            sheet_rows: list[tuple[str, list[list[str]]]] = []

            for sheet in workbook_root.findall("main:sheets/main:sheet", _XML_NS)[
                :max_sheets
            ]:
                name = sheet.attrib.get("name", "Sheet")
                rel_id = sheet.attrib.get(
                    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
                )
                if not rel_id or rel_id not in workbook_rels:
                    continue
                target = workbook_rels[rel_id]
                sheet_xml = archive.read(_resolve_xlsx_part_path(target))
                rows = _sheet_rows(sheet_xml, shared_strings)
                if rows:
                    sheet_rows.append((name, rows))
            return sheet_rows


def _parse_xml(content: bytes) -> ET.Element:
    return ET.fromstring(content)


def _relationship_map(content: bytes) -> dict[str, str]:
    root = ET.fromstring(content)
    mapping: dict[str, str] = {}
    for rel in root.findall("rel:Relationship", _XML_NS):
        rel_id = rel.attrib.get("Id")
        target = rel.attrib.get("Target")
        if rel_id and target:
            mapping[rel_id] = target
    return mapping


def _resolve_xlsx_part_path(target: str) -> str:
    normalized_target = target.replace("\\", "/").strip()
    if not normalized_target:
        raise KeyError("Empty worksheet target path in XLSX relationships.")

    normalized_target = normalized_target.lstrip("/")
    if normalized_target.startswith("xl/"):
        return posixpath.normpath(normalized_target)
    return posixpath.normpath(f"xl/{normalized_target}")


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = _parse_xml(archive.read("xl/sharedStrings.xml"))
    values: list[str] = []
    for si in root.findall("main:si", _XML_NS):
        text_parts = [node.text or "" for node in si.findall(".//main:t", _XML_NS)]
        values.append("".join(text_parts))
    return values


def _sheet_rows(sheet_xml: bytes, shared_strings: list[str]) -> list[list[str]]:
    root = _parse_xml(sheet_xml)
    sheet_data = root.find("main:sheetData", _XML_NS)
    if sheet_data is None:
        return []

    rows: list[list[str]] = []
    for row in sheet_data.findall("main:row", _XML_NS):
        cells: dict[int, str] = {}
        max_index = -1
        for cell in row.findall("main:c", _XML_NS):
            ref = cell.attrib.get("r", "")
            column_index = _column_index_from_ref(ref)
            max_index = max(max_index, column_index)
            cells[column_index] = _cell_value(cell, shared_strings)
        if max_index < 0:
            continue
        materialized = [cells.get(index, "").strip() for index in range(max_index + 1)]
        if any(materialized):
            rows.append(materialized)
    return rows


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    value_node = cell.find("main:v", _XML_NS)
    inline_node = cell.find("main:is/main:t", _XML_NS)

    if cell_type == "inlineStr" and inline_node is not None:
        return inline_node.text or ""
    if value_node is None or value_node.text is None:
        return ""
    raw = value_node.text
    if cell_type == "s":
        try:
            return shared_strings[int(raw)]
        except (ValueError, IndexError):
            return raw
    if cell_type == "b":
        return "true" if raw == "1" else "false"
    return raw


def _column_index_from_ref(reference: str) -> int:
    letters = "".join(char for char in reference if char.isalpha())
    index = 0
    for char in letters:
        index = index * 26 + (ord(char.upper()) - ord("A") + 1)
    return max(index - 1, 0)


def _normalize_row(row: list[str], width: int) -> list[str]:
    normalized = [cell.strip() for cell in row]
    if len(normalized) < width:
        normalized.extend([""] * (width - len(normalized)))
    return normalized


def _context_before_header(rows: list[list[str]], header_index: int) -> list[list[str]]:
    if header_index <= 0:
        return []

    context_rows: list[list[str]] = []
    for row in rows[:header_index]:
        if any(cell.strip() for cell in row):
            context_rows.append(row)
    return context_rows[-3:]


def _annotate_workbook_context(
    tables: list[TableProfile],
) -> dict[str, str]:
    if not tables:
        return {}

    sheet_name_map = {
        _normalize_identifier(table.sheet_name or table.table_name): table
        for table in tables
        if table.sheet_name or table.table_name
    }
    workbook_sheet_descriptions: dict[str, str] = {}
    for table in tables:
        described_sheets = _extract_sheet_descriptions(table, sheet_name_map)
        score = _context_sheet_score(table, described_sheets)
        if score < 4:
            continue

        table.is_context_sheet = True
        for target_key, description in described_sheets.items():
            target = sheet_name_map.get(target_key)
            if target is None:
                continue
            target.sheet_description = description
            workbook_sheet_descriptions[target.sheet_name or target.table_name] = description

    return workbook_sheet_descriptions


def _context_sheet_score(
    table: TableProfile,
    described_sheets: dict[str, str],
) -> int:
    score = 0
    sheet_name = _normalize_identifier(table.sheet_name or table.table_name)
    if sheet_name in _CONTEXT_SHEET_NAME_KEYWORDS:
        score += 3

    header_tokens = {_normalize_identifier(value) for value in table.raw_header if value.strip()}
    if header_tokens & _SHEET_HEADER_TOKENS:
        score += 2
    if header_tokens & _DESCRIPTION_HEADER_TOKENS:
        score += 2
    if table.column_count <= 3:
        score += 1
    if len(described_sheets) >= 2:
        score += 4
    elif len(described_sheets) == 1:
        score += 2
    return score


def _extract_sheet_descriptions(
    table: TableProfile,
    sheet_name_map: dict[str, TableProfile],
) -> dict[str, str]:
    descriptions: dict[str, str] = {}
    for row in table.preview_rows:
        if len(row) < 2:
            continue
        candidate_name = _normalize_identifier(row[0])
        description = row[1].strip()
        if not candidate_name or not description:
            continue
        if candidate_name in sheet_name_map:
            descriptions[candidate_name] = description
    return descriptions


def _normalize_headers(raw_headers: Iterable[str]) -> list[str]:
    headers: list[str] = []
    seen: dict[str, int] = {}
    for index, raw_header in enumerate(raw_headers, start=1):
        base_name = raw_header.strip() or f"column_{index}"
        candidate = base_name
        if candidate in seen:
            seen[candidate] += 1
            candidate = f"{candidate}_{seen[candidate]}"
        else:
            seen[candidate] = 1
        headers.append(candidate)
    return headers


def _looks_like_title(first_cell: str, non_empty_cells: list[str]) -> bool:
    if len(non_empty_cells) != 1:
        return False
    if not first_cell:
        return False
    letters = [char for char in first_cell if char.isalpha()]
    return bool(letters) and first_cell.upper() == first_cell


def _looks_like_title_row(row: list[str]) -> bool:
    non_empty = [cell.strip() for cell in row if cell.strip()]
    if not non_empty:
        return False
    return _looks_like_title(non_empty[0], non_empty)


def _next_non_empty_row(rows: list[list[str]], *, start: int) -> list[str] | None:
    for row in rows[start:]:
        if any(cell.strip() for cell in row):
            return row
    return None


def _looks_like_data_row(non_empty: list[str]) -> bool:
    if len(non_empty) < 2:
        return False
    data_like_count = sum(_looks_like_data_value(cell) for cell in non_empty)
    return data_like_count >= max(2, len(non_empty) - 1)


def _looks_like_sparse_unit_row(row: list[str]) -> bool:
    if not row:
        return False
    non_empty = [cell.strip() for cell in row if cell.strip()]
    if len(non_empty) < 2:
        return False
    if _empty_ratio(row) < 0.4:
        return False
    unit_like_count = sum(_looks_like_unit_token(cell) for cell in non_empty)
    return unit_like_count >= max(2, len(non_empty) - 1)


def _empty_ratio(row: list[str]) -> float:
    if not row:
        return 1.0
    empty_count = sum(1 for cell in row if not cell.strip())
    return empty_count / len(row)


def _looks_like_unit_token(value: str) -> bool:
    token = value.strip().lower()
    if not token:
        return False
    if token.startswith("(") and token.endswith(")"):
        return True
    return token in {
        "%",
        "ppm",
        "ppb",
        "mg/kg",
        "ug/g",
        "μg/g",
        "cm",
        "mm",
        "m",
        "ky",
        "ka",
        "ma",
        "wt%",
    }


def _looks_like_header_name(value: str) -> bool:
    if not value:
        return False
    if _looks_like_data_value(value):
        return False
    words = value.replace("_", " ").replace("-", " ").split()
    if not words:
        return False
    return len(words) <= 4


def _looks_like_data_value(value: str) -> bool:
    return any(
        checker(value)
        for checker in (_is_boolean, _is_integer, _is_float, _is_date, _is_datetime)
    )


def _infer_type(values: list[str]) -> ScalarType:
    if not values:
        return "unknown"
    if all(_is_boolean(value) for value in values):
        return "boolean"
    if all(_is_integer(value) for value in values):
        return "integer"
    if all(_is_float(value) for value in values):
        return "float"
    if all(_is_datetime(value) for value in values):
        return "datetime"
    if all(_is_date(value) for value in values):
        return "date"
    return "string"


def _is_boolean(value: str) -> bool:
    return value.lower() in {"true", "false", "yes", "no", "0", "1"}


def _is_integer(value: str) -> bool:
    try:
        int(value)
    except ValueError:
        return False
    return "." not in value


def _is_float(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _is_date(value: str) -> bool:
    return _parse_iso_date(value) is not None


def _is_datetime(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return "T" in value or " " in value


def _parse_iso_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _stable_id(value: str, *, prefix: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"
def _normalize_identifier(value: str) -> str:
    return "".join(char.lower() for char in value if char.isalnum())
