from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


ScalarType = Literal[
    "string",
    "integer",
    "float",
    "boolean",
    "date",
    "datetime",
    "unknown",
]

TableFormat = Literal["csv", "tsv", "xls", "xlsx"]


@dataclass(slots=True)
class ColumnProfile:
    name: str
    ordinal: int
    inferred_type: ScalarType = "unknown"
    nullable: bool | None = None
    distinct_ratio: float | None = None
    sample_values: list[str] = field(default_factory=list)
    categorical_values: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TableProfile:
    table_id: str
    table_name: str
    sheet_name: str | None = None
    is_context_sheet: bool = False
    sheet_description: str | None = None
    header_row_index: int | None = None
    context_before_header: list[list[str]] = field(default_factory=list)
    raw_header: list[str] = field(default_factory=list)
    row_count: int | None = None
    column_count: int = 0
    columns: list[ColumnProfile] = field(default_factory=list)
    preview_rows: list[list[str]] = field(default_factory=list)
    summary: str | None = None
    keywords: list[str] = field(default_factory=list)
    table_search_text: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TabularIndexResult:
    source_id: str
    relative_path: str
    filename: str
    file_format: TableFormat
    tables: list[TableProfile] = field(default_factory=list)
    parser_version: str = "v1"
    parse_warnings: list[str] = field(default_factory=list)
    workbook_sheet_descriptions: dict[str, str] = field(default_factory=dict)
    file_summary: str | None = None
    file_keywords: list[str] = field(default_factory=list)
    lexical_text: str | None = None
    semantic_text: str | None = None
