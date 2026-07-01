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

ColumnRole = Literal["dimension", "metric", "date", "id", "unknown"]

TableFormat = Literal["csv", "tsv", "xls", "xlsx"]


@dataclass(slots=True)
class ColumnProfile:
    # deterministic
    name: str
    ordinal: int
    inferred_type: ScalarType = "unknown"
    nullable: bool | None = None
    distinct_ratio: float | None = None
    sample_values: list[str] = field(default_factory=list)
    categorical_values: list[str] = field(default_factory=list)
    # llm enrich
    semantic_label: str | None = None
    description: str | None = None
    aliases: list[str] = field(default_factory=list)
    role: ColumnRole = "unknown"
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TableProfile:
    # deterministic
    table_id: str
    table_name: str
    sheet_name: str | None = None
    header_row_index: int | None = None
    raw_header: list[str] = field(default_factory=list)
    row_count: int | None = None
    column_count: int = 0
    columns: list[ColumnProfile] = field(default_factory=list)
    preview_rows: list[list[str]] = field(default_factory=list)
    # llm enrich
    summary: str | None = None
    business_purpose: str | None = None
    keywords: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TabularIndexResult:
    # deterministic
    source_id: str
    relative_path: str
    filename: str
    file_format: TableFormat
    tables: list[TableProfile] = field(default_factory=list)
    parser_version: str = "v1"
    parse_warnings: list[str] = field(default_factory=list)
    # llm enrich
    file_summary: str | None = None
    file_keywords: list[str] = field(default_factory=list)
    lexical_text: str | None = None
    semantic_text: str | None = None
