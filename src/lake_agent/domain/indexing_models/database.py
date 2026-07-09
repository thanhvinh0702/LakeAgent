from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from lake_agent.domain.indexing_models.tabular import ScalarType


DatabaseFormat = Literal["db", "sqlite"]


@dataclass(slots=True)
class DbColumnProfile:
    name: str
    ordinal: int
    inferred_type: ScalarType = "unknown"
    nullable: bool | None = None
    distinct_ratio: float | None = None
    sample_values: list[str] = field(default_factory=list)
    null_count: int | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DbTableProfile:
    table_id: str
    table_name: str
    row_count: int | None = None
    column_count: int = 0
    columns: list[DbColumnProfile] = field(default_factory=list)
    preview_rows: list[list[str]] = field(default_factory=list)
    summary: str | None = None
    keywords: list[str] = field(default_factory=list)
    table_search_text: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DatabaseIndexResult:
    source_id: str
    relative_path: str
    filename: str
    file_format: DatabaseFormat
    tables: list[DbTableProfile] = field(default_factory=list)
    parser_version: str = "v1"
    parse_warnings: list[str] = field(default_factory=list)
    file_summary: str | None = None
    file_keywords: list[str] = field(default_factory=list)
    file_search_text: str | None = None
