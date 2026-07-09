from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


SqlScriptFormat = Literal["sql"]


@dataclass(slots=True)
class SqlScriptSection:
    section_id: str
    chunk_index: int
    heading: str | None = None
    content: str = ""
    char_count: int = 0
    search_text: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SqlScriptIndexResult:
    source_id: str
    relative_path: str
    filename: str
    file_format: SqlScriptFormat
    sections: list[SqlScriptSection] = field(default_factory=list)
    parser_version: str = "v1"
    parse_warnings: list[str] = field(default_factory=list)
    file_summary: str | None = None
    file_keywords: list[str] = field(default_factory=list)
    file_search_text: str | None = None
