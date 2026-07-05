from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


TextFormat = Literal["txt", "md"]


@dataclass(slots=True)
class TextSection:
    section_id: str
    chunk_index: int
    heading: str | None = None
    content: str = ""
    line_start: int | None = None
    line_end: int | None = None
    char_count: int = 0
    search_text: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TextIndexResult:
    source_id: str
    relative_path: str
    filename: str
    file_format: TextFormat
    sections: list[TextSection] = field(default_factory=list)
    parser_version: str = "v1"
    parse_warnings: list[str] = field(default_factory=list)
    file_summary: str | None = None
    file_keywords: list[str] = field(default_factory=list)
    file_search_text: str | None = None
