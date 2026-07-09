from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


JsonFormat = Literal["json", "jsonl", "ndjson"]


@dataclass(slots=True)
class JsonSection:
    section_id: str
    chunk_index: int
    path_start: str | None = None
    path_end: str | None = None
    entry_count: int = 0
    content: str = ""
    char_count: int = 0
    search_text: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class JsonIndexResult:
    source_id: str
    relative_path: str
    filename: str
    file_format: JsonFormat
    sections: list[JsonSection] = field(default_factory=list)
    parser_version: str = "flattened_json_v1"
    parse_warnings: list[str] = field(default_factory=list)
    top_level_type: str | None = None
    entry_count: int = 0
    max_depth: int = 0
    file_summary: str | None = None
    file_keywords: list[str] = field(default_factory=list)
    file_search_text: str | None = None
