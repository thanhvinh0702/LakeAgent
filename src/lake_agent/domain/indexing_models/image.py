from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


ImageFormat = Literal["jpeg", "png", "gif", "webp", "tiff"]


@dataclass(slots=True)
class ImageSection:
    section_id: str
    section_type: str
    chunk_index: int
    heading: str | None = None
    content: str = ""
    line_start: int | None = None
    line_end: int | None = None
    char_count: int = 0
    search_text: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ImageIndexResult:
    source_id: str
    relative_path: str
    filename: str
    file_format: ImageFormat
    width: int
    height: int
    color_mode: str
    has_alpha: bool
    is_animated: bool
    frame_count: int
    parser_version: str = "v1"
    parse_warnings: list[str] = field(default_factory=list)
    sections: list[ImageSection] = field(default_factory=list)
    file_summary: str | None = None
    file_keywords: list[str] = field(default_factory=list)
    file_search_text: str | None = None
