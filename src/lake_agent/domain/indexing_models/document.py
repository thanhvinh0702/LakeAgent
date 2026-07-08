from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


DocumentFormat = Literal["pdf", "docx", "doc", "rtf"]


@dataclass(slots=True)
class DocumentEmbeddedImage:
    image_id: str
    image_index: int
    path: str
    filename: str
    width: int
    height: int
    color_mode: str
    page_start: int | None = None
    page_end: int | None = None
    caption: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DocumentSection:
    section_id: str
    section_type: str
    chunk_index: int
    heading: str | None = None
    content: str = ""
    page_start: int | None = None
    page_end: int | None = None
    char_count: int = 0
    search_text: str | None = None
    image_id: str | None = None
    image_index: int | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DocumentIndexResult:
    source_id: str
    relative_path: str
    filename: str
    file_format: DocumentFormat
    sections: list[DocumentSection] = field(default_factory=list)
    parser_version: str = "docling_hierarchical_v1"
    parse_warnings: list[str] = field(default_factory=list)
    file_summary: str | None = None
    file_keywords: list[str] = field(default_factory=list)
    file_search_text: str | None = None
    embedded_images: list[DocumentEmbeddedImage] = field(default_factory=list)
    artifact_dir: str | None = None
