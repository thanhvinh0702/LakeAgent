from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


SlideshowFormat = Literal["ppt", "pptx"]


@dataclass(slots=True)
class SlideshowEmbeddedImage:
    image_id: str
    image_index: int
    path: str
    filename: str
    width: int
    height: int
    color_mode: str
    slide_start: int | None = None
    slide_end: int | None = None
    caption: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SlideshowSection:
    section_id: str
    section_type: str
    chunk_index: int
    heading: str | None = None
    content: str = ""
    slide_start: int | None = None
    slide_end: int | None = None
    char_count: int = 0
    search_text: str | None = None
    image_id: str | None = None
    image_index: int | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SlideshowIndexResult:
    source_id: str
    relative_path: str
    filename: str
    file_format: SlideshowFormat
    sections: list[SlideshowSection] = field(default_factory=list)
    parser_version: str = "docling_hierarchical_v1"
    parse_warnings: list[str] = field(default_factory=list)
    file_summary: str | None = None
    file_keywords: list[str] = field(default_factory=list)
    file_search_text: str | None = None
    embedded_images: list[SlideshowEmbeddedImage] = field(default_factory=list)
    artifact_dir: str | None = None
