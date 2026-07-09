from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


EpubFormat = Literal["epub"]


@dataclass(slots=True)
class EpubEmbeddedImage:
    image_id: str
    image_index: int
    href: str
    path: str
    filename: str
    media_type: str | None = None
    width: int | None = None
    height: int | None = None
    color_mode: str | None = None
    caption: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EpubSection:
    section_id: str
    section_type: str
    chunk_index: int
    content: str
    heading: str | None = None
    chapter_index: int | None = None
    chapter_title: str | None = None
    chapter_href: str | None = None
    image_id: str | None = None
    image_index: int | None = None
    image_href: str | None = None
    char_count: int = 0
    search_text: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EpubIndexResult:
    source_id: str
    relative_path: str
    filename: str
    file_format: EpubFormat
    sections: list[EpubSection] = field(default_factory=list)
    embedded_images: list[EpubEmbeddedImage] = field(default_factory=list)
    title: str | None = None
    creators: list[str] = field(default_factory=list)
    language: str | None = None
    publisher: str | None = None
    identifier: str | None = None
    chapter_count: int = 0
    image_count: int = 0
    vl_model_name: str | None = None
    artifact_dir: str | None = None
    parser_version: str = "epub_zip_xhtml_v1"
    parse_warnings: list[str] = field(default_factory=list)
    file_summary: str | None = None
    file_keywords: list[str] = field(default_factory=list)
    file_search_text: str | None = None
