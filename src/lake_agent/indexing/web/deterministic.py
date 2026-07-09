from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath

from lake_agent.domain.indexing_models import WebIndexResult, WebSection
from lake_agent.indexing.text.chunking import (
    StructuredTextChunk,
    build_basic_search_text,
    chunk_markdown_text,
    normalize_text,
)

_SUPPORTED_FORMATS = {"html", "htm"}
_BLOCK_TAGS = {
    "p",
    "div",
    "section",
    "article",
    "main",
    "aside",
    "blockquote",
    "pre",
    "li",
    "td",
    "th",
}
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_SKIP_TAGS = {"script", "style", "noscript", "svg"}


@dataclass(frozen=True, slots=True)
class WebParseOptions:
    max_chars_per_chunk: int = 2400
    min_chunk_chars: int = 400


@dataclass(frozen=True, slots=True)
class _HTMLBlock:
    block_type: str
    text: str
    line_start: int | None


class DeterministicWebParser:
    def __init__(self, options: WebParseOptions | None = None) -> None:
        self._options = options or WebParseOptions()

    def parse_file(
        self,
        file_path: str | Path,
        *,
        relative_path: str | None = None,
        source_id: str | None = None,
    ) -> WebIndexResult:
        path = Path(file_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(path)

        extension = path.suffix.lower().removeprefix(".")
        if extension not in _SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported deterministic web format: {extension}")

        normalized_relative_path = relative_path or path.name
        normalized_relative_path = PurePosixPath(
            normalized_relative_path.replace("\\", "/")
        ).as_posix()
        normalized_source_id = source_id or _stable_id(
            normalized_relative_path,
            prefix="source",
        )

        raw_html = path.read_text(encoding="utf-8-sig", errors="replace")
        extracted = _extract_html_blocks(raw_html)
        markdown_like = _render_markdown_like_document(extracted)
        normalized_text = normalize_text(markdown_like)
        warnings: list[str] = list(extracted.warnings)

        chunks = chunk_markdown_text(
            normalized_text,
            max_chars=self._options.max_chars_per_chunk,
            min_chars=self._options.min_chunk_chars,
        )
        sections = self._build_sections(chunks, normalized_source_id)
        if not sections:
            warnings.append("The HTML file did not contain any readable text sections.")

        result = WebIndexResult(
            source_id=normalized_source_id,
            relative_path=normalized_relative_path,
            filename=path.name,
            file_format=extension,
            sections=sections,
            parse_warnings=warnings,
        )
        for section in result.sections:
            section.search_text = build_basic_search_text(section.heading, section.content)
        result.file_search_text = _build_file_search_text(result)
        return result

    def _build_sections(
        self,
        chunks: list[StructuredTextChunk],
        source_id: str,
    ) -> list[WebSection]:
        return [
            WebSection(
                section_id=_stable_id(
                    f"{source_id}:{chunk.chunk_index}:{chunk.heading or ''}",
                    prefix="websec",
                ),
                chunk_index=chunk.chunk_index,
                heading=chunk.heading,
                content=chunk.content.strip(),
                line_start=chunk.line_start,
                line_end=chunk.line_end,
                char_count=len(chunk.content.strip()),
                search_text=build_basic_search_text(chunk.heading, chunk.content.strip()),
            )
            for chunk in chunks
            if chunk.content.strip()
        ]


@dataclass(slots=True)
class _ExtractedHTML:
    title: str | None
    meta_description: str | None
    blocks: list[_HTMLBlock]
    warnings: list[str]


class _HTMLStructureExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title: str | None = None
        self.meta_description: str | None = None
        self.blocks: list[_HTMLBlock] = []
        self.warnings: list[str] = []
        self._skip_depth = 0
        self._in_title = False
        self._current_heading_tag: str | None = None
        self._current_heading_line: int | None = None
        self._current_heading_parts: list[str] = []
        self._current_block_tag: str | None = None
        self._current_block_line: int | None = None
        self._current_block_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return

        if tag == "title":
            self._in_title = True
            return

        if tag == "meta":
            attr_map = {key.lower(): value for key, value in attrs}
            name = (attr_map.get("name") or attr_map.get("property") or "").lower()
            if name in {"description", "og:description"} and not self.meta_description:
                content = (attr_map.get("content") or "").strip()
                if content:
                    self.meta_description = _clean_html_text(content)
            return

        if tag in _HEADING_TAGS:
            self._flush_block()
            self._flush_heading()
            self._current_heading_tag = tag
            self._current_heading_line = self.getpos()[0]
            self._current_heading_parts = []
            return

        if tag in _BLOCK_TAGS and self._current_block_tag is None:
            self._current_block_tag = tag
            self._current_block_line = self.getpos()[0]
            self._current_block_parts = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if self._skip_depth > 0:
            return

        if tag == "title":
            self._in_title = False
            return

        if self._current_heading_tag == tag:
            self._flush_heading()
            return

        if self._current_block_tag == tag:
            self._flush_block()

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        cleaned = _clean_html_text(data)
        if not cleaned:
            return
        if self._in_title:
            self.title = cleaned if self.title is None else f"{self.title} {cleaned}".strip()
            return
        if self._current_heading_tag is not None:
            self._current_heading_parts.append(cleaned)
            return
        if self._current_block_tag is not None:
            self._current_block_parts.append(cleaned)

    def close(self) -> None:
        super().close()
        self._flush_heading()
        self._flush_block()

    def _flush_heading(self) -> None:
        if self._current_heading_tag is None:
            return
        text = _join_text_parts(self._current_heading_parts)
        if text:
            self.blocks.append(
                _HTMLBlock(
                    block_type=self._current_heading_tag,
                    text=text,
                    line_start=self._current_heading_line,
                )
            )
        self._current_heading_tag = None
        self._current_heading_line = None
        self._current_heading_parts = []

    def _flush_block(self) -> None:
        if self._current_block_tag is None:
            return
        text = _join_text_parts(self._current_block_parts)
        if text:
            self.blocks.append(
                _HTMLBlock(
                    block_type=self._current_block_tag,
                    text=text,
                    line_start=self._current_block_line,
                )
            )
        self._current_block_tag = None
        self._current_block_line = None
        self._current_block_parts = []


def _extract_html_blocks(raw_html: str) -> _ExtractedHTML:
    parser = _HTMLStructureExtractor()
    try:
        parser.feed(raw_html)
        parser.close()
    except Exception as exc:
        parser.warnings.append(f"HTML parsing warning: {exc}")
    return _ExtractedHTML(
        title=parser.title,
        meta_description=parser.meta_description,
        blocks=parser.blocks,
        warnings=parser.warnings,
    )


def _render_markdown_like_document(extracted: _ExtractedHTML) -> str:
    parts: list[str] = []
    if extracted.title:
        parts.append(f"# {extracted.title}")
    if extracted.meta_description:
        parts.append(extracted.meta_description)

    for block in extracted.blocks:
        if block.block_type in _HEADING_TAGS:
            level = int(block.block_type[1])
            hashes = "#" * max(1, min(6, level))
            parts.append(f"{hashes} {block.text}")
        else:
            parts.append(block.text)

    return "\n\n".join(part.strip() for part in parts if part.strip())


def _build_file_search_text(result: WebIndexResult) -> str | None:
    parts = [result.filename, result.relative_path]
    for section in result.sections[:3]:
        if section.heading:
            parts.append(section.heading)
    return "\n".join(part for part in parts if part).strip() or None


def _clean_html_text(value: str) -> str:
    value = unescape(value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _join_text_parts(parts: list[str]) -> str:
    return _clean_html_text(" ".join(part for part in parts if part.strip()))


def _stable_id(value: str, *, prefix: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"
