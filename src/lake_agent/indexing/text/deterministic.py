from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from lake_agent.domain.indexing_models import TextIndexResult, TextSection
from lake_agent.indexing.text.chunking import (
    build_basic_search_text,
    chunk_markdown_text,
    chunk_plain_text,
    normalize_text,
)

_SUPPORTED_FORMATS = {"txt", "md"}


@dataclass(frozen=True, slots=True)
class TextParseOptions:
    max_chars_per_chunk: int = 2400
    min_chunk_chars: int = 400


class DeterministicTextParser:
    def __init__(self, options: TextParseOptions | None = None) -> None:
        self._options = options or TextParseOptions()

    def parse_file(
        self,
        file_path: str | Path,
        *,
        relative_path: str | None = None,
        source_id: str | None = None,
    ) -> TextIndexResult:
        path = Path(file_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(path)

        extension = path.suffix.lower().removeprefix(".")
        if extension not in _SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported deterministic text format: {extension}")

        normalized_relative_path = relative_path or path.name
        normalized_relative_path = PurePosixPath(
            normalized_relative_path.replace("\\", "/")
        ).as_posix()
        normalized_source_id = source_id or _stable_id(
            normalized_relative_path,
            prefix="source",
        )

        text = path.read_text(encoding="utf-8-sig", errors="replace")
        normalized_text = normalize_text(text)
        warnings: list[str] = []

        if extension == "md":
            sections = self._build_sections(
                chunk_markdown_text(
                    normalized_text,
                    max_chars=self._options.max_chars_per_chunk,
                    min_chars=self._options.min_chunk_chars,
                ),
                normalized_source_id,
            )
        else:
            sections = self._build_sections(
                chunk_plain_text(
                    normalized_text,
                    max_chars=self._options.max_chars_per_chunk,
                    min_chars=self._options.min_chunk_chars,
                ),
                normalized_source_id,
            )

        if not sections:
            warnings.append("The file did not contain any readable text sections.")

        result = TextIndexResult(
            source_id=normalized_source_id,
            relative_path=normalized_relative_path,
            filename=path.name,
            file_format=extension,
            sections=sections,
            parse_warnings=warnings,
        )
        for section in result.sections:
            section.search_text = build_basic_search_text(
                section.heading,
                section.content,
            )
        result.file_search_text = _build_file_search_text(result)
        return result

    def _build_sections(self, chunks, source_id: str) -> list[TextSection]:
        return [
            _build_section(
                source_id=source_id,
                chunk_index=chunk.chunk_index,
                heading=chunk.heading,
                content=chunk.content,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
            )
            for chunk in chunks
        ]


def _build_section(
    *,
    source_id: str,
    chunk_index: int,
    heading: str | None,
    content: str,
    line_start: int | None,
    line_end: int | None,
) -> TextSection:
    normalized_content = content.strip()
    return TextSection(
        section_id=_stable_id(f"{source_id}:{chunk_index}:{heading or ''}", prefix="section"),
        chunk_index=chunk_index,
        heading=heading,
        content=normalized_content,
        line_start=line_start,
        line_end=line_end,
        char_count=len(normalized_content),
        search_text=build_basic_search_text(
            heading,
            normalized_content,
        ),
    )


def _build_file_search_text(result: TextIndexResult) -> str | None:
    parts = [result.filename, result.relative_path]
    for section in result.sections[:3]:
        if section.heading:
            parts.append(section.heading)
    return "\n".join(part for part in parts if part).strip() or None


def _stable_id(value: str, *, prefix: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"
