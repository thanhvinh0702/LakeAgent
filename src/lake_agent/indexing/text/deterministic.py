from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from lake_agent.domain.indexing_models import TextIndexResult, TextSection

_SUPPORTED_FORMATS = {"txt", "md"}
_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


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
        normalized_text = _normalize_text(text)
        warnings: list[str] = []

        if extension == "md":
            sections = self._parse_markdown(normalized_text, normalized_source_id)
        else:
            sections = self._parse_plain_text(normalized_text, normalized_source_id)

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
            section.search_text = _build_section_search_text(
                section.heading,
                section.content,
            )
        result.file_search_text = _build_file_search_text(result)
        return result

    def _parse_markdown(self, text: str, source_id: str) -> list[TextSection]:
        lines = text.split("\n")
        sections: list[tuple[str | None, list[str], int]] = []
        current_heading: str | None = None
        current_start = 1
        current_lines: list[str] = []

        for line_number, line in enumerate(lines, start=1):
            heading_match = _MARKDOWN_HEADING_RE.match(line)
            if heading_match:
                if current_lines:
                    sections.append((current_heading, current_lines, current_start))
                current_heading = heading_match.group(2).strip()
                current_lines = []
                current_start = line_number + 1
                continue
            current_lines.append(line)

        if current_lines:
            sections.append((current_heading, current_lines, current_start))

        built_sections: list[TextSection] = []
        chunk_index = 0
        for heading, raw_lines, line_start in sections:
            body = "\n".join(raw_lines).strip()
            if not body:
                continue
            for chunk in _chunk_paragraphs(
                body,
                max_chars=self._options.max_chars_per_chunk,
                min_chars=self._options.min_chunk_chars,
            ):
                chunk_index += 1
                built_sections.append(
                    _build_section(
                        source_id=source_id,
                        chunk_index=chunk_index,
                        heading=heading,
                        content=chunk,
                        line_start=line_start,
                        line_end=line_start + chunk.count("\n"),
                    )
                )
        return built_sections

    def _parse_plain_text(self, text: str, source_id: str) -> list[TextSection]:
        cleaned = text.strip()
        if not cleaned:
            return []

        sections: list[TextSection] = []
        for chunk_index, chunk in enumerate(
            _chunk_paragraphs(
                cleaned,
                max_chars=self._options.max_chars_per_chunk,
                min_chars=self._options.min_chunk_chars,
            ),
            start=1,
        ):
            sections.append(
                _build_section(
                    source_id=source_id,
                    chunk_index=chunk_index,
                    heading=None,
                    content=chunk,
                    line_start=None,
                    line_end=None,
                )
            )
        return sections


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
        search_text=_build_section_search_text(
            heading,
            normalized_content,
        ),
    )


def _chunk_paragraphs(
    text: str,
    *,
    max_chars: int,
    min_chars: int,
) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = paragraph
            continue
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_large_paragraph(paragraph, max_chars))
            continue
        current = candidate

    if current:
        if chunks and len(current) < min_chars and len(chunks[-1]) + len(current) + 2 <= max_chars:
            chunks[-1] = f"{chunks[-1]}\n\n{current}"
        else:
            chunks.append(current)
    return chunks


def _split_large_paragraph(paragraph: str, max_chars: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    if len(sentences) == 1:
        return [paragraph[index : index + max_chars].strip() for index in range(0, len(paragraph), max_chars)]

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        candidate = sentence if not current else f"{current} {sentence}"
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = sentence
            continue
        current = candidate
    if current:
        chunks.append(current)
    return chunks


def _normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()

def _build_section_search_text(heading: str | None, content: str) -> str:
    parts = []
    if heading:
        parts.append(heading)
    parts.append(content)
    return "\n".join(parts)


def _build_file_search_text(result: TextIndexResult) -> str | None:
    parts = [result.filename, result.relative_path]
    for section in result.sections[:3]:
        if section.heading:
            parts.append(section.heading)
    return "\n".join(part for part in parts if part).strip() or None


def _stable_id(value: str, *, prefix: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"
