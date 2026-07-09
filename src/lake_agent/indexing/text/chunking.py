from __future__ import annotations

import re
from dataclasses import dataclass

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - compatibility fallback
    RecursiveCharacterTextSplitter = None

_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


@dataclass(frozen=True, slots=True)
class StructuredTextChunk:
    chunk_index: int
    heading: str | None
    content: str
    line_start: int | None
    line_end: int | None


def normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def chunk_markdown_text(
    text: str,
    *,
    max_chars: int,
    min_chars: int,
) -> list[StructuredTextChunk]:
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

    chunks: list[StructuredTextChunk] = []
    chunk_index = 0
    for heading, raw_lines, line_start in sections:
        body = "\n".join(raw_lines).strip()
        if not body:
            continue
        for chunk in _chunk_paragraphs(body, max_chars=max_chars, min_chars=min_chars):
            chunk_index += 1
            chunks.append(
                StructuredTextChunk(
                    chunk_index=chunk_index,
                    heading=heading,
                    content=chunk,
                    line_start=line_start,
                    line_end=line_start + chunk.count("\n"),
                )
            )
    return chunks


def chunk_plain_text(
    text: str,
    *,
    max_chars: int,
    min_chars: int,
    target_chars: int | None = None,
) -> list[StructuredTextChunk]:
    cleaned = text.strip()
    if not cleaned:
        return []

    chunk_size = min(target_chars or max_chars, max_chars)
    chunk_overlap = min(150, max(0, chunk_size // 5))
    splitter = _build_recursive_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True,
        length_function=len,
    )
    chunks = [chunk.strip() for chunk in splitter.split_text(cleaned) if chunk.strip()]

    return [
        StructuredTextChunk(
            chunk_index=index,
            heading=None,
            content=chunk,
            line_start=None,
            line_end=None,
        )
        for index, chunk in enumerate(chunks, start=1)
    ]


def _build_recursive_text_splitter(
    *,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
    keep_separator: bool,
    length_function,
):
    if RecursiveCharacterTextSplitter is not None:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            keep_separator=keep_separator,
            length_function=length_function,
        )
    return _FallbackRecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        keep_separator=keep_separator,
        length_function=length_function,
    )


def build_basic_search_text(heading: str | None, content: str) -> str:
    parts = []
    if heading:
        parts.append(heading)
    parts.append(content)
    return "\n".join(parts)


def _chunk_paragraphs(
    text: str,
    *,
    max_chars: int,
    min_chars: int,
    target_chars: int | None = None,
) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if (
            current
            and target_chars is not None
            and len(current) >= min_chars
            and len(current) + len(paragraph) + 2 > target_chars
        ):
            chunks.append(current)
            current = ""

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
        return [
            paragraph[index : index + max_chars].strip()
            for index in range(0, len(paragraph), max_chars)
        ]

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


class _FallbackRecursiveCharacterTextSplitter:
    def __init__(
        self,
        *,
        chunk_size: int,
        chunk_overlap: int,
        separators: list[str],
        keep_separator: bool,
        length_function,
    ) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = separators
        self._keep_separator = keep_separator
        self._length_function = length_function

    def split_text(self, text: str) -> list[str]:
        return self._split_text(text, self._separators)

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        stripped = text.strip()
        if not stripped:
            return []
        if self._length_function(stripped) <= self._chunk_size:
            return [stripped]
        if not separators:
            return self._split_fixed_width(stripped)

        separator = separators[0]
        if separator and separator in stripped:
            parts = self._split_with_separator(stripped, separator)
            return self._merge_parts(parts, separators)
        return self._split_text(stripped, separators[1:])

    def _merge_parts(self, parts: list[str], separators: list[str]) -> list[str]:
        chunks: list[str] = []
        current = ""

        for part in parts:
            candidate = part if not current else f"{current}{part}"
            if self._length_function(candidate) <= self._chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(current.strip())
            if self._length_function(part.strip()) <= self._chunk_size:
                current = part
            else:
                chunks.extend(self._split_text(part, separators[1:]))
                current = ""

        if current:
            chunks.append(current.strip())

        return self._apply_overlap([chunk for chunk in chunks if chunk])

    def _split_with_separator(self, text: str, separator: str) -> list[str]:
        if not self._keep_separator:
            return [part for part in text.split(separator) if part]

        raw_parts = text.split(separator)
        parts: list[str] = []
        for index, raw_part in enumerate(raw_parts):
            if not raw_part:
                continue
            prefix = "" if index == 0 else separator
            parts.append(f"{prefix}{raw_part}")
        return parts

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        if self._chunk_overlap <= 0 or len(chunks) < 2:
            return chunks

        overlapped: list[str] = [chunks[0]]
        for chunk in chunks[1:]:
            overlap = overlapped[-1][-self._chunk_overlap :].strip()
            if overlap and not chunk.startswith(overlap):
                combined = f"{overlap} {chunk}".strip()
                if self._length_function(combined) <= self._chunk_size:
                    overlapped.append(combined)
                    continue
            overlapped.append(chunk)
        return overlapped

    def _split_fixed_width(self, text: str) -> list[str]:
        step = max(1, self._chunk_size - self._chunk_overlap)
        return [
            text[index : index + self._chunk_size].strip()
            for index in range(0, len(text), step)
            if text[index : index + self._chunk_size].strip()
        ]
