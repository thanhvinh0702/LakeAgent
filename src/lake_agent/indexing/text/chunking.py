from __future__ import annotations

import re
from dataclasses import dataclass

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
) -> list[StructuredTextChunk]:
    cleaned = text.strip()
    if not cleaned:
        return []

    return [
        StructuredTextChunk(
            chunk_index=index,
            heading=None,
            content=chunk,
            line_start=None,
            line_end=None,
        )
        for index, chunk in enumerate(
            _chunk_paragraphs(cleaned, max_chars=max_chars, min_chars=min_chars),
            start=1,
        )
    ]


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
