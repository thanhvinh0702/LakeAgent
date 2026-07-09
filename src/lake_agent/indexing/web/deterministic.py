from __future__ import annotations

import hashlib
import html.parser
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from lake_agent.domain.indexing_models import WebFormat, WebIndexResult, WebSection
from lake_agent.indexing.text.chunking import build_basic_search_text


_SUPPORTED_FORMATS = {"html", "htm"}


@dataclass(frozen=True, slots=True)
class WebParseOptions:
    max_chars_per_chunk: int = 2400
    min_chunk_chars: int = 400


class WebHTMLParser(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.blocks = []  # List of tuples: (block_type, text_content)
        self.current_tag = None
        self.current_data = []
        self.heading_level = None
        self.in_list = False
        self.list_items = []
        self.in_table = False
        self.current_table = []
        self.current_row = []
        self.current_cell = None

        # Inline tags to ignore (just let data accumulate)
        self.inline_tags = {
            "a", "b", "i", "u", "span", "strong", "em", "code", "mark",
            "small", "sub", "sup", "ins", "del", "abbr", "cite", "q", "time"
        }

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in self.inline_tags:
            return

        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._flush()
            self.heading_level = int(tag[1])
            self.current_tag = "heading"
        elif tag in {"p", "blockquote", "pre", "div"}:
            self._flush()
            self.current_tag = "paragraph"
        elif tag in {"ul", "ol"}:
            self._flush()
            self.in_list = True
            self.list_items = []
        elif tag == "li" and self.in_list:
            self.current_tag = "li"
        elif tag == "table":
            self._flush()
            self.in_table = True
            self.current_table = []
        elif tag == "tr" and self.in_table:
            self.current_row = []
        elif tag in {"td", "th"} and self.in_table:
            self.current_cell = []
            self.current_tag = "cell"

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in self.inline_tags:
            return

        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"} and self.current_tag == "heading":
            self._flush()
            self.heading_level = None
        elif tag in {"p", "blockquote", "pre", "div"} and self.current_tag == "paragraph":
            self._flush()
        elif tag in {"ul", "ol"} and self.in_list:
            self._flush()
            self.in_list = False
        elif tag == "li" and self.current_tag == "li":
            data = "".join(self.current_data).strip()
            if data:
                self.list_items.append(f"- {data}")
            self.current_data = []
            self.current_tag = None
        elif tag == "table" and self.in_table:
            self._flush()
            self.in_table = False
        elif tag == "tr" and self.in_table:
            if self.current_row:
                self.current_table.append(self.current_row)
            self.current_row = []
        elif tag in {"td", "th"} and self.in_table:
            cell_data = "".join(self.current_cell).strip() if self.current_cell else ""
            self.current_row.append(cell_data)
            self.current_cell = None
            self.current_tag = None

    def handle_data(self, data):
        if self.current_tag in {"heading", "paragraph", "li"}:
            self.current_data.append(data)
        elif self.current_tag == "cell" and self.current_cell is not None:
            self.current_cell.append(data)
        elif not self.current_tag and not self.in_list and not self.in_table:
            trimmed = data.strip()
            if trimmed:
                self.blocks.append(("text", trimmed))

    def _flush(self):
        if self.current_tag == "heading":
            data = "".join(self.current_data).strip()
            if data:
                self.blocks.append((f"h{self.heading_level or 1}", data))
            self.current_data = []
            self.current_tag = None
        elif self.current_tag == "paragraph":
            data = "".join(self.current_data).strip()
            if data:
                self.blocks.append(("p", data))
            self.current_data = []
            self.current_tag = None
        elif self.in_list and self.list_items:
            self.blocks.append(("list", "\n".join(self.list_items)))
            self.list_items = []
        elif self.in_table and self.current_table:
            markdown_table = []
            for row in self.current_table:
                markdown_table.append("| " + " | ".join(row) + " |")
            if markdown_table:
                if len(markdown_table) > 1:
                    header_len = len(self.current_table[0])
                    separator = "|" + "|".join(["---"] * header_len) + "|"
                    markdown_table.insert(1, separator)
                self.blocks.append(("table", "\n".join(markdown_table)))
            self.current_table = []


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

        html_content = path.read_text(encoding="utf-8-sig", errors="replace")
        parser = WebHTMLParser()
        parser.feed(html_content)
        parser._flush()  # Final flush for any remaining state

        # Now group the parsed semantic blocks by heading
        # A section is a collection of content blocks under a heading
        sections: list[WebSection] = []
        warnings: list[str] = []

        current_heading: str | None = None
        current_content_parts: list[str] = []
        chunk_index = 1

        def add_section(heading, content_parts):
            nonlocal chunk_index
            content = "\n\n".join(content_parts).strip()
            if not content:
                return
            # Chunk the content if it exceeds max_chars_per_chunk
            chunks = _chunk_content(content, self._options.max_chars_per_chunk, self._options.min_chunk_chars)
            for chunk in chunks:
                sec_id = _stable_id(f"{normalized_source_id}:{chunk_index}:{heading or ''}", prefix="section")
                sections.append(
                    WebSection(
                        section_id=sec_id,
                        chunk_index=chunk_index,
                        heading=heading,
                        content=chunk,
                        char_count=len(chunk),
                    )
                )
                chunk_index += 1

        for block_type, text in parser.blocks:
            if block_type.startswith("h") and len(block_type) == 2 and block_type[1].isdigit():
                if current_content_parts:
                    add_section(current_heading, current_content_parts)
                current_heading = text
                current_content_parts = []
            else:
                current_content_parts.append(text)

        if current_content_parts or (current_heading and not sections):
            add_section(current_heading, current_content_parts)

        if not sections:
            warnings.append("The file did not contain any readable semantic HTML blocks.")

        result = WebIndexResult(
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


def _chunk_content(content: str, max_chars: int, min_chars: int) -> list[str]:
    # Simple block chunking: splits by double newline, tries to fit as much as possible
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for p in paragraphs:
        candidate = p if not current else f"{current}\n\n{p}"
        if len(candidate) > max_chars:
            if current:
                chunks.append(current)
                current = p
            else:
                # If a single paragraph is too large, split it by sentences
                sub_chunks = _split_paragraph(p, max_chars)
                for sc in sub_chunks[:-1]:
                    chunks.append(sc)
                current = sub_chunks[-1]
        else:
            current = candidate
    if current:
        if chunks and len(current) < min_chars and len(chunks[-1]) + len(current) + 2 <= max_chars:
            chunks[-1] = f"{chunks[-1]}\n\n{current}"
        else:
            chunks.append(current)
    return chunks


def _split_paragraph(p: str, max_chars: int) -> list[str]:
    import re
    sentences = re.split(r"(?<=[.!?])\s+", p)
    chunks = []
    current = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        candidate = s if not current else f"{current} {s}"
        if len(candidate) > max_chars:
            if current:
                chunks.append(current)
                current = s
            else:
                # Force chunk
                chunks.append(s[:max_chars])
                current = s[max_chars:]
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _build_file_search_text(result: WebIndexResult) -> str | None:
    parts = [result.filename, result.relative_path]
    for section in result.sections[:3]:
        if section.heading:
            parts.append(section.heading)
    return "\n".join(part for part in parts if part).strip() or None


def _stable_id(value: str, *, prefix: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"
